/*
 * SquareSumV1: y = sum(x^2, dim=axis, keepdim).  dtype fp16/bf16/float.
 * Rewritten per plan.md: two regimes after folding to (outer L, reduceLen, inner).
 *
 *  ROW  (inner==1, reduce the last/contiguous axis, n = reduceLen):
 *    rows split over cores; per n-range batched reduce
 *      n<=8   : batch 512 segs -> (gather to 8-blocks if n<8) -> BlockReduceSum
 *      n<=64  : batch 64  segs -> (gather to 64-blocks if n<64) -> WholeReduceSum
 *      n<=512 : batch 8   segs -> (gather to 512-blocks if n<512) -> Whole+BlockReduceSum
 *      n<=4096: 2 segs padded to 4096 -> per-seg ReduceSum
 *      n>4096 : per row, chunk by 4096, accumulate, final ReduceSum
 *    square in fp32 (cast bf16/fp16 -> fp32 first).
 *
 *  COL  (inner>1, reduce a middle axis m=reduceLen, keep n=inner):  (l,m,n)->(l,n)
 *    fp32 workspace[l*n] (or output gm directly when dtype==float); split (l, m-chunks)
 *    over cores, each accumulates dst[n] over its m-rows (binary fold), atomic-adds to
 *    workspace[l*n]; SyncAll; then cast workspace -> output.
 *
 *  GENERAL (non-contiguous axis): per kept-output row (kept from earlier impl).
 *  RPAR    (inner==1, few rows / reduce-all): split reduce over cores + combine.
 */
#include "kernel_operator.h"
#include "lib/reduce/reduce.h"

constexpr uint32_t F32_BLK = 8;     // fp32 per 32B datablock
constexpr uint32_t MAX_DIM = 8;
constexpr uint32_t ROW_TILE = 4096;  // fp32 elements per row-batch working set
// mode 4: decouple the contiguous READ tile from the gather-align output width. The read/accum
// tile (ktCap) is capped by tileElems/IN_R;
// the gather output [ktCap, INp] then needs a bigger scratch than ROW_TILE. GATHER_CAP fp32
// sizes gthBuf/idxBuf for mode 4 ONLY so ktCap can reach tileElems/IN_R without the old
// ROW_TILE/INp burst cap (was 51 rows = 7KB). The gather runs ONCE per tile, but ReduceSum<AR>
// then runs in REDUCE_CHUNK sub-slices of the already-aligned output so the lib's internal
// scratch stays small (a single ReduceSum over the whole [118,72] tile blew the UB budget).
constexpr uint32_t GATHER_CAP = 8544;   // fp32 elems for mode-4 gather scratch (>= 118*72)
constexpr uint32_t REDUCE_CHUNK = 56;   // rows per ReduceSum<AR> call (bounds its UB scratch)
constexpr uint32_t MODE4_OUTQ = 512;    // mode-4 only writes <=REDUCE_CHUNK vals -> tiny outQue
// ---- FLAT (mode 9) ----
constexpr uint32_t FLAT_POS = 256u;             // positions per read
constexpr uint32_t FLAT_BATCH = 2048u;          // row sums per atomic store (a multiple of FLAT_POS)
constexpr uint32_t FLAT_CLEAR_BYTES = 8192u;    // output bytes zeroed per store at start
constexpr int32_t FLAT_EVENT = 3;

// Each tiling mode is compiled as its own kernel binary (host key = mode + 1, dispatched in the entry),
// so SS_MODE folds to a constant and a binary carries only its own mode's path.
#define SS_MODE (TILING_KEY_IS(1) ? 0u : TILING_KEY_IS(3) ? 2u : TILING_KEY_IS(4) ? 3u : \
                 TILING_KEY_IS(5) ? 4u : TILING_KEY_IS(10) ? 9u : 1u)

__aicore__ inline uint32_t Align64(uint32_t x) { return (x + 63) / 64 * 64; }
__aicore__ inline uint32_t Align8(uint32_t x) { return (x + 7) / 8 * 8; }
__aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t a) { return (x + a - 1) / a * a; }
__aicore__ inline uint32_t MinU(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline uint64_t MinU2(uint64_t a, uint64_t b) { return a < b ? a : b; }

template <typename T>
__aicore__ inline void PadIn(const AscendC::LocalTensor<T>& dst,
                             const AscendC::GlobalTensor<T>& src, uint32_t n) {
  AscendC::DataCopyExtParams p{1, (uint32_t)(n * sizeof(T)), 0, 0, 0};
  AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
  AscendC::DataCopyPad(dst, src, p, pad);
}
template <typename T>
__aicore__ inline void PadOut(const AscendC::GlobalTensor<T>& dst,
                              const AscendC::LocalTensor<T>& src, uint32_t n) {
  AscendC::DataCopyExtParams p{1, (uint32_t)(n * sizeof(T)), 0, 0, 0};
  AscendC::DataCopyPad(dst, src, p);
}

template <typename typeT>
class KernelSquareSumV1 {
  static constexpr uint32_t ALN = 32 / sizeof(typeT);

 public:
  __aicore__ inline KernelSquareSumV1() {}
  template <typename TT>
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR y, GM_ADDR workspace,
                              TT t, AscendC::TPipe* pipeIn) {
    pipe = pipeIn;
    tileElems = t->tileElems;
    outer = t->outer;
    reduceLen = t->reduceLen;
    inner = t->inner;
    keptPerCore = t->outerPerCore;  // mode 4: kept rows per core
    x_gm.SetGlobalBuffer((__gm__ typeT*)input);
    y_gm.SetGlobalBuffer((__gm__ typeT*)y);
    uint32_t bidx = AscendC::GetBlockIdx();

    if (SS_MODE == 9u) {  // FLAT: plane-order aligned reads, fp16 per-row block reduce
      mPerCore = t->mPerCore;  // positions per core
      colCores = t->colCores;
      if ((uint64_t)outer * reduceLen * inner * sizeof(typeT) >= (8ULL << 20)) {
        x_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
      }
      usrWs.SetGlobalBuffer((__gm__ float*)(AscendC::GetUserWorkspace(workspace)));  // fp32 sums
      uint32_t es = (uint32_t)sizeof(typeT);
      // the last row of a read is reduced through a window reaching 48 elements past it; the
      // same slots later carry this core's fp32 share of the sums back
      pipe->InitBuffer(inQue, 2, AlignUp(FLAT_POS * reduceLen * es, 32u) + 128u);
      pipe->InitBuffer(outQue, 1, FLAT_CLEAR_BYTES + 32u);  // clears, fp32 batches, fp16 results
      pipe->InitBuffer(accBuf, FLAT_POS * 16u * es + 256u);  // one 16-element slot per row
      pipe->InitBuffer(sqBuf, FLAT_BATCH * es + 256u);       // fp16 row sums of one batch
      pipe->InitBuffer(gthBuf, FLAT_BATCH * 4u + 256u);      // fp32 sums that follow a plane wrap
      pipe->InitBuffer(idxBuf, FLAT_BATCH * sizeof(uint32_t));  // 4B offsets moving them
      return;
    }

    if (SS_MODE == 0u) {  // ROW
      oStart = bidx * t->outerPerCore;
      oCount = oStart >= outer ? 0 : MinU(outer - oStart, t->outerPerCore);
    } else if (SS_MODE == 3u) {  // COLUMN
      colCores = t->colCores;
      mPerCore = t->mPerCore;     // m-rows per core within one l
      nP = t->innerP;             // Align8(inner) for fp32 dst
      usrWs.SetGlobalBuffer((__gm__ float*)(AscendC::GetUserWorkspace(workspace)));
    } else if (SS_MODE == 2u) {  // RPAR
      O = t->O; Oslot = t->Oslot; rCores = t->rCores; rPerCore = t->rPerCore;
      usrWs.SetGlobalBuffer((__gm__ float*)(AscendC::GetUserWorkspace(workspace)));
    } else if (SS_MODE == 4u) {  // prefix/suffix reduction, contiguous keep axis
      keptPerCore = t->outerPerCore;
    } else {  // GENERAL
      unitReduced = t->unitReduced; unit = t->unit; klProd = t->klProd; rlProd = t->rlProd;
      nKL = t->nKL; nRL = t->nRL;
      koStart = bidx * t->koPerCore; koEnd = MinU(koStart + t->koPerCore, klProd);
      for (uint32_t d = 0; d < MAX_DIM; d++) {
        klDim[d] = t->klDim[d]; klStr[d] = t->klStr[d];
        rlDim[d] = t->rlDim[d]; rlStr[d] = t->rlStr[d];
      }
    }
    // Per-dtype buffer sizing: queues sized for typeT (fp16/bf16 -> half the bytes
    // of fp32), so double-buffering fits in UB.  fp32-typed workspace loads/stores
    // (COLUMN/RPAR) reuse these queues capped at fltChunk = (queue bytes)/4 floats.
    // DB lets MTE2 load overlap the vector square/reduce (else ~2x serial).
    fltChunk = (tileElems * (uint32_t)sizeof(typeT)) / (uint32_t)sizeof(float);
    // Large ROW reductions whose post-cast row stride is only a few lanes away
    // from a 32B boundary can load a whole phase cycle as one contiguous GM
    // transaction.  For d = reduceLen mod F32_BLK, a cycle of F32_BLK/d rows
    // returns to the original phase.  Keep aligned interiors in place and
    // Gather only the boundary values into 16-lane slots.  The guards below
    // are geometric/resource predicates; no challenge-shape literal is used.
    rowBoundaryRows_ = 0u;
    if (sizeof(typeT) == 2 && SS_MODE == 0u) {
      uint32_t d = reduceLen % F32_BLK;
      // A phase cycle is F32_BLK/d rows, and every batch writes its results at
      // accOut[done] with done a multiple of that cycle length.  A vector store
      // needs a 32B (= F32_BLK float) destination, so only d == 1 keeps every
      // batch's output slot aligned; d == 2 and d == 4 satisfy the old
      // F32_BLK % d == 0 test but fault on device with 507015 "The UB address
      // accessed by the VEC instruction is not aligned".  Those reduceLens take
      // the generic RowPadBatch/RowTier path instead.
      if (d == 1u) {
        uint32_t rows = F32_BLK / d;
        uint64_t rowElems64 = (uint64_t)rows * reduceLen;
        // At least one aligned fp32 block must remain in every row.  Without
        // this guard (for example R=4,d=4,Q=0), ReduceSum would receive a
        // zero-width descriptor and the boundary-only layout is invalid.
        uint32_t q = reduceLen - d;
        if (q >= F32_BLK && rowElems64 <= 0xffffffffULL - (ALN - 1u)) {
          uint32_t rowTile = AlignUp((uint32_t)rowElems64, ALN);
          // Keep the enlarged input/square tile comfortably below the UB limit;
          // this also leaves room for ReduceSum scratch and the boundary table.
          uint64_t estimated = 2ULL * (uint64_t)rowTile * sizeof(typeT) +
                               (uint64_t)rowTile * sizeof(float) +
                               (uint64_t)tileElems * sizeof(float) +
                               (uint64_t)ROW_TILE * sizeof(float) +
                               (uint64_t)tileElems * sizeof(typeT) +
                               3ULL * 256ULL;
          if (rows >= 2u && outer >= 8u * rows && estimated <= 190ULL * 1024ULL) {
            rowBoundaryRows_ = rows;
          }
        }
      }
    }
    rowBoundary_ = rowBoundaryRows_ != 0u;
    bool needGather = false;
    if (SS_MODE == 0u) {
      uint32_t n = reduceLen;
      bool exact = (n == 8 || n == 64 || n == 512);
      // tiny non-aligned R: gather-to-block (rightPad's sub-256B blocks starve MTE2)
      if (!exact && n <= 512 && n * (uint32_t)sizeof(typeT) < 256) needGather = true;
      if (rowBoundary_) needGather = true;
    } else if (SS_MODE == 3u) {
      uint32_t n = inner;
      // COLUMN gather: any non-aligned inner whose padded row fits ROW_TILE (colGR>=1).
      // Extends past the old n<512 so medium-N (e.g. 521) reads contiguously + accumulates
      // via MulAddDst instead of the per-row strided pad-load.
      if ((n % ALN != 0) && AlignUp(n, ALN) <= ROW_TILE) needGather = true;
    } else if (SS_MODE == 4u) {
      needGather = true;  // mode 4: contiguous big-block read + gather-align inner reduce
    }
    // fp32 paths calling ReduceSum<AR> (ROW rightPad/big, RPAR, GENERAL) need UB headroom
    // for the lib's internal scratch -> disable double-buffer there. COLUMN (fold-based,
    // non-gather) keeps DB.
    // Descriptor-bound non-aligned column reductions use a small phase table but
    // load several phase cycles in one contiguous transaction.  Eligibility and
    // burst depth are derived from dtype, alignment cycle, traffic, and UB budget.
    bool f32ColGather = (sizeof(typeT) == 4 && SS_MODE == 3u && needGather);
    uint64_t inputBytes = (uint64_t)outer * reduceLen * inner * sizeof(typeT);
    f32ColBurst_ = f32ColGather && outer == 1 && (inner & 7u) == 1u &&
                   inner <= 768u && inputBytes >= (64ULL << 20);
    colBurstMul_ = 1u;
    uint32_t phaseRows = f32ColBurst_ ? 8u : 0u;
    if (f32ColBurst_) {
      // Keep the burst as large as the complete one-buffer geometry permits.
      // For the fp32 C5 geometry, eight phase groups fit below the 192 KiB UB
      // limit (the former 120 KiB cap unnecessarily stopped at seven).
      uint32_t fit = (140u * 1024u) / (phaseRows * inner * sizeof(typeT));
      if (fit > 4u) fit = 4u;
      if (fit < 2u) { f32ColBurst_ = false; phaseRows = 0u; }
      else colBurstMul_ = fit;
    }
    bool halfColBurst = sizeof(typeT) == 2 && SS_MODE == 3u && needGather &&
                        inner <= 512u && inputBytes >= (64ULL << 20);
    if (halfColBurst) colBurstMul_ = 8u;

    // Only the half path accumulates a paired read in place; bf16 squares each plane through
    // sqBuf, which the pair geometry sizes to MODE4_OUTQ, so pairing bf16 overran it (507015).
    mode4Pair_ = std::is_same_v<typeT, half> && SS_MODE == 4u && (outer & 1u) == 0u &&
                 reduceLen >= 32u && reduceLen <= 256u &&
                 inputBytes >= (64ULL << 20);
    mode4Kt_ = 0u;
    if (mode4Pair_) {
      uint32_t inp = AlignUp(reduceLen, F32_BLK);
      for (uint32_t cand = 256u; cand >= 2u; cand -= 2u) {
        uint32_t packed = cand * reduceLen;
        uint32_t pairPitch = AlignUp(packed, 16u);
        uint32_t gather = cand * inp;
        uint64_t ub = 2ULL * (2ULL * pairPitch) * sizeof(typeT) +
                      (uint64_t)MODE4_OUTQ * sizeof(typeT) +
                      ((uint64_t)MODE4_OUTQ + packed + 2ULL * gather) * sizeof(float) +
                      3ULL * 256ULL;
        if (((uint64_t)pairPitch * sizeof(typeT)) % 512ULL == 0ULL &&
            ub <= 190ULL * 1024ULL) {
          mode4Kt_ = cand;
          break;
        }
      }
      if (mode4Kt_ == 0u) mode4Pair_ = false;
    }
    if (inputBytes >= (8ULL << 20)) {
      x_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    uint8_t inNum = f32ColBurst_ ? 2 : f32ColGather ? 2
                    : (sizeof(typeT) == 4 && (SS_MODE != 3u || needGather)) ? 1 : 2;
    uint32_t colRows = (SS_MODE == 3u && needGather) ? ROW_TILE / AlignUp(inner, ALN) : 1u;
    if (colRows < 1u) colRows = 1u;
    uint32_t mode4PairPitch = mode4Pair_ ? AlignUp(mode4Kt_ * reduceLen, 16u) : 0u;
    // mode 4 planes read one at a time: a tile rounded to whole 32B-aligned reads sizes its queue
    uint32_t mode4Rows = (SS_MODE == 4u && !mode4Pair_) ? Mode4AlignedRows(Mode4TileRows()) : 0u;
    uint32_t rowBoundaryTile = rowBoundary_ ? AlignUp(rowBoundaryRows_ * reduceLen, ALN) : 0u;
    uint32_t inTile = rowBoundary_ ? rowBoundaryTile
                      : mode4Pair_ ? 2u * mode4PairPitch
                      : mode4Rows != 0u ? AlignUp(mode4Rows * reduceLen, 16u)
                      : f32ColBurst_ ? phaseRows * colBurstMul_ * inner
                      : halfColBurst ? colRows * colBurstMul_ * inner
                      : f32ColGather ? ROW_TILE : tileElems;
    pipe->InitBuffer(inQue, inNum, inTile * (uint32_t)sizeof(typeT));
    // mode 4 writes at most REDUCE_CHUNK results per ReduceSum call -> a tiny outQue frees ~15KB
    // of UB for the enlarged mode-4 gather scratch + ReduceSum tmp. Other modes keep tileElems.
    uint32_t outTile = halfColBurst ? (AlignUp(inner, ALN) * 2u)
                       : (SS_MODE == 4u) ? MODE4_OUTQ
                       : f32ColBurst_ ? AlignUp(inner, ALN) : tileElems;
    pipe->InitBuffer(outQue, 1, outTile * sizeof(typeT));
    // +256B bank stagger: UB is 512B rows; the two operands of a vec op must be offset by 256B
    // (mod 512) to avoid a read bank conflict. Natural sizes are 512-multiples -> adjacent TBufs
    // land at the same phase (offset 0 -> conflict). Padding each fp32 TBuf by 256B offsets the
    // next -> Gather(sqBuf,idxBuf) & dst-Add(accBuf,gthBuf) become conflict-free.
    uint32_t sqCap = halfColBurst
                         ? (64u + AlignUp(colRows * inner, 8u) + ALN)
                     : rowBoundary_ ? rowBoundaryTile
                         : mode4Pair_ ? MODE4_OUTQ
                         : f32ColBurst_
                         ? 64u + AlignUp(phaseRows * inner, 8u) + ALN
                         : tileElems;
    uint32_t sqBytes = sqCap * (uint32_t)sizeof(float);
    pipe->InitBuffer(sqBuf, sqBytes + (f32ColBurst_ ? 0u
                                      : ((sqBytes % 512u) == 256u ? 0u : 256u)));
    uint32_t accCap = halfColBurst ? AlignUp(inner, ALN)
                      : mode4Pair_ ? mode4Kt_ * reduceLen
                      : f32ColBurst_ ? AlignUp(inner, ALN)
                      : tileElems;
    pipe->InitBuffer(accBuf, accCap * sizeof(float) + 256);
    // mode 4 uses a wider gather scratch (GATHER_CAP) so its read tile can reach tileElems/IN_R
    // (16KB burst) instead of the old ROW_TILE/INp cap; all other modes keep ROW_TILE.
    uint32_t gthCap = mode4Pair_ ? mode4Kt_ * AlignUp(reduceLen, F32_BLK)
                      : (SS_MODE == 4u) ? GATHER_CAP
                      : f32ColBurst_ ? phaseRows * AlignUp(inner, ALN)
                      : ROW_TILE;
    pipe->InitBuffer(gthBuf, gthCap * sizeof(float) + 256);
    // Boundary ROW needs one 16-lane slot per phase-cycle row plus one output
    // assembly offset per row; do not pay for the generic 4096-entry table.
    // The bf16 aligned path keeps each row's head and tail values in separate slots (BuildRowBoundaryIdxAt).
    constexpr uint32_t BOUNDARY_SLOT =
        (sizeof(typeT) == 2 && !std::is_same_v<typeT, half>) ? 32u : 16u;
    uint32_t idxCap = rowBoundary_ ? rowBoundaryRows_ * BOUNDARY_SLOT + rowBoundaryRows_ : gthCap;
    if (needGather) pipe->InitBuffer(idxBuf, AlignUp(idxCap, F32_BLK) * sizeof(uint32_t));
  }

  // bits [lo, hi) of one 64-lane mask word
  __aicore__ inline uint64_t LaneBits(uint32_t lo, uint32_t hi) {
    uint64_t upper = (hi >= 64u) ? ~0ULL : ((1ULL << hi) - 1ULL);
    uint64_t lower = (lo >= 64u) ? ~0ULL : ((1ULL << lo) - 1ULL);
    return upper & ~lower;
  }

  // FLAT: this core's equal, FLAT_BATCH-aligned share [z0, z1) of the kept positions (cores past
  // the end own none).  The core zeroes that share of the fp32 sums first and casts it to y last.
  __aicore__ inline bool FlatShare(uint32_t b, uint64_t totalOut, uint64_t& z0, uint64_t& z1) {
    uint64_t share = (totalOut + colCores - 1u) / colCores;
    uint64_t clrElems = (share + FLAT_BATCH - 1u) / FLAT_BATCH * FLAT_BATCH;
    z0 = (uint64_t)b * clrElems;
    z1 = MinU2(z0 + clrElems, totalOut);
    return z0 < totalOut;
  }

  // FLAT: zero this core's share of the fp32 sums and wait for the store to land before this
  // core's first atomic add.
  __aicore__ inline void ClearFlatShare(uint32_t b, uint64_t totalOut) {
    uint64_t z0 = 0u, z1 = 0u;
    if (!FlatShare(b, totalOut, z0, z1)) return;
    AscendC::LocalTensor<float> zo = outQue.template AllocTensor<float>();
    AscendC::Duplicate(zo, 0.0f, (int32_t)FLAT_BATCH);
    outQue.template EnQue<float>(zo);
    zo = outQue.template DeQue<float>();
    for (uint64_t z = z0; z < z1; z += FLAT_BATCH) {
      PadOut(usrWs[z], zo, (uint32_t)MinU2(FLAT_BATCH, z1 - z));
    }
    outQue.FreeTensor(zo);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(FLAT_EVENT);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(FLAT_EVENT);
  }

  // FLAT (mode 9, fp16): each core streams its plane-order share of positions in FLAT_POS-position
  // reads that start and end on 32B boundaries.  A read is squared in place.  Rows k = c + 16m of a
  // read start IN_R*m blocks after row c at the same 16-element phase, so one masked block reduce per
  // phase class writes each row's (at most 8) block sums into its 16-element slot, and one more
  // block reduce turns the slots into fp16 row sums.  FLAT_BATCH of those at a time are cast to fp32
  // and atomic-added to the workspace (this core's share zeroed first); the part of a batch past a
  // plane wrap moves to aligned stores that start at kept position 0.  Once every core's adds have
  // landed (SyncAll), each core casts its share of the fp32 sums back to y.  Summing the planes in
  // fp16 instead lets each atomic add round to the output's own ulp, and those errors accumulate.
  __aicore__ inline void ProcessFlat() {
    if constexpr (std::is_same_v<typeT, half>) {
      uint32_t OUT_R = outer, IN_R = reduceLen, KEPT = inner;
      uint32_t b = AscendC::GetBlockIdx();
      ClearFlatShare(b, (uint64_t)KEPT);
      uint64_t totalPos = (uint64_t)OUT_R * KEPT;
      uint64_t per = mPerCore;
      uint64_t p0 = (uint64_t)b * per;
      if (p0 < totalPos) {
        uint64_t p1 = MinU2(p0 + per, totalPos);
        uint64_t cmask[16][2];
        for (uint32_t c = 0; c < 16u; c++) {
          uint32_t phi = (IN_R * c) & 15u;
          cmask[c][0] = LaneBits(phi, phi + IN_R);
          cmask[c][1] = (phi + IN_R > 64u) ? LaneBits(0u, phi + IN_R - 64u) : 0ULL;
        }
        AscendC::LocalTensor<uint32_t> ofsU = idxBuf.Get<uint32_t>();
        AscendC::LocalTensor<int32_t> ofs = ofsU.template ReinterpretCast<int32_t>();
        AscendC::ArithProgression<int32_t>(ofs, 0, 4, (int32_t)FLAT_BATCH);
        // slot lanes a class never writes (fully masked blocks, lanes 8..15) stay zero from here on
        AscendC::LocalTensor<half> slots = accBuf.Get<half>();
        AscendC::LocalTensor<int16_t> slotz = slots.template ReinterpretCast<int16_t>();
        AscendC::Duplicate(slotz, (int16_t)0, (int32_t)(FLAT_POS * 16u));
        AscendC::LocalTensor<half> rs = sqBuf.Get<half>();
        AscendC::SetAtomicAdd<float>();
        uint32_t filled = 0u;
        uint64_t bp = p0;
        for (uint64_t pp = p0; pp < p1; pp += FLAT_POS) {
          uint32_t rows = (uint32_t)MinU2((uint64_t)FLAT_POS, p1 - pp);
          uint32_t cnt = rows * IN_R;
          AscendC::LocalTensor<half> xin = inQue.template AllocTensor<half>();
          if (((cnt * 2u) & 31u) == 0u) {
            AscendC::DataCopy(xin, x_gm[pp * IN_R], cnt);
          } else {
            PadIn(xin, x_gm[pp * IN_R], cnt);
          }
          inQue.EnQue(xin);
          xin = inQue.template DeQue<half>();
          AscendC::Mul(xin, xin, xin, (int32_t)cnt);
          for (uint32_t c = 0; c < 16u && c < rows; c++) {
            AscendC::BlockReduceSum<half>(slots[c * 16u], xin[((IN_R * c) >> 4) * 16u],
                                          (int32_t)((rows - c + 15u) / 16u), cmask[c], 32, 1,
                                          (int32_t)IN_R);
          }
          AscendC::BlockReduceSum<half>(rs[filled], slots, (int32_t)((rows + 7u) / 8u), 128, 1, 1, 8);
          inQue.FreeTensor(xin);
          filled += rows;
          if (filled + FLAT_POS <= FLAT_BATCH && pp + rows < p1) continue;
          // positions [bp, bp + filled): up to the plane wrap in place, the rest in stores that
          // each start at kept position 0
          uint32_t ks = (uint32_t)(bp % KEPT);
          uint32_t w = MinU(filled, KEPT - ks);
          AscendC::LocalTensor<float> yo = outQue.template AllocTensor<float>();
          AscendC::Cast(yo, rs, AscendC::RoundMode::CAST_NONE, (int32_t)filled);
          AscendC::LocalTensor<float> tail = gthBuf.Get<float>();
          if (w < filled) AscendC::Gather(tail, yo, ofsU, w * 4u, filled - w);
          outQue.template EnQue<float>(yo);
          yo = outQue.template DeQue<float>();
          PadOut(usrWs[ks], yo, w);
          outQue.FreeTensor(yo);
          for (uint32_t done = w; done < filled;) {
            uint32_t seg = MinU(filled - done, KEPT);
            AscendC::LocalTensor<float> yo2 = outQue.template AllocTensor<float>();
            AscendC::Gather(yo2, tail, ofsU, (done - w) * 4u, seg);
            outQue.template EnQue<float>(yo2);
            yo2 = outQue.template DeQue<float>();
            PadOut(usrWs, yo2, seg);
            outQue.FreeTensor(yo2);
            done += seg;
          }
          bp += filled;
          filled = 0u;
        }
        AscendC::SetAtomicNone();
      }
      // every core's adds have landed: cast this core's share of the fp32 sums back to y
      AscendC::SyncAll();
      uint64_t z0 = 0u, z1 = 0u;
      if (FlatShare(b, (uint64_t)KEPT, z0, z1)) {
        // Send the share back in as few FLAT_BATCH multiples as the queues hold (an inQue slot takes the
        // fp32 read, the outQue slot the fp16 result): one read, cast and store per core where it fits.
        const uint32_t inSlot = AlignUp(FLAT_POS * IN_R * 2u, 32u) + 128u;
        uint32_t wbStep = MinU(inSlot / 4u, (uint32_t)((FLAT_CLEAR_BYTES + 32u) / 2u)) / FLAT_BATCH * FLAT_BATCH;
        if (wbStep < FLAT_BATCH) wbStep = FLAT_BATCH;
        for (uint64_t z = z0; z < z1; z += wbStep) {
          uint32_t c = (uint32_t)MinU2(wbStep, z1 - z);
          AscendC::LocalTensor<float> wb = inQue.template AllocTensor<float>();
          PadIn(wb, usrWs[z], c);
          inQue.template EnQue<float>(wb);
          wb = inQue.template DeQue<float>();
          AscendC::LocalTensor<half> yh = outQue.template AllocTensor<half>();
          AscendC::Cast(yh, wb, AscendC::RoundMode::CAST_RINT, (int32_t)c);
          outQue.template EnQue<half>(yh);
          yh = outQue.template DeQue<half>();
          PadOut(y_gm[z], yh, c);
          outQue.FreeTensor(yh);
          inQue.FreeTensor(wb);
        }
      }
    }
  }

  __aicore__ inline void Process() {
    if (SS_MODE == 0u) ProcessRow();
    else if (SS_MODE == 3u) ProcessCol();
    else if (SS_MODE == 2u) ProcessRPar();
    else if (SS_MODE == 4u) ProcessMode4();
    else if (SS_MODE == 9u) ProcessFlat();
    else ProcessGeneral();
  }

 private:
  __aicore__ inline void SquareToF32(const AscendC::LocalTensor<float>& sq,
                                     const AscendC::LocalTensor<typeT>& xin, uint32_t n) {
    if constexpr (std::is_same_v<typeT, float>) {
      AscendC::Mul(sq, xin, xin, n);
    } else if constexpr (std::is_same_v<typeT, half>) {
      AscendC::Mul(xin, xin, xin, n);
      AscendC::Cast(sq, xin, AscendC::RoundMode::CAST_NONE, n);
    } else {  // bf16: square in fp32, NO bf16 round-trip. torch rounds each square to bf16
      // (~0.2%/elem, random sign) -> fp32-sum differs from golden by ~0.004% << 1e-2 gate.
      // Drops 2 Casts/elem -> ~halves the square vec (wide-row workload is 88% vec-bound: big win).
      AscendC::Cast(sq, xin, AscendC::RoundMode::CAST_NONE, n);
      AscendC::Mul(sq, sq, sq, n);
    }
  }
  __aicore__ inline void CastF32To(const AscendC::LocalTensor<typeT>& yout,
                                   const AscendC::LocalTensor<float>& src, uint32_t n) {
    if constexpr (std::is_same_v<typeT, float>) AscendC::Adds(yout, src, 0.0f, n);
    else AscendC::Cast(yout, src, AscendC::RoundMode::CAST_RINT, n);
  }
  // cast raw typeT -> fp32 (for the square-in-fp32 path); for fp32 just copy
  __aicore__ inline void RawToF32(const AscendC::LocalTensor<float>& dst,
                                  const AscendC::LocalTensor<typeT>& xin, uint32_t n) {
    if constexpr (std::is_same_v<typeT, float>) AscendC::Adds(dst, xin, 0.0f, n);
    else AscendC::Cast(dst, xin, AscendC::RoundMode::CAST_NONE, n);
  }

  // ============================ ROW (inner==1) ============================
  // build a gather offset table that scatters `segs` segments of `n` (<block)
  // into `block`-aligned slots: idx[i*block+k] = (k<n)? (i*n+k):ZERO ; *sizeof(float)
  __aicore__ inline void BuildRowGather(uint32_t segs, uint32_t n, uint32_t block,
                                        uint32_t zeroOff) {
    BuildRowGatherE(segs, n, block, zeroOff, (uint32_t)sizeof(float));
  }
  // element-sized variant: gather byte-offsets use `esz` (sizeof gathered dtype)
  __aicore__ inline void BuildRowGatherE(uint32_t segs, uint32_t n, uint32_t block,
                                         uint32_t zeroOff, uint32_t esz) {
    AscendC::LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    for (uint32_t i = 0; i < segs; i++)
      for (uint32_t k = 0; k < block; k++)
        idx.SetValue(i * block + k, (k < n ? (i * n + k) : zeroOff) * esz);
  }

  // VECTORIZED gather-idx build: replaces segs*block scalar SetValue (~27us serial on the
  // critical path before the gather loop) with Duplicate + per-row ArithProgression.  ArithProgression
  // rounds its count up to a 32B multiple, so it can overwrite the pad lanes [n..block) with progression
  // values (not zeroOff); reduce paths (mode 4) SUM the pad -> re-zero [n..block) with a few scalar
  // SetValue (still ~block/(block-n) fewer than the full build).  PIPE_ALL orders V build -> scalar
  // fix -> the V gather that reads idx.
  __aicore__ inline void BuildRowGatherVec(uint32_t segs, uint32_t n, uint32_t block,
                                           uint32_t zeroOff, uint32_t esz) {
    AscendC::LocalTensor<int32_t> idx =
        idxBuf.Get<uint32_t>().template ReinterpretCast<int32_t>();
    AscendC::Duplicate(idx, (int32_t)(zeroOff * esz), segs * block);
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < segs; i++)
      AscendC::ArithProgression<int32_t>(idx[i * block], (int32_t)(i * n * esz),
                                         (int32_t)esz, (int32_t)n);
    AscendC::PipeBarrier<PIPE_ALL>();
    if (block > n) {  // re-zero pad lanes the ArithProgression may have overrun
      AscendC::LocalTensor<uint32_t> idxU = idxBuf.Get<uint32_t>();
      for (uint32_t i = 0; i < segs; i++)
        for (uint32_t k = n; k < block; k++) idxU.SetValue(i * block + k, zeroOff * esz);
      AscendC::PipeBarrier<PIPE_ALL>();
    }
  }

  // Lean fp32 gather table: zero-slot prefill plus one progression per row, with no scalar pad
  // fix-up and no pipe barrier ahead of the first read.  A progression rounds its count up to
  // 32B and spills into the row's first pad lanes, so only consumers that never use pad lanes
  // may take it: column folds drop the pad columns, masked whole reduces skip them.
  __aicore__ inline void BuildRowGatherLean(uint32_t segs, uint32_t n, uint32_t block,
                                            uint32_t zeroOff) {
    AscendC::LocalTensor<int32_t> idx =
        idxBuf.Get<uint32_t>().template ReinterpretCast<int32_t>();
    AscendC::Duplicate(idx, (int32_t)(zeroOff * 4u), segs * block);
    for (uint32_t i = 0; i < segs; i++)
      AscendC::ArithProgression<int32_t>(idx[i * block], (int32_t)(i * n * 4u), 4, (int32_t)n);
  }

  __aicore__ inline void ProcessRow() {
    if (oCount == 0) return;
    uint32_t n = reduceLen;
    if (rowBoundary_) RowBoundaryBatch(n);
    else if (n == 8) RowTier(n, 8, 512);     // exact: BlockReduceSum
    else if (n == 64) RowTier(n, 64, 64);    // exact: WholeReduceSum
    else if (n == 512) RowTier(n, 512, 8);   // exact: Whole+BlockReduceSum
    else if (n <= 512 && n * (uint32_t)sizeof(typeT) < 256) {  // tiny -> gather-to-block
      // ...unless the AlignUp pad waste is small (<=25%): then RowPadBatch is leaner (no
      // gather index table, no Gather op, no scalar segs*block SetValue loop). n=31 fp16:
      // AlignUp(31,16)=32, pad=1 (3%) -> RowPadBatch kills the 4096-store scalar critical
      // path on the launch/scalar-bound tiny case (tiny workload is scalar 0.94).
      if (AlignUp(n, ALN) - n <= n / 4) RowPadBatch(n);
      else if (n < 8) RowTier(n, 8, 512);
      else if (n < 64) RowTier(n, 64, 64);
      else RowTier(n, 512, 8);
    } else if (n <= tileElems) {
      RowPadBatch(n);                        // medium non-exact: rightPad batch + ReduceSum<AR>
    } else {
      RowBig(n);                             // streaming per row
    }
  }

  // A phase-cycle batch of bf16/fp16 rows, R=8q+d where d divides 8:
  // aligned interiors are packed with stride Q=R-d; the removed head/tail
  // values fit in one 16-lane boundary slot per row.  One contiguous read then
  // replaces the per-row padded transactions while Gather touches only the
  // small boundary table.
  __aicore__ inline void BuildRowBoundaryIdx(uint32_t R) {
    uint32_t ROWS = rowBoundaryRows_;
    constexpr uint32_t BOUNDARY_SLOT = 16u;
    uint32_t OUT_IDX_BASE = ROWS * BOUNDARY_SLOT;
    uint32_t D = F32_BLK / ROWS;
    uint32_t Q = R - D;
    uint32_t zeroOff = ROWS * R;
    AscendC::LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    for (uint32_t i = 0; i < OUT_IDX_BASE; ++i) {
      idx.SetValue(i, zeroOff * (uint32_t)sizeof(float));
    }
    // Give every row the SAME interior window [r*Q, (r+1)*Q) so the whole batch
    // reduces in one ReduceSum call (the split 1 + (rb-1) form costs an extra
    // library call per batch and cannot use the basic-block path).  Window r then
    // over-counts row r-1's trailing r*D values -- removed by zeroing the window's
    // leading mixed block -- and under-counts row r at both ends:
    //   head  F32_BLK - r*D values at [r*Q + r*D, r*Q + F32_BLK)  (zeroed with it)
    //   tail  (r+1)*D values at [(r+1)*Q, (r+1)*Q + (r+1)*D)      (next window's block)
    // That is F32_BLK + D values per row, which is why BOUNDARY_SLOT is 16.
    for (uint32_t r = 0; r < ROWS; ++r) {
      uint32_t slot = r * BOUNDARY_SLOT;
      uint32_t p = 0;
      if (r != 0u) {
        for (uint32_t e = r * Q + r * D; e < r * Q + F32_BLK; ++e) {
          idx.SetValue(slot + p++, e * (uint32_t)sizeof(float));
        }
      }
      for (uint32_t e = (r + 1u) * Q; e < (r + 1u) * Q + (r + 1u) * D; ++e) {
        idx.SetValue(slot + p++, e * (uint32_t)sizeof(float));
      }
    }
    // One merged ReduceSum packs the interior sums contiguously at acc[0..ROWS).
    for (uint32_t r = 0; r < ROWS; ++r) {
      idx.SetValue(OUT_IDX_BASE + r, r * (uint32_t)sizeof(float));
    }
  }

  // BuildRowBoundaryIdx with the gather's zero slot at element offset zeroOff from the rows' start.
  // Each row takes two 16-lane slots: its head values (the 32B block at row*Q) in the first and its tail values
  // (the block at (row+1)*Q) in the second.  When Q is a multiple of 128 floats both blocks sit in the same UB
  // bank group, and a gather list with such addresses next to each other takes the bank-conflict path.
  // Unused lanes read the zero slot, filled with one vector call rather than per-entry scalar writes.
  __aicore__ inline void BuildRowBoundaryIdxAt(uint32_t R, uint32_t zeroOff, uint32_t Q) {
    uint32_t ROWS = rowBoundaryRows_;
    constexpr uint32_t BOUNDARY_SLOT = 16u;
    uint32_t D = R - Q;
    AscendC::LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    AscendC::LocalTensor<int32_t> idxI = idx.template ReinterpretCast<int32_t>();
    AscendC::Duplicate(idxI, (int32_t)(zeroOff * (uint32_t)sizeof(float)), (int32_t)(ROWS * 2u * BOUNDARY_SLOT));
    for (uint32_t r = 0; r < ROWS; ++r) {
      uint32_t slot = r * 2u * BOUNDARY_SLOT;
      uint32_t p = 0;
      if (r != 0u) {
        for (uint32_t e = r * Q + r * D; e < r * Q + F32_BLK; ++e) {
          idx.SetValue(slot + p++, e * (uint32_t)sizeof(float));
        }
      }
      p = BOUNDARY_SLOT;
      for (uint32_t e = (r + 1u) * Q; e < (r + 1u) * Q + (r + 1u) * D; ++e) {
        idx.SetValue(slot + p++, e * (uint32_t)sizeof(float));
      }
    }
  }

  // One phase cycle of rb rows whose fp32 squares start fl lanes into sq0: the boundary gather, the
  // reduce of the aligned interior windows, and both sums added into out[0, rb).
  // Q (the interior window) and whether it reduces in 64-element tiers are per-core constants from the caller.
  __aicore__ inline void RowBoundaryReduce(const AscendC::LocalTensor<float>& sq0, uint32_t fl, uint32_t rb,
                                           uint32_t Q, bool tiered, const AscendC::LocalTensor<float>& out) {
    uint32_t ROWS = rowBoundaryRows_;
    constexpr uint32_t BOUNDARY_SLOT = 32u;  // head slot + tail slot per row (BuildRowBoundaryIdxAt)
    AscendC::LocalTensor<float> sq = sq0[fl];
    AscendC::LocalTensor<float> gathered = gthBuf.Get<float>();
    AscendC::LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    AscendC::Gather(gathered, sq0, idx, fl * 4u, rb * BOUNDARY_SLOT);
    if (rb > 1u && (Q / F32_BLK) <= 255u) {
      AscendC::Duplicate<float>(sq[Q], 0.0f, (uint64_t)F32_BLK, (uint8_t)(rb - 1u), 1u,
                                (uint8_t)(Q / F32_BLK));
    } else {
      for (uint32_t r = 1; r < rb; ++r) {
        AscendC::Duplicate(sq[r * Q], 0.0f, F32_BLK);
      }
    }
    AscendC::LocalTensor<float> acc = accBuf.Get<float>();
    uint32_t intShape[2] = {rb, Q};
    uint32_t sub = Q / 64u;
    if (tiered && rb * sub <= 255u) {
      uint32_t o1 = AlignUp(ROWS * BOUNDARY_SLOT, F32_BLK);
      uint32_t n1 = rb * (Q / F32_BLK);
      AscendC::LocalTensor<float> t1 = gthBuf.Get<float>()[o1];
      AscendC::LocalTensor<float> t2 = gthBuf.Get<float>()[o1 + n1];
      AscendC::BlockReduceSum<float>(t1, sq, (int32_t)(rb * sub), 64, 1, 1, 8);
      AscendC::BlockReduceSum<float>(t2, t1, (int32_t)(n1 / 64u), 64, 1, 1, 8);
      AscendC::WholeReduceSum<float>(acc, t2, (int32_t)sub, (int32_t)rb, 1, 1,
                                     (int32_t)(sub / F32_BLK));
    } else if (rb == ROWS && (ROWS % 8u) == 0u && (Q % 64u) == 0u) {
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(acc, sq, intShape, true);
    } else {
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(acc, sq, intShape, false);
    }
    uint32_t boundaryShape[2] = {rb, BOUNDARY_SLOT};
    if (rb == ROWS) {
      AscendC::WholeReduceSum<float>(acc[16], gathered, (int32_t)BOUNDARY_SLOT, rb, 1, 1,
                                     (int32_t)(BOUNDARY_SLOT / F32_BLK));
    } else {
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(acc[16], gathered, boundaryShape,
                                                                     false);
    }
    AscendC::Add(out, acc, acc[16], rb);
  }

  // bf16 phase cycles read whole 32B frames (a read whose GM start or end is off a 32B boundary is
  // slower).  A share starts on a multiple of eight rows, so a frame starts 0 or 8 elements before its
  // first row and runs on to the next boundary.  `cycles` phase cycles share one read; each is squared
  // from its own frame boundary and reduced through its lead, and the boundary gather's zero slot sits
  // past the squares of either lead.  A frame that would start off an fp32 block or run past the
  // input end falls back to reading exactly the rows.
  __aicore__ inline void RowBoundaryAligned(uint32_t R, uint32_t cycles) {
    uint32_t ROWS = rowBoundaryRows_;
    uint32_t zA = AlignUp(ROWS * R, ALN);
    const uint32_t Q = R - F32_BLK / ROWS;               // per-core constants of RowBoundaryReduce
    const bool tiered = (Q % 512u) == 0u && Q <= 1024u;
    BuildRowBoundaryIdxAt(R, zA, Q);
    AscendC::LocalTensor<float> accOut = accBuf.Get<float>()[64];
    AscendC::Duplicate(accOut, 0.0f, AlignUp(oCount, F32_BLK));
    AscendC::LocalTensor<float> sq0 = sqBuf.Get<float>();
    AscendC::Duplicate(sq0[zA], 0.0f, 2u * F32_BLK);
    uint64_t totalElems = (uint64_t)outer * R;
    uint32_t done = 0;
    while (done < oCount) {
      uint32_t rr = MinU(oCount - done, cycles * ROWS);
      uint64_t s = (uint64_t)(oStart + done) * R;
      uint32_t lead = (uint32_t)(s & (uint64_t)(ALN - 1u));
      uint32_t readN = AlignUp(lead + rr * R, ALN);
      AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
      if ((lead % F32_BLK) == 0u && s - lead + readN <= totalElems) {
        AscendC::DataCopy(xin, x_gm[s - lead], readN);
      } else {
        lead = 0u;
        readN = rr * R;
        PadIn(xin, x_gm[s], readN);
      }
      inQue.EnQue(xin);
      xin = inQue.template DeQue<typeT>();
      for (uint32_t r0 = 0; r0 < rr; r0 += ROWS) {
        uint32_t rb = MinU(rr - r0, ROWS);
        uint32_t e0 = lead + r0 * R;
        uint32_t fl = e0 & (ALN - 1u);
        uint32_t cnt = MinU(AlignUp(fl + rb * R, ALN), readN - (e0 - fl));
        SquareToF32(sq0, xin[e0 - fl], cnt);
        RowBoundaryReduce(sq0, fl, rb, Q, tiered, accOut[done + r0]);
      }
      inQue.FreeTensor(xin);
      done += rr;
    }
    AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
    CastF32To(yout, accOut, oCount);
    outQue.template EnQue<typeT>(yout);
    yout = outQue.template DeQue<typeT>();
    PadOut(y_gm[oStart], yout, oCount);
    outQue.FreeTensor(yout);
  }

  __aicore__ inline void RowBoundaryBatch(uint32_t R) {
    if constexpr (sizeof(typeT) == 2 && !std::is_same_v<typeT, half>) {
      RowBoundaryAligned(R, 1u);
      return;
    }
    uint32_t ROWS = rowBoundaryRows_;
    constexpr uint32_t BOUNDARY_SLOT = 16u;
    uint32_t OUT_IDX_BASE = ROWS * BOUNDARY_SLOT;
    uint32_t D = F32_BLK / ROWS;
    uint32_t Q = R - D;
    uint32_t zeroOff = ROWS * R;
    BuildRowBoundaryIdx(R);

    // The sentinel block is outside every valid full/partial input burst and is
    // invariant across batches.  Keep it initialized once, then accumulate each
    // batch directly into an output-sized fp32 scratch.  This removes the old
    // per-batch cast/queue/short-write epilogue while preserving the row order
    // (each output row belongs to exactly one batch).
    AscendC::LocalTensor<float> accOut = accBuf.Get<float>()[64];
    AscendC::Duplicate(accOut, 0.0f, AlignUp(oCount, F32_BLK));
    AscendC::LocalTensor<float> sqInit = sqBuf.Get<float>();
    AscendC::Duplicate(sqInit[zeroOff], 0.0f, F32_BLK);

    uint32_t done = 0;
    while (done < oCount) {
      uint32_t rb = MinU(oCount - done, ROWS);
      uint32_t flat = rb * R;
      AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
      PadIn(xin, x_gm[(uint64_t)(oStart + done) * R], flat);
      inQue.EnQue(xin);
      xin = inQue.template DeQue<typeT>();

      AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
      SquareToF32(sq, xin, flat);
      inQue.FreeTensor(xin);

      AscendC::LocalTensor<float> gathered = gthBuf.Get<float>();
      AscendC::LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
      // The final batch may contain fewer than a full phase cycle.  Limit the
      // gather count to rb so unused-row indices (which point at the zero
      // sentinel only in the full table) are never dereferenced.
      AscendC::Gather(gathered, sq, idx, (uint32_t)0, rb * BOUNDARY_SLOT);

      // Remove the one mixed aligned block at the HEAD of windows 1..rb-1.  Its
      // valid elements (row r-1's tail and row r's head) were captured by the
      // boundary Gather above.  Q is a multiple of eight by construction; use one
      // repeated Duplicate when its datablock stride fits the API's uint8 stride
      // field, retaining the scalar fallback for a wider pitch.
      if (rb > 1u && (Q / F32_BLK) <= 255u) {
        AscendC::Duplicate<float>(sq[Q], 0.0f, (uint64_t)F32_BLK,
                                  (uint8_t)(rb - 1u), 1u,
                                  (uint8_t)(Q / F32_BLK));
      } else {
        for (uint32_t r = 1; r < rb; ++r) {
          AscendC::Duplicate(sq[r * Q], 0.0f, F32_BLK);
        }
      }

      AscendC::LocalTensor<float> acc = accBuf.Get<float>();
      uint32_t intShape[2] = {rb, Q};
      uint32_t sub = Q / 64u;  // 64-float windows per interior row
      if ((Q % 512u) == 0u && Q <= 1024u && rb * sub <= 255u) {
        // BlockReduceSum cascade instead of ReduceSum<AR>: 8 floats -> 1 over the contiguous
        // [rb, Q] interior, again over those block sums, then one WholeReduceSum of each row's
        // remaining Q/64 sums (C3: vec 31.6 -> 24.3 us, no library call per batch).
        uint32_t o1 = AlignUp(ROWS * BOUNDARY_SLOT, F32_BLK);
        uint32_t n1 = rb * (Q / F32_BLK);
        AscendC::LocalTensor<float> t1 = gthBuf.Get<float>()[o1];
        AscendC::LocalTensor<float> t2 = gthBuf.Get<float>()[o1 + n1];
        AscendC::BlockReduceSum<float>(t1, sq, (int32_t)(rb * sub), 64, 1, 1, 8);
        AscendC::BlockReduceSum<float>(t2, t1, (int32_t)(n1 / 64u), 64, 1, 1, 8);
        AscendC::WholeReduceSum<float>(acc, t2, (int32_t)sub, (int32_t)rb, 1, 1,
                                       (int32_t)(sub / F32_BLK));
      } else if (rb == ROWS && (ROWS % 8u) == 0u && (Q % 64u) == 0u) {
        // isBasicBlock needs a whole batch whose row count is a multiple of eight
        // and a 64-float row; a short trailing batch takes the generic form.
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            acc, sq, intShape, true);
      } else {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            acc, sq, intShape, false);
      }

      uint32_t boundaryShape[2] = {rb, BOUNDARY_SLOT};
      if (rb == ROWS) {
        // Boundary rows are exactly 16 fp32 lanes (two datablocks).  The
        // fixed-width whole reduction avoids the generic AR descriptor and
        // its per-call shape handling on C3.
        AscendC::WholeReduceSum<float>(acc[16], gathered, 16, rb, 1, 1, 2);
      } else {
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            acc[16], gathered, boundaryShape, false);
      }

      // The merged reduction already packs the interior sums at acc[0..rb), so
      // the old repack Gather is gone; acc[16..16+rb) holds the boundary sums.
      AscendC::Add(accOut[done], acc, acc[16], rb);
      done += rb;
    }

    AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
    CastF32To(yout, accOut, oCount);
    outQue.template EnQue<typeT>(yout);
    yout = outQue.template DeQue<typeT>();
    PadOut(y_gm[oStart], yout, oCount);
    outQue.FreeTensor(yout);
  }

  // non-exact n<=tileElems: batch G rows, rightPad each to RpT=AlignUp(n,ALN) so the
  // batched ReduceSum<AR> rows stay 32B-aligned. Leaner than gather-to-block (one big
  // strided DataCopyPad + one ReduceSum vs gather op + 2-stage reduce). [opt_v7 path]
  __aicore__ inline void RowPadBatch(uint32_t R) {
    uint32_t RpT = AlignUp(R, ALN);
    uint32_t G = tileElems / RpT;
    if (G < 1) G = 1;
    uint8_t rightPad = (uint8_t)(RpT - R);
    uint32_t done = 0;
    while (done < oCount) {
      uint32_t rb = MinU(oCount - done, G);
      AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
      AscendC::DataCopyExtParams p{(uint16_t)rb, (uint32_t)(R * sizeof(typeT)), 0, 0, 0};
      AscendC::DataCopyPadExtParams<typeT> pad{true, 0, rightPad, (typeT)0};
      AscendC::DataCopyPad(xin, x_gm[(uint64_t)(oStart + done) * R], p, pad);
      inQue.EnQue(xin); xin = inQue.template DeQue<typeT>();
      AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
      SquareToF32(sq, xin, rb * RpT);
      AscendC::LocalTensor<float> outF = accBuf.Get<float>();
      uint32_t shape[2] = {rb, RpT};
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(outF, sq, shape, false);
      AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
      CastF32To(yout, outF, rb);
      outQue.template EnQue<typeT>(yout); yout = outQue.template DeQue<typeT>();
      PadOut(y_gm[oStart + done], yout, rb);
      outQue.FreeTensor(yout);
      inQue.FreeTensor(xin);
      done += rb;
    }
  }

  // n<=block(8/64/512); batch `segs` rows; gather-align when n<block; reduce per block.
  // Always process a FULL `segs` batch (zero-padded tail) so the block/whole reduces
  // consume whole repeats; store only the `rb` valid results.
  __aicore__ inline void RowTier(uint32_t n, uint32_t block, uint32_t segs) {
    bool exact = (n == block);
    uint32_t full = segs * block;
    if (!exact) BuildRowGather(segs, n, block, segs * n);  // zero slot index = segs*n
    uint32_t done = 0;
    while (done < oCount) {
      uint32_t rb = MinU(oCount - done, segs);
      AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
      PadIn(xin, x_gm[(uint64_t)(oStart + done) * n], rb * n);
      inQue.EnQue(xin);
      xin = inQue.template DeQue<typeT>();
      AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
      AscendC::LocalTensor<float> outF = accBuf.Get<float>();
      if (exact) {
        if (rb < segs) AscendC::Duplicate(sq, 0.0f, full);
        SquareToF32(sq, xin, rb * n);  // tight [rb,block]; tail rows already 0
        ReduceBlocks(outF, sq, segs, block);
      } else {
        AscendC::LocalTensor<float> raw = gthBuf.Get<float>();
        AscendC::Duplicate(raw, 0.0f, full);  // pad segs + zero slot all 0
        RawToF32(raw, xin, rb * n);           // valid [0:rb*n]
        AscendC::LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
        AscendC::Gather(sq, raw, idx, (uint32_t)0, full);  // [segs,block]
        AscendC::Mul(sq, sq, sq, full);
        ReduceBlocks(outF, sq, segs, block);
      }
      AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
      CastF32To(yout, outF, rb);
      outQue.template EnQue<typeT>(yout);
      yout = outQue.template DeQue<typeT>();
      PadOut(y_gm[oStart + done], yout, rb);
      outQue.FreeTensor(yout);
      inQue.FreeTensor(xin);
      done += rb;
    }
  }

  // reduce `rb` blocks of `block` fp32 in sq -> outF[rb] (one sum per block)
  __aicore__ inline void ReduceBlocks(const AscendC::LocalTensor<float>& outF,
                                      const AscendC::LocalTensor<float>& sq,
                                      uint32_t rb, uint32_t block) {
    if (block == 8) {
      // BlockReduceSum: each 8-fp32 datablock -> 1.  mask=64 (8 blocks/repeat).
      uint32_t reps = (rb + 7) / 8;
      AscendC::BlockReduceSum<float>(outF, sq, reps, 64, 1, 1, 8);
    } else if (block == 64) {
      // WholeReduceSum: each 64-fp32 repeat -> 1.
      AscendC::WholeReduceSum<float>(outF, sq, 64, rb, 1, 1, 8);
    } else {  // block==512: Whole(64-> [rb*8]) then Block(8-> [rb])
      AscendC::LocalTensor<float> tmp = gthBuf.Get<float>();
      AscendC::WholeReduceSum<float>(tmp, sq, 64, rb * 8, 1, 1, 8);  // rb*8 sub-sums
      uint32_t reps = (rb + 7) / 8;
      AscendC::BlockReduceSum<float>(outF, tmp, reps, 64, 1, 1, 8);  // 8 sub -> 1
    }
  }

  // n>4096: per row, stream 4096-chunks into chunk-sums, final ReduceSum -> 1.
  __aicore__ inline void RowBig(uint32_t n) {
    for (uint32_t i = 0; i < oCount; i++) {
      AscendC::LocalTensor<float> acc = accBuf.Get<float>();
      AscendC::Duplicate(acc, 0.0f, F32_BLK);
      uint64_t base = (uint64_t)(oStart + i) * n;
      uint32_t off = 0;
      while (off < n) {
        uint32_t c = MinU(n - off, tileElems);
        uint32_t cPad = Align64(c);
        AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
        PadIn(xin, x_gm[base + off], c);
        inQue.EnQue(xin);
        xin = inQue.template DeQue<typeT>();
        AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
        AscendC::Duplicate(sq, 0.0f, cPad);
        SquareToF32(sq, xin, c);
        AscendC::LocalTensor<float> t1 = gthBuf.Get<float>();
        uint32_t shape[2] = {1, cPad};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(t1, sq, shape, false);
        AscendC::Add(acc, acc, t1, 1);
        inQue.FreeTensor(xin);
        off += c;
      }
      AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
      CastF32To(yout, acc, 1);
      outQue.template EnQue<typeT>(yout);
      yout = outQue.template DeQue<typeT>();
      PadOut(y_gm[oStart + i], yout, 1);
      outQue.FreeTensor(yout);
    }
  }

  // ============================ COLUMN (inner>1) ============================
  // (l,m,n) reduce m. workspace[l*n] fp32 (or output gm if fp32). split (l,m-chunk)
  // over cores, accumulate dst[n] (binary fold), atomic-add to ws[l*n]; sync; cast out.
  __aicore__ inline void ProcessCol() {
    constexpr bool F32 = std::is_same_v<typeT, float>;
    uint32_t L = outer, M = reduceLen, N = inner;
    uint32_t bidx = AscendC::GetBlockIdx();
    uint64_t wsTotal = (uint64_t)L * N;
    uint32_t nPad = AlignUp(N, ALN);
    // ntCap = largest N-slice that fits dst(accBuf=tileElems floats) AND the fp32 write
    // staging (outQue holds tileElems*sizeof(typeT)/4 floats): fp32 8192, fp16/bf16 4096.
    uint32_t ntCap = F32 ? tileElems : (tileElems / 2);
    uint32_t numNtiles = (N <= ntCap) ? 1u : ((N + ntCap - 1) / ntCap);
    uint32_t chunksPerL = (M + mPerCore - 1) / mPerCore;
    // chunksPerL==1 => each work-item folds ALL M rows for its (l) -> dst is the COMPLETE
    // column sum -> cast & write y_gm DIRECTLY (no fp32 workspace, no atomic, no pre-zero,
    // no SyncAll, no cast-roundtrip).  chunksPerL>1 (m-split) -> partial sums -> atomic-add
    // into a pre-zeroed fp32 target (workspace for fp16/bf16, y_gm for fp32) + final cast.
    bool direct = (chunksPerL == 1);
    if (!direct) {  // pre-zero the atomic target, distributed over cores, then barrier
      uint64_t per = (wsTotal + colCores - 1) / colCores;
      uint64_t zs = (uint64_t)bidx * per, ze = MinU2(zs + per, wsTotal);
      for (uint64_t z = zs; z < ze;) {
        uint32_t c = (uint32_t)MinU2(ze - z, fltChunk);
        AscendC::LocalTensor<float> zb = accBuf.Get<float>();
        AscendC::Duplicate(zb, 0.0f, c);
        AscendC::LocalTensor<float> zo = outQue.template AllocTensor<float>();
        AscendC::Adds(zo, zb, 0.0f, c);
        outQue.template EnQue<float>(zo);
        zo = outQue.template DeQue<float>();
        AscendC::DataCopyExtParams wp{1, (uint32_t)(c * sizeof(float)), 0, 0, 0};
        if constexpr (F32) AscendC::DataCopyPad(y_gm[z], zo, wp);
        else AscendC::DataCopyPad(usrWs[z], zo, wp);
        outQue.FreeTensor(zo);
        z += c;
      }
      // The large fp32 stream reads several MB per core before its first atomic add, so its
      // zero stores land first without a cross-core barrier (C5 -3.3 us); others keep it.
      if (!f32ColBurst_) AscendC::SyncAll();
    }
    // build the per-n reuse table ONCE (tiny n -> transpose offsets, medium n -> gather idx)
    bool colTiny = (N % ALN != 0) && (N * (uint32_t)sizeof(typeT) < 64);
    bool colGather = (N % ALN != 0) && !colTiny && (nPad <= ROW_TILE);
    if (colTiny) {
      colTransMc = (ROW_TILE / N) / 8 * 8;
      if (colTransMc < 8) colTransMc = 8;
      AscendC::LocalTensor<int32_t> offI =
          idxBuf.Get<uint32_t>().template ReinterpretCast<int32_t>();
      for (uint32_t c = 0; c < N; c++)  // off[c*Mc+r] = (r*N+c)*4  (transpose offsets)
        AscendC::ArithProgression<int32_t>(offI[c * colTransMc], (int32_t)(c * sizeof(float)),
                                           (int32_t)(N * sizeof(float)), (int32_t)colTransMc);
      AscendC::PipeBarrier<PIPE_V>();
    } else if (colGather) {
      colGatherRows = f32ColBurst_ ? 8u : ROW_TILE / nPad;
      if (colGatherRows < 1) colGatherRows = 1;
      // the fold drops the pad columns, so the lean table's spilled pad entries are harmless
      BuildRowGatherLean(colGatherRows, N, nPad, AlignUp(colGatherRows * N, 8u));
    }
    // distribute (work-item = (l, m-chunk)) x (n-tile) units over cores.  For small M and
    // huge N this fills all cores via the N dimension (B_ax0_wide); chunksPerL stays 1 so
    // each unit writes a complete, disjoint slice directly.
    uint64_t totalWi = (uint64_t)L * chunksPerL;
    uint64_t totalUnits = totalWi * numNtiles;
    uint64_t uPer = (totalUnits + colCores - 1) / colCores;
    uint64_t u0 = (uint64_t)bidx * uPer, u1 = MinU2(u0 + uPer, totalUnits);
    for (uint64_t u = u0; u < u1; u++) {
      uint64_t wi = u / numNtiles;
      uint32_t ntile = (uint32_t)(u % numNtiles);
      uint32_t l = (uint32_t)(wi / chunksPerL);
      uint32_t mc = (uint32_t)(wi % chunksPerL);
      uint32_t m0 = mc * mPerCore;
      uint32_t mrows = MinU(mPerCore, M - m0);
      uint64_t base = ((uint64_t)l * M + m0) * N;
      if (numNtiles == 1) {
        AscendC::LocalTensor<float> dst = accBuf.Get<float>();
        AscendC::Duplicate(dst, 0.0f, nPad);
        ColAccum(dst, base, mrows, N, nPad);
        if (direct) DirectWriteOut(dst, l, 0, N); else AtomicAddOut(dst, l, 0, N);
      } else {
        uint32_t n0 = ntile * ntCap;
        uint32_t nt = MinU(ntCap, N - n0);
        uint32_t nPadTile = AlignUp(nt, ALN);
        AscendC::LocalTensor<float> dst = accBuf.Get<float>();
        AscendC::Duplicate(dst, 0.0f, nPadTile);
        ColAccumSlice(dst, base, mrows, N, n0, nt, nPadTile);
        if (direct) DirectWriteOut(dst, l, n0, nt); else AtomicAddOut(dst, l, n0, nt);
      }
    }
    if (!F32 && !direct) {  // direct already cast & wrote y_gm; only the atomic path uses ws
      AscendC::SyncAll();
      // cast workspace -> output gm
      uint64_t per = (wsTotal + colCores - 1) / colCores;
      uint64_t cs = (uint64_t)bidx * per, ce = MinU(cs + per, wsTotal);
      for (uint64_t z = cs; z < ce;) {
        uint32_t c = (uint32_t)MinU(ce - z, fltChunk);
        AscendC::LocalTensor<float> wb = inQue.template AllocTensor<float>();
        AscendC::DataCopyExtParams rp{1, (uint32_t)(c * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> rpad{false, 0, 0, 0};
        AscendC::DataCopyPad(wb, usrWs[z], rp, rpad);
        inQue.template EnQue<float>(wb);
        wb = inQue.template DeQue<float>();
        AscendC::LocalTensor<typeT> yo = outQue.template AllocTensor<typeT>();
        CastF32To(yo, wb, c);
        outQue.template EnQue<typeT>(yo);
        yo = outQue.template DeQue<typeT>();
        PadOut(y_gm[z], yo, c);
        outQue.FreeTensor(yo);
        inQue.FreeTensor(wb);
        z += c;
      }
    }
  }

  // dst[nPad] += sum over `mrows` rows of square(row[N]); rows contiguous, stride N.
  __aicore__ inline void ColAccum(const AscendC::LocalTensor<float>& dst, uint64_t gmBase,
                                  uint32_t mrows, uint32_t N, uint32_t nPad) {
    bool aligned = (N % ALN == 0);
    if (!aligned) {
      if (N * (uint32_t)sizeof(typeT) < 64) {  // tiny: contiguous load + gather-transpose + AR
        ColAccumTranspose(dst, gmBase, mrows, N);
        return;
      }
      if (nPad <= ROW_TILE) {  // non-aligned, row fits ROW_TILE: contiguous load + MulAddDst
        ColAccumGather(dst, gmBase, mrows, N, nPad);  // accumulate + realign once (beats pad-load)
        return;
      }
    }
    uint32_t rowsPer = aligned ? MinU(tileElems / N, mrows) : MinU(tileElems / nPad, mrows);
    if (rowsPer < 1) rowsPer = 1;
    uint32_t done = 0;
    while (done < mrows) {
      uint32_t rb = MinU(mrows - done, rowsPer);
      AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
      AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
      if (aligned) {
        PadIn(xin, x_gm[gmBase + (uint64_t)done * N], rb * N);
        inQue.EnQue(xin); xin = inQue.template DeQue<typeT>();
        SquareToF32(sq, xin, rb * N);
        FoldRowsInto(dst, sq, rb, N);  // tight [rb,N], N aligned
      } else {
        // strided pad-load each row to nPad (32B aligned) then fold
        AscendC::DataCopyExtParams p{(uint16_t)rb, (uint32_t)(N * sizeof(typeT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<typeT> pad{true, 0, (uint8_t)(nPad - N), (typeT)0};
        AscendC::DataCopyPad(xin, x_gm[gmBase + (uint64_t)done * N], p, pad);
        inQue.EnQue(xin); xin = inQue.template DeQue<typeT>();
        if (rb < rowsPer) AscendC::Duplicate(sq, 0.0f, rowsPer * nPad);
        SquareToF32(sq, xin, rb * nPad);
        FoldRowsInto(dst, sq, rowsPer, nPad);  // includes zeroed tail rows
      }
      inQue.FreeTensor(xin);
      done += rb;
    }
  }
  // column tiny non-aligned n: load Mc rows CONTIGUOUS (full HBM bw), square, Gather
  // UB-transpose [Mc,n]->[n,Mc], ReduceSum<AR> reduces Mc (wide) -> [n], add to dst.
  // Beats gather-fold for very small n (fold's nPad-wide vector is underutilized).
  // off table (idxBuf) built once in ProcessCol: off[c*Mc+r]=(r*n+c)*4.
  __aicore__ inline void ColAccumTranspose(const AscendC::LocalTensor<float>& dst,
                                           uint64_t base, uint32_t mrows, uint32_t n) {
    uint32_t Mc = colTransMc;
    uint32_t cap = n * Mc;
    AscendC::LocalTensor<uint32_t> off = idxBuf.Get<uint32_t>();
    uint32_t done = 0;
    while (done < mrows) {
      uint32_t m = MinU(mrows - done, Mc);
      AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
      PadIn(xin, x_gm[base + (uint64_t)done * n], m * n);
      inQue.EnQue(xin); xin = inQue.template DeQue<typeT>();
      AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
      AscendC::Duplicate(sq, 0.0f, Mc * n);   // zero whole (incl pad rows), 32B-aligned at 0
      SquareToF32(sq, xin, m * n);            // square valid rows; pad rows stay 0
      AscendC::LocalTensor<float> trans = gthBuf.Get<float>();
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Gather(trans, sq, off, (uint32_t)0, cap);  // [n, Mc]
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::LocalTensor<float> tmp = sqBuf.Get<float>();  // reuse sqBuf (sq consumed)
      uint32_t shape[2] = {n, Mc};
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(tmp, trans, shape, false);
      AscendC::Add(dst, dst, tmp, n);
      inQue.FreeTensor(xin);
      done += m;
    }
  }

  // plan column case-1 (n<512 non-aligned): load colGatherRows rows CONTIGUOUS (tight, no
  // per-row 32B gaps) and accumulate their squares into a PACKED [colGR,N] fp32 lane
  // accumulator (lane r sums rows r, r+colGR, ...) via the fused MulAddDst (acc += x*x,
  // one vec pass replacing the old square+Mul). The 255->256 Gather-realign + fold run
  // ONCE per column at the end instead of per tile -> hot loop drops from 4 vec passes
  // {Cast,Mul,Gather,fold} to 2 {Cast,MulAddDst}, flipping the case from vec- to read-bound.
  // The gather index was built ONCE in ProcessCol (idx[i*nPad+k]=(k<N)?i*N+k:colGR*N).
  // accP lives at sqBuf[+64 fp32] so it is 256B phase-staggered vs xf(gthBuf): the two
  // MulAddDst read operands then land in different UB banks (no read-bank conflict).
  __aicore__ inline void ColAccumGather(const AscendC::LocalTensor<float>& dst,
                                        uint64_t gmBase, uint32_t mrows,
                                        uint32_t N, uint32_t nPad) {
    uint32_t colGR = colGatherRows;
    uint32_t zSlot = AlignUp(colGR * N, 8u);  // 32B-aligned gather zero-slot (idx built w/ same)
    AscendC::LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    AscendC::LocalTensor<float> accP = sqBuf.Get<float>()[64];  // packed lane accumulator (phase 256)
    AscendC::Duplicate(accP, 0.0f, zSlot + ALN);  // zero accumulator + gather zero-slot ONCE
    // Burst-decouple: load burstMul*colGR rows in ONE DataCopyPad (bigger contiguous HBM
    // burst amortizes descriptor latency), then run burstMul MulAddDst sub-passes of colGR
    // rows each into the SAME
    // packed lane accumulator.  Lane r still accumulates rows r, r+colGR, r+2*colGR, ... in
    // the identical order as the old colGR-tile loop -> BIT-EXACT; only DMA granularity changes.
    uint32_t burstMul = colBurstMul_;
    if (sizeof(typeT) == 2 && burstMul == 1u) burstMul = 2u;
    // The sub-pass loop below indexes xin at p*N with p a multiple of colGR, and
    // a UB address handed to a vector or DMA instruction has to be 32B aligned.
    // N is the raw row length here, so colGR*N*sizeof(typeT) is not always a
    // multiple of 32 -- and bursting is exactly what creates a second sub-pass
    // (p>0). Where the product is unaligned, keep one sub-pass per load; the
    // device reports "The UB address accessed by the VEC instruction is not
    // aligned" otherwise. fp32 leaves burstMul at 1 already, which is why only
    // the 2-byte dtypes ever faulted.
    if ((((uint64_t)colGR * N * sizeof(typeT)) & 31ULL) != 0ULL) burstMul = 1u;
    uint32_t bigRows = colGR * burstMul;
    uint32_t cyc = colGR * N;  // accumulator lanes, elements
    // bf16 only (C2): keeping this path out of the half and float kernels leaves their code as it was
    if constexpr (!std::is_same_v<typeT, half> && !std::is_same_v<typeT, float>) {
      if ((cyc & 7u) == 0u && (((uint64_t)cyc * sizeof(typeT)) & 31ULL) == 0ULL) {
        // Aligned reads (a read whose GM start or end is off a 32B boundary is slower: 3% for 2-byte
        // elements, 18% for fp32).  The frame starts at the 32B boundary at or below the first
        // element, so every burst starts aligned.  Frame offset f accumulates into lane f mod cyc:
        // the `lead` elements before gmBase are zeroed after the first sub-pass, the last `lead`
        // elements of every lane cycle wrap to lanes [0, lead) and are copied back past the cycle,
        // and the realign gather reads the lanes shifted by `lead`.
        uint32_t es = (uint32_t)sizeof(typeT);
        uint32_t lead = (uint32_t)((gmBase * es) & 31ULL) / es;
        uint64_t totalElems = (uint64_t)outer * reduceLen * inner;
        uint64_t endElem = gmBase + (uint64_t)mrows * N;
        uint64_t frame = gmBase - lead;
        uint64_t span = (uint64_t)mrows * N + lead;
        uint64_t B = (uint64_t)bigRows * N;
        for (uint64_t off = 0; off < span; off += B) {
          uint32_t want = (uint32_t)MinU2(B, span - off);
          // past the range the read may run on to the next 32B boundary, except at the input end
          uint32_t readN = (off + want == span && endElem < totalElems) ? AlignUp(want, 32u / es) : want;
          AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
          if (((readN * es) & 31u) == 0u) {
            AscendC::DataCopy(xin, x_gm[frame + off], readN);
          } else {
            PadIn(xin, x_gm[frame + off], readN);
          }
          inQue.EnQue(xin);
          xin = inQue.template DeQue<typeT>();
          AscendC::LocalTensor<float> xf = gthBuf.Get<float>();
          for (uint32_t p = 0; p < want; p += cyc) {
            uint32_t cnt = MinU(cyc, want - p);
            if constexpr (std::is_same_v<typeT, float>) {
              AscendC::MulAddDst(accP, xin[p], xin[p], (int32_t)cnt);
            } else {
              RawToF32(xf, xin[p], cnt);
              AscendC::MulAddDst(accP, xf, xf, (int32_t)cnt);
            }
            if (off == 0u && p == 0u && lead != 0u) AscendC::Duplicate(accP, 0.0f, (int32_t)lead);
          }
          inQue.FreeTensor(xin);
        }
        if (lead != 0u) AscendC::Adds(accP[cyc], accP, 0.0f, (int32_t)lead);
        AscendC::LocalTensor<float> g = gthBuf.Get<float>();
        AscendC::Gather(g, accP, idx, lead * 4u, colGR * nPad);
        FoldRowsInto(dst, g, colGR, nPad);
        return;
      }
    }
    uint32_t done = 0;
    while (done < mrows) {
      uint32_t rb = MinU(mrows - done, bigRows);
      AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
      // Full phase-group bursts are 32B aligned at both ends on the large
      // fp32 stream.  Use the native contiguous DMA there; retain PadIn for
      // the final short/unaligned burst so no tail bytes are read.
      uint64_t srcElem = gmBase + (uint64_t)done * N;
      uint32_t burstBytes = rb * N * (uint32_t)sizeof(typeT);
      if ((burstBytes & 31u) == 0u && ((srcElem * sizeof(typeT)) & 31u) == 0u) {
        AscendC::DataCopy(xin, x_gm[srcElem], rb * N);
      } else {
        PadIn(xin, x_gm[srcElem], rb * N);
      }
      inQue.EnQue(xin); xin = inQue.template DeQue<typeT>();
      AscendC::LocalTensor<float> xf = gthBuf.Get<float>();
      uint32_t p = 0;
      while (p < rb) {                                   // sub-passes into the 16-lane accP
        uint32_t sub = MinU(rb - p, colGR);
        if constexpr (std::is_same_v<typeT, float>) {
          // fp32 is already in the arithmetic format; avoid the historical
          // Adds(x, 0) UB copy before the fused square-and-accumulate.
          AscendC::MulAddDst(accP, xin[p * N], xin[p * N], (int32_t)(sub * N));
        } else {
          RawToF32(xf, xin[p * N], sub * N);             // cast bf16/fp16 -> fp32
          AscendC::MulAddDst(accP, xf, xf, (int32_t)(sub * N));
        }
        p += sub;
      }
      inQue.FreeTensor(xin);
      done += rb;
    }
    AscendC::LocalTensor<float> g = gthBuf.Get<float>();  // reuse gthBuf (xf consumed)
    AscendC::Gather(g, accP, idx, (uint32_t)0, colGR * nPad);  // realign -> [colGR, nPad] ONCE
    FoldRowsInto(dst, g, colGR, nPad);                        // sum colGR lanes -> dst ONCE
  }
  // atomic-add dst[0:len] (fp32) into the accumulation target at row l, column offset n0.
  __aicore__ inline void AtomicAddOut(const AscendC::LocalTensor<float>& dst, uint32_t l,
                                      uint32_t n0, uint32_t len) {
    constexpr bool F32 = std::is_same_v<typeT, float>;
    AscendC::SetAtomicAdd<float>();
    AscendC::LocalTensor<float> po = outQue.template AllocTensor<float>();
    AscendC::Adds(po, dst, 0.0f, AlignUp(len, F32_BLK));  // whole fp32 lanes; only len elements are stored
    outQue.template EnQue<float>(po);
    po = outQue.template DeQue<float>();
    AscendC::DataCopyExtParams wp{1, (uint32_t)(len * sizeof(float)), 0, 0, 0};
    uint64_t off = (uint64_t)l * inner + n0;
    if constexpr (F32) AscendC::DataCopyPad(y_gm[off], po, wp);
    else AscendC::DataCopyPad(usrWs[off], po, wp);
    outQue.FreeTensor(po);
    AscendC::SetAtomicNone();
  }

  // dst[0:len] is a COMPLETE column sum (fp32) -> cast to the output dtype and write y_gm
  // DIRECTLY (no fp32 workspace, no atomic, no SyncAll, no cast-roundtrip). Used when each
  // (l, n-tile) is owned by exactly one core (chunksPerL==1).
  __aicore__ inline void DirectWriteOut(const AscendC::LocalTensor<float>& dst, uint32_t l,
                                        uint32_t n0, uint32_t len) {
    uint64_t off = (uint64_t)l * inner + n0;
    AscendC::LocalTensor<typeT> po = outQue.template AllocTensor<typeT>();
    CastF32To(po, dst, AlignUp(len, ALN));  // whole lanes (no masked tail); only len elements are stored
    outQue.template EnQue<typeT>(po);
    po = outQue.template DeQue<typeT>();
    PadOut(y_gm[off], po, len);
    outQue.FreeTensor(po);
  }

  // large-N column: dst[nt] += sum over `mrows` rows of square(row[n0:n0+nt]).
  // rows have full stride Nfull; load each row's nt-slice via strided multi-block copy.
  // nt aligned -> tight multi-row load + binary fold; else 1 row/batch with rightPad.
  __aicore__ inline void ColAccumSlice(const AscendC::LocalTensor<float>& dst, uint64_t base,
                                       uint32_t mrows, uint32_t Nfull, uint32_t n0,
                                       uint32_t nt, uint32_t nPadTile) {
    bool aligned = (nt % ALN == 0);
    uint32_t rowsPer = aligned ? MinU(tileElems / nt, mrows) : 1;
    if (rowsPer < 1) rowsPer = 1;
    uint32_t done = 0;
    while (done < mrows) {
      uint32_t rb = MinU(mrows - done, rowsPer);
      uint64_t rowBase = base + (uint64_t)done * Nfull + n0;
      AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
      AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
      if (aligned) {
        // rb blocks of nt elems, src gap (Nfull-nt) elems between rows, dst tight (nt 32B-aligned)
        AscendC::DataCopyExtParams p{(uint16_t)rb, (uint32_t)(nt * sizeof(typeT)),
                                     (uint32_t)((uint64_t)(Nfull - nt) * sizeof(typeT)), 0, 0};
        AscendC::DataCopyPadExtParams<typeT> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(xin, x_gm[rowBase], p, pad);
        inQue.EnQue(xin); xin = inQue.template DeQue<typeT>();
        SquareToF32(sq, xin, rb * nt);
        FoldRowsInto(dst, sq, rb, nt);
      } else {
        AscendC::DataCopyExtParams p{1, (uint32_t)(nt * sizeof(typeT)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<typeT> pad{true, 0, (uint8_t)(nPadTile - nt), (typeT)0};
        AscendC::DataCopyPad(xin, x_gm[rowBase], p, pad);
        inQue.EnQue(xin); xin = inQue.template DeQue<typeT>();
        SquareToF32(sq, xin, nPadTile);
        AscendC::Add(dst, dst, sq, nPadTile);
      }
      inQue.FreeTensor(xin);
      done += rb;
    }
  }

  // dst[width] += column-sum of [rows, width] (binary fold over rows, width 32B-aligned)
  __aicore__ inline void FoldRowsInto(const AscendC::LocalTensor<float>& dst,
                                      const AscendC::LocalTensor<float>& sq,
                                      uint32_t rows, uint32_t width) {
    uint32_t f = 1;
    while (f * 2 <= rows) f *= 2;  // largest pow2 <= rows
    // fold the pow2 part
    uint32_t leftover = rows - f;
    for (uint32_t r = 0; r < leftover; r++)  // add the non-pow2 tail rows into row 0..
      AscendC::Add(sq, sq, sq[(f + r) * width], width);
    uint32_t cur = f;
    while (cur > 1) {
      uint32_t h = cur / 2;
      AscendC::Add(sq, sq, sq[h * width], h * width);
      cur = h;
    }
    AscendC::Add(dst, dst, sq, width);
  }

  // ============================ RPAR (inner==1, few rows) ============================
  __aicore__ inline void ReduceRunInto(const AscendC::LocalTensor<float>& acc,
                                       uint64_t gmOff, uint32_t len) {
    uint32_t off = 0;
    while (off < len) {
      uint32_t c = MinU(len - off, tileElems);
      uint32_t cPad = Align64(c);
      AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
      PadIn(xin, x_gm[gmOff + off], c);
      inQue.EnQue(xin); xin = inQue.template DeQue<typeT>();
      AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
      AscendC::Duplicate(sq, 0.0f, cPad);
      SquareToF32(sq, xin, c);
      uint32_t shape[2] = {1, cPad};
      AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(acc[F32_BLK], sq, shape, false);
      AscendC::Add(acc, acc, acc[F32_BLK], 1);
      inQue.FreeTensor(xin);
      off += c;
    }
  }
  __aicore__ inline void ProcessRPar() {
    uint32_t bidx = AscendC::GetBlockIdx();
    uint32_t r0 = bidx * rPerCore, r1 = MinU(r0 + rPerCore, reduceLen);
    if (r1 > r0) {
      for (uint32_t a = 0; a < outer; a++) {
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::Duplicate(acc, 0.0f, F32_BLK);
        ReduceRunInto(acc, (uint64_t)a * reduceLen + r0, r1 - r0);
        AscendC::LocalTensor<float> po = outQue.template AllocTensor<float>();
        AscendC::Adds(po, acc, 0.0f, 1);
        outQue.template EnQue<float>(po);
        po = outQue.template DeQue<float>();
        AscendC::DataCopyExtParams wp{1, (uint32_t)sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(usrWs[(uint64_t)bidx * Oslot + a], po, wp);
        outQue.FreeTensor(po);
      }
    } else {
      AscendC::LocalTensor<float> acc = accBuf.Get<float>();
      AscendC::Duplicate(acc, 0.0f, Oslot);
      AscendC::LocalTensor<float> po = outQue.template AllocTensor<float>();
      AscendC::Adds(po, acc, 0.0f, Oslot);
      outQue.template EnQue<float>(po);
      po = outQue.template DeQue<float>();
      AscendC::DataCopyExtParams wp{1, (uint32_t)(Oslot * sizeof(float)), 0, 0, 0};
      AscendC::DataCopyPad(usrWs[(uint64_t)bidx * Oslot], po, wp);
      outQue.FreeTensor(po);
    }
    AscendC::SyncAll();
    if (bidx == 0) {
      uint32_t nfloat = rCores * Oslot;
      AscendC::LocalTensor<float> fin = inQue.template AllocTensor<float>();
      AscendC::DataCopyExtParams rp{1, (uint32_t)(nfloat * sizeof(float)), 0, 0, 0};
      AscendC::DataCopyPadExtParams<float> rpad{false, 0, 0, 0};
      AscendC::DataCopyPad(fin, usrWs, rp, rpad);
      inQue.template EnQue<float>(fin); fin = inQue.template DeQue<float>();
      AscendC::LocalTensor<float> acc = accBuf.Get<float>();
      AscendC::Duplicate(acc, 0.0f, Oslot);
      for (uint32_t c = 0; c < rCores; c++) AscendC::Add(acc, acc, fin[c * Oslot], Oslot);
      inQue.FreeTensor(fin);
      AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
      CastF32To(yout, acc, O);
      outQue.template EnQue<typeT>(yout); yout = outQue.template DeQue<typeT>();
      PadOut(y_gm, yout, O);
      outQue.FreeTensor(yout);
    }
  }

  // ============================ MODE 4 (reduce prefix+suffix, keep contig middle) ====
  // mode 4, planes read one at a time: kept rows per tile, as many as the read queue (tileElems input
  // elements) and the gather scratch (GATHER_CAP fp32 lanes over fp32-aligned rows) hold.
  __aicore__ inline uint32_t Mode4TileRows() {
    uint32_t rows = GATHER_CAP / AlignUp(reduceLen, F32_BLK);
    if (rows > tileElems / reduceLen) rows = tileElems / reduceLen;
    return rows < 1u ? 1u : rows;
  }
  // 2-byte input: when a tile of `rows` rows reads a length off a 32B boundary, the largest smaller
  // tile whose reads end on one (such reads are faster, and fp16 then takes one plain DataCopy per
  // plane); 0 when the tile is already aligned or no smaller tile can be.
  __aicore__ inline uint32_t Mode4AlignedRows(uint32_t rows) {
    if constexpr (sizeof(typeT) == 2) {
      uint32_t low = reduceLen & (~reduceLen + 1u);  // largest power of two dividing reduceLen
      uint32_t cycle = low >= 16u ? 1u : 16u / low;  // rows whose bytes make whole 32B
      if ((rows * reduceLen) % 16u == 0u || rows < cycle) return 0u;
      return rows / cycle * cycle;
    } else {
      (void)rows;
      return 0u;
    }
  }

  // input folded to [OUT_R planes(outer=OUT_R), KEPT kept(inner=KEPT), IN_R inner(reduceLen)].
  // each core owns a KEPT-range; per plane read [keptTile, IN_R] CONTIGUOUS (BW-bound big
  // block), square, reduce the inner IN_R -> partial[keptTile], accumulate over OUT_R planes.
  __aicore__ inline void ProcessMode4() {
    uint32_t OUT_R = outer, IN_R = reduceLen, KEPT = inner;
    // INp = fp32-reduce-aligned inner row. The gather output is fp32 and consumed by ReduceSum<AR>,
    // which only needs the reduce axis 32B-aligned (= 8 fp32), NOT ALN(=16 for fp16). Aligning to 8
    // instead of 16 (up to 8 fewer lanes per row) shrinks the gather/reduce width AND lets ktCap run higher.
    uint32_t INp = AlignUp(IN_R, F32_BLK);  // gather-aligned inner row (8 fp32 = 32B)
    uint32_t k0 = AscendC::GetBlockIdx() * keptPerCore;
    if (k0 >= KEPT) return;
    uint32_t k1 = MinU(k0 + keptPerCore, KEPT);
    // NOTE: ReduceSum<AR> supports ONLY fp32 (SupportType<half,float>() assert) -> the gather-
    // align + ReduceSum path is inherently fp32; fp16 no-cast is not viable here. Square+cast
    // to fp32, gather, ReduceSum<AR>. (GM read stays the input dtype, so BW is unaffected.)
    // gathered[kt,INp] lives in gthBuf(<=ROW_TILE); the contiguous flat read [kt*IN_R] in accBuf.
    // Read-burst decouple: the read/accum tile is capped by tileElems/IN_R (accsq [ktCap,IN_R]
    // must fit accBuf; inQue holds one plane's kt*IN_R). The gather scratch GATHER_CAP/INp is
    // wide enough that tileElems/IN_R is usually the binding cap => ~16KB fp16 bursts (was
    // ROW_TILE/INp => ~7KB). Big-stride 16KB bursts hit ~730GB/s (probe-verified). For 2-byte
    // input the tile then rounds down to whole 32B-aligned reads (Mode4AlignedRows).
    uint32_t ktCap = mode4Pair_ ? mode4Kt_ : Mode4TileRows();
    if (!mode4Pair_) {
      uint32_t alignedRows = Mode4AlignedRows(ktCap);
      if (alignedRows != 0u) ktCap = alignedRows;
    }
    if (ktCap < 1) ktCap = 1;
    // build the row-gather index ONCE: idx[i*INp+k] = (k<IN_R? i*IN_R+k : zeroSlot)*4.  Whole
    // reduces masked to IN_R never read the pad lanes, so they take the lean table (no scalar
    // pad fix-up, no pipe barrier ahead of the reads).
    bool wholeReduce = INp <= 128u && ktCap <= 252u;
    if (wholeReduce) {
      BuildRowGatherLean(ktCap, IN_R, INp, ktCap * IN_R);
    } else {
      BuildRowGatherVec(ktCap, IN_R, INp, ktCap * IN_R, (uint32_t)sizeof(float));  // vectorized idx build
    }
    AscendC::LocalTensor<uint32_t> gidx = idxBuf.Get<uint32_t>();
    for (uint32_t kt0 = k0; kt0 < k1; kt0 += ktCap) {
      uint32_t kt = MinU(ktCap, k1 - kt0);
      // Accumulate the OUT_R planes' SQUARES into accsq[kt, IN_R] (packed), THEN gather-align
      // + ReduceSum the inner IN_R ONCE per tile.  Per-plane work is only square+Add (cheap),
      // so the contiguous read dominates => BW-bound (was: gather+reduce PER plane = compute).
      AscendC::LocalTensor<float> accsq = accBuf.Get<float>();
      AscendC::Duplicate(accsq, 0.0f, ktCap * IN_R + ALN);  // zero data + gather zero-slot
      if constexpr (std::is_same_v<typeT, half>) {
        if (mode4Pair_) {
          // Fetch two distant planes with one MTE2 descriptor. DataCopyPad packs each
          // block to a 32B boundary in UB, so the second plane begins at planePitch.
          // Pairing halves the number of input descriptors for an even plane count.
          uint32_t planeElems = kt * IN_R;
          uint32_t planePitch = AlignUp(planeElems, 16u);  // 16 half = 32B
          uint32_t srcGapBytes = (KEPT - kt) * IN_R * (uint32_t)sizeof(typeT);
          for (uint32_t o = 0; o < OUT_R; o += 2) {
            uint64_t base = ((uint64_t)o * KEPT + kt0) * IN_R;
            AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
            AscendC::DataCopyExtParams p{2, planeElems * (uint32_t)sizeof(typeT),
                                         srcGapBytes, 0, 0};
            AscendC::DataCopyPadExtParams<typeT> pad{false, 0, 0, (typeT)0};
            AscendC::DataCopyPad(xin, x_gm[base], p, pad);
            inQue.EnQue(xin);
            xin = inQue.template DeQue<typeT>();
            AscendC::MulAddDst(accsq, xin, xin, (int32_t)planeElems);
            AscendC::MulAddDst(accsq, xin[planePitch], xin[planePitch], (int32_t)planeElems);
            inQue.FreeTensor(xin);
          }
        } else {
          for (uint32_t o = 0; o < OUT_R; o++) {
            uint64_t base = ((uint64_t)o * KEPT + kt0) * IN_R;
            AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
            uint32_t bytes = kt * IN_R * (uint32_t)sizeof(typeT);
            if ((bytes & 31u) == 0u && kt == ktCap) {
              AscendC::DataCopy(xin, x_gm[base], kt * IN_R);
            } else {
              PadIn(xin, x_gm[base], kt * IN_R);
            }
            inQue.EnQue(xin);
            xin = inQue.template DeQue<typeT>();
            AscendC::MulAddDst(accsq, xin, xin, (int32_t)(kt * IN_R));
            inQue.FreeTensor(xin);
          }
        }
      } else {
        for (uint32_t o = 0; o < OUT_R; o++) {
          uint64_t base = ((uint64_t)o * KEPT + kt0) * IN_R;
          AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
          PadIn(xin, x_gm[base], kt * IN_R);
          inQue.EnQue(xin);
          xin = inQue.template DeQue<typeT>();
          AscendC::LocalTensor<float> sqf = sqBuf.Get<float>();
          SquareToF32(sqf, xin, kt * IN_R);              // square in input dtype -> fp32
          AscendC::Add(accsq, accsq, sqf, kt * IN_R);    // accumulate squares across planes
          inQue.FreeTensor(xin);
        }
      }
      AscendC::LocalTensor<float> g = gthBuf.Get<float>();
      AscendC::Gather(g, accsq, gidx, (uint32_t)0, kt * INp);  // align -> [kt, INp] (pad 0) ONCE
      if (wholeReduce) {
        // WholeReduceSum the first min(IN_R, 64) lanes of every row, then the rest, and store
        // the tile once: no ReduceSum<AR> scratch and one write per tile instead of per chunk.
        AscendC::LocalTensor<float> outv = sqBuf.Get<float>();
        int32_t rowBlk = (int32_t)(INp / F32_BLK);
        AscendC::WholeReduceSum<float>(outv, g, (int32_t)MinU(IN_R, 64u), (int32_t)kt, 1, 1, rowBlk);
        if (IN_R > 64u) {
          AscendC::LocalTensor<float> tail = outv[AlignUp(kt, F32_BLK)];
          AscendC::WholeReduceSum<float>(tail, g[64], (int32_t)(IN_R - 64u), (int32_t)kt, 1, 1,
                                         rowBlk);
          AscendC::Add(outv, outv, tail, (int32_t)kt);
        }
        AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
        CastF32To(yout, outv, kt);
        outQue.template EnQue<typeT>(yout);
        yout = outQue.template DeQue<typeT>();
        PadOut(y_gm[kt0], yout, kt);
        outQue.FreeTensor(yout);
        continue;
      }
      // ReduceSum the aligned [kt, INp] in bounded row slices: the lib scratch for one
      // ReduceSum<AR> over the WHOLE 118-row tile overflows UB, so reduce sub-slices of the
      // already-gathered g (no re-gather, no idx rebuild -> cheap loop) and write each slice.
      for (uint32_t c0 = 0; c0 < kt; c0 += REDUCE_CHUNK) {
        uint32_t cc = MinU(REDUCE_CHUNK, kt - c0);
        AscendC::LocalTensor<float> outv = sqBuf.Get<float>();   // reduce output
        uint32_t shp[2] = {cc, INp};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            outv, g[(uint64_t)c0 * INp], shp, false);
        AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
        CastF32To(yout, outv, cc);
        outQue.template EnQue<typeT>(yout);
        yout = outQue.template DeQue<typeT>();
        PadOut(y_gm[kt0 + c0], yout, cc);
        outQue.FreeTensor(yout);
      }
    }
  }

  // ============================ GENERAL (non-contiguous) ============================
  __aicore__ inline uint64_t LeadOffset(uint32_t idx, const uint32_t* dim,
                                        const uint32_t* str, uint32_t nd) {
    uint64_t off = 0; uint32_t tmp = idx;
    for (int d = (int)nd - 1; d >= 0; d--) { off += (uint64_t)(tmp % dim[d]) * str[d]; tmp /= dim[d]; }
    return off;
  }
  __aicore__ inline void ProcessGeneral() {
    for (uint32_t ko = koStart; ko < koEnd; ko++) {
      uint64_t base = LeadOffset(ko, klDim, klStr, nKL);
      if (unitReduced) {
        AscendC::LocalTensor<float> acc = accBuf.Get<float>();
        AscendC::Duplicate(acc, 0.0f, F32_BLK);
        for (uint32_t rc = 0; rc < rlProd; rc++)
          ReduceRunInto(acc, base + LeadOffset(rc, rlDim, rlStr, nRL), unit);
        AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
        CastF32To(yout, acc, 1);
        outQue.template EnQue<typeT>(yout); yout = outQue.template DeQue<typeT>();
        PadOut(y_gm[ko], yout, 1);
        outQue.FreeTensor(yout);
      } else {
        uint32_t icStart = 0;
        while (icStart < unit) {
          uint32_t ic = MinU(unit - icStart, tileElems);
          AscendC::LocalTensor<float> acc = accBuf.Get<float>();
          AscendC::Duplicate(acc, 0.0f, ic);
          for (uint32_t rc = 0; rc < rlProd; rc++) {
            uint64_t roff = LeadOffset(rc, rlDim, rlStr, nRL);
            AscendC::LocalTensor<typeT> xin = inQue.template AllocTensor<typeT>();
            PadIn(xin, x_gm[base + roff + icStart], ic);
            inQue.EnQue(xin); xin = inQue.template DeQue<typeT>();
            AscendC::LocalTensor<float> sq = sqBuf.Get<float>();
            SquareToF32(sq, xin, ic);
            AscendC::Add(acc, acc, sq, ic);
            inQue.FreeTensor(xin);
          }
          AscendC::LocalTensor<typeT> yout = outQue.template AllocTensor<typeT>();
          CastF32To(yout, acc, ic);
          outQue.template EnQue<typeT>(yout); yout = outQue.template DeQue<typeT>();
          PadOut(y_gm[(uint64_t)ko * unit + icStart], yout, ic);
          outQue.FreeTensor(yout);
          icStart += ic;
        }
      }
    }
  }

 private:
  // The TPipe lives in the kernel entry: constructing one publishes its address through a global
  // pointer, and inside this object that would keep the compiler from folding the scalar members.
  AscendC::TPipe* pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> sqBuf, accBuf, gthBuf, idxBuf;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQue;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQue;
  AscendC::GlobalTensor<typeT> x_gm, y_gm;
  AscendC::GlobalTensor<float> usrWs;
  uint32_t tileElems, fltChunk, outer, reduceLen, inner, oStart, oCount;
  uint32_t keptPerCore;
  uint32_t colCores, mPerCore, nP, colGatherRows, colTransMc;
  uint32_t colBurstMul_, mode4Kt_;
  uint32_t rowBoundaryRows_;
  bool f32ColBurst_, mode4Pair_, rowBoundary_;
  uint32_t O, Oslot, rCores, rPerCore;
  uint32_t unitReduced, unit, klProd, rlProd, nKL, nRL, koStart, koEnd;
  uint32_t klDim[MAX_DIM], klStr[MAX_DIM], rlDim[MAX_DIM], rlStr[MAX_DIM];
};

extern "C" __global__ __aicore__ void square_sum_v1(GM_ADDR input, GM_ADDR y,
                                                    GM_ADDR workspace, GM_ADDR tiling) {
  // One kernel binary per tiling key (host key = mode + 1); each key keeps its own branch here so
  // opbuild emits its entry.
  // Init reads at most a dozen tiling fields, and only the ones its mode needs.
  // GET_TILING_DATA instead copies the whole struct into local memory, which
  // measures 4.2us of a 40-core launch -- the cost is the copy's own setup, not
  // its 208 bytes, since Concat pays 8.2us for a 4180-byte struct.  Reading the
  // fields straight out of GM skips that setup.  GENERAL walks klDim/klStr/rlDim/
  // rlStr -- 32 array entries -- and pulling those one at a time out of GM costs
  // more than the copy saves (case 4 scalar 53.0 -> 56.4us), so only GENERAL copies.
  AscendC::TPipe pipe;  // outside the kernel object (see KernelSquareSumV1::pipe)
  if (TILING_KEY_IS(1)) {  // ROW
    KernelSquareSumV1<DTYPE_INPUT> op;
    op.Init(input, y, workspace, (__gm__ const TilingData*)tiling, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(3)) {  // RPAR
    KernelSquareSumV1<DTYPE_INPUT> op;
    op.Init(input, y, workspace, (__gm__ const TilingData*)tiling, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(4)) {  // COLUMN
    KernelSquareSumV1<DTYPE_INPUT> op;
    op.Init(input, y, workspace, (__gm__ const TilingData*)tiling, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(5)) {  // reduced prefix and suffix, contiguous kept middle
    KernelSquareSumV1<DTYPE_INPUT> op;
    op.Init(input, y, workspace, (__gm__ const TilingData*)tiling, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(10)) {  // FLAT
    KernelSquareSumV1<DTYPE_INPUT> op;
    op.Init(input, y, workspace, (__gm__ const TilingData*)tiling, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(2)) {  // GENERAL
    GET_TILING_DATA(tiling_data, tiling);
    KernelSquareSumV1<DTYPE_INPUT> op;
    op.Init(input, y, workspace, &tiling_data, &pipe);
    op.Process();
  }
}
