/*
 * IndexAdd: out = input.clone(); out.select(dim, index[k]) += source.select(dim, k)
 * aclnnIndexAdd(input, index, source, dim, out).  Index may repeat (collisions
 * accumulate).  Plan: copy input->out, cross-core barrier, then each (outer,k)
 * unit atomically adds source[.,k,.] onto out[.,index[k],.].  Hardware atomic
 * add reproduces torch's collision-sum and int wrap-around for free.
 */
#include <cstdint>

#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

// wide-row workload (fp16 last-axis) phase-1/3 N-tile cap.  Effectively UNCAPPED now: the unified-pool rewrite
// below decouples the phase-1/3 row width from the phase-2 source tile, so a whole-row Nt no
// longer forces B down (device printf confirms B stays 192, 1 block/core).  The old cap of 64
// existed only because the pre-decouple layout charged a wide Nt 3x (2 xfer slots + srcT) and
// shrank B to 96, doubling the per-core add-steps.  Measured on devenv: uncapping (whole-row
// 446 B phase-1 reads + the wide phase-3 store) is -5.37 +/- 0.33 us on wide-row workload, ABBA 14 pairs,
// against a +-0.86 us null band; tiny workload/fused-axis workload/large-strided workload/large-stream workload .o are byte-identical so they cannot be affected.
#ifndef IA_HALF_WIDE_NTCAP
#define IA_HALF_WIDE_NTCAP 4096u
#endif

// key-7 fp32 phase-1/3 N-tile (multiple of 16).  112 => 448 B x/y blocks; 224 => 896 B.
// The unified-pool rewrite below decouples Nt from the phase-2 source tile, so Nt is free
// up to pool/(2*B): at B=64 the 190 KB guard allows Nt=336.
#ifndef IA_F32_NTILE
#define IA_F32_NTILE 336u
#endif

// key-7 fp32 phase-2 source tile width (multiple of 16).  128 => 512 B src blocks; 144 => 576 B.
#ifndef IA_F32_MTILE
#define IA_F32_MTILE 160u
#endif

namespace optiling {

constexpr uint32_t MAX_TILE = 8192;
constexpr uint32_t STAGE_CAP = 64;
enum class IndexAddKey : uint32_t {
  KEY_INT_ATOMIC = 1,
  KEY_FP32_DIRECT = 2,
  KEY_FPWS_ATOMIC = 3,
  KEY_FP_DIRECT = 4,
  KEY_STEPWISE = 5,
  KEY_STEPWISE_ALL_OUTER = 6,
  KEY_VEC_TRANSPOSE = 7,
  KEY_BATCHED_READ = 12,
  KEY_PACKED_ROWS = 13,
};

static uint32_t ElementSizeBytes(uint32_t dt) {
  switch (dt) {
    case ge::DT_FLOAT16:
    case ge::DT_BF16:
      return 2;
    case ge::DT_INT8:
      return 1;
    default:
      return 4;
  }
}

static bool IsIntegerType(uint32_t dt) {
  return dt == ge::DT_INT8 || dt == ge::DT_INT32;
}

static bool IsFloat32Type(uint32_t dt) { return dt == ge::DT_FLOAT; }

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  IndexAddTilingData tiling;
  auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t maxCores = plat.GetCoreNumAiv();
  if (maxCores < 1) maxCores = 1;
  uint32_t dtype = context->GetInputDesc(0)->GetDataType();
  uint32_t dsize = ElementSizeBytes(dtype);
  uint32_t elAlign = 256 / dsize;
  if (elAlign == 0) elAlign = 1;

  auto inShape = context->GetInputShape(0)->GetStorageShape();
  uint32_t r = inShape.GetDimNum();
  if (r == 0) r = 1;

  auto* attrs = context->GetAttrs();
  const int64_t* dimPtr = (attrs != nullptr) ? attrs->GetInt(0) : nullptr;
  int64_t dim = (dimPtr != nullptr) ? *dimPtr : 0;
  if (dim < 0) dim += (int64_t)r;
  if (dim < 0) dim = 0;
  if (dim >= (int64_t)r) dim = (int64_t)r - 1;

  uint64_t outer = 1, inner = 1;
  for (uint32_t d = 0; d < (uint32_t)dim; d++)
    outer *= (uint64_t)inShape.GetDim(d);
  uint64_t dimLen = (uint64_t)inShape.GetDim((uint32_t)dim);
  for (uint32_t d = (uint32_t)dim + 1; d < r; d++)
    inner *= (uint64_t)inShape.GetDim(d);
  if (dimLen == 0) dimLen = 1;
  if (inner == 0) inner = 1;
  if (outer == 0) outer = 1;

  uint64_t m = (uint64_t)context->GetInputShape(1)->GetStorageShape().GetShapeSize();
  if (m == 0) m = 1;

  uint64_t copyTotal = outer * dimLen * inner;
  uint64_t units = outer * m;

  uint32_t coreNum = maxCores;
  // Launch/synchronization-bound int8 dim-0 work benefits from fewer even
  // cores.  Select by useful rows per core rather than by an exact tensor.
  if (dtype == ge::DT_INT8 && outer == 1 && units < (uint64_t)maxCores * 4u &&
      copyTotal < (1ULL << 20)) {
    uint32_t useful = (uint32_t)((units + 7u) / 8u);
    if (useful < 2u) useful = 2u;
    useful = (useful + 1u) & ~1u;
    if (useful > 16u) useful = 16u;
    if (useful < coreNum) coreNum = useful;
  }
  uint64_t copyPerCore = (copyTotal + coreNum - 1) / coreNum;
  copyPerCore = ((copyPerCore + elAlign - 1) / elAlign) * elAlign;
  if (copyPerCore < elAlign) copyPerCore = elAlign;
  uint32_t copyTileLen = MAX_TILE;
  if ((uint64_t)copyTileLen > copyPerCore) copyTileLen = (uint32_t)copyPerCore;
  copyTileLen = ((copyTileLen + elAlign - 1) / elAlign) * elAlign;
  if (copyTileLen < elAlign) copyTileLen = elAlign;

  uint64_t unitsPerCore = (units + coreNum - 1) / coreNum;
  if (unitsPerCore < 1) unitsPerCore = 1;

  uint32_t innerTile = (inner < MAX_TILE) ? (uint32_t)inner : MAX_TILE;

  // ---- route by dtype + dim-position (verified vs torch 2.5.1 = contest golden) ----
  //   int8/int32 -> K_INT_ATOMIC(1); float -> K_FP32_DIRECT(2);
  //   fp16/bf16 & (dim0 || last-axis: outer==1 || inner==1):
  //       huge numel (fp32 ws > 256MB) -> K_FP_DIRECT(4) else K_FPWS_ATOMIC(3);
  //   fp16/bf16 & middle dim (outer>1 && inner>1) -> K_STEPWISE(5).
  bool isInt = IsIntegerType(dtype);
  bool isFp32 = IsFloat32Type(dtype);
  IndexAddKey tilingKey = IndexAddKey::KEY_INT_ATOMIC;
  if (isInt) {
    // int32/int8: default atomic scatter (key 1).  Large int32 middle-dim streams use
    // key 12/13 when their UB budget permits batched contiguous reads; otherwise the
    // generic path is retained.  These routes keep all cores busy and preserve source order.
    // key 12 (batched-read + Gather-realign): read K consecutive source rows in ONE ~32KB DMA
    // (vs per-row 10.6KB), Gather-repack them to a 32B-aligned padded pitch (fixes the
    // non-32B-aligned write-source problem), then atomic-write each aligned row.
    tilingKey = IndexAddKey::KEY_INT_ATOMIC;
    bool int32Mid = (dtype == ge::DT_INT32 && inner > 1 && outer > 1);
    if (int32Mid) {
      uint32_t packedK = 8u;
      uint32_t packedStride = ((uint32_t)inner + 7u) / 8u * 8u;
      uint64_t packedUb = 2ULL * packedK * packedStride * 4ULL;
      uint64_t indexUb = ((m + 7ULL) / 8ULL * 8ULL) * sizeof(int32_t);
      uint64_t streamedSource = (uint64_t)outer * m * inner * 4ULL;
      bool packedRows = inner * 4ULL >= 4096ULL && (inner & 7ULL) != 0ULL &&
                        packedUb + indexUb + 1024ULL <= 190ULL * 1024ULL &&
                        streamedSource >= 64ULL * 1024ULL * 1024ULL;
      if (packedRows) {
        // Multi-block DataCopyPad directly materializes eight source rows at a
        // 32B-aligned UB pitch. This removes the full-row Gather and lets the
        // queue overlap packed reads with atomic writes.
        tilingKey = IndexAddKey::KEY_PACKED_ROWS;
      }
      uint32_t K = 32768u / ((uint32_t)inner * 4u);
      if (K < 1) K = 1;
      if (K > 8) K = 8;
      uint32_t bstr = ((uint32_t)inner + 7) / 8 * 8;
      uint64_t need = (uint64_t)K * inner * 4 * 2 + (uint64_t)K * bstr * 4 * 2;  // qBind + buf2 + off
      if (!packedRows && K >= 2 && need + 32ULL * 1024 <= 190ULL * 1024) tilingKey = IndexAddKey::KEY_BATCHED_READ;
    }
  } else if (isFp32) {
    if (inner == 1 && outer > 1) {
      // fp32 last-axis: vectorized transpose (B outer-rows -> lanes), GM transposed workspace
      // + SetAtomicAdd.  A UB-accumulator variant (no workspace, so only the minimal 59.6 MB of
      // traffic) was built and LOSES 218us vs 176us: an fp32 resident accT forces B down to 32,
      // which doubles the per-core vector-Add count and flips the kernel vec-bound.
      tilingKey = IndexAddKey::KEY_VEC_TRANSPOSE;
    } else {
      tilingKey = IndexAddKey::KEY_FP32_DIRECT;
    }
  } else {  // fp16 / bf16
    bool middleDim = (outer > 1 && inner > 1);
    if (middleDim) {
      tilingKey = IndexAddKey::KEY_STEPWISE;
    } else if (inner == 1 && outer > 1) {
      tilingKey = IndexAddKey::KEY_VEC_TRANSPOSE;  // last-axis: vectorized transpose, per-step round (fp16 int32-idx semantics)
    } else if ((uint64_t)copyTotal * 4ULL > (256ULL << 20)) {
      tilingKey = IndexAddKey::KEY_FP_DIRECT;
    } else {
      tilingKey = IndexAddKey::KEY_FPWS_ATOMIC;  // dim0 (outer==1)
    }
  }

  // owned-rows (keys 3 & 5): each core owns a slice of Dim rows, holds their fp32
  // accumulators in UB, scans M once in k-order.  key 3 (dim0/last) rounds once
  // (fp32-once); key 5 (middle dim) rounds each step.
  uint32_t accRowsPerCore = 1, accInnerTile = (uint32_t)inner;
  if (tilingKey == IndexAddKey::KEY_STEPWISE) {
    accRowsPerCore = (uint32_t)((dimLen + maxCores - 1) / maxCores);
    if (accRowsPerCore < 1) accRowsPerCore = 1;
    uint32_t accCore = (uint32_t)((dimLen + accRowsPerCore - 1) / accRowsPerCore);
    if (accCore < 1) accCore = 1;
    // all-outer IA-2 layout is [ownedRow, outer, spPitch].  Account every UB buffer here: using
    // only the fp32 acc size is unsafe when bf16 spPitch (16 elems) differs from fp32 acc pitch
    // (8 elems), e.g. inner=120.  Keep headroom below the 192 KiB physical UB limit.
    uint64_t spPitch = ((uint64_t)inner + 15) / 16 * 16;
    uint64_t wide = (uint64_t)outer * spPitch;
    uint64_t layoutElems = (uint64_t)accRowsPerCore * wide;
    uint64_t stageElems = (uint64_t)STAGE_CAP * spPitch;
    uint64_t qLoadElems = layoutElems > stageElems ? layoutElems : stageElems;
    uint64_t idxElems = ((uint64_t)m + 7) / 8 * 8;
    uint64_t alloBytes = 2 * qLoadElems * dsize + 2 * layoutElems * dsize +
                         layoutElems * 4 + wide * 4 + wide * dsize + 2 * idxElems * 4;
    if (outer > 1 && alloBytes <= 180ULL * 1024) {
      accInnerTile = (uint32_t)inner;  // whole inner, single tile
      // the all-outer layout is its own kernel binary, so key 5 never carries its code
      tilingKey = IndexAddKey::KEY_STEPWISE_ALL_OUTER;
    } else {
      uint64_t ait = (90ULL * 1024) / ((uint64_t)accRowsPerCore * 4);
      if (ait > inner) ait = inner;
      if (ait >= 8) ait = ait / 8 * 8;  // keep acc row pitch 32B aligned (fp32)
      if (ait < 1) ait = 1;
      // Kept strictly below inner.  Before the all-outer layout had its own key,
      // the kernel took accInnerTile == inner as the host committing to that
      // layout and sized its UB from outer*spPitch -- exactly the allocation the
      // gate above had just rejected (507015, "VEC instruction error: the ub
      // address out of bounds", at inner=8 and inner=16).
      // key 5 is the middle-dim route, so inner > 1 holds here.
      if (outer > 1 && ait >= inner) {
        ait = inner - 1;
        if (ait >= 8) ait = ait / 8 * 8;
        if (ait < 1) ait = 1;
      }
      accInnerTile = (uint32_t)ait;
    }
    coreNum = accCore;  // owned-rows uses one core per Dim-slice
  }

  // key 7: vectorized transpose (inner==1, outer>1).  Block B outer-rows into
  // lanes; accT[N,B] resident fp32; per k a B-wide vector Add (fp16/bf16 per-step
  // round in k-order, fp32 no round).  B chosen so the accumulate-phase peak fits.
  uint32_t k7B = 0, k7Ntile = 0, k7Mtile = 0, k7BlocksPerCore = 0;
  if (tilingKey == IndexAddKey::KEY_VEC_TRANSPOSE) {
    uint32_t N = (uint32_t)dimLen;
    k7Ntile = 64;                              // fixed, mult of 8 (output-transpose pitch)
    k7Mtile = (dsize == 2) ? 64u : 128u;       // smaller M-tile -> smaller buffers -> bigger B
    uint32_t maxTile = k7Ntile > k7Mtile ? k7Ntile : k7Mtile;
    uint32_t Npad = ((N + k7Ntile - 1) / k7Ntile) * k7Ntile;
    uint64_t idxSz = (((uint64_t)m + 7) / 8 * 8) * 4;
    uint32_t B = 16;
    bool fp16Fast = (dtype == ge::DT_FLOAT16);  // wide-row workload: fp16 fast path uses fp16 accT (half UB)
    bool fp32Fast = (dtype == ge::DT_FLOAT);    // large-stream workload: fp32 TILED accT[Nt,B] -> bigger B
    if (fp16Fast) {
      // ---- WIDE-BURST + PIPELINED retile (wide-row workload) -------------------------------------------
      // Measured on this machine: with the shipped 64-wide tiles every DMA block is
      // tile*sizeof(T) = 128 B, and a NOVEC strip (all vector calls deleted) gives 79us of pure
      // DMA against an 87us kernel -- i.e. load and compute are strictly SERIAL (the single xfer
      // buffer forces WaitFlag<MTE2_V> before every transpose).  Two independent fixes:
      //   Nt = alignUp(N,16) makes phase 1/3 a SINGLE whole-row transfer (blockLen = N*2, GM
      //     stride 0 -> the B rows are one contiguous B*N*2 run); NOVEC then drops 79 -> 70us.
      //   2x xfer lets phase 2 issue tile i+1's read before tile i's transpose, overlapping the
      //     two.  Row-REALIGN cost is ZERO (the hw 16x16 transpose takes any tile width, no
      //     Gather -- that realign cost is what sank the SquareSumV1 fused-axis workload attempt).
      // B must not shrink far: halving B doubles the per-core M add-steps (measured +21us of
      // exposed compute at B=96 vs B=192), so pick B by MIN per-core outer rows
      // (= ceil(ceil(outer/B)/cores)*B), not by "largest that fits"; ties go to the larger B.
      // WHOLE-ROW DECOUPLE: the kernel's phase-1/3 staging buffer is now sized INDEPENDENTLY of
      // the phase-2 source tile.  UB layout = accT[Npad,B] + ONE pool of B*max(Nt, 3*Mt) elems
      // (phase 1 staging = phase 3 pack = B*Nt; phase 2 = 2 load slots + srcT = 3*B*Mt).  Because
      // Nt no longer inflates the phase-2 slots, Nt can be alignUp(N,16) -- the whole-row mode --
      // WITHOUT forcing B down.  Effect: phases 1 and 3 become ONE contiguous B*N*2 GM run each
      // (446 B blocks, srcStride 0) instead of ceil(N/64) strided 128 B passes, and the ragged
      // per-row PadOutT store loop disappears.
      // ---- ARM L (L1e): PHASE-3-WIDE ONLY --------------------------------------------
      // Phase 1/2 keep the shipped 64-wide N-tile (128 B GM read blocks).  What changes is the
      // OUTPUT geometry: accT rows drop from alignUp(N,Nt)=256 to Nrow=alignUp(N,16)=224, and the
      // freed 12,288 B lets the unified pool hold a FULL-WIDTH outT[B,Nrow] pack, so phase 3 is
      // ONE DataCopyPad of curB contiguous N*2 (=446 B) blocks instead of ceil(N/Nt)=4 passes of
      // 192 x 128 B blocks with a 318 B GM gap.
      uint32_t NtC = ((N + 15) / 16) * 16;
      if (NtC > IA_HALF_WIDE_NTCAP) NtC = IA_HALF_WIDE_NTCAP;
      k7Ntile = NtC;
      Npad = ((N + 15) / 16) * 16;               // accT rows == phase-3 pack pitch (Nrow)
      uint32_t peMin = Npad;                     // pool floor: phase 3 needs B*Nrow elems
      if (peMin < k7Ntile) peMin = k7Ntile;      // phase 1 stages [B,Nt]
      if (peMin < 48u) peMin = 48u;              // phase 2 floor at Mt=16 (3*Mt=48)
      uint64_t bestRows = ~0ULL;
      for (uint32_t cand = 256; cand >= 16; cand -= 16) {
        uint64_t base = (uint64_t)Npad * cand * 2 + (uint64_t)cand * peMin * 2 + idxSz + 1024;
        if (base > 180ULL * 1024) continue;
        uint64_t nb = (outer + cand - 1) / cand;
        uint64_t bpc = (nb + maxCores - 1) / maxCores;
        uint64_t rows = bpc * cand;
        if (rows < bestRows) { bestRows = rows; B = cand; }
      }
      // then grow the source tile Mt as far as the residual pool allows (wider phase-2 blocks)
      uint32_t bestMt = 16;
      for (uint32_t mt = 512; mt >= 16; mt -= 16) {
        uint32_t pe = 3 * mt;
        if (pe < peMin) pe = peMin;
        uint64_t peak = (uint64_t)Npad * B * 2 + (uint64_t)B * pe * 2 + idxSz + 1024;
        if (peak <= 190ULL * 1024) { bestMt = mt; break; }
      }
      k7Mtile = bestMt;
      maxTile = k7Ntile > k7Mtile ? k7Ntile : k7Mtile;
    } else if (fp32Fast) {
      // fp32 TILED buffers: accT[Nt,B] + 2x xfer[B,maxTile] + 2x srcT[Mt,B] (all fp32, no Gather
      // offsets).  BOTH the load staging and the transposed source are ping-ponged so phase 2 can
      // issue tile i+1's MTE2 read before tile i's V/MTE3 barrier (the shipped single-slot xfer
      // serialised the pipes: measured mte2 106us + mte3 67us == aiv_time 178us).
      // Cap B<=64: phase-3 FastTransB32 transposes accT[Nt,B]->[B,Nt] with channelAlign=B<=64.
      // Bigger B => fewer blocks => half the phase-2 atomic-store count (the scalar bound).
      // Nt widened 64 -> IA_C5_NTILE: accT is [Nt,B] (not [Npad,B]) so Nt costs UB linearly, and
      // it sets the phase-1 x-read / phase-3 y-write block size (Nt*4 B).  Must be a multiple of
      // 16: phase 3 calls FastTransB32 with R=Nt, which reads ceil(Nt/16)*16 accT rows.
      // UNIFIED POOL (T1, transferred from the fp16 wide-row workload path): the three fp32 buffers
      // accT[Nt,B], 2x xfer[B,maxTile] and 2x srcT[Mt,B] are STRICTLY SEQUENTIAL in liveness --
      //   phase 1 touches accT + one staging slot,   (kernel: PipeBarrier<PIPE_ALL> hand-off)
      //   phase 2 touches 2 xfer + 2 srcT and never accT,
      //   phase 3 touches accT + outF(=one srcT slot).
      // Charging them separately paid for 28,672 dead bytes at the phase-2 peak AND coupled the
      // phase-1/3 row width Nt to the phase-2 tile through `maxTile`.  One pool of
      //   max(2*Nt*B, 4*Mt*B) elems
      // removes both: Nt now costs nothing until 2*Nt exceeds 4*Mt, so the phase-1 x read and the
      // phase-3 y write can use 896 B blocks (Nt=224, 3 tiles) instead of 448 B (Nt=112, 6 tiles),
      // and the residue still buys Mt 128->144 (576 B phase-2 source blocks).
      k7Ntile = IA_F32_NTILE;
      k7Mtile = IA_F32_MTILE;
      for (uint32_t cand = 64; cand >= 16; cand -= 16) {
        uint64_t p13 = 2ULL * k7Ntile * cand;   // accT[Nt,B] + staging/pack [B,Nt]
        uint64_t p2 = 4ULL * k7Mtile * cand;    // 2 xfer slots [B,Mt] + 2 srcT slots [Mt,B]
        uint64_t pool = p13 > p2 ? p13 : p2;
        uint64_t peak = pool * 4 + idxSz + 1024;
        // This phase's buffers are strictly sequential, so the whole 192 KiB is usable (index and safety
        // slack included).  On the 910B4 allocations up to 196,608 B pass while 200,064 B fails with
        // 507015; 253,952 B is the Ascend 950 figure.  192 KiB admits B=64 for M up to about 5,880 and
        // falls back to B=48 above that.
        if (peak <= 192ULL * 1024) { B = cand; break; }
      }
      maxTile = k7Ntile > k7Mtile ? k7Ntile : k7Mtile;  // (fp32 kernel no longer uses it)
    } else {
      // bf16 old gather path (unused): fp32 accT[Npad,B] + off + srcT + gth + rnd.
      for (uint32_t cand = 128; cand >= 16; cand -= 16) {
        uint64_t peak = (uint64_t)Npad * cand * 4 + (uint64_t)cand * maxTile * 2 +
                        (uint64_t)maxTile * cand * 4 + (uint64_t)k7Mtile * cand * 4 +
                        (uint64_t)maxTile * cand * 2 + idxSz + 1024;
        if (peak <= 180ULL * 1024) { B = cand; break; }
      }
    }
    k7B = B;
    uint64_t nBlocks = (outer + B - 1) / B;
    uint32_t cn = (uint32_t)(nBlocks < maxCores ? nBlocks : maxCores);
    if (cn < 1) cn = 1;
    k7BlocksPerCore = (uint32_t)((nBlocks + cn - 1) / cn);
    cn = (uint32_t)((nBlocks + k7BlocksPerCore - 1) / k7BlocksPerCore);
    if (cn < 1) cn = 1;
    coreNum = cn;
    // For fp32 last-axis streams, keep all AIVs resident even when the block
    // count is only slightly above 1x core count.  Extra cores take one block
    // while the tail cores take two; this improves MTE issue concurrency.
    if (coreNum < maxCores) coreNum = maxCores;
  }

  // key 23: fused (rank,row) lanes + cross-core prologue, fp16 last-axis.  Measured on the 910B4
  // (C2 [305,24,223] M1200, 40 cores): 70.46 us vs 77.98 for key 7, 3 rotated rounds, acc pass.
  // Every condition is a HARD invariant of the key-23 kernel -- stack tables, uint8 repeat counts,
  // and the UB carve -- so any shape that fails one keeps the shipped key-7 path.
  bool useFused = false;
  if (tilingKey == IndexAddKey::KEY_VEC_TRANSPOSE && dtype == ge::DT_FLOAT16 && inner == 1 &&
      outer >= 1 && k7B != 0 && coreNum >= 1) {
    uint64_t fN = (uint64_t)dimLen;
    uint64_t fM = (uint64_t)m;
    uint64_t fNR = (fN + 15) / 16 * 16;
    uint64_t fChK = (fM + coreNum - 1) / coreNum;
    uint64_t fMPad = (fM + 7) / 8 * 8;
    useFused = fN >= 1 && fN <= 255 && fNR <= 256 &&            // rank tables uint8 / [256]
               fM >= 8 && (fM % 8) == 0 && fM <= 2040 &&          // repM = M/8 is uint8; LCAP = 2048
               fChK <= 256 &&                                     // per-core chunk cache [256]
               (uint64_t)coreNum * fNR <= 8 * fM &&               // chunk histograms fit srcT (32M B)
               12 * fNR + 32 * fChK + 4 * fM <= 32 * fM &&        // hW/pre/tot + value slots + offsets fit plane
               132 * fM + 136 * fNR + 4 * fMPad <= 193664ULL;     // pool + accBuf + idxBuf <= measured-safe UB
  }

  // key 12: int32 batched-read.  accInnerTile = K (rows per DMA); coreNum stays maxCores.
  if (tilingKey == IndexAddKey::KEY_BATCHED_READ) {
    uint32_t K = 32768u / ((uint32_t)inner * 4u);
    if (K < 1) K = 1;
    if (K > 8) K = 8;
    accInnerTile = K;
    copyTileLen = 4096;
    if ((uint64_t)copyTileLen > copyPerCore) copyTileLen = (uint32_t)copyPerCore;
    if (copyTileLen < 1) copyTileLen = 1;
  }

  // Packed-row read. Round the unit span to a 32-row boundary so successive
  // core starts retain a stable 512B phase; K=8 keeps the padded row batch in UB.
  if (tilingKey == IndexAddKey::KEY_PACKED_ROWS) {
    accInnerTile = 8;
    unitsPerCore = ((unitsPerCore + 31) / 32) * 32;
    // qBind is already sized for one packed 8-row read; reuse that capacity for
    // the initial x->y passthrough so CopyDirect emits four large transfers per
    // core instead of many 4-K-element descriptors.
    copyTileLen = 8u * (((uint32_t)inner + 7u) / 8u * 8u);
    if ((uint64_t)copyTileLen > copyPerCore) copyTileLen = (uint32_t)copyPerCore;
    if (copyTileLen < 1) copyTileLen = 1;
  }
  context->SetBlockDim(coreNum);
  // The key selects the compiled kernel binary; it is metadata, not tiling data.
  context->SetTilingKey(useFused ? 23u : static_cast<uint32_t>(tilingKey));
  tiling.set_outer(outer);
  tiling.set_dimLen(dimLen);
  tiling.set_inner(inner);
  tiling.set_m((uint32_t)m);
  tiling.set_accInnerTile(accInnerTile);
  tiling.set_k7B(k7B);
  tiling.set_k7Ntile(k7Ntile);
  tiling.set_k7Mtile(k7Mtile);

  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  size_t* currentWorkspace = context->GetWorkspaceSizes(1);
  // 16MB system-reserved prefix (GetUserWorkspace skips it) + K_FPWS fp32 ws.
  if (tilingKey == IndexAddKey::KEY_FPWS_ATOMIC) {
    currentWorkspace[0] = (16ULL * 1024 * 1024) + copyTotal * 4ULL;
  } else if (tilingKey == IndexAddKey::KEY_VEC_TRANSPOSE && dsize == 4) {
    // large-stream workload fp32: transposed atomic accumulator outT[dimLen, OUTER_pad], OUTER_pad = ceil(outer/k7B)*k7B.
    uint64_t outerPad = (uint64_t)((outer + k7B - 1) / k7B) * k7B;
    currentWorkspace[0] = (16ULL * 1024 * 1024) + (uint64_t)dimLen * outerPad * 4ULL;
  } else if (useFused) {
    // key 23 GM ws after the 16MB prefix: [0, coreNum*NR) int32 chunk histograms, then
    // alignUp(m,8) 32 B gbOff records (one full block per slot: a 4 B store corrupts neighbours).
    uint64_t wNR = ((uint64_t)dimLen + 15) / 16 * 16;
    uint64_t wMPad = ((uint64_t)m + 7) / 8 * 8;
    currentWorkspace[0] = (16ULL * 1024 * 1024) + (uint64_t)coreNum * wNR * 4ULL + wMPad * 32ULL + 4096ULL;  } else {
    currentWorkspace[0] = 16 * 1024 * 1024;
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  const gert::Shape* xShape = context->GetInputShape(0);
  gert::Shape* yShape = context->GetOutputShape(0);
  *yShape = *xShape;
  return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class IndexAdd : public OpDef {
 public:
  explicit IndexAdd(const char* name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32,
                   ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("indices")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
                   ge::DT_INT32})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Input("source")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32,
                   ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32,
                   ge::DT_INT8})
        .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                 ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                             ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("dim").AttrType(REQUIRED).Int();

    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(IndexAdd);
}  // namespace ops
