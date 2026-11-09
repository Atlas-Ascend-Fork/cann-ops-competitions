/*
 * IndexAdd kernel.  out = input.clone(); out[.,index[k],.] += source[.,k,.]
 * Index may repeat (collisions accumulate).
 *
 * ★torch semantics (verified locally, torch 2.5.1 = contest golden):
 *   - int8/int32 (any dim): integer exact, order-independent (int8 wraps mod256).
 *   - fp16/bf16/fp32, dim==0 OR inner==1(last axis): fp32 accumulate, round ONCE
 *     (order-independent).  -> parallel fp32 accumulation matches torch bit-exact.
 *   - fp16/bf16, MIDDLE dim (outer>1 && inner>1): native-dtype stepwise, index order.
 *
 * Routing (tilingKey, decided host-side):
 *   1 K_INT_ATOMIC   int8/int32 any dim : copy + SyncAll + SetAtomicAdd<T> scatter.
 *   2 K_FP32_DIRECT  float any dim      : same, SetAtomicAdd<float>.
 *   3 K_FPWS_ATOMIC  fp16/bf16 (dim0|last), ws fits : init fp32 workspace from input,
 *                    SyncAll, SetAtomicAdd<float> scatter source->ws, SyncAll,
 *                    Cast ws(fp32)->T->y.  bit-exact torch fp32-once (incl hot4).
 *   4 K_FP_DIRECT    fp16/bf16 (dim0|last), huge numel : native-dtype atomic to y
 *                    (only reached when Dim huge => collision~0 => passes 0.1% gate).
 *   5 K_STEPWISE     fp16/bf16 MIDDLE dim : owned-rows stepwise index-order (GatherAcc).
 *   6 K_STEPWISE_ALL_OUTER  same, outer>1 and every outer's owned rows fit UB at once (GatherAccAllO).
 *  13 K_INT_PACKED   large-strided workload int32 middle dim   : 8-row double-buffered padded read + atomic write.
 */
#include "kernel_operator.h"

namespace {
constexpr uint32_t FWS_TILE = 4096;             // ProcFpWs phase tile (elements)
constexpr uint32_t STAGE_CAP = 64;              // key5 V2: matched-source rows staged per group
constexpr uint32_t SCATTER_BATCH_BYTES = 8192;  // int8 key 1: consecutive source rows read at once
constexpr uint64_t WS_PREFIX_BYTES = 16ULL << 20;  // reserved prefix for FFTS SyncAll
}  // namespace

template <typename T>
__aicore__ inline void PadInT(const AscendC::LocalTensor<T>& dst,
                              const AscendC::GlobalTensor<T>& src, uint32_t n) {
  const uint32_t blk = 32u / sizeof(T);
  if (n % blk == 0) {
    AscendC::DataCopy(dst, src, n);
  } else {
    AscendC::DataCopyExtParams params{1, n * (uint32_t)sizeof(T), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, params, padParams);
  }
}

template <typename T>
__aicore__ inline void PadOutT(const AscendC::GlobalTensor<T>& dst,
                               const AscendC::LocalTensor<T>& src, uint32_t n) {
  const uint32_t blk = 32u / sizeof(T);
  if (n % blk == 0) {
    AscendC::DataCopy(dst, src, n);
  } else {
    AscendC::DataCopyExtParams params{1, n * (uint32_t)sizeof(T), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, params);
  }
}

template <typename T>
class KernelIndexAdd {
 public:
  __aicore__ inline KernelIndexAdd() {}

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR source,
                              GM_ADDR y, GM_ADDR workspace,
                              __gm__ const IndexAddTilingData* t, AscendC::TPipe* pipeIn) {
    pipe = pipeIn;
    // Eight scalars read straight from the GM tiling blob, once each: GET_TILING_DATA's copy costs
    // its own setup on every core (a few us at 40 cores) regardless of the struct's 32 bytes.
    outer = t->outer;
    dimLen = t->dimLen;
    inner = t->inner;
    m = t->m;
    accInnerTile = t->accInnerTile;
    k7B = t->k7B;
    k7Ntile = t->k7Ntile;
    k7Mtile = t->k7Mtile;
    const uint32_t cores = AscendC::GetBlockNum() < 1 ? 1 : AscendC::GetBlockNum();
    copyTotal = outer * dimLen * inner;
    units = outer * (uint64_t)m;
    copyPerCore = (copyTotal + cores - 1) / cores;
    const uint32_t align = 256u / (uint32_t)sizeof(T);
    copyPerCore = (copyPerCore + align - 1) / align * align;
    if (copyPerCore < align) copyPerCore = align;
    copyTileLen = 8192u < copyPerCore ? 8192u : (uint32_t)copyPerCore;
    copyTileLen = (copyTileLen + align - 1) / align * align;
    innerTile = inner < 8192u ? (uint32_t)inner : 8192u;
    unitsPerCore = (units + cores - 1) / cores;
    if (TILING_KEY_IS(13)) {
      unitsPerCore = (unitsPerCore + 31) / 32 * 32;
      copyTileLen = 8u * (((uint32_t)inner + 7) / 8 * 8);
      if (copyTileLen > copyPerCore) copyTileLen = (uint32_t)copyPerCore;
    } else if (TILING_KEY_IS(12)) {
      copyTileLen = 4096u < copyPerCore ? 4096u : (uint32_t)copyPerCore;
    }
    if (copyTileLen < 1) copyTileLen = 1;
    if (TILING_KEY_IS(5) || TILING_KEY_IS(6)) {
      accRowsPerCore = ((uint32_t)dimLen + cores - 1) / cores;
      if (accRowsPerCore < 1) accRowsPerCore = 1;
    } else {
      accRowsPerCore = 1;
    }
    k7BlocksPerCore = 1;
    if ((TILING_KEY_IS(7) || TILING_KEY_IS(23)) && k7B != 0) {
      uint64_t nBlocks = (outer + k7B - 1) / k7B;
      k7BlocksPerCore = (uint32_t)((nBlocks + cores - 1) / cores);
      if (k7BlocksPerCore < 1) k7BlocksPerCore = 1;
    }
    accStride = (accInnerTile + 7) / 8 * 8;  // 32B-aligned acc row pitch (fp32)
    // key 3 = owned-rows fp32-once (dim0/last); key 5 = owned-rows stepwise (middle dim)
    roundStep = TILING_KEY_IS(5);

    xGm.SetGlobalBuffer((__gm__ T*)x);
    srcGm.SetGlobalBuffer((__gm__ T*)source);
    yGm.SetGlobalBuffer((__gm__ T*)y);
    idxGm.SetGlobalBuffer((__gm__ int32_t*)indices);
    uint64_t sourceBytes = (uint64_t)outer * m * inner * sizeof(T);
    bool onePassInputs = sourceBytes >= (8ULL << 20) &&
                         (TILING_KEY_IS(13) ||
                          (TILING_KEY_IS(7) && std::is_same_v<T, float>));
    if (onePassInputs) {
      xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
      srcGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    // Key 23 replaces C2's STRIDED 160 B column read with a CONTIGUOUS whole-row read.  On that shape
    // srcGm DISABLE measured 480 -> 850 GB/s and -12.7 us of wall; on the strided key-7 fp16 path the
    // same hint is a measured LOSS (+7.7 / +10.4), because a 512 B L2 line spans 256 fp16 and adjacent
    // k-tiles reuse it.  srcGm only: xGm's read shape here was never measured with the hint.
    if (TILING_KEY_IS(23) && sourceBytes >= (8ULL << 20)) {
      srcGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }

    pipe->InitBuffer(idxBuf, ((m + 7) / 8 * 8) * sizeof(int32_t));
    if (TILING_KEY_IS(3)) {
      // K_FPWS: fp32 workspace path.  Use GetUserWorkspace (skips the framework's
      // system-reserved prefix at the exact size, instead of a hardcoded offset).
      wsGm.SetGlobalBuffer((__gm__ float*)AscendC::GetUserWorkspace(workspace));
      pipe->InitBuffer(qLoad, 2, FWS_TILE * sizeof(float));
      pipe->InitBuffer(qStore, 2, FWS_TILE * sizeof(float));
    } else if (TILING_KEY_IS(5)) {
      // owned-rows buffers (middle-dim stepwise). V1: qLoad/qStore hold the WHOLE owned
      // acc-row batch (nRows*accInnerTile) -> per-o acc-row load/store collapse from nRows
      // synced DMAs to ONE contiguous DataCopyPad each. V2: qLoad ALSO stages STAGE_CAP
      // matched source rows in one buffer (many DMAs -> one EnQue/DeQue sync -> vector adds),
      // killing the ~152 per-source synced DMAs (the scalar-bound killer). Size = max of both.
      uint32_t spPitch = (accInnerTile + 15) / 16 * 16;  // bf16 32B-aligned stage-row pitch
      uint32_t qElems = accRowsPerCore * accInnerTile;
      uint32_t stElems = STAGE_CAP * spPitch;
      if (stElems > qElems) qElems = stElems;
      pipe->InitBuffer(qLoad, 2, qElems * sizeof(T));
      pipe->InitBuffer(qStore, 2, accRowsPerCore * accInnerTile * sizeof(T));
      pipe->InitBuffer(accBuf, accRowsPerCore * accStride * sizeof(float));
      pipe->InitBuffer(tmpBuf, accInnerTile * sizeof(float));
      pipe->InitBuffer(rndBuf, accInnerTile * sizeof(T));
      pipe->InitBuffer(matchBuf, ((m + 7) / 8 * 8) * sizeof(int32_t));
    } else if (TILING_KEY_IS(6)) {
      // all-outer owned rows: the host picks this key only when accInnerTile == inner, outer > 1
      // and the whole layout fits UB.  IA-2 layout: [ownedRow, outer, spPitch].  qLoad/qStore
      // hold the complete layout so step 1/3 need one queue synchronization and one wide Cast
      // rather than nRows*outer short Casts.  The source stage for one matched k is its
      // [outer, spPitch] prefix.
      uint32_t spPitch = (accInnerTile + 15) / 16 * 16;  // bf16 32B-aligned stage-row pitch
      uint32_t qElems = accRowsPerCore * accInnerTile;
      uint32_t stElems = STAGE_CAP * spPitch;
      if (stElems > qElems) qElems = stElems;
      uint32_t wide = (uint32_t)outer * spPitch;
      uint32_t layoutElems = accRowsPerCore * wide;
      if (layoutElems > qElems) qElems = layoutElems;
      pipe->InitBuffer(qLoad, 2, qElems * sizeof(T));
      pipe->InitBuffer(qStore, 2, layoutElems * sizeof(T));
      pipe->InitBuffer(accBuf, layoutElems * sizeof(float));
      pipe->InitBuffer(tmpBuf, wide * sizeof(float));
      pipe->InitBuffer(rndBuf, wide * sizeof(T));
    } else if (TILING_KEY_IS(7) || TILING_KEY_IS(23)) {
      // vectorized transpose (inner==1, outer>1): B outer-rows -> lanes.
      uint32_t B = k7B, Nt = k7Ntile, Mt = k7Mtile;
      uint32_t maxTile = Nt > Mt ? Nt : Mt;
      uint32_t Npad = ((uint32_t)dimLen + Nt - 1) / Nt * Nt;
      if constexpr (std::is_same_v<T, half>) {
        // FAST fp16 path (wide-row workload): fp16 accT + hw 16x16 transpose (no Gather offsets, no fp32 acc).
        // WIDE-BURST + PIPELINED retile: host sets Nt=alignUp(N,16) so phase 1/3 move the B rows
        // as ONE contiguous B*N*2 run (446 B rows instead of 128 B blocks) and Mt as wide as UB
        // allows (448 B source blocks).  k7xfer holds TWO load slots so phase 2 issues tile i+1's
        // MTE2 read before tile i's transpose+add barrier -> DMA overlaps compute (the shipped
        // single-slot version is strictly serial: NOVEC 79us + compute 8us == 87us measured).
        // The phase-3 pack outT[B,Nt] ALIASES xfer slot 0 (both load slots are dead in phase 3),
        // so no separate k7gth buffer is needed.
        // WHOLE-ROW DECOUPLE: ONE unified pool replaces (2x xfer + srcT).  Phase 1 stages the
        // B rows at pitch alignUp(N,16)==Nt (B*Nt elems) and phase 3 packs outT[B,Nt] into the
        // same region; phase 2 carves it into 2 load slots + srcT (3*B*Mt elems).  The pool is
        // therefore B*max(Nt, 3*Mt), which UNTIES the phase-1/3 row width from the phase-2 tile
        // (the old 2*B*maxTile + B*Mt made a wide Nt cost 3x and forced B down).
        // accT rows = Nrow = alignUp(N,16): phase 1's last tile writes rows
        // n0_last + alignUp(N-n0_last,16) == alignUp(N,16) when Nt is a multiple of 16, so
        // alignUp(N,Nt) over-allocates.  Nrow*2 == alignUp(N*2,32) is exactly the DMA's
        // 32B-rounded row size, so a [B,Nrow] pack stores as ONE contiguous curB*N*2 run.
        uint32_t Nrow = ((uint32_t)dimLen + 15) / 16 * 16;
        uint32_t p2E = 3u * Mt;                      // phase 2: 2 load slots + srcT
        uint32_t poolE = Nt > p2E ? Nt : p2E;        // phase 1: [B,Nt] staging
        if (poolE < Nrow) poolE = Nrow;              // phase 3: [B,Nrow] pack
        poolE *= B;
        if (TILING_KEY_IS(23)) {
          // key 23 FUSED-RANK layout (ProcFusedRank): lanes of RB=16 rows, accumulator held in RANK
          // order.  pool = 4*SR + 2*m + 4*NR + 3*NR*RB half elems (srcRaw, srcT, plane, gbOff, permF,
          // permI, accNat, xB0, xB1, sB1); C2: 181,696 B + accBuf 7,168 B + idxBuf 4,800 B = 193,664 B.
          // The GM workspace carries the cross-core prologue tables (chunk histograms + gbOff records).
          const uint32_t RBk = 16u;
          const uint32_t NRk = ((uint32_t)dimLen + 15u) / 16u * 16u;
          const uint32_t SRk = RBk * m;
          pipe->InitBuffer(accBuf, NRk * RBk * sizeof(T));
          pipe->InitBuffer(k7xfer, (4u * SRk + 2u * m + 4u * NRk + 3u * NRk * RBk) * sizeof(T));
          wsGm.SetGlobalBuffer((__gm__ float*)AscendC::GetUserWorkspace(workspace));
          (void)poolE;
        } else {
          pipe->InitBuffer(accBuf, Nrow * B * sizeof(T));  // fp16 accT[Nrow,B]
          pipe->InitBuffer(k7xfer, poolE * sizeof(T));     // unified staging pool
        }
      } else if constexpr (std::is_same_v<T, float>) {
        // FAST fp32 path (large-stream workload): TILED accT[Nt,B] (not [Npad,B]) + FastTransB32 everywhere (no Gather
        // offsets).  Small accT frees UB for ~2x B -> half the phase-2 atomic-store count (scalar bound).
        // PIPELINED phase 2: k7xfer holds TWO load slots so the MTE2 read of tile i+1 is issued
        // BEFORE tile i's transpose/atomic-store barrier -> MTE2 overlaps V+MTE3 (measured: the
        // shipped single-slot version serialises them, mte2 106us + mte3 67us == aiv_time 178us).
        // Mt is the phase-2 source tile WIDTH (was hardwired to Nt): Mt=128 fp32 = 512 B DMA
        // blocks instead of 256 B.  k7src holds two Mt*B slots (ping-pong for the stores).
        // UNIFIED POOL (T1): accT / xfer / srcT are strictly sequential in liveness (phase 1 =
        // accT + 1 staging slot; phase 2 = 2 xfer + 2 srcT, accT dead; phase 3 = accT + outF),
        // and the phases already hand off through PipeBarrier<PIPE_ALL>.  ONE pool of
        // max(2*Nt, 4*Mt)*B elems replaces the three separate buffers, which (a) recovers the
        // 28 KB that used to sit dead at the phase-2 peak and (b) UNTIES Nt from the phase-2
        // tile (the old `maxTile` made a wide Nt inflate both xfer slots).
        uint32_t p13 = 2u * Nt;                       // phase 1/3: accT[Nt,B] + [B,Nt] stage/pack
        uint32_t p2 = 4u * Mt;                        // phase 2: 2 xfer[B,Mt] + 2 srcT[Mt,B]
        uint32_t poolE = (p13 > p2 ? p13 : p2) * B;
        pipe->InitBuffer(k7xfer, poolE * sizeof(float));  // unified pool
        (void)maxTile;
        wsGm.SetGlobalBuffer((__gm__ float*)AscendC::GetUserWorkspace(workspace));  // GM atomic accumulator
      } else {
        // bf16 path: fp32 accT[Npad,B] + element-Gather transpose (unused by the 5 cases).
        pipe->InitBuffer(accBuf, (uint32_t)Npad * B * sizeof(float));
        pipe->InitBuffer(k7xfer, B * maxTile * sizeof(T));
        pipe->InitBuffer(k7off, maxTile * B * sizeof(uint32_t));
        pipe->InitBuffer(k7src, Mt * B * sizeof(float));
        pipe->InitBuffer(k7gth, maxTile * B * sizeof(T));
        pipe->InitBuffer(rndBuf, B * sizeof(T));
      }
    } else if (TILING_KEY_IS(12)) {
      // int32 batched-read + Gather-realign: qBind = CopyDirect passthrough; qLoad = K source
      // rows read in one DMA; accBuf = buf2 (Gather-repacked to 32B-aligned rows); k7off =
      // gather offset table.
      uint32_t bstr = ((uint32_t)inner + 7) / 8 * 8;
      pipe->InitBuffer(qBind, 2, copyTileLen * sizeof(T));
      pipe->InitBuffer(qLoad, 2, accInnerTile * inner * sizeof(int32_t));
      pipe->InitBuffer(accBuf, accInnerTile * bstr * sizeof(int32_t));
      pipe->InitBuffer(k7off, accInnerTile * bstr * sizeof(uint32_t));
    } else if (TILING_KEY_IS(13)) {
      // DataCopyPad packs eight source rows to a 32B UB pitch per instruction.
      // Reuse a double-buffered TQueBind for both
      // phases so MTE2->MTE3->reuse dependencies stay framework-managed.
      uint32_t bstr = ((uint32_t)inner + 7) / 8 * 8;
      uint32_t packedElems = accInnerTile * bstr;
      uint32_t bindElems = copyTileLen > packedElems ? copyTileLen : packedElems;
      pipe->InitBuffer(qBind, 2, bindElems * sizeof(T));
      // CONTIGUOUS-READ head slots: one 32B-aligned 8-element slot per row of the batch.
      // The read is now ONE unbroken block (UB pitch == inner*4, not the 32B-rounded 32B pitch),
      // so a row whose UB byte offset is not a multiple of 32 has its leading few elements
      // stored from here instead.  K <= 8 rows * 8 elems * 4 B = 256 B.
      pipe->InitBuffer(rndBuf, accInnerTile * 8u * sizeof(int32_t));
    } else {
      // ProcAtomic (keys 1/2/4): passthrough copy + scatter.
      uint32_t passLen = copyTileLen > innerTile ? copyTileLen : innerTile;
      if constexpr (std::is_same_v<T, int8_t>) {
        // the int8 scatter reads a core's consecutive source rows together, one row per 32B pitch
        uint32_t batch = Int8ScatterBatch();
        uint32_t pitch = ((uint32_t)inner + 31u) / 32u * 32u;
        if (batch >= 2u && passLen < batch * pitch) passLen = batch * pitch;
      }
      pipe->InitBuffer(qBind, 2, passLen * sizeof(T));
    }

    idxLocal = idxBuf.Get<int32_t>();
    AscendC::DataCopyExtParams ip{1, m * (uint32_t)sizeof(int32_t), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> ipad{false, 0, 0, 0};
    AscendC::DataCopyPad(idxLocal, idxGm, ip, ipad);
  }

  __aicore__ inline void Process() {
    if (TILING_KEY_IS(3)) {
      ProcFpWs();   // fp32-workspace atomic scatter (bandwidth-bound, dim0/last)
    } else if (TILING_KEY_IS(5)) {
      GatherAcc();  // owned-rows stepwise (middle dim)
    } else if (TILING_KEY_IS(6)) {
      GatherAccAllO();  // owned-rows stepwise with every outer's rows held at once (middle dim)
    } else if (TILING_KEY_IS(23)) {
      if constexpr (std::is_same_v<T, half>) {
        ProcFusedRank();  // fp16 last-axis: contiguous whole-row read + Gatherb rank lanes + cross-core prologue
      } else if constexpr (sizeof(T) == 4) {
        ProcVecTransAtomic();  // unreachable: the host emits key 23 for fp16 only
      } else {
        ProcVecTranspose();    // unreachable: the host emits key 23 for fp16 only
      }    } else if (TILING_KEY_IS(7)) {
      if constexpr (sizeof(T) == 4) {
        ProcVecTransAtomic();  // large-stream workload fp32 last-axis: transposed GM accumulator + atomic scatter (add on MTE3)
      } else if constexpr (std::is_same_v<T, half>) {
        ProcVecTransposeF16Fast();  // wide-row workload fp16 last-axis: hw 16x16 TransDataTo5HD transpose + fp16 direct add
      } else {
        ProcVecTranspose();  // bf16 last-axis fallback (element-Gather + fp32 per-step round)
      }
    } else if (TILING_KEY_IS(12)) {
      CopyDirect();
      AscendC::SyncAll();
      ScatterBatchedRead();  // batched ~32KB reads (K rows/DMA) then per-row atomic write
    } else if (TILING_KEY_IS(13)) {
      CopyDirect();
      AscendC::SyncAll();
      ScatterPackedRows();  // 8 padded rows/read, no Gather
    } else {
      CopyDirect();
      AscendC::SyncAll();
      ScatterDirect();
    }
  }

 private:
  __aicore__ inline uint64_t DstBase(uint64_t o, uint32_t k) {
    int32_t idx = idxLocal.GetValue(k);
    if (idx < 0) idx = 0;
    if (idx >= (int32_t)dimLen) idx = (int32_t)dimLen - 1;
    return (o * dimLen + (uint64_t)idx) * inner;
  }

  // int8 key 1: source rows per scatter read, each on a 32B pitch within SCATTER_BATCH_BYTES
  // (< 2: one row at a time)
  __aicore__ inline uint32_t Int8ScatterBatch() {
    uint32_t pitch = ((uint32_t)inner + 31u) / 32u * 32u;
    uint64_t fit = inner <= SCATTER_BATCH_BYTES ? SCATTER_BATCH_BYTES / pitch : 0u;
    if (fit > 255u) fit = 255u;
    return (uint32_t)(fit < unitsPerCore ? fit : unitsPerCore);
  }

  // ---- keys 1/2/4: copy input->y, barrier, atomic scatter source->y ----
  __aicore__ inline void CopyDirect() {
    uint64_t start = (uint64_t)AscendC::GetBlockIdx() * copyPerCore;
    uint64_t end = start + copyPerCore;
    if (end > copyTotal) end = copyTotal;
    for (uint64_t c = start; c < end; c += copyTileLen) {
      uint32_t len = (uint32_t)((end - c) < copyTileLen ? (end - c) : copyTileLen);
      AscendC::LocalTensor<T> a = qBind.AllocTensor<T>();
      PadInT(a, xGm[c], len);
      qBind.EnQue(a);
      a = qBind.DeQue<T>();
      PadOutT(yGm[c], a, len);
      qBind.FreeTensor(a);
    }
  }

  __aicore__ inline void ScatterDirect() {
    uint64_t u0 = (uint64_t)AscendC::GetBlockIdx() * unitsPerCore;
    uint64_t u1 = u0 + unitsPerCore;
    if (u1 > units) u1 = units;
    AscendC::SetAtomicAdd<T>();
    if constexpr (std::is_same_v<T, int8_t>) {
      const uint32_t batch = Int8ScatterBatch();
      if (batch >= 2u) {
        // One read for up to `batch` consecutive source rows of one outer, then only the per-row
        // atomic stores: a queue round trip per row was most of this core's scalar work.
        const uint32_t IN = (uint32_t)inner;
        const uint32_t pitch = (IN + 31u) / 32u * 32u;
        for (uint64_t u = u0; u < u1;) {
          uint64_t o = u / m;
          uint32_t k = (uint32_t)(u % m);
          uint64_t segEnd = (o + 1) * m;
          if (segEnd > u1) segEnd = u1;
          uint32_t kb = (segEnd - u) < batch ? (uint32_t)(segEnd - u) : batch;
          AscendC::LocalTensor<T> a = qBind.AllocTensor<T>();
          if (pitch == IN) {
            AscendC::DataCopy(a, srcGm[(o * m + k) * inner], kb * IN);
          } else {
            AscendC::DataCopyExtParams rp{(uint16_t)kb, IN, 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> rpad{false, 0, 0, 0};
            AscendC::DataCopyPad(a, srcGm[(o * m + k) * inner], rp, rpad);
          }
          qBind.EnQue(a);
          a = qBind.DeQue<T>();
          for (uint32_t i = 0; i < kb; i++)
            PadOutT(yGm[DstBase(o, k + i)], a[(uint64_t)i * pitch], IN);
          qBind.FreeTensor(a);
          u += kb;
        }
        AscendC::SetAtomicNone();
        return;
      }
    }
    for (uint64_t u = u0; u < u1; u++) {
      uint64_t o = u / m;
      uint32_t k = (uint32_t)(u % m);
      uint64_t dst = DstBase(o, k);
      uint64_t src = (o * m + k) * inner;
      for (uint64_t c = 0; c < inner; c += innerTile) {
        uint32_t len = (uint32_t)((inner - c) < innerTile ? (inner - c) : innerTile);
        AscendC::LocalTensor<T> a = qBind.AllocTensor<T>();
        PadInT(a, srcGm[src + c], len);
        qBind.EnQue(a);
        a = qBind.DeQue<T>();
        PadOutT(yGm[dst + c], a, len);
        qBind.FreeTensor(a);
      }
    }
    AscendC::SetAtomicNone();
  }

  // ---- key 12: int32 middle-dim -> batched-read atomic scatter ----
  // Same 40-core contiguous-unit partition + atomic writes as ScatterDirect, but reads K
  // consecutive source rows in ONE ~32KB DMA (vs per-row 10.6KB), then atomic-adds each row
  // to y[o, idx[k+i], :].  Bigger, better-amortized reads for the read-bound large-strided workload.
  __aicore__ inline void ScatterBatchedRead() {
   if constexpr (std::is_same_v<T, int32_t>) {
    uint64_t u0 = (uint64_t)AscendC::GetBlockIdx() * unitsPerCore;
    uint64_t u1 = u0 + unitsPerCore;
    if (u1 > units) u1 = units;
    uint32_t N = (uint32_t)dimLen, IN = (uint32_t)inner, K = accInnerTile;
    uint32_t bstr = (IN + 7) / 8 * 8;  // 32B-aligned repacked row pitch
    AscendC::LocalTensor<int32_t> buf2 = accBuf.Get<int32_t>();
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    // Gather offset table (data-independent, built once): repacks contiguous [K,IN] -> 32B-
    // aligned [K,bstr].  off[i*bstr + c] = (i*IN + c)*4 (byte) for c<IN; 0 (unused) for padding.
    AscendC::LocalTensor<int32_t> offI = k7off.Get<uint32_t>().template ReinterpretCast<int32_t>();
    AscendC::Duplicate(offI, (int32_t)0, K * bstr);
    for (uint32_t i = 0; i < K; i++)
      AscendC::ArithProgression<int32_t>(offI[(uint64_t)i * bstr],
                                         (int32_t)(i * IN * 4), (int32_t)4, IN);
    AscendC::SetAtomicAdd<int32_t>();
    uint64_t u = u0;
    while (u < u1) {
      uint32_t o = (uint32_t)(u / m);
      uint64_t segEnd = (uint64_t)(o + 1) * m;  // stay within one o (contiguous source rows)
      if (segEnd > u1) segEnd = u1;
      uint32_t k = (uint32_t)(u % m);
      uint32_t kEnd = k + (uint32_t)(segEnd - u);
      while (k < kEnd) {
        uint32_t Kb = (kEnd - k) < K ? (kEnd - k) : K;
        AscendC::LocalTensor<int32_t> buf = qLoad.template AllocTensor<int32_t>();
        AscendC::DataCopyExtParams rp{1, Kb * IN * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(buf, srcGm[((uint64_t)o * m + k) * IN], rp, pad);  // ONE big ~32KB read
        qLoad.EnQue(buf);
        buf = qLoad.template DeQue<int32_t>();
        AscendC::Gather(buf2, buf, k7off.Get<uint32_t>(), (uint32_t)0, Kb * bstr);  // repack -> aligned rows
        qLoad.FreeTensor(buf);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);  // repack done -> atomic write
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        for (uint32_t i = 0; i < Kb; i++) {
          int32_t j = idxLocal.GetValue(k + i);
          if (j < 0) j = 0;
          if (j >= (int32_t)N) j = (int32_t)N - 1;
          AscendC::DataCopyExtParams wp{1, IN * (uint32_t)sizeof(int32_t), 0, 0, 0};
          AscendC::DataCopyPad(yGm[((uint64_t)o * N + (uint32_t)j) * IN], buf2[(uint64_t)i * bstr], wp);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);  // write done -> next Gather reuse buf2
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        k += Kb;
      }
      u = segEnd;
    }
    AscendC::SetAtomicNone();
   }
  }

  // ---- key 13: int32 middle-dim, batch of K source rows -> atomic scatter ----
  // ONE-DESCRIPTOR CONTIGUOUS READ.  Measured on this machine (279 MB stream, 40 cores):
  //   blockCount=Kb, blockLen=inner*4  -> mte2 396.9 us (703 GB/s), wall 441
  //   Kb SEPARATE blockCount=1 calls   -> mte2 397.0 us (703 GB/s), wall 441   <- same!
  //   ONE blockCount=1 of Kb*inner*4 B -> mte2 374.3 us (746 GB/s), wall 392
  // So the win is not descriptor COUNT -- it is that a single descriptor covering one unbroken
  // GM range AND one unbroken UB range hits the streaming path.  The multi-block form breaks the
  // UB side: it advances by ceil(inner*4/32)*32 per row, and that 16 B of rounding costs 22.5 us
  // of mte2 plus a 26 us wall-minus-aiv tax.
  // The rounding was load-bearing, though: it is what made every store source 32B-aligned.  With
  // a contiguous UB pitch, row i sits at byte i*inner*4, which is 32B-aligned only when
  // (i*inner*4) % 32 == 0.  Fix without giving up contiguity: split such a row's store in two.
  // Let hE = ((32 - (i*inner*4) % 32) % 32)/4  (0..7 elements).  Then buf[i*inner + hE] is at
  // byte i*inner*4 + (32 - mi), which is exactly 0 mod 32 -- the BODY is always aligned.  The
  // hE-element HEAD is scalar-copied into a 32B-aligned slot and stored separately.  Head and
  // body are disjoint halves of the same row, and SetAtomicAdd is order-independent, so the
  // result is bit-identical.  Cost: <=7 GetValue+SetValue pairs and one extra descriptor per
  // misaligned row (mte3 is fully hidden here -- deleting every store moved the wall 1.5 us).
  __aicore__ inline void ScatterPackedRows() {
   if constexpr (std::is_same_v<T, int32_t>) {
    uint64_t u0 = (uint64_t)AscendC::GetBlockIdx() * unitsPerCore;
    uint64_t u1 = u0 + unitsPerCore;
    if (u1 > units) u1 = units;
    uint32_t N = (uint32_t)dimLen, IN = (uint32_t)inner, K = accInnerTile;
    const uint32_t rowB = IN * (uint32_t)sizeof(int32_t);
    AscendC::LocalTensor<int32_t> hd = rndBuf.Get<int32_t>();  // [K,8], each slot 32B-aligned
    AscendC::LocalTensor<int64_t> hd64 = hd.template ReinterpretCast<int64_t>();
    // Clamp all indices once on the vector pipe; the hot per-row loop then
    // performs a single UB read without scalar compare/branch overhead.  idxBuf is a TBuf, so the
    // idx load -> clamp hand-off is not inserted automatically.
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Maxs(idxLocal, idxLocal, (int32_t)0, (int32_t)m);
    AscendC::Mins(idxLocal, idxLocal, (int32_t)N - 1, (int32_t)m);
    AscendC::SetAtomicAdd<int32_t>();
    uint64_t u = u0;
    while (u < u1) {
      uint32_t o = (uint32_t)(u / m);
      uint64_t segEnd = (uint64_t)(o + 1) * m;
      if (segEnd > u1) segEnd = u1;
      uint32_t k = (uint32_t)(u % m);
      uint32_t kEnd = k + (uint32_t)(segEnd - u);
      while (k < kEnd) {
        uint32_t Kb = (kEnd - k) < K ? (kEnd - k) : K;
        AscendC::LocalTensor<int32_t> buf = qBind.template AllocTensor<int32_t>();
        AscendC::DataCopyExtParams rp{1, Kb * rowB, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> rpad{false, 0, 0, 0};
        AscendC::DataCopyPad(buf, srcGm[((uint64_t)o * m + k) * IN], rp, rpad);
        qBind.EnQue(buf);
        buf = qBind.template DeQue<int32_t>();
        AscendC::LocalTensor<int64_t> buf64 = buf.template ReinterpretCast<int64_t>();
        if ((IN & 1u) == 0u) {
          for (uint32_t i = 0; i < Kb; i++) {
            uint32_t hE = ((32u - ((i * rowB) & 31u)) & 31u) >> 2;
            // Even row pitch keeps every row 8B aligned and hE even: pack two
            // int32 head values into one scalar int64 transfer.
            for (uint32_t t = 0; t < hE / 2u; t++)
              hd64.SetValue(i * 4u + t, buf64.GetValue((uint64_t)i * IN / 2u + t));
          }
        } else {
          for (uint32_t i = 0; i < Kb; i++) {
            uint32_t hE = ((32u - ((i * rowB) & 31u)) & 31u) >> 2;
            for (uint32_t t = 0; t < hE; t++)
              hd.SetValue(i * 8u + t, buf.GetValue((uint64_t)i * IN + t));
          }
        }
        for (uint32_t i = 0; i < Kb; i++) {
          int32_t j = idxLocal.GetValue(k + i);
          uint64_t dst = ((uint64_t)o * N + (uint32_t)j) * IN;
          uint32_t hE = ((32u - ((i * rowB) & 31u)) & 31u) >> 2;
          if (hE != 0u) {
            AscendC::DataCopyExtParams hp{1, hE * (uint32_t)sizeof(int32_t), 0, 0, 0};
            AscendC::DataCopyPad(yGm[dst], hd[i * 8u], hp);
          }
          uint32_t body = IN - hE;
          if ((body & 7u) == 0u) {
            AscendC::DataCopy(yGm[dst + hE], buf[(uint64_t)i * IN + hE], body);
          } else {
            AscendC::DataCopyExtParams wp{1, body * (uint32_t)sizeof(int32_t), 0, 0, 0};
            AscendC::DataCopyPad(yGm[dst + hE], buf[(uint64_t)i * IN + hE], wp);
          }
        }
        qBind.FreeTensor(buf);
        k += Kb;
      }
      u = segEnd;
    }
    AscendC::SetAtomicNone();
   }
  }

  // ---- key 3: fp16/bf16 dim0/last -> fp32 workspace, atomic accumulate, cast out ----
  // body guarded: Cast<float,T>/Cast<T,float> only valid for half/bf16; this path is
  // only ever invoked (host tilingKey==3) for sizeof(T)==2.
  __aicore__ inline void ProcFpWs() {
   if constexpr (sizeof(T) == 2) {
    uint64_t start = (uint64_t)AscendC::GetBlockIdx() * copyPerCore;
    uint64_t end = start + copyPerCore;
    if (end > copyTotal) end = copyTotal;

    // phase 1: ws[g] = (fp32) input[g]  (per-core slice of flat numel)
    for (uint64_t g = start; g < end; g += FWS_TILE) {
      uint32_t len = (uint32_t)((end - g) < FWS_TILE ? (end - g) : FWS_TILE);
      AscendC::LocalTensor<T> a = qLoad.AllocTensor<T>();
      PadInT(a, xGm[g], len);
      qLoad.EnQue(a);
      a = qLoad.DeQue<T>();
      AscendC::LocalTensor<float> o = qStore.AllocTensor<float>();
      AscendC::Cast(o, a, AscendC::RoundMode::CAST_NONE, len);
      qLoad.FreeTensor(a);
      qStore.EnQue(o);
      o = qStore.DeQue<float>();
      PadOutT(wsGm[g], o, len);
      qStore.FreeTensor(o);
    }
    AscendC::SyncAll();

    // phase 2: atomic-add (fp32) each source row onto ws
    uint64_t u0 = (uint64_t)AscendC::GetBlockIdx() * unitsPerCore;
    uint64_t u1 = u0 + unitsPerCore;
    if (u1 > units) u1 = units;
    AscendC::SetAtomicAdd<float>();
    for (uint64_t u = u0; u < u1; u++) {
      uint64_t o = u / m;
      uint32_t k = (uint32_t)(u % m);
      uint64_t dst = DstBase(o, k);
      uint64_t src = (o * m + k) * inner;
      for (uint64_t c = 0; c < inner; c += FWS_TILE) {
        uint32_t len = (uint32_t)((inner - c) < FWS_TILE ? (inner - c) : FWS_TILE);
        AscendC::LocalTensor<T> a = qLoad.AllocTensor<T>();
        PadInT(a, srcGm[src + c], len);
        qLoad.EnQue(a);
        a = qLoad.DeQue<T>();
        AscendC::LocalTensor<float> of = qStore.AllocTensor<float>();
        AscendC::Cast(of, a, AscendC::RoundMode::CAST_NONE, len);
        qLoad.FreeTensor(a);
        qStore.EnQue(of);
        of = qStore.DeQue<float>();
        PadOutT(wsGm[dst + c], of, len);  // atomic add (fp32)
        qStore.FreeTensor(of);
      }
    }
    AscendC::SetAtomicNone();
    AscendC::SyncAll();

    // phase 3: y[g] = (T) round(ws[g])  (per-core slice)
    AscendC::DataCopyPadExtParams<float> padf{false, 0, 0, 0};
    for (uint64_t g = start; g < end; g += FWS_TILE) {
      uint32_t len = (uint32_t)((end - g) < FWS_TILE ? (end - g) : FWS_TILE);
      AscendC::LocalTensor<float> a = qLoad.AllocTensor<float>();
      AscendC::DataCopyExtParams ld{1, len * (uint32_t)sizeof(float), 0, 0, 0};
      AscendC::DataCopyPad(a, wsGm[g], ld, padf);
      qLoad.EnQue(a);
      a = qLoad.DeQue<float>();
      AscendC::LocalTensor<T> o = qStore.AllocTensor<T>();
      AscendC::Cast(o, a, AscendC::RoundMode::CAST_RINT, len);
      qLoad.FreeTensor(a);
      qStore.EnQue(o);
      o = qStore.DeQue<T>();
      PadOutT(yGm[g], o, len);
      qStore.FreeTensor(o);
    }
   }  // if constexpr sizeof(T)==2
  }

  // ---- key 5: fp16/bf16 MIDDLE dim -> owned-rows stepwise (index order) ----
  __aicore__ inline void AddOneSource(uint64_t srcOff, uint32_t row, uint32_t ic,
                                      AscendC::LocalTensor<float>& acc,
                                      AscendC::LocalTensor<float>& tmpF,
                                      AscendC::LocalTensor<T>& rnd) {
    AscendC::LocalTensor<T> b = qLoad.AllocTensor<T>();
    PadInT(b, srcGm[srcOff], ic);
    qLoad.EnQue(b);
    b = qLoad.DeQue<T>();
    AscendC::Cast(tmpF, b, AscendC::RoundMode::CAST_NONE, ic);
    qLoad.FreeTensor(b);
    AscendC::Add(acc[row * accStride], acc[row * accStride], tmpF, ic);
    if (roundStep) {
      // middle-dim: torch rounds to T after every add (stepwise, index order).
      AscendC::Cast(rnd, acc[row * accStride], AscendC::RoundMode::CAST_RINT, ic);
      AscendC::Cast(acc[row * accStride], rnd, AscendC::RoundMode::CAST_NONE, ic);
    }
    // dim0/last: accumulate in fp32, round ONCE at write-out (matches torch fp32-once).
  }

  // key 6 all-outer restructure (bf16/fp16 middle-dim, outer>1): acc holds EVERY outer's owned
  // rows [nRows, outer, spPitch] in fp32.  The match list (k where idx[k] in [r0,r1)) is built
  // once; then each matched source j is processed ONCE -- k and dst-row are read a single time
  // (not once per outer), and its column is added to all `outer` accumulators.  This collapses the
  // old outer*cnt scalar index reads to cnt, killing the scalar wall.  Per-step bf16 rounding is
  // preserved: j iterates in k-order and each acc[o,row] receives its adds in k-order.
  __aicore__ inline void GatherAccAllO() {
   if constexpr (sizeof(T) == 2) {
    uint32_t r0 = (uint32_t)AscendC::GetBlockIdx() * accRowsPerCore;
    if (r0 >= dimLen) return;
    uint32_t r1 = r0 + accRowsPerCore;
    if (r1 > (uint32_t)dimLen) r1 = (uint32_t)dimLen;
    uint32_t nRows = r1 - r0;
    uint32_t IN = (uint32_t)inner;
    uint32_t spPitch = (accInnerTile + 15) / 16 * 16;  // bf16 32B-aligned stage-row pitch
    uint32_t wide = (uint32_t)outer * spPitch;          // one owned row across every outer
    constexpr uint32_t VEC_CHUNK = 64 * 255;            // fp32-width repeats, repeatTime <= 255

    AscendC::LocalTensor<float> acc = accBuf.Get<float>();
    AscendC::LocalTensor<float> tmpF = tmpBuf.Get<float>();
    AscendC::LocalTensor<T> rnd = rndBuf.Get<T>();

    // 1. (issued ahead of the match scan so the x read and its cast overlap the scan)
    //    Interleave x[o, owned rows, :] directly as [row, o, spPitch].  Twelve strided DMAs
    //    share one queue synchronization, then one wide Cast initializes the whole acc layout.
    AscendC::LocalTensor<T> a = qLoad.template AllocTensor<T>();
    uint32_t rowGapBlocks = ((uint32_t)outer - 1) * spPitch * sizeof(T) / 32;
    AscendC::DataCopyExtParams ip{(uint16_t)nRows, IN * (uint32_t)sizeof(T), 0,
                                  rowGapBlocks, 0};
    AscendC::DataCopyPadExtParams<T> ipad{true, 0, 0, (T)0};
    for (uint32_t o = 0; o < outer; o++) {
      AscendC::DataCopyPad(a[(uint64_t)o * spPitch],
                           xGm[((uint64_t)o * dimLen + r0) * IN], ip, ipad);
    }
    qLoad.EnQue(a);
    a = qLoad.template DeQue<T>();
    uint32_t layoutElems = nRows * wide;
    for (uint32_t base = 0; base < layoutElems; base += VEC_CHUNK) {
      uint32_t len = layoutElems - base;
      if (len > VEC_CHUNK) len = VEC_CHUNK;
      AscendC::Cast(acc[base], a[base], AscendC::RoundMode::CAST_NONE, len);
    }
    qLoad.FreeTensor(a);

    // 2. Scan idx in k-order and stage matched sources as soon as a group is ready, so their reads
    //    start while the rest of the index is still being scanned (the whole scan used to finish
    //    before the first source read).  A source is src[o,k,:] for all o ([outer, spPitch], each
    //    block 32B-aligned), then vector Add + per-step bf16 round.  A group is at most G sources
    //    (a qLoad buffer holds >= accRowsPerCore*wide elements, i.e. nRows staged sources) and its
    //    strided DMAs share ONE EnQue/DeQue sync; the second qLoad buffer keeps the next group's
    //    reads in flight while V consumes this one.  The 8-wide scan step (4 int64 loads carry 8
    //    int32 indices) overshoots a group by at most 7 matches, which stay pending.  (k, row)
    //    pairs stay in scalar locals, so no match list is written to UB and read back.  Adds run
    //    strictly in match (k) order, so bf16 stepwise rounding is unchanged.  acc is only touched
    //    by vector instructions, which run in issue order, so step 1's Cast needs no barrier.
    AscendC::DataCopyExtParams sp0{(uint16_t)outer, IN * (uint32_t)sizeof(T),
                                   (uint32_t)(((uint64_t)m - 1) * IN * sizeof(T)), 0, 0};
    AscendC::DataCopyPadExtParams<T> pad0{true, 0, 0, (T)0};
    constexpr uint32_t GROUP_CAP = 64;
    const uint32_t G = nRows < GROUP_CAP ? nRows : GROUP_CAP;
    uint32_t pendK[GROUP_CAP + 8], pendRow[GROUP_CAP + 8];
    uint32_t np = 0, k = 0;
    AscendC::LocalTensor<int64_t> idx64 = idxLocal.template ReinterpretCast<int64_t>();
    const int32_t lo = (int32_t)r0, hi = (int32_t)r1;
    while (true) {
      for (; np < G && k + 8 <= m; k += 8) {
        int64_t q0 = idx64.GetValue((k >> 1)), q1 = idx64.GetValue((k >> 1) + 1);
        int64_t q2 = idx64.GetValue((k >> 1) + 2), q3 = idx64.GetValue((k >> 1) + 3);
        int32_t i0 = (int32_t)q0, i1 = (int32_t)(q0 >> 32), i2 = (int32_t)q1, i3 = (int32_t)(q1 >> 32);
        int32_t i4 = (int32_t)q2, i5 = (int32_t)(q2 >> 32), i6 = (int32_t)q3, i7 = (int32_t)(q3 >> 32);
        if (i0 >= lo && i0 < hi) { pendK[np] = k; pendRow[np++] = (uint32_t)(i0 - lo); }
        if (i1 >= lo && i1 < hi) { pendK[np] = k + 1; pendRow[np++] = (uint32_t)(i1 - lo); }
        if (i2 >= lo && i2 < hi) { pendK[np] = k + 2; pendRow[np++] = (uint32_t)(i2 - lo); }
        if (i3 >= lo && i3 < hi) { pendK[np] = k + 3; pendRow[np++] = (uint32_t)(i3 - lo); }
        if (i4 >= lo && i4 < hi) { pendK[np] = k + 4; pendRow[np++] = (uint32_t)(i4 - lo); }
        if (i5 >= lo && i5 < hi) { pendK[np] = k + 5; pendRow[np++] = (uint32_t)(i5 - lo); }
        if (i6 >= lo && i6 < hi) { pendK[np] = k + 6; pendRow[np++] = (uint32_t)(i6 - lo); }
        if (i7 >= lo && i7 < hi) { pendK[np] = k + 7; pendRow[np++] = (uint32_t)(i7 - lo); }
      }
      for (; np < G && k < m; k++) {
        int32_t i = idxLocal.GetValue(k);
        if (i >= lo && i < hi) { pendK[np] = k; pendRow[np++] = (uint32_t)(i - lo); }
      }
      if (np == 0) break;  // index exhausted and nothing pending
      const uint32_t gc = np < G ? np : G;
      AscendC::LocalTensor<T> stg = qLoad.template AllocTensor<T>();
      for (uint32_t t = 0; t < gc; t++)
        AscendC::DataCopyPad(stg[(uint64_t)t * wide], srcGm[(uint64_t)pendK[t] * IN], sp0, pad0);
      qLoad.EnQue(stg);
      stg = qLoad.template DeQue<T>();
      for (uint32_t t = 0; t < gc; t++) {
        AscendC::LocalTensor<float> ar = acc[(uint64_t)pendRow[t] * wide];
        AscendC::LocalTensor<T> st = stg[(uint64_t)t * wide];
        for (uint32_t base = 0; base < wide; base += VEC_CHUNK) {
          uint32_t len = wide - base;
          if (len > VEC_CHUNK) len = VEC_CHUNK;
          AscendC::Cast(tmpF[base], st[base], AscendC::RoundMode::CAST_NONE, len);
          AscendC::Add(ar[base], ar[base], tmpF[base], len);
          AscendC::Cast(rnd[base], ar[base], AscendC::RoundMode::CAST_RINT, len);
          AscendC::Cast(ar[base], rnd[base], AscendC::RoundMode::CAST_NONE, len);
        }
      }
      qLoad.FreeTensor(stg);
      for (uint32_t t = gc; t < np; t++) {  // the scan's overshoot stays pending, in k-order
        pendK[t - gc] = pendK[t];
        pendRow[t - gc] = pendRow[t];
      }
      np -= gc;
    }

    // 3. One wide Cast packs [row, outer, spPitch], then one strided UB->GM DMA per outer.
    AscendC::LocalTensor<T> out = qStore.template AllocTensor<T>();
    for (uint32_t base = 0; base < layoutElems; base += VEC_CHUNK) {
      uint32_t len = layoutElems - base;
      if (len > VEC_CHUNK) len = VEC_CHUNK;
      AscendC::Cast(out[base], acc[base], AscendC::RoundMode::CAST_RINT, len);
    }
    qStore.EnQue(out);
    out = qStore.template DeQue<T>();
    AscendC::DataCopyExtParams op{(uint16_t)nRows, IN * (uint32_t)sizeof(T),
                                  rowGapBlocks, 0, 0};
    for (uint32_t o = 0; o < outer; o++) {
      AscendC::DataCopyPad(yGm[((uint64_t)o * dimLen + r0) * IN],
                           out[(uint64_t)o * spPitch], op);
    }
    qStore.FreeTensor(out);
   }
  }

  __aicore__ inline void GatherAcc() {
   if constexpr (sizeof(T) == 2) {
    uint32_t r0 = (uint32_t)AscendC::GetBlockIdx() * accRowsPerCore;
    if (r0 >= dimLen) return;
    uint32_t r1 = r0 + accRowsPerCore;
    if (r1 > (uint32_t)dimLen) r1 = (uint32_t)dimLen;
    uint32_t nRows = r1 - r0;

    AscendC::LocalTensor<float> acc = accBuf.Get<float>();
    AscendC::LocalTensor<float> tmpF = tmpBuf.Get<float>();
    AscendC::LocalTensor<T> rnd = rndBuf.Get<T>();
    AscendC::LocalTensor<int32_t> matchK = matchBuf.Get<int32_t>();

    bool multiScan = (outer > 1) || ((uint64_t)accInnerTile < inner);
    uint32_t cnt = 0;
    if (multiScan) {
      // manual 4-unroll of the scalar GetValue scan (4 consecutive UB reads pipeline far
      // better than one-at-a-time). idxLocal is UB int32 (8/32B-block).
      uint32_t k = 0;
      for (; k + 4 <= m; k += 4) {
        int32_t i0 = idxLocal.GetValue(k), i1 = idxLocal.GetValue(k + 1);
        int32_t i2 = idxLocal.GetValue(k + 2), i3 = idxLocal.GetValue(k + 3);
        if (i0 >= (int32_t)r0 && i0 < (int32_t)r1) matchK.SetValue(cnt++, (int32_t)k);
        if (i1 >= (int32_t)r0 && i1 < (int32_t)r1) matchK.SetValue(cnt++, (int32_t)(k + 1));
        if (i2 >= (int32_t)r0 && i2 < (int32_t)r1) matchK.SetValue(cnt++, (int32_t)(k + 2));
        if (i3 >= (int32_t)r0 && i3 < (int32_t)r1) matchK.SetValue(cnt++, (int32_t)(k + 3));
      }
      for (; k < m; k++) {
        int32_t i = idxLocal.GetValue(k);
        if (i >= (int32_t)r0 && i < (int32_t)r1) matchK.SetValue(cnt++, (int32_t)k);
      }
    }

    for (uint64_t o = 0; o < outer; o++) {
      uint64_t oRow = o * dimLen;
      uint64_t oSrc = o * m;
      for (uint64_t c = 0; c < inner; c += accInnerTile) {
        uint32_t ic = (uint32_t)((inner - c) < accInnerTile ? (inner - c) : accInnerTile);
        if (ic == inner) {
          // V1: whole inner in one tile -> x[o, r0..r1, :] is nRows*inner CONTIGUOUS in GM.
          // ONE DataCopyPad (single block) instead of nRows queue-synced DMAs; then nRows
          // cheap vector Casts scatter each row to the 32B-aligned accStride pitch.
          AscendC::LocalTensor<T> a = qLoad.AllocTensor<T>();
          PadInT(a, xGm[(oRow + r0) * inner], nRows * ic);
          qLoad.EnQue(a);
          a = qLoad.DeQue<T>();
          for (uint32_t r = 0; r < nRows; r++)
            AscendC::Cast(acc[r * accStride], a[r * ic], AscendC::RoundMode::CAST_NONE, ic);
          qLoad.FreeTensor(a);
        } else {
          for (uint32_t r = 0; r < nRows; r++) {  // c-tiled fallback (inner > accInnerTile)
            AscendC::LocalTensor<T> a = qLoad.AllocTensor<T>();
            PadInT(a, xGm[(oRow + r0 + r) * inner + c], ic);
            qLoad.EnQue(a);
            a = qLoad.DeQue<T>();
            AscendC::Cast(acc[r * accStride], a, AscendC::RoundMode::CAST_NONE, ic);
            qLoad.FreeTensor(a);
          }
        }
        if (multiScan) {
          // V2: stage matched source rows into ONE qLoad buffer (many DMAs -> one EnQue/DeQue
          // sync), then vector-only Add + per-step round reading from UB. Kills the ~152
          // per-source synced tiny DMAs. Grouped by STAGE_CAP; adds run in matchK (index)
          // order so the bf16 per-step rounding is bit-identical.
          uint32_t sp = (accInnerTile + 15) / 16 * 16;
          for (uint32_t g = 0; g < cnt; g += STAGE_CAP) {
            uint32_t gc = (cnt - g) < STAGE_CAP ? (cnt - g) : STAGE_CAP;
            AscendC::LocalTensor<T> stg = qLoad.AllocTensor<T>();
            AscendC::DataCopyExtParams sp0{1, ic * (uint32_t)sizeof(T), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> pad0{false, 0, 0, 0};
            for (uint32_t j = 0; j < gc; j++) {
              uint32_t k = (uint32_t)matchK.GetValue(g + j);
              AscendC::DataCopyPad(stg[j * sp], srcGm[(oSrc + (uint64_t)k) * inner + c], sp0, pad0);
            }
            qLoad.EnQue(stg);
            stg = qLoad.DeQue<T>();
            for (uint32_t j = 0; j < gc; j++) {
              uint32_t k = (uint32_t)matchK.GetValue(g + j);
              uint32_t row = (uint32_t)idxLocal.GetValue(k) - r0;
              AscendC::Cast(tmpF, stg[j * sp], AscendC::RoundMode::CAST_NONE, ic);
              AscendC::Add(acc[row * accStride], acc[row * accStride], tmpF, ic);
              AscendC::Cast(rnd, acc[row * accStride], AscendC::RoundMode::CAST_RINT, ic);
              AscendC::Cast(acc[row * accStride], rnd, AscendC::RoundMode::CAST_NONE, ic);
            }
            qLoad.FreeTensor(stg);
          }
        } else {
          for (uint32_t k = 0; k < m; k++) {
            int32_t i = idxLocal.GetValue(k);
            if (i < (int32_t)r0 || i >= (int32_t)r1) continue;
            AddOneSource((oSrc + (uint64_t)k) * inner + c, (uint32_t)i - r0, ic, acc,
                         tmpF, rnd);
          }
        }
        if (ic == inner) {
          // V1: nRows cheap Casts into a tight buffer, then ONE contiguous DataCopyPad store
          // (y[o, r0..r1, :] is nRows*inner contiguous).
          AscendC::LocalTensor<T> out = qStore.AllocTensor<T>();
          for (uint32_t r = 0; r < nRows; r++)
            AscendC::Cast(out[r * ic], acc[r * accStride], AscendC::RoundMode::CAST_RINT, ic);
          qStore.EnQue(out);
          out = qStore.DeQue<T>();
          PadOutT(yGm[(oRow + r0) * inner], out, nRows * ic);
          qStore.FreeTensor(out);
        } else {
          for (uint32_t r = 0; r < nRows; r++) {  // c-tiled fallback
            AscendC::LocalTensor<T> out = qStore.AllocTensor<T>();
            AscendC::Cast(out, acc[r * accStride], AscendC::RoundMode::CAST_RINT, ic);
            qStore.EnQue(out);
            out = qStore.DeQue<T>();
            PadOutT(yGm[(oRow + r0 + r) * inner + c], out, ic);
            qStore.FreeTensor(out);
          }
        }
      }
    }
   }  // if constexpr sizeof(T)==2
  }

  // ---- key 7: vectorized transpose (inner==1, outer>1) ----
  // load curRows rows of curCols elems (GM rows gmStride apart) into ub, packed;
  // returns UB row pitch = AlignUp(curCols, 32/sizeof(T)) (auto 32B pad, DMA UB side).
  __aicore__ inline uint32_t LoadRowsPacked(const AscendC::LocalTensor<T>& ub,
                                            const AscendC::GlobalTensor<T>& gm,
                                            uint32_t curRows, uint32_t curCols,
                                            uint32_t gmStride) {
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);  // prior transpose(V) read done -> overwrite
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
    AscendC::DataCopyExtParams p{(uint16_t)curRows, curCols * (uint32_t)sizeof(T),
                                 (gmStride - curCols) * (uint32_t)sizeof(T), 0, 0};
    AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    AscendC::DataCopyPad(ub, gm, p, pad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);  // load(MTE2) -> transpose(V)
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
    uint32_t lg = 32u / (uint32_t)sizeof(T);
    if (lg < 1) lg = 1;
    return (curCols + lg - 1) / lg * lg;
  }

  // gather-transpose src[srcRows, srcCols(pitch)] -> dst[srcCols, srcRows]:
  // dst[c*srcRows+r] = src[r*srcCols+c].  ArithProgression dst pitch = srcRows must
  // be a multiple of 8 (32B-aligned).  Gather offsets are byte offsets into src.
  template <typename ST>
  __aicore__ inline void GTr(const AscendC::LocalTensor<ST>& dst,
                             const AscendC::LocalTensor<ST>& src, uint32_t srcRows,
                             uint32_t srcCols) {
    uint32_t sz = (uint32_t)sizeof(ST);
    // The offset table is data-independent (only (srcRows,srcCols,sz)).  Cache it:
    // consecutive same-geometry transposes (all full N/M tiles) reuse it, turning
    // srcCols ArithProgression calls per tile into one build per geometry change.
    if (!(srcRows == gRows_ && srcCols == gCols_ && sz == gSz_)) {
      AscendC::LocalTensor<int32_t> offI =
          k7off.Get<uint32_t>().template ReinterpretCast<int32_t>();
      for (uint32_t c = 0; c < srcCols; c++)
        AscendC::ArithProgression<int32_t>(offI[c * srcRows], (int32_t)(c * sz),
                                           (int32_t)(srcCols * sz), srcRows);
      gRows_ = srcRows;
      gCols_ = srcCols;
      gSz_ = sz;
    }
    AscendC::Gather(dst, src, k7off.Get<uint32_t>(), (uint32_t)0, srcCols * srcRows);
  }

  __aicore__ inline void ProcVecTranspose() {
    const uint32_t B = k7B, Nt = k7Ntile, Mt = k7Mtile;
    const uint32_t N = (uint32_t)dimLen, M = m;
    constexpr bool F16 = (sizeof(T) == 2);
    uint64_t nBlocks = (outer + B - 1) / B;
    uint64_t blk0 = (uint64_t)AscendC::GetBlockIdx() * k7BlocksPerCore;
    uint64_t blk1 = blk0 + k7BlocksPerCore;
    if (blk1 > nBlocks) blk1 = nBlocks;
    AscendC::LocalTensor<float> accT = accBuf.Get<float>();
    AscendC::LocalTensor<float> srcT = k7src.Get<float>();
    AscendC::LocalTensor<T> xfer = k7xfer.Get<T>();

    for (uint64_t blk = blk0; blk < blk1; blk++) {
      uint32_t o0 = (uint32_t)(blk * B);
      uint32_t curB = (o0 + B > outer) ? (uint32_t)(outer - o0) : B;
      // phase 1: input[curB,N] -> accT[N,B] (transpose per N-tile; lanes b>=curB garbage, unused)
      for (uint32_t n0 = 0; n0 < N; n0 += Nt) {
        uint32_t vN = (n0 + Nt > N) ? (N - n0) : Nt;
        uint32_t P = LoadRowsPacked(xfer, xGm[(uint64_t)o0 * N + n0], curB, vN, N);
        if constexpr (F16) {
          AscendC::LocalTensor<T> gth = k7gth.Get<T>();
          GTr<T>(gth, xfer, B, P);
          AscendC::Cast(accT[(uint64_t)n0 * B], gth, AscendC::RoundMode::CAST_NONE, P * B);
        } else {
          GTr<float>(accT[(uint64_t)n0 * B], xfer.template ReinterpretCast<float>(), B, P);
        }
      }
      // phase 2: per M-tile, transpose source -> srcT[Mt,B]; per-k B-wide add (k-order)
      for (uint32_t m0 = 0; m0 < M; m0 += Mt) {
        uint32_t vM = (m0 + Mt > M) ? (M - m0) : Mt;
        uint32_t Pm = LoadRowsPacked(xfer, srcGm[(uint64_t)o0 * M + m0], curB, vM, M);
        if constexpr (F16) {
          AscendC::LocalTensor<T> gth = k7gth.Get<T>();
          GTr<T>(gth, xfer, B, Pm);
          AscendC::Cast(srcT, gth, AscendC::RoundMode::CAST_NONE, Pm * B);
        } else {
          GTr<float>(srcT, xfer.template ReinterpretCast<float>(), B, Pm);
        }
        // Q1: unroll by 4 with the four idx scalar reads hoisted to the top, so the scalar
        // pipe issues them ahead of (and overlapping) the vector Adds.  Per-step round order
        // is preserved: each add is immediately followed by its round in k-order, so even a
        // j-collision within the group stays add,round,add,round.
        uint32_t kc = 0;
        for (; kc + 4 <= vM; kc += 4) {
          int32_t j0 = idxLocal.GetValue(m0 + kc), j1 = idxLocal.GetValue(m0 + kc + 1);
          int32_t j2 = idxLocal.GetValue(m0 + kc + 2), j3 = idxLocal.GetValue(m0 + kc + 3);
          j0 = j0 < 0 ? 0 : (j0 >= (int32_t)N ? (int32_t)N - 1 : j0);
          j1 = j1 < 0 ? 0 : (j1 >= (int32_t)N ? (int32_t)N - 1 : j1);
          j2 = j2 < 0 ? 0 : (j2 >= (int32_t)N ? (int32_t)N - 1 : j2);
          j3 = j3 < 0 ? 0 : (j3 >= (int32_t)N ? (int32_t)N - 1 : j3);
          AscendC::Add(accT[(uint64_t)j0 * B], accT[(uint64_t)j0 * B], srcT[(uint64_t)kc * B], B);
          if constexpr (F16) { AscendC::LocalTensor<T> r = rndBuf.Get<T>(); AscendC::Cast(r, accT[(uint64_t)j0 * B], AscendC::RoundMode::CAST_RINT, B); AscendC::Cast(accT[(uint64_t)j0 * B], r, AscendC::RoundMode::CAST_NONE, B); }
          AscendC::Add(accT[(uint64_t)j1 * B], accT[(uint64_t)j1 * B], srcT[(uint64_t)(kc + 1) * B], B);
          if constexpr (F16) { AscendC::LocalTensor<T> r = rndBuf.Get<T>(); AscendC::Cast(r, accT[(uint64_t)j1 * B], AscendC::RoundMode::CAST_RINT, B); AscendC::Cast(accT[(uint64_t)j1 * B], r, AscendC::RoundMode::CAST_NONE, B); }
          AscendC::Add(accT[(uint64_t)j2 * B], accT[(uint64_t)j2 * B], srcT[(uint64_t)(kc + 2) * B], B);
          if constexpr (F16) { AscendC::LocalTensor<T> r = rndBuf.Get<T>(); AscendC::Cast(r, accT[(uint64_t)j2 * B], AscendC::RoundMode::CAST_RINT, B); AscendC::Cast(accT[(uint64_t)j2 * B], r, AscendC::RoundMode::CAST_NONE, B); }
          AscendC::Add(accT[(uint64_t)j3 * B], accT[(uint64_t)j3 * B], srcT[(uint64_t)(kc + 3) * B], B);
          if constexpr (F16) { AscendC::LocalTensor<T> r = rndBuf.Get<T>(); AscendC::Cast(r, accT[(uint64_t)j3 * B], AscendC::RoundMode::CAST_RINT, B); AscendC::Cast(accT[(uint64_t)j3 * B], r, AscendC::RoundMode::CAST_NONE, B); }
        }
        for (; kc < vM; kc++) {
          int32_t j = idxLocal.GetValue(m0 + kc);
          if (j < 0) j = 0;
          if (j >= (int32_t)N) j = (int32_t)N - 1;
          AscendC::Add(accT[(uint64_t)j * B], accT[(uint64_t)j * B], srcT[(uint64_t)kc * B], B);
          if constexpr (F16) {
            AscendC::LocalTensor<T> rnd = rndBuf.Get<T>();
            AscendC::Cast(rnd, accT[(uint64_t)j * B], AscendC::RoundMode::CAST_RINT, B);
            AscendC::Cast(accT[(uint64_t)j * B], rnd, AscendC::RoundMode::CAST_NONE, B);
          }
        }
      }
      // phase 3: accT[N,B] -> y[curB,N] (transpose back per N-tile)
      for (uint32_t n0 = 0; n0 < N; n0 += Nt) {
        uint32_t vN = (n0 + Nt > N) ? (N - n0) : Nt;
        AscendC::LocalTensor<float> outF = srcT;  // reuse [Mt,B] as [B,Nt] (Nt<=Mt)
        GTr<float>(outF, accT[(uint64_t)n0 * B], Nt, B);  // outF[B,Nt] = transpose accT tile
        AscendC::LocalTensor<T> outT = xfer;
        if constexpr (F16) {
          AscendC::Cast(outT, outF, AscendC::RoundMode::CAST_RINT, B * Nt);
        }
        // store curB rows of vN (UB pitch Nt) to y (GM rows N apart)
        AscendC::LocalTensor<T> src = F16 ? outT : outF.template ReinterpretCast<T>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);  // compute(V) -> store(MTE3)
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        if (vN == Nt) {
          AscendC::DataCopyExtParams sp{(uint16_t)curB, Nt * (uint32_t)sizeof(T), 0,
                                        (N - Nt) * (uint32_t)sizeof(T), 0};
          AscendC::DataCopyPad(yGm[(uint64_t)o0 * N + n0], src, sp);
        } else {
          for (uint32_t b = 0; b < curB; b++)
            PadOutT(yGm[(uint64_t)(o0 + b) * N + n0], src[(uint64_t)b * Nt], vN);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);  // store done -> reuse buf
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
      }
    }
  }

  // ===== wide-row workload fp16 FAST path: hardware 16x16 TransDataTo5HD transpose + fp16 direct add =====
  // Replaces the element-Gather transpose (GTr ~1 elem/cyc = the 296us vec-bound wall) with the
  // hardware 16x16 b16 transpose (~16 elem/cyc).  Accumulator stays fp16: a single fp16 hardware
  // Add == round_fp16(a+b) == torch per-step fp16 round (bit-identical to the old fp32-add +
  // RINT-cast).  So 1 vec op/step replaces 3, and the transposes need NO cast.
  //   src[R,C](rowpitch sp) -> dst[C,R](rowpitch dp).  All list addresses are 16-elem (32B)
  //   aligned because sp,dp are multiples of 16 (P=alignUp(.,16), B/Nt mult of 16).  Partial
  //   R/C blocks transpose garbage into pad rows/cols that are never stored.
  __aicore__ inline void FastTransB16(const AscendC::LocalTensor<T>& dst,
                                      const AscendC::LocalTensor<T>& src, uint32_t R,
                                      uint32_t C, uint32_t sp, uint32_t dp) {
    uint32_t rb = (R + 15) / 16, cb = (C + 15) / 16;
    // repeatTimes=cb folds all cb column-blocks of a 16-row strip into ONE call, so the
    // scalar address-list build (16+16 GetPhyAddr) happens once per row-block, not per
    // 16x16 block (cb-fold cut = the scalar bound after the transpose fix).  Strides are in
    // 32B(=16 fp16) block units: src next col-block = +16 elem = 1 block; dst next row-group
    // = +16*dp elem = dp blocks.
    AscendC::TransDataTo5HDParams p;
    p.repeatTimes = (uint8_t)cb;
    p.srcRepStride = (cb == 1) ? 0 : 1;
    p.dstRepStride = (cb == 1) ? 0 : (uint16_t)dp;
    uint64_t srcList[16], dstList[16];
    const uint64_t srcBase = (uint64_t)src.GetPhyAddr();
    const uint64_t dstBase = (uint64_t)dst.GetPhyAddr();
    for (uint32_t bi = 0; bi < rb; bi++) {
      for (uint32_t r = 0; r < 16; r++)
        srcList[r] = srcBase + ((uint64_t)(bi * 16 + r) * sp) * sizeof(T);
      for (uint32_t c = 0; c < 16; c++)
        dstList[c] = dstBase + ((uint64_t)c * dp + bi * 16) * sizeof(T);
      AscendC::TransDataTo5HD<T>(dstList, srcList, p);
    }
  }

  __aicore__ inline void ProcFusedRank() {
    const uint32_t B = k7B;
    const uint32_t N = (uint32_t)dimLen;
    const uint32_t M = m;
    const uint32_t RB = 16u;
    const uint32_t NR = (N + 15u) / 16u * 16u;
    const uint32_t SR = RB * M;
    uint64_t nBlocks = (outer + B - 1) / B;
    uint64_t blk0 = (uint64_t)AscendC::GetBlockIdx() * k7BlocksPerCore;
    uint64_t blk1 = blk0 + k7BlocksPerCore;
    if (blk1 > nBlocks) blk1 = nBlocks;
    AscendC::LocalTensor<T> accR = accBuf.Get<T>();
    AscendC::LocalTensor<T> pool = k7xfer.Get<T>();
    AscendC::LocalTensor<T> srcRaw = pool;
    AscendC::LocalTensor<T> srcT = pool[SR];
    AscendC::LocalTensor<T> plane = pool[2u * SR];
    AscendC::LocalTensor<uint32_t> gbOff = pool[3u * SR].template ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> permF =
        pool[3u * SR + 2u * M].template ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<uint32_t> permI =
        pool[3u * SR + 2u * M + 2u * NR].template ReinterpretCast<uint32_t>();
    AscendC::LocalTensor<T> accNat = pool[3u * SR + 2u * M + 4u * NR];
    // iaq5: two x slots and two source slots.  sB0 aliases srcRaw (pool[0]).
    AscendC::LocalTensor<T> xB0 = pool[3u * SR + 2u * M + 4u * NR + NR * RB];
    AscendC::LocalTensor<T> xB1 = pool[3u * SR + 2u * M + 4u * NR + 2u * NR * RB];
    AscendC::LocalTensor<T> sB1 = pool[3u * SR + 2u * M + 4u * NR + 3u * NR * RB];
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Maxs(idxLocal, idxLocal, (int32_t)0, M);
    AscendC::Mins(idxLocal, idxLocal, (int32_t)N - 1, M);
    // iaq13: issue the FIRST sub-block's loads BEFORE the scalar prologue.  The prologue is
    // pure scalar and the loads are pure MTE2, so they can overlap -- but today they do not:
    // iaq5's wall 110.83 is almost exactly iaq6's 62.53 plus the prologue's 48.30, a fully
    // serial addition.  iaq6 is transfer bound (mte2 51.9 of a 62.5 wall), so there is real
    // transfer time available to hide under this scalar.
    // Only the FIRST block is hoisted; the in-loop prime still runs for every later block.
    {
      AscendC::DataCopyPadExtParams<T> pd0{false, 0, 0, 0};
      if (blk0 < blk1) {
        uint32_t o0f = (uint32_t)(blk0 * B);
        uint32_t curBf = (o0f + B > outer) ? (uint32_t)(outer - o0f) : B;
        uint32_t rows0 = (curBf < RB) ? curBf : RB;
        AscendC::DataCopyExtParams xp0{(uint16_t)rows0, N * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(xB0, xGm[(uint64_t)o0f * N], xp0, pd0);
        AscendC::DataCopyExtParams s0{1, rows0 * M * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(srcRaw, srcGm[(uint64_t)o0f * M], s0, pd0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      }
    }
    uint16_t cnt[256];
    uint16_t rankOf[256];
    // L (the largest repeat count of any one column) is DATA-dependent and can reach M, so the
    // per-level tables are sized for the host bound M <= 2040.  hist/nextR/cL hold column counts
    // <= N <= 255 (host-gated), so they are uint8; sL holds element offsets <= M, so it is uint16.
    constexpr uint32_t LCAP = 2048u;
    uint8_t hist[LCAP];
    uint8_t nextR[LCAP];
    uint8_t cL[LCAP];
    uint16_t sL[LCAP];
    // ===== iaq13: CROSS-CORE PROLOGUE =====================================================
    // Measured (iaq6/iaq8): the serial prologue costs 49.5 us PER CORE = the two 1200-iteration
    // loops (index scan+histogram, gbOff build) 41.5 us + the 224-column group 8.03 us.  Every
    // core needs the SAME global tables, which is why they were computed redundantly 40 times.
    // The two 1200-loops are now split by k across the cores and shared through GM workspace:
    //   R1: core j scans only k in [k0,k1) and stores that chunk's histogram at ws[j*NR]; SyncAll.
    //   R2: every core loads all nCh chunk histograms (ONE DMA), vector-reduces the global count
    //       and its own chunk prefix, runs the 224-group on the global count, and for each k in
    //       its chunk computes occ = prefix[c] + local repeats -- exactly the serial occurrence
    //       count -- so gbOff[sL[occ] + rankOf[c]] = k*32 is bit-identical to the serial loop.
    //       Each entry is stored at its own ws[gbBase + slot]; SyncAll; every core loads the
    //       complete gbOff (ONE DMA).
    // Coverage: the chunks partition [0,M) and slot = sL[occ]+rankOf[c] is a bijection onto
    // [0,M), so every gbOff entry is written by exactly one core.  (Degenerate-permutation rule:
    // Gatherb must never see an unwritten entry -- 507015 "ub address out of bounds".)
    // UB scratch: srcT (hAll, nCh*NR*4 = 35,840 B <= 38,400 for C2) and plane (hW/pre/tot/val)
    // are dead until the block loop; the hoisted first-block loads live in xB0/srcRaw.
    // Flags: every new flag is an ADJACENT Set+Wait pair on EVENT_ID6/7, so none can match the
    // hoisted MTE2_V(EVENT_ID0) that stays pending until the first sub-block.
    // Stack / uint8-repeat / UB-carve invariants (chK <= CH_CAP, M <= 2040, M % 8 == 0, N <= 255, ...) are enforced by the host gate.
    const uint32_t nCores = ((uint32_t)AscendC::GetBlockNum() < 1u) ? 1u : (uint32_t)AscendC::GetBlockNum();
    const uint32_t jCore = (uint32_t)AscendC::GetBlockIdx();
    const uint32_t chK = (M + nCores - 1u) / nCores;
    const uint32_t nCh = (M + chK - 1u) / chK;
    const uint32_t k0 = jCore * chK;
    const uint32_t k1 = (k0 >= M) ? k0 : ((k0 + chK > M) ? M : k0 + chK);
    const uint64_t gbBase = (uint64_t)nCores * NR;
    AscendC::LocalTensor<int32_t> hAll = srcT.template ReinterpretCast<int32_t>();
    AscendC::LocalTensor<float> hAllF = srcT.template ReinterpretCast<float>();
    AscendC::LocalTensor<int32_t> pI = plane.template ReinterpretCast<int32_t>();
    AscendC::LocalTensor<int32_t> hW = pI;
    AscendC::LocalTensor<float> hWF = plane.template ReinterpretCast<float>();
    AscendC::LocalTensor<int32_t> pre = pI[NR];
    AscendC::LocalTensor<int32_t> tot = pI[2u * NR];
    AscendC::LocalTensor<int32_t> val = pI[3u * NR];
    AscendC::LocalTensor<float> valF = hWF[3u * NR];
    AscendC::LocalTensor<float> gbOffF = gbOff.template ReinterpretCast<float>();
    constexpr uint32_t CH_CAP = 256u;
    uint16_t cK[CH_CAP];
    uint16_t jK[CH_CAP];
    uint16_t lh[256];
    for (uint32_t i = 0; i < NR; i++) { lh[i] = 0; rankOf[i] = 0; }
    // ---- R1: own-chunk histogram -> ws[jCore*NR] ----
    if (k0 < k1) {
      AscendC::Duplicate(hW, (int32_t)0, NR);
      for (uint32_t k = k0; k < k1; k++) {
        uint32_t c = (uint32_t)idxLocal.GetValue(k);
        cK[k - k0] = (uint16_t)c;
        lh[c] = (uint16_t)(lh[c] + 1u);
      }
      for (uint32_t k = k0; k < k1; k++) {
        uint32_t c = (uint32_t)cK[k - k0];
        if (lh[c] != 0u) { hW.SetValue(c, (int32_t)lh[c]); lh[c] = 0; }
      }
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID6);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID6);
      AscendC::DataCopy(wsGm[(uint64_t)jCore * NR], hWF, NR);
    }
    AscendC::SyncAll();
    // ---- R2a: load every chunk histogram, reduce global count and own prefix ----
    {
      AscendC::DataCopyExtParams hp{1, nCh * NR * 4u, 0, 0, 0};
      AscendC::DataCopyPadExtParams<float> hpd{false, 0, 0, 0};
      AscendC::DataCopyPad(hAllF, wsGm[0], hp, hpd);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
    }
    const uint32_t jj = (jCore < nCh) ? jCore : nCh;
    AscendC::Duplicate(pre, (int32_t)0, NR);
    for (uint32_t i = 0; i < jj; i++) AscendC::Add(pre, pre, hAll[i * NR], NR);
    AscendC::Duplicate(tot, (int32_t)0, NR);
    AscendC::Add(tot, tot, pre, NR);
    for (uint32_t i = jj; i < nCh; i++) AscendC::Add(tot, tot, hAll[i * NR], NR);
    for (uint32_t i = 0; i < N; i++) cnt[i] = (uint16_t)tot.GetValue(i);
    for (uint32_t i = N; i < NR; i++) cnt[i] = 0;
    uint32_t L = 0;
    for (uint32_t i = 0; i < N; i++) if ((uint32_t)cnt[i] > L) L = (uint32_t)cnt[i];
    for (uint32_t v = 0; v <= L; v++) hist[v] = 0;
    for (uint32_t i = 0; i < N; i++) hist[cnt[i]] = (uint16_t)(hist[cnt[i]] + 1u);
    uint32_t racc = 0;
    for (uint32_t d = 0; d <= L; d++) {
      uint32_t v = L - d;
      nextR[v] = (uint16_t)racc;
      racc += (uint32_t)hist[v];
    }
    for (uint32_t i = 0; i < N; i++) {
      uint32_t v = (uint32_t)cnt[i];
      rankOf[i] = nextR[v];
      nextR[v] = (uint16_t)(nextR[v] + 1u);
    }
    // cL[l] = number of columns with MORE than l occurrences = suffix sum of hist: O(L), not O(L^2).
    {
      uint32_t suf = 0;
      for (uint32_t v = L; v >= 1u; v--) { suf += (uint32_t)hist[v]; cL[v - 1u] = (uint8_t)suf; }
    }
    uint32_t sacc = 0;
    for (uint32_t l = 0; l < L; l++) { sL[l] = (uint16_t)sacc; sacc += (uint32_t)cL[l]; }
    // iaq13: rankOf is a bijection of [0,N) onto [0,N) -- each count level gets a disjoint rank
    // range and the levels sum to N -- so the fill loop below writes every permF/permI row < N.
    // Only the NR-N padding rows need zeroing: 2 SetValues for C2 instead of 448 (~2.8 us).
    for (uint32_t r = N; r < NR; r++) { permF.SetValue(r, 0u); permI.SetValue(r, 0u); }
    for (uint32_t i = 0; i < N; i++) {
      permF.SetValue((uint32_t)rankOf[i], i * 32u);
      permI.SetValue(i, (uint32_t)rankOf[i] * 32u);
    }
    // ---- R2b: own-chunk gbOff entries -> their scattered ws slots ----
    // lh[] is all-zero again here (R1 zeroed each bin as it wrote it), so it now counts repeats
    // of a column WITHIN this chunk, while pre[c] counts every earlier chunk.
    uint32_t nV = 0;
    if (k0 < k1) {
      for (uint32_t k = k0; k < k1; k++) {
        uint32_t c = (uint32_t)cK[k - k0];
        uint32_t occ = (uint32_t)pre.GetValue(c) + (uint32_t)lh[c];
        lh[c] = (uint16_t)(lh[c] + 1u);
        jK[nV] = (uint16_t)((uint32_t)sL[occ] + (uint32_t)rankOf[c]);
        val.SetValue(8u * nV, (int32_t)(k * 32u));  // 32 B pitch: every store source is 32B-aligned
        nV++;
      }
      AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID7);
      AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID7);
      // iaq13: ONE FULL 32 B RECORD PER SLOT.  iaq10 stored a single float (4 B) per slot and
      // failed acc while the histogram half was proven correct (iaq12 pass, iaq11 +V_S still
      // fail).  A 4 B store appears to be issued as 32 B, overwriting the next 7 slots that
      // other cores write concurrently.  Here slot s owns GM bytes [8s*4, 8s*4+32): the store is
      // exactly 32 B from this value's own 32B-aligned UB block, so no two stores can overlap.
      for (uint32_t i = 0; i < nV; i++) {
        AscendC::DataCopy(wsGm[gbBase + 8u * (uint64_t)jK[i]], valF[8u * i], 8u);
      }
    }
    AscendC::SyncAll();
    // ---- R2c: every core loads the sparse record table and compacts it into gbOff ----
    // M*32 B == 2*SR bytes == exactly the srcT region, for any M (SR = 16*M half elements).
    // Offsets 0,32,64.. live in plane right after the value slots: int32 index 3NR + 8*chK
    // (32B-aligned), 4*M bytes, disjoint from hW/pre/tot/val.  Host gate: 12NR + 32chK + 4M <= 32M.
    {
      AscendC::LocalTensor<float> sprF = srcT.template ReinterpretCast<float>();
      AscendC::LocalTensor<int32_t> sprI = srcT.template ReinterpretCast<int32_t>();
      AscendC::DataCopyExtParams gl{1, M * 32u, 0, 0, 0};
      AscendC::DataCopyPadExtParams<float> gpd{false, 0, 0, 0};
      AscendC::DataCopyPad(sprF, wsGm[gbBase], gl, gpd);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID7);
      AscendC::LocalTensor<int32_t> offI = pI[3u * NR + 8u * chK];
      AscendC::LocalTensor<uint32_t> offU = offI.template ReinterpretCast<uint32_t>();
      AscendC::LocalTensor<int32_t> gbOffI = gbOff.template ReinterpretCast<int32_t>();
      AscendC::ArithProgression<int32_t>(offI, (int32_t)0, (int32_t)32, M);
      AscendC::Gather(gbOffI, sprI, offU, (uint32_t)0, M);
      (void)gbOffF;
    }
    // scalar writes of permF/permI -> vector reads them as Gatherb address lists.
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID1);
    AscendC::GatherRepeatParams gp{1, 8};  // dstBlkStride, dstRepStride -- matches the shipped grid_sampler reference
    AscendC::DataCopyPadExtParams<T> pd{false, 0, 0, 0};
    const uint8_t repNR = (uint8_t)(NR / 8u);
    const uint8_t repM = (uint8_t)(M / 8u);
    for (uint64_t blk = blk0; blk < blk1; blk++) {
      uint32_t o0 = (uint32_t)(blk * B);
      uint32_t curB = (o0 + B > outer) ? (uint32_t)(outer - o0) : B;
      // iaq5: software pipeline ACROSS sub-blocks.  iaq4 showed the pipe sum still equals the
      // wall (116.6 vs 116.8) because within one sub-block load->transpose->gather->add->pack->
      // store is a true dependency chain; merging the two loads only bought -1.1 us.  Here the
      // NEXT sub-block's loads are issued before the CURRENT one's transposes, so transfer for
      // si+1 runs under compute for si.  Flags: MTE2_V id = slot, V_MTE2 id = 2 + slot.
      const uint32_t nSub = (curB + RB - 1u) / RB;
      if (blk != blk0) {  // iaq13: the first block was already primed before the prologue
        uint32_t rows0 = (curB < RB) ? curB : RB;
        AscendC::DataCopyExtParams xp0{(uint16_t)rows0, N * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(xB0, xGm[(uint64_t)o0 * N], xp0, pd);
        AscendC::DataCopyExtParams s0{1, rows0 * M * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(srcRaw, srcGm[(uint64_t)o0 * M], s0, pd);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      }
      for (uint32_t si = 0; si < nSub; si++) {
        uint32_t g = si * RB;
        uint32_t rows = (g + RB > curB) ? (curB - g) : RB;
        uint64_t row0 = (uint64_t)o0 + g;
        uint32_t p = si & 1u;
        if (si + 1u < nSub) {
          uint32_t q = p ^ 1u;
          uint32_t g2 = g + RB;
          uint32_t rows2 = (g2 + RB > curB) ? (curB - g2) : RB;
          uint64_t row2 = (uint64_t)o0 + g2;
          if (si >= 1u) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(q == 0u ? EVENT_ID2 : EVENT_ID3);
          AscendC::DataCopyExtParams xp2{(uint16_t)rows2, N * (uint32_t)sizeof(T), 0, 0, 0};
          AscendC::DataCopyPad(q == 0u ? xB0 : xB1, xGm[row2 * N], xp2, pd);
          AscendC::DataCopyExtParams s2b{1, rows2 * M * (uint32_t)sizeof(T), 0, 0, 0};
          AscendC::DataCopyPad(q == 0u ? srcRaw : sB1, srcGm[row2 * M], s2b, pd);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(q == 0u ? EVENT_ID0 : EVENT_ID1);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(p == 0u ? EVENT_ID0 : EVENT_ID1);
        FastTransB16(accNat, p == 0u ? xB0 : xB1, RB, NR, NR, RB);
        AscendC::Gatherb(accR.template ReinterpretCast<uint16_t>(),
                         accNat.template ReinterpretCast<uint16_t>(), permF, repNR, gp);
        FastTransB16(srcT, p == 0u ? srcRaw : sB1, RB, M, M, RB);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(p == 0u ? EVENT_ID2 : EVENT_ID3);
        AscendC::Gatherb(plane.template ReinterpretCast<uint16_t>(),
                         srcT.template ReinterpretCast<uint16_t>(), gbOff, repM, gp);
        for (uint32_t l = 0; l < L; l++) {
          uint32_t cl = (uint32_t)cL[l];
          if (cl == 0u) break;
          AscendC::Add(accR, accR, plane[(uint32_t)sL[l] * RB], cl * RB);
        }
        AscendC::Gatherb(accNat.template ReinterpretCast<uint16_t>(),
                         accR.template ReinterpretCast<uint16_t>(), permI, repNR, gp);
        FastTransB16(srcT, accNat, NR, RB, RB, NR);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID4);
        AscendC::DataCopyExtParams yp{(uint16_t)rows, N * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(yGm[row0 * N], srcT, yp);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
      }
      // Exactly two V_MTE2 sets have no matching wait: the last iteration never prefetches,
      // and the second to last one's prefetch is the last wait issued.  Drain both.
      if (nSub >= 1u) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(((nSub - 1u) & 1u) == 0u ? EVENT_ID2 : EVENT_ID3);
      if (nSub >= 2u) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(((nSub - 2u) & 1u) == 0u ? EVENT_ID2 : EVENT_ID3);
    }
  }

  __aicore__ inline void ProcVecTransposeF16Fast() {
    const uint32_t B = k7B, Nt = k7Ntile, Mt = k7Mtile;
    const uint32_t N = (uint32_t)dimLen, M = m;
    uint64_t nBlocks = (outer + B - 1) / B;
    uint64_t blk0 = (uint64_t)AscendC::GetBlockIdx() * k7BlocksPerCore;
    uint64_t blk1 = blk0 + k7BlocksPerCore;
    if (blk1 > nBlocks) blk1 = nBlocks;
    AscendC::LocalTensor<T> accT = accBuf.Get<T>();   // fp16 [Npad,B]
    // ONE unified pool of B*max(Nt,3*Mt) elems.  Phase 1 stages [B,Nt] at pool[0]; phase 2 uses
    // pool[0] / pool[xfSz] as the two [B,Mt] load slots and pool[2*xfSz] as srcT[Mt,B]; phase 3
    // packs outT[B,Nt] back at pool[0].  The three phases are strictly sequential within a block
    // and every hand-off is already guarded (V_MTE2 before phase 2's first load, MTE2_V/PIPE_V
    // inside phase 2, MTE3_V at the end of phase 3), so the aliasing is safe.  Decoupling the
    // phase-1/3 row width from the phase-2 tile is what lets Nt = alignUp(N,16) keep B large.
    AscendC::LocalTensor<T> pool = k7xfer.Get<T>();
    const uint32_t xfSz = B * Mt;                     // one phase-2 load slot (elements)
    AscendC::LocalTensor<T> xfer = pool;              // phase-1 staging / phase-2 slot 0
    AscendC::LocalTensor<T> srcT = pool[2u * xfSz];   // fp16 srcT[Mt,B]
    AscendC::LocalTensor<T> outT = pool;              // fp16 [B,Nrow] phase-3 pack
    // Nrow = alignUp(N,16) = the accT row count AND the phase-3 pack row pitch.  Nrow*2 ==
    // alignUp(N*2,32) == the DMA's 32B-rounded row size, so the phase-3 store is ONE
    // DataCopyPad of curB contiguous N*2-byte blocks (srcStride/dstStride both 0).
    const uint32_t Nrow = (N + 15) / 16 * 16;
    // clamp indices to [0,N-1] ONCE (vectorized): the per-k loop then skips 3 scalar cmp/step.
    // (wide-row workload became scalar-bound after the transpose fix; hoisting the clamp cuts scalar ~half.)
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);  // idx load(MTE2) -> clamp(V)
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::Maxs(idxLocal, idxLocal, (int32_t)0, M);
    AscendC::Mins(idxLocal, idxLocal, (int32_t)N - 1, M);
    // premultiply the clamped index by B on the VECTOR unit (off the scalar path): the per-k scatter
    // then reads accT[j] directly with NO per-add scalar `j*B` multiply (idxLocal[k] = clamp*B).
    AscendC::Muls(idxLocal, idxLocal, (int32_t)B, M);

    for (uint64_t blk = blk0; blk < blk1; blk++) {
      uint32_t o0 = (uint32_t)(blk * B);
      uint32_t curB = (o0 + B > outer) ? (uint32_t)(outer - o0) : B;
      // phase 1: input[curB,N] -> accT[N,B]  (transpose all B lanes; lanes>=curB garbage, unused)
      for (uint32_t n0 = 0; n0 < N; n0 += Nt) {
        uint32_t vN = (n0 + Nt > N) ? (N - n0) : Nt;
        uint32_t P = LoadRowsPacked(xfer, xGm[(uint64_t)o0 * N + n0], curB, vN, N);
        FastTransB16(accT[(uint64_t)n0 * B], xfer, B, P, P, B);
      }
      // phase 2: per M-tile transpose source -> srcT[Mt,B]; per-k B-wide fp16 direct add (k-order).
      // PIPELINED: tile i+1's load is issued before tile i's transpose, so MTE2 streams the next
      // (wide) source block while V does this tile's transpose + vM adds.
      const uint32_t nT = (M + Mt - 1) / Mt;
      AscendC::DataCopyPadExtParams<T> lpad{false, 0, 0, 0};
      {  // prologue: phase 1's last transpose still reads xfer slot 0 -> guard, then issue tile 0
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        uint32_t vM0 = (Mt > M) ? M : Mt;
        AscendC::DataCopyExtParams p0{(uint16_t)curB, vM0 * (uint32_t)sizeof(T),
                                      (M - vM0) * (uint32_t)sizeof(T), 0, 0};
        AscendC::DataCopyPad(xfer, srcGm[(uint64_t)o0 * M], p0, lpad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      }
      for (uint32_t ti = 0; ti < nT; ti++) {
        const uint32_t m0 = ti * Mt;
        const uint32_t vM = (m0 + Mt > M) ? (M - m0) : Mt;
        const uint32_t slot = ti & 1;
        if (ti + 1 < nT) {  // prefetch next tile into the other slot
          const uint32_t os = slot ^ 1;
          const uint32_t m1 = m0 + Mt;
          const uint32_t vM1 = (m1 + Mt > M) ? (M - m1) : Mt;
          if (ti >= 1)
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(os == 0 ? EVENT_ID3 : EVENT_ID4);
          AscendC::DataCopyExtParams p1{(uint16_t)curB, vM1 * (uint32_t)sizeof(T),
                                        (M - vM1) * (uint32_t)sizeof(T), 0, 0};
          AscendC::DataCopyPad(xfer[os * xfSz], srcGm[(uint64_t)o0 * M + m1], p1, lpad);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(os == 0 ? EVENT_ID0 : EVENT_ID1);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot == 0 ? EVENT_ID0 : EVENT_ID1);
        const uint32_t Pm = (vM + 15) / 16 * 16;  // fp16 UB row pitch (DataCopyPad 32B-rounds it)
        FastTransB16(srcT, xfer[slot * xfSz], B, Pm, Pm, B);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot == 0 ? EVENT_ID3 : EVENT_ID4);  // slot free
        AscendC::LocalTensor<int64_t> idx64 = idxLocal.template ReinterpretCast<int64_t>();
        uint32_t kc = 0;
        for (; kc + 8 <= vM; kc += 8) {  // packed read: 4 int64 loads -> 8 int32, fewer loop branches
          int64_t p0 = idx64.GetValue((m0 + kc) >> 1), p1 = idx64.GetValue(((m0 + kc) >> 1) + 1);
          int64_t p2 = idx64.GetValue(((m0 + kc) >> 1) + 2), p3 = idx64.GetValue(((m0 + kc) >> 1) + 3);
          int32_t j0 = (int32_t)p0, j1 = (int32_t)(p0 >> 32);  // j* already = clamp(index)*B
          int32_t j2 = (int32_t)p1, j3 = (int32_t)(p1 >> 32);
          int32_t j4 = (int32_t)p2, j5 = (int32_t)(p2 >> 32);
          int32_t j6 = (int32_t)p3, j7 = (int32_t)(p3 >> 32);
          AscendC::Add(accT[(uint64_t)j0], accT[(uint64_t)j0], srcT[(uint64_t)kc * B], B);
          AscendC::Add(accT[(uint64_t)j1], accT[(uint64_t)j1], srcT[(uint64_t)(kc + 1) * B], B);
          AscendC::Add(accT[(uint64_t)j2], accT[(uint64_t)j2], srcT[(uint64_t)(kc + 2) * B], B);
          AscendC::Add(accT[(uint64_t)j3], accT[(uint64_t)j3], srcT[(uint64_t)(kc + 3) * B], B);
          AscendC::Add(accT[(uint64_t)j4], accT[(uint64_t)j4], srcT[(uint64_t)(kc + 4) * B], B);
          AscendC::Add(accT[(uint64_t)j5], accT[(uint64_t)j5], srcT[(uint64_t)(kc + 5) * B], B);
          AscendC::Add(accT[(uint64_t)j6], accT[(uint64_t)j6], srcT[(uint64_t)(kc + 6) * B], B);
          AscendC::Add(accT[(uint64_t)j7], accT[(uint64_t)j7], srcT[(uint64_t)(kc + 7) * B], B);
        }
        for (; kc < vM; kc++) {
          int32_t j = idxLocal.GetValue(m0 + kc);  // = clamp(index)*B
          AscendC::Add(accT[(uint64_t)j], accT[(uint64_t)j], srcT[(uint64_t)kc * B], B);
        }
      }
      // drain the 2 trailing "xfer slot free" flags (set nT times, waited nT-2 times)
      if (nT >= 1)
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(((nT - 1) & 1) == 0 ? EVENT_ID3 : EVENT_ID4);
      if (nT >= 2)
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(((nT - 2) & 1) == 0 ? EVENT_ID3 : EVENT_ID4);
      // phase 3: accT[Nrow,B] -> outT[B,Nrow] (all N-tiles, pure V) -> ONE contiguous store.
      // Every tile packs into its own column slice of the SAME full-width buffer (dst pitch
      // Nrow, tile offset n0 -- both multiples of 16 so every TransDataTo5HD dst address stays
      // 32B aligned).  Column N..Nrow-1 of the last tile is transpose pad and is never stored.
      for (uint32_t n0 = 0; n0 < N; n0 += Nt) {
        uint32_t vN = (n0 + Nt > N) ? (N - n0) : Nt;
        FastTransB16(outT[n0], accT[(uint64_t)n0 * B], vN, B, B, Nrow);
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);  // transpose(V) -> store(MTE3)
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
      {  // ONE store of curB blocks x N*2 bytes; UB row advances by Nrow*2 == the 32B-rounded
         // block size, so srcStride 0 is exact, and GM rows are contiguous so dstStride 0 too.
        AscendC::DataCopyExtParams sp{(uint16_t)curB, N * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyPad(yGm[(uint64_t)o0 * N], outT, sp);
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);  // store done -> reuse outT
      AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    }
  }

  // b32 (fp32) transpose src[R,C](pitch sp) -> dst[C,R](pitch dp).  C<=64 (channelAlign),
  // R mult of 16.  Recipe from grid_sampler_2d OutTranspose (proven fp32 hw transpose):
  // repeatTimes=cAlign/8 folds the C cols in 8-groups; each j-group of 16 source-rows ->
  // 16 output-cols (b32 C0=8 -> dstList 8 pairs, each output 8-col half addressed separately).
  // strides in 32B(=8 fp32) block units: src next 8-col group = 1 block; dst next 8-row
  // group = 8*dp elem = dp blocks.  Replaces GTr Gather (~1 elem/cyc) with hw ~8 elem/cyc.
  __aicore__ inline void FastTransB32(const AscendC::LocalTensor<float>& dst,
                                      const AscendC::LocalTensor<float>& src, uint32_t R,
                                      uint32_t C, uint32_t sp, uint32_t dp) {
    uint32_t rg = (R + 15) / 16;
    uint32_t cAlign = (C + 7) / 8 * 8;
    AscendC::TransDataTo5HDParams p;
    p.repeatTimes = (uint8_t)(cAlign / 8);
    p.dstRepStride = (cAlign / 8 == 1) ? 0 : (uint16_t)dp;
    p.srcRepStride = (cAlign / 8 == 1) ? 0 : 1;
    uint64_t srcList[16], dstList[16];
    const uint64_t srcBase = (uint64_t)src.GetPhyAddr();
    const uint64_t dstBase = (uint64_t)dst.GetPhyAddr();
    for (uint32_t j = 0; j < rg; j++) {
      for (uint32_t i = 0; i < 16; i++)
        srcList[i] = srcBase + ((uint64_t)(j * 16 + i) * sp) * sizeof(float);
      for (uint32_t i = 0; i < 8; i++) {
        dstList[i * 2] = dstBase + ((uint64_t)i * dp + j * 16) * sizeof(float);
        dstList[i * 2 + 1] = dstBase + ((uint64_t)i * dp + 8 + j * 16) * sizeof(float);
      }
      AscendC::TransDataTo5HD<float>(dstList, srcList, p);
    }
  }

  // ---- key 7 fp32 (large-stream workload): transposed GM accumulator outT[N,OUTER] + SetAtomicAdd<float> scatter.
  // per-k source column is added to outT[index[k], oBlock] via an outer-wide CONTIGUOUS atomic
  // store on the MTE3/DMA unit -> the accumulate leaves the Vector unit.  Vector does only the
  // transposes (now FastTransB32 hw b32, ~8 elem/cyc vs GTr Gather ~1).  Each core owns a DISJOINT
  // outer-column block of outT -> no cross-core atomic contention.  fp32-once (order-independent)
  // -> SetAtomicAdd<float> RMW is bit-exact to torch.  (UB-accumulate variant is vec-bound and
  // LOSES: transpose+add serialize on the vector pipe; atomic overlaps transpose(V) with scatter(MTE3).)
  __aicore__ inline void ProcVecTransAtomic() {
    const uint32_t B = k7B, Nt = k7Ntile;
    const uint32_t N = (uint32_t)dimLen, M = m;
    uint64_t nBlocks = (outer + B - 1) / B;
    // T6 BLOCK-MAJOR workspace: ws is [nBlocks, N, B] instead of [N, nBlocks*B].  SAME total size
    // (nBlocks*N*B == N*OUTER) so op_host's workspace sizing is unchanged.  Each core owns block
    // `blk` EXCLUSIVELY (k7BlocksPerCore==1, coreNum==nBlocks), so its whole [N,B] slab is now GM-
    // CONTIGUOUS: phase 1's store and phase 3's load collapse from vN blocks of 256 B (GM gap
    // (OUTER-B)*4 = 9,472 B) into ONE block of vN*B*4 B.  Phase 2's atomic store stays a single
    // 256 B block (row n of this core's slab is at blkBase + n*B).  The per-word atomic-add
    // sequence per core is unchanged -> bit-exact.
    uint64_t blk0 = (uint64_t)AscendC::GetBlockIdx() * k7BlocksPerCore;
    uint64_t blk1 = blk0 + k7BlocksPerCore;
    if (blk1 > nBlocks) blk1 = nBlocks;
    // UNIFIED POOL (T1).  Layout, re-bound per phase (liveness is strictly sequential and the
    // hand-offs are already PipeBarrier<PIPE_ALL>):
    //   phase 1: accT   = pool[0]        [Nt,B]   stage = pool[Nt*B]   [B,Nt]
    //   phase 2: xfer0  = pool[0]        [B,Kt]   xfer1 = pool[B*Kt]   [B,Kt]
    //            srcT0  = pool[2*B*Kt]   [Kt,B]   srcT1 = pool[3*B*Kt] [Kt,B]
    //   phase 3: accT   = pool[0]        [Nt,B]   outF  = pool[Nt*B]   [B,Nt]
    AscendC::LocalTensor<float> pool = k7xfer.Get<float>();
    const uint32_t p13Half = Nt * B;  // accT / stage / outF stride within the pool
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
    // precompute clamped scatter offsets on the VECTOR unit (off the scalar critical path):
    // idxLocal[k] = clamp(index[k], 0, N-1) * B  -> per-k inner loop just reads it (no clamp/mul).
    AscendC::Maxs(idxLocal, idxLocal, 0, (int32_t)M);
    AscendC::Mins(idxLocal, idxLocal, (int32_t)(N - 1), (int32_t)M);
    AscendC::Muls(idxLocal, idxLocal, (int32_t)B, (int32_t)M);

    for (uint64_t blk = blk0; blk < blk1; blk++) {
      uint32_t o0 = (uint32_t)(blk * B);
      uint32_t curB = (o0 + B > outer) ? (uint32_t)(outer - o0) : B;
      const uint64_t blkBase = blk * (uint64_t)N * B;  // T6: base of this core's [N,B] ws slab
      // phase 1: ws[:, o0:o0+B] = transpose(input).  TILED accT[Nt,B] (not [Npad,B]) -> frees UB
      // for ~2x B -> half the phase-2 atomic-store count (the scalar bound).  Store per N-tile.
      {  // ---- phase 1 scope: pool = accT[Nt,B] | stage[B,Nt] ----
      AscendC::LocalTensor<float> accT = pool;                 // [Nt,B]
      AscendC::LocalTensor<T> xfer =
          pool[p13Half].template ReinterpretCast<T>();          // [B,Nt] staging slot
      for (uint32_t n0 = 0; n0 < N; n0 += Nt) {
        uint32_t vN = (n0 + Nt > N) ? (N - n0) : Nt;
        uint32_t P = LoadRowsPacked(xfer, xGm[(uint64_t)o0 * N + n0], curB, vN, N);
        FastTransB32(accT, xfer.template ReinterpretCast<float>(), B, P, P, B);  // accT[Nt,B]
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);  // transpose(V) -> store(MTE3)
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        // T6: accT's UB pitch is exactly B, and the slab is block-major, so the whole vN-row tile
        // is ONE contiguous vN*B*4 B run in BOTH UB and GM -> 1 descriptor instead of vN.
        AscendC::DataCopyExtParams sp{1, vN * B * 4u, 0, 0, 0};
        AscendC::DataCopyPad(wsGm[blkBase + (uint64_t)n0 * B], accT, sp);  // store this tile's vN rows
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);  // store done -> accT reusable
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
      }
      }  // end phase 1 scope (accT + stage now dead; pool is free for phase 2)
      AscendC::PipeBarrier<PIPE_ALL>();  // init store visible before atomic RMW reads it
      // phase 2: atomic scatter source columns into outT rows.
      // DOUBLE-BUFFERED srcT (ping-pong the two halves of the 128*B buffer, each Nt*B): tile
      // N+1's transpose(V) overlaps tile N's atomic stores(MTE3) -> hide the store latency
      // under the next transpose.  Reuse guard: before overwriting slot s, wait its stores
      // from 2 tiles ago drained (MTE3_V id5/id6).  Stores across slots are independent atomic
      // adds (hardware serializes same-address RMW) -> overlap is correctness-neutral.
      AscendC::SetAtomicAdd<float>();
      const uint32_t Kt = k7Mtile;          // phase-2 source tile width (fp32 elems) -> Kt*4 B blocks
      const uint32_t slotSz = Kt * B;       // one srcT ping-pong slot
      const uint32_t xfSz = B * Kt;         // one xfer load slot (Kt-wide, no longer max(Nt,Kt))
      // pool = [xfer0 | xfer1 | srcT0 | srcT1], each B*Kt elems
      AscendC::LocalTensor<T> xfer = pool.template ReinterpretCast<T>();
      AscendC::LocalTensor<float> srcT = pool[2u * xfSz];
      const uint32_t nT = (M + Kt - 1) / Kt;
      AscendC::DataCopyPadExtParams<T> lpad{false, 0, 0, 0};
      // PROLOGUE: issue tile 0's load into xfer slot 0.  (phase 1 ended with PipeBarrier<PIPE_ALL>,
      // so xfer is free and no V_MTE2 guard is needed for the first use of either slot.)
      {
        uint32_t vM0 = (Kt > M) ? M : Kt;
        AscendC::DataCopyExtParams p0{(uint16_t)curB, vM0 * (uint32_t)sizeof(T),
                                      (M - vM0) * (uint32_t)sizeof(T), 0, 0};
        AscendC::DataCopyPad(xfer, srcGm[(uint64_t)o0 * M], p0, lpad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      }
      for (uint32_t ti = 0; ti < nT; ti++) {
        const uint32_t m0 = ti * Kt;
        const uint32_t vM = (m0 + Kt > M) ? (M - m0) : Kt;
        const uint32_t slot = ti & 1;
        // PREFETCH tile ti+1 into the other xfer slot FIRST: this DataCopyPad is issued before
        // the V_MTE3 barrier below, so MTE2 streams the next 512 B-block read while V transposes
        // and MTE3 drains this tile's atomic stores.
        if (ti + 1 < nT) {
          const uint32_t os = slot ^ 1;
          const uint32_t m1 = m0 + Kt;
          const uint32_t vM1 = (m1 + Kt > M) ? (M - m1) : Kt;
          if (ti >= 1)  // that slot's previous transpose must have finished reading it
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(os == 0 ? EVENT_ID3 : EVENT_ID4);
          AscendC::DataCopyExtParams p1{(uint16_t)curB, vM1 * (uint32_t)sizeof(T),
                                        (M - vM1) * (uint32_t)sizeof(T), 0, 0};
          AscendC::DataCopyPad(xfer[os * xfSz], srcGm[(uint64_t)o0 * M + m1], p1, lpad);
          AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(os == 0 ? EVENT_ID0 : EVENT_ID1);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(slot == 0 ? EVENT_ID0 : EVENT_ID1);
        const uint32_t Pm = (vM + 7) / 8 * 8;  // fp32 UB row pitch (DataCopyPad 32B-rounds it)
        AscendC::LocalTensor<float> srcS = srcT[slot * slotSz];
        if (ti >= 2)  // slot reused: wait its previous tile's stores to drain before overwrite
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(slot == 0 ? EVENT_ID5 : EVENT_ID6);
        FastTransB32(srcS, xfer[slot * xfSz].template ReinterpretCast<float>(), B, Pm, Pm, B);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(slot == 0 ? EVENT_ID3 : EVENT_ID4);  // xfer slot free
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);  // transpose(V) done -> stores read srcS
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        uint32_t kc = 0;
        AscendC::LocalTensor<int64_t> idx64 = idxLocal.template ReinterpretCast<int64_t>();
        for (; kc + 4 <= vM; kc += 4) {  // packed read: 2 int64 loads yield 4 int32 offsets (half the scalar loads)
          int64_t p0 = idx64.GetValue((m0 + kc) >> 1);      // idx[m0+kc], idx[m0+kc+1] packed (=clamp*OUTER)
          int64_t p1 = idx64.GetValue(((m0 + kc) >> 1) + 1);  // idx[m0+kc+2], idx[m0+kc+3]
          int32_t f0 = (int32_t)p0, f1 = (int32_t)(p0 >> 32);
          int32_t f2 = (int32_t)p1, f3 = (int32_t)(p1 >> 32);
          AscendC::DataCopy(wsGm[blkBase + (uint64_t)f0], srcS[(uint64_t)kc * B], B);
          AscendC::DataCopy(wsGm[blkBase + (uint64_t)f1], srcS[(uint64_t)(kc + 1) * B], B);
          AscendC::DataCopy(wsGm[blkBase + (uint64_t)f2], srcS[(uint64_t)(kc + 2) * B], B);
          AscendC::DataCopy(wsGm[blkBase + (uint64_t)f3], srcS[(uint64_t)(kc + 3) * B], B);
        }
        for (; kc < vM; kc++) {
          int32_t off = idxLocal.GetValue(m0 + kc);
          AscendC::DataCopy(wsGm[blkBase + (uint64_t)off], srcS[(uint64_t)kc * B], B);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(slot == 0 ? EVENT_ID5 : EVENT_ID6);  // stores issued
      }
      const uint32_t ti = nT;
      // epilogue: drain the last (up to 2) tiles' stores before SetAtomicNone / read-back
      if (ti >= 1)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(((ti - 1) & 1) == 0 ? EVENT_ID5 : EVENT_ID6);
      if (ti >= 2)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(((ti - 2) & 1) == 0 ? EVENT_ID5 : EVENT_ID6);
      // drain the 2 trailing V_MTE2 "xfer slot free" flags (set nT times, waited nT-2 times)
      if (ti >= 1)
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(((ti - 1) & 1) == 0 ? EVENT_ID3 : EVENT_ID4);
      if (ti >= 2)
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(((ti - 2) & 1) == 0 ? EVENT_ID3 : EVENT_ID4);
      AscendC::SetAtomicNone();
      AscendC::PipeBarrier<PIPE_ALL>();  // atomic stores visible before read-back
      // phase 3: ws[:, o0:o0+curB] -> transpose -> y.  TILED accT[Nt,B]: load+transpose+store per tile.
      {  // ---- phase 3 scope: pool = accT[Nt,B] | outF[B,Nt] ----
      AscendC::LocalTensor<float> accT = pool;        // [Nt,B]
      AscendC::LocalTensor<float> outP = pool[p13Half];  // [B,Nt] pack
      for (uint32_t n0 = 0; n0 < N; n0 += Nt) {
        uint32_t vN = (n0 + Nt > N) ? (N - n0) : Nt;
        if (n0 > 0) AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);  // prior transpose read accT -> overwrite
        AscendC::DataCopyExtParams lp{1, vN * B * 4u, 0, 0, 0};  // T6: one contiguous run
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(accT, wsGm[blkBase + (uint64_t)n0 * B], lp, pad);  // load this tile into accT[0..]
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);  // load(MTE2) -> transpose(V)
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID3);
        AscendC::LocalTensor<float> outF = outP;
        FastTransB32(outF, accT, Nt, B, B, Nt);  // accT[Nt,B] -> outF[B,Nt], B<=64
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);  // transpose done reading accT (next-tile overwrite)
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);  // transpose(V) -> store(MTE3)
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID2);
        AscendC::LocalTensor<T> src = outF.template ReinterpretCast<T>();
        if (vN == Nt) {
          AscendC::DataCopyExtParams sp{(uint16_t)curB, Nt * 4u, 0, (N - Nt) * 4u, 0};
          AscendC::DataCopyPad(yGm[(uint64_t)o0 * N + n0], src, sp);
        } else {
          for (uint32_t b = 0; b < curB; b++)
            PadOutT(yGm[(uint64_t)(o0 + b) * N + n0], src[(uint64_t)b * Nt], vN);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);  // store done -> reuse outF(srcT)
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
      }
      }  // end phase 3 scope
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID4);  // drain last transpose's accT-read flag
    }
  }

  // ---- key 9: int32 middle-dim -> owned-outer, no atomics ----
  // Each core owns a disjoint slice of `outer`.  For each o it holds the full acc[dimLen,
  // inner] (row pitch padded to accStride for 32B alignment) in UB, loads input[o] into it,
  // adds every source row  acc[idx[k]] += source[o,k]  (int32 exact, order-independent),
  // then writes y[o].  Each o's output is owned by ONE core -> no cross-core atomic
  // contention (unlike key 1's contiguous-unit atomic scatter).
  __aicore__ inline void ProcInt32OwnedOuter() {
   if constexpr (std::is_same_v<T, int32_t>) {
    uint32_t o0 = (uint32_t)AscendC::GetBlockIdx() * accRowsPerCore;
    if ((uint64_t)o0 >= outer) return;
    uint64_t o1_64 = (uint64_t)o0 + accRowsPerCore;
    if (o1_64 > outer) o1_64 = outer;
    uint32_t o1 = (uint32_t)o1_64;
    uint32_t N = (uint32_t)dimLen, IN = (uint32_t)inner;
    AscendC::LocalTensor<int32_t> acc = accBuf.Get<int32_t>();
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    for (uint32_t o = o0; o < o1; o++) {
      // load input[o] -> acc (per dim-row into the padded pitch)
      for (uint32_t r = 0; r < N; r++) {
        AscendC::DataCopyExtParams lp{1, IN * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(acc[(uint64_t)r * accStride],
                             xGm[((uint64_t)o * N + r) * IN], lp, pad);
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);  // input loaded -> adds
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      for (uint32_t k = 0; k < m; k++) {
        int32_t j = idxLocal.GetValue(k);
        if (j < 0) j = 0;
        if (j >= (int32_t)N) j = (int32_t)N - 1;
        AscendC::LocalTensor<int32_t> s = qLoad.template AllocTensor<int32_t>();
        AscendC::DataCopyExtParams sp{1, IN * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(s, srcGm[((uint64_t)o * m + k) * IN], sp, pad);
        qLoad.EnQue(s);
        s = qLoad.template DeQue<int32_t>();
        AscendC::Add(acc[(uint64_t)j * accStride], acc[(uint64_t)j * accStride], s, IN);
        qLoad.FreeTensor(s);
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);  // adds done -> write
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      for (uint32_t r = 0; r < N; r++) {
        AscendC::DataCopyExtParams wp{1, IN * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(yGm[((uint64_t)o * N + r) * IN],
                             acc[(uint64_t)r * accStride], wp);
      }
      AscendC::PipeBarrier<PIPE_ALL>();  // WAR: acc reused across o -> write done before reload
    }
   }
  }

  // ---- key 10: int32 middle-dim -> inner-split, no atomics, all cores busy ----
  // inner is split into S = ceil(inner/accInnerTile) column slices of width W = accInnerTile.
  // A unit = (o, slice); units are round-robin'd over all `cores` (so 40 cores stay busy even
  // when outer < cores).  Each core, per unit, holds acc[dimLen, W] for that (o,slice), loads
  // input[o,:,slice], adds every source row's slice  acc[idx[k],:] += source[o,k,slice], writes
  // y[o,:,slice].  Cores own DISJOINT output columns -> no cross-core atomics.  Slice reads are
  // contiguous (cW-wide sub-rows), giving large-block DMA.
  __aicore__ inline void ProcInt32InnerSplit() {
   if constexpr (std::is_same_v<T, int32_t>) {
    uint32_t N = (uint32_t)dimLen, IN = (uint32_t)inner, W = accInnerTile;
    uint32_t S = (IN + W - 1) / W;
    uint64_t unitsK = outer * (uint64_t)S;
    uint32_t cores = (uint32_t)AscendC::GetBlockNum();
    uint32_t bid = (uint32_t)AscendC::GetBlockIdx();
    AscendC::LocalTensor<int32_t> acc = accBuf.Get<int32_t>();
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    for (uint64_t u = bid; u < unitsK; u += cores) {
      uint32_t o = (uint32_t)(u / S);
      uint32_t sl = (uint32_t)(u % S);
      uint32_t c0 = sl * W;
      uint32_t cW = (c0 + W > IN) ? (IN - c0) : W;
      for (uint32_t r = 0; r < N; r++) {
        AscendC::DataCopyExtParams lp{1, cW * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(acc[(uint64_t)r * accStride],
                             xGm[((uint64_t)o * N + r) * IN + c0], lp, pad);
      }
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);  // input slice loaded -> adds
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
      for (uint32_t k = 0; k < m; k++) {
        int32_t j = idxLocal.GetValue(k);
        if (j < 0) j = 0;
        if (j >= (int32_t)N) j = (int32_t)N - 1;
        AscendC::LocalTensor<int32_t> s = qLoad.template AllocTensor<int32_t>();
        AscendC::DataCopyExtParams sp{1, cW * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(s, srcGm[((uint64_t)o * m + k) * IN + c0], sp, pad);
        qLoad.EnQue(s);
        s = qLoad.template DeQue<int32_t>();
        AscendC::Add(acc[(uint64_t)j * accStride], acc[(uint64_t)j * accStride], s, cW);
        qLoad.FreeTensor(s);
      }
      AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);  // adds done -> write
      AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
      for (uint32_t r = 0; r < N; r++) {
        AscendC::DataCopyExtParams wp{1, cW * (uint32_t)sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(yGm[((uint64_t)o * N + r) * IN + c0],
                             acc[(uint64_t)r * accStride], wp);
      }
      AscendC::PipeBarrier<PIPE_ALL>();  // WAR: acc reused across units
    }
   }
  }

  // ---- key 6: fp16/bf16 last-axis (inner==1) -> owned-outer fp32-once (no atomics) ----
  // torch fp16/bf16 last-axis index_add == fp32-accumulate-round-once (verified: 0 err vs
  // golden).  Each core owns a disjoint slice of `outer` rows -> its outputs are disjoint,
  // no cross-core atomics.  Per row: load input[dimLen] + src[m] into UB, cast fp32,
  // scatter-sum acc[idx[k]] += srcF[k] (order-independent), round-cast once, write.
  __aicore__ inline void ProcLastFp32Once() {
   // fp16/bf16 -> per-step round to T (torch int32-index semantics); fp32 -> exact (no round).
   if constexpr (sizeof(T) == 2 || std::is_same_v<T, float>) {
    uint32_t o0 = (uint32_t)AscendC::GetBlockIdx() * accRowsPerCore;  // accRowsPerCore = outerPerCore
    if ((uint64_t)o0 >= outer) return;
    uint64_t o1_64 = (uint64_t)o0 + accRowsPerCore;
    if (o1_64 > outer) o1_64 = outer;
    uint32_t o1 = (uint32_t)o1_64;
    uint32_t N = (uint32_t)dimLen;
    for (uint32_t o = o0; o < o1; o++) {
      AscendC::LocalTensor<T> xin = qLoad.template AllocTensor<T>();
      PadInT(xin, xGm[(uint64_t)o * N], N);
      qLoad.EnQue(xin);
      xin = qLoad.template DeQue<T>();
      AscendC::LocalTensor<float> acc = accBuf.Get<float>();
      if constexpr (sizeof(T) == 2)
        AscendC::Cast(acc, xin, AscendC::RoundMode::CAST_NONE, N);
      else
        AscendC::Adds(acc, xin, 0.0f, N);  // fp32 copy input -> acc
      qLoad.FreeTensor(xin);
      AscendC::LocalTensor<T> sin = qLoad.template AllocTensor<T>();
      PadInT(sin, srcGm[(uint64_t)o * m], m);
      qLoad.EnQue(sin);
      sin = qLoad.template DeQue<T>();
      AscendC::LocalTensor<float> srcF = tmpBuf.Get<float>();
      if constexpr (sizeof(T) == 2)
        AscendC::Cast(srcF, sin, AscendC::RoundMode::CAST_NONE, m);
      else
        AscendC::Adds(srcF, sin, 0.0f, m);
      qLoad.FreeTensor(sin);
      for (uint32_t k = 0; k < m; k++) {
        int32_t j = idxLocal.GetValue(k);
        if (j < 0) j = 0;
        if (j >= (int32_t)N) j = (int32_t)N - 1;
        float v = acc.GetValue((uint32_t)j) + srcF.GetValue(k);
        // torch index_add w/ int32 index rounds to T after EACH add (fp16 only; fp32 exact,
        // bf16 scalar cast unsupported+no bf16 last-axis case).
        if constexpr (std::is_same_v<T, half>) v = (float)((half)v);
        acc.SetValue((uint32_t)j, v);
      }
      AscendC::LocalTensor<T> out = qStore.template AllocTensor<T>();
      if constexpr (sizeof(T) == 2)
        AscendC::Cast(out, acc, AscendC::RoundMode::CAST_RINT, N);
      else
        AscendC::Adds(out, acc, 0.0f, N);  // fp32 copy acc -> out
      qStore.EnQue(out);
      out = qStore.template DeQue<T>();
      PadOutT(yGm[(uint64_t)o * N], out, N);
      qStore.FreeTensor(out);
    }
   }
  }

 private:
  // The TPipe lives in the kernel entry: constructing one publishes its address through a global
  // pointer, and inside this object that would keep the compiler from folding the scalar members.
  AscendC::TPipe* pipe;
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 2>
      qBind;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> qLoad;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 2> qStore;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> idxBuf, accBuf, tmpBuf, rndBuf, matchBuf,
      k7xfer, k7off, k7src, k7gth;
  AscendC::LocalTensor<int32_t> idxLocal;
  AscendC::GlobalTensor<T> xGm, srcGm, yGm;
  AscendC::GlobalTensor<float> wsGm;
  AscendC::GlobalTensor<int32_t> idxGm;

  bool roundStep;
  uint64_t outer, dimLen, inner, copyTotal, copyPerCore, units, unitsPerCore;
  uint32_t m, copyTileLen, innerTile, accRowsPerCore, accInnerTile, accStride;
  uint32_t k7B, k7Ntile, k7Mtile, k7BlocksPerCore;
  uint32_t gRows_ = 0, gCols_ = 0, gSz_ = 0;  // GTr offset-table cache key
};

extern "C" __global__ __aicore__ void index_add(GM_ADDR x, GM_ADDR indices,
                                                GM_ADDR source, GM_ADDR y,
                                                GM_ADDR workspace,
                                                GM_ADDR tiling) {
  // Keep each key's complete dispatch in the entry so opbuild emits and
  // registers the corresponding runtime function entry.
  AscendC::TPipe pipe;  // outside the kernel object (see KernelIndexAdd::pipe)
  if (TILING_KEY_IS(1)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  } else if (TILING_KEY_IS(2)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  } else if (TILING_KEY_IS(3)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  } else if (TILING_KEY_IS(4)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  } else if (TILING_KEY_IS(5)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  } else if (TILING_KEY_IS(6)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  } else if (TILING_KEY_IS(7)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  } else if (TILING_KEY_IS(23)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  } else if (TILING_KEY_IS(12)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  } else if (TILING_KEY_IS(13)) {
    KernelIndexAdd<DTYPE_X> op;
    op.Init(x, indices, source, y, workspace, (__gm__ const IndexAddTilingData*)tiling, &pipe);
    op.Process(); return;
  }
}
