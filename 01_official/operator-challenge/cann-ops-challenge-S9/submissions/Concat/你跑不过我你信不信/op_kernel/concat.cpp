/*
 * Concat: concatenate a dynamic list of tensors along concat_dim -> output.
 * Model: outer = prod(dims before dim), inner = prod(dims after dim),
 *   each active input k contributes width widths[k] along dim at output
 *   offset offsets[k]. For each (outer row o, active input k) a contiguous run
 *   of widths[k]*inner elements is copied: src input_k[o*segLen ..], dst output
 *   at o*sumW*inner + offsets[k]*inner. Runs are distributed across cores.
 * Dynamic input "x": ListTensorDesc is built in the kernel entry from x and
 *   passed by value (matching the Unpack reference); GetDataPtr<T>(idx) gives
 *   each input tensor's GM base. Pure data movement (GM->UB->GM), DataCopyPad.
 */
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

constexpr int32_t BUFFER_NUM = 2;  // Concat = MTE2 load + MTE3 store: DB overlaps the two engines (A/B: beats BN=1)
constexpr uint32_t MAX_T = 256;
constexpr uint32_t GATHER_WBATCH = 8;  // Route G: gather this many rows, then ONE strided store (vs per-row store)


template <typename T>
__aicore__ inline void PadIn(const AscendC::LocalTensor<T>& dst,
                             const AscendC::GlobalTensor<T>& src, uint32_t n) {
  AscendC::DataCopyExtParams params{1, n * (uint32_t)sizeof(T), 0, 0, 0};
  AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  AscendC::DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void PadOut(const AscendC::GlobalTensor<T>& dst,
                              const AscendC::LocalTensor<T>& src, uint32_t n) {
  AscendC::DataCopyExtParams params{1, n * (uint32_t)sizeof(T), 0, 0, 0};
  AscendC::DataCopyPad(dst, src, params);
}

// Store `blockCount` rows of `segLen` elems each from packed UB to GM with a
// per-row gap (output row stride). Requires segLen*sizeof(T) % 32 == 0 so the
// packed UB rows are 32B-aligned (srcStride=0). dstStride is in GM bytes.
template <typename T>
__aicore__ inline void StoreStrided(const AscendC::GlobalTensor<T>& dst,
                                    const AscendC::LocalTensor<T>& src,
                                    uint32_t blockCount, uint32_t segLen,
                                    uint32_t outRowStride) {
  AscendC::DataCopyExtParams params{(uint16_t)blockCount, segLen * (uint32_t)sizeof(T),
                                    0, (outRowStride - segLen) * (uint32_t)sizeof(T), 0};
  AscendC::DataCopyPad(dst, src, params);
}

// Route-6 variant: Gather rows use the maximum-group UB pitch, which can be
// wider than AlignUp(segLen*sizeof(T), 32B). Carry that gap in srcStride.
template <typename T>
__aicore__ inline void StoreStridedPitched(const AscendC::GlobalTensor<T>& dst,
                                           const AscendC::LocalTensor<T>& src,
                                           uint32_t blockCount, uint32_t segLen,
                                           uint32_t outRowStride,
                                           uint32_t srcStrideBlocks) {
  AscendC::DataCopyExtParams params{(uint16_t)blockCount, segLen * (uint32_t)sizeof(T),
                                    srcStrideBlocks,
                                    (outRowStride - segLen) * (uint32_t)sizeof(T), 0};
  AscendC::DataCopyPad(dst, src, params);
}

// Route U load: read `blockCount` rows (each `cb` valid elems) from GM into UB.
// Each GM row spans `segLen` elems (so consecutive rows' col-blocks are segLen apart);
// Normal mode auto-pads each block to 32B in UB (dstStride=0). For cb<segLen this is a
// column-block of a wider segment. Non-aligned cb is fine (Normal byte-blockLen).
template <typename T>
__aicore__ inline void LoadStrided(const AscendC::LocalTensor<T>& dst,
                                   const AscendC::GlobalTensor<T>& src,
                                   uint32_t blockCount, uint32_t cb, uint32_t segLen) {
  AscendC::DataCopyExtParams params{(uint16_t)blockCount, cb * (uint32_t)sizeof(T),
                                    (segLen - cb) * (uint32_t)sizeof(T), 0, 0};
  AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
  AscendC::DataCopyPad(dst, src, params, padParams);
}

template <typename T>
class KernelConcat {
 public:
  __aicore__ inline KernelConcat() {}
  __aicore__ inline void Init(GM_ADDR y, const ConcatTilingData& t,
                              AscendC::ListTensorDesc inputList) {
    inputList_ = inputList;
    y_gm.SetGlobalBuffer((__gm__ T*)y);
    numActive = t.numActive;
    outer = t.outer;
    inner = t.inner;
    sumW = t.sumW;
    tileLen = t.tileLen;
    totalRuns = t.totalRuns;
    route = t.route;
    perCoreElems = t.perCoreElems;
    totalElems = t.totalElems;
    nRowChunks = t.nRowChunks;
    rowsPerChunk = t.rowsPerChunk;
    gR = t.gR;
    gCoreRows = t.gCoreRows;
    gUbSrcElems = t.gUbSrcElems;
    // Only [0,numActive) entries are ever read (all routes index k<numActive).
    // Copying all MAX_T=256 was ~1024 scalar UB ops/core -- the fixed scalar cost
    // that dominated tiny/overhead-bound cases (e.g. int8 axis0). Bound to numActive.
    for (uint32_t i = 0; i < numActive; i++) {
      origIdx[i] = t.origIdx[i];
      widths[i] = t.widths[i];
      offsets[i] = t.offsets[i];
      slotElem[i] = t.slotElem[i];
    }
    uint32_t bi = AscendC::GetBlockIdx();
    runStart = bi * t.runsPerCore;
    runEnd = runStart + t.runsPerCore;
    if (runEnd > totalRuns) runEnd = totalRuns;
    if (route == 3) {
      uint32_t offElems = ((sumW + 63) / 64) * 64;
      uint32_t pitchElem = ((sumW + 127) / 128) * 128;
      pipe.InitBuffer(gLoadQue, 1, gUbSrcElems * sizeof(T));
      pipe.InitBuffer(gStoreQue, 2, GATHER_WBATCH * pitchElem * sizeof(T));  // GATHER_WBATCH gathered rows/store batch
      pipe.InitBuffer(gBaseBuf, offElems * sizeof(int32_t));
      pipe.InitBuffer(gIncBuf, offElems * sizeof(int32_t));
      pipe.InitBuffer(gCurBuf, offElems * sizeof(int32_t));
    } else if (route == 5) {
      const uint32_t PAD = 32768u / (uint32_t)sizeof(T);   // ~32KB per buffer
      pipe.InitBuffer(pReadQ, 2, PAD * sizeof(T));    // packed contiguous read (full BW), double-buffered
      pipe.InitBuffer(pStoreQ, 2, PAD * sizeof(T));  // Gather output -> strided store, double-buffered
      pipe.InitBuffer(pIdxBuf, PAD * sizeof(uint32_t));     // repack byte-offset index
    } else {
      pipe.InitBuffer(que, BUFFER_NUM, tileLen * sizeof(T));
    }
  }

  __aicore__ inline void Process() {
    ProcessImpl();
  }

  __aicore__ inline void ProcessImpl() {
    if (route == 1) {
      ProcessDim0Split();
      return;
    }
    if (route == 2) {
      ProcessStrided();
      return;
    }
    if (route == 3) {
      ProcessGatherLast();
      return;
    }
    if (route == 4) {
      ProcessUniversal();
      return;
    }
    if (route == 5) {
      ProcessUniversalPacked();
      return;
    }
    if (numActive == 0) return;
    for (uint32_t r = runStart; r < runEnd; r++) {
      uint32_t o = r / numActive;
      uint32_t k = r % numActive;
      uint32_t wi = widths[k];
      uint64_t segLen = (uint64_t)wi * inner;
      if (segLen == 0) continue;
      x_gm.SetGlobalBuffer(inputList_.GetDataPtr<T>(origIdx[k]));
      uint64_t srcBase = (uint64_t)o * segLen;
      uint64_t dstOff =
          (uint64_t)o * sumW * inner + (uint64_t)offsets[k] * inner;
      for (uint64_t c = 0; c < segLen; c += tileLen) {
        uint32_t len =
            (segLen - c) < tileLen ? (uint32_t)(segLen - c) : tileLen;
        AscendC::LocalTensor<T> a = que.AllocTensor<T>();
        PadIn(a, x_gm[srcBase + c], len);
        que.EnQue(a);
        a = que.DeQue<T>();
        PadOut(y_gm[dstOff + c], a, len);
        que.FreeTensor(a);
      }
    }
  }

  // Route P: dim0 concat -> output is the inputs laid end-to-end contiguously.
  // This core owns output element range [gStart,gEnd); for each input whose
  // output span [eo,eo+ln) overlaps, copy the overlap (contiguous in both src
  // and dst). Fills all cores regardless of how few inputs there are.
  __aicore__ inline void ProcessDim0Split() {
    if (numActive == 0) return;
    uint64_t gStart = (uint64_t)AscendC::GetBlockIdx() * perCoreElems;
    uint64_t gEnd = gStart + perCoreElems;
    if (gEnd > totalElems) gEnd = totalElems;
    if (gStart >= gEnd) return;
    for (uint32_t k = 0; k < numActive; k++) {
      uint64_t eo = (uint64_t)offsets[k] * inner;   // output elem offset of input k
      uint64_t ln = (uint64_t)widths[k] * inner;     // input k length (outer==1)
      if (ln == 0) continue;
      uint64_t lo = eo > gStart ? eo : gStart;
      uint64_t hiK = eo + ln;
      uint64_t hi = hiK < gEnd ? hiK : gEnd;
      if (lo >= hi) continue;
      x_gm.SetGlobalBuffer(inputList_.GetDataPtr<T>(origIdx[k]));
      for (uint64_t c = lo; c < hi; c += tileLen) {
        uint32_t len = (hi - c) < tileLen ? (uint32_t)(hi - c) : tileLen;
        AscendC::LocalTensor<T> a = que.AllocTensor<T>();
        PadIn(a, x_gm[c - eo], len);   // src offset within input k = c - eo
        que.EnQue(a);
        a = que.DeQue<T>();
        PadOut(y_gm[c], a, len);
        que.FreeTensor(a);
      }
    }
  }

  // Route S: aligned strided collapse (non-first axis). Each work unit u maps to
  // (input k, row-chunk rc). For that input's row range, load a batch of rows
  // contiguously into UB (1 DataCopyPad), then store the batch to the output with
  // a per-row gap in a single multi-row strided DataCopyPad -- collapsing v1's
  // per-row tiny copies. Requires segLen 32B-aligned (gated in host).
  __aicore__ inline void ProcessStrided() {
    if (numActive == 0) return;
    // route 2 reuse: totalRuns=units, runsPerCore=unitsPerCore; runStart/runEnd
    // (set in Init) already hold this core's unit range.
    uint32_t uStart = runStart;
    uint32_t uEnd = runEnd;
    uint64_t outRowStride = (uint64_t)sumW * inner;
    for (uint32_t u = uStart; u < uEnd; u++) {
      uint32_t k = u % numActive;
      uint32_t rc = u / numActive;
      uint32_t r0 = rc * rowsPerChunk;
      uint32_t r1 = r0 + rowsPerChunk;
      if (r1 > outer) r1 = outer;
      if (r0 >= r1) continue;
      uint32_t segLen = widths[k] * inner;
      if (segLen == 0) continue;
      x_gm.SetGlobalBuffer(inputList_.GetDataPtr<T>(origIdx[k]));
      uint64_t dstColOff = (uint64_t)offsets[k] * inner;
      uint32_t rb = tileLen / segLen;     // rows per batch (segLen <= tileLen, gated)
      if (rb < 1) rb = 1;
      if (rb > 4095) rb = 4095;           // DataCopyPad blockCount limit
      for (uint32_t rs = r0; rs < r1; rs += rb) {
        uint32_t R = (r1 - rs) < rb ? (r1 - rs) : rb;
        uint64_t srcBase = (uint64_t)rs * segLen;
        uint64_t dstBase = (uint64_t)rs * outRowStride + dstColOff;
        AscendC::LocalTensor<T> a = que.AllocTensor<T>();
        PadIn(a, x_gm[srcBase], R * segLen);                 // contiguous load
        que.EnQue(a);
        a = que.DeQue<T>();
        StoreStrided(y_gm[dstBase], a, R, segLen, (uint32_t)outRowStride);
        que.FreeTensor(a);
      }
    }
  }


  // Route U: universal strided-collapse for ANY non-first-axis (any alignment/dtype).
  // Unit = (input k, row-chunk). For that input's rows [r0,r1): batch R rows + col-block
  // cb (cb<segLen only when a row's segment exceeds the UB tile). Each batch =
  // 1 blockCount=R Normal-mode load (auto-pad rows to 32B in UB) + 1 multi-row strided
  // store (GM-byte dstStride skips the other inputs' columns). Collapses the v1 flood.
  __aicore__ inline void ProcessUniversal() {
    if (numActive == 0) return;
    uint32_t e32 = 32u / (uint32_t)sizeof(T);
    if (e32 < 1) e32 = 1;
    uint64_t rowW = (uint64_t)sumW * inner;
    for (uint32_t u = runStart; u < runEnd; u++) {
      uint32_t k = u % numActive;
      uint32_t rc = u / numActive;
      uint32_t r0 = rc * rowsPerChunk;
      uint32_t r1 = r0 + rowsPerChunk;
      if (r1 > outer) r1 = outer;
      if (r0 >= r1) continue;
      uint32_t segLen = widths[k] * inner;
      if (segLen == 0) continue;
      uint64_t dstColOff = (uint64_t)offsets[k] * inner;
      x_gm.SetGlobalBuffer(inputList_.GetDataPtr<T>(origIdx[k]));
      // pick col-block cbLen and rows-per-batch R so R*AlignUp(cbLen,e32) <= tileLen
      uint32_t pitch = ((segLen + e32 - 1) / e32) * e32;   // AlignUp(segLen, e32)
      uint32_t cbLen, R;
      if (pitch <= tileLen) {
        cbLen = segLen;
        R = tileLen / pitch;
        if (R < 1) R = 1;
        if (R > 4095u) R = 4095u;
      } else {
        cbLen = (tileLen / e32) * e32;   // largest e32-multiple <= tileLen
        if (cbLen < e32) cbLen = e32;
        R = 1;
      }
      for (uint32_t rs = r0; rs < r1; rs += R) {
        uint32_t Rb = (r1 - rs) < R ? (r1 - rs) : R;
        for (uint32_t c0 = 0; c0 < segLen; c0 += cbLen) {
          uint32_t cb = (segLen - c0) < cbLen ? (segLen - c0) : cbLen;
          uint64_t srcBase = (uint64_t)rs * segLen + c0;
          uint64_t dstBase = (uint64_t)rs * rowW + dstColOff + c0;
          AscendC::LocalTensor<T> a = que.AllocTensor<T>();
          LoadStrided(a, x_gm[srcBase], Rb, cb, segLen);
          que.EnQue(a);
          a = que.DeQue<T>();
          StoreStrided(y_gm[dstBase], a, Rb, cb, (uint32_t)rowW);
          que.FreeTensor(a);
        }
      }
    }
  }

  // Route 5: read-efficient route-U. Same (input,row-chunk) unit + strided store as
  // route 4, but the READ is a blockCount=1 PACKED load (full HBM BW) instead of route
  // 4's Rb padded descriptors (which throttle mte2 to ~184 GB/s). A UB->UB Gather then
  // expands the packed rows into the exact 32B-padded pitch StoreStrided consumes (for
  // 32B-aligned segments the Gather is an identity copy, kept for a uniform pipeline).
  // Host gate guarantees maxSeg <= PAD so a whole segment always fits one packed read.
  // Gather has no int8 on A2 -> host keeps int8 on route 4.
  __aicore__ inline void ProcessUniversalPacked() {
    if (numActive == 0) return;
    const uint32_t e32 = 32u / (uint32_t)sizeof(T);
    const uint32_t PAD = 32768u / (uint32_t)sizeof(T);
    uint64_t rowW = (uint64_t)sumW * inner;
    for (uint32_t u = runStart; u < runEnd; u++) {
      uint32_t k = u % numActive;
      uint32_t rc = u / numActive;
      uint32_t r0 = rc * rowsPerChunk;
      uint32_t r1 = r0 + rowsPerChunk;
      if (r1 > outer) r1 = outer;
      if (r0 >= r1) continue;
      uint32_t segLen = widths[k] * inner;
      if (segLen == 0) continue;
      uint64_t dstColOff = (uint64_t)offsets[k] * inner;
      x_gm.SetGlobalBuffer(inputList_.GetDataPtr<T>(origIdx[k]));
      uint32_t pitch = ((segLen + e32 - 1) / e32) * e32;   // AlignUp(segLen, e32) <= PAD

      uint32_t R = PAD / pitch;
      if (R > (r1 - r0)) R = r1 - r0;
      if (R < 1) R = 1;
      if (R > 4095u) R = 4095u;

      // Build the repack byte-offset index once per unit:
      //   idx[r*pitch + c] = (r*segLen + c) * sizeof(T)
      // valid data at c<segLen; c in [segLen,pitch) reads harmlessly inside readBuf
      // (max index (R-1)*segLen+(pitch-1) < R*pitch <= PAD) and is never stored. pitch
      // is a multiple of e32 so every row start idx[r*pitch] is 32B-aligned for Adds.
      AscendC::LocalTensor<int32_t> idx = pIdxBuf.Get<int32_t>();
      AscendC::ArithProgression(idx, (int32_t)0, (int32_t)sizeof(T), pitch);  // row 0: c*sizeof
      AscendC::PipeBarrier<PIPE_V>();
      for (uint32_t r = 1; r < R; r++) {
        AscendC::Adds(idx[r * pitch], idx, (int32_t)(r * segLen * (uint32_t)sizeof(T)), pitch);
      }
      AscendC::PipeBarrier<PIPE_V>();  // idx WAR is vec-in-order safe; PIPE_V keeps mte2/mte3 overlapping across units (PIPE_ALL flushed the pipeline 1080x)

      for (uint32_t rs = r0; rs < r1; rs += R) {
        uint32_t Rb = (r1 - rs) < R ? (r1 - rs) : R;
        uint64_t srcBase = (uint64_t)rs * segLen;
        uint64_t dstBase = (uint64_t)rs * rowW + dstColOff;
        AscendC::LocalTensor<T> a = pReadQ.AllocTensor<T>();
        AscendC::DataCopyExtParams cp{1, Rb * segLen * (uint32_t)sizeof(T), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pd{false, 0, 0, 0};
        AscendC::DataCopyPad(a, x_gm[srcBase], cp, pd);   // ONE packed contiguous read
        pReadQ.EnQue(a);
        a = pReadQ.DeQue<T>();
        AscendC::LocalTensor<T> b = pStoreQ.AllocTensor<T>();
        AscendC::Gather(b, a, idx.template ReinterpretCast<uint32_t>(), (uint32_t)0, Rb * pitch);
        pReadQ.FreeTensor(a);
        pStoreQ.EnQue(b);
        b = pStoreQ.DeQue<T>();
        StoreStrided(y_gm[dstBase], b, Rb, segLen, (uint32_t)rowW);
        pStoreQ.FreeTensor(b);
      }
    }
  }



  // Route G: non-aligned last-axis (inner==1) via Gather column-assembly.
  // Build, once per core, a byte-offset table per output column (base[c]+row*inc[c]).
  // Per row-batch: load each input's gR rows contiguously into ubSrc (each input block
  // 32B-aligned), Gather each output row into ubDst (vector handles non-aligned column
  // scatter), then store the batch's rows contiguous. Collapses the tiny-copy flood.
  __aicore__ inline void ProcessGatherLast() {
    if (numActive == 0) return;
    uint32_t bi = AscendC::GetBlockIdx();
    uint32_t ro0 = bi * gCoreRows;
    uint32_t ro1 = ro0 + gCoreRows;
    if (ro1 > outer) ro1 = outer;
    if (ro0 >= ro1) return;
    const int32_t sz = (int32_t)sizeof(T);
    AscendC::LocalTensor<int32_t> base = gBaseBuf.Get<int32_t>();
    AscendC::LocalTensor<int32_t> inc = gIncBuf.Get<int32_t>();
    // scalar-fill the per-column byte-offset base/increment (once per core)
    for (uint32_t k = 0; k < numActive; k++) {
      uint32_t w = widths[k];
      uint32_t oc = offsets[k];
      int32_t so = (int32_t)slotElem[k];
      int32_t incv = (int32_t)w * sz;
      for (uint32_t i = 0; i < w; i++) {
        base.SetValue(oc + i, (so + (int32_t)i) * sz);
        inc.SetValue(oc + i, incv);
      }
    }
    AscendC::PipeBarrier<PIPE_ALL>();   // scalar fill -> vector consumers
    AscendC::LocalTensor<int32_t> cur = gCurBuf.Get<int32_t>();
    for (uint32_t rb = ro0; rb < ro1; rb += gR) {
      uint32_t Rb = (ro1 - rb) < gR ? (ro1 - rb) : gR;
      AscendC::LocalTensor<T> src = gLoadQue.AllocTensor<T>();
      for (uint32_t k = 0; k < numActive; k++) {
        uint32_t w = widths[k];
        if (w == 0) continue;
        x_gm.SetGlobalBuffer(inputList_.GetDataPtr<T>(origIdx[k]));
        PadIn(src[slotElem[k]], x_gm[(uint64_t)rb * w], Rb * w);
      }
      gLoadQue.EnQue(src);
      src = gLoadQue.DeQue<T>();
      AscendC::Adds(cur, base, (int32_t)0, sumW);          // cur = base (row 0 byte offsets)
      if ((sumW * (uint32_t)sizeof(T)) % 32u == 0u) {
        // sumW 32B-aligned -> gather rows PACKED at jj*sumW (aligned) and write cnt rows in ONE
        // contiguous PadOut -> writes & per-row scalar overhead /= GATHER_WBATCH.
        for (uint32_t j = 0; j < Rb;) {
          uint32_t cnt = (Rb - j) < GATHER_WBATCH ? (Rb - j) : GATHER_WBATCH;
          AscendC::LocalTensor<T> rows = gStoreQue.AllocTensor<T>();
          for (uint32_t jj = 0; jj < cnt; jj++) {
            AscendC::Gather(rows[jj * sumW], src, cur.template ReinterpretCast<uint32_t>(),
                            (uint32_t)0, sumW);
            if (j + jj + 1 < Rb) AscendC::Add(cur, cur, inc, sumW);
          }
          gStoreQue.EnQue(rows);
          rows = gStoreQue.DeQue<T>();
          PadOut(y_gm[(uint64_t)(rb + j) * sumW], rows, cnt * sumW);  // cnt packed rows contiguous
          gStoreQue.FreeTensor(rows);
          j += cnt;
        }
      } else {
        // non-32B-aligned sumW: per-row gather+store (packed batching not possible)
        for (uint32_t j = 0; j < Rb; j++) {
          AscendC::LocalTensor<T> row = gStoreQue.AllocTensor<T>();
          AscendC::Gather(row, src, cur.template ReinterpretCast<uint32_t>(), (uint32_t)0, sumW);
          if (j + 1 < Rb) AscendC::Add(cur, cur, inc, sumW);
          gStoreQue.EnQue(row);
          row = gStoreQue.DeQue<T>();
          PadOut(y_gm[(uint64_t)(rb + j) * sumW], row, sumW);
          gStoreQue.FreeTensor(row);
        }
      }
      gLoadQue.FreeTensor(src);
    }
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT,
                    BUFFER_NUM>
      que;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> gLoadQue;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> gStoreQue;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> gBaseBuf, gIncBuf, gCurBuf;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> pReadQ;    // route 5: packed contiguous read
  AscendC::TQue<AscendC::QuePosition::VECOUT, 2> pStoreQ; // route 5: Gather output -> store
  AscendC::TBuf<AscendC::QuePosition::VECCALC> pIdxBuf;    // route 5: repack index
  AscendC::GlobalTensor<T> x_gm, y_gm;
  AscendC::ListTensorDesc inputList_;
  uint32_t numActive, outer, inner, sumW, tileLen, totalRuns, runStart, runEnd;
  uint32_t route, perCoreElems, totalElems, nRowChunks, rowsPerChunk;
  uint32_t gR, gCoreRows, gUbSrcElems;
  uint32_t origIdx[MAX_T], widths[MAX_T], offsets[MAX_T], slotElem[MAX_T];
};

// Lean route-1 kernel: dim-0 concat is one contiguous output stream split by
// output element range. Keep only the fields and code this hot path uses.
template <typename T>
class KernelConcatRoute1 {
 public:
  __aicore__ inline KernelConcatRoute1() {}
  __aicore__ inline void Init(AscendC::TPipe* pipeIn, GM_ADDR y, const ConcatTilingData& t,
                              AscendC::ListTensorDesc inputList) {
    pipe = pipeIn;
    inputList_ = inputList;
    y_gm.SetGlobalBuffer((__gm__ T*)y);
    numActive = t.numActive;
    inner = t.inner;
    tileLen = t.tileLen;
    for (uint32_t i = 0; i < numActive; i++) {
      origIdx[i] = t.origIdx[i];
      widths[i] = t.widths[i];
      offsets[i] = t.offsets[i];
    }
    uint32_t blockIdx = AscendC::GetBlockIdx();
    gStart = t.slotElem[blockIdx];
    gEnd = t.slotElem[blockIdx + 1];
    pipe->InitBuffer(que, BUFFER_NUM, tileLen * sizeof(T));
  }

  __aicore__ inline void Process() {
    if (numActive == 0) return;
    if (gStart >= gEnd) return;
    for (uint32_t k = 0; k < numActive; k++) {
      uint64_t eo = (uint64_t)offsets[k] * inner;
      uint64_t ln = (uint64_t)widths[k] * inner;
      if (ln == 0) continue;
      uint64_t lo = eo > gStart ? eo : gStart;
      uint64_t hiK = eo + ln;
      uint64_t hi = hiK < gEnd ? hiK : gEnd;
      if (lo >= hi) continue;
      x_gm.SetGlobalBuffer(inputList_.GetDataPtr<T>(origIdx[k]));
      for (uint64_t c = lo; c < hi; c += tileLen) {
        uint32_t len = (hi - c) < tileLen ? (uint32_t)(hi - c) : tileLen;
        AscendC::LocalTensor<T> a = que.AllocTensor<T>();
        PadIn(a, x_gm[c - eo], len);
        que.EnQue(a);
        a = que.DeQue<T>();
        PadOut(y_gm[c], a, len);
        que.FreeTensor(a);
      }
    }
  }

 private:
  AscendC::TPipe* pipe;
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT,
                    BUFFER_NUM> que;
  AscendC::GlobalTensor<T> x_gm, y_gm;
  AscendC::ListTensorDesc inputList_;
  uint32_t numActive, inner, tileLen;
  uint64_t gStart, gEnd;
  uint32_t origIdx[MAX_T], widths[MAX_T], offsets[MAX_T];
};

// Predecoded dim-0 path: the host packs up to three contiguous pieces for
// each core into the existing tiling arrays. This removes the generic route-1
// scan over every input and leaves only large GM->UB->GM transfers on device.
template <typename T>
class KernelConcatDim0Direct {
 public:
  __aicore__ inline KernelConcatDim0Direct() {}
  __aicore__ inline void Init(AscendC::TPipe* pipeIn, GM_ADDR x, GM_ADDR y,
                              __gm__ const ConcatTilingData* t,
                              uint32_t blockIdx) {
    pipe = pipeIn;
    __gm__ uint64_t* listHead = reinterpret_cast<__gm__ uint64_t*>(x);
    dataPtrTable = listHead + ((*listHead) >> 3);
    y_gm.SetGlobalBuffer((__gm__ T*)y);
    tileLen = t->tileLen;
    uint32_t base = blockIdx * 3u;
    for (uint32_t j = 0; j < 3; j++) {
      inputIdx[j] = t->origIdx[base + j];
      srcOff[j] = t->widths[base + j];
      dstOff[j] = t->offsets[base + j];
      length[j] = t->slotElem[base + j];
    }
    pipe->InitBuffer(que, 1, tileLen * sizeof(T));
  }

  __aicore__ inline void Process() {
    CopyPiece(0);
    CopyPiece(1);
    CopyPiece(2);
  }

 private:
  __aicore__ inline void CopyPiece(uint32_t j) {
    uint32_t n = length[j];
    if (n == 0) return;
    x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(*(dataPtrTable + inputIdx[j])));
    x_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    for (uint32_t c = 0; c < n; c += tileLen) {
      uint32_t len = (n - c) < tileLen ? (n - c) : tileLen;
      AscendC::LocalTensor<T> local = que.AllocTensor<T>();
      PadIn(local, x_gm[srcOff[j] + c], len);
      que.EnQue(local);
      local = que.DeQue<T>();
      PadOut(y_gm[dstOff[j] + c], local, len);
      que.FreeTensor(local);
    }
  }

  AscendC::TPipe* pipe;
  // One queue buffer: with two, a chunk's read overlaps the previous chunk's store, and on the
  // dim-0 int8 case that overlap measured slower (C3 12.90 -> 11.84 us with a single buffer).
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> que;
  AscendC::GlobalTensor<T> x_gm, y_gm;
  __gm__ uint64_t* dataPtrTable;
  uint32_t tileLen;
  uint32_t inputIdx[3], srcOff[3], dstOff[3], length[3];
};

// Three-unit lean path. Host tiling has already decoded the units owned by
// each core into {input, srcBase, dstBase, segmentLength} tuples.
template <typename T>
class KernelConcatPredecoded3 {
 public:
  __aicore__ inline KernelConcatPredecoded3() {}
  __aicore__ inline void Init(AscendC::TPipe* pipeIn, GM_ADDR x, GM_ADDR y,
                              __gm__ const ConcatTilingData* t,
                              uint32_t blockIdx) {
    pipe = pipeIn;
    __gm__ uint64_t* listHead = reinterpret_cast<__gm__ uint64_t*>(x);
    dataPtrTable = listHead + ((*listHead) >> 3);
    y_gm.SetGlobalBuffer((__gm__ T*)y);
    rowBatch = t->rowsPerChunk;
    tRowWidth = t->sumW * t->inner;
    uint32_t base = blockIdx * 3u;
    for (uint32_t j = 0; j < 3; j++) {
      inputIdx[j] = t->origIdx[base + j];
      srcBase[j] = t->widths[base + j];
      dstBase[j] = t->offsets[base + j];
      segLen[j] = t->slotElem[base + j];
    }
    pipe->InitBuffer(que, BUFFER_NUM, 4096u * sizeof(T));
  }

  __aicore__ inline void Process() {
    CopyUnit(0);
    CopyUnit(1);
    CopyUnit(2);
  }

 private:
  __aicore__ inline void CopyUnit(uint32_t j) {
    x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(*(dataPtrTable + inputIdx[j])));
    AscendC::LocalTensor<T> local = que.AllocTensor<T>();
    LoadStrided(local, x_gm[srcBase[j]], rowBatch, segLen[j], segLen[j]);
    que.EnQue(local);
    local = que.DeQue<T>();
    StoreStrided(y_gm[dstBase[j]], local, rowBatch, segLen[j], tRowWidth);
    que.FreeTensor(local);
  }

  AscendC::TPipe* pipe;
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT,
                    BUFFER_NUM> que;
  AscendC::GlobalTensor<T> x_gm, y_gm;
  __gm__ uint64_t* dataPtrTable;
  uint32_t inputIdx[3], srcBase[3], dstBase[3], segLen[3];
  uint32_t rowBatch, tRowWidth;
};

// Four-unit lean path. Host tiling expands the units owned by each core and
// packs the one-row/two-row DMA choice in the high byte of segLen.
template <typename T>
class KernelConcatPredecoded4 {
 public:
  __aicore__ inline KernelConcatPredecoded4() {}
  __aicore__ inline void Init(AscendC::TPipe* pipeIn, GM_ADDR x, GM_ADDR y,
                              __gm__ const ConcatTilingData* t,
                              uint32_t blockIdx) {
    pipe = pipeIn;
    __gm__ uint64_t* listHead = reinterpret_cast<__gm__ uint64_t*>(x);
    dataPtrTable = listHead + ((*listHead) >> 3);
    y_gm.SetGlobalBuffer((__gm__ T*)y);
    rowW = t->sumW * t->inner;
    tileLen = t->tileLen;
    uint32_t base = blockIdx * 4u;
    for (uint32_t j = 0; j < 4; j++) {
      inputIdx[j] = t->origIdx[base + j];
      srcBase[j] = t->widths[base + j];
      dstBase[j] = t->offsets[base + j];
      uint32_t packed = t->slotElem[base + j];
      segLen[j] = packed & 0x00ffffffu;
      rowBatch[j] = packed >> 24;
    }
    pipe->InitBuffer(que, BUFFER_NUM, tileLen * sizeof(T));
  }

  __aicore__ inline void Process() {
    CopyUnit(0);
    CopyUnit(1);
    CopyUnit(2);
    CopyUnit(3);
  }

 private:
  __aicore__ inline void CopyOneRow(uint32_t j, uint32_t row) {
    uint32_t src = srcBase[j] + row * segLen[j];
    uint32_t dst = dstBase[j] + row * rowW;
    AscendC::LocalTensor<T> local = que.AllocTensor<T>();
    LoadStrided(local, x_gm[src], 1u, segLen[j], segLen[j]);
    que.EnQue(local);
    local = que.DeQue<T>();
    StoreStrided(y_gm[dst], local, 1u, segLen[j], rowW);
    que.FreeTensor(local);
  }

  __aicore__ inline void CopyUnit(uint32_t j) {
    if (segLen[j] == 0) return;
    x_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(*(dataPtrTable + inputIdx[j])));
    x_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    if (rowBatch[j] == 2u) {
      AscendC::LocalTensor<T> local = que.AllocTensor<T>();
      LoadStrided(local, x_gm[srcBase[j]], 2u, segLen[j], segLen[j]);
      que.EnQue(local);
      local = que.DeQue<T>();
      StoreStrided(y_gm[dstBase[j]], local, 2u, segLen[j], rowW);
      que.FreeTensor(local);
    } else {
      CopyOneRow(j, 0u);
      CopyOneRow(j, 1u);
    }
  }



  AscendC::TPipe* pipe;
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT,
                    BUFFER_NUM> que;

  AscendC::GlobalTensor<T> x_gm, y_gm;
  __gm__ uint64_t* dataPtrTable;
  uint32_t rowW, tileLen;
  uint32_t inputIdx[4], srcBase[4], dstBase[4], segLen[4], rowBatch[4];

};

// Lean route-4 kernel: universal non-first-axis strided copy. The host tiling,
// unit order, row-chunk geometry, and DMA calls are identical to the proven
// general-kernel route. Keeping only the fields used by route 4 isolates
// tiny workload/wide-row workload/large-strided workload from the route-2/3/5 instruction footprint.
template <typename T, bool DISABLE_INPUT_L2 = false>
class KernelConcatRoute4 {
 public:
  __aicore__ inline KernelConcatRoute4() {}
  // Templated on the tiling handle so key 11 can hand over a __gm__ pointer and
  // skip GET_TILING_DATA's whole-struct copy: Init reads 8 scalars plus three
  // numActive-long arrays, far under the ~55-field break-even against a 4180 B
  // bulk copy. key 4 still passes a pointer to the local copy.
  template <typename TD>
  __aicore__ inline void Init(AscendC::TPipe* pipeIn, GM_ADDR y, TD t,
                              AscendC::ListTensorDesc inputList) {
    pipe = pipeIn;
    inputList_ = inputList;
    y_gm.SetGlobalBuffer((__gm__ T*)y);
    numActive = t->numActive;
    outer = t->outer;
    inner = t->inner;
    sumW = t->sumW;
    tileLen = t->tileLen;
    rowsPerChunk = t->rowsPerChunk;
    for (uint32_t i = 0; i < numActive; i++) {
      origIdx[i] = t->origIdx[i];
      widths[i] = t->widths[i];
      offsets[i] = t->offsets[i];
    }
    uint32_t blockIdx = AscendC::GetBlockIdx();
    runStart = blockIdx * t->runsPerCore;
    runEnd = runStart + t->runsPerCore;
    if (runEnd > t->totalRuns) runEnd = t->totalRuns;
    // CONTIGUOUS-READ head slots: one 32B slot per row of a batch (HD_ROWS * 32 B).
    // Route 4's host tiling already spends the WHOLE 192 KiB UB on two 96 KiB ping-pong
    // tiles, so this scratch must come OUT of the tile, not on top of it -- a first attempt
    // that simply added it overflowed UB and faulted with "MPU address access is invalid".
    {
      uint32_t hdElems = HD_ROWS * 32u / (uint32_t)sizeof(T);
      if (tileLen > 2u * hdElems) tileLen -= hdElems;
    }
    pipe->InitBuffer(que, BUFFER_NUM, tileLen * sizeof(T));
    pipe->InitBuffer(hdBuf, HD_ROWS * 32u);
  }

  __aicore__ inline void Process() {
    if (numActive == 0) return;
    uint32_t e32 = 32u / (uint32_t)sizeof(T);
    if (e32 < 1) e32 = 1;
    uint64_t rowW = (uint64_t)sumW * inner;
    for (uint32_t u = runStart; u < runEnd; u++) {
      uint32_t k = u % numActive;
      uint32_t rowChunk = u / numActive;
      uint32_t r0 = rowChunk * rowsPerChunk;
      uint32_t r1 = r0 + rowsPerChunk;
      if (r1 > outer) r1 = outer;
      if (r0 >= r1) continue;
      uint32_t segLen = widths[k] * inner;
      if (segLen == 0) continue;
      uint64_t dstColOff = (uint64_t)offsets[k] * inner;
      x_gm.SetGlobalBuffer(inputList_.GetDataPtr<T>(origIdx[k]));
      if constexpr (DISABLE_INPUT_L2) {
        x_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
      }
      uint32_t pitch = ((segLen + e32 - 1) / e32) * e32;
      uint32_t cbLen;
      uint32_t rowsPerBatch;
      if (pitch <= tileLen) {
        cbLen = segLen;
        rowsPerBatch = tileLen / pitch;
        if (rowsPerBatch < 1) rowsPerBatch = 1;
        if (rowsPerBatch > 4095u) rowsPerBatch = 4095u;
      } else {
        cbLen = (tileLen / e32) * e32;
        if (cbLen < e32) cbLen = e32;
        rowsPerBatch = 1;
      }
      // CONTIGUOUS-READ PATH.  When cbLen >= segLen the GM source for `rowBatch` consecutive
      // rows of this input is ONE unbroken run (LoadStrided's srcStride is 0), yet it is
      // described as rowBatch separate blocks purely so each UB row lands on a 32B boundary --
      // StoreStrided needs that, because a multi-block UB->GM descriptor advances its UB side by
      // ceil(blockLen/32)*32.  Measured on this machine (IndexAdd c4, 279 MB): that rounding
      // costs ~6% of read bandwidth AND a wall-minus-aiv tax the per-pipe table does not show;
      // one blockCount=1 descriptor over the same bytes ran 746 GB/s vs 703.
      // So: read it as a SINGLE block, widened to whole 32B lines at both ends (up to e32-1
      // extra elements of this same input, never stored), because a GM->UB read whose first or
      // last byte is off a 32B boundary misses the fast path (C4 int32: mte2 64 -> 58 us through
      // DataCopyPad; plain DataCopy on the same window only reached 61).  Row r then starts at
      // UB element leadE + r*segLen, and the store's alignment is bought back with a head split:
      // with hE = ((32 - ((leadE + r*segLen)*sizeof(T)) % 32) % 32) / sizeof(T) the body
      // local[leadE + r*segLen + hE] is exactly 32B-aligned and stores straight out of UB, while
      // the <= e32-1 leading elements detour through a 32B-aligned scratch slot.  A window that
      // would run past the input keeps the exact bytes (leadE = trailE = 0).
      const bool contig = (cbLen >= segLen) && (segLen >= e32) && (rowsPerBatch <= HD_ROWS) &&
                          ((uint64_t)rowsPerBatch * segLen + 2u * e32 <= (uint64_t)tileLen);
      const uint64_t inputElems = (uint64_t)outer * segLen;
      for (uint32_t rs = r0; rs < r1; rs += rowsPerBatch) {
        uint32_t rowBatch = (r1 - rs) < rowsPerBatch ? (r1 - rs) : rowsPerBatch;
        if (contig) {
          uint64_t srcBase = (uint64_t)rs * segLen;
          uint64_t dstBase = (uint64_t)rs * rowW + dstColOff;
          uint64_t readElems = (uint64_t)rowBatch * segLen;
          uint32_t leadE = (uint32_t)((srcBase * sizeof(T)) & 31u) / (uint32_t)sizeof(T);
          uint32_t trailE =
              (uint32_t)((32u - (((srcBase + readElems) * sizeof(T)) & 31u)) & 31u) /
              (uint32_t)sizeof(T);
          AscendC::LocalTensor<T> local = que.AllocTensor<T>();
          if (srcBase + readElems + trailE > inputElems) {
            leadE = 0u;
            trailE = 0u;
          }
          AscendC::DataCopyExtParams rp{
              1, (uint32_t)((leadE + readElems + trailE) * sizeof(T)), 0, 0, 0};
          AscendC::DataCopyPadExtParams<T> pd{false, 0, 0, 0};
          AscendC::DataCopyPad(local, x_gm[srcBase - leadE], rp, pd);
          que.EnQue(local);
          local = que.DeQue<T>();
          AscendC::LocalTensor<T> hd = hdBuf.Get<T>();
          for (uint32_t r = 0; r < rowBatch; r++) {
            uint64_t rowStart = leadE + (uint64_t)r * segLen;
            uint32_t hE = ((32u - (uint32_t)((rowStart * sizeof(T)) & 31u)) & 31u) /
                          (uint32_t)sizeof(T);
            for (uint32_t t = 0; t < hE; t++)
              hd.SetValue(r * e32 + t, local.GetValue(rowStart + t));
          }
          for (uint32_t r = 0; r < rowBatch; r++) {
            uint64_t rowStart = leadE + (uint64_t)r * segLen;
            uint32_t hE = ((32u - (uint32_t)((rowStart * sizeof(T)) & 31u)) & 31u) /
                          (uint32_t)sizeof(T);
            uint64_t d = dstBase + (uint64_t)r * rowW;
            if (hE != 0u) {
              AscendC::DataCopyExtParams hp{1, hE * (uint32_t)sizeof(T), 0, 0, 0};
              AscendC::DataCopyPad(y_gm[d], hd[r * e32], hp);
            }
            AscendC::DataCopyExtParams bp{1, (segLen - hE) * (uint32_t)sizeof(T), 0, 0, 0};
            AscendC::DataCopyPad(y_gm[d + hE], local[rowStart + hE], bp);
          }
          que.FreeTensor(local);
          continue;
        }
        for (uint32_t c0 = 0; c0 < segLen; c0 += cbLen) {
          uint32_t cb = (segLen - c0) < cbLen ? (segLen - c0) : cbLen;
          uint64_t srcBase = (uint64_t)rs * segLen + c0;
          uint64_t dstBase = (uint64_t)rs * rowW + dstColOff + c0;
          AscendC::LocalTensor<T> local = que.AllocTensor<T>();
          LoadStrided(local, x_gm[srcBase], rowBatch, cb, segLen);
          que.EnQue(local);
          local = que.DeQue<T>();
          StoreStrided(y_gm[dstBase], local, rowBatch, cb, (uint32_t)rowW);
          que.FreeTensor(local);
        }
      }
    }
  }

 private:
  static constexpr uint32_t HD_ROWS = 128u;  // max rows per contiguous batch
  AscendC::TPipe* pipe;
  AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT,
                    BUFFER_NUM> que;
  AscendC::TBuf<AscendC::TPosition::VECCALC> hdBuf;
  AscendC::GlobalTensor<T> x_gm, y_gm;
  AscendC::ListTensorDesc inputList_;
  uint32_t numActive, outer, inner, sumW, tileLen, runStart, runEnd, rowsPerChunk;
  uint32_t origIdx[MAX_T], widths[MAX_T], offsets[MAX_T];
};

// Smallest offset >= minOff whose value mod 512 B is `residue` (bank-group placement of UB tables).
__aicore__ inline uint32_t PlaceMod512(uint32_t minOff, uint32_t residue) {
  uint32_t off = (minOff / 512u) * 512u + residue;
  return off >= minOff ? off : off + 512u;
}

// Key 13 table buffer (byte offsets): base at 0, the rows-row current-index table on a 512 B
// boundary, inc at +128 B mod 512 and the rows-row increment table at +256 B mod 512 (residues are 32B-aligned), so no Add of
// the per-band rebuild or of the group advance has two operands in one bank group.
__aicore__ inline uint32_t TallBandMr4CurOff(uint32_t pitch) {
  const uint32_t rowB = pitch * (uint32_t)sizeof(int32_t);
  return ((256u + rowB + 511u) / 512u) * 512u;  // base sits at 256 B
}
__aicore__ inline uint32_t TallBandMr4IncOff(uint32_t pitch, uint32_t rows) {
  const uint32_t rowB = pitch * (uint32_t)sizeof(int32_t);
  return PlaceMod512(TallBandMr4CurOff(pitch) + rows * rowB, 128u);
}
__aicore__ inline uint32_t TallBandMr4Inc4Off(uint32_t pitch, uint32_t rows) {
  const uint32_t rowB = pitch * (uint32_t)sizeof(int32_t);
  return PlaceMod512(TallBandMr4IncOff(pitch, rows) + rowB, 256u);
}

template <typename T, bool PRETABLE = false, bool MULTIROW = false>
class KernelConcatTallBand {
 public:
  __aicore__ inline KernelConcatTallBand() {}

  // Reads its fields straight out of GM, as keys 7/8/9/11 do, instead of copying
  // the 4180 B tiling struct and four numActive-long arrays on every core.  For
  // this key the host packs origIdx | width << 9 per active input, so a core reads
  // seven scalars, its plan and group words, the group's first column, and one
  // word per group member.  Member columns and source slots are the prefix sums
  // the host lays the group out with (AlignUp(rowsPerBand * w, 32 B) per member).
  __aicore__ inline void Init(AscendC::TPipe* pipeIn, GM_ADDR y,
                              __gm__ const ConcatTilingData* t,
                              AscendC::ListTensorDesc inputList,
                              uint32_t blockIdx) {
    pipe = pipeIn;
    inputList_ = inputList;
    y_gm.SetGlobalBuffer((__gm__ T*)y);
    const uint32_t numActive = t->numActive;
    outer = t->outer;
    sumW = t->sumW;
    rowsPerBand = t->gcX;
    const uint32_t numGroups = t->gcWg;
    maxSrcElems = t->gcStg;
    maxGroupW = t->gcChk;
    uint32_t plan = t->slotElem[numActive + numGroups + blockIdx];
    bandStart = (plan >> 6) & 0x1fffu;
    bandCount = plan >> 19;
    uint32_t meta = t->slotElem[numActive + (plan & 0x3fu)];
    k0 = meta & 0x1ffu;
    k1 = (meta >> 9) & 0x1ffu;
    const uint32_t e32 = 32u / (uint32_t)sizeof(T);
    colBase = 0;
    groupW = 0;
    if (k1 > k0) {
      colBase = t->offsets[k0];
      uint32_t slot = 0;
      for (uint32_t m = k0; m < k1; m++) {
        uint32_t packed = t->origIdx[m];
        uint32_t w = packed >> 9;
        memberW[m - k0] = w;
        memberCol[m - k0] = groupW;
        memberSlot[m - k0] = slot;
        groupBase[m - k0] = inputList_.GetDataPtr<T>(packed & 0x1ffu);
        groupW += w;
        slot += ((rowsPerBand * w + e32 - 1) / e32) * e32;
      }
    }
    pitch = ((maxGroupW + e32 - 1) / e32) * e32;
    uint32_t tableCap = ((maxGroupW + 63) / 64) * 64;
    // The host encodes the wide-read pipeline choice in rowsPerBand: it only
    // asks for 32-row source bands when the wide configuration fits UB.
    const bool c5WideReadPipeline = rowsPerBand == 32u;
    const uint32_t sourceBufferNum = c5WideReadPipeline ? 1u : 2u;
    const uint32_t outputRowsPerTile = c5WideReadPipeline ? (MULTIROW ? MR_ROWS : 8u) : 4u;
    const uint32_t outputBufferNum = c5WideReadPipeline ? 2u : 3u;
    pipe->InitBuffer(srcQueue, sourceBufferNum, maxSrcElems * sizeof(T));
    pipe->InitBuffer(tileQueue, outputBufferNum,
                     outputRowsPerTile * pitch * sizeof(T));
    uint32_t tableBytes = tableCap * sizeof(int32_t);
    uint32_t bankPad = (tableBytes & 511u) == 0u ? 256u : 0u;
    if constexpr (MULTIROW) {
      pipe->InitBuffer(tableBuf, TallBandMr4Inc4Off(pitch, MR_ROWS) +
                                     MR_ROWS * pitch * (uint32_t)sizeof(int32_t));
    } else {
      pipe->InitBuffer(tableBuf, 3u * tableBytes + 2u * bankPad);
    }
    if constexpr (PRETABLE) {
      // Small single-group plans (key 12) ship the finished Gather tables in the tiling arrays:
      // offsets[] holds the row-0 source byte offset of every output column, widths[] the
      // per-row increment.  Two small DMA reads replace the per-member table build.
      AscendC::GlobalTensor<int32_t> tblGm;
      AscendC::DataCopyExtParams tp{1, tableBytes, 0, 0, 0};
      AscendC::DataCopyPadExtParams<int32_t> tpad{false, 0, 0, 0};
      tblGm.SetGlobalBuffer((__gm__ int32_t*)(t->offsets));
      AscendC::DataCopyPad(tableBuf.GetWithOffset<int32_t>(tableCap, 0), tblGm[0], tp, tpad);
      tblGm.SetGlobalBuffer((__gm__ int32_t*)(t->widths));
      AscendC::DataCopyPad(
          tableBuf.GetWithOffset<int32_t>(tableCap, 2u * tableBytes + 2u * bankPad),
          tblGm[0], tp, tpad);
    }
  }

  // Key 13: one Gather per MR_ROWS output rows instead of a Gather and an Add per row.  cur4 holds
  // MR_ROWS rows of source byte offsets at the tile pitch (entries past groupW stay 0, a valid
  // index whose tile bytes are never stored); inc4 holds MR_ROWS * inc in every row block, so one
  // Add advances the whole group.  The tiles are two MR_ROWS-row buffers -- the bytes of key 10's
  // single 8-row tile -- which leaves room for the two tables beside the 32-row source band.
  static constexpr uint32_t MR_ROWS = 4u;
  __aicore__ inline void ProcessMultiRow() {
    if (bandCount == 0 || k0 >= k1) return;
    const uint32_t e32 = 32u / (uint32_t)sizeof(T);
    const int32_t elemBytes = (int32_t)sizeof(T);
    const uint32_t groupElems = MR_ROWS * pitch;
    AscendC::LocalTensor<int32_t> base = tableBuf.GetWithOffset<int32_t>(pitch, 256u);
    AscendC::LocalTensor<int32_t> cur4 =
        tableBuf.GetWithOffset<int32_t>(groupElems, TallBandMr4CurOff(pitch));
    AscendC::LocalTensor<int32_t> inc =
        tableBuf.GetWithOffset<int32_t>(pitch, TallBandMr4IncOff(pitch, MR_ROWS));
    AscendC::LocalTensor<int32_t> inc4 =
        tableBuf.GetWithOffset<int32_t>(groupElems, TallBandMr4Inc4Off(pitch, MR_ROWS));

    // Start the first tall read before building the tables.
    uint32_t firstRow = bandStart * rowsPerBand;
    uint32_t firstRows = (outer - firstRow) < rowsPerBand ? (outer - firstRow) : rowsPerBand;
    uint32_t phase = bandStart % (k1 - k0);
    AscendC::LocalTensor<T> firstSrc = srcQueue.AllocTensor<T>();
    LoadBand(firstSrc, firstRow, firstRows, phase);
    srcQueue.EnQue(firstSrc);

    const uint32_t alignElems = 32u / (uint32_t)sizeof(int32_t);
    for (uint32_t m = k0; m < k1; m++) {
      uint32_t w = memberW[m - k0];
      uint32_t localCol = memberCol[m - k0];
      uint32_t localSlot = memberSlot[m - k0];
      int32_t increment = (int32_t)w * elemBytes;
      uint32_t alignedBegin = (localCol + alignElems - 1u) & ~(alignElems - 1u);
      uint32_t alignedEnd = (localCol + w) & ~(alignElems - 1u);
      if (alignedEnd > alignedBegin) {
        for (uint32_t i = localCol; i < alignedBegin; i++) {
          base.SetValue(i, (int32_t)(localSlot + i - localCol) * elemBytes);
          inc.SetValue(i, increment);
        }
        AscendC::ArithProgression(
            base[alignedBegin], (int32_t)(localSlot + alignedBegin - localCol) * elemBytes,
            elemBytes, (int32_t)(alignedEnd - alignedBegin));
        AscendC::Duplicate(inc[alignedBegin], increment, (int32_t)(alignedEnd - alignedBegin));
        for (uint32_t i = alignedEnd; i < localCol + w; i++) {
          base.SetValue(i, (int32_t)(localSlot + i - localCol) * elemBytes);
          inc.SetValue(i, increment);
        }
      } else {
        for (uint32_t i = 0; i < w; i++) {
          base.SetValue(localCol + i, (int32_t)(localSlot + i) * elemBytes);
          inc.SetValue(localCol + i, increment);
        }
      }
    }

    AscendC::Duplicate(cur4, (int32_t)0, (int32_t)groupElems);
    AscendC::Duplicate(inc4, (int32_t)0, (int32_t)groupElems);
    AscendC::Adds(inc4, inc, (int32_t)0, groupW);
    for (uint32_t q = 1; q < MR_ROWS; q++) {
      AscendC::Add(inc4, inc4, inc, groupW);
    }
    for (uint32_t q = 1; q < MR_ROWS; q++) {
      AscendC::Adds(inc4[q * pitch], inc4, (int32_t)0, groupW);
    }

    uint32_t groupPitch = ((groupW + e32 - 1) / e32) * e32;
    uint32_t srcStrideBlocks = (pitch - groupPitch) / e32;
    for (uint32_t band = 0; band < bandCount; band++) {
      uint32_t row = (bandStart + band) * rowsPerBand;
      uint32_t rowCount = (outer - row) < rowsPerBand ? (outer - row) : rowsPerBand;
      AscendC::LocalTensor<T> src = srcQueue.DeQue<T>();
      AscendC::Adds(cur4, base, (int32_t)0, groupW);
      for (uint32_t q = 1; q < MR_ROWS; q++) {
        AscendC::Add(cur4[q * pitch], cur4[(q - 1u) * pitch], inc, groupW);
      }
      for (uint32_t outRow = 0; outRow < rowCount; outRow += MR_ROWS) {
        uint32_t outRows = (rowCount - outRow) < MR_ROWS ? (rowCount - outRow) : MR_ROWS;
        AscendC::LocalTensor<T> tile = tileQueue.AllocTensor<T>();
        AscendC::Gather(tile, src, cur4.template ReinterpretCast<uint32_t>(), (uint32_t)0,
                        outRows * pitch);
        if (outRow + outRows < rowCount) AscendC::Add(cur4, cur4, inc4, (int32_t)groupElems);
        tileQueue.EnQue(tile);
        tile = tileQueue.DeQue<T>();
        StoreStridedPitched(y_gm[(uint64_t)(row + outRow) * sumW + colBase], tile, outRows,
                            groupW, sumW, srcStrideBlocks);
        tileQueue.FreeTensor(tile);
      }
      srcQueue.FreeTensor(src);
      if (band + 1 < bandCount) {
        uint32_t nextRow = row + rowsPerBand;
        uint32_t nextRows = (outer - nextRow) < rowsPerBand ? (outer - nextRow) : rowsPerBand;
        AscendC::LocalTensor<T> nextSrc = srcQueue.AllocTensor<T>();
        LoadBand(nextSrc, nextRow, nextRows, phase);
        srcQueue.EnQue(nextSrc);
      }
    }
  }

  __aicore__ inline void Process() {
    if constexpr (MULTIROW) {
      ProcessMultiRow();
      return;
    }
    if (bandCount == 0 || k0 >= k1) return;
    const uint32_t e32 = 32u / (uint32_t)sizeof(T);
    const int32_t elemBytes = (int32_t)sizeof(T);
    uint32_t tableCap = ((maxGroupW + 63) / 64) * 64;
    uint32_t tableBytes = tableCap * sizeof(int32_t);
    uint32_t bankPad = (tableBytes & 511u) == 0u ? 256u : 0u;
    AscendC::LocalTensor<int32_t> base =
        tableBuf.GetWithOffset<int32_t>(tableCap, 0);
    AscendC::LocalTensor<int32_t> cur =
        tableBuf.GetWithOffset<int32_t>(tableCap, tableBytes + bankPad);
    AscendC::LocalTensor<int32_t> inc =
        tableBuf.GetWithOffset<int32_t>(tableCap, 2u * tableBytes + 2u * bankPad);

    // Start the first tall read before building the immutable Gather tables.
    // Its MTE2 latency is therefore hidden under the one-time scalar/vector setup.
    uint32_t firstRow = bandStart * rowsPerBand;
    uint32_t firstRows = (outer - firstRow) < rowsPerBand ? (outer - firstRow) : rowsPerBand;
    uint32_t phase = (k1 > k0) ? bandStart % (k1 - k0) : 0;
    AscendC::LocalTensor<T> firstSrc = srcQueue.AllocTensor<T>();
    LoadBand(firstSrc, firstRow, firstRows, phase);
    srcQueue.EnQue(firstSrc);

    if constexpr (!PRETABLE) {
    const uint32_t alignElems = 32u / (uint32_t)sizeof(int32_t);
    for (uint32_t m = k0; m < k1; m++) {
      uint32_t w = memberW[m - k0];
      uint32_t localCol = memberCol[m - k0];
      uint32_t localSlot = memberSlot[m - k0];
      int32_t increment = (int32_t)w * elemBytes;
      uint32_t alignedBegin =
          (localCol + alignElems - 1u) & ~(alignElems - 1u);
      uint32_t alignedEnd = (localCol + w) & ~(alignElems - 1u);
      if (alignedEnd > alignedBegin) {
        for (uint32_t i = localCol; i < alignedBegin; i++) {
          base.SetValue(i, (int32_t)(localSlot + i - localCol) * elemBytes);
          inc.SetValue(i, increment);
        }
        AscendC::ArithProgression(
            base[alignedBegin],
            (int32_t)(localSlot + alignedBegin - localCol) * elemBytes,
            elemBytes, (int32_t)(alignedEnd - alignedBegin));
        AscendC::Duplicate(inc[alignedBegin], increment,
                           (int32_t)(alignedEnd - alignedBegin));
        for (uint32_t i = alignedEnd; i < localCol + w; i++) {
          base.SetValue(i, (int32_t)(localSlot + i - localCol) * elemBytes);
          inc.SetValue(i, increment);
        }
      } else {
        for (uint32_t i = 0; i < w; i++) {
          base.SetValue(localCol + i, (int32_t)(localSlot + i) * elemBytes);
          inc.SetValue(localCol + i, increment);
        }
      }
    }
    }

    // The host encodes the wide-read pipeline choice in rowsPerBand: it only
    // asks for 32-row source bands when the wide configuration fits UB.
    const bool c5WideReadPipeline = rowsPerBand == 32u;
    uint32_t groupPitch = ((groupW + e32 - 1) / e32) * e32;
    uint32_t srcStrideBlocks = (pitch - groupPitch) / e32;
    for (uint32_t band = 0; band < bandCount; band++) {
      uint32_t row = (bandStart + band) * rowsPerBand;
      uint32_t rowCount = (outer - row) < rowsPerBand ? (outer - row) : rowsPerBand;
      if (!c5WideReadPipeline && band + 1 < bandCount) {
        uint32_t nextRow = row + rowsPerBand;
        uint32_t nextRows =
            (outer - nextRow) < rowsPerBand ? (outer - nextRow) : rowsPerBand;
        AscendC::LocalTensor<T> nextSrc = srcQueue.AllocTensor<T>();
        LoadBand(nextSrc, nextRow, nextRows, phase);
        srcQueue.EnQue(nextSrc);
      }
      AscendC::LocalTensor<T> src = srcQueue.DeQue<T>();
      const uint32_t outputRowsPerTile = c5WideReadPipeline ? 8u : 4u;
      AscendC::Adds(cur, base, (int32_t)0, groupW);
      for (uint32_t outRow = 0; outRow < rowCount; outRow += outputRowsPerTile) {
        uint32_t outRows = (rowCount - outRow) < outputRowsPerTile
                               ? (rowCount - outRow)
                               : outputRowsPerTile;
        AscendC::LocalTensor<T> tile = tileQueue.AllocTensor<T>();
        for (uint32_t r = 0; r < outRows; r++) {
          AscendC::Gather(tile[r * pitch], src,
                          cur.template ReinterpretCast<uint32_t>(),
                          (uint32_t)0, groupW);
          if (outRow + r + 1 < rowCount) AscendC::Add(cur, cur, inc, groupW);
        }
        tileQueue.EnQue(tile);
        tile = tileQueue.DeQue<T>();
        StoreStridedPitched(
            y_gm[(uint64_t)(row + outRow) * sumW + colBase], tile,
            outRows, groupW, sumW, srcStrideBlocks);
        tileQueue.FreeTensor(tile);
      }
      srcQueue.FreeTensor(src);
      if (c5WideReadPipeline && band + 1 < bandCount) {
        uint32_t nextRow = row + rowsPerBand;
        uint32_t nextRows =
            (outer - nextRow) < rowsPerBand ? (outer - nextRow) : rowsPerBand;
        AscendC::LocalTensor<T> nextSrc = srcQueue.AllocTensor<T>();
        LoadBand(nextSrc, nextRow, nextRows, phase);
        srcQueue.EnQue(nextSrc);
      }
    }
  }

 private:
  __aicore__ inline void LoadOne(const AscendC::LocalTensor<T>& dst,
                                 uint32_t member, uint32_t row,
                                 uint32_t rowCount) {
    uint32_t w = memberW[member - k0];
    uint32_t slot = memberSlot[member - k0];
    x_gm.SetGlobalBuffer(groupBase[member - k0]);
    if constexpr (!PRETABLE) {
      // Small plans read a few KB per band; normal L2 mode is kept for them.
      x_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    uint64_t srcByteOffset = (uint64_t)row * w * sizeof(T);
    uint32_t copyBytes = rowCount * w * sizeof(T);
    if (rowCount == rowsPerBand && ((srcByteOffset | copyBytes) & 31u) == 0u) {
      AscendC::DataCopy(dst[slot], x_gm[(uint64_t)row * w], rowCount * w);
    } else {
      PadIn(dst[slot], x_gm[(uint64_t)row * w], rowCount * w);
    }
  }

  __aicore__ inline void LoadBand(const AscendC::LocalTensor<T>& dst,
                                  uint32_t row, uint32_t rowCount,
                                  uint32_t phase) {
    uint32_t members = k1 - k0;
    uint32_t member = k0 + phase;
    for (uint32_t i = 0; i < members; i++) {
      LoadOne(dst, member, row, rowCount);
      member++;
      if (member == k1) member = k0;
    }
  }

  AscendC::TPipe* pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> srcQueue;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> tileQueue;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> tableBuf;
  AscendC::GlobalTensor<T> x_gm, y_gm;
  AscendC::ListTensorDesc inputList_;
  uint32_t outer, sumW, rowsPerBand;
  uint32_t maxSrcElems, maxGroupW, pitch;
  uint32_t bandStart, bandCount, k0, k1, colBase, groupW;
  uint32_t memberW[32], memberCol[32], memberSlot[32];
  __gm__ T* groupBase[32];
};

// Key 12: one small Gather group whose row-0 byte offsets (offsets[]) and per-row increments
// (widths[]) the host ships in the tiling arrays.  Per band a core reads each member once, builds
// the band's whole index table (row r = row r-1 + inc) and issues ONE Gather and ONE strided store
// for all rows of the band, instead of a Gather and an Add per row and a store per 4-row tile.
// Table rows sit on a pitch of 64 mod 128 elements (256 mod 512 B), inc at +128 and bandInc at
// +384 B mod 512: every Add's operands then fall in different bank groups.
template <typename T>
class KernelConcatSmallShot {
 public:
  __aicore__ inline KernelConcatSmallShot() {}

  __aicore__ inline void Init(AscendC::TPipe* pipeIn, GM_ADDR y,
                              __gm__ const ConcatTilingData* t,
                              AscendC::ListTensorDesc inputList,
                              uint32_t blockIdx) {
    pipe = pipeIn;
    inputList_ = inputList;
    y_gm.SetGlobalBuffer((__gm__ T*)y);
    const uint32_t numActive = t->numActive;
    outer = t->outer;
    sumW = t->sumW;
    rowsPerBand = t->gcX;
    const uint32_t numGroups = t->gcWg;
    const uint32_t maxSrcElems = t->gcStg;
    const uint32_t maxGroupW = t->gcChk;
    uint32_t plan = t->slotElem[numActive + numGroups + blockIdx];
    bandStart = (plan >> 6) & 0x1fffu;
    bandCount = plan >> 19;
    uint32_t meta = t->slotElem[numActive + (plan & 0x3fu)];
    k0 = meta & 0x1ffu;
    k1 = (meta >> 9) & 0x1ffu;
    const uint32_t e32 = 32u / (uint32_t)sizeof(T);
    groupW = 0;
    if (k1 > k0) {
      uint32_t slot = 0;
      for (uint32_t m = k0; m < k1; m++) {
        uint32_t packed = t->origIdx[m];
        uint32_t w = packed >> 9;
        memberW[m - k0] = w;
        memberSlot[m - k0] = slot;
        groupBase[m - k0] = inputList_.GetDataPtr<T>(packed & 0x1ffu);
        groupW += w;
        slot += ((rowsPerBand * w + e32 - 1) / e32) * e32;
      }
    }
    // The host gate keeps the single group at most MAX_T columns, so tableCap entries of the
    // tiling arrays are valid; rowPitch >= tableCap and differs from it by 0 or 64.
    rowPitch = ((maxGroupW + 63u) / 128u) * 128u + 64u;
    tableCap = ((maxGroupW + 63u) / 64u) * 64u;
    const uint32_t rowBytes = rowPitch * sizeof(int32_t);
    // Table buffer: inc copy (128 mod 512 B), bandInc (384), index table rows (0 / 256).
    tableOff[0] = Place(0u, 128u);
    tableOff[1] = Place(tableOff[0] + rowBytes, 384u);
    tableOff[2] = Place(tableOff[1] + rowBytes, 0u);
    tilingGm = t;
    pipe->InitBuffer(srcQueue, bandCount > 1u ? 2u : 1u, maxSrcElems * sizeof(T));
    pipe->InitBuffer(tileQueue, 1u, rowsPerBand * rowPitch * sizeof(T));
    // base | inc as shipped in the tiling arrays; they reach the vector pipe through a queue
    // hand-off (a DMA write into a plain TBuf read by the vector pipe is not synchronised).
    pipe->InitBuffer(shipQueue, 1u, 2u * rowBytes);
    pipe->InitBuffer(tableBuf, tableOff[2] + rowsPerBand * rowBytes);
  }

  __aicore__ inline void Process() {
    if (bandCount == 0 || k0 >= k1) return;
    const uint32_t e32 = 32u / (uint32_t)sizeof(T);
    AscendC::LocalTensor<int32_t> inc = tableBuf.GetWithOffset<int32_t>(rowPitch, tableOff[0]);
    AscendC::LocalTensor<int32_t> table =
        tableBuf.GetWithOffset<int32_t>(rowsPerBand * rowPitch, tableOff[2]);

    // Ship the two tables, then start the first band's read; the index table is built from the
    // dequeued tables while that read runs.
    AscendC::LocalTensor<int32_t> shipped = shipQueue.AllocTensor<int32_t>();
    AscendC::GlobalTensor<int32_t> tblGm;
    AscendC::DataCopyExtParams tp{1, tableCap * (uint32_t)sizeof(int32_t), 0, 0, 0};
    AscendC::DataCopyPadExtParams<int32_t> tpad{false, 0, 0, 0};
    tblGm.SetGlobalBuffer((__gm__ int32_t*)(tilingGm->offsets));
    AscendC::DataCopyPad(shipped, tblGm[0], tp, tpad);
    tblGm.SetGlobalBuffer((__gm__ int32_t*)(tilingGm->widths));
    AscendC::DataCopyPad(shipped[rowPitch], tblGm[0], tp, tpad);
    shipQueue.EnQue(shipped);

    uint32_t phase = bandStart % (k1 - k0);
    uint32_t row = bandStart * rowsPerBand;
    uint32_t rows = (outer - row) < rowsPerBand ? (outer - row) : rowsPerBand;
    AscendC::LocalTensor<T> firstSrc = srcQueue.AllocTensor<T>();
    LoadBand(firstSrc, row, rows, phase);
    srcQueue.EnQue(firstSrc);

    shipped = shipQueue.DeQue<int32_t>();
    if (rowPitch > tableCap) {
      AscendC::Duplicate(shipped[tableCap], (int32_t)0, (int32_t)(rowPitch - tableCap));
      AscendC::Duplicate(shipped[rowPitch + tableCap], (int32_t)0, (int32_t)(rowPitch - tableCap));
    }
    AscendC::Adds(table, shipped, (int32_t)0, (int32_t)rowPitch);
    AscendC::Adds(inc, shipped[rowPitch], (int32_t)0, (int32_t)rowPitch);
    shipQueue.FreeTensor(shipped);
    // Every band's source buffer restarts its rows at the member slots, so one table serves
    // all bands of the core.
    for (uint32_t r = 1; r < rowsPerBand; r++) {
      AscendC::Add(table[r * rowPitch], table[(r - 1u) * rowPitch], inc, (int32_t)rowPitch);
    }

    const uint32_t groupPitch = ((groupW + e32 - 1u) / e32) * e32;
    const uint32_t srcStrideBlocks = (rowPitch - groupPitch) / e32;
    for (uint32_t band = 0; band < bandCount; band++) {
      row = (bandStart + band) * rowsPerBand;
      if (row >= outer) break;
      rows = (outer - row) < rowsPerBand ? (outer - row) : rowsPerBand;
      AscendC::LocalTensor<T> src = srcQueue.DeQue<T>();
      AscendC::LocalTensor<T> tile = tileQueue.AllocTensor<T>();
      AscendC::Gather(tile, src, table.template ReinterpretCast<uint32_t>(), (uint32_t)0,
                      rows * rowPitch);
      tileQueue.EnQue(tile);
      tile = tileQueue.DeQue<T>();
      StoreStridedPitched(y_gm[(uint64_t)row * sumW], tile, rows, groupW, sumW, srcStrideBlocks);
      tileQueue.FreeTensor(tile);
      srcQueue.FreeTensor(src);
      if (band + 1u < bandCount && row + rowsPerBand < outer) {
        uint32_t nextRow = row + rowsPerBand;
        uint32_t nextRows = (outer - nextRow) < rowsPerBand ? (outer - nextRow) : rowsPerBand;
        AscendC::LocalTensor<T> nextSrc = srcQueue.AllocTensor<T>();
        LoadBand(nextSrc, nextRow, nextRows, phase);
        srcQueue.EnQue(nextSrc);
      }
    }
  }

 private:
  // Smallest offset >= minOff whose value mod 512 B is `residue`.
  __aicore__ inline uint32_t Place(uint32_t minOff, uint32_t residue) {
    uint32_t off = (minOff / 512u) * 512u + residue;
    return off >= minOff ? off : off + 512u;
  }

  __aicore__ inline void LoadBand(const AscendC::LocalTensor<T>& dst, uint32_t row,
                                  uint32_t rowCount, uint32_t phase) {
    uint32_t members = k1 - k0;
    uint32_t member = k0 + phase;
    for (uint32_t i = 0; i < members; i++) {
      uint32_t w = memberW[member - k0];
      uint32_t slot = memberSlot[member - k0];
      x_gm.SetGlobalBuffer(groupBase[member - k0]);
      uint64_t srcByteOffset = (uint64_t)row * w * sizeof(T);
      uint32_t copyBytes = rowCount * w * sizeof(T);
      if (rowCount == rowsPerBand && ((srcByteOffset | copyBytes) & 31u) == 0u) {
        AscendC::DataCopy(dst[slot], x_gm[(uint64_t)row * w], rowCount * w);
      } else {
        PadIn(dst[slot], x_gm[(uint64_t)row * w], rowCount * w);
      }
      member++;
      if (member == k1) member = k0;
    }
  }

  AscendC::TPipe* pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> srcQueue;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> tileQueue;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> shipQueue;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> tableBuf;
  AscendC::GlobalTensor<T> x_gm, y_gm;
  AscendC::ListTensorDesc inputList_;
  __gm__ const ConcatTilingData* tilingGm;
  uint32_t outer, sumW, rowsPerBand, rowPitch, tableCap;
  uint32_t bandStart, bandCount, k0, k1, groupW;
  uint32_t tableOff[3];
  uint32_t memberW[32], memberSlot[32];
  __gm__ T* groupBase[32];
};

template <typename T>
class KernelConcatRoute6 {
 public:
  __aicore__ inline KernelConcatRoute6() {}
  __aicore__ inline void Init(GM_ADDR y, const ConcatTilingData& t,
                              AscendC::ListTensorDesc inputList) {
    inputList_ = inputList;
    y_gm.SetGlobalBuffer((__gm__ T*)y);
    numActive = t.numActive;
    outer = t.outer;
    sumW = t.sumW;
    totalRuns = t.totalRuns;
    rowsPerChunk = t.rowsPerChunk;
    gcX = t.gcX;
    gcWg = t.gcWg;
    gcStg = t.gcStg;
    gcChk = t.gcChk;
    gcTbl = t.gcTbl;
    for (uint32_t i = 0; i < numActive; i++) {
      origIdx[i] = t.origIdx[i];
      widths[i] = t.widths[i];
      offsets[i] = t.offsets[i];
      slotElem[i] = t.slotElem[i];
    }
    metaMode = (gcTbl >> 30) & 1u;
    if (metaMode != 0u) {
      for (uint32_t i = 0; i < gcWg; i++) {
        groupMeta[i] = t.slotElem[numActive + i];
      }
    }
    uint32_t bi = AscendC::GetBlockIdx();
    runStart = bi * t.runsPerCore;
    runEnd = runStart + t.runsPerCore;
    if (runEnd > totalRuns) runEnd = totalRuns;

    const uint32_t e32 = 32u / (uint32_t)sizeof(T);
    uint32_t pitch = ((gcChk + e32 - 1) / e32) * e32;
    uint32_t tblCap = ((gcChk + 63) / 64) * 64;
    pipe.InitBuffer(gcSrcQ, 2, gcStg * sizeof(T));
    pipe.InitBuffer(gcTileQ, 2, gcX * pitch * sizeof(T));
    uint32_t tblBytes = tblCap * sizeof(int32_t);
    constexpr uint32_t bankPad = 256u;
    pipe.InitBuffer(gcTableBuf, 3u * tblBytes + 2u * bankPad);
  }

  __aicore__ inline void Process() {
    ProcessGroupedGather();
  }

  // Route 6: grouped-Gather 4KB-burst store (inner==1, non-aligned, big fp/int32 -- the large-stream workload
  // class). Adjacent segments are grouped on host into ~4KB-wide column-groups. A work unit
  // u = (group g, row-chunk rc). For that group's rows [r0,r1) in bands of gR6:
  //   READ  : one contiguous PadIn per member (full BW, blockCount=1) into a packed src UB.
  //   ASSEMBLE: Gather each output row of the group into a [gR6,pitch] tile in UB (vector
  //             handles the non-aligned column scatter, using a per-column byte-offset table).
  //   WRITE : each assembled row as one contiguous ~groupW*sizeof(T) PadOut at colBase.
  // The ~4KB contiguous write fills HBM lines (vs route 5's 256B-fragmented strided store).
  // Pad columns [Wg,pitch) are never gathered nor stored -> bit-exact. colBase is byte-
  // granular in GM (no alignment needed, same as route 5's non-32B dstStride).
  __aicore__ inline void ProcessGroupedGather() {
    if (numActive == 0) return;
    const uint32_t e32 = 32u / (uint32_t)sizeof(T);
    const int32_t sz = (int32_t)sizeof(T);
    uint32_t numGroups = gcWg;
    uint32_t Rb0 = gcX;            // gR6
    uint32_t pitch = ((gcChk + e32 - 1) / e32) * e32;   // AlignUp(maxGroupW, e32)
    uint32_t WG = gcTbl & 0x3fffffffu;      // grouping threshold (replay fallback)
    uint32_t balMode = gcTbl >> 31;         // 0 = legacy "acc>=WG"; 1 = capacity-greedy "acc+w<=WG"
    uint32_t tblCap = ((gcChk + 63) / 64) * 64;
    uint32_t tblBytes = tblCap * sizeof(int32_t);
    constexpr uint32_t bankPad = 256u;
    AscendC::LocalTensor<int32_t> base = gcTableBuf.GetWithOffset<int32_t>(tblCap, 0);
    AscendC::LocalTensor<int32_t> cur =
        gcTableBuf.GetWithOffset<int32_t>(tblCap, tblBytes + bankPad);
    AscendC::LocalTensor<int32_t> inc =
        gcTableBuf.GetWithOffset<int32_t>(tblCap, 2u * tblBytes + 2u * bankPad);
    uint32_t numRowChunks = numGroups > 0 ? totalRuns / numGroups : 0;
    uint32_t u = runStart;
    while (u < runEnd) {
      uint32_t g = u / numRowChunks;
      // In group-major order, consecutive units belong to the same column group.
      // Keep that run together so descriptor pointers and Gather tables are built once.
      uint32_t unitEnd = (g + 1) * numRowChunks;
      if (unitEnd > runEnd) unitEnd = runEnd;
      // Locate group g by replaying the WG-accumulation scan over widths[].
      uint32_t k0 = 0, nk = 0, Wg = 0, colBase = 0;
      if (metaMode != 0u) {
        uint32_t meta = groupMeta[g];
        k0 = meta & 0x1ffu;
        uint32_t k1 = (meta >> 9) & 0x1ffu;
        nk = k1 - k0;
        colBase = offsets[k0];
        uint32_t colEnd = (k1 < numActive) ? offsets[k1] : sumW;
        Wg = colEnd - colBase;
      } else {
        uint32_t curG = 0, acc = 0, gStart = 0;
        for (uint32_t k = 0; k < numActive; k++) {
          if (balMode == 1u && acc > 0 && acc + widths[k] > WG) {   // close [gStart, k-1]
            if (curG == g) { k0 = gStart; nk = k - gStart; Wg = acc; colBase = offsets[gStart]; break; }
            curG++; gStart = k; acc = 0;
          }
          acc += widths[k];
          bool close = (balMode == 0u) ? (acc >= WG || k == numActive - 1) : (k == numActive - 1);
          if (close) {
            if (curG == g) { k0 = gStart; nk = k - gStart + 1; Wg = acc; colBase = offsets[gStart]; break; }
            curG++; gStart = k + 1; acc = 0;
          }
        }
      }
      // Hoist per-member GM base pointers out of the unit/band loops. GetDataPtr(index) is a
      // GM descriptor load (*(dataPtr_+index)); calling it per band (~nk*bands/core) put
      // its scalar GM-load latency on the MTE2 critical path -- each contiguous read
      // stalled waiting for the next member's base pointer to resolve, bubbling MTE2
      // (measured large-stream workload: aiv 222us vs mte2-active 154us -> ~68us exposed). Resolve once per
      // group run so the band loop issues its reads back-to-back and MTE2 stays saturated.
      for (uint32_t m = k0; m < k0 + nk; m++) {
        groupBase[m - k0] = inputList_.GetDataPtr<T>(origIdx[m]);
      }
      uint32_t groupPitch = ((Wg + e32 - 1) / e32) * e32;
      uint32_t srcStrideBlocks = (pitch - groupPitch) / e32;
      uint32_t groupSlotBase = slotElem[k0];
      bool tableReady = false;
      for (; u < unitEnd; u++) {
        uint32_t rc = u % numRowChunks;
        uint32_t r0 = rc * rowsPerChunk;
        uint32_t r1 = r0 + rowsPerChunk;
        if (r1 > outer) r1 = outer;
        if (r0 >= r1) continue;
        // (b) row bands -- software-pipelined 3-stage chain (READ mte2 -> GATHER vec ->
        // WRITE mte3). One-ahead source prefetch keeps MTE2 active during Gather/store.
        uint32_t nb = (r1 - r0 + Rb0 - 1) / Rb0;
        // Rotate member issue order by row chunk to spread concurrent MTE2 requests.
        uint32_t phase = nk > 0 ? rc % nk : 0;
        // Prologue: issue band 0 first. For the first unit of this group, its MTE2
        // transfer overlaps the one-time offset-table build below.
        if (nb > 0) {
          uint32_t Rb = (r1 - r0) < Rb0 ? (r1 - r0) : Rb0;
          AscendC::LocalTensor<T> src = gcSrcQ.AllocTensor<T>();
          LoadBand(src, k0, nk, phase, groupSlotBase, r0, Rb, Rb0);
          gcSrcQ.EnQue(src);
        }
        if (!tableReady) {
      // (a) build per-column byte-offset base/inc tables (once per persistent group run).
      //   member m, local col i -> packed src slot (slotE+i); row stride = w_m elems.
      //     base[lc0+i] = (slotE+i)*sz  -- arithmetic progression, first (slotE+hd)*sz, step sz
      //     inc [lc0+i] = w_m*sz        -- constant
      // VECTORIZED: the 32B-aligned interior [a,e) of each member is filled by ONE
      // ArithProgression + ONE Duplicate; only the <=7 head elems [lc0,a) and <=7 tail
      // elems [e,lc0+w) still need scalar SetValue. Interior and edge writes are on
      // DISJOINT 32B blocks by construction, so there is no cross-pipe WAW on any word.
      // large-stream workload: 2*Wg ~= 2274 scalar UB stores/unit -> 2*nk = 34 vector ops + <=476 edge stores.
      // Placed AFTER the band-0 read is issued so this build (and the PIPE_ALL that ends
      // it) runs under band 0's MTE2 read, which the band loop's DeQue waits for anyway.
          const uint32_t ALN = 32u / (uint32_t)sizeof(int32_t);   // 8 int32 = 32B
          for (uint32_t m = k0; m < k0 + nk; m++) {
            uint32_t w = widths[m];
            uint32_t lc0 = offsets[m] - colBase;
            uint32_t slotE = slotElem[m] - groupSlotBase;
            int32_t so = (int32_t)slotE;
            int32_t incv = (int32_t)w * sz;
            uint32_t a = (lc0 + ALN - 1u) & ~(ALN - 1u);
            uint32_t e = (lc0 + w) & ~(ALN - 1u);
            if (e > a) {
              for (uint32_t i = lc0; i < a; i++) {
                base.SetValue(i, (so + (int32_t)(i - lc0)) * sz);
                inc.SetValue(i, incv);
              }
              AscendC::ArithProgression(base[a],
                                        (int32_t)((so + (int32_t)(a - lc0)) * sz), sz,
                                        (int32_t)(e - a));
              AscendC::Duplicate(inc[a], incv, (int32_t)(e - a));
              for (uint32_t i = e; i < lc0 + w; i++) {
                base.SetValue(i, (so + (int32_t)(i - lc0)) * sz);
                inc.SetValue(i, incv);
              }
            } else {
              for (uint32_t i = 0; i < w; i++) {
                base.SetValue(lc0 + i, (so + (int32_t)i) * sz);
                inc.SetValue(lc0 + i, incv);
              }
            }
          }
          AscendC::PipeBarrier<PIPE_ALL>();
          tableReady = true;
        }
        for (uint32_t bi = 0; bi < nb; bi++) {
        uint32_t rs = r0 + bi * Rb0;
        uint32_t Rb = (r1 - rs) < Rb0 ? (r1 - rs) : Rb0;
        // With two source slots, issue band bi+1 before waiting for band bi. This
        // keeps its MTE2 commands queued behind the current transfer and removes
        // the per-band DeQue-return -> scalar-issue bubble.
        if (bi + 1 < nb) {
          uint32_t rsN = rs + Rb0;
          uint32_t RbN = (r1 - rsN) < Rb0 ? (r1 - rsN) : Rb0;
          AscendC::LocalTensor<T> srcN = gcSrcQ.AllocTensor<T>();
          LoadBand(srcN, k0, nk, phase, groupSlotBase, rsN, RbN, Rb0);
          gcSrcQ.EnQue(srcN);
        }
        AscendC::LocalTensor<T> src = gcSrcQ.DeQue<T>();        // band bi's read (ready)
        AscendC::LocalTensor<T> tile = gcTileQ.AllocTensor<T>();
        AscendC::Adds(cur, base, (int32_t)0, Wg);   // cur = row-0 byte offsets
        for (uint32_t j = 0; j < Rb; j++) {          // ASSEMBLE
          AscendC::Gather(tile[j * pitch], src, cur.template ReinterpretCast<uint32_t>(),
                          (uint32_t)0, Wg);
          if (j + 1 < Rb) AscendC::Add(cur, cur, inc, Wg);
        }
        gcSrcQ.FreeTensor(src);
        gcTileQ.EnQue(tile);
        tile = gcTileQ.DeQue<T>();
        StoreStridedPitched(y_gm[(uint64_t)rs * sumW + colBase], tile, Rb, Wg, sumW,
                            srcStrideBlocks);
        gcTileQ.FreeTensor(tile);
        }
      }
    }
  }

  __aicore__ inline void LoadOne(const AscendC::LocalTensor<T>& dst, uint32_t k0,
                                 uint32_t m, uint32_t groupSlotBase,
                                 uint32_t row, uint32_t rowCount, uint32_t fullRows) {
    uint32_t w = widths[m];
    uint32_t slot = slotElem[m] - groupSlotBase;
    x_gm.SetGlobalBuffer(groupBase[m - k0]);
    uint64_t srcByteOffset = (uint64_t)row * w * sizeof(T);
    uint32_t copyBytes = rowCount * w * sizeof(T);
    if (rowCount == fullRows && ((srcByteOffset | copyBytes) & 31u) == 0u) {
      AscendC::DataCopy(dst[slot], x_gm[(uint64_t)row * w], rowCount * w);
    } else {
      PadIn(dst[slot], x_gm[(uint64_t)row * w], rowCount * w);
    }
  }

  __aicore__ inline void LoadBand(const AscendC::LocalTensor<T>& dst, uint32_t k0,
                                  uint32_t nk, uint32_t phase,
                                  uint32_t groupSlotBase, uint32_t row,
                                  uint32_t rowCount, uint32_t fullRows) {
    uint32_t m = k0 + phase;
    uint32_t end = k0 + nk;
    for (uint32_t q = 0; q < nk; q++) {
      LoadOne(dst, k0, m, groupSlotBase, row, rowCount, fullRows);
      if (++m == end) m = k0;
    }
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> gcSrcQ;    // route 6: packed contiguous reads
  AscendC::TQue<AscendC::QuePosition::VECOUT, 2> gcTileQ; // route 6: gather output -> store
  AscendC::TBuf<AscendC::QuePosition::VECCALC> gcTableBuf;  // base/cur/inc with 256B bank phase padding
  AscendC::GlobalTensor<T> x_gm, y_gm;
  AscendC::ListTensorDesc inputList_;
  uint32_t numActive, outer, sumW, totalRuns, runStart, runEnd, rowsPerChunk;
  uint32_t gcX, gcWg, gcStg, gcChk, gcTbl, metaMode;
  uint32_t origIdx[MAX_T], widths[MAX_T], offsets[MAX_T], slotElem[MAX_T];
  uint32_t groupMeta[64];
  __gm__ T* groupBase[MAX_T];
};

extern "C" __global__ __aicore__ void concat(GM_ADDR x, GM_ADDR y,
                                             GM_ADDR workspace,
                                             GM_ADDR tiling) {
  // The three predecoded keys need six scalars and this core's three or four
  // slots -- 88 bytes.  GET_TILING_DATA copies all 4180 bytes of the struct into
  // local memory on every core, which measures 5.5us of a 10.1us case-1 launch;
  // reading the handful of fields straight out of GM costs 0.6us instead.
  // TILING_KEY_IS compiles one binary per key, so returning before the macro
  // keeps the copy out of these three binaries entirely.
  if (TILING_KEY_IS(9)) {
    AscendC::TPipe pipe;
    KernelConcatPredecoded4<DTYPE_X> op;
    op.Init(&pipe, x, y, (__gm__ const ConcatTilingData*)tiling,
            AscendC::GetBlockIdx());
    op.Process();
    return;
  }
  if (TILING_KEY_IS(8)) {
    AscendC::TPipe pipe;
    KernelConcatPredecoded3<DTYPE_X> op;
    op.Init(&pipe, x, y, (__gm__ const ConcatTilingData*)tiling,
            AscendC::GetBlockIdx());
    op.Process();
    return;
  }
  if (TILING_KEY_IS(7)) {
    AscendC::TPipe pipe;
    KernelConcatDim0Direct<DTYPE_X> op;
    op.Init(&pipe, x, y, (__gm__ const ConcatTilingData*)tiling,
            AscendC::GetBlockIdx());
    op.Process();
    return;
  }
  if (TILING_KEY_IS(11)) {
    AscendC::ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(x));
    AscendC::TPipe pipe;
    KernelConcatRoute4<DTYPE_X, true> op;
    op.Init(&pipe, y, (__gm__ const ConcatTilingData*)tiling, inputList);
    op.Process();
    return;
  }
  if (TILING_KEY_IS(12)) {
    AscendC::ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(x));
    AscendC::TPipe pipe;
    KernelConcatSmallShot<DTYPE_X> op;
    op.Init(&pipe, y, (__gm__ const ConcatTilingData*)tiling, inputList,
            AscendC::GetBlockIdx());
    op.Process();
    return;
  }
  if (TILING_KEY_IS(13)) {
    AscendC::ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(x));
    AscendC::TPipe pipe;
    KernelConcatTallBand<DTYPE_X, false, true> op;
    op.Init(&pipe, y, (__gm__ const ConcatTilingData*)tiling, inputList,
            AscendC::GetBlockIdx());
    op.Process();
    return;
  }
  if (TILING_KEY_IS(10)) {
    AscendC::ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(x));
    AscendC::TPipe pipe;
    KernelConcatTallBand<DTYPE_X> op;
    op.Init(&pipe, y, (__gm__ const ConcatTilingData*)tiling, inputList,
            AscendC::GetBlockIdx());
    op.Process();
    return;
  }
  GET_TILING_DATA(tiling_data, tiling);
  if (TILING_KEY_IS(1)) {
    AscendC::ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(x));
    AscendC::TPipe pipe;
    KernelConcatRoute1<DTYPE_X> op;
    op.Init(&pipe, y, tiling_data, inputList);
    op.Process();
  } else if (TILING_KEY_IS(4)) {
    AscendC::ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(x));
    AscendC::TPipe pipe;
    KernelConcatRoute4<DTYPE_X> op;
    op.Init(&pipe, y, &tiling_data, inputList);
    op.Process();
  } else if (TILING_KEY_IS(0)) {
    AscendC::ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(x));
    KernelConcat<DTYPE_X> op;
    op.Init(y, tiling_data, inputList);
    op.Process();
  } else if (TILING_KEY_IS(6)) {
    AscendC::ListTensorDesc inputList(reinterpret_cast<__gm__ void*>(x));
    KernelConcatRoute6<DTYPE_X> op;
    op.Init(y, tiling_data, inputList);
    op.Process();
  }
}
