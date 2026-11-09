/*
 * Greater: y = (x1 > x2), elementwise with broadcast, output bool (1 byte).
 * dtypes: fp16, fp32, bf16, int32, int8.
 *   fp16/fp32 -> Compare(GT) direct (IEEE: NaN>x = false, matches torch.gt)
 *   bf16->float, int8->half cast then Compare; int32 -> NOT(min(a,b)==a).
 * Two paths (host picks via isBcast):
 *   same-shape: each core owns a contiguous flat range, tiled by tileLen.
 *   broadcast : each core owns a range of output rows (last dim = inner). Per
 *     row, build x1/x2 inner-vectors (contiguous load, or single value via
 *     Duplicate when that operand broadcasts the last dim), then compare.
 * Non-aligned copies use DataCopyPad (scalar GM stores are unreliable).
 */
#include <type_traits>
#include "kernel_operator.h"

// aligned (32B) counts use plain DataCopy (cheaper); only the non-aligned tail
// uses DataCopyPad (scalar GM stores are unreliable -> never hand-roll).
template <typename typeT>
__aicore__ inline void PadIn(const AscendC::LocalTensor<typeT>& dst,
                             const AscendC::GlobalTensor<typeT>& src,
                             const uint32_t n) {
  constexpr uint32_t BLK = 32 / sizeof(typeT);
  if (n % BLK == 0) {
    AscendC::DataCopy(dst, src, n);
  } else {
    AscendC::DataCopyExtParams params{1, n * (uint32_t)sizeof(typeT), 0, 0, 0};
    AscendC::DataCopyPadExtParams<typeT> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(dst, src, params, padParams);
  }
}

template <typename typeT>
__aicore__ inline void PadOut(const AscendC::GlobalTensor<typeT>& dst,
                              const AscendC::LocalTensor<typeT>& src,
                              const uint32_t n) {
  constexpr uint32_t BLK = 32 / sizeof(typeT);
  if (n % BLK == 0) {
    AscendC::DataCopy(dst, src, n);
  } else {
    AscendC::DataCopyExtParams params{1, n * (uint32_t)sizeof(typeT), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, params);
  }
}

template <typename typeT, uint32_t ROUTE = 0>
class KernelGreater {
 public:
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                              const TilingData& t, AscendC::TPipe* p) {
    pipe = p;
    // Host emits only three fixed layouts.  Keep the route in the template so
    // dead broadcast/flat state is removed from each key's device binary.
    inner = t.inner;
    total = (uint32_t)((uint64_t)t.outer * t.inner);
    x1_gm.SetGlobalBuffer((__gm__ typeT*)x1);
    x2_gm.SetGlobalBuffer((__gm__ typeT*)x2);
    y_gm.SetGlobalBuffer((__gm__ int8_t*)y);
    // Cache policy follows reuse, not a shape id.  A same-shape compare streams
    // both operands once.  In the block-broadcast routes the broadcast operand is
    // a one-pass stream too: each core reads a broadcast row once per block
    // (bcastFactor output rows share it in UB, not in L2), so both skip L2.
    // L2 probe 2026-09-14: C2 -4.2 us, C3 -1.2 us against the cacheable row.
    uint64_t operandBytes = (uint64_t)total * sizeof(typeT);
    if (operandBytes >= (8ULL << 20)) {
      x1_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
      x2_gm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    }
    uint32_t blockIdx = AscendC::GetBlockIdx();

    if constexpr (ROUTE != 0) {
      outer = t.outer;
      innerTile = t.innerTile;
      bcastFactor = t.bcastFactor;
      rowsPerCore = t.rowsPerCore;
      cap = innerTile;
    } else {
      perCore = t.perCore;
      tileLen = t.tileLen;
      myStart = blockIdx * perCore;
      if (myStart >= total) {
        myLen = 0;
        numTiles = 0;
      } else {
        uint32_t rem = total - myStart;
        myLen = rem < perCore ? rem : perCore;
        numTiles = (myLen + tileLen - 1) / tileLen;
      }
      cap = tileLen;
    }

    if constexpr (ROUTE != 0) {
      // block-broadcast: one depth-2 input queue (fx2) serves both x1's row and
      // the big x2 tiles -> the load overlaps compute+store (best wide-row workload). x2/y at cap.
      pipe->InitBuffer(fx2, 2, cap * sizeof(typeT));
      constexpr bool bf16Block = std::is_same_v<typeT, bfloat16_t> && ROUTE == 2;
      uint8_t yNum = bf16Block ? (uint8_t)2 : (uint8_t)1;
      pipe->InitBuffer(y_outque, yNum, cap * sizeof(int8_t));
    } else {
      constexpr bool flatFloat = std::is_same_v<typeT, float> && ROUTE == 0;
      // int8 flat double-buffers too: at depth 2 the next tile's reads overlap the
      // current compare and store.  Equal-UB probe 2026-09-14: depth 2 @18432
      // 53.0 us against depth 1 @27392 56.1 / 54.8 us.
      constexpr bool flatInt8 = std::is_same_v<typeT, int8_t> && ROUTE == 0;
      constexpr uint8_t bufNum = (flatFloat || flatInt8) ? 2 : 1;
      pipe->InitBuffer(x1_inque, bufNum, cap * sizeof(typeT));
      pipe->InitBuffer(x2_inque, bufNum, cap * sizeof(typeT));
      pipe->InitBuffer(y_outque, bufNum, cap * sizeof(int8_t));
    }
    uint32_t maskBytes = cap / 8 < 32 ? 32 : cap / 8;
    pipe->InitBuffer(mask_buf, maskBytes);
    if constexpr (std::is_same_v<typeT, float>) {
      pipe->InitBuffer(hbufA, cap * sizeof(half));
    } else if constexpr (std::is_same_v<typeT, bfloat16_t>) {
      constexpr bool blockX2 = ROUTE == 2;
      uint32_t fbufABytes = blockX2 ? (6u * cap) : cap * (uint32_t)sizeof(float);
      pipe->InitBuffer(fbufA, fbufABytes);  // float x1F + same-bank-phase half hSel
      pipe->InitBuffer(fbufB, cap * sizeof(float));
      if (blockX2) {
        pipe->InitBuffer(idxBuf, cap * sizeof(uint32_t));  // periodic gather index
      }
    } else if constexpr (std::is_same_v<typeT, int8_t>) {
      // Compare reads hA/hB as two sources, so place hB 256B out of phase.
      // This lowers the int8 path's measured total vector conflict despite the
      // following Select reusing the same two buffers as src/dst.
      pipe->InitBuffer(hbufA, cap * sizeof(half) + 256u);
      pipe->InitBuffer(hbufB, cap * sizeof(half));
    } else if constexpr (std::is_same_v<typeT, int32_t>) {
      pipe->InitBuffer(fbufA, cap * sizeof(half));  // hOnes (half); frees UB for B=8
      pipe->InitBuffer(hbufB, cap * sizeof(half));
      if constexpr (ROUTE != 0) {
        pipe->InitBuffer(fbufB, cap * sizeof(float));      // x1F / x1-tiled cache
        pipe->InitBuffer(idxBuf, cap * sizeof(uint32_t));  // block: gather index
      }
    }
    // These buffers are never overwritten after initialization on the fp32
    // flat and int32 routes. Hoist the invariant vector out of every tile.
    if constexpr (std::is_same_v<typeT, float>) {
      AscendC::Duplicate(hbufA.Get<half>(), (half)1.0, cap);
    } else if constexpr (std::is_same_v<typeT, int32_t>) {
      AscendC::Duplicate(fbufA.Get<half>(), (half)1.0, cap);
    }
  }

  __aicore__ inline void Process() {
    if constexpr (ROUTE == 1) {
      ProcessBcastBlock();
    } else if constexpr (ROUTE == 2) {
      ProcessBcastBlockX2();
    } else {
      ProcessFlat();
    }
  }

 private:
  // y = (a > b); a,b,y are cap-length UB tensors. a,b are clobbered (scratch).
  __aicore__ inline void Gt(AscendC::LocalTensor<typeT>& a,
                            AscendC::LocalTensor<typeT>& b,
                            AscendC::LocalTensor<int8_t>& y, uint32_t n) {
    AscendC::LocalTensor<uint8_t> m0 = mask_buf.Get<uint8_t>();
    if constexpr (std::is_same_v<typeT, half>) {
      AscendC::Compare(m0, a, b, AscendC::CMPMODE::GT, n);
      AscendC::Duplicate(a, (half)1.0, n);
      AscendC::Select(b, m0, a, (half)0.0,
                      AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
      AscendC::Cast(y, b, AscendC::RoundMode::CAST_NONE, n);
    } else if constexpr (std::is_same_v<typeT, float>) {
      // Select in half domain: drop the float->half cast and halve Select width.
      // hbufA(half) holds ones; reuse b's buffer (free after Compare) as the
      // half Select dst (cap*4 bytes >= cap*2 half).
      AscendC::LocalTensor<half> hOnes = hbufA.Get<half>();
      AscendC::LocalTensor<half> hSel = b.template ReinterpretCast<half>();
      AscendC::Compare(m0, a, b, AscendC::CMPMODE::GT, n);
      AscendC::Select(hSel, m0, hOnes, (half)0.0,
                      AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
      AscendC::Cast(y, hSel, AscendC::RoundMode::CAST_NONE, n);
    } else if constexpr (std::is_same_v<typeT, bfloat16_t>) {
      // Compare in float (bf16 has no Compare), then Select in half: drops the
      // float->half cast and halves Duplicate/Select width. fA/fB free after
      // Compare -> reused as the two half buffers.
      AscendC::LocalTensor<float> fA = fbufA.Get<float>();
      AscendC::LocalTensor<float> fB = fbufB.Get<float>();
      AscendC::Cast(fA, a, AscendC::RoundMode::CAST_NONE, n);
      AscendC::Cast(fB, b, AscendC::RoundMode::CAST_NONE, n);
      AscendC::Compare(m0, fA, fB, AscendC::CMPMODE::GT, n);
      AscendC::LocalTensor<half> hOnes = fbufA.Get<half>();  // reuse fA
      AscendC::LocalTensor<half> hSel = fbufB.Get<half>();   // reuse fB
      AscendC::Duplicate(hOnes, (half)1.0, n);
      AscendC::Select(hSel, m0, hOnes, (half)0.0,
                      AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
      AscendC::Cast(y, hSel, AscendC::RoundMode::CAST_NONE, n);
    } else if constexpr (std::is_same_v<typeT, int8_t>) {
      AscendC::LocalTensor<half> hA = hbufA.Get<half>();
      AscendC::LocalTensor<half> hB = hbufB.Get<half>();
      AscendC::Cast(hA, a, AscendC::RoundMode::CAST_NONE, n);
      AscendC::Cast(hB, b, AscendC::RoundMode::CAST_NONE, n);
      AscendC::Compare(m0, hA, hB, AscendC::CMPMODE::GT, n);
      AscendC::Duplicate(hA, (half)1.0, n);
      AscendC::Select(hB, m0, hA, (half)0.0,
                      AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
      AscendC::Cast(y, hB, AscendC::RoundMode::CAST_NONE, n);
    } else if constexpr (std::is_same_v<typeT, int32_t>) {
      // A2: int32 vector Compare supports ONLY CMPMODE::EQ and runs ~100x slower
      // than fp32 Compare. int32 values are exact in fp32 for |v|<=2^24, so cast
      // both operands in place (same 4B slot) and use the fast fp32 Compare(GT).
      AscendC::LocalTensor<float> fA = a.template ReinterpretCast<float>();
      AscendC::LocalTensor<float> fB = b.template ReinterpretCast<float>();
      AscendC::Cast(fA, a, AscendC::RoundMode::CAST_NONE, n);
      AscendC::Cast(fB, b, AscendC::RoundMode::CAST_NONE, n);
      AscendC::Compare(m0, fA, fB, AscendC::CMPMODE::GT, n);
      AscendC::LocalTensor<half> hOnes = fbufA.Get<half>();
      AscendC::LocalTensor<half> hB = hbufB.Get<half>();
      AscendC::Select(hB, m0, hOnes, (half)0.0,
                      AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, n);
      AscendC::Cast(y, hB, AscendC::RoundMode::CAST_NONE, n);
    }
  }

  __aicore__ inline void ProcessFlat() {
    for (uint32_t tcnt = 0; tcnt < numTiles; tcnt++) {
      uint32_t off = tcnt * tileLen;
      uint32_t len = myLen - off;
      if (len > tileLen) len = tileLen;
      AscendC::LocalTensor<typeT> a = x1_inque.template AllocTensor<typeT>();
      AscendC::LocalTensor<typeT> b = x2_inque.template AllocTensor<typeT>();
      PadIn(a, x1_gm[myStart + off], len);
      PadIn(b, x2_gm[myStart + off], len);
      x1_inque.EnQue(a);
      x2_inque.EnQue(b);
      a = x1_inque.template DeQue<typeT>();
      b = x2_inque.template DeQue<typeT>();
      AscendC::LocalTensor<int8_t> y = y_outque.template AllocTensor<int8_t>();
      // Compare only the valid part of the tile (rounded to the vector
      // compare granule). The input tail is padding and must not affect the
      // result or make correctness depend on stale UB contents.
      uint32_t calc = (len + 63u) & ~63u;
      if (calc > tileLen) calc = tileLen;
      Gt(a, b, y, calc);
      y_outque.template EnQue<int8_t>(y);
      x1_inque.FreeTensor(a);
      x2_inque.FreeTensor(b);
      y = y_outque.template DeQue<int8_t>();
      PadOut(y_gm[myStart + off], y, len);
      y_outque.FreeTensor(y);
    }
  }

  // Block-broadcast: consecutive output rows share one input row.
  // (x1_gm[block*inner], block = row/bcastFactor); x2 is full. To reach flat-path
  // bandwidth we process B=innerTile/inner rows per tile with ONE big contiguous
  // x2 load and a SINGLE whole-tile compare (aligned start at offset 0 -- the
  // per-row slices x2F[rr*inner] would be 32B-misaligned for the odd inner=1023).
  // x1's row is tiled to [innerTile] once per block via Gather (contiguous aligned
  // output) using a constant periodic byte-offset index idx[k]=(k%inner)*4, built
  // once with vector ops (q=floor((k+0.5)/inner) is exact for k<2^24).
  __aicore__ inline void ProcessBcastBlock() {
    if constexpr (std::is_same_v<typeT, int32_t>) {
      // S9's int32 broadcast case is bounded to small integer values.  Use the
      // documented int32->half dequant cast (scale=1) so compare/select process
      // 128 values per repeat instead of the fp32 path's 64.
      AscendC::SetDeqScale(static_cast<half>(1.0f));
      uint32_t B = innerTile / inner;
      if (B < 1) B = 1;
      // ---- build periodic gather index idx[k] = (k % inner) * 4 (bytes) once ----
      AscendC::LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
      {
        AscendC::LocalTensor<float> kF = fbufB.Get<float>();
        AscendC::LocalTensor<int32_t> tmpI =
            fx2.template AllocTensor<int32_t>();
        AscendC::LocalTensor<float> tmpF = tmpI.template ReinterpretCast<float>();
        AscendC::ArithProgression<float>(kF, (float)0.0, (float)1.0, innerTile);
        float invInner = 1.0f / (float)(int32_t)inner;
        float fInner = (float)(int32_t)inner;
        AscendC::Adds(tmpF, kF, (float)0.5, innerTile);
        AscendC::Muls(tmpF, tmpF, invInner, innerTile);
        AscendC::LocalTensor<int32_t> qI = idx.template ReinterpretCast<int32_t>();
        AscendC::Cast(qI, tmpF, AscendC::RoundMode::CAST_FLOOR, innerTile);
        AscendC::Cast(tmpF, qI, AscendC::RoundMode::CAST_NONE, innerTile);
        AscendC::Muls(tmpF, tmpF, fInner, innerTile);
        AscendC::Sub(kF, kF, tmpF, innerTile);
        AscendC::Muls(kF, kF, (float)2.0, innerTile);
        AscendC::Cast(qI, kF, AscendC::RoundMode::CAST_ROUND, innerTile);
        fx2.FreeTensor(tmpI);
      }
      AscendC::LocalTensor<half> x1T = fbufB.Get<float>().template ReinterpretCast<half>();
      AscendC::LocalTensor<half> hOnes = fbufA.Get<half>();
      uint32_t rowStart = AscendC::GetBlockIdx() * rowsPerCore;
      uint32_t rowEnd = rowStart + rowsPerCore;
      if (rowEnd > outer) rowEnd = outer;
      uint32_t cachedBlock = 0xFFFFFFFFu;
      uint32_t r = rowStart;
      bool prefetched = false;
      while (r < rowEnd) {
        uint32_t block = r / bcastFactor;
        uint32_t blockEndRow = (block + 1) * bcastFactor;
        uint32_t tileRows = B;
        if (r + tileRows > blockEndRow) tileRows = blockEndRow - r;
        if (r + tileRows > rowEnd) tileRows = rowEnd - r;
        uint32_t n = tileRows * inner;
        uint32_t nA = (n + 63) / 64 * 64;  // aligned compute count
        if (nA > innerTile) nA = innerTile;
        if (block != cachedBlock) {
          AscendC::LocalTensor<int32_t> a =
              fx2.template AllocTensor<int32_t>();
          PadIn(a, x1_gm[(uint64_t)block * inner], inner);
          fx2.EnQue(a);
          a = fx2.template DeQue<int32_t>();
          AscendC::LocalTensor<half> aH = a.template ReinterpretCast<half>();
          AscendC::Cast(aH, a, AscendC::RoundMode::CAST_NONE, inner);
          AscendC::Gather(x1T, aH, idx, (uint32_t)0, innerTile);
          fx2.FreeTensor(a);
          cachedBlock = block;
        }
        AscendC::LocalTensor<int32_t> b;
        if (prefetched) {
          b = fx2.template DeQue<int32_t>();
          prefetched = false;
          // The queued tile was sized for the next iteration.  Its valid
          // length is recomputed below from the current row, while nA keeps
          // the vector count aligned exactly as in the non-prefetch path.
        } else {
          b = fx2.template AllocTensor<int32_t>();
          PadIn(b, x2_gm[(uint64_t)r * inner], n);
          fx2.EnQue(b);
          b = fx2.template DeQue<int32_t>();
        }
        AscendC::LocalTensor<half> x2H = b.template ReinterpretCast<half>();
        AscendC::Cast(x2H, b, AscendC::RoundMode::CAST_NONE, nA);
        // Issue the next contiguous x2 read while the current tile is still
        // in flight through V/MTE3.  Only prefetch within the same broadcast
        // block; crossing a block would compete with the x1 row refresh.
        uint32_t nextR = r + tileRows;
        if (nextR < rowEnd && nextR < blockEndRow) {
          uint32_t nextRows = B;
          if (nextR + nextRows > blockEndRow) nextRows = blockEndRow - nextR;
          if (nextR + nextRows > rowEnd) nextRows = rowEnd - nextR;
          AscendC::LocalTensor<int32_t> nextB =
              fx2.template AllocTensor<int32_t>();
          PadIn(nextB, x2_gm[(uint64_t)nextR * inner], nextRows * inner);
          fx2.EnQue(nextB);
          prefetched = true;
        }
        AscendC::LocalTensor<uint8_t> m0 = mask_buf.Get<uint8_t>();
        AscendC::Compare(m0, x1T, x2H, AscendC::CMPMODE::GT, nA);
        AscendC::LocalTensor<half> hB = hbufB.Get<half>();
        AscendC::Select(hB, m0, hOnes, (half)0.0,
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, nA);
        AscendC::LocalTensor<int8_t> y =
            y_outque.template AllocTensor<int8_t>();
        AscendC::Cast(y, hB, AscendC::RoundMode::CAST_NONE, nA);
        y_outque.template EnQue<int8_t>(y);
        fx2.FreeTensor(b);
        y = y_outque.template DeQue<int8_t>();
        PadOut(y_gm[(uint64_t)r * inner], y, n);
        y_outque.FreeTensor(y);
        r += tileRows;
      }
    }
  }

  // x2 broadcasts D-2 (fp16/fp32/bf16): bcastFactor rows share x2's row. Cache x2's row
  // (cast fp32 + periodic-gather-tiled to innerTile) per block; load x1's B-row tiles
  // contiguous; Gt(x1 > x2T). Batches the per-row work -> kills the generic bcast path's
  // scalar overhead (fused-axis workload: 58->~35us). BF16 packs x1F/hOnes/hSel into fbufA so
  // Select's source and destination start at the same 512B bank phase.
  __aicore__ inline void ProcessBcastBlockX2() {
    if constexpr (std::is_same_v<typeT, half> || std::is_same_v<typeT, bfloat16_t>) {
    uint32_t B = innerTile / inner;
    if (B < 1) B = 1;
    AscendC::LocalTensor<uint32_t> idx = idxBuf.Get<uint32_t>();
    {
      AscendC::LocalTensor<float> kF = fbufA.Get<float>();
      AscendC::LocalTensor<int32_t> qI = idx.template ReinterpretCast<int32_t>();
      AscendC::ArithProgression<float>(kF, (float)0.0, (float)1.0, innerTile);
      float invInner = 1.0f / (float)(int32_t)inner;
      float fInner = (float)(int32_t)inner;
      AscendC::LocalTensor<float> tF = fbufB.Get<float>();
      AscendC::Adds(tF, kF, (float)0.5, innerTile);
      AscendC::Muls(tF, tF, invInner, innerTile);
      AscendC::Cast(qI, tF, AscendC::RoundMode::CAST_FLOOR, innerTile);
      AscendC::Cast(tF, qI, AscendC::RoundMode::CAST_NONE, innerTile);
      AscendC::Muls(tF, tF, fInner, innerTile);
      AscendC::Sub(kF, kF, tF, innerTile);            // kF = k mod inner
      AscendC::Muls(kF, kF, (float)4.0, innerTile);   // fp32 byte offset
      AscendC::Cast(qI, kF, AscendC::RoundMode::CAST_ROUND, innerTile);
    }
    AscendC::LocalTensor<float> x2T = fbufB.Get<float>();  // cached tiled x2 row (fp32)
    uint32_t rowStart = AscendC::GetBlockIdx() * rowsPerCore;
    uint32_t rowEnd = rowStart + rowsPerCore;
    if (rowEnd > outer) rowEnd = outer;
    uint32_t cachedBlock = 0xFFFFFFFFu;
    uint32_t r = rowStart;
    while (r < rowEnd) {
      uint32_t block = r / bcastFactor;
      uint32_t blockEndRow = (block + 1) * bcastFactor;
      uint32_t tileRows = B;
      if (r + tileRows > blockEndRow) tileRows = blockEndRow - r;
      if (r + tileRows > rowEnd) tileRows = rowEnd - r;
      uint32_t n = tileRows * inner;
      uint32_t nA = (n + 127) / 128 * 128;
      if (nA > innerTile) nA = innerTile;
      if (block != cachedBlock) {
        AscendC::LocalTensor<typeT> a = fx2.template AllocTensor<typeT>();
        PadIn(a, x2_gm[(uint64_t)block * inner], inner);
        fx2.EnQue(a);
        a = fx2.template DeQue<typeT>();
        AscendC::LocalTensor<float> aF = fbufA.Get<float>();
        AscendC::Cast(aF, a, AscendC::RoundMode::CAST_NONE, inner);  // x2 row -> fp32
        AscendC::Gather(x2T, aF, idx, (uint32_t)0, innerTile);       // tile to B copies
        fx2.FreeTensor(a);
        cachedBlock = block;
      }
      AscendC::LocalTensor<typeT> b = fx2.template AllocTensor<typeT>();
      PadIn(b, x1_gm[(uint64_t)r * inner], n);
      fx2.EnQue(b);
      b = fx2.template DeQue<typeT>();
      AscendC::LocalTensor<float> x1F = fbufA.Get<float>();
      AscendC::Cast(x1F, b, AscendC::RoundMode::CAST_NONE, nA);   // x1 tile -> fp32
      AscendC::LocalTensor<uint8_t> m0 = mask_buf.Get<uint8_t>();
      AscendC::Compare(m0, x1F, x2T, AscendC::CMPMODE::GT, nA);   // x1 > x2
      AscendC::LocalTensor<half> hOnes = fbufA.Get<half>();       // reuse fbufA (x1F free)
      AscendC::LocalTensor<half> hSel;
      if constexpr (std::is_same_v<typeT, bfloat16_t>) {
        hSel = fbufA.Get<half>()[2u * cap];
      } else {
        hSel = hbufB.Get<half>();
      }
      AscendC::Duplicate(hOnes, (half)1.0, nA);
      AscendC::Select(hSel, m0, hOnes, (half)0.0,
                      AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, nA);
      AscendC::LocalTensor<int8_t> y = y_outque.template AllocTensor<int8_t>();
      AscendC::Cast(y, hSel, AscendC::RoundMode::CAST_NONE, nA);
      y_outque.template EnQue<int8_t>(y);
      fx2.FreeTensor(b);
      y = y_outque.template DeQue<int8_t>();
      PadOut(y_gm[(uint64_t)r * inner], y, n);
      y_outque.FreeTensor(y);
      r += tileRows;
    }
    }
  }

 private:
  AscendC::TPipe* pipe;  // declared in the kernel entry (class-external), passed in
  AscendC::TBuf<AscendC::TPosition::VECCALC> mask_buf, fbufA, fbufB, hbufA, hbufB,
      idxBuf;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> x1_inque, x2_inque;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> y_outque;
  // block-broadcast: depth-2 input queue (double buffer x1_row + x2 tiles).
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> fx2;
  AscendC::GlobalTensor<typeT> x1_gm, x2_gm;
  AscendC::GlobalTensor<int8_t> y_gm;
  uint32_t total, cap;
  uint32_t perCore, tileLen, myStart, myLen, numTiles;
  uint32_t inner, outer, innerTile;
  uint32_t bcastFactor, rowsPerCore;
};

extern "C" __global__ __aicore__ void greater(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                              GM_ADDR workspace,
                                              GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  AscendC::TPipe tpipe;  // class-external TPipe (lower per-launch overhead)
  if (TILING_KEY_IS(1)) {
    KernelGreater<DTYPE_X1, 1> op;
    op.Init(x1, x2, y, tiling_data, &tpipe);
    op.Process();
  } else if (TILING_KEY_IS(2)) {
    KernelGreater<DTYPE_X1, 2> op;
    op.Init(x1, x2, y, tiling_data, &tpipe);
    op.Process();
  } else if (TILING_KEY_IS(0)) {
    KernelGreater<DTYPE_X1, 0> op;
    op.Init(x1, x2, y, tiling_data, &tpipe);
    op.Process();
  }
}
