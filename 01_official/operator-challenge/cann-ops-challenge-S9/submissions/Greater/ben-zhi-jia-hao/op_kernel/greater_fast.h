#ifndef GREATER_FAST_H
#define GREATER_FAST_H
/*
 * Aligned, one-tile Greater kernel. The host selects this route only when
 * there is no broadcast, totalNum == usedCores * perCore, perCore is divisible
 * by 32, and the per-core work fits GT_FAST_TILE and the available UB.
 * The kernel relies on these prerequisites and has no tail path.
 * Float and half inputs use direct comparison; bfloat16 casts to float, int8
 * casts to half, and int32 comparison uses !(min(x1, x2) == x1). Selection
 * and conversion produce one Boolean byte per output element.
 */

namespace {

// Keep the fixed tile bound aligned with the host FAST_TILE_ELEMS.
constexpr uint32_t GT_FAST_TILE = 2048;

// The route reads a single uint64_t perCore field at offset zero.
using GtFastTiling = const __gm__ uint64_t *;

template <typename T>
class KernelGreaterFast {
public:
    using CT = typename GtTraits<T>::CT;
    static constexpr uint32_t S = sizeof(T);
    static constexpr uint32_t CS = sizeof(CT);
    static constexpr bool NEED_CAST = GtTraits<T>::kNeedCast;
    static constexpr bool INT_PATH = GtTraits<T>::kIntPath;

    // The fixed UB regions are aligned to 32 bytes and use 256-element vector bounds.
    static constexpr uint32_t N = GT_FAST_TILE;
    static constexpr uint32_t OFF_X = 0;                        // operand 1, T
    static constexpr uint32_t OFF_Y = OFF_X + N * S;            // operand 2, T
    static constexpr uint32_t OFF_O = OFF_Y + N * S;            // result, int8
    static constexpr uint32_t OFF_ONES = OFF_O + N;             // half 1.0
    static constexpr uint32_t OFF_SEL = OFF_ONES + N * 2;       // half 1.0/0.0
    static constexpr uint32_t OFF_MASK = OFF_SEL + N * 2;       // compare bitmask
    static constexpr uint32_t MASK_BYTES = (N / 8 + 255) / 256 * 256;
    static constexpr uint32_t OFF_CA = OFF_MASK + MASK_BYTES;   // cast/min scratch
    static constexpr uint32_t OFF_CB = OFF_CA + N * CS;
    static constexpr uint32_t UB_BYTES =
        NEED_CAST ? (OFF_CB + N * CS) : (INT_PATH ? (OFF_CA + N * CS) : OFF_CA);

    __aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        // The one thing TPipe's constructor does that any kernel needs:
        // set_atomic_none / set_mask_norm / set_vector_mask. This route builds
        // no TPipe, so it issues that call itself and skips the pool
        // bookkeeping, the buffer records and the destructor's record walk.
        InitSocState();

        // Read the tiling field and bind the input/output buffers before copying.
        const uint32_t n = static_cast<uint32_t>(reinterpret_cast<GtFastTiling>(tiling)[0]);
        x1Gm_.SetGlobalBuffer((__gm__ T *)x1);
        x2Gm_.SetGlobalBuffer((__gm__ T *)x2);
        yGm_.SetGlobalBuffer((__gm__ int8_t *)y);

        // The host guarantees an exact split. Each core processes the payload perCore
        // count without a tail clamp.
        const uint64_t start = static_cast<uint64_t>(GetBlockIdx()) * n;

        // Vector length, rounded up to a whole 256-element repeat. It reaches
        // past `n` into this route's own UB, never into a neighbour's - N is the
        // host's bound on n and every region above is sized for N - and nothing
        // beyond the first n bytes of the result is ever stored.
        const uint32_t na = (n + 255) / 256 * 256;

        LocalTensor<T> xL(TPosition::VECCALC, OFF_X, N);
        LocalTensor<T> yL(TPosition::VECCALC, OFF_Y, N);
        DataCopy(xL, x1Gm_[start], n);
        DataCopy(yL, x2Gm_[start], n);

        // Vector scratch is disjoint from the DMA input regions. Keep this vector
        // segment consistent with the optional diagnostic ladder.
        // ---- GT_VEC_BEGIN ----
        LocalTensor<half> ones(TPosition::VECCALC, OFF_ONES, N);
        Duplicate(ones, (half)1.0, na);

        // Wait for both MTE2 input transfers before the first vector operand read.
        SetFlag<HardEvent::MTE2_V>(EVT);
        WaitFlag<HardEvent::MTE2_V>(EVT);

        LocalTensor<uint8_t> mask(TPosition::VECCALC, OFF_MASK, MASK_BYTES);
        if constexpr (INT_PATH) {
            LocalTensor<int32_t> ca(TPosition::VECCALC, OFF_CA, N);
            Min(ca, xL, yL, na);                        // min(x1, x2)
            Compare(mask, ca, xL, CMPMODE::EQ, na);     // (min == x1) <=> (x1 <= x2)
            LocalTensor<uint16_t> m16 = mask.template ReinterpretCast<uint16_t>();
            Not(m16, m16, MASK_BYTES / 2);              // negate -> (x1 > x2)
        } else if constexpr (NEED_CAST) {
            LocalTensor<CT> ca(TPosition::VECCALC, OFF_CA, N);
            LocalTensor<CT> cb(TPosition::VECCALC, OFF_CB, N);
            Cast(ca, xL, RoundMode::CAST_NONE, na);
            Cast(cb, yL, RoundMode::CAST_NONE, na);
            Compare(mask, ca, cb, CMPMODE::GT, na);
        } else {
            Compare(mask, xL, yL, CMPMODE::GT, na);
        }
        LocalTensor<half> sel(TPosition::VECCALC, OFF_SEL, N);
        Select(sel, mask, ones, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, na);
        LocalTensor<int8_t> oL(TPosition::VECCALC, OFF_O, N);
        Cast(oL, sel, RoundMode::CAST_NONE, na);
        // ---- GT_VEC_END ----

        // Wait for vector conversion before copying the Boolean output through MTE3.
        SetFlag<HardEvent::V_MTE3>(EVT);
        WaitFlag<HardEvent::V_MTE3>(EVT);
        DataCopy(yGm_[start], oL, n);   // bool output: one byte per element

        // Drain all pipelines before returning.
        pipe_barrier(PIPE_ALL);
    }

private:
    // EVENT IDS. Id 0 for both edges. The event pool is indexed per HardEvent
    // (kernel_tpipe_impl.h, `eventPool_ + EventToIndex(evt)`), so MTE2_V id 0
    // and V_MTE3 id 0 are different hardware flags; and this route constructs no
    // TPipe and allocates no TQue, so nothing else in the variant has taken an
    // id or could collide with these two.
    static constexpr int32_t EVT = 0;

    GlobalTensor<T> x1Gm_, x2Gm_;
    GlobalTensor<int8_t> yGm_;
};

}  // namespace

#endif  // GREATER_FAST_H
