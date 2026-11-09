#ifndef GREATER_LADDER_H
#define GREATER_LADDER_H
/*
 * Optional diagnostic prefix kernels for the aligned fp16 route. The host
 * gates them with S9_VARIANT 10 through 16 and the aligned-route prerequisites.
 * Keys 900 through 906 select progressively longer prefixes: entry, observation,
 * tiling read, address setup, first input load, second input load and vector work.
 * These kernels intentionally do not produce ordinary Greater output.
 * Nonempty prefixes write a 32-byte observation per core after draining pending
 * pipelines. Keep the vector segment consistent with greater_fast.h.
 * Prefix timings can expose different overlap and are not independent stage costs.
 */

namespace {

// Ladder levels. The tiling key is GT_LADDER_KEY0 + the level, and the host
// arm is GT_VAR_LADDER0 + the level; all three are kept in step by
// greater_tiling_core.h, which is the authority - this file cannot check it.
constexpr uint32_t GT_LV_EMPTY  = 0;   // L0   enter, return
constexpr uint32_t GT_LV_WRITE  = 1;   // L0w  the observation write only
constexpr uint32_t GT_LV_TILING = 2;   // L1   + the 8-byte tiling read
constexpr uint32_t GT_LV_SETUP  = 3;   // L2   + SetGlobalBuffer, start, na
constexpr uint32_t GT_LV_IN1    = 4;   // L3   + DataCopy in x1
constexpr uint32_t GT_LV_IN2    = 5;   // L4   + DataCopy in x2
constexpr uint32_t GT_LV_VEC    = 6;   // L5   + the compare/select/cast
constexpr uint32_t GT_LV_MAX    = GT_LV_VEC;

template <typename T, uint32_t LEVEL>
class KernelGreaterLadder {
public:
    // Reuse the fixed UB layout declared by the aligned kernel.
    using F = KernelGreaterFast<T>;
    using CT = typename F::CT;
    static constexpr uint32_t N = F::N;

    // The additional 32-byte observation block follows F::UB_BYTES. This diagnostic
    // route relies on the host aligned-route UB reservation covering that block.
    static constexpr uint32_t OFF_STUB = (F::UB_BYTES + 31u) / 32u * 32u;
    static constexpr uint32_t STUB_BYTES = 32;   // one whole aligned datablock

    __aicore__ inline void Run(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR tiling)
    {
        // Level zero returns immediately without producing output.
        if constexpr (LEVEL == GT_LV_EMPTY) {
            return;
        } else {
            // ---- L0w and up ------------------------------------------------
            // What `TPipe`'s constructor does that any kernel needs:
            // set_atomic_none / set_mask_norm / set_vector_mask. set_atomic_none
            // is load-bearing even here - the epilogue's store would ACCUMULATE
            // rather than overwrite if a previous kernel left atomic mode set.
            InitSocState();
            yGm_.SetGlobalBuffer((__gm__ int8_t *)y);

            // The accumulator that keeps every stage alive. Seeded from
            // GetBlockIdx() rather than a literal so that no part of the chain
            // is a compile-time constant the compiler could fold through.
            uint32_t sink = static_cast<uint32_t>(GetBlockIdx());

            uint32_t n = 0;
            if constexpr (LEVEL >= GT_LV_TILING) {
                // Read perCore from the uint64_t field at tiling offset zero.
                n = static_cast<uint32_t>(reinterpret_cast<GtFastTiling>(tiling)[0]);
                sink += n;
            }

            uint64_t start = 0;
            uint32_t na = 0;
            if constexpr (LEVEL >= GT_LV_SETUP) {
                // Bind buffers, compute the core offset and round the vector extent as in
                // the aligned kernel.
                x1Gm_.SetGlobalBuffer((__gm__ T *)x1);
                x2Gm_.SetGlobalBuffer((__gm__ T *)x2);
                start = static_cast<uint64_t>(GetBlockIdx()) * n;
                na = (n + 255u) / 256u * 256u;
                sink += static_cast<uint32_t>(start) + na;
            }

            LocalTensor<T> xL(TPosition::VECCALC, F::OFF_X, N);
            LocalTensor<T> yL(TPosition::VECCALC, F::OFF_Y, N);
            if constexpr (LEVEL >= GT_LV_IN1) {
                // ---- L3: first operand in ---------------------------------
                DataCopy(xL, x1Gm_[start], n);
            }
            if constexpr (LEVEL >= GT_LV_IN2) {
                // Submit the second input load after the first on MTE2.
                DataCopy(yL, x2Gm_[start], n);
            }

            if constexpr (LEVEL >= GT_LV_VEC) {
                // The vector segment mirrors the aligned kernel.
                // ---- GT_VEC_BEGIN ----
                LocalTensor<half> ones(TPosition::VECCALC, F::OFF_ONES, N);
                Duplicate(ones, (half)1.0, na);
                SetFlag<HardEvent::MTE2_V>(EVT);
                WaitFlag<HardEvent::MTE2_V>(EVT);
                LocalTensor<uint8_t> mask(TPosition::VECCALC, F::OFF_MASK, F::MASK_BYTES);
                if constexpr (F::INT_PATH) {
                    LocalTensor<int32_t> ca(TPosition::VECCALC, F::OFF_CA, N);
                    Min(ca, xL, yL, na);
                    Compare(mask, ca, xL, CMPMODE::EQ, na);
                    LocalTensor<uint16_t> m16 = mask.template ReinterpretCast<uint16_t>();
                    Not(m16, m16, F::MASK_BYTES / 2);
                } else if constexpr (F::NEED_CAST) {
                    LocalTensor<CT> ca(TPosition::VECCALC, F::OFF_CA, N);
                    LocalTensor<CT> cb(TPosition::VECCALC, F::OFF_CB, N);
                    Cast(ca, xL, RoundMode::CAST_NONE, na);
                    Cast(cb, yL, RoundMode::CAST_NONE, na);
                    Compare(mask, ca, cb, CMPMODE::GT, na);
                } else {
                    Compare(mask, xL, yL, CMPMODE::GT, na);
                }
                LocalTensor<half> sel(TPosition::VECCALC, F::OFF_SEL, N);
                Select(sel, mask, ones, (half)0.0, SELMODE::VSEL_TENSOR_SCALAR_MODE, na);
                LocalTensor<int8_t> oL(TPosition::VECCALC, F::OFF_O, N);
                Cast(oL, sel, RoundMode::CAST_NONE, na);
                // ---- GT_VEC_END ----
            }

            Epilogue(sink);
        }
    }

private:
    // Drain pending input and vector work before scalar observation reads.
    // The observation retains data dependencies on the reached prefix. Lower
    // levels can read untouched UB values; this is diagnostic data, not output.
    // Each core stores 32 bytes at blockIdx * 32. The host exact split and
    // perCore >= 32 prerequisite keep these stores within the output allocation.
    __aicore__ inline void Epilogue(uint32_t sink)
    {
        pipe_barrier(PIPE_ALL);                                     // (a)

        LocalTensor<uint16_t> pX(TPosition::VECCALC, F::OFF_X, 16);  // (b)
        LocalTensor<uint16_t> pY(TPosition::VECCALC, F::OFF_Y, 16);
        LocalTensor<uint8_t> pO(TPosition::VECCALC, F::OFF_O, 32);
        sink += static_cast<uint32_t>(pX.GetValue(0));
        sink += static_cast<uint32_t>(pY.GetValue(0));
        sink += static_cast<uint32_t>(pO.GetValue(0));

        LocalTensor<int32_t> sW(TPosition::VECCALC, OFF_STUB, STUB_BYTES / 4);
        sW.SetValue(0, static_cast<int32_t>(sink));
        pipe_barrier(PIPE_ALL);                                     // S -> MTE3

        LocalTensor<int8_t> sB(TPosition::VECCALC, OFF_STUB, STUB_BYTES);  // (c)
        DataCopy(yGm_[static_cast<uint64_t>(GetBlockIdx()) * STUB_BYTES], sB, STUB_BYTES);

        pipe_barrier(PIPE_ALL);   // Drain all pipelines after the observation store.
    }

    // Event IDs are independent for each HardEvent type. This route constructs
    // no TPipe or TQue, so no queue allocation consumes this event ID.
    static constexpr int32_t EVT = 0;

    GlobalTensor<T> x1Gm_, x2Gm_;
    GlobalTensor<int8_t> yGm_;
};

}  // namespace

#endif  // GREATER_LADDER_H
