/**
 * Transpose 设备侧实现。
 * 按分块信息执行连续片段搬运、整块转置或二维 tile 转置，并维护双缓冲流水线依赖。
 */

#include "kernel_operator.h"
#include "lib/transpose/confusion_transpose.h"
#include "../op_host/transpose_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kRunTileElements = 8192U;
constexpr uint32_t kLargeRunTileElements = 24576U;
constexpr uint32_t kMaxTile = 64U;
constexpr uint32_t kUbBlockBytes = 32U;
constexpr int32_t kTileBufferDepth = 2;

template <typename A, typename B>
struct SameType {
    static constexpr bool value = false;
};
template <typename A>
struct SameType<A, A> {
    static constexpr bool value = true;
};

template <typename T>
struct GatherTraits {
    using Type = T;
};
template <>
struct GatherTraits<int8_t> {
    using Type = half;
};

__aicore__ inline uint32_t AlignUpBlock(uint32_t bytes)
{
    return ((bytes + kUbBlockBytes - 1) / kUbBlockBytes) * kUbBlockBytes;
}

struct TileMeta {
    uint64_t inOffset;
    uint64_t outOffset;
    uint32_t countA;
    uint32_t countB;
};

struct RunMeta {
    uint32_t batch;
    uint64_t inOffset;
    uint64_t outOffset;
    uint32_t rows;
    uint32_t count;
};

template <typename T>
// 搬运置换后仍连续的片段，减少不必要的逐元素索引计算。
class KernelRunTranspose {
public:
    static constexpr uint32_t kRunElements =
        SameType<T, float>::value || SameType<T, int8_t>::value ?
        kLargeRunTileElements : kRunTileElements;

    static constexpr uint32_t kPackRunElements =
        sizeof(T) == 1 ? 20480U : 8192U;

    static constexpr uint32_t kRunPackUbBytes = 180U * 1024U;

    __aicore__ static inline uint32_t PackUbBytes(
        uint32_t exactBytes, uint32_t rows, uint32_t batch, uint32_t perRow)
    {
        const uint32_t pitch = AlignUpBlock(exactBytes * batch);
        return 2U * AlignUpBlock(rows * pitch) +
               2U * AlignUpBlock(batch * rows * exactBytes) +
               AlignUpBlock(batch * rows * perRow *
                            static_cast<uint32_t>(sizeof(int32_t)));
    }

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const TransposeTilingData& tiling)
    {
        in_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputs));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        if (tiling.bypassOutput != 0U) {

            out_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }


        for (uint32_t i = 0; i < S9_TRANSPOSE_MAX_GROUP; ++i) {
            outerLen_[i] = tiling.outerLen[i];
            outerInStride_[i] = tiling.outerInStride[i];
            outerOutStride_[i] = tiling.outerOutStride[i];
        }
        outerCount_ = tiling.outerCount;
        taskCount_ = tiling.taskCount;
        runLen_ = tiling.runLen;
        runChunk_ = tiling.runChunk;
        runChunkCount_ = tiling.runChunkCount;
        midLen_ = tiling.midLen;
        midInStride_ = tiling.midInStride;
        midBatch_ = tiling.midBatch;
        midChunkCount_ = tiling.midChunkCount;

        outerTotal_ = tiling.outerTotal;

        outerBatch_ = tiling.tileA >= 1ULL ?
            static_cast<uint32_t>(tiling.tileA) : 1U;
        usePacked_ = 0U;
        {
            const uint32_t exact =
                static_cast<uint32_t>(runChunk_) * static_cast<uint32_t>(sizeof(T));

            packUnit_ = (exact % 4U == 0U) ? 4U :
                (sizeof(T) >= 2U ? static_cast<uint32_t>(sizeof(T)) : 0U);

            if (runChunkCount_ == 1U && (exact % kUbBlockBytes) != 0U &&
                midBatch_ > 1U && packUnit_ >= 2U &&
                PackUbBytes(exact, static_cast<uint32_t>(midBatch_),
                            outerBatch_, exact / packUnit_) <= kRunPackUbBytes) {
                usePacked_ = 1U;
                packPerRow_ = exact / packUnit_;
            }
        }
        if (usePacked_ == 0U) {
            pipe_.InitBuffer(runBufA_, kRunElements * sizeof(T));
            pipe_.InitBuffer(runBufB_, kRunElements * sizeof(T));
        } else {
            const uint32_t exactBytes =
                static_cast<uint32_t>(runChunk_) * static_cast<uint32_t>(sizeof(T));
            const uint32_t pitchBytes = AlignUpBlock(exactBytes * outerBatch_);
            packPitchBytes_ = pitchBytes;

            const uint32_t rawBytes = static_cast<uint32_t>(midBatch_) * pitchBytes;
            const uint32_t packBytes =
                outerBatch_ * static_cast<uint32_t>(midBatch_) * exactBytes;
            const uint32_t offBytes = outerBatch_ *
                static_cast<uint32_t>(midBatch_) * packPerRow_ *
                static_cast<uint32_t>(sizeof(int32_t));

            pipe_.InitBuffer(runBufA_, AlignUpBlock(rawBytes));
            pipe_.InitBuffer(runBufB_, AlignUpBlock(rawBytes));
            pipe_.InitBuffer(packBufA_, AlignUpBlock(packBytes));
            pipe_.InitBuffer(packBufB_, AlignUpBlock(packBytes));
            pipe_.InitBuffer(offsetBuf_, AlignUpBlock(offBytes));
            const uint32_t perRow = packPerRow_;
            LocalTensor<int32_t> off = offsetBuf_.template Get<int32_t>();
            const uint32_t rowsMax = static_cast<uint32_t>(midBatch_);

            const uint32_t nOff = outerBatch_ * rowsMax * perRow;
            LocalTensor<float> f0 = runBufA_.template Get<float>();
            uint32_t kChunk = f0.GetSize() / 3U;
            if (kChunk > 2048U) { kChunk = 2048U; }
            kChunk = (kChunk / 64U) * 64U;
            if (kChunk == 0U) {
                for (uint32_t q = 0; q < outerBatch_; ++q) {
                    for (uint32_t r = 0; r < rowsMax; ++r) {
                        const uint32_t bse = r * pitchBytes + q * exactBytes;
                        for (uint32_t j = 0; j < perRow; ++j) {
                            off.SetValue((q * rowsMax + r) * perRow + j,
                                static_cast<int32_t>(bse + j * packUnit_));
                        }
                    }
                }
            } else {
                LocalTensor<float> f1 = f0[kChunk];
                LocalTensor<float> f2 = f0[2U * kChunk];
                const float invPer =
                    1.0f / static_cast<float>(static_cast<int32_t>(perRow));
                const float invRows =
                    1.0f / static_cast<float>(static_cast<int32_t>(rowsMax));
                const float fA = static_cast<float>(static_cast<int32_t>(packUnit_));
                const float fP = static_cast<float>(
                    static_cast<int32_t>(pitchBytes) -
                    static_cast<int32_t>(exactBytes));
                const float fQ = static_cast<float>(
                    static_cast<int32_t>(exactBytes) *
                    (1 - static_cast<int32_t>(rowsMax)));
                const float fRows = static_cast<float>(static_cast<int32_t>(rowsMax));
                const float fPer = static_cast<float>(static_cast<int32_t>(perRow));
                for (uint32_t base = 0; base < nOff; base += kChunk) {
                    const int32_t c = static_cast<int32_t>(
                        (nOff - base) < kChunk ? (nOff - base) : kChunk);
                    CreateVecIndex(f0, static_cast<float>(static_cast<int32_t>(base)), c);
                    PipeBarrier<PIPE_V>();
                    Muls(f1, f0, invPer, c);
                    PipeBarrier<PIPE_V>();
                    Adds(f1, f1, 1.0e-4f, c);
                    PipeBarrier<PIPE_V>();
                    Cast(f1, f1, RoundMode::CAST_FLOOR, c);
                    PipeBarrier<PIPE_V>();
                    Muls(f2, f1, invRows, c);
                    PipeBarrier<PIPE_V>();
                    Adds(f2, f2, 1.0e-4f, c);
                    PipeBarrier<PIPE_V>();
                    Cast(f2, f2, RoundMode::CAST_FLOOR, c);
                    PipeBarrier<PIPE_V>();

                    Muls(f0, f2, -fRows, c);
                    PipeBarrier<PIPE_V>();
                    Add(f1, f1, f0, c);
                    PipeBarrier<PIPE_V>();

                    CreateVecIndex(f0, static_cast<float>(static_cast<int32_t>(base)), c);
                    PipeBarrier<PIPE_V>();
                    Muls(f0, f0, fA, c);
                    PipeBarrier<PIPE_V>();
                    Muls(f1, f1, fP, c);
                    PipeBarrier<PIPE_V>();
                    Add(f0, f0, f1, c);
                    PipeBarrier<PIPE_V>();
                    Muls(f2, f2, fQ, c);
                    PipeBarrier<PIPE_V>();
                    Add(f0, f0, f2, c);
                    PipeBarrier<PIPE_V>();
                    Cast(off[base], f0, RoundMode::CAST_RINT, c);
                    PipeBarrier<PIPE_V>();
                }
                (void)fPer;
            }
            SetFlag<HardEvent::S_V>(EVENT_ID1);
            WaitFlag<HardEvent::S_V>(EVENT_ID1);
        }
    }

    __aicore__ inline void Process()
    {
        if (taskCount_ == 0) {
            return;
        }
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t share = taskCount_ / blocks;
        const uint64_t extra = taskCount_ % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1U : 0U);
        if (begin >= end) {
            return;
        }

        if (usePacked_ != 0U) {
            ProcessPacked(begin, end);
            return;
        }
        RunMeta meta[2];
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        ComputeMeta(begin, meta[0]);
        IssueLoad(meta[0], 0U);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);

        const uint64_t count = end - begin;
        for (uint64_t i = 0; i < count; ++i) {
            const uint32_t ping = static_cast<uint32_t>(i & 1U);
            const event_t event = static_cast<event_t>(ping);
            if (i + 1U < count) {
                const uint32_t next = ping ^ 1U;
                const event_t nextEvent = static_cast<event_t>(next);
                WaitFlag<HardEvent::MTE3_MTE2>(nextEvent);
                ComputeMeta(begin + i + 1U, meta[next]);
                IssueLoad(meta[next], next);
                SetFlag<HardEvent::MTE2_MTE3>(nextEvent);
            }
            WaitFlag<HardEvent::MTE2_MTE3>(event);
            IssueStore(meta[ping], ping);
            SetFlag<HardEvent::MTE3_MTE2>(event);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

private:

    __aicore__ inline void ProcessPacked(uint64_t begin, uint64_t end)
    {
        LocalTensor<T> packs[2] = {
            packBufA_.template Get<T>(), packBufB_.template Get<T>()};
        LocalTensor<uint32_t> offs =
            offsetBuf_.template Get<int32_t>().template ReinterpretCast<uint32_t>();
        RunMeta meta[2];
        const uint64_t n = end - begin;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        ComputeMeta(begin, meta[0]);
        IssueLoad(meta[0], 0U);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint64_t i = 0; i < n; ++i) {
            const uint32_t p = static_cast<uint32_t>(i & 1U);
            const event_t ep = static_cast<event_t>(p);
            if (i + 1U < n) {
                const uint32_t np = p ^ 1U;
                const event_t nextEvent = static_cast<event_t>(np);
                WaitFlag<HardEvent::V_MTE2>(nextEvent);
                ComputeMeta(begin + i + 1U, meta[np]);
                IssueLoad(meta[np], np);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }
            WaitFlag<HardEvent::MTE2_V>(ep);
            WaitFlag<HardEvent::MTE3_V>(ep);
            LocalTensor<T> tile =
                p == 0U ? runBufA_.template Get<T>() : runBufB_.template Get<T>();
            const uint32_t total =
                meta[p].batch * meta[p].rows * meta[p].count;

            const uint32_t gcount =
                meta[p].batch * meta[p].rows * packPerRow_;
            if (packUnit_ == 4U) {
                LocalTensor<int32_t> d = packs[p].template ReinterpretCast<int32_t>();
                LocalTensor<int32_t> sT = tile.template ReinterpretCast<int32_t>();
                Gather(d, sT, offs, 0U, gcount);
            } else {
                Gather(packs[p], tile, offs, 0U, gcount);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(ep);
            SetFlag<HardEvent::V_MTE3>(ep);
            WaitFlag<HardEvent::V_MTE3>(ep);
            const DataCopyExtParams store(
                1U, static_cast<uint32_t>(total * sizeof(T)), 0U, 0U, 0U);
            DataCopyPad(out_[meta[p].outOffset], packs[p], store);
            SetFlag<HardEvent::MTE3_V>(ep);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    __aicore__ inline void OuterOffsets(
        uint64_t index, uint64_t& inOffset, uint64_t& outOffset) const
    {
        inOffset = 0;
        outOffset = 0;
        for (int32_t i = static_cast<int32_t>(outerCount_) - 1; i >= 0; --i) {
            const uint64_t len = outerLen_[i];
            const uint64_t digit = index % len;
            index /= len;
            inOffset += digit * outerInStride_[i];
            outOffset += digit * outerOutStride_[i];
        }
    }

    __aicore__ inline void ComputeMeta(uint64_t task, RunMeta& meta) const
    {
        const uint64_t perOuter = midChunkCount_ * runChunkCount_;
        const uint64_t outerSlot = task / perOuter;
        const uint64_t outerIndex = outerSlot * outerBatch_;
        const uint64_t rest = task - outerSlot * perOuter;
        const uint64_t midChunk = rest / runChunkCount_;
        const uint64_t runChunk = rest - midChunk * runChunkCount_;
        uint64_t inBase = 0;
        uint64_t outBase = 0;
        OuterOffsets(outerIndex, inBase, outBase);
        const uint64_t midStart = midChunk * midBatch_;
        const uint64_t leftOuter = outerTotal_ - outerIndex;
        meta.batch = static_cast<uint32_t>(
            leftOuter < outerBatch_ ? leftOuter : outerBatch_);
        meta.rows = static_cast<uint32_t>(
            (midLen_ - midStart) < midBatch_ ? (midLen_ - midStart) : midBatch_);
        const uint64_t runStart = runChunk * runChunk_;
        meta.count = static_cast<uint32_t>(
            (runLen_ - runStart) < runChunk_ ? (runLen_ - runStart) : runChunk_);
        meta.inOffset = inBase + midStart * midInStride_ + runStart;
        meta.outOffset = outBase + midStart * runLen_ + runStart;
    }

    __aicore__ inline void IssueLoad(const RunMeta& meta, uint32_t ping)
    {
        LocalTensor<T> tile = ping == 0U ? runBufA_.template Get<T>() : runBufB_.template Get<T>();
        const DataCopyPadExtParams<T> pad;
        if (usePacked_ != 0U && outerBatch_ > 1U) {
            const uint32_t wide =
                meta.batch * meta.count * static_cast<uint32_t>(sizeof(T));

            const uint32_t widePad = AlignUpBlock(wide);
            const DataCopyExtParams wl(
                meta.rows, wide,
                static_cast<uint32_t>(midInStride_ * sizeof(T)) - wide,
                (packPitchBytes_ - widePad) / kUbBlockBytes, 0);
            DataCopyPad(tile, in_[meta.inOffset], wl, pad);
            return;
        }

        const uint32_t exact = meta.count * static_cast<uint32_t>(sizeof(T));
        const uint32_t padded = AlignUpBlock(exact);
        const uint64_t strideBytes = midInStride_ * sizeof(T);
        if (padded != exact && meta.rows > 1U && strideBytes >= padded) {
            const uint32_t pitch = padded / static_cast<uint32_t>(sizeof(T));
            const DataCopyExtParams head(
                meta.rows - 1U, padded,
                static_cast<uint32_t>(strideBytes - padded), 0, 0);
            DataCopyPad(tile, in_[meta.inOffset], head, pad);
            const DataCopyExtParams tail(1U, exact, 0, 0, 0);
            DataCopyPad(
                tile[(meta.rows - 1U) * pitch],
                in_[meta.inOffset + (meta.rows - 1U) * midInStride_], tail, pad);
            return;
        }
        const DataCopyExtParams load(
            meta.rows, exact,
            (midInStride_ - meta.count) * sizeof(T), 0, 0);
        DataCopyPad(tile, in_[meta.inOffset], load, pad);
    }

    __aicore__ inline void IssueStore(const RunMeta& meta, uint32_t ping)
    {
        LocalTensor<T> tile = ping == 0U ? runBufA_.template Get<T>() : runBufB_.template Get<T>();

        const uint32_t exact = meta.count * static_cast<uint32_t>(sizeof(T));
        if (meta.count == runLen_ && (exact % kUbBlockBytes) == 0U) {
            const DataCopyExtParams whole(
                1U, static_cast<uint32_t>(meta.rows) * exact, 0U, 0U, 0U);
            DataCopyPad(out_[meta.outOffset], tile, whole);
            return;
        }
        const DataCopyExtParams store(
            meta.rows, meta.count * sizeof(T), 0,
            (runLen_ - meta.count) * sizeof(T), 0);
        DataCopyPad(out_[meta.outOffset], tile, store);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> runBufA_;
    TBuf<TPosition::VECCALC> runBufB_;
    TBuf<TPosition::VECCALC> packBufA_;
    TBuf<TPosition::VECCALC> packBufB_;
    TBuf<TPosition::VECCALC> offsetBuf_;
    uint32_t usePacked_ = 0U;
    uint32_t outerBatch_ = 1U;
    uint64_t outerTotal_ = 1U;
    uint32_t packPitchBytes_ = 0U;
    uint32_t packUnit_ = 0U;
    uint32_t packPerRow_ = 0U;
    GlobalTensor<T> in_;
    GlobalTensor<T> out_;
    uint64_t outerLen_[S9_TRANSPOSE_MAX_GROUP];
    uint64_t outerInStride_[S9_TRANSPOSE_MAX_GROUP];
    uint64_t outerOutStride_[S9_TRANSPOSE_MAX_GROUP];
    uint64_t taskCount_;
    uint64_t runLen_;
    uint64_t runChunk_;
    uint64_t runChunkCount_;
    uint64_t midLen_;
    uint64_t midInStride_;
    uint64_t midBatch_;
    uint64_t midChunkCount_;
    uint32_t outerCount_;
};

template<typename T, uint32_t kBatchCap>
// 将多个连续片段组成批次，复用分块元数据并合并搬运。
class KernelBatchRunTranspose {
public:
    __aicore__ static inline bool Select(const TransposeTilingData& t,
        uint32_t& axis, uint32_t& batch, uint32_t& mid)
    {
        if constexpr (sizeof(T) != 2U) return false;
        if (t.regime != S9_TRANSPOSE_REGIME_RUN || t.runChunkCount != 1U ||
            t.midChunkCount != 1U || t.tileA != 1U || t.midLen < 2U ||
            t.runLen == 0U || t.runLen * sizeof(T) > 512U ||
            (t.runLen * sizeof(T)) % 32U == 0U || t.midInStride * sizeof(T) > 0xFFFFFFFFULL) return false;
        axis = S9_TRANSPOSE_MAX_GROUP;
        for (uint32_t i = 0; i < t.outerCount; ++i) {
            if (t.outerInStride[i] == t.runLen && t.outerLen[i] > 1U &&
                t.outerOutStride[i] * sizeof(T) <= 0xFFFFFFFFULL) { axis = i; break; }
        }
        if (axis == S9_TRANSPOSE_MAX_GROUP) return false;
        batch = static_cast<uint32_t>(t.outerLen[axis] < kBatchCap ? t.outerLen[axis] : kBatchCap);
        if (batch < 2U || t.midInStride < batch * t.runLen) return false;
        const uint32_t exact = static_cast<uint32_t>(t.runLen * sizeof(T));
        const uint32_t rawPitch = AlignUpBlock(batch * exact + LanePad(t));
        const uint32_t bound = (180U * 1024U) / (6U * batch * exact);
        mid = static_cast<uint32_t>(t.midLen < bound ? t.midLen : bound);
        while (mid > 0U && 2U * mid * rawPitch + 4U * batch * AlignUpBlock(mid * exact) > 180U * 1024U) --mid;
        if (mid == 0U) return false;
        const uint64_t groups = (t.midLen + mid - 1U) / mid;
        mid = static_cast<uint32_t>((t.midLen + groups - 1U) / groups);
        return t.outerOutStride[axis] >= mid * t.runLen;
    }

    __aicore__ static inline uint32_t LanePad(const TransposeTilingData& t)
    {
        return ((t.midInStride * sizeof(T)) % 32U) != 0U ? 62U : 0U;
    }
    __aicore__ static inline bool CanUse(const TransposeTilingData& t)
    {
        uint32_t axis, batch, mid; return Select(t, axis, batch, mid);
    }
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, const TransposeTilingData& t)
    {
        in_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        if (t.bypassOutput != 0U) out_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        Select(t, axis_, batch_, mid_);
        run_ = t.runLen; midLen_ = t.midLen; midStride_ = t.midInStride; outerCount_ = t.outerCount;
        inTotal_ = 1U;
        for (uint32_t i = 0; i < t.outerCount; ++i) inTotal_ *= t.outerLen[i];
        inTotal_ *= t.midLen * t.runLen;
        for (uint32_t i = 0; i < S9_TRANSPOSE_MAX_GROUP; ++i) {
            lens_[i] = t.outerLen[i]; istrides_[i] = t.outerInStride[i]; ostrides_[i] = t.outerOutStride[i];
        }
        axisGroups_ = (lens_[axis_] + batch_ - 1U) / batch_;
        midGroups_ = (midLen_ + mid_ - 1U) / mid_;
        tasks_ = (t.outerTotal / lens_[axis_]) * axisGroups_ * midGroups_;
        rawPitch_ = AlignUpBlock(static_cast<uint32_t>(batch_ * run_ * sizeof(T)) + LanePad(t));
        laneL_ = 1U;
        {
            const uint32_t r = static_cast<uint32_t>((midStride_ * sizeof(T)) % 32U);
            uint32_t g = 32U, x = r;
            while (x != 0U) { const uint32_t t2 = g % x; g = x; x = t2; }
            const bool baseAligned = (reinterpret_cast<uint64_t>(input) % 32U) == 0U;
            if (r != 0U && baseAligned &&
                (32U / g) * midStride_ * sizeof(T) <= 0xFFFFFFFFULL) {
                laneL_ = 32U / g;
            }
        }
        const uint32_t grp = laneL_ > 8U ? laneL_ : 8U;
        outPitch_ = AlignUpBlock(static_cast<uint32_t>(mid_ * run_ * sizeof(T)));
        pipe_.InitBuffer(raw0_, mid_ * rawPitch_); pipe_.InitBuffer(raw1_, mid_ * rawPitch_);
        pipe_.InitBuffer(packed0_, batch_ * outPitch_); pipe_.InitBuffer(packed1_, batch_ * outPitch_);
        pipe_.InitBuffer(offsets_, batch_ * outPitch_ / sizeof(T) * sizeof(int32_t));
        LocalTensor<int32_t> off = offsets_.template Get<int32_t>();
        const uint32_t firstRows = mid_ < grp ? mid_ : grp;
        for (uint32_t m = 0; m < firstRows; ++m) {
            const uint32_t ph = laneL_ > 1U ? static_cast<uint32_t>((m * midStride_ * sizeof(T)) % 32U) : 0U;
            for (uint32_t r = 0; r < run_; ++r) off.SetValue(m * run_ + r, static_cast<int32_t>(m * rawPitch_ + r * sizeof(T) + ph));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID3); WaitFlag<HardEvent::S_V>(EVENT_ID3);
        for (uint32_t k = 0; k < batch_; ++k) {
            for (uint32_t m = 0; m < mid_; m += grp) {
                if (k == 0U && m == 0U) continue;
                const uint32_t rows = (mid_ - m) < grp ? mid_ - m : grp;
                Adds(off[k * outPitch_ / sizeof(T) + m * run_], off,
                    static_cast<int32_t>(m * rawPitch_ + k * run_ * sizeof(T)), static_cast<int32_t>(rows * run_));
            }
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Meta(uint64_t task, RunMeta& m) const
    {
        const uint64_t midStart = (task % midGroups_) * mid_;
        task /= midGroups_;
        const uint64_t axisStart = (task % axisGroups_) * batch_;
        task /= axisGroups_;
        uint64_t inBase = 0U, outBase = 0U;
        for (int32_t i = static_cast<int32_t>(outerCount_) - 1; i >= 0; --i) {
            if (static_cast<uint32_t>(i) == axis_) continue;
            const uint64_t digit = task % lens_[i]; task /= lens_[i];
            inBase += digit * istrides_[i]; outBase += digit * ostrides_[i];
        }
        m.inOffset = inBase + axisStart * istrides_[axis_] + midStart * midStride_;
        m.outOffset = outBase + axisStart * ostrides_[axis_] + midStart * run_;
        m.rows = static_cast<uint32_t>((midLen_ - midStart) < mid_ ? midLen_ - midStart : mid_);
        m.batch = static_cast<uint32_t>((lens_[axis_] - axisStart) < batch_ ? lens_[axis_] - axisStart : batch_);
        m.count = static_cast<uint32_t>(run_);
    }
    __aicore__ inline void Load(const RunMeta& m, uint32_t p)
    {
        LocalTensor<T> raw = p == 0U ? raw0_.template Get<T>() : raw1_.template Get<T>();
        const uint32_t bytes = static_cast<uint32_t>(m.batch * run_ * sizeof(T));
        if (laneL_ > 1U) {
            const DataCopyPadExtParams<T> lpad;
            const uint32_t st = static_cast<uint32_t>((m.inOffset * sizeof(T)) % 32U);
            const uint32_t laneStride = static_cast<uint32_t>(laneL_ * midStride_ * sizeof(T));
            for (uint32_t j = 0U; j < laneL_ && j < m.rows; ++j) {
                const uint32_t blocks = (m.rows - j + laneL_ - 1U) / laneL_;
                const uint32_t sh = st + static_cast<uint32_t>((j * midStride_ * sizeof(T)) % 32U);

                const uint32_t len = AlignUpBlock(bytes + sh);
                const uint64_t firstEl = m.inOffset + j * midStride_ - sh / sizeof(T);
                const uint64_t lastEl = firstEl + static_cast<uint64_t>(blocks - 1U) * laneL_ * midStride_;
                const bool exactTail = lastEl + len / sizeof(T) > inTotal_;
                const uint32_t regular = exactTail ? blocks - 1U : blocks;
                if (regular != 0U) {
                    const DataCopyExtParams lane(regular, len, laneStride - len,
                        (laneL_ * rawPitch_ - len) / 32U, 0U);
                    DataCopyPad(raw[j * rawPitch_ / sizeof(T)], in_[firstEl], lane, lpad);
                }
                if (exactTail) {
                    const DataCopyExtParams tail(1U, bytes + sh, 0U, 0U, 0U);
                    DataCopyPad(raw[(j + (blocks - 1U) * laneL_) * rawPitch_ / sizeof(T)], in_[lastEl], tail, lpad);
                }
            }
            return;
        }
        const DataCopyExtParams load(m.rows, bytes, midStride_ * sizeof(T) - bytes,
            (rawPitch_ - AlignUpBlock(bytes)) / 32U, 0U);
        const DataCopyPadExtParams<T> pad; DataCopyPad(raw, in_[m.inOffset], load, pad);
    }
    __aicore__ inline void Process()
    {
        const uint64_t blocks = GetBlockNum(), block = GetBlockIdx();
        const uint64_t begin = tasks_ * block / blocks, end = tasks_ * (block + 1U) / blocks;
        if (begin >= end) return;
        LocalTensor<uint32_t> off = offsets_.template Get<int32_t>().template ReinterpretCast<uint32_t>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0); SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
        RunMeta meta[2]; Meta(begin, meta[0]);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); Load(meta[0], 0U); SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint64_t task = begin; task < end; ++task) {
            const uint32_t p = static_cast<uint32_t>((task - begin) & 1U); const event_t ep = static_cast<event_t>(p);
            if (task + 1U < end) {
                const uint32_t next = p ^ 1U; Meta(task + 1U, meta[next]);
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(next)); Load(meta[next], next);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(next));
            }
            WaitFlag<HardEvent::MTE2_V>(ep); WaitFlag<HardEvent::MTE3_V>(ep);
            LocalTensor<T> raw = p == 0U ? raw0_.template Get<T>() : raw1_.template Get<T>();
            LocalTensor<T> packed = p == 0U ? packed0_.template Get<T>() : packed1_.template Get<T>();
            const RunMeta& m = meta[p];
            for (uint32_t k = 0; k < m.batch; ++k) {
                Gather(packed[k * outPitch_ / sizeof(T)], raw, off[k * outPitch_ / sizeof(T)],
                    laneL_ > 1U ? static_cast<uint32_t>((m.inOffset * sizeof(T)) % 32U) : 0U,
                    static_cast<uint32_t>(m.rows * run_));
            }
            PipeBarrier<PIPE_V>(); SetFlag<HardEvent::V_MTE2>(ep);
            SetFlag<HardEvent::V_MTE3>(ep); WaitFlag<HardEvent::V_MTE3>(ep);
            const uint32_t bytes = static_cast<uint32_t>(m.rows * run_ * sizeof(T));
            const DataCopyExtParams store(m.batch, bytes, (outPitch_ - AlignUpBlock(bytes)) / 32U,
                ostrides_[axis_] * sizeof(T) - bytes, 0U);
            DataCopyPad(out_[m.outOffset], packed, store); SetFlag<HardEvent::MTE3_V>(ep);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }
private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> raw0_, raw1_, packed0_, packed1_, offsets_;
    GlobalTensor<T> in_, out_;
    uint64_t lens_[S9_TRANSPOSE_MAX_GROUP], istrides_[S9_TRANSPOSE_MAX_GROUP], ostrides_[S9_TRANSPOSE_MAX_GROUP];
    uint64_t run_, midLen_, midStride_, axisGroups_, midGroups_, tasks_;
    uint32_t axis_, batch_, mid_, rawPitch_, outPitch_, outerCount_;
    uint32_t laneL_ = 1U;
    uint64_t inTotal_ = 0U;
};

template <typename T>
// 当完整转置能够放入 UB 时，执行整块搬入、转置和搬出。
class KernelWholeTranspose {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const TransposeTilingData& tiling)
    {
        in_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputs));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        tileA_ = static_cast<uint32_t>(tiling.tileA);
        tileB_ = static_cast<uint32_t>(tiling.tileB);
        const uint32_t elements = tileA_ * tileB_;
        LocalTensor<T> src(TPosition::VECCALC, 0, elements);
        LocalTensor<T> dst(TPosition::VECCALC, elements * sizeof(T), elements);
        const uint64_t srcBase = src.GetPhyAddr();
        const uint64_t dstBase = dst.GetPhyAddr();
        const uint32_t hb = tileB_ / 16U;
        for (uint32_t i = 0; i < hb; ++i) {
            for (uint32_t n = 0; n < 16U; ++n) {
                const uint32_t offset = i * tileA_ * 16U + tileA_ * n;
                srcAddr_[i * 16U + n] = srcBase + offset * sizeof(T);
            }
            for (uint32_t m = 0; m < 16U; ++m) {
                const uint32_t offset = i * 16U + tileB_ * m;
                dstAddr_[i * 16U + m] = dstBase + offset * sizeof(T);
            }
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t elements = tileA_ * tileB_;
        LocalTensor<T> src(TPosition::VECCALC, 0, elements);
        LocalTensor<T> dst(TPosition::VECCALC, elements * sizeof(T), elements);
        DataCopy(src, in_, elements);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        TransDataTo5HDParams params;
        params.repeatTimes = static_cast<uint8_t>(tileA_ / 16U);
        params.dstRepStride = params.repeatTimes > 1 ? static_cast<uint16_t>(tileB_) : 0;
        params.srcRepStride = params.repeatTimes > 1 ? 1 : 0;
        const uint32_t hb = tileB_ / 16U;
        for (uint32_t i = 0; i < hb; ++i) {
            TransDataTo5HD<T>(*reinterpret_cast<uint64_t(*)[16]>(&dstAddr_[i * 16U]),
                              *reinterpret_cast<uint64_t(*)[16]>(&srcAddr_[i * 16U]), params);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(out_, dst, elements);
    }

private:
    GlobalTensor<T> in_;
    GlobalTensor<T> out_;
    uint64_t srcAddr_[256];
    uint64_t dstAddr_[256];
    uint32_t tileA_;
    uint32_t tileB_;
};

template <typename T, uint32_t kF32BulkMode = 0U>
// 执行二维 tile 转置，通过硬件转置或 Gather 重排数据，并合并输出写回。
class KernelTranspose {
public:
    using G = typename GatherTraits<T>::Type;
    static constexpr bool kIsInt8 = SameType<T, int8_t>::value;
    static constexpr bool kIsInt32 = SameType<T, int32_t>::value;
    static constexpr bool kIsFloat = SameType<T, float>::value;
    static constexpr bool kIsB32 = (sizeof(T) == 4);

    static constexpr uint32_t kTileCap = (sizeof(T) == 2) ? 128U : 64U;
    static constexpr bool kHwTranspose = (sizeof(T) == 2) || kIsFloat || kIsInt8;

    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const TransposeTilingData& tiling)
    {
        in_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(inputs));
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        if (tiling.bypassOutput != 0U) {

            out_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        for (uint32_t i = 0; i < S9_TRANSPOSE_MAX_GROUP; ++i) {
            outerLen_[i] = tiling.outerLen[i];
            outerInStride_[i] = tiling.outerInStride[i];
            outerOutStride_[i] = tiling.outerOutStride[i];
        }
        outerCount_ = tiling.outerCount;
        taskCount_ = tiling.taskCount;
        storeGroup_ = tiling.storeGroup;
        fullRowStore_ = tiling.fullRowStore;
        wholeTile_ = tiling.wholeTile;
        aFastShift_ = tiling.aFastShift;
        regime_ = tiling.regime;
        runLen_ = tiling.runLen;
        runChunk_ = tiling.runChunk;
        runChunkCount_ = tiling.runChunkCount;
        midLen_ = tiling.midLen;
        midInStride_ = tiling.midInStride;
        midBatch_ = tiling.midBatch;
        midChunkCount_ = tiling.midChunkCount;
        lenA_ = tiling.lenA;
        lenB_ = tiling.lenB;
        shiftRead_ = 0U;
        if constexpr (kIsFloat && kF32BulkMode == 2U) {
            if (tiling.storeGroup == 64U && tiling.tileB == 16U &&
                tiling.tileA == 32U &&
                ((tiling.inStrideB * sizeof(T)) % kUbBlockBytes) != 0U) {
                shiftRead_ = 1U;
            }
        }
        {
            const uint32_t g = (kF32BulkMode == 2U) ? 64U : 16U;
            const uint32_t full = g * static_cast<uint32_t>(tiling.tileB);
            const uint32_t need =
                ((static_cast<uint32_t>(tiling.lenB) + 15U) / 16U) * 16U;
            wideRowLen_ = (tiling.tileCountB <= g && need < full) ? need : full;
        }
        tileA_ = static_cast<uint32_t>(tiling.tileA);
        tileB_ = static_cast<uint32_t>(tiling.tileB);
        tileCountA_ = tiling.tileCountA;
        tileCountB_ = tiling.tileCountB;
        inStrideB_ = tiling.inStrideB;
        outStrideA_ = tiling.outStrideA;

        if constexpr (kIsFloat && kF32BulkMode == 2U) {
            if (storeGroup_ == 64U && tileA_ == 32U && tileB_ == 16U) {

                const uint32_t outputBytes = tileA_ * wideRowLen_ * sizeof(T);
                const uint32_t tableBytes = shiftRead_ != 0U ? 8192U : 0U;
                const uint32_t bytesPerSlot = 2U * 16U * sizeof(T) *
                    (shiftRead_ != 0U ? 32U + kShiftRawPitch : 32U);
                uint32_t loadGroup = (180U * 1024U - outputBytes - tableBytes - (CanPackF32Whole() ? AlignUpBlock((tileA_ * wideRowLen_ + 7U) / 8U) : 0U)) / bytesPerSlot;
                const uint32_t maximum = shiftRead_ != 0U ? 8U : 16U;
                if (loadGroup > maximum) loadGroup = maximum;
                runChunk_ = loadGroup;
            }
        }

        outerTotalK_ = 1U;
        for (uint32_t i = 0U; i < outerCount_; ++i) {
            outerTotalK_ *= outerLen_[i];
        }
        outerFast_ = 0U;
        if (outerCount_ >= 1U && outerTotalK_ > 1U &&
            (static_cast<uint32_t>(lenB_) * sizeof(T)) % kUbBlockBytes != 0U &&
            outStrideA_ != lenB_ &&
            outerOutStride_[outerCount_ - 1U] == lenB_) {
            outerFast_ = 1U;
        }

        if (regime_ == S9_TRANSPOSE_REGIME_RUN) {
            pipe_.InitBuffer(runBuf_, kRunTileElements * sizeof(T));
            return;
        }
        if constexpr (kIsInt32 && kF32BulkMode == 0U) {
            if (CanCollapsedFullOutput()) {
                tileA_ = 32U; tileB_ = 128U;
                tileCountA_ = (lenA_ + tileA_ - 1U) / tileA_;
                tileCountB_ = 1U; taskCount_ = tileCountA_;
                wideRowLen_ = (static_cast<uint32_t>(outStrideA_) + 15U) / 16U * 16U;
                if ((wideRowLen_ / 8U) % 4U == 0U) wideRowLen_ += 16U;
                shiftRead_ = 12U;
                laneL_ = 1U;
                {
                    const uint32_t r = static_cast<uint32_t>((inStrideB_ * sizeof(T)) % kUbBlockBytes);
                    uint32_t g = kUbBlockBytes, x = r;
                    while (x != 0U) { const uint32_t t2 = g % x; g = x; x = t2; }
                    const bool baseAligned = (reinterpret_cast<uint64_t>(inputs) % kUbBlockBytes) == 0U;
                    if (r != 0U && baseAligned && inStrideB_ * sizeof(T) * 8U <= 0xFFFFFFFFULL) {
                        laneL_ = kUbBlockBytes / g;
                    }
                }
                pipe_.InitBuffer(srcTileA_, 128U * 40U * sizeof(T));
                pipe_.InitBuffer(srcTileB_, 128U * 40U * sizeof(T));
                pipe_.InitBuffer(dstWide_, 32U * wideRowLen_ * sizeof(T));
                if (outStrideA_ < wideRowLen_) {
                    pipe_.InitBuffer(packOffBuf_, AlignUpBlock((32U * wideRowLen_ + 7U) / 8U));
                    BuildCollapsedStoreMask();
                }
                return;
            }
            if (CanCollapseOutput()) {
                tileA_ = 32U; tileB_ = 256U;
                tileCountA_ = (lenA_ + tileA_ - 1U) / tileA_;
                tileCountB_ = (outStrideA_ + tileB_ - 1U) / tileB_;
                taskCount_ = tileCountA_ * tileCountB_;
                const uint32_t bytes = tileA_ * tileB_ * sizeof(T);
                pipe_.InitBuffer(srcTileA_, bytes); pipe_.InitBuffer(srcTileB_, bytes);
                pipe_.InitBuffer(dstTileA_, bytes); pipe_.InitBuffer(dstTileB_, bytes);
                return;
            }
        }
        if constexpr (kIsFloat && kF32BulkMode == 2U) {
            if (CanPhaseF32()) {
                wideRowLen_ = ((static_cast<uint32_t>(lenB_) + 127U) / 128U) * 128U;
                shiftRead_ = 4U;
                pipe_.InitBuffer(srcTileA_, 128U * 40U * sizeof(T));
                pipe_.InitBuffer(srcTileB_, 128U * 40U * sizeof(T));
                pipe_.InitBuffer(dstTileA_, 128U * 40U * sizeof(T));
                pipe_.InitBuffer(dstWide_, tileA_ * wideRowLen_ * sizeof(T));
                pipe_.InitBuffer(packOffBuf_, AlignUpBlock((tileA_ * wideRowLen_ + 7U) / 8U));
                BuildF32StoreMask();
                return;
            }
        }

        if constexpr (kHwTranspose) {

            if (wholeTile_ != 0U) {
                const uint32_t tileBytes = tileA_ * tileB_ * sizeof(T);
                pipe_.InitBuffer(srcTileA_, tileBytes);
                pipe_.InitBuffer(dstTileA_, tileBytes);
            } else {
                const bool f32Bulk16 = kF32BulkMode == 2U && kIsFloat &&
                    storeGroup_ == 64U && tileA_ == 32U && tileB_ == 16U;
                const uint32_t tileBytes = (kIsInt8 || fullRowStore_ == 2U) ?
                    tileA_ * tileB_ * sizeof(T) :
                    (f32Bulk16 ?
                     (static_cast<uint32_t>(runChunk_) * 16U *
                      (shiftRead_ != 0U ? 32U + kShiftRawPitch : 32U) * sizeof(T)) :
                     kTileCap * kTileCap * sizeof(T));
                pipe_.InitBuffer(srcTileA_, tileBytes);
                pipe_.InitBuffer(srcTileB_, tileBytes);
            }
            if (wholeTile_ != 0U) {

            } else if (storeGroup_ > 1U) {

                const uint32_t wideBytes = fullRowStore_ != 0U ?
                    static_cast<uint32_t>(tileA_ * WideRowLen() * sizeof(T)) :
                    (kIsInt8 ?
                    tileA_ * storeGroup_ * tileB_ * sizeof(T) :
                    ((kF32BulkMode != 0U && kIsFloat) ?
                     tileA_ * wideRowLen_ * sizeof(T) :
                     kTileCap * storeGroup_ * kTileCap * sizeof(T)));
                pipe_.InitBuffer(dstWide_, wideBytes);
                if (shiftRead_ != 0U) {
                    pipe_.InitBuffer(offsetBuf_, 8U * 256U * sizeof(int32_t));
                    BuildShiftOffsets();
                }
                if constexpr (kIsFloat && kF32BulkMode == 2U) {
                    if (CanPackF32Whole()) {
                        pipe_.InitBuffer(packOffBuf_, AlignUpBlock((tileA_ * wideRowLen_ + 7U) / 8U));
                        BuildF32StoreMask();
                    }
                }
                if (fullRowStore_ == 2U) {
                    BuildPackOffsets();
                }
            } else {
                const uint32_t tileBytes = kIsInt8 ? tileA_ * tileB_ * sizeof(T) :
                    kTileCap * kTileCap * sizeof(T);
                pipe_.InitBuffer(dstTileA_, tileBytes);
                pipe_.InitBuffer(dstTileB_, tileBytes);
            }
            PrecomputeHwAddrs();
        } else {
            if constexpr (kIsInt32) {
                if (storeGroup_ > 1U) {

                    const uint32_t wideElements = fullRowStore_ != 0U ?
                        static_cast<uint32_t>(tileA_ * lenB_) :
                        kTileCap * storeGroup_ * kTileCap;
                    pipe_.InitBuffer(srcWideBuf_, wideElements * sizeof(T));
                    pipe_.InitBuffer(srcWideBufB_, wideElements * sizeof(T));
                    pipe_.InitBuffer(dstWideBuf_, wideElements * sizeof(T));
                    pipe_.InitBuffer(offsetBuf_, wideElements * sizeof(uint32_t));
                } else {
                    pipe_.InitBuffer(
                        srcQue_, kTileBufferDepth,
                        kTileCap * kTileCap * sizeof(T));
                    pipe_.InitBuffer(
                        dstQue_, kTileBufferDepth,
                        kTileCap * kTileCap * sizeof(T));
                    pipe_.InitBuffer(
                        offsetBuf_, kTileCap * kTileCap * sizeof(uint32_t));
                }
            } else {
                pipe_.InitBuffer(
                    srcQue_, kTileBufferDepth,
                    kTileCap * kTileCap * sizeof(T));
                pipe_.InitBuffer(
                    dstQue_, kTileBufferDepth,
                    kTileCap * kTileCap * sizeof(T));
                pipe_.InitBuffer(
                    offsetBuf_, kTileCap * kTileCap * sizeof(uint32_t));
            }
        }
        if constexpr (kIsInt8 && !kHwTranspose) {
            pipe_.InitBuffer(srcWideBuf_, kTileCap * kTileCap * sizeof(G));
            pipe_.InitBuffer(dstWideBuf_, kTileCap * kTileCap * sizeof(G));
        }
    }

    __aicore__ inline void Process()
    {
        if (taskCount_ == 0) {
            return;
        }
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());

        const uint64_t units = taskCount_;
        const uint64_t share = units / blocks;
        const uint64_t extra = units % blocks;
        const uint64_t ub = block * share + (block < extra ? block : extra);
        const uint64_t ue = ub + share + (block < extra ? 1 : 0);
        const uint64_t begin = ub;
        const uint64_t end = ue;
        if (begin >= end) {
            return;
        }
        if constexpr (kIsInt32 && kF32BulkMode == 0U) {
            if (shiftRead_ == 12U) { ProcessCollapsedFullOutput(begin, end); return; }
            if (CanCollapseOutput()) { ProcessCollapsedOutput(begin, end); return; }
        }
        if constexpr (kIsFloat && kF32BulkMode == 2U) {
            if (shiftRead_ == 4U) { ProcessPhaseF32(block, taskCount_, blocks); return; }
        }
        if (regime_ == S9_TRANSPOSE_REGIME_RUN) {
            for (uint64_t task = begin; task < end; ++task) {
                RunTaskCopy(task);
            }
        } else if constexpr (kHwTranspose) {
            if (wholeTile_ != 0U) {
                ProcessWholeTileHw();
            } else if (storeGroup_ > 1U) {
                ProcessTileHwGrouped(block, taskCount_, blocks);
            } else {
                ProcessTileHw(begin, end);
            }
        } else {
            if constexpr (kIsInt32) {
                if (storeGroup_ > 1U) {
                    ProcessTileGatherGrouped(block, taskCount_, blocks);
                    return;
                }
            }
            BuildGatherOffsets();
            for (uint64_t task = begin; task < end; ++task) {
                TileMeta meta;
                ComputeTileMeta(task, meta);
                CopyInTile(meta);
                ComputeTile(meta);
                CopyOutTile(meta);
            }
        }
    }

private:

    __aicore__ inline bool CanCollapsedFullOutput() const
    {
        if constexpr (kIsInt32 && kF32BulkMode == 0U) {
            return CanCollapseOutput() && outStrideA_ > 256U && outStrideA_ <= 1024U;
        }
        return false;
    }

    __aicore__ inline void IssueCollapsedFullLoad(bool lanes, uint32_t ping, uint64_t a,
        uint32_t startB, uint32_t rows, uint32_t countA)
    {
        LocalTensor<T> src = ping == 0U ? srcTileA_.template Get<T>() : srcTileB_.template Get<T>();
        const uint32_t dstStride = (40U * sizeof(T) - AlignUpBlock(countA * sizeof(T))) / kUbBlockBytes;
        const DataCopyPadExtParams<T> pad;
        uint32_t filled = 0U;
        while (filled < rows) {
            const uint64_t column = startB + filled;
            const uint64_t outer = column / lenB_;
            const uint64_t inner = column % lenB_;
            const uint32_t count = static_cast<uint32_t>((lenB_ - inner) < (rows - filled) ? lenB_ - inner : rows - filled);
            uint64_t inBase = 0U, unusedOut = 0U;
            OuterOffsets(outer, inBase, unusedOut);
            if (lanes) {
                const uint32_t laneStride = laneL_ * static_cast<uint32_t>(inStrideB_ * sizeof(T));
                for (uint32_t j = 0U; j < laneL_ && j < count; ++j) {
                    const uint32_t blocks = (count - j + laneL_ - 1U) / laneL_;
                    const uint64_t first = inBase + (inner + j) * inStrideB_ + a;
                    const uint32_t ph = static_cast<uint32_t>((first * sizeof(T)) % kUbBlockBytes);
                    const uint32_t len = ph + countA * sizeof(T);
                    const DataCopyExtParams load(blocks, len, laneStride - len,
                        (laneL_ * 40U * sizeof(T) - AlignUpBlock(len)) / kUbBlockBytes, 0U);
                    DataCopyPad(src[(filled + j) * 40U], in_[first - ph / sizeof(T)], load, pad);
                }
                filled += count;
                continue;
            }
            const DataCopyExtParams load(count, countA * sizeof(T),
                (inStrideB_ - countA) * sizeof(T), dstStride, 0U);
            DataCopyPad(src[filled * 40U], in_[inBase + inner * inStrideB_ + a], load, pad);
            filled += count;
        }
    }

    __aicore__ inline void ProcessCollapsedFullOutput(uint64_t begin, uint64_t end)
    {
        if constexpr (kIsInt32 && kF32BulkMode == 0U) {
            if (begin >= end) return;
            LocalTensor<T> dst = dstWide_.template Get<T>();
            uint64_t srcAddr[2][8][16], dstAddr[64][16];
            for (uint32_t p = 0U; p < 2U; ++p) {
                LocalTensor<T> src = p == 0U ? srcTileA_.template Get<T>() : srcTileB_.template Get<T>();
                for (uint32_t h = 0U; h < 8U; ++h) {
                    for (uint32_t n = 0U; n < 16U; ++n) {
                        srcAddr[p][h][n] = reinterpret_cast<uint64_t>(src[(h * 16U + n) * 40U].GetPhyAddr());
                    }
                }
            }
            const uint32_t fullB = static_cast<uint32_t>(outStrideA_);
            for (uint32_t h = 0U; h < (fullB + 15U) / 16U; ++h) {
                for (uint32_t n = 0U; n < 16U; ++n) {
                    dstAddr[h][n] = reinterpret_cast<uint64_t>(dst[h * 16U + (n / 2U) * wideRowLen_ + (n % 2U) * 8U].GetPhyAddr());
                }
            }

            uint8_t lanePh[8][128];
            uint64_t laneRow[16];
            constexpr bool lanes = false;
            if (lanes) {
                for (uint32_t column = 0U; column < 1024U; ++column) {
                    uint64_t ph = 0U;
                    if (column < fullB) {
                        uint64_t inBase = 0U, unusedOut = 0U;
                        OuterOffsets(column / lenB_, inBase, unusedOut);
                        ph = ((inBase + (column % lenB_) * inStrideB_) * sizeof(T)) % kUbBlockBytes;
                    }
                    lanePh[column / 128U][column % 128U] = static_cast<uint8_t>(ph);
                }
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            const uint32_t chunks = (fullB + 127U) / 128U;
            for (uint64_t task = begin; task < end; ++task) {
                const uint64_t a = task * tileA_;
                const uint32_t countA = static_cast<uint32_t>((lenA_ - a) < tileA_ ? lenA_ - a : tileA_);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                IssueCollapsedFullLoad(lanes, 0U, a, 0U, fullB < 128U ? fullB : 128U, countA);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                TransDataTo5HDParams tp;
                tp.repeatTimes = static_cast<uint8_t>((countA + 7U) / 8U);
                tp.srcRepStride = tp.repeatTimes == 1U ? 0U : 1U;
                tp.dstRepStride = tp.repeatTimes == 1U ? 0U : static_cast<uint16_t>(wideRowLen_);
                for (uint32_t chunk = 0U; chunk < chunks; ++chunk) {
                    const uint32_t p = chunk & 1U, startB = chunk * 128U;
                    const uint32_t rows = (fullB - startB) < 128U ? fullB - startB : 128U;
                    const event_t event = static_cast<event_t>(p);
                    if (chunk + 1U < chunks) {
                        const uint32_t next = p ^ 1U, nextB = startB + 128U;
                        const uint32_t nextRows = (fullB - nextB) < 128U ? fullB - nextB : 128U;
                        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(next));
                        IssueCollapsedFullLoad(lanes, next, a, nextB, nextRows, countA);
                        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(next));
                    }
                    WaitFlag<HardEvent::MTE2_V>(event);
                    for (uint32_t h = 0U; h < (rows + 15U) / 16U; ++h) {

                        if (lanes) {
                            for (uint32_t n = 0U; n < 16U; ++n) {
                                laneRow[n] = srcAddr[p][h][n] + lanePh[chunk][h * 16U + n];
                            }
                            HwTranspose(dstAddr[startB / 16U + h], laneRow, tp);
                        } else {
                            HwTranspose(dstAddr[startB / 16U + h], srcAddr[p][h], tp);
                        }

                    }
                    PipeBarrier<PIPE_V>(); SetFlag<HardEvent::V_MTE2>(event);
                }
                if (fullB < wideRowLen_) {
                    uint64_t retained = 0U;
                    GatherMask(dst, dst, packOffBuf_.template Get<int32_t>().template ReinterpretCast<uint32_t>(),
                        true, countA * wideRowLen_, GatherMaskParams(1, 1, 8, 0), retained);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2); WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                const DataCopyExtParams store(1U, countA * fullB * sizeof(T), 0U, 0U, 0U);
                DataCopyPad(out_[a * outStrideA_], dst, store);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
    }

    __aicore__ inline void BuildCollapsedStoreMask()
    {
        LocalTensor<int32_t> mask = packOffBuf_.template Get<int32_t>();
        uint32_t a = wideRowLen_, b = 256U;
        while (b != 0U) { const uint32_t r = a % b; a = b; b = r; }
        const uint32_t periodRows = 256U / a;
        const uint32_t words = periodRows * wideRowLen_ / 32U;

        if (wideRowLen_ >= 32U) {
            const uint32_t valid = static_cast<uint32_t>(outStrideA_);
            uint32_t column = 0U;
            for (uint32_t w = 0U; w < words; ++w) {
                uint32_t first = column < valid ? valid - column : 0U;
                if (first > 32U) first = 32U;
                uint32_t bits = first == 32U ? 0xffffffffU : ((1U << first) - 1U);
                const uint32_t untilNextRow = wideRowLen_ - column;
                if (untilNextRow < 32U) {
                    uint32_t second = 32U - untilNextRow;
                    if (second > valid) second = valid;
                    bits |= ((1U << second) - 1U) << untilNextRow;
                }
                mask.SetValue(w, static_cast<int32_t>(bits));
                column += 32U;
                if (column >= wideRowLen_) column -= wideRowLen_;
            }
        } else {

            for (uint32_t w = 0U; w < words; ++w) {
                uint32_t bits = 0U;
                for (uint32_t bit = 0U; bit < 32U; ++bit) {
                    if ((w * 32U + bit) % wideRowLen_ < outStrideA_) bits |= 1U << bit;
                }
                mask.SetValue(w, static_cast<int32_t>(bits));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID2); WaitFlag<HardEvent::S_V>(EVENT_ID2);
        for (uint32_t row = periodRows; row < tileA_; row += periodRows) {
            Adds(mask[row * wideRowLen_ / 32U], mask, static_cast<int32_t>(0), static_cast<int32_t>(words));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool CanCollapseOutput() const
    {
        if constexpr (!(kIsInt32 && kF32BulkMode == 0U)) {
            return false;
        } else {
            if (regime_ != S9_TRANSPOSE_REGIME_TILE || wholeTile_ != 0U ||
                storeGroup_ > 1U || fullRowStore_ != 0U || outerCount_ == 0U ||
                outerTotalK_ <= 1U || lenA_ < 64U || lenB_ <= 1U || lenB_ > 64U ||
                inStrideB_ != lenA_ || outStrideA_ != outerTotalK_ * lenB_ ||
                outStrideA_ > 0xFFFFFFFFULL / sizeof(T)) {
                return false;
            }
            uint64_t stride = lenB_;
            for (int32_t i = static_cast<int32_t>(outerCount_) - 1; i >= 0; --i) {
                if (outerOutStride_[i] != stride) return false;
                stride *= outerLen_[i];
            }
            return stride == outStrideA_;
        }
    }

    __aicore__ inline void CollapsedTileMeta(uint64_t task, TileMeta& m) const
    {
        const uint64_t a = (task / tileCountB_) * tileA_;
        const uint64_t b = (task % tileCountB_) * tileB_;
        m.inOffset = a;
        m.outOffset = a * outStrideA_ + b;
        m.countA = static_cast<uint32_t>((lenA_ - a) < tileA_ ? lenA_ - a : tileA_);
        m.countB = static_cast<uint32_t>((outStrideA_ - b) < tileB_ ? outStrideA_ - b : tileB_);
    }

    __aicore__ inline void IssueCollapsedLoad(const TileMeta& m, uint32_t ping)
    {
        LocalTensor<T> src = ping == 0U ? srcTileA_.template Get<T>() : srcTileB_.template Get<T>();
        const uint64_t first = m.outOffset % outStrideA_;
        const uint32_t dstStride = (tileA_ * sizeof(T) - AlignUpBlock(m.countA * sizeof(T))) / kUbBlockBytes;
        const DataCopyPadExtParams<T> pad;
        uint32_t filled = 0U;
        while (filled < m.countB) {
            const uint64_t column = first + filled;
            const uint64_t outer = column / lenB_;
            const uint64_t inner = column % lenB_;
            const uint32_t rows = static_cast<uint32_t>(
                (lenB_ - inner) < (m.countB - filled) ? lenB_ - inner : m.countB - filled);
            uint64_t inBase = 0U, outBase = 0U;
            OuterOffsets(outer, inBase, outBase);
            const DataCopyExtParams load(rows, m.countA * sizeof(T),
                (inStrideB_ - m.countA) * sizeof(T), dstStride, 0U);
            DataCopyPad(src[filled * tileA_], in_[inBase + inner * inStrideB_ + m.inOffset], load, pad);
            filled += rows;
        }
    }

    __aicore__ inline void ProcessCollapsedOutput(uint64_t begin, uint64_t end)
    {
        if constexpr (kIsInt32 && kF32BulkMode == 0U) {
            if (begin >= end) return;

            uint64_t srcAddr[2][256];
            uint64_t dstAddr[2][256];
            const uint32_t hb = tileB_ / 16U;
            for (uint32_t p = 0; p < 2U; ++p) {
                LocalTensor<T> src = p == 0U ? srcTileA_.template Get<T>() : srcTileB_.template Get<T>();
                LocalTensor<T> dst = p == 0U ? dstTileA_.template Get<T>() : dstTileB_.template Get<T>();
                for (uint32_t h = 0; h < hb; ++h) {
                    for (uint32_t n = 0; n < 16U; ++n) {
                        srcAddr[p][h * 16U + n] = reinterpret_cast<uint64_t>(src[h * 16U * tileA_ + n * tileA_].GetPhyAddr());
                        dstAddr[p][h * 16U + n] = reinterpret_cast<uint64_t>(dst[h * 16U + (n / 2U) * tileB_ + (n % 2U) * 8U].GetPhyAddr());
                    }
                }
            }
            TransDataTo5HDParams tp;
            tp.repeatTimes = static_cast<uint8_t>(tileA_ / 8U);
            tp.srcRepStride = 1U;
            tp.dstRepStride = static_cast<uint16_t>(tileB_);
            TileMeta meta[2];
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0); SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CollapsedTileMeta(begin, meta[0]); IssueCollapsedLoad(meta[0], 0U);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            const uint64_t count = end - begin;
            for (uint64_t k = 0; k < count; ++k) {
                const uint32_t p = static_cast<uint32_t>(k & 1U);
                const event_t ep = static_cast<event_t>(p);
                if (k + 1U < count) {
                    const uint32_t next = p ^ 1U;
                    WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(next));
                    CollapsedTileMeta(begin + k + 1U, meta[next]); IssueCollapsedLoad(meta[next], next);
                    SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(next));
                }
                WaitFlag<HardEvent::MTE2_V>(ep); WaitFlag<HardEvent::MTE3_V>(ep);
                for (uint32_t h = 0; h < hb; ++h) HwTranspose(dstAddr[p] + h * 16U, srcAddr[p] + h * 16U, tp);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(ep);
                SetFlag<HardEvent::V_MTE3>(ep); WaitFlag<HardEvent::V_MTE3>(ep);
                LocalTensor<T> dst = p == 0U ? dstTileA_.template Get<T>() : dstTileB_.template Get<T>();
                const TileMeta& m = meta[p];
                const uint32_t stride = (tileB_ * sizeof(T) - AlignUpBlock(m.countB * sizeof(T))) / kUbBlockBytes;
                const DataCopyExtParams store(m.countA, m.countB * sizeof(T), stride,
                    (outStrideA_ - m.countB) * sizeof(T), 0U);
                DataCopyPad(out_[m.outOffset], dst, store);
                SetFlag<HardEvent::MTE3_V>(ep);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
        }
    }

    __aicore__ inline bool CanPackF32Whole() const
    {
        if constexpr (kIsFloat && kF32BulkMode == 2U) {
            return storeGroup_ == 64U && tileA_ == 32U && tileB_ == 16U &&
                fullRowStore_ == 0U && tileCountB_ <= storeGroup_ &&
                outStrideA_ == lenB_ && lenB_ < wideRowLen_;
        }
        return false;
    }

    __aicore__ inline void BuildF32StoreMask()
    {
        LocalTensor<int32_t> mask = packOffBuf_.template Get<int32_t>();
        uint32_t a = wideRowLen_, b = 256U;
        while (b != 0U) { const uint32_t r = a % b; a = b; b = r; }
        const uint32_t periodRows = 256U / a;
        const uint32_t words = periodRows * wideRowLen_ / 32U;

        if (wideRowLen_ >= 32U) {
            const uint32_t valid = static_cast<uint32_t>(lenB_);
            uint32_t column = 0U;
            for (uint32_t w = 0U; w < words; ++w) {
                uint32_t first = column < valid ? valid - column : 0U;
                if (first > 32U) first = 32U;
                uint32_t bits = first == 32U ? 0xffffffffU : ((1U << first) - 1U);
                const uint32_t untilNextRow = wideRowLen_ - column;
                if (untilNextRow < 32U) {
                    uint32_t second = 32U - untilNextRow;
                    if (second > valid) second = valid;
                    bits |= ((1U << second) - 1U) << untilNextRow;
                }
                mask.SetValue(w, static_cast<int32_t>(bits));
                column += 32U;
                if (column >= wideRowLen_) column -= wideRowLen_;
            }
        } else {

            for (uint32_t w = 0U; w < words; ++w) {
                uint32_t bits = 0U;
                for (uint32_t bit = 0U; bit < 32U; ++bit) {
                    if ((w * 32U + bit) % wideRowLen_ < lenB_) bits |= 1U << bit;
                }
                mask.SetValue(w, static_cast<int32_t>(bits));
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID2); WaitFlag<HardEvent::S_V>(EVENT_ID2);
        for (uint32_t row = periodRows; row < tileA_; row += periodRows) {
            Adds(mask[row * wideRowLen_ / 32U], mask, static_cast<int32_t>(0), static_cast<int32_t>(words));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline bool CanPhaseF32() const
    {
        if constexpr (kIsFloat && kF32BulkMode == 2U) {
            if (regime_ != S9_TRANSPOSE_REGIME_TILE || wholeTile_ != 0U ||
                storeGroup_ != 64U || tileA_ != 32U || tileB_ != 16U ||
                fullRowStore_ != 0U || shiftRead_ != 1U ||
                tileCountB_ > storeGroup_ || lenA_ < 32U || lenB_ == 0U ||
                lenB_ > 1024U || inStrideB_ != lenA_ || outStrideA_ != lenB_ ||
                inStrideB_ > 0xffffffffULL / (8U * sizeof(T))) return false;
            const uint32_t row = ((static_cast<uint32_t>(lenB_) + 127U) / 128U) * 128U;
            const uint32_t maskBytes = true ? AlignUpBlock((32U * row + 7U) / 8U) : 0U;

            return 3U * 128U * 40U * sizeof(T) + 32U * row * sizeof(T) + maskBytes <= 180U * 1024U;
        }
        return false;
    }

    __aicore__ inline void ProcessPhaseF32(uint64_t begin, uint64_t end, uint64_t step)
    {
        if constexpr (kIsFloat && kF32BulkMode == 2U) {
            if (begin >= end) return;
            LocalTensor<T> raw0 = srcTileA_.template Get<T>();
            LocalTensor<T> raw1 = srcTileB_.template Get<T>();
            LocalTensor<T> phaseBuffer = dstTileA_.template Get<T>();
            LocalTensor<T> output = dstWide_.template Get<T>();
            uint64_t firstSrc[2][8][16], firstDst[8][16];
            uint64_t secondSrc[8][2][16], secondDst[8][2][16];
            for (uint32_t j = 0U; j < 8U; ++j) {
                for (uint32_t n = 0U; n < 16U; ++n) {
                    firstSrc[0][j][n] = reinterpret_cast<uint64_t>(raw0[(j * 16U + n) * 40U].GetPhyAddr());
                    firstSrc[1][j][n] = reinterpret_cast<uint64_t>(raw1[(j * 16U + n) * 40U].GetPhyAddr());
                    firstDst[j][n] = reinterpret_cast<uint64_t>(phaseBuffer[j * 40U * 16U + (n / 2U) * 16U + (n % 2U) * 8U].GetPhyAddr());
                }
            }
            for (uint32_t p = 0U; p < 8U; ++p) {
                for (uint32_t k = 0U; k < 2U; ++k) {
                    for (uint32_t n = 0U; n < 16U; ++n) {
                        const uint32_t j = n % 8U;
                        const uint32_t r = static_cast<uint32_t>((p + j * inStrideB_) % 8U);
                        secondSrc[p][k][n] = reinterpret_cast<uint64_t>(phaseBuffer[j * 40U * 16U + (r + n / 8U) * 16U + k * 8U].GetPhyAddr());
                    }
                }
            }
            const uint32_t chunks = (static_cast<uint32_t>(lenB_) + 127U) / 128U;
            for (uint32_t chunk = 0U; chunk < chunks; ++chunk) {
                for (uint32_t k = 0U; k < 2U; ++k) {
                    for (uint32_t n = 0U; n < 16U; ++n) {
                        secondDst[chunk][k][n] = reinterpret_cast<uint64_t>(output[(n % 2U) * wideRowLen_ + chunk * 128U + (k * 8U + n / 2U) * 8U].GetPhyAddr());
                    }
                }
            }
            TransDataTo5HDParams first;
            first.repeatTimes = 5U; first.srcRepStride = 1U; first.dstRepStride = 16U;
            TransDataTo5HDParams second;
            second.repeatTimes = 16U; second.srcRepStride = 4U;
            second.dstRepStride = static_cast<uint16_t>(wideRowLen_ / 4U);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0); SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            for (uint64_t task = begin; task < end; task += step) {
                const uint64_t outer = task / tileCountA_;
                const uint64_t startA = (task % tileCountA_) * tileA_;
                const uint32_t countA = static_cast<uint32_t>((lenA_ - startA) < tileA_ ? lenA_ - startA : tileA_);
                uint64_t inBase = 0U, outBase = 0U; OuterOffsets(outer, inBase, outBase);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                IssuePhaseF32Load(0U, inBase, startA, 0U, lenB_ < 128U ? static_cast<uint32_t>(lenB_) : 128U, countA);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                for (uint32_t chunk = 0U; chunk < chunks; ++chunk) {
                    const uint32_t ping = chunk & 1U;
                    const event_t event = static_cast<event_t>(ping);
                    if (chunk + 1U < chunks) {
                        const uint32_t next = ping ^ 1U, nextB = (chunk + 1U) * 128U;
                        const uint32_t nextRows = (lenB_ - nextB) < 128U ? static_cast<uint32_t>(lenB_ - nextB) : 128U;
                        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(next));
                        IssuePhaseF32Load(next, inBase, startA, nextB, nextRows, countA);
                        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(next));
                    }
                    WaitFlag<HardEvent::MTE2_V>(event);
                    for (uint32_t j = 0U; j < 8U; ++j) HwTranspose(firstDst[j], firstSrc[ping][j], first);
                    PipeBarrier<PIPE_V>();

                    SetFlag<HardEvent::V_MTE2>(event);
                    const uint32_t p = static_cast<uint32_t>((inBase + startA + chunk * 128U * inStrideB_) % 8U);
                    for (uint32_t k = 0U; k < 2U; ++k) HwTranspose(secondDst[chunk][k], secondSrc[p][k], second);
                    PipeBarrier<PIPE_V>();
                }
                uint64_t retained = 0U;
                GatherMask(output, output, packOffBuf_.template Get<int32_t>().template ReinterpretCast<uint32_t>(),
                    true, countA * wideRowLen_, GatherMaskParams(1, 1, 8, 0), retained);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2); WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                const DataCopyExtParams store(1U, countA * static_cast<uint32_t>(lenB_) * sizeof(T), 0U, 0U, 0U);
                DataCopyPad(out_[outBase + startA * outStrideA_], output, store);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
    }

    __aicore__ inline void IssuePhaseF32Load(uint32_t ping, uint64_t inBase,
        uint64_t startA, uint64_t startB, uint32_t rows, uint32_t countA)
    {
        LocalTensor<T> raw = ping == 0U ? srcTileA_.template Get<T>() : srcTileB_.template Get<T>();
        const uint32_t pitch = tileA_ + 8U;
        const uint64_t base = inBase + startA + startB * inStrideB_;
        const uint64_t total = outerTotalK_ * lenA_ * lenB_;
        const DataCopyPadExtParams<T> pad;
        for (uint32_t j = 0; j < 8U && j < rows; ++j) {
            const uint32_t count = (rows - j + 7U) / 8U;
            const uint64_t first = base + j * inStrideB_;
            const uint32_t phase = static_cast<uint32_t>(first % 8U);
            const uint32_t bytes = AlignUpBlock((countA + phase) * sizeof(T));
            const uint64_t last = first + static_cast<uint64_t>(count - 1U) * 8U * inStrideB_;
            const bool exactTail = last - phase + bytes / sizeof(T) > total;
            const uint32_t regular = exactTail ? count - 1U : count;
            if (regular != 0U) {
                const DataCopyExtParams load(regular, bytes, 8U * inStrideB_ * sizeof(T) - bytes,
                    (pitch * sizeof(T) - bytes) / kUbBlockBytes, 0U);
                DataCopyPad(raw[j * 16U * pitch], in_[first - phase], load, pad);
            }
            if (exactTail) {
                const DataCopyExtParams tail(1U, (countA + phase) * sizeof(T), 0U, 0U, 0U);
                DataCopyPad(raw[(j * 16U + count - 1U) * pitch], in_[last - phase], tail, pad);
            }
        }
    }
    __aicore__ inline void OuterOffsets(uint64_t index, uint64_t& inOffset, uint64_t& outOffset) const
    {
        inOffset = 0;
        outOffset = 0;
        for (int32_t i = static_cast<int32_t>(outerCount_) - 1; i >= 0; --i) {
            const uint64_t len = outerLen_[i];
            const uint64_t digit = index % len;
            index /= len;
            inOffset += digit * outerInStride_[i];
            outOffset += digit * outerOutStride_[i];
        }
    }

    __aicore__ inline uint32_t WideRowLen() const
    {
        if (fullRowStore_ == 1U) {
            return static_cast<uint32_t>(lenB_);
        }
        if (fullRowStore_ == 2U) {
            const uint32_t gran = kIsInt8 ? 64U : 16U;
            return (static_cast<uint32_t>(lenB_) + gran - 1U) / gran * gran;
        }
        return tileB_ * storeGroup_;
    }

    __aicore__ inline uint32_t GatherDstPitch() const
    {
        return fullRowStore_ != 0U ? static_cast<uint32_t>(lenB_) : tileB_;
    }

    __aicore__ inline void BuildGatherOffsets()
    {
        const uint32_t pitch = GatherDstPitch();
        LocalTensor<int32_t> offsets = offsetBuf_.template Get<int32_t>();

        if ((pitch * sizeof(int32_t)) % kUbBlockBytes == 0U) {
            for (uint32_t b = 0; b < pitch; ++b) {
                offsets.SetValue(b, static_cast<int32_t>(b * tileA_ * sizeof(G)));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            for (uint32_t a = 1; a < tileA_; ++a) {
                Adds(offsets[a * pitch], offsets, static_cast<int32_t>(a * sizeof(G)),
                     static_cast<int32_t>(pitch));
            }
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t a = 0; a < tileA_; ++a) {
                for (uint32_t b = 0; b < pitch; ++b) {
                    offsets.SetValue(
                        a * pitch + b,
                        static_cast<int32_t>((b * tileA_ + a) * sizeof(G)));
                }
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void ComputeTileMeta(uint64_t task, TileMeta& meta) const
    {
        const uint64_t perOuter = tileCountA_ * tileCountB_;

        uint64_t outerIndex;
        uint64_t rest;
        if (outerFast_ != 0U) {
            outerIndex = task % outerTotalK_;
            rest = task / outerTotalK_;
        } else {
            outerIndex = task / perOuter;
            rest = task - outerIndex * perOuter;
        }
        uint64_t tileIndexA;
        uint64_t tileIndexB;
        if (aFastShift_ != 0U && storeGroup_ > 1U) {
            const uint32_t groupShift = storeGroup_ == 8U ? 3U :
                (storeGroup_ == 4U ? 2U : 1U);
            const uint64_t group = rest >> groupShift;
            const uint64_t slot = rest & (storeGroup_ - 1U);
            tileIndexA = group & (tileCountA_ - 1U);
            const uint64_t groupB = group >> aFastShift_;
            tileIndexB = (groupB << groupShift) + slot;
        } else {
            tileIndexA = rest / tileCountB_;
            tileIndexB = rest - tileIndexA * tileCountB_;
        }

        uint64_t inBase = 0;
        uint64_t outBase = 0;
        OuterOffsets(outerIndex, inBase, outBase);
        const uint64_t startA = tileIndexA * tileA_;
        meta.countA = static_cast<uint32_t>((lenA_ - startA) < tileA_ ? (lenA_ - startA) : tileA_);
        const uint64_t startB = tileIndexB * tileB_;
        meta.countB = static_cast<uint32_t>((lenB_ - startB) < tileB_ ? (lenB_ - startB) : tileB_);
        meta.inOffset = inBase + startA + startB * inStrideB_;
        meta.outOffset = outBase + startA * outStrideA_ + startB;
    }

    __aicore__ inline void CopyInTile(const TileMeta& meta)
    {
        LocalTensor<T> src = srcQue_.template AllocTensor<T>();
        const DataCopyPadExtParams<T> pad;

        const uint32_t loadStride =
            (tileA_ * sizeof(T) - AlignUpBlock(meta.countA * sizeof(T))) / kUbBlockBytes;
        const DataCopyExtParams load(
            meta.countB, meta.countA * sizeof(T), (inStrideB_ - meta.countA) * sizeof(T), loadStride, 0);
        DataCopyPad(src, in_[meta.inOffset], load, pad);
        srcQue_.EnQue(src);
    }

    __aicore__ inline void PrecomputeHwAddrs()
    {
        if constexpr (kHwTranspose) {
            if constexpr (kIsInt8) {
                PrecomputeB8Addrs();
                return;
            }
            if constexpr (kIsFloat && kF32BulkMode == 2U) {
                if (storeGroup_ == 64U && tileA_ == 32U && tileB_ == 16U) {
                    PrecomputeF32Bulk2Addrs();
                    return;
                }
            } else if constexpr (kIsFloat && kF32BulkMode == 1U) {
                if (outerCount_ == 1U && storeGroup_ == 16U &&
                    tileA_ == 128U && tileB_ == 16U) {
                    PrecomputeF32Bulk2Addrs();
                    return;
                }
            }
            LocalTensor<T> s0 = srcTileA_.template Get<T>();
            LocalTensor<T> d0 = (wholeTile_ != 0U || storeGroup_ <= 1U) ? dstTileA_.template Get<T>()
                                                   : dstWide_.template Get<T>();
            if (wholeTile_ != 0U) {
                const uint32_t hb = tileB_ / 16U;
                for (uint32_t i = 0; i < hb; ++i) {
                    for (uint32_t n = 0; n < 16U; ++n) {
                        const uint32_t so = i * tileA_ * 16U + tileA_ * n;
                        srcAddr0_[i * 16U + n] = reinterpret_cast<uint64_t>(s0[so].GetPhyAddr());
                    }
                    if constexpr (kIsB32) {
                        for (uint32_t m = 0; m < 8U; ++m) {
                            for (uint32_t k = 0; k < 2U; ++k) {
                                const uint32_t doff = i * 16U + tileB_ * m + k * 8U;
                                dstAddr0_[i * 16U + m * 2U + k] =
                                    reinterpret_cast<uint64_t>(d0[doff].GetPhyAddr());
                            }
                        }
                    } else {
                        for (uint32_t m = 0; m < 16U; ++m) {
                            const uint32_t doff = i * 16U + tileB_ * m;
                            dstAddr0_[i * 16U + m] = reinterpret_cast<uint64_t>(d0[doff].GetPhyAddr());
                        }
                    }
                }
                return;
            }
            LocalTensor<T> s1 = srcTileB_.template Get<T>();
            LocalTensor<T> d1 = (storeGroup_ > 1U) ? dstWide_.template Get<T>()
                                                   : dstTileA_.template Get<T>();
            if (storeGroup_ <= 1U) {
                d1 = dstTileB_.template Get<T>();
            }
            const uint32_t hb = tileB_ / 16U;
            for (uint32_t i = 0; i < hb; ++i) {
                for (uint32_t n = 0; n < 16U; ++n) {
                    const uint32_t so = i * tileA_ * 16U + tileA_ * n;
                    srcAddr0_[i * 16U + n] = reinterpret_cast<uint64_t>(s0[so].GetPhyAddr());
                    srcAddr1_[i * 16U + n] = reinterpret_cast<uint64_t>(s1[so].GetPhyAddr());
                }
                if constexpr (kIsB32) {
                    for (uint32_t m = 0; m < 8U; ++m) {
                        for (uint32_t k = 0; k < 2U; ++k) {
                            const uint32_t doff = i * 16U + tileB_ * m + k * 8U;
                            const uint32_t addr = i * 16U + m * 2U + k;
                            dstAddr0_[addr] = reinterpret_cast<uint64_t>(d0[doff].GetPhyAddr());
                            dstAddr1_[addr] = reinterpret_cast<uint64_t>(d1[doff].GetPhyAddr());
                        }
                    }
                } else {
                    for (uint32_t m = 0; m < 16U; ++m) {
                        const uint32_t doff = i * 16U + tileB_ * m;
                        dstAddr0_[i * 16U + m] = reinterpret_cast<uint64_t>(d0[doff].GetPhyAddr());
                        dstAddr1_[i * 16U + m] = reinterpret_cast<uint64_t>(d1[doff].GetPhyAddr());
                    }
                }
                if (storeGroup_ > 1U) {

                    LocalTensor<T> dw = dstWide_.template Get<T>();
                    const uint32_t rowLen = WideRowLen();
                    for (uint32_t slot = 0; slot < storeGroup_; ++slot) {
                        if constexpr (kIsB32) {
                            for (uint32_t m = 0; m < 8U; ++m) {
                                for (uint32_t k = 0; k < 2U; ++k) {
                                    const uint32_t base = i * 16U + rowLen * m + k * 8U + slot * tileB_;
                                    if constexpr (kIsFloat) {

                                        uint64_t (*dstF32Wide)[16] =
                                            reinterpret_cast<uint64_t (*)[16]>(dstAddrWide_);
                                        dstF32Wide[slot][i * 16U + m * 2U + k] =
                                            reinterpret_cast<uint64_t>(dw[base].GetPhyAddr());
                                    } else {
                                        dstAddrWide_[slot][i * 16U + m * 2U + k] =
                                            reinterpret_cast<uint64_t>(dw[base].GetPhyAddr());
                                    }
                                }
                            }
                        } else {
                            for (uint32_t m = 0; m < 16U; ++m) {
                                const uint32_t base = i * 16U + rowLen * m + slot * tileB_;
                                dstAddrWide_[slot][i * 16U + m] =
                                    reinterpret_cast<uint64_t>(dw[base].GetPhyAddr());
                            }
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void BuildPackOffsets()
    {
        const uint32_t lb = static_cast<uint32_t>(lenB_);
        const uint32_t rowLen = WideRowLen();
        const uint32_t exact = lb * static_cast<uint32_t>(sizeof(T));

        packUnit_ = (sizeof(T) == 1U || (exact % 4U == 0U)) ?
            4U : static_cast<uint32_t>(sizeof(T));
        packPerRow_ = exact / packUnit_;

        constexpr uint32_t L = 32U / static_cast<uint32_t>(sizeof(T));
        uint32_t g = L;
        uint32_t x = lb % L;
        while (x != 0U) { const uint32_t t = g % x; g = x; x = t; }
        uint32_t minRows = L / g;

        packRows_ = minRows;
        {

            const uint32_t cap = 32U * 1024U / sizeof(int32_t);
            while (packRows_ * 2U <= tileA_ &&
                   packRows_ * 2U * packPerRow_ <= cap) {
                packRows_ *= 2U;
            }
        }
        pipe_.InitBuffer(packBuf_, tileA_ * lb * sizeof(T));
        pipe_.InitBuffer(packOffBuf_, packRows_ * packPerRow_ * sizeof(int32_t));
        LocalTensor<int32_t> off = packOffBuf_.template Get<int32_t>();

        {
            const uint32_t N = packRows_ * packPerRow_;
            const int32_t A = static_cast<int32_t>(packUnit_);
            const int32_t B = static_cast<int32_t>(rowLen * sizeof(T)) -
                static_cast<int32_t>(packPerRow_ * packUnit_);

            const float invPer =
                1.0f / static_cast<float>(static_cast<int32_t>(packPerRow_));

            LocalTensor<float> f0 = dstWide_.template Get<float>();

            uint32_t kChunk = f0.GetSize() / 3U;
            if (kChunk > 2048U) { kChunk = 2048U; }
            kChunk = (kChunk / 64U) * 64U;
            LocalTensor<float> f1 = f0[kChunk];
            LocalTensor<float> f2 = f0[2U * kChunk];
            for (uint32_t base = 0; base < N; base += kChunk) {
                const int32_t c = static_cast<int32_t>(
                    (N - base) < kChunk ? (N - base) : kChunk);
                CreateVecIndex(f0, static_cast<float>(static_cast<int32_t>(base)), c);
                PipeBarrier<PIPE_V>();
                Muls(f1, f0, invPer, c);
                PipeBarrier<PIPE_V>();
                Adds(f1, f1, 1.0e-4f, c);
                PipeBarrier<PIPE_V>();
                Cast(f2, f1, RoundMode::CAST_FLOOR, c);
                PipeBarrier<PIPE_V>();
                Muls(f0, f0, static_cast<float>(A), c);
                PipeBarrier<PIPE_V>();
                Muls(f2, f2, static_cast<float>(B), c);
                PipeBarrier<PIPE_V>();
                Add(f0, f0, f2, c);
                PipeBarrier<PIPE_V>();
                Cast(off[base], f0, RoundMode::CAST_RINT, c);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void PrecomputeF32Bulk2Addrs()
    {
        if constexpr (kF32BulkMode == 2U) {

        if constexpr (kIsFloat && kF32BulkMode != 0U) {
            LocalTensor<T> src0 = srcTileA_.template Get<T>();
            LocalTensor<T> src1 = srcTileB_.template Get<T>();
            LocalTensor<T> dst = dstWide_.template Get<T>();
            uint64_t (*srcTable0)[16] =
                reinterpret_cast<uint64_t (*)[16]>(srcAddr0_);
            uint64_t (*srcTable1)[16] =
                reinterpret_cast<uint64_t (*)[16]>(dstAddr0_);
            uint64_t (*dstTable)[16] =
                reinterpret_cast<uint64_t (*)[16]>(dstAddrWide_);
            const uint32_t rowLen = wideRowLen_;

            const uint32_t ordinaryGroup = shiftRead_ != 0U ? 8U : 16U;
            const uint32_t sourceSlots = kF32BulkMode == 2U ?
                (runChunk_ < ordinaryGroup ? static_cast<uint32_t>(runChunk_) : 16U) : 2U;
            for (uint32_t slot = 0U; slot < sourceSlots; ++slot) {
                for (uint32_t lane = 0U; lane < 16U; ++lane) {
                    const uint32_t offset =
                        (slot * tileB_ + lane) * tileA_;
                    srcTable0[slot][lane] =
                        reinterpret_cast<uint64_t>(src0[offset].GetPhyAddr());
                    srcTable1[slot][lane] =
                        reinterpret_cast<uint64_t>(src1[offset].GetPhyAddr());
                }
            }
            for (uint32_t slot = 0U; slot < storeGroup_; ++slot) {
                for (uint32_t m = 0U; m < 8U; ++m) {
                    for (uint32_t k = 0U; k < 2U; ++k) {
                        const uint32_t index = m * 2U + k;
                        const uint32_t offset =
                            rowLen * m + k * 8U + slot * tileB_;
                        dstTable[slot][index] =
                            reinterpret_cast<uint64_t>(dst[offset].GetPhyAddr());
                    }
                }
            }
        }

        } else {

        if constexpr (kIsFloat && kF32BulkMode != 0U) {
            LocalTensor<T> src0 = srcTileA_.template Get<T>();
            LocalTensor<T> src1 = srcTileB_.template Get<T>();
            LocalTensor<T> dst = dstWide_.template Get<T>();
            uint64_t (*srcTable0)[16] =
                reinterpret_cast<uint64_t (*)[16]>(srcAddr0_);
            uint64_t (*srcTable1)[16] =
                reinterpret_cast<uint64_t (*)[16]>(dstAddr0_);
            uint64_t (*dstTable)[16] =
                reinterpret_cast<uint64_t (*)[16]>(dstAddrWide_);
            const uint32_t rowLen = wideRowLen_;
            constexpr uint32_t sourceSlots =
                kF32BulkMode == 2U ? 16U : 2U;
            for (uint32_t slot = 0U; slot < sourceSlots; ++slot) {
                for (uint32_t lane = 0U; lane < 16U; ++lane) {
                    const uint32_t offset =
                        (slot * tileB_ + lane) * tileA_;
                    srcTable0[slot][lane] =
                        reinterpret_cast<uint64_t>(src0[offset].GetPhyAddr());
                    srcTable1[slot][lane] =
                        reinterpret_cast<uint64_t>(src1[offset].GetPhyAddr());
                }
            }
            for (uint32_t slot = 0U; slot < storeGroup_; ++slot) {
                for (uint32_t m = 0U; m < 8U; ++m) {
                    for (uint32_t k = 0U; k < 2U; ++k) {
                        const uint32_t index = m * 2U + k;
                        const uint32_t offset =
                            rowLen * m + k * 8U + slot * tileB_;
                        dstTable[slot][index] =
                            reinterpret_cast<uint64_t>(dst[offset].GetPhyAddr());
                    }
                }
            }
        }

        }
    }

    __aicore__ inline void PrecomputeB8Addrs()
    {
        if constexpr (kIsInt8) {

            uint64_t (*srcB8)[2][2][16] =
                reinterpret_cast<uint64_t (*)[2][2][16]>(srcAddr0_);
            uint64_t (*dstB8)[2][2][16] =
                reinterpret_cast<uint64_t (*)[2][2][16]>(dstAddr0_);
            uint64_t (*dstB8Wide)[2][2][16] =
                reinterpret_cast<uint64_t (*)[2][2][16]>(dstAddrWide_);
            LocalTensor<T> s0 = srcTileA_.template Get<T>();
            LocalTensor<T> s1 = srcTileB_.template Get<T>();
            const uint64_t srcBase0 = s0.GetPhyAddr();
            const uint64_t srcBase1 = s1.GetPhyAddr();
            const uint32_t bBlocks = tileB_ / 32U;
            for (uint32_t b = 0; b < bBlocks; ++b) {
                for (uint32_t i = 0; i < 16U; ++i) {
                    const uint32_t offset = (b * 32U + i) * tileA_;
                    srcB8[0][b][0][i] = srcBase0 + offset;
                    srcB8[1][b][0][i] = srcBase1 + offset;
                    srcB8[0][b][1][i] = srcBase0 + offset + tileA_ * 16U;
                    srcB8[1][b][1][i] = srcBase1 + offset + tileA_ * 16U;
                }
            }

            if (storeGroup_ > 1U) {
                LocalTensor<T> dst = dstWide_.template Get<T>();
                const uint64_t dstBase = dst.GetPhyAddr();
                const uint32_t rowLen = WideRowLen();
                for (uint32_t slot = 0; slot < storeGroup_; ++slot) {
                    for (uint32_t b = 0; b < bBlocks; ++b) {
                        for (uint32_t i = 0; i < 16U; ++i) {
                            const uint32_t offset = i * rowLen + slot * tileB_ + b * 32U;
                            dstB8Wide[slot][b][0][i] = dstBase + offset;
                            dstB8Wide[slot][b][1][i] = dstBase + offset + rowLen * 16U;
                        }
                    }
                }
            } else {
                LocalTensor<T> d0 = dstTileA_.template Get<T>();
                LocalTensor<T> d1 = dstTileB_.template Get<T>();
                const uint64_t dstBase0 = d0.GetPhyAddr();
                const uint64_t dstBase1 = d1.GetPhyAddr();
                for (uint32_t b = 0; b < bBlocks; ++b) {
                    for (uint32_t i = 0; i < 16U; ++i) {
                        const uint32_t offset = i * tileB_ + b * 32U;
                        dstB8[0][b][0][i] = dstBase0 + offset;
                        dstB8[1][b][0][i] = dstBase1 + offset;
                        dstB8[0][b][1][i] = dstBase0 + offset + tileB_ * 16U;
                        dstB8[1][b][1][i] = dstBase1 + offset + tileB_ * 16U;
                    }
                }
            }
        }
    }

    __aicore__ inline void ProcessWholeTileHw()
    {
        const uint32_t elements = tileA_ * tileB_;
        LocalTensor<T> src = srcTileA_.template Get<T>();
        LocalTensor<T> dst = dstTileA_.template Get<T>();
        DataCopy(src, in_, elements);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        TransDataTo5HDParams tp;
        constexpr uint32_t elementsPerBlock = kIsB32 ? 8U : 16U;
        tp.repeatTimes = static_cast<uint8_t>(tileA_ / elementsPerBlock);
        tp.dstRepStride = tp.repeatTimes > 1 ? static_cast<uint16_t>(tileB_) : 0;
        tp.srcRepStride = tp.repeatTimes > 1 ? 1 : 0;
        const uint32_t hb = tileB_ / 16U;
        for (uint32_t i = 0; i < hb; ++i) {
            HwTranspose(&dstAddr0_[i * 16U], &srcAddr0_[i * 16U], tp);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(out_, dst, elements);
    }

    __aicore__ inline void IssueLoadHw(const TileMeta& m, uint32_t p)
    {
        LocalTensor<T> src = (p == 0) ? srcTileA_.template Get<T>() : srcTileB_.template Get<T>();
        const DataCopyPadExtParams<T> pad;
        const uint32_t ls = (tileA_ * sizeof(T) - AlignUpBlock(m.countA * sizeof(T))) / kUbBlockBytes;
        const DataCopyExtParams load(m.countB, m.countA * sizeof(T),
            (inStrideB_ - m.countA) * sizeof(T), ls, 0);
        DataCopyPad(src, in_[m.inOffset], load, pad);
    }

    __aicore__ inline void HwTranspose(
        uint64_t* dstAddr, uint64_t* srcAddr, const TransDataTo5HDParams& params)
    {
        if constexpr (kIsB32) {

            TransDataTo5HD<float>(*reinterpret_cast<uint64_t(*)[16]>(dstAddr),
                                  *reinterpret_cast<uint64_t(*)[16]>(srcAddr), params);
        } else {
            TransDataTo5HD<T>(*reinterpret_cast<uint64_t(*)[16]>(dstAddr),
                              *reinterpret_cast<uint64_t(*)[16]>(srcAddr), params);
        }
    }

    __aicore__ inline void HwTransposeB8(
        uint64_t* dstLow, uint64_t* dstHigh, uint64_t* srcLow, uint64_t* srcHigh,
        uint32_t dstPitch)
    {
        const uint8_t repeat = static_cast<uint8_t>(tileA_ / 32U);
        const uint16_t dstRepeatStride = repeat > 1 ? static_cast<uint16_t>(dstPitch) : 0;
        const uint16_t srcRepeatStride = repeat > 1 ? 1U : 0U;
        TransDataTo5HD<uint8_t>(*reinterpret_cast<uint64_t(*)[16]>(dstLow),
                                *reinterpret_cast<uint64_t(*)[16]>(srcLow),
                                {false, false, repeat, dstRepeatStride, srcRepeatStride});
        TransDataTo5HD<uint8_t>(*reinterpret_cast<uint64_t(*)[16]>(dstHigh),
                                *reinterpret_cast<uint64_t(*)[16]>(srcLow),
                                {false, true, repeat, dstRepeatStride, srcRepeatStride});
        TransDataTo5HD<uint8_t>(*reinterpret_cast<uint64_t(*)[16]>(dstHigh),
                                *reinterpret_cast<uint64_t(*)[16]>(srcHigh),
                                {true, true, repeat, dstRepeatStride, srcRepeatStride});
        TransDataTo5HD<uint8_t>(*reinterpret_cast<uint64_t(*)[16]>(dstLow),
                                *reinterpret_cast<uint64_t(*)[16]>(srcHigh),
                                {true, false, repeat, dstRepeatStride, srcRepeatStride});
    }

    __aicore__ inline void ComputeGroupedSlotMeta(
        uint64_t inBase, uint64_t outBase, uint64_t startA, uint64_t firstB,
        uint32_t slot, uint32_t countA, TileMeta& meta) const
    {
        const uint64_t startB = firstB + static_cast<uint64_t>(slot) * tileB_;
        meta.countA = countA;
        meta.countB = static_cast<uint32_t>(
            (lenB_ - startB) < tileB_ ? (lenB_ - startB) : tileB_);
        meta.inOffset = inBase + startA + startB * inStrideB_;
        meta.outOffset = outBase + startA * outStrideA_ + startB;
    }

    __aicore__ inline void ProcessTileHwGrouped(
        uint64_t begin, uint64_t end, uint64_t step)
    {
        if constexpr (kIsFloat && kF32BulkMode == 2U) {
            if (storeGroup_ == 64U && tileA_ == 32U && tileB_ == 16U) {
                ProcessTileHwGroupedF32Bulk2(begin, end, step);
                return;
            }
        } else if constexpr (kIsFloat && kF32BulkMode == 1U) {
            if (outerCount_ == 1U && storeGroup_ == 16U &&
                tileA_ == 128U && tileB_ == 16U) {
                ProcessTileHwGroupedF32Bulk2(begin, end, step);
                return;
            }
        }
        const uint32_t G = storeGroup_;
        const uint32_t hb = tileB_ / 16U;
        const uint32_t rowLen = WideRowLen();

        const uint32_t lastHb = fullRowStore_ != 0U ?
            ((rowLen - (tileCountB_ - 1U) * tileB_) + 15U) / 16U : hb;
        TransDataTo5HDParams tp;
        constexpr uint32_t elementsPerBlock = kIsB32 ? 8U : 16U;
        tp.repeatTimes = static_cast<uint8_t>(tileA_ / elementsPerBlock);
        tp.dstRepStride = tp.repeatTimes > 1 ? static_cast<uint16_t>(rowLen) : 0;
        tp.srcRepStride = tp.repeatTimes > 1 ? 1 : 0;
        if (begin >= end) { return; }
        LocalTensor<T> dw = dstWide_.template Get<T>();
        TileMeta meta[2];
        const uint64_t groupCountB = (tileCountB_ + G - 1U) / G;
        const uint64_t perOuter = tileCountA_ * groupCountB;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        for (uint64_t groupTask = begin; groupTask < end; groupTask += step) {
            const uint64_t outerIndex = groupTask / perOuter;
            const uint64_t rest = groupTask - outerIndex * perOuter;
            uint64_t tileIndexA;
            uint64_t groupIndexB;
            if (aFastShift_ != 0U) {
                tileIndexA = rest & (tileCountA_ - 1U);
                groupIndexB = rest >> aFastShift_;
            } else {
                tileIndexA = rest / groupCountB;
                groupIndexB = rest - tileIndexA * groupCountB;
            }
            const uint64_t firstTileB = groupIndexB * G;
            const uint32_t filled = static_cast<uint32_t>(
                (tileCountB_ - firstTileB) < G ? (tileCountB_ - firstTileB) : G);
            uint64_t inBase = 0;
            uint64_t outBase = 0;
            OuterOffsets(outerIndex, inBase, outBase);
            const uint64_t startA = tileIndexA * tileA_;
            const uint32_t countA = static_cast<uint32_t>(
                (lenA_ - startA) < tileA_ ? (lenA_ - startA) : tileA_);
            const uint64_t firstB = firstTileB * tileB_;
            const uint64_t groupOutOffset =
                outBase + startA * outStrideA_ + firstB;

            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            ComputeGroupedSlotMeta(
                inBase, outBase, startA, firstB, 0U, countA, meta[0]);
            IssueLoadHw(meta[0], 0U);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            for (uint32_t slot = 0; slot < filled; ++slot) {
                const uint32_t p = slot & 1U;
                const event_t ep = static_cast<event_t>(p);
                if (slot + 1U < filled) {
                    const uint32_t np = p ^ 1U;
                    const event_t nextEvent = static_cast<event_t>(np);
                    WaitFlag<HardEvent::V_MTE2>(nextEvent);
                    ComputeGroupedSlotMeta(
                        inBase, outBase, startA, firstB, slot + 1U, countA,
                        meta[np]);
                    IssueLoadHw(meta[np], np);
                    SetFlag<HardEvent::MTE2_V>(nextEvent);
                }
                WaitFlag<HardEvent::MTE2_V>(ep);
                if constexpr (kIsInt8) {
                    uint64_t (*srcB8)[2][2][16] =
                        reinterpret_cast<uint64_t (*)[2][2][16]>(srcAddr0_);
                    uint64_t (*dstB8Wide)[2][2][16] =
                        reinterpret_cast<uint64_t (*)[2][2][16]>(dstAddrWide_);
                    const uint32_t bBlocks = tileB_ / 32U;
                    for (uint32_t b = 0; b < bBlocks; ++b) {
                        HwTransposeB8(
                            dstB8Wide[slot][b][0], dstB8Wide[slot][b][1],
                            srcB8[p][b][0], srcB8[p][b][1], rowLen);
                    }
                } else {
                    uint64_t* srcAddr = p == 0U ? srcAddr0_ : srcAddr1_;
                    uint64_t* dstAddr;
                    if constexpr (kIsFloat) {
                        uint64_t (*dstF32Wide)[16] =
                            reinterpret_cast<uint64_t (*)[16]>(dstAddrWide_);
                        dstAddr = dstF32Wide[slot];
                    } else {
                        dstAddr = dstAddrWide_[slot];
                    }
                    const uint32_t hbSlot =
                        (slot + 1U == filled) ? lastHb : hb;
                    for (uint32_t i = 0; i < hbSlot; ++i) {
                        HwTranspose(&dstAddr[i * 16U], &srcAddr[i * 16U], tp);
                    }
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(ep);
            }
            const TileMeta& lastMeta = meta[(filled - 1U) & 1U];
            const uint32_t wide = (filled - 1U) * tileB_ + lastMeta.countB;
            if (fullRowStore_ == 2U) {

                LocalTensor<T> pk = packBuf_.template Get<T>();
                LocalTensor<uint32_t> po = packOffBuf_.template Get<int32_t>()
                    .template ReinterpretCast<uint32_t>();
                const uint32_t lb = static_cast<uint32_t>(lenB_);
                for (uint32_t r0 = 0; r0 < countA; r0 += packRows_) {
                    const uint32_t rr =
                        (countA - r0) < packRows_ ? (countA - r0) : packRows_;
                    const uint32_t gc = rr * packPerRow_;
                    if (packUnit_ == 4U) {
                        LocalTensor<int32_t> pd =
                            pk.template ReinterpretCast<int32_t>();
                        LocalTensor<int32_t> sv =
                            dw.template ReinterpretCast<int32_t>();

                        Gather(pd[(r0 * lb * sizeof(T)) / 4U],
                               sv[(r0 * rowLen * sizeof(T)) / 4U],
                               po, 0U, gc);
                    } else {
                        Gather(pk[r0 * lb], dw[r0 * rowLen], po, 0U, gc);
                    }
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                const DataCopyExtParams store(
                    1U, static_cast<uint32_t>(countA * lb * sizeof(T)),
                    0U, 0U, 0U);
                DataCopyPad(out_[groupOutOffset], pk, store);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
                continue;
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            if (fullRowStore_ != 0U) {

                const DataCopyExtParams store(
                    1U, static_cast<uint32_t>(countA * wide * sizeof(T)),
                    0U, 0U, 0U);
                DataCopyPad(out_[groupOutOffset], dw, store);
            } else {
                const uint32_t srcStride =
                    (rowLen * sizeof(T) - AlignUpBlock(wide * sizeof(T))) /
                    kUbBlockBytes;
                const DataCopyExtParams store(
                    countA, wide * sizeof(T), srcStride,
                    (outStrideA_ - wide) * sizeof(T), 0);
                DataCopyPad(out_[groupOutOffset], dw, store);
            }
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
    }

    __aicore__ inline void BuildShiftOffsets()
    {
        if constexpr (kIsFloat && kF32BulkMode == 2U) {
            LocalTensor<int32_t> off = offsetBuf_.template Get<int32_t>();
            const int32_t rawBase = static_cast<int32_t>(static_cast<uint32_t>(runChunk_) * 16U * 32U * sizeof(T));
            const uint32_t sMod = static_cast<uint32_t>(inStrideB_ % 8U);
            for (uint32_t a = 0U; a < 32U; ++a) {
                off.SetValue(a, rawBase + static_cast<int32_t>(a * sizeof(T)));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID3);
            WaitFlag<HardEvent::S_V>(EVENT_ID3);
            for (uint32_t m = 0U; m < 8U; ++m) {
                for (uint32_t u = 0U; u < 8U; ++u) {
                    if (m == 0U && u == 0U) continue;
                    const uint32_t r = (m + u * sMod) % 8U;
                    Adds(off[m * 256U + u * 32U], off,
                         static_cast<int32_t>((u * kShiftRawPitch + r) * sizeof(T)), 32);
                }
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void IssueF32Bulk2Load(
        uint32_t ping, uint64_t inBase, uint64_t startA, uint64_t firstB,
        uint32_t slotStart, uint32_t slots, uint32_t countA)
    {
        if constexpr (kF32BulkMode == 2U) {

        if constexpr (kIsFloat && kF32BulkMode != 0U) {
            const uint64_t startB =
                firstB + static_cast<uint64_t>(slotStart) * tileB_;
            const uint64_t lastB =
                startB + static_cast<uint64_t>(slots - 1U) * tileB_;
            const uint32_t lastCount = static_cast<uint32_t>(
                (lenB_ - lastB) < tileB_ ? (lenB_ - lastB) : tileB_);
            const uint32_t rows = (slots - 1U) * tileB_ + lastCount;
            const uint32_t loadStride =
                (tileA_ * sizeof(T) -
                 AlignUpBlock(countA * sizeof(T))) / kUbBlockBytes;
            const DataCopyPadExtParams<T> pad;
            LocalTensor<T> src = ping == 0U ?
                srcTileA_.template Get<T>() : srcTileB_.template Get<T>();
            const uint64_t base = inBase + startA + startB * inStrideB_;
            if (shiftRead_ != 0U) {

                for (uint32_t j = 0U; j < 8U && j < rows; ++j) {
                    const uint32_t cnt = (rows - j + 7U) / 8U;
                    const uint64_t bj = base + static_cast<uint64_t>(j) * inStrideB_;
                    const uint32_t r = static_cast<uint32_t>(bj % 8U);
                    const uint32_t blk =
                        AlignUpBlock((countA + r) * sizeof(T));
                    const uint32_t rowBytes = kShiftRawPitch * sizeof(T);
                    const uint32_t rawBase = static_cast<uint32_t>(runChunk_) * 16U * 32U;
                    const uint64_t total = outerTotalK_ * lenA_ * lenB_;
                    const uint64_t last = bj + static_cast<uint64_t>(cnt - 1U) * 8U * inStrideB_;
                    const bool exactTail = last - r + blk / sizeof(T) > total;
                    const uint32_t regular = exactTail ? cnt - 1U : cnt;
                    if (regular != 0U) {
                        const DataCopyExtParams ld(regular, blk,
                            8U * inStrideB_ * sizeof(T) - blk,
                            (8U * rowBytes - blk) / kUbBlockBytes, 0);
                        DataCopyPad(src[rawBase + j * kShiftRawPitch], in_[bj - r], ld, pad);
                    }
                    if (exactTail) {
                        const DataCopyExtParams tail(1U, (countA + r) * sizeof(T), 0U, 0U, 0U);
                        DataCopyPad(src[rawBase + (j + (cnt - 1U) * 8U) * kShiftRawPitch],
                                    in_[last - r], tail, pad);
                    }
                }
                return;
            }
            const DataCopyExtParams load(
                rows, countA * sizeof(T),
                (inStrideB_ - countA) * sizeof(T), loadStride, 0);
            DataCopyPad(src, in_[base], load, pad);
        }

        } else {

        if constexpr (kIsFloat && kF32BulkMode != 0U) {
            const uint64_t startB =
                firstB + static_cast<uint64_t>(slotStart) * tileB_;
            const uint64_t lastB =
                startB + static_cast<uint64_t>(slots - 1U) * tileB_;
            const uint32_t lastCount = static_cast<uint32_t>(
                (lenB_ - lastB) < tileB_ ? (lenB_ - lastB) : tileB_);
            const uint32_t rows = (slots - 1U) * tileB_ + lastCount;
            const uint32_t loadStride =
                (tileA_ * sizeof(T) -
                 AlignUpBlock(countA * sizeof(T))) / kUbBlockBytes;
            const DataCopyPadExtParams<T> pad;
            LocalTensor<T> src = ping == 0U ?
                srcTileA_.template Get<T>() : srcTileB_.template Get<T>();
            const uint64_t base = inBase + startA + startB * inStrideB_;
            if (shiftRead_ != 0U) {

                for (uint32_t j = 0U; j < 8U && j < rows; ++j) {
                    const uint32_t cnt = (rows - j + 7U) / 8U;
                    const uint64_t bj = base + static_cast<uint64_t>(j) * inStrideB_;
                    const uint32_t r = static_cast<uint32_t>(bj % 8U);
                    const uint32_t blk =
                        AlignUpBlock((countA + r) * sizeof(T));
                    const uint32_t rowBytes = kShiftRawPitch * sizeof(T);
                    const DataCopyExtParams ld(
                        cnt, blk,
                        8U * inStrideB_ * sizeof(T) - blk,
                        (8U * rowBytes - blk) / kUbBlockBytes, 0);
                    DataCopyPad(
                        src[kShiftAlnElems + j * kShiftRawPitch],
                        in_[bj - r], ld, pad);
                }
                return;
            }
            const DataCopyExtParams load(
                rows, countA * sizeof(T),
                (inStrideB_ - countA) * sizeof(T), loadStride, 0);
            DataCopyPad(src, in_[base], load, pad);
        }

        }
    }

    __aicore__ inline void ProcessTileHwGroupedF32Bulk2(
        uint64_t begin, uint64_t end, uint64_t step)
    {
        if constexpr (kF32BulkMode == 2U) {

        if constexpr (kIsFloat && kF32BulkMode != 0U) {
            if (begin >= end) { return; }
            constexpr uint32_t G = kF32BulkMode == 2U ? 64U : 16U;

            const uint32_t kLoadGroup =
                kF32BulkMode == 2U ? static_cast<uint32_t>(runChunk_) : 2U;
            const uint32_t rowLen = wideRowLen_;
            TransDataTo5HDParams tp;
            tp.repeatTimes = static_cast<uint8_t>(tileA_ / 8U);
            tp.dstRepStride = static_cast<uint16_t>(rowLen);
            tp.srcRepStride = 1U;
            LocalTensor<T> dst = dstWide_.template Get<T>();
            uint64_t (*srcTable0)[16] =
                reinterpret_cast<uint64_t (*)[16]>(srcAddr0_);
            uint64_t (*srcTable1)[16] =
                reinterpret_cast<uint64_t (*)[16]>(dstAddr0_);
            uint64_t (*dstTable)[16] =
                reinterpret_cast<uint64_t (*)[16]>(dstAddrWide_);
            const uint64_t groupCountB = (tileCountB_ + G - 1U) / G;
            const uint64_t perOuter = tileCountA_ * groupCountB;
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            for (uint64_t groupTask = begin; groupTask < end;
                 groupTask += step) {
                const uint64_t outerIndex = groupTask / perOuter;
                const uint64_t rest = groupTask - outerIndex * perOuter;
                uint64_t tileIndexA;
                uint64_t groupIndexB;
                if (aFastShift_ != 0U) {
                    tileIndexA = rest & (tileCountA_ - 1U);
                    groupIndexB = rest >> aFastShift_;
                } else {
                    tileIndexA = rest / groupCountB;
                    groupIndexB = rest - tileIndexA * groupCountB;
                }
                const uint64_t firstTileB = groupIndexB * G;
                const uint32_t filled = static_cast<uint32_t>(
                    (tileCountB_ - firstTileB) < G ?
                    (tileCountB_ - firstTileB) : G);
                uint64_t inBase = 0U;
                uint64_t outBase = 0U;
                OuterOffsets(outerIndex, inBase, outBase);
                const uint64_t startA = tileIndexA * tileA_;
                const uint32_t countA = static_cast<uint32_t>(
                    (lenA_ - startA) < tileA_ ?
                    (lenA_ - startA) : tileA_);
                const uint64_t firstB = firstTileB * tileB_;
                const uint32_t firstSlots =
                    filled < kLoadGroup ? filled : kLoadGroup;
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                IssueF32Bulk2Load(
                    0U, inBase, startA, firstB, 0U, firstSlots, countA);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);

                const uint32_t chunks =
                    (filled + kLoadGroup - 1U) / kLoadGroup;
                for (uint32_t chunk = 0U; chunk < chunks; ++chunk) {
                    const uint32_t ping = chunk & 1U;
                    const event_t event = static_cast<event_t>(ping);
                    if (chunk + 1U < chunks) {
                        const uint32_t nextPing = ping ^ 1U;
                        const event_t nextEvent =
                            static_cast<event_t>(nextPing);
                        const uint32_t nextStart =
                            (chunk + 1U) * kLoadGroup;
                        const uint32_t nextLeft = filled - nextStart;
                        const uint32_t nextSlots = nextLeft < kLoadGroup ?
                            nextLeft : kLoadGroup;
                        WaitFlag<HardEvent::V_MTE2>(nextEvent);
                        IssueF32Bulk2Load(
                            nextPing, inBase, startA, firstB,
                            nextStart, nextSlots, countA);
                        SetFlag<HardEvent::MTE2_V>(nextEvent);
                    }
                    WaitFlag<HardEvent::MTE2_V>(event);
                    const uint32_t slotStart = chunk * kLoadGroup;
                    const uint32_t left = filled - slotStart;
                    const uint32_t slots =
                        left < kLoadGroup ? left : kLoadGroup;
                    if (shiftRead_ != 0U) {
                        LocalTensor<T> sb = ping == 0U ?
                            srcTileA_.template Get<T>() :
                            srcTileB_.template Get<T>();
                        LocalTensor<uint32_t> offs =
                            offsetBuf_.template Get<int32_t>()
                                .template ReinterpretCast<uint32_t>();
                        const uint64_t sB =
                            firstB + static_cast<uint64_t>(slotStart) * tileB_;
                        const uint64_t lB =
                            sB + static_cast<uint64_t>(slots - 1U) * tileB_;
                        const uint32_t lC = static_cast<uint32_t>(
                            (lenB_ - lB) < tileB_ ? (lenB_ - lB) : tileB_);
                        const uint32_t nrow = (slots - 1U) * tileB_ + lC;
                        const uint32_t m = static_cast<uint32_t>(
                            (inBase + startA + sB * inStrideB_) % 8U);
                        const uint32_t ng = (nrow + 7U) / 8U;
                        for (uint32_t g = 0U; g < ng; ++g) {
                            Gather(sb[g * 256U], sb, offs[m * 256U],
                                   static_cast<uint32_t>(
                                       g * 8U * kShiftRawPitch * sizeof(T)),
                                   256U);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                    uint64_t (*srcTable)[16] =
                        ping == 0U ? srcTable0 : srcTable1;
                    for (uint32_t slot = 0U; slot < slots; ++slot) {
                        HwTranspose(
                            dstTable[slotStart + slot], srcTable[slot], tp);
                    }
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(event);
                }

                const uint64_t lastB =
                    firstB + static_cast<uint64_t>(filled - 1U) * tileB_;
                const uint32_t lastCount = static_cast<uint32_t>(
                    (lenB_ - lastB) < tileB_ ?
                    (lenB_ - lastB) : tileB_);
                const uint32_t wide = (filled - 1U) * tileB_ + lastCount;
                if (CanPackF32Whole()) {

                    uint64_t retained = 0U;
                    GatherMask(dst, dst, packOffBuf_.template Get<int32_t>().template ReinterpretCast<uint32_t>(),
                        true, countA * rowLen, GatherMaskParams(1, 1, 8, 0), retained);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID2); WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                    const DataCopyExtParams packedStore(1U, countA * wide * sizeof(T), 0U, 0U, 0U);
                    DataCopyPad(out_[outBase + startA * outStrideA_ + firstB], dst, packedStore);
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
                    continue;
                }
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                const uint32_t srcStride =
                    (rowLen * sizeof(T) - AlignUpBlock(wide * sizeof(T))) /
                    kUbBlockBytes;
                const DataCopyExtParams store(
                    countA, wide * sizeof(T), srcStride,
                    (outStrideA_ - wide) * sizeof(T), 0);
                DataCopyPad(
                    out_[outBase + startA * outStrideA_ + firstB], dst, store);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }

        } else {

        if constexpr (kIsFloat && kF32BulkMode != 0U) {
            if (begin >= end) { return; }
            constexpr uint32_t G = kF32BulkMode == 2U ? 64U : 16U;

            const uint32_t kLoadGroup =
                kF32BulkMode == 2U ? (shiftRead_ != 0U ? 8U : 16U) : 2U;
            const uint32_t rowLen = wideRowLen_;
            TransDataTo5HDParams tp;
            tp.repeatTimes = static_cast<uint8_t>(tileA_ / 8U);
            tp.dstRepStride = static_cast<uint16_t>(rowLen);
            tp.srcRepStride = 1U;
            LocalTensor<T> dst = dstWide_.template Get<T>();
            uint64_t (*srcTable0)[16] =
                reinterpret_cast<uint64_t (*)[16]>(srcAddr0_);
            uint64_t (*srcTable1)[16] =
                reinterpret_cast<uint64_t (*)[16]>(dstAddr0_);
            uint64_t (*dstTable)[16] =
                reinterpret_cast<uint64_t (*)[16]>(dstAddrWide_);
            const uint64_t groupCountB = (tileCountB_ + G - 1U) / G;
            const uint64_t perOuter = tileCountA_ * groupCountB;
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            for (uint64_t groupTask = begin; groupTask < end;
                 groupTask += step) {
                const uint64_t outerIndex = groupTask / perOuter;
                const uint64_t rest = groupTask - outerIndex * perOuter;
                uint64_t tileIndexA;
                uint64_t groupIndexB;
                if (aFastShift_ != 0U) {
                    tileIndexA = rest & (tileCountA_ - 1U);
                    groupIndexB = rest >> aFastShift_;
                } else {
                    tileIndexA = rest / groupCountB;
                    groupIndexB = rest - tileIndexA * groupCountB;
                }
                const uint64_t firstTileB = groupIndexB * G;
                const uint32_t filled = static_cast<uint32_t>(
                    (tileCountB_ - firstTileB) < G ?
                    (tileCountB_ - firstTileB) : G);
                uint64_t inBase = 0U;
                uint64_t outBase = 0U;
                OuterOffsets(outerIndex, inBase, outBase);
                const uint64_t startA = tileIndexA * tileA_;
                const uint32_t countA = static_cast<uint32_t>(
                    (lenA_ - startA) < tileA_ ?
                    (lenA_ - startA) : tileA_);
                const uint64_t firstB = firstTileB * tileB_;
                const uint32_t firstSlots =
                    filled < kLoadGroup ? filled : kLoadGroup;
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                IssueF32Bulk2Load(
                    0U, inBase, startA, firstB, 0U, firstSlots, countA);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);

                const uint32_t chunks =
                    (filled + kLoadGroup - 1U) / kLoadGroup;
                for (uint32_t chunk = 0U; chunk < chunks; ++chunk) {
                    const uint32_t ping = chunk & 1U;
                    const event_t event = static_cast<event_t>(ping);
                    if (chunk + 1U < chunks) {
                        const uint32_t nextPing = ping ^ 1U;
                        const event_t nextEvent =
                            static_cast<event_t>(nextPing);
                        const uint32_t nextStart =
                            (chunk + 1U) * kLoadGroup;
                        const uint32_t nextLeft = filled - nextStart;
                        const uint32_t nextSlots = nextLeft < kLoadGroup ?
                            nextLeft : kLoadGroup;
                        WaitFlag<HardEvent::V_MTE2>(nextEvent);
                        IssueF32Bulk2Load(
                            nextPing, inBase, startA, firstB,
                            nextStart, nextSlots, countA);
                        SetFlag<HardEvent::MTE2_V>(nextEvent);
                    }
                    WaitFlag<HardEvent::MTE2_V>(event);
                    const uint32_t slotStart = chunk * kLoadGroup;
                    const uint32_t left = filled - slotStart;
                    const uint32_t slots =
                        left < kLoadGroup ? left : kLoadGroup;
                    if (shiftRead_ != 0U) {
                        LocalTensor<T> sb = ping == 0U ?
                            srcTileA_.template Get<T>() :
                            srcTileB_.template Get<T>();
                        LocalTensor<uint32_t> offs =
                            offsetBuf_.template Get<int32_t>()
                                .template ReinterpretCast<uint32_t>();
                        const uint64_t sB =
                            firstB + static_cast<uint64_t>(slotStart) * tileB_;
                        const uint64_t lB =
                            sB + static_cast<uint64_t>(slots - 1U) * tileB_;
                        const uint32_t lC = static_cast<uint32_t>(
                            (lenB_ - lB) < tileB_ ? (lenB_ - lB) : tileB_);
                        const uint32_t nrow = (slots - 1U) * tileB_ + lC;
                        const uint32_t m = static_cast<uint32_t>(
                            (inBase + startA + sB * inStrideB_) % 8U);
                        const uint32_t ng = (nrow + 7U) / 8U;
                        for (uint32_t g = 0U; g < ng; ++g) {
                            Gather(sb[g * 256U], sb, offs[m * 256U],
                                   static_cast<uint32_t>(
                                       g * 8U * kShiftRawPitch * sizeof(T)),
                                   256U);
                        }
                        PipeBarrier<PIPE_V>();
                    }
                    uint64_t (*srcTable)[16] =
                        ping == 0U ? srcTable0 : srcTable1;
                    for (uint32_t slot = 0U; slot < slots; ++slot) {
                        HwTranspose(
                            dstTable[slotStart + slot], srcTable[slot], tp);
                    }
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(event);
                }

                const uint64_t lastB =
                    firstB + static_cast<uint64_t>(filled - 1U) * tileB_;
                const uint32_t lastCount = static_cast<uint32_t>(
                    (lenB_ - lastB) < tileB_ ?
                    (lenB_ - lastB) : tileB_);
                const uint32_t wide = (filled - 1U) * tileB_ + lastCount;
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                const uint32_t srcStride =
                    (rowLen * sizeof(T) - AlignUpBlock(wide * sizeof(T))) /
                    kUbBlockBytes;
                const DataCopyExtParams store(
                    countA, wide * sizeof(T), srcStride,
                    (outStrideA_ - wide) * sizeof(T), 0);
                DataCopyPad(
                    out_[outBase + startA * outStrideA_ + firstB], dst, store);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }

        }
    }

    __aicore__ inline void BuildWideGatherOffsets(uint32_t wideB)
    {
        LocalTensor<int32_t> offsets = offsetBuf_.template Get<int32_t>();

        if ((wideB * sizeof(int32_t)) % kUbBlockBytes == 0U) {
            for (uint32_t b = 0U; b < wideB; ++b) {
                offsets.SetValue(
                    b, static_cast<int32_t>(b * tileA_ * sizeof(T)));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            for (uint32_t a = 1U; a < tileA_; ++a) {
                Adds(
                    offsets[a * wideB], offsets,
                    static_cast<int32_t>(a * sizeof(T)),
                    static_cast<int32_t>(wideB));
            }
            PipeBarrier<PIPE_V>();
        } else {
            for (uint32_t a = 0U; a < tileA_; ++a) {
                for (uint32_t b = 0U; b < wideB; ++b) {
                    offsets.SetValue(
                        a * wideB + b,
                        static_cast<int32_t>((b * tileA_ + a) * sizeof(T)));
                }
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
    }

    __aicore__ inline void ComputeWideGroupMeta(
        uint64_t groupTask, uint64_t groupCountB, uint32_t wideB,
        TileMeta& meta) const
    {
        const uint64_t perOuter = tileCountA_ * groupCountB;
        const uint64_t outerIndex = groupTask / perOuter;
        const uint64_t rest = groupTask - outerIndex * perOuter;
        uint64_t tileIndexA;
        uint64_t groupIndexB;
        if (aFastShift_ != 0U) {
            tileIndexA = rest & (tileCountA_ - 1U);
            groupIndexB = rest >> aFastShift_;
        } else {
            tileIndexA = rest / groupCountB;
            groupIndexB = rest - tileIndexA * groupCountB;
        }
        uint64_t inBase = 0U;
        uint64_t outBase = 0U;
        OuterOffsets(outerIndex, inBase, outBase);
        const uint64_t startA = tileIndexA * tileA_;
        meta.countA = static_cast<uint32_t>(
            (lenA_ - startA) < tileA_ ? (lenA_ - startA) : tileA_);
        const uint64_t startB =
            groupIndexB * static_cast<uint64_t>(wideB);
        meta.countB = wideB;
        meta.inOffset = inBase + startA + startB * inStrideB_;
        meta.outOffset = outBase + startA * outStrideA_ + startB;
    }

    __aicore__ inline void IssueWideGroupLoad(
        const TileMeta& meta, const LocalTensor<T>& src, uint32_t wideB)
    {
        const uint32_t loadStride =
            (tileA_ * sizeof(T) -
             AlignUpBlock(meta.countA * sizeof(T))) / kUbBlockBytes;
        const DataCopyExtParams load(
            wideB, meta.countA * sizeof(T),
            (inStrideB_ - meta.countA) * sizeof(T), loadStride, 0);
        const DataCopyPadExtParams<T> pad;
        DataCopyPad(src, in_[meta.inOffset], load, pad);
    }

    __aicore__ inline void ProcessTileGatherGrouped(
        uint64_t begin, uint64_t end, uint64_t step)
    {
        if constexpr (kIsInt32) {

            const uint32_t wideB = fullRowStore_ != 0U ?
                static_cast<uint32_t>(lenB_) : storeGroup_ * tileB_;
            BuildWideGatherOffsets(wideB);
            LocalTensor<T> src[2] = {
                srcWideBuf_.template Get<T>(),
                srcWideBufB_.template Get<T>()};
            LocalTensor<T> dst = dstWideBuf_.template Get<T>();
            LocalTensor<uint32_t> offsets =
                offsetBuf_.template Get<int32_t>().template ReinterpretCast<uint32_t>();
            const uint64_t groupCountB = tileCountB_ / storeGroup_;
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            const uint64_t n =
                (end - begin + step - 1U) / step;
            if (n == 0U) {
                return;
            }
            TileMeta meta[2];
            ComputeWideGroupMeta(begin, groupCountB, wideB, meta[0]);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            IssueWideGroupLoad(meta[0], src[0], wideB);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (uint64_t k = 0U; k < n; ++k) {
                const uint32_t p = static_cast<uint32_t>(k & 1U);
                const event_t ep = static_cast<event_t>(p);
                if (k + 1U < n) {
                    const uint32_t np = p ^ 1U;
                    const event_t nextEvent = static_cast<event_t>(np);
                    ComputeWideGroupMeta(
                        begin + (k + 1U) * step,
                        groupCountB, wideB, meta[np]);
                    WaitFlag<HardEvent::V_MTE2>(nextEvent);
                    IssueWideGroupLoad(meta[np], src[np], wideB);
                    SetFlag<HardEvent::MTE2_V>(nextEvent);
                }
                WaitFlag<HardEvent::MTE2_V>(ep);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
                Gather(
                    dst, src[p], offsets, 0U,
                    static_cast<uint32_t>(meta[p].countA * wideB));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(ep);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                if (fullRowStore_ != 0U) {

                    const DataCopyExtParams whole(
                        1U,
                        static_cast<uint32_t>(meta[p].countA) * wideB *
                            static_cast<uint32_t>(sizeof(T)),
                        0U, 0U, 0U);
                    DataCopyPad(out_[meta[p].outOffset], dst, whole);
                } else {
                    const DataCopyExtParams store(
                        meta[p].countA, wideB * sizeof(T), 0,
                        (outStrideA_ - wideB) * sizeof(T), 0);
                    DataCopyPad(out_[meta[p].outOffset], dst, store);
                }
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
    }

    __aicore__ inline void ProcessTileHw(uint64_t begin, uint64_t end)
    {
        const uint32_t hb = tileB_ / 16U;
        TransDataTo5HDParams tp;
        constexpr uint32_t elementsPerBlock = kIsB32 ? 8U : 16U;
        tp.repeatTimes = static_cast<uint8_t>(tileA_ / elementsPerBlock);
        tp.dstRepStride = tp.repeatTimes > 1 ? static_cast<uint16_t>(tileB_) : 0;
        tp.srcRepStride = tp.repeatTimes > 1 ? 1 : 0;
        const DataCopyPadExtParams<T> pad;
        const uint64_t n = end - begin;
        if (n == 0) { return; }
        TileMeta meta[2];

        SetFlag<HardEvent::V_MTE2>(EVENT_ID0); SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0); SetFlag<HardEvent::MTE3_V>(EVENT_ID1);

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        ComputeTileMeta(begin, meta[0]);
        IssueLoadHw(meta[0], 0);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (uint64_t k = 0; k < n; ++k) {
            const uint32_t p = static_cast<uint32_t>(k & 1U);
            const event_t ep = static_cast<event_t>(p);
            if (k + 1 < n) {
                const uint32_t np = p ^ 1U;
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(np));
                ComputeTileMeta(begin + k + 1U, meta[np]);
                IssueLoadHw(meta[np], np);
                SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(np));
            }
            const TileMeta& current = meta[p];
            LocalTensor<T> dst = (p == 0) ? dstTileA_.template Get<T>() : dstTileB_.template Get<T>();
            WaitFlag<HardEvent::MTE2_V>(ep);
            WaitFlag<HardEvent::MTE3_V>(ep);
            if constexpr (kIsInt8) {
                uint64_t (*srcB8)[2][2][16] =
                    reinterpret_cast<uint64_t (*)[2][2][16]>(srcAddr0_);
                uint64_t (*dstB8)[2][2][16] =
                    reinterpret_cast<uint64_t (*)[2][2][16]>(dstAddr0_);
                const uint32_t bBlocks = tileB_ / 32U;
                for (uint32_t b = 0; b < bBlocks; ++b) {
                    HwTransposeB8(dstB8[p][b][0], dstB8[p][b][1],
                                  srcB8[p][b][0], srcB8[p][b][1], tileB_);
                }
            } else {
                uint64_t* srcAddr = (p == 0) ? srcAddr0_ : srcAddr1_;
                uint64_t* dstAddr = (p == 0) ? dstAddr0_ : dstAddr1_;
                for (uint32_t i = 0; i < hb; ++i) {
                    HwTranspose(&dstAddr[i * 16U], &srcAddr[i * 16U], tp);
                }
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(ep);
            SetFlag<HardEvent::V_MTE3>(ep);
            WaitFlag<HardEvent::V_MTE3>(ep);
            const uint32_t ss =
                (tileB_ * sizeof(T) - AlignUpBlock(current.countB * sizeof(T))) /
                kUbBlockBytes;
            const DataCopyExtParams store(current.countA, current.countB * sizeof(T), ss,
                (outStrideA_ - current.countB) * sizeof(T), 0);
            DataCopyPad(out_[current.outOffset], dst, store);
            SetFlag<HardEvent::MTE3_V>(ep);
        }
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0); WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0); WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    }

    __aicore__ inline void ComputeTile(const TileMeta& meta)
    {
        LocalTensor<T> src = srcQue_.template DeQue<T>();
        LocalTensor<T> dst = dstQue_.template AllocTensor<T>();
        const uint32_t gathered = meta.countA * GatherDstPitch();
        if constexpr (kIsInt8) {
            LocalTensor<uint32_t> offsets =
                offsetBuf_.template Get<int32_t>().template ReinterpretCast<uint32_t>();
            LocalTensor<G> srcWide = srcWideBuf_.template Get<G>();
            LocalTensor<G> dstWide = dstWideBuf_.template Get<G>();
            Cast(srcWide, src, RoundMode::CAST_NONE, static_cast<int32_t>(kTileCap * kTileCap));
            PipeBarrier<PIPE_V>();
            Gather(dstWide, srcWide, offsets, 0U, gathered);
            PipeBarrier<PIPE_V>();
            Cast(dst, dstWide, RoundMode::CAST_RINT, static_cast<int32_t>(gathered));
            PipeBarrier<PIPE_V>();
        } else if constexpr (sizeof(T) == 2) {

            LocalTensor<uint8_t> ctTmp = ctTmpBuf_.template Get<uint8_t>();
            AscendC::tiling::ConfusionTransposeTiling ctTiling;
            ctTiling.param0 = 0U;
            ctTiling.param1 = tileB_;
            ctTiling.param2 = tileA_;
            ctTiling.param3 = tileB_ / 16U;
            ctTiling.param4 = tileB_;
            ctTiling.param5 = tileA_ / 16U;
            Transpose(dst, src, ctTmp, TransposeType::TRANSPOSE_ND2ND_ONLY, ctTiling);
            PipeBarrier<PIPE_V>();
        } else if (fullRowStore_ == 0U && (tileA_ % 16U) == 0U &&
                   (tileB_ % 16U) == 0U) {

            LocalTensor<uint8_t> ctTmp =
                offsetBuf_.template Get<int32_t>().template ReinterpretCast<uint8_t>();
            AscendC::tiling::ConfusionTransposeTiling ctTiling;
            ctTiling.param0 = 8U;
            ctTiling.param1 = tileB_;
            ctTiling.param2 = tileA_;
            ctTiling.param3 = tileB_ / 16U;
            ctTiling.param4 = tileB_;
            ctTiling.param5 = tileA_ / 8U;
            Transpose(dst.template ReinterpretCast<float>(),
                      src.template ReinterpretCast<float>(),
                      ctTmp, TransposeType::TRANSPOSE_ND2ND_ONLY, ctTiling);
            PipeBarrier<PIPE_V>();
            (void)gathered;
        } else {
            LocalTensor<uint32_t> offsets =
                offsetBuf_.template Get<int32_t>().template ReinterpretCast<uint32_t>();
            Gather(dst, src, offsets, 0U, gathered);
            PipeBarrier<PIPE_V>();
        }
        srcQue_.FreeTensor(src);
        dstQue_.EnQue(dst);
    }

    __aicore__ inline void CopyOutTile(const TileMeta& meta)
    {
        LocalTensor<T> dst = dstQue_.template DeQue<T>();
        if (fullRowStore_ != 0U) {

            const DataCopyExtParams store(
                1U, static_cast<uint32_t>(meta.countA * meta.countB * sizeof(T)),
                0U, 0U, 0U);
            DataCopyPad(out_[meta.outOffset], dst, store);
        } else {
            const uint32_t storeStride =
                (tileB_ * sizeof(T) - AlignUpBlock(meta.countB * sizeof(T))) / kUbBlockBytes;
            const DataCopyExtParams store(
                meta.countA, meta.countB * sizeof(T), storeStride, (outStrideA_ - meta.countB) * sizeof(T), 0);
            DataCopyPad(out_[meta.outOffset], dst, store);
        }
        dstQue_.FreeTensor(dst);
    }

    __aicore__ inline void RunTaskCopy(uint64_t task)
    {
        const uint64_t perOuter = midChunkCount_ * runChunkCount_;
        const uint64_t outerIndex = task / perOuter;
        const uint64_t rest = task - outerIndex * perOuter;
        const uint64_t midChunk = rest / runChunkCount_;
        const uint64_t runChunk = rest - midChunk * runChunkCount_;

        uint64_t inBase = 0;
        uint64_t outBase = 0;
        OuterOffsets(outerIndex, inBase, outBase);
        const uint64_t midStart = midChunk * midBatch_;
        const uint32_t rows =
            static_cast<uint32_t>((midLen_ - midStart) < midBatch_ ? (midLen_ - midStart) : midBatch_);
        const uint64_t runStart = runChunk * runChunk_;
        const uint32_t count =
            static_cast<uint32_t>((runLen_ - runStart) < runChunk_ ? (runLen_ - runStart) : runChunk_);
        if (rows == 0 || count == 0) {
            return;
        }
        const uint64_t inOffset = inBase + midStart * midInStride_ + runStart;
        const uint64_t outOffset = outBase + midStart * runLen_ + runStart;

        LocalTensor<T> tile = runBuf_.template Get<T>();
        const DataCopyPadExtParams<T> pad;
        const DataCopyExtParams load(
            rows, count * sizeof(T), (midInStride_ - count) * sizeof(T), 0, 0);
        DataCopyPad(tile, in_[inOffset], load, pad);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        const DataCopyExtParams store(
            rows, count * sizeof(T), 0, (runLen_ - count) * sizeof(T), 0);
        DataCopyPad(out_[outOffset], tile, store);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> runBuf_;
    TQue<TPosition::VECIN, kTileBufferDepth> srcQue_;
    TQue<TPosition::VECOUT, kTileBufferDepth> dstQue_;
    TBuf<TPosition::VECCALC> offsetBuf_;
    TBuf<TPosition::VECCALC> srcTileA_;
    TBuf<TPosition::VECCALC> srcTileB_;
    TBuf<TPosition::VECCALC> dstTileA_;
    TBuf<TPosition::VECCALC> dstTileB_;
    TBuf<TPosition::VECCALC> dstWide_;
    uint64_t srcAddr0_[256];
    uint64_t srcAddr1_[128];
    uint64_t dstAddr0_[256];
    uint64_t dstAddr1_[128];
    uint64_t dstAddrWide_[8][128];
    uint32_t fullRowStore_ = 0U;
    uint32_t packUnit_ = 0U;
    uint32_t packPerRow_ = 0U;
    uint32_t packRows_ = 0U;
    TBuf<TPosition::VECCALC> packBuf_;
    TBuf<TPosition::VECCALC> packOffBuf_;
    TBuf<TPosition::VECCALC> ctTmpBuf_;
    TBuf<TPosition::VECCALC> srcWideBuf_;
    TBuf<TPosition::VECCALC> srcWideBufB_;
    TBuf<TPosition::VECCALC> dstWideBuf_;
    GlobalTensor<T> in_;
    GlobalTensor<T> out_;
    uint64_t outerLen_[S9_TRANSPOSE_MAX_GROUP];
    uint64_t outerInStride_[S9_TRANSPOSE_MAX_GROUP];
    uint64_t outerOutStride_[S9_TRANSPOSE_MAX_GROUP];
    uint64_t taskCount_;
    uint32_t storeGroup_;

    uint32_t wideRowLen_ = 0U;
    uint32_t laneL_ = 1U;

    uint32_t shiftRead_ = 0U;
    static constexpr uint32_t kShiftRawPitch = 40U;
    static constexpr uint32_t kShiftAlnElems = 128U * 32U;
    uint32_t wholeTile_;
    uint32_t aFastShift_;
    uint64_t runLen_;
    uint64_t runChunk_;
    uint64_t runChunkCount_;
    uint64_t midLen_;
    uint64_t midInStride_;
    uint64_t midBatch_;
    uint64_t midChunkCount_;
    uint64_t lenA_;
    uint64_t lenB_;
    uint64_t tileCountA_;
    uint64_t tileCountB_;
    uint64_t inStrideB_;
    uint64_t outStrideA_;
    uint32_t outerFast_ = 0U;
    uint64_t outerTotalK_ = 1U;
    uint32_t outerCount_;
    uint32_t regime_;
    uint32_t tileA_;
    uint32_t tileB_;
};
}

extern "C" __global__ __aicore__ void transpose(
    GM_ADDR inputs, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TransposeTilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelWholeTranspose<DTYPE_INPUTS> op;
        op.Init(inputs, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        if constexpr (sizeof(DTYPE_INPUTS) == 2U) {
            if (KernelBatchRunTranspose<DTYPE_INPUTS, 8U>::CanUse(tilingData)) {
                KernelBatchRunTranspose<DTYPE_INPUTS, 8U> batch;
                batch.Init(inputs, y, tilingData); batch.Process(); return;
            }
        }
        KernelRunTranspose<DTYPE_INPUTS> op;
        op.Init(inputs, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelTranspose<DTYPE_INPUTS, 1U> op;
        op.Init(inputs, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelTranspose<DTYPE_INPUTS, 2U> op;
        op.Init(inputs, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KernelTranspose<DTYPE_INPUTS, 0U> op;
        op.Init(inputs, y, tilingData);
        op.Process();
    }
}
