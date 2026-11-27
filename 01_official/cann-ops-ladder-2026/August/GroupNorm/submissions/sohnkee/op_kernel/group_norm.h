/*!
 * \file group_norm.h
 * \brief Generic correctness-first GroupNorm kernel for Ascend 910B.
 */

#ifndef GROUP_NORM_H
#define GROUP_NORM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "group_norm_tiling_data.h"
#include "group_norm_tiling_key.h"

namespace NsGroupNorm {

using namespace AscendC;

constexpr int32_t REDUCE_HALF_ELEMENTS = 4096;
constexpr int32_t REDUCE_MAX_ELEMENTS = 8192;
constexpr int32_t REDUCE_DOUBLE_ELEMENTS = 16384;
constexpr int32_t BLOCK_BYTES = 32;
constexpr int32_t TINY_STATIC_MAX_ELEMENTS = 64;
constexpr int32_t TINY_STATIC_MAX_CHANNELS = 512;
constexpr uint32_t TINY_X_ADDR = 0;
constexpr uint32_t TINY_X1_ADDR = 512;
constexpr uint32_t TINY_X2_ADDR = 1024;
constexpr uint32_t TINY_REDUCE_ADDR = 1536;
constexpr uint32_t TINY_GAMMA_ADDR = 2048;
constexpr uint32_t TINY_BETA_ADDR = 4096;
constexpr uint8_t TINY_EVENT_IO = 0;
constexpr uint8_t TINY_EVENT_REDUCE = 1;

template <typename T>
class TinyStaticGroupNorm {
public:
    __aicore__ inline TinyStaticGroupNorm() = default;
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y,
        const GroupNormTilingData* tilingData)
    {
        tiling_ = tilingData;
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= tiling_->totalGroups) {
            return;
        }
        groupStart_ = blockIdx_ * tiling_->blockFactor;
        const int64_t gmOffset = groupStart_ * tiling_->groupElements;
        xGm_.SetGlobalBuffer((__gm__ T*)x + gmOffset);
        yGm_.SetGlobalBuffer((__gm__ T*)y + gmOffset);
        gammaGm_.SetGlobalBuffer((__gm__ float*)gamma);
        betaGm_.SetGlobalBuffer((__gm__ float*)beta);
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= tiling_->totalGroups) {
            return;
        }
        const int64_t count = tiling_->groupElements;
        const int64_t channelStart =
            (groupStart_ % tiling_->numGroups) * tiling_->channelsPerGroup;
        LocalTensor<T> ioLocal(TPosition::VECCALC, TINY_X_ADDR, TINY_STATIC_MAX_ELEMENTS);
        LocalTensor<float> x1Local(
            TPosition::VECCALC, TINY_X1_ADDR, TINY_STATIC_MAX_ELEMENTS);
        LocalTensor<float> x2Local(
            TPosition::VECCALC, TINY_X2_ADDR, TINY_STATIC_MAX_ELEMENTS);
        LocalTensor<float> reduceLocal(
            TPosition::VECCALC, TINY_REDUCE_ADDR, BLOCK_BYTES / sizeof(float));
        LocalTensor<float> gammaLocal(
            TPosition::VECCALC, TINY_GAMMA_ADDR, TINY_STATIC_MAX_CHANNELS);
        LocalTensor<float> betaLocal(
            TPosition::VECCALC, TINY_BETA_ADDR, TINY_STATIC_MAX_CHANNELS);

        DataCopyPad(ioLocal, xGm_,
            {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0}, {false, 0, 0, 0});
        DataCopyPad(gammaLocal, gammaGm_,
            {1, static_cast<uint16_t>(tiling_->channels * sizeof(float)), 0, 0},
            {false, 0, 0, 0});
        DataCopyPad(betaLocal, betaGm_,
            {1, static_cast<uint16_t>(tiling_->channels * sizeof(float)), 0, 0},
            {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(TINY_EVENT_IO);
        SetFlag<HardEvent::MTE2_S>(TINY_EVENT_REDUCE);
        WaitFlag<HardEvent::MTE2_V>(TINY_EVENT_IO);
        WaitFlag<HardEvent::MTE2_S>(TINY_EVENT_REDUCE);

        if constexpr (sizeof(T) == sizeof(float)) {
            Adds(x1Local, ioLocal, 0.0f, count);
        } else {
            Cast(x1Local, ioLocal, RoundMode::CAST_NONE, count);
        }
        PipeBarrier<PIPE_V>();

        WholeReduceSum(reduceLocal, x1Local, static_cast<int32_t>(count), 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(TINY_EVENT_REDUCE);
        WaitFlag<HardEvent::V_S>(TINY_EVENT_REDUCE);
        const float mean = reduceLocal.GetValue(0) / static_cast<float>(count);
        Adds(x2Local, x1Local, -mean, count);
        PipeBarrier<PIPE_V>();
        Mul(x2Local, x2Local, x2Local, count);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(reduceLocal, x2Local, static_cast<int32_t>(count), 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(TINY_EVENT_REDUCE);
        WaitFlag<HardEvent::V_S>(TINY_EVENT_REDUCE);
        const float variance = reduceLocal.GetValue(0) / static_cast<float>(count);
        const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);

        if (tiling_->spatial % (BLOCK_BYTES / sizeof(float)) == 0) {
            for (int64_t channel = 0; channel < tiling_->channelsPerGroup; ++channel) {
                const float gamma = gammaLocal.GetValue(channelStart + channel);
                const float beta = betaLocal.GetValue(channelStart + channel);
                const float scale = rstd * gamma;
                const float bias = beta - mean * scale;
                const int64_t localOffset = channel * tiling_->spatial;
                Muls(x1Local[localOffset], x1Local[localOffset], scale, tiling_->spatial);
                Adds(x1Local[localOffset], x1Local[localOffset], bias, tiling_->spatial);
            }
            PipeBarrier<PIPE_V>();
            Store(ioLocal, x1Local, count);
            SetFlag<HardEvent::V_MTE3>(TINY_EVENT_IO);
            WaitFlag<HardEvent::V_MTE3>(TINY_EVENT_IO);
            DataCopyPad(yGm_, ioLocal,
                {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0});
            SetFlag<HardEvent::MTE3_MTE2>(TINY_EVENT_IO);
            WaitFlag<HardEvent::MTE3_MTE2>(TINY_EVENT_IO);
            return;
        }

        SetFlag<HardEvent::V_MTE2>(TINY_EVENT_IO);
        WaitFlag<HardEvent::V_MTE2>(TINY_EVENT_IO);
        for (int64_t channel = 0; channel < tiling_->channelsPerGroup; ++channel) {
            const int64_t channelOffset = channel * tiling_->spatial;
            if (channel != 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(TINY_EVENT_IO);
                DataCopyPad(ioLocal, xGm_[channelOffset],
                    {1, static_cast<uint16_t>(tiling_->spatial * sizeof(T)), 0, 0},
                    {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(TINY_EVENT_IO);
                WaitFlag<HardEvent::MTE2_V>(TINY_EVENT_IO);
                if constexpr (sizeof(T) == sizeof(float)) {
                    Adds(x1Local, ioLocal, 0.0f, tiling_->spatial);
                } else {
                    Cast(x1Local, ioLocal, RoundMode::CAST_NONE, tiling_->spatial);
                }
                PipeBarrier<PIPE_V>();
            }
            const float gamma = gammaLocal.GetValue(channelStart + channel);
            const float beta = betaLocal.GetValue(channelStart + channel);
            const float scale = rstd * gamma;
            const float bias = beta - mean * scale;
            Muls(x1Local, x1Local, scale, tiling_->spatial);
            Adds(x1Local, x1Local, bias, tiling_->spatial);
            PipeBarrier<PIPE_V>();
            Store(ioLocal, x1Local, tiling_->spatial);
            SetFlag<HardEvent::V_MTE3>(TINY_EVENT_IO);
            WaitFlag<HardEvent::V_MTE3>(TINY_EVENT_IO);
            DataCopyPad(yGm_[channelOffset], ioLocal,
                {1, static_cast<uint16_t>(tiling_->spatial * sizeof(T)), 0, 0});
            SetFlag<HardEvent::MTE3_MTE2>(TINY_EVENT_IO);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(TINY_EVENT_IO);
    }

private:
    __aicore__ inline void Store(
        const LocalTensor<T>& dst, const LocalTensor<float>& src, int64_t count)
    {
        if constexpr (sizeof(T) == sizeof(float)) {
            Adds(dst, src, 0.0f, count);
        } else {
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
            Cast(dst, src, RoundMode::CAST_RINT, count);
#else
            Cast(dst, src, RoundMode::CAST_NONE, count);
#endif
        }
        PipeBarrier<PIPE_V>();
    }

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<float> betaGm_;
    const GroupNormTilingData* tiling_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t groupStart_ = 0;
};

template <typename T>
class GroupNorm {
public:
    __aicore__ inline GroupNorm() = default;
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, const GroupNormTilingData* tilingData);
    __aicore__ inline void Process();
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    __aicore__ inline void ProcessLowAFirstTile();
#endif
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_3
    __aicore__ inline void ProcessTwoByteShiftedMoments();
#endif

private:
    __aicore__ inline int64_t CeilDiv(int64_t value, int64_t divisor) const
    {
        return (value + divisor - 1) / divisor;
    }

    __aicore__ inline int64_t Min(int64_t lhs, int64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline RoundMode OutputRoundMode() const
    {
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        return RoundMode::CAST_RINT;
#else
        return RoundMode::CAST_NONE;
#endif
    }

    __aicore__ inline void CopyInAffine();
    __aicore__ inline void CopyInX(int64_t offset, int64_t count);
    __aicore__ inline void CopyOutY(int64_t offset, int64_t count);
    __aicore__ inline void LoadAsFloat(int64_t offset, int64_t count);
    __aicore__ inline void LoadPrefetchedAsFloat(int64_t count);
    __aicore__ inline int64_t FirstOutputCount() const;
    __aicore__ inline void LoadStatisticsAsFloat(int64_t offset, int64_t count);
    __aicore__ inline void ReleaseStatisticsInput();
    __aicore__ inline void StoreFromFloat(int64_t offset, int64_t count);
    __aicore__ inline void ReduceSumSmall(
        const LocalTensor<float>& dst, const LocalTensor<float>& src,
        const LocalTensor<float>& work, int32_t count);
    __aicore__ inline void ReduceSum(
        const LocalTensor<float>& dst, const LocalTensor<float>& src,
        const LocalTensor<float>& work, int32_t count);
    __aicore__ inline float ReadReduceResult(int32_t count);
    __aicore__ inline void KahanAdd(float value, float& sum, float& correction) const;
    __aicore__ inline float ComputeMean(int64_t groupOffset);
    __aicore__ inline float ComputeVariance(int64_t groupOffset, float mean);
    template <int32_t PrefetchMode, bool FirstInputReady, bool PrefetchFirstOutput>
    __aicore__ inline void ComputeStatistics(
        int64_t groupOffset, float& mean, float& variance);
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    __aicore__ inline void ComputeLowATwoByteMoments(float& mean, float& variance);
#endif
    __aicore__ inline void ProcessFullGroup(
        int64_t localGroup, int64_t globalGroup,
        const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal);
    __aicore__ inline void ProcessResidentPair(
        int64_t globalGroup,
        const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal);
    __aicore__ inline void AffineUnalignedResidentSlice(
        const LocalTensor<float>& tensor, int64_t offset, int32_t count,
        float scale, float bias);
    template <bool FirstInputReady>
    __aicore__ inline void NormalizeGroup(
        int64_t localGroup, int64_t globalGroup, float mean, float rstd,
        const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueueX_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQueue_;
    TQue<QuePosition::VECIN, 1> inputQueueGamma_;
    TQue<QuePosition::VECIN, 1> inputQueueBeta_;
    TQue<QuePosition::VECOUT, 1> outputQueueY_;
    TBuf<QuePosition::VECCALC> x1Buf_;
    TBuf<QuePosition::VECCALC> x2Buf_;
    TBuf<QuePosition::VECCALC> reduceBuf_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<float> betaGm_;

    LocalTensor<float> x1Local_;
    LocalTensor<float> x2Local_;
    LocalTensor<float> reduceLocal_;
    LocalTensor<T> inputLocal_;

    const GroupNormTilingData* tiling_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t groupStart_ = 0;
    int64_t outputChannelBegin_ = 0;
    int64_t outputChannelEnd_ = 0;

    constexpr static int64_t ELEMENTS_PER_BLOCK = BLOCK_BYTES / sizeof(T);
};

template <typename T>
__aicore__ inline void GroupNorm<T>::Init(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, const GroupNormTilingData* tilingData)
{
    tiling_ = tilingData;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tiling_->totalGroups * tiling_->outputSplitWorkers) {
        return;
    }
    const int64_t logicalBlockIdx = blockIdx_ / tiling_->outputSplitWorkers;
    groupStart_ = logicalBlockIdx * tiling_->blockFactor;
    outputChannelBegin_ = 0;
    outputChannelEnd_ = tiling_->channelsPerGroup;
    if (tiling_->outputSplitWorkers == 2) {
        const int64_t channelsPerWorker = tiling_->channelsPerGroup / 2;
        outputChannelBegin_ = (blockIdx_ % 2) * channelsPerWorker;
        outputChannelEnd_ = outputChannelBegin_ + channelsPerWorker;
    }
    const int64_t gmOffset = groupStart_ * tiling_->groupElements;

    xGm_.SetGlobalBuffer((__gm__ T*)x + gmOffset);
    yGm_.SetGlobalBuffer((__gm__ T*)y + gmOffset);
    gammaGm_.SetGlobalBuffer((__gm__ float*)gamma);
    betaGm_.SetGlobalBuffer((__gm__ float*)beta);

#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    if constexpr (sizeof(T) == sizeof(float)) {
        pipe_.InitBuffer(inOutQueue_, 2, tiling_->outputFactor * sizeof(T));
    } else {
#endif
        const int32_t inputQueueDepth = tiling_->statisticsPrefetchMode == 2 ? 2 : 1;
        pipe_.InitBuffer(inputQueueX_, inputQueueDepth, tiling_->outputFactor * sizeof(T));
        pipe_.InitBuffer(outputQueueY_, 1, tiling_->outputFactor * sizeof(T));
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    }
#endif
    pipe_.InitBuffer(inputQueueGamma_, 1, tiling_->channelsAlign * sizeof(float));
    pipe_.InitBuffer(inputQueueBeta_, 1, tiling_->channelsAlign * sizeof(float));
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    if constexpr (sizeof(T) != sizeof(float)) {
#endif
        pipe_.InitBuffer(x1Buf_, tiling_->outputFactor * sizeof(float));
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    }
#endif
    pipe_.InitBuffer(x2Buf_, Min(tiling_->ubFactor, REDUCE_MAX_ELEMENTS) * sizeof(float));
    pipe_.InitBuffer(reduceBuf_, BLOCK_BYTES);

#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    if constexpr (sizeof(T) != sizeof(float)) {
#endif
        x1Local_ = x1Buf_.Get<float>();
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    }
#endif
    x2Local_ = x2Buf_.Get<float>();
    reduceLocal_ = reduceBuf_.Get<float>();
}

template <typename T>
__aicore__ inline void GroupNorm<T>::CopyInAffine()
{
    LocalTensor<float> gammaLocal = inputQueueGamma_.AllocTensor<float>();
    LocalTensor<float> betaLocal = inputQueueBeta_.AllocTensor<float>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    DataCopyPad(gammaLocal, gammaGm_,
        {1, static_cast<uint16_t>(tiling_->channels * sizeof(float)), 0, 0}, {false, 0, 0, 0});
    DataCopyPad(betaLocal, betaGm_,
        {1, static_cast<uint16_t>(tiling_->channels * sizeof(float)), 0, 0}, {false, 0, 0, 0});
#else
    DataCopy(gammaLocal, gammaGm_, tiling_->channelsAlign);
    DataCopy(betaLocal, betaGm_, tiling_->channelsAlign);
#endif
    inputQueueGamma_.EnQue(gammaLocal);
    inputQueueBeta_.EnQue(betaLocal);
}

template <typename T>
__aicore__ inline void GroupNorm<T>::CopyInX(int64_t offset, int64_t count)
{
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    LocalTensor<T> xLocal;
    if constexpr (sizeof(T) == sizeof(float)) {
        xLocal = inOutQueue_.AllocTensor<T>();
    } else {
        xLocal = inputQueueX_.AllocTensor<T>();
    }
#else
    LocalTensor<T> xLocal = inputQueueX_.AllocTensor<T>();
#endif
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    if constexpr (sizeof(T) == sizeof(float)) {
        if (count == REDUCE_DOUBLE_ELEMENTS) {
            // 16K FP32 is 65536B, one byte beyond uint16 blockLen.  Express
            // the same contiguous transfer as two legal 32KiB blocks.
            DataCopyPad(xLocal, xGm_[offset],
                {2, static_cast<uint16_t>(32768), 0, 0}, {false, 0, 0, 0});
        } else {
            DataCopyPad(xLocal, xGm_[offset],
                {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0}, {false, 0, 0, 0});
        }
    } else {
        DataCopyPad(xLocal, xGm_[offset],
            {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0}, {false, 0, 0, 0});
    }
#else
    const int64_t alignedCount = CeilDiv(count, ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    DataCopy(xLocal, xGm_[offset], alignedCount);
#endif
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    if constexpr (sizeof(T) == sizeof(float)) {
        inOutQueue_.template EnQue<QuePosition::GM, QuePosition::VECIN, T>(xLocal);
    } else {
        inputQueueX_.EnQue(xLocal);
    }
#else
    inputQueueX_.EnQue(xLocal);
#endif
}

template <typename T>
__aicore__ inline void GroupNorm<T>::CopyOutY(int64_t offset, int64_t count)
{
    LocalTensor<T> yLocal = outputQueueY_.DeQue<T>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    DataCopyPad(yGm_[offset], yLocal,
        {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0});
#else
    const int64_t alignedCount = CeilDiv(count, ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
    if (count == alignedCount) {
        DataCopy(yGm_[offset], yLocal, count);
    } else {
        const int64_t prefix = alignedCount - ELEMENTS_PER_BLOCK;
        if (prefix > 0) {
            DataCopy(yGm_[offset], yLocal, prefix);
        }
        SetAtomicAdd<T>();
        DataCopy(yGm_[offset + prefix], yLocal[prefix], ELEMENTS_PER_BLOCK);
        SetAtomicNone();
    }
#endif
    outputQueueY_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void GroupNorm<T>::LoadAsFloat(int64_t offset, int64_t count)
{
    CopyInX(offset, count);
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    if constexpr (sizeof(T) == sizeof(float)) {
        inputLocal_ = inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
        x1Local_ = inputLocal_;
        return;
    }
#endif
    LocalTensor<T> xLocal = inputQueueX_.DeQue<T>();
    if constexpr (sizeof(T) == sizeof(float)) {
        Adds(x1Local_, xLocal, 0.0f, count);
    } else {
        Cast(x1Local_, xLocal, RoundMode::CAST_NONE, count);
    }
    PipeBarrier<PIPE_V>();
    inputQueueX_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void GroupNorm<T>::LoadPrefetchedAsFloat(int64_t count)
{
    LocalTensor<T> xLocal = inputQueueX_.DeQue<T>();
    Cast(x1Local_, xLocal, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    inputQueueX_.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline int64_t GroupNorm<T>::FirstOutputCount() const
{
    if (tiling_->spatial % (BLOCK_BYTES / sizeof(float)) == 0 &&
        tiling_->spatial < tiling_->outputFactor) {
        const int64_t channelsPerTile = tiling_->outputFactor / tiling_->spatial;
        return Min(channelsPerTile, tiling_->channelsPerGroup) * tiling_->spatial;
    }
    return Min(tiling_->outputFactor, tiling_->spatial);
}

template <typename T>
__aicore__ inline void GroupNorm<T>::LoadStatisticsAsFloat(int64_t offset, int64_t count)
{
    if constexpr (sizeof(T) == sizeof(float)) {
        CopyInX(offset, count);
        inputLocal_ = inputQueueX_.DeQue<T>();
        x1Local_ = inputLocal_;
    } else {
        LoadAsFloat(offset, count);
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::ReleaseStatisticsInput()
{
    if constexpr (sizeof(T) == sizeof(float)) {
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
        inOutQueue_.FreeTensor(inputLocal_);
#else
        inputQueueX_.FreeTensor(inputLocal_);
        x1Local_ = x1Buf_.Get<float>();
#endif
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::StoreFromFloat(int64_t offset, int64_t count)
{
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    if constexpr (sizeof(T) == sizeof(float)) {
        inOutQueue_.template EnQue<QuePosition::VECOUT, QuePosition::GM, T>(inputLocal_);
        LocalTensor<T> yLocal =
            inOutQueue_.template DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
#if __CCE_AICORE__ == 220 || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
        if (count == REDUCE_DOUBLE_ELEMENTS) {
            DataCopyPad(yGm_[offset], yLocal,
                {2, static_cast<uint16_t>(32768), 0, 0});
        } else {
            DataCopyPad(yGm_[offset], yLocal,
                {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0});
        }
#else
        const int64_t alignedCount = CeilDiv(count, ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
        if (count == alignedCount) {
            DataCopy(yGm_[offset], yLocal, count);
        } else {
            const int64_t prefix = alignedCount - ELEMENTS_PER_BLOCK;
            if (prefix > 0) {
                DataCopy(yGm_[offset], yLocal, prefix);
            }
            SetAtomicAdd<T>();
            DataCopy(yGm_[offset + prefix], yLocal[prefix], ELEMENTS_PER_BLOCK);
            SetAtomicNone();
        }
#endif
        inOutQueue_.FreeTensor(yLocal);
        return;
    }
#endif
    LocalTensor<T> yLocal = outputQueueY_.AllocTensor<T>();
    if constexpr (sizeof(T) == sizeof(float)) {
        Adds(yLocal, x1Local_, 0.0f, count);
    } else {
        Cast(yLocal, x1Local_, OutputRoundMode(), count);
    }
    PipeBarrier<PIPE_V>();
    outputQueueY_.EnQue(yLocal);
    CopyOutY(offset, count);
}

template <typename T>
__aicore__ inline void GroupNorm<T>::ReduceSumSmall(
    const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& work, int32_t count)
{
    const int32_t repeat = count / 64;
    const int32_t tail = count % 64;
    if (repeat > 0) {
        WholeReduceSum(work, src, 64, repeat, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }
    if (tail != 0) {
        WholeReduceSum(work[repeat], src[count - tail], tail, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dst, work, repeat + 1, 1, 1, 1, 8);
    } else {
        WholeReduceSum(dst, work, repeat, 1, 1, 1, 8);
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::ReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src,
    const LocalTensor<float>& work, int32_t count)
{
    if (tiling_->groupElements <= REDUCE_MAX_ELEMENTS) {
        // Whole-group resident inputs use the vendor reduction directly.
        // Keep the hand-tuned tree for large-group full/tail tiles, where the
        // generic API previously regressed MTE/Vector-heavy cases.
        AscendC::ReduceSum<float, true>(dst, src, work, count);
        PipeBarrier<PIPE_V>();
        return;
    }
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    if constexpr (sizeof(T) == sizeof(float)) {
        if (count == REDUCE_DOUBLE_ELEMENTS) {
            // Keep both 8K first-level trees in the existing 8K scratch and
            // reduce their concatenated partials together.  Each 8K half
            // produces 1024 FP32 partial elements, so the two ranges are
            // disjoint and occupy only the first quarter of `work`.
            constexpr int32_t HALF_PARTIAL_ELEMENTS =
                REDUCE_MAX_ELEMENTS / (BLOCK_BYTES / sizeof(float));
            BlockReduceSum(work, src, 128, 64, 1, 1, 8);
            BlockReduceSum(work[HALF_PARTIAL_ELEMENTS], src[REDUCE_MAX_ELEMENTS],
                128, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            BlockReduceSum(work, work, 32, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(dst, work, 64, 4, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            return;
        }
    }
#endif
    if (count > REDUCE_MAX_ELEMENTS) {
        BlockReduceSum(work, src, 128, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        BlockReduceSum(work, work, 16, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dst, work, 64, 2, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        const int32_t tailCount = count - REDUCE_MAX_ELEMENTS;
        if (tailCount == REDUCE_MAX_ELEMENTS) {
            BlockReduceSum(work, src[REDUCE_MAX_ELEMENTS], 128, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            BlockReduceSum(work, work, 16, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(dst[2], work, 64, 2, 1, 1, 8);
        } else if (tailCount <= REDUCE_HALF_ELEMENTS) {
            dst(3) = 0.0f;
            ReduceSumSmall(dst[2], src[REDUCE_MAX_ELEMENTS], work, tailCount);
        } else {
            BlockReduceSum(work, src[REDUCE_MAX_ELEMENTS], 64, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            BlockReduceSum(work, work, 8, 64, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            WholeReduceSum(dst[2], work, 64, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            ReduceSumSmall(dst[3], src[REDUCE_MAX_ELEMENTS + REDUCE_HALF_ELEMENTS],
                work, tailCount - REDUCE_HALF_ELEMENTS);
        }
    } else if (count == REDUCE_MAX_ELEMENTS) {
        BlockReduceSum(work, src, 128, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        BlockReduceSum(work, work, 16, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dst, work, 64, 2, 1, 1, 8);
    } else if (count <= REDUCE_HALF_ELEMENTS) {
        ReduceSumSmall(dst, src, work, count);
    } else {
        BlockReduceSum(work, src, 64, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        BlockReduceSum(work, work, 8, 64, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(dst, work, 64, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        ReduceSumSmall(dst[1], src[REDUCE_HALF_ELEMENTS], work, count - REDUCE_HALF_ELEMENTS);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline float GroupNorm<T>::ReadReduceResult(int32_t count)
{
    const event_t eventId = static_cast<event_t>(0);
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);
    const float first = reduceLocal_.GetValue(0);
    if (tiling_->groupElements <= REDUCE_MAX_ELEMENTS) {
        return first;
    }
    if (count <= REDUCE_HALF_ELEMENTS) {
        return first;
    }
    const float firstPair = first + reduceLocal_.GetValue(1);
    return count > REDUCE_MAX_ELEMENTS ?
        firstPair + reduceLocal_.GetValue(2) + reduceLocal_.GetValue(3) : firstPair;
}

template <typename T>
__aicore__ inline void GroupNorm<T>::KahanAdd(float value, float& sum, float& correction) const
{
    const float adjusted = value - correction;
    const float next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
}

template <typename T>
__aicore__ inline float GroupNorm<T>::ComputeMean(int64_t groupOffset)
{
    float sum = 0.0f;
    float correction = 0.0f;
    for (int64_t offset = 0; offset < tiling_->groupElements; offset += tiling_->ubFactor) {
        const int64_t count = Min(tiling_->ubFactor, tiling_->groupElements - offset);
        LoadAsFloat(groupOffset + offset, count);
        ReduceSum(reduceLocal_, x1Local_, x2Local_, static_cast<int32_t>(count));
        const float tileSum = ReadReduceResult(static_cast<int32_t>(count));
        KahanAdd(tileSum, sum, correction);
    }
    return sum / static_cast<float>(tiling_->groupElements);
}

template <typename T>
__aicore__ inline float GroupNorm<T>::ComputeVariance(int64_t groupOffset, float mean)
{
    float sum = 0.0f;
    float correction = 0.0f;
    for (int64_t offset = 0; offset < tiling_->groupElements; offset += tiling_->ubFactor) {
        const int64_t count = Min(tiling_->ubFactor, tiling_->groupElements - offset);
        LoadAsFloat(groupOffset + offset, count);
        Adds(x2Local_, x1Local_, -mean, count);
        PipeBarrier<PIPE_V>();
        Mul(x2Local_, x2Local_, x2Local_, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal_, x2Local_, x1Local_, static_cast<int32_t>(count));
        const float tileSum = ReadReduceResult(static_cast<int32_t>(count));
        KahanAdd(tileSum, sum, correction);
    }
    return sum / static_cast<float>(tiling_->groupElements);
}

template <typename T>
template <int32_t PrefetchMode, bool FirstInputReady, bool PrefetchFirstOutput>
__aicore__ inline void GroupNorm<T>::ComputeStatistics(
    int64_t groupOffset, float& mean, float& variance)
{
    float sum = 0.0f;
    float correction = 0.0f;
    float mergedVariance = 0.0f;
    int64_t processed = 0;
    if constexpr ((PrefetchMode == 2 && sizeof(T) == sizeof(float)) ||
                  (PrefetchMode == 3 && sizeof(T) != sizeof(float))) {
        if constexpr (!FirstInputReady) {
            const int64_t firstCount = Min(tiling_->ubFactor, tiling_->groupElements);
            CopyInX(groupOffset, firstCount);
        }
    }
    for (int64_t offset = 0; offset < tiling_->groupElements; offset += tiling_->ubFactor) {
        const int64_t count = Min(tiling_->ubFactor, tiling_->groupElements - offset);
        if constexpr (PrefetchMode == 2 && sizeof(T) == sizeof(float)) {
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
            inputLocal_ =
                inOutQueue_.template DeQue<QuePosition::GM, QuePosition::VECIN, T>();
#else
            inputLocal_ = inputQueueX_.DeQue<T>();
#endif
            x1Local_ = inputLocal_;
            const int64_t nextOffset = offset + tiling_->ubFactor;
            if (nextOffset < tiling_->groupElements) {
                const int64_t nextCount = Min(
                    tiling_->ubFactor, tiling_->groupElements - nextOffset);
                CopyInX(groupOffset + nextOffset, nextCount);
            }
        } else if constexpr (PrefetchMode == 3 && sizeof(T) != sizeof(float)) {
            inputLocal_ = inputQueueX_.DeQue<T>();
            Cast(x1Local_, inputLocal_, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            inputQueueX_.FreeTensor(inputLocal_);
            const int64_t nextOffset = offset + tiling_->ubFactor;
            if (nextOffset < tiling_->groupElements) {
                const int64_t nextCount = Min(
                    tiling_->ubFactor, tiling_->groupElements - nextOffset);
                CopyInX(groupOffset + nextOffset, nextCount);
            } else if constexpr (PrefetchFirstOutput) {
                // The mode-3 input slot is free after the final Cast.  Refill
                // it with the first normalization tile while the final FP32
                // reductions are still running, without allocating more UB.
                CopyInX(groupOffset, FirstOutputCount());
            }
        } else {
            LoadStatisticsAsFloat(groupOffset + offset, count);
        }

        ReduceSum(reduceLocal_, x1Local_, x2Local_, static_cast<int32_t>(count));
        const float tileSum = ReadReduceResult(static_cast<int32_t>(count));
        const float tileMean = tileSum / static_cast<float>(count);
        Adds(x1Local_, x1Local_, -tileMean, count);
        PipeBarrier<PIPE_V>();
        Mul(x1Local_, x1Local_, x1Local_, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal_, x1Local_, x2Local_, static_cast<int32_t>(count));
        const float tileVariance = ReadReduceResult(static_cast<int32_t>(count)) /
            static_cast<float>(count);
        ReleaseStatisticsInput();

        if (processed == 0) {
            mergedVariance = tileVariance;
        } else {
            const float newCount = static_cast<float>(processed + count);
            const float weightA = static_cast<float>(processed) / newCount;
            const float weightB = static_cast<float>(count) / newCount;
            const float meanA = sum / static_cast<float>(processed);
            const float delta = meanA - tileMean;
            mergedVariance = mergedVariance * weightA + tileVariance * weightB +
                weightA * weightB * delta * delta;
        }
        KahanAdd(tileSum, sum, correction);
        processed += count;
    }
    mean = sum / static_cast<float>(tiling_->groupElements);
    variance = mergedVariance;
}

#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
template <typename T>
__aicore__ inline void GroupNorm<T>::ComputeLowATwoByteMoments(
    float& mean, float& variance)
{
    float anchor = 0.0f;
    bool anchorReady = false;
    float shiftedSum = 0.0f;
    float shiftedSumCorrection = 0.0f;
    float shiftedSquareSum = 0.0f;
    float shiftedSquareCorrection = 0.0f;
    for (int64_t offset = 0; offset < tiling_->groupElements; offset += tiling_->ubFactor) {
        const int64_t count = Min(tiling_->ubFactor, tiling_->groupElements - offset);
        inputLocal_ = inputQueueX_.DeQue<T>();
        Cast(x1Local_, inputLocal_, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        inputQueueX_.FreeTensor(inputLocal_);

        const int64_t nextOffset = offset + tiling_->ubFactor;
        if (nextOffset < tiling_->groupElements) {
            const int64_t nextCount = Min(
                tiling_->ubFactor, tiling_->groupElements - nextOffset);
            CopyInX(nextOffset, nextCount);
        } else {
            CopyInX(0, FirstOutputCount());
        }

        if (!anchorReady) {
            // Reuse the Cast-completion event and sample the first tile at
            // uniform offsets.  This is more robust to one arbitrary outlier
            // than x[0], without adding a Vector reduction or GM traffic.
            const event_t anchorEventId = static_cast<event_t>(0);
            SetFlag<HardEvent::V_S>(anchorEventId);
            WaitFlag<HardEvent::V_S>(anchorEventId);
            const int64_t sampleStride = count / 8;
            float anchorSum = 0.0f;
            for (int32_t sample = 0; sample < 8; ++sample) {
                anchorSum += x1Local_.GetValue(sample * sampleStride);
            }
            anchor = anchorSum * 0.125f;
            anchorReady = true;
        }

        // Reduce z=x-anchor instead of raw x.  The shift preserves variance
        // accuracy for large-offset inputs while two reduction results still
        // share one Vector-to-Scalar publication per tile.
        Adds(x1Local_, x1Local_, -anchor, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal_, x1Local_, x2Local_, static_cast<int32_t>(count));
        Mul(x1Local_, x1Local_, x1Local_, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal_[4], x1Local_, x2Local_, static_cast<int32_t>(count));

        const event_t eventId = static_cast<event_t>(0);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        float tileSum = reduceLocal_.GetValue(0);
        float tileSquareSum = reduceLocal_.GetValue(4);
        if (count > REDUCE_HALF_ELEMENTS) {
            tileSum += reduceLocal_.GetValue(1);
            tileSquareSum += reduceLocal_.GetValue(5);
        }
        if (count > REDUCE_MAX_ELEMENTS) {
            tileSum += reduceLocal_.GetValue(2) + reduceLocal_.GetValue(3);
            tileSquareSum += reduceLocal_.GetValue(6) + reduceLocal_.GetValue(7);
        }
        KahanAdd(tileSum, shiftedSum, shiftedSumCorrection);
        KahanAdd(tileSquareSum, shiftedSquareSum, shiftedSquareCorrection);
    }
    const float invCount = 1.0f / static_cast<float>(tiling_->groupElements);
    const float shiftedMean = shiftedSum * invCount;
    mean = anchor + shiftedMean;
    variance = shiftedSquareSum * invCount - shiftedMean * shiftedMean;
    if (variance < 0.0f) {
        variance = 0.0f;
    }
}
#endif

template <typename T>
__aicore__ inline void GroupNorm<T>::AffineUnalignedResidentSlice(
    const LocalTensor<float>& tensor, int64_t offset, int32_t count,
    float scale, float bias)
{
    constexpr int32_t FP32_ELEMENTS_PER_BLOCK = BLOCK_BYTES / sizeof(float);
    constexpr int32_t FP32_ELEMENTS_PER_REPEAT = 256 / sizeof(float);
    const int32_t leading = static_cast<int32_t>(offset % FP32_ELEMENTS_PER_BLOCK);
    if (leading == 0) {
        Muls(tensor[offset], tensor[offset], scale, count);
        Adds(tensor[offset], tensor[offset], bias, count);
        return;
    }

    // Vector sources must start on a 32-byte boundary.  Cover the unaligned
    // channel head from the preceding aligned address with a bit mask, then
    // continue from the next 256-byte repeat boundary with the count API.
    const int64_t alignedOffset = offset - leading;
    const int32_t headCount = Min(count, FP32_ELEMENTS_PER_REPEAT - leading);
    const uint64_t headBits = (static_cast<uint64_t>(1) << headCount) - 1;
    uint64_t headMask[2] = {headBits << leading, 0};
    const UnaryRepeatParams repeatParams(1, 1, 8, 8);
    Muls(tensor[alignedOffset], tensor[alignedOffset], scale, headMask, 1, repeatParams);
    Adds(tensor[alignedOffset], tensor[alignedOffset], bias, headMask, 1, repeatParams);

    const int32_t tailCount = count - headCount;
    if (tailCount > 0) {
        const int64_t tailOffset = offset + headCount;
        Muls(tensor[tailOffset], tensor[tailOffset], scale, tailCount);
        Adds(tensor[tailOffset], tensor[tailOffset], bias, tailCount);
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::ProcessFullGroup(
    int64_t localGroup, int64_t globalGroup,
    const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal)
{
    const int64_t count = tiling_->groupElements;
    const int64_t groupOffset = localGroup * count;
    LoadAsFloat(groupOffset, count);

    // Keep the original FP32 input in x1.  x2 is first the reduction
    // workspace, then the centered square buffer and its in-place workspace.
    ReduceSum(reduceLocal_, x1Local_, x2Local_, static_cast<int32_t>(count));
    const float mean = ReadReduceResult(static_cast<int32_t>(count)) / static_cast<float>(count);
    Adds(x2Local_, x1Local_, -mean, count);
    PipeBarrier<PIPE_V>();
    Mul(x2Local_, x2Local_, x2Local_, count);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduceLocal_, x2Local_, x2Local_, static_cast<int32_t>(count));
    const float variance = ReadReduceResult(static_cast<int32_t>(count)) / static_cast<float>(count);
    const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);

    // A local tensor slice used by vector instructions must start at a 32-byte
    // boundary.  When one channel is FP32-aligned, normalize the resident group
    // in place.  Otherwise retain the one-load statistics path and reuse the
    // aligned per-channel output path.
    if (tiling_->spatial % (BLOCK_BYTES / sizeof(float)) == 0) {
        const int64_t channelStart =
            (globalGroup % tiling_->numGroups) * tiling_->channelsPerGroup;
        for (int64_t channel = 0; channel < tiling_->channelsPerGroup; ++channel) {
            const float gamma = gammaLocal.GetValue(channelStart + channel);
            const float beta = betaLocal.GetValue(channelStart + channel);
            const float scale = rstd * gamma;
            const float bias = beta - mean * scale;
            const int64_t channelOffset = channel * tiling_->spatial;
            Muls(x1Local_[channelOffset], x1Local_[channelOffset], scale, tiling_->spatial);
            Adds(x1Local_[channelOffset], x1Local_[channelOffset], bias, tiling_->spatial);
        }
        PipeBarrier<PIPE_V>();
        StoreFromFloat(groupOffset, count);
    } else if constexpr (sizeof(T) == sizeof(uint16_t)) {
        // Resident two-byte groups already keep the original FP32 input in
        // x1Local_.  Preserve that load for non-aligned channels as well:
        // masked aligned heads avoid illegal UB addresses, and one contiguous
        // output Cast/copy replaces the padded GM reload plus per-row Casts.
        const int64_t channelStart =
            (globalGroup % tiling_->numGroups) * tiling_->channelsPerGroup;
        for (int64_t channel = 0; channel < tiling_->channelsPerGroup; ++channel) {
            const float gamma = gammaLocal.GetValue(channelStart + channel);
            const float beta = betaLocal.GetValue(channelStart + channel);
            const float scale = rstd * gamma;
            const float bias = beta - mean * scale;
            AffineUnalignedResidentSlice(
                x1Local_, channel * tiling_->spatial,
                static_cast<int32_t>(tiling_->spatial), scale, bias);
        }
        PipeBarrier<PIPE_V>();
        StoreFromFloat(groupOffset, count);
    } else {
        NormalizeGroup<false>(localGroup, globalGroup, mean, rstd, gammaLocal, betaLocal);
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::ProcessResidentPair(
    int64_t globalGroup,
    const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal)
{
    const int64_t count = tiling_->groupElements;
    const int64_t pairCount = count * 2;
    LoadAsFloat(0, pairCount);

    ReduceSum(reduceLocal_, x1Local_, x2Local_, static_cast<int32_t>(count));
    const float mean0 =
        ReadReduceResult(static_cast<int32_t>(count)) / static_cast<float>(count);
    ReduceSum(reduceLocal_, x1Local_[count], x2Local_, static_cast<int32_t>(count));
    const float mean1 =
        ReadReduceResult(static_cast<int32_t>(count)) / static_cast<float>(count);

    Adds(x2Local_, x1Local_, -mean0, count);
    Adds(x2Local_[count], x1Local_[count], -mean1, count);
    PipeBarrier<PIPE_V>();
    Mul(x2Local_, x2Local_, x2Local_, pairCount);
    PipeBarrier<PIPE_V>();

    ReduceSum(reduceLocal_, x2Local_, x2Local_, static_cast<int32_t>(count));
    const float variance0 =
        ReadReduceResult(static_cast<int32_t>(count)) / static_cast<float>(count);
    ReduceSum(reduceLocal_, x2Local_[count], x2Local_, static_cast<int32_t>(count));
    const float variance1 =
        ReadReduceResult(static_cast<int32_t>(count)) / static_cast<float>(count);
    const float rstd0 = 1.0f / sqrt(variance0 + tiling_->epsilon);
    const float rstd1 = 1.0f / sqrt(variance1 + tiling_->epsilon);

    const int64_t channel0 = globalGroup % tiling_->numGroups;
    const int64_t channel1 = (globalGroup + 1) % tiling_->numGroups;
    const float gamma0 = gammaLocal.GetValue(channel0);
    const float gamma1 = gammaLocal.GetValue(channel1);
    const float scale0 = rstd0 * gamma0;
    const float scale1 = rstd1 * gamma1;
    Muls(x1Local_, x1Local_, scale0, count);
    Adds(x1Local_, x1Local_, betaLocal.GetValue(channel0) - mean0 * scale0, count);
    Muls(x1Local_[count], x1Local_[count], scale1, count);
    Adds(x1Local_[count], x1Local_[count],
        betaLocal.GetValue(channel1) - mean1 * scale1, count);
    PipeBarrier<PIPE_V>();
    StoreFromFloat(0, pairCount);
}

template <typename T>
template <bool FirstInputReady>
__aicore__ inline void GroupNorm<T>::NormalizeGroup(
    int64_t localGroup, int64_t globalGroup, float mean, float rstd,
    const LocalTensor<float>& gammaLocal, const LocalTensor<float>& betaLocal)
{
    const int64_t channelStart =
        (globalGroup % tiling_->numGroups) * tiling_->channelsPerGroup;
    const int64_t groupOffset = localGroup * tiling_->groupElements;
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        // A non-aligned channel normally pays one GM->UB and one UB->GM queue
        // transaction per channel.  Batch consecutive channels with
        // DataCopyPad's row layout: every UB row is 32-byte aligned while the
        // effective vector count and per-channel affine order stay unchanged.
        if (tiling_->spatial % ELEMENTS_PER_BLOCK != 0 &&
            tiling_->spatial < tiling_->outputFactor) {
            const int64_t rowStride =
                CeilDiv(tiling_->spatial, ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK;
            const int64_t channelsPerTile = tiling_->outputFactor / rowStride;
            int64_t channelBase = 0;
            if constexpr (FirstInputReady) {
                // The low-A statistics path already prefetched channel zero.
                LoadPrefetchedAsFloat(tiling_->spatial);
                const float gamma = gammaLocal.GetValue(channelStart);
                const float beta = betaLocal.GetValue(channelStart);
                const float scale = rstd * gamma;
                const float bias = beta - mean * scale;
                Muls(x1Local_, x1Local_, scale, tiling_->spatial);
                Adds(x1Local_, x1Local_, bias, tiling_->spatial);
                PipeBarrier<PIPE_V>();
                if (tiling_->channelsPerGroup == 1) {
                    StoreFromFloat(groupOffset, tiling_->spatial);
                    return;
                }
                if (channelsPerTile == 1) {
                    StoreFromFloat(groupOffset, tiling_->spatial);
                }
                channelBase = 1;
            }
            for (; channelBase < tiling_->channelsPerGroup;) {
                const int64_t localRowBase =
                    FirstInputReady && channelsPerTile > 1 && channelBase == 1 ? 1 : 0;
                const int64_t channelCount = Min(
                    channelsPerTile - localRowBase,
                    tiling_->channelsPerGroup - channelBase);
                const int64_t channelOffset = groupOffset + channelBase * tiling_->spatial;
                DataCopyExtParams copyParams;
                copyParams.blockCount = static_cast<uint16_t>(channelCount);
                copyParams.blockLen = static_cast<uint32_t>(tiling_->spatial * sizeof(T));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<T> padParams;
                padParams.isPad = true;
                padParams.leftPadding = 0;
                padParams.rightPadding = static_cast<uint8_t>(rowStride - tiling_->spatial);
                padParams.paddingValue = static_cast<T>(0);

                LocalTensor<T> xBatch = inputQueueX_.AllocTensor<T>();
                DataCopyPad(xBatch, xGm_[channelOffset], copyParams, padParams);
                inputQueueX_.EnQue(xBatch);
                xBatch = inputQueueX_.DeQue<T>();
                const int64_t paddedCount = channelCount * rowStride;
                if constexpr (FirstInputReady) {
                    Cast(x1Local_[localRowBase * rowStride], xBatch,
                        RoundMode::CAST_NONE, paddedCount);
                } else {
                    for (int64_t channel = 0; channel < channelCount; ++channel) {
                        const int64_t sourceOffset = channel * rowStride;
                        const int64_t localOffset = channel * rowStride;
                        Cast(x1Local_[localOffset], xBatch[sourceOffset],
                            RoundMode::CAST_NONE, tiling_->spatial);
                    }
                }
                PipeBarrier<PIPE_V>();
                inputQueueX_.FreeTensor(xBatch);

                for (int64_t channel = 0; channel < channelCount; ++channel) {
                    const float gamma =
                        gammaLocal.GetValue(channelStart + channelBase + channel);
                    const float beta =
                        betaLocal.GetValue(channelStart + channelBase + channel);
                    const float scale = rstd * gamma;
                    const float bias = beta - mean * scale;
                    const int64_t localOffset = (localRowBase + channel) * rowStride;
                    Muls(x1Local_[localOffset], x1Local_[localOffset],
                        scale, tiling_->spatial);
                    Adds(x1Local_[localOffset], x1Local_[localOffset],
                        bias, tiling_->spatial);
                }
                PipeBarrier<PIPE_V>();

                LocalTensor<T> yBatch = outputQueueY_.AllocTensor<T>();
                const int64_t outputCount = (localRowBase + channelCount) * rowStride;
                if constexpr (FirstInputReady) {
                    Cast(yBatch, x1Local_, OutputRoundMode(), outputCount);
                } else {
                    for (int64_t channel = 0; channel < channelCount; ++channel) {
                        const int64_t localOffset = channel * rowStride;
                        Cast(yBatch[localOffset], x1Local_[localOffset],
                            OutputRoundMode(), tiling_->spatial);
                    }
                }
                PipeBarrier<PIPE_V>();
                outputQueueY_.EnQue(yBatch);
                yBatch = outputQueueY_.DeQue<T>();
                DataCopyExtParams outputParams = copyParams;
                outputParams.blockCount =
                    static_cast<uint16_t>(localRowBase + channelCount);
                const int64_t outputOffset =
                    localRowBase == 0 ? channelOffset : groupOffset;
                DataCopyPad(yGm_[outputOffset], yBatch, outputParams);
                outputQueueY_.FreeTensor(yBatch);
                channelBase += channelCount;
            }
            return;
        }
    }
    if (tiling_->spatial % (BLOCK_BYTES / sizeof(float)) == 0 &&
        tiling_->spatial < tiling_->outputFactor) {
        const int64_t channelsPerTile = tiling_->outputFactor / tiling_->spatial;
        for (int64_t channelBase = 0; channelBase < tiling_->channelsPerGroup;
             channelBase += channelsPerTile) {
            const int64_t channelCount = Min(
                channelsPerTile, tiling_->channelsPerGroup - channelBase);
            const int64_t count = channelCount * tiling_->spatial;
            const int64_t channelOffset = groupOffset + channelBase * tiling_->spatial;
            if constexpr (FirstInputReady) {
                if (channelBase == 0) {
                    LoadPrefetchedAsFloat(count);
                } else {
                    LoadAsFloat(channelOffset, count);
                }
            } else {
                LoadAsFloat(channelOffset, count);
            }
            for (int64_t channel = 0; channel < channelCount; ++channel) {
                const float gamma = gammaLocal.GetValue(channelStart + channelBase + channel);
                const float beta = betaLocal.GetValue(channelStart + channelBase + channel);
                const float scale = rstd * gamma;
                const float bias = beta - mean * scale;
                const int64_t localOffset = channel * tiling_->spatial;
                Muls(x1Local_[localOffset], x1Local_[localOffset], scale, tiling_->spatial);
                Adds(x1Local_[localOffset], x1Local_[localOffset], bias, tiling_->spatial);
            }
            PipeBarrier<PIPE_V>();
            StoreFromFloat(channelOffset, count);
        }
        return;
    }
    for (int64_t channel = outputChannelBegin_; channel < outputChannelEnd_; ++channel) {
        const float gamma = gammaLocal.GetValue(channelStart + channel);
        const float beta = betaLocal.GetValue(channelStart + channel);
        const float scale = rstd * gamma;
        const float bias = beta - mean * scale;
        const int64_t channelOffset = groupOffset + channel * tiling_->spatial;
        for (int64_t offset = 0; offset < tiling_->spatial; offset += tiling_->outputFactor) {
            const int64_t count = Min(tiling_->outputFactor, tiling_->spatial - offset);
            if constexpr (FirstInputReady) {
                if (channel == 0 && offset == 0) {
                    LoadPrefetchedAsFloat(count);
                } else {
                    LoadAsFloat(channelOffset + offset, count);
                }
            } else {
                LoadAsFloat(channelOffset + offset, count);
            }
            Muls(x1Local_, x1Local_, scale, count);
            Adds(x1Local_, x1Local_, bias, count);
            PipeBarrier<PIPE_V>();
            StoreFromFloat(channelOffset + offset, count);
        }
    }
}

template <typename T>
__aicore__ inline void GroupNorm<T>::Process()
{
    if (blockIdx_ >= tiling_->totalGroups * tiling_->outputSplitWorkers) {
        return;
    }
    CopyInAffine();
    LocalTensor<float> gammaLocal = inputQueueGamma_.DeQue<float>();
    LocalTensor<float> betaLocal = inputQueueBeta_.DeQue<float>();

    const int64_t groupCount = tiling_->outputSplitWorkers == 2 ? 1 :
        ((blockIdx_ == GetBlockNum() - 1) ?
            tiling_->lastCoreGroups : tiling_->blockFactor);
    if (tiling_->groupElements <= REDUCE_MAX_ELEMENTS) {
        if constexpr (sizeof(T) == sizeof(float)) {
            // Fuse only the two already-owned adjacent single-channel groups.
            // Per-group reductions and formulas remain independent; the pair
            // shares the existing UB allocation and one GM queue transaction.
            if (groupCount == 2 && tiling_->channelsPerGroup == 1 &&
                tiling_->spatial % ELEMENTS_PER_BLOCK == 0 &&
                tiling_->groupElements * 2 <= tiling_->outputFactor) {
                ProcessResidentPair(groupStart_, gammaLocal, betaLocal);
                inputQueueGamma_.FreeTensor(gammaLocal);
                inputQueueBeta_.FreeTensor(betaLocal);
                return;
            }
        }
        for (int64_t localGroup = 0; localGroup < groupCount; ++localGroup) {
            ProcessFullGroup(localGroup, groupStart_ + localGroup, gammaLocal, betaLocal);
        }
    } else {
        for (int64_t localGroup = 0; localGroup < groupCount; ++localGroup) {
            const int64_t groupOffset = localGroup * tiling_->groupElements;
            float mean = 0.0f;
            float variance = 0.0f;
            if (tiling_->statisticsPrefetchMode == 2) {
                ComputeStatistics<2, false, false>(groupOffset, mean, variance);
            } else if (tiling_->statisticsPrefetchMode == 3) {
                ComputeStatistics<3, false, false>(groupOffset, mean, variance);
            } else {
                ComputeStatistics<1, false, false>(groupOffset, mean, variance);
            }
            const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);
            NormalizeGroup<false>(
                localGroup, groupStart_ + localGroup, mean, rstd, gammaLocal, betaLocal);
        }
    }

    inputQueueGamma_.FreeTensor(gammaLocal);
    inputQueueBeta_.FreeTensor(betaLocal);
}

#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_3
template <typename T>
__aicore__ inline void GroupNorm<T>::ProcessTwoByteShiftedMoments()
{
    if constexpr (sizeof(T) != sizeof(uint16_t)) {
        Process();
        return;
    }
    if (blockIdx_ >= tiling_->totalGroups) {
        return;
    }

    CopyInAffine();
    LocalTensor<float> gammaLocal = inputQueueGamma_.DeQue<float>();
    LocalTensor<float> betaLocal = inputQueueBeta_.DeQue<float>();

    // Use the already-proven shifted-moment organization from the Low-A leaf,
    // but retain this leaf's original affine-first issue order.  The anchor
    // keeps the first and second moments well conditioned, while both tile
    // reductions share one Vector-to-Scalar publication.
    const int64_t firstCount = Min(tiling_->ubFactor, tiling_->groupElements);
    CopyInX(0, firstCount);
    float anchor = 0.0f;
    bool anchorReady = false;
    float shiftedSum = 0.0f;
    float shiftedSumCorrection = 0.0f;
    float shiftedSquareSum = 0.0f;
    float shiftedSquareCorrection = 0.0f;
    for (int64_t offset = 0; offset < tiling_->groupElements; offset += tiling_->ubFactor) {
        const int64_t count = Min(tiling_->ubFactor, tiling_->groupElements - offset);
        inputLocal_ = inputQueueX_.DeQue<T>();
        Cast(x1Local_, inputLocal_, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        inputQueueX_.FreeTensor(inputLocal_);

        const int64_t nextOffset = offset + tiling_->ubFactor;
        if (nextOffset < tiling_->groupElements) {
            const int64_t nextCount = Min(
                tiling_->ubFactor, tiling_->groupElements - nextOffset);
            CopyInX(nextOffset, nextCount);
        }

        if (!anchorReady) {
            const event_t anchorEventId = static_cast<event_t>(0);
            SetFlag<HardEvent::V_S>(anchorEventId);
            WaitFlag<HardEvent::V_S>(anchorEventId);
            const int64_t sampleStride = count / 8;
            float anchorSum = 0.0f;
            for (int32_t sample = 0; sample < 8; ++sample) {
                anchorSum += x1Local_.GetValue(sample * sampleStride);
            }
            anchor = anchorSum * 0.125f;
            anchorReady = true;
        }

        Adds(x1Local_, x1Local_, -anchor, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal_, x1Local_, x2Local_, static_cast<int32_t>(count));
        Mul(x1Local_, x1Local_, x1Local_, count);
        PipeBarrier<PIPE_V>();
        ReduceSum(reduceLocal_[4], x1Local_, x2Local_, static_cast<int32_t>(count));

        const event_t eventId = static_cast<event_t>(0);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);
        float tileSum = reduceLocal_.GetValue(0);
        float tileSquareSum = reduceLocal_.GetValue(4);
        if (count > REDUCE_HALF_ELEMENTS) {
            tileSum += reduceLocal_.GetValue(1);
            tileSquareSum += reduceLocal_.GetValue(5);
        }
        if (count > REDUCE_MAX_ELEMENTS) {
            tileSum += reduceLocal_.GetValue(2) + reduceLocal_.GetValue(3);
            tileSquareSum += reduceLocal_.GetValue(6) + reduceLocal_.GetValue(7);
        }
        KahanAdd(tileSum, shiftedSum, shiftedSumCorrection);
        KahanAdd(tileSquareSum, shiftedSquareSum, shiftedSquareCorrection);
    }

    const float invCount = 1.0f / static_cast<float>(tiling_->groupElements);
    const float shiftedMean = shiftedSum * invCount;
    const float mean = anchor + shiftedMean;
    float variance = shiftedSquareSum * invCount - shiftedMean * shiftedMean;
    if (variance < 0.0f) {
        variance = 0.0f;
    }
    const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);
    NormalizeGroup<false>(0, groupStart_, mean, rstd, gammaLocal, betaLocal);
    inputQueueGamma_.FreeTensor(gammaLocal);
    inputQueueBeta_.FreeTensor(betaLocal);
}
#endif

#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
template <typename T>
__aicore__ inline void GroupNorm<T>::ProcessLowAFirstTile()
{
    if (blockIdx_ >= tiling_->totalGroups * tiling_->outputSplitWorkers) {
        return;
    }
    const int64_t firstCount = Min(tiling_->ubFactor, tiling_->groupElements);
    CopyInX(0, firstCount);
    CopyInAffine();
    if constexpr (sizeof(T) == sizeof(float)) {
        const int64_t groupCount = tiling_->outputSplitWorkers == 2 ? 1 :
            ((blockIdx_ == GetBlockNum() - 1) ?
                tiling_->lastCoreGroups : tiling_->blockFactor);
        float mean = 0.0f;
        float variance = 0.0f;
        ComputeStatistics<2, true, false>(0, mean, variance);
        LocalTensor<float> gammaLocal = inputQueueGamma_.DeQue<float>();
        LocalTensor<float> betaLocal = inputQueueBeta_.DeQue<float>();
        float rstd = 1.0f / sqrt(variance + tiling_->epsilon);
        NormalizeGroup<false>(0, groupStart_, mean, rstd, gammaLocal, betaLocal);
        for (int64_t localGroup = 1; localGroup < groupCount; ++localGroup) {
            const int64_t groupOffset = localGroup * tiling_->groupElements;
            mean = 0.0f;
            variance = 0.0f;
            ComputeStatistics<2, false, false>(groupOffset, mean, variance);
            rstd = 1.0f / sqrt(variance + tiling_->epsilon);
            NormalizeGroup<false>(
                localGroup, groupStart_ + localGroup, mean, rstd, gammaLocal, betaLocal);
        }
        inputQueueGamma_.FreeTensor(gammaLocal);
        inputQueueBeta_.FreeTensor(betaLocal);
    } else {
        // Low-A 2-byte groups retain O37's first-tile ordering.
        float mean = 0.0f;
        float variance = 0.0f;
        ComputeLowATwoByteMoments(mean, variance);
        LocalTensor<float> gammaLocal = inputQueueGamma_.DeQue<float>();
        LocalTensor<float> betaLocal = inputQueueBeta_.DeQue<float>();
        const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);
        NormalizeGroup<true>(0, groupStart_, mean, rstd, gammaLocal, betaLocal);
        inputQueueGamma_.FreeTensor(gammaLocal);
        inputQueueBeta_.FreeTensor(betaLocal);
    }
}
#endif

} // namespace NsGroupNorm

#endif // GROUP_NORM_H
