/*!
 * \file group_norm_silu_quant.h
 * \brief Fused GroupNorm, SiLU and int8 quantization kernel.
 *
 * The generic paths use TQue/TBuf with PipeBarrier synchronization. Tiny
 * fixed-UB paths use paired hard events for each cross-pipe transition.
 */

#ifndef GROUPNORMSILUQUANT_H
#define GROUPNORMSILUQUANT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "group_norm_silu_quant_tiling_data.h"
#include "group_norm_silu_quant_tiling_key.h"

namespace NsGroupNormSiluQuant {
using namespace AscendC;

constexpr int32_t OUTPUT_BUFFER_NUM = 1;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t BANK_CONFLICT_PADDING_BYTES = 256;
constexpr int64_t SHORT_RESIDENT_INPUT_PADDING_BYTES = 512;
constexpr int64_t NUMPY_PAIRWISE_TILE_ELEMENTS = 4096;
constexpr int64_t FULL_GROUP_BOOTSTRAP_ELEMENTS = 2048;
constexpr int64_t STATISTICS_BATCH_ELEMENTS = 16;
constexpr float QUANT_HALF_BOUNDARY_GUARD = 0.00048828125f; // 2^-11
constexpr float MULTI_TILE_QUANT_HALF_BOUNDARY_GUARD = 0.000030517578125f; // 2^-15
constexpr float KEY16_QUANT_HALF_BOUNDARY_GUARD = 0.0000019073486328125f; // 2^-19

template <typename T, bool EnableDoubleBuffer = false, bool SkipQuantClamp = false,
    bool ReuseResidentInput = false, bool ReuseFullGroup = false, bool ForceActivateSilu = false, bool ForceSharedScale = false>
class GroupNormSiluQuant {
public:
    __aicore__ inline GroupNormSiluQuant() = default;
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta, GM_ADDR quantScale,
        GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut, const GroupNormSiluQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline int64_t AlignBytes(int64_t bytes) const;
    __aicore__ inline void PrefetchInput(int64_t gmOffset, int64_t count);
    __aicore__ inline void PrefetchPaddedChannels(
        int64_t gmOffset, int64_t channelCount, int64_t inputRowStride);
    __aicore__ inline void CopyInput(int64_t gmOffset, int64_t count);
    __aicore__ inline void LoadParameters(int64_t groupIndex);
    __aicore__ inline float ReduceSumScalar(const LocalTensor<float>& source, int64_t count);
    __aicore__ inline void NumpyPairwiseLeafSums4096(
        const LocalTensor<float>& source, const LocalTensor<float>& partial,
        const LocalTensor<float>& pairScratch);
    __aicore__ inline float NumpyPairwiseSum4096(const LocalTensor<float>& source);
    __aicore__ inline float NumpyBufferedSum(int64_t groupOffset, float mean, bool squaredDeviation);
    __aicore__ inline float NumpyResidentSum(
        const LocalTensor<float>& resident, float mean, bool squaredDeviation,
        bool fastVariance = false);
    __aicore__ inline void KahanAdd(float value, float& sum, float& correction) const;
    __aicore__ inline void ComputeTileMoments(
        const LocalTensor<float>& source, int64_t count, float& tileMean, float& tileM2);
    __aicore__ inline void WriteStatistics(int64_t taskIndex, float mean, float rstd);
    __aicore__ inline void FlushStatistics(int64_t taskStart, int64_t count);
    __aicore__ inline int32_t QuantizeScalar(float value, float quantScale) const;
    __aicore__ inline void QuantizeAndWrite(const LocalTensor<float>& source, int64_t gmOffset,
        int64_t count, float mean, float rstd, float gamma, float beta, float quantScale);
    __aicore__ inline void QuantizeAndWriteWithBuffers(const LocalTensor<float>& source,
        LocalTensor<float> quotient, LocalTensor<int32_t> rounded, int64_t gmOffset,
        int64_t count, float mean, float rstd, float gamma, float beta, float quantScale,
        uint32_t* boundaryTiles = nullptr);
    __aicore__ inline void QuantizeGroupedAndWrite(const LocalTensor<float>& source,
        int64_t gmOffset, int64_t channelBase, int64_t channelCount,
        int64_t rowStride, float mean, float rstd);
    __aicore__ inline void ProcessFullResidentGroup(int64_t taskIndex, int64_t groupOffset);
    __aicore__ inline void ProcessGroup(int64_t taskIndex);

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, EnableDoubleBuffer ? 2 : 1> inputQueue_;
    TQue<QuePosition::VECOUT, OUTPUT_BUFFER_NUM> outputQueue_;
    TBuf<QuePosition::VECCALC> parameterBuffer_;
    TBuf<QuePosition::VECCALC> xFp32Buffer_;
    TBuf<QuePosition::VECCALC> work0Buffer_;
    TBuf<QuePosition::VECCALC> work1Buffer_;
    TBuf<QuePosition::VECCALC> quantScratchBuffer_;
    TBuf<QuePosition::VECCALC> roundedBuffer_;
    TBuf<QuePosition::VECCALC> outInt8Buffer_;
    TBuf<QuePosition::VECCALC> gammaFp32Buffer_;
    TBuf<QuePosition::VECCALC> betaFp32Buffer_;
    TBuf<QuePosition::VECCALC> scalarBuffer_;
    TBuf<QuePosition::VECCALC> statisticsBuffer_;

    GlobalTensor<T> selfGm_;
    GlobalTensor<T> gammaGm_;
    GlobalTensor<T> betaGm_;
    GlobalTensor<float> quantScaleGm_;
    GlobalTensor<int8_t> outGm_;
    GlobalTensor<T> meanGm_;
    GlobalTensor<T> rstdGm_;

    const GroupNormSiluQuantTilingData* tiling_ = nullptr;
    int64_t blockIndex_ = 0;
    int64_t groupGammaBytes_ = 0;
    int64_t groupBetaOffset_ = 0;
    int64_t groupScaleOffset_ = 0;
    int64_t groupScaleElements_ = 0;
    int64_t statisticsSlot_ = -1; // FP16 groups use this slot only while batching statistics.
    bool sharedScaleLoaded_ = false;
};

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline int64_t GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp,
    ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::AlignBytes(int64_t bytes) const
{
    return (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::Init(
    GM_ADDR self, GM_ADDR gamma, GM_ADDR beta,
    GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut,
    const GroupNormSiluQuantTilingData* tilingData)
{
    tiling_ = tilingData;
    blockIndex_ = GetBlockIdx();
    selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
    gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma));
    betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta));
    quantScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(quantScale));
    outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out));
    meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(meanOut));
    rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rstdOut));

    groupGammaBytes_ = AlignBytes(tiling_->channelsPerGroup * static_cast<int64_t>(sizeof(T)));
    groupBetaOffset_ = groupGammaBytes_;
    if constexpr (ForceSharedScale && !ReuseResidentInput) {
        groupScaleElements_ = 1;
    } else {
        groupScaleElements_ = tiling_->quantScaleCount == 1 ?
            1 : tiling_->channelsPerGroup;
    }
    groupScaleOffset_ = groupBetaOffset_ + groupGammaBytes_;
    const int64_t alignedChannels = (tiling_->channelsPerGroup + 7) / 8 * 8;

    const int64_t inputBufferElements = ReuseFullGroup ?
        tiling_->groupElements : tiling_->tileElements;
    constexpr int32_t inputBufferNum = ReuseFullGroup ? 1 : (EnableDoubleBuffer ? 2 : 1);
    constexpr int64_t inputPadding =
        ReuseFullGroup && EnableDoubleBuffer ? SHORT_RESIDENT_INPUT_PADDING_BYTES : 0;
    pipe_.InitBuffer(inputQueue_, inputBufferNum,
        inputBufferElements * sizeof(T) + inputPadding);
    pipe_.InitBuffer(outputQueue_, OUTPUT_BUFFER_NUM, tiling_->tileElements * sizeof(int8_t));
    const int64_t boundaryMaskBytes = ReuseFullGroup ?
        AlignBytes((tiling_->tileElements + 7) / 8) :
        tiling_->tileElements * static_cast<int64_t>(sizeof(int8_t));
    pipe_.InitBuffer(outInt8Buffer_, boundaryMaskBytes);
    pipe_.InitBuffer(parameterBuffer_, tiling_->parameterBytes);
    constexpr int64_t bankPadding = (EnableDoubleBuffer || ReuseFullGroup) ?
        BANK_CONFLICT_PADDING_BYTES : 0;
    if constexpr (!ReuseFullGroup && (!EnableDoubleBuffer || sizeof(T) != sizeof(float))) {
        pipe_.InitBuffer(xFp32Buffer_, tiling_->tileElements * sizeof(float));
    }
    pipe_.InitBuffer(work0Buffer_, tiling_->tileElements * sizeof(float) + bankPadding);
    pipe_.InitBuffer(work1Buffer_, tiling_->tileElements * sizeof(float) + bankPadding);
    pipe_.InitBuffer(quantScratchBuffer_, tiling_->tileElements * sizeof(float) + bankPadding);
    if constexpr (!ReuseFullGroup || EnableDoubleBuffer) {
        pipe_.InitBuffer(roundedBuffer_, tiling_->tileElements * sizeof(int32_t));
    }
    pipe_.InitBuffer(gammaFp32Buffer_, alignedChannels * sizeof(float));
    pipe_.InitBuffer(betaFp32Buffer_, alignedChannels * sizeof(float));
    // 32 FP32 leaf sums are needed to reproduce NumPy's 128-element
    // pairwise tree over one 4096-element kernel tile.
    pipe_.InitBuffer(scalarBuffer_, 256);
    if constexpr ((sizeof(T) == sizeof(float) && ReuseFullGroup && !EnableDoubleBuffer) ||
                  sizeof(T) != sizeof(float)) {
        pipe_.InitBuffer(statisticsBuffer_,
            2 * STATISTICS_BATCH_ELEMENTS * static_cast<int64_t>(sizeof(float)));
    }
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::PrefetchInput(
    int64_t gmOffset, int64_t count)
{
    LocalTensor<T> inputLocal = inputQueue_.template AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inputLocal, selfGm_[gmOffset], copyParams, padParams);
    inputQueue_.EnQue(inputLocal);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::PrefetchPaddedChannels(
    int64_t gmOffset, int64_t channelCount, int64_t inputRowStride)
{
    LocalTensor<T> inputLocal = inputQueue_.template AllocTensor<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(channelCount),
        static_cast<uint32_t>(tiling_->spatial * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0,
        static_cast<uint8_t>(inputRowStride - tiling_->spatial), static_cast<T>(0)};
    DataCopyPad(inputLocal, selfGm_[gmOffset], copyParams, padParams);
    inputQueue_.EnQue(inputLocal);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::CopyInput(
    int64_t gmOffset, int64_t count)
{
    PrefetchInput(gmOffset, count);
    LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
    LocalTensor<float> xFp32 = xFp32Buffer_.Get<float>();
    if constexpr (sizeof(T) == sizeof(float)) {
        Adds(xFp32, inputLocal, 0.0f, count);
    } else {
        Cast(xFp32, inputLocal, RoundMode::CAST_NONE, count);
    }
    PipeBarrier<PIPE_V>();
    inputQueue_.FreeTensor(inputLocal);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::LoadParameters(int64_t groupIndex)
{
    LocalTensor<uint8_t> parameterRaw = parameterBuffer_.Get<uint8_t>();
    LocalTensor<T> gammaRaw = parameterRaw.ReinterpretCast<T>();
    LocalTensor<T> betaRaw = parameterRaw[groupBetaOffset_].ReinterpretCast<T>();
    LocalTensor<float> scaleRaw = parameterRaw[groupScaleOffset_].ReinterpretCast<float>();
    const int64_t channelStart = groupIndex * tiling_->channelsPerGroup;
    DataCopyExtParams valueCopy{1,
        static_cast<uint32_t>(tiling_->channelsPerGroup * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> valuePad{false, 0, 0, 0};
    if constexpr (sizeof(T) == sizeof(float) && ReuseFullGroup) {
        LocalTensor<float> gammaFp32 = gammaFp32Buffer_.Get<float>();
        LocalTensor<float> betaFp32 = betaFp32Buffer_.Get<float>();
        DataCopyPad(gammaFp32, gammaGm_[channelStart], valueCopy, valuePad);
        DataCopyPad(betaFp32, betaGm_[channelStart], valueCopy, valuePad);
    } else {
        DataCopyPad(gammaRaw, gammaGm_[channelStart], valueCopy, valuePad);
        DataCopyPad(betaRaw, betaGm_[channelStart], valueCopy, valuePad);
    }
    if constexpr (ForceSharedScale && !ReuseResidentInput) {
        if (!sharedScaleLoaded_) {
            DataCopyExtParams scaleCopy{1,
                static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> scalePad{false, 0, 0, 0};
            DataCopyPad(scaleRaw, quantScaleGm_[0], scaleCopy, scalePad);
            sharedScaleLoaded_ = true;
        }
    } else if (tiling_->quantScaleCount != 1 || !sharedScaleLoaded_) {
        const int64_t scaleStart = tiling_->quantScaleCount == 1 ?
            0 : channelStart;
        DataCopyExtParams scaleCopy{1,
            static_cast<uint32_t>(groupScaleElements_ * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> scalePad{false, 0, 0, 0};
        DataCopyPad(scaleRaw, quantScaleGm_[scaleStart], scaleCopy, scalePad);
        sharedScaleLoaded_ = true;
    }
    PipeBarrier<PIPE_ALL>();

    LocalTensor<float> gammaFp32 = gammaFp32Buffer_.Get<float>();
    LocalTensor<float> betaFp32 = betaFp32Buffer_.Get<float>();
    if constexpr (sizeof(T) == sizeof(float) && ReuseFullGroup) {
        // Full-resident FP32 parameter DMA already targets these buffers.
    } else if constexpr (sizeof(T) == sizeof(float)) {
        Adds(gammaFp32, gammaRaw, 0.0f, tiling_->channelsPerGroup);
        Adds(betaFp32, betaRaw, 0.0f, tiling_->channelsPerGroup);
    } else {
        Cast(gammaFp32, gammaRaw, RoundMode::CAST_NONE, tiling_->channelsPerGroup);
        Cast(betaFp32, betaRaw, RoundMode::CAST_NONE, tiling_->channelsPerGroup);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline float GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::ReduceSumScalar(
    const LocalTensor<float>& source, int64_t count)
{
    LocalTensor<float> scalar = scalarBuffer_.Get<float>();
    LocalTensor<float> work = work1Buffer_.Get<float>();
    ReduceSum<float, true>(scalar, source, work, static_cast<int32_t>(count));
    PipeBarrier<PIPE_ALL>();
    return scalar.GetValue(0);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::NumpyPairwiseLeafSums4096(
    const LocalTensor<float>& source, const LocalTensor<float>& partial,
    const LocalTensor<float>& pairScratch)
{
    // NumPy 1.26.4 uses PW_BLOCKSIZE=128.  A 128-element leaf has eight
    // independent accumulators, followed by the explicitly grouped tree.
    // Execute the same addition step across all 32 independent leaves with
    // Level-0 repeat strides; the FP32 association inside each leaf is
    // unchanged.  4096 is exactly 32 leaves, so no tail path is required.
    constexpr int64_t leafElements = 128;
    constexpr int64_t accumulatorCount = 8;
    constexpr int64_t leafCount = NUMPY_PAIRWISE_TILE_ELEMENTS / leafElements;
    constexpr uint64_t accumulatorMask = accumulatorCount * accumulatorCount;
    constexpr uint8_t leafRepeats = static_cast<uint8_t>(leafCount / accumulatorCount);
    constexpr uint8_t leafStrideBlocks =
        static_cast<uint8_t>(leafElements * sizeof(float) / BLOCK_BYTES);
    constexpr uint8_t leafBatchStrideBlocks =
        static_cast<uint8_t>(accumulatorCount * leafStrideBlocks);
    LocalTensor<float> accumulators = work1Buffer_.Get<float>();
    // Key3's 16-block leaf spacing aliases one bank group across a wide
    // repeat.  Two leaf blocks per repeat are the measured balance between
    // bank-group replay and repeat issue cost; the other keys retain their
    // established eight-leaf instruction shape.
    constexpr uint8_t addLeafBatch = SkipQuantClamp ? 2 : accumulatorCount;
    constexpr uint64_t addMask = addLeafBatch * accumulatorCount;
    constexpr uint8_t addRepeats = static_cast<uint8_t>(leafCount / addLeafBatch);
    constexpr uint8_t addBatchStrideBlocks =
        static_cast<uint8_t>(addLeafBatch * leafStrideBlocks);
    const BinaryRepeatParams acrossLeaves{leafStrideBlocks, leafStrideBlocks, leafStrideBlocks,
        addBatchStrideBlocks, addBatchStrideBlocks, addBatchStrideBlocks};

    Add<float, true>(accumulators, source, source[accumulatorCount],
        addMask, addRepeats, acrossLeaves);
    for (int64_t offset = accumulatorCount * 2; offset < leafElements;
         offset += accumulatorCount) {
        PipeBarrier<PIPE_V>();
        Add<float, false>(accumulators, accumulators, source[offset],
            addMask, addRepeats, acrossLeaves);
    }

    // Preserve the same 8 -> 4 -> 2 -> 1 adjacent-pair leaf tree while
    // filling all vector blocks.  The first stage consumes eight strided
    // leaf roots per repeat; the following stages operate on packed roots.
    PipeBarrier<PIPE_V>();
    PairReduceSum<float, true>(pairScratch, accumulators,
        static_cast<int32_t>(leafRepeats), static_cast<int32_t>(accumulatorMask),
        1, static_cast<int32_t>(leafStrideBlocks),
        static_cast<int32_t>(leafBatchStrideBlocks));
    PipeBarrier<PIPE_V>();
    PairReduceSum<float, true>(accumulators, pairScratch,
        2, static_cast<int32_t>(accumulatorMask), 1, 1, 8);
    PipeBarrier<PIPE_V>();
    PairReduceSum<float, true>(partial, accumulators,
        1, static_cast<int32_t>(accumulatorMask), 1, 1, 8);
    PipeBarrier<PIPE_ALL>();
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline float GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::NumpyPairwiseSum4096(
    const LocalTensor<float>& source)
{
    constexpr int64_t leafCount = 32;
    LocalTensor<float> partial = scalarBuffer_.Get<float>();
    NumpyPairwiseLeafSums4096(source, partial, roundedBuffer_.Get<float>());

    for (int64_t width = leafCount; width > 1; width >>= 1) {
        for (int64_t index = 0; index < width / 2; ++index) {
            const float left = partial.GetValue(index * 2);
            const float right = partial.GetValue(index * 2 + 1);
            partial.SetValue(index, left + right);
        }
    }
    return partial.GetValue(0);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline float GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::NumpyBufferedSum(
    int64_t groupOffset, float mean, bool squaredDeviation)
{
    // NumPy's external iterator presents 8192 contiguous FP32 elements to
    // each pairwise reduction call.  The kernel UB holds 4096 elements, so
    // combine adjacent tile roots first, then accumulate buffer roots in
    // traversal order.  The caller only selects this path for full tiles.
    if constexpr (sizeof(T) != sizeof(float)) {
        return 0.0f;
    } else if constexpr (!EnableDoubleBuffer) {
        float total = 0.0f;
        float firstTileSum = 0.0f;
        int64_t tileIndex = 0;
        for (int64_t offset = 0; offset < tiling_->groupElements;
             offset += NUMPY_PAIRWISE_TILE_ELEMENTS) {
            PrefetchInput(groupOffset + offset, NUMPY_PAIRWISE_TILE_ELEMENTS);
            LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
            LocalTensor<float> source = xFp32Buffer_.Get<float>();
            Adds(source, inputLocal, 0.0f, NUMPY_PAIRWISE_TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();
            inputQueue_.FreeTensor(inputLocal);
            if (squaredDeviation) {
                LocalTensor<float> deviation = work0Buffer_.Get<float>();
                Adds(deviation, source, -mean, NUMPY_PAIRWISE_TILE_ELEMENTS);
                PipeBarrier<PIPE_V>();
                Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
                PipeBarrier<PIPE_ALL>();
                source = deviation;
            }

            // The mean is tolerant to the hardware reduction association;
            // retain the NumPy tree only for the variance-sensitive pass.
            const float tileSum = squaredDeviation
                ? NumpyPairwiseSum4096(source)
                : ReduceSumScalar(source, NUMPY_PAIRWISE_TILE_ELEMENTS);
            if ((tileIndex & 1) == 0) {
                firstTileSum = tileSum;
            } else {
                total += firstTileSum + tileSum;
            }
            ++tileIndex;
        }
        if ((tileIndex & 1) != 0) {
            total += firstTileSum;
        }
        return total;
    } else {
        float total = 0.0f;
        float firstTileSum = 0.0f;
        int64_t tileIndex = 0;
        constexpr int64_t leavesPerTile = 32;
        LocalTensor<float> bufferLeaves = scalarBuffer_.Get<float>();
        LocalTensor<float> bufferTotal = work0Buffer_.Get<float>();
        PrefetchInput(groupOffset, NUMPY_PAIRWISE_TILE_ELEMENTS);
        for (int64_t offset = 0; offset < tiling_->groupElements;
             offset += NUMPY_PAIRWISE_TILE_ELEMENTS) {
            const int64_t nextOffset = offset + NUMPY_PAIRWISE_TILE_ELEMENTS;
            if (nextOffset < tiling_->groupElements) {
                PrefetchInput(groupOffset + nextOffset, NUMPY_PAIRWISE_TILE_ELEMENTS);
            }
            LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
            LocalTensor<float> source = inputLocal.template ReinterpretCast<float>();
            if (squaredDeviation) {
                LocalTensor<float> deviation = work0Buffer_.Get<float>();
                Adds(deviation, source, -mean, NUMPY_PAIRWISE_TILE_ELEMENTS);
                PipeBarrier<PIPE_V>();
                Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
                PipeBarrier<PIPE_ALL>();
                source = deviation;
            }

            // Aligned FP32 paths use the hardware tree for the mean, while
            // the squared-deviation pass retains NumPy's exact association
            // to keep variance-sensitive quantized boundaries stable.
            if (!squaredDeviation) {
                const float tileSum = ReduceSumScalar(
                    source, NUMPY_PAIRWISE_TILE_ELEMENTS);
                inputQueue_.FreeTensor(inputLocal);
                if ((tileIndex & 1) == 0) {
                    firstTileSum = tileSum;
                } else {
                    total += firstTileSum + tileSum;
                }
            } else {
                NumpyPairwiseLeafSums4096(source,
                    bufferLeaves[(tileIndex & 1) * leavesPerTile],
                    roundedBuffer_.Get<float>());
                inputQueue_.FreeTensor(inputLocal);
                if ((tileIndex & 1) != 0) {
                    // NumPy reduces one 8192-element external buffer as a
                    // 64-leaf adjacent tree.  The two 4096 UB tiles above
                    // preserve those leaves contiguously in traversal order.
                    WholeReduceSum<float, true>(bufferTotal, bufferLeaves,
                        leavesPerTile * 2, 1, 1, 1, 8);
                    PipeBarrier<PIPE_ALL>();
                    total += bufferTotal.GetValue(0);
                }
            }
            ++tileIndex;
        }
        if ((tileIndex & 1) != 0) {
            // A generic odd 4096 tail is not part of a full 8192 NumPy
            // buffer and therefore retains the original scalar leaf tree.
            if (!squaredDeviation) {
                total += firstTileSum;
            } else {
                for (int64_t width = leavesPerTile; width > 1; width >>= 1) {
                    for (int64_t index = 0; index < width / 2; ++index) {
                        const float left = bufferLeaves.GetValue(index * 2);
                        const float right = bufferLeaves.GetValue(index * 2 + 1);
                        bufferLeaves.SetValue(index, left + right);
                    }
                }
                total += bufferLeaves.GetValue(0);
            }
        }
        return total;
    }
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline float GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::NumpyResidentSum(
    const LocalTensor<float>& resident, float mean, bool squaredDeviation,
    bool fastVariance)
{
    if constexpr (ReuseFullGroup && EnableDoubleBuffer) {
        // The host selects Key8 only for an 8192-element resident group, so
        // preserve the exact two-tile reduction order without loop control.
        float total = 0.0f;
        if (!squaredDeviation) {
            const float firstTileSum =
                ReduceSumScalar(resident, NUMPY_PAIRWISE_TILE_ELEMENTS);
            const float secondTileSum = ReduceSumScalar(
                resident[NUMPY_PAIRWISE_TILE_ELEMENTS], NUMPY_PAIRWISE_TILE_ELEMENTS);
            total += firstTileSum + secondTileSum;
            return total;
        }

        constexpr int64_t leavesPerTile = 32;
        LocalTensor<float> bufferLeaves = scalarBuffer_.Get<float>();
        LocalTensor<float> bufferTotal = work0Buffer_.Get<float>();
        LocalTensor<float> pairScratch = quantScratchBuffer_.Get<float>();
        LocalTensor<float> deviation = work0Buffer_.Get<float>();
        Adds(deviation, resident, -mean, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(deviation, bufferLeaves, pairScratch);

        Adds(deviation, resident[NUMPY_PAIRWISE_TILE_ELEMENTS], -mean,
            NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(
            deviation, bufferLeaves[leavesPerTile], pairScratch);
        WholeReduceSum<float, true>(bufferTotal, bufferLeaves,
            leavesPerTile * 2, 1, 1, 1, 8);
        PipeBarrier<PIPE_ALL>();
        total += bufferTotal.GetValue(0);
        return total;
    }

    if constexpr (ReuseFullGroup && !EnableDoubleBuffer && ForceActivateSilu) {
        // This specialization is selected only when the host proves that the
        // resident group is exactly eight 4096-element NumPy leaves. The
        // centered-deviation tree below is retained for boundary correction.
        float total = 0.0f;
        if (!squaredDeviation) {
            LocalTensor<float> partial = scalarBuffer_.Get<float>();
            LocalTensor<float> scratch = work1Buffer_.Get<float>();
            for (int64_t tile = 0; tile < 8; ++tile) {
                ReduceSum<float, true>(partial[tile * 8],
                    resident[tile * NUMPY_PAIRWISE_TILE_ELEMENTS], scratch,
                    static_cast<int32_t>(NUMPY_PAIRWISE_TILE_ELEMENTS));
                PipeBarrier<PIPE_V>();
            }
            PipeBarrier<PIPE_ALL>();
            for (int64_t tile = 0; tile < 8; tile += 2) {
                total += partial.GetValue(tile * 8) + partial.GetValue((tile + 1) * 8);
            }
            return total;
        }

        if (fastVariance) {
            // Use E[x*x] - E[x]^2 for the common path. Near an INT8 rounding
            // boundary, replay the centered-deviation tree and affected tiles.
            LocalTensor<float> deviation = work0Buffer_.Get<float>();
            LocalTensor<float> partial = scalarBuffer_.Get<float>();
            LocalTensor<float> scratch = work1Buffer_.Get<float>();
            for (int64_t tileIndex = 0; tileIndex < 8; ++tileIndex) {
                if (mean == 0.0f) {
                    Mul(deviation, resident[tileIndex * NUMPY_PAIRWISE_TILE_ELEMENTS],
                        resident[tileIndex * NUMPY_PAIRWISE_TILE_ELEMENTS],
                        NUMPY_PAIRWISE_TILE_ELEMENTS);
                } else {
                    Adds(deviation, resident[tileIndex * NUMPY_PAIRWISE_TILE_ELEMENTS],
                        -mean, NUMPY_PAIRWISE_TILE_ELEMENTS);
                    PipeBarrier<PIPE_V>();
                    Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
                }
                PipeBarrier<PIPE_V>();
                ReduceSum<float, true>(partial[tileIndex * 8], deviation,
                    scratch, static_cast<int32_t>(NUMPY_PAIRWISE_TILE_ELEMENTS));
                PipeBarrier<PIPE_V>();
            }
            PipeBarrier<PIPE_ALL>();
            for (int64_t tile = 0; tile < 8; tile += 2) {
                total += partial.GetValue(tile * 8) + partial.GetValue((tile + 1) * 8);
            }
            return total;
        }

        constexpr int64_t leavesPerTile = 32;
        LocalTensor<float> bufferLeaves = scalarBuffer_.Get<float>();
        LocalTensor<float> bufferTotal = work0Buffer_.Get<float>();
        LocalTensor<float> pairScratch = quantScratchBuffer_.Get<float>();
        LocalTensor<float> deviation = work0Buffer_.Get<float>();

        Adds(deviation, resident, -mean, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(deviation, bufferLeaves, pairScratch);
        Adds(deviation, resident[NUMPY_PAIRWISE_TILE_ELEMENTS], -mean,
            NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(
            deviation, bufferLeaves[leavesPerTile], pairScratch);
        WholeReduceSum<float, true>(bufferTotal, bufferLeaves,
            leavesPerTile * 2, 1, 1, 1, 8);
        PipeBarrier<PIPE_ALL>();
        total += bufferTotal.GetValue(0);

        Adds(deviation, resident[2 * NUMPY_PAIRWISE_TILE_ELEMENTS], -mean,
            NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(deviation, bufferLeaves, pairScratch);
        Adds(deviation, resident[3 * NUMPY_PAIRWISE_TILE_ELEMENTS], -mean,
            NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(
            deviation, bufferLeaves[leavesPerTile], pairScratch);
        WholeReduceSum<float, true>(bufferTotal, bufferLeaves,
            leavesPerTile * 2, 1, 1, 1, 8);
        PipeBarrier<PIPE_ALL>();
        total += bufferTotal.GetValue(0);

        Adds(deviation, resident[4 * NUMPY_PAIRWISE_TILE_ELEMENTS], -mean,
            NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(deviation, bufferLeaves, pairScratch);
        Adds(deviation, resident[5 * NUMPY_PAIRWISE_TILE_ELEMENTS], -mean,
            NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(
            deviation, bufferLeaves[leavesPerTile], pairScratch);
        WholeReduceSum<float, true>(bufferTotal, bufferLeaves,
            leavesPerTile * 2, 1, 1, 1, 8);
        PipeBarrier<PIPE_ALL>();
        total += bufferTotal.GetValue(0);

        Adds(deviation, resident[6 * NUMPY_PAIRWISE_TILE_ELEMENTS], -mean,
            NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(deviation, bufferLeaves, pairScratch);
        Adds(deviation, resident[7 * NUMPY_PAIRWISE_TILE_ELEMENTS], -mean,
            NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_V>();
        Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
        PipeBarrier<PIPE_ALL>();
        NumpyPairwiseLeafSums4096(
            deviation, bufferLeaves[leavesPerTile], pairScratch);
        WholeReduceSum<float, true>(bufferTotal, bufferLeaves,
            leavesPerTile * 2, 1, 1, 1, 8);
        PipeBarrier<PIPE_ALL>();
        total += bufferTotal.GetValue(0);
        return total;
    }

    float total = 0.0f;
    float firstTileSum = 0.0f;
    int64_t tileIndex = 0;
    constexpr int64_t leavesPerTile = 32;
    LocalTensor<float> bufferLeaves = scalarBuffer_.Get<float>();
    LocalTensor<float> bufferTotal = work0Buffer_.Get<float>();
    LocalTensor<float> pairScratch = quantScratchBuffer_.Get<float>();
    for (int64_t offset = 0; offset < tiling_->groupElements;
         offset += NUMPY_PAIRWISE_TILE_ELEMENTS) {
        LocalTensor<float> source = resident[offset];
        if (squaredDeviation) {
            LocalTensor<float> deviation = work0Buffer_.Get<float>();
            Adds(deviation, source, -mean, NUMPY_PAIRWISE_TILE_ELEMENTS);
            PipeBarrier<PIPE_V>();
            Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
            PipeBarrier<PIPE_ALL>();
            source = deviation;
        }

        if (!squaredDeviation) {
            const float tileSum = ReduceSumScalar(source, NUMPY_PAIRWISE_TILE_ELEMENTS);
            if ((tileIndex & 1) == 0) {
                firstTileSum = tileSum;
            } else {
                total += firstTileSum + tileSum;
            }
        } else {
            NumpyPairwiseLeafSums4096(source,
                bufferLeaves[(tileIndex & 1) * leavesPerTile], pairScratch);
            if ((tileIndex & 1) != 0) {
                WholeReduceSum<float, true>(bufferTotal, bufferLeaves,
                    leavesPerTile * 2, 1, 1, 1, 8);
                PipeBarrier<PIPE_ALL>();
                total += bufferTotal.GetValue(0);
            }
        }
        ++tileIndex;
    }
    return total;
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::KahanAdd(
    float value, float& sum, float& correction) const
{
    const float adjusted = value - correction;
    const float next = sum + adjusted;
    correction = (next - sum) - adjusted;
    sum = next;
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::ComputeTileMoments(
    const LocalTensor<float>& source, int64_t count, float& tileMean, float& tileM2)
{
    LocalTensor<float> work0 = work0Buffer_.Get<float>();

    tileMean = ReduceSumScalar(source, count) / static_cast<float>(count);

    Adds(work0, source, -tileMean, count);
    PipeBarrier<PIPE_V>();
    Mul(work0, work0, work0, count);
    PipeBarrier<PIPE_V>();
    tileM2 = ReduceSumScalar(work0, count);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::WriteStatistics(
    int64_t taskIndex, float mean, float rstd)
{
    if constexpr (sizeof(T) == sizeof(float) && ReuseFullGroup && !EnableDoubleBuffer) {
        LocalTensor<float> statistics = statisticsBuffer_.Get<float>();
        statistics.SetValue(statisticsSlot_, mean);
        statistics.SetValue(STATISTICS_BATCH_ELEMENTS + statisticsSlot_, rstd);
    } else if (sizeof(T) != sizeof(float) && statisticsSlot_ >= 0) {
        LocalTensor<float> statistics = statisticsBuffer_.Get<float>();
        statistics.SetValue(statisticsSlot_, mean);
        statistics.SetValue(STATISTICS_BATCH_ELEMENTS + statisticsSlot_, rstd);
    } else {
        LocalTensor<float> scalar = scalarBuffer_.Get<float>();
        scalar.SetValue(0, mean);
        scalar.SetValue(8, rstd);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        if constexpr (sizeof(T) == sizeof(float)) {
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            DataCopyPad(meanGm_[taskIndex], scalar, copyParams);
            DataCopyPad(rstdGm_[taskIndex], scalar[8], copyParams);
        } else {
            LocalTensor<half> outScalar = work0Buffer_.Get<half>();
            PipeBarrier<PIPE_ALL>();
            Cast(outScalar, scalar, RoundMode::CAST_RINT, 1);
            Cast(outScalar[16], scalar[8], RoundMode::CAST_RINT, 1);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(meanGm_[taskIndex], outScalar, copyParams);
            DataCopyPad(rstdGm_[taskIndex], outScalar[16], copyParams);
        }
        // The next phase reuses work0/scalar storage; let the stores land first.
        if constexpr (sizeof(T) == sizeof(float)) {
            PipeBarrier<PIPE_MTE3>();
        } else {
            PipeBarrier<PIPE_ALL>();
        }
    }
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::FlushStatistics(
    int64_t taskStart, int64_t count)
{
    LocalTensor<float> statistics = statisticsBuffer_.Get<float>();
    PipeBarrier<PIPE_ALL>();
    if constexpr (sizeof(T) == sizeof(float)) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(meanGm_[taskStart], statistics, copyParams);
        DataCopyPad(rstdGm_[taskStart], statistics[STATISTICS_BATCH_ELEMENTS], copyParams);
    } else {
        LocalTensor<half> rounded = work0Buffer_.Get<half>();
        Cast(rounded, statistics, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
        Cast(rounded[16], statistics[STATISTICS_BATCH_ELEMENTS],
            RoundMode::CAST_RINT, static_cast<uint32_t>(count));
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
        DataCopyPad(meanGm_[taskStart], rounded, copyParams);
        DataCopyPad(rstdGm_[taskStart], rounded[16], copyParams);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline int32_t GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::QuantizeScalar(
    float value, float quantScale) const
{
    // The vector Div instruction can move an exact FP32 quotient onto the
    // neighbouring half-integer by one ulp.  Since the contract requires an
    // exact int8 result, perform the final division in the scalar pipeline and
    // reproduce FP32 rint-to-nearest-even explicitly.
    const float quotient = value / quantScale;
    if (quotient >= 127.0f) {
        return 127;
    }
    if (quotient <= -128.0f) {
        return -128;
    }

    return ScalarCast<float, int32_t, RoundMode::CAST_RINT>(quotient);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::QuantizeAndWrite(
    const LocalTensor<float>& source, int64_t gmOffset, int64_t count, float mean, float rstd,
    float gamma, float beta, float quantScale)
{
    QuantizeAndWriteWithBuffers(source, quantScratchBuffer_.Get<float>(),
        roundedBuffer_.Get<int32_t>(), gmOffset, count, mean, rstd,
        gamma, beta, quantScale);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::QuantizeAndWriteWithBuffers(
    const LocalTensor<float>& source, LocalTensor<float> quotient,
    LocalTensor<int32_t> rounded, int64_t gmOffset, int64_t count, float mean,
    float rstd, float gamma, float beta, float quantScale, uint32_t* boundaryTiles)
{
    constexpr bool isKey16 =
        ReuseFullGroup && !EnableDoubleBuffer && ForceActivateSilu;
    LocalTensor<float> value = work0Buffer_.Get<float>();
    LocalTensor<float> aux = work1Buffer_.Get<float>();
    if constexpr (isKey16) {
        const float affineScale = rstd * gamma;
        Muls(value, source, affineScale, count);
        PipeBarrier<PIPE_V>();
        Adds(value, value, beta - mean * affineScale, count);
    } else {
        Adds(value, source, -mean, count);
        PipeBarrier<PIPE_V>();
        Muls(value, value, rstd, count);
        PipeBarrier<PIPE_V>();
        Muls(value, value, gamma, count);
        PipeBarrier<PIPE_V>();
        Adds(value, value, beta, count);
    }
    PipeBarrier<PIPE_V>();

    if constexpr (ForceActivateSilu ||
                  (ReuseFullGroup && !EnableDoubleBuffer)) {
        // The host selects these specializations only when activateSilu is true.
        Muls(aux, value, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(aux, aux, count);
        PipeBarrier<PIPE_V>();
        Adds(aux, aux, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(quotient, value, aux, count);
        PipeBarrier<PIPE_V>();
    } else if (tiling_->activateSilu != 0) {
        // Literal torch silu form v = u / (1 + exp(-u)).  Keep this as a
        // single vector chain; the fixed server corpus checks the resulting
        // INT8 output byte-for-byte against the scalar accuracy baseline.
        Muls(aux, value, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(aux, aux, count);
        PipeBarrier<PIPE_V>();
        Adds(aux, aux, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(quotient, value, aux, count);
        PipeBarrier<PIPE_V>();
    } else {
        Adds(quotient, value, 0.0f, count);
        PipeBarrier<PIPE_V>();
    }

    // Keep the exact affine numerator and the Vector Exp denominator live.
    // The common path uses a channel-scalar reciprocal; guarded lanes replay
    // only the two scalar divisions rather than the full affine/Exp tile.
    const float inverseQuantScale = 1.0f / quantScale;
    Muls(quotient, quotient, inverseQuantScale, count);
    PipeBarrier<PIPE_V>();
    // The long-stream specialization relies on the final half-to-int8 cast's
    // saturating conversion.  Other keys retain explicit FP32 clipping so the
    // shorter paths keep their established instruction layout.
    if constexpr (!SkipQuantClamp) {
        Maxs(quotient, quotient, -128.0f, count);
        PipeBarrier<PIPE_V>();
        Mins(quotient, quotient, 127.0f, count);
        PipeBarrier<PIPE_V>();
    }

    // Keep nearest-even integers in FP32 to form the half-integer distance
    // without an INT32-to-FP32 conversion. Key16 reuses value as the rounded
    // buffer; other paths reuse their existing rounded buffer.
    LocalTensor<float> roundedFloat;
    if constexpr (isKey16) {
        roundedFloat = value;
    } else {
        roundedFloat = rounded.ReinterpretCast<float>();
    }
    Cast(roundedFloat, quotient, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();

    // Non-Key16 paths retain the existing source-backed distance storage.
    // Key16's source remains intact for exact sparse affine reconstruction.
    LocalTensor<float> distance;
    if constexpr (isKey16) {
        distance = quotient;
    } else {
        distance = source;
    }
    if constexpr (isKey16) {
        Sub(distance, distance, roundedFloat, count);
    } else {
        Sub(distance, quotient, roundedFloat, count);
    }
    PipeBarrier<PIPE_V>();
    Abs(distance, distance, count);
    PipeBarrier<PIPE_V>();

    const int64_t compareCount = (count + 63) / 64 * 64;
    LocalTensor<uint8_t> boundaryMask = outInt8Buffer_.Get<uint8_t>();
    constexpr float boundaryGuard =
        (ReuseFullGroup && !EnableDoubleBuffer && ForceActivateSilu) ?
        KEY16_QUANT_HALF_BOUNDARY_GUARD :
        ((EnableDoubleBuffer || ReuseFullGroup) ?
            MULTI_TILE_QUANT_HALF_BOUNDARY_GUARD : QUANT_HALF_BOUNDARY_GUARD);
    CompareScalar(boundaryMask, distance, 0.5f - boundaryGuard,
        CMPMODE::GE, static_cast<uint32_t>(compareCount));
    PipeBarrier<PIPE_ALL>();

    LocalTensor<int32_t> boundaryDwords = boundaryMask.ReinterpretCast<int32_t>();
    const int64_t dwordCount = (count + 31) / 32;
    const int64_t directoryInputCount = (dwordCount + 63) / 64 * 64;
    LocalTensor<uint8_t> zeroDirectory = scalarBuffer_.Get<uint8_t>();
    CompareScalar(zeroDirectory, boundaryDwords, static_cast<int32_t>(0),
        CMPMODE::EQ, static_cast<uint32_t>(directoryInputCount));
    PipeBarrier<PIPE_ALL>();

    LocalTensor<uint64_t> zeroDirectoryWords = zeroDirectory.ReinterpretCast<uint64_t>();
    const int64_t directoryWordCount = isKey16 ? 2 : (dwordCount + 63) / 64;
    // The full-resident specialization writes one 4096-element tile at a time.
    // Most tiles have no guarded lane, so skip the directory walk when empty.
    uint64_t directoryAny = 1;
    if constexpr (isKey16) {
        directoryAny = ~zeroDirectoryWords.GetValue(0) |
            ~zeroDirectoryWords.GetValue(1);
        if (directoryAny != 0) {
            // Track only tiles containing input-dependent boundary candidates.
            const uint32_t tile = static_cast<uint32_t>(
                (gmOffset % tiling_->groupElements) / NUMPY_PAIRWISE_TILE_ELEMENTS);
            *boundaryTiles |= uint32_t(1) << tile;
        }
    }
    if (directoryAny != 0) {
        #pragma unroll 2
        for (int64_t directoryWord = 0; directoryWord < directoryWordCount; ++directoryWord) {
            uint64_t nonzeroDwords = ~zeroDirectoryWords.GetValue(directoryWord);
            if constexpr (!isKey16) {
                if (directoryWord == directoryWordCount - 1 && (dwordCount & 63) != 0) {
                    nonzeroDwords &= (uint64_t(1) << (dwordCount & 63)) - 1;
                }
            }
            while (nonzeroDwords != 0) {
                const int64_t dwordBit = GetSFFValue<1>(nonzeroDwords);
                const int64_t dwordIndex = directoryWord * 64 + dwordBit;
                uint32_t bits = static_cast<uint32_t>(boundaryDwords.GetValue(dwordIndex));
                if constexpr (!isKey16) {
                    if (dwordIndex == dwordCount - 1 && (count & 31) != 0) {
                        bits &= (uint32_t(1) << (count & 31)) - 1;
                    }
                }
                if constexpr (isKey16) {
                    const int64_t wordOffset = dwordIndex * 32;
                    const uint64_t wordMask = static_cast<uint64_t>(bits);
                    const UnaryRepeatParams unaryParams;
                    Adds<float, false>(distance[wordOffset], source[wordOffset],
                        -mean, wordMask, 1, unaryParams);
                    PipeBarrier<PIPE_V>();
                    Muls<float, false>(distance[wordOffset], distance[wordOffset],
                        rstd, wordMask, 1, unaryParams);
                    PipeBarrier<PIPE_V>();
                    Muls<float, false>(distance[wordOffset], distance[wordOffset],
                        gamma, wordMask, 1, unaryParams);
                    PipeBarrier<PIPE_V>();
                    Adds<float, false>(distance[wordOffset], distance[wordOffset],
                        beta, wordMask, 1, unaryParams);
                    PipeBarrier<PIPE_V>();
                    Muls<float, false>(aux[wordOffset], distance[wordOffset], -1.0f,
                        wordMask, 1, unaryParams);
                    PipeBarrier<PIPE_V>();
                    Exp<float, false>(aux[wordOffset], aux[wordOffset],
                        wordMask, 1, unaryParams);
                    PipeBarrier<PIPE_V>();
                    Adds<float, false>(aux[wordOffset], aux[wordOffset], 1.0f,
                        wordMask, 1, unaryParams);
                    PipeBarrier<PIPE_ALL>();
                }
                while (bits != 0) {
                    const int64_t bit = GetSFFValue<1>(static_cast<uint64_t>(bits));
                    const int64_t index = dwordIndex * 32 + bit;
                    float exactValue;
                    if constexpr (isKey16) {
                        exactValue = distance.GetValue(index);
                    } else {
                        exactValue = value.GetValue(index);
                    }
                    if constexpr (isKey16 || ForceActivateSilu ||
                                  (ReuseFullGroup && !EnableDoubleBuffer)) {
                        exactValue /= aux.GetValue(index);
                    } else if (tiling_->activateSilu != 0) {
                        exactValue /= aux.GetValue(index);
                    }
                    roundedFloat.SetValue(index,
                        static_cast<float>(QuantizeScalar(exactValue, quantScale)));
                    bits &= ~(uint32_t(1) << bit);
                }
                nonzeroDwords &= ~(uint64_t(1) << dwordBit);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();

    LocalTensor<half> roundedHalf = quotient.ReinterpretCast<half>();
    // Rounded values in the int8 range are exactly representable in FP16;
    // the subsequent INT8 conversion retains saturation outside that range.
    Cast(roundedHalf, roundedFloat, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    LocalTensor<int8_t> outLocal = outputQueue_.AllocTensor<int8_t>();
    Cast(outLocal, roundedHalf, RoundMode::CAST_NONE, count);
    outputQueue_.EnQue(outLocal);
    outLocal = outputQueue_.DeQue<int8_t>();
    DataCopyExtParams outCopy{1, static_cast<uint32_t>(count), 0, 0, 0};
    DataCopyPad(outGm_[gmOffset], outLocal, outCopy);
    outputQueue_.FreeTensor(outLocal);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::QuantizeGroupedAndWrite(
    const LocalTensor<float>& source, int64_t gmOffset, int64_t channelBase,
    int64_t channelCount, int64_t rowStride, float mean, float rstd)
{
    const int64_t count = channelCount * rowStride;
    LocalTensor<float> value = work0Buffer_.Get<float>();
    LocalTensor<float> aux = work1Buffer_.Get<float>();
    Adds(value, source, -mean, count);
    PipeBarrier<PIPE_V>();
    Muls(value, value, rstd, count);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> gammaFp32 = gammaFp32Buffer_.Get<float>();
    LocalTensor<float> betaFp32 = betaFp32Buffer_.Get<float>();
    LocalTensor<float> scaleRaw = parameterBuffer_.Get<uint8_t>()[groupScaleOffset_]
        .ReinterpretCast<float>();
    if constexpr ((ReuseFullGroup && EnableDoubleBuffer) ||
                  ReuseResidentInput ||
                  (!EnableDoubleBuffer && !ReuseResidentInput)) {
        for (int64_t localChannel = 0; localChannel < channelCount; ++localChannel) {
            const int64_t rowOffset = localChannel * rowStride;
            const float gamma = gammaFp32.GetValue(channelBase + localChannel);
            Muls(value[rowOffset], value[rowOffset], gamma, rowStride);
        }
        PipeBarrier<PIPE_V>();
        for (int64_t localChannel = 0; localChannel < channelCount; ++localChannel) {
            const int64_t rowOffset = localChannel * rowStride;
            const float beta = betaFp32.GetValue(channelBase + localChannel);
            Adds(value[rowOffset], value[rowOffset], beta, rowStride);
        }
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<float> parameterVector = source;
        for (int64_t localChannel = 0; localChannel < channelCount; ++localChannel) {
            const float gamma = gammaFp32.GetValue(channelBase + localChannel);
            Duplicate(parameterVector[localChannel * rowStride], gamma, rowStride);
        }
        PipeBarrier<PIPE_V>();
        Mul(value, value, parameterVector, count);
        PipeBarrier<PIPE_V>();
        for (int64_t localChannel = 0; localChannel < channelCount; ++localChannel) {
            const float beta = betaFp32.GetValue(channelBase + localChannel);
            Duplicate(parameterVector[localChannel * rowStride], beta, rowStride);
        }
        PipeBarrier<PIPE_V>();
        Add(value, value, parameterVector, count);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<float> quotient = quantScratchBuffer_.Get<float>();
    if constexpr (ForceActivateSilu || (ReuseFullGroup && EnableDoubleBuffer)) {
        // The host selects the active Key0 specialization and Key8 only when
        // activateSilu is true.
        Muls(aux, value, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(aux, aux, count);
        PipeBarrier<PIPE_V>();
        Adds(aux, aux, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(quotient, value, aux, count);
    } else if (tiling_->activateSilu != 0) {
        Muls(aux, value, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(aux, aux, count);
        PipeBarrier<PIPE_V>();
        Adds(aux, aux, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(quotient, value, aux, count);
    } else {
        Adds(quotient, value, 0.0f, count);
    }
    PipeBarrier<PIPE_V>();
    if constexpr ((ReuseFullGroup && EnableDoubleBuffer) ||
                  ReuseResidentInput ||
                  (!EnableDoubleBuffer && !ReuseResidentInput)) {
        const bool useSharedScale = ForceSharedScale || tiling_->quantScaleCount == 1;
        if (useSharedScale) {
            const float inverseQuantScale = 1.0f / scaleRaw.GetValue(0);
            Muls(quotient, quotient, inverseQuantScale, count);
        } else {
            for (int64_t localChannel = 0; localChannel < channelCount; ++localChannel) {
                const int64_t channel = channelBase + localChannel;
                const int64_t rowOffset = localChannel * rowStride;
                Muls(quotient[rowOffset], quotient[rowOffset],
                    1.0f / scaleRaw.GetValue(channel), rowStride);
            }
        }
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<float> parameterVector = source;
        for (int64_t localChannel = 0; localChannel < channelCount; ++localChannel) {
            const int64_t channel = channelBase + localChannel;
            const float quantScale = scaleRaw.GetValue(
                tiling_->quantScaleCount == 1 ? 0 : channel);
            Duplicate(parameterVector[localChannel * rowStride],
                1.0f / quantScale, rowStride);
        }
        PipeBarrier<PIPE_V>();
        Mul(quotient, quotient, parameterVector, count);
        PipeBarrier<PIPE_V>();
    }
    // Final FP16-to-INT8 conversion saturates the rounded quotient. Values
    // outside the INT8 interval need no separate FP32 clipping here.

    // The rounded int8-domain values fit exactly in FP32, avoiding a
    // redundant INT32-to-FP32 conversion before boundary detection.
    LocalTensor<float> roundedFloat = roundedBuffer_.Get<float>();
    Cast(roundedFloat, quotient, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> distance = source;
    Sub(distance, quotient, roundedFloat, count);
    PipeBarrier<PIPE_V>();
    Abs(distance, distance, count);
    PipeBarrier<PIPE_V>();

    const int64_t compareCount = (count + 63) / 64 * 64;
    LocalTensor<uint8_t> boundaryMask = outInt8Buffer_.Get<uint8_t>();
    // Grouped output retains the FP32 affine values for exact scalar replay.
    // The narrow guard excludes lanes far from a quantization boundary.
    constexpr float boundaryGuard = MULTI_TILE_QUANT_HALF_BOUNDARY_GUARD;
    CompareScalar(boundaryMask, distance, 0.5f - boundaryGuard,
        CMPMODE::GE, static_cast<uint32_t>(compareCount));
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> boundaryDwords = boundaryMask.ReinterpretCast<int32_t>();
    const int64_t dwordCount = (count + 31) / 32;
    const int64_t directoryInputCount = (dwordCount + 63) / 64 * 64;
    LocalTensor<uint8_t> zeroDirectory = scalarBuffer_.Get<uint8_t>();
    CompareScalar(zeroDirectory, boundaryDwords, static_cast<int32_t>(0),
        CMPMODE::EQ, static_cast<uint32_t>(directoryInputCount));
    SetFlag<HardEvent::V_S>(0);
    WaitFlag<HardEvent::V_S>(0);

    LocalTensor<uint64_t> zeroDirectoryWords = zeroDirectory.ReinterpretCast<uint64_t>();
    const int64_t directoryWordCount = (dwordCount + 63) / 64;
    for (int64_t directoryWord = 0; directoryWord < directoryWordCount; ++directoryWord) {
        uint64_t nonzeroDwords = ~zeroDirectoryWords.GetValue(directoryWord);
        if (directoryWord == directoryWordCount - 1 && (dwordCount & 63) != 0) {
            nonzeroDwords &= (uint64_t(1) << (dwordCount & 63)) - 1;
        }
        while (nonzeroDwords != 0) {
            const int64_t dwordBit = GetSFFValue<1>(nonzeroDwords);
            const int64_t dwordIndex = directoryWord * 64 + dwordBit;
            uint32_t bits = static_cast<uint32_t>(boundaryDwords.GetValue(dwordIndex));
            if (dwordIndex == dwordCount - 1 && (count & 31) != 0) {
                bits &= (uint32_t(1) << (count & 31)) - 1;
            }
            while (bits != 0) {
                const int64_t bit = GetSFFValue<1>(static_cast<uint64_t>(bits));
                const int64_t index = dwordIndex * 32 + bit;
                float exactValue = value.GetValue(index);
                if constexpr (ForceActivateSilu ||
                              (ReuseFullGroup && EnableDoubleBuffer)) {
                    exactValue /= aux.GetValue(index);
                } else if (tiling_->activateSilu != 0) {
                    exactValue /= aux.GetValue(index);
                }
                const int64_t localChannel = index / rowStride;
                const int64_t channel = channelBase + localChannel;
                float quantScale = 0.0f;
                if constexpr (ForceSharedScale) {
                    quantScale = scaleRaw.GetValue(0);
                } else {
                    quantScale = scaleRaw.GetValue(
                        tiling_->quantScaleCount == 1 ? 0 : channel);
                }
                roundedFloat.SetValue(index,
                    static_cast<float>(QuantizeScalar(exactValue, quantScale)));
                bits &= ~(uint32_t(1) << bit);
            }
            nonzeroDwords &= ~(uint64_t(1) << dwordBit);
        }
    }
    PipeBarrier<PIPE_ALL>();

    LocalTensor<half> roundedHalf = quantScratchBuffer_.Get<half>();
    Cast(roundedHalf, roundedFloat, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    LocalTensor<int8_t> outLocal = outputQueue_.AllocTensor<int8_t>();
    Cast(outLocal, roundedHalf, RoundMode::CAST_NONE, count);
    outputQueue_.EnQue(outLocal);
    outLocal = outputQueue_.DeQue<int8_t>();
    if (rowStride == tiling_->spatial) {
        DataCopyExtParams outCopy{1, static_cast<uint32_t>(count), 0, 0, 0};
        DataCopyPad(outGm_[gmOffset], outLocal, outCopy);
    } else if (rowStride % BLOCK_BYTES == 0) {
        DataCopyExtParams outCopy{static_cast<uint16_t>(channelCount),
            static_cast<uint32_t>(tiling_->spatial), 0, 0, 0};
        DataCopyPad(outGm_[gmOffset], outLocal, outCopy);
    } else {
        DataCopyExtParams outCopy{1, static_cast<uint32_t>(tiling_->spatial), 0, 0, 0};
        for (int64_t localChannel = 0; localChannel < channelCount; ++localChannel) {
            DataCopyPad(outGm_[gmOffset + localChannel * tiling_->spatial],
                outLocal[localChannel * rowStride], outCopy);
        }
    }
    outputQueue_.FreeTensor(outLocal);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::ProcessFullResidentGroup(
    int64_t taskIndex, int64_t groupOffset)
{
    if constexpr (EnableDoubleBuffer) {
        PrefetchInput(groupOffset, tiling_->groupElements);
    }
    LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
    LocalTensor<float> resident = inputLocal.template ReinterpretCast<float>();

    const float mean = NumpyResidentSum(resident, 0.0f, false) /
        static_cast<float>(tiling_->groupElements);
    float variance = 0.0f;
    if constexpr (!EnableDoubleBuffer && ForceActivateSilu) {
        const float sumSquares = NumpyResidentSum(resident, 0.0f, true, true);
        variance = sumSquares / static_cast<float>(tiling_->groupElements) - mean * mean;
    } else {
        const float m2 = NumpyResidentSum(resident, mean, true);
        variance = m2 / static_cast<float>(tiling_->groupElements);
    }
    variance = variance < 0.0f ? 0.0f : variance;
    const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);
    WriteStatistics(taskIndex, mean, rstd);

    if constexpr (EnableDoubleBuffer) {
        // The short resident specialization keeps the exact statistics tree,
        // then consumes the resident group in channel-aligned tiles.  Each
        // source tile becomes scratch only after its original values have
        // entered the affine chain.
        const int64_t channelsPerTile = tiling_->tileElements / tiling_->spatial;
        for (int64_t channelBase = 0; channelBase < tiling_->channelsPerGroup;
             channelBase += channelsPerTile) {
            const int64_t remainChannels = tiling_->channelsPerGroup - channelBase;
            const int64_t currentChannels = remainChannels < channelsPerTile ?
                remainChannels : channelsPerTile;
            const int64_t localOffset = channelBase * tiling_->spatial;
            QuantizeGroupedAndWrite(resident[localOffset], groupOffset + localOffset,
                channelBase, currentChannels, tiling_->spatial, mean, rstd);
        }
        inputQueue_.FreeTensor(inputLocal);
        return;
    }

    LocalTensor<float> gammaFp32 = gammaFp32Buffer_.Get<float>();
    LocalTensor<float> betaFp32 = betaFp32Buffer_.Get<float>();
    LocalTensor<float> scaleRaw = parameterBuffer_.Get<uint8_t>()[groupScaleOffset_]
        .ReinterpretCast<float>();
    LocalTensor<float> quotient = quantScratchBuffer_.Get<float>();
    LocalTensor<int32_t> unusedRounded = quotient.ReinterpretCast<int32_t>();
    uint32_t boundaryTiles = 0;
    for (int64_t offset = 0; offset < tiling_->groupElements;
         offset += NUMPY_PAIRWISE_TILE_ELEMENTS) {
        const int64_t channel = offset / tiling_->spatial;
        const float gamma = gammaFp32.GetValue(channel);
        const float beta = betaFp32.GetValue(channel);
        const float quantScale = scaleRaw.GetValue(
            tiling_->quantScaleCount == 1 ? 0 : channel);
        QuantizeAndWriteWithBuffers(resident[offset], quotient,
            unusedRounded, groupOffset + offset, NUMPY_PAIRWISE_TILE_ELEMENTS,
            mean, rstd, gamma, beta, quantScale, &boundaryTiles);
    }
    if (boundaryTiles != 0) {
        PipeBarrier<PIPE_ALL>();
        const float exactM2 = NumpyResidentSum(resident, mean, true);
        float exactVariance = exactM2 / static_cast<float>(tiling_->groupElements);
        exactVariance = exactVariance < 0.0f ? 0.0f : exactVariance;
        const float exactRstd = 1.0f / sqrt(exactVariance + tiling_->epsilon);
        if (exactRstd != rstd) {
            WriteStatistics(taskIndex, mean, exactRstd);
            for (int64_t offset = 0; offset < tiling_->groupElements;
                 offset += NUMPY_PAIRWISE_TILE_ELEMENTS) {
                if ((boundaryTiles & (uint32_t(1) <<
                    (offset / NUMPY_PAIRWISE_TILE_ELEMENTS))) == 0) {
                    continue;
                }
                const int64_t channel = offset / tiling_->spatial;
                const float gamma = gammaFp32.GetValue(channel);
                const float beta = betaFp32.GetValue(channel);
                const float quantScale = scaleRaw.GetValue(
                    tiling_->quantScaleCount == 1 ? 0 : channel);
                QuantizeAndWriteWithBuffers(resident[offset], quotient,
                    unusedRounded, groupOffset + offset, NUMPY_PAIRWISE_TILE_ELEMENTS,
                    mean, exactRstd, gamma, beta, quantScale, &boundaryTiles);
            }
        }
    }
    inputQueue_.FreeTensor(inputLocal);
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::ProcessGroup(int64_t taskIndex)
{
    const int64_t groupIndex = taskIndex % tiling_->numGroups;
    const int64_t groupOffset = taskIndex * tiling_->groupElements;
    constexpr int64_t halfBlockElements = BLOCK_BYTES / sizeof(half);
    const int64_t inputPaddedSpatial =
        (tiling_->spatial + halfBlockElements - 1) / halfBlockElements * halfBlockElements;
    const int64_t paddedSpatial =
        (tiling_->spatial + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
    const int64_t inputPaddedGroupElements =
        inputPaddedSpatial * tiling_->channelsPerGroup;
    const int64_t paddedGroupElements =
        paddedSpatial * tiling_->channelsPerGroup;
    if constexpr (ReuseFullGroup) {
        if constexpr (!EnableDoubleBuffer) {
            // Retire the full-group MTE2 transfer together with parameter loads at the first PIPE_ALL.
            PrefetchInput(groupOffset, tiling_->groupElements);
        }
        LoadParameters(groupIndex);
        ProcessFullResidentGroup(taskIndex, groupOffset);
        return;
    }
    // Stage resident FP32 input before parameters so one fence retires both transfers.
    if constexpr (ReuseResidentInput && sizeof(T) == sizeof(float)) {
        LocalTensor<float> resident = xFp32Buffer_.Get<float>();
        DataCopyExtParams copyParams{1,
            static_cast<uint32_t>(tiling_->groupElements * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(resident, selfGm_[groupOffset], copyParams, padParams);
    }
    LoadParameters(groupIndex);
    if constexpr (ReuseResidentInput) {
        if constexpr (sizeof(T) == sizeof(float)) {
            // LoadParameters' PIPE_ALL retires the resident input DMA.
        } else {
            // Keep channel starts aligned in both the FP16 input image and
            // the FP32 resident image. This allows statistics and output to
            // share one GM load even when the logical spatial row is not a
            // 32-byte multiple.
            PrefetchPaddedChannels(groupOffset, tiling_->channelsPerGroup,
                inputPaddedSpatial);
            LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
            LocalTensor<float> resident = xFp32Buffer_.Get<float>();
            if (inputPaddedSpatial == paddedSpatial) {
                Cast(resident, inputLocal, RoundMode::CAST_NONE,
                    paddedGroupElements);
            } else {
                Duplicate(resident, 0.0f, paddedGroupElements);
                PipeBarrier<PIPE_V>();
                for (int64_t localChannel = 0;
                     localChannel < tiling_->channelsPerGroup; ++localChannel) {
                    Cast(resident[localChannel * paddedSpatial],
                        inputLocal[localChannel * inputPaddedSpatial],
                        RoundMode::CAST_NONE, inputPaddedSpatial);
                }
            }
            PipeBarrier<PIPE_V>();
            inputQueue_.FreeTensor(inputLocal);
        }
    }

    float mean = 0.0f;
    float m2 = 0.0f;
    if constexpr (sizeof(T) == sizeof(float)) {
        if constexpr (ReuseResidentInput) {
            LocalTensor<float> source = xFp32Buffer_.Get<float>();
            if (tiling_->groupElements == NUMPY_PAIRWISE_TILE_ELEMENTS) {
                mean = ReduceSumScalar(source, NUMPY_PAIRWISE_TILE_ELEMENTS) /
                    static_cast<float>(NUMPY_PAIRWISE_TILE_ELEMENTS);
                LocalTensor<float> deviation = work0Buffer_.Get<float>();
                Adds(deviation, source, -mean, NUMPY_PAIRWISE_TILE_ELEMENTS);
                PipeBarrier<PIPE_V>();
                Mul(deviation, deviation, deviation, NUMPY_PAIRWISE_TILE_ELEMENTS);
                PipeBarrier<PIPE_ALL>();
                m2 = NumpyPairwiseSum4096(deviation);
            } else {
                ComputeTileMoments(source, tiling_->groupElements, mean, m2);
            }
        } else if (tiling_->tileElements == NUMPY_PAIRWISE_TILE_ELEMENTS &&
            tiling_->groupElements >= NUMPY_PAIRWISE_TILE_ELEMENTS &&
            tiling_->groupElements % NUMPY_PAIRWISE_TILE_ELEMENTS == 0) {
            mean = NumpyBufferedSum(groupOffset, 0.0f, false) /
                static_cast<float>(tiling_->groupElements);
            m2 = NumpyBufferedSum(groupOffset, mean, true);
        } else {
            if constexpr (EnableDoubleBuffer) {
                const int64_t firstCount = tiling_->groupElements < tiling_->tileElements ?
                    tiling_->groupElements : tiling_->tileElements;
                PrefetchInput(groupOffset, firstCount);
            }
            int64_t processed = 0;
            for (int64_t offset = 0; offset < tiling_->groupElements; offset += tiling_->tileElements) {
                const int64_t remain = tiling_->groupElements - offset;
                const int64_t count = remain < tiling_->tileElements ? remain : tiling_->tileElements;
                float tileMean = 0.0f;
                float tileM2 = 0.0f;
                if constexpr (EnableDoubleBuffer) {
                    const int64_t nextOffset = offset + tiling_->tileElements;
                    if (nextOffset < tiling_->groupElements) {
                        const int64_t nextRemain = tiling_->groupElements - nextOffset;
                        const int64_t nextCount = nextRemain < tiling_->tileElements ?
                            nextRemain : tiling_->tileElements;
                        PrefetchInput(groupOffset + nextOffset, nextCount);
                    }
                    LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
                    LocalTensor<float> source = inputLocal.template ReinterpretCast<float>();
                    ComputeTileMoments(source, count, tileMean, tileM2);
                    inputQueue_.FreeTensor(inputLocal);
                } else {
                    CopyInput(groupOffset + offset, count);
                    ComputeTileMoments(
                        xFp32Buffer_.Get<float>(), count, tileMean, tileM2);
                }
                if (processed == 0) {
                    mean = tileMean;
                    m2 = tileM2;
                } else {
                    const float newCount = static_cast<float>(processed + count);
                    const float delta = tileMean - mean;
                    mean += delta * static_cast<float>(count) / newCount;
                    m2 += tileM2 + delta * delta * static_cast<float>(processed) *
                        static_cast<float>(count) / newCount;
                }
                processed += count;
            }
        }
    } else if constexpr (ReuseResidentInput) {
        // The resident FP16 group has zero-filled channel padding. Reduce it in
        // one pass, then remove the padding contribution from both moments.
        float paddedMean = 0.0f;
        float paddedM2 = 0.0f;
        ComputeTileMoments(xFp32Buffer_.Get<float>(),
            paddedGroupElements, paddedMean, paddedM2);
        mean = paddedMean * (static_cast<float>(paddedGroupElements) /
            static_cast<float>(tiling_->groupElements));
        m2 = paddedM2 + static_cast<float>(paddedGroupElements) * paddedMean * paddedMean -
            static_cast<float>(tiling_->groupElements) * mean * mean;
    } else {
        int64_t processed = 0;
        if constexpr (EnableDoubleBuffer) {
            const int64_t firstCount = tiling_->groupElements < tiling_->tileElements ?
                tiling_->groupElements : tiling_->tileElements;
            PrefetchInput(groupOffset, firstCount);
        }
        for (int64_t offset = 0; offset < tiling_->groupElements; offset += tiling_->tileElements) {
            const int64_t remain = tiling_->groupElements - offset;
            const int64_t count = remain < tiling_->tileElements ? remain : tiling_->tileElements;
            if constexpr (EnableDoubleBuffer) {
                const int64_t nextOffset = offset + tiling_->tileElements;
                if (nextOffset < tiling_->groupElements) {
                    const int64_t nextRemain = tiling_->groupElements - nextOffset;
                    const int64_t nextCount = nextRemain < tiling_->tileElements ?
                        nextRemain : tiling_->tileElements;
                    PrefetchInput(groupOffset + nextOffset, nextCount);
                }
                LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
                Cast(xFp32Buffer_.Get<float>(), inputLocal, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                inputQueue_.FreeTensor(inputLocal);
            } else {
                CopyInput(groupOffset + offset, count);
            }
            float tileMean = 0.0f;
            float tileM2 = 0.0f;
            ComputeTileMoments(xFp32Buffer_.Get<float>(), count,
                tileMean, tileM2);
            if (processed == 0) {
                mean = tileMean;
                m2 = tileM2;
            } else {
                const float newCount = static_cast<float>(processed + count);
                const float delta = tileMean - mean;
                mean += delta * static_cast<float>(count) / newCount;
                m2 += tileM2 + delta * delta * static_cast<float>(processed) *
                    static_cast<float>(count) / newCount;
            }
            processed += count;
        }
    }
    float variance = m2 / static_cast<float>(tiling_->groupElements);
    variance = variance < 0.0f ? 0.0f : variance;
    const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);
    WriteStatistics(taskIndex, mean, rstd);

    LocalTensor<float> gammaFp32 = gammaFp32Buffer_.Get<float>();
    LocalTensor<float> betaFp32 = betaFp32Buffer_.Get<float>();
    LocalTensor<float> scaleRaw = parameterBuffer_.Get<uint8_t>()[groupScaleOffset_]
        .ReinterpretCast<float>();
    const bool useGroupedOutput = sizeof(T) == sizeof(float) &&
        tiling_->spatial % 8 == 0 &&
        tiling_->spatial <= tiling_->tileElements / 2;
    const bool usePaddedGroupedOutput = sizeof(T) != sizeof(float) &&
        tiling_->channelsPerGroup >= 4 && tiling_->channelsPerGroup <= UINT16_MAX &&
        tiling_->spatial >= 128 &&
        inputPaddedGroupElements <= tiling_->tileElements &&
        paddedGroupElements <= tiling_->tileElements;
    const bool usePaddedStreamOutput = sizeof(T) != sizeof(float) &&
        tiling_->channelsPerGroup >= 4 && tiling_->spatial >= 128 &&
        paddedSpatial <= tiling_->tileElements / 2;
    if constexpr (ReuseResidentInput && sizeof(T) != sizeof(float)) {
        QuantizeGroupedAndWrite(xFp32Buffer_.Get<float>(), groupOffset, 0,
            tiling_->channelsPerGroup, paddedSpatial, mean, rstd);
    } else if (usePaddedGroupedOutput) {
        LocalTensor<T> inputLocal = inputQueue_.template AllocTensor<T>();
        DataCopyExtParams inCopy{static_cast<uint16_t>(tiling_->channelsPerGroup),
            static_cast<uint32_t>(tiling_->spatial * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0,
            static_cast<uint8_t>(inputPaddedSpatial - tiling_->spatial), static_cast<T>(0)};
        DataCopyPad(inputLocal, selfGm_[groupOffset], inCopy, padParams);
        inputQueue_.EnQue(inputLocal);
        inputLocal = inputQueue_.template DeQue<T>();
        LocalTensor<float> source = xFp32Buffer_.Get<float>();
        if (inputPaddedSpatial == paddedSpatial) {
            Cast(source, inputLocal, RoundMode::CAST_NONE, paddedGroupElements);
        } else {
            Duplicate(source, 0.0f, paddedGroupElements);
            PipeBarrier<PIPE_V>();
            for (int64_t localChannel = 0; localChannel < tiling_->channelsPerGroup;
                 ++localChannel) {
                Cast(source[localChannel * paddedSpatial],
                    inputLocal[localChannel * inputPaddedSpatial],
                    RoundMode::CAST_NONE, inputPaddedSpatial);
            }
        }
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(inputLocal);
        QuantizeGroupedAndWrite(source, groupOffset, 0,
            tiling_->channelsPerGroup, paddedSpatial, mean, rstd);
    } else if (usePaddedStreamOutput) {
        const int64_t channelsPerTile = tiling_->tileElements / paddedSpatial;
        if constexpr (EnableDoubleBuffer) {
            const int64_t firstChannels = tiling_->channelsPerGroup < channelsPerTile ?
                tiling_->channelsPerGroup : channelsPerTile;
            PrefetchPaddedChannels(groupOffset, firstChannels, inputPaddedSpatial);
        }
        for (int64_t channelBase = 0; channelBase < tiling_->channelsPerGroup;
             channelBase += channelsPerTile) {
            const int64_t remainChannels = tiling_->channelsPerGroup - channelBase;
            const int64_t currentChannels = remainChannels < channelsPerTile ?
                remainChannels : channelsPerTile;
            if constexpr (EnableDoubleBuffer) {
                const int64_t nextChannelBase = channelBase + currentChannels;
                if (nextChannelBase < tiling_->channelsPerGroup) {
                    const int64_t nextRemainChannels =
                        tiling_->channelsPerGroup - nextChannelBase;
                    const int64_t nextChannels = nextRemainChannels < channelsPerTile ?
                        nextRemainChannels : channelsPerTile;
                    PrefetchPaddedChannels(
                        groupOffset + nextChannelBase * tiling_->spatial,
                        nextChannels, inputPaddedSpatial);
                }
            } else {
                PrefetchPaddedChannels(groupOffset + channelBase * tiling_->spatial,
                    currentChannels, inputPaddedSpatial);
            }
            LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
            LocalTensor<float> source = xFp32Buffer_.Get<float>();
            const int64_t paddedElements = currentChannels * paddedSpatial;
            if (inputPaddedSpatial == paddedSpatial) {
                Cast(source, inputLocal, RoundMode::CAST_NONE, paddedElements);
            } else {
                Duplicate(source, 0.0f, paddedElements);
                PipeBarrier<PIPE_V>();
                for (int64_t localChannel = 0; localChannel < currentChannels;
                     ++localChannel) {
                    Cast(source[localChannel * paddedSpatial],
                        inputLocal[localChannel * inputPaddedSpatial],
                        RoundMode::CAST_NONE, inputPaddedSpatial);
                }
            }
            PipeBarrier<PIPE_V>();
            inputQueue_.FreeTensor(inputLocal);
            QuantizeGroupedAndWrite(source,
                groupOffset + channelBase * tiling_->spatial,
                channelBase, currentChannels, paddedSpatial, mean, rstd);
        }
    } else if (useGroupedOutput) {
        const int64_t channelsPerTile = tiling_->tileElements / tiling_->spatial;
        if constexpr (EnableDoubleBuffer) {
            const int64_t firstChannels = tiling_->channelsPerGroup < channelsPerTile ?
                tiling_->channelsPerGroup : channelsPerTile;
            PrefetchInput(groupOffset, firstChannels * tiling_->spatial);
            for (int64_t channelBase = 0; channelBase < tiling_->channelsPerGroup;
                 channelBase += channelsPerTile) {
                const int64_t remainChannels = tiling_->channelsPerGroup - channelBase;
                const int64_t currentChannels = remainChannels < channelsPerTile ?
                    remainChannels : channelsPerTile;
                const int64_t nextChannelBase = channelBase + currentChannels;
                if (nextChannelBase < tiling_->channelsPerGroup) {
                    const int64_t nextRemainChannels = tiling_->channelsPerGroup - nextChannelBase;
                    const int64_t nextChannels = nextRemainChannels < channelsPerTile ?
                        nextRemainChannels : channelsPerTile;
                    PrefetchInput(groupOffset + nextChannelBase * tiling_->spatial,
                        nextChannels * tiling_->spatial);
                }
                LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
                LocalTensor<float> source = inputLocal.template ReinterpretCast<float>();
                QuantizeGroupedAndWrite(source,
                    groupOffset + channelBase * tiling_->spatial,
                    channelBase, currentChannels, tiling_->spatial, mean, rstd);
                inputQueue_.FreeTensor(inputLocal);
            }
        } else {
            for (int64_t channelBase = 0; channelBase < tiling_->channelsPerGroup;
                 channelBase += channelsPerTile) {
                const int64_t remainChannels = tiling_->channelsPerGroup - channelBase;
                const int64_t currentChannels = remainChannels < channelsPerTile ?
                    remainChannels : channelsPerTile;
                LocalTensor<float> source;
                if constexpr (ReuseResidentInput) {
                    source = xFp32Buffer_.Get<float>()[channelBase * tiling_->spatial];
                } else {
                    CopyInput(groupOffset + channelBase * tiling_->spatial,
                        currentChannels * tiling_->spatial);
                    source = xFp32Buffer_.Get<float>();
                }
                QuantizeGroupedAndWrite(source,
                    groupOffset + channelBase * tiling_->spatial,
                    channelBase, currentChannels, tiling_->spatial, mean, rstd);
            }
        }
    } else if constexpr (EnableDoubleBuffer) {
        const int64_t firstCount = tiling_->spatial < tiling_->tileElements ?
            tiling_->spatial : tiling_->tileElements;
        PrefetchInput(groupOffset, firstCount);
        for (int64_t channel = 0; channel < tiling_->channelsPerGroup; ++channel) {
            const float gamma = gammaFp32.GetValue(channel);
            const float beta = betaFp32.GetValue(channel);
            float quantScale;
            if constexpr (ForceSharedScale && !ReuseResidentInput) {
                quantScale = scaleRaw.GetValue(0);
            } else {
                quantScale = scaleRaw.GetValue(tiling_->quantScaleCount == 1 ? 0 : channel);
            }
            for (int64_t spatialOffset = 0; spatialOffset < tiling_->spatial;
                 spatialOffset += tiling_->tileElements) {
                const int64_t remain = tiling_->spatial - spatialOffset;
                const int64_t count = remain < tiling_->tileElements ? remain : tiling_->tileElements;
                const int64_t localGroupOffset = channel * tiling_->spatial + spatialOffset;
                const int64_t nextSpatialOffset = spatialOffset + tiling_->tileElements;
                if (nextSpatialOffset < tiling_->spatial) {
                    const int64_t nextRemain = tiling_->spatial - nextSpatialOffset;
                    const int64_t nextCount = nextRemain < tiling_->tileElements ?
                        nextRemain : tiling_->tileElements;
                    PrefetchInput(groupOffset + channel * tiling_->spatial + nextSpatialOffset,
                        nextCount);
                } else if (channel + 1 < tiling_->channelsPerGroup) {
                    PrefetchInput(groupOffset + (channel + 1) * tiling_->spatial, firstCount);
                }
                LocalTensor<T> inputLocal = inputQueue_.template DeQue<T>();
                LocalTensor<float> source;
                if constexpr (sizeof(T) == sizeof(float)) {
                    source = inputLocal.template ReinterpretCast<float>();
                } else {
                    source = xFp32Buffer_.Get<float>();
                    Cast(source, inputLocal, RoundMode::CAST_NONE, count);
                    PipeBarrier<PIPE_V>();
                }
                QuantizeAndWrite(source, groupOffset + localGroupOffset,
                    count, mean, rstd, gamma, beta, quantScale);
                inputQueue_.FreeTensor(inputLocal);
            }
        }
    } else {
        for (int64_t channel = 0; channel < tiling_->channelsPerGroup; ++channel) {
            const float gamma = gammaFp32.GetValue(channel);
            const float beta = betaFp32.GetValue(channel);
            float quantScale;
            if constexpr (ForceSharedScale && !ReuseResidentInput) {
                quantScale = scaleRaw.GetValue(0);
            } else {
                quantScale = scaleRaw.GetValue(tiling_->quantScaleCount == 1 ? 0 : channel);
            }
            for (int64_t spatialOffset = 0; spatialOffset < tiling_->spatial;
                 spatialOffset += tiling_->tileElements) {
                const int64_t remain = tiling_->spatial - spatialOffset;
                const int64_t count = remain < tiling_->tileElements ? remain : tiling_->tileElements;
                const int64_t localGroupOffset = channel * tiling_->spatial + spatialOffset;
                // The resident key guarantees 32-byte channel starts. Other keys
                // reload each slice into the aligned buffer base.
                LocalTensor<float> source;
                if constexpr (ReuseResidentInput) {
                    source = xFp32Buffer_.Get<float>()[localGroupOffset];
                } else {
                    CopyInput(groupOffset + localGroupOffset, count);
                    source = xFp32Buffer_.Get<float>();
                }
                QuantizeAndWrite(source, groupOffset + localGroupOffset,
                    count, mean, rstd, gamma, beta, quantScale);
            }
        }
    }
}

template <typename T, bool EnableDoubleBuffer, bool SkipQuantClamp, bool ReuseResidentInput, bool ReuseFullGroup, bool ForceActivateSilu, bool ForceSharedScale>
__aicore__ inline void GroupNormSiluQuant<T, EnableDoubleBuffer, SkipQuantClamp, ReuseResidentInput, ReuseFullGroup, ForceActivateSilu, ForceSharedScale>::Process()
{
    const int64_t base = tiling_->totalGroups / tiling_->workerNum;
    const int64_t remainder = tiling_->totalGroups % tiling_->workerNum;
    const int64_t start = blockIndex_ * base + (blockIndex_ < remainder ? blockIndex_ : remainder);
    const int64_t count = base + (blockIndex_ < remainder ? 1 : 0);
    if constexpr (sizeof(T) == sizeof(float) && ReuseFullGroup && !EnableDoubleBuffer) {
        for (int64_t batchStart = 0; batchStart < count; batchStart += STATISTICS_BATCH_ELEMENTS) {
            const int64_t remain = count - batchStart;
            const int64_t batchCount = remain < STATISTICS_BATCH_ELEMENTS ?
                remain : STATISTICS_BATCH_ELEMENTS;
            for (int64_t index = 0; index < batchCount; ++index) {
                statisticsSlot_ = index;
                ProcessGroup(start + batchStart + index);
            }
            FlushStatistics(start + batchStart, batchCount);
        }
    } else if constexpr (sizeof(T) != sizeof(float)) {
        if (count >= 2) {
            for (int64_t batchStart = 0; batchStart < count; batchStart += STATISTICS_BATCH_ELEMENTS) {
                const int64_t remain = count - batchStart;
                const int64_t batchCount = remain < STATISTICS_BATCH_ELEMENTS ?
                    remain : STATISTICS_BATCH_ELEMENTS;
                for (int64_t index = 0; index < batchCount; ++index) {
                    statisticsSlot_ = index;
                    ProcessGroup(start + batchStart + index);
                }
                FlushStatistics(start + batchStart, batchCount);
            }
        } else {
            statisticsSlot_ = -1;
            for (int64_t index = 0; index < count; ++index) {
                ProcessGroup(start + index);
            }
        }
    } else {
        for (int64_t index = 0; index < count; ++index) {
            ProcessGroup(start + index);
        }
    }
}

// A fixed UB image avoids queue and TPipe setup for short, aligned FP32
// groups. The Host checks all bounds and selects this path only for SiLU with
// a shared quantization scale.
class Fp32TinyGroupNormSiluQuant {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta,
        GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut,
        const GroupNormSiluQuantTilingData* tilingData)
    {
        tiling_ = tilingData;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(self));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(quantScale));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out));
        meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(meanOut));
        rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rstdOut));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskIndex = GetBlockIdx();
        const int64_t groupIndex = taskIndex % tiling_->numGroups;
        const int64_t groupOffset = taskIndex * tiling_->groupElements;
        const int64_t channelStart = groupIndex * tiling_->channelsPerGroup;
        const int32_t count = static_cast<int32_t>(tiling_->groupElements);
        const int32_t spatial = static_cast<int32_t>(tiling_->spatial);
        LocalTensor<float> raw(TPosition::VECCALC, 0U, 256U);
        LocalTensor<float> value(TPosition::VECCALC, 1024U, 256U);
        LocalTensor<float> aux(TPosition::VECCALC, 2048U, 256U);
        LocalTensor<float> quotient(TPosition::VECCALC, 3072U, 256U);
        LocalTensor<float> rounded(TPosition::VECCALC, 4096U, 256U);
        LocalTensor<int8_t> output(TPosition::VECCALC, 5120U, 256U);
        LocalTensor<float> gammaLocal(TPosition::VECCALC, 5632U, 64U);
        LocalTensor<float> betaLocal(TPosition::VECCALC, 5888U, 64U);
        LocalTensor<float> scaleLocal(TPosition::VECCALC, 6144U, 64U);
        LocalTensor<float> reduction(TPosition::VECCALC, 6400U, 64U);
        LocalTensor<float> scratch(TPosition::VECCALC, 6656U, 256U);
        LocalTensor<uint8_t> mask(TPosition::VECCALC, 7680U, 256U);
        LocalTensor<float> stat(TPosition::VECCALC, 7936U, 64U);
        const DataCopyPadExtParams<float> noPad{false, 0, 0, 0};
        const DataCopyExtParams inCopy{1,
            static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        const DataCopyExtParams paramCopy{1,
            static_cast<uint32_t>(tiling_->channelsPerGroup * sizeof(float)), 0, 0, 0};
        const DataCopyExtParams scaleCopy{1, sizeof(float), 0, 0, 0};
        DataCopyPad(raw, selfGm_[groupOffset], inCopy, noPad);
        DataCopyPad(gammaLocal, gammaGm_[channelStart], paramCopy, noPad);
        DataCopyPad(betaLocal, betaGm_[channelStart], paramCopy, noPad);
        DataCopyPad(scaleLocal, scaleGm_, scaleCopy, noPad);
        SetFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::MTE2_S>(1);
        WaitFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_S>(1);

        WholeReduceSum(reduction, raw, count, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        const float mean = reduction.GetValue(0) / static_cast<float>(count);
        Adds(value, raw, -mean, count);
        PipeBarrier<PIPE_V>();
        Mul(value, value, value, count);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(reduction, value, count, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        const float variance = reduction.GetValue(0) / static_cast<float>(count);
        const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);
        stat.SetValue(0, mean);
        stat.SetValue(8, rstd);
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        const DataCopyExtParams scalarCopy{1, sizeof(float), 0, 0, 0};
        DataCopyPad(meanGm_[taskIndex], stat, scalarCopy);
        DataCopyPad(rstdGm_[taskIndex], stat[8], scalarCopy);

        Adds(value, raw, -mean, count);
        PipeBarrier<PIPE_V>();
        Muls(value, value, rstd, count);
        PipeBarrier<PIPE_V>();
        for (int32_t channel = 0; channel < tiling_->channelsPerGroup; ++channel) {
            Duplicate(raw[channel * spatial], gammaLocal.GetValue(channel), spatial);
        }
        PipeBarrier<PIPE_V>();
        Mul(value, value, raw, count);
        PipeBarrier<PIPE_V>();
        for (int32_t channel = 0; channel < tiling_->channelsPerGroup; ++channel) {
            Duplicate(raw[channel * spatial], betaLocal.GetValue(channel), spatial);
        }
        PipeBarrier<PIPE_V>();
        Add(value, value, raw, count);
        PipeBarrier<PIPE_V>();
        Muls(aux, value, -1.0f, count);
        PipeBarrier<PIPE_V>();
        Exp(aux, aux, count);
        PipeBarrier<PIPE_V>();
        Adds(aux, aux, 1.0f, count);
        PipeBarrier<PIPE_V>();
        Div(quotient, value, aux, count);
        PipeBarrier<PIPE_V>();
        const float quantScale = scaleLocal.GetValue(0);
        Muls(quotient, quotient, 1.0f / quantScale, count);
        PipeBarrier<PIPE_V>();
        Cast(rounded, quotient, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Sub(raw, quotient, rounded, count);
        PipeBarrier<PIPE_V>();
        Abs(raw, raw, count);
        PipeBarrier<PIPE_V>();
        const int32_t compareCount = (count + 63) / 64 * 64;
        CompareScalar(mask, raw, 0.5f - MULTI_TILE_QUANT_HALF_BOUNDARY_GUARD,
            CMPMODE::GE, compareCount);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<uint64_t> maskWords = mask.ReinterpretCast<uint64_t>();
        const int32_t wordCount = (count + 63) / 64;
        for (int32_t word = 0; word < wordCount; ++word) {
            uint64_t bits = maskWords.GetValue(word);
            if (word == wordCount - 1 && (count & 63) != 0) {
                bits &= (uint64_t(1) << (count & 63)) - 1;
            }
            while (bits != 0) {
                const int32_t bit = static_cast<int32_t>(GetSFFValue<1>(bits));
                const int32_t index = word * 64 + bit;
                const float exactValue = value.GetValue(index) / aux.GetValue(index);
                const float scalarQuotient = exactValue / quantScale;
                int32_t roundedValue = scalarQuotient >= 127.0f ? 127 :
                    (scalarQuotient <= -128.0f ? -128 :
                    ScalarCast<float, int32_t, RoundMode::CAST_RINT>(scalarQuotient));
                rounded.SetValue(index, static_cast<float>(roundedValue));
                bits &= ~(uint64_t(1) << bit);
            }
        }
        PipeBarrier<PIPE_ALL>();
        LocalTensor<half> roundedHalf = quotient.ReinterpretCast<half>();
        Cast(roundedHalf, rounded, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Cast(output, roundedHalf, RoundMode::CAST_NONE, count);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const DataCopyExtParams outCopy{1, static_cast<uint32_t>(count), 0, 0, 0};
        DataCopyPad(outGm_[groupOffset], output, outCopy);
    }

private:
    GlobalTensor<float> selfGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<float> betaGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<int8_t> outGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;
    const GroupNormSiluQuantTilingData* tiling_ = nullptr;
};

class Fp16TinyGroupNormSiluQuant {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta,
        GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut,
        const GroupNormSiluQuantTilingData* tilingData)
    {
        tiling_ = tilingData;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(self));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(gamma));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(beta));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(quantScale));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out));
        meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(meanOut));
        rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(rstdOut));
    }

    __aicore__ inline void Process()
    {
        const int64_t taskIndex = GetBlockIdx();
        const int64_t groupIndex = taskIndex % tiling_->numGroups;
        const int64_t groupOffset = taskIndex * tiling_->groupElements;
        const int64_t channelStart = groupIndex * tiling_->channelsPerGroup;
        const int32_t count = static_cast<int32_t>(tiling_->groupElements);
        const int32_t spatial = static_cast<int32_t>(tiling_->spatial);
        LocalTensor<half> raw(TPosition::VECCALC, 0U, 512U);
        LocalTensor<float> input(TPosition::VECCALC, 1024U, 1024U);
        LocalTensor<float> value(TPosition::VECCALC, 2048U, 1024U);
        LocalTensor<float> aux(TPosition::VECCALC, 3072U, 1024U);
        LocalTensor<float> quotient(TPosition::VECCALC, 4096U, 1024U);
        LocalTensor<float> rounded(TPosition::VECCALC, 5120U, 1024U);
        LocalTensor<int8_t> output(TPosition::VECCALC, 6144U, 256U);
        LocalTensor<half> gammaLocal(TPosition::VECCALC, 6400U, 128U);
        LocalTensor<half> betaLocal(TPosition::VECCALC, 6656U, 128U);
        LocalTensor<float> scaleLocal(TPosition::VECCALC, 6912U, 64U);
        LocalTensor<float> reduction(TPosition::VECCALC, 7168U, 64U);
        LocalTensor<uint8_t> mask(TPosition::VECCALC, 7424U, 256U);
        LocalTensor<half> stat(TPosition::VECCALC, 7680U, 128U);
        LocalTensor<float> parameterVector(TPosition::VECCALC, 8192U, 1024U);
        LocalTensor<float> betaVector(TPosition::VECCALC, 9216U, 1024U);
        const DataCopyPadExtParams<half> noHalfPad{false, 0, 0, 0};
        const DataCopyPadExtParams<float> noFloatPad{false, 0, 0, 0};
        const DataCopyExtParams inCopy{1,
            static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
        const DataCopyExtParams paramCopy{1,
            static_cast<uint32_t>(tiling_->channelsPerGroup * sizeof(half)), 0, 0, 0};
        const DataCopyExtParams scaleCopy{1, sizeof(float), 0, 0, 0};
        DataCopyPad(raw, selfGm_[groupOffset], inCopy, noHalfPad);
        DataCopyPad(gammaLocal, gammaGm_[channelStart], paramCopy, noHalfPad);
        DataCopyPad(betaLocal, betaGm_[channelStart], paramCopy, noHalfPad);
        DataCopyPad(scaleLocal, scaleGm_, scaleCopy, noFloatPad);
        SetFlag<HardEvent::MTE2_V>(0);
        SetFlag<HardEvent::MTE2_S>(1);
        WaitFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_S>(1);
        Cast(input, raw, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();

        float sum = 0.0f;
        if (count <= 64) {
            WholeReduceSum(reduction, input, count, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(1);
            WaitFlag<HardEvent::V_S>(1);
            sum = reduction.GetValue(0);
        } else {
            WholeReduceSum(reduction, input, 64, 1, 1, 1, 8);
            WholeReduceSum(reduction[8], input[64], count - 64, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(1);
            WaitFlag<HardEvent::V_S>(1);
            sum = reduction.GetValue(0) + reduction.GetValue(8);
        }
        const float mean = sum / static_cast<float>(count);
        Adds(value, input, -mean, count);
        PipeBarrier<PIPE_V>();
        Mul(quotient, value, value, count);
        PipeBarrier<PIPE_V>();
        float squaredSum = 0.0f;
        if (count <= 64) {
            WholeReduceSum(reduction, quotient, count, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(1);
            WaitFlag<HardEvent::V_S>(1);
            squaredSum = reduction.GetValue(0);
        } else {
            WholeReduceSum(reduction, quotient, 64, 1, 1, 1, 8);
            WholeReduceSum(reduction[8], quotient[64], count - 64, 1, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(1);
            WaitFlag<HardEvent::V_S>(1);
            squaredSum = reduction.GetValue(0) + reduction.GetValue(8);
        }
        const float variance = squaredSum / static_cast<float>(count);
        const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);
        stat.SetValue(0, static_cast<half>(mean));
        stat.SetValue(16, static_cast<half>(rstd));
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        const DataCopyExtParams scalarCopy{1, sizeof(half), 0, 0, 0};
        DataCopyPad(meanGm_[taskIndex], stat, scalarCopy);
        DataCopyPad(rstdGm_[taskIndex], stat[16], scalarCopy);

        Muls(value, value, rstd, count);
        PipeBarrier<PIPE_V>();
        if ((spatial & 7) == 0) {
            for (int32_t channel = 0; channel < tiling_->channelsPerGroup; ++channel) {
                Duplicate(parameterVector[channel * spatial],
                    static_cast<float>(gammaLocal.GetValue(channel)), spatial);
                Duplicate(betaVector[channel * spatial],
                    static_cast<float>(betaLocal.GetValue(channel)), spatial);
            }
        } else {
            // Fill the last channel first, then overwrite only the first
            // channel. The prefix fits in one repeat for this two-channel
            // tiny path, even when the full group crosses 64 FP32 elements.
            const int32_t lastChannel = tiling_->channelsPerGroup - 1;
            Duplicate(parameterVector,
                static_cast<float>(gammaLocal.GetValue(lastChannel)), count);
            Duplicate(betaVector,
                static_cast<float>(betaLocal.GetValue(lastChannel)), count);
            if (tiling_->channelsPerGroup == 2) {
                PipeBarrier<PIPE_V>();
                Duplicate(parameterVector,
                    static_cast<float>(gammaLocal.GetValue(0)), spatial);
                Duplicate(betaVector,
                    static_cast<float>(betaLocal.GetValue(0)), spatial);
            }
        }
        PipeBarrier<PIPE_V>();
        Mul(value, value, parameterVector, count);
        PipeBarrier<PIPE_V>();
        Add(value, value, betaVector, count);
        PipeBarrier<PIPE_V>();
        // A single external counter mask covers this contiguous FP32
        // arithmetic stage. Restore normal mode before comparisons/casts.
        const UnaryRepeatParams counterUnary{1, 1, 8, 8};
        const BinaryRepeatParams counterBinary{1, 1, 1, 8, 8, 8};
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(count);
        Muls<float, false>(aux, value, -1.0f, MASK_PLACEHOLDER, 1, counterUnary);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(aux, aux, MASK_PLACEHOLDER, 1, counterUnary);
        PipeBarrier<PIPE_V>();
        Adds<float, false>(aux, aux, 1.0f, MASK_PLACEHOLDER, 1, counterUnary);
        PipeBarrier<PIPE_V>();
        Div<float, false>(quotient, value, aux, MASK_PLACEHOLDER, 1, counterBinary);
        PipeBarrier<PIPE_V>();
        const float quantScale = scaleLocal.GetValue(0);
        Muls<float, false>(quotient, quotient, 1.0f / quantScale, MASK_PLACEHOLDER, 1, counterUnary);
        PipeBarrier<PIPE_V>();
        Cast<float, float, false>(rounded, quotient, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, counterUnary);
        PipeBarrier<PIPE_V>();
        Sub<float, false>(input, quotient, rounded, MASK_PLACEHOLDER, 1, counterBinary);
        PipeBarrier<PIPE_V>();
        Abs<float, false>(input, input, MASK_PLACEHOLDER, 1, counterUnary);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        CompareScalar(mask, input, 0.5f - MULTI_TILE_QUANT_HALF_BOUNDARY_GUARD,
            CMPMODE::GE, (count + 63) / 64 * 64);
        PipeBarrier<PIPE_ALL>();
        LocalTensor<uint64_t> maskWords = mask.ReinterpretCast<uint64_t>();
        const int32_t wordCount = (count + 63) / 64;
        for (int32_t word = 0; word < wordCount; ++word) {
            uint64_t bits = maskWords.GetValue(word);
            if (word == wordCount - 1 && (count & 63) != 0) {
                bits &= (uint64_t(1) << (count & 63)) - 1;
            }
            while (bits != 0) {
                const int32_t bit = static_cast<int32_t>(GetSFFValue<1>(bits));
                const int32_t index = word * 64 + bit;
                const float exactValue = value.GetValue(index) / aux.GetValue(index);
                const float scalarQuotient = exactValue / quantScale;
                const int32_t roundedValue = scalarQuotient >= 127.0f ? 127 :
                    (scalarQuotient <= -128.0f ? -128 :
                    ScalarCast<float, int32_t, RoundMode::CAST_RINT>(scalarQuotient));
                rounded.SetValue(index, static_cast<float>(roundedValue));
                bits &= ~(uint64_t(1) << bit);
            }
        }
        PipeBarrier<PIPE_ALL>();
        LocalTensor<half> roundedHalf = quotient.ReinterpretCast<half>();
        Cast(roundedHalf, rounded, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Cast(output, roundedHalf, RoundMode::CAST_NONE, count);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const DataCopyExtParams outCopy{1, static_cast<uint32_t>(count), 0, 0, 0};
        DataCopyPad(outGm_[groupOffset], output, outCopy);
    }

private:
    GlobalTensor<half> selfGm_;
    GlobalTensor<half> gammaGm_;
    GlobalTensor<half> betaGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<int8_t> outGm_;
    GlobalTensor<half> meanGm_;
    GlobalTensor<half> rstdGm_;
    const GroupNormSiluQuantTilingData* tiling_ = nullptr;
};

// Queue-free resident path. Host tile capacity provides at least 104 KiB UB;
// padded groups contain at most 4096 elements and hence at most 128 channels.
// Other groups retain the original resident or streaming implementation.
template <typename T>
class FixedResidentGroupNormSiluQuant {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta,
        GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut,
        const GroupNormSiluQuantTilingData* tilingData)
    {
        tiling_ = tilingData;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gamma));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(beta));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(quantScale));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out));
        meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(meanOut));
        rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rstdOut));
    }

    __aicore__ inline void Process()
    {
        const int64_t worker = GetBlockIdx();
        if (tiling_->workerNum == tiling_->totalGroups) {
            ProcessGroup(worker);
            return;
        }
        const int64_t base = tiling_->totalGroups / tiling_->workerNum;
        const int64_t extra = tiling_->totalGroups % tiling_->workerNum;
        const int64_t start = worker * base + (worker < extra ? worker : extra);
        const int64_t groups = base + (worker < extra ? 1 : 0);
        for (int64_t i = 0; i < groups; ++i) {
            ProcessGroup(start + i);
            if (i + 1 < groups) {
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessGroup(int64_t taskIndex)
    {
        const int64_t groupIndex = taskIndex % tiling_->numGroups;
        const int64_t groupOffset = taskIndex * tiling_->groupElements;
        const int64_t channelStart = groupIndex * tiling_->channelsPerGroup;
        const int32_t spatial = static_cast<int32_t>(tiling_->spatial);
        const int32_t channels = static_cast<int32_t>(tiling_->channelsPerGroup);
        const int32_t inputBlock = 32 / sizeof(T);
        const int32_t inputStride = (spatial + inputBlock - 1) / inputBlock * inputBlock;
        const int32_t rowStride = (spatial + 31) / 32 * 32;
        const int32_t count = rowStride * channels;
        // Capacity: 4096 padded elements, 128 channel parameters, 104 KiB UB.
        LocalTensor<T> raw(TPosition::VECCALC, 0U, 4096U);
        LocalTensor<float> input(TPosition::VECCALC, 16384U, 4096U);
        LocalTensor<float> value(TPosition::VECCALC, 32768U, 4096U);
        LocalTensor<float> aux(TPosition::VECCALC, 49152U, 4096U);
        LocalTensor<float> quotient(TPosition::VECCALC, 65536U, 4096U);
        LocalTensor<float> rounded(TPosition::VECCALC, 81920U, 4096U);
        // FP16 preserves centered values in value while the already allocated
        // quotient buffer holds squared deviations until statistics complete.
        LocalTensor<float> varianceValue(TPosition::VECCALC,
            sizeof(T) == sizeof(float) ? 32768U : 65536U, 4096U);
        LocalTensor<int8_t> output(TPosition::VECCALC, 98304U, 4096U);
        LocalTensor<T> gammaLocal(TPosition::VECCALC, 102400U, 128U);
        LocalTensor<T> betaLocal(TPosition::VECCALC, 102912U, 128U);
        LocalTensor<float> scaleLocal(TPosition::VECCALC, 103424U, 128U);
        LocalTensor<float> reduction(TPosition::VECCALC, 103936U, 64U);
        LocalTensor<uint8_t> mask(TPosition::VECCALC, 104192U, 512U);
        LocalTensor<T> stat(TPosition::VECCALC, 104704U, 32U);
        LocalTensor<uint8_t> directory(TPosition::VECCALC, 104832U, 256U);
        const DataCopyPadExtParams<T> noHalfPad{false, 0, 0, 0};
        const DataCopyPadExtParams<float> noFloatPad{false, 0, 0, 0};
        const DataCopyExtParams inCopy{static_cast<uint16_t>(channels),
            static_cast<uint32_t>(spatial * sizeof(T)), 0, 0, 0};
        const DataCopyPadExtParams<T> inputPad{true, 0,
            static_cast<uint8_t>(inputStride - spatial), static_cast<T>(0)};
        const DataCopyExtParams paramCopy{1,
            static_cast<uint32_t>(channels * sizeof(T)), 0, 0, 0};
        const int32_t scaleCount = tiling_->quantScaleCount == 1 ? 1 : channels;
        const DataCopyExtParams scaleCopy{1,
            static_cast<uint32_t>(scaleCount * sizeof(float)), 0, 0, 0};
        if constexpr (sizeof(T) == sizeof(float)) {
            const DataCopyPadExtParams<T> fp32Pad{true, 0,
                static_cast<uint8_t>(rowStride - spatial), static_cast<T>(0)};
            DataCopyPad(input, selfGm_[groupOffset], inCopy, fp32Pad);
        } else {
            DataCopyPad(raw, selfGm_[groupOffset], inCopy, inputPad);
        }
        if constexpr (sizeof(T) != sizeof(float)) {
            SetFlag<HardEvent::MTE2_V>(0);
        }
        DataCopyPad(gammaLocal, gammaGm_[channelStart], paramCopy, noHalfPad);
        DataCopyPad(betaLocal, betaGm_[channelStart], paramCopy, noHalfPad);
        DataCopyPad(scaleLocal, scaleGm_[tiling_->quantScaleCount == 1 ? 0 : channelStart], scaleCopy, noFloatPad);
        if constexpr (sizeof(T) == sizeof(float)) {
            SetFlag<HardEvent::MTE2_V>(0);
        }
        SetFlag<HardEvent::MTE2_S>(1);
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr (sizeof(T) == sizeof(float)) {
            WaitFlag<HardEvent::MTE2_S>(1);
        }
        if constexpr (sizeof(T) != sizeof(float)) {
            if (inputStride == rowStride) {
                Cast(input, raw, RoundMode::CAST_NONE, count);
            } else {
                Duplicate(input, 0.0f, count);
                PipeBarrier<PIPE_V>();
                for (int32_t channel = 0; channel < channels; ++channel) {
                    Cast(input[channel * rowStride], raw[channel * inputStride],
                        RoundMode::CAST_NONE, inputStride);
                }
            }
            PipeBarrier<PIPE_V>();
        }
        const float mean = Sum(input, reduction, count) /
            static_cast<float>(tiling_->groupElements);
        Adds(value, input, -mean, count);
        PipeBarrier<PIPE_V>();
        Mul(varianceValue, value, value, count);
        PipeBarrier<PIPE_V>();
        // Zero the padded lanes before the variance reduction. Subtracting
        // their contribution afterwards loses accuracy for large means.
        if (rowStride != spatial) {
            const int32_t validTail = spatial & 31;
            uint64_t padMask[2] = {
                ((uint64_t(1) << (32 - validTail)) - 1) << validTail, 0U};
            for (int32_t channel = 0; channel < channels; ++channel) {
                Duplicate(varianceValue[channel * rowStride + rowStride - 32],
                    0.0f, padMask, 1, 1, 8);
            }
            PipeBarrier<PIPE_V>();
        }
        float squaredSum = 0.0f;
        if constexpr (sizeof(T) == sizeof(float)) {
            if (count == NUMPY_PAIRWISE_TILE_ELEMENTS) {
                squaredSum = PairwiseVarianceSum(varianceValue, aux, rounded, reduction);
            } else {
                squaredSum = Sum(varianceValue, reduction, count);
            }
        } else {
            squaredSum = Sum(varianceValue, reduction, count);
        }
        const float variance = squaredSum / static_cast<float>(tiling_->groupElements);
        const float rstd = 1.0f / sqrt(variance + tiling_->epsilon);
        stat.SetValue(0, static_cast<T>(mean));
        stat.SetValue(16, static_cast<T>(rstd));
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        const DataCopyExtParams scalarCopy{1, sizeof(T), 0, 0, 0};
        DataCopyPad(meanGm_[taskIndex], stat, scalarCopy);
        DataCopyPad(rstdGm_[taskIndex], stat[16], scalarCopy);

        if constexpr (sizeof(T) == sizeof(float)) {
            Adds(value, input, -mean, count);
            PipeBarrier<PIPE_V>();
        }
        Muls(value, value, rstd, count);
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) != sizeof(float)) {
            // Parameters are first consumed here, after both statistics.
            WaitFlag<HardEvent::MTE2_S>(1);
        }
        for (int32_t channel = 0; channel < channels; ++channel) {
            Muls(value[channel * rowStride], value[channel * rowStride],
                static_cast<float>(gammaLocal.GetValue(channel)), rowStride);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t channel = 0; channel < channels; ++channel) {
            Adds(value[channel * rowStride], value[channel * rowStride],
                static_cast<float>(betaLocal.GetValue(channel)), rowStride);
        }
        PipeBarrier<PIPE_V>();
        if (tiling_->activateSilu != 0) {
            Muls(aux, value, -1.0f, count);
            PipeBarrier<PIPE_V>();
            Exp(aux, aux, count);
            PipeBarrier<PIPE_V>();
            Adds(aux, aux, 1.0f, count);
            PipeBarrier<PIPE_V>();
            Div(quotient, value, aux, count);
        } else {
            Adds(quotient, value, 0.0f, count);
        }
        PipeBarrier<PIPE_V>();
        if (scaleCount == 1) {
            Muls(quotient, quotient, 1.0f / scaleLocal.GetValue(0), count);
        } else {
            for (int32_t channel = 0; channel < channels; ++channel) {
                Muls(quotient[channel * rowStride], quotient[channel * rowStride],
                    1.0f / scaleLocal.GetValue(channel), rowStride);
            }
        }
        PipeBarrier<PIPE_V>();
        Cast(rounded, quotient, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Sub(input, quotient, rounded, count);
        PipeBarrier<PIPE_V>();
        Abs(input, input, count);
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, input, 0.5f - MULTI_TILE_QUANT_HALF_BOUNDARY_GUARD,
            CMPMODE::GE, (count + 63) / 64 * 64);
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> maskDwords = mask.ReinterpretCast<int32_t>();
        const int32_t dwordCount = (count + 31) / 32;
        CompareScalar(directory, maskDwords, static_cast<int32_t>(0), CMPMODE::EQ,
            (dwordCount + 63) / 64 * 64);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        LocalTensor<uint64_t> directoryWords = directory.ReinterpretCast<uint64_t>();
        for (int32_t word = 0; word < (dwordCount + 63) / 64; ++word) {
            uint64_t dwords = ~directoryWords.GetValue(word);
            if (word == (dwordCount - 1) / 64 && (dwordCount & 63) != 0) {
                dwords &= (uint64_t(1) << (dwordCount & 63)) - 1;
            }
            while (dwords != 0) {
                const int32_t dwordBit = static_cast<int32_t>(GetSFFValue<1>(dwords));
                const int32_t dwordIndex = word * 64 + dwordBit;
                uint32_t bits = static_cast<uint32_t>(maskDwords.GetValue(dwordIndex));
                while (bits != 0) {
                    const int32_t bit = static_cast<int32_t>(GetSFFValue<1>(static_cast<uint64_t>(bits)));
                    const int32_t index = dwordIndex * 32 + bit;
                    float exactValue = value.GetValue(index);
                    if (tiling_->activateSilu != 0) {
                        exactValue /= aux.GetValue(index);
                    }
                    const float scalarQuotient = exactValue /
                        scaleLocal.GetValue(scaleCount == 1 ? 0 : index / rowStride);
                    const int32_t roundedValue = scalarQuotient >= 127.0f ? 127 :
                        (scalarQuotient <= -128.0f ? -128 :
                        ScalarCast<float, int32_t, RoundMode::CAST_RINT>(scalarQuotient));
                    rounded.SetValue(index, static_cast<float>(roundedValue));
                    bits &= ~(uint32_t(1) << bit);
                }
                dwords &= ~(uint64_t(1) << dwordBit);
            }
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        LocalTensor<half> roundedHalf = quotient.ReinterpretCast<half>();
        Cast(roundedHalf, rounded, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Cast(output, roundedHalf, RoundMode::CAST_NONE, count);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const DataCopyExtParams outCopy{static_cast<uint16_t>(channels),
            static_cast<uint32_t>(spatial), 0, 0, 0};
        DataCopyPad(outGm_[groupOffset], output, outCopy);
    }

private:
    __aicore__ inline float PairwiseVarianceSum(const LocalTensor<float>& source,
        const LocalTensor<float>& accumulators, const LocalTensor<float>& pairScratch,
        const LocalTensor<float>& partial)
    {
        // NumPy 1.26.4 uses PW_BLOCKSIZE=128.  A 128-element leaf has eight
        // independent accumulators, followed by the explicitly grouped tree.
        // Execute the same addition step across all 32 independent leaves with
        // Level-0 repeat strides; the FP32 association inside each leaf is
        // unchanged.  4096 is exactly 32 leaves, so no tail path is required.
        constexpr int64_t leafElements = 128;
        constexpr int64_t accumulatorCount = 8;
        constexpr int64_t leafCount = NUMPY_PAIRWISE_TILE_ELEMENTS / leafElements;
        constexpr uint64_t accumulatorMask = accumulatorCount * accumulatorCount;
        constexpr uint8_t leafRepeats = static_cast<uint8_t>(leafCount / accumulatorCount);
        constexpr uint8_t leafStrideBlocks =
            static_cast<uint8_t>(leafElements * sizeof(float) / BLOCK_BYTES);
        constexpr uint8_t leafBatchStrideBlocks =
            static_cast<uint8_t>(accumulatorCount * leafStrideBlocks);

        constexpr uint8_t addLeafBatch = accumulatorCount;
        constexpr uint64_t addMask = addLeafBatch * accumulatorCount;
        constexpr uint8_t addRepeats = static_cast<uint8_t>(leafCount / addLeafBatch);
        constexpr uint8_t addBatchStrideBlocks =
            static_cast<uint8_t>(addLeafBatch * leafStrideBlocks);
        const BinaryRepeatParams acrossLeaves{leafStrideBlocks, leafStrideBlocks, leafStrideBlocks,
            addBatchStrideBlocks, addBatchStrideBlocks, addBatchStrideBlocks};

        Add<float, true>(accumulators, source, source[accumulatorCount],
            addMask, addRepeats, acrossLeaves);
        for (int64_t offset = accumulatorCount * 2; offset < leafElements;
             offset += accumulatorCount) {
            PipeBarrier<PIPE_V>();
            Add<float, false>(accumulators, accumulators, source[offset],
                addMask, addRepeats, acrossLeaves);
        }

        // Preserve the same 8 -> 4 -> 2 -> 1 adjacent-pair leaf tree while
        // filling all vector blocks.  The first stage consumes eight strided
        // leaf roots per repeat; the following stages operate on packed roots.
        PipeBarrier<PIPE_V>();
        PairReduceSum<float, true>(pairScratch, accumulators,
            static_cast<int32_t>(leafRepeats), static_cast<int32_t>(accumulatorMask),
            1, static_cast<int32_t>(leafStrideBlocks),
            static_cast<int32_t>(leafBatchStrideBlocks));
        PipeBarrier<PIPE_V>();
        PairReduceSum<float, true>(accumulators, pairScratch,
            2, static_cast<int32_t>(accumulatorMask), 1, 1, 8);
        PipeBarrier<PIPE_V>();
        PairReduceSum<float, true>(partial, accumulators,
            1, static_cast<int32_t>(accumulatorMask), 1, 1, 8);
        PipeBarrier<PIPE_V>();

        PairReduceSum<float, true>(pairScratch, partial, 1, 32, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        PairReduceSum<float, true>(partial, pairScratch, 1, 16, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        PairReduceSum<float, true>(pairScratch, partial, 1, 8, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        PairReduceSum<float, true>(partial, pairScratch, 1, 4, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        PairReduceSum<float, true>(pairScratch, partial, 1, 2, 1, 1, 8);
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        return pairScratch.GetValue(0);
    }

    __aicore__ inline float Sum(const LocalTensor<float>& src,
        const LocalTensor<float>& dst, uint32_t count)
    {
        set_mask_count();
        set_vector_mask(0, count);
        vcadd(reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()), 1, 1, 1, 8, 1);
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        int64_t accumulator = get_acc_val();
        const float result = *reinterpret_cast<float*>(&accumulator);
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        return result;
    }
    GlobalTensor<T> selfGm_;
    GlobalTensor<T> gammaGm_;
    GlobalTensor<T> betaGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<int8_t> outGm_;
    GlobalTensor<T> meanGm_;
    GlobalTensor<T> rstdGm_;
    const GroupNormSiluQuantTilingData* tiling_ = nullptr;
};

// FP32 resident execution with the reference reduction association. Groups
// share vector instructions only while fitting the fixed UB capacity. Input
// prefetch, parameter wrapping, odd group tails and non-power-of-two reductions
// all follow the runtime metadata; no complete input shape is used as a key.
template <int32_t GroupElements = 0, bool PadLeaves = false>
class ExactResidentFp32GroupNormSiluQuant {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR gamma, GM_ADDR beta,
        GM_ADDR quantScale, GM_ADDR out, GM_ADDR meanOut, GM_ADDR rstdOut,
        const GroupNormSiluQuantTilingData* tilingData)
    {
        tiling_ = tilingData;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(self));
        gammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gamma));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(quantScale));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out));
        meanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(meanOut));
        rstdGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(rstdOut));
    }

    // The compiler inserts same-pipeline vector dependencies for fixed short
    // groups. Keep the explicit barriers for larger and dynamic groups to
    // preserve their measured instruction schedule.
    __aicore__ inline void Process()
    {
        const int64_t worker = GetBlockIdx();
        if (tiling_->workerNum == tiling_->totalGroups) {
            ProcessGroup<1>(worker, -1, 0, false);
            WaitFlag<HardEvent::MTE3_S>(2);
            return;
        }
        // One full resident batch per worker needs neither a loop nor
        // next-batch prefetch. This is a task-count/resource condition.
        if ((GroupElements == 0 ? tiling_->groupElements : GroupElements) <= 4096 &&
            tiling_->totalGroups == 2 * tiling_->workerNum) {
            ProcessGroup<2>(worker * 2, -1, 0, false);
            WaitFlag<HardEvent::MTE3_S>(2);
            return;
        }
        const int64_t base = tiling_->totalGroups / tiling_->workerNum;
        const int64_t extra = tiling_->totalGroups % tiling_->workerNum;
        const int64_t start = worker * base + (worker < extra ? worker : extra);
        const int64_t groups = base + (worker < extra ? 1 : 0);
        const int32_t capacity = (GroupElements == 0 ? tiling_->groupElements : GroupElements) <= 4096 ? 2 : 1;
        for (int64_t i = 0; i < groups; i += capacity) {
            const int64_t next = i + capacity < groups ? start + i + capacity : -1;
            const int32_t nextBatch = next < 0 ? 0 : (groups - i - capacity < capacity ? 1 : capacity);
            if (capacity == 2 && i + 1 < groups) ProcessGroup<2>(start + i, next, nextBatch, i != 0);
            else ProcessGroup<1>(start + i, next, nextBatch, i != 0);
        }
        WaitFlag<HardEvent::MTE3_S>(2);
    }

    template <int32_t batchGroups>
    __aicore__ inline void ProcessGroup(int64_t taskIndex, int64_t nextTask, int32_t nextBatch, bool prefetched)
    {
        const int64_t groupIndex = taskIndex % tiling_->numGroups;
        const int64_t groupOffset = taskIndex * tiling_->groupElements;
        const int64_t channelStart = groupIndex * tiling_->channelsPerGroup;
        const int32_t spatial = static_cast<int32_t>(tiling_->spatial);
        const int32_t groupChannels = static_cast<int32_t>(tiling_->channelsPerGroup);
        const int32_t channels = groupChannels * batchGroups;
        const int32_t rowStride = PadLeaves ? spatial / 128 * 136 : (spatial + 31) / 32 * 32;
        const int32_t groupCount = PadLeaves ? GroupElements / 128 * 136 :
            (GroupElements == 0 ? rowStride * groupChannels : GroupElements);
        const int32_t count = groupCount * batchGroups;
        const int32_t validCount = (GroupElements == 0 ?
            static_cast<int32_t>(tiling_->groupElements) : GroupElements) * batchGroups;
        const bool contiguousParams = groupIndex + batchGroups <= tiling_->numGroups;
        const int32_t paramStride = contiguousParams ? groupChannels : (groupChannels + 7) / 8 * 8;
        // A 17-block input leaf stride breaks the
        // repeated bank address of 16-block leaves. The padded layout ends
        // at 186240 bytes, with the host reserving a further 8192 bytes.
        constexpr uint32_t bufferStride = PadLeaves ? 35072U : 33024U;
        constexpr uint32_t bufferElements = PadLeaves ? 8704U : 8192U;
        constexpr uint32_t parameterBase = bufferStride * 5 + 8192U;
        constexpr uint32_t maskBytes = PadLeaves ? 1280U : 1024U;
        constexpr uint32_t parameterElements = PadLeaves ? 64U : 256U;
        constexpr uint32_t parameterStride = parameterElements * sizeof(float);
        LocalTensor<float> input(TPosition::VECCALC, 0U, bufferElements);
        LocalTensor<float> value(TPosition::VECCALC, bufferStride, bufferElements);
        LocalTensor<float> aux(TPosition::VECCALC, bufferStride * 2, bufferElements);
        LocalTensor<float> quotient(TPosition::VECCALC, bufferStride * 3, bufferElements);
        LocalTensor<float> rounded(TPosition::VECCALC, bufferStride * 4, bufferElements);
        LocalTensor<int8_t> output(TPosition::VECCALC, bufferStride * 5, 8192U);
        LocalTensor<float> gammaLocal(TPosition::VECCALC, parameterBase, parameterElements);
        LocalTensor<float> betaLocal(TPosition::VECCALC, parameterBase + parameterStride, parameterElements);
        LocalTensor<float> scaleLocal(TPosition::VECCALC, parameterBase + 2 * parameterStride, parameterElements);
        LocalTensor<float> reduction(TPosition::VECCALC, parameterBase + 3 * parameterStride, 64U);
        LocalTensor<uint8_t> mask(TPosition::VECCALC, parameterBase + 3 * parameterStride + 256U, maskBytes);
        LocalTensor<float> stat(TPosition::VECCALC, parameterBase + 3 * parameterStride + 256U + maskBytes, 32U);
        LocalTensor<uint8_t> directory(TPosition::VECCALC, parameterBase + 3 * parameterStride + 384U + maskBytes, 256U);
        const DataCopyPadExtParams<float> noFloatPad{false, 0, 0, 0};
        const DataCopyExtParams inCopy{
            static_cast<uint16_t>(PadLeaves ? validCount / 128 : 1),
            static_cast<uint32_t>(PadLeaves ? 512 : validCount * sizeof(float)),
            0, static_cast<uint32_t>(PadLeaves ? 1 : 0), 0};
        const DataCopyExtParams paramCopy{1,
            static_cast<uint32_t>(groupChannels * sizeof(float)), 0, 0, 0};
        const int32_t scaleCount = tiling_->quantScaleCount == 1 ? 1 : channels;
        const DataCopyExtParams scaleCopy{1,
            static_cast<uint32_t>(scaleCount * sizeof(float)), 0, 0, 0};
        if (!prefetched) {
            DataCopyPad(input, selfGm_[groupOffset], inCopy, noFloatPad);
            SetFlag<HardEvent::MTE2_V>(0);
        }
        if (contiguousParams) {
            const DataCopyExtParams allParamCopy{1, static_cast<uint32_t>(channels * sizeof(float)), 0, 0, 0};
            DataCopyPad(gammaLocal, gammaGm_[channelStart], allParamCopy, noFloatPad);
            DataCopyPad(betaLocal, betaGm_[channelStart], allParamCopy, noFloatPad);
            if (tiling_->quantScaleCount != 1) {
                DataCopyPad(scaleLocal, scaleGm_[channelStart], allParamCopy, noFloatPad);
            }
        } else {
            for (int32_t g = 0; g < batchGroups; ++g) {
                const int64_t paramStart = ((taskIndex + g) % tiling_->numGroups) * groupChannels;
                DataCopyPad(gammaLocal[g * paramStride], gammaGm_[paramStart], paramCopy, noFloatPad);
                DataCopyPad(betaLocal[g * paramStride], betaGm_[paramStart], paramCopy, noFloatPad);
                if (tiling_->quantScaleCount != 1) {
                    const DataCopyExtParams groupScaleCopy{1,
                        static_cast<uint32_t>(groupChannels * sizeof(float)), 0, 0, 0};
                    DataCopyPad(scaleLocal[g * paramStride], scaleGm_[paramStart], groupScaleCopy, noFloatPad);
                }
            }
        }
        if (tiling_->quantScaleCount == 1) {
            DataCopyPad(scaleLocal, scaleGm_, scaleCopy, noFloatPad);
        }
        SetFlag<HardEvent::MTE2_S>(1);
        WaitFlag<HardEvent::MTE2_V>(0);
        if constexpr (PadLeaves) {
            Duplicate<float>(input[128], 0.0f, 8,
                static_cast<uint8_t>(validCount / 128), 1, 17);
            VectorBarrier();
        }

        float means[2];
        float rstds[2];
        NumpySums<batchGroups>(input, aux, rounded, reduction, groupCount, means);
        for (int32_t g = 0; g < batchGroups; ++g) {
            means[g] /= static_cast<float>(tiling_->groupElements);
            Adds(value[g * groupCount], input[g * groupCount], -means[g], groupCount);
        }
        VectorBarrier();
        if (nextTask >= 0) {
            // The centered values now own every remaining arithmetic input.
            // Transfer the next batch while the current variance and output run.
            SetFlag<HardEvent::V_MTE2>(2);
            WaitFlag<HardEvent::V_MTE2>(2);
            const DataCopyExtParams nextCopy{
                static_cast<uint16_t>(PadLeaves ? nextBatch * tiling_->groupElements / 128 : 1),
                static_cast<uint32_t>(PadLeaves ? 512 : nextBatch * tiling_->groupElements * sizeof(float)),
                0, static_cast<uint32_t>(PadLeaves ? 1 : 0), 0};
            DataCopyPad(input, selfGm_[nextTask * tiling_->groupElements], nextCopy, noFloatPad);
            SetFlag<HardEvent::MTE2_V>(0);
        }
        // Keep the centered values for normalization while the second buffer
        // holds the squared deviations used by the variance tree.
        Mul(quotient, value, value, count);
        VectorBarrier();
        NumpySums<batchGroups>(quotient, aux, rounded, reduction, groupCount, rstds);
        for (int32_t g = 0; g < batchGroups; ++g) {
            const float variance = rstds[g] / static_cast<float>(tiling_->groupElements);
            rstds[g] = 1.0f / sqrt(variance + tiling_->epsilon);
        }
        if (prefetched) WaitFlag<HardEvent::MTE3_S>(2);
        for (int32_t g = 0; g < batchGroups; ++g) {
            stat.SetValue(g, static_cast<float>(means[g]));
            stat.SetValue(16 + g, static_cast<float>(rstds[g]));
        }
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        const DataCopyExtParams scalarCopy{1, static_cast<uint32_t>(sizeof(float) * batchGroups), 0, 0, 0};
        DataCopyPad(meanGm_[taskIndex], stat, scalarCopy);
        DataCopyPad(rstdGm_[taskIndex], stat[16], scalarCopy);
        const UnaryRepeatParams counterUnary{1, 1, 8, 8};
        const BinaryRepeatParams counterBinary{1, 1, 1, 8, 8, 8};
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(groupCount);
        for (int32_t g = 0; g < batchGroups; ++g) {
            Muls<float, false>(value[g * groupCount], value[g * groupCount], rstds[g],
                MASK_PLACEHOLDER, 1, counterUnary);
        }
        VectorBarrier();
        WaitFlag<HardEvent::MTE2_S>(1);
        SetVectorMask<float, MaskMode::COUNTER>(rowStride);
        for (int32_t channel = 0; channel < channels; ++channel) {
            Muls<float, false>(value[channel * rowStride], value[channel * rowStride],
                static_cast<float>(gammaLocal.GetValue(channel / groupChannels * paramStride + channel % groupChannels)),
                MASK_PLACEHOLDER, 1, counterUnary);
        }
        VectorBarrier();
        for (int32_t channel = 0; channel < channels; ++channel) {
            Adds<float, false>(value[channel * rowStride], value[channel * rowStride],
                static_cast<float>(betaLocal.GetValue(channel / groupChannels * paramStride + channel % groupChannels)),
                MASK_PLACEHOLDER, 1, counterUnary);
        }
        VectorBarrier();
        // Widen the affine row mask to the complete resident batch.
        // Normal mode is restored after the final boundary-distance Abs.
        SetVectorMask<float, MaskMode::COUNTER>(count);
        if (tiling_->activateSilu != 0) {
            Muls<float, false>(aux, value, -1.0f, MASK_PLACEHOLDER, 1, counterUnary);
            VectorBarrier();
            Exp<float, false>(aux, aux, MASK_PLACEHOLDER, 1, counterUnary);
            VectorBarrier();
            Adds<float, false>(aux, aux, 1.0f, MASK_PLACEHOLDER, 1, counterUnary);
            VectorBarrier();
            Div<float, false>(quotient, value, aux, MASK_PLACEHOLDER, 1, counterBinary);
        } else {
            Adds<float, false>(quotient, value, 0.0f, MASK_PLACEHOLDER, 1, counterUnary);
        }
        VectorBarrier();
        if (scaleCount == 1) {
            Muls<float, false>(quotient, quotient, 1.0f / scaleLocal.GetValue(0), MASK_PLACEHOLDER, 1, counterUnary);
        } else {
            SetVectorMask<float, MaskMode::COUNTER>(rowStride);
            for (int32_t channel = 0; channel < channels; ++channel) {
                Muls<float, false>(quotient[channel * rowStride], quotient[channel * rowStride],
                    1.0f / scaleLocal.GetValue(channel / groupChannels * paramStride + channel % groupChannels),
                    MASK_PLACEHOLDER, 1, counterUnary);
            }
            SetVectorMask<float, MaskMode::COUNTER>(count);
        }
        VectorBarrier();
        Cast<float, float, false>(rounded, quotient, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, counterUnary);
        VectorBarrier();
        Sub<float, false>(quotient, quotient, rounded, MASK_PLACEHOLDER, 1, counterBinary);
        VectorBarrier();
        Abs<float, false>(quotient, quotient, MASK_PLACEHOLDER, 1, counterUnary);
        VectorBarrier();
        SetMaskNorm();
        ResetMask();
        LocalTensor<int32_t> maskDwords = mask.ReinterpretCast<int32_t>();
        const int32_t dwordCount = ((PadLeaves ? validCount : count) + 31) / 32;
        const int32_t directoryCount = (dwordCount + 63) / 64 * 64;
        if constexpr (PadLeaves && (GroupElements * batchGroups) % 2048 != 0) {
            // Each comparison writes all bits of its packed plane. Initialize
            // only when the later 64-dword directory comparison has a tail.
            Duplicate(maskDwords, static_cast<int32_t>(0), directoryCount);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (PadLeaves) {
            // Compare the two valid 64-element halves of each padded leaf
            // separately. Each plane is a packed bitset; padding is not read.
            const UnaryRepeatParams compareStrides{1, 1, 8, 17};
            const uint8_t leafRepeats = static_cast<uint8_t>(validCount / 128);
            CompareScalar(mask, quotient, 0.5f - MULTI_TILE_QUANT_HALF_BOUNDARY_GUARD,
                CMPMODE::GE, 64, leafRepeats, compareStrides);
            CompareScalar(mask[validCount / 16], quotient[64],
                0.5f - MULTI_TILE_QUANT_HALF_BOUNDARY_GUARD,
                CMPMODE::GE, 64, leafRepeats, compareStrides);
        } else {
        CompareScalar(mask, quotient, 0.5f - MULTI_TILE_QUANT_HALF_BOUNDARY_GUARD,
            CMPMODE::GE, (count + 63) / 64 * 64);
        }
        VectorBarrier();
        CompareScalar(directory, maskDwords, static_cast<int32_t>(0), CMPMODE::EQ,
            directoryCount);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
        LocalTensor<uint64_t> directoryWords = directory.ReinterpretCast<uint64_t>();
        for (int32_t word = 0; word < (dwordCount + 63) / 64; ++word) {
            uint64_t dwords = ~directoryWords.GetValue(word);
            if (word == (dwordCount - 1) / 64 && (dwordCount & 63) != 0) {
                dwords &= (uint64_t(1) << (dwordCount & 63)) - 1;
            }
            while (dwords != 0) {
                const int32_t dwordBit = static_cast<int32_t>(GetSFFValue<1>(dwords));
                const int32_t dwordIndex = word * 64 + dwordBit;
                uint32_t bits = static_cast<uint32_t>(maskDwords.GetValue(dwordIndex));
                while (bits != 0) {
                    const int32_t bit = static_cast<int32_t>(GetSFFValue<1>(static_cast<uint64_t>(bits)));
                    int32_t index = dwordIndex * 32 + bit;
                    if constexpr (PadLeaves) {
                        const int32_t halfElements = validCount / 2;
                        const bool upperHalf = index >= halfElements;
                        const int32_t planeIndex = upperHalf ? index - halfElements : index;
                        index = (planeIndex / 64) * 136 + planeIndex % 64 + (upperHalf ? 64 : 0);
                    }
                    float exactValue = value.GetValue(index);
                    if (tiling_->activateSilu != 0) {
                        // Resolve near-half quantization with an independent
                        // scalar exponential using fixed-point range reduction.
                        const float exponent = -exactValue;
                        const float denominator = exponent > -80.0f && exponent < 80.0f ?
                            1.0f + ScalarExpQ56(exponent) : aux.GetValue(index);
                        exactValue /= denominator;
                    }
                    const float scalarQuotient = exactValue /
                        scaleLocal.GetValue(scaleCount == 1 ? 0 : index / groupCount * paramStride + (index / rowStride) % groupChannels);
                    const int32_t roundedValue = scalarQuotient >= 127.0f ? 127 :
                        (scalarQuotient <= -128.0f ? -128 :
                        ScalarCast<float, int32_t, RoundMode::CAST_RINT>(scalarQuotient));
                    rounded.SetValue(index, static_cast<float>(roundedValue));
                    bits &= ~(uint32_t(1) << bit);
                }
                dwords &= ~(uint64_t(1) << dwordBit);
            }
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        LocalTensor<half> roundedHalf = quotient.ReinterpretCast<half>();
        if constexpr (PadLeaves) {
            const UnaryRepeatParams compactCast{1, 1, 8, 17};
            const uint8_t leafRepeats = static_cast<uint8_t>(validCount / 128);
            Cast(roundedHalf, rounded, RoundMode::CAST_NONE, 64, leafRepeats, compactCast);
            Cast(roundedHalf[64], rounded[64], RoundMode::CAST_NONE, 64, leafRepeats, compactCast);
        } else {
            Cast(roundedHalf, rounded, RoundMode::CAST_NONE, count);
        }
        VectorBarrier();
        Cast(output, roundedHalf, RoundMode::CAST_NONE, validCount);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const DataCopyExtParams outCopy{1, static_cast<uint32_t>(validCount), 0, 0, 0};
        DataCopyPad(outGm_[groupOffset], output, outCopy);
        SetFlag<HardEvent::MTE3_S>(2);
    }

private:
    __aicore__ inline int64_t MultiplyQ56(int64_t a, int64_t b)
    {
        constexpr int64_t mask = (int64_t(1) << 28) - 1;
        const int64_t ah = a >> 28;
        const int64_t al = a & mask;
        const int64_t bh = b >> 28;
        const int64_t bl = b & mask;
        const int64_t middle = ah * bl + al * bh + ((al * bl) >> 28);
        return ah * bh + ((middle + (int64_t(1) << 27)) >> 28);
    }

    __aicore__ inline float ScalarExpQ56(float input)
    {
        // Range reduction and a degree-12 Taylor polynomial in Q56 arithmetic.
        // The caller limits |input| to 80, keeping all products in int64 range.
        const uint32_t inputBits = *reinterpret_cast<const uint32_t*>(&input);
        const uint32_t absoluteBits = inputBits & 0x7fffffffU;
        if (absoluteBits < 0x32800000U) return 1.0f;
        const int32_t exponent = static_cast<int32_t>((absoluteBits >> 23) & 255U) - 127;
        const int64_t mantissa = static_cast<int64_t>((absoluteBits & 0x7fffffU) | 0x800000U);
        const int32_t shift = exponent + 33;
        int64_t x = shift >= 0 ? mantissa << shift : mantissa >> (-shift);
        if ((inputBits >> 31) != 0) x = -x;
        const int32_t power = static_cast<int32_t>(input * 1.4426950408889634f +
            (input >= 0.0f ? 0.5f : -0.5f));
        const int64_t r = x - static_cast<int64_t>(power) * 49946518145322874LL;
        int64_t p = 150432888LL;
        p = 1805194656LL + MultiplyQ56(p, r);
        p = 19857141214LL + MultiplyQ56(p, r);
        p = 198571412142LL + MultiplyQ56(p, r);
        p = 1787142709274LL + MultiplyQ56(p, r);
        p = 14297141674192LL + MultiplyQ56(p, r);
        p = 100079991719344LL + MultiplyQ56(p, r);
        p = 600479950316066LL + MultiplyQ56(p, r);
        p = 3002399751580331LL + MultiplyQ56(p, r);
        p = 12009599006321323LL + MultiplyQ56(p, r);
        p = 36028797018963968LL + MultiplyQ56(p, r);
        p = 72057594037927936LL + MultiplyQ56(p, r);
        p = 72057594037927936LL + MultiplyQ56(p, r);
        const bool belowOne = p < (int64_t(1) << 56);
        const int32_t resultShift = belowOne ? 32 : 33;
        int64_t significand = p >> resultShift;
        const int64_t remainder = p & ((int64_t(1) << resultShift) - 1);
        const int64_t halfway = int64_t(1) << (resultShift - 1);
        if (remainder > halfway || (remainder == halfway && (significand & 1) != 0)) ++significand;
        int32_t resultExponent = power + 127 - (belowOne ? 1 : 0);
        if (significand >= (int64_t(1) << 24)) {
            significand >>= 1;
            ++resultExponent;
        }
        const uint32_t resultBits = (static_cast<uint32_t>(resultExponent) << 23) |
            (static_cast<uint32_t>(significand) & 0x7fffffU);
        return *reinterpret_cast<const float*>(&resultBits);
    }

    __aicore__ inline void VectorBarrier()
    {
        if constexpr (GroupElements == 0 || GroupElements > 2048) {
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline float ScalarPairwiseSum(const LocalTensor<float>& source, int32_t count)
    {
        // Iterative form of the adjacent NumPy tree, including non-power-of-two tails.
        int32_t offsets[16], sizes[16], phases[16];
        float lefts[16];
        int32_t top = 0;
        offsets[0] = 0; sizes[0] = count; phases[0] = 0;
        float result = 0.0f;
        while (top >= 0) {
            const int32_t n = sizes[top], offset = offsets[top];
            if (n <= 128) {
                if (n < 8) {
                    result = -0.0f;
                    for (int32_t j = 0; j < n; ++j) result += source.GetValue(offset + j);
                } else {
                    float a[8];
                    for (int32_t j = 0; j < 8; ++j) a[j] = source.GetValue(offset + j);
                    int32_t j = 8;
                    for (; j <= n - 8; j += 8) {
                        for (int32_t k = 0; k < 8; ++k) a[k] += source.GetValue(offset + j + k);
                    }
                    result = ((a[0] + a[1]) + (a[2] + a[3])) +
                             ((a[4] + a[5]) + (a[6] + a[7]));
                    for (; j < n; ++j) result += source.GetValue(offset + j);
                }
                --top;
            } else if (phases[top] == 0) {
                const int32_t split = (n / 2) / 8 * 8;
                phases[top] = 1;
                ++top; offsets[top] = offset; sizes[top] = split; phases[top] = 0;
            } else if (phases[top] == 1) {
                const int32_t split = (n / 2) / 8 * 8;
                lefts[top] = result; phases[top] = 2;
                ++top; offsets[top] = offset + split; sizes[top] = n - split; phases[top] = 0;
            } else {
                result = lefts[top] + result;
                --top;
            }
        }
        return result;
    }

    template <int32_t batchGroups>
    __aicore__ inline void NumpySums(const LocalTensor<float>& source,
        const LocalTensor<float>& accumulators, const LocalTensor<float>& pairScratch,
        const LocalTensor<float>& partial, int32_t dynamicCount, float* sums)
    {
        const int32_t groupCount = GroupElements == 0 ? dynamicCount : GroupElements;
        if ((groupCount & (groupCount - 1)) != 0 || groupCount < 128) {
            SetFlag<HardEvent::V_S>(1);
            WaitFlag<HardEvent::V_S>(1);
            for (int32_t g = 0; g < batchGroups; ++g) sums[g] = ScalarPairwiseSum(source[g * groupCount], groupCount);
            return;
        }
        const int32_t leaves = groupCount / 128;
        const int32_t totalLeaves = leaves * batchGroups;
        constexpr uint8_t leafBatch = PadLeaves ? 8 : 2;
        constexpr uint8_t leafBlocks = PadLeaves ? 17 : 16;
        const uint64_t addMask = leafBatch * 8;
        const uint8_t repeats = static_cast<uint8_t>(totalLeaves / leafBatch);
        const BinaryRepeatParams firstStrides{1, leafBlocks, leafBlocks,
            leafBatch, leafBatch * leafBlocks, leafBatch * leafBlocks};
        const BinaryRepeatParams strides{1, 1, leafBlocks,
            leafBatch, leafBatch, leafBatch * leafBlocks};
        Add<float, true>(accumulators, source, source[8], addMask, repeats, firstStrides);
        #pragma unroll
        for (int32_t offset = 16; offset < 128; offset += 8) {
            VectorBarrier();
            Add<float, false>(accumulators, accumulators, source[offset], addMask, repeats, strides);
        }
        VectorBarrier();
        if constexpr (PadLeaves) {
            // Each packed block contains the eight accumulators of one leaf.
            // Reduce eight independent adjacent-pair trees per instruction
            // repeat, retaining the same leaf order in the packed roots.
            BlockReduceSum<float, true>(partial, accumulators,
                totalLeaves / 8, 64, 1, 1, 8);
        } else {
            WholeReduceSum<float, true>(partial, accumulators, 8, totalLeaves, 1, 1, 1);
        }
        if (leaves >= 8) {
            VectorBarrier();
            if (leaves == 8) {
                // One group's eight leaf roots occupy one 32B block.
                // A block reduction handles all resident groups together
                // with the same adjacent-pair association inside each group.
                BlockReduceSum<float, true>(pairScratch, partial,
                    1, leaves * batchGroups, 1, 1, 8);
            } else {
                WholeReduceSum<float, true>(pairScratch, partial, leaves, batchGroups, 1, 1, leaves / 8);
            }
            SetFlag<HardEvent::V_S>(1);
            WaitFlag<HardEvent::V_S>(1);
            for (int32_t g = 0; g < batchGroups; ++g) sums[g] = pairScratch.GetValue(g);
            return;
        }
        bool inPartial = true;
        #pragma unroll
        for (int32_t width = totalLeaves; width > batchGroups; width /= 2) {
            VectorBarrier();
            if (inPartial) PairReduceSum<float, true>(pairScratch, partial, 1, width, 1, 1, 8);
            else PairReduceSum<float, true>(partial, pairScratch, 1, width, 1, 1, 8);
            inPartial = !inPartial;
        }
        SetFlag<HardEvent::V_S>(1);
        WaitFlag<HardEvent::V_S>(1);
        for (int32_t g = 0; g < batchGroups; ++g) sums[g] = inPartial ? partial.GetValue(g) : pairScratch.GetValue(g);
    }


    GlobalTensor<float> selfGm_;
    GlobalTensor<float> gammaGm_;
    GlobalTensor<float> betaGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<int8_t> outGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;
    const GroupNormSiluQuantTilingData* tiling_ = nullptr;
};

} // namespace NsGroupNormSiluQuant
#endif // GROUPNORMSILUQUANT_H
