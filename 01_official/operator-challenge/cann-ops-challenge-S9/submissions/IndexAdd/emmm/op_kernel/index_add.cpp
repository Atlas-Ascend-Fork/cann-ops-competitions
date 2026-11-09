#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kTileBytes = 16 * 1024;
constexpr uint32_t kSmallInnerFourWayLimit = 256;
constexpr uint32_t kBalancedWideInnerFourWayLimit = 1024;
constexpr uint32_t kBalancedSmallInnerEightWayLimit = 128;
constexpr uint32_t kBalancedMediumInnerEightWayLimit = 256;
constexpr uint32_t kDeepSmallInnerMinGroupSize = 128;
constexpr uint32_t kDeepSmallInnerMaxGroupSize = 1024;
constexpr uint32_t kDeepSmallInnerStride = 32;
constexpr uint32_t kWholeAxisOuterBatch = 32;
constexpr uint32_t kWholeAxisMaxIndexLength = 2040;
constexpr uint32_t kWholeAxisAccumulatorBytes = 18 * 1024;
constexpr uint32_t kChunkedFloatMaxIndexLength = 8000;
constexpr uint32_t kWholeInt32RowMaxElements = 8192;
constexpr uint32_t kWholeInt32RowMaxIndices = 8000;
constexpr uint32_t kFlatWholeInt32MaxRows = 7;
constexpr uint32_t kFlatWholeInt32MaxElements = 4096;
constexpr uint32_t kAtomicInt32SourceChunkRows = 5;
constexpr uint32_t kAtomicInt32MaxInnerLength = 4096;
constexpr uint32_t kAtomicInt32HeadScratchBytes = 64;
constexpr uint32_t kAtomicInt32OffsetBytes = 32;
constexpr uint32_t kAtomicInt32PhaseBytes = 4U * sizeof(int32_t);
constexpr uint32_t kColumnOwnedInt32SourceBatchRows = 8;
constexpr uint32_t kCoalescedAtomicInt32SourceChunkRows = 8;
constexpr uint32_t kWideInt32AxisGroupRows = 8;
constexpr uint32_t kWideInt32MaxInnerLength = 5136;
constexpr uint32_t kWideInt32MaxIndexLength = 1216;
constexpr uint32_t kWideInt32InnerTileMaxAxisLength = 320;
constexpr uint32_t kWideInt32InnerTileElements = 129;
constexpr uint32_t kWideInt32InnerTileStride = 136;
constexpr uint32_t kWideInt32InnerTileSourceChunk = 32;
constexpr uint32_t kBfloatAxisGroupRows = 128;
constexpr uint32_t kBfloatGroupMaxInnerLength = 128;
constexpr uint32_t kBfloatGroupMaxIndexLength = 512;
constexpr uint32_t kBfloatGroupSourceChunkRows = 120;
constexpr uint32_t kLargeFloatMaxAxisLength = 655;
constexpr uint32_t kLargeFloatOuterBatch = 64;
constexpr uint32_t kLargeFloatSourceChunk = 96;
constexpr uint32_t kSmallInt8AxisGroupRows = 4;
constexpr uint32_t kSmallInt8MaxInnerLength = 128;
constexpr uint32_t kSmallInt8MaxIndexLength = 128;
constexpr uint32_t kVnchwFloatMaxAxisLength = 655;
constexpr uint32_t kVnchwFloatAlignedAxisLength = 656;
constexpr uint32_t kVnchwFloatOuterBatch = 32;
constexpr uint32_t kVnchwFloatColumnChunk = 368;
// Keep the row-major and column-major staging matrices adjacent. Index-cache
// double buffering is evaluated independently of any UB bank-phase padding.
constexpr uint32_t kVnchwFloatMatrixGapElements = 0;
constexpr uint32_t kVnchwFloatWideColumnChunk =
    2U * kVnchwFloatColumnChunk;
constexpr uint32_t kVnchwFloatOuterGroups =
    kVnchwFloatOuterBatch / 16U;
constexpr uint32_t kVnchwFloatVaWordsPerGroup = 4U;
constexpr uint32_t kVnchwFloatVaConfigWords =
    kVnchwFloatOuterGroups * kVnchwFloatVaWordsPerGroup;
constexpr uint32_t kVnchwFloatMaxAxisChunks =
    (kVnchwFloatMaxAxisLength + kVnchwFloatColumnChunk - 1U) /
    kVnchwFloatColumnChunk;
constexpr uint32_t kVnchwFloatTotalVaConfigWords =
    (4U + kVnchwFloatMaxAxisChunks) * kVnchwFloatVaConfigWords;
// The source transpose keeps only its destination/source VA tables live after
// the self tensor has been materialized.  Reuse the remaining VA-table UB for
// an even-sized index window so the hot accumulation loop reads packed index
// pairs from UB instead of issuing thousands of scalar GM loads.
constexpr uint32_t kVnchwFloatIndexCacheElements =
    5U * kVnchwFloatColumnChunk;
constexpr uint32_t kVnchwFloatPersistentVaBytes =
    2U * kVnchwFloatVaConfigWords * sizeof(uint64_t);
constexpr uint32_t kVnchwFloatConfigBytes =
    kVnchwFloatTotalVaConfigWords * sizeof(uint64_t);
constexpr uint32_t kVnchwFloatIndexCacheBytes =
    kVnchwFloatPersistentVaBytes +
    2U * kVnchwFloatIndexCacheElements * sizeof(int32_t);
constexpr uint32_t kVnchwFloatVaStorageBytes =
    kVnchwFloatConfigBytes > kVnchwFloatIndexCacheBytes
        ? kVnchwFloatConfigBytes
        : kVnchwFloatIndexCacheBytes;
static_assert(
    kVnchwFloatPersistentVaBytes +
            2U * kVnchwFloatIndexCacheElements * sizeof(int32_t) <=
        kVnchwFloatVaStorageBytes,
    "float VNCHW double index cache must fit VA-table storage");
constexpr uint32_t kVnchwFloatUbBytes =
    kVnchwFloatAlignedAxisLength * kVnchwFloatOuterBatch *
        sizeof(float) +
    (2U * kVnchwFloatOuterBatch * kVnchwFloatColumnChunk +
     kVnchwFloatMatrixGapElements) * sizeof(float) +
    kVnchwFloatVaStorageBytes;
static_assert(kVnchwFloatUbBytes <= 192U * 1024U,
              "float VNCHW buffers must fit the C220 UB");
constexpr uint32_t kVnchwHalfMaxAxisLength = 224;
constexpr uint32_t kVnchwHalfOuterBatch = 192;
constexpr uint32_t kVnchwHalfColumnChunk = 128;
constexpr uint32_t kVnchwHalfMaxIndexLength = 2040;
constexpr uint32_t kVnchwHalfOuterGroups =
    kVnchwHalfOuterBatch / 16U;
constexpr uint32_t kVnchwHalfVaWordsPerGroup = 4U;
constexpr uint32_t kVnchwHalfVaConfigWords =
    kVnchwHalfOuterGroups * kVnchwHalfVaWordsPerGroup;
constexpr uint64_t kVnchwVaAddressMask = 0x1fffULL;
template <typename T>
struct IndexAddAccumulator {
    using Type = T;
};

template <>
struct IndexAddAccumulator<bfloat16_t> {
    using Type = float;
};

template <>
struct IndexAddAccumulator<int8_t> {
    using Type = int16_t;
};

template <typename T, bool UsePackedSmallIndices = false, bool UseBalancedWideInner = false,
          bool UseBalancedSmallEightWay = false, bool UseAxisTailSmallEightWay = false,
          bool UseBalancedMediumEightWay = false, bool UseAxisTailMediumEightWay = false,
          bool UseDeepSmallOneTwentyEightWay = false, bool CacheDeepIndices = false,
          bool UseDeepScalarFloat = false, bool UseWholeAxisOuterBatch = false,
          uint32_t ChunkedFloatOuterBatch = 0>
class IndexAddVectorBaseline {
    using AccT = typename IndexAddAccumulator<T>::Type;
    static constexpr uint32_t kTileElements = kTileBytes / sizeof(T);
    static constexpr uint32_t kWorkingTileElements =
        UseDeepSmallOneTwentyEightWay ? kDeepSmallInnerStride : kTileElements;
    static constexpr bool kUseDeepScalarFloat =
        UseDeepScalarFloat && IsSameType<T, float>::value;
    static constexpr bool kUseWholeAxisOuterBatch =
        UseWholeAxisOuterBatch && IsSameType<T, half>::value;
    static constexpr bool kUseChunkedOuterBatchFloat =
        ChunkedFloatOuterBatch != 0 && IsSameType<T, AccT>::value;
    static constexpr uint32_t kChunkedFloatBatch =
        ChunkedFloatOuterBatch;
    static constexpr uint32_t kChunkedFloatMaxAxis =
        ChunkedFloatOuterBatch == 256
            ? 256
            : (ChunkedFloatOuterBatch == 32 ? 672 : 1024);
    static constexpr uint32_t kChunkedFloatMaxIndexLength =
        ChunkedFloatOuterBatch == 256 ? 2048 : 8000;
    static constexpr uint32_t kChunkedFloatChunkUpdates =
        ChunkedFloatOuterBatch == 256
            ? 96
            : (ChunkedFloatOuterBatch == 32 ? 512 : 1024);
    static constexpr uint32_t kChunkedFloatSourceBytes =
        kChunkedFloatBatch * kChunkedFloatChunkUpdates * sizeof(T);
    static constexpr uint32_t kChunkedFloatAccumulatorBytes =
        ChunkedFloatOuterBatch == 256
            ? 140 * 1024
            : (ChunkedFloatOuterBatch == 32 ? 120 * 1024 : 100 * 1024);
    static constexpr uint32_t kDeepAccumulatorStride =
        kUseDeepScalarFloat ? 1 : kDeepSmallInnerStride;
    static constexpr uint32_t kDeepSmallSourceBatchRows =
        IsSameType<T, int8_t>::value
            ? 4080
            : (IsSameType<T, half>::value
                   ? 2040
                   : (IsSameType<T, bfloat16_t>::value ? 1016 : 504));
    static constexpr uint32_t kDeepSmallSourceBufferRows =
        kDeepSmallSourceBatchRows / 2;
    static constexpr uint32_t kFourWayInnerLimit =
        UseBalancedWideInner ? kBalancedWideInnerFourWayLimit : kSmallInnerFourWayLimit;

public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ / (static_cast<uint64_t>(axisLength_) * innerLength_);
        innerTileCount_ = (innerLength_ + kTileElements - 1) / kTileElements;

        self_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), totalLength_);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index), indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output), totalLength_);

        if constexpr (kUseChunkedOuterBatchFloat) {
            pipe_.InitBuffer(stagedSourceBuffer_, kChunkedFloatSourceBytes);
            pipe_.InitBuffer(accumulatorBuffer_,
                             kChunkedFloatAccumulatorBytes);
            return;
        }
        if constexpr (kUseWholeAxisOuterBatch) {
            pipe_.InitBuffer(
                stagedSourceBuffer_,
                kWholeAxisOuterBatch * kWholeAxisMaxIndexLength * sizeof(T));
            pipe_.InitBuffer(accumulatorBuffer_, kWholeAxisAccumulatorBytes);
            return;
        }

        pipe_.InitBuffer(inputQueue_, 1, kWorkingTileElements * sizeof(T));
        pipe_.InitBuffer(sourceQueue_, 1, kWorkingTileElements * sizeof(T));
        pipe_.InitBuffer(outputQueue_, 1, kWorkingTileElements * sizeof(T));
        if constexpr (IsSameType<T, int8_t>::value) {
            if constexpr (UseDeepSmallOneTwentyEightWay) {
                pipe_.InitBuffer(accumulatorBuffer_,
                                 kDeepSmallInnerMaxGroupSize *
                                     kDeepSmallInnerStride * sizeof(AccT));
            } else {
                pipe_.InitBuffer(accumulatorBuffer_,
                                 kTileElements * sizeof(AccT));
            }
            if constexpr (UseDeepSmallOneTwentyEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_,
                                 kWorkingTileElements * sizeof(AccT));
                pipe_.InitBuffer(stagedSourceBuffer_,
                                 kDeepSmallSourceBatchRows *
                                     kDeepSmallInnerStride * sizeof(T));
            } else if constexpr (UseBalancedSmallEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_,
                                 kTileElements * sizeof(AccT));
            } else if constexpr (UseAxisTailSmallEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_,
                                 kTileElements * sizeof(AccT));
            } else if constexpr (UseBalancedMediumEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_,
                                 kTileElements * sizeof(AccT));
            } else if constexpr (UseAxisTailMediumEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_,
                                 kTileElements * sizeof(AccT));
            } else {
                pipe_.InitBuffer(secondAccumulatorBuffer_,
                                 kTileElements * sizeof(AccT));
                pipe_.InitBuffer(sourceAccumulatorBuffer_,
                                 kTileElements * sizeof(AccT));
                pipe_.InitBuffer(thirdAccumulatorBuffer_,
                                 kFourWayInnerLimit * sizeof(AccT));
                pipe_.InitBuffer(fourthAccumulatorBuffer_,
                                 kFourWayInnerLimit * sizeof(AccT));
            }
            pipe_.InitBuffer(conversionBuffer_,
                             kWorkingTileElements * sizeof(half));
        } else {
            if constexpr (UseDeepSmallOneTwentyEightWay) {
                pipe_.InitBuffer(accumulatorBuffer_,
                                 kDeepSmallInnerMaxGroupSize *
                                     kDeepSmallInnerStride * sizeof(AccT));
            } else {
                pipe_.InitBuffer(accumulatorBuffer_, kTileElements * sizeof(AccT));
            }
            if constexpr (UseDeepSmallOneTwentyEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_,
                                 kWorkingTileElements * sizeof(AccT));
                pipe_.InitBuffer(stagedSourceBuffer_,
                                 kDeepSmallSourceBatchRows *
                                     kDeepSmallInnerStride * sizeof(T));
            } else if constexpr (UseBalancedSmallEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_, kTileElements * sizeof(AccT));
            } else if constexpr (UseAxisTailSmallEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_, kTileElements * sizeof(AccT));
            } else if constexpr (UseBalancedMediumEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_, kTileElements * sizeof(AccT));
            } else if constexpr (UseAxisTailMediumEightWay) {
                pipe_.InitBuffer(sourceAccumulatorBuffer_, kTileElements * sizeof(AccT));
            } else {
                pipe_.InitBuffer(secondAccumulatorBuffer_, kTileElements * sizeof(AccT));
                pipe_.InitBuffer(sourceAccumulatorBuffer_, kTileElements * sizeof(AccT));
                pipe_.InitBuffer(thirdAccumulatorBuffer_,
                                 kFourWayInnerLimit * sizeof(AccT));
                pipe_.InitBuffer(fourthAccumulatorBuffer_,
                                 kFourWayInnerLimit * sizeof(AccT));
            }
            pipe_.InitBuffer(conversionBuffer_,
                             kWorkingTileElements * sizeof(half));
        }
    }

    __aicore__ inline void Process()
    {
        const uint64_t workCount = outerLength_ * axisLength_ * innerTileCount_;
        const uint64_t blockStride = GetBlockNum();
        if constexpr (kUseWholeAxisOuterBatch) {
            ProcessWholeAxisOuterBatch();
            return;
        }
        if constexpr (kUseChunkedOuterBatchFloat) {
            ProcessChunkedOuterBatchFloat();
            return;
        }
        if constexpr (UseDeepSmallOneTwentyEightWay) {
            ProcessDeepSmallInnerOneTwentyEightWay();
            return;
        }
        if constexpr (UseBalancedMediumEightWay) {
            ProcessMediumInnerEightWay(workCount);
            return;
        }
        if constexpr (UseAxisTailSmallEightWay || UseAxisTailMediumEightWay) {
            ProcessAxisTailEightWay();
            return;
        }
        if constexpr (UseBalancedSmallEightWay) {
            ProcessSmallInnerEightWay(workCount);
            return;
        }
        if (innerLength_ <= kFourWayInnerLimit && workCount >= 4 * blockStride) {
            ProcessSmallInnerFourWay(workCount);
            return;
        }

        LocalTensor<AccT> accumulator = accumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> secondAccumulator = secondAccumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> sourceAccumulator = sourceAccumulatorBuffer_.Get<AccT>();

        for (uint64_t firstWork = GetBlockIdx(); firstWork < workCount; firstWork += 2 * blockStride) {
            const uint64_t firstTileIndex = firstWork % innerTileCount_;
            const uint64_t firstRowIndex = firstWork / innerTileCount_;
            const uint32_t firstAxisIndex = static_cast<uint32_t>(firstRowIndex % axisLength_);
            const uint64_t firstOuterIndex = firstRowIndex / axisLength_;
            const uint64_t firstInnerOffset = firstTileIndex * kTileElements;
            const uint32_t firstLength = static_cast<uint32_t>(
                innerLength_ - firstInnerOffset < kTileElements ? innerLength_ - firstInnerOffset : kTileElements);
            const uint64_t firstOutputOffset = firstRowIndex * innerLength_ + firstInnerOffset;

            const uint64_t secondWork = firstWork + blockStride;
            const bool hasSecondWork = secondWork < workCount;
            uint32_t secondAxisIndex = 0;
            uint64_t secondOuterIndex = 0;
            uint64_t secondInnerOffset = 0;
            uint32_t secondLength = 0;
            uint64_t secondOutputOffset = 0;
            if (hasSecondWork) {
                const uint64_t secondTileIndex = secondWork % innerTileCount_;
                const uint64_t secondRowIndex = secondWork / innerTileCount_;
                secondAxisIndex = static_cast<uint32_t>(secondRowIndex % axisLength_);
                secondOuterIndex = secondRowIndex / axisLength_;
                secondInnerOffset = secondTileIndex * kTileElements;
                secondLength = static_cast<uint32_t>(innerLength_ - secondInnerOffset < kTileElements
                                                         ? innerLength_ - secondInnerOffset
                                                         : kTileElements);
                secondOutputOffset = secondRowIndex * innerLength_ + secondInnerOffset;
            }

            LoadAccumulator(accumulator, inputQueue_, self_[firstOutputOffset], firstLength);
            if (hasSecondWork) {
                LoadAccumulator(secondAccumulator, inputQueue_, self_[secondOutputOffset], secondLength);
            }
            for (uint32_t update = 0; update < indexLength_; ++update) {
                const int32_t indexValue = index_.GetValue(update);
                if (indexValue == static_cast<int32_t>(firstAxisIndex)) {
                    AccumulateSource(accumulator, sourceAccumulator, update, firstOuterIndex, firstInnerOffset,
                                     firstLength);
                }
                if (hasSecondWork && indexValue == static_cast<int32_t>(secondAxisIndex)) {
                    AccumulateSource(secondAccumulator, sourceAccumulator, update, secondOuterIndex,
                                     secondInnerOffset, secondLength);
                }
            }
            StoreAccumulator(output_[firstOutputOffset], accumulator, firstLength);
            if (hasSecondWork) {
                StoreAccumulator(output_[secondOutputOffset], secondAccumulator, secondLength);
            }
        }
    }

private:
    __aicore__ inline void ProcessChunkedOuterBatchFloat()
    {
        constexpr uint32_t maxAccumulatorElements =
            kChunkedFloatMaxAxis * kChunkedFloatBatch;
        constexpr uint32_t cachedIndexOffset =
            maxAccumulatorElements * sizeof(T) / sizeof(uint32_t);
        constexpr uint32_t selfOffsetsOffset =
            cachedIndexOffset + kChunkedFloatMaxIndexLength;
        constexpr uint32_t sourceOffsetsOffset =
            selfOffsetsOffset + kChunkedFloatBatch;
        constexpr uint32_t axisOffsetsOffset =
            sourceOffsetsOffset + kChunkedFloatBatch;
        constexpr uint32_t sourceScratchOffset =
            (axisOffsetsOffset + kChunkedFloatMaxAxis) * sizeof(uint32_t) /
            sizeof(T);

        LocalTensor<T> accumulators = accumulatorBuffer_.Get<T>();
        LocalTensor<uint32_t> storage =
            accumulators.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> cachedIndices =
            storage[cachedIndexOffset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> selfOffsets = storage[selfOffsetsOffset];
        LocalTensor<uint32_t> sourceOffsets = storage[sourceOffsetsOffset];
        LocalTensor<uint32_t> axisOffsets = storage[axisOffsetsOffset];
        LocalTensor<T> sourceScratch = accumulators[sourceScratchOffset];
        LocalTensor<T> staged = stagedSourceBuffer_.Get<T>();

        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        const TEventID mte2ToScalarEvent =
            pipe_.AllocEventID<HardEvent::MTE2_S>();
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(mte2ToScalarEvent));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(mte2ToScalarEvent));
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(mte2ToScalarEvent);

        constexpr uint32_t ioOuterBatch = 16;
        for (uint32_t lane = 0; lane < kChunkedFloatBatch; ++lane) {
            selfOffsets.SetValue(
                lane, lane * axisLength_ * static_cast<uint32_t>(sizeof(T)));
            sourceOffsets.SetValue(
                lane, lane * kChunkedFloatChunkUpdates *
                          static_cast<uint32_t>(sizeof(T)));
        }
        for (uint32_t axis = 0; axis < axisLength_; ++axis) {
            axisOffsets.SetValue(
                axis, axis * kChunkedFloatBatch *
                          static_cast<uint32_t>(sizeof(T)));
        }
        const TEventID scalarToVectorEvent =
            pipe_.AllocEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(static_cast<event_t>(scalarToVectorEvent));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(scalarToVectorEvent));
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent);

        const TEventID mte2ToVectorEvent =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID vectorToMte2Event =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID vectorToMte3Event =
            pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID mte3ToVectorEvent =
            pipe_.AllocEventID<HardEvent::MTE3_V>();
        const TEventID mte3ToMte2Event =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const uint64_t blockStride = GetBlockNum();
        uint32_t outerBatch = kChunkedFloatBatch;
        if constexpr (kChunkedFloatBatch == 256) {
            const uint64_t balancedOuterBatch =
                (outerLength_ + blockStride - 1) / blockStride;
            if (balancedOuterBatch < outerBatch) {
                outerBatch = static_cast<uint32_t>(balancedOuterBatch);
            }
        } else {
            while (outerBatch > 1 &&
                   (outerLength_ + outerBatch - 1) / outerBatch < blockStride) {
                outerBatch /= 2;
            }
        }
        const uint64_t batchCount =
            (outerLength_ + outerBatch - 1) / outerBatch;

        for (uint64_t batch = GetBlockIdx(); batch < batchCount;
             batch += blockStride) {
            const uint64_t firstOuter = batch * outerBatch;
            const uint32_t activeOuter = static_cast<uint32_t>(
                outerLength_ - firstOuter < outerBatch
                    ? outerLength_ - firstOuter
                    : outerBatch);
            const uint64_t firstSelfOffset = firstOuter * axisLength_;
            for (uint32_t firstLane = 0; firstLane < activeOuter;
                 firstLane += ioOuterBatch) {
                const uint32_t laneCount =
                    activeOuter - firstLane < ioOuterBatch
                        ? activeOuter - firstLane
                        : ioOuterBatch;
                const DataCopyExtParams selfCopy{
                    1, laneCount * axisLength_ *
                           static_cast<uint32_t>(sizeof(T)),
                    0, 0, 0};
                DataCopyPad(
                    staged,
                    self_[firstSelfOffset +
                          static_cast<uint64_t>(firstLane) * axisLength_],
                    selfCopy,
                    DataCopyPadExtParams<T>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(mte2ToVectorEvent));
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(mte2ToVectorEvent));
                for (uint32_t axis = 0; axis < axisLength_; ++axis) {
                    Gather(
                        accumulators[axis * kChunkedFloatBatch + firstLane],
                        staged, selfOffsets,
                        axis * static_cast<uint32_t>(sizeof(T)), laneCount);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(vectorToMte2Event));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(vectorToMte2Event));
            }

            for (uint32_t firstUpdate = 0; firstUpdate < indexLength_;
                 firstUpdate += kChunkedFloatChunkUpdates) {
                const uint32_t updateCount =
                    indexLength_ - firstUpdate < kChunkedFloatChunkUpdates
                        ? indexLength_ - firstUpdate
                        : kChunkedFloatChunkUpdates;
                const uint32_t alignedUpdateBytes =
                    (updateCount * static_cast<uint32_t>(sizeof(T)) + 31U) &
                    ~31U;
                const uint32_t destinationStrideBlocks =
                    (kChunkedFloatChunkUpdates *
                         static_cast<uint32_t>(sizeof(T)) -
                     alignedUpdateBytes) /
                    32U;
                const DataCopyExtParams sourceCopy{
                    static_cast<uint16_t>(activeOuter),
                    updateCount * static_cast<uint32_t>(sizeof(T)),
                    (indexLength_ - updateCount) *
                        static_cast<uint32_t>(sizeof(T)),
                    destinationStrideBlocks, 0};
                const uint64_t firstSourceOffset =
                    firstOuter * indexLength_ + firstUpdate;
                DataCopyPad(staged, source_[firstSourceOffset], sourceCopy,
                            DataCopyPadExtParams<T>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(mte2ToVectorEvent));
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(mte2ToVectorEvent));

                for (uint32_t localUpdate = 0; localUpdate < updateCount;
                     ++localUpdate) {
                    Gather(sourceScratch, staged, sourceOffsets,
                           localUpdate *
                               static_cast<uint32_t>(sizeof(T)),
                           activeOuter);
                    PipeBarrier<PIPE_V>();
                    const uint32_t axis = static_cast<uint32_t>(
                        cachedIndices.GetValue(firstUpdate + localUpdate));
                    Add(accumulators[axis * kChunkedFloatBatch],
                        accumulators[axis * kChunkedFloatBatch],
                        sourceScratch, activeOuter);
                    PipeBarrier<PIPE_V>();
                }

                const uint32_t nextUpdate =
                    firstUpdate + kChunkedFloatChunkUpdates;
                if (nextUpdate < indexLength_) {
                    SetFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(vectorToMte2Event));
                    WaitFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(vectorToMte2Event));
                }
            }

            const uint32_t outputRowBytes =
                (axisLength_ * static_cast<uint32_t>(sizeof(T)) + 31U) &
                ~31U;
            const uint32_t outputRowElements =
                outputRowBytes / static_cast<uint32_t>(sizeof(T));
            for (uint32_t firstLane = 0; firstLane < activeOuter;
                 firstLane += ioOuterBatch) {
                const uint32_t laneCount =
                    activeOuter - firstLane < ioOuterBatch
                        ? activeOuter - firstLane
                        : ioOuterBatch;
                for (uint32_t lane = 0; lane < laneCount; ++lane) {
                    Gather(staged[lane * outputRowElements], accumulators,
                           axisOffsets,
                           (firstLane + lane) *
                               static_cast<uint32_t>(sizeof(T)),
                           axisLength_);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(
                    static_cast<event_t>(vectorToMte3Event));
                WaitFlag<HardEvent::V_MTE3>(
                    static_cast<event_t>(vectorToMte3Event));
                const DataCopyExtParams outputCopy{
                    static_cast<uint16_t>(laneCount),
                    axisLength_ * static_cast<uint32_t>(sizeof(T)),
                    0, 0, 0};
                DataCopyPad(
                    output_[firstSelfOffset +
                            static_cast<uint64_t>(firstLane) * axisLength_],
                    staged, outputCopy);
                if (firstLane + laneCount < activeOuter) {
                    SetFlag<HardEvent::MTE3_V>(
                        static_cast<event_t>(mte3ToVectorEvent));
                    WaitFlag<HardEvent::MTE3_V>(
                        static_cast<event_t>(mte3ToVectorEvent));
                } else {
                    SetFlag<HardEvent::MTE3_MTE2>(
                        static_cast<event_t>(mte3ToMte2Event));
                    WaitFlag<HardEvent::MTE3_MTE2>(
                        static_cast<event_t>(mte3ToMte2Event));
                }
            }
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(mte2ToVectorEvent);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(vectorToMte2Event);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(vectorToMte3Event);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(mte3ToVectorEvent);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
    }

    __aicore__ inline void ProcessWholeAxisOuterBatch()
    {
        constexpr uint32_t maxAccumulatorElements =
            128 * kWholeAxisOuterBatch;
        constexpr uint32_t cachedIndexOffset =
            maxAccumulatorElements * sizeof(T) / sizeof(uint32_t);
        constexpr uint32_t selfOffsetsOffset =
            cachedIndexOffset + kWholeAxisMaxIndexLength;
        constexpr uint32_t sourceOffsetsOffset =
            selfOffsetsOffset + kWholeAxisOuterBatch;
        constexpr uint32_t axisOffsetsOffset =
            sourceOffsetsOffset + kWholeAxisOuterBatch;
        constexpr uint32_t sourceScratchOffset =
            (axisOffsetsOffset + 128) * sizeof(uint32_t) / sizeof(T);

        LocalTensor<T> accumulators = accumulatorBuffer_.Get<T>();
        LocalTensor<uint32_t> storage =
            accumulators.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> cachedIndices =
            storage[cachedIndexOffset].template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> selfOffsets = storage[selfOffsetsOffset];
        LocalTensor<uint32_t> sourceOffsets = storage[sourceOffsetsOffset];
        LocalTensor<uint32_t> axisOffsets = storage[axisOffsetsOffset];
        LocalTensor<T> sourceScratch = accumulators[sourceScratchOffset];
        LocalTensor<T> staged = stagedSourceBuffer_.Get<T>();

        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        const TEventID mte2ToScalarEvent =
            pipe_.AllocEventID<HardEvent::MTE2_S>();
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(mte2ToScalarEvent));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(mte2ToScalarEvent));
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(mte2ToScalarEvent);

        for (uint32_t lane = 0; lane < kWholeAxisOuterBatch; ++lane) {
            selfOffsets.SetValue(
                lane, lane * axisLength_ * static_cast<uint32_t>(sizeof(T)));
            sourceOffsets.SetValue(
                lane, lane * indexLength_ * static_cast<uint32_t>(sizeof(T)));
        }
        for (uint32_t axis = 0; axis < axisLength_; ++axis) {
            axisOffsets.SetValue(
                axis, axis * kWholeAxisOuterBatch *
                          static_cast<uint32_t>(sizeof(T)));
        }
        const TEventID scalarToVectorEvent =
            pipe_.AllocEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(static_cast<event_t>(scalarToVectorEvent));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(scalarToVectorEvent));
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVectorEvent);

        const TEventID mte2ToVectorEvent =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID vectorToMte2Event =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID vectorToMte3Event =
            pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID mte3ToMte2Event =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const uint64_t blockStride = GetBlockNum();
        uint32_t outerBatch = kWholeAxisOuterBatch;
        while (outerBatch > 1 &&
               (outerLength_ + outerBatch - 1) / outerBatch < blockStride) {
            outerBatch /= 2;
        }
        const uint64_t batchCount =
            (outerLength_ + outerBatch - 1) / outerBatch;
        LocalTensor<uint64_t> cachedIndexPairs =
            cachedIndices.template ReinterpretCast<uint64_t>();

        for (uint64_t batch = GetBlockIdx(); batch < batchCount;
             batch += blockStride) {
            const uint64_t firstOuter = batch * outerBatch;
            const uint32_t activeOuter = static_cast<uint32_t>(
                outerLength_ - firstOuter < outerBatch
                    ? outerLength_ - firstOuter
                    : outerBatch);
            const uint64_t firstSelfOffset = firstOuter * axisLength_;
            const DataCopyExtParams selfCopy{
                1, activeOuter * axisLength_ *
                       static_cast<uint32_t>(sizeof(T)),
                0, 0, 0};
            DataCopyPad(staged, self_[firstSelfOffset], selfCopy,
                        DataCopyPadExtParams<T>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(
                static_cast<event_t>(mte2ToVectorEvent));
            WaitFlag<HardEvent::MTE2_V>(
                static_cast<event_t>(mte2ToVectorEvent));
            for (uint32_t axis = 0; axis < axisLength_; ++axis) {
                Gather(accumulators[axis * kWholeAxisOuterBatch], staged,
                       selfOffsets, axis * static_cast<uint32_t>(sizeof(T)),
                       activeOuter);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(
                static_cast<event_t>(vectorToMte2Event));
            WaitFlag<HardEvent::V_MTE2>(
                static_cast<event_t>(vectorToMte2Event));

            const uint64_t firstSourceOffset =
                firstOuter * indexLength_;
            const DataCopyExtParams sourceCopy{
                1, activeOuter * indexLength_ *
                       static_cast<uint32_t>(sizeof(T)),
                0, 0, 0};
            DataCopyPad(staged, source_[firstSourceOffset], sourceCopy,
                        DataCopyPadExtParams<T>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(
                static_cast<event_t>(mte2ToVectorEvent));
            WaitFlag<HardEvent::MTE2_V>(
                static_cast<event_t>(mte2ToVectorEvent));

            const uint32_t pairCount = indexLength_ / 2;
            uint64_t packedIndex =
                pairCount == 0 ? 0 : cachedIndexPairs.GetValue(0);
            for (uint32_t pair = 0; pair < pairCount; ++pair) {
                const uint64_t nextPackedIndex =
                    pair + 1 < pairCount
                        ? cachedIndexPairs.GetValue(pair + 1)
                        : 0;
#pragma unroll
                for (uint32_t packedLane = 0; packedLane < 2;
                     ++packedLane) {
                    const uint32_t update = pair * 2 + packedLane;
                    Gather(sourceScratch, staged, sourceOffsets,
                           update * static_cast<uint32_t>(sizeof(T)),
                           activeOuter);
                    PipeBarrier<PIPE_V>();
                    const uint32_t axis =
                        static_cast<uint32_t>(packedIndex);
                    packedIndex >>= 32;
                    Add(accumulators[axis * kWholeAxisOuterBatch],
                        accumulators[axis * kWholeAxisOuterBatch],
                        sourceScratch, activeOuter);
                    PipeBarrier<PIPE_V>();
                }
                packedIndex = nextPackedIndex;
            }
            if ((indexLength_ & 1U) != 0) {
                const uint32_t update = 2 * pairCount;
                Gather(sourceScratch, staged, sourceOffsets,
                       update * static_cast<uint32_t>(sizeof(T)),
                       activeOuter);
                PipeBarrier<PIPE_V>();
                const uint32_t axis = static_cast<uint32_t>(
                    cachedIndices.GetValue(update));
                Add(accumulators[axis * kWholeAxisOuterBatch],
                    accumulators[axis * kWholeAxisOuterBatch],
                    sourceScratch, activeOuter);
                PipeBarrier<PIPE_V>();
            }

            const uint32_t outputRowBytes =
                (axisLength_ * static_cast<uint32_t>(sizeof(T)) + 31U) &
                ~31U;
            const uint32_t outputRowElements = outputRowBytes / sizeof(T);
            for (uint32_t lane = 0; lane < activeOuter; ++lane) {
                Gather(staged[lane * outputRowElements], accumulators,
                       axisOffsets,
                       lane * static_cast<uint32_t>(sizeof(T)),
                       axisLength_);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(
                static_cast<event_t>(vectorToMte3Event));
            WaitFlag<HardEvent::V_MTE3>(
                static_cast<event_t>(vectorToMte3Event));
            const DataCopyExtParams outputCopy{
                static_cast<uint16_t>(activeOuter),
                axisLength_ * static_cast<uint32_t>(sizeof(T)),
                0, 0, 0};
            DataCopyPad(output_[firstSelfOffset], staged, outputCopy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(mte3ToMte2Event));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(mte3ToMte2Event));
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(mte2ToVectorEvent);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(vectorToMte2Event);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(vectorToMte3Event);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
    }

    __aicore__ inline TEventID AllocateMte2ToDeepComputeEvent()
    {
        if constexpr (kUseDeepScalarFloat) {
            return pipe_.AllocEventID<HardEvent::MTE2_S>();
        } else {
            return pipe_.AllocEventID<HardEvent::MTE2_V>();
        }
    }

    __aicore__ inline void ReleaseMte2ToDeepComputeEvent(TEventID event)
    {
        if constexpr (kUseDeepScalarFloat) {
            pipe_.ReleaseEventID<HardEvent::MTE2_S>(event);
        } else {
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(event);
        }
    }

    __aicore__ inline void SetMte2ToDeepComputeEvent(TEventID event)
    {
        if constexpr (kUseDeepScalarFloat) {
            SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(event));
        } else {
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(event));
        }
    }

    __aicore__ inline void WaitMte2ToDeepComputeEvent(TEventID event)
    {
        if constexpr (kUseDeepScalarFloat) {
            WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(event));
        } else {
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(event));
        }
    }

    __aicore__ inline TEventID AllocateDeepComputeToMte2Event()
    {
        if constexpr (kUseDeepScalarFloat) {
            return pipe_.AllocEventID<HardEvent::S_MTE2>();
        } else {
            return pipe_.AllocEventID<HardEvent::V_MTE2>();
        }
    }

    __aicore__ inline void ReleaseDeepComputeToMte2Event(TEventID event)
    {
        if constexpr (kUseDeepScalarFloat) {
            pipe_.ReleaseEventID<HardEvent::S_MTE2>(event);
        } else {
            pipe_.ReleaseEventID<HardEvent::V_MTE2>(event);
        }
    }

    __aicore__ inline void SetDeepComputeToMte2Event(TEventID event)
    {
        if constexpr (kUseDeepScalarFloat) {
            SetFlag<HardEvent::S_MTE2>(static_cast<event_t>(event));
        } else {
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(event));
        }
    }

    __aicore__ inline void WaitDeepComputeToMte2Event(TEventID event)
    {
        if constexpr (kUseDeepScalarFloat) {
            WaitFlag<HardEvent::S_MTE2>(static_cast<event_t>(event));
        } else {
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(event));
        }
    }

    __aicore__ inline TEventID AllocateDeepComputeToMte3Event()
    {
        if constexpr (kUseDeepScalarFloat) {
            return pipe_.AllocEventID<HardEvent::S_MTE3>();
        } else {
            return pipe_.AllocEventID<HardEvent::V_MTE3>();
        }
    }

    __aicore__ inline void ReleaseDeepComputeToMte3Event(TEventID event)
    {
        if constexpr (kUseDeepScalarFloat) {
            pipe_.ReleaseEventID<HardEvent::S_MTE3>(event);
        } else {
            pipe_.ReleaseEventID<HardEvent::V_MTE3>(event);
        }
    }

    __aicore__ inline void SetDeepComputeToMte3Event(TEventID event)
    {
        if constexpr (kUseDeepScalarFloat) {
            SetFlag<HardEvent::S_MTE3>(static_cast<event_t>(event));
        } else {
            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(event));
        }
    }

    __aicore__ inline void WaitDeepComputeToMte3Event(TEventID event)
    {
        if constexpr (kUseDeepScalarFloat) {
            WaitFlag<HardEvent::S_MTE3>(static_cast<event_t>(event));
        } else {
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(event));
        }
    }

    __aicore__ inline void ProcessDeepSmallInnerOneTwentyEightWay()
    {
        LocalTensor<AccT> groupedAccumulators = accumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> sourceAccumulator = sourceAccumulatorBuffer_.Get<AccT>();
        const uint32_t length = static_cast<uint32_t>(innerLength_);
        uint32_t rowsPerGroup = kDeepSmallInnerMaxGroupSize;
        const uint64_t minimumGroupCount = 2 * static_cast<uint64_t>(GetBlockNum());
        while (rowsPerGroup > kDeepSmallInnerMinGroupSize &&
               outerLength_ *
                       ((axisLength_ + rowsPerGroup - 1) / rowsPerGroup) <
                   minimumGroupCount) {
            rowsPerGroup /= 2;
        }
        const uint32_t groupsPerOuter =
            (axisLength_ + rowsPerGroup - 1) / rowsPerGroup;
        const uint64_t groupCount = outerLength_ * groupsPerOuter;
        const uint64_t blockStride = GetBlockNum();
        LocalTensor<int32_t> cachedIndices =
            groupedAccumulators.template ReinterpretCast<int32_t>();
        if constexpr (CacheDeepIndices) {
            const uint32_t accumulatorBytes =
                axisLength_ * kDeepAccumulatorStride * sizeof(AccT);
            const uint32_t cacheOffset =
                ((accumulatorBytes + 31U) & ~31U) / sizeof(int32_t);
            cachedIndices = cachedIndices[cacheOffset];
            const DataCopyExtParams indexCopy{
                1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
            const TEventID mte2ToScalarEvent =
                pipe_.AllocEventID<HardEvent::MTE2_S>();
            DataCopyPad(cachedIndices, index_, indexCopy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_S>(
                static_cast<event_t>(mte2ToScalarEvent));
            WaitFlag<HardEvent::MTE2_S>(
                static_cast<event_t>(mte2ToScalarEvent));
            pipe_.ReleaseEventID<HardEvent::MTE2_S>(mte2ToScalarEvent);
        }
        const TEventID mte2ToVectorEvent = AllocateMte2ToDeepComputeEvent();
        const TEventID secondMte2ToVectorEvent =
            AllocateMte2ToDeepComputeEvent();
        const TEventID vectorToMte2Event = AllocateDeepComputeToMte2Event();
        const TEventID secondVectorToMte2Event =
            AllocateDeepComputeToMte2Event();
        const TEventID vectorToMte3Event = AllocateDeepComputeToMte3Event();
        const TEventID mte3ToMte2Event = pipe_.AllocEventID<HardEvent::MTE3_MTE2>();

        for (uint64_t group = GetBlockIdx(); group < groupCount; group += blockStride) {
            const uint64_t outerIndex = group / groupsPerOuter;
            const uint32_t groupWithinOuter =
                static_cast<uint32_t>(group - outerIndex * groupsPerOuter);
            const uint32_t firstAxisIndex = groupWithinOuter * rowsPerGroup;
            const uint32_t rowCount =
                axisLength_ - firstAxisIndex < rowsPerGroup
                    ? axisLength_ - firstAxisIndex
                    : rowsPerGroup;
            const uint64_t firstWork = outerIndex * axisLength_ + firstAxisIndex;

            if constexpr (IsSameType<T, AccT>::value) {
                LoadDeepSmallAccumulators(groupedAccumulators, firstWork,
                                          rowCount, mte2ToVectorEvent);
            } else {
                for (uint32_t lane = 0; lane < rowCount; ++lane) {
                    LoadAccumulator(
                        groupedAccumulators[lane * kDeepAccumulatorStride],
                        inputQueue_, self_[(firstWork + lane) * innerLength_], length);
                }
            }

            if (groupsPerOuter == 1) {
                ProcessDeepSmallInnerStagedSource(
                    groupedAccumulators, sourceAccumulator, cachedIndices, outerIndex,
                    firstAxisIndex, rowCount, length,
                    mte2ToVectorEvent, secondMte2ToVectorEvent,
                    vectorToMte2Event, secondVectorToMte2Event);
            } else if constexpr (IsSameType<T, half>::value ||
                                 IsSameType<T, int8_t>::value) {
                const uint32_t pairCount = indexLength_ / 2;
                const __gm__ uint64_t* indexPairs =
                    reinterpret_cast<const __gm__ uint64_t*>(index_.GetPhyAddr());
                uint64_t packedIndex = pairCount == 0 ? 0 : indexPairs[0];
                for (uint32_t pair = 0; pair < pairCount; ++pair) {
                    const uint64_t nextPackedIndex =
                        pair + 1 < pairCount ? indexPairs[pair + 1] : 0;
#pragma unroll
                    for (uint32_t packedLane = 0; packedLane < 2; ++packedLane) {
                        const uint32_t update = pair * 2 + packedLane;
                        const int32_t indexValue =
                            static_cast<int32_t>(static_cast<uint32_t>(packedIndex));
                        packedIndex >>= 32;
                        const uint32_t outputLane =
                            static_cast<uint32_t>(indexValue) - firstAxisIndex;
                        if (outputLane < rowCount) {
                            AccumulateSource(
                                groupedAccumulators[
                                    outputLane * kDeepAccumulatorStride],
                                sourceAccumulator, update, outerIndex, 0, length);
                        }
                    }
                    packedIndex = nextPackedIndex;
                }
                if ((indexLength_ & 1U) != 0) {
                    const uint32_t update = 2 * pairCount;
                    const int32_t indexValue = index_.GetValue(update);
                    const uint32_t outputLane =
                        static_cast<uint32_t>(indexValue) - firstAxisIndex;
                    if (outputLane < rowCount) {
                        AccumulateSource(
                            groupedAccumulators[
                                outputLane * kDeepAccumulatorStride],
                            sourceAccumulator, update, outerIndex, 0, length);
                    }
                }
            } else {
                for (uint32_t update = 0; update < indexLength_; ++update) {
                    const int32_t indexValue = index_.GetValue(update);
                    const uint32_t outputLane =
                        static_cast<uint32_t>(indexValue) - firstAxisIndex;
                    if (outputLane < rowCount) {
                        AccumulateSource(
                            groupedAccumulators[
                                outputLane * kDeepAccumulatorStride],
                            sourceAccumulator, update, outerIndex, 0, length);
                    }
                }
            }

            if constexpr (IsSameType<T, AccT>::value) {
                StoreDeepSmallAccumulators(groupedAccumulators, firstWork,
                                           rowCount, vectorToMte3Event,
                                           mte3ToMte2Event);
            } else {
                for (uint32_t lane = 0; lane < rowCount; ++lane) {
                    StoreAccumulator(
                        output_[(firstWork + lane) * innerLength_],
                        groupedAccumulators[lane * kDeepAccumulatorStride],
                        length);
                }
            }
        }

        ReleaseMte2ToDeepComputeEvent(mte2ToVectorEvent);
        ReleaseMte2ToDeepComputeEvent(secondMte2ToVectorEvent);
        ReleaseDeepComputeToMte2Event(vectorToMte2Event);
        ReleaseDeepComputeToMte2Event(secondVectorToMte2Event);
        ReleaseDeepComputeToMte3Event(vectorToMte3Event);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
    }

    __aicore__ inline void LoadDeepSmallAccumulators(
        LocalTensor<AccT> groupedAccumulators, uint64_t firstWork,
        uint32_t rowCount, TEventID mte2ToVectorEvent)
    {
        if constexpr (kUseDeepScalarFloat) {
            const DataCopyExtParams copy{
                1, rowCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(groupedAccumulators,
                        self_[firstWork * innerLength_], copy,
                        DataCopyPadExtParams<T>{false, 0, 0, 0});
        } else {
            const uint32_t rowBytes = innerLength_ * sizeof(T);
            const uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
            const uint32_t accumulatorRowBytes =
                kDeepSmallInnerStride * sizeof(AccT);
            const uint32_t destinationStrideBlocks =
                (accumulatorRowBytes - alignedRowBytes) / 32U;
            const DataCopyExtParams copy{
                static_cast<uint16_t>(rowCount), rowBytes, 0,
                destinationStrideBlocks, 0};
            DataCopyPad(groupedAccumulators,
                        self_[firstWork * innerLength_], copy,
                        DataCopyPadExtParams<T>{false, 0, 0, 0});
        }
        SetMte2ToDeepComputeEvent(mte2ToVectorEvent);
        WaitMte2ToDeepComputeEvent(mte2ToVectorEvent);
    }

    __aicore__ inline void StoreDeepSmallAccumulators(
        LocalTensor<AccT> groupedAccumulators, uint64_t firstWork,
        uint32_t rowCount, TEventID vectorToMte3Event,
        TEventID mte3ToMte2Event)
    {
        SetDeepComputeToMte3Event(vectorToMte3Event);
        WaitDeepComputeToMte3Event(vectorToMte3Event);
        if constexpr (kUseDeepScalarFloat) {
            const DataCopyExtParams copy{
                1, rowCount * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(output_[firstWork * innerLength_], groupedAccumulators,
                        copy);
        } else {
            const uint32_t rowBytes = innerLength_ * sizeof(T);
            const uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
            const uint32_t accumulatorRowBytes =
                kDeepSmallInnerStride * sizeof(AccT);
            const uint32_t sourceStrideBlocks =
                (accumulatorRowBytes - alignedRowBytes) / 32U;
            const DataCopyExtParams copy{
                static_cast<uint16_t>(rowCount), rowBytes,
                sourceStrideBlocks, 0, 0};
            DataCopyPad(output_[firstWork * innerLength_], groupedAccumulators,
                        copy);
        }
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(mte3ToMte2Event));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(mte3ToMte2Event));
    }

    __aicore__ inline void ProcessDeepSmallInnerStagedSource(
        LocalTensor<AccT> groupedAccumulators,
        LocalTensor<AccT> sourceAccumulator, LocalTensor<int32_t> cachedIndices,
        uint64_t outerIndex,
        uint32_t firstAxisIndex, uint32_t rowCount, uint32_t length,
        TEventID mte2ToVectorEvent, TEventID secondMte2ToVectorEvent,
        TEventID vectorToMte2Event, TEventID secondVectorToMte2Event)
    {
        LocalTensor<T> firstStagedSource = stagedSourceBuffer_.Get<T>();
        LocalTensor<T> secondStagedSource =
            firstStagedSource[kDeepSmallSourceBufferRows * kDeepSmallInnerStride];
        const uint64_t outerSourceBase = outerIndex * indexLength_ * innerLength_;
        const uint32_t sourceRowBytes = innerLength_ * sizeof(T);
        const uint32_t alignedSourceRowBytes = (sourceRowBytes + 31U) & ~31U;
        const uint32_t stagedRowBytes =
            CacheDeepIndices ? alignedSourceRowBytes
                             : kDeepSmallInnerStride * sizeof(T);
        const uint32_t stagedRowElements = stagedRowBytes / sizeof(T);
        const uint32_t sourceBufferRows =
            kDeepSmallSourceBufferRows * kDeepSmallInnerStride * sizeof(T) /
            stagedRowBytes;
        const uint32_t destinationStrideBlocks =
            (stagedRowBytes - alignedSourceRowBytes) / 32U;

        const uint32_t firstUpdateCount =
            indexLength_ < sourceBufferRows
                ? indexLength_
                : sourceBufferRows;
        CopyDeepSmallSourceBatch(firstStagedSource, outerSourceBase, 0,
                                 firstUpdateCount, sourceRowBytes,
                                 destinationStrideBlocks,
                                 mte2ToVectorEvent);

        uint32_t batchIndex = 0;
        for (uint32_t firstUpdate = 0; firstUpdate < indexLength_;
             firstUpdate += sourceBufferRows, ++batchIndex) {
            const bool useSecondBuffer = (batchIndex & 1U) != 0;
            const uint32_t updateCount =
                indexLength_ - firstUpdate < sourceBufferRows
                    ? indexLength_ - firstUpdate
                    : sourceBufferRows;
            if (useSecondBuffer) {
                WaitMte2ToDeepComputeEvent(secondMte2ToVectorEvent);
            } else {
                WaitMte2ToDeepComputeEvent(mte2ToVectorEvent);
            }

            const uint32_t nextUpdate = firstUpdate + sourceBufferRows;
            if (nextUpdate < indexLength_) {
                const uint32_t nextUpdateCount =
                    indexLength_ - nextUpdate < sourceBufferRows
                        ? indexLength_ - nextUpdate
                        : sourceBufferRows;
                if (useSecondBuffer) {
                    if (batchIndex >= 1) {
                        WaitDeepComputeToMte2Event(vectorToMte2Event);
                    }
                    CopyDeepSmallSourceBatch(
                        firstStagedSource, outerSourceBase, nextUpdate,
                        nextUpdateCount, sourceRowBytes,
                        destinationStrideBlocks, mte2ToVectorEvent);
                } else {
                    if (batchIndex >= 1) {
                        WaitDeepComputeToMte2Event(secondVectorToMte2Event);
                    }
                    CopyDeepSmallSourceBatch(
                        secondStagedSource, outerSourceBase, nextUpdate,
                        nextUpdateCount, sourceRowBytes,
                        destinationStrideBlocks, secondMte2ToVectorEvent);
                }
            }

            if constexpr (CacheDeepIndices) {
                LocalTensor<uint64_t> cachedIndexPairs =
                    cachedIndices.template ReinterpretCast<uint64_t>();
                const uint32_t pairCount = updateCount / 2;
                const uint32_t firstPair = firstUpdate / 2;
                uint64_t packedIndex =
                    pairCount == 0 ? 0 : cachedIndexPairs.GetValue(firstPair);
                for (uint32_t pair = 0; pair < pairCount; ++pair) {
                    const uint64_t nextPackedIndex =
                        pair + 1 < pairCount
                            ? cachedIndexPairs.GetValue(firstPair + pair + 1)
                            : 0;
#pragma unroll
                    for (uint32_t packedLane = 0; packedLane < 2;
                         ++packedLane) {
                        const uint32_t localUpdate = pair * 2 + packedLane;
                        const int32_t indexValue = static_cast<int32_t>(
                            static_cast<uint32_t>(packedIndex));
                        packedIndex >>= 32;
                        const uint32_t outputLane =
                            static_cast<uint32_t>(indexValue) - firstAxisIndex;
                        if (outputLane < rowCount) {
                            AccumulateStagedDeepSource(
                                groupedAccumulators[
                                    outputLane * kDeepAccumulatorStride],
                                sourceAccumulator,
                                useSecondBuffer
                                    ? secondStagedSource[
                                          localUpdate * stagedRowElements]
                                    : firstStagedSource[
                                          localUpdate * stagedRowElements],
                                length);
                        }
                    }
                    packedIndex = nextPackedIndex;
                }
                if ((updateCount & 1U) != 0) {
                    const uint32_t localUpdate = 2 * pairCount;
                    const uint32_t update = firstUpdate + localUpdate;
                    const uint32_t outputLane = static_cast<uint32_t>(
                                                    cachedIndices.GetValue(update)) -
                                                firstAxisIndex;
                    if (outputLane < rowCount) {
                        AccumulateStagedDeepSource(
                            groupedAccumulators[
                                outputLane * kDeepAccumulatorStride],
                            sourceAccumulator,
                            useSecondBuffer
                                ? secondStagedSource[
                                      localUpdate * stagedRowElements]
                                : firstStagedSource[
                                      localUpdate * stagedRowElements],
                            length);
                    }
                }
            } else {
                for (uint32_t localUpdate = 0; localUpdate < updateCount;
                     ++localUpdate) {
                    const uint32_t update = firstUpdate + localUpdate;
                    const uint32_t outputLane = static_cast<uint32_t>(
                                                    index_.GetValue(update)) -
                                                firstAxisIndex;
                    if (outputLane < rowCount) {
                        AccumulateStagedDeepSource(
                            groupedAccumulators[
                                outputLane * kDeepAccumulatorStride],
                            sourceAccumulator,
                            useSecondBuffer
                                ? secondStagedSource[
                                      localUpdate * stagedRowElements]
                                : firstStagedSource[
                                      localUpdate * stagedRowElements],
                            length);
                    }
                }
            }

            if (useSecondBuffer) {
                SetDeepComputeToMte2Event(secondVectorToMte2Event);
            } else {
                SetDeepComputeToMte2Event(vectorToMte2Event);
            }
        }

        WaitDeepComputeToMte2Event(vectorToMte2Event);
        if (indexLength_ > sourceBufferRows) {
            WaitDeepComputeToMte2Event(secondVectorToMte2Event);
        }
    }

    __aicore__ inline void CopyDeepSmallSourceBatch(
        LocalTensor<T> stagedSource, uint64_t outerSourceBase,
        uint32_t firstUpdate, uint32_t updateCount, uint32_t sourceRowBytes,
        uint32_t destinationStrideBlocks, TEventID mte2ToVectorEvent)
    {
        const DataCopyExtParams sourceCopy{
            static_cast<uint16_t>(updateCount), sourceRowBytes, 0,
            destinationStrideBlocks, 0};
        const uint64_t firstSourceOffset =
            outerSourceBase + static_cast<uint64_t>(firstUpdate) * innerLength_;
        DataCopyPad(stagedSource, source_[firstSourceOffset], sourceCopy,
                    DataCopyPadExtParams<T>{false, 0, 0, 0});
        SetMte2ToDeepComputeEvent(mte2ToVectorEvent);
    }

    __aicore__ inline void AccumulateStagedDeepSource(
        LocalTensor<AccT> accumulator, LocalTensor<AccT> sourceAccumulator,
        LocalTensor<T> stagedSource, uint32_t length)
    {
        if constexpr (kUseDeepScalarFloat) {
            accumulator.SetValue(
                0, accumulator.GetValue(0) + stagedSource.GetValue(0));
            return;
        } else if constexpr (IsSameType<T, AccT>::value) {
            Add(accumulator, accumulator, stagedSource, length);
        } else if constexpr (IsSameType<T, int8_t>::value) {
            LocalTensor<half> conversion = conversionBuffer_.Get<half>();
            Cast(conversion, stagedSource, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Cast(sourceAccumulator, conversion, RoundMode::CAST_RINT, length);
            PipeBarrier<PIPE_V>();
            Add(accumulator, accumulator, sourceAccumulator, length);
        } else {
            Cast(sourceAccumulator, stagedSource, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Add(accumulator, accumulator, sourceAccumulator, length);
        }
        PipeBarrier<PIPE_V>();
        RoundBfloatAccumulator(accumulator, length);
    }

    __aicore__ inline void DispatchSmallInnerEightWay(
        uint32_t lane, LocalTensor<AccT> firstAccumulator,
        LocalTensor<AccT> secondAccumulator, LocalTensor<AccT> thirdAccumulator,
        LocalTensor<AccT> fourthAccumulator, LocalTensor<AccT> fifthAccumulator,
        LocalTensor<AccT> sixthAccumulator, LocalTensor<AccT> seventhAccumulator,
        LocalTensor<AccT> eighthAccumulator, LocalTensor<AccT> sourceAccumulator,
        uint32_t update, uint64_t outerIndex, uint32_t length)
    {
        switch (lane) {
            case 0:
                AccumulateSource(firstAccumulator, sourceAccumulator, update,
                                 outerIndex, 0, length);
                break;
            case 1:
                AccumulateSource(secondAccumulator, sourceAccumulator, update,
                                 outerIndex, 0, length);
                break;
            case 2:
                AccumulateSource(thirdAccumulator, sourceAccumulator, update,
                                 outerIndex, 0, length);
                break;
            case 3:
                AccumulateSource(fourthAccumulator, sourceAccumulator, update,
                                 outerIndex, 0, length);
                break;
            case 4:
                AccumulateSource(fifthAccumulator, sourceAccumulator, update,
                                 outerIndex, 0, length);
                break;
            case 5:
                AccumulateSource(sixthAccumulator, sourceAccumulator, update,
                                 outerIndex, 0, length);
                break;
            case 6:
                AccumulateSource(seventhAccumulator, sourceAccumulator, update,
                                 outerIndex, 0, length);
                break;
            case 7:
                AccumulateSource(eighthAccumulator, sourceAccumulator, update,
                                 outerIndex, 0, length);
                break;
            default:
                break;
        }
    }

    __aicore__ inline void ProcessSmallInnerEightWay(uint64_t workCount)
    {
        LocalTensor<AccT> groupedAccumulators = accumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> firstAccumulator = groupedAccumulators[0];
        LocalTensor<AccT> secondAccumulator =
            groupedAccumulators[kBalancedSmallInnerEightWayLimit];
        LocalTensor<AccT> thirdAccumulator =
            groupedAccumulators[2 * kBalancedSmallInnerEightWayLimit];
        LocalTensor<AccT> fourthAccumulator =
            groupedAccumulators[3 * kBalancedSmallInnerEightWayLimit];
        LocalTensor<AccT> fifthAccumulator =
            groupedAccumulators[4 * kBalancedSmallInnerEightWayLimit];
        LocalTensor<AccT> sixthAccumulator =
            groupedAccumulators[5 * kBalancedSmallInnerEightWayLimit];
        LocalTensor<AccT> seventhAccumulator =
            groupedAccumulators[6 * kBalancedSmallInnerEightWayLimit];
        LocalTensor<AccT> eighthAccumulator =
            groupedAccumulators[7 * kBalancedSmallInnerEightWayLimit];
        LocalTensor<AccT> sourceAccumulator = sourceAccumulatorBuffer_.Get<AccT>();
        const uint32_t length = static_cast<uint32_t>(innerLength_);
        const uint64_t groupCount = workCount / 8;
        const uint64_t blockStride = GetBlockNum();

        for (uint64_t group = GetBlockIdx(); group < groupCount; group += blockStride) {
            const uint64_t firstWork = 8 * group;
            const uint32_t firstAxisIndex = static_cast<uint32_t>(firstWork % axisLength_);
            const uint64_t outerIndex = firstWork / axisLength_;
            const uint64_t firstOutputOffset = firstWork * innerLength_;

            LoadAccumulator(firstAccumulator, inputQueue_, self_[firstOutputOffset], length);
            LoadAccumulator(secondAccumulator, inputQueue_,
                            self_[firstOutputOffset + innerLength_], length);
            LoadAccumulator(thirdAccumulator, inputQueue_,
                            self_[firstOutputOffset + 2 * innerLength_], length);
            LoadAccumulator(fourthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 3 * innerLength_], length);
            LoadAccumulator(fifthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 4 * innerLength_], length);
            LoadAccumulator(sixthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 5 * innerLength_], length);
            LoadAccumulator(seventhAccumulator, inputQueue_,
                            self_[firstOutputOffset + 6 * innerLength_], length);
            LoadAccumulator(eighthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 7 * innerLength_], length);

            if constexpr (UsePackedSmallIndices || IsSameType<T, half>::value ||
                          (UseBalancedSmallEightWay && IsSameType<T, int8_t>::value)) {
                const uint32_t pairCount = indexLength_ / 2;
                const __gm__ uint64_t* indexPairs =
                    reinterpret_cast<const __gm__ uint64_t*>(index_.GetPhyAddr());
                uint64_t packedIndex = pairCount == 0 ? 0 : indexPairs[0];
                for (uint32_t pair = 0; pair < pairCount; ++pair) {
                    const uint64_t nextPackedIndex =
                        pair + 1 < pairCount ? indexPairs[pair + 1] : 0;
#pragma unroll
                    for (uint32_t packedLane = 0; packedLane < 2; ++packedLane) {
                        const uint32_t update = pair * 2 + packedLane;
                        const int32_t indexValue =
                            static_cast<int32_t>(static_cast<uint32_t>(packedIndex));
                        packedIndex >>= 32;
                        const uint32_t outputLane =
                            static_cast<uint32_t>(indexValue) - firstAxisIndex;
                        if (outputLane < 8) {
                            DispatchSmallInnerEightWay(
                                outputLane, firstAccumulator, secondAccumulator,
                                thirdAccumulator, fourthAccumulator, fifthAccumulator,
                                sixthAccumulator, seventhAccumulator, eighthAccumulator,
                                sourceAccumulator, update, outerIndex, length);
                        }
                    }
                    packedIndex = nextPackedIndex;
                }
                if ((indexLength_ & 1U) != 0) {
                    const uint32_t update = 2 * pairCount;
                    const int32_t indexValue = index_.GetValue(update);
                    const uint32_t outputLane =
                        static_cast<uint32_t>(indexValue) - firstAxisIndex;
                    if (outputLane < 8) {
                        DispatchSmallInnerEightWay(
                            outputLane, firstAccumulator, secondAccumulator,
                            thirdAccumulator, fourthAccumulator, fifthAccumulator,
                            sixthAccumulator, seventhAccumulator, eighthAccumulator,
                            sourceAccumulator, update, outerIndex, length);
                    }
                }
            } else {
                for (uint32_t update = 0; update < indexLength_; ++update) {
                    const int32_t indexValue = index_.GetValue(update);
                    const uint32_t outputLane =
                        static_cast<uint32_t>(indexValue) - firstAxisIndex;
                    if (outputLane < 8) {
                        DispatchSmallInnerEightWay(
                            outputLane, firstAccumulator, secondAccumulator,
                            thirdAccumulator, fourthAccumulator, fifthAccumulator,
                            sixthAccumulator, seventhAccumulator, eighthAccumulator,
                            sourceAccumulator, update, outerIndex, length);
                    }
                }
            }

            StoreAccumulator(output_[firstOutputOffset], firstAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + innerLength_],
                             secondAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 2 * innerLength_],
                             thirdAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 3 * innerLength_],
                             fourthAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 4 * innerLength_],
                             fifthAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 5 * innerLength_],
                             sixthAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 6 * innerLength_],
                             seventhAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 7 * innerLength_],
                             eighthAccumulator, length);
        }
    }

    __aicore__ inline void ProcessAxisTailEightWay()
    {
        constexpr uint32_t accumulatorStride =
            UseAxisTailMediumEightWay ? kBalancedMediumInnerEightWayLimit
                                      : kBalancedSmallInnerEightWayLimit;
        LocalTensor<AccT> groupedAccumulators = accumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> firstAccumulator = groupedAccumulators[0];
        LocalTensor<AccT> secondAccumulator =
            groupedAccumulators[accumulatorStride];
        LocalTensor<AccT> thirdAccumulator =
            groupedAccumulators[2 * accumulatorStride];
        LocalTensor<AccT> fourthAccumulator =
            groupedAccumulators[3 * accumulatorStride];
        LocalTensor<AccT> fifthAccumulator =
            groupedAccumulators[4 * accumulatorStride];
        LocalTensor<AccT> sixthAccumulator =
            groupedAccumulators[5 * accumulatorStride];
        LocalTensor<AccT> seventhAccumulator =
            groupedAccumulators[6 * accumulatorStride];
        LocalTensor<AccT> eighthAccumulator =
            groupedAccumulators[7 * accumulatorStride];
        LocalTensor<AccT> sourceAccumulator = sourceAccumulatorBuffer_.Get<AccT>();
        const uint32_t length = static_cast<uint32_t>(innerLength_);
        const uint32_t groupsPerOuter = axisLength_ / 8;
        const uint64_t fullGroupCount = outerLength_ * groupsPerOuter;
        const uint64_t blockStride = GetBlockNum();

        for (uint64_t group = GetBlockIdx(); group < fullGroupCount; group += blockStride) {
            const uint64_t outerIndex = group / groupsPerOuter;
            const uint32_t groupWithinOuter =
                static_cast<uint32_t>(group - outerIndex * groupsPerOuter);
            const uint32_t firstAxisIndex = 8 * groupWithinOuter;
            const uint64_t firstWork = outerIndex * axisLength_ + firstAxisIndex;
            const uint64_t firstOutputOffset = firstWork * innerLength_;

            LoadAccumulator(firstAccumulator, inputQueue_, self_[firstOutputOffset], length);
            LoadAccumulator(secondAccumulator, inputQueue_,
                            self_[firstOutputOffset + innerLength_], length);
            LoadAccumulator(thirdAccumulator, inputQueue_,
                            self_[firstOutputOffset + 2 * innerLength_], length);
            LoadAccumulator(fourthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 3 * innerLength_], length);
            LoadAccumulator(fifthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 4 * innerLength_], length);
            LoadAccumulator(sixthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 5 * innerLength_], length);
            LoadAccumulator(seventhAccumulator, inputQueue_,
                            self_[firstOutputOffset + 6 * innerLength_], length);
            LoadAccumulator(eighthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 7 * innerLength_], length);

            if constexpr (IsSameType<T, half>::value || IsSameType<T, int8_t>::value) {
                const uint32_t pairCount = indexLength_ / 2;
                const __gm__ uint64_t* indexPairs =
                    reinterpret_cast<const __gm__ uint64_t*>(index_.GetPhyAddr());
                uint64_t packedIndex = pairCount == 0 ? 0 : indexPairs[0];
                for (uint32_t pair = 0; pair < pairCount; ++pair) {
                    const uint64_t nextPackedIndex =
                        pair + 1 < pairCount ? indexPairs[pair + 1] : 0;
#pragma unroll
                    for (uint32_t packedLane = 0; packedLane < 2; ++packedLane) {
                        const uint32_t update = pair * 2 + packedLane;
                        const int32_t indexValue =
                            static_cast<int32_t>(static_cast<uint32_t>(packedIndex));
                        packedIndex >>= 32;
                        const uint32_t outputLane =
                            static_cast<uint32_t>(indexValue) - firstAxisIndex;
                        if (outputLane < 8) {
                            DispatchSmallInnerEightWay(
                                outputLane, firstAccumulator, secondAccumulator,
                                thirdAccumulator, fourthAccumulator, fifthAccumulator,
                                sixthAccumulator, seventhAccumulator, eighthAccumulator,
                                sourceAccumulator, update, outerIndex, length);
                        }
                    }
                    packedIndex = nextPackedIndex;
                }
                if ((indexLength_ & 1U) != 0) {
                    const uint32_t update = 2 * pairCount;
                    const int32_t indexValue = index_.GetValue(update);
                    const uint32_t outputLane =
                        static_cast<uint32_t>(indexValue) - firstAxisIndex;
                    if (outputLane < 8) {
                        DispatchSmallInnerEightWay(
                            outputLane, firstAccumulator, secondAccumulator,
                            thirdAccumulator, fourthAccumulator, fifthAccumulator,
                            sixthAccumulator, seventhAccumulator, eighthAccumulator,
                            sourceAccumulator, update, outerIndex, length);
                    }
                }
            } else {
                for (uint32_t update = 0; update < indexLength_; ++update) {
                    const int32_t indexValue = index_.GetValue(update);
                    const uint32_t outputLane =
                        static_cast<uint32_t>(indexValue) - firstAxisIndex;
                    if (outputLane < 8) {
                        DispatchSmallInnerEightWay(
                            outputLane, firstAccumulator, secondAccumulator,
                            thirdAccumulator, fourthAccumulator, fifthAccumulator,
                            sixthAccumulator, seventhAccumulator, eighthAccumulator,
                            sourceAccumulator, update, outerIndex, length);
                    }
                }
            }

            StoreAccumulator(output_[firstOutputOffset], firstAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + innerLength_],
                             secondAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 2 * innerLength_],
                             thirdAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 3 * innerLength_],
                             fourthAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 4 * innerLength_],
                             fifthAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 5 * innerLength_],
                             sixthAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 6 * innerLength_],
                             seventhAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 7 * innerLength_],
                             eighthAccumulator, length);
        }

        const uint32_t tailRowCount = axisLength_ % 8;
        const uint32_t tailAxisBegin = 8 * groupsPerOuter;
        const uint64_t tailWorkCount = outerLength_ * tailRowCount;
        for (uint64_t tailWork = GetBlockIdx(); tailWork < tailWorkCount;
             tailWork += blockStride) {
            const uint64_t outerIndex = tailWork / tailRowCount;
            const uint32_t axisIndex =
                tailAxisBegin + static_cast<uint32_t>(tailWork % tailRowCount);
            const uint64_t outputOffset =
                (outerIndex * axisLength_ + axisIndex) * innerLength_;

            LoadAccumulator(firstAccumulator, inputQueue_, self_[outputOffset], length);
            for (uint32_t update = 0; update < indexLength_; ++update) {
                if (index_.GetValue(update) == static_cast<int32_t>(axisIndex)) {
                    AccumulateSource(firstAccumulator, sourceAccumulator, update,
                                     outerIndex, 0, length);
                }
            }
            StoreAccumulator(output_[outputOffset], firstAccumulator, length);
        }
    }

    __aicore__ inline void ProcessMediumInnerEightWay(uint64_t workCount)
    {
        LocalTensor<AccT> groupedAccumulators = accumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> firstAccumulator = groupedAccumulators[0];
        LocalTensor<AccT> secondAccumulator =
            groupedAccumulators[kBalancedMediumInnerEightWayLimit];
        LocalTensor<AccT> thirdAccumulator =
            groupedAccumulators[2 * kBalancedMediumInnerEightWayLimit];
        LocalTensor<AccT> fourthAccumulator =
            groupedAccumulators[3 * kBalancedMediumInnerEightWayLimit];
        LocalTensor<AccT> fifthAccumulator =
            groupedAccumulators[4 * kBalancedMediumInnerEightWayLimit];
        LocalTensor<AccT> sixthAccumulator =
            groupedAccumulators[5 * kBalancedMediumInnerEightWayLimit];
        LocalTensor<AccT> seventhAccumulator =
            groupedAccumulators[6 * kBalancedMediumInnerEightWayLimit];
        LocalTensor<AccT> eighthAccumulator =
            groupedAccumulators[7 * kBalancedMediumInnerEightWayLimit];
        LocalTensor<AccT> sourceAccumulator = sourceAccumulatorBuffer_.Get<AccT>();
        const uint32_t length = static_cast<uint32_t>(innerLength_);
        const uint64_t groupCount = workCount / 8;
        const uint64_t blockStride = GetBlockNum();

        for (uint64_t group = GetBlockIdx(); group < groupCount; group += blockStride) {
            const uint64_t firstWork = 8 * group;
            const uint32_t firstAxisIndex = static_cast<uint32_t>(firstWork % axisLength_);
            const uint64_t outerIndex = firstWork / axisLength_;
            const uint64_t firstOutputOffset = firstWork * innerLength_;

            LoadAccumulator(firstAccumulator, inputQueue_, self_[firstOutputOffset], length);
            LoadAccumulator(secondAccumulator, inputQueue_,
                            self_[firstOutputOffset + innerLength_], length);
            LoadAccumulator(thirdAccumulator, inputQueue_,
                            self_[firstOutputOffset + 2 * innerLength_], length);
            LoadAccumulator(fourthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 3 * innerLength_], length);
            LoadAccumulator(fifthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 4 * innerLength_], length);
            LoadAccumulator(sixthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 5 * innerLength_], length);
            LoadAccumulator(seventhAccumulator, inputQueue_,
                            self_[firstOutputOffset + 6 * innerLength_], length);
            LoadAccumulator(eighthAccumulator, inputQueue_,
                            self_[firstOutputOffset + 7 * innerLength_], length);

            if constexpr (IsSameType<T, half>::value || IsSameType<T, int8_t>::value) {
                const uint32_t pairCount = indexLength_ / 2;
                const __gm__ uint64_t* indexPairs =
                    reinterpret_cast<const __gm__ uint64_t*>(index_.GetPhyAddr());
                uint64_t packedIndex = pairCount == 0 ? 0 : indexPairs[0];
                for (uint32_t pair = 0; pair < pairCount; ++pair) {
                    const uint64_t nextPackedIndex =
                        pair + 1 < pairCount ? indexPairs[pair + 1] : 0;
#pragma unroll
                    for (uint32_t packedLane = 0; packedLane < 2; ++packedLane) {
                        const uint32_t update = pair * 2 + packedLane;
                        const int32_t indexValue =
                            static_cast<int32_t>(static_cast<uint32_t>(packedIndex));
                        packedIndex >>= 32;
                        const uint32_t outputLane =
                            static_cast<uint32_t>(indexValue) - firstAxisIndex;
                        if (outputLane < 8) {
                            DispatchSmallInnerEightWay(
                                outputLane, firstAccumulator, secondAccumulator,
                                thirdAccumulator, fourthAccumulator, fifthAccumulator,
                                sixthAccumulator, seventhAccumulator, eighthAccumulator,
                                sourceAccumulator, update, outerIndex, length);
                        }
                    }
                    packedIndex = nextPackedIndex;
                }
                if ((indexLength_ & 1U) != 0) {
                    const uint32_t update = 2 * pairCount;
                    const int32_t indexValue = index_.GetValue(update);
                    const uint32_t outputLane =
                        static_cast<uint32_t>(indexValue) - firstAxisIndex;
                    if (outputLane < 8) {
                        DispatchSmallInnerEightWay(
                            outputLane, firstAccumulator, secondAccumulator,
                            thirdAccumulator, fourthAccumulator, fifthAccumulator,
                            sixthAccumulator, seventhAccumulator, eighthAccumulator,
                            sourceAccumulator, update, outerIndex, length);
                    }
                }
            } else {
                for (uint32_t update = 0; update < indexLength_; ++update) {
                    const int32_t indexValue = index_.GetValue(update);
                    const uint32_t outputLane =
                        static_cast<uint32_t>(indexValue) - firstAxisIndex;
                    if (outputLane < 8) {
                        DispatchSmallInnerEightWay(
                            outputLane, firstAccumulator, secondAccumulator,
                            thirdAccumulator, fourthAccumulator, fifthAccumulator,
                            sixthAccumulator, seventhAccumulator, eighthAccumulator,
                            sourceAccumulator, update, outerIndex, length);
                    }
                }
            }

            StoreAccumulator(output_[firstOutputOffset], firstAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + innerLength_],
                             secondAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 2 * innerLength_],
                             thirdAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 3 * innerLength_],
                             fourthAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 4 * innerLength_],
                             fifthAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 5 * innerLength_],
                             sixthAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 6 * innerLength_],
                             seventhAccumulator, length);
            StoreAccumulator(output_[firstOutputOffset + 7 * innerLength_],
                             eighthAccumulator, length);
        }
    }

    __aicore__ inline void ProcessSmallInnerFourWay(uint64_t workCount)
    {
        LocalTensor<AccT> accumulator = accumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> secondAccumulator = secondAccumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> thirdAccumulator = thirdAccumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> fourthAccumulator = fourthAccumulatorBuffer_.Get<AccT>();
        LocalTensor<AccT> sourceAccumulator = sourceAccumulatorBuffer_.Get<AccT>();
        const uint32_t length = static_cast<uint32_t>(innerLength_);
        const uint64_t groupCount = (workCount + 3) / 4;
        const uint64_t blockStride = GetBlockNum();

        for (uint64_t group = GetBlockIdx(); group < groupCount; group += blockStride) {
            const uint64_t firstWork = 4 * group;
            const uint64_t secondWork = firstWork + 1;
            const uint64_t thirdWork = firstWork + 2;
            const uint64_t fourthWork = firstWork + 3;
            const bool hasSecondWork = secondWork < workCount;
            const bool hasThirdWork = thirdWork < workCount;
            const bool hasFourthWork = fourthWork < workCount;

            const uint32_t firstAxisIndex = static_cast<uint32_t>(firstWork % axisLength_);
            const uint64_t firstOuterIndex = firstWork / axisLength_;
            const uint64_t firstOutputOffset = firstWork * innerLength_;
            uint32_t secondAxisIndex = 0;
            uint32_t thirdAxisIndex = 0;
            uint32_t fourthAxisIndex = 0;
            uint64_t secondOuterIndex = 0;
            uint64_t thirdOuterIndex = 0;
            uint64_t fourthOuterIndex = 0;
            if (hasSecondWork) {
                secondAxisIndex = static_cast<uint32_t>(secondWork % axisLength_);
                secondOuterIndex = secondWork / axisLength_;
            }
            if (hasThirdWork) {
                thirdAxisIndex = static_cast<uint32_t>(thirdWork % axisLength_);
                thirdOuterIndex = thirdWork / axisLength_;
            }
            if (hasFourthWork) {
                fourthAxisIndex = static_cast<uint32_t>(fourthWork % axisLength_);
                fourthOuterIndex = fourthWork / axisLength_;
            }

            LoadAccumulator(accumulator, inputQueue_, self_[firstOutputOffset], length);
            if (hasSecondWork) {
                LoadAccumulator(secondAccumulator, inputQueue_, self_[secondWork * innerLength_], length);
            }
            if (hasThirdWork) {
                LoadAccumulator(thirdAccumulator, inputQueue_, self_[thirdWork * innerLength_], length);
            }
            if (hasFourthWork) {
                LoadAccumulator(fourthAccumulator, inputQueue_, self_[fourthWork * innerLength_], length);
            }

            if constexpr (UsePackedSmallIndices || IsSameType<T, half>::value) {
                const uint32_t pairCount = indexLength_ / 2;
                const __gm__ uint64_t* indexPairs =
                    reinterpret_cast<const __gm__ uint64_t*>(index_.GetPhyAddr());
                uint64_t packedIndex = pairCount == 0 ? 0 : indexPairs[0];
                for (uint32_t pair = 0; pair < pairCount; ++pair) {
                    const uint64_t nextPackedIndex =
                        pair + 1 < pairCount ? indexPairs[pair + 1] : 0;
#pragma unroll
                    for (uint32_t lane = 0; lane < 2; ++lane) {
                        const uint32_t update = pair * 2 + lane;
                        const int32_t indexValue =
                            static_cast<int32_t>(static_cast<uint32_t>(packedIndex));
                        packedIndex >>= 32;
                        if (indexValue == static_cast<int32_t>(firstAxisIndex)) {
                            AccumulateSource(accumulator, sourceAccumulator, update,
                                             firstOuterIndex, 0, length);
                        }
                        if (hasSecondWork &&
                            indexValue == static_cast<int32_t>(secondAxisIndex)) {
                            AccumulateSource(secondAccumulator, sourceAccumulator, update,
                                             secondOuterIndex, 0, length);
                        }
                        if (hasThirdWork &&
                            indexValue == static_cast<int32_t>(thirdAxisIndex)) {
                            AccumulateSource(thirdAccumulator, sourceAccumulator, update,
                                             thirdOuterIndex, 0, length);
                        }
                        if (hasFourthWork &&
                            indexValue == static_cast<int32_t>(fourthAxisIndex)) {
                            AccumulateSource(fourthAccumulator, sourceAccumulator, update,
                                             fourthOuterIndex, 0, length);
                        }
                    }
                    packedIndex = nextPackedIndex;
                }
                if ((indexLength_ & 1U) != 0) {
                    const uint32_t update = 2 * pairCount;
                    const int32_t indexValue = index_.GetValue(update);
                    if (indexValue == static_cast<int32_t>(firstAxisIndex)) {
                        AccumulateSource(accumulator, sourceAccumulator, update,
                                         firstOuterIndex, 0, length);
                    }
                    if (hasSecondWork &&
                        indexValue == static_cast<int32_t>(secondAxisIndex)) {
                        AccumulateSource(secondAccumulator, sourceAccumulator, update,
                                         secondOuterIndex, 0, length);
                    }
                    if (hasThirdWork &&
                        indexValue == static_cast<int32_t>(thirdAxisIndex)) {
                        AccumulateSource(thirdAccumulator, sourceAccumulator, update,
                                         thirdOuterIndex, 0, length);
                    }
                    if (hasFourthWork &&
                        indexValue == static_cast<int32_t>(fourthAxisIndex)) {
                        AccumulateSource(fourthAccumulator, sourceAccumulator, update,
                                         fourthOuterIndex, 0, length);
                    }
                }
            } else {
                for (uint32_t update = 0; update < indexLength_; ++update) {
                    const int32_t indexValue = index_.GetValue(update);
                    if (indexValue == static_cast<int32_t>(firstAxisIndex)) {
                        AccumulateSource(accumulator, sourceAccumulator, update,
                                         firstOuterIndex, 0, length);
                    }
                    if (hasSecondWork &&
                        indexValue == static_cast<int32_t>(secondAxisIndex)) {
                        AccumulateSource(secondAccumulator, sourceAccumulator, update,
                                         secondOuterIndex, 0, length);
                    }
                    if (hasThirdWork &&
                        indexValue == static_cast<int32_t>(thirdAxisIndex)) {
                        AccumulateSource(thirdAccumulator, sourceAccumulator, update,
                                         thirdOuterIndex, 0, length);
                    }
                    if (hasFourthWork &&
                        indexValue == static_cast<int32_t>(fourthAxisIndex)) {
                        AccumulateSource(fourthAccumulator, sourceAccumulator, update,
                                         fourthOuterIndex, 0, length);
                    }
                }
            }

            StoreAccumulator(output_[firstOutputOffset], accumulator, length);
            if (hasSecondWork) {
                StoreAccumulator(output_[secondWork * innerLength_], secondAccumulator, length);
            }
            if (hasThirdWork) {
                StoreAccumulator(output_[thirdWork * innerLength_], thirdAccumulator, length);
            }
            if (hasFourthWork) {
                StoreAccumulator(output_[fourthWork * innerLength_], fourthAccumulator, length);
            }
        }
    }

    __aicore__ inline void AccumulateSource(LocalTensor<AccT> accumulator,
                                             LocalTensor<AccT> sourceAccumulator, uint32_t update,
                                             uint64_t outerIndex, uint64_t innerOffset, uint32_t length)
    {
        const uint64_t sourceOffset =
            (outerIndex * indexLength_ + update) * innerLength_ + innerOffset;
        LoadAccumulator(sourceAccumulator, sourceQueue_, source_[sourceOffset], length);
        Add(accumulator, accumulator, sourceAccumulator, length);
        PipeBarrier<PIPE_V>();
        RoundBfloatAccumulator(accumulator, length);
    }

    __aicore__ inline void RoundBfloatAccumulator(LocalTensor<AccT> accumulator,
                                                   uint32_t length)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            LocalTensor<T> rounded = conversionBuffer_.Get<T>();
            Cast(rounded, accumulator, RoundMode::CAST_RINT, length);
            PipeBarrier<PIPE_V>();
            Cast(accumulator, rounded, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void LoadAccumulator(LocalTensor<AccT> destination,
                                            TQue<QuePosition::VECIN, 1>& queue,
                                            GlobalTensor<T> source, uint32_t length)
    {
        LocalTensor<T> staging = queue.AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        DataCopyPad(staging, source, copy, DataCopyPadExtParams<T>{false, 0, 0, 0});
        queue.EnQue<T>(staging);
        staging = queue.DeQue<T>();
        if constexpr (IsSameType<T, AccT>::value) {
            Adds(destination, staging, static_cast<T>(0), length);
        } else if constexpr (IsSameType<T, int8_t>::value) {
            LocalTensor<half> conversion = conversionBuffer_.Get<half>();
            Cast(conversion, staging, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Cast(destination, conversion, RoundMode::CAST_RINT, length);
        } else {
            Cast(destination, staging, RoundMode::CAST_NONE, length);
        }
        PipeBarrier<PIPE_V>();
        queue.FreeTensor(staging);
    }

    __aicore__ inline void StoreAccumulator(GlobalTensor<T> destination, LocalTensor<AccT> source,
                                             uint32_t length)
    {
        LocalTensor<T> staging = outputQueue_.AllocTensor<T>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
        if constexpr (IsSameType<T, AccT>::value) {
            Adds(staging, source, static_cast<T>(0), length);
        } else if constexpr (IsSameType<T, int8_t>::value) {
            ShiftLeft(source, source, static_cast<int16_t>(8), length);
            PipeBarrier<PIPE_V>();
            ShiftRight(source, source, static_cast<int16_t>(8), length);
            PipeBarrier<PIPE_V>();
            LocalTensor<half> conversion = conversionBuffer_.Get<half>();
            Cast(conversion, source, RoundMode::CAST_NONE, length);
            PipeBarrier<PIPE_V>();
            Cast(staging, conversion, RoundMode::CAST_RINT, length);
        } else {
            Cast(staging, source, RoundMode::CAST_RINT, length);
        }
        outputQueue_.EnQue<T>(staging);
        staging = outputQueue_.DeQue<T>();
        DataCopyPad(destination, staging, copy);
        outputQueue_.FreeTensor(staging);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECIN, 1> sourceQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> stagedSourceBuffer_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> secondAccumulatorBuffer_;
    TBuf<QuePosition::VECCALC> sourceAccumulatorBuffer_;
    TBuf<QuePosition::VECCALC> thirdAccumulatorBuffer_;
    TBuf<QuePosition::VECCALC> fourthAccumulatorBuffer_;
    TBuf<QuePosition::VECCALC> conversionBuffer_;
    GlobalTensor<T> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<T> source_;
    GlobalTensor<T> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint64_t innerTileCount_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

template <uint32_t GroupRows = 1,
          uint32_t MaxRowElements = kWholeInt32RowMaxElements>
class IndexAddWholeInt32Row {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(self),
                              totalLength_);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output),
                                totalLength_);
        pipe_.InitBuffer(accumulatorBuffer_,
                         GroupRows * MaxRowElements * sizeof(int32_t));
        pipe_.InitBuffer(sourceBuffer_,
                         MaxRowElements * sizeof(int32_t));
        pipe_.InitBuffer(indexBuffer_,
                         kWholeInt32RowMaxIndices * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> accumulator =
            accumulatorBuffer_.Get<int32_t>();
        LocalTensor<int32_t> stagedSource = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();
        LocalTensor<uint64_t> cachedIndexPairs =
            cachedIndices.template ReinterpretCast<uint64_t>();

        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
            0};
        const TEventID mte2ToScalarEvent =
            pipe_.AllocEventID<HardEvent::MTE2_S>();
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(mte2ToScalarEvent));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(mte2ToScalarEvent));
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(mte2ToScalarEvent);

        const TEventID mte2ToVectorEvent =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID vectorToMte2Event =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID vectorToMte3Event =
            pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID mte3ToMte2Event =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const uint32_t rowBytes =
            static_cast<uint32_t>(innerLength_) * sizeof(int32_t);
        const uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
        const uint32_t accumulatorRowBytes =
            MaxRowElements * static_cast<uint32_t>(sizeof(int32_t));
        const uint32_t accumulatorStrideBlocks =
            (accumulatorRowBytes - alignedRowBytes) / 32U;
        const DataCopyExtParams sourceRowCopy{1, rowBytes, 0, 0, 0};
        const uint32_t pairCount = indexLength_ / 2;
        const uint32_t groupsPerOuter =
            (axisLength_ + GroupRows - 1) / GroupRows;
        const uint64_t workCount = outerLength_ * groupsPerOuter;
        const uint64_t blockStride = GetBlockNum();

        for (uint64_t work = GetBlockIdx(); work < workCount;
             work += blockStride) {
            const uint64_t outer = work / groupsPerOuter;
            const uint32_t group =
                static_cast<uint32_t>(work % groupsPerOuter);
            const uint32_t firstAxis = group * GroupRows;
            const uint32_t rowCount =
                axisLength_ - firstAxis < GroupRows
                    ? axisLength_ - firstAxis
                    : GroupRows;
            const uint64_t selfOffset =
                (outer * axisLength_ + firstAxis) * innerLength_;
            const DataCopyExtParams selfCopy{
                static_cast<uint16_t>(rowCount), rowBytes, 0,
                accumulatorStrideBlocks, 0};
            DataCopyPad(accumulator, self_[selfOffset], selfCopy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(
                static_cast<event_t>(mte2ToVectorEvent));
            WaitFlag<HardEvent::MTE2_V>(
                static_cast<event_t>(mte2ToVectorEvent));

            bool sourceBufferInUse = false;
            uint64_t packedIndex =
                pairCount == 0 ? 0 : cachedIndexPairs.GetValue(0);
            for (uint32_t pair = 0; pair < pairCount; ++pair) {
                const uint64_t nextPackedIndex =
                    pair + 1 < pairCount
                        ? cachedIndexPairs.GetValue(pair + 1)
                        : 0;
#pragma unroll
                for (uint32_t packedLane = 0; packedLane < 2;
                     ++packedLane) {
                    const uint32_t update = pair * 2 + packedLane;
                    const uint32_t updateAxis =
                        static_cast<uint32_t>(packedIndex);
                    packedIndex >>= 32;
                    const uint32_t row = updateAxis - firstAxis;
                    if (row >= rowCount) {
                        continue;
                    }
                    AccumulateUpdate(
                        accumulator[row * MaxRowElements], stagedSource, outer,
                        update, sourceRowCopy, sourceBufferInUse,
                        mte2ToVectorEvent, vectorToMte2Event);
                }
                packedIndex = nextPackedIndex;
            }
            if ((indexLength_ & 1U) != 0) {
                const uint32_t updateAxis = static_cast<uint32_t>(
                    cachedIndices.GetValue(indexLength_ - 1));
                const uint32_t row = updateAxis - firstAxis;
                if (row < rowCount) {
                    AccumulateUpdate(
                        accumulator[row * MaxRowElements], stagedSource, outer,
                        indexLength_ - 1, sourceRowCopy, sourceBufferInUse,
                        mte2ToVectorEvent, vectorToMte2Event);
                }
            }
            if (sourceBufferInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(vectorToMte2Event));
            }

            SetFlag<HardEvent::V_MTE3>(
                static_cast<event_t>(vectorToMte3Event));
            WaitFlag<HardEvent::V_MTE3>(
                static_cast<event_t>(vectorToMte3Event));
            const DataCopyExtParams outputCopy{
                static_cast<uint16_t>(rowCount), rowBytes,
                accumulatorStrideBlocks, 0, 0};
            DataCopyPad(output_[selfOffset], accumulator, outputCopy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(mte3ToMte2Event));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(mte3ToMte2Event));
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(mte2ToVectorEvent);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(vectorToMte2Event);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(vectorToMte3Event);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(mte3ToMte2Event);
    }

private:
    __aicore__ inline void AccumulateUpdate(
        LocalTensor<int32_t> accumulator, LocalTensor<int32_t> stagedSource,
        uint64_t outer, uint32_t update, const DataCopyExtParams& rowCopy,
        bool& sourceBufferInUse, TEventID mte2ToVectorEvent,
        TEventID vectorToMte2Event)
    {
        if (sourceBufferInUse) {
            WaitFlag<HardEvent::V_MTE2>(
                static_cast<event_t>(vectorToMte2Event));
        }
        const uint64_t sourceOffset =
            (outer * indexLength_ + update) * innerLength_;
        DataCopyPad(stagedSource, source_[sourceOffset], rowCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(
            static_cast<event_t>(mte2ToVectorEvent));
        WaitFlag<HardEvent::MTE2_V>(
            static_cast<event_t>(mte2ToVectorEvent));
        Add(accumulator, accumulator, stagedSource,
            static_cast<uint32_t>(innerLength_));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(
            static_cast<event_t>(vectorToMte2Event));
        sourceBufferInUse = true;
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> sourceBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    GlobalTensor<int32_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<int32_t> source_;
    GlobalTensor<int32_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

class IndexAddFlatWholeInt32Rows {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(self),
                              totalLength_);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output),
                                totalLength_);
        pipe_.InitBuffer(
            accumulatorBuffer_,
            kFlatWholeInt32MaxRows * kFlatWholeInt32MaxElements *
                sizeof(int32_t));
        pipe_.InitBuffer(sourceBuffer_,
                         2U * kFlatWholeInt32MaxElements * sizeof(int32_t));
        pipe_.InitBuffer(indexBuffer_,
                         kWholeInt32RowMaxIndices * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> accumulator =
            accumulatorBuffer_.Get<int32_t>();
        LocalTensor<int32_t> sourceStorage = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> firstSource = sourceStorage;
        LocalTensor<int32_t> secondSource =
            sourceStorage[kFlatWholeInt32MaxElements];
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();
        LocalTensor<uint64_t> cachedIndexPairs =
            cachedIndices.template ReinterpretCast<uint64_t>();

        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
            0};
        const TEventID indexReady = pipe_.AllocEventID<HardEvent::MTE2_S>();
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);

        const TEventID inputReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID firstSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID secondSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID firstSourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID secondSourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady =
            pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const uint32_t rowBytes =
            static_cast<uint32_t>(innerLength_) * sizeof(int32_t);
        const uint32_t alignedRowBytes = (rowBytes + 31U) & ~31U;
        const uint32_t accumulatorRowBytes =
            kFlatWholeInt32MaxElements * sizeof(int32_t);
        const uint32_t accumulatorStrideBlocks =
            (accumulatorRowBytes - alignedRowBytes) / 32U;
        const DataCopyExtParams sourceRowCopy{1, rowBytes, 0, 0, 0};
        const uint64_t flatRowCount = outerLength_ * axisLength_;
        const uint64_t blockIdx = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        uint64_t flatRow = flatRowCount * blockIdx / blockNum;
        const uint64_t pastFlatRow =
            flatRowCount * (blockIdx + 1U) / blockNum;

        while (flatRow < pastFlatRow) {
            const uint64_t outer = flatRow / axisLength_;
            const uint32_t firstAxis =
                static_cast<uint32_t>(flatRow % axisLength_);
            const uint32_t rowsUntilOuterEnd = axisLength_ - firstAxis;
            const uint64_t rowsUntilBlockEnd = pastFlatRow - flatRow;
            const uint32_t rowCount = static_cast<uint32_t>(
                rowsUntilBlockEnd < rowsUntilOuterEnd
                    ? rowsUntilBlockEnd
                    : rowsUntilOuterEnd);
            const uint64_t selfOffset = flatRow * innerLength_;
            const DataCopyExtParams selfCopy{
                static_cast<uint16_t>(rowCount), rowBytes, 0,
                accumulatorStrideBlocks, 0};
            DataCopyPad(accumulator, self_[selfOffset], selfCopy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(inputReady));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(inputReady));

            uint32_t cursor = 0U;
            uint64_t packedIndex = 0U;
            uint32_t update = 0U;
            uint32_t row = 0U;
            bool hasUpdate = FindNextUpdate(
                cachedIndexPairs, cursor, packedIndex, firstAxis, rowCount,
                update, row);
            uint32_t sourceGeneration = 0U;
            bool firstSourceInUse = false;
            bool secondSourceInUse = false;
            if (hasUpdate) {
                CopySource(firstSource, outer, update, sourceRowCopy,
                           firstSourceReady);
            }
            while (hasUpdate) {
                uint32_t nextUpdate = 0U;
                uint32_t nextRow = 0U;
                const bool hasNextUpdate = FindNextUpdate(
                    cachedIndexPairs, cursor, packedIndex, firstAxis,
                    rowCount, nextUpdate, nextRow);
                if (sourceGeneration == 0U) {
                    WaitFlag<HardEvent::MTE2_V>(
                        static_cast<event_t>(firstSourceReady));
                } else {
                    WaitFlag<HardEvent::MTE2_V>(
                        static_cast<event_t>(secondSourceReady));
                }

                const uint32_t nextGeneration = sourceGeneration ^ 1U;
                if (hasNextUpdate) {
                    if (nextGeneration == 0U) {
                        if (firstSourceInUse) {
                            WaitFlag<HardEvent::V_MTE2>(
                                static_cast<event_t>(firstSourceReusable));
                            firstSourceInUse = false;
                        }
                        CopySource(firstSource, outer, nextUpdate,
                                   sourceRowCopy, firstSourceReady);
                    } else {
                        if (secondSourceInUse) {
                            WaitFlag<HardEvent::V_MTE2>(
                                static_cast<event_t>(secondSourceReusable));
                            secondSourceInUse = false;
                        }
                        CopySource(secondSource, outer, nextUpdate,
                                   sourceRowCopy, secondSourceReady);
                    }
                }

                LocalTensor<int32_t> activeSource =
                    sourceGeneration == 0U ? firstSource : secondSource;
                Add(accumulator[row * kFlatWholeInt32MaxElements],
                    accumulator[row * kFlatWholeInt32MaxElements],
                    activeSource, static_cast<uint32_t>(innerLength_));
                PipeBarrier<PIPE_V>();
                if (sourceGeneration == 0U) {
                    SetFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(firstSourceReusable));
                    firstSourceInUse = true;
                } else {
                    SetFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(secondSourceReusable));
                    secondSourceInUse = true;
                }
                sourceGeneration = nextGeneration;
                update = nextUpdate;
                row = nextRow;
                hasUpdate = hasNextUpdate;
            }
            if (firstSourceInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(firstSourceReusable));
            }
            if (secondSourceInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(secondSourceReusable));
            }

            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            const DataCopyExtParams outputCopy{
                static_cast<uint16_t>(rowCount), rowBytes,
                accumulatorStrideBlocks, 0, 0};
            DataCopyPad(output_[selfOffset], accumulator, outputCopy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
            flatRow += rowCount;
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(firstSourceReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(secondSourceReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(firstSourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(secondSourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
    }

private:
    __aicore__ inline bool FindNextUpdate(
        LocalTensor<uint64_t> cachedIndexPairs, uint32_t& cursor,
        uint64_t& packedIndex, uint32_t firstAxis, uint32_t rowCount,
        uint32_t& update, uint32_t& row)
    {
        while (cursor < indexLength_) {
            const uint32_t candidate = cursor++;
            if ((candidate & 1U) == 0U) {
                packedIndex = cachedIndexPairs.GetValue(candidate / 2U);
            }
            const uint32_t axis = static_cast<uint32_t>(packedIndex);
            packedIndex >>= 32U;
            const uint32_t candidateRow = axis - firstAxis;
            if (candidateRow < rowCount) {
                update = candidate;
                row = candidateRow;
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void CopySource(
        LocalTensor<int32_t> destination, uint64_t outer, uint32_t update,
        const DataCopyExtParams& rowCopy, TEventID sourceReady)
    {
        const uint64_t sourceOffset =
            (outer * indexLength_ + update) * innerLength_;
        DataCopyPad(destination, source_[sourceOffset], rowCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(sourceReady));
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> sourceBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    GlobalTensor<int32_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<int32_t> source_;
    GlobalTensor<int32_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

class IndexAddColumnOwnedInt32 {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(self),
                              totalLength_);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output),
                                totalLength_);

        const uint64_t flatColumnCount = outerLength_ * innerLength_;
        const uint32_t blockNum = GetBlockNum();
        maxColumnLength_ = static_cast<uint32_t>(
            (flatColumnCount + blockNum - 1U) / blockNum);
        maxColumnLength_ = (maxColumnLength_ + 7U) & ~7U;
        const uint32_t alignedIndexBytes =
            (indexLength_ * static_cast<uint32_t>(sizeof(int32_t)) + 31U) &
            ~31U;
        pipe_.InitBuffer(
            accumulatorBuffer_,
            axisLength_ * maxColumnLength_ * sizeof(int32_t));
        pipe_.InitBuffer(
            sourceBuffer_,
            2U * kColumnOwnedInt32SourceBatchRows * maxColumnLength_ *
                sizeof(int32_t));
        pipe_.InitBuffer(indexBuffer_, alignedIndexBytes);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> accumulator =
            accumulatorBuffer_.Get<int32_t>();
        LocalTensor<int32_t> sourceStorage =
            sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> firstSource = sourceStorage;
        LocalTensor<int32_t> secondSource =
            sourceStorage[kColumnOwnedInt32SourceBatchRows *
                          maxColumnLength_];
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();

        const TEventID indexReady =
            pipe_.AllocEventID<HardEvent::MTE2_S>();
        const TEventID selfReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID firstSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID secondSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID firstSourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID secondSourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady =
            pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();

        const DataCopyExtParams indexCopy{
            1U,
            indexLength_ * static_cast<uint32_t>(sizeof(int32_t)),
            0U, 0U, 0U};
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));

        const uint64_t flatColumnCount = outerLength_ * innerLength_;
        const uint64_t blockIdx = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        uint64_t flatColumn = flatColumnCount * blockIdx / blockNum;
        const uint64_t pastFlatColumn =
            flatColumnCount * (blockIdx + 1U) / blockNum;

        while (flatColumn < pastFlatColumn) {
            const uint64_t outer = flatColumn / innerLength_;
            const uint32_t firstColumn = static_cast<uint32_t>(
                flatColumn - outer * innerLength_);
            const uint64_t remainingBlockColumns =
                pastFlatColumn - flatColumn;
            const uint32_t remainingOuterColumns =
                static_cast<uint32_t>(innerLength_) - firstColumn;
            const uint32_t columnCount = static_cast<uint32_t>(
                remainingBlockColumns < remainingOuterColumns
                    ? remainingBlockColumns
                    : remainingOuterColumns);
            const uint32_t columnBytes =
                columnCount * static_cast<uint32_t>(sizeof(int32_t));

            const uint64_t selfOffset =
                (outer * axisLength_) * innerLength_ + firstColumn;
            const DataCopyExtParams rowCopy{
                1U, columnBytes, 0U, 0U, 0U};
            for (uint32_t axis = 0U; axis < axisLength_; ++axis) {
                DataCopyPad(accumulator[axis * maxColumnLength_],
                            self_[selfOffset + axis * innerLength_],
                            rowCopy,
                            DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            }
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(selfReady));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(selfReady));

            uint32_t currentUpdate = 0U;
            uint32_t currentCount =
                indexLength_ < kColumnOwnedInt32SourceBatchRows
                    ? indexLength_
                    : kColumnOwnedInt32SourceBatchRows;
            bool useFirst = true;
            bool firstInUse = false;
            bool secondInUse = false;
            if (currentCount != 0U) {
                CopySourceBatch(firstSource, outer, firstColumn,
                                currentUpdate, currentCount, columnBytes,
                                firstSourceReady);
                firstInUse = true;
            }

            while (currentCount != 0U) {
                const TEventID currentReady =
                    useFirst ? firstSourceReady : secondSourceReady;
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(currentReady));

                const uint32_t nextUpdate = currentUpdate + currentCount;
                const uint32_t remainingUpdates = indexLength_ - nextUpdate;
                const uint32_t nextCount =
                    remainingUpdates < kColumnOwnedInt32SourceBatchRows
                        ? remainingUpdates
                        : kColumnOwnedInt32SourceBatchRows;
                if (nextCount != 0U) {
                    if (useFirst) {
                        if (secondInUse) {
                            WaitFlag<HardEvent::V_MTE2>(
                                static_cast<event_t>(secondSourceReusable));
                            secondInUse = false;
                        }
                        CopySourceBatch(
                            secondSource, outer, firstColumn, nextUpdate,
                            nextCount, columnBytes, secondSourceReady);
                        secondInUse = true;
                    } else {
                        if (firstInUse) {
                            WaitFlag<HardEvent::V_MTE2>(
                                static_cast<event_t>(firstSourceReusable));
                            firstInUse = false;
                        }
                        CopySourceBatch(
                            firstSource, outer, firstColumn, nextUpdate,
                            nextCount, columnBytes, firstSourceReady);
                        firstInUse = true;
                    }
                }

                LocalTensor<int32_t> currentSource =
                    useFirst ? firstSource : secondSource;
                for (uint32_t localUpdate = 0U;
                     localUpdate < currentCount; ++localUpdate) {
                    const uint32_t axis = static_cast<uint32_t>(
                        cachedIndices.GetValue(currentUpdate + localUpdate));
                    Add(accumulator[axis * maxColumnLength_],
                        accumulator[axis * maxColumnLength_],
                        currentSource[localUpdate * maxColumnLength_],
                        columnCount);
                    PipeBarrier<PIPE_V>();
                }
                const TEventID currentReusable =
                    useFirst ? firstSourceReusable : secondSourceReusable;
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(currentReusable));

                currentUpdate = nextUpdate;
                currentCount = nextCount;
                useFirst = !useFirst;
            }
            if (firstInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(firstSourceReusable));
            }
            if (secondInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(secondSourceReusable));
            }

            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            for (uint32_t axis = 0U; axis < axisLength_; ++axis) {
                DataCopyPad(output_[selfOffset + axis * innerLength_],
                            accumulator[axis * maxColumnLength_], rowCopy);
            }
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));

            flatColumn += columnCount;
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(selfReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(firstSourceReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(secondSourceReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(firstSourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(secondSourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
    }

private:
    __aicore__ inline void CopySourceBatch(
        LocalTensor<int32_t> destination, uint64_t outer,
        uint32_t firstColumn, uint32_t firstUpdate, uint32_t updateCount,
        uint32_t columnBytes, TEventID sourceReady)
    {
        const uint32_t alignedColumnBytes = (columnBytes + 31U) & ~31U;
        // C220 GM-to-UB align moves encode the GM gap in bytes and the UB
        // gap in 32-byte blocks.  They deliberately use different units,
        // so consecutive source rows do not need a parity split merely
        // because their byte pitch is not 32-byte aligned.
        const uint32_t sourceStrideBytes =
            (static_cast<uint32_t>(innerLength_) *
                 static_cast<uint32_t>(sizeof(int32_t)) -
             alignedColumnBytes);
        const uint32_t destinationStrideBlocks =
            (maxColumnLength_ *
                 static_cast<uint32_t>(sizeof(int32_t)) -
             alignedColumnBytes) /
            32U;
        const bool protectFinalSourceRow =
            outer + 1U == outerLength_ &&
            firstUpdate + updateCount == indexLength_ &&
            firstColumn * static_cast<uint32_t>(sizeof(int32_t)) +
                    alignedColumnBytes >
                static_cast<uint32_t>(innerLength_) *
                    static_cast<uint32_t>(sizeof(int32_t));
        const uint32_t finalLocalUpdate = updateCount - 1U;
        const uint32_t copyCount =
            updateCount - (protectFinalSourceRow ? 1U : 0U);
        if (copyCount != 0U) {
            const uint64_t sourceOffset =
                (outer * indexLength_ + firstUpdate) *
                    innerLength_ +
                firstColumn;
            const DataCopyExtParams sourceCopy{
                static_cast<uint16_t>(copyCount), alignedColumnBytes,
                sourceStrideBytes, destinationStrideBlocks, 0U};
            DataCopyPad(destination, source_[sourceOffset], sourceCopy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        if (protectFinalSourceRow) {
            const uint64_t finalSourceOffset =
                (outer * indexLength_ + firstUpdate + finalLocalUpdate) *
                    innerLength_ +
                firstColumn;
            const DataCopyExtParams finalCopy{
                1U, columnBytes, 0U, 0U, 0U};
            DataCopyPad(destination[finalLocalUpdate * maxColumnLength_],
                        source_[finalSourceOffset], finalCopy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(sourceReady));
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> sourceBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    GlobalTensor<int32_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<int32_t> source_;
    GlobalTensor<int32_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
    uint32_t maxColumnLength_;
};

template <bool CachePhaseZeroAxis = false>
class IndexAddCoalescedAtomicInt32 {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output, GM_ADDR workspace,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(self),
                              totalLength_);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(source));
        if constexpr (CachePhaseZeroAxis) {
            // This path streams a source tensor much larger than the output
            // working set exactly once. Bypass L2 for those read-only source
            // loads so the interleaved atomic output traffic can retain its
            // cache residency. The policy is attached only to source_; self,
            // index, and output keep the normal cache mode.
            source_.SetL2CacheHint<CacheRwMode::READ>(
                CacheMode::CACHE_MODE_DISABLE);
        }
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output),
                                totalLength_);
        SetSysWorkspace(workspace);

        sourceBufferElements_ =
            kCoalescedAtomicInt32SourceChunkRows *
            static_cast<uint32_t>(innerLength_);
        const uint32_t alignedIndexBytes =
            (indexLength_ * static_cast<uint32_t>(sizeof(int32_t)) + 31U) &
            ~31U;
        pipe_.InitBuffer(firstSourceBuffer_,
                         sourceBufferElements_ * sizeof(int32_t));
        pipe_.InitBuffer(secondSourceBuffer_,
                         sourceBufferElements_ * sizeof(int32_t));
        pipe_.InitBuffer(indexBuffer_, alignedIndexBytes);
        pipe_.InitBuffer(headScratchBuffer_, kAtomicInt32HeadScratchBytes);
        pipe_.InitBuffer(gatherOffsetBuffer_, kAtomicInt32OffsetBytes);
        if constexpr (CachePhaseZeroAxis) {
            alignedRowElements_ =
                (static_cast<uint32_t>(innerLength_) + 7U) & ~7U;
            pipe_.InitBuffer(cachedPhaseZeroBuffer_,
                             alignedRowElements_ * sizeof(int32_t));
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> firstSource =
            firstSourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> secondSource =
            secondSourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();
        LocalTensor<int32_t> firstHeadScratch =
            headScratchBuffer_.Get<int32_t>();
        LocalTensor<int32_t> secondHeadScratch = firstHeadScratch[8U];
        LocalTensor<int32_t> gatherOffsetValues =
            gatherOffsetBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> gatherOffsets =
            gatherOffsetValues.template ReinterpretCast<uint32_t>();
        LocalTensor<int32_t> cachedPhaseZero;
        if constexpr (CachePhaseZeroAxis) {
            cachedPhaseZero = cachedPhaseZeroBuffer_.Get<int32_t>();
        }

        const TEventID indexReady =
            pipe_.AllocEventID<HardEvent::MTE2_S>();
        const TEventID selfReady =
            pipe_.AllocEventID<HardEvent::MTE2_MTE3>();
        const TEventID firstSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID secondSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID firstSourceReusable =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const TEventID secondSourceReusable =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const TEventID vectorToStore =
            pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID sourceStoreComplete =
            pipe_.AllocEventID<HardEvent::MTE3_V>();

        ArithProgression(gatherOffsetValues, static_cast<int32_t>(0),
                         static_cast<int32_t>(sizeof(int32_t)), 8U);

        const DataCopyExtParams indexCopy{
            1U,
            indexLength_ * static_cast<uint32_t>(sizeof(int32_t)),
            0U, 0U, 0U};
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        const uint32_t rowBytes =
            static_cast<uint32_t>(innerLength_) * sizeof(int32_t);
        const DataCopyExtParams rowCopy{1U, rowBytes, 0U, 0U, 0U};

        __builtin_cce_set_atomic_none();
        const uint64_t selfRows = outerLength_ * axisLength_;
        const uint64_t selfRowStart = selfRows * blockIdx / blockNum;
        const uint64_t selfRowEnd =
            selfRows * (blockIdx + 1U) / blockNum;
        for (uint64_t firstRow = selfRowStart; firstRow < selfRowEnd;) {
            const uint32_t rowCount = static_cast<uint32_t>(
                selfRowEnd - firstRow <
                        kCoalescedAtomicInt32SourceChunkRows
                    ? selfRowEnd - firstRow
                    : kCoalescedAtomicInt32SourceChunkRows);
            const DataCopyExtParams copy{
                1U, rowCount * rowBytes, 0U, 0U, 0U};
            const uint64_t offset = firstRow * innerLength_;
            DataCopyPad(firstSource, self_[offset], copy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_MTE3>(
                static_cast<event_t>(selfReady));
            WaitFlag<HardEvent::MTE2_MTE3>(
                static_cast<event_t>(selfReady));
            DataCopyPad(output_[offset], firstSource, copy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(firstSourceReusable));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(firstSourceReusable));
            firstRow += rowCount;
        }

        const uint64_t sourceRows = outerLength_ * indexLength_;
        const bool usePairedSourceOwnership =
            blockNum >= 2U && (blockNum & 1U) == 0U;
        const uint64_t sourceRowStart =
            sourceRows * blockIdx / blockNum;
        const uint64_t sourceRowEnd =
            sourceRows * (blockIdx + 1U) / blockNum;
        const uint64_t sourceGenerationCount =
            (sourceRows + kCoalescedAtomicInt32SourceChunkRows - 1U) /
            kCoalescedAtomicInt32SourceChunkRows;
        const uint32_t sourcePairCount = blockNum >> 1U;
        const uint32_t sourcePair = blockIdx >> 1U;
        const uint32_t sourcePairLane = blockIdx & 1U;
        const uint64_t sourcePairGenerationStart =
            usePairedSourceOwnership
                ? sourceGenerationCount * sourcePair / sourcePairCount
                : 0U;
        const uint64_t sourcePairGenerationEnd =
            usePairedSourceOwnership
                ? sourceGenerationCount * (sourcePair + 1U) /
                      sourcePairCount
                : 0U;
        uint32_t currentPairLane = sourcePairLane;
        uint64_t currentGeneration =
            sourcePairGenerationStart + sourcePairLane;
        uint64_t currentStart =
            usePairedSourceOwnership
                ? currentGeneration * kCoalescedAtomicInt32SourceChunkRows
                : sourceRowStart;
        const uint64_t currentRemainingRows =
            usePairedSourceOwnership
                ? (currentGeneration < sourcePairGenerationEnd
                       ? sourceRows - currentStart
                       : 0U)
                : sourceRowEnd - currentStart;
        uint32_t currentCount = static_cast<uint32_t>(
            currentRemainingRows < kCoalescedAtomicInt32SourceChunkRows
                ? currentRemainingRows
                : kCoalescedAtomicInt32SourceChunkRows);
        bool useFirst = true;
        bool firstInUse = false;
        bool secondInUse = false;
        if (currentCount != 0U) {
            CopySourceGeneration(firstSource, currentStart, currentCount,
                                 rowBytes);
            SetFlag<HardEvent::MTE2_V>(
                static_cast<event_t>(firstSourceReady));
            firstInUse = true;
        }

        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        SyncAll();

        const uint32_t cachedAxis =
            CachePhaseZeroAxis ? blockIdx % axisLength_ : 0U;
        uint64_t cachedOuter = ~static_cast<uint64_t>(0U);
        uint64_t cachedOutputOffset = 0U;
        bool cachedPhaseZeroUsed = false;
        if constexpr (CachePhaseZeroAxis) {
            Duplicate(cachedPhaseZero, static_cast<int32_t>(0),
                      alignedRowElements_);
        }

        __builtin_cce_set_atomic_s32();
        __builtin_cce_set_atomic_add();
        while (currentCount != 0U) {
            const TEventID currentReady =
                useFirst ? firstSourceReady : secondSourceReady;
            WaitFlag<HardEvent::MTE2_V>(
                static_cast<event_t>(currentReady));

            const uint64_t nextGeneration =
                currentGeneration + (currentPairLane == 0U ? 3U : 1U);
            const uint64_t nextStart =
                usePairedSourceOwnership
                    ? nextGeneration *
                          kCoalescedAtomicInt32SourceChunkRows
                    : currentStart + currentCount;
            const uint64_t nextRemainingRows =
                usePairedSourceOwnership
                    ? (nextGeneration < sourcePairGenerationEnd
                           ? sourceRows - nextStart
                           : 0U)
                    : sourceRowEnd - nextStart;
            const uint32_t nextCount = static_cast<uint32_t>(
                nextRemainingRows < kCoalescedAtomicInt32SourceChunkRows
                    ? nextRemainingRows
                    : kCoalescedAtomicInt32SourceChunkRows);
            if (nextCount != 0U) {
                if (useFirst) {
                    if (secondInUse) {
                        WaitFlag<HardEvent::MTE3_MTE2>(
                            static_cast<event_t>(secondSourceReusable));
                        secondInUse = false;
                    }
                    CopySourceGeneration(secondSource, nextStart, nextCount,
                                         rowBytes);
                    SetFlag<HardEvent::MTE2_V>(
                        static_cast<event_t>(secondSourceReady));
                    secondInUse = true;
                } else {
                    if (firstInUse) {
                        WaitFlag<HardEvent::MTE3_MTE2>(
                            static_cast<event_t>(firstSourceReusable));
                        firstInUse = false;
                    }
                    CopySourceGeneration(firstSource, nextStart, nextCount,
                                         rowBytes);
                    SetFlag<HardEvent::MTE2_V>(
                        static_cast<event_t>(firstSourceReady));
                    firstInUse = true;
                }
            }

            LocalTensor<int32_t> currentSource =
                useFirst ? firstSource : secondSource;
            LocalTensor<int32_t> currentHeadScratch =
                useFirst ? firstHeadScratch : secondHeadScratch;
            const uint64_t generationOuter =
                currentStart / indexLength_;
            if constexpr (CachePhaseZeroAxis) {
                if (generationOuter != cachedOuter) {
                    if (cachedOuter != ~static_cast<uint64_t>(0U)) {
                        FlushCachedPhaseZero(
                            cachedPhaseZero, cachedOutputOffset,
                            cachedPhaseZeroUsed, vectorToStore,
                            sourceStoreComplete, rowBytes);
                        Duplicate(cachedPhaseZero,
                                  static_cast<int32_t>(0),
                                  alignedRowElements_);
                    }
                    cachedOuter = generationOuter;
                    cachedOutputOffset =
                        (cachedOuter * axisLength_ + cachedAxis) *
                        innerLength_;
                    cachedPhaseZeroUsed = false;
                }
            }
            uint64_t outputOffsets[kCoalescedAtomicInt32SourceChunkRows];
            bool representatives[kCoalescedAtomicInt32SourceChunkRows];
            for (uint32_t localRow = 0U; localRow < currentCount;
                 ++localRow) {
                const uint64_t flatSourceRow = currentStart + localRow;
                const uint64_t outer = flatSourceRow / indexLength_;
                const uint32_t update = static_cast<uint32_t>(
                    flatSourceRow - outer * indexLength_);
                const uint32_t axis = static_cast<uint32_t>(
                    cachedIndices.GetValue(update));
                outputOffsets[localRow] =
                    (outer * axisLength_ + axis) * innerLength_;
                representatives[localRow] = true;
            }

            // Eight consecutive rows occupy an integral number of 32-byte
            // blocks when innerLength % 8 == 4.  Their UB starts alternate
            // between phases 0 and 16 bytes, so merge equal destinations
            // within each phase without introducing a strided GM-to-UB copy.
            for (uint32_t localRow = 0U; localRow < currentCount;
                 localRow += 2U) {
                for (uint32_t previous = 0U; previous < localRow;
                     previous += 2U) {
                    if (representatives[previous] &&
                        outputOffsets[previous] ==
                            outputOffsets[localRow]) {
                        Add(currentSource[previous * innerLength_],
                            currentSource[previous * innerLength_],
                            currentSource[localRow * innerLength_],
                            static_cast<uint32_t>(innerLength_));
                        representatives[localRow] = false;
                        break;
                    }
                }
                if constexpr (CachePhaseZeroAxis) {
                    if (representatives[localRow] &&
                        outputOffsets[localRow] == cachedOutputOffset) {
                        Add(cachedPhaseZero, cachedPhaseZero,
                            currentSource[localRow * innerLength_],
                            static_cast<uint32_t>(innerLength_));
                        representatives[localRow] = false;
                        cachedPhaseZeroUsed = true;
                    }
                }
            }

            SetFlag<HardEvent::V_MTE3>(
                static_cast<event_t>(vectorToStore));
            WaitFlag<HardEvent::V_MTE3>(
                static_cast<event_t>(vectorToStore));
            for (uint32_t localRow = 0U; localRow < currentCount;
                 localRow += 2U) {
                if (!representatives[localRow]) {
                    continue;
                }
                const uint64_t outputOffset = outputOffsets[localRow];
                DataCopyPad(output_[outputOffset],
                            currentSource[localRow * innerLength_], rowCopy);
            }

            if (currentCount > 1U) {
                SetFlag<HardEvent::MTE3_V>(
                    static_cast<event_t>(sourceStoreComplete));
                WaitFlag<HardEvent::MTE3_V>(
                    static_cast<event_t>(sourceStoreComplete));

                // Each phase-16 row is represented by an aligned 16-byte
                // zero prefix followed by its exact row.  The prefix aliases
                // the preceding phase-0 row tail, which is why those stores
                // are completed before the prefixes are cleared.
                for (uint32_t localRow = 1U; localRow < currentCount;
                     localRow += 2U) {
                    const uint32_t prefixStart =
                        localRow * static_cast<uint32_t>(innerLength_) - 4U;
                    Duplicate(currentSource[prefixStart],
                              static_cast<int32_t>(0), 4U);
                }

                const uint32_t prefixedLength =
                    static_cast<uint32_t>(innerLength_) + 4U;
                for (uint32_t localRow = 1U; localRow < currentCount;
                     localRow += 2U) {
                    for (uint32_t previous = 1U; previous < localRow;
                         previous += 2U) {
                        if (representatives[previous] &&
                            outputOffsets[previous] ==
                                outputOffsets[localRow]) {
                            const uint32_t previousPrefix =
                                previous *
                                    static_cast<uint32_t>(innerLength_) -
                                4U;
                            const uint32_t localPrefix =
                                localRow *
                                    static_cast<uint32_t>(innerLength_) -
                                4U;
                            Add(currentSource[previousPrefix],
                                currentSource[previousPrefix],
                                currentSource[localPrefix], prefixedLength);
                            representatives[localRow] = false;
                            break;
                        }
                    }
                }

                for (uint32_t localRow = 1U; localRow < currentCount;
                     localRow += 2U) {
                    if (representatives[localRow] &&
                        outputOffsets[localRow] == 0U) {
                        const uint32_t prefixStart =
                            localRow *
                                static_cast<uint32_t>(innerLength_) -
                            4U;
                        Gather(currentHeadScratch,
                               currentSource[prefixStart],
                               gatherOffsets, kAtomicInt32PhaseBytes, 4U);
                    }
                }

                SetFlag<HardEvent::V_MTE3>(
                    static_cast<event_t>(vectorToStore));
                WaitFlag<HardEvent::V_MTE3>(
                    static_cast<event_t>(vectorToStore));

                for (uint32_t localRow = 1U; localRow < currentCount;
                     localRow += 2U) {
                    if (!representatives[localRow]) {
                        continue;
                    }
                    const uint64_t outputOffset = outputOffsets[localRow];
                    const uint32_t packedRowStart =
                        localRow * static_cast<uint32_t>(innerLength_);
                    const uint32_t prefixStart = packedRowStart - 4U;
                    if (outputOffset == 0U) {
                        const DataCopyExtParams headCopy{
                            1U, kAtomicInt32PhaseBytes, 0U, 0U, 0U};
                        const DataCopyExtParams tailCopy{
                            1U, rowBytes - kAtomicInt32PhaseBytes,
                            0U, 0U, 0U};
                        DataCopyPad(output_[0U], currentHeadScratch,
                                    headCopy);
                        DataCopyPad(output_[4U],
                                    currentSource[packedRowStart + 4U],
                                    tailCopy);
                    } else {
                        const DataCopyExtParams prefixedCopy{
                            1U, rowBytes + kAtomicInt32PhaseBytes,
                            0U, 0U, 0U};
                        DataCopyPad(output_[outputOffset - 4U],
                                    currentSource[prefixStart],
                                    prefixedCopy);
                    }
                }
            }
            const TEventID currentReusable =
                useFirst ? firstSourceReusable : secondSourceReusable;
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(currentReusable));

            currentStart = nextStart;
            currentCount = nextCount;
            currentGeneration = nextGeneration;
            currentPairLane ^= 1U;
            useFirst = !useFirst;
        }
        if constexpr (CachePhaseZeroAxis) {
            FlushCachedPhaseZero(
                cachedPhaseZero, cachedOutputOffset,
                cachedPhaseZeroUsed, vectorToStore,
                sourceStoreComplete, rowBytes);
        }
        if (firstInUse) {
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(firstSourceReusable));
        }
        if (secondInUse) {
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(secondSourceReusable));
        }
        __builtin_cce_set_atomic_none();

        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_MTE3>(selfReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(firstSourceReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(secondSourceReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(firstSourceReusable);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(secondSourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(vectorToStore);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(sourceStoreComplete);
    }

private:
    __aicore__ inline void FlushCachedPhaseZero(
        LocalTensor<int32_t> cachedPhaseZero, uint64_t outputOffset,
        bool used, TEventID vectorToStore, TEventID storeComplete,
        uint32_t rowBytes)
    {
        if (!used) {
            return;
        }
        SetFlag<HardEvent::V_MTE3>(
            static_cast<event_t>(vectorToStore));
        WaitFlag<HardEvent::V_MTE3>(
            static_cast<event_t>(vectorToStore));
        const DataCopyExtParams rowCopy{
            1U, rowBytes, 0U, 0U, 0U};
        DataCopyPad(output_[outputOffset], cachedPhaseZero, rowCopy);
        SetFlag<HardEvent::MTE3_V>(
            static_cast<event_t>(storeComplete));
        WaitFlag<HardEvent::MTE3_V>(
            static_cast<event_t>(storeComplete));
    }

    __aicore__ inline void CopySourceGeneration(
        LocalTensor<int32_t> destination, uint64_t firstRow,
        uint32_t rowCount, uint32_t rowBytes)
    {
        const DataCopyExtParams copy{
            1U, rowCount * rowBytes, 0U, 0U, 0U};
        DataCopyPad(destination, source_[firstRow * innerLength_], copy,
                    DataCopyPadExtParams<int32_t>{false, 0U, 0U, 0});
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> firstSourceBuffer_;
    TBuf<QuePosition::VECCALC> secondSourceBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    TBuf<QuePosition::VECCALC> headScratchBuffer_;
    TBuf<QuePosition::VECCALC> gatherOffsetBuffer_;
    TBuf<QuePosition::VECCALC> cachedPhaseZeroBuffer_;
    GlobalTensor<int32_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<int32_t> source_;
    GlobalTensor<int32_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
    uint32_t sourceBufferElements_;
    uint32_t alignedRowElements_;
};

class IndexAddAtomicInt32 {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output, GM_ADDR workspace,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(self),
                              totalLength_);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output),
                                totalLength_);
        SetSysWorkspace(workspace);
        pipe_.InitBuffer(
            firstSourceBuffer_,
            kAtomicInt32SourceChunkRows * kAtomicInt32MaxInnerLength *
                sizeof(int32_t));
        pipe_.InitBuffer(
            secondSourceBuffer_,
            kAtomicInt32SourceChunkRows * kAtomicInt32MaxInnerLength *
                sizeof(int32_t));
        pipe_.InitBuffer(indexBuffer_,
                         kWholeInt32RowMaxIndices * sizeof(int32_t));
        pipe_.InitBuffer(firstHeadScratchBuffer_,
                         kAtomicInt32HeadScratchBytes);
        pipe_.InitBuffer(secondHeadScratchBuffer_,
                         kAtomicInt32HeadScratchBytes);
        pipe_.InitBuffer(gatherOffsetBuffer_, kAtomicInt32OffsetBytes);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> firstSource =
            firstSourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> secondSource =
            secondSourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();
        LocalTensor<int32_t> firstHeadScratch =
            firstHeadScratchBuffer_.Get<int32_t>();
        LocalTensor<int32_t> secondHeadScratch =
            secondHeadScratchBuffer_.Get<int32_t>();
        LocalTensor<int32_t> gatherOffsetValues =
            gatherOffsetBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> gatherOffsets =
            gatherOffsetValues.template ReinterpretCast<uint32_t>();
        const TEventID indexReady = pipe_.AllocEventID<HardEvent::MTE2_S>();
        const TEventID firstSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_MTE3>();
        const TEventID secondSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_MTE3>();
        const TEventID firstSourceReusable =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const TEventID secondSourceReusable =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const TEventID sourceStoreComplete =
            pipe_.AllocEventID<HardEvent::MTE3_V>();
        const TEventID sourceVectorComplete =
            pipe_.AllocEventID<HardEvent::V_MTE3>();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        const uint32_t rowBytes =
            static_cast<uint32_t>(innerLength_) * sizeof(int32_t);
        const uint32_t alignedInnerLength =
            (static_cast<uint32_t>(innerLength_) + 7U) & ~7U;
        const bool usePackedSource =
            (static_cast<uint32_t>(innerLength_) & 7U) == 4U;

        if (usePackedSource) {
            ArithProgression(gatherOffsetValues, static_cast<int32_t>(0),
                             static_cast<int32_t>(sizeof(int32_t)), 8U);
        }

        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));

        __builtin_cce_set_atomic_none();
        const uint64_t selfRows = outerLength_ * axisLength_;
        const uint64_t selfRowStart = selfRows * blockIdx / blockNum;
        const uint64_t selfRowEnd = selfRows * (blockIdx + 1U) / blockNum;
        for (uint64_t firstRow = selfRowStart; firstRow < selfRowEnd;) {
            const uint32_t rowCount = static_cast<uint32_t>(
                selfRowEnd - firstRow < kAtomicInt32SourceChunkRows
                    ? selfRowEnd - firstRow
                    : kAtomicInt32SourceChunkRows);
            const DataCopyExtParams copy{
                static_cast<uint16_t>(rowCount), rowBytes, 0, 0, 0};
            const uint64_t offset = firstRow * innerLength_;
            DataCopyPad(firstSource, self_[offset], copy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_MTE3>(
                static_cast<event_t>(firstSourceReady));
            WaitFlag<HardEvent::MTE2_MTE3>(
                static_cast<event_t>(firstSourceReady));
            DataCopyPad(output_[offset], firstSource, copy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(firstSourceReusable));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(firstSourceReusable));
            firstRow += rowCount;
        }

        const uint64_t sourceRows = outerLength_ * indexLength_;
        const bool usePairedSourceOwnership =
            blockNum >= 2U && (blockNum & 1U) == 0U;
        const uint64_t sourceRowStart = sourceRows * blockIdx / blockNum;
        const uint64_t sourceRowEnd = sourceRows * (blockIdx + 1U) / blockNum;
        const uint64_t sourceGenerationCount =
            (sourceRows + kAtomicInt32SourceChunkRows - 1U) /
            kAtomicInt32SourceChunkRows;
        const uint32_t sourcePairCount = blockNum >> 1U;
        const uint32_t sourcePair = blockIdx >> 1U;
        const uint32_t sourcePairLane = blockIdx & 1U;
        const uint64_t sourcePairGenerationStart =
            usePairedSourceOwnership
                ? sourceGenerationCount * sourcePair / sourcePairCount
                : 0U;
        const uint64_t sourcePairGenerationEnd =
            usePairedSourceOwnership
                ? sourceGenerationCount * (sourcePair + 1U) /
                      sourcePairCount
                : 0U;
        uint32_t currentPairLane = sourcePairLane;
        uint64_t currentGeneration =
            sourcePairGenerationStart + sourcePairLane;
        uint64_t currentStart =
            usePairedSourceOwnership
                ? currentGeneration * kAtomicInt32SourceChunkRows
                : sourceRowStart;
        const uint64_t currentRemainingRows =
            usePairedSourceOwnership
                ? (currentGeneration < sourcePairGenerationEnd
                       ? sourceRows - currentStart
                       : 0U)
                : sourceRowEnd - currentStart;
        uint32_t currentCount = static_cast<uint32_t>(
            currentRemainingRows < kAtomicInt32SourceChunkRows
                ? currentRemainingRows
                : kAtomicInt32SourceChunkRows);
        bool useFirst = true;
        bool firstInUse = false;
        bool secondInUse = false;
        if (currentCount != 0U) {
            CopySourceGeneration(firstSource, currentStart, currentCount,
                                 rowBytes, alignedInnerLength,
                                 usePackedSource);
            SetFlag<HardEvent::MTE2_MTE3>(
                static_cast<event_t>(firstSourceReady));
            firstInUse = true;
        }

        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        SyncAll();

        __builtin_cce_set_atomic_s32();
        __builtin_cce_set_atomic_add();
        while (currentCount != 0U) {
            const TEventID currentReady =
                useFirst ? firstSourceReady : secondSourceReady;
            WaitFlag<HardEvent::MTE2_MTE3>(
                static_cast<event_t>(currentReady));

            const uint64_t nextGeneration =
                currentGeneration + (currentPairLane == 0U ? 3U : 1U);
            const uint64_t nextStart =
                usePairedSourceOwnership
                    ? nextGeneration * kAtomicInt32SourceChunkRows
                    : currentStart + currentCount;
            const uint64_t nextRemainingRows =
                usePairedSourceOwnership
                    ? (nextGeneration < sourcePairGenerationEnd
                           ? sourceRows - nextStart
                           : 0U)
                    : sourceRowEnd - nextStart;
            const uint32_t nextCount = static_cast<uint32_t>(
                nextRemainingRows < kAtomicInt32SourceChunkRows
                    ? nextRemainingRows
                    : kAtomicInt32SourceChunkRows);
            if (nextCount != 0U) {
                if (useFirst) {
                    if (secondInUse) {
                        WaitFlag<HardEvent::MTE3_MTE2>(
                            static_cast<event_t>(secondSourceReusable));
                        secondInUse = false;
                    }
                    CopySourceGeneration(
                        secondSource, nextStart, nextCount, rowBytes,
                        alignedInnerLength, usePackedSource);
                    SetFlag<HardEvent::MTE2_MTE3>(
                        static_cast<event_t>(secondSourceReady));
                    secondInUse = true;
                } else {
                    if (firstInUse) {
                        WaitFlag<HardEvent::MTE3_MTE2>(
                            static_cast<event_t>(firstSourceReusable));
                        firstInUse = false;
                    }
                    CopySourceGeneration(
                        firstSource, nextStart, nextCount, rowBytes,
                        alignedInnerLength, usePackedSource);
                    SetFlag<HardEvent::MTE2_MTE3>(
                        static_cast<event_t>(firstSourceReady));
                    firstInUse = true;
                }
            }

            LocalTensor<int32_t> currentSource =
                useFirst ? firstSource : secondSource;
            if (usePackedSource) {
                LocalTensor<int32_t> currentHeadScratch =
                    useFirst ? firstHeadScratch : secondHeadScratch;
                uint64_t outputOffsets[kAtomicInt32SourceChunkRows];
                for (uint32_t localRow = 0; localRow < currentCount;
                     ++localRow) {
                    const uint64_t flatSourceRow = currentStart + localRow;
                    const uint64_t outer = flatSourceRow / indexLength_;
                    const uint32_t update = static_cast<uint32_t>(
                        flatSourceRow - outer * indexLength_);
                    const uint32_t axis = static_cast<uint32_t>(
                        cachedIndices.GetValue(update));
                    outputOffsets[localRow] =
                        (outer * axisLength_ + axis) * innerLength_;
                }

                const bool leadingSingleRow =
                    currentCount == kAtomicInt32SourceChunkRows &&
                    ((currentStart * static_cast<uint64_t>(rowBytes)) &
                     31U) != 0U;

                // Store every already aligned row first. INT32 accumulation
                // is exact modulo 2^32, so the later prefixed group may be
                // reordered without changing the result.
                if (leadingSingleRow) {
                    const DataCopyExtParams rowCopy{
                        1, rowBytes, 0, 0, 0};
                    DataCopyPad(output_[outputOffsets[0U]], currentSource,
                                rowCopy);
                    for (uint32_t localRow = 1U;
                         localRow < currentCount; localRow += 2U) {
                        const uint32_t packedRowStart =
                            localRow * static_cast<uint32_t>(innerLength_) +
                            4U;
                        DataCopyPad(output_[outputOffsets[localRow]],
                                    currentSource[packedRowStart], rowCopy);
                    }
                } else {
                    for (uint32_t localRow = 0; localRow < currentCount;
                         localRow += 2U) {
                        const uint32_t packedRowStart =
                            localRow * static_cast<uint32_t>(innerLength_);
                        const DataCopyExtParams rowCopy{
                            1, rowBytes, 0, 0, 0};
                        DataCopyPad(output_[outputOffsets[localRow]],
                                    currentSource[packedRowStart], rowCopy);
                    }
                }

                if (currentCount > 1U) {
                    SetFlag<HardEvent::MTE3_V>(
                        static_cast<event_t>(sourceStoreComplete));
                    WaitFlag<HardEvent::MTE3_V>(
                        static_cast<event_t>(sourceStoreComplete));

                    // All aligned-row MTE3 reads are now complete. Clear both
                    // prefixes and gather any row-zero exception in one
                    // Vector generation before releasing prefixed stores.
                    const uint32_t firstPrefixedRow =
                        leadingSingleRow ? 2U : 1U;
                    for (uint32_t localRow = firstPrefixedRow;
                         localRow < currentCount;
                         localRow += 2U) {
                        const uint32_t packedRowStart =
                            localRow * static_cast<uint32_t>(innerLength_) +
                            (leadingSingleRow ? 4U : 0U);
                        const uint32_t prefixStart = packedRowStart - 4U;
                        Duplicate(currentSource[prefixStart],
                                  static_cast<int32_t>(0), 4);
                        if (outputOffsets[localRow] == 0U) {
                            const uint32_t scratchOffset =
                                (localRow - firstPrefixedRow) * 4U;
                            Gather(currentHeadScratch[scratchOffset],
                                   currentSource[prefixStart], gatherOffsets,
                                   kAtomicInt32PhaseBytes, 4U);
                        }
                    }

                    SetFlag<HardEvent::V_MTE3>(
                        static_cast<event_t>(sourceVectorComplete));
                    WaitFlag<HardEvent::V_MTE3>(
                        static_cast<event_t>(sourceVectorComplete));

                    for (uint32_t localRow = firstPrefixedRow;
                         localRow < currentCount;
                         localRow += 2U) {
                        const uint32_t packedRowStart =
                            localRow * static_cast<uint32_t>(innerLength_) +
                            (leadingSingleRow ? 4U : 0U);
                        const uint32_t prefixStart = packedRowStart - 4U;
                        const uint64_t outputOffset =
                            outputOffsets[localRow];
                        if (outputOffset == 0U) {
                            const uint32_t scratchOffset =
                                (localRow - firstPrefixedRow) * 4U;
                            const DataCopyExtParams headCopy{
                                1, kAtomicInt32PhaseBytes, 0, 0, 0};
                            const DataCopyExtParams tailCopy{
                                1, rowBytes - kAtomicInt32PhaseBytes,
                                0, 0, 0};
                            DataCopyPad(output_[0U],
                                        currentHeadScratch[scratchOffset],
                                        headCopy);
                            DataCopyPad(output_[4U],
                                        currentSource[packedRowStart + 4U],
                                        tailCopy);
                        } else {
                            const DataCopyExtParams prefixedRowCopy{
                                1, rowBytes + kAtomicInt32PhaseBytes,
                                0, 0, 0};
                            DataCopyPad(output_[outputOffset - 4U],
                                        currentSource[prefixStart],
                                        prefixedRowCopy);
                        }
                    }
                }
            } else {
                for (uint32_t localRow = 0; localRow < currentCount;
                     ++localRow) {
                    const uint64_t flatSourceRow = currentStart + localRow;
                    const uint64_t outer = flatSourceRow / indexLength_;
                    const uint32_t update = static_cast<uint32_t>(
                        flatSourceRow - outer * indexLength_);
                    const uint32_t axis = static_cast<uint32_t>(
                        cachedIndices.GetValue(update));
                    const uint64_t outputOffset =
                        (outer * axisLength_ + axis) * innerLength_;
                    const DataCopyExtParams rowCopy{
                        1, rowBytes, 0, 0, 0};
                    DataCopyPad(output_[outputOffset],
                                currentSource[localRow * alignedInnerLength],
                                rowCopy);
                }
            }
            const TEventID currentReusable =
                useFirst ? firstSourceReusable : secondSourceReusable;
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(currentReusable));

            currentStart = nextStart;
            currentCount = nextCount;
            currentGeneration = nextGeneration;
            currentPairLane ^= 1U;
            useFirst = !useFirst;
        }
        if (firstInUse) {
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(firstSourceReusable));
        }
        if (secondInUse) {
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(secondSourceReusable));
        }
        __builtin_cce_set_atomic_none();

        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_MTE3>(firstSourceReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_MTE3>(secondSourceReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(firstSourceReusable);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(secondSourceReusable);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(sourceStoreComplete);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(sourceVectorComplete);
    }

private:
    __aicore__ inline void CopySourceGeneration(
        LocalTensor<int32_t> destination, uint64_t firstRow,
        uint32_t rowCount, uint32_t rowBytes,
        uint32_t alignedInnerLength, bool usePackedSource)
    {
        if (!usePackedSource ||
            rowCount != kAtomicInt32SourceChunkRows) {
            const DataCopyExtParams copy{
                static_cast<uint16_t>(usePackedSource ? 1U : rowCount),
                usePackedSource ? rowCount * rowBytes : rowBytes, 0, 0,
                0};
            DataCopyPad(destination, source_[firstRow * innerLength_], copy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            return;
        }

        const bool leadingSingleRow =
            ((firstRow * static_cast<uint64_t>(rowBytes)) & 31U) != 0U;
        if (leadingSingleRow) {
            const DataCopyExtParams firstCopy{1, rowBytes, 0, 0, 0};
            const DataCopyExtParams finalCopy{1, 4U * rowBytes, 0, 0, 0};
            DataCopyPad(destination, source_[firstRow * innerLength_],
                        firstCopy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            DataCopyPad(destination[alignedInnerLength],
                        source_[(firstRow + 1U) * innerLength_], finalCopy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        } else {
            const DataCopyExtParams firstCopy{1, 4U * rowBytes, 0, 0, 0};
            const DataCopyExtParams finalCopy{1, rowBytes, 0, 0, 0};
            DataCopyPad(destination, source_[firstRow * innerLength_],
                        firstCopy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            DataCopyPad(
                destination[4U * static_cast<uint32_t>(innerLength_)],
                source_[(firstRow + 4U) * innerLength_], finalCopy,
                DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> firstSourceBuffer_;
    TBuf<QuePosition::VECCALC> secondSourceBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    TBuf<QuePosition::VECCALC> firstHeadScratchBuffer_;
    TBuf<QuePosition::VECCALC> secondHeadScratchBuffer_;
    TBuf<QuePosition::VECCALC> gatherOffsetBuffer_;
    GlobalTensor<int32_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<int32_t> source_;
    GlobalTensor<int32_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

class IndexAddWideInt32AxisGroups {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(self),
                              totalLength_);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output),
                                totalLength_);
        pipe_.InitBuffer(
            accumulatorBuffer_,
            kWideInt32AxisGroupRows * kWideInt32MaxInnerLength *
                sizeof(int32_t));
        pipe_.InitBuffer(sourceBuffer_,
                         kWideInt32MaxInnerLength * sizeof(int32_t));
        pipe_.InitBuffer(indexBuffer_,
                         kWideInt32MaxIndexLength * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> accumulator =
            accumulatorBuffer_.Get<int32_t>();
        LocalTensor<int32_t> stagedSources = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();
        LocalTensor<uint64_t> cachedIndexPairs =
            cachedIndices.template ReinterpretCast<uint64_t>();

        const TEventID indexReady = pipe_.AllocEventID<HardEvent::MTE2_S>();
        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);

        const TEventID selfReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID sourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID sourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady =
            pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();

        const uint32_t rowBytes =
            static_cast<uint32_t>(innerLength_) * sizeof(int32_t);
        const uint32_t accumulatorRowElements =
            (static_cast<uint32_t>(innerLength_) + 7U) & ~7U;
        const DataCopyExtParams sourceCopy{1, rowBytes, 0, 0, 0};
        const uint32_t groupsPerOuter =
            (axisLength_ + kWideInt32AxisGroupRows - 1U) /
            kWideInt32AxisGroupRows;
        const uint64_t workCount = outerLength_ * groupsPerOuter;
        const uint32_t blockStride = GetBlockNum();
        for (uint64_t work = GetBlockIdx(); work < workCount;
             work += blockStride) {
            const uint64_t outer = work / groupsPerOuter;
            const uint32_t group =
                static_cast<uint32_t>(work % groupsPerOuter);
            const uint32_t firstAxis =
                group * kWideInt32AxisGroupRows;
            const uint32_t rowCount =
                axisLength_ - firstAxis < kWideInt32AxisGroupRows
                    ? axisLength_ - firstAxis
                    : kWideInt32AxisGroupRows;
            const uint64_t selfOffset =
                (outer * axisLength_ + firstAxis) * innerLength_;
            const DataCopyExtParams selfCopy{
                static_cast<uint16_t>(rowCount), rowBytes, 0, 0, 0};
            DataCopyPad(accumulator, self_[selfOffset], selfCopy,
                        DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(selfReady));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(selfReady));

            bool sourceInUse = false;
            uint32_t cursor = 0U;
            uint64_t packedIndex = 0U;
            uint32_t update = 0U;
            uint32_t row = 0U;
            bool hasUpdate = FindNextUpdate(
                cachedIndexPairs, cursor, packedIndex, firstAxis, rowCount,
                update, row);
            while (hasUpdate) {
                if (sourceInUse) {
                    WaitFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(sourceReusable));
                }
                const uint64_t sourceOffset =
                    (outer * indexLength_ + update) * innerLength_;
                DataCopyPad(stagedSources, source_[sourceOffset], sourceCopy,
                            DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(sourceReady));
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(sourceReady));
                Add(accumulator[row * accumulatorRowElements],
                    accumulator[row * accumulatorRowElements],
                    stagedSources,
                    static_cast<uint32_t>(innerLength_));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(sourceReusable));
                sourceInUse = true;
                hasUpdate = FindNextUpdate(
                    cachedIndexPairs, cursor, packedIndex, firstAxis,
                    rowCount, update, row);
            }
            if (sourceInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(sourceReusable));
            }

            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            DataCopyPad(output_[selfOffset], accumulator, selfCopy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(selfReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(sourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
    }

private:
    __aicore__ inline bool FindNextUpdate(
        LocalTensor<uint64_t> cachedIndexPairs, uint32_t& cursor,
        uint64_t& packedIndex, uint32_t firstAxis, uint32_t rowCount,
        uint32_t& update, uint32_t& row)
    {
        while (cursor < indexLength_) {
            const uint32_t candidate = cursor++;
            if ((candidate & 1U) == 0U) {
                packedIndex = cachedIndexPairs.GetValue(candidate / 2U);
            }
            const uint32_t axis = static_cast<uint32_t>(packedIndex);
            packedIndex >>= 32U;
            const uint32_t candidateRow = axis - firstAxis;
            if (candidateRow < rowCount) {
                update = candidate;
                row = candidateRow;
                return true;
            }
        }
        return false;
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> sourceBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    GlobalTensor<int32_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<int32_t> source_;
    GlobalTensor<int32_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

class IndexAddWideInt32InnerTiles {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(self),
                              totalLength_);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output),
                                totalLength_);
        pipe_.InitBuffer(
            accumulatorBuffer_,
            kWideInt32InnerTileMaxAxisLength *
                kWideInt32InnerTileStride * sizeof(int32_t));
        pipe_.InitBuffer(
            sourceBuffer_,
            kWideInt32InnerTileSourceChunk *
                kWideInt32InnerTileStride * sizeof(int32_t));
        pipe_.InitBuffer(indexBuffer_,
                         kWideInt32MaxIndexLength * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> accumulators =
            accumulatorBuffer_.Get<int32_t>();
        LocalTensor<int32_t> stagedSources = sourceBuffer_.Get<int32_t>();
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();

        const TEventID indexReady = pipe_.AllocEventID<HardEvent::MTE2_S>();
        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));

        const TEventID selfReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID sourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID sourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady =
            pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();

        const uint32_t tilesPerOuter = static_cast<uint32_t>(
            (innerLength_ + kWideInt32InnerTileElements - 1U) /
            kWideInt32InnerTileElements);
        const uint64_t workCount = outerLength_ * tilesPerOuter;
        for (uint64_t work = GetBlockIdx(); work < workCount;
             work += GetBlockNum()) {
            const uint64_t outer = work / tilesPerOuter;
            const uint32_t tile =
                static_cast<uint32_t>(work % tilesPerOuter);
            const uint32_t firstInner =
                tile * kWideInt32InnerTileElements;
            const uint32_t activeInner = static_cast<uint32_t>(
                innerLength_ - firstInner < kWideInt32InnerTileElements
                    ? innerLength_ - firstInner
                    : kWideInt32InnerTileElements);
            const uint32_t activeBytes = activeInner * sizeof(int32_t);
            const uint32_t alignedBytes = (activeBytes + 31U) & ~31U;
            const uint32_t ubStrideBlocks =
                (kWideInt32InnerTileStride * sizeof(int32_t) -
                 alignedBytes) /
                32U;
            const uint32_t gmStrideBytes =
                static_cast<uint32_t>(innerLength_ - activeInner) *
                sizeof(int32_t);

            const uint64_t selfOffset =
                outer * axisLength_ * innerLength_ + firstInner;
            const DataCopyExtParams selfCopy{
                static_cast<uint16_t>(axisLength_), activeBytes,
                gmStrideBytes, ubStrideBlocks, 0};
            DataCopyPad(
                accumulators, self_[selfOffset], selfCopy,
                DataCopyPadExtParams<int32_t>{
                    true, 0,
                    static_cast<uint8_t>((alignedBytes - activeBytes) /
                                         sizeof(int32_t)),
                    0});
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(selfReady));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(selfReady));

            for (uint32_t firstUpdate = 0; firstUpdate < indexLength_;
                 firstUpdate += kWideInt32InnerTileSourceChunk) {
                const uint32_t updateCount =
                    indexLength_ - firstUpdate <
                            kWideInt32InnerTileSourceChunk
                        ? indexLength_ - firstUpdate
                        : kWideInt32InnerTileSourceChunk;
                const uint64_t sourceOffset =
                    (outer * indexLength_ + firstUpdate) * innerLength_ +
                    firstInner;
                const DataCopyExtParams sourceCopy{
                    static_cast<uint16_t>(updateCount), activeBytes,
                    gmStrideBytes, ubStrideBlocks, 0};
                DataCopyPad(
                    stagedSources, source_[sourceOffset], sourceCopy,
                    DataCopyPadExtParams<int32_t>{
                        true, 0,
                        static_cast<uint8_t>((alignedBytes - activeBytes) /
                                             sizeof(int32_t)),
                        0});
                SetFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(sourceReady));
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(sourceReady));

                for (uint32_t localUpdate = 0;
                     localUpdate < updateCount; ++localUpdate) {
                    const uint32_t axis = static_cast<uint32_t>(
                        cachedIndices.GetValue(firstUpdate + localUpdate));
                    Add(accumulators[
                            axis * kWideInt32InnerTileStride],
                        accumulators[
                            axis * kWideInt32InnerTileStride],
                        stagedSources[
                            localUpdate * kWideInt32InnerTileStride],
                        activeInner);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(sourceReusable));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(sourceReusable));
            }

            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            const DataCopyExtParams outputCopy{
                static_cast<uint16_t>(axisLength_), activeBytes,
                ubStrideBlocks, gmStrideBytes, 0};
            DataCopyPad(output_[selfOffset], accumulators, outputCopy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(selfReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(sourceReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(sourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> sourceBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    GlobalTensor<int32_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<int32_t> source_;
    GlobalTensor<int32_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

// Retain the historical instruction layout of non-BF16 objects. The packed
// path below is selected only for a real BF16 compilation, not by tensor shape.
class IndexAddLegacyBfloat16AxisGroups {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        constexpr uint32_t groupElements =
            kBfloatAxisGroupRows * kBfloatGroupMaxInnerLength;
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(self));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
        pipe_.InitBuffer(accumulatorBuffer_, groupElements * sizeof(float));
        pipe_.InitBuffer(stagingBuffer_, groupElements * sizeof(bfloat16_t));
        pipe_.InitBuffer(secondSourceStagingBuffer_,
                         kBfloatGroupSourceChunkRows *
                             kBfloatGroupMaxInnerLength *
                             sizeof(bfloat16_t));
        pipe_.InitBuffer(convertedSourceBuffer_,
                         kBfloatGroupSourceChunkRows *
                             kBfloatGroupMaxInnerLength * sizeof(float));
        pipe_.InitBuffer(indexBuffer_,
                         kBfloatGroupMaxIndexLength * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        LocalTensor<bfloat16_t> staging = stagingBuffer_.Get<bfloat16_t>();
        // Self/output staging and the first BF16 source generation have
        // disjoint lifetimes. Reusing their 32-KiB region leaves enough UB
        // to convert one complete source generation to FP32 at once.
        LocalTensor<bfloat16_t> firstStagedSources = staging;
        LocalTensor<bfloat16_t> secondStagedSources =
            secondSourceStagingBuffer_.Get<bfloat16_t>();
        LocalTensor<float> convertedSources =
            convertedSourceBuffer_.Get<float>();
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();

        const TEventID indexReady = pipe_.AllocEventID<HardEvent::MTE2_S>();
        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);

        const TEventID inputReady = pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID firstSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID secondSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID firstSourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID secondSourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady = pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const uint32_t balancedGroupsPerOuter = static_cast<uint32_t>(
            (GetBlockNum() + outerLength_ - 1U) / outerLength_);
        const uint32_t targetGroupsPerOuter =
            balancedGroupsPerOuter < 2U ? balancedGroupsPerOuter : 2U;
        const uint32_t balancedGroupRows =
            (axisLength_ + targetGroupsPerOuter - 1U) /
            targetGroupsPerOuter;
        const uint32_t runtimeGroupRows =
            balancedGroupRows < kBfloatAxisGroupRows
                ? balancedGroupRows
                : kBfloatAxisGroupRows;
        const uint32_t runtimeGroupsPerOuter =
            (axisLength_ + runtimeGroupRows - 1U) /
            runtimeGroupRows;
        const uint64_t workCount =
            outerLength_ * runtimeGroupsPerOuter;
        const uint32_t blockStride = GetBlockNum();

        for (uint64_t work = GetBlockIdx(); work < workCount;
             work += blockStride) {
            const uint64_t outer = work / runtimeGroupsPerOuter;
            const uint32_t group =
                static_cast<uint32_t>(work % runtimeGroupsPerOuter);
            const uint32_t firstAxis = group * runtimeGroupRows;
            const uint32_t rowCount =
                axisLength_ - firstAxis < runtimeGroupRows
                    ? axisLength_ - firstAxis
                    : runtimeGroupRows;
            const uint32_t accumulatorRowElements =
                (static_cast<uint32_t>(innerLength_) + 15U) & ~15U;
            const uint32_t paddedGroupElements =
                rowCount * accumulatorRowElements;
            const uint64_t selfOffset =
                (outer * axisLength_ + firstAxis) * innerLength_;
            const DataCopyExtParams groupCopy{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(innerLength_) *
                    static_cast<uint32_t>(sizeof(bfloat16_t)),
                0, 0, 0};
            DataCopyPad(staging, self_[selfOffset], groupCopy,
                        DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(inputReady));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(inputReady));
            Cast(accumulator, staging, RoundMode::CAST_NONE,
                 paddedGroupElements);
            PipeBarrier<PIPE_V>();

            const uint32_t alignedInnerElements =
                (static_cast<uint32_t>(innerLength_) + 15U) & ~15U;
            const DataCopyPadExtParams<bfloat16_t> sourcePadding{
                true, 0,
                static_cast<uint8_t>(alignedInnerElements - innerLength_),
                static_cast<bfloat16_t>(0)};
            const uint64_t firstSourceOffset =
                outer * indexLength_ * innerLength_;
            uint32_t sourceFirst = 0;
            uint32_t sourceRows =
                indexLength_ < kBfloatGroupSourceChunkRows
                    ? indexLength_
                    : kBfloatGroupSourceChunkRows;
            bool useFirstSource = true;
            bool firstSourceInUse = false;
            bool secondSourceInUse = false;
            if (sourceRows != 0U) {
                // staging aliases the first source buffer. PIPE_V orders the
                // self Cast within Vector, but only V_MTE2 makes its read
                // completion visible before MTE2 overwrites that region.
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(firstSourceReusable));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(firstSourceReusable));
                CopySourceChunk(firstStagedSources, firstSourceOffset,
                                sourceFirst, sourceRows, sourcePadding,
                                firstSourceReady);
            }
            while (sourceRows != 0U) {
                const TEventID currentReady =
                    useFirstSource ? firstSourceReady : secondSourceReady;
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(currentReady));

                const uint32_t nextSourceFirst = sourceFirst + sourceRows;
                const uint32_t nextSourceRows =
                    indexLength_ - nextSourceFirst <
                            kBfloatGroupSourceChunkRows
                        ? indexLength_ - nextSourceFirst
                        : kBfloatGroupSourceChunkRows;
                if (nextSourceRows != 0U) {
                    if (useFirstSource) {
                        if (secondSourceInUse) {
                            WaitFlag<HardEvent::V_MTE2>(
                                static_cast<event_t>(
                                    secondSourceReusable));
                            secondSourceInUse = false;
                        }
                        CopySourceChunk(
                            secondStagedSources, firstSourceOffset,
                            nextSourceFirst, nextSourceRows, sourcePadding,
                            secondSourceReady);
                    } else {
                        if (firstSourceInUse) {
                            WaitFlag<HardEvent::V_MTE2>(
                                static_cast<event_t>(
                                    firstSourceReusable));
                            firstSourceInUse = false;
                        }
                        CopySourceChunk(
                            firstStagedSources, firstSourceOffset,
                            nextSourceFirst, nextSourceRows, sourcePadding,
                            firstSourceReady);
                    }
                }

                LocalTensor<bfloat16_t> stagedSources =
                    useFirstSource ? firstStagedSources
                                   : secondStagedSources;
                const uint32_t convertedElements =
                    sourceRows * alignedInnerElements;
                Cast(convertedSources, stagedSources, RoundMode::CAST_NONE,
                     convertedElements);
                PipeBarrier<PIPE_V>();

                // The current BF16 generation is dead after the bulk Cast,
                // so its first padded row can serve as the per-update
                // round-trip scratch while MTE2 fills the other generation.
                LocalTensor<bfloat16_t> roundedSource = stagedSources;
                for (uint32_t localUpdate = 0; localUpdate < sourceRows;
                     ++localUpdate) {
                    const uint32_t update = sourceFirst + localUpdate;
                    const uint32_t axis = static_cast<uint32_t>(
                        cachedIndices.GetValue(update));
                    const uint32_t localAxis = axis - firstAxis;
                    if (localAxis >= rowCount) {
                        continue;
                    }
                    LocalTensor<float> target =
                        accumulator[localAxis * accumulatorRowElements];
                    Add(target, target,
                        convertedSources[localUpdate * alignedInnerElements],
                        static_cast<uint32_t>(innerLength_));
                    PipeBarrier<PIPE_V>();
                    Cast(roundedSource, target, RoundMode::CAST_RINT,
                         static_cast<uint32_t>(innerLength_));
                    PipeBarrier<PIPE_V>();
                    Cast(target, roundedSource, RoundMode::CAST_NONE,
                         static_cast<uint32_t>(innerLength_));
                    PipeBarrier<PIPE_V>();
                }
                if (useFirstSource) {
                    SetFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(firstSourceReusable));
                    firstSourceInUse = true;
                } else {
                    SetFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(secondSourceReusable));
                    secondSourceInUse = true;
                }
                sourceFirst = nextSourceFirst;
                sourceRows = nextSourceRows;
                useFirstSource = !useFirstSource;
            }
            if (firstSourceInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(firstSourceReusable));
            }
            if (secondSourceInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(secondSourceReusable));
            }

            Cast(staging, accumulator, RoundMode::CAST_RINT,
                 paddedGroupElements);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            DataCopyPad(output_[selfOffset], staging, groupCopy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(firstSourceReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(secondSourceReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(firstSourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(secondSourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
    }

private:
    __aicore__ inline void CopySourceChunk(
        LocalTensor<bfloat16_t> destination, uint64_t firstSourceOffset,
        uint32_t sourceFirst, uint32_t sourceRows,
        const DataCopyPadExtParams<bfloat16_t>& sourcePadding,
        TEventID sourceReady)
    {
        const DataCopyExtParams sourcesCopy{
            static_cast<uint16_t>(sourceRows),
            static_cast<uint32_t>(innerLength_) *
                static_cast<uint32_t>(sizeof(bfloat16_t)),
            0, 0, 0};
        DataCopyPad(destination,
                    source_[firstSourceOffset +
                            static_cast<uint64_t>(sourceFirst) *
                                innerLength_],
                    sourcesCopy, sourcePadding);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(sourceReady));
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> stagingBuffer_;
    TBuf<QuePosition::VECCALC> secondSourceStagingBuffer_;
    TBuf<QuePosition::VECCALC> convertedSourceBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    GlobalTensor<bfloat16_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<bfloat16_t> source_;
    GlobalTensor<bfloat16_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};


class IndexAddBfloat16AxisGroups {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        constexpr uint32_t groupElements =
            kBfloatAxisGroupRows * kBfloatGroupMaxInnerLength;
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(self));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
        pipe_.InitBuffer(accumulatorBuffer_, groupElements * sizeof(float));
        pipe_.InitBuffer(stagingBuffer_, groupElements * sizeof(bfloat16_t));
        pipe_.InitBuffer(secondSourceStagingBuffer_,
                         kBfloatGroupSourceChunkRows *
                             kBfloatGroupMaxInnerLength *
                             sizeof(bfloat16_t));
        pipe_.InitBuffer(convertedSourceBuffer_,
                         4U * kBfloatGroupMaxInnerLength * sizeof(float));
        pipe_.InitBuffer(indexBuffer_,
                         kBfloatGroupMaxIndexLength * sizeof(int32_t));
        pipe_.InitBuffer(gatheredSourceBuffer_,
                         4U * kBfloatGroupMaxInnerLength *
                             sizeof(bfloat16_t));
        pipe_.InitBuffer(gatherOffsetBuffer_,
                         kBfloatGroupMaxInnerLength * sizeof(uint32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        LocalTensor<bfloat16_t> staging = stagingBuffer_.Get<bfloat16_t>();
        // Self/output staging and the first packed BF16 source generation
        // have disjoint lifetimes. Source staging remains read-only until
        // every owned row in that generation has been gathered.
        LocalTensor<bfloat16_t> firstStagedSources = staging;
        LocalTensor<bfloat16_t> secondStagedSources =
            secondSourceStagingBuffer_.Get<bfloat16_t>();
        LocalTensor<float> convertedSources =
            convertedSourceBuffer_.Get<float>();
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();

        LocalTensor<bfloat16_t> gatheredSources =
            gatheredSourceBuffer_.Get<bfloat16_t>();
        LocalTensor<int32_t> gatherOffsetValues =
            gatherOffsetBuffer_.Get<int32_t>();
        LocalTensor<uint32_t> gatherOffsets =
            gatherOffsetValues.ReinterpretCast<uint32_t>();
        ArithProgression(gatherOffsetValues, static_cast<int32_t>(0),
                         static_cast<int32_t>(sizeof(bfloat16_t)),
                         kBfloatGroupMaxInnerLength);
        Duplicate(gatheredSources.ReinterpretCast<uint16_t>(),
                  static_cast<uint16_t>(0),
                  4U * kBfloatGroupMaxInnerLength);
        PipeBarrier<PIPE_V>();

        const TEventID indexReady = pipe_.AllocEventID<HardEvent::MTE2_S>();
        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);

        const TEventID inputReady = pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID firstSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID secondSourceReady =
            pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID firstSourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID secondSourceReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady = pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const uint32_t balancedGroupsPerOuter = static_cast<uint32_t>(
            (GetBlockNum() + outerLength_ - 1U) / outerLength_);
        const uint32_t targetGroupsPerOuter =
            balancedGroupsPerOuter < 2U ? balancedGroupsPerOuter : 2U;
        const uint32_t balancedGroupRows =
            (axisLength_ + targetGroupsPerOuter - 1U) /
            targetGroupsPerOuter;
        const uint32_t runtimeGroupRows =
            balancedGroupRows < kBfloatAxisGroupRows
                ? balancedGroupRows
                : kBfloatAxisGroupRows;
        const uint32_t runtimeGroupsPerOuter =
            (axisLength_ + runtimeGroupRows - 1U) /
            runtimeGroupRows;
        const uint64_t workCount =
            outerLength_ * runtimeGroupsPerOuter;
        const uint32_t blockStride = GetBlockNum();

        for (uint64_t work = GetBlockIdx(); work < workCount;
             work += blockStride) {
            const uint64_t outer = work / runtimeGroupsPerOuter;
            const uint32_t group =
                static_cast<uint32_t>(work % runtimeGroupsPerOuter);
            const uint32_t firstAxis = group * runtimeGroupRows;
            const uint32_t rowCount =
                axisLength_ - firstAxis < runtimeGroupRows
                    ? axisLength_ - firstAxis
                    : runtimeGroupRows;
            const uint32_t accumulatorRowElements =
                (static_cast<uint32_t>(innerLength_) + 15U) & ~15U;
            const uint32_t paddedGroupElements =
                rowCount * accumulatorRowElements;
            const uint64_t selfOffset =
                (outer * axisLength_ + firstAxis) * innerLength_;
            const DataCopyExtParams groupCopy{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(innerLength_) *
                    static_cast<uint32_t>(sizeof(bfloat16_t)),
                0, 0, 0};
            DataCopyPad(staging, self_[selfOffset], groupCopy,
                        DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(inputReady));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(inputReady));
            Cast(accumulator, staging, RoundMode::CAST_NONE,
                 paddedGroupElements);
            PipeBarrier<PIPE_V>();

            const uint32_t alignedInnerElements =
                (static_cast<uint32_t>(innerLength_) + 15U) & ~15U;
            const uint64_t firstSourceOffset =
                outer * indexLength_ * innerLength_;
            uint32_t sourceFirst = 0;
            uint32_t sourceRows =
                indexLength_ < kBfloatGroupSourceChunkRows
                    ? indexLength_
                    : kBfloatGroupSourceChunkRows;
            bool useFirstSource = true;
            bool firstSourceInUse = false;
            bool secondSourceInUse = false;
            if (sourceRows != 0U) {
                // staging aliases the first source buffer. PIPE_V orders the
                // self Cast within Vector, but only V_MTE2 makes its read
                // completion visible before MTE2 overwrites that region.
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(firstSourceReusable));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(firstSourceReusable));
                CopySourceChunk(firstStagedSources, firstSourceOffset,
                                sourceFirst, sourceRows,
                                firstSourceReady);
            }
            while (sourceRows != 0U) {
                const TEventID currentReady =
                    useFirstSource ? firstSourceReady : secondSourceReady;
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(currentReady));

                const uint32_t nextSourceFirst = sourceFirst + sourceRows;
                const uint32_t nextSourceRows =
                    indexLength_ - nextSourceFirst <
                            kBfloatGroupSourceChunkRows
                        ? indexLength_ - nextSourceFirst
                        : kBfloatGroupSourceChunkRows;
                if (nextSourceRows != 0U) {
                    if (useFirstSource) {
                        if (secondSourceInUse) {
                            WaitFlag<HardEvent::V_MTE2>(
                                static_cast<event_t>(
                                    secondSourceReusable));
                            secondSourceInUse = false;
                        }
                        CopySourceChunk(
                            secondStagedSources, firstSourceOffset,
                            nextSourceFirst, nextSourceRows,
                            secondSourceReady);
                    } else {
                        if (firstSourceInUse) {
                            WaitFlag<HardEvent::V_MTE2>(
                                static_cast<event_t>(
                                    firstSourceReusable));
                            firstSourceInUse = false;
                        }
                        CopySourceChunk(
                            firstStagedSources, firstSourceOffset,
                            nextSourceFirst, nextSourceRows,
                            firstSourceReady);
                    }
                }

                LocalTensor<bfloat16_t> stagedSources =
                    useFirstSource ? firstStagedSources
                                   : secondStagedSources;
                // The packed BF16 generation stays live until all owned rows
                // have been gathered. Only the next quartet is expanded.
                uint32_t cursor = 0;
                uint32_t p0, p1, p2, p3, a0, a1, a2, a3;
                while (NextOwned(cachedIndices, sourceFirst, sourceRows,
                                 firstAxis, rowCount, cursor, p0, a0)) {
                    const bool have1 = NextOwned(
                        cachedIndices, sourceFirst, sourceRows,
                        firstAxis, rowCount, cursor, p1, a1);
                    const bool have2 = have1 && NextOwned(
                        cachedIndices, sourceFirst, sourceRows,
                        firstAxis, rowCount, cursor, p2, a2);
                    const bool have3 = have2 && NextOwned(
                        cachedIndices, sourceFirst, sourceRows,
                        firstAxis, rowCount, cursor, p3, a3);
                    SetMaskNorm();
                    ResetMask();
                    const uint32_t inner =
                        static_cast<uint32_t>(innerLength_);
                    Gather(gatheredSources, stagedSources, gatherOffsets,
                           p0 * inner * sizeof(bfloat16_t), inner);
                    if (have1) {
                        Gather(gatheredSources[alignedInnerElements],
                               stagedSources, gatherOffsets,
                               p1 * inner * sizeof(bfloat16_t), inner);
                    }
                    if (have2) {
                        Gather(gatheredSources[2U * alignedInnerElements],
                               stagedSources, gatherOffsets,
                               p2 * inner * sizeof(bfloat16_t), inner);
                    }
                    if (have3) {
                        Gather(gatheredSources[3U * alignedInnerElements],
                               stagedSources, gatherOffsets,
                               p3 * inner * sizeof(bfloat16_t), inner);
                    }
                    PipeBarrier<PIPE_V>();
                    const uint32_t gatheredRows =
                        1U + static_cast<uint32_t>(have1) +
                        static_cast<uint32_t>(have2) +
                        static_cast<uint32_t>(have3);
                    Cast(convertedSources, gatheredSources,
                         RoundMode::CAST_NONE,
                         gatheredRows * alignedInnerElements);
                    PipeBarrier<PIPE_V>();
                    // Gathered BF16 scratch is now dead and can hold rounding
                    // results; packed input staging must remain unmodified.
                    SetMaskCount();
                    SetVectorMask<float, MaskMode::COUNTER>(
                        0, static_cast<uint64_t>(innerLength_));
                    if (have3 && a0 != a1 && a0 != a2 && a0 != a3 &&
                        a1 != a2 && a1 != a3 && a2 != a3) {
                        UpdateIndependentQuartet(
                            accumulator, convertedSources, gatheredSources,
                            accumulatorRowElements, alignedInnerElements,
                            0U, 1U, 2U, 3U, a0, a1, a2, a3);
                    } else {
                        // Collisions and incomplete quartets retain exact
                        // chronological rounding after every contribution.
                        UpdateOne(accumulator[a0 * accumulatorRowElements],
                                  convertedSources, gatheredSources);
                        if (have1) {
                            UpdateOne(accumulator[a1 * accumulatorRowElements],
                                      convertedSources[alignedInnerElements],
                                      gatheredSources);
                        }
                        if (have2) {
                            UpdateOne(accumulator[a2 * accumulatorRowElements],
                                      convertedSources[2U * alignedInnerElements],
                                      gatheredSources);
                        }
                        if (have3) {
                            UpdateOne(accumulator[a3 * accumulatorRowElements],
                                      convertedSources[3U * alignedInnerElements],
                                      gatheredSources);
                        }
                    }
                }
                PipeBarrier<PIPE_V>();
                SetMaskNorm();
                ResetMask();
                if (useFirstSource) {
                    SetFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(firstSourceReusable));
                    firstSourceInUse = true;
                } else {
                    SetFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(secondSourceReusable));
                    secondSourceInUse = true;
                }
                sourceFirst = nextSourceFirst;
                sourceRows = nextSourceRows;
                useFirstSource = !useFirstSource;
            }
            if (firstSourceInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(firstSourceReusable));
            }
            if (secondSourceInUse) {
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(secondSourceReusable));
            }

            Cast(staging, accumulator, RoundMode::CAST_RINT,
                 paddedGroupElements);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            DataCopyPad(output_[selfOffset], staging, groupCopy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(firstSourceReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(secondSourceReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(firstSourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(secondSourceReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
    }

private:
    __aicore__ inline bool NextOwned(
        LocalTensor<int32_t> indices, uint32_t sourceFirst,
        uint32_t sourceRows, uint32_t firstAxis, uint32_t rowCount,
        uint32_t& cursor, uint32_t& position, uint32_t& localAxis)
    {
        while (cursor < sourceRows) {
            position = cursor++;
            localAxis = static_cast<uint32_t>(
                indices.GetValue(sourceFirst + position)) - firstAxis;
            if (localAxis < rowCount) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void UpdateOne(
        LocalTensor<float> target, LocalTensor<float> source,
        LocalTensor<bfloat16_t> rounded)
    {
        AddRowRaw(target, source);
        PipeBarrier<PIPE_V>();
        RoundRowRaw(rounded, target);
        PipeBarrier<PIPE_V>();
        BackRowRaw(target, rounded);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UpdateIndependentQuartet(
        LocalTensor<float> accumulator, LocalTensor<float> sources,
        LocalTensor<bfloat16_t> scratch, uint32_t rowStride,
        uint32_t sourceStride, uint32_t p0, uint32_t p1, uint32_t p2,
        uint32_t p3, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3)
    {
        LocalTensor<float> t0 = accumulator[a0 * rowStride];
        LocalTensor<float> t1 = accumulator[a1 * rowStride];
        LocalTensor<float> t2 = accumulator[a2 * rowStride];
        LocalTensor<float> t3 = accumulator[a3 * rowStride];
        AddRowRaw(t0, sources[p0 * sourceStride]);
        AddRowRaw(t1, sources[p1 * sourceStride]);
        AddRowRaw(t2, sources[p2 * sourceStride]);
        AddRowRaw(t3, sources[p3 * sourceStride]);
        PipeBarrier<PIPE_V>();
        RoundRowRaw(scratch, t0);
        RoundRowRaw(scratch[rowStride], t1);
        RoundRowRaw(scratch[2U * rowStride], t2);
        RoundRowRaw(scratch[3U * rowStride], t3);
        PipeBarrier<PIPE_V>();
        BackRowRaw(t0, scratch);
        BackRowRaw(t1, scratch[rowStride]);
        BackRowRaw(t2, scratch[2U * rowStride]);
        BackRowRaw(t3, scratch[3U * rowStride]);
        PipeBarrier<PIPE_V>();
    }

    // Direct C220 equivalents of the preceding Level-0 API descriptors.
    // The caller owns the counter mask and proves disjoint quartet rows;
    // explicit phase/collision barriers remain responsible for every RAW.
    __aicore__ inline void AddRowRaw(LocalTensor<float> target,
                                    LocalTensor<float> source)
    {
        auto* dst = reinterpret_cast<__ubuf__ float*>(target.GetPhyAddr());
        auto* src = reinterpret_cast<__ubuf__ float*>(source.GetPhyAddr());
        vadd(dst, dst, src, 1U, 1U, 1U, 1U, 8U, 8U, 8U);
    }

    __aicore__ inline void RoundRowRaw(LocalTensor<bfloat16_t> rounded,
                                      LocalTensor<float> target)
    {
        auto* dst = reinterpret_cast<__ubuf__ bfloat16_t*>(rounded.GetPhyAddr());
        auto* src = reinterpret_cast<__ubuf__ float*>(target.GetPhyAddr());
        vconv_f322bf16r(dst, src, 1U, 1U, 1U, 4U, 8U);
    }

    __aicore__ inline void BackRowRaw(LocalTensor<float> target,
                                     LocalTensor<bfloat16_t> rounded)
    {
        auto* dst = reinterpret_cast<__ubuf__ float*>(target.GetPhyAddr());
        auto* src = reinterpret_cast<__ubuf__ bfloat16_t*>(rounded.GetPhyAddr());
        vconv_bf162f32(dst, src, 1U, 1U, 1U, 8U, 4U);
    }

    __aicore__ inline void CopySourceChunk(
        LocalTensor<bfloat16_t> destination, uint64_t firstSourceOffset,
        uint32_t sourceFirst, uint32_t sourceRows,
        TEventID sourceReady)
    {
        const DataCopyExtParams sourcesCopy{
            1,
            sourceRows * static_cast<uint32_t>(innerLength_) *
                static_cast<uint32_t>(sizeof(bfloat16_t)),
            0, 0, 0};
        DataCopyPad(destination,
                    source_[firstSourceOffset +
                            static_cast<uint64_t>(sourceFirst) *
                                innerLength_],
                    sourcesCopy,
                    DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(sourceReady));
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> stagingBuffer_;
    TBuf<QuePosition::VECCALC> secondSourceStagingBuffer_;
    TBuf<QuePosition::VECCALC> convertedSourceBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    TBuf<QuePosition::VECCALC> gatheredSourceBuffer_;
    TBuf<QuePosition::VECCALC> gatherOffsetBuffer_;
    GlobalTensor<bfloat16_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<bfloat16_t> source_;
    GlobalTensor<bfloat16_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

class IndexAddHalfVnchwOuterBatch {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ / axisLength_;
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(self));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        pipe_.InitBuffer(
            accumulatorBuffer_,
            kVnchwHalfMaxAxisLength * kVnchwHalfOuterBatch * sizeof(half));
        pipe_.InitBuffer(
            matrixBuffer_,
            2U * kVnchwHalfOuterBatch * kVnchwHalfColumnChunk *
                sizeof(half));
        pipe_.InitBuffer(indexBuffer_,
                         kVnchwHalfMaxIndexLength * sizeof(int32_t));
        pipe_.InitBuffer(
            sourceVaConfigBuffer_,
            2U * kVnchwHalfVaConfigWords * sizeof(uint64_t));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t matrixElements =
            kVnchwHalfOuterBatch * kVnchwHalfColumnChunk;
        LocalTensor<half> accumulators = accumulatorBuffer_.Get<half>();
        LocalTensor<half> matrixStorage = matrixBuffer_.Get<half>();
        LocalTensor<half> rowMajorMatrix = matrixStorage;
        LocalTensor<half> columnMajorMatrix =
            matrixStorage[matrixElements];
        LocalTensor<half> sourceRowMajorMatrix = columnMajorMatrix;
        LocalTensor<half> sourceColumnMajorMatrix = rowMajorMatrix;
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();
        LocalTensor<uint64_t> cachedIndexPairs =
            cachedIndices.template ReinterpretCast<uint64_t>();
        LocalTensor<uint64_t> sourceDestinationVaConfig =
            sourceVaConfigBuffer_.Get<uint64_t>();
        LocalTensor<uint64_t> sourceSourceVaConfig =
            sourceDestinationVaConfig[kVnchwHalfVaConfigWords];

        const TEventID indexReady = pipe_.AllocEventID<HardEvent::MTE2_S>();
        const TEventID inputReady = pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID inputReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady = pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputReusable =
            pipe_.AllocEventID<HardEvent::MTE3_V>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const TEventID sourceVaConfigReady =
            pipe_.AllocEventID<HardEvent::S_V>();
        uint32_t outerBatch = static_cast<uint32_t>(
            (outerLength_ + GetBlockNum() - 1U) / GetBlockNum());
        if (outerBatch > kVnchwHalfOuterBatch) {
            outerBatch = kVnchwHalfOuterBatch;
        }
        const uint32_t batchCount = static_cast<uint32_t>(
            (outerLength_ + outerBatch - 1U) / outerBatch);

        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));

        for (uint32_t batch = GetBlockIdx(); batch < batchCount;
             batch += GetBlockNum()) {
            const uint32_t firstOuter = batch * outerBatch;
            const uint32_t activeOuter =
                outerLength_ - firstOuter < outerBatch
                    ? static_cast<uint32_t>(outerLength_ - firstOuter)
                    : outerBatch;
            const uint64_t firstSelfOffset =
                static_cast<uint64_t>(firstOuter) * axisLength_;

            for (uint32_t firstAxis = 0; firstAxis < axisLength_;
                 firstAxis += kVnchwHalfColumnChunk) {
                const uint32_t columnCount =
                    axisLength_ - firstAxis < kVnchwHalfColumnChunk
                        ? axisLength_ - firstAxis
                        : kVnchwHalfColumnChunk;
                CopyRowsToUb(rowMajorMatrix,
                             self_[firstSelfOffset + firstAxis], activeOuter,
                             columnCount, axisLength_);
                SetFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                if (firstAxis + columnCount == axisLength_ &&
                    indexLength_ != 0U) {
                    const uint64_t firstSourceOffset =
                        static_cast<uint64_t>(firstOuter) * indexLength_;
                    CopyRowsToUb(
                        sourceRowMajorMatrix,
                        source_[firstSourceOffset], activeOuter,
                        indexLength_ < kVnchwHalfColumnChunk
                            ? indexLength_
                            : kVnchwHalfColumnChunk,
                        indexLength_);
                    SetFlag<HardEvent::MTE2_V>(
                        static_cast<event_t>(inputReady));
                }
                TransposeOuterToColumns(
                    accumulators[firstAxis * kVnchwHalfOuterBatch],
                    rowMajorMatrix, activeOuter, columnCount);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(inputReusable));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(inputReusable));
            }

            PrepareOuterToColumnsVaConfig(
                sourceDestinationVaConfig, sourceSourceVaConfig,
                sourceColumnMajorMatrix, sourceRowMajorMatrix, activeOuter);
            SetFlag<HardEvent::S_V>(
                static_cast<event_t>(sourceVaConfigReady));
            WaitFlag<HardEvent::S_V>(
                static_cast<event_t>(sourceVaConfigReady));

            for (uint32_t firstUpdate = 0; firstUpdate < indexLength_;
                 firstUpdate += kVnchwHalfColumnChunk) {
                const uint32_t updateCount =
                    indexLength_ - firstUpdate < kVnchwHalfColumnChunk
                        ? indexLength_ - firstUpdate
                        : kVnchwHalfColumnChunk;
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                TransposeOuterToColumnsCached(
                    sourceDestinationVaConfig, sourceSourceVaConfig,
                    updateCount);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(inputReusable));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(inputReusable));
                const uint32_t nextFirstUpdate =
                    firstUpdate + kVnchwHalfColumnChunk;
                if (nextFirstUpdate < indexLength_) {
                    const uint32_t nextUpdateCount =
                        indexLength_ - nextFirstUpdate <
                                kVnchwHalfColumnChunk
                            ? indexLength_ - nextFirstUpdate
                            : kVnchwHalfColumnChunk;
                    const uint64_t nextSourceOffset =
                        static_cast<uint64_t>(firstOuter) * indexLength_ +
                        nextFirstUpdate;
                    CopyRowsToUb(sourceRowMajorMatrix,
                                 source_[nextSourceOffset], activeOuter,
                                 nextUpdateCount, indexLength_);
                    SetFlag<HardEvent::MTE2_V>(
                        static_cast<event_t>(inputReady));
                }
                const uint32_t pairCount = updateCount / 2U;
                const uint32_t firstPair = firstUpdate / 2U;
                const uint32_t octetCount = pairCount / 4U;
#pragma unroll 8
                for (uint32_t octet = 0; octet < octetCount; ++octet) {
                    uint64_t firstPackedIndex =
                        cachedIndexPairs.GetValue(firstPair + 4U * octet);
                    uint64_t secondPackedIndex =
                        cachedIndexPairs.GetValue(firstPair + 4U * octet + 1U);
                    uint64_t thirdPackedIndex =
                        cachedIndexPairs.GetValue(firstPair + 4U * octet + 2U);
                    uint64_t fourthPackedIndex =
                        cachedIndexPairs.GetValue(firstPair + 4U * octet + 3U);
#pragma unroll
                    for (uint32_t lane = 0; lane < 2U; ++lane) {
                        const uint32_t localUpdate = octet * 8U + lane;
                        const uint32_t axis =
                            static_cast<uint32_t>(firstPackedIndex);
                        firstPackedIndex >>= 32U;
                        Add(accumulators[axis * kVnchwHalfOuterBatch],
                            accumulators[axis * kVnchwHalfOuterBatch],
                            sourceColumnMajorMatrix[
                                localUpdate * kVnchwHalfOuterBatch],
                            kVnchwHalfOuterBatch);
                        PipeBarrier<PIPE_V>();
                    }
#pragma unroll
                    for (uint32_t lane = 0; lane < 2U; ++lane) {
                        const uint32_t localUpdate = octet * 8U + 2U + lane;
                        const uint32_t axis =
                            static_cast<uint32_t>(secondPackedIndex);
                        secondPackedIndex >>= 32U;
                        Add(accumulators[axis * kVnchwHalfOuterBatch],
                            accumulators[axis * kVnchwHalfOuterBatch],
                            sourceColumnMajorMatrix[
                                localUpdate * kVnchwHalfOuterBatch],
                            kVnchwHalfOuterBatch);
                        PipeBarrier<PIPE_V>();
                    }
#pragma unroll
                    for (uint32_t lane = 0; lane < 2U; ++lane) {
                        const uint32_t localUpdate = octet * 8U + 4U + lane;
                        const uint32_t axis =
                            static_cast<uint32_t>(thirdPackedIndex);
                        thirdPackedIndex >>= 32U;
                        Add(accumulators[axis * kVnchwHalfOuterBatch],
                            accumulators[axis * kVnchwHalfOuterBatch],
                            sourceColumnMajorMatrix[
                                localUpdate * kVnchwHalfOuterBatch],
                            kVnchwHalfOuterBatch);
                        PipeBarrier<PIPE_V>();
                    }
#pragma unroll
                    for (uint32_t lane = 0; lane < 2U; ++lane) {
                        const uint32_t localUpdate = octet * 8U + 6U + lane;
                        const uint32_t axis =
                            static_cast<uint32_t>(fourthPackedIndex);
                        fourthPackedIndex >>= 32U;
                        Add(accumulators[axis * kVnchwHalfOuterBatch],
                            accumulators[axis * kVnchwHalfOuterBatch],
                            sourceColumnMajorMatrix[
                                localUpdate * kVnchwHalfOuterBatch],
                            kVnchwHalfOuterBatch);
                        PipeBarrier<PIPE_V>();
                    }
                }
                for (uint32_t pair = 4U * octetCount; pair < pairCount;
                     ++pair) {
                    uint64_t packedIndex =
                        cachedIndexPairs.GetValue(firstPair + pair);
#pragma unroll
                    for (uint32_t lane = 0; lane < 2U; ++lane) {
                        const uint32_t localUpdate = pair * 2U + lane;
                        const uint32_t axis =
                            static_cast<uint32_t>(packedIndex);
                        packedIndex >>= 32U;
                        Add(accumulators[axis * kVnchwHalfOuterBatch],
                            accumulators[axis * kVnchwHalfOuterBatch],
                            sourceColumnMajorMatrix[
                                localUpdate * kVnchwHalfOuterBatch],
                            kVnchwHalfOuterBatch);
                        PipeBarrier<PIPE_V>();
                    }
                }
                if ((updateCount & 1U) != 0U) {
                    const uint32_t localUpdate = 2U * pairCount;
                    const uint32_t axis = static_cast<uint32_t>(
                        cachedIndices.GetValue(firstUpdate + localUpdate));
                    Add(accumulators[axis * kVnchwHalfOuterBatch],
                        accumulators[axis * kVnchwHalfOuterBatch],
                        sourceColumnMajorMatrix[
                            localUpdate * kVnchwHalfOuterBatch],
                        kVnchwHalfOuterBatch);
                    PipeBarrier<PIPE_V>();
                }
            }

            for (uint32_t firstAxis = 0; firstAxis < axisLength_;
                 firstAxis += kVnchwHalfColumnChunk) {
                const uint32_t columnCount =
                    axisLength_ - firstAxis < kVnchwHalfColumnChunk
                        ? axisLength_ - firstAxis
                        : kVnchwHalfColumnChunk;
                if (axisLength_ > kVnchwHalfColumnChunk) {
                    TransposeColumnsToOuter<kVnchwHalfMaxAxisLength>(
                        matrixStorage[firstAxis],
                        accumulators[firstAxis * kVnchwHalfOuterBatch],
                        columnCount);
                } else {
                    TransposeColumnsToOuter<kVnchwHalfColumnChunk>(
                        matrixStorage,
                        accumulators[firstAxis * kVnchwHalfOuterBatch],
                        columnCount);
                }
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(
                static_cast<event_t>(outputReady));
            WaitFlag<HardEvent::V_MTE3>(
                static_cast<event_t>(outputReady));
            if (axisLength_ > kVnchwHalfColumnChunk) {
                CopyWholeRowsToGm<kVnchwHalfMaxAxisLength>(
                    output_[firstSelfOffset], matrixStorage, activeOuter,
                    axisLength_);
            } else {
                CopyWholeRowsToGm<kVnchwHalfColumnChunk>(
                    output_[firstSelfOffset], matrixStorage, activeOuter,
                    axisLength_);
            }
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputReusable);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
        pipe_.ReleaseEventID<HardEvent::S_V>(sourceVaConfigReady);
    }

private:
    __aicore__ inline void CopyRowsToUb(
        LocalTensor<half> destination, GlobalTensor<half> source,
        uint32_t rowCount, uint32_t columnCount, uint32_t sourceRowLength)
    {
        const uint32_t columnBytes = columnCount * sizeof(half);
        const uint32_t alignedColumnBytes = (columnBytes + 31U) & ~31U;
        const DataCopyExtParams copy{
            static_cast<uint16_t>(rowCount), columnBytes,
            static_cast<uint32_t>((sourceRowLength - columnCount) *
                                  sizeof(half)),
            static_cast<uint32_t>(
                (kVnchwHalfColumnChunk * sizeof(half) -
                 alignedColumnBytes) /
                32U),
            0};
        DataCopyPad(
            destination, source, copy,
            DataCopyPadExtParams<half>{
                true, 0,
                static_cast<uint8_t>((alignedColumnBytes - columnBytes) /
                                     sizeof(half)),
                static_cast<half>(0)});
    }

    template <uint32_t sourceRowLength>
    __aicore__ inline void CopyWholeRowsToGm(
        GlobalTensor<half> destination, LocalTensor<half> source,
        uint32_t rowCount, uint32_t columnCount)
    {
        const uint32_t columnBytes = columnCount * sizeof(half);
        const uint32_t alignedColumnBytes = (columnBytes + 31U) & ~31U;
        const DataCopyExtParams copy{
            static_cast<uint16_t>(rowCount), columnBytes,
            static_cast<uint32_t>(
                (sourceRowLength * sizeof(half) - alignedColumnBytes) /
                32U),
            0U,
            0};
        DataCopyPad(destination, source, copy);
    }

    __aicore__ inline void TransposeOuterToColumns(
        LocalTensor<half> destination, LocalTensor<half> source,
        uint32_t activeOuter, uint32_t columnCount)
    {
        const uint8_t repeatTimes = static_cast<uint8_t>(
            (columnCount + 15U) / 16U);
        const TransDataTo5HDParams params(
            false, false, repeatTimes, kVnchwHalfOuterBatch, 1);
        uint64_t destinationList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t sourceList[NCHW_CONV_ADDR_LIST_SIZE];
        for (uint32_t firstRow = 0; firstRow < kVnchwHalfOuterBatch;
             firstRow += 16U) {
            for (uint32_t lane = 0; lane < 16U; ++lane) {
                const uint32_t row =
                    firstRow + lane < activeOuter
                        ? firstRow + lane
                        : activeOuter - 1U;
                sourceList[lane] = reinterpret_cast<uint64_t>(
                    source[row * kVnchwHalfColumnChunk].GetPhyAddr());
                destinationList[lane] = reinterpret_cast<uint64_t>(
                    destination[firstRow +
                                lane * kVnchwHalfOuterBatch]
                        .GetPhyAddr());
            }
            TransDataTo5HD<half>(destinationList, sourceList, params);
        }
    }

    __aicore__ inline void PrepareOuterToColumnsVaConfig(
        LocalTensor<uint64_t> destinationConfig,
        LocalTensor<uint64_t> sourceConfig,
        LocalTensor<half> destination, LocalTensor<half> source,
        uint32_t activeOuter)
    {
        const uint64_t destinationBase =
            (reinterpret_cast<uint64_t>(destination.GetPhyAddr()) >> 5U) &
            kVnchwVaAddressMask;
        const uint64_t sourceBase =
            (reinterpret_cast<uint64_t>(source.GetPhyAddr()) >> 5U) &
            kVnchwVaAddressMask;
#pragma unroll
        for (uint32_t group = 0; group < kVnchwHalfOuterGroups; ++group) {
#pragma unroll
            for (uint32_t word = 0; word < kVnchwHalfVaWordsPerGroup;
                 ++word) {
                uint64_t packedDestination = 0U;
                uint64_t packedSource = 0U;
#pragma unroll
                for (uint32_t packedLane = 0; packedLane < 4U;
                     ++packedLane) {
                    const uint32_t lane = word * 4U + packedLane;
                    const uint32_t requestedRow = group * 16U + lane;
                    const uint32_t row = requestedRow < activeOuter
                                             ? requestedRow
                                             : activeOuter - 1U;
                    const uint64_t destinationBlock =
                        (destinationBase + group +
                         lane * kVnchwHalfOuterGroups) &
                        kVnchwVaAddressMask;
                    const uint64_t sourceBlock =
                        (sourceBase +
                         row * (kVnchwHalfColumnChunk / 16U)) &
                        kVnchwVaAddressMask;
                    const uint32_t shift = packedLane * 16U;
                    packedDestination |= destinationBlock << shift;
                    packedSource |= sourceBlock << shift;
                }
                const uint32_t configOffset =
                    group * kVnchwHalfVaWordsPerGroup + word;
                destinationConfig.SetValue(configOffset,
                                           packedDestination);
                sourceConfig.SetValue(configOffset, packedSource);
            }
        }
    }

    __aicore__ inline void TransposeOuterToColumnsCached(
        LocalTensor<uint64_t> destinationConfig,
        LocalTensor<uint64_t> sourceConfig, uint32_t columnCount)
    {
        const uint8_t repeatTimes = static_cast<uint8_t>(
            (columnCount + 15U) / 16U);
#pragma unroll
        for (uint32_t group = 0; group < kVnchwHalfOuterGroups; ++group) {
            const uint32_t configOffset =
                group * kVnchwHalfVaWordsPerGroup;
            __ubuf__ uint64_t* destinationAddress =
                reinterpret_cast<__ubuf__ uint64_t*>(
                    destinationConfig[configOffset].GetPhyAddr());
            __ubuf__ uint64_t* sourceAddress =
                reinterpret_cast<__ubuf__ uint64_t*>(
                    sourceConfig[configOffset].GetPhyAddr());
            vld_va_reg(VA0, destinationAddress, L128);
            vld_va_reg(VA1, destinationAddress, H128);
            vld_va_reg(VA2, sourceAddress, L128);
            vld_va_reg(VA3, sourceAddress, H128);
            scatter_vnchwconv_b16(
                VA0, VA2, repeatTimes, kVnchwHalfOuterBatch, 1);
        }
    }

    template <uint32_t destinationRowLength>
    __aicore__ inline void TransposeColumnsToOuter(
        LocalTensor<half> destination, LocalTensor<half> source,
        uint32_t columnCount)
    {
        const TransDataTo5HDParams params(
            false, false, kVnchwHalfOuterBatch / 16U,
            destinationRowLength, 1);
        uint64_t destinationList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t sourceList[NCHW_CONV_ADDR_LIST_SIZE];
        const uint32_t alignedColumns = (columnCount + 15U) & ~15U;
        for (uint32_t firstColumn = 0; firstColumn < alignedColumns;
             firstColumn += 16U) {
            for (uint32_t lane = 0; lane < 16U; ++lane) {
                const uint32_t column =
                    firstColumn + lane < columnCount
                        ? firstColumn + lane
                        : columnCount - 1U;
                sourceList[lane] = reinterpret_cast<uint64_t>(
                    source[column * kVnchwHalfOuterBatch].GetPhyAddr());
                destinationList[lane] = reinterpret_cast<uint64_t>(
                    destination[firstColumn +
                                lane * destinationRowLength]
                        .GetPhyAddr());
            }
            TransDataTo5HD<half>(destinationList, sourceList, params);
        }
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> matrixBuffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    TBuf<QuePosition::VECCALC> sourceVaConfigBuffer_;
    GlobalTensor<half> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<half> source_;
    GlobalTensor<half> output_;
    uint64_t totalLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

class IndexAddLargeFloatOuterBatch {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ / axisLength_;
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(self));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        pipe_.InitBuffer(accumulatorBuffer_, 165U * 1024U);
        pipe_.InitBuffer(
            stagingBuffer_,
            kLargeFloatOuterBatch * kLargeFloatSourceChunk *
                sizeof(float));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t accumulatorElements =
            kLargeFloatMaxAxisLength * kLargeFloatOuterBatch;
        constexpr uint32_t selfOffsetsOffset = accumulatorElements;
        constexpr uint32_t sourceOffsetsOffset =
            selfOffsetsOffset + kLargeFloatOuterBatch;
        constexpr uint32_t sourceScratchOffset =
            sourceOffsetsOffset + kLargeFloatOuterBatch;
        constexpr uint32_t ioOuterBatch = 8;
        constexpr uint32_t outputRowElements =
            (kLargeFloatMaxAxisLength + 7U) & ~7U;
        constexpr uint32_t axisOffsetsOffset =
            ioOuterBatch * outputRowElements;

        LocalTensor<float> storage = accumulatorBuffer_.Get<float>();
        LocalTensor<float> accumulators = storage;
        LocalTensor<uint32_t> uintStorage =
            storage.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> selfOffsets =
            uintStorage[selfOffsetsOffset];
        LocalTensor<uint32_t> sourceOffsets =
            uintStorage[sourceOffsetsOffset];
        LocalTensor<float> sourceScratch = storage[sourceScratchOffset];
        LocalTensor<float> staged = stagingBuffer_.Get<float>();
        LocalTensor<uint32_t> axisOffsets =
            staged[axisOffsetsOffset].template ReinterpretCast<uint32_t>();

        for (uint32_t lane = 0; lane < kLargeFloatOuterBatch; ++lane) {
            selfOffsets.SetValue(
                lane, lane * axisLength_ * sizeof(float));
            sourceOffsets.SetValue(
                lane, lane * kLargeFloatSourceChunk * sizeof(float));
        }
        const TEventID scalarToVector =
            pipe_.AllocEventID<HardEvent::S_V>();
        SetFlag<HardEvent::S_V>(static_cast<event_t>(scalarToVector));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(scalarToVector));
        pipe_.ReleaseEventID<HardEvent::S_V>(scalarToVector);

        const TEventID inputReady = pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID stagingReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady = pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputStagingReusable =
            pipe_.AllocEventID<HardEvent::MTE3_V>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const uint32_t batchCount = static_cast<uint32_t>(
            (outerLength_ + kLargeFloatOuterBatch - 1U) /
            kLargeFloatOuterBatch);
        const uint32_t blockStride = GetBlockNum();

        for (uint32_t batch = GetBlockIdx(); batch < batchCount;
             batch += blockStride) {
            const uint32_t firstOuter = batch * kLargeFloatOuterBatch;
            const uint32_t activeOuter =
                outerLength_ - firstOuter < kLargeFloatOuterBatch
                    ? static_cast<uint32_t>(outerLength_ - firstOuter)
                    : kLargeFloatOuterBatch;
            const uint64_t firstSelfOffset =
                static_cast<uint64_t>(firstOuter) * axisLength_;
            for (uint32_t firstLane = 0; firstLane < activeOuter;
                 firstLane += ioOuterBatch) {
                const uint32_t laneCount =
                    activeOuter - firstLane < ioOuterBatch
                        ? activeOuter - firstLane
                        : ioOuterBatch;
                const DataCopyExtParams selfCopy{
                    1, laneCount * axisLength_ *
                           static_cast<uint32_t>(sizeof(float)),
                    0, 0, 0};
                DataCopyPad(
                    staged,
                    self_[firstSelfOffset +
                              static_cast<uint64_t>(firstLane) *
                              axisLength_],
                    selfCopy,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                for (uint32_t axis = 0; axis < axisLength_;
                     ++axis) {
                    Gather(
                        accumulators[axis * kLargeFloatOuterBatch +
                                     firstLane],
                        staged, selfOffsets, axis * sizeof(float), laneCount);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(stagingReusable));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(stagingReusable));
            }

            for (uint32_t firstUpdate = 0;
                 firstUpdate < indexLength_;
                 firstUpdate += kLargeFloatSourceChunk) {
                const uint32_t updateCount =
                    indexLength_ - firstUpdate < kLargeFloatSourceChunk
                        ? indexLength_ - firstUpdate
                        : kLargeFloatSourceChunk;
                const uint32_t alignedUpdateBytes =
                    (updateCount * static_cast<uint32_t>(sizeof(float)) +
                     31U) &
                    ~31U;
                const uint32_t destinationStrideBlocks =
                    (kLargeFloatSourceChunk *
                         static_cast<uint32_t>(sizeof(float)) -
                     alignedUpdateBytes) /
                    32U;
                const DataCopyExtParams sourceCopy{
                    static_cast<uint16_t>(activeOuter),
                    updateCount * static_cast<uint32_t>(sizeof(float)),
                    (indexLength_ - updateCount) *
                        static_cast<uint32_t>(sizeof(float)),
                    destinationStrideBlocks, 0};
                const uint64_t firstSourceOffset =
                    static_cast<uint64_t>(firstOuter) *
                        indexLength_ +
                    firstUpdate;
                DataCopyPad(staged, source_[firstSourceOffset], sourceCopy,
                            DataCopyPadExtParams<float>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                for (uint32_t localUpdate = 0; localUpdate < updateCount;
                     ++localUpdate) {
                    Gather(sourceScratch, staged, sourceOffsets,
                           localUpdate * sizeof(float), activeOuter);
                    PipeBarrier<PIPE_V>();
                    const uint32_t axis = static_cast<uint32_t>(
                        index_.GetValue(firstUpdate + localUpdate));
                    Add(accumulators[axis * kLargeFloatOuterBatch],
                        accumulators[axis * kLargeFloatOuterBatch],
                        sourceScratch, activeOuter);
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(stagingReusable));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(stagingReusable));
            }

            for (uint32_t axis = 0; axis < axisLength_; ++axis) {
                axisOffsets.SetValue(
                    axis,
                    axis * kLargeFloatOuterBatch * sizeof(float));
            }
            const TEventID outputOffsetsReady =
                pipe_.AllocEventID<HardEvent::S_V>();
            SetFlag<HardEvent::S_V>(
                static_cast<event_t>(outputOffsetsReady));
            WaitFlag<HardEvent::S_V>(
                static_cast<event_t>(outputOffsetsReady));
            pipe_.ReleaseEventID<HardEvent::S_V>(outputOffsetsReady);
            for (uint32_t firstLane = 0; firstLane < activeOuter;
                 firstLane += ioOuterBatch) {
                const uint32_t laneCount =
                    activeOuter - firstLane < ioOuterBatch
                        ? activeOuter - firstLane
                        : ioOuterBatch;
                for (uint32_t lane = 0; lane < laneCount; ++lane) {
                    Gather(staged[lane * outputRowElements], accumulators,
                           axisOffsets, (firstLane + lane) * sizeof(float),
                           axisLength_);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(
                    static_cast<event_t>(outputReady));
                WaitFlag<HardEvent::V_MTE3>(
                    static_cast<event_t>(outputReady));
                const DataCopyExtParams outputCopy{
                    static_cast<uint16_t>(laneCount),
                    axisLength_ * static_cast<uint32_t>(sizeof(float)), 0, 0,
                    0};
                DataCopyPad(
                    output_[firstSelfOffset +
                                static_cast<uint64_t>(firstLane) *
                                axisLength_],
                    staged, outputCopy);
                if (firstLane + laneCount < activeOuter) {
                    SetFlag<HardEvent::MTE3_V>(
                        static_cast<event_t>(outputStagingReusable));
                    WaitFlag<HardEvent::MTE3_V>(
                        static_cast<event_t>(outputStagingReusable));
                } else {
                    SetFlag<HardEvent::MTE3_MTE2>(
                        static_cast<event_t>(outputComplete));
                    WaitFlag<HardEvent::MTE3_MTE2>(
                        static_cast<event_t>(outputComplete));
                }
            }
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(stagingReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputStagingReusable);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> stagingBuffer_;
    GlobalTensor<float> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<float> source_;
    GlobalTensor<float> output_;
    uint64_t totalLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

template <bool SpecializeFullUpdates = false>
class IndexAddLargeFloatVnchwOuterBatch {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ / axisLength_;
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(self));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        indexPairs_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(index),
                                    (indexLength_ + 1U) / 2U);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        pipe_.InitBuffer(
            accumulatorBuffer_,
            kVnchwFloatAlignedAxisLength * kVnchwFloatOuterBatch *
                sizeof(float));
        pipe_.InitBuffer(
            matrixBuffer_,
            (2U * kVnchwFloatOuterBatch * kVnchwFloatColumnChunk +
             kVnchwFloatMatrixGapElements) * sizeof(float));
        pipe_.InitBuffer(
            sourceVaConfigBuffer_,
            kVnchwFloatVaStorageBytes);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t matrixElements =
            kVnchwFloatOuterBatch * kVnchwFloatColumnChunk;
        LocalTensor<float> accumulators = accumulatorBuffer_.Get<float>();
        LocalTensor<float> matrixStorage = matrixBuffer_.Get<float>();
        LocalTensor<float> rowMajorMatrix = matrixStorage;
        LocalTensor<float> columnMajorMatrix = matrixStorage[
            matrixElements + kVnchwFloatMatrixGapElements];
        LocalTensor<uint32_t> uintAccumulators =
            accumulators.template ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> sourceOffsets = uintAccumulators[
            kVnchwFloatMaxAxisLength * kVnchwFloatOuterBatch];
        LocalTensor<uint64_t> sourceDestinationVaConfig =
            sourceVaConfigBuffer_.Get<uint64_t>();
        LocalTensor<uint64_t> sourceSourceVaConfig =
            sourceDestinationVaConfig[kVnchwFloatVaConfigWords];
        LocalTensor<int32_t> firstCachedIndices =
            sourceVaConfigBuffer_.Get<int32_t>()[
                kVnchwFloatPersistentVaBytes / sizeof(int32_t)];
        LocalTensor<int32_t> secondCachedIndices =
            firstCachedIndices[kVnchwFloatIndexCacheElements];
        LocalTensor<uint64_t> selfSourceVaConfig =
            sourceSourceVaConfig[kVnchwFloatVaConfigWords];
        LocalTensor<uint64_t> secondSelfSourceVaConfig =
            selfSourceVaConfig[kVnchwFloatVaConfigWords];
        LocalTensor<uint64_t> selfDestinationVaConfigs =
            secondSelfSourceVaConfig[kVnchwFloatVaConfigWords];
        const TEventID offsetsReady = pipe_.AllocEventID<HardEvent::S_V>();
        constexpr HardEvent indexConsumerEvent = SpecializeFullUpdates
            ? HardEvent::MTE2_V : HardEvent::MTE2_S;
        const TEventID indicesReady =
            pipe_.AllocEventID<indexConsumerEvent>();
        TEventID indicesScaled = 0;
        if constexpr (SpecializeFullUpdates) {
            indicesScaled = pipe_.AllocEventID<HardEvent::V_S>();
        }

        const TEventID inputReady = pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID inputReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady = pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputReusable =
            pipe_.AllocEventID<HardEvent::MTE3_V>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const uint32_t blockCount = GetBlockNum();
        const uint32_t nominalBatchCount = static_cast<uint32_t>(
            (outerLength_ + kVnchwFloatOuterBatch - 1U) /
            kVnchwFloatOuterBatch);
        uint32_t activeOuterBatch = kVnchwFloatOuterBatch;
        if (nominalBatchCount > blockCount &&
            nominalBatchCount < 2U * blockCount) {
            activeOuterBatch = static_cast<uint32_t>(
                (outerLength_ + 2U * blockCount - 1U) /
                (2U * blockCount));
        }
        const uint32_t batchCount = static_cast<uint32_t>(
            (outerLength_ + activeOuterBatch - 1U) / activeOuterBatch);
        const __gm__ uint64_t* indexPairs =
            reinterpret_cast<const __gm__ uint64_t*>(index_.GetPhyAddr());

        for (uint32_t batch = GetBlockIdx(); batch < batchCount;
             batch += GetBlockNum()) {
            const uint32_t firstOuter = batch * activeOuterBatch;
            const uint32_t activeOuter =
                outerLength_ - firstOuter < activeOuterBatch
                    ? static_cast<uint32_t>(outerLength_ - firstOuter)
                    : activeOuterBatch;
            const uint64_t firstSelfOffset =
                static_cast<uint64_t>(firstOuter) * axisLength_;

            PrepareOuterToColumnsSourceVaConfig(
                selfSourceVaConfig, rowMajorMatrix, activeOuter,
                kVnchwFloatWideColumnChunk);
            PrepareOuterToColumnsSourceVaConfig(
                secondSelfSourceVaConfig,
                rowMajorMatrix[kVnchwFloatColumnChunk], activeOuter,
                kVnchwFloatWideColumnChunk);
            const uint32_t axisChunkCount =
                (axisLength_ + kVnchwFloatColumnChunk - 1U) /
                kVnchwFloatColumnChunk;
            for (uint32_t axisChunk = 0; axisChunk < axisChunkCount;
                 ++axisChunk) {
                PrepareOuterToColumnsDestinationVaConfig(
                    selfDestinationVaConfigs[
                        axisChunk * kVnchwFloatVaConfigWords],
                    accumulators[axisChunk * kVnchwFloatColumnChunk *
                                 kVnchwFloatOuterBatch]);
            }
            SetFlag<HardEvent::S_V>(
                static_cast<event_t>(offsetsReady));
            WaitFlag<HardEvent::S_V>(
                static_cast<event_t>(offsetsReady));

            for (uint32_t firstAxis = 0; firstAxis < axisLength_;
                 firstAxis += kVnchwFloatWideColumnChunk) {
                const uint32_t remainingColumns =
                    axisLength_ - firstAxis;
                const uint32_t firstColumnCount =
                    remainingColumns < kVnchwFloatColumnChunk
                        ? remainingColumns
                        : kVnchwFloatColumnChunk;
                const uint32_t secondColumnCount =
                    remainingColumns - firstColumnCount <
                            kVnchwFloatColumnChunk
                        ? remainingColumns - firstColumnCount
                        : kVnchwFloatColumnChunk;
                CopyRowsToUb(rowMajorMatrix,
                             self_[firstSelfOffset + firstAxis], activeOuter,
                             firstColumnCount + secondColumnCount,
                             axisLength_, kVnchwFloatWideColumnChunk);
                SetFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                const bool useGather =
                    firstColumnCount <= 8U ||
                    (secondColumnCount != 0U &&
                     secondColumnCount <= 8U);
                if (useGather) {
                    for (uint32_t row = 0;
                         row < kVnchwFloatOuterBatch; ++row) {
                        sourceOffsets.SetValue(
                            row,
                            row * kVnchwFloatWideColumnChunk *
                                sizeof(float));
                    }
                    SetFlag<HardEvent::S_V>(
                        static_cast<event_t>(offsetsReady));
                    WaitFlag<HardEvent::S_V>(
                        static_cast<event_t>(offsetsReady));
                }
                if (firstColumnCount <= 8U) {
                    for (uint32_t column = 0;
                         column < firstColumnCount;
                         ++column) {
                        Gather(
                            accumulators[
                                (firstAxis + column) *
                                kVnchwFloatOuterBatch],
                            rowMajorMatrix, sourceOffsets,
                            column * sizeof(float), activeOuter);
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    const uint32_t axisChunk =
                        firstAxis / kVnchwFloatColumnChunk;
                    TransposeOuterToColumnsCached(
                        selfDestinationVaConfigs[
                            axisChunk * kVnchwFloatVaConfigWords],
                        selfSourceVaConfig, firstColumnCount);
                }
                if (secondColumnCount != 0U) {
                    if (secondColumnCount <= 8U) {
                        for (uint32_t column = 0;
                             column < secondColumnCount; ++column) {
                            Gather(
                                accumulators[
                                    (firstAxis +
                                     kVnchwFloatColumnChunk + column) *
                                    kVnchwFloatOuterBatch],
                                rowMajorMatrix[
                                    kVnchwFloatColumnChunk],
                                sourceOffsets,
                                column * sizeof(float), activeOuter);
                            PipeBarrier<PIPE_V>();
                        }
                    } else {
                        const uint32_t secondAxisChunk =
                            firstAxis / kVnchwFloatColumnChunk + 1U;
                        TransposeOuterToColumnsCached(
                            selfDestinationVaConfigs[
                                secondAxisChunk *
                                kVnchwFloatVaConfigWords],
                            secondSelfSourceVaConfig,
                            secondColumnCount);
                    }
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(inputReusable));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(inputReusable));
            }

            PrepareOuterToColumnsVaConfig(
                sourceDestinationVaConfig, sourceSourceVaConfig,
                columnMajorMatrix, rowMajorMatrix, activeOuter);
            SetFlag<HardEvent::S_V>(
                static_cast<event_t>(offsetsReady));
            WaitFlag<HardEvent::S_V>(
                static_cast<event_t>(offsetsReady));

            bool lowSourceVaPairLoaded = true;
            if (indexLength_ > 8U) {
                LoadOuterToColumnsVaPair(
                    sourceDestinationVaConfig, sourceSourceVaConfig, 0U);
            }

            uint32_t cachedFirstUpdate = 0U;
            uint32_t cachedIndexCount =
                indexLength_ < kVnchwFloatIndexCacheElements
                    ? indexLength_
                    : kVnchwFloatIndexCacheElements;
            uint32_t prefetchedIndexCount = 0U;
            uint32_t cachedGeneration = 0U;
            if (indexLength_ > 8U) {
                const DataCopyExtParams indexCopy{
                    1U,
                    cachedIndexCount *
                        static_cast<uint32_t>(sizeof(int32_t)),
                    0U, 0U, 0U};
                DataCopyPad(
                    firstCachedIndices, index_[0U], indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                SetFlag<indexConsumerEvent>(
                    static_cast<event_t>(indicesReady));
            }
            if (indexLength_ != 0U) {
                const uint64_t firstSourceOffset =
                    static_cast<uint64_t>(firstOuter) * indexLength_;
                CopyRowsToUb(rowMajorMatrix,
                             source_[firstSourceOffset], activeOuter,
                             indexLength_ < kVnchwFloatColumnChunk
                                 ? indexLength_
                                 : kVnchwFloatColumnChunk,
                             indexLength_, kVnchwFloatColumnChunk);
                SetFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
            }
            // Full source generations always materialize 64 lanes, including
            // clamped padding lanes in the final outer batch.  Keep the full
            // FP32 mask live so the hot loop can issue the underlying VADD
            // without the level-2 Add API's per-update mask-count setup and
            // restore sequence.  Only active lanes are later copied to GM.
            set_mask_norm();
            constexpr uint64_t outerMask =
                (1ULL << kVnchwFloatOuterBatch) - 1ULL;
            set_vector_mask(0U, outerMask);

            for (uint32_t firstUpdate = 0; firstUpdate < indexLength_;
                 firstUpdate += kVnchwFloatColumnChunk) {
                const uint32_t updateCount =
                    indexLength_ - firstUpdate < kVnchwFloatColumnChunk
                        ? indexLength_ - firstUpdate
                        : kVnchwFloatColumnChunk;
                WaitFlag<HardEvent::MTE2_V>(
                    static_cast<event_t>(inputReady));
                if (updateCount > 8U &&
                    prefetchedIndexCount != 0U &&
                    firstUpdate == cachedFirstUpdate + cachedIndexCount) {
                    cachedFirstUpdate += cachedIndexCount;
                    cachedIndexCount = prefetchedIndexCount;
                    prefetchedIndexCount = 0U;
                    cachedGeneration ^= 1U;
                }
                if (updateCount > 8U &&
                    firstUpdate == cachedFirstUpdate) {
                    WaitFlag<indexConsumerEvent>(
                        static_cast<event_t>(indicesReady));
                    if constexpr (SpecializeFullUpdates) {
                        // Scale each newly active cache window exactly once.
                        // Cached consumers use byte offsets; the <=8-update
                        // GM fallback below still consumes raw axis indices.
                        LocalTensor<int32_t> indicesToScale =
                            cachedGeneration == 0U
                                ? firstCachedIndices : secondCachedIndices;
                        Muls(indicesToScale, indicesToScale,
                             static_cast<int32_t>(
                                 kVnchwFloatOuterBatch * sizeof(float)),
                             static_cast<int32_t>(cachedIndexCount));
                        SetFlag<HardEvent::V_S>(
                            static_cast<event_t>(indicesScaled));
                        WaitFlag<HardEvent::V_S>(
                            static_cast<event_t>(indicesScaled));
                        set_mask_norm();
                        set_vector_mask(0U, outerMask);
                    }
                }
                if (updateCount <= 8U) {
                    if (batch == GetBlockIdx()) {
                        DataCachePreload(
                            indexPairs_,
                            static_cast<int64_t>(firstUpdate) *
                                static_cast<int64_t>(sizeof(int32_t)));
                    }
                    for (uint32_t row = 0;
                         row < kVnchwFloatOuterBatch; ++row) {
                        sourceOffsets.SetValue(
                            row,
                            row * kVnchwFloatColumnChunk * sizeof(float));
                    }
                    SetFlag<HardEvent::S_V>(
                        static_cast<event_t>(offsetsReady));
                    WaitFlag<HardEvent::S_V>(
                        static_cast<event_t>(offsetsReady));
                    const uint32_t pairCount = updateCount / 2U;
                    const uint32_t firstPair = firstUpdate / 2U;
                    uint64_t packedIndex = pairCount == 0U
                                               ? 0U
                                               : indexPairs[firstPair];
                    for (uint32_t pair = 0; pair < pairCount; ++pair) {
                        const uint64_t nextPackedIndex =
                            pair + 1U < pairCount
                                ? indexPairs[firstPair + pair + 1U]
                                : 0U;
#pragma unroll
                        for (uint32_t lane = 0; lane < 2U; ++lane) {
                            const uint32_t localUpdate = pair * 2U + lane;
                            Gather(columnMajorMatrix, rowMajorMatrix,
                                   sourceOffsets,
                                   localUpdate * sizeof(float), activeOuter);
                            PipeBarrier<PIPE_V>();
                            const uint32_t axis =
                                static_cast<uint32_t>(packedIndex);
                            packedIndex >>= 32U;
                            Add(accumulators[
                                    axis * kVnchwFloatOuterBatch],
                                accumulators[
                                    axis * kVnchwFloatOuterBatch],
                                columnMajorMatrix, activeOuter);
                            PipeBarrier<PIPE_V>();
                        }
                        packedIndex = nextPackedIndex;
                    }
                    if ((updateCount & 1U) != 0U) {
                        const uint32_t localUpdate = 2U * pairCount;
                        Gather(columnMajorMatrix, rowMajorMatrix,
                               sourceOffsets,
                               localUpdate * sizeof(float), activeOuter);
                        PipeBarrier<PIPE_V>();
                        const uint32_t axis = static_cast<uint32_t>(
                            index_.GetValue(firstUpdate + localUpdate));
                        Add(accumulators[axis * kVnchwFloatOuterBatch],
                            accumulators[axis * kVnchwFloatOuterBatch],
                            columnMajorMatrix, activeOuter);
                        PipeBarrier<PIPE_V>();
                    }
                    SetFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(inputReusable));
                    WaitFlag<HardEvent::V_MTE2>(
                        static_cast<event_t>(inputReusable));
                    continue;
                }
                TransposeOuterToColumnsLoadedPair(updateCount);
                if (kVnchwFloatOuterGroups > 2U) {
                    const uint32_t nextFirstGroup =
                        lowSourceVaPairLoaded ? 2U : 0U;
                    LoadOuterToColumnsVaPair(
                        sourceDestinationVaConfig, sourceSourceVaConfig,
                        nextFirstGroup);
                    TransposeOuterToColumnsLoadedPair(updateCount);
                    lowSourceVaPairLoaded = !lowSourceVaPairLoaded;
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(inputReusable));
                WaitFlag<HardEvent::V_MTE2>(
                    static_cast<event_t>(inputReusable));
                const uint32_t nextFirstUpdate =
                    firstUpdate + kVnchwFloatColumnChunk;
                if (nextFirstUpdate < indexLength_) {
                    const uint32_t nextUpdateCount =
                        indexLength_ - nextFirstUpdate <
                                kVnchwFloatColumnChunk
                            ? indexLength_ - nextFirstUpdate
                            : kVnchwFloatColumnChunk;
                    const uint64_t nextSourceOffset =
                        static_cast<uint64_t>(firstOuter) * indexLength_ +
                        nextFirstUpdate;
                    CopyRowsToUb(rowMajorMatrix,
                                 source_[nextSourceOffset], activeOuter,
                                 nextUpdateCount, indexLength_,
                                 kVnchwFloatColumnChunk);
                    SetFlag<HardEvent::MTE2_V>(
                        static_cast<event_t>(inputReady));
                }
                if (firstUpdate == cachedFirstUpdate) {
                    const uint32_t nextCachedFirstUpdate =
                        cachedFirstUpdate + cachedIndexCount;
                    const uint32_t remainingIndices =
                        nextCachedFirstUpdate < indexLength_
                            ? indexLength_ - nextCachedFirstUpdate
                            : 0U;
                    if (remainingIndices > 8U) {
                        prefetchedIndexCount =
                            remainingIndices < kVnchwFloatIndexCacheElements
                                ? remainingIndices
                                : kVnchwFloatIndexCacheElements;
                        LocalTensor<int32_t> nextCachedIndices =
                            cachedGeneration == 0U
                                ? secondCachedIndices
                                : firstCachedIndices;
                        const DataCopyExtParams indexCopy{
                            1U,
                            prefetchedIndexCount *
                                static_cast<uint32_t>(sizeof(int32_t)),
                            0U, 0U, 0U};
                        DataCopyPad(
                            nextCachedIndices,
                            index_[nextCachedFirstUpdate], indexCopy,
                            DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                        SetFlag<indexConsumerEvent>(
                            static_cast<event_t>(indicesReady));
                    }
                }
                LocalTensor<int32_t> activeCachedIndices =
                    cachedGeneration == 0U
                        ? firstCachedIndices
                        : secondCachedIndices;
                LocalTensor<uint64_t> activeCachedIndexPairs =
                    activeCachedIndices.template ReinterpretCast<uint64_t>();
                const uint32_t firstPair =
                    (firstUpdate - cachedFirstUpdate) / 2U;
                const __ubuf__ uint64_t* activeCachedIndexPairAddress =
                    reinterpret_cast<const __ubuf__ uint64_t*>(
                        activeCachedIndexPairs[firstPair].GetPhyAddr());
                if constexpr (SpecializeFullUpdates) {
                    if (updateCount == kVnchwFloatColumnChunk) {
                        AccumulateFullFloatGeneration(
                            accumulators, columnMajorMatrix,
                            activeCachedIndexPairAddress);
                        continue;
                    }
                }
                const uint32_t pairCount = updateCount / 2U;
                const uint32_t octetCount = pairCount / 4U;
                for (uint32_t octet = 0; octet < octetCount; ++octet) {
                    const __ubuf__ uint64_t* octetIndexPairs =
                        activeCachedIndexPairAddress + 4U * octet;
                    const uint64_t firstPackedIndex =
                        octetIndexPairs[0];
                    const uint64_t secondPackedIndex =
                        octetIndexPairs[1];
                    const uint64_t thirdPackedIndex =
                        octetIndexPairs[2];
                    const uint64_t fourthPackedIndex =
                        octetIndexPairs[3];
                    const uint32_t firstLocalUpdate = octet * 8U;
                    AddFourIndependentFloatOuter(
                        accumulators,
                        columnMajorMatrix[
                            firstLocalUpdate * kVnchwFloatOuterBatch],
                        static_cast<uint32_t>(firstPackedIndex),
                        static_cast<uint32_t>(firstPackedIndex >> 32U),
                        static_cast<uint32_t>(secondPackedIndex),
                        static_cast<uint32_t>(secondPackedIndex >> 32U));
                    AddFourIndependentFloatOuter(
                        accumulators,
                        columnMajorMatrix[
                            (firstLocalUpdate + 4U) *
                            kVnchwFloatOuterBatch],
                        static_cast<uint32_t>(thirdPackedIndex),
                        static_cast<uint32_t>(thirdPackedIndex >> 32U),
                        static_cast<uint32_t>(fourthPackedIndex),
                        static_cast<uint32_t>(fourthPackedIndex >> 32U));
                }
                for (uint32_t pair = 4U * octetCount; pair < pairCount;
                     ++pair) {
                    uint64_t packedIndex =
                        activeCachedIndexPairs.GetValue(firstPair + pair);
#pragma unroll
                    for (uint32_t lane = 0; lane < 2U; ++lane) {
                        const uint32_t localUpdate = pair * 2U + lane;
                        const uint32_t axis =
                            static_cast<uint32_t>(packedIndex);
                        packedIndex >>= 32U;
                        AddCachedFloatOuter(
                            accumulators, axis,
                            columnMajorMatrix[
                                localUpdate * kVnchwFloatOuterBatch]);
                        PipeBarrier<PIPE_V>();
                    }
                }
                if ((updateCount & 1U) != 0U) {
                    const uint32_t localUpdate = 2U * pairCount;
                    const uint32_t axis = static_cast<uint32_t>(
                        activeCachedIndices.GetValue(
                            firstUpdate - cachedFirstUpdate + localUpdate));
                    AddCachedFloatOuter(
                        accumulators, axis,
                        columnMajorMatrix[
                            localUpdate * kVnchwFloatOuterBatch]);
                    PipeBarrier<PIPE_V>();
                }
            }

            constexpr uint32_t outputColumnChunk =
                2U * kVnchwFloatColumnChunk;
            for (uint32_t firstAxis = 0; firstAxis < axisLength_;
                 firstAxis += outputColumnChunk) {
                const uint32_t remainingColumns =
                    axisLength_ - firstAxis;
                const uint32_t firstColumnCount =
                    remainingColumns < kVnchwFloatColumnChunk
                        ? remainingColumns
                        : kVnchwFloatColumnChunk;
                TransposeColumnsToOuter(
                    rowMajorMatrix,
                    accumulators[firstAxis * kVnchwFloatOuterBatch],
                    firstColumnCount, outputColumnChunk);
                const uint32_t secondColumnCount =
                    remainingColumns - firstColumnCount <
                            kVnchwFloatColumnChunk
                        ? remainingColumns - firstColumnCount
                        : kVnchwFloatColumnChunk;
                if (secondColumnCount != 0U) {
                    TransposeColumnsToOuter(
                        rowMajorMatrix[kVnchwFloatColumnChunk],
                        accumulators[
                            (firstAxis + kVnchwFloatColumnChunk) *
                            kVnchwFloatOuterBatch],
                        secondColumnCount, outputColumnChunk);
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(
                    static_cast<event_t>(outputReady));
                WaitFlag<HardEvent::V_MTE3>(
                    static_cast<event_t>(outputReady));
                const uint32_t outputColumnCount =
                    firstColumnCount + secondColumnCount;
                CopyRowsToGm(
                    output_[firstSelfOffset + firstAxis], rowMajorMatrix,
                    activeOuter, outputColumnCount, outputColumnChunk,
                    axisLength_);
                if (firstAxis + outputColumnCount < axisLength_) {
                    SetFlag<HardEvent::MTE3_V>(
                        static_cast<event_t>(outputReusable));
                    WaitFlag<HardEvent::MTE3_V>(
                        static_cast<event_t>(outputReusable));
                } else {
                    SetFlag<HardEvent::MTE3_MTE2>(
                        static_cast<event_t>(outputComplete));
                    WaitFlag<HardEvent::MTE3_MTE2>(
                        static_cast<event_t>(outputComplete));
                }
            }
        }

        pipe_.ReleaseEventID<HardEvent::S_V>(offsetsReady);
        pipe_.ReleaseEventID<indexConsumerEvent>(indicesReady);
        if constexpr (SpecializeFullUpdates) {
            pipe_.ReleaseEventID<HardEvent::V_S>(indicesScaled);
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputReusable);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
    }

private:
    __aicore__ inline void AccumulateFullFloatGeneration(
        LocalTensor<float> accumulators, LocalTensor<float> columns,
        const __ubuf__ uint64_t* cachedIndexPairs)
    {
        constexpr uint32_t pairCount =
            kVnchwFloatColumnChunk / 2U;
        constexpr uint32_t octetCount = pairCount / 4U;
        for (uint32_t octet = 0U; octet < octetCount; ++octet) {
            const __ubuf__ uint64_t* octetIndexPairs =
                cachedIndexPairs + 4U * octet;
            const uint64_t firstPackedIndex = octetIndexPairs[0U];
            const uint64_t secondPackedIndex = octetIndexPairs[1U];
            const uint64_t thirdPackedIndex = octetIndexPairs[2U];
            const uint64_t fourthPackedIndex = octetIndexPairs[3U];
            const uint32_t firstLocalUpdate = octet * 8U;
            AddFourIndependentFloatOuter(
                accumulators,
                columns[firstLocalUpdate * kVnchwFloatOuterBatch],
                static_cast<uint32_t>(firstPackedIndex),
                static_cast<uint32_t>(firstPackedIndex >> 32U),
                static_cast<uint32_t>(secondPackedIndex),
                static_cast<uint32_t>(secondPackedIndex >> 32U));
            AddFourIndependentFloatOuter(
                accumulators,
                columns[(firstLocalUpdate + 4U) *
                        kVnchwFloatOuterBatch],
                static_cast<uint32_t>(thirdPackedIndex),
                static_cast<uint32_t>(thirdPackedIndex >> 32U),
                static_cast<uint32_t>(fourthPackedIndex),
                static_cast<uint32_t>(fourthPackedIndex >> 32U));
        }
    }

    __aicore__ inline void AddFourIndependentFloatOuter(
        LocalTensor<float> accumulators, LocalTensor<float> columns,
        uint32_t firstAxis, uint32_t secondAxis, uint32_t thirdAxis,
        uint32_t fourthAxis)
    {
        const bool hasCollision =
            firstAxis == secondAxis || firstAxis == thirdAxis ||
            firstAxis == fourthAxis || secondAxis == thirdAxis ||
            secondAxis == fourthAxis || thirdAxis == fourthAxis;
        // Byte scaling is injective for valid indices, so the collision
        // predicate and every chronological reduction barrier are unchanged.
        AddCachedFloatOuter(accumulators, firstAxis, columns);
        if (hasCollision) {
            PipeBarrier<PIPE_V>();
        }
        AddCachedFloatOuter(
            accumulators, secondAxis,
            columns[kVnchwFloatOuterBatch]);
        if (hasCollision) {
            PipeBarrier<PIPE_V>();
        }
        AddCachedFloatOuter(
            accumulators, thirdAxis,
            columns[2U * kVnchwFloatOuterBatch]);
        if (hasCollision) {
            PipeBarrier<PIPE_V>();
        }
        AddCachedFloatOuter(
            accumulators, fourthAxis,
            columns[3U * kVnchwFloatOuterBatch]);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddCachedFloatOuter(
        LocalTensor<float> accumulators, uint32_t axisOrByteOffset,
        LocalTensor<float> source)
    {
        if constexpr (SpecializeFullUpdates) {
            __ubuf__ uint8_t* accumulatorBytes =
                reinterpret_cast<__ubuf__ uint8_t*>(
                    accumulators.GetPhyAddr());
            __ubuf__ float* destination = reinterpret_cast<__ubuf__ float*>(
                accumulatorBytes + axisOrByteOffset);
            __ubuf__ float* sourceAddress =
                reinterpret_cast<__ubuf__ float*>(source.GetPhyAddr());
            vadd(destination, destination, sourceAddress,
                 1U, 1U, 1U, 1U, 8U, 8U, 8U);
        } else {
            AddFullFloatOuter(
                accumulators[axisOrByteOffset * kVnchwFloatOuterBatch],
                source);
        }
    }

    __aicore__ inline void AddFullFloatOuter(
        LocalTensor<float> accumulator, LocalTensor<float> source)
    {
        __ubuf__ float* accumulatorAddress =
            reinterpret_cast<__ubuf__ float*>(
                accumulator.GetPhyAddr());
        __ubuf__ float* sourceAddress =
            reinterpret_cast<__ubuf__ float*>(source.GetPhyAddr());
        vadd(accumulatorAddress, accumulatorAddress, sourceAddress,
             1U, 1U, 1U, 1U, 8U, 8U, 8U);
    }

    __aicore__ inline void CopyRowsToUb(
        LocalTensor<float> destination, GlobalTensor<float> source,
        uint32_t rowCount, uint32_t columnCount, uint32_t sourceRowLength,
        uint32_t destinationRowLength)
    {
        const uint32_t columnBytes = columnCount * sizeof(float);
        const uint32_t alignedColumnBytes = (columnBytes + 31U) & ~31U;
        const DataCopyExtParams copy{
            static_cast<uint16_t>(rowCount), columnBytes,
            static_cast<uint32_t>((sourceRowLength - columnCount) *
                                  sizeof(float)),
            static_cast<uint32_t>(
                (destinationRowLength * sizeof(float) -
                 alignedColumnBytes) /
                32U),
            0};
        DataCopyPad(destination, source, copy,
                    DataCopyPadExtParams<float>{false, 0, 0, 0});
    }

    __aicore__ inline void CopyRowsToGm(
        GlobalTensor<float> destination, LocalTensor<float> source,
        uint32_t rowCount, uint32_t columnCount,
        uint32_t sourceRowLength,
        uint32_t destinationRowLength)
    {
        const uint32_t columnBytes = columnCount * sizeof(float);
        const uint32_t alignedColumnBytes = (columnBytes + 31U) & ~31U;
        const DataCopyExtParams copy{
            static_cast<uint16_t>(rowCount), columnBytes,
            static_cast<uint32_t>(
                (sourceRowLength * sizeof(float) -
                 alignedColumnBytes) /
                32U),
            static_cast<uint32_t>((destinationRowLength - columnCount) *
                                  sizeof(float)),
            0};
        DataCopyPad(destination, source, copy);
    }

    __aicore__ inline void TransposeOuterToColumns(
        LocalTensor<float> destination, LocalTensor<float> source,
        uint32_t activeOuter, uint32_t columnCount)
    {
        const uint8_t repeatTimes = static_cast<uint8_t>(
            (columnCount + 7U) / 8U);
        const TransDataTo5HDParams params(
            false, false, repeatTimes, kVnchwFloatOuterBatch, 1);
        uint64_t destinationList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t sourceList[NCHW_CONV_ADDR_LIST_SIZE];
        for (uint32_t firstRow = 0;
             firstRow < kVnchwFloatOuterBatch; firstRow += 16U) {
            for (uint32_t lane = 0; lane < 16U; ++lane) {
                const uint32_t row =
                    firstRow + lane < activeOuter
                        ? firstRow + lane
                        : activeOuter - 1U;
                sourceList[lane] = reinterpret_cast<uint64_t>(
                    source[row * kVnchwFloatColumnChunk].GetPhyAddr());
            }
            for (uint32_t column = 0; column < 8U; ++column) {
                destinationList[2U * column] = reinterpret_cast<uint64_t>(
                    destination[firstRow +
                                column * kVnchwFloatOuterBatch]
                        .GetPhyAddr());
                destinationList[2U * column + 1U] =
                    reinterpret_cast<uint64_t>(
                        destination[firstRow + 8U +
                                    column * kVnchwFloatOuterBatch]
                            .GetPhyAddr());
            }
            TransDataTo5HD<float>(destinationList, sourceList, params);
        }
    }

    __aicore__ inline void PrepareOuterToColumnsVaConfig(
        LocalTensor<uint64_t> destinationConfig,
        LocalTensor<uint64_t> sourceConfig,
        LocalTensor<float> destination, LocalTensor<float> source,
        uint32_t activeOuter)
    {
        const uint64_t destinationBase =
            (reinterpret_cast<uint64_t>(destination.GetPhyAddr()) >> 5U) &
            kVnchwVaAddressMask;
        const uint64_t sourceBase =
            (reinterpret_cast<uint64_t>(source.GetPhyAddr()) >> 5U) &
            kVnchwVaAddressMask;
#pragma unroll
        for (uint32_t group = 0; group < kVnchwFloatOuterGroups; ++group) {
#pragma unroll
            for (uint32_t word = 0; word < kVnchwFloatVaWordsPerGroup;
                 ++word) {
                uint64_t packedDestination = 0U;
                uint64_t packedSource = 0U;
#pragma unroll
                for (uint32_t packedLane = 0; packedLane < 4U;
                     ++packedLane) {
                    const uint32_t addressIndex =
                        word * 4U + packedLane;
                    const uint32_t requestedRow =
                        group * 16U + addressIndex;
                    const uint32_t row = requestedRow < activeOuter
                                             ? requestedRow
                                             : activeOuter - 1U;
                    const uint64_t destinationBlock =
                        (destinationBase + group * 2U +
                         (addressIndex & 1U) +
                         (addressIndex >> 1U) *
                             (kVnchwFloatOuterBatch / 8U)) &
                        kVnchwVaAddressMask;
                    const uint64_t sourceBlock =
                        (sourceBase +
                         row * (kVnchwFloatColumnChunk / 8U)) &
                        kVnchwVaAddressMask;
                    const uint32_t shift = packedLane * 16U;
                    packedDestination |= destinationBlock << shift;
                    packedSource |= sourceBlock << shift;
                }
                const uint32_t configOffset =
                    group * kVnchwFloatVaWordsPerGroup + word;
                destinationConfig.SetValue(configOffset,
                                           packedDestination);
                sourceConfig.SetValue(configOffset, packedSource);
            }
        }
    }

    __aicore__ inline void PrepareOuterToColumnsDestinationVaConfig(
        LocalTensor<uint64_t> destinationConfig,
        LocalTensor<float> destination)
    {
        const uint64_t destinationBase =
            (reinterpret_cast<uint64_t>(destination.GetPhyAddr()) >> 5U) &
            kVnchwVaAddressMask;
#pragma unroll
        for (uint32_t group = 0; group < kVnchwFloatOuterGroups; ++group) {
#pragma unroll
            for (uint32_t word = 0; word < kVnchwFloatVaWordsPerGroup;
                 ++word) {
                uint64_t packedDestination = 0U;
#pragma unroll
                for (uint32_t packedLane = 0; packedLane < 4U;
                     ++packedLane) {
                    const uint32_t addressIndex =
                        word * 4U + packedLane;
                    const uint64_t destinationBlock =
                        (destinationBase + group * 2U +
                         (addressIndex & 1U) +
                         (addressIndex >> 1U) *
                             (kVnchwFloatOuterBatch / 8U)) &
                        kVnchwVaAddressMask;
                    packedDestination |=
                        destinationBlock << (packedLane * 16U);
                }
                destinationConfig.SetValue(
                    group * kVnchwFloatVaWordsPerGroup + word,
                    packedDestination);
            }
        }
    }

    __aicore__ inline void PrepareOuterToColumnsSourceVaConfig(
        LocalTensor<uint64_t> sourceConfig, LocalTensor<float> source,
        uint32_t activeOuter, uint32_t sourceRowLength)
    {
        const uint64_t sourceBase =
            (reinterpret_cast<uint64_t>(source.GetPhyAddr()) >> 5U) &
            kVnchwVaAddressMask;
#pragma unroll
        for (uint32_t group = 0; group < kVnchwFloatOuterGroups; ++group) {
#pragma unroll
            for (uint32_t word = 0; word < kVnchwFloatVaWordsPerGroup;
                 ++word) {
                uint64_t packedSource = 0U;
#pragma unroll
                for (uint32_t packedLane = 0; packedLane < 4U;
                     ++packedLane) {
                    const uint32_t addressIndex =
                        word * 4U + packedLane;
                    const uint32_t requestedRow =
                        group * 16U + addressIndex;
                    const uint32_t row = requestedRow < activeOuter
                                             ? requestedRow
                                             : activeOuter - 1U;
                    const uint64_t sourceBlock =
                        (sourceBase +
                         row * (sourceRowLength / 8U)) &
                        kVnchwVaAddressMask;
                    packedSource |=
                        sourceBlock << (packedLane * 16U);
                }
                sourceConfig.SetValue(
                    group * kVnchwFloatVaWordsPerGroup + word,
                    packedSource);
            }
        }
    }

    __aicore__ inline void LoadOuterToColumnsVaPair(
        LocalTensor<uint64_t> destinationConfig,
        LocalTensor<uint64_t> sourceConfig, uint32_t firstGroup)
    {
        const uint32_t firstConfigOffset =
            firstGroup * kVnchwFloatVaWordsPerGroup;
        const uint32_t secondConfigOffset =
            (firstGroup + 1U) * kVnchwFloatVaWordsPerGroup;
        __ubuf__ uint64_t* firstDestinationAddress =
            reinterpret_cast<__ubuf__ uint64_t*>(
                destinationConfig[firstConfigOffset].GetPhyAddr());
        __ubuf__ uint64_t* firstSourceAddress =
            reinterpret_cast<__ubuf__ uint64_t*>(
                sourceConfig[firstConfigOffset].GetPhyAddr());
        __ubuf__ uint64_t* secondDestinationAddress =
            reinterpret_cast<__ubuf__ uint64_t*>(
                destinationConfig[secondConfigOffset].GetPhyAddr());
        __ubuf__ uint64_t* secondSourceAddress =
            reinterpret_cast<__ubuf__ uint64_t*>(
                sourceConfig[secondConfigOffset].GetPhyAddr());
        vld_va_reg(VA0, firstDestinationAddress, L128);
        vld_va_reg(VA1, firstDestinationAddress, H128);
        vld_va_reg(VA2, firstSourceAddress, L128);
        vld_va_reg(VA3, firstSourceAddress, H128);
        vld_va_reg(VA4, secondDestinationAddress, L128);
        vld_va_reg(VA5, secondDestinationAddress, H128);
        vld_va_reg(VA6, secondSourceAddress, L128);
        vld_va_reg(VA7, secondSourceAddress, H128);
    }

    __aicore__ inline void TransposeOuterToColumnsLoadedPair(
        uint32_t columnCount)
    {
        const uint8_t repeatTimes = static_cast<uint8_t>(
            (columnCount + 7U) / 8U);
        scatter_vnchwconv_b32(
            VA0, VA2, repeatTimes, kVnchwFloatOuterBatch, 1U);
        scatter_vnchwconv_b32(
            VA4, VA6, repeatTimes, kVnchwFloatOuterBatch, 1U);
    }

    __aicore__ inline void TransposeOuterToColumnsCached(
        LocalTensor<uint64_t> destinationConfig,
        LocalTensor<uint64_t> sourceConfig, uint32_t columnCount)
    {
        const uint8_t repeatTimes = static_cast<uint8_t>(
            (columnCount + 7U) / 8U);
#pragma unroll
        for (uint32_t group = 0; group < kVnchwFloatOuterGroups; ++group) {
            const uint32_t configOffset =
                group * kVnchwFloatVaWordsPerGroup;
            __ubuf__ uint64_t* destinationAddress =
                reinterpret_cast<__ubuf__ uint64_t*>(
                    destinationConfig[configOffset].GetPhyAddr());
            __ubuf__ uint64_t* sourceAddress =
                reinterpret_cast<__ubuf__ uint64_t*>(
                    sourceConfig[configOffset].GetPhyAddr());
            vld_va_reg(VA0, destinationAddress, L128);
            vld_va_reg(VA1, destinationAddress, H128);
            vld_va_reg(VA2, sourceAddress, L128);
            vld_va_reg(VA3, sourceAddress, H128);
            scatter_vnchwconv_b32(
                VA0, VA2, repeatTimes, kVnchwFloatOuterBatch, 1U);
        }
    }

    __aicore__ inline void TransposeColumnsToOuter(
        LocalTensor<float> destination, LocalTensor<float> source,
        uint32_t columnCount, uint32_t destinationRowLength)
    {
        const TransDataTo5HDParams params(
            false, false, kVnchwFloatOuterBatch / 8U,
            destinationRowLength, 1);
        uint64_t destinationList[NCHW_CONV_ADDR_LIST_SIZE];
        uint64_t sourceList[NCHW_CONV_ADDR_LIST_SIZE];
        const uint32_t alignedColumns = (columnCount + 15U) & ~15U;
        for (uint32_t firstColumn = 0; firstColumn < alignedColumns;
             firstColumn += 16U) {
            for (uint32_t lane = 0; lane < 16U; ++lane) {
                const uint32_t column =
                    firstColumn + lane < columnCount
                        ? firstColumn + lane
                        : columnCount - 1U;
                sourceList[lane] = reinterpret_cast<uint64_t>(
                    source[column * kVnchwFloatOuterBatch].GetPhyAddr());
            }
            for (uint32_t row = 0; row < 8U; ++row) {
                destinationList[2U * row] = reinterpret_cast<uint64_t>(
                    destination[firstColumn +
                                row * destinationRowLength]
                        .GetPhyAddr());
                destinationList[2U * row + 1U] =
                    reinterpret_cast<uint64_t>(
                        destination[firstColumn + 8U +
                                    row * destinationRowLength]
                            .GetPhyAddr());
            }
            TransDataTo5HD<float>(destinationList, sourceList, params);
        }
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> matrixBuffer_;
    TBuf<QuePosition::VECCALC> sourceVaConfigBuffer_;
    GlobalTensor<float> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<uint64_t> indexPairs_;
    GlobalTensor<float> source_;
    GlobalTensor<float> output_;
    uint64_t totalLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};

class IndexAddSmallInt8AxisGroups {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                GM_ADDR output,
                                const IndexAddTilingData& tilingData)
    {
        totalLength_ = tilingData.totalLength;
        innerLength_ = tilingData.innerLength;
        axisLength_ = tilingData.axisLength;
        indexLength_ = tilingData.indexLength;
        outerLength_ = totalLength_ /
                       (static_cast<uint64_t>(axisLength_) * innerLength_);
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(self));
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),
                               indexLength_);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(source));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(output));
        pipe_.InitBuffer(
            sourceStagingBuffer_,
            kSmallInt8MaxIndexLength * kSmallInt8MaxInnerLength);
        pipe_.InitBuffer(
            rowStagingBuffer_,
            kSmallInt8AxisGroupRows * kSmallInt8MaxInnerLength);
        pipe_.InitBuffer(
            halfRowsBuffer_,
            kSmallInt8AxisGroupRows * kSmallInt8MaxInnerLength *
                sizeof(half));
        pipe_.InitBuffer(
            accumulatorBuffer_,
            kSmallInt8AxisGroupRows * kSmallInt8MaxInnerLength *
                sizeof(int16_t));
        pipe_.InitBuffer(sourceHalfBuffer_,
                         kSmallInt8MaxInnerLength * sizeof(half));
        pipe_.InitBuffer(sourceInt16Buffer_,
                         kSmallInt8MaxInnerLength * sizeof(int16_t));
        pipe_.InitBuffer(indexBuffer_,
                         kSmallInt8MaxIndexLength * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int8_t> stagedSource =
            sourceStagingBuffer_.Get<int8_t>();
        LocalTensor<int8_t> stagedRows = rowStagingBuffer_.Get<int8_t>();
        LocalTensor<half> halfRows = halfRowsBuffer_.Get<half>();
        LocalTensor<int16_t> accumulators =
            accumulatorBuffer_.Get<int16_t>();
        LocalTensor<half> sourceHalf = sourceHalfBuffer_.Get<half>();
        LocalTensor<int16_t> sourceInt16 =
            sourceInt16Buffer_.Get<int16_t>();
        LocalTensor<int32_t> cachedIndices = indexBuffer_.Get<int32_t>();

        const TEventID indexReady = pipe_.AllocEventID<HardEvent::MTE2_S>();
        const DataCopyExtParams indexCopy{
            1, indexLength_ * static_cast<uint32_t>(sizeof(int32_t)), 0, 0,
            0};
        DataCopyPad(cachedIndices, index_, indexCopy,
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        WaitFlag<HardEvent::MTE2_S>(static_cast<event_t>(indexReady));
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(indexReady);

        const TEventID inputReady = pipe_.AllocEventID<HardEvent::MTE2_V>();
        const TEventID inputReusable =
            pipe_.AllocEventID<HardEvent::V_MTE2>();
        const TEventID outputReady = pipe_.AllocEventID<HardEvent::V_MTE3>();
        const TEventID outputComplete =
            pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
        const uint32_t groupsPerOuter =
            (axisLength_ + kSmallInt8AxisGroupRows - 1U) /
            kSmallInt8AxisGroupRows;
        const uint64_t workCount = outerLength_ * groupsPerOuter;
        const uint32_t alignedInnerLength =
            (static_cast<uint32_t>(innerLength_) + 31U) & ~31U;

        for (uint64_t work = GetBlockIdx(); work < workCount;
             work += GetBlockNum()) {
            const uint64_t outer = work / groupsPerOuter;
            const uint32_t group =
                static_cast<uint32_t>(work % groupsPerOuter);
            const uint32_t firstAxis = group * kSmallInt8AxisGroupRows;
            const uint32_t rowCount =
                axisLength_ - firstAxis < kSmallInt8AxisGroupRows
                    ? axisLength_ - firstAxis
                    : kSmallInt8AxisGroupRows;
            const uint32_t paddedRowElements =
                rowCount * alignedInnerLength;
            const uint64_t selfOffset =
                (outer * axisLength_ + firstAxis) * innerLength_;
            const DataCopyExtParams rowCopy{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(innerLength_), 0, 0, 0};
            DataCopyPad(stagedRows, self_[selfOffset], rowCopy,
                        DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
            const uint64_t firstSourceOffset =
                outer * indexLength_ * innerLength_;
            const DataCopyExtParams sourceCopy{
                static_cast<uint16_t>(indexLength_),
                static_cast<uint32_t>(innerLength_), 0, 0, 0};
            DataCopyPad(stagedSource, source_[firstSourceOffset], sourceCopy,
                        DataCopyPadExtParams<int8_t>{false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(inputReady));
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(inputReady));
            Cast(halfRows, stagedRows, RoundMode::CAST_NONE,
                 paddedRowElements);
            PipeBarrier<PIPE_V>();
            Cast(accumulators, halfRows, RoundMode::CAST_RINT,
                 paddedRowElements);
            PipeBarrier<PIPE_V>();

            for (uint32_t update = 0; update < indexLength_; ++update) {
                const uint32_t axis =
                    static_cast<uint32_t>(cachedIndices.GetValue(update));
                const uint32_t localAxis = axis - firstAxis;
                if (localAxis >= rowCount) {
                    continue;
                }
                Cast(sourceHalf,
                     stagedSource[update * alignedInnerLength],
                     RoundMode::CAST_NONE,
                     static_cast<uint32_t>(innerLength_));
                PipeBarrier<PIPE_V>();
                Cast(sourceInt16, sourceHalf, RoundMode::CAST_RINT,
                     static_cast<uint32_t>(innerLength_));
                PipeBarrier<PIPE_V>();
                Add(accumulators[localAxis * alignedInnerLength],
                    accumulators[localAxis * alignedInnerLength],
                    sourceInt16, static_cast<uint32_t>(innerLength_));
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE2>(
                static_cast<event_t>(inputReusable));
            WaitFlag<HardEvent::V_MTE2>(
                static_cast<event_t>(inputReusable));

            ShiftLeft(accumulators, accumulators, static_cast<int16_t>(8),
                      paddedRowElements);
            PipeBarrier<PIPE_V>();
            ShiftRight(accumulators, accumulators, static_cast<int16_t>(8),
                       paddedRowElements);
            PipeBarrier<PIPE_V>();
            Cast(halfRows, accumulators, RoundMode::CAST_NONE,
                 paddedRowElements);
            PipeBarrier<PIPE_V>();
            Cast(stagedRows, halfRows, RoundMode::CAST_RINT,
                 paddedRowElements);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(outputReady));
            DataCopyPad(output_[selfOffset], stagedRows, rowCopy);
            SetFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
            WaitFlag<HardEvent::MTE3_MTE2>(
                static_cast<event_t>(outputComplete));
        }

        pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady);
        pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputReusable);
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady);
        pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(outputComplete);
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> sourceStagingBuffer_;
    TBuf<QuePosition::VECCALC> rowStagingBuffer_;
    TBuf<QuePosition::VECCALC> halfRowsBuffer_;
    TBuf<QuePosition::VECCALC> accumulatorBuffer_;
    TBuf<QuePosition::VECCALC> sourceHalfBuffer_;
    TBuf<QuePosition::VECCALC> sourceInt16Buffer_;
    TBuf<QuePosition::VECCALC> indexBuffer_;
    GlobalTensor<int8_t> self_;
    GlobalTensor<int32_t> index_;
    GlobalTensor<int8_t> source_;
    GlobalTensor<int8_t> output_;
    uint64_t totalLength_;
    uint64_t innerLength_;
    uint64_t outerLength_;
    uint32_t axisLength_;
    uint32_t indexLength_;
};
#include "index_add_half_prefix.h"
}  // namespace

extern "C" __global__ __aicore__ void index_add(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(11)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, false, false,
                               false, true, true, false, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(15)) {
        IndexAddWholeInt32Row<> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(16)) {
        IndexAddWholeInt32Row<5, 6144> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(25)) {
        IndexAddWholeInt32Row<4, 6144> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(26)) {
        IndexAddFlatWholeInt32Rows op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(29)) {
        IndexAddColumnOwnedInt32 op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(30)) {
        IndexAddCoalescedAtomicInt32<false> op;
        op.Init(self, index, source, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(33)) {
        IndexAddCoalescedAtomicInt32<true> op;
        op.Init(self, index, source, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(17)) {
        IndexAddAtomicInt32 op;
        op.Init(self, index, source, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(18)) {
        IndexAddWideInt32AxisGroups op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(19)) {
        if constexpr (IsSameType<DTYPE_SELF, bfloat16_t>::value) {
            IndexAddBfloat16AxisGroups op;
            op.Init(self, index, source, output, tilingData);
            op.Process();
        } else {
            IndexAddLegacyBfloat16AxisGroups op;
            op.Init(self, index, source, output, tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(20)) {
        IndexAddLargeFloatOuterBatch op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(21)) {
        IndexAddSmallInt8AxisGroups op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(22)) {
        IndexAddLargeFloatVnchwOuterBatch op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(35)) {
        IndexAddLargeFloatVnchwOuterBatch<true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(23)) {
        if constexpr (IsSameType<DTYPE_SELF, half>::value) {
            bool handled = false;
            {
                TPipe pipe;
                IndexAddHalfSourcePrefetchPrefix op;
                op.Init(self, index, source, output, tilingData, &pipe);
                handled = op.Prepare();
                if (handled) {
                    op.Process();
                }
            }
            if (!handled) {
                IndexAddHalfVnchwOuterBatch fallback;
                fallback.Init(self, index, source, output, tilingData);
                fallback.Process();
            }
        } else {
            IndexAddHalfVnchwOuterBatch op;
            op.Init(self, index, source, output, tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(24)) {
        IndexAddWideInt32InnerTiles op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(14)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, false, false,
                               false, false, false, false, false, 256> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(12)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, false, false,
                               false, false, false, false, false, 32> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(13)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, false, false,
                               false, false, false, false, false, 16> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, false, false, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(8)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, false, false, false, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(9)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, false, false, false, true, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, false, false, false, true, true, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, false, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, false, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, false, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        IndexAddVectorBaseline<DTYPE_SELF, false, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        IndexAddVectorBaseline<DTYPE_SELF, true> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        IndexAddVectorBaseline<DTYPE_SELF> op;
        op.Init(self, index, source, output, tilingData);
        op.Process();
    }
}
