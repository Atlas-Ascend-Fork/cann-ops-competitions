
#define S9_TINY_TYPE_half 1
#define S9_TINY_TYPE_float 0
#define S9_TINY_TYPE_int8_t 0
#define S9_TINY_TYPE_int32_t 0
#define S9_TINY_TYPE_bfloat16_t 0
#define S9_HAS_TINY S9_ROW_JOIN(S9_TINY_TYPE_,DTYPE_X1)

#define S9_DENSE_TYPE_float 1
#define S9_DENSE_TYPE_half 0
#define S9_DENSE_TYPE_int8_t 1
#define S9_DENSE_TYPE_int32_t 0
#define S9_DENSE_TYPE_bfloat16_t 0
#define S9_HAS_DENSE S9_ROW_JOIN(S9_DENSE_TYPE_,DTYPE_X1)

#define S9_ROW_TYPE_float 0
#define S9_ROW_TYPE_half 0
#define S9_ROW_TYPE_int8_t 0
#define S9_ROW_TYPE_int32_t 1
#define S9_ROW_TYPE_bfloat16_t 1
#define S9_ROW_JOIN_(a,b) a##b
#define S9_ROW_JOIN(a,b) S9_ROW_JOIN_(a,b)
#define S9_HAS_ROW S9_ROW_JOIN(S9_ROW_TYPE_,DTYPE_X1)
#include <type_traits>

#include "kernel_operator.h"
#include "greater_tiling.h"

using namespace AscendC;

namespace {
constexpr uint64_t kMaxShortBroadcastCacheBytes = 32 * 1024;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}


__aicore__ inline void GreaterIndexRange(LocalTensor<int32_t> dst, int32_t first, uint32_t count)
{
    if(count==0)return;
    const uint32_t seed=count<8?count:8;
    CreateVecIndex(dst,first,seed);PipeBarrier<PIPE_V>();
    uint32_t filled=seed;
    while(filled<count){
        const uint32_t step=filled<count-filled?filled:count-filled;
        Adds(dst[filled],dst,static_cast<int32_t>(filled),step);PipeBarrier<PIPE_V>();filled+=step;
    }
}

template <typename T>
struct CompareType {
    using Type = T;
};

template <>
struct CompareType<int8_t> {
    using Type = half;
};

template <>
struct CompareType<bfloat16_t> {
    using Type = float;
};

template <typename T>
class KernelGreater {
public:
    using CompareT = typename CompareType<T>::Type;
    static constexpr bool kNeedsConversion =
        std::is_same<T, int8_t>::value || std::is_same<T, bfloat16_t>::value;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const GreaterTilingData& data)
    {
        tiling_ = data;
        x1_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), tiling_.totalElements);

        pipe_.InitBuffer(x1Queue_, 1, tiling_.tileElements * sizeof(T));
        pipe_.InitBuffer(x2Queue_, 1, tiling_.tileElements * sizeof(T));
        pipe_.InitBuffer(yQueue_, 1, tiling_.tileElements);
        if constexpr (kNeedsConversion) {
            pipe_.InitBuffer(conversionBuffer_, 3 * tiling_.tileElements * sizeof(CompareT));
        } else {
            pipe_.InitBuffer(tempBuffer_, tiling_.tileElements * sizeof(T));
        }
        // Select mode 2 consumes a continuous packed mask across repeats and
        // requires 8 KiB of immediate-data storage on Atlas A2.
        pipe_.InitBuffer(maskBuffer_, 8 * 1024);
        if constexpr (sizeof(T) > 1) {
            const uint64_t groupElements = data.broadcastRepeat * data.broadcastInner;
            const uint64_t vectorElements = 256 / sizeof(CompareT);
            const uint64_t vectorCount =
                (groupElements + vectorElements - 1) / vectorElements * vectorElements;
            const uint64_t dmaElements = 32 / sizeof(T);
            const uint64_t innerDmaCount =
                (data.broadcastInner + dmaElements - 1) / dmaElements * dmaElements;
            const bool shortRepeatMode = data.mode >= 9 && data.mode <= 12;
            const uint64_t segmentElements = MinU64(
                groupElements, static_cast<uint64_t>(kMaxReliableCompareElements));
            const uint64_t segmentVectorCount =
                (segmentElements + vectorElements - 1) / vectorElements * vectorElements;
            // A repeated suffix just above the 8 KiB Gather-index capacity
            // used to fall back to scalar UB expansion for every group.  A
            // fixed-size Gather window only needs one 8 KiB index buffer and
            // can cover an arbitrarily larger group in several segments.
            const bool gatherWindowFits =
                segmentVectorCount + innerDmaCount <= data.tileElements;
            if (shortRepeatMode && groupElements != 0 && gatherWindowFits) {
                pipe_.InitBuffer(indexBuffer_, 8 * 1024);
                if (groupElements <= kMaxReliableCompareElements &&
                    CanCacheRhsBroadcast(vectorCount, innerDmaCount)) {
                    const uint64_t cacheBytes =
                        RhsCacheSlots() * BroadcastCacheStride(
                            vectorCount, innerDmaCount) * sizeof(T);
                    pipe_.InitBuffer(broadcastCacheBuffer_, cacheBytes);
                }
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (tiling_.mode == 2) {
            scalarValue_ = x1_.GetValue(0);
            PipeBarrier<PIPE_ALL>();
        } else if (tiling_.mode == 3) {
            scalarValue_ = x2_.GetValue(0);
            PipeBarrier<PIPE_ALL>();
        }
        const uint64_t coreBegin = static_cast<uint64_t>(GetBlockIdx()) * tiling_.blockElements;
        if (coreBegin >= tiling_.totalElements) {
            return;
        }
        const uint64_t coreEnd = MinU64(coreBegin + tiling_.blockElements, tiling_.totalElements);
        if (tiling_.mode == 13 || tiling_.mode == 14) {
            ProcessCrossedCartesianSuffix(coreBegin, coreEnd);
            return;
        }
        for (uint64_t begin = coreBegin; begin < coreEnd; begin += tiling_.tileElements) {
            const uint32_t count = static_cast<uint32_t>(MinU64(tiling_.tileElements, coreEnd - begin));
            if (tiling_.mode == 0) {
                if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value ||
                              std::is_same<T, int8_t>::value || std::is_same<T, bfloat16_t>::value ||
                              std::is_same<T, int32_t>::value) {
                    ProcessVector(begin, count);
                } else {
                    ProcessScalar(begin, count);
                }
            } else if (tiling_.mode >= 9 && tiling_.mode <= 12) {
                ProcessShortRepeatedSuffix(coreBegin, coreEnd);
                return;
            } else if (tiling_.mode == 4 || tiling_.mode == 5) {
                ProcessContiguousBroadcast(begin, count);
            } else if (tiling_.mode == 6 || tiling_.mode == 7) {
                ProcessSuffixScalarBroadcast(begin, count);
            } else if (tiling_.mode == 8) {
                ProcessSharedSuffixBroadcast(begin, count);
            } else if (tiling_.mode == 2 || tiling_.mode == 3) {
                if constexpr (std::is_same<T, half>::value || std::is_same<T, float>::value ||
                              std::is_same<T, int8_t>::value || std::is_same<T, bfloat16_t>::value ||
                              std::is_same<T, int32_t>::value) {
                    ProcessScalarVector(begin, count);
                } else {
                    ProcessScalar(begin, count);
                }
            } else {
                ProcessScalar(begin, count);
            }
        }
    }

private:
    // A single packed Compare -> Select invocation above this span can lose
    // true mask lanes on 910B-family devices.  Reusing one small mask window
    // for multiple vector spans keeps the proven instruction contract while
    // allowing one larger MTE tile.
    static constexpr uint32_t kMaxReliableCompareElements = 2 * 1024;

    __aicore__ inline uint64_t RhsSourceElements() const
    {
        uint64_t elements = 1;
        for (uint32_t axis = 0; axis < tiling_.rank; ++axis) {
            elements *= tiling_.x2Shape[axis];
        }
        return elements;
    }

    __aicore__ inline uint64_t RhsCacheSlots() const
    {
        return tiling_.broadcastInner == 0
            ? 0
            : RhsSourceElements() / tiling_.broadcastInner;
    }

    __aicore__ inline uint32_t BroadcastCacheStride(
        uint32_t vectorCount, uint32_t innerDmaCount) const
    {
        constexpr uint32_t kCacheAlignmentElements = 256;
        const uint32_t required = vectorCount + innerDmaCount;
        return (required + kCacheAlignmentElements - 1) /
                   kCacheAlignmentElements * kCacheAlignmentElements;
    }

    __aicore__ inline bool CanCacheRhsBroadcast(
        uint32_t vectorCount, uint32_t innerDmaCount) const
    {
        // ProcessInt32 reuses its rhs tensor as FP16 result scratch, so an
        // INT32 entry cannot safely remain resident across groups.  Keep the
        // proven per-group route for INT8/INT32 and cache only read-only rhs
        // types here.
        if constexpr (sizeof(T) == 1 || std::is_same<T, int32_t>::value) {
            return false;
        } else {
            if (tiling_.mode != 9) {
                return false;
            }
            const uint64_t slots = RhsCacheSlots();
            const uint64_t stride = BroadcastCacheStride(
                vectorCount, innerDmaCount);
            return slots != 0 && stride != 0 &&
                   slots <= kMaxShortBroadcastCacheBytes /
                                (stride * sizeof(T));
        }
    }

    __aicore__ inline void CopyOut(uint64_t begin, uint32_t count)
    {
        LocalTensor<uint8_t> output = yQueue_.DeQue<uint8_t>();
        DataCopyExtParams params{1, count, 0, 0, 0};
        DataCopyPad(y_[begin], output, params);
        yQueue_.FreeTensor(output);
    }

    __aicore__ inline void ProcessVector(uint64_t begin, uint32_t count)
    {
        ProcessVectorOffsets(begin, begin, begin, count);
    }

    __aicore__ inline void ProcessVectorOffsets(
        uint64_t outputBegin, uint64_t lhsBegin, uint64_t rhsBegin, uint32_t count)
    {
        const uint32_t vectorElements = 256 / sizeof(CompareT);
        const uint32_t vectorCount =
            (count + vectorElements - 1) / vectorElements * vectorElements;
        // DataCopyPad's left/right padding is an MTE block-tail facility, not
        // a way to fill an entire 256-byte vector repeat.  Asking it to pad
        // vectorCount-count elements makes tiny tensors (for example two
        // FP16 values) encode an illegal MTE instruction on 910B.  Pad only
        // to the 32-byte DMA block; vector instructions may consume the rest
        // of the allocated UB repeat because those lanes are never copied
        // back to GM.
        const uint32_t dmaElements = 32 / sizeof(T);
        const uint32_t dmaCount =
            (count + dmaElements - 1) / dmaElements * dmaElements;
        LocalTensor<T> lhs = x1Queue_.AllocTensor<T>();
        LocalTensor<T> rhs = x2Queue_.AllocTensor<T>();
        DataCopyExtParams inputParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{
            true, 0, static_cast<uint8_t>(dmaCount - count), static_cast<T>(0)};
        DataCopyPad(lhs, x1_[lhsBegin], inputParams, pad);
        DataCopyPad(rhs, x2_[rhsBegin], inputParams, pad);
        x1Queue_.EnQue(lhs);
        x2Queue_.EnQue(rhs);

        lhs = x1Queue_.DeQue<T>();
        rhs = x2Queue_.DeQue<T>();
        LocalTensor<uint8_t> output = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        if constexpr (std::is_same<T, int32_t>::value) {
            ProcessInt32(lhs, rhs, output, mask, vectorCount);
        } else if constexpr (kNeedsConversion) {
            ProcessConverted(lhs, rhs, output, mask, vectorCount);
        } else {
            ProcessNative(lhs, rhs, output, mask, vectorCount);
        }
        yQueue_.EnQue(output);
        x1Queue_.FreeTensor(lhs);
        x2Queue_.FreeTensor(rhs);
        CopyOut(outputBegin, count);
    }

    __aicore__ inline void ProcessContiguousBroadcast(uint64_t begin, uint32_t count)
    {
        const uint64_t intervalEnd = begin + count;
        const uint64_t groupElements = tiling_.broadcastRepeat * tiling_.broadcastInner;
        while (begin < intervalEnd) {
            const uint64_t innerOffset = begin % tiling_.broadcastInner;
            const uint64_t available = tiling_.broadcastInner - innerOffset;
            const uint32_t once = static_cast<uint32_t>(MinU64(
                MinU64(tiling_.tileElements, intervalEnd - begin), available));
            const uint64_t outerIndex = begin / groupElements;
            const uint64_t broadcastOffset = outerIndex * tiling_.broadcastInner + innerOffset;
            const uint64_t lhsBegin = tiling_.mode == 4 ? broadcastOffset : begin;
            const uint64_t rhsBegin = tiling_.mode == 5 ? broadcastOffset : begin;
            ProcessVectorOffsets(begin, lhsBegin, rhsBegin, once);
            begin += once;
        }
    }

    __aicore__ inline void ProcessSuffixScalarBroadcast(uint64_t begin, uint32_t count)
    {
        const uint64_t intervalEnd = begin + count;
        const bool lhsScalar = tiling_.mode == 7;
        while (begin < intervalEnd) {
            const uint64_t innerOffset = begin % tiling_.broadcastInner;
            const uint64_t available = tiling_.broadcastInner - innerOffset;
            const uint32_t once = static_cast<uint32_t>(MinU64(
                MinU64(tiling_.tileElements, intervalEnd - begin), available));
            const uint64_t lhsOffset = BroadcastOffset(begin, tiling_.x1Stride);
            const uint64_t rhsOffset = BroadcastOffset(begin, tiling_.x2Stride);
            scalarValue_ = lhsScalar ? x1_.GetValue(lhsOffset) : x2_.GetValue(rhsOffset);
            PipeBarrier<PIPE_ALL>();
            ProcessScalarVectorAt(begin, lhsScalar ? rhsOffset : lhsOffset,
                                  once, lhsScalar);
            begin += once;
        }
    }

    __aicore__ inline uint32_t CrossScalarCount(bool lhsScalar) const
    {
        const uint64_t* vectorShape = lhsScalar
            ? tiling_.x2Shape : tiling_.x1Shape;
        const uint64_t* scalarShape = lhsScalar
            ? tiling_.x1Shape : tiling_.x2Shape;
        int32_t axis = S9_GREATER_MAX_RANK - 1;
        for (; axis >= 0; --axis) {
            if (vectorShape[axis] != tiling_.outputShape[axis] ||
                scalarShape[axis] != 1) {
                break;
            }
        }
        return axis >= 0 ? static_cast<uint32_t>(tiling_.outputShape[axis]) : 0;
    }

    __aicore__ inline void ProcessCrossedCartesianSuffix(
        uint64_t coreBegin, uint64_t coreEnd)
    {
        const bool lhsScalar = tiling_.mode == 14;
        const uint32_t inner = static_cast<uint32_t>(tiling_.broadcastInner);
        const uint32_t cartesian = static_cast<uint32_t>(tiling_.broadcastRepeat);
        const uint32_t scalarCount = CrossScalarCount(lhsScalar);
        if (inner == 0 || cartesian == 0 || scalarCount == 0) {
            return;
        }
        const uint64_t rowElements =
            static_cast<uint64_t>(scalarCount) * inner;
        const uint64_t taskBegin = coreBegin / rowElements;
        const uint64_t taskEnd = (coreEnd + rowElements - 1) / rowElements;
        const uint64_t totalTasks = tiling_.totalElements / rowElements;
        const uint64_t stop = MinU64(taskEnd, totalTasks);
        for (uint64_t task = taskBegin; task < stop; ++task) {
            const uint64_t outputRowBegin = task * rowElements;
            for (uint32_t scalarBegin = 0; scalarBegin < scalarCount;) {
                uint32_t consumed;
                if constexpr (std::is_same<T, int8_t>::value) {
                    consumed = ProcessInt8CrossedCartesianBatch(
                        outputRowBegin, scalarBegin, scalarCount, inner, lhsScalar);
                } else {
                    consumed = ProcessCrossedCartesianBatch(
                        outputRowBegin, scalarBegin, scalarCount, inner, lhsScalar);
                }
                scalarBegin += consumed;
            }
        }
    }

    __aicore__ inline uint32_t ProcessInt8CrossedCartesianBatch(
        uint64_t outputRowBegin, uint32_t scalarBegin,
        uint32_t scalarCount, uint32_t inner, bool lhsScalar)
    {
        const uint32_t paddedInner = (inner + 31) / 32 * 32;
        const uint32_t vectorElements = 256 / sizeof(CompareT);
        uint32_t batchGroups = scalarCount - scalarBegin;
        const uint32_t maxByCompare = kMaxReliableCompareElements / paddedInner;
        if (batchGroups > maxByCompare) {
            batchGroups = maxByCompare;
        }
        while (batchGroups > 1) {
            const uint32_t paddedCount = batchGroups * paddedInner;
            const uint32_t vectorCount =
                (paddedCount + vectorElements - 1) / vectorElements * vectorElements;
            if (static_cast<uint64_t>(vectorCount) + paddedInner <=
                tiling_.tileElements) {
                break;
            }
            --batchGroups;
        }
        const uint32_t paddedCount = batchGroups * paddedInner;
        const uint32_t vectorCount =
            (paddedCount + vectorElements - 1) / vectorElements * vectorElements;
        const uint32_t sourceOffset = vectorCount;
        const uint32_t dmaElements = 32 / sizeof(T);
        const uint32_t scalarDmaCount =
            (batchGroups + dmaElements - 1) / dmaElements * dmaElements;
        const uint32_t vectorDmaCount =
            (inner + dmaElements - 1) / dmaElements * dmaElements;

        LocalTensor<T> lhs = x1Queue_.AllocTensor<T>();
        LocalTensor<T> rhs = x2Queue_.AllocTensor<T>();
        LocalTensor<T> scalarTensor = lhsScalar ? lhs : rhs;
        LocalTensor<T> vectorTensor = lhsScalar ? rhs : lhs;
        GlobalTensor<T>& scalarGm = lhsScalar ? x1_ : x2_;
        GlobalTensor<T>& vectorGm = lhsScalar ? x2_ : x1_;
        const uint64_t scalarOutput =
            outputRowBegin + static_cast<uint64_t>(scalarBegin) * inner;
        const uint64_t scalarOffset = BroadcastOffset(
            scalarOutput, lhsScalar ? tiling_.x1Stride : tiling_.x2Stride);
        const uint64_t vectorOffset = BroadcastOffset(
            outputRowBegin, lhsScalar ? tiling_.x2Stride : tiling_.x1Stride);
        DataCopyExtParams scalarParams{
            1, static_cast<uint32_t>(batchGroups * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> scalarPad{
            true, 0, static_cast<uint8_t>(scalarDmaCount - batchGroups),
            static_cast<T>(0)};
        DataCopyExtParams vectorParams{
            1, static_cast<uint32_t>(inner * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> vectorPad{
            true, 0, static_cast<uint8_t>(vectorDmaCount - inner),
            static_cast<T>(0)};
        DataCopyPad(scalarTensor[sourceOffset], scalarGm[scalarOffset],
                    scalarParams, scalarPad);
        DataCopyPad(vectorTensor[sourceOffset], vectorGm[vectorOffset],
                    vectorParams, vectorPad);
        x1Queue_.EnQue(lhs);
        x2Queue_.EnQue(rhs);
        lhs = x1Queue_.DeQue<T>();
        rhs = x2Queue_.DeQue<T>();
        scalarTensor = lhsScalar ? lhs : rhs;
        vectorTensor = lhsScalar ? rhs : lhs;

        uint32_t dstShape[2] = {batchGroups, paddedInner};
        uint32_t scalarShape[2] = {batchGroups, 1};
        uint32_t vectorShape[2] = {1, paddedInner};
        LocalTensor<uint8_t> sharedTmp = maskBuffer_.Get<uint8_t>();
        LocalTensor<uint8_t> output = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        if constexpr (std::is_same<T, int8_t>::value) {
            // Every INT8 value is represented exactly by FP16. Widen the
            // compact input rows once, then broadcast in a supported type.
            // This batches many short scalar/vector comparisons without
            // changing the comparison or depending on the input values.
            LocalTensor<half> converted = conversionBuffer_.Get<half>();
            LocalTensor<half> lhsHalf = converted;
            LocalTensor<half> rhsHalf = converted[tiling_.tileElements];
            LocalTensor<half> temp = converted[2 * tiling_.tileElements];
            LocalTensor<half> scalarHalf = lhsScalar ? lhsHalf : rhsHalf;
            LocalTensor<half> vectorHalf = lhsScalar ? rhsHalf : lhsHalf;
            Cast(scalarHalf[sourceOffset], scalarTensor[sourceOffset],
                 RoundMode::CAST_NONE, scalarDmaCount);
            Cast(vectorHalf[sourceOffset], vectorTensor[sourceOffset],
                 RoundMode::CAST_NONE, vectorDmaCount);
            PipeBarrier<PIPE_V>();
            BroadCast<half, 2, 1>(scalarHalf, scalarHalf[sourceOffset],
                                  dstShape, scalarShape, sharedTmp);
            BroadCast<half, 2, 0>(vectorHalf, vectorHalf[sourceOffset],
                                  dstShape, vectorShape, sharedTmp);
            PipeBarrier<PIPE_V>();
            Compare(mask, lhsHalf, rhsHalf, CMPMODE::GT, vectorCount);
            PipeBarrier<PIPE_V>();
            Duplicate(lhsHalf, static_cast<half>(1), vectorCount);
            PipeBarrier<PIPE_V>();
            Select(temp, mask, lhsHalf, static_cast<half>(0),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, vectorCount);
            PipeBarrier<PIPE_V>();
            Cast(output, temp, RoundMode::CAST_NONE, vectorCount);
        } else {
            BroadCast<T, 2, 1>(scalarTensor, scalarTensor[sourceOffset],
                                dstShape, scalarShape, sharedTmp);
            BroadCast<T, 2, 0>(vectorTensor, vectorTensor[sourceOffset],
                                dstShape, vectorShape, sharedTmp);
            if constexpr (std::is_same<T, int32_t>::value) {
                ProcessInt32(lhs, rhs, output, mask, vectorCount);
            } else if constexpr (kNeedsConversion) {
                ProcessConverted(lhs, rhs, output, mask, vectorCount);
            } else {
                ProcessNative(lhs, rhs, output, mask, vectorCount);
            }
        }
        yQueue_.EnQue(output);
        x1Queue_.FreeTensor(lhs);
        x2Queue_.FreeTensor(rhs);
        output = yQueue_.DeQue<uint8_t>();
        // UB rows are padded to 32 bytes; MTE3 consumes that implicit
        // padding and packs all valid rows into one contiguous GM interval.
        DataCopyExtParams outputParams{
            static_cast<uint16_t>(batchGroups), inner, 0, 0, 0};
        const uint64_t destination = outputRowBegin +
            static_cast<uint64_t>(scalarBegin) * inner;
        DataCopyPad(y_[destination], output, outputParams);
        PipeBarrier<PIPE_MTE3>();
        yQueue_.FreeTensor(output);
        return batchGroups;
    }

    __aicore__ inline uint32_t ProcessCrossedCartesianBatch(
        uint64_t outputRowBegin, uint32_t scalarBegin,
        uint32_t scalarCount, uint32_t inner, bool lhsScalar)
    {
        const uint32_t paddedInner = (inner + 31) / 32 * 32;
        const uint32_t vectorElements = 256 / sizeof(CompareT);
        uint32_t batchGroups = scalarCount - scalarBegin;
        const uint32_t maxByCompare = kMaxReliableCompareElements / paddedInner;
        if (batchGroups > maxByCompare) {
            batchGroups = maxByCompare;
        }
        while (batchGroups > 1) {
            const uint32_t paddedCount = batchGroups * paddedInner;
            const uint32_t vectorCount =
                (paddedCount + vectorElements - 1) / vectorElements * vectorElements;
            if (static_cast<uint64_t>(vectorCount) + paddedInner <=
                tiling_.tileElements) {
                break;
            }
            --batchGroups;
        }
        const uint32_t paddedCount = batchGroups * paddedInner;
        const uint32_t vectorCount =
            (paddedCount + vectorElements - 1) / vectorElements * vectorElements;
        const uint32_t sourceOffset = vectorCount;
        const uint32_t dmaElements = 32 / sizeof(T);
        const uint32_t scalarDmaCount =
            (batchGroups + dmaElements - 1) / dmaElements * dmaElements;
        const uint32_t vectorDmaCount =
            (inner + dmaElements - 1) / dmaElements * dmaElements;

        LocalTensor<T> lhs = x1Queue_.AllocTensor<T>();
        LocalTensor<T> rhs = x2Queue_.AllocTensor<T>();
        LocalTensor<T> scalarTensor = lhsScalar ? lhs : rhs;
        LocalTensor<T> vectorTensor = lhsScalar ? rhs : lhs;
        GlobalTensor<T>& scalarGm = lhsScalar ? x1_ : x2_;
        GlobalTensor<T>& vectorGm = lhsScalar ? x2_ : x1_;
        const uint64_t scalarOutput =
            outputRowBegin + static_cast<uint64_t>(scalarBegin) * inner;
        const uint64_t scalarOffset = BroadcastOffset(
            scalarOutput, lhsScalar ? tiling_.x1Stride : tiling_.x2Stride);
        const uint64_t vectorOffset = BroadcastOffset(
            outputRowBegin, lhsScalar ? tiling_.x2Stride : tiling_.x1Stride);
        DataCopyExtParams scalarParams{
            1, static_cast<uint32_t>(batchGroups * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> scalarPad{
            true, 0, static_cast<uint8_t>(scalarDmaCount - batchGroups),
            static_cast<T>(0)};
        DataCopyExtParams vectorParams{
            1, static_cast<uint32_t>(inner * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> vectorPad{
            true, 0, static_cast<uint8_t>(vectorDmaCount - inner),
            static_cast<T>(0)};
        DataCopyPad(scalarTensor[sourceOffset], scalarGm[scalarOffset],
                    scalarParams, scalarPad);
        DataCopyPad(vectorTensor[sourceOffset], vectorGm[vectorOffset],
                    vectorParams, vectorPad);
        x1Queue_.EnQue(lhs);
        x2Queue_.EnQue(rhs);
        lhs = x1Queue_.DeQue<T>();
        rhs = x2Queue_.DeQue<T>();
        scalarTensor = lhsScalar ? lhs : rhs;
        vectorTensor = lhsScalar ? rhs : lhs;

        uint32_t dstShape[2] = {batchGroups, paddedInner};
        uint32_t scalarShape[2] = {batchGroups, 1};
        uint32_t vectorShape[2] = {1, paddedInner};
        LocalTensor<uint8_t> sharedTmp = maskBuffer_.Get<uint8_t>();
        BroadCast<T, 2, 1>(scalarTensor, scalarTensor[sourceOffset],
                            dstShape, scalarShape, sharedTmp);
        BroadCast<T, 2, 0>(vectorTensor, vectorTensor[sourceOffset],
                            dstShape, vectorShape, sharedTmp);

        LocalTensor<uint8_t> output = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        if constexpr (std::is_same<T, int32_t>::value) {
            ProcessInt32(lhs, rhs, output, mask, vectorCount);
        } else if constexpr (kNeedsConversion) {
            ProcessConverted(lhs, rhs, output, mask, vectorCount);
        } else {
            ProcessNative(lhs, rhs, output, mask, vectorCount);
        }
        yQueue_.EnQue(output);
        x1Queue_.FreeTensor(lhs);
        x2Queue_.FreeTensor(rhs);
        output = yQueue_.DeQue<uint8_t>();
        // UB rows are padded to 32 bytes; MTE3 consumes that implicit
        // padding and packs all valid rows into one contiguous GM interval.
        DataCopyExtParams outputParams{
            static_cast<uint16_t>(batchGroups), inner, 0, 0, 0};
        const uint64_t destination = outputRowBegin +
            static_cast<uint64_t>(scalarBegin) * inner;
        DataCopyPad(y_[destination], output, outputParams);
        PipeBarrier<PIPE_MTE3>();
        yQueue_.FreeTensor(output);
        return batchGroups;
    }

    __aicore__ inline void ProcessSharedSuffixBroadcast(uint64_t begin, uint32_t count)
    {
        const uint64_t intervalEnd = begin + count;
        while (begin < intervalEnd) {
            const uint64_t innerOffset = begin % tiling_.broadcastInner;
            const uint64_t available = tiling_.broadcastInner - innerOffset;
            const uint32_t once = static_cast<uint32_t>(MinU64(
                MinU64(tiling_.tileElements, intervalEnd - begin), available));
            const uint64_t lhsOffset = BroadcastOffset(begin, tiling_.x1Stride);
            const uint64_t rhsOffset = BroadcastOffset(begin, tiling_.x2Stride);
            ProcessVectorOffsets(begin, lhsOffset, rhsOffset, once);
            begin += once;
        }
    }

    __aicore__ inline void ProcessShortRepeatedSuffix(uint64_t begin, uint64_t end)
    {
        const uint32_t inner = static_cast<uint32_t>(tiling_.broadcastInner);
        const uint32_t repeat = static_cast<uint32_t>(tiling_.broadcastRepeat);
        const uint32_t groupElements = inner * repeat;
        const bool lhsBroadcast = tiling_.mode == 10 || tiling_.mode == 12;
        const bool bothNeedOffset = tiling_.mode == 11 || tiling_.mode == 12;
        const uint32_t vectorElements = 256 / sizeof(CompareT);
        const uint32_t vectorCount =
            (groupElements + vectorElements - 1) / vectorElements * vectorElements;
        const uint32_t dmaElements = 32 / sizeof(T);
        const uint32_t innerDmaCount =
            (inner + dmaElements - 1) / dmaElements * dmaElements;
        if constexpr (sizeof(T) > 1) {
            const uint32_t segmentElements = static_cast<uint32_t>(MinU64(
                groupElements, static_cast<uint64_t>(kMaxReliableCompareElements)));
            const uint32_t segmentVectorCount =
                (segmentElements + vectorElements - 1) / vectorElements * vectorElements;
            if (groupElements > kMaxReliableCompareElements &&
                static_cast<uint64_t>(segmentVectorCount) + innerDmaCount <=
                    tiling_.tileElements) {
                ProcessSegmentedShortRepeatedSuffix(
                    begin, end, groupElements, inner, lhsBroadcast,
                    bothNeedOffset);
                return;
            }
        }
        const bool useAlignedGather =
            sizeof(T) > 1 && groupElements <= kMaxReliableCompareElements &&
            static_cast<uint64_t>(vectorCount) + innerDmaCount <= tiling_.tileElements;
        LocalTensor<uint32_t> offsets = maskBuffer_.Get<uint32_t>();
        if (useAlignedGather) {
            offsets = indexBuffer_.Get<uint32_t>();
            for (uint32_t element = 0; element < groupElements; ++element) {
                offsets.SetValue(element, (element % inner) * sizeof(T));
            }
            const event_t scalarToVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
        }
        const uint64_t localGroups = (end - begin) / groupElements;
        const uint64_t cacheSlotsForCore = RhsCacheSlots();
        // Every AIV owns a disjoint output interval and therefore builds its
        // own UB cache.  Do not pre-expand the complete RHS when that core
        // will consume only a handful of groups (the published small rank-six
        // anchor is such a case).  Two local reuses per cache slot is a
        // conservative break-even guard for one suffix DMA plus one Gather.
        const bool useRhsCache = useAlignedGather &&
            CanCacheRhsBroadcast(vectorCount, innerDmaCount) &&
            localGroups >= cacheSlotsForCore * 2;
        if (useRhsCache) {
            LocalTensor<T> cache = broadcastCacheBuffer_.Get<T>();
            const uint32_t cacheStride = BroadcastCacheStride(
                vectorCount, innerDmaCount);
            const uint64_t cacheSlots = cacheSlotsForCore;
            BuildRhsBroadcastCache(
                cache, offsets, cacheSlots, cacheStride, vectorCount,
                inner, innerDmaCount, groupElements);
            while (begin < end) {
                const uint64_t rhsOffset =
                    BroadcastOffset(begin, tiling_.x2Stride);
                const uint64_t cacheSlot = rhsOffset / inner;
                if (cacheSlot < cacheSlots) {
                    ProcessShortRepeatedCachedRhsGroup(
                        begin, begin, groupElements, cache,
                        cacheSlot, cacheStride);
                } else {
                    ProcessShortRepeatedGroup(
                        begin, begin, rhsOffset, groupElements, inner, repeat,
                        false, useAlignedGather, offsets);
                }
                begin += groupElements;
            }
            return;
        }
        while (begin < end) {
            if constexpr (sizeof(T) > 1 &&
                          !std::is_same<T, int32_t>::value) {
                if (useAlignedGather && !bothNeedOffset &&
                    (tiling_.mode == 9 || tiling_.mode == 10)) {
                    const uint32_t batchGroups = ContiguousShortBatchGroups(
                        begin, end, groupElements, inner, lhsBroadcast);
                    if (batchGroups >= 2) {
                        const uint64_t shortOffset = BroadcastOffset(
                            begin, lhsBroadcast ? tiling_.x1Stride
                                                : tiling_.x2Stride);
                        ProcessContiguousShortBatch(
                            begin, shortOffset, batchGroups, groupElements,
                            inner, lhsBroadcast, offsets);
                        begin += static_cast<uint64_t>(batchGroups) *
                                 groupElements;
                        continue;
                    }
                }
            }
            const uint64_t lhsOffset = lhsBroadcast || bothNeedOffset
                ? BroadcastOffset(begin, tiling_.x1Stride) : begin;
            const uint64_t rhsOffset = !lhsBroadcast || bothNeedOffset
                ? BroadcastOffset(begin, tiling_.x2Stride) : begin;
            ProcessShortRepeatedGroup(begin, lhsOffset, rhsOffset, groupElements,
                                      inner, repeat, lhsBroadcast, useAlignedGather, offsets);
            begin += groupElements;
        }
    }

    __aicore__ inline uint32_t ContiguousShortBatchGroups(
        uint64_t begin, uint64_t end, uint32_t groupElements,
        uint32_t inner, bool lhsBroadcast) const
    {
        if (groupElements == 0 || inner == 0 || begin >= end) {
            return 0;
        }
        const uint64_t availableGroups = (end - begin) / groupElements;
        uint32_t maxGroups = static_cast<uint32_t>(MinU64(
            availableGroups,
            static_cast<uint64_t>(kMaxReliableCompareElements / groupElements)));
        const uint32_t vectorElements = 256 / sizeof(CompareT);
        const uint32_t dmaElements = 32 / sizeof(T);
        while (maxGroups >= 2) {
            const uint32_t batchElements = maxGroups * groupElements;
            const uint32_t vectorCount =
                (batchElements + vectorElements - 1) /
                vectorElements * vectorElements;
            const uint32_t shortElements = maxGroups * inner;
            const uint32_t shortDmaCount =
                (shortElements + dmaElements - 1) /
                dmaElements * dmaElements;
            if (static_cast<uint64_t>(vectorCount) + shortDmaCount <=
                tiling_.tileElements) {
                break;
            }
            --maxGroups;
        }
        if (maxGroups < 2) {
            return 1;
        }
        const uint64_t* shortStride = lhsBroadcast
            ? tiling_.x1Stride : tiling_.x2Stride;
        const uint64_t first = BroadcastOffset(begin, shortStride);
        uint32_t contiguous = 1;
        for (; contiguous < maxGroups; ++contiguous) {
            const uint64_t next = BroadcastOffset(
                begin + static_cast<uint64_t>(contiguous) * groupElements,
                shortStride);
            if (next != first + static_cast<uint64_t>(contiguous) * inner) {
                break;
            }
        }
        return contiguous;
    }

    __aicore__ inline void ProcessContiguousShortBatch(
        uint64_t outputBegin, uint64_t shortBegin, uint32_t batchGroups,
        uint32_t groupElements, uint32_t inner, bool lhsBroadcast,
        LocalTensor<uint32_t> offsets)
    {
        const uint32_t batchElements = batchGroups * groupElements;
        const uint32_t shortElements = batchGroups * inner;
        const uint32_t vectorElements = 256 / sizeof(CompareT);
        const uint32_t vectorCount =
            (batchElements + vectorElements - 1) /
            vectorElements * vectorElements;
        const uint32_t dmaElements = 32 / sizeof(T);
        const uint32_t batchDmaCount =
            (batchElements + dmaElements - 1) /
            dmaElements * dmaElements;
        const uint32_t shortDmaCount =
            (shortElements + dmaElements - 1) /
            dmaElements * dmaElements;

        uint32_t output = 0;
        for (uint32_t group = 0; group < batchGroups; ++group) {
            const uint32_t sourceBase = group * inner;
            for (uint32_t repetition = 0;
                 repetition < tiling_.broadcastRepeat; ++repetition) {
                for (uint32_t element = 0; element < inner; ++element) {
                    offsets.SetValue(
                        output++, (sourceBase + element) * sizeof(T));
                }
            }
        }
        const event_t scalarToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);

        LocalTensor<T> lhs = x1Queue_.AllocTensor<T>();
        LocalTensor<T> rhs = x2Queue_.AllocTensor<T>();
        DataCopyExtParams fullParams{
            1, static_cast<uint32_t>(batchElements * sizeof(T)), 0, 0, 0};
        DataCopyExtParams shortParams{
            1, static_cast<uint32_t>(shortElements * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> fullPad{
            true, 0, static_cast<uint8_t>(batchDmaCount - batchElements),
            static_cast<T>(0)};
        DataCopyPadExtParams<T> shortPad{
            true, 0, static_cast<uint8_t>(shortDmaCount - shortElements),
            static_cast<T>(0)};
        if (lhsBroadcast) {
            DataCopyPad(lhs[vectorCount], x1_[shortBegin],
                        shortParams, shortPad);
            DataCopyPad(rhs, x2_[outputBegin], fullParams, fullPad);
        } else {
            DataCopyPad(lhs, x1_[outputBegin], fullParams, fullPad);
            DataCopyPad(rhs[vectorCount], x2_[shortBegin],
                        shortParams, shortPad);
        }
        x1Queue_.EnQue(lhs);
        x2Queue_.EnQue(rhs);
        lhs = x1Queue_.DeQue<T>();
        rhs = x2Queue_.DeQue<T>();

        LocalTensor<T> repeated = lhsBroadcast ? lhs : rhs;
        Gather(repeated, repeated[vectorCount], offsets, 0, batchElements);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> result = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        if constexpr (kNeedsConversion) {
            ProcessConverted(lhs, rhs, result, mask, vectorCount);
        } else {
            ProcessNative(lhs, rhs, result, mask, vectorCount);
        }
        yQueue_.EnQue(result);
        x1Queue_.FreeTensor(lhs);
        x2Queue_.FreeTensor(rhs);
        CopyOut(outputBegin, batchElements);
    }

    __aicore__ inline void ProcessSegmentedShortRepeatedSuffix(
        uint64_t begin, uint64_t end, uint32_t groupElements,
        uint32_t inner, bool lhsBroadcast, bool bothNeedOffset)
    {
        // A group-aligned interval can reuse one offset vector across every
        // group owned by this AIV.  The host deliberately returns to this
        // topology above the measured group-count crossover; keep that
        // amortized loop separate from the arbitrary-boundary fast path.
        if (begin % groupElements == 0 &&
            (end - begin) % groupElements == 0) {
            ProcessGroupAlignedSegmentedShortRepeatedSuffix(
                begin, end, groupElements, inner, lhsBroadcast,
                bothNeedOffset);
            return;
        }
        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        // Core intervals are 32-byte output aligned but may begin inside a
        // logical repeated-suffix group.  Cap every window at both the proven
        // Gather span and the current group boundary, and phase the offsets
        // from the core's actual position inside that suffix.
        while (begin < end) {
            const uint32_t groupOffset =
                static_cast<uint32_t>(begin % groupElements);
            const uint64_t groupBegin = begin - groupOffset;
            const uint32_t groupAvailable = groupElements - groupOffset;
            const uint32_t segmentCount = static_cast<uint32_t>(MinU64(
                MinU64(kMaxReliableCompareElements, groupAvailable),
                end - begin));
            const uint32_t suffixPhase = groupOffset % inner;
            for (uint32_t element = 0; element < segmentCount; ++element) {
                offsets.SetValue(
                    element,
                    ((suffixPhase + element) % inner) * sizeof(T));
            }
            const event_t scalarToVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            const uint64_t shortOffset = lhsBroadcast
                ? BroadcastOffset(groupBegin, tiling_.x1Stride)
                : BroadcastOffset(groupBegin, tiling_.x2Stride);
            const uint64_t lhsOffset = lhsBroadcast
                ? shortOffset
                : (bothNeedOffset
                       ? BroadcastOffset(begin, tiling_.x1Stride)
                       : begin);
            const uint64_t rhsOffset = lhsBroadcast
                ? (bothNeedOffset
                       ? BroadcastOffset(begin, tiling_.x2Stride)
                       : begin)
                : shortOffset;
            // repeat is unused on the Gather branch; zero makes that
            // contract explicit and prevents accidental scalar reuse.
            ProcessShortRepeatedGroup(
                begin, lhsOffset, rhsOffset, segmentCount, inner, 0,
                lhsBroadcast, true, offsets);
            begin += segmentCount;
        }
    }

    __aicore__ inline void ProcessGroupAlignedSegmentedShortRepeatedSuffix(
        uint64_t begin, uint64_t end, uint32_t groupElements,
        uint32_t inner, bool lhsBroadcast, bool bothNeedOffset)
    {
        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        for (uint32_t segmentOffset = 0; segmentOffset < groupElements;
             segmentOffset += kMaxReliableCompareElements) {
            const uint32_t segmentCount =
                groupElements - segmentOffset < kMaxReliableCompareElements
                    ? groupElements - segmentOffset
                    : kMaxReliableCompareElements;
            const uint32_t suffixPhase = segmentOffset % inner;
            for (uint32_t element = 0; element < segmentCount; ++element) {
                offsets.SetValue(
                    element,
                    ((suffixPhase + element) % inner) * sizeof(T));
            }
            const event_t scalarToVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);

            for (uint64_t groupBegin = begin; groupBegin < end;
                 groupBegin += groupElements) {
                const uint64_t shortOffset = lhsBroadcast
                    ? BroadcastOffset(groupBegin, tiling_.x1Stride)
                    : BroadcastOffset(groupBegin, tiling_.x2Stride);
                const uint64_t outputBegin = groupBegin + segmentOffset;
                const uint64_t lhsOffset = lhsBroadcast
                    ? shortOffset
                    : (bothNeedOffset
                           ? BroadcastOffset(outputBegin, tiling_.x1Stride)
                           : outputBegin);
                const uint64_t rhsOffset = lhsBroadcast
                    ? (bothNeedOffset
                           ? BroadcastOffset(outputBegin, tiling_.x2Stride)
                           : outputBegin)
                    : shortOffset;
                ProcessShortRepeatedGroup(
                    outputBegin, lhsOffset, rhsOffset, segmentCount, inner, 0,
                    lhsBroadcast, true, offsets);
            }
        }
    }

    __aicore__ inline void BuildRhsBroadcastCache(
        LocalTensor<T> cache, LocalTensor<uint32_t> offsets,
        uint64_t cacheSlots, uint32_t cacheStride, uint32_t vectorCount,
        uint32_t inner, uint32_t innerDmaCount, uint32_t groupElements)
    {
        DataCopyExtParams suffixParams{
            1, static_cast<uint32_t>(inner * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> suffixPad{
            true, 0, static_cast<uint8_t>(innerDmaCount - inner),
            static_cast<T>(0)};
        const event_t inputReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        for (uint64_t slot = 0; slot < cacheSlots; ++slot) {
            const uint64_t base = slot * cacheStride;
            DataCopyPad(
                cache[base + vectorCount],
                x2_[slot * static_cast<uint64_t>(inner)],
                suffixParams, suffixPad);
            SetFlag<HardEvent::MTE2_V>(inputReady);
            WaitFlag<HardEvent::MTE2_V>(inputReady);
            Gather(cache[base], cache[base + vectorCount], offsets, 0,
                   groupElements);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessShortRepeatedCachedRhsGroup(
        uint64_t outputBegin, uint64_t lhsBegin, uint32_t groupElements,
        LocalTensor<T> cache, uint64_t cacheSlot, uint32_t cacheStride)
    {
        const uint32_t vectorElements = 256 / sizeof(CompareT);
        const uint32_t vectorCount =
            (groupElements + vectorElements - 1) / vectorElements * vectorElements;
        const uint32_t dmaElements = 32 / sizeof(T);
        const uint32_t groupDmaCount =
            (groupElements + dmaElements - 1) / dmaElements * dmaElements;

        LocalTensor<T> lhs = x1Queue_.AllocTensor<T>();
        DataCopyExtParams fullParams{
            1, static_cast<uint32_t>(groupElements * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> fullPad{
            true, 0, static_cast<uint8_t>(groupDmaCount - groupElements),
            static_cast<T>(0)};
        DataCopyPad(lhs, x1_[lhsBegin], fullParams, fullPad);
        x1Queue_.EnQue(lhs);
        lhs = x1Queue_.DeQue<T>();
        LocalTensor<T> rhs = cache[cacheSlot * cacheStride];

        LocalTensor<uint8_t> output = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        if constexpr (std::is_same<T, int32_t>::value) {
            ProcessInt32(lhs, rhs, output, mask, vectorCount);
        } else if constexpr (kNeedsConversion) {
            ProcessConverted(lhs, rhs, output, mask, vectorCount);
        } else {
            ProcessNative(lhs, rhs, output, mask, vectorCount);
        }
        yQueue_.EnQue(output);
        x1Queue_.FreeTensor(lhs);
        CopyOut(outputBegin, groupElements);
    }

    __aicore__ inline void ProcessShortRepeatedGroup(
        uint64_t outputBegin, uint64_t lhsBegin, uint64_t rhsBegin,
        uint32_t groupElements, uint32_t inner, uint32_t repeat,
        bool lhsBroadcast, bool useAlignedGather, LocalTensor<uint32_t> offsets)
    {
        const uint32_t vectorElements = 256 / sizeof(CompareT);
        const uint32_t vectorCount =
            (groupElements + vectorElements - 1) / vectorElements * vectorElements;
        const uint32_t dmaElements = 32 / sizeof(T);
        const uint32_t groupDmaCount =
            (groupElements + dmaElements - 1) / dmaElements * dmaElements;
        const uint32_t innerDmaCount =
            (inner + dmaElements - 1) / dmaElements * dmaElements;
        // Gather requires aligned UB bases.  Keep the short source suffix at
        // an aligned offset after the rounded comparison span, then expand it
        // into the aligned start of the same queue tensor.  The 8 KiB mask
        // buffer can temporarily hold at most 2,048 byte-offset indices.
        // CANN 8.5 byte-element Gather is not bit-exact for every non-block
        // suffix, so INT8 retains the established scalar expansion.
        const uint32_t suffixOffset = useAlignedGather ? vectorCount : 0;

        LocalTensor<T> lhs = x1Queue_.AllocTensor<T>();
        LocalTensor<T> rhs = x2Queue_.AllocTensor<T>();
        DataCopyExtParams fullParams{
            1, static_cast<uint32_t>(groupElements * sizeof(T)), 0, 0, 0};
        DataCopyExtParams suffixParams{
            1, static_cast<uint32_t>(inner * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> fullPad{
            true, 0, static_cast<uint8_t>(groupDmaCount - groupElements), static_cast<T>(0)};
        DataCopyPadExtParams<T> suffixPad{
            true, 0, static_cast<uint8_t>(innerDmaCount - inner), static_cast<T>(0)};
        if (lhsBroadcast) {
            DataCopyPad(lhs[suffixOffset], x1_[lhsBegin], suffixParams, suffixPad);
            DataCopyPad(rhs, x2_[rhsBegin], fullParams, fullPad);
        } else {
            DataCopyPad(lhs, x1_[lhsBegin], fullParams, fullPad);
            DataCopyPad(rhs[suffixOffset], x2_[rhsBegin], suffixParams, suffixPad);
        }
        x1Queue_.EnQue(lhs);
        x2Queue_.EnQue(rhs);
        lhs = x1Queue_.DeQue<T>();
        rhs = x2Queue_.DeQue<T>();

        LocalTensor<T> repeated = lhsBroadcast ? lhs : rhs;
        if (useAlignedGather) {
            Gather(repeated, repeated[suffixOffset], offsets, 0, groupElements);
            PipeBarrier<PIPE_V>();
        } else {
            const event_t mte2ToScalar = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
            for (uint32_t repetition = 1; repetition < repeat; ++repetition) {
                const uint32_t destination = repetition * inner;
                for (uint32_t offset = 0; offset < inner; ++offset) {
                    repeated.SetValue(destination + offset, repeated.GetValue(offset));
                }
            }
            const event_t scalarToVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
        }

        LocalTensor<uint8_t> output = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        if constexpr (std::is_same<T, int32_t>::value) {
            ProcessInt32(lhs, rhs, output, mask, vectorCount);
        } else if constexpr (kNeedsConversion) {
            ProcessConverted(lhs, rhs, output, mask, vectorCount);
        } else {
            ProcessNative(lhs, rhs, output, mask, vectorCount);
        }
        yQueue_.EnQue(output);
        x1Queue_.FreeTensor(lhs);
        x2Queue_.FreeTensor(rhs);
        CopyOut(outputBegin, groupElements);
    }

    __aicore__ inline void ProcessNative(
        LocalTensor<T>& lhs, LocalTensor<T>& rhs, LocalTensor<uint8_t>& output,
        LocalTensor<uint8_t>& mask, uint32_t vectorCount)
    {
        LocalTensor<T> temp = tempBuffer_.Get<T>();
        for (uint32_t offset = 0; offset < vectorCount;
             offset += kMaxReliableCompareElements) {
            const uint32_t once = vectorCount - offset < kMaxReliableCompareElements
                                      ? vectorCount - offset
                                      : kMaxReliableCompareElements;
            Compare(mask, lhs[offset], rhs[offset], CMPMODE::GT, once);
            PipeBarrier<PIPE_V>();
            Duplicate(lhs[offset], static_cast<T>(1), once);
            PipeBarrier<PIPE_V>();
            Select(temp[offset], mask, lhs[offset], static_cast<T>(0),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, once);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<T, float>::value) {
                LocalTensor<half> halfTemp = lhs.template ReinterpretCast<half>();
                Cast(halfTemp[offset], temp[offset], RoundMode::CAST_NONE, once);
                PipeBarrier<PIPE_V>();
                Cast(output[offset], halfTemp[offset], RoundMode::CAST_NONE, once);
            } else {
                Cast(output[offset], temp[offset], RoundMode::CAST_NONE, once);
            }
        }
    }

    __aicore__ inline void ProcessConverted(
        LocalTensor<T>& lhsRaw, LocalTensor<T>& rhsRaw, LocalTensor<uint8_t>& output,
        LocalTensor<uint8_t>& mask, uint32_t vectorCount)
    {
        LocalTensor<CompareT> converted = conversionBuffer_.Get<CompareT>();
        LocalTensor<CompareT> lhs = converted;
        LocalTensor<CompareT> rhs = converted[tiling_.tileElements];
        LocalTensor<CompareT> temp = converted[2 * tiling_.tileElements];
        Cast(lhs, lhsRaw, RoundMode::CAST_NONE, vectorCount);
        Cast(rhs, rhsRaw, RoundMode::CAST_NONE, vectorCount);
        PipeBarrier<PIPE_V>();
        for (uint32_t offset = 0; offset < vectorCount;
             offset += kMaxReliableCompareElements) {
            const uint32_t once = vectorCount - offset < kMaxReliableCompareElements
                                      ? vectorCount - offset
                                      : kMaxReliableCompareElements;
            Compare(mask, lhs[offset], rhs[offset], CMPMODE::GT, once);
            PipeBarrier<PIPE_V>();
            Duplicate(lhs[offset], static_cast<CompareT>(1), once);
            PipeBarrier<PIPE_V>();
            Select(temp[offset], mask, lhs[offset], static_cast<CompareT>(0),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, once);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<CompareT, float>::value) {
                // A2 has no float -> uint8 Cast.  BF16 input storage is no
                // longer needed after widening and can hold the FP16 bridge.
                LocalTensor<half> halfTemp = lhsRaw.template ReinterpretCast<half>();
                Cast(halfTemp[offset], temp[offset], RoundMode::CAST_NONE, once);
                PipeBarrier<PIPE_V>();
                Cast(output[offset], halfTemp[offset], RoundMode::CAST_NONE, once);
            } else {
                Cast(output[offset], temp[offset], RoundMode::CAST_NONE, once);
            }
        }
    }

    __aicore__ inline void ProcessInt32(
        LocalTensor<T>& lhs, LocalTensor<T>& rhs, LocalTensor<uint8_t>& output,
        LocalTensor<uint8_t>& greaterMask, uint32_t vectorCount)
    {
        // For every signed INT32 pair, lhs > rhs iff max(lhs,rhs) != rhs.
        // EQ is the native A2 integer comparison. Invert at Select, so no
        // second Compare, Not, And, subtraction, or float conversion is needed.
        LocalTensor<int32_t> maximum = tempBuffer_.Get<int32_t>();
        Max(maximum, lhs, rhs, vectorCount);
        PipeBarrier<PIPE_V>();
        Compare(greaterMask, rhs, maximum, CMPMODE::EQ, vectorCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> zeros = lhs.template ReinterpretCast<half>();
        LocalTensor<half> result = rhs.template ReinterpretCast<half>();
        Duplicate(zeros, static_cast<half>(0), vectorCount);
        PipeBarrier<PIPE_V>();
        Select(result, greaterMask, zeros, static_cast<half>(1),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, vectorCount);
        PipeBarrier<PIPE_V>();
        Cast(output, result, RoundMode::CAST_NONE, vectorCount);
    }

    __aicore__ inline void ProcessScalarVector(uint64_t begin, uint32_t count)
    {
        ProcessScalarVectorAt(begin, begin, count, tiling_.mode == 2);
    }

    __aicore__ inline void ProcessScalarVectorAt(
        uint64_t outputBegin, uint64_t valueBegin, uint32_t count, bool lhsScalar)
    {
        const uint32_t vectorElements = 256 / sizeof(CompareT);
        const uint32_t vectorCount =
            (count + vectorElements - 1) / vectorElements * vectorElements;
        const uint32_t dmaElements = 32 / sizeof(T);
        const uint32_t dmaCount =
            (count + dmaElements - 1) / dmaElements * dmaElements;
        LocalTensor<T> values = lhsScalar ? x2Queue_.AllocTensor<T>() : x1Queue_.AllocTensor<T>();
        GlobalTensor<T>& source = lhsScalar ? x2_ : x1_;
        DataCopyExtParams inputParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{
            true, 0, static_cast<uint8_t>(dmaCount - count), static_cast<T>(0)};
        DataCopyPad(values, source[valueBegin], inputParams, pad);
        if (lhsScalar) {
            x2Queue_.EnQue(values);
            values = x2Queue_.DeQue<T>();
        } else {
            x1Queue_.EnQue(values);
            values = x1Queue_.DeQue<T>();
        }

        LocalTensor<uint8_t> output = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        if constexpr (std::is_same<T, int32_t>::value) {
            // A2 lacks INT32 GT and INT32->UINT8 Cast, but the exact
            // Max/EQ/mask route used by the tensor-tensor kernel is still
            // fully vectorizable. Materialize the changing scalar once per
            // contiguous suffix in the otherwise-unused input queue, then
            // reuse that route. This removes rank div/mod and two scalar GM
            // reads per output element from large scalar broadcasts.
            LocalTensor<T> scalarTensor = lhsScalar
                ? x1Queue_.AllocTensor<T>()
                : x2Queue_.AllocTensor<T>();
            Duplicate(scalarTensor, scalarValue_, vectorCount);
            PipeBarrier<PIPE_V>();
            if (lhsScalar) {
                ProcessInt32(scalarTensor, values, output, mask, vectorCount);
            } else {
                ProcessInt32(values, scalarTensor, output, mask, vectorCount);
            }
            if (lhsScalar) {
                x1Queue_.FreeTensor(scalarTensor);
            } else {
                x2Queue_.FreeTensor(scalarTensor);
            }
        } else if constexpr (kNeedsConversion) {
            ProcessConvertedScalar(values, output, mask, vectorCount, lhsScalar);
        } else {
            ProcessNativeScalar(values, output, mask, vectorCount, lhsScalar);
        }
        PipeBarrier<PIPE_ALL>();
        yQueue_.EnQue(output);
        if (lhsScalar) {
            x2Queue_.FreeTensor(values);
        } else {
            x1Queue_.FreeTensor(values);
        }
        CopyOut(outputBegin, count);
    }

    __aicore__ inline void ProcessNativeScalar(
        LocalTensor<T>& values, LocalTensor<uint8_t>& output, LocalTensor<uint8_t>& mask,
        uint32_t vectorCount, bool lhsScalar)
    {
        LocalTensor<T> temp = tempBuffer_.Get<T>();
        for (uint32_t offset = 0; offset < vectorCount;
             offset += kMaxReliableCompareElements) {
            const uint32_t once = vectorCount - offset < kMaxReliableCompareElements
                                      ? vectorCount - offset
                                      : kMaxReliableCompareElements;
            CompareScalar(mask, values[offset], scalarValue_,
                          lhsScalar ? CMPMODE::LT : CMPMODE::GT, once);
            PipeBarrier<PIPE_V>();
            Duplicate(values[offset], static_cast<T>(1), once);
            PipeBarrier<PIPE_V>();
            Select(temp[offset], mask, values[offset], static_cast<T>(0),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, once);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<T, float>::value) {
                LocalTensor<half> halfTemp = values.template ReinterpretCast<half>();
                Cast(halfTemp[offset], temp[offset], RoundMode::CAST_NONE, once);
                PipeBarrier<PIPE_V>();
                Cast(output[offset], halfTemp[offset], RoundMode::CAST_NONE, once);
            } else {
                Cast(output[offset], temp[offset], RoundMode::CAST_NONE, once);
            }
        }
    }

    __aicore__ inline CompareT ConvertedScalar() const
    {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            return AscendC::ToFloat(scalarValue_);
        } else {
            return static_cast<CompareT>(scalarValue_);
        }
    }

    __aicore__ inline void ProcessConvertedScalar(
        LocalTensor<T>& raw, LocalTensor<uint8_t>& output, LocalTensor<uint8_t>& mask,
        uint32_t vectorCount, bool lhsScalar)
    {
        LocalTensor<CompareT> converted = conversionBuffer_.Get<CompareT>();
        LocalTensor<CompareT> values = converted;
        LocalTensor<CompareT> temp = converted[tiling_.tileElements];
        Cast(values, raw, RoundMode::CAST_NONE, vectorCount);
        PipeBarrier<PIPE_V>();
        const CompareT scalar = ConvertedScalar();
        for (uint32_t offset = 0; offset < vectorCount;
             offset += kMaxReliableCompareElements) {
            const uint32_t once = vectorCount - offset < kMaxReliableCompareElements
                                      ? vectorCount - offset
                                      : kMaxReliableCompareElements;
            CompareScalar(mask, values[offset], scalar,
                          lhsScalar ? CMPMODE::LT : CMPMODE::GT, once);
            PipeBarrier<PIPE_V>();
            Duplicate(values[offset], static_cast<CompareT>(1), once);
            PipeBarrier<PIPE_V>();
            Select(temp[offset], mask, values[offset], static_cast<CompareT>(0),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, once);
            PipeBarrier<PIPE_V>();
            if constexpr (std::is_same<CompareT, float>::value) {
                LocalTensor<half> halfTemp = raw.template ReinterpretCast<half>();
                Cast(halfTemp[offset], temp[offset], RoundMode::CAST_NONE, once);
                PipeBarrier<PIPE_V>();
                Cast(output[offset], halfTemp[offset], RoundMode::CAST_NONE, once);
            } else {
                Cast(output[offset], temp[offset], RoundMode::CAST_NONE, once);
            }
        }
    }

    __aicore__ inline uint64_t BroadcastOffset(uint64_t linear, const uint64_t* strides) const
    {
        uint64_t offset = 0;
        for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= 0; --axis) {
            const uint64_t extent = tiling_.outputShape[axis];
            const uint64_t coordinate = extent == 0 ? 0 : linear % extent;
            linear = extent == 0 ? 0 : linear / extent;
            offset += coordinate * strides[axis];
        }
        return offset;
    }

    __aicore__ inline bool ScalarGreater(T lhs, T rhs) const
    {
        if constexpr (std::is_same<T, half>::value) {
            // AICore does not support scalar FP16 comparison directly.
            return static_cast<float>(lhs) > static_cast<float>(rhs);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            // BF16 scalar arithmetic is not native on A2.
            return AscendC::ToFloat(lhs) > AscendC::ToFloat(rhs);
        } else {
            return lhs > rhs;
        }
    }

    __aicore__ inline void ProcessScalar(uint64_t begin, uint32_t count)
    {
        LocalTensor<uint8_t> output = yQueue_.AllocTensor<uint8_t>();
        for (uint32_t local = 0; local < count; ++local) {
            const uint64_t linear = begin + local;
            const uint64_t lhsOffset = tiling_.mode == 0 ? linear : BroadcastOffset(linear, tiling_.x1Stride);
            const uint64_t rhsOffset = tiling_.mode == 0 ? linear : BroadcastOffset(linear, tiling_.x2Stride);
            output.SetValue(local, ScalarGreater(x1_.GetValue(lhsOffset), x2_.GetValue(rhsOffset)) ? 1 : 0);
        }
        PipeBarrier<PIPE_ALL>();
        yQueue_.EnQue(output);
        CopyOut(begin, count);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> x1Queue_;
    TQue<TPosition::VECIN, 1> x2Queue_;
    TQue<TPosition::VECOUT, 1> yQueue_;
    TBuf<TPosition::VECCALC> tempBuffer_;
    TBuf<TPosition::VECCALC> conversionBuffer_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    TBuf<TPosition::VECCALC> broadcastCacheBuffer_;
    GlobalTensor<T> x1_;
    GlobalTensor<T> x2_;
    GlobalTensor<uint8_t> y_;
    T scalarValue_{};
    GreaterTilingData tiling_{};
};
// A topology-driven broadcast tile. Both operands remain immutable in UB,
// allowing repeated outer broadcast addresses to reuse their expanded tile.
template <typename T, bool GENERATED = false>
class KernelGreaterGather {
public:
    using C = typename CompareType<T>::Type;
    __aicore__ inline void Init(GM_ADDR lhs, GM_ADDR rhs, GM_ADDR out, GM_ADDR tiling,
                               const GreaterGatherMeta& meta) {
        meta_ = meta;
        x1_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(lhs));
        x2_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rhs));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(out));
        mapsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling + sizeof(GreaterGatherMeta)));
        const uint32_t n = meta_.paddedSpan;
        pipe_.InitBuffer(raw1_, 1, n * sizeof(T));
        // Distinct DMA staging buffers allow the second load to overlap
        // the first operand's vector Gather.
        pipe_.InitBuffer(raw2_, 1, n * sizeof(T));
        pipe_.InitBuffer(out_, 1, n);
        if constexpr (GENERATED) pipe_.InitBuffer(generatedMaps_, 2 * S9_GREATER_GATHER_CAP * sizeof(uint32_t));
        else pipe_.InitBuffer(maps_, 1, 2 * S9_GREATER_GATHER_CAP * sizeof(uint32_t));
        pipe_.InitBuffer(values_, 2 * n * sizeof(C));
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, bfloat16_t>::value)
            pipe_.InitBuffer(convert_, n * sizeof(C));
        pipe_.InitBuffer(mask_, 8 * 1024);
        if constexpr (std::is_same<T, int32_t>::value)
            pipe_.InitBuffer(maximum_, 2048 * sizeof(int32_t));
        pipe_.InitBuffer(half_, 3 * n * sizeof(half));
        pipe_.InitBuffer(shift_, n * sizeof(uint32_t));
    }
    __aicore__ inline void Process() {
        uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * meta_.base.blockElements;
        const uint64_t end = MinU64(begin + meta_.base.blockElements, meta_.base.totalElements);
        if (begin >= end) return;
        const uint32_t n = meta_.paddedSpan;
        LocalTensor<uint32_t> map;
        if constexpr (GENERATED) {
            map = generatedMaps_.Get<uint32_t>();
            BuildSimpleMaps(map);
        } else {
        map = maps_.AllocTensor<uint32_t>();
        if (meta_.lhsCount != meta_.span) DataCopy(map, mapsGm_, n);
        if (meta_.rhsCount != meta_.span) DataCopy(map[S9_GREATER_GATHER_CAP], mapsGm_[S9_GREATER_GATHER_CAP], n);
        maps_.EnQue(map); map = maps_.DeQue<uint32_t>();
        }
        LocalTensor<C> lhs = values_.Get<C>();
        LocalTensor<C> rhs = lhs[n];
        LocalTensor<half> ones = half_.Get<half>();
        Duplicate(ones, static_cast<half>(std::is_same<T, int32_t>::value ? 0 : 1), n);
        PipeBarrier<PIPE_V>();
        uint64_t lastLhs = UINT64_MAX, lastRhs = UINT64_MAX;
        uint32_t leftCoverage = 0, rightCoverage = 0;
        uint64_t lastGroup = UINT64_MAX, outerLhs = 0, outerRhs = 0;
        while (begin < end) {
            const uint64_t group = begin / meta_.groupSpan;
            const uint64_t withinGroup = begin % meta_.groupSpan;
            const uint64_t chunk = withinGroup / meta_.span;
            const uint64_t rowBegin = group * meta_.groupSpan + chunk * meta_.span;
            const uint32_t validSpan = static_cast<uint32_t>(MinU64(meta_.span, meta_.groupSpan - chunk * meta_.span));
            const uint32_t actualSteps = validSpan / meta_.inner;
            const uint32_t start = static_cast<uint32_t>(begin - rowBegin);
            const uint32_t count = static_cast<uint32_t>(MinU64(validSpan - start, end - begin));
            const uint32_t prefix = start + count;
            const uint32_t compareCount = (prefix + 255) / 256 * 256;
            // The shape map already describes the suffix. Decode only the
            // outer mixed-radix group, and reuse it for all adjacent chunks.
            if (group != lastGroup) {
                outerLhs = 0; outerRhs = 0;
                uint64_t linear = group;
                for (int32_t axis = static_cast<int32_t>(meta_.splitAxis) - 1;
                     axis >= static_cast<int32_t>(S9_GREATER_MAX_RANK) - static_cast<int32_t>(meta_.base.rank); --axis) {
                    const uint64_t coord = linear % meta_.base.outputShape[axis];
                    linear /= meta_.base.outputShape[axis];
                    outerLhs += coord * meta_.base.x1Stride[axis];
                    outerRhs += coord * meta_.base.x2Stride[axis];
                }
                lastGroup = group;
            }
            const uint64_t leftOffset = outerLhs + chunk * meta_.steps * meta_.lhsStep;
            const uint64_t rightOffset = outerRhs + chunk * meta_.steps * meta_.rhsStep;
            if (leftOffset != lastLhs || prefix > leftCoverage) {
                const uint32_t expand = meta_.lhsStep == 0 ? n : compareCount;
                Load(x1_, raw1_, lhs, map, leftOffset, meta_.lhsStep == 0 ? meta_.lhsCount : meta_.lhsStep * actualSteps, meta_.lhsCount == meta_.span, expand);
                leftCoverage = meta_.lhsStep == 0 ? n : static_cast<uint32_t>(MinU64(expand, validSpan));
                lastLhs = leftOffset;
            }
            if (rightOffset != lastRhs || prefix > rightCoverage) {
                const uint32_t expand = meta_.rhsStep == 0 ? n : compareCount;
                Load(x2_, raw2_, rhs, map[S9_GREATER_GATHER_CAP], rightOffset, meta_.rhsStep == 0 ? meta_.rhsCount : meta_.rhsStep * actualSteps, meta_.rhsCount == meta_.span, expand);
                rightCoverage = meta_.rhsStep == 0 ? n : static_cast<uint32_t>(MinU64(expand, validSpan));
                lastRhs = rightOffset;
            }
            LocalTensor<half> selected = ones[n];
            // Keep packed comparison/Select spans within the validated 2048
            // lanes while amortizing DMA, index decoding and writes over 4K.
            for (uint32_t j = 0; j < compareCount; j += 2048) {
                const uint32_t once = compareCount - j < 2048 ? compareCount - j : 2048;
                LocalTensor<uint8_t> mask = mask_.Get<uint8_t>();
                if constexpr (std::is_same<T, int32_t>::value) {
                    LocalTensor<int32_t> mx = maximum_.Get<int32_t>();
                    Max(mx, lhs[j], rhs[j], once);
                    PipeBarrier<PIPE_V>();
                    Compare(mask, rhs[j], mx, CMPMODE::EQ, once);
                } else {
                    Compare(mask, lhs[j], rhs[j], CMPMODE::GT, once);
                }
                PipeBarrier<PIPE_V>();
                Select(selected[j], mask, ones,
                       static_cast<half>(std::is_same<T, int32_t>::value ? 1 : 0),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE, once);
                PipeBarrier<PIPE_V>();
            }
            LocalTensor<uint8_t> output = out_.AllocTensor<uint8_t>();
            if (start != 0) {
                // A core may begin inside a logical row, but owns complete
                // 32-byte output units. Gather makes its UB slice aligned.
                LocalTensor<int32_t> shift = shift_.Get<int32_t>();
                const uint32_t sliceCount = (count + 255) / 256 * 256;
                GreaterIndexRange(shift, static_cast<int32_t>(start), sliceCount);
                PipeBarrier<PIPE_V>();
                Muls(shift, shift, static_cast<int32_t>(sizeof(half)), sliceCount);
                PipeBarrier<PIPE_V>();
                LocalTensor<uint32_t> shiftU = shift.template ReinterpretCast<uint32_t>();
                LocalTensor<half> sliced = ones[2 * n];
                Gather(sliced, selected, shiftU, 0, count);
                PipeBarrier<PIPE_V>();
                Cast(output, sliced, RoundMode::CAST_NONE, count);
            } else {
                Cast(output, selected, RoundMode::CAST_NONE, compareCount);
            }
            out_.EnQue(output); output = out_.DeQue<uint8_t>();
            DataCopyExtParams p{1, count, 0, 0, 0};
            DataCopyPad(y_[begin], output, p);
            out_.FreeTensor(output);
            begin += count;
        }
        if constexpr (!GENERATED) maps_.FreeTensor(map);
    }
private:
    __aicore__ inline void ProduceSimpleMap(LocalTensor<int32_t> dst, LocalTensor<int32_t> index,
            LocalTensor<int32_t> quotient, LocalTensor<int32_t> remainder, uint32_t side) {
        const uint32_t n = meta_.paddedSpan;
        if (meta_.mapKinds[side] == 2) {
            Muls(remainder, quotient, static_cast<int32_t>(meta_.divisors[side]), n);
            PipeBarrier<PIPE_V>();
            Sub(remainder, index, remainder, n); PipeBarrier<PIPE_V>();
            Muls(dst, remainder, static_cast<int32_t>(sizeof(C)), n);
        } else {
            Muls(dst, quotient, static_cast<int32_t>(sizeof(C)), n);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void BuildSimpleMaps(LocalTensor<uint32_t> maps) {
        const uint32_t n = meta_.paddedSpan;
        LocalTensor<int32_t> left = maps.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> right = left[S9_GREATER_GATHER_CAP];
        if (meta_.mapKinds[0] == 0) Duplicate(left, static_cast<int32_t>(0), n);
        if (meta_.mapKinds[1] == 0) Duplicate(right, static_cast<int32_t>(0), n);
        const bool leftMath = meta_.mapKinds[0] >= 2;
        const bool rightMath = meta_.mapKinds[1] >= 2;
        if (!leftMath && !rightMath) { PipeBarrier<PIPE_V>(); return; }
        LocalTensor<int32_t> index = shift_.Get<int32_t>();
        LocalTensor<int32_t> quotient = values_.Get<int32_t>();
        LocalTensor<int32_t> remainder = half_.Get<int32_t>();
        GreaterIndexRange(index, 0, n); PipeBarrier<PIPE_V>();
        if (leftMath) {
            Muls(quotient, index, static_cast<int32_t>(meta_.reciprocals[0]), n);
            PipeBarrier<PIPE_V>();
            ShiftRight(quotient, quotient, static_cast<int32_t>(meta_.shifts[0]), static_cast<int32_t>(n));
            PipeBarrier<PIPE_V>();
            ProduceSimpleMap(left, index, quotient, remainder, 0);
        }
        if (rightMath) {
            if (!leftMath || meta_.divisors[0] != meta_.divisors[1]) {
                Muls(quotient, index, static_cast<int32_t>(meta_.reciprocals[1]), n);
                PipeBarrier<PIPE_V>();
                ShiftRight(quotient, quotient, static_cast<int32_t>(meta_.shifts[1]), static_cast<int32_t>(n));
                PipeBarrier<PIPE_V>();
            }
            ProduceSimpleMap(right, index, quotient, remainder, 1);
        }
    }
    __aicore__ inline void Load(GlobalTensor<T>& source, TQue<TPosition::VECIN, 1>& queue,
            LocalTensor<C> dst, LocalTensor<uint32_t> map, uint64_t offset, uint32_t count, bool identity, uint32_t expand) {
        LocalTensor<T> raw = queue.template AllocTensor<T>();
        const uint32_t dma = 32 / sizeof(T);
        const uint32_t padded = (count + dma - 1) / dma * dma;
        DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(padded - count), static_cast<T>(0)};
        DataCopyPad(raw, source[offset], p, pad);
        queue.EnQue(raw); raw = queue.template DeQue<T>();
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, bfloat16_t>::value) {
            if (identity) {
                // A contiguous operand needs only exact widening, no gather.
                Cast(dst, raw, RoundMode::CAST_NONE, expand);
            } else {
                LocalTensor<C> cv = convert_.Get<C>();
                const uint32_t vectors = 256 / sizeof(C);
                const uint32_t castCount = (count + vectors - 1) / vectors * vectors;
                Cast(cv, raw, RoundMode::CAST_NONE, castCount);
                PipeBarrier<PIPE_V>();
                Gather(dst, cv, map, 0, expand);
            }
        } else {
            if (identity) {
                // Vector Copy is bit-preserving, including NaN and signed zero.
                constexpr uint32_t lanes = 256 / sizeof(T);
                AscendC::Copy(dst, raw, static_cast<uint64_t>(lanes),
                    static_cast<uint8_t>(expand / lanes), {1, 1, 8, 8});
            } else {
                Gather(dst, raw, map, 0, expand);
            }
        }
        PipeBarrier<PIPE_V>();
        queue.FreeTensor(raw);
    }
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> raw1_, raw2_, maps_;
    TQue<TPosition::VECOUT, 1> out_;
    TBuf<TPosition::VECCALC> generatedMaps_, values_, convert_, mask_, maximum_, half_, shift_;
    GlobalTensor<T> x1_, x2_;
    GlobalTensor<uint8_t> y_;
    GlobalTensor<uint32_t> mapsGm_;
    GreaterGatherMeta meta_{};
};
// Dense and scalar comparisons use their own compile-time kernel and UB plan.
template <typename T, uint32_t BUFFER_COUNT = 1>
class KernelGreaterLinear {
public:
    using C = typename CompareType<T>::Type;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, const GreaterTilingData& d) {
        total_ = d.totalElements; block_ = d.blockElements; tile_ = d.tileElements; mode_ = d.mode;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        constexpr uint32_t buffers = BUFFER_COUNT;
        pipe_.InitBuffer(aq_, buffers, tile_ * sizeof(T));
        pipe_.InitBuffer(bq_, buffers, tile_ * sizeof(T));
        pipe_.InitBuffer(yq_, buffers, tile_);
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, bfloat16_t>::value)
            pipe_.InitBuffer(convert_, 2 * tile_ * sizeof(C));
        pipe_.InitBuffer(mask_, 8 * 1024);
        pipe_.InitBuffer(half_, 2 * 4096 * sizeof(half));
        if constexpr (std::is_same<T, int32_t>::value) {
            pipe_.InitBuffer(max_, 4096 * sizeof(int32_t));
            pipe_.InitBuffer(scalar_, 4096 * sizeof(int32_t));
        }
    }
    __aicore__ inline void Process() {
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * block_;
        const uint64_t end = MinU64(begin + block_, total_);
        if (begin >= end) return;
        const bool scalar = mode_ == 2 || mode_ == 3;
        C scalarValue{};
        if (scalar) {
            const T raw = mode_ == 2 ? a_.GetValue(0) : b_.GetValue(0);
            if constexpr (std::is_same<T, bfloat16_t>::value) scalarValue = ToFloat(raw);
            else scalarValue = static_cast<C>(raw);
            PipeBarrier<PIPE_ALL>();
        }
        const uint32_t constants = tile_ < 4096 ? tile_ : 4096;
        LocalTensor<half> ones = half_.Get<half>();
        Duplicate(ones, static_cast<half>(std::is_same<T, int32_t>::value ? 0 : 1), constants);
        if constexpr (std::is_same<T, int32_t>::value) {
            if (scalar) Duplicate(scalar_.Get<int32_t>(), scalarValue, constants);
        }
        PipeBarrier<PIPE_V>();
        for (uint64_t offset = begin; offset < end; offset += tile_) {
            const uint32_t count = static_cast<uint32_t>(MinU64(tile_, end - offset));
            const uint32_t n = (count + 255) / 256 * 256;
            LocalTensor<T> ar = aq_.AllocTensor<T>();
            if (mode_ == 2) Copy(ar, b_, offset, count); else Copy(ar, a_, offset, count);
            aq_.EnQue(ar);
            LocalTensor<T> br;
            if (!scalar) { br = bq_.AllocTensor<T>(); Copy(br, b_, offset, count); bq_.EnQue(br); }
            ar = aq_.DeQue<T>();
            if (!scalar) br = bq_.DeQue<T>();
            LocalTensor<C> av, bv;
            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, bfloat16_t>::value) {
                av = convert_.Get<C>(); bv = av[tile_];
                Cast(av, ar, RoundMode::CAST_NONE, n);
                if (!scalar) Cast(bv, br, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
            } else { av = ar; if (!scalar) bv = br; }
            LocalTensor<uint8_t> output = yq_.AllocTensor<uint8_t>();
            for (uint32_t j = 0; j < n; j += 4096) {
                const uint32_t once = n - j < 4096 ? n - j : 4096;
                LocalTensor<uint8_t> mask = mask_.Get<uint8_t>();
                if constexpr (std::is_same<T, int32_t>::value) {
                    LocalTensor<int32_t> left = av[j];
                    LocalTensor<int32_t> right;
                    if (scalar) { right = scalar_.Get<int32_t>(); if (mode_ == 2) { left = right; right = av[j]; } }
                    else right = bv[j];
                    LocalTensor<int32_t> maximum = max_.Get<int32_t>();
                    Max(maximum, left, right, once); PipeBarrier<PIPE_V>();
                    Compare(mask, right, maximum, CMPMODE::EQ, once);
                } else {
                    if (scalar) CompareScalar(mask, av[j], scalarValue, mode_ == 2 ? CMPMODE::LT : CMPMODE::GT, once);
                    else Compare(mask, av[j], bv[j], CMPMODE::GT, once);
                }
                PipeBarrier<PIPE_V>();
                LocalTensor<half> selected = ones[4096];
                Select(selected, mask, ones, static_cast<half>(std::is_same<T, int32_t>::value ? 1 : 0), SELMODE::VSEL_TENSOR_SCALAR_MODE, once);
                PipeBarrier<PIPE_V>();
                Cast(output[j], selected, RoundMode::CAST_NONE, once);
                PipeBarrier<PIPE_V>();
            }
            yq_.EnQue(output);
            aq_.FreeTensor(ar); if (!scalar) bq_.FreeTensor(br);
            output = yq_.DeQue<uint8_t>();
            DataCopyExtParams p{1, count, 0, 0, 0}; DataCopyPad(y_[offset], output, p);
            yq_.FreeTensor(output);
        }
    }
private:
    __aicore__ inline void Copy(LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t offset, uint32_t n) {
        constexpr uint32_t dma = 32 / sizeof(T);
        const uint32_t padCount = (n + dma - 1) / dma * dma - n;
        DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(padCount), static_cast<T>(0)};
        DataCopyPad(dst, src[offset], p, pad);
    }
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> aq_, bq_;
    TQue<TPosition::VECOUT, 1> yq_;
    TBuf<TPosition::VECCALC> convert_, mask_, half_, max_, scalar_;
    GlobalTensor<T> a_, b_;
    GlobalTensor<uint8_t> y_;
    uint64_t total_, block_;
    uint32_t tile_, mode_;
};
// Dense and scalar comparisons use their own compile-time kernel and UB plan.
template <typename T>
class KernelGreaterRuns {
public:
    using C = typename CompareType<T>::Type;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, const GreaterTilingData& d) {
        data_ = d; total_ = d.totalElements; block_ = d.blockElements; tile_ = d.tileElements; mode_ = d.mode;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        pipe_.InitBuffer(aq_, 1, tile_ * sizeof(T));
        pipe_.InitBuffer(bq_, 1, tile_ * sizeof(T));
        pipe_.InitBuffer(yq_, 1, tile_);
        if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, bfloat16_t>::value)
            pipe_.InitBuffer(convert_, 2 * tile_ * sizeof(C));
        pipe_.InitBuffer(mask_, 8 * 1024);
        pipe_.InitBuffer(half_, 2 * 4096 * sizeof(half));
        if constexpr (std::is_same<T, int32_t>::value) {
            pipe_.InitBuffer(max_, 4096 * sizeof(int32_t));
            pipe_.InitBuffer(scalar_, 4096 * sizeof(int32_t));
        }
    }
    __aicore__ inline void Process() {
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * block_;
        const uint64_t end = MinU64(begin + block_, total_);
        if (begin >= end) return;
        const bool scalar = mode_ == 6 || mode_ == 7;
        const bool lhsScalar = mode_ == 7;
        uint64_t lastScalar = UINT64_MAX;
        C scalarValue{};
        const uint32_t constants = tile_ < 4096 ? tile_ : 4096;
        LocalTensor<half> ones = half_.Get<half>();
        Duplicate(ones, static_cast<half>(std::is_same<T, int32_t>::value ? 0 : 1), constants);
        PipeBarrier<PIPE_V>();
        for (uint64_t offset = begin; offset < end;) {
            const uint64_t rowRemaining = data_.broadcastInner - offset % data_.broadcastInner;
            const uint32_t count = static_cast<uint32_t>(MinU64(MinU64(tile_, end - offset), rowRemaining));
            uint64_t offsetA = 0, offsetB = 0;
            if (mode_ == 4 || mode_ == 5) {
                // A contiguous repeated interval has a closed-form address.
                // Preserve the large DMA/vector path without rank divisions.
                const uint64_t innerOffset = data_.broadcastInner - rowRemaining;
                const uint64_t group = data_.broadcastInner * data_.broadcastRepeat;
                const uint64_t repeated = (offset / group) * data_.broadcastInner + innerOffset;
                offsetA = mode_ == 4 ? repeated : offset;
                offsetB = mode_ == 5 ? repeated : offset;
            } else {
                uint64_t linear = offset;
                for (int32_t axis = S9_GREATER_MAX_RANK - 1; axis >= static_cast<int32_t>(S9_GREATER_MAX_RANK) - static_cast<int32_t>(data_.rank); --axis) {
                    const uint64_t c = linear % data_.outputShape[axis];
                    linear /= data_.outputShape[axis];
                    offsetA += c * data_.x1Stride[axis]; offsetB += c * data_.x2Stride[axis];
                }
            }
            const uint64_t scalarOffset = lhsScalar ? offsetA : offsetB;
            if (scalar && scalarOffset != lastScalar) {
                const T raw = lhsScalar ? a_.GetValue(scalarOffset) : b_.GetValue(scalarOffset);
                if constexpr (std::is_same<T, bfloat16_t>::value) scalarValue = ToFloat(raw);
                else scalarValue = static_cast<C>(raw);
                PipeBarrier<PIPE_ALL>();
                if constexpr (std::is_same<T, int32_t>::value) {
                    Duplicate(scalar_.Get<int32_t>(), scalarValue, constants); PipeBarrier<PIPE_V>();
                }
                lastScalar = scalarOffset;
            }
            const uint32_t n = (count + 255) / 256 * 256;
            LocalTensor<T> ar = aq_.AllocTensor<T>();
            if (lhsScalar) Copy(ar, b_, offsetB, count); else Copy(ar, a_, offsetA, count);
            aq_.EnQue(ar);
            LocalTensor<T> br;
            if (!scalar) { br = bq_.AllocTensor<T>(); Copy(br, b_, offsetB, count); bq_.EnQue(br); }
            ar = aq_.DeQue<T>();
            if (!scalar) br = bq_.DeQue<T>();
            LocalTensor<C> av, bv;
            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, bfloat16_t>::value) {
                av = convert_.Get<C>(); bv = av[tile_];
                Cast(av, ar, RoundMode::CAST_NONE, n);
                if (!scalar) Cast(bv, br, RoundMode::CAST_NONE, n);
                PipeBarrier<PIPE_V>();
            } else { av = ar; if (!scalar) bv = br; }
            LocalTensor<uint8_t> output = yq_.AllocTensor<uint8_t>();
            for (uint32_t j = 0; j < n; j += 4096) {
                const uint32_t once = n - j < 4096 ? n - j : 4096;
                LocalTensor<uint8_t> mask = mask_.Get<uint8_t>();
                if constexpr (std::is_same<T, int32_t>::value) {
                    LocalTensor<int32_t> left = av[j];
                    LocalTensor<int32_t> right;
                    if (scalar) { right = scalar_.Get<int32_t>(); if (lhsScalar) { left = right; right = av[j]; } }
                    else right = bv[j];
                    LocalTensor<int32_t> maximum = max_.Get<int32_t>();
                    Max(maximum, left, right, once); PipeBarrier<PIPE_V>();
                    Compare(mask, right, maximum, CMPMODE::EQ, once);
                } else {
                    if (scalar) CompareScalar(mask, av[j], scalarValue, lhsScalar ? CMPMODE::LT : CMPMODE::GT, once);
                    else Compare(mask, av[j], bv[j], CMPMODE::GT, once);
                }
                PipeBarrier<PIPE_V>();
                LocalTensor<half> selected = ones[4096];
                Select(selected, mask, ones, static_cast<half>(std::is_same<T, int32_t>::value ? 1 : 0), SELMODE::VSEL_TENSOR_SCALAR_MODE, once);
                PipeBarrier<PIPE_V>();
                Cast(output[j], selected, RoundMode::CAST_NONE, once);
                PipeBarrier<PIPE_V>();
            }
            yq_.EnQue(output);
            aq_.FreeTensor(ar); if (!scalar) bq_.FreeTensor(br);
            output = yq_.DeQue<uint8_t>();
            DataCopyExtParams p{1, count, 0, 0, 0}; DataCopyPad(y_[offset], output, p);
            yq_.FreeTensor(output);
            offset += count;
        }
    }
private:
    __aicore__ inline void Copy(LocalTensor<T> dst, GlobalTensor<T>& src, uint64_t offset, uint32_t n) {
        constexpr uint32_t dma = 32 / sizeof(T);
        const uint32_t padCount = (n + dma - 1) / dma * dma - n;
        DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(padCount), static_cast<T>(0)};
        DataCopyPad(dst, src[offset], p, pad);
    }
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> aq_, bq_;
    TQue<TPosition::VECOUT, 1> yq_;
    TBuf<TPosition::VECCALC> convert_, mask_, half_, max_, scalar_;
    GlobalTensor<T> a_, b_;
    GlobalTensor<uint8_t> y_;
    uint64_t total_, block_;
    uint32_t tile_, mode_;
    GreaterTilingData data_;
};
// One small dense output needs no queue lifecycle or outer loop.
template <typename T>
class KernelGreaterTiny {
public:
    using C = typename CompareType<T>::Type;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, const GreaterTilingData& d) {
        count_ = static_cast<uint32_t>(d.totalElements);
        n_ = (count_ + 255) / 256 * 256;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        pipe_.InitBuffer(raw_, 2 * n_ * sizeof(T));
        pipe_.InitBuffer(out_, n_);
        if constexpr (std::is_same<T,int8_t>::value || std::is_same<T,bfloat16_t>::value)
            pipe_.InitBuffer(convert_, 2 * n_ * sizeof(C));
        pipe_.InitBuffer(mask_, 8192);
        pipe_.InitBuffer(half_, 8192);
        if constexpr (std::is_same<T,int32_t>::value) pipe_.InitBuffer(max_, 8192);
    }
    __aicore__ inline void Process() {
        if (count_ == 0) return;
        LocalTensor<T> ar = raw_.Get<T>(), br = ar[n_];
        const uint32_t dma = 32 / sizeof(T);
        const uint32_t padded = (count_ + dma - 1) / dma * dma;
        DataCopyExtParams p{1, static_cast<uint32_t>(count_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(padded - count_), static_cast<T>(0)};
        DataCopyPad(ar, a_, p, pad); DataCopyPad(br, b_, p, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        LocalTensor<C> av, bv;
        if constexpr (std::is_same<T,int8_t>::value || std::is_same<T,bfloat16_t>::value) {
            av = convert_.Get<C>(); bv = av[n_];
            Cast(av, ar, RoundMode::CAST_NONE, n_); Cast(bv, br, RoundMode::CAST_NONE, n_);
            PipeBarrier<PIPE_V>();
        } else { av = ar; bv = br; }
        LocalTensor<half> ones = half_.Get<half>();
        Duplicate(ones, static_cast<half>(std::is_same<T, int32_t>::value ? 0 : 1), 2048); PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> output = out_.Get<uint8_t>();
        for (uint32_t j = 0; j < n_; j += 2048) {
            const uint32_t once = n_ - j < 2048 ? n_ - j : 2048;
            LocalTensor<uint8_t> mask = mask_.Get<uint8_t>();
            if constexpr (std::is_same<T,int32_t>::value) {
                LocalTensor<int32_t> maximum = max_.Get<int32_t>();
                Max(maximum, av[j], bv[j], once); PipeBarrier<PIPE_V>();
                Compare(mask, bv[j], maximum, CMPMODE::EQ, once);
            } else Compare(mask, av[j], bv[j], CMPMODE::GT, once);
            PipeBarrier<PIPE_V>();
            LocalTensor<half> selected = ones[2048];
            Select(selected, mask, ones, static_cast<half>(std::is_same<T, int32_t>::value ? 1 : 0), SELMODE::VSEL_TENSOR_SCALAR_MODE, once);
            PipeBarrier<PIPE_V>();
            Cast(output[j], selected, RoundMode::CAST_NONE, once); PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams outp{1, count_, 0, 0, 0}; DataCopyPad(y_, output, outp);
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0); WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
    }
private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> raw_, out_, convert_, mask_, half_, max_;
    GlobalTensor<T> a_, b_;
    GlobalTensor<uint8_t> y_;
    uint32_t count_, n_;
};

// Compile-time dense/scalar memory plans; ordinary kernels above stay intact.
template <typename T, bool IS_SCALAR, bool STREAM_OUTPUT>
class KernelGreaterStream {
public:
    using C = typename CompareType<T>::Type;
    static constexpr bool CONVERT = std::is_same<T,int8_t>::value || std::is_same<T,bfloat16_t>::value;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, const GreaterTilingData& d) {
        total_=d.totalElements; block_=d.blockElements; tile_=static_cast<uint32_t>(d.tileElements); leftScalar_=d.mode==2;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a)); b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if constexpr (STREAM_OUTPUT) y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(aq_,2,tile_*sizeof(T));
        if constexpr (!IS_SCALAR) pipe_.InitBuffer(bq_,2,tile_*sizeof(T));
        pipe_.InitBuffer(yq_,2,tile_);
        if constexpr (CONVERT) pipe_.InitBuffer(convert_,(IS_SCALAR?1:2)*tile_*sizeof(C));
        pipe_.InitBuffer(mask_,8192); pipe_.InitBuffer(half_,2*4096*sizeof(half));
        if constexpr (std::is_same<T,int32_t>::value) pipe_.InitBuffer(max_,4096*sizeof(int32_t));
    }
    __aicore__ inline void Process() {
        uint64_t begin=static_cast<uint64_t>(GetBlockIdx())*block_;
        if (begin>=total_) return;
        const uint64_t end=begin+MinU64(block_,total_-begin);
        C scalar{};
        if constexpr (IS_SCALAR) {
            const T raw=leftScalar_?a_.GetValue(0):b_.GetValue(0);
            if constexpr (std::is_same<T,bfloat16_t>::value) scalar=ToFloat(raw); else scalar=static_cast<C>(raw);
            PipeBarrier<PIPE_ALL>();
        }
        LocalTensor<half> ones=half_.Get<half>();
        Duplicate(ones,static_cast<half>(std::is_same<T,int32_t>::value?0:1),tile_<4096?tile_:4096); PipeBarrier<PIPE_V>();
        for (uint64_t offset=begin;offset<end;offset+=tile_) {
            const uint32_t count=static_cast<uint32_t>(MinU64(tile_,end-offset));
            const uint32_t n=(count+255)/256*256;
            LocalTensor<T> ar=aq_.AllocTensor<T>();
            if constexpr (IS_SCALAR) { if(leftScalar_) Copy(ar,b_,offset,count); else Copy(ar,a_,offset,count); }
            else Copy(ar,a_,offset,count);
            aq_.EnQue(ar);
            LocalTensor<T> br;
            if constexpr (!IS_SCALAR) { br=bq_.AllocTensor<T>(); Copy(br,b_,offset,count); bq_.EnQue(br); }
            ar=aq_.DeQue<T>(); if constexpr (!IS_SCALAR) br=bq_.DeQue<T>();
            LocalTensor<C> av,bv;
            if constexpr (CONVERT) {
                av=convert_.Get<C>(); Cast(av,ar,RoundMode::CAST_NONE,n);
                if constexpr (!IS_SCALAR) { bv=av[tile_]; Cast(bv,br,RoundMode::CAST_NONE,n); }
                PipeBarrier<PIPE_V>();
            } else { av=ar; if constexpr (!IS_SCALAR) bv=br; }
            LocalTensor<uint8_t> output=yq_.AllocTensor<uint8_t>();
            for(uint32_t j=0;j<n;j+=4096) {
                const uint32_t once=n-j<4096?n-j:4096;
                LocalTensor<uint8_t> mask=mask_.Get<uint8_t>();
                if constexpr (std::is_same<T,int32_t>::value) {
                    LocalTensor<int32_t> maximum=max_.Get<int32_t>();
                    if constexpr (IS_SCALAR) {
                        if(leftScalar_) Mins(maximum,av[j],scalar,once); else Maxs(maximum,av[j],scalar,once); PipeBarrier<PIPE_V>();
                        CompareScalar(mask,maximum,scalar,CMPMODE::EQ,once);
                    } else { Max(maximum,av[j],bv[j],once); PipeBarrier<PIPE_V>(); Compare(mask,bv[j],maximum,CMPMODE::EQ,once); }
                } else {
                    if constexpr (IS_SCALAR) CompareScalar(mask,av[j],scalar,leftScalar_?CMPMODE::LT:CMPMODE::GT,once);
                    else Compare(mask,av[j],bv[j],CMPMODE::GT,once);
                }
                PipeBarrier<PIPE_V>();
                LocalTensor<half> selected=ones[4096];
                Select(selected,mask,ones,static_cast<half>(std::is_same<T,int32_t>::value?1:0),SELMODE::VSEL_TENSOR_SCALAR_MODE,once);
                PipeBarrier<PIPE_V>(); Cast(output[j],selected,RoundMode::CAST_NONE,once); PipeBarrier<PIPE_V>();
            }
            yq_.EnQue(output); aq_.FreeTensor(ar); if constexpr (!IS_SCALAR) bq_.FreeTensor(br);
            output=yq_.DeQue<uint8_t>(); DataCopyExtParams p{1,count,0,0,0}; DataCopyPad(y_[offset],output,p); yq_.FreeTensor(output);
        }
    }
private:
    __aicore__ inline void Copy(LocalTensor<T> dst, GlobalTensor<T>& src,uint64_t off,uint32_t n) {
        constexpr uint32_t dma=32/sizeof(T); const uint32_t padding=(n+dma-1)/dma*dma-n;
        DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(padding),static_cast<T>(0)}; DataCopyPad(dst,src[off],p,pad);
    }
    TPipe pipe_; TQue<TPosition::VECIN,1> aq_,bq_; TQue<TPosition::VECOUT,1> yq_;
    TBuf<TPosition::VECCALC> convert_,mask_,half_,max_; GlobalTensor<T> a_,b_; GlobalTensor<uint8_t> y_;
    uint64_t total_,block_; uint32_t tile_; bool leftScalar_;
};

template <typename T, bool STREAM_OUTPUT>
class KernelGreaterScalarRows {
public:
    using C=typename CompareType<T>::Type;
    static constexpr bool CONVERT=std::is_same<T,int8_t>::value||std::is_same<T,bfloat16_t>::value;
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR y,const GreaterTilingData& d) {
        d_=d; tile_=static_cast<uint32_t>(d.tileElements); leftScalar_=d.mode==7;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a)); b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b)); y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if constexpr (STREAM_OUTPUT) y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        scalarCount_=1; for(uint32_t i=0;i<8;++i) scalarCount_*=leftScalar_?d.x1Shape[i]:d.x2Shape[i];
        prefix_=8; uint64_t span=1; while(prefix_>0&&span<d.broadcastInner) span*=d.outputShape[--prefix_];
        pipe_.InitBuffer(in_,2,tile_*sizeof(T)); pipe_.InitBuffer(out_,2,tile_);
        pipe_.InitBuffer(pages_,512*sizeof(T));
        if constexpr(CONVERT) pipe_.InitBuffer(convert_,tile_*sizeof(C));
        pipe_.InitBuffer(mask_,8192); pipe_.InitBuffer(half_,2*4096*sizeof(half));
        if constexpr(std::is_same<T,int32_t>::value) pipe_.InitBuffer(max_,4096*sizeof(int32_t));
    }
    __aicore__ inline void Process() {
        uint64_t pos=static_cast<uint64_t>(GetBlockIdx())*d_.blockElements;
        if(pos>=d_.totalElements)return;
        const uint64_t end=pos+MinU64(d_.blockElements,d_.totalElements-pos);
        uint64_t row=pos/d_.broadcastInner,phase=pos-row*d_.broadcastInner;
        InitRow(row);
        LocalTensor<half> ones=half_.Get<half>();
        Duplicate(ones,static_cast<half>(std::is_same<T,int32_t>::value?0:1),tile_<4096?tile_:4096); PipeBarrier<PIPE_V>();
        uint64_t lastScalar=UINT64_MAX; C value{};
        while(pos<end) {
            const uint32_t count=static_cast<uint32_t>(MinU64(MinU64(tile_,end-pos),d_.broadcastInner-phase));
            const uint32_t n=(count+255)/256*256;
            const uint64_t scalarOffset=leftScalar_?baseA_:baseB_;
            if(scalarOffset!=lastScalar) { value=ScalarAt(scalarOffset); lastScalar=scalarOffset; }
            const uint64_t vectorOffset=(leftScalar_?baseB_:baseA_)+phase;
            if(!live_||vectorOffset!=vectorOffset_||count>vectorCount_) {
                if(live_)in_.FreeTensor(raw_);
                raw_=in_.AllocTensor<T>();
                if(leftScalar_)Copy(raw_,b_,vectorOffset,count);else Copy(raw_,a_,vectorOffset,count);
                in_.EnQue(raw_); raw_=in_.DeQue<T>();
                if constexpr(CONVERT) { Cast(convert_.Get<C>(),raw_,RoundMode::CAST_NONE,n); PipeBarrier<PIPE_V>(); }
                live_=true; vectorOffset_=vectorOffset; vectorCount_=count;
            }
            LocalTensor<C> av;
            if constexpr(CONVERT)av=convert_.Get<C>();else av=raw_;
            LocalTensor<uint8_t> output=out_.AllocTensor<uint8_t>();
            for(uint32_t j=0;j<n;j+=4096) {
                uint32_t once=n-j<4096?n-j:4096; LocalTensor<uint8_t> mask=mask_.Get<uint8_t>();
                if constexpr(std::is_same<T,int32_t>::value) {
                    auto maximum=max_.Get<int32_t>(); if(leftScalar_)Mins(maximum,av[j],value,once);else Maxs(maximum,av[j],value,once); PipeBarrier<PIPE_V>();
                    CompareScalar(mask,maximum,value,CMPMODE::EQ,once);
                } else CompareScalar(mask,av[j],value,leftScalar_?CMPMODE::LT:CMPMODE::GT,once);
                PipeBarrier<PIPE_V>(); auto selected=ones[4096];
                Select(selected,mask,ones,static_cast<half>(std::is_same<T,int32_t>::value?1:0),SELMODE::VSEL_TENSOR_SCALAR_MODE,once);
                PipeBarrier<PIPE_V>(); Cast(output[j],selected,RoundMode::CAST_NONE,once); PipeBarrier<PIPE_V>();
            }
            out_.EnQue(output); output=out_.DeQue<uint8_t>(); DataCopyExtParams p{1,count,0,0,0};
            // Consecutive unaligned rows can share one GM32B block.
            // Order writes on MTE3 without serializing the next row computation.
            if(pos&31)PipeBarrier<PIPE_MTE3>();
            DataCopyPad(y_[pos],output,p);out_.FreeTensor(output);
            pos+=count;phase+=count;
            if(phase==d_.broadcastInner&&pos<end){phase=0;NextRow();}
        }
        if(live_)in_.FreeTensor(raw_);
    }
private:
    __aicore__ inline void InitRow(uint64_t row) {
        baseA_=baseB_=0;
        for(int32_t i=static_cast<int32_t>(prefix_)-1;i>=0;--i) {
            if(d_.outputShape[i]<=1){coords_[i]=0;continue;}
            const uint64_t c=row%d_.outputShape[i];row/=d_.outputShape[i];coords_[i]=c;
            baseA_+=c*d_.x1Stride[i];baseB_+=c*d_.x2Stride[i];
        }
    }
    __aicore__ inline void NextRow() {
        for(int32_t i=static_cast<int32_t>(prefix_)-1;i>=0;--i) {
            if(d_.outputShape[i]<=1)continue;
            ++coords_[i];baseA_+=d_.x1Stride[i];baseB_+=d_.x2Stride[i];
            if(coords_[i]<d_.outputShape[i])break;
            coords_[i]=0;baseA_-=d_.outputShape[i]*d_.x1Stride[i];baseB_-=d_.outputShape[i]*d_.x2Stride[i];
        }
    }
    __aicore__ inline C ScalarAt(uint64_t offset) {
        const uint64_t page=offset/256*256; const uint32_t slot=static_cast<uint32_t>((offset/256)&1);
        auto cache=pages_.Get<T>()[slot*256];
        if(page!=pageTags_[slot]) {
            SetFlag<HardEvent::S_MTE2>(EVENT_ID3);WaitFlag<HardEvent::S_MTE2>(EVENT_ID3);
            const uint32_t n=static_cast<uint32_t>(MinU64(256,scalarCount_-page));
            if(leftScalar_)Copy(cache,a_,page,n);else Copy(cache,b_,page,n);
            SetFlag<HardEvent::MTE2_S>(EVENT_ID3);WaitFlag<HardEvent::MTE2_S>(EVENT_ID3);pageTags_[slot]=page;
        }
        const T raw=cache.GetValue(static_cast<uint32_t>(offset-page));
        if constexpr(std::is_same<T,bfloat16_t>::value)return ToFloat(raw);else return static_cast<C>(raw);
    }
    __aicore__ inline void Copy(LocalTensor<T> dst,GlobalTensor<T>& src,uint64_t off,uint32_t n) {
        constexpr uint32_t dma=32/sizeof(T);DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((n+dma-1)/dma*dma-n),static_cast<T>(0)};DataCopyPad(dst,src[off],p,pad);
    }
    TPipe pipe_;TQue<TPosition::VECIN,1> in_;TQue<TPosition::VECOUT,1> out_;
    TBuf<TPosition::VECCALC> pages_,convert_,mask_,half_,max_;LocalTensor<T> raw_;
    GlobalTensor<T> a_,b_;GlobalTensor<uint8_t> y_;GreaterTilingData d_{};
    uint64_t scalarCount_,baseA_=0,baseB_=0,coords_[8]{},pageTags_[2]{UINT64_MAX,UINT64_MAX},vectorOffset_=0;
    uint32_t tile_,prefix_,vectorCount_=0;bool leftScalar_=false,live_=false;
};


// A group contains a repeated vector and one input contiguous within that group.
template <typename T, bool STREAM_OUTPUT>
class KernelGreaterPeriodic {
public:
    using C=typename CompareType<T>::Type;
    static constexpr bool CONVERT=std::is_same<T,int8_t>::value||std::is_same<T,bfloat16_t>::value;
    __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR y,const GreaterTilingData& d) {
        d_=d;tile_=static_cast<uint32_t>(d.tileElements);period_=static_cast<uint32_t>(d.broadcastInner);
        padded_=(period_+255)/256*256;groupSpan_=d.broadcastInner*d.broadcastRepeat;leftRepeat_=d.mode==15;
        first_=8;uint64_t span=1;while(first_>0&&span<groupSpan_)span*=d.outputShape[--first_];
        uint32_t power=1;shift_=15;while(power<period_){power<<=1;++shift_;}
        reciprocal_=((uint32_t{1}<<shift_)+period_-1)/period_;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if constexpr(STREAM_OUTPUT)y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(in_,2,tile_*sizeof(T));pipe_.InitBuffer(out_,2,tile_);
        pipe_.InitBuffer(rawCache_,padded_*sizeof(T));
        if constexpr(CONVERT){pipe_.InitBuffer(convert_,tile_*sizeof(C));pipe_.InitBuffer(convertCache_,padded_*sizeof(C));}
        pipe_.InitBuffer(expanded_,tile_*sizeof(C));pipe_.InitBuffer(indices_,2*tile_*sizeof(int32_t));
        pipe_.InitBuffer(mask_,8192);pipe_.InitBuffer(half_,2*tile_*sizeof(half));
        if constexpr(std::is_same<T,int32_t>::value)pipe_.InitBuffer(max_,tile_*sizeof(int32_t));
    }
    __aicore__ inline void Process() {
        uint64_t pos=static_cast<uint64_t>(GetBlockIdx())*d_.blockElements;
        if(pos>=d_.totalElements)return;
        const uint64_t end=pos+MinU64(d_.blockElements,d_.totalElements-pos);
        uint64_t group=pos<groupSpan_?0:pos/groupSpan_,groupBase=group*groupSpan_,groupEnd=groupBase+groupSpan_;
        uint32_t phase=static_cast<uint32_t>(pos%period_);uint64_t activeGroup=UINT64_MAX,baseA=0,baseB=0;
        auto ones=half_.Get<half>();Duplicate(ones,static_cast<half>(std::is_same<T,int32_t>::value?0:1),tile_);PipeBarrier<PIPE_V>();
        while(pos<end) {
            if(pos==groupEnd){++group;groupBase=groupEnd;groupEnd+=groupSpan_;}
            if(group!=activeGroup){Decode(group,baseA,baseB);LoadRepeat(leftRepeat_?baseA:baseB);activeGroup=group;}
            const uint32_t count=static_cast<uint32_t>(MinU64(MinU64(tile_,end-pos),groupEnd-pos));
            const uint32_t n=(count+255)/256*256;
            auto ar=in_.AllocTensor<T>();const uint64_t denseOffset=(leftRepeat_?baseB:baseA)+(pos-groupBase);
            if(leftRepeat_)Copy(ar,b_,denseOffset,count);else Copy(ar,a_,denseOffset,count);
            in_.EnQue(ar);
            LocalTensor<C> repeated=Repeated(phase,n);
            ar=in_.DeQue<T>();LocalTensor<C> dense;
            if constexpr(CONVERT){dense=convert_.Get<C>();Cast(dense,ar,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}else dense=ar;
            auto output=out_.AllocTensor<uint8_t>();auto mask=mask_.Get<uint8_t>();
            if constexpr(std::is_same<T,int32_t>::value){
                auto maximum=max_.Get<int32_t>();Max(maximum,dense,repeated,n);PipeBarrier<PIPE_V>();
                if(leftRepeat_)Compare(mask,dense,maximum,CMPMODE::EQ,n);else Compare(mask,repeated,maximum,CMPMODE::EQ,n);
            }else{
                if(leftRepeat_)Compare(mask,repeated,dense,CMPMODE::GT,n);else Compare(mask,dense,repeated,CMPMODE::GT,n);
            }
            PipeBarrier<PIPE_V>();auto selected=ones[tile_];
            Select(selected,mask,ones,static_cast<half>(std::is_same<T,int32_t>::value?1:0),SELMODE::VSEL_TENSOR_SCALAR_MODE,n);
            PipeBarrier<PIPE_V>();Cast(output,selected,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
            out_.EnQue(output);in_.FreeTensor(ar);output=out_.DeQue<uint8_t>();
            DataCopyExtParams p{1,count,0,0,0};DataCopyPad(y_[pos],output,p);out_.FreeTensor(output);
            // Adjacent writes inside one core may share an unaligned GM block.
            if((pos|count)&31){SetFlag<HardEvent::MTE3_S>(EVENT_ID6);WaitFlag<HardEvent::MTE3_S>(EVENT_ID6);}
            pos+=count;phase+=count;
            if(phase>=period_){if(tile_<=period_)phase-=period_;else phase-=((phase*reciprocal_)>>shift_)*period_;}
        }
    }
private:
    __aicore__ inline void Decode(uint64_t group,uint64_t& a,uint64_t& b){
        a=b=0;
        for(int32_t axis=static_cast<int32_t>(first_)-1;axis>=0;--axis){
            if(d_.outputShape[axis]<=1)continue;
            uint64_t c=group%d_.outputShape[axis];group/=d_.outputShape[axis];
            a+=c*d_.x1Stride[axis];b+=c*d_.x2Stride[axis];
        }
    }
    __aicore__ inline void LoadRepeat(uint64_t offset){
        if(offset==repeatOffset_)return;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID5);WaitFlag<HardEvent::V_MTE2>(EVENT_ID5);
        auto raw=rawCache_.Get<T>();if(leftRepeat_)Copy(raw,a_,offset,period_);else Copy(raw,b_,offset,period_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID5);WaitFlag<HardEvent::MTE2_V>(EVENT_ID5);
        if constexpr(CONVERT){Cast(convertCache_.Get<C>(),raw,RoundMode::CAST_NONE,padded_);PipeBarrier<PIPE_V>();}
        repeatOffset_=offset;expandedPhase_=UINT32_MAX;expandedCount_=0;
    }
    __aicore__ inline LocalTensor<C> Repeated(uint32_t phase,uint32_t n){
        LocalTensor<C> cached;
        if constexpr(CONVERT)cached=convertCache_.Get<C>();else cached=rawCache_.Get<C>();
        if((phase*sizeof(C))%32==0&&phase+n<=period_)return cached[phase];
        auto dst=expanded_.Get<C>();
        if(expandedPhase_==phase&&expandedCount_>=n)return dst;
        auto index=indices_.Get<int32_t>();auto q=index[tile_];
        if(mapPhase_!=phase||mapCount_<n){
            const uint32_t count=tile_;
            if(mapPhase_!=UINT32_MAX){
                // Stored offsets already include sizeof(C). A negative modular
                // displacement needs at most one correction per lane.
                int32_t delta=static_cast<int32_t>(phase)-static_cast<int32_t>(mapPhase_);
                if(delta>0)delta-=static_cast<int32_t>(period_);
                Adds(index,index,delta*static_cast<int32_t>(sizeof(C)),count);PipeBarrier<PIPE_V>();
                ShiftRight(q.template ReinterpretCast<uint32_t>(),index.template ReinterpretCast<uint32_t>(),static_cast<uint32_t>(31),static_cast<int32_t>(count));PipeBarrier<PIPE_V>();
                Muls(q,q,static_cast<int32_t>(period_*sizeof(C)),count);PipeBarrier<PIPE_V>();
                Add(index,index,q,count);PipeBarrier<PIPE_V>();
            }else{
                if(tile_<=period_){
                    GreaterIndexRange(index,static_cast<int32_t>(phase)-static_cast<int32_t>(period_),count);PipeBarrier<PIPE_V>();
                    ShiftRight(q.template ReinterpretCast<uint32_t>(),index.template ReinterpretCast<uint32_t>(),static_cast<uint32_t>(31),static_cast<int32_t>(count));PipeBarrier<PIPE_V>();
                    Muls(q,q,static_cast<int32_t>(period_),count);PipeBarrier<PIPE_V>();Add(index,index,q,count);PipeBarrier<PIPE_V>();
                }else{
                    GreaterIndexRange(index,static_cast<int32_t>(phase),count);PipeBarrier<PIPE_V>();
                    Muls(q,index,static_cast<int32_t>(reciprocal_),count);PipeBarrier<PIPE_V>();
                    ShiftRight(q,q,static_cast<int32_t>(shift_),static_cast<int32_t>(count));PipeBarrier<PIPE_V>();
                    Muls(q,q,static_cast<int32_t>(period_),count);PipeBarrier<PIPE_V>();Sub(index,index,q,count);PipeBarrier<PIPE_V>();
                }
                Muls(index,index,static_cast<int32_t>(sizeof(C)),count);PipeBarrier<PIPE_V>();
            }
            mapPhase_=phase;mapCount_=count;
        }
        Gather(dst,cached,index.template ReinterpretCast<uint32_t>(),0,n);PipeBarrier<PIPE_V>();expandedPhase_=phase;expandedCount_=n;return dst;
    }
    __aicore__ inline void Copy(LocalTensor<T> dst,GlobalTensor<T>& src,uint64_t off,uint32_t n){
        constexpr uint32_t dma=32/sizeof(T);DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>((n+dma-1)/dma*dma-n),static_cast<T>(0)};DataCopyPad(dst,src[off],p,pad);
    }
    TPipe pipe_;TQue<TPosition::VECIN,1> in_;TQue<TPosition::VECOUT,1> out_;
    TBuf<TPosition::VECCALC> rawCache_,convertCache_,convert_,expanded_,indices_,mask_,half_,max_;
    GlobalTensor<T> a_,b_;GlobalTensor<uint8_t> y_;GreaterTilingData d_{};
    uint64_t groupSpan_,repeatOffset_=UINT64_MAX;uint32_t period_,padded_,tile_,first_,shift_,reciprocal_;
    uint32_t mapPhase_=UINT32_MAX,mapCount_=0,expandedPhase_=UINT32_MAX,expandedCount_=0;bool leftRepeat_;
};


#if S9_HAS_ROW
template<typename T,bool BYPASS_INPUT=false,bool BYPASS_OUTPUT=false>
class KernelGreaterRowPad {
 using C=typename CompareType<T>::Type;
 static constexpr bool CONVERT=std::is_same<T,bfloat16_t>::value;
public:
 __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR y,const GreaterRowTilingData& d){
  d_=d;width_=static_cast<uint32_t>(d.width);pad_=d.padded;rows_=d.rows;tile_=pad_*rows_;left_=d.leftRepeated!=0;
  a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
  if constexpr(BYPASS_INPUT){a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);}
  if constexpr(BYPASS_OUTPUT)y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  pipe_.InitBuffer(bq_,2,tile_*sizeof(T));pipe_.InitBuffer(yq_,2,tile_);
  pipe_.InitBuffer(cacheRaw_,pad_*sizeof(T));pipe_.InitBuffer(expanded_,tile_*sizeof(C));
  if constexpr(CONVERT){pipe_.InitBuffer(cacheC_,pad_*sizeof(C));pipe_.InitBuffer(denseC_,tile_*sizeof(C));}
  if constexpr(std::is_same<T,int32_t>::value)pipe_.InitBuffer(maximum_,tile_*sizeof(int32_t));
  pipe_.InitBuffer(mask_,512);pipe_.InitBuffer(half_,8192*sizeof(half));
 }
 __aicore__ inline void Process(){
  uint64_t row=static_cast<uint64_t>(GetBlockIdx())*d_.blockRows;if(row>=d_.totalRows)return;
  const uint64_t end=MinU64(row+d_.blockRows,d_.totalRows);
  auto ones=half_.Get<half>();Duplicate(ones,static_cast<half>(std::is_same<T,int32_t>::value?0:1),tile_<4096?tile_:4096);PipeBarrier<PIPE_V>();
  uint32_t nrows=Batch(row,end);LoadDense(row,nrows);
  while(row<end){
   EnsureVector(row/d_.repeat);auto raw=bq_.DeQue<T>();
   uint64_t next=row+nrows;uint32_t nextRows=0;
   if(next<end){nextRows=Batch(next,end);LoadDense(next,nextRows);}
   const uint32_t n=nrows*pad_;LocalTensor<C> dense;
   if constexpr(CONVERT){dense=denseC_.Get<C>();Cast(dense,raw,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}else dense=raw;
   auto expanded=expanded_.Get<C>();auto output=yq_.AllocTensor<uint8_t>();
   for(uint32_t j=0;j<n;j+=4096){
    const uint32_t count=n-j<4096?n-j:4096;auto mask=mask_.Get<uint8_t>();
    if constexpr(std::is_same<T,int32_t>::value){
     auto mx=maximum_.Get<int32_t>();Max(mx,expanded[j],dense[j],count);PipeBarrier<PIPE_V>();
     if(left_)Compare(mask,dense[j],mx,CMPMODE::EQ,count);else Compare(mask,expanded[j],mx,CMPMODE::EQ,count);
    }else{if(left_)Compare(mask,expanded[j],dense[j],CMPMODE::GT,count);else Compare(mask,dense[j],expanded[j],CMPMODE::GT,count);}
    PipeBarrier<PIPE_V>();auto selected=ones[4096];Select(selected,mask,ones,static_cast<half>(std::is_same<T,int32_t>::value?1:0),SELMODE::VSEL_TENSOR_SCALAR_MODE,count);
    PipeBarrier<PIPE_V>();Cast(output[j],selected,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
   }
   yq_.EnQue(output);bq_.FreeTensor(raw);output=yq_.DeQue<uint8_t>();
   const uint64_t pos=row*d_.width;const uint32_t rounded=(width_+31)/32*32;
   DataCopyExtParams cp{static_cast<uint16_t>(nrows),width_,(pad_-rounded)/32,0,0};
   if(pos&31)PipeBarrier<PIPE_MTE3>();DataCopyPad(y_[pos],output,cp);yq_.FreeTensor(output);
   row=next;nrows=nextRows;
  }
 }
private:
 __aicore__ inline uint32_t Batch(uint64_t row,uint64_t end){return static_cast<uint32_t>(MinU64(MinU64(rows_,end-row),d_.repeat-row%d_.repeat));}
 __aicore__ inline void ZeroExtra(LocalTensor<T> raw,uint32_t nr){
  constexpr uint32_t dma=32/sizeof(T);if(pad_==(width_+dma-1)/dma*dma)return;
  auto words=raw.template ReinterpretCast<int32_t>();
  for(uint32_t i=0;i<nr;++i)Duplicate(words[(i*pad_+pad_-128)*sizeof(T)/4],static_cast<int32_t>(0),128*sizeof(T)/4);
 }
 __aicore__ inline void LoadDense(uint64_t row,uint32_t nr){
  auto raw=bq_.AllocTensor<T>();ZeroExtra(raw,nr);SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
  constexpr uint32_t dma=32/sizeof(T);uint32_t rounded=(width_+dma-1)/dma*dma;
  DataCopyExtParams cp{static_cast<uint16_t>(nr),static_cast<uint32_t>(width_*sizeof(T)),0,static_cast<uint32_t>((pad_-rounded)*sizeof(T)/32),0};
  DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>(rounded-width_),static_cast<T>(0)};
  if(left_)DataCopyPad(raw,b_[row*d_.width],cp,pp);else DataCopyPad(raw,a_[row*d_.width],cp,pp);
  bq_.EnQue(raw);
 }
 __aicore__ inline void EnsureVector(uint64_t group){
  if(group==cachedGroup_)return;auto raw=cacheRaw_.Get<T>();ZeroExtra(raw,1);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
  constexpr uint32_t dma=32/sizeof(T);uint32_t rounded=(width_+dma-1)/dma*dma;
  DataCopyExtParams cp{1,static_cast<uint32_t>(width_*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>(rounded-width_),static_cast<T>(0)};
  if(left_)DataCopyPad(raw,a_[group*d_.width],cp,pp);else DataCopyPad(raw,b_[group*d_.width],cp,pp);
  SetFlag<HardEvent::MTE2_V>(EVENT_ID6);WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
  LocalTensor<C> cached;if constexpr(CONVERT){cached=cacheC_.Get<C>();Cast(cached,raw,RoundMode::CAST_NONE,pad_);PipeBarrier<PIPE_V>();}else cached=raw;
  auto dst=expanded_.Get<C>().template ReinterpretCast<int32_t>();auto src=cached.template ReinterpretCast<int32_t>();
  for(uint32_t i=0;i<rows_;++i)Adds(dst[i*pad_*sizeof(C)/4],src,static_cast<int32_t>(0),pad_*sizeof(C)/4);
  PipeBarrier<PIPE_V>();cachedGroup_=group;
 }
 TPipe pipe_;TQue<TPosition::VECIN,1>bq_;TQue<TPosition::VECOUT,1>yq_;
 TBuf<TPosition::VECCALC>cacheRaw_,cacheC_,expanded_,denseC_,maximum_,mask_,half_;
 GlobalTensor<T>a_,b_;GlobalTensor<uint8_t>y_;GreaterRowTilingData d_{};
 uint32_t width_,pad_,rows_,tile_;bool left_;uint64_t cachedGroup_=UINT64_MAX;
};
#endif


#if S9_HAS_ROW
template<typename T,bool BYPASS_INPUT=false,bool BYPASS_OUTPUT=false>
class KernelGreaterFlatRows {
 using C=typename CompareType<T>::Type;static constexpr bool CONVERT=std::is_same<T,bfloat16_t>::value;
public:
 __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR y,GM_ADDR tiling,const GreaterRowTilingData& d){
  d_=d;width_=static_cast<uint32_t>(d.width);pad_=d.padded;rows_=d.rows;tile_=static_cast<uint32_t>(d.base.tileElements);left_=d.leftRepeated!=0;
  a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));mapsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling+sizeof(GreaterRowTilingData)));
  if constexpr(BYPASS_INPUT){a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);}if constexpr(BYPASS_OUTPUT)y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  pipe_.InitBuffer(bq_,2,tile_*sizeof(T)+32);pipe_.InitBuffer(yq_,2,tile_+32);pipe_.InitBuffer(cacheRaw_,pad_*sizeof(T));pipe_.InitBuffer(expanded_,tile_*sizeof(C));pipe_.InitBuffer(maps_,tile_*sizeof(uint32_t));
  if constexpr(CONVERT){pipe_.InitBuffer(cacheC_,pad_*sizeof(C));pipe_.InitBuffer(denseC_,tile_*sizeof(C));}else pipe_.InitBuffer(maximum_,4096*sizeof(int32_t));
  pipe_.InitBuffer(mask_,512);pipe_.InitBuffer(half_,8192*sizeof(half));
 }
 __aicore__ inline void Process(){
  uint64_t row=static_cast<uint64_t>(GetBlockIdx())*d_.blockRows;if(row>=d_.totalRows)return;const uint64_t end=MinU64(row+d_.blockRows,d_.totalRows);
  auto map=maps_.Get<uint32_t>();DataCopy(map,mapsGm_,tile_);SetFlag<HardEvent::MTE2_V>(EVENT_ID6);WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
  auto ones=half_.Get<half>();Duplicate(ones,static_cast<half>(std::is_same<T,int32_t>::value?0:1),tile_<4096?tile_:4096);PipeBarrier<PIPE_V>();
  uint64_t group=row/d_.repeat,remain=d_.repeat-row%d_.repeat;uint32_t nr=static_cast<uint32_t>(MinU64(MinU64(rows_,remain),end-row));LoadDense(row,nr);
  while(row<end){
   EnsureVector(group);auto raw=bq_.DeQue<T>();uint64_t next=row+nr;uint64_t nextGroup=group,nextRemain=remain-nr;if(nextRemain==0){++nextGroup;nextRemain=d_.repeat;}
   uint32_t nextRows=0;if(next<end){nextRows=static_cast<uint32_t>(MinU64(MinU64(rows_,nextRemain),end-next));LoadDense(next,nextRows);}
   uint32_t count=nr*width_,n=(count+255)/256*256;LocalTensor<C> dense;
   if constexpr(CONVERT){dense=denseC_.Get<C>();Cast(dense,raw,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}else dense=raw;
   auto expanded=expanded_.Get<C>();auto output=yq_.AllocTensor<uint8_t>();
   for(uint32_t j=0;j<n;j+=4096){uint32_t once=n-j<4096?n-j:4096;auto mask=mask_.Get<uint8_t>();
    if constexpr(std::is_same<T,int32_t>::value){auto mx=maximum_.Get<int32_t>();Max(mx,expanded[j],dense[j],once);PipeBarrier<PIPE_V>();if(left_)Compare(mask,dense[j],mx,CMPMODE::EQ,once);else Compare(mask,expanded[j],mx,CMPMODE::EQ,once);}
    else{if(left_)Compare(mask,expanded[j],dense[j],CMPMODE::GT,once);else Compare(mask,dense[j],expanded[j],CMPMODE::GT,once);}
    PipeBarrier<PIPE_V>();auto selected=ones[4096];Select(selected,mask,ones,static_cast<half>(std::is_same<T,int32_t>::value?1:0),SELMODE::VSEL_TENSOR_SCALAR_MODE,once);PipeBarrier<PIPE_V>();Cast(output[j],selected,RoundMode::CAST_NONE,once);PipeBarrier<PIPE_V>();
   }
   yq_.EnQue(output);bq_.FreeTensor(raw);output=yq_.DeQue<uint8_t>();const uint64_t pos=row*d_.width;DataCopyExtParams cp{1,count,0,0,0};if(pos&31)PipeBarrier<PIPE_MTE3>();DataCopyPad(y_[pos],output,cp);yq_.FreeTensor(output);
   row=next;nr=nextRows;group=nextGroup;remain=nextRemain;
  }
 }
private:
 __aicore__ inline void ZeroTail(LocalTensor<T> raw,uint32_t count,uint32_t rounded){
  if(count==rounded)return;auto bits=raw.template ReinterpretCast<int32_t>();uint32_t base=rounded>=256?rounded-256:0;
  Duplicate(bits[base*sizeof(T)/4],static_cast<int32_t>(0),(rounded-base)*sizeof(T)/4);
 }
 __aicore__ inline void LoadDense(uint64_t row,uint32_t nr){
  uint32_t count=nr*width_,rounded=(count+255)/256*256;auto raw=bq_.AllocTensor<T>();const uint64_t off=row*d_.width;
  uint32_t loadCount=count;const bool canReadRounded=off+rounded<=d_.base.totalElements;
  if(canReadRounded)loadCount=rounded;else{ZeroTail(raw,count,rounded);SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);}
  constexpr uint32_t dma=32/sizeof(T);
  if((off*sizeof(T))%32==0&&loadCount%dma==0){if(left_)DataCopy(raw,b_[off],loadCount);else DataCopy(raw,a_[off],loadCount);}
  else{DataCopyExtParams cp{1,static_cast<uint32_t>(loadCount*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>((loadCount+dma-1)/dma*dma-loadCount),static_cast<T>(0)};if(left_)DataCopyPad(raw,b_[off],cp,pp);else DataCopyPad(raw,a_[off],cp,pp);}bq_.EnQue(raw);
 }
 __aicore__ inline void EnsureVector(uint64_t g){
  if(g==cached_)return;auto raw=cacheRaw_.Get<T>();auto words=raw.template ReinterpretCast<int32_t>();Duplicate(words[(pad_-128)*sizeof(T)/4],static_cast<int32_t>(0),128*sizeof(T)/4);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);constexpr uint32_t dma=32/sizeof(T);DataCopyExtParams cp{1,static_cast<uint32_t>(width_*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>((width_+dma-1)/dma*dma-width_),static_cast<T>(0)};
  if(left_)DataCopyPad(raw,a_[g*d_.width],cp,pp);else DataCopyPad(raw,b_[g*d_.width],cp,pp);SetFlag<HardEvent::MTE2_V>(EVENT_ID6);WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
  LocalTensor<C> cached;if constexpr(CONVERT){cached=cacheC_.Get<C>();Cast(cached,raw,RoundMode::CAST_NONE,pad_);PipeBarrier<PIPE_V>();}else cached=raw;
  Gather(expanded_.Get<C>(),cached,maps_.Get<uint32_t>(),0,tile_);PipeBarrier<PIPE_V>();cached_=g;
 }
 TPipe pipe_;TQue<TPosition::VECIN,1>bq_;TQue<TPosition::VECOUT,1>yq_;TBuf<TPosition::VECCALC>cacheRaw_,cacheC_,denseC_,expanded_,maps_,maximum_,mask_,half_;GlobalTensor<T>a_,b_;GlobalTensor<uint8_t>y_;GlobalTensor<uint32_t>mapsGm_;GreaterRowTilingData d_{};uint32_t width_,pad_,rows_,tile_;bool left_;uint64_t cached_=UINT64_MAX;
};
#endif
#if S9_HAS_ROW
template<typename T,bool BYPASS_INPUT=false,bool BYPASS_OUTPUT=false>
class KernelGreaterDeferredWorkspaceRows {
 using C=typename CompareType<T>::Type;static constexpr bool CONVERT=std::is_same<T,bfloat16_t>::value;
public:
 __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR y,GM_ADDR tiling,GM_ADDR workspace,const GreaterRowTilingData& d){
  ws_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(workspace));
  d_=d;width_=static_cast<uint32_t>(d.width);pad_=d.padded;rows_=d.rows;tile_=static_cast<uint32_t>(d.base.tileElements);left_=d.leftRepeated!=0;
  a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));mapsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling+sizeof(GreaterRowTilingData)));
  if constexpr(BYPASS_INPUT){a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);}if constexpr(BYPASS_OUTPUT)y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  pipe_.InitBuffer(bq_,2,tile_*sizeof(T)+32);pipe_.InitBuffer(yq_,2,tile_+32);pipe_.InitBuffer(cacheRaw_,pad_*sizeof(T));pipe_.InitBuffer(expanded_,tile_*sizeof(C));pipe_.InitBuffer(maps_,tile_*sizeof(uint32_t));
  if constexpr(CONVERT){pipe_.InitBuffer(cacheC_,pad_*sizeof(C));pipe_.InitBuffer(denseC_,tile_*sizeof(C));}else pipe_.InitBuffer(maximum_,4096*sizeof(int32_t));
  pipe_.InitBuffer(maskq_,2,tile_/8+32);pipe_.InitBuffer(mask_,512);pipe_.InitBuffer(half_,8192*sizeof(half));
 }
 __aicore__ inline void Process(){
  const uint64_t begin=static_cast<uint64_t>(GetBlockIdx())*d_.blockRows;if(begin>=d_.totalRows)return;const uint64_t end=MinU64(begin+d_.blockRows,d_.totalRows),payload=rows_*width_/8,wsStride=(d_.blockRows*d_.width/8+31)/32*32,wsBase=static_cast<uint64_t>(GetBlockIdx())*wsStride;
  auto map=maps_.Get<uint32_t>();DataCopy(map,mapsGm_,tile_);SetFlag<HardEvent::MTE2_V>(EVENT_ID6);WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
  uint64_t row=begin,group=row/d_.repeat,remain=d_.repeat-row%d_.repeat;uint32_t nr=rows_;LoadDense(row,nr);uint64_t packet=0;
  while(row<end){
   EnsureVector(group);auto raw=bq_.DeQue<T>();uint64_t next=row+nr,nextGroup=group,nextRemain=remain-nr;if(nextRemain==0){++nextGroup;nextRemain=d_.repeat;}if(next<end)LoadDense(next,nr);
   auto expanded=expanded_.Get<int32_t>();auto dense=raw.template ReinterpretCast<int32_t>();auto bits=maskq_.AllocTensor<uint8_t>();
   for(uint32_t j=0;j<tile_;j+=4096){uint32_t count=tile_-j<4096?tile_-j:4096;auto mx=maximum_.Get<int32_t>();Max(mx,expanded[j],dense[j],count);PipeBarrier<PIPE_V>();if(left_)Compare(bits[j/8],dense[j],mx,CMPMODE::EQ,count);else Compare(bits[j/8],expanded[j],mx,CMPMODE::EQ,count);PipeBarrier<PIPE_V>();}
   maskq_.EnQue(bits);bq_.FreeTensor(raw);bits=maskq_.DeQue<uint8_t>();uint64_t bitPos=wsBase+packet*payload;DataCopyExtParams bp{1,static_cast<uint32_t>(payload),0,0,0};if(bitPos&31)PipeBarrier<PIPE_MTE3>();DataCopyPad(ws_[bitPos],bits,bp);maskq_.FreeTensor(bits);
   ++packet;row=next;group=nextGroup;remain=nextRemain;
  }
  auto event=pipe_.AllocEventID<HardEvent::MTE3_MTE2>();SetFlag<HardEvent::MTE3_MTE2>(event);WaitFlag<HardEvent::MTE3_MTE2>(event);pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(event);
  auto zeros=half_.Get<half>();Duplicate(zeros,static_cast<half>(0),4096);PipeBarrier<PIPE_V>();
  const uint64_t outBegin=begin*d_.width,outEnd=end*d_.width;uint64_t pos=outBegin;const uint32_t capacity=tile_*4;uint32_t count=static_cast<uint32_t>(MinU64(capacity,outEnd-pos));LoadCompactBits(wsBase,count/8);
  auto free0=pipe_.AllocEventID<HardEvent::MTE3_V>();auto free1=pipe_.AllocEventID<HardEvent::MTE3_V>();auto ready=pipe_.AllocEventID<HardEvent::V_MTE3>();SetFlag<HardEvent::MTE3_V>(free0);SetFlag<HardEvent::MTE3_V>(free1);uint32_t turn=0;
  while(pos<outEnd){auto raw=bq_.DeQue<T>();auto bits=raw.template ReinterpretCast<uint8_t>();uint64_t next=pos+count;uint32_t nextCount=0;if(next<outEnd){nextCount=static_cast<uint32_t>(MinU64(capacity,outEnd-next));LoadCompactBits(wsBase+(next-outBegin)/8,nextCount/8);}
   auto freeEvent=(turn&1)?free1:free0;WaitFlag<HardEvent::MTE3_V>(freeEvent);auto output=(turn&1)?maps_.Get<uint8_t>():expanded_.Get<uint8_t>();uint32_t n=(count+255)/256*256;
   for(uint32_t j=0;j<n;j+=4096){uint32_t once=n-j<4096?n-j:4096;auto selected=zeros[4096];Select(selected,bits[j/8],zeros,static_cast<half>(1),SELMODE::VSEL_TENSOR_SCALAR_MODE,once);PipeBarrier<PIPE_V>();Cast(output[j],selected,RoundMode::CAST_NONE,once);PipeBarrier<PIPE_V>();}
   bq_.FreeTensor(raw);SetFlag<HardEvent::V_MTE3>(ready);WaitFlag<HardEvent::V_MTE3>(ready);DataCopyExtParams cp{1,count,0,0,0};if(pos&31)PipeBarrier<PIPE_MTE3>();DataCopyPad(y_[pos],output,cp);SetFlag<HardEvent::MTE3_V>(freeEvent);pos=next;count=nextCount;++turn;
  }
  WaitFlag<HardEvent::MTE3_V>(free0);WaitFlag<HardEvent::MTE3_V>(free1);pipe_.ReleaseEventID<HardEvent::MTE3_V>(free0);pipe_.ReleaseEventID<HardEvent::MTE3_V>(free1);pipe_.ReleaseEventID<HardEvent::V_MTE3>(ready);
 }

private:
 __aicore__ inline void LoadCompactBits(uint64_t off,uint32_t count){auto raw=bq_.AllocTensor<T>();DataCopyExtParams cp{1,count,0,0,0};DataCopyPadExtParams<uint8_t> pp{true,0,static_cast<uint8_t>((count+31)/32*32-count),0};DataCopyPad(raw.template ReinterpretCast<uint8_t>(),ws_[off],cp,pp);bq_.EnQue(raw);}
 __aicore__ inline void ZeroTail(LocalTensor<T> raw,uint32_t count,uint32_t rounded){
  if(count==rounded)return;auto bits=raw.template ReinterpretCast<int32_t>();uint32_t base=rounded>=256?rounded-256:0;
  Duplicate(bits[base*sizeof(T)/4],static_cast<int32_t>(0),(rounded-base)*sizeof(T)/4);
 }
 __aicore__ inline void LoadDense(uint64_t row,uint32_t nr){
  uint32_t count=nr*width_,rounded=(count+255)/256*256;auto raw=bq_.AllocTensor<T>();const uint64_t off=row*d_.width;
  uint32_t loadCount=count;const bool canReadRounded=off+rounded<=d_.base.totalElements;
  if(canReadRounded)loadCount=rounded;else{ZeroTail(raw,count,rounded);SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);}
  constexpr uint32_t dma=32/sizeof(T);
  if((off*sizeof(T))%32==0&&loadCount%dma==0){if(left_)DataCopy(raw,b_[off],loadCount);else DataCopy(raw,a_[off],loadCount);}
  else{DataCopyExtParams cp{1,static_cast<uint32_t>(loadCount*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>((loadCount+dma-1)/dma*dma-loadCount),static_cast<T>(0)};if(left_)DataCopyPad(raw,b_[off],cp,pp);else DataCopyPad(raw,a_[off],cp,pp);}bq_.EnQue(raw);
 }
 __aicore__ inline void EnsureVector(uint64_t g){
  if(g==cached_)return;auto raw=cacheRaw_.Get<T>();auto words=raw.template ReinterpretCast<int32_t>();Duplicate(words[(pad_-128)*sizeof(T)/4],static_cast<int32_t>(0),128*sizeof(T)/4);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);constexpr uint32_t dma=32/sizeof(T);DataCopyExtParams cp{1,static_cast<uint32_t>(width_*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>((width_+dma-1)/dma*dma-width_),static_cast<T>(0)};
  if(left_)DataCopyPad(raw,a_[g*d_.width],cp,pp);else DataCopyPad(raw,b_[g*d_.width],cp,pp);SetFlag<HardEvent::MTE2_V>(EVENT_ID6);WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
  LocalTensor<C> cached;if constexpr(CONVERT){cached=cacheC_.Get<C>();Cast(cached,raw,RoundMode::CAST_NONE,pad_);PipeBarrier<PIPE_V>();}else cached=raw;
  Gather(expanded_.Get<C>(),cached,maps_.Get<uint32_t>(),0,tile_);PipeBarrier<PIPE_V>();cached_=g;
 }
 TPipe pipe_;TQue<TPosition::VECIN,1>bq_;TQue<TPosition::VECOUT,1>yq_,maskq_;TBuf<TPosition::VECCALC>cacheRaw_,cacheC_,denseC_,expanded_,maps_,maximum_,mask_,half_;GlobalTensor<T>a_,b_;GlobalTensor<uint8_t>y_,ws_;GlobalTensor<uint32_t>mapsGm_;GreaterRowTilingData d_{};uint32_t width_,pad_,rows_,tile_;bool left_;uint64_t cached_=UINT64_MAX;
};
#endif

#if S9_HAS_DENSE
template<typename T,bool BYPASS_INPUT,bool BYPASS_OUTPUT>
class KernelGreaterPrefetchDense {
 using C=typename CompareType<T>::Type;static constexpr bool CONVERT=std::is_same<T,int8_t>::value;
public:
 __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR y,const GreaterTilingData& d){
  d_=d;tile_=static_cast<uint32_t>(d.tileElements);a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
  if constexpr(BYPASS_INPUT){a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);}if constexpr(BYPASS_OUTPUT)y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  pipe_.InitBuffer(aq_,2,tile_*sizeof(T)+32);pipe_.InitBuffer(bq_,2,tile_*sizeof(T)+32);pipe_.InitBuffer(yq_,2,tile_+32);if constexpr(CONVERT)pipe_.InitBuffer(convert_,2*tile_*sizeof(C));pipe_.InitBuffer(mask_,512);pipe_.InitBuffer(half_,8192*sizeof(half));
 }
 __aicore__ inline void Process(){
  uint64_t pos=static_cast<uint64_t>(GetBlockIdx())*d_.blockElements;if(pos>=d_.totalElements)return;const uint64_t end=MinU64(pos+d_.blockElements,d_.totalElements);
  auto ones=half_.Get<half>();Duplicate(ones,static_cast<half>(1),4096);PipeBarrier<PIPE_V>();uint32_t count=static_cast<uint32_t>(MinU64(tile_,end-pos));Load(pos,count);
  while(pos<end){auto ar=aq_.DeQue<T>();auto br=bq_.DeQue<T>();uint64_t next=pos+count;uint32_t nextCount=0;if(next<end){nextCount=static_cast<uint32_t>(MinU64(tile_,end-next));Load(next,nextCount);}
   const uint32_t n=(count+255)/256*256;LocalTensor<C> a,b;if constexpr(CONVERT){a=convert_.Get<C>();b=a[tile_];Cast(a,ar,RoundMode::CAST_NONE,n);Cast(b,br,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}else{a=ar;b=br;}
   auto out=yq_.AllocTensor<uint8_t>();for(uint32_t j=0;j<n;j+=4096){uint32_t once=n-j<4096?n-j:4096;auto m=mask_.Get<uint8_t>();Compare(m,a[j],b[j],CMPMODE::GT,once);PipeBarrier<PIPE_V>();auto selected=ones[4096];Select(selected,m,ones,static_cast<half>(0),SELMODE::VSEL_TENSOR_SCALAR_MODE,once);PipeBarrier<PIPE_V>();Cast(out[j],selected,RoundMode::CAST_NONE,once);PipeBarrier<PIPE_V>();}
   yq_.EnQue(out);aq_.FreeTensor(ar);bq_.FreeTensor(br);out=yq_.DeQue<uint8_t>();DataCopyExtParams p{1,count,0,0,0};DataCopyPad(y_[pos],out,p);yq_.FreeTensor(out);pos=next;count=nextCount;
  }
 }
private:
 __aicore__ inline void Load(uint64_t off,uint32_t count){
  auto a=aq_.AllocTensor<T>();auto b=bq_.AllocTensor<T>();uint32_t n=(count+255)/256*256;
  if(n!=count){uint32_t start=n-256;Duplicate(a.template ReinterpretCast<int32_t>()[start*sizeof(T)/4],static_cast<int32_t>(0),256*sizeof(T)/4);Duplicate(b.template ReinterpretCast<int32_t>()[start*sizeof(T)/4],static_cast<int32_t>(0),256*sizeof(T)/4);SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);}
  constexpr uint32_t dma=32/sizeof(T);DataCopyExtParams p{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>((count+dma-1)/dma*dma-count),static_cast<T>(0)};DataCopyPad(a,a_[off],p,pp);aq_.EnQue(a);DataCopyPad(b,b_[off],p,pp);bq_.EnQue(b);
 }
 TPipe pipe_;TQue<TPosition::VECIN,1>aq_,bq_;TQue<TPosition::VECOUT,1>yq_;TBuf<TPosition::VECCALC>convert_,mask_,half_;GlobalTensor<T>a_,b_;GlobalTensor<uint8_t>y_;GreaterTilingData d_{};uint32_t tile_;
};
#endif
#if S9_HAS_DENSE
template<typename T,bool BYPASS_INPUT,bool BYPASS_OUTPUT>
class KernelGreaterDeferredDense {
 using C=typename CompareType<T>::Type;static constexpr bool CONVERT=std::is_same<T,int8_t>::value;
public:
 __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR y,const GreaterTilingData& d){
  d_=d;tile_=static_cast<uint32_t>(d.tileElements);a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
  if constexpr(BYPASS_INPUT){a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);}if constexpr(BYPASS_OUTPUT)y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  const uint64_t stride=static_cast<uint64_t>(GetBlockNum())*tile_;const uint64_t capacity=(d.totalElements/stride+(d.totalElements%stride!=0))*tile_;pipe_.InitBuffer(packed_,(capacity+255)/256*32);pipe_.InitBuffer(aq_,2,tile_*sizeof(T)+32);pipe_.InitBuffer(bq_,2,tile_*sizeof(T)+32);pipe_.InitBuffer(yq_,2,tile_+32);if constexpr(CONVERT)pipe_.InitBuffer(convert_,2*tile_*sizeof(C));pipe_.InitBuffer(mask_,512);pipe_.InitBuffer(half_,8192*sizeof(half));
 }
 __aicore__ inline void Process(){
  uint64_t begin=static_cast<uint64_t>(GetBlockIdx())*tile_;if(begin>=d_.totalElements)return;const uint64_t end=d_.totalElements,stride=static_cast<uint64_t>(GetBlockNum())*tile_;uint64_t packet=0;
  uint64_t pos=begin;uint32_t count=static_cast<uint32_t>(MinU64(tile_,end-pos));Load(pos,count);
  auto packed=packed_.Get<uint8_t>();
  // Keep one bit per output in UB until this core has finished reading its input span.
  while(pos<end){auto ar=aq_.DeQue<T>();auto br=bq_.DeQue<T>();uint64_t next=pos+stride;uint32_t nextCount=0;if(next<end){nextCount=static_cast<uint32_t>(MinU64(tile_,end-next));Load(next,nextCount);}
   uint32_t n=(count+255)/256*256;LocalTensor<C> a,b;if constexpr(CONVERT){a=convert_.Get<C>();b=a[tile_];Cast(a,ar,RoundMode::CAST_NONE,n);Cast(b,br,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}else{a=ar;b=br;}Compare(packed[packet*tile_/8],a,b,CMPMODE::GT,n);PipeBarrier<PIPE_V>();aq_.FreeTensor(ar);bq_.FreeTensor(br);pos=next;count=nextCount;++packet;
  }
  auto ones=half_.Get<half>();Duplicate(ones,static_cast<half>(1),4096);PipeBarrier<PIPE_V>();
  pos=begin;packet=0;
  while(pos<end){count=static_cast<uint32_t>(MinU64(tile_,end-pos));uint32_t n=(count+255)/256*256;auto output=yq_.AllocTensor<uint8_t>();
   for(uint32_t j=0;j<n;j+=4096){uint32_t once=n-j<4096?n-j:4096;auto selected=ones[4096];Select(selected,packed[(packet*tile_+j)/8],ones,static_cast<half>(0),SELMODE::VSEL_TENSOR_SCALAR_MODE,once);PipeBarrier<PIPE_V>();Cast(output[j],selected,RoundMode::CAST_NONE,once);PipeBarrier<PIPE_V>();}
   yq_.EnQue(output);output=yq_.DeQue<uint8_t>();DataCopyExtParams cp{1,count,0,0,0};DataCopyPad(y_[pos],output,cp);yq_.FreeTensor(output);pos+=stride;++packet;
  }
 }

private:
 __aicore__ inline void Load(uint64_t off,uint32_t count){
  auto a=aq_.AllocTensor<T>();auto b=bq_.AllocTensor<T>();uint32_t n=(count+255)/256*256;
  if(n!=count){uint32_t start=n-256;Duplicate(a.template ReinterpretCast<int32_t>()[start*sizeof(T)/4],static_cast<int32_t>(0),256*sizeof(T)/4);Duplicate(b.template ReinterpretCast<int32_t>()[start*sizeof(T)/4],static_cast<int32_t>(0),256*sizeof(T)/4);SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);}
  constexpr uint32_t dma=32/sizeof(T);DataCopyExtParams p{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>((count+dma-1)/dma*dma-count),static_cast<T>(0)};DataCopyPad(a,a_[off],p,pp);aq_.EnQue(a);DataCopyPad(b,b_[off],p,pp);bq_.EnQue(b);
 }
 TPipe pipe_;TQue<TPosition::VECIN,1>aq_,bq_;TQue<TPosition::VECOUT,1>yq_;TBuf<TPosition::VECCALC>convert_,mask_,half_,packed_;GlobalTensor<T>a_,b_;GlobalTensor<uint8_t>y_;GreaterTilingData d_{};uint32_t tile_;
};
#endif


#if S9_HAS_TINY
class KernelGreaterCompactTiny {
public:
 __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR y,uint32_t count){count_=count;a_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(a));b_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(b));y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));pipe_.InitBuffer(buf_,40960);}
 __aicore__ inline void Process(){
  uint32_t n=(count_+127)/128*128;auto all=buf_.Get<half>();auto a=all;auto b=all[4096];auto ones=all[8192];auto selected=all[12288];auto bytes=buf_.Get<uint8_t>();auto out=bytes[32768];auto mask=bytes[36864];
  if(n!=count_){Duplicate(a[n-128],static_cast<half>(0),128);Duplicate(b[n-128],static_cast<half>(0),128);SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);}
  DataCopyExtParams cp{1,count_*2,0,0,0};DataCopyPadExtParams<half> pp{true,0,static_cast<uint8_t>((count_+15)/16*16-count_),static_cast<half>(0)};DataCopyPad(a,a_,cp,pp);DataCopyPad(b,b_,cp,pp);
  Duplicate(ones,static_cast<half>(1),n);SetFlag<HardEvent::MTE2_V>(EVENT_ID6);WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);Compare(mask,a,b,CMPMODE::GT,n);PipeBarrier<PIPE_V>();Select(selected,mask,ones,static_cast<half>(0),SELMODE::VSEL_TENSOR_SCALAR_MODE,n);PipeBarrier<PIPE_V>();Cast(out,selected,RoundMode::CAST_NONE,n);SetFlag<HardEvent::V_MTE3>(EVENT_ID6);WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);DataCopyExtParams op{1,count_,0,0,0};DataCopyPad(y_,out,op);PipeBarrier<PIPE_ALL>();
 }
private:TPipe pipe_;TBuf<TPosition::VECCALC>buf_;GlobalTensor<half>a_,b_;GlobalTensor<uint8_t>y_;uint32_t count_;
};
#endif

} // namespace


#if S9_HAS_ROW
template<typename T,bool BYPASS_INPUT=false,bool BYPASS_OUTPUT=false>
class KernelGreaterUbRows {
 using C=int32_t;static constexpr bool CONVERT=false;
public:
 __aicore__ inline void Init(GM_ADDR a,GM_ADDR b,GM_ADDR y,GM_ADDR tiling,const GreaterRowTilingData& d){
  d_=d;width_=static_cast<uint32_t>(d.width);pad_=d.padded;rows_=d.rows;tile_=static_cast<uint32_t>(d.base.tileElements);left_=d.leftRepeated!=0;
  a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a));b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(b));y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));mapsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling+sizeof(GreaterRowTilingData)));
  if constexpr(BYPASS_INPUT){a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);}
  pipe_.InitBuffer(bq_,2,tile_*sizeof(T)+32);pipe_.InitBuffer(cacheRaw_,pad_*sizeof(T));pipe_.InitBuffer(expanded_,tile_*4);
  pipe_.InitBuffer(packed_,(d.blockRows/rows_)*(tile_/8));
 }
 __aicore__ inline void Process(){
  const uint64_t quantum=d_.reserved,units=(d_.totalRows+quantum-1)/quantum,workers=GetBlockNum(),id=GetBlockIdx(),small=units/workers,extra=units%workers;
  const uint64_t begin=(id*small+MinU64(id,extra))*quantum;
  if(begin>=d_.totalRows)return;
  const uint64_t end=MinU64(begin+(small+(id<extra))*quantum,d_.totalRows);
  auto packed=packed_.Get<uint8_t>();uint64_t row=begin,group=row/d_.repeat,remain=d_.repeat-row%d_.repeat,packet=0;LoadDense(row,rows_);
  while(row<end){
   EnsureVector(group);auto raw=bq_.DeQue<T>();const uint64_t next=row+rows_;uint64_t nextGroup=group,nextRemain=remain-rows_;if(nextRemain==0){++nextGroup;nextRemain=d_.repeat;}if(next<end)LoadDense(next,rows_);
   auto dense=raw.template ReinterpretCast<int32_t>();auto expanded=expanded_.Get<int32_t>();
   for(uint32_t j=0;j<tile_;j+=4096){const uint32_t n=tile_-j<4096?tile_-j:4096;if(left_)Min(dense[j],expanded[j],dense[j],n);else Max(dense[j],expanded[j],dense[j],n);PipeBarrier<PIPE_V>();Compare(packed[packet*(tile_/8)+j/8],expanded[j],dense[j],CMPMODE::EQ,n);PipeBarrier<PIPE_V>();}
   bq_.FreeTensor(raw);++packet;row=next;group=nextGroup;remain=nextRemain;
  }
  // Decode four padded bit packets, then let the multi-block DMA remove
  // each packet's padding. All full output transfers start on a 32-byte boundary.
  const uint64_t packets=packet;const uint32_t valid=rows_*width_;
  auto output0=bq_.AllocTensor<T>();auto output1=bq_.AllocTensor<T>();auto zeros=expanded_.Get<half>();Duplicate(zeros,static_cast<half>(0),4096);PipeBarrier<PIPE_V>();
  auto free0=pipe_.AllocEventID<HardEvent::MTE3_V>();auto free1=pipe_.AllocEventID<HardEvent::MTE3_V>();auto ready=pipe_.AllocEventID<HardEvent::V_MTE3>();SetFlag<HardEvent::MTE3_V>(free0);SetFlag<HardEvent::MTE3_V>(free1);
  uint32_t turn=0;
  for(packet=0;packet<packets;packet+=4){
   const uint32_t batch=static_cast<uint32_t>(MinU64(4,packets-packet));auto freeEvent=(turn&1)?free1:free0;WaitFlag<HardEvent::MTE3_V>(freeEvent);auto output=(turn&1)?output1.template ReinterpretCast<uint8_t>():output0.template ReinterpretCast<uint8_t>();
   for(uint32_t p=0;p<batch;++p)for(uint32_t j=0;j<tile_;j+=4096){const uint32_t n=tile_-j<4096?tile_-j:4096;auto selected=zeros[4096];Select(selected,packed[(packet+p)*(tile_/8)+j/8],zeros,static_cast<half>(1),SELMODE::VSEL_TENSOR_SCALAR_MODE,n);PipeBarrier<PIPE_V>();Cast(output[p*tile_+j],selected,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();}
   SetFlag<HardEvent::V_MTE3>(ready);WaitFlag<HardEvent::V_MTE3>(ready);
   DataCopyExtParams cp{static_cast<uint16_t>(batch),valid,static_cast<uint32_t>((tile_-(valid+31)/32*32)/32),0,0};DataCopyPad(y_[begin*d_.width+packet*valid],output,cp);SetFlag<HardEvent::MTE3_V>(freeEvent);++turn;
  }
  WaitFlag<HardEvent::MTE3_V>(free0);WaitFlag<HardEvent::MTE3_V>(free1);pipe_.ReleaseEventID<HardEvent::MTE3_V>(free0);pipe_.ReleaseEventID<HardEvent::MTE3_V>(free1);pipe_.ReleaseEventID<HardEvent::V_MTE3>(ready);bq_.FreeTensor(output0);bq_.FreeTensor(output1);
 }
private:
 __aicore__ inline void ZeroTail(LocalTensor<T> raw,uint32_t count,uint32_t rounded){
  if(count==rounded)return;auto bits=raw.template ReinterpretCast<int32_t>();uint32_t base=rounded>=256?rounded-256:0;
  Duplicate(bits[base*sizeof(T)/4],static_cast<int32_t>(0),(rounded-base)*sizeof(T)/4);
 }
 __aicore__ inline void LoadDense(uint64_t row,uint32_t nr){
  uint32_t count=nr*width_,rounded=(count+255)/256*256;auto raw=bq_.AllocTensor<T>();const uint64_t off=row*d_.width;
  uint32_t loadCount=count;const bool canReadRounded=off+rounded<=d_.base.totalElements;
  if(canReadRounded)loadCount=rounded;else{ZeroTail(raw,count,rounded);SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);}
  constexpr uint32_t dma=32/sizeof(T);
  if((off*sizeof(T))%32==0&&loadCount%dma==0){if(left_)DataCopy(raw,b_[off],loadCount);else DataCopy(raw,a_[off],loadCount);}
  else{DataCopyExtParams cp{1,static_cast<uint32_t>(loadCount*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>((loadCount+dma-1)/dma*dma-loadCount),static_cast<T>(0)};if(left_)DataCopyPad(raw,b_[off],cp,pp);else DataCopyPad(raw,a_[off],cp,pp);}bq_.EnQue(raw);
 }
 __aicore__ inline void EnsureVector(uint64_t g){
  if(g==cached_)return;auto indexStorage=bq_.AllocTensor<T>();auto map=indexStorage.template ReinterpretCast<uint32_t>();DataCopy(map,mapsGm_,tile_);auto raw=cacheRaw_.Get<T>();auto words=raw.template ReinterpretCast<int32_t>();Duplicate(words[(pad_-128)*sizeof(T)/4],static_cast<int32_t>(0),128*sizeof(T)/4);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID6);WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);constexpr uint32_t dma=32/sizeof(T);DataCopyExtParams cp{1,static_cast<uint32_t>(width_*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pp{true,0,static_cast<uint8_t>((width_+dma-1)/dma*dma-width_),static_cast<T>(0)};
  if(left_)DataCopyPad(raw,a_[g*d_.width],cp,pp);else DataCopyPad(raw,b_[g*d_.width],cp,pp);SetFlag<HardEvent::MTE2_V>(EVENT_ID6);WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
  LocalTensor<C> cached;if constexpr(CONVERT){cached=cacheC_.Get<C>();Cast(cached,raw,RoundMode::CAST_NONE,pad_);PipeBarrier<PIPE_V>();}else cached=raw;
  Gather(expanded_.Get<C>(),cached,map,0,tile_);PipeBarrier<PIPE_V>();bq_.FreeTensor(indexStorage);cached_=g;
 }
 TPipe pipe_;TQue<TPosition::VECIN,1>bq_;TBuf<TPosition::VECCALC>cacheRaw_,cacheC_,expanded_,maps_,packed_;GlobalTensor<T>a_,b_;GlobalTensor<uint8_t>y_;GlobalTensor<uint32_t>mapsGm_;GreaterRowTilingData d_{};uint32_t width_,pad_,rows_,tile_;bool left_;uint64_t cached_=UINT64_MAX;
};
#endif

extern "C" __global__ __aicore__ void greater(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(GreaterTilingData);
#if S9_HAS_DENSE
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 277", GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 284", GreaterTilingData);
#endif
#if S9_HAS_TINY
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 26", GreaterTinyTilingData);
#endif
#if S9_HAS_ROW
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 30", GreaterFlatRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 31", GreaterFlatRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 287", GreaterFlatRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 281", GreaterFlatRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 286", GreaterFlatRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 22", GreaterFlatRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 23", GreaterFlatRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 24", GreaterFlatRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 25", GreaterFlatRowTilingData);
#endif
#if S9_HAS_ROW
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 15", GreaterRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 16", GreaterRowTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 17", GreaterRowTilingData);
#endif
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", GreaterGatherTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 6", GreaterGatherMeta);
    // Keep the legacy entry at the original code address. Its large hot
    // control-flow graph can be sensitive to instruction-cache placement.
#if S9_HAS_TINY
    if(TILING_KEY_IS(26)){GET_TILING_DATA_WITH_STRUCT(GreaterTinyTilingData,d,tiling);KernelGreaterCompactTiny op;op.Init(x1,x2,y,static_cast<uint32_t>(d.total));op.Process();return;}
#endif

#if S9_HAS_ROW
    if(TILING_KEY_IS(31)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);KernelGreaterUbRows<int32_t,true,false> op;op.Init(x1,x2,y,tiling,d);op.Process();return;}
    if(TILING_KEY_IS(287)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);const uint64_t offset=d.base.inputL2BypassOffset;KernelGreaterUbRows<int32_t,false,false> op;op.Init(x1+offset,x2+offset,y,tiling,d);op.Process();return;}
    if(TILING_KEY_IS(281)){
      GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);
      const uint64_t offset=d.base.inputL2BypassOffset;
      KernelGreaterFlatRows<DTYPE_X1,false,false> op;op.Init(x1+offset,x2+offset,y,tiling,d);op.Process();return;
    }
    if(TILING_KEY_IS(286)){
      GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);
      const uint64_t offset=d.base.inputL2BypassOffset;
      KernelGreaterDeferredWorkspaceRows<int32_t,false,false> op;op.Init(x1+offset,x2+offset,y,tiling,GetUserWorkspace(workspace),d);op.Process();return;
    }
#endif
#if S9_HAS_DENSE
    if(TILING_KEY_IS(277)){
      GET_TILING_DATA_WITH_STRUCT(GreaterTilingData,d,tiling);const uint64_t offset=d.inputL2BypassOffset;
      KernelGreaterPrefetchDense<DTYPE_X1,false,false> op;op.Init(x1+offset,x2+offset,y,d);op.Process();return;
    }
    if(TILING_KEY_IS(284)){
      GET_TILING_DATA_WITH_STRUCT(GreaterTilingData,d,tiling);const uint64_t offset=d.inputL2BypassOffset;
      KernelGreaterDeferredDense<DTYPE_X1,false,false> op;op.Init(x1+offset,x2+offset,y,d);op.Process();return;
    }
#endif

    if (TILING_KEY_IS(0)) {
        GET_TILING_DATA(tilingData, tiling);
        KernelGreater<DTYPE_X1> kernel;
        kernel.Init(x1, x2, y, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterGatherMeta, data, tiling);
        KernelGreaterGather<DTYPE_X1> kernel;
        kernel.Init(x1, x2, y, tiling, data);
        kernel.Process();
    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA(tilingData, tiling);
        KernelGreaterLinear<DTYPE_X1, 1> kernel;
        kernel.Init(x1, x2, y, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA(tilingData, tiling);
        KernelGreaterRuns<DTYPE_X1> kernel;
        kernel.Init(x1, x2, y, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(4)) {
        GET_TILING_DATA(tilingData, tiling);
        KernelGreaterTiny<DTYPE_X1> kernel;
        kernel.Init(x1, x2, y, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(5)) {
        GET_TILING_DATA(tilingData, tiling);
        KernelGreaterLinear<DTYPE_X1, 2> kernel;
        kernel.Init(x1, x2, y, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(7)) {
        GET_TILING_DATA(data, tiling); KernelGreaterScalarRows<DTYPE_X1,false> kernel; kernel.Init(x1,x2,y,data); kernel.Process();
    } else if (TILING_KEY_IS(8)) {
        GET_TILING_DATA(data, tiling); KernelGreaterStream<DTYPE_X1,true,false> kernel; kernel.Init(x1,x2,y,data); kernel.Process();
    } else if (TILING_KEY_IS(9)) {
        GET_TILING_DATA(data, tiling); KernelGreaterStream<DTYPE_X1,true,true> kernel; kernel.Init(x1,x2,y,data); kernel.Process();
    } else if (TILING_KEY_IS(10)) {
        GET_TILING_DATA(data, tiling); KernelGreaterStream<DTYPE_X1,false,false> kernel; kernel.Init(x1,x2,y,data); kernel.Process();
    } else if (TILING_KEY_IS(11)) {
        GET_TILING_DATA(data, tiling); KernelGreaterStream<DTYPE_X1,false,true> kernel; kernel.Init(x1,x2,y,data); kernel.Process();
    } else if (TILING_KEY_IS(12)) {
        GET_TILING_DATA(data, tiling); KernelGreaterPeriodic<DTYPE_X1,false> kernel; kernel.Init(x1,x2,y,data); kernel.Process();
    } else if (TILING_KEY_IS(13)) {
        GET_TILING_DATA(data, tiling); KernelGreaterPeriodic<DTYPE_X1,true> kernel; kernel.Init(x1,x2,y,data); kernel.Process();
    } else if (TILING_KEY_IS(14)) {
        GET_TILING_DATA(data, tiling); KernelGreaterScalarRows<DTYPE_X1,true> kernel; kernel.Init(x1,x2,y,data); kernel.Process();
    } else if (TILING_KEY_IS(6)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterGatherMeta, data, tiling);
        KernelGreaterGather<DTYPE_X1, true> kernel;
        kernel.Init(x1, x2, y, tiling, data);
        kernel.Process();
    }
#if S9_HAS_ROW
    else if(TILING_KEY_IS(15)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);KernelGreaterRowPad<DTYPE_X1,false,false> op;op.Init(x1,x2,y,d);op.Process();}
    else if(TILING_KEY_IS(16)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);KernelGreaterRowPad<DTYPE_X1,false,true> op;op.Init(x1,x2,y,d);op.Process();}
    else if(TILING_KEY_IS(17)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);KernelGreaterRowPad<DTYPE_X1,true,true> op;op.Init(x1,x2,y,d);op.Process();}
#endif

#if S9_HAS_ROW
    else if(TILING_KEY_IS(30)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);KernelGreaterDeferredWorkspaceRows<int32_t,true,false> op;op.Init(x1,x2,y,tiling,GetUserWorkspace(workspace),d);op.Process();}
    else if(TILING_KEY_IS(22)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);KernelGreaterFlatRows<DTYPE_X1,false,false> op;op.Init(x1,x2,y,tiling,d);op.Process();}
    else if(TILING_KEY_IS(23)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);KernelGreaterFlatRows<DTYPE_X1,false,true> op;op.Init(x1,x2,y,tiling,d);op.Process();}
    else if(TILING_KEY_IS(24)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);KernelGreaterFlatRows<DTYPE_X1,true,true> op;op.Init(x1,x2,y,tiling,d);op.Process();}
    else if(TILING_KEY_IS(25)){GET_TILING_DATA_WITH_STRUCT(GreaterRowTilingData,d,tiling);KernelGreaterFlatRows<DTYPE_X1,true,false> op;op.Init(x1,x2,y,tiling,d);op.Process();}
#endif
#if S9_HAS_DENSE
    else if(TILING_KEY_IS(28)){GET_TILING_DATA(d,tiling);KernelGreaterDeferredDense<DTYPE_X1,true,false> op;op.Init(x1,x2,y,d);op.Process();}
    else if(TILING_KEY_IS(18)){GET_TILING_DATA(d,tiling);KernelGreaterPrefetchDense<DTYPE_X1,false,false> op;op.Init(x1,x2,y,d);op.Process();}
    else if(TILING_KEY_IS(19)){GET_TILING_DATA(d,tiling);KernelGreaterPrefetchDense<DTYPE_X1,false,true> op;op.Init(x1,x2,y,d);op.Process();}
    else if(TILING_KEY_IS(20)){GET_TILING_DATA(d,tiling);KernelGreaterPrefetchDense<DTYPE_X1,true,true> op;op.Init(x1,x2,y,d);op.Process();}
    else if(TILING_KEY_IS(21)){GET_TILING_DATA(d,tiling);KernelGreaterPrefetchDense<DTYPE_X1,true,false> op;op.Init(x1,x2,y,d);op.Process();}
#endif

}
