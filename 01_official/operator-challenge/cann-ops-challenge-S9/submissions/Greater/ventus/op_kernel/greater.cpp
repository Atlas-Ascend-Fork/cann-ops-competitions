/**
 * Greater 设备侧实现。
 * 按广播模式执行逐元素比较并生成布尔结果；输入转换和输出存储使用配套的缓冲区生命周期。
 */

/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#define K_MAX_SHAPE_DIM 0
#include "kernel_operator.h"
#include "../op_host/greater_tiling.h"

#if defined(ASCENDC_CPU_DEBUG) && !defined(GET_TILING_DATA_WITH_STRUCT)
#define GET_TILING_DATA_WITH_STRUCT(type, variable, pointer) \
    type variable = *reinterpret_cast<const __gm__ type*>(pointer)
#endif

using namespace AscendC;

namespace {
constexpr uint32_t kInputTileBytes = 8U * 1024U;
constexpr uint32_t kBroadcastScratchBytes = 64U * 1024U;
constexpr uint32_t kBufferNum = 1U;
constexpr uint32_t kDenseBufferNum = 2U;

template <typename T>
struct DenseTilePolicy {
    static constexpr uint32_t kElements = 7168U;
};

template <>
struct DenseTilePolicy<half> {
    static constexpr uint32_t kElements = 11264U;
};

template <>
struct DenseTilePolicy<int8_t> {
    static constexpr uint32_t kElements = 11264U;
};

template <typename T>
struct BoxTilePolicy {
    static constexpr uint32_t kElements =
        S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32;
};

template <>
struct BoxTilePolicy<half> {
    static constexpr uint32_t kElements = S9_GREATER_BOX_ELEMENTS_FLOAT16;
};

template <>
struct BoxTilePolicy<bfloat16_t> {
    static constexpr uint32_t kElements = S9_GREATER_BOX_ELEMENTS_BFLOAT16;
};

template <>
struct BoxTilePolicy<int8_t> {
    static constexpr uint32_t kElements = S9_GREATER_BOX_ELEMENTS_INT8;
};

template <typename T>
struct IsBfloat16 {
    static constexpr bool kValue = false;
};

template <>
struct IsBfloat16<bfloat16_t> {
    static constexpr bool kValue = true;
};

template <typename T>
struct IsInt8 {
    static constexpr bool kValue = false;
};

template <>
struct IsInt8<int8_t> {
    static constexpr bool kValue = true;
};

template <typename T>
struct SupportsHighLevelBroadcast {
    static constexpr bool kValue = false;
};

template <>
struct SupportsHighLevelBroadcast<half> {
    static constexpr bool kValue = true;
};

template <>
struct SupportsHighLevelBroadcast<float> {
    static constexpr bool kValue = true;
};

template <>
struct SupportsHighLevelBroadcast<int8_t> {
    static constexpr bool kValue = true;
};

template <typename T>
struct BroadcastRange {
    __aicore__ static inline void Run(LocalTensor<T> tensor, uint32_t start,
                                      uint32_t count, T scalar)
    {
        constexpr uint32_t kBlockElements = 32U / sizeof(T);
        const uint32_t misalignment = start % kBlockElements;
        const uint32_t head = misalignment == 0U ? 0U :
            (count < kBlockElements - misalignment ?
                count : kBlockElements - misalignment);
        for (uint32_t index = 0U; index < head; ++index) {
            tensor.SetValue(start + index, scalar);
        }
        start += head;
        count -= head;
        if (count == 0U) {
            return;
        }
        Duplicate(tensor[start], scalar, count);
    }
};

template <>
struct BroadcastRange<int8_t> {
    __aicore__ static inline void Run(LocalTensor<int8_t> tensor, uint32_t start,
                                      uint32_t count, int8_t scalar)
    {
        const uint16_t byte = static_cast<uint8_t>(scalar);
        const uint16_t packed = byte | static_cast<uint16_t>(byte << 8);
        constexpr uint32_t kBlockElements = 32U / sizeof(int8_t);
        const uint32_t misalignment = start % kBlockElements;
        const uint32_t head = misalignment == 0U ? 0U :
            (count < kBlockElements - misalignment ?
                count : kBlockElements - misalignment);
        for (uint32_t index = 0U; index < head; ++index) {
            tensor.SetValue(start + index, scalar);
        }
        start += head;
        count -= head;
        const uint32_t paired = count & ~1U;
        if (paired != 0U) {
            LocalTensor<uint16_t> words = tensor[start].ReinterpretCast<uint16_t>();
            Duplicate(words, packed, paired / 2U);
        }
        if ((count & 1U) != 0U) {
            tensor.SetValue(start + paired, scalar);
        }
    }
};

template <typename T>
struct CompareRunner {
    __aicore__ static inline void Run(LocalTensor<uint8_t> mask,
                                      LocalTensor<T> x1, LocalTensor<T> x2,
                                      LocalTensor<float>, LocalTensor<float>,
                                      uint32_t count)
    {
        Compare(mask, x1, x2, CMPMODE::GT, count);
    }
};

template <typename T>
struct OutputSelector {
    __aicore__ static inline void Run(LocalTensor<half> output,
                                      LocalTensor<uint8_t> mask,
                                      LocalTensor<half> tensorValues,
                                      uint32_t count)
    {
        Select(output, mask, tensorValues, static_cast<half>(0),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    }

    __aicore__ static inline void InitTensor(LocalTensor<half> tensorValues,
                                             uint32_t count)
    {
        Duplicate(tensorValues, static_cast<half>(1), count);
    }
};

template <>
struct OutputSelector<int32_t> {
    __aicore__ static inline void Run(LocalTensor<half> output,
                                      LocalTensor<uint8_t> lessOrEqualMask,
                                      LocalTensor<half> tensorValues,
                                      uint32_t count)
    {
        Select(output, lessOrEqualMask, tensorValues, static_cast<half>(1),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    }

    __aicore__ static inline void InitTensor(LocalTensor<half> tensorValues,
                                             uint32_t count)
    {
        Duplicate(tensorValues, static_cast<half>(0), count);
    }
};

template <typename T>
struct ConversionStorage {
    static constexpr uint32_t kBytesPerElement = 0U;
};

template <>
struct ConversionStorage<bfloat16_t> {
    static constexpr uint32_t kBytesPerElement = sizeof(float);
};

template <>
struct ConversionStorage<int8_t> {
    static constexpr uint32_t kBytesPerElement = sizeof(half);
};

template <typename T>
struct DenseUbBudget {
    static constexpr uint32_t kElements = DenseTilePolicy<T>::kElements;
    static constexpr uint32_t kConvertedBytes =
        ConversionStorage<T>::kBytesPerElement == 0U ?
            32U : kElements * ConversionStorage<T>::kBytesPerElement;
    static constexpr uint32_t kBytes =
        4U * kElements * sizeof(T) +
        2U * kElements * sizeof(uint8_t) + kElements / 8U +
        2U * kElements * sizeof(half) + 2U * kConvertedBytes;
};

static_assert(DenseTilePolicy<half>::kElements % 256U == 0U);
static_assert(DenseTilePolicy<float>::kElements % 256U == 0U);
static_assert(DenseTilePolicy<bfloat16_t>::kElements % 256U == 0U);
static_assert(DenseTilePolicy<int32_t>::kElements % 256U == 0U);
static_assert(DenseTilePolicy<int8_t>::kElements % 256U == 0U);
static_assert(DenseUbBudget<half>::kBytes <= 192U * 1024U);
static_assert(DenseUbBudget<float>::kBytes <= 192U * 1024U);
static_assert(DenseUbBudget<bfloat16_t>::kBytes <= 192U * 1024U);
static_assert(DenseUbBudget<int32_t>::kBytes <= 192U * 1024U);
static_assert(DenseUbBudget<int8_t>::kBytes <= 192U * 1024U);

template <typename T>
struct BoxUbBudget {
    static constexpr uint32_t kElements = BoxTilePolicy<T>::kElements;
    static constexpr uint32_t kConvertedBytes =
        ConversionStorage<T>::kBytesPerElement == 0U ?
            32U : kElements * ConversionStorage<T>::kBytesPerElement;
    static constexpr uint32_t kBytes =
        4U * kElements * sizeof(T) +
        kElements * sizeof(uint8_t) + kElements / 8U +
        2U * kElements * sizeof(half) + 2U * kConvertedBytes +
        kBroadcastScratchBytes;
};

static_assert(BoxTilePolicy<half>::kElements % 256U == 0U);
static_assert(BoxTilePolicy<float>::kElements % 256U == 0U);
static_assert(BoxTilePolicy<bfloat16_t>::kElements % 256U == 0U);
static_assert(BoxTilePolicy<int32_t>::kElements % 256U == 0U);
static_assert(BoxTilePolicy<int8_t>::kElements % 256U == 0U);
static_assert(BoxUbBudget<half>::kBytes <= 192U * 1024U);
static_assert(BoxUbBudget<float>::kBytes <= 192U * 1024U);
static_assert(BoxUbBudget<bfloat16_t>::kBytes <= 192U * 1024U);
static_assert(BoxUbBudget<int32_t>::kBytes <= 192U * 1024U);
static_assert(BoxUbBudget<int8_t>::kBytes <= 192U * 1024U);

template <typename T>
struct RowStripePolicy {
    using ComputeType = T;
    static constexpr uint32_t kChunkElements = 256U / sizeof(T);
};

template <>
struct RowStripePolicy<bfloat16_t> {
    using ComputeType = float;

    static constexpr uint32_t kChunkElements = 64U;
};

template <>
struct RowStripePolicy<int8_t> {
    using ComputeType = half;

    static constexpr uint32_t kChunkElements = 128U;
};

template <typename T>
struct RowStripeMaximumStorage {
    static constexpr uint32_t kBytes = 32U;
};

template <typename T>
struct IsInt32 {
    static constexpr bool kValue = false;
};

template <>
struct IsInt32<int32_t> {
    static constexpr bool kValue = true;
};

template <>
struct RowStripeMaximumStorage<int32_t> {
    static constexpr uint32_t kBytes =
        S9_GREATER_ROW_STRIPE_MAX_ROWS *
        RowStripePolicy<int32_t>::kChunkElements * sizeof(int32_t);
};

template <typename T>
struct RowStripeUbBudget {
    static constexpr uint32_t kChunkElements =
        RowStripePolicy<T>::kChunkElements;
    static constexpr uint32_t kStripeElements =
        S9_GREATER_ROW_STRIPE_MAX_ROWS * kChunkElements;
    static constexpr uint32_t kConvertedBytesPerElement =
        ConversionStorage<T>::kBytesPerElement;
    static constexpr uint32_t kDenseConvertedBytes =
        kConvertedBytesPerElement == 0U ?
            32U : kStripeElements * kConvertedBytesPerElement;
    static constexpr uint32_t kRowConvertedBytes =
        kConvertedBytesPerElement == 0U ?
            32U : kChunkElements * kConvertedBytesPerElement;
    static constexpr uint32_t kBytes =
        BoxTilePolicy<T>::kElements * sizeof(T) +
        kStripeElements * sizeof(T) +
        kStripeElements * sizeof(uint8_t) +
        kStripeElements / 8U +
        2U * kStripeElements * sizeof(half) +
        kDenseConvertedBytes + kRowConvertedBytes +
        RowStripeMaximumStorage<T>::kBytes;
};

static_assert(RowStripeUbBudget<half>::kBytes <= 192U * 1024U);
static_assert(RowStripeUbBudget<float>::kBytes <= 192U * 1024U);
static_assert(RowStripeUbBudget<bfloat16_t>::kBytes <= 192U * 1024U);
static_assert(RowStripeUbBudget<int32_t>::kBytes <= 192U * 1024U);
static_assert(RowStripeUbBudget<int8_t>::kBytes <= 192U * 1024U);

template <>
struct CompareRunner<bfloat16_t> {
    __aicore__ static inline void Run(LocalTensor<uint8_t> mask,
                                      LocalTensor<bfloat16_t> x1,
                                      LocalTensor<bfloat16_t> x2,
                                      LocalTensor<float> convertedX1,
                                      LocalTensor<float> convertedX2,
                                      uint32_t count)
    {
        Cast(convertedX1, x1, RoundMode::CAST_NONE, count);
        Cast(convertedX2, x2, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Compare(mask, convertedX1, convertedX2, CMPMODE::GT, count);
    }
};

template <>
struct CompareRunner<int32_t> {
    __aicore__ static inline void Run(LocalTensor<uint8_t> mask,
                                      LocalTensor<int32_t> x1,
                                      LocalTensor<int32_t> x2,
                                      LocalTensor<float> convertedX1,
                                      LocalTensor<float> convertedX2,
                                      uint32_t count)
    {

        Max(x1, x1, x2, count);
        PipeBarrier<PIPE_V>();
        Compare(mask, x1, x2, CMPMODE::EQ, count);
    }
};

template <>
struct CompareRunner<int8_t> {
    __aicore__ static inline void Run(LocalTensor<uint8_t> mask,
                                      LocalTensor<int8_t> x1,
                                      LocalTensor<int8_t> x2,
                                      LocalTensor<float> convertedX1,
                                      LocalTensor<float> convertedX2,
                                      uint32_t count)
    {
        LocalTensor<half> halfX1 = convertedX1.ReinterpretCast<half>();
        LocalTensor<half> halfX2 = convertedX2.ReinterpretCast<half>();
        Cast(halfX1, x1, RoundMode::CAST_NONE, count);
        Cast(halfX2, x2, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Compare(mask, halfX1, halfX2, CMPMODE::GT, count);
    }
};

template <typename T, typename ComputeT, bool RowIsX1>
__aicore__ inline void CompareWithCachedConvertedRow(
    LocalTensor<uint8_t> mask, LocalTensor<T> row, LocalTensor<T> dense,
    LocalTensor<float> convertedX1, LocalTensor<float> convertedX2,
    uint32_t count, uint32_t& convertedRowValidCount)
{
    LocalTensor<ComputeT> convertedRow = (RowIsX1 ? convertedX1 : convertedX2)
        .template ReinterpretCast<ComputeT>();
    LocalTensor<ComputeT> convertedDense = (RowIsX1 ? convertedX2 : convertedX1)
        .template ReinterpretCast<ComputeT>();

    if constexpr (!RowIsX1) {
        Cast(convertedDense, dense, RoundMode::CAST_NONE, count);
    }
    if (convertedRowValidCount < count) {
        Cast(convertedRow, row, RoundMode::CAST_NONE, count);
        convertedRowValidCount = count;
    }
    if constexpr (RowIsX1) {
        Cast(convertedDense, dense, RoundMode::CAST_NONE, count);
    }
    PipeBarrier<PIPE_V>();
    if constexpr (RowIsX1) {
        Compare(mask, convertedRow, convertedDense, CMPMODE::GT, count);
    } else {
        Compare(mask, convertedDense, convertedRow, CMPMODE::GT, count);
    }
}

template <typename T>
// 处理通用广播比较，并区分标量、连续片段与非对齐边界。
class KernelGreater {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;
        tileElements_ = kInputTileBytes / sizeof(T);

        pipe_.InitBuffer(x1Queue_, kBufferNum, kInputTileBytes);
        pipe_.InitBuffer(x2Queue_, kBufferNum, kInputTileBytes);
        pipe_.InitBuffer(yQueue_, kBufferNum, tileElements_ * sizeof(uint8_t));
        pipe_.InitBuffer(maskBuffer_, tileElements_ / 8U);
        pipe_.InitBuffer(selectionValueBuffer_, tileElements_ * sizeof(half));
        pipe_.InitBuffer(selectedBuffer_, tileElements_ * sizeof(half));
        constexpr uint32_t kMinimumBufferBytes = 32U;
        const uint32_t convertedBytes = ConversionStorage<T>::kBytesPerElement == 0U ?
            kMinimumBufferBytes : tileElements_ * ConversionStorage<T>::kBytesPerElement;
        pipe_.InitBuffer(convertedX1Buffer_, convertedBytes);
        pipe_.InitBuffer(convertedX2Buffer_, convertedBytes);
        pipe_.InitBuffer(scalarQueue_, kBufferNum, kInputTileBytes);
        pipe_.InitBuffer(broadcastScratchBuffer_, kBroadcastScratchBytes);
    }

    __aicore__ inline void Process()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        uint64_t start = 0;
        uint64_t end = 0;
        uint32_t maximumChunk = tileElements_;
        if (tiling_->rowElements <= tileElements_) {
            const uint64_t totalRows = tiling_->totalElements / tiling_->rowElements;
            const uint64_t rowsPerBlock = totalRows / blocks +
                (totalRows % blocks != 0U ? 1U : 0U);
            const uint64_t startRow = rowsPerBlock * block;
            const uint64_t endRow = startRow + rowsPerBlock < totalRows ?
                startRow + rowsPerBlock : totalRows;
            start = startRow * tiling_->rowElements;
            end = endRow * tiling_->rowElements;
            const uint32_t rowsPerTile = static_cast<uint32_t>(
                tileElements_ / tiling_->rowElements);
            maximumChunk = rowsPerTile * static_cast<uint32_t>(tiling_->rowElements);
        } else {
            constexpr uint64_t kBlockAlignment = 256U;
            const uint64_t average = tiling_->totalElements / blocks +
                (tiling_->totalElements % blocks != 0U ? 1U : 0U);
            const uint64_t blockElements =
                (average + kBlockAlignment - 1U) & ~(kBlockAlignment - 1U);
            start = blockElements * block;
            const uint64_t blockEnd = start + blockElements;
            end = blockEnd < tiling_->totalElements ?
                blockEnd : tiling_->totalElements;
        }
        if (start >= tiling_->totalElements) {
            return;
        }

        LocalTensor<half> selectionValues = selectionValueBuffer_.Get<half>();
        OutputSelector<T>::InitTensor(selectionValues, tileElements_);
        PipeBarrier<PIPE_V>();

        uint64_t current = start;
        while (current < end) {
            uint64_t chunk64 = end - current;
            if (chunk64 > maximumChunk) {
                chunk64 = maximumChunk;
            }
            const uint32_t chunk = static_cast<uint32_t>(chunk64);
            CopyIn(current, chunk);
            Compute(current, chunk);
            CopyOut(current, chunk);
            current += chunk;
        }
    }

private:
    __aicore__ inline void InputRowOffsets(uint64_t row, uint64_t& x1Offset,
                                           uint64_t& x2Offset) const
    {
        x1Offset = 0;
        x2Offset = 0;
        for (uint32_t reverse = tiling_->outerRank; reverse > 0; --reverse) {
            const uint32_t axis = reverse - 1;
            const uint64_t coordinate = row % tiling_->outputDims[axis];
            row /= tiling_->outputDims[axis];
            if (tiling_->x1Dims[axis] != 1) {
                x1Offset += coordinate * tiling_->x1Strides[axis];
            }
            if (tiling_->x2Dims[axis] != 1) {
                x2Offset += coordinate * tiling_->x2Strides[axis];
            }
        }
    }

    __aicore__ inline void FlushCopyRun(LocalTensor<T> local,
                                         const GlobalTensor<T>& global,
                                         uint64_t sourceOffset,
                                         uint32_t localOffset,
                                         uint32_t count,
                                         const DataCopyPadExtParams<T>& pad,
                                         bool scalarPass,
                                         bool& needsScalarHead)
    {
        if (count == 0U) {
            return;
        }

        constexpr uint32_t kBlockElements = 32U / sizeof(T);
        const uint32_t misalignment = localOffset % kBlockElements;
        const uint32_t scalarHead = misalignment == 0U ? 0U :
            (count < kBlockElements - misalignment ?
                count : kBlockElements - misalignment);
        if (scalarHead != 0U) {
            needsScalarHead = true;
            if (scalarPass) {
                for (uint32_t index = 0U; index < scalarHead; ++index) {
                    local.SetValue(localOffset + index,
                                   global.GetValue(sourceOffset + index));
                }
            }
        }
        if (scalarPass) {
            return;
        }

        sourceOffset += scalarHead;
        localOffset += scalarHead;
        count -= scalarHead;
        if (count == 0U) {
            return;
        }
        const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
        DataCopyPad(local[localOffset], global[sourceOffset], copy, pad);
    }

    __aicore__ inline void AppendCopyRun(LocalTensor<T> local,
                                          const GlobalTensor<T>& global,
                                          uint64_t sourceOffset,
                                          uint32_t localOffset,
                                          uint32_t count,
                                          uint64_t& runSource,
                                          uint32_t& runLocal,
                                          uint32_t& runCount,
                                          const DataCopyPadExtParams<T>& pad,
                                          bool scalarPass,
                                          bool& needsScalarHead)
    {
        if (runCount != 0U && sourceOffset == runSource + runCount &&
            localOffset == runLocal + runCount) {
            runCount += count;
            return;
        }
        FlushCopyRun(local, global, runSource, runLocal, runCount, pad,
                     scalarPass, needsScalarHead);
        runSource = sourceOffset;
        runLocal = localOffset;
        runCount = count;
    }

    __aicore__ inline void MaterializeInputRuns(
        LocalTensor<T> x1Destination, LocalTensor<T> x2Destination,
        uint64_t outputOffset, uint32_t count,
        const DataCopyPadExtParams<T>& pad, bool scalarPass,
        bool& needsScalarHead)
    {
        uint64_t row = outputOffset / tiling_->rowElements;
        uint64_t column = outputOffset - row * tiling_->rowElements;
        uint32_t localOffset = 0U;
        uint32_t segmentIndex = 0U;
        uint64_t x1RunSource = 0U;
        uint64_t x2RunSource = 0U;
        uint32_t x1RunLocal = 0U;
        uint32_t x2RunLocal = 0U;
        uint32_t x1RunCount = 0U;
        uint32_t x2RunCount = 0U;
        while (localOffset < count) {
            const uint64_t rowRemaining = tiling_->rowElements - column;
            const uint32_t remaining = count - localOffset;
            const uint32_t segment = static_cast<uint32_t>(
                rowRemaining < remaining ? rowRemaining : remaining);
            uint64_t x1Offset = 0U;
            uint64_t x2Offset = 0U;
            InputRowOffsets(row, x1Offset, x2Offset);
            if (tiling_->x1InnerBroadcast == 0U) {
                x1Offset += column;
            }
            if (tiling_->x2InnerBroadcast == 0U) {
                x2Offset += column;
            }
            const uint32_t x1Count =
                tiling_->x1InnerBroadcast != 0U ? 1U : segment;
            const uint32_t x2Count =
                tiling_->x2InnerBroadcast != 0U ? 1U : segment;
            const uint32_t x1LocalOffset =
                tiling_->x1InnerBroadcast != 0U ? segmentIndex : localOffset;
            const uint32_t x2LocalOffset =
                tiling_->x2InnerBroadcast != 0U ? segmentIndex : localOffset;
            AppendCopyRun(x1Destination, x1Global_, x1Offset, x1LocalOffset,
                          x1Count, x1RunSource, x1RunLocal, x1RunCount, pad,
                          scalarPass, needsScalarHead);
            AppendCopyRun(x2Destination, x2Global_, x2Offset, x2LocalOffset,
                          x2Count, x2RunSource, x2RunLocal, x2RunCount, pad,
                          scalarPass, needsScalarHead);

            localOffset += segment;
            ++segmentIndex;
            column += segment;
            if (column == tiling_->rowElements) {
                column = 0U;
                ++row;
            }
        }
        FlushCopyRun(x1Destination, x1Global_, x1RunSource, x1RunLocal,
                     x1RunCount, pad, scalarPass, needsScalarHead);
        FlushCopyRun(x2Destination, x2Global_, x2RunSource, x2RunLocal,
                     x2RunCount, pad, scalarPass, needsScalarHead);
    }

    __aicore__ inline void CopyIn(uint64_t outputOffset,
                                  uint32_t count)
    {
        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        LocalTensor<T> x1Destination = x1Local;
        LocalTensor<T> x2Destination = x2Local;
        if (tiling_->x1InnerBroadcast != 0) {
            x1Destination = scalarQueue_.AllocTensor<T>();
        } else if (tiling_->x2InnerBroadcast != 0) {

            x2Destination = scalarQueue_.AllocTensor<T>();
        }
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        bool needsScalarHead = false;
        MaterializeInputRuns(x1Destination, x2Destination, outputOffset,
                             count, pad, false, needsScalarHead);
        if (needsScalarHead) {
            const event_t mte2ToScalar = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
            MaterializeInputRuns(x1Destination, x2Destination, outputOffset,
                                 count, pad, true, needsScalarHead);
            const event_t scalarToVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
        }
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
        if (tiling_->x1InnerBroadcast != 0) {
            scalarQueue_.EnQue(x1Destination);
        } else if (tiling_->x2InnerBroadcast != 0) {
            scalarQueue_.EnQue(x2Destination);
        }
    }

    __aicore__ inline void ExpandBroadcast(LocalTensor<T> tensor,
                                            LocalTensor<T> scalars,
                                            uint64_t outputOffset,
                                            uint32_t count)
    {
        if (CanUseBatchBroadcast(outputOffset, count)) {
            const uint32_t rowElements = static_cast<uint32_t>(tiling_->rowElements);
            const uint32_t rows = count / rowElements;
            const uint32_t outputShape[2] = {rows, rowElements};
            const uint32_t inputShape[2] = {rows, 1U};
            LocalTensor<uint8_t> scratch = broadcastScratchBuffer_.Get<uint8_t>();
            Broadcast<T, 2, 1, false>(tensor, scalars, outputShape, inputShape, scratch);
            return;
        }

        const uint64_t startColumn = outputOffset % tiling_->rowElements;
        const uint64_t firstAvailable = tiling_->rowElements - startColumn;
        const uint64_t afterFirst = count > firstAvailable ?
            count - firstAvailable : 0U;
        uint32_t segmentIndex = 1U + static_cast<uint32_t>(
            afterFirst / tiling_->rowElements +
            (afterFirst % tiling_->rowElements != 0U ? 1U : 0U));
        uint32_t remaining = count;
        uint64_t globalEnd = outputOffset + count;
        while (remaining != 0U) {
            const uint64_t endColumn =
                (globalEnd - 1U) % tiling_->rowElements;
            const uint32_t available = static_cast<uint32_t>(endColumn + 1U);
            const uint32_t segment = available < remaining ? available : remaining;
            const uint32_t localOffset = remaining - segment;
            --segmentIndex;
            const T scalar = scalars.GetValue(segmentIndex);
            BroadcastRange<T>::Run(tensor, localOffset, segment, scalar);
            remaining -= segment;
            globalEnd -= segment;
        }
    }

    __aicore__ inline bool CanUseBatchBroadcast(uint64_t outputOffset,
                                                 uint32_t count) const
    {
        if constexpr (!SupportsHighLevelBroadcast<T>::kValue) {
            return false;
        }
        if (tiling_->rowElements > tileElements_ ||
            outputOffset % tiling_->rowElements != 0U ||
            count % tiling_->rowElements != 0U) {
            return false;
        }
        const uint32_t rowElements = static_cast<uint32_t>(tiling_->rowElements);
        const uint32_t rows = count / rowElements;
        constexpr uint32_t kMinimumRows = 8U;
        if (rows < kMinimumRows) {
            return false;
        }

        constexpr uint32_t kHalfBlockElements = 32U / sizeof(half);
        constexpr uint32_t kTypeBlockElements = 32U / sizeof(T);
        if constexpr (sizeof(T) == 1U) {
            const uint32_t alignedRows =
                (rows + kHalfBlockElements - 1U) & ~(kHalfBlockElements - 1U);
            const uint32_t alignedOutput =
                (count + kHalfBlockElements - 1U) & ~(kHalfBlockElements - 1U);
            const uint32_t scratchHalfElements =
                kBroadcastScratchBytes / sizeof(half);
            if (alignedRows + alignedOutput >= scratchHalfElements) {
                return false;
            }
            const uint32_t algorithmElements =
                scratchHalfElements - alignedRows - alignedOutput;
            const uint32_t alignedRow =
                (rowElements + kHalfBlockElements - 1U) & ~(kHalfBlockElements - 1U);
            const uint32_t minimumElements = kHalfBlockElements * kHalfBlockElements +
                (rowElements % kHalfBlockElements == 0U ?
                    0U : kHalfBlockElements * alignedRow);
            return algorithmElements >= minimumElements;
        } else {
            const uint32_t scratchElements = kBroadcastScratchBytes / sizeof(T);
            const uint32_t alignedRow =
                (rowElements + kTypeBlockElements - 1U) & ~(kTypeBlockElements - 1U);
            const uint32_t minimumElements = kTypeBlockElements * kTypeBlockElements +
                (rowElements % kTypeBlockElements == 0U ?
                    0U : kTypeBlockElements * alignedRow);
            return scratchElements >= minimumElements;
        }
    }

    __aicore__ inline void Compute(uint64_t outputOffset, uint32_t count)
    {
        constexpr uint32_t kVectorAlignment = 256U;
        const uint32_t vectorCount =
            (count + kVectorAlignment - 1U) & ~(kVectorAlignment - 1U);
        LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        LocalTensor<half> selectionValues = selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected = selectedBuffer_.Get<half>();
        LocalTensor<float> convertedX1 = convertedX1Buffer_.Get<float>();
        LocalTensor<float> convertedX2 = convertedX2Buffer_.Get<float>();
        LocalTensor<T> scalars;

        if (tiling_->x1InnerBroadcast != 0) {
            scalars = scalarQueue_.DeQue<T>();
            ExpandBroadcast(x1Local, scalars, outputOffset, count);
        } else if (tiling_->x2InnerBroadcast != 0) {
            scalars = scalarQueue_.DeQue<T>();
            ExpandBroadcast(x2Local, scalars, outputOffset, count);
        }
        PipeBarrier<PIPE_ALL>();
        CompareRunner<T>::Run(mask, x1Local, x2Local, convertedX1,
                              convertedX2, vectorCount);
        PipeBarrier<PIPE_V>();
        OutputSelector<T>::Run(selected, mask, selectionValues, vectorCount);
        PipeBarrier<PIPE_V>();
        Cast(yLocal.ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, vectorCount);

        yQueue_.EnQue(yLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
        if (tiling_->x1InnerBroadcast != 0) {
            scalarQueue_.FreeTensor(scalars);
        } else if (tiling_->x2InnerBroadcast != 0) {
            scalarQueue_.FreeTensor(scalars);
        }
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset, uint32_t count)
    {
        LocalTensor<uint8_t> yLocal = yQueue_.DeQue<uint8_t>();
        const DataCopyExtParams copy(1, count * sizeof(uint8_t), 0, 0, 0);
        DataCopyPad(yGlobal_[outputOffset], yLocal, copy);
        yQueue_.FreeTensor(yLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> x1Queue_;
    TQue<QuePosition::VECIN, kBufferNum> x2Queue_;
    TQue<QuePosition::VECIN, kBufferNum> scalarQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> yQueue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectedBuffer_;
    TBuf<TPosition::VECCALC> convertedX1Buffer_;
    TBuf<TPosition::VECCALC> convertedX2Buffer_;
    TBuf<TPosition::VECCALC> broadcastScratchBuffer_;
    GlobalTensor<T> x1Global_;
    GlobalTensor<T> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterTilingData* tiling_;
    uint32_t tileElements_;
};

template <typename T>
// 将能够容纳的广播片段搬入 UB，复用输入生成连续输出。
class KernelGreaterBox {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        constexpr uint64_t kLargeBytes = 1ULL * 1024ULL * 1024ULL;
        constexpr uint64_t kBytesPerElement = sizeof(T) + 1ULL;
        const bool pureRowLeft = tiling.matrixX1Pattern ==
            S9_GREATER_MATRIX_ROW_BROADCAST && tiling.matrixX2Pattern == 0U;
        const bool pureRowRight = tiling.matrixX2Pattern ==
            S9_GREATER_MATRIX_ROW_BROADCAST && tiling.matrixX1Pattern == 0U;
        if (tiling.matrixEnabled != 0U && (pureRowLeft || pureRowRight) &&
            tiling.totalElements >=
                (kLargeBytes + kBytesPerElement - 1U) / kBytesPerElement) {
            if (pureRowLeft) {
                x2Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            } else {
                x1Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
        }
        constexpr uint32_t kBoxTileElements = BoxTilePolicy<T>::kElements;
        tileElements_ = kBoxTileElements;

        pipe_.InitBuffer(x1Queue_, kBufferNum,
                         kBoxTileElements * sizeof(T));
        pipe_.InitBuffer(x2Queue_, kBufferNum,
                         kBoxTileElements * sizeof(T));
        pipe_.InitBuffer(x1AuxQueue_, kBufferNum,
                         kBoxTileElements * sizeof(T));
        pipe_.InitBuffer(x2AuxQueue_, kBufferNum,
                         kBoxTileElements * sizeof(T));
        pipe_.InitBuffer(yQueue_, kBufferNum,
                         kBoxTileElements * sizeof(uint8_t));
        pipe_.InitBuffer(maskBuffer_, kBoxTileElements / 8U);
        pipe_.InitBuffer(selectionValueBuffer_,
                         kBoxTileElements * sizeof(half));
        pipe_.InitBuffer(selectedBuffer_,
                         kBoxTileElements * sizeof(half));
        constexpr uint32_t kMinimumBufferBytes = 32U;
        const uint32_t convertedBytes =
            ConversionStorage<T>::kBytesPerElement == 0U ?
                kMinimumBufferBytes :
                kBoxTileElements * ConversionStorage<T>::kBytesPerElement;
        pipe_.InitBuffer(convertedX1Buffer_, convertedBytes);
        pipe_.InitBuffer(convertedX2Buffer_, convertedBytes);
        pipe_.InitBuffer(broadcastScratchBuffer_, kBroadcastScratchBytes);
    }

    __aicore__ inline void Process()
    {
        if (tiling_->matrixEnabled != 0U) {
            ProcessMatrix();
            return;
        }

        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseBoxes = tiling_->boxCount / blocks;
        const uint64_t remainder = tiling_->boxCount % blocks;
        const uint64_t beginBox = baseBoxes * block +
            (block < remainder ? block : remainder);
        const uint64_t boxCount = baseBoxes +
            (block < remainder ? 1U : 0U);
        if (boxCount == 0U) {
            return;
        }

        LocalTensor<half> selectionValues = selectionValueBuffer_.Get<half>();
        OutputSelector<T>::InitTensor(selectionValues, tileElements_);
        PipeBarrier<PIPE_V>();

        const uint32_t elements = static_cast<uint32_t>(tiling_->boxElements);
        for (uint64_t relative = 0U; relative < boxCount; ++relative) {
            const uint64_t boxIndex = beginBox + relative;
            CopyIn(boxIndex);
            Compute(elements);
            CopyOut(boxIndex * tiling_->boxElements, elements);
        }
    }

private:
    __aicore__ inline uint64_t MatrixInputOffset(
        uint64_t outerIndex, uint32_t rowOffset, uint32_t pattern,
        const uint64_t* inputDims, const uint64_t* inputStrides) const
    {
        uint64_t inputOffset = 0U;
        for (uint32_t reverse = tiling_->matrixRowStart;
             reverse > 0U; --reverse) {
            const uint32_t axis = reverse - 1U;
            const uint64_t coordinate =
                outerIndex % tiling_->outputDims[axis];
            outerIndex /= tiling_->outputDims[axis];
            if (inputDims[axis] != 1U) {
                inputOffset += coordinate * inputStrides[axis];
            }
        }
        if ((pattern & S9_GREATER_MATRIX_ROW_BROADCAST) == 0U) {
            inputOffset += static_cast<uint64_t>(rowOffset) *
                inputStrides[tiling_->outerRank - 1U];
        }
        return inputOffset;
    }

    __aicore__ inline uint32_t MatrixInputCount(uint32_t rows,
                                                uint32_t pattern) const
    {
        const bool rowBroadcast =
            (pattern & S9_GREATER_MATRIX_ROW_BROADCAST) != 0U;
        const bool colBroadcast =
            (pattern & S9_GREATER_MATRIX_COL_BROADCAST) != 0U;
        const uint32_t inputRows = rowBroadcast ? 1U : rows;
        const uint32_t inputCols = colBroadcast ? 1U :
            static_cast<uint32_t>(tiling_->matrixCols);
        return inputRows * inputCols;
    }

    __aicore__ inline void CopyInMatrix(uint64_t outerIndex,
                                        uint32_t rowOffset,
                                        uint32_t rows)
    {
        LocalTensor<T> x1Primary = x1Queue_.AllocTensor<T>();
        LocalTensor<T> x2Primary = x2Queue_.AllocTensor<T>();
        LocalTensor<T> x1Aux = x1AuxQueue_.AllocTensor<T>();
        LocalTensor<T> x2Aux = x2AuxQueue_.AllocTensor<T>();
        LocalTensor<T> x1Destination =
            tiling_->matrixX1Pattern == 0U ? x1Primary : x1Aux;
        LocalTensor<T> x2Destination =
            tiling_->matrixX2Pattern == 0U ? x2Primary : x2Aux;

        const uint64_t x1Offset = MatrixInputOffset(
            outerIndex, rowOffset, tiling_->matrixX1Pattern,
            tiling_->x1Dims, tiling_->x1Strides);
        const uint64_t x2Offset = MatrixInputOffset(
            outerIndex, rowOffset, tiling_->matrixX2Pattern,
            tiling_->x2Dims, tiling_->x2Strides);
        const uint32_t x1Count =
            MatrixInputCount(rows, tiling_->matrixX1Pattern);
        const uint32_t x2Count =
            MatrixInputCount(rows, tiling_->matrixX2Pattern);
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        const DataCopyExtParams x1Copy(
            1, x1Count * sizeof(T), 0, 0, 0);
        const DataCopyExtParams x2Copy(
            1, x2Count * sizeof(T), 0, 0, 0);
        DataCopyPad(x1Destination, x1Global_[x1Offset], x1Copy, pad);
        DataCopyPad(x2Destination, x2Global_[x2Offset], x2Copy, pad);
        x1Queue_.EnQue(x1Primary);
        x2Queue_.EnQue(x2Primary);
        x1AuxQueue_.EnQue(x1Aux);
        x2AuxQueue_.EnQue(x2Aux);
    }

    __aicore__ inline void ProcessMatrix()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        OutputSelector<T>::InitTensor(selectionValues, tileElements_);
        PipeBarrier<PIPE_V>();

        if (tileCount > 1U) {
            if (tiling_->matrixX1Pattern == S9_GREATER_MATRIX_ROW_BROADCAST &&
                tiling_->matrixX2Pattern == 0U) {
                ProcessCachedMatrixRow<true>(beginTile, tileCount);
                return;
            }
            if (tiling_->matrixX2Pattern == S9_GREATER_MATRIX_ROW_BROADCAST &&
                tiling_->matrixX1Pattern == 0U) {
                ProcessCachedMatrixRow<false>(beginTile, tileCount);
                return;
            }
        }

        for (uint64_t relative = 0U; relative < tileCount; ++relative) {
            const uint64_t tile = beginTile + relative;
            const uint64_t outerIndex =
                tile / tiling_->matrixRowTiles;
            const uint32_t rowTile = static_cast<uint32_t>(
                tile - outerIndex * tiling_->matrixRowTiles);
            const uint32_t rowOffset =
                rowTile * tiling_->matrixRowsPerTile;
            const uint64_t remainingRows =
                tiling_->matrixRows - rowOffset;
            const uint32_t rows = static_cast<uint32_t>(
                remainingRows < tiling_->matrixRowsPerTile ?
                    remainingRows : tiling_->matrixRowsPerTile);
            const uint32_t count =
                rows * static_cast<uint32_t>(tiling_->matrixCols);
            const uint64_t outputOffset =
                outerIndex * tiling_->matrixRows * tiling_->matrixCols +
                static_cast<uint64_t>(rowOffset) * tiling_->matrixCols;

            CopyInMatrix(outerIndex, rowOffset, rows);
            ComputeMatrix(rows, count);
            CopyOut(outputOffset, count);
        }
    }

    __aicore__ inline uint32_t BroadcastGroupCount(
        const uint64_t* inputDims) const
    {
        uint32_t groups = 0U;
        bool previousBroadcast = false;
        for (uint32_t axis = tiling_->boxStart; axis < tiling_->rank; ++axis) {
            if (tiling_->outputDims[axis] == 1U) {
                continue;
            }
            const bool broadcast = inputDims[axis] == 1U;
            if (broadcast && !previousBroadcast) {
                ++groups;
            }
            previousBroadcast = broadcast;
        }
        return groups;
    }

    __aicore__ inline uint32_t CompactElementCount(
        const uint64_t* inputDims) const
    {
        uint32_t elements = 1U;
        for (uint32_t axis = tiling_->boxStart; axis < tiling_->rank; ++axis) {
            elements *= static_cast<uint32_t>(inputDims[axis]);
        }
        return elements;
    }

    __aicore__ inline uint64_t InputOffset(
        uint64_t boxIndex, const uint64_t* inputDims,
        const uint64_t* inputStrides) const
    {
        uint64_t inputOffset = 0U;
        for (uint32_t reverse = tiling_->boxStart; reverse > 0U; --reverse) {
            const uint32_t axis = reverse - 1U;
            const uint64_t coordinate = boxIndex % tiling_->outputDims[axis];
            boxIndex /= tiling_->outputDims[axis];
            if (inputDims[axis] != 1U) {
                inputOffset += coordinate * inputStrides[axis];
            }
        }
        return inputOffset;
    }

    __aicore__ inline void CopyIn(uint64_t boxIndex)
    {
        LocalTensor<T> x1Primary = x1Queue_.AllocTensor<T>();
        LocalTensor<T> x2Primary = x2Queue_.AllocTensor<T>();
        LocalTensor<T> x1Aux = x1AuxQueue_.AllocTensor<T>();
        LocalTensor<T> x2Aux = x2AuxQueue_.AllocTensor<T>();
        const uint32_t x1Groups = BroadcastGroupCount(tiling_->x1Dims);
        const uint32_t x2Groups = BroadcastGroupCount(tiling_->x2Dims);
        LocalTensor<T> x1Destination =
            (x1Groups & 1U) == 0U ? x1Primary : x1Aux;
        LocalTensor<T> x2Destination =
            (x2Groups & 1U) == 0U ? x2Primary : x2Aux;
        const uint32_t x1Elements = CompactElementCount(tiling_->x1Dims);
        const uint32_t x2Elements = CompactElementCount(tiling_->x2Dims);
        const uint64_t x1Offset = InputOffset(
            boxIndex, tiling_->x1Dims, tiling_->x1Strides);
        const uint64_t x2Offset = InputOffset(
            boxIndex, tiling_->x2Dims, tiling_->x2Strides);
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        const DataCopyExtParams x1Copy(1, x1Elements * sizeof(T), 0, 0, 0);
        const DataCopyExtParams x2Copy(1, x2Elements * sizeof(T), 0, 0, 0);
        DataCopyPad(x1Destination, x1Global_[x1Offset], x1Copy, pad);
        DataCopyPad(x2Destination, x2Global_[x2Offset], x2Copy, pad);
        x1Queue_.EnQue(x1Primary);
        x2Queue_.EnQue(x2Primary);
        x1AuxQueue_.EnQue(x1Aux);
        x2AuxQueue_.EnQue(x2Aux);
    }

    __aicore__ inline void CopyVector(LocalTensor<T> destination,
                                      LocalTensor<T> source,
                                      uint32_t count)
    {
        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(count);
        Copy<T, false>(destination, source, MASK_PLACEHOLDER, 1,
                       {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
    }

    template <typename U>
    __aicore__ inline void RepeatAlignedRow(LocalTensor<U> destination,
                                             LocalTensor<U> source,
                                             LocalTensor<U> zeros,
                                             uint32_t rows,
                                             uint32_t alignedCols)
    {
        constexpr uint32_t kRowsPerMask = 8U;
        constexpr uint32_t kUint16PerBlock = 16U;
        constexpr uint32_t kBlockElements = 32U / sizeof(U);
        const uint32_t totalRepeatTimes =
            alignedCols / kBlockElements;
        const uint32_t destinationBlockStride =
            alignedCols * (sizeof(U) == sizeof(float) ? 2U : 1U) /
            kUint16PerBlock;

        if (destinationBlockStride > 255U) {

            static_assert(kBroadcastScratchBytes / 32U <= 4095U);
            const CopyRepeatParams copyParams(
                1U, 1U, static_cast<uint16_t>(totalRepeatTimes), 0U);
            SetMaskCount();
            SetVectorMask<uint16_t, MaskMode::COUNTER>(
                alignedCols * sizeof(U) / sizeof(uint16_t));
            Copy<uint16_t, false>(
                destination.template ReinterpretCast<uint16_t>(),
                source.template ReinterpretCast<uint16_t>(),
                MASK_PLACEHOLDER, static_cast<uint8_t>(rows), copyParams);
            PipeBarrier<PIPE_V>();
            SetMaskNorm();
            ResetMask();
            return;
        }
        const BinaryRepeatParams params(
            static_cast<uint8_t>(destinationBlockStride), 0, 0,
            1, 1, 0);
        const uint32_t groups = rows / kRowsPerMask;
        uint32_t destinationOffset = 0U;
        SetMaskNorm();
        SetVectorMask<uint16_t, MaskMode::NORMAL>(
            kRowsPerMask * kUint16PerBlock);
        for (uint32_t group = 0U; group < groups; ++group) {
            Or<uint16_t, false>(
                destination[destinationOffset].template ReinterpretCast<uint16_t>(),
                source.template ReinterpretCast<uint16_t>(),
                zeros.template ReinterpretCast<uint16_t>(),
                MASK_PLACEHOLDER,
                static_cast<uint8_t>(totalRepeatTimes), params);
            destinationOffset += kRowsPerMask * alignedCols;
        }
        const uint32_t tailRows = rows - groups * kRowsPerMask;
        if (tailRows != 0U) {
            SetVectorMask<uint16_t, MaskMode::NORMAL>(
                tailRows * kUint16PerBlock);
            Or<uint16_t, false>(
                destination[destinationOffset].template ReinterpretCast<uint16_t>(),
                source.template ReinterpretCast<uint16_t>(),
                zeros.template ReinterpretCast<uint16_t>(),
                MASK_PLACEHOLDER,
                static_cast<uint8_t>(totalRepeatTimes), params);
        }
        PipeBarrier<PIPE_V>();
    }

    template <typename U>
    __aicore__ inline void CompactPaddedRows(LocalTensor<U> destination,
                                              LocalTensor<U> padded,
                                              uint32_t rows,
                                              uint32_t cols,
                                              uint32_t alignedCols)
    {
        constexpr uint8_t kGatherPattern = 7U;
        constexpr uint32_t kBlockElements = 32U / sizeof(U);
        const GatherMaskParams params = {
            1, static_cast<uint16_t>(rows),
            static_cast<uint16_t>(alignedCols / kBlockElements), 0};
        uint64_t reservedCount = 0U;
        if constexpr (IsBfloat16<U>::kValue) {
            GatherMask(destination.template ReinterpretCast<uint16_t>(),
                       padded.template ReinterpretCast<uint16_t>(),
                       kGatherPattern, true, cols, params, reservedCount);
        } else {
            GatherMask(destination, padded, kGatherPattern, true, cols,
                       params, reservedCount);
        }
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void RepeatRowsScalar(LocalTensor<T> destination,
                                             LocalTensor<T> source,
                                             uint32_t rows,
                                             uint32_t cols)
    {
        for (uint32_t row = 0U; row < rows; ++row) {
            for (uint32_t col = 0U; col < cols; ++col) {
                destination.SetValue(row * cols + col, source.GetValue(col));
            }
        }
    }

    __aicore__ inline uint32_t AlignedRowsPerChunk(
        uint32_t rows, uint32_t maximumRows, uint32_t cols,
        uint32_t blockElements) const
    {
        if (rows <= maximumRows) {
            return rows;
        }
        uint32_t lhs = cols;
        uint32_t rhs = blockElements;
        while (rhs != 0U) {
            const uint32_t remainder = lhs % rhs;
            lhs = rhs;
            rhs = remainder;
        }
        const uint32_t rowQuantum = blockElements / lhs;
        return maximumRows / rowQuantum * rowQuantum;
    }

    __aicore__ inline bool CanUseLastBroadcast(uint32_t rows,
                                                uint32_t cols) const
    {
        if constexpr (!SupportsHighLevelBroadcast<T>::kValue) {
            return false;
        }
        constexpr uint32_t kMinimumRows = 8U;
        if (rows < kMinimumRows) {
            return false;
        }
        const uint32_t count = rows * cols;
        constexpr uint32_t kHalfBlockElements = 32U / sizeof(half);
        constexpr uint32_t kTypeBlockElements = 32U / sizeof(T);
        if constexpr (sizeof(T) == 1U) {
            const uint32_t alignedRows =
                (rows + kHalfBlockElements - 1U) &
                ~(kHalfBlockElements - 1U);
            const uint32_t alignedOutput =
                (count + kHalfBlockElements - 1U) &
                ~(kHalfBlockElements - 1U);
            const uint32_t scratchHalfElements =
                kBroadcastScratchBytes / sizeof(half);
            if (alignedRows + alignedOutput >= scratchHalfElements) {
                return false;
            }
            const uint32_t algorithmElements =
                scratchHalfElements - alignedRows - alignedOutput;
            const uint32_t alignedCols =
                (cols + kHalfBlockElements - 1U) &
                ~(kHalfBlockElements - 1U);
            const uint32_t minimumElements =
                kHalfBlockElements * kHalfBlockElements +
                (cols % kHalfBlockElements == 0U ?
                    0U : kHalfBlockElements * alignedCols);
            return algorithmElements >= minimumElements;
        } else {
            const uint32_t scratchElements =
                kBroadcastScratchBytes / sizeof(T);
            const uint32_t alignedCols =
                (cols + kTypeBlockElements - 1U) &
                ~(kTypeBlockElements - 1U);
            const uint32_t minimumElements =
                kTypeBlockElements * kTypeBlockElements +
                (cols % kTypeBlockElements == 0U ?
                    0U : kTypeBlockElements * alignedCols);
            return scratchElements >= minimumElements;
        }
    }

    __aicore__ inline void BroadcastLastAxis(LocalTensor<T> destination,
                                              LocalTensor<T> source,
                                              uint32_t rows,
                                              uint32_t cols)
    {
        if (CanUseLastBroadcast(rows, cols)) {
            const uint32_t outputShape[2] = {rows, cols};
            const uint32_t inputShape[2] = {rows, 1U};
            LocalTensor<uint8_t> scratch =
                broadcastScratchBuffer_.Get<uint8_t>();
            Broadcast<T, 2, 1, false>(destination, source,
                                      outputShape, inputShape, scratch);
            return;
        }
        for (uint32_t row = 0U; row < rows; ++row) {
            const T scalar = source.GetValue(row);
            BroadcastRange<T>::Run(destination, row * cols, cols, scalar);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BroadcastFirstAxis(LocalTensor<T> destination,
                                               LocalTensor<T> source,
                                               uint32_t rows,
                                               uint32_t cols,
                                               uint32_t destinationOffset,
                                               uint32_t sourceOffset)
    {
        constexpr uint32_t kTypeBlockElements = 32U / sizeof(T);
        if (destinationOffset % kTypeBlockElements != 0U ||
            sourceOffset % kTypeBlockElements != 0U) {
            RepeatRowsScalar(destination, source, rows, cols);
            return;
        }
        if (rows == 1U) {
            if constexpr (sizeof(T) == 1U) {

                LocalTensor<half> copied = broadcastScratchBuffer_
                    .Get<uint8_t>().ReinterpretCast<half>();
                Cast(copied, source, RoundMode::CAST_NONE, cols);
                PipeBarrier<PIPE_V>();
                Cast(destination.template ReinterpretCast<int8_t>(), copied,
                     RoundMode::CAST_RINT, cols);
                PipeBarrier<PIPE_V>();
            } else {
                CopyVector(destination, source, cols);
            }
            return;
        }
        if (cols == 1U) {
            const T scalar = source.GetValue(0U);
            BroadcastRange<T>::Run(destination, 0U, rows, scalar);
            PipeBarrier<PIPE_V>();
            return;
        }

        if constexpr (sizeof(T) == 1U) {
            constexpr uint32_t kBlockElements = 32U / sizeof(half);
            const uint32_t alignedCols =
                (cols + kBlockElements - 1U) & ~(kBlockElements - 1U);
            const uint32_t scratchHalfElements =
                kBroadcastScratchBytes / sizeof(half);

            const uint32_t fixedElements = alignedCols + 2U * kBlockElements;
            const uint32_t maximumRows =
                (scratchHalfElements - fixedElements) /
                (alignedCols + cols);
            const uint32_t rowsPerChunk = AlignedRowsPerChunk(
                rows, maximumRows, cols, 32U / sizeof(int8_t));
            if (rowsPerChunk == 0U) {
                RepeatRowsScalar(destination, source, rows, cols);
                return;
            }
            LocalTensor<half> scratch =
                broadcastScratchBuffer_.Get<uint8_t>().ReinterpretCast<half>();
            LocalTensor<half> convertedSource = scratch;
            LocalTensor<half> padded = convertedSource[alignedCols];
            LocalTensor<half> dense = padded[rowsPerChunk * alignedCols];
            const uint32_t denseElements = rowsPerChunk * cols;
            const uint32_t alignedDenseElements =
                (denseElements + kBlockElements - 1U) &
                ~(kBlockElements - 1U);
            LocalTensor<half> zeros = dense[alignedDenseElements];
            Cast(convertedSource, source, RoundMode::CAST_NONE, cols);
            Duplicate(zeros.template ReinterpretCast<uint16_t>(),
                      static_cast<uint16_t>(0), kBlockElements);
            PipeBarrier<PIPE_V>();
            uint32_t rowOffset = 0U;
            while (rowOffset < rows) {
                const uint32_t remaining = rows - rowOffset;
                const uint32_t chunkRows =
                    remaining < rowsPerChunk ? remaining : rowsPerChunk;
                RepeatAlignedRow(padded, convertedSource, zeros,
                                 chunkRows, alignedCols);
                CompactPaddedRows(dense, padded, chunkRows,
                                  cols, alignedCols);
                Cast(destination[rowOffset * cols].template ReinterpretCast<int8_t>(),
                     dense, RoundMode::CAST_RINT, chunkRows * cols);
                PipeBarrier<PIPE_V>();
                rowOffset += chunkRows;
            }
        } else {
            if constexpr (SupportsHighLevelBroadcast<T>::kValue) {
                if ((cols * sizeof(T)) % 32U == 0U) {
                    const uint32_t outputShape[2] = {rows, cols};
                    const uint32_t inputShape[2] = {1U, cols};
                    LocalTensor<uint8_t> scratch =
                        broadcastScratchBuffer_.Get<uint8_t>();
                    Broadcast<T, 2, 0, false>(destination, source,
                                              outputShape, inputShape, scratch);
                    return;
                }
            }
            constexpr uint32_t kBlockElements = 32U / sizeof(T);
            const uint32_t alignedCols =
                (cols + kBlockElements - 1U) & ~(kBlockElements - 1U);
            const uint32_t scratchElements =
                kBroadcastScratchBytes / sizeof(T);
            const uint32_t maximumRows =
                (scratchElements - kBlockElements) / alignedCols;
            const uint32_t rowsPerChunk = AlignedRowsPerChunk(
                rows, maximumRows, cols, kBlockElements);
            if (rowsPerChunk == 0U) {
                RepeatRowsScalar(destination, source, rows, cols);
                return;
            }
            LocalTensor<T> scratch =
                broadcastScratchBuffer_.Get<uint8_t>().ReinterpretCast<T>();
            LocalTensor<T> padded = scratch;
            LocalTensor<T> zeros = padded[rowsPerChunk * alignedCols];
            Duplicate(zeros.template ReinterpretCast<uint16_t>(),
                      static_cast<uint16_t>(0), 32U / sizeof(uint16_t));
            PipeBarrier<PIPE_V>();
            uint32_t rowOffset = 0U;
            while (rowOffset < rows) {
                const uint32_t remaining = rows - rowOffset;
                const uint32_t chunkRows =
                    remaining < rowsPerChunk ? remaining : rowsPerChunk;
                RepeatAlignedRow(padded, source, zeros,
                                 chunkRows, alignedCols);
                CompactPaddedRows(destination[rowOffset * cols], padded,
                                  chunkRows, cols, alignedCols);
                rowOffset += chunkRows;
            }
        }
    }

    __aicore__ inline void ExpandGroup(LocalTensor<T> destination,
                                        LocalTensor<T> source,
                                        uint32_t outerCompact,
                                        uint32_t repeats,
                                        uint32_t innerElements)
    {
        if (innerElements == 1U) {
            BroadcastLastAxis(destination, source, outerCompact, repeats);
            return;
        }
        for (uint32_t outer = 0U; outer < outerCompact; ++outer) {
            BroadcastFirstAxis(
                destination[outer * repeats * innerElements],
                source[outer * innerElements], repeats, innerElements,
                outer * repeats * innerElements,
                outer * innerElements);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ExpandMatrixInput(LocalTensor<T> destination,
                                              LocalTensor<T> source,
                                              uint32_t rows,
                                              uint32_t pattern)
    {
        const uint32_t cols =
            static_cast<uint32_t>(tiling_->matrixCols);
        if (pattern == S9_GREATER_MATRIX_COL_BROADCAST) {
            BroadcastLastAxis(destination, source, rows, cols);
        } else if (pattern == S9_GREATER_MATRIX_ROW_BROADCAST) {
            BroadcastFirstAxis(destination, source, rows, cols, 0U, 0U);
        } else {
            const T scalar = source.GetValue(0U);
            BroadcastRange<T>::Run(destination, 0U, rows * cols, scalar);
        }
    }

    __aicore__ inline void ComputeMatrix(uint32_t rows, uint32_t count)
    {
        constexpr uint32_t kVectorAlignment = 256U;
        const uint32_t vectorCount =
            (count + kVectorAlignment - 1U) & ~(kVectorAlignment - 1U);
        LocalTensor<T> x1Primary = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Primary = x2Queue_.DeQue<T>();
        LocalTensor<T> x1Aux = x1AuxQueue_.DeQue<T>();
        LocalTensor<T> x2Aux = x2AuxQueue_.DeQue<T>();
        if (tiling_->matrixX1Pattern != 0U) {
            ExpandMatrixInput(x1Primary, x1Aux, rows,
                              tiling_->matrixX1Pattern);
        }
        if (tiling_->matrixX2Pattern != 0U) {
            ExpandMatrixInput(x2Primary, x2Aux, rows,
                              tiling_->matrixX2Pattern);
        }
        PipeBarrier<PIPE_ALL>();

        LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected = selectedBuffer_.Get<half>();
        LocalTensor<float> convertedX1 = convertedX1Buffer_.Get<float>();
        LocalTensor<float> convertedX2 = convertedX2Buffer_.Get<float>();
        CompareRunner<T>::Run(mask, x1Primary, x2Primary,
                              convertedX1, convertedX2, vectorCount);
        PipeBarrier<PIPE_V>();
        OutputSelector<T>::Run(selected, mask, selectionValues, vectorCount);
        PipeBarrier<PIPE_V>();
        Cast(yLocal.ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, vectorCount);

        yQueue_.EnQue(yLocal);
        x1Queue_.FreeTensor(x1Primary);
        x2Queue_.FreeTensor(x2Primary);
        x1AuxQueue_.FreeTensor(x1Aux);
        x2AuxQueue_.FreeTensor(x2Aux);
    }

    template <bool RowIsX1>
    __aicore__ inline void PrefetchCachedMatrixDense(uint64_t tile, bool auxiliary)
    {
        auto& primaryQueue = RowIsX1 ? x2Queue_ : x1Queue_;
        auto& auxiliaryQueue = RowIsX1 ? x2AuxQueue_ : x1AuxQueue_;
        auto& queue = auxiliary ? auxiliaryQueue : primaryQueue;
        auto& global = RowIsX1 ? x2Global_ : x1Global_;
        const uint64_t* dims = RowIsX1 ? tiling_->x2Dims : tiling_->x1Dims;
        const uint64_t* strides = RowIsX1 ? tiling_->x2Strides : tiling_->x1Strides;
        const uint64_t outer = tile / tiling_->matrixRowTiles;
        const uint32_t rowTile = static_cast<uint32_t>(tile - outer * tiling_->matrixRowTiles);
        const uint32_t row = rowTile * tiling_->matrixRowsPerTile;
        const uint64_t remaining = tiling_->matrixRows - row;
        const uint32_t rows = static_cast<uint32_t>(remaining < tiling_->matrixRowsPerTile ?
            remaining : tiling_->matrixRowsPerTile);
        const uint32_t count = rows * static_cast<uint32_t>(tiling_->matrixCols);
        const uint64_t offset = MatrixInputOffset(outer, row, 0U, dims, strides);
        LocalTensor<T> dense = queue.template AllocTensor<T>();
        const DataCopyExtParams copy(1U, count * sizeof(T), 0U, 0U, 0U);
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        DataCopyPad(dense, global[offset], copy, pad);
        queue.EnQue(dense);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessCachedMatrixRow(uint64_t beginTile,
                                                 uint64_t tileCount)
    {
        auto& rowQueue = RowIsX1 ? x1Queue_ : x2Queue_;
        auto& compactQueue = RowIsX1 ? x1AuxQueue_ : x2AuxQueue_;
        auto& densePrimaryQueue = RowIsX1 ? x2Queue_ : x1Queue_;
        auto& denseAuxiliaryQueue = RowIsX1 ? x2AuxQueue_ : x1AuxQueue_;
        auto& rowGlobal = RowIsX1 ? x1Global_ : x2Global_;
        const uint64_t* rowDims = RowIsX1 ? tiling_->x1Dims : tiling_->x2Dims;
        const uint64_t* rowStrides = RowIsX1 ? tiling_->x1Strides : tiling_->x2Strides;
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        LocalTensor<T> compact;
        LocalTensor<T> expanded;
        uint64_t cachedOuter = ~static_cast<uint64_t>(0U);
        uint32_t convertedRowValidCount = 0U;
        PrefetchCachedMatrixDense<RowIsX1>(beginTile, false);
        for (uint64_t relative = 0U; relative < tileCount; ++relative) {
            const uint64_t tile = beginTile + relative;
            const uint64_t outer = tile / tiling_->matrixRowTiles;
            const uint32_t rowTile = static_cast<uint32_t>(tile - outer * tiling_->matrixRowTiles);
            const uint32_t rowOffset = rowTile * tiling_->matrixRowsPerTile;
            const uint64_t remaining = tiling_->matrixRows - rowOffset;
            const uint32_t rows = static_cast<uint32_t>(remaining < tiling_->matrixRowsPerTile ?
                remaining : tiling_->matrixRowsPerTile);
            const uint32_t count = rows * cols;
            if (outer != cachedOuter) {
                if (cachedOuter != ~static_cast<uint64_t>(0U)) {
                    PipeBarrier<PIPE_V>();
                    rowQueue.FreeTensor(expanded);
                    compactQueue.FreeTensor(compact);
                }
                expanded = rowQueue.template AllocTensor<T>();
                compact = compactQueue.template AllocTensor<T>();
                const uint64_t offset = MatrixInputOffset(outer, 0U,
                    S9_GREATER_MATRIX_ROW_BROADCAST, rowDims, rowStrides);
                const DataCopyExtParams copy(1U, cols * sizeof(T), 0U, 0U, 0U);
                DataCopyPad(compact, rowGlobal[offset], copy, pad);
                rowQueue.EnQue(expanded);
                compactQueue.EnQue(compact);
                expanded = rowQueue.template DeQue<T>();
                compact = compactQueue.template DeQue<T>();
                ExpandMatrixInput(expanded, compact, tiling_->matrixRowsPerTile,
                                  S9_GREATER_MATRIX_ROW_BROADCAST);
                PipeBarrier<PIPE_V>();
                cachedOuter = outer;
                convertedRowValidCount = 0U;
            }

            auto& denseQueue = (relative & 1U) != 0U ? denseAuxiliaryQueue : densePrimaryQueue;
            LocalTensor<T> dense = denseQueue.template DeQue<T>();
            if (relative + 1U < tileCount) {
                PrefetchCachedMatrixDense<RowIsX1>(tile + 1U, ((relative + 1U) & 1U) != 0U);
            }

            const uint32_t vectorCount = (count + 255U) & ~255U;
            LocalTensor<uint8_t> yLocal = yQueue_.template AllocTensor<uint8_t>();
            LocalTensor<uint8_t> mask = maskBuffer_.template Get<uint8_t>();
            LocalTensor<half> selectionValues = selectionValueBuffer_.template Get<half>();
            LocalTensor<half> selected = selectedBuffer_.template Get<half>();
            LocalTensor<float> convertedX1 = convertedX1Buffer_.template Get<float>();
            LocalTensor<float> convertedX2 = convertedX2Buffer_.template Get<float>();
            if constexpr (IsBfloat16<T>::kValue) {
                CompareWithCachedConvertedRow<T, float, RowIsX1>(
                    mask, expanded, dense, convertedX1, convertedX2,
                    vectorCount, convertedRowValidCount);
            } else if constexpr (IsInt8<T>::kValue) {
                CompareWithCachedConvertedRow<T, half, RowIsX1>(
                    mask, expanded, dense, convertedX1, convertedX2,
                    vectorCount, convertedRowValidCount);
            } else if constexpr (RowIsX1 && IsInt32<T>::kValue) {

                Min(dense, dense, expanded, vectorCount);
                PipeBarrier<PIPE_V>();
                Compare(mask, dense, expanded, CMPMODE::EQ, vectorCount);
            } else if constexpr (RowIsX1) {
                CompareRunner<T>::Run(mask, expanded, dense, convertedX1, convertedX2, vectorCount);
            } else {
                CompareRunner<T>::Run(mask, dense, expanded, convertedX1, convertedX2, vectorCount);
            }
            PipeBarrier<PIPE_V>();
            OutputSelector<T>::Run(selected, mask, selectionValues, vectorCount);
            PipeBarrier<PIPE_V>();
            Cast(yLocal.template ReinterpretCast<int8_t>(), selected, RoundMode::CAST_RINT, vectorCount);
            yQueue_.EnQue(yLocal);
            denseQueue.FreeTensor(dense);
            CopyOut((outer * tiling_->matrixRows + rowOffset) * cols, count);
        }
        PipeBarrier<PIPE_V>();
        rowQueue.FreeTensor(expanded);
        compactQueue.FreeTensor(compact);
    }

    __aicore__ inline void ExpandInput(LocalTensor<T> primary,
                                        LocalTensor<T> auxiliary,
                                        const uint64_t* inputDims)
    {
        const uint32_t groups = BroadcastGroupCount(inputDims);
        bool sourceIsPrimary = (groups & 1U) == 0U;
        uint32_t outerCompact = CompactElementCount(inputDims);
        uint32_t innerElements = 1U;
        int32_t axis = static_cast<int32_t>(tiling_->rank) - 1;
        const int32_t boxStart = static_cast<int32_t>(tiling_->boxStart);
        while (axis >= boxStart) {
            if (tiling_->outputDims[axis] == 1U) {
                --axis;
                continue;
            }
            if (inputDims[axis] == tiling_->outputDims[axis]) {
                const uint32_t dimension =
                    static_cast<uint32_t>(inputDims[axis]);
                outerCompact /= dimension;
                innerElements *= dimension;
                --axis;
                continue;
            }

            uint32_t repeats = 1U;
            while (axis >= boxStart) {
                if (tiling_->outputDims[axis] == 1U) {
                    --axis;
                    continue;
                }
                if (inputDims[axis] != 1U) {
                    break;
                }
                repeats *= static_cast<uint32_t>(tiling_->outputDims[axis]);
                --axis;
            }
            if (sourceIsPrimary) {
                ExpandGroup(auxiliary, primary, outerCompact,
                            repeats, innerElements);
            } else {
                ExpandGroup(primary, auxiliary, outerCompact,
                            repeats, innerElements);
            }
            sourceIsPrimary = !sourceIsPrimary;
            innerElements *= repeats;
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        constexpr uint32_t kVectorAlignment = 256U;
        const uint32_t vectorCount =
            (count + kVectorAlignment - 1U) & ~(kVectorAlignment - 1U);
        LocalTensor<T> x1Primary = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Primary = x2Queue_.DeQue<T>();
        LocalTensor<T> x1Aux = x1AuxQueue_.DeQue<T>();
        LocalTensor<T> x2Aux = x2AuxQueue_.DeQue<T>();
        ExpandInput(x1Primary, x1Aux, tiling_->x1Dims);
        ExpandInput(x2Primary, x2Aux, tiling_->x2Dims);
        PipeBarrier<PIPE_ALL>();

        LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        LocalTensor<half> selectionValues = selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected = selectedBuffer_.Get<half>();
        LocalTensor<float> convertedX1 = convertedX1Buffer_.Get<float>();
        LocalTensor<float> convertedX2 = convertedX2Buffer_.Get<float>();
        CompareRunner<T>::Run(mask, x1Primary, x2Primary,
                              convertedX1, convertedX2, vectorCount);
        PipeBarrier<PIPE_V>();
        OutputSelector<T>::Run(selected, mask, selectionValues, vectorCount);
        PipeBarrier<PIPE_V>();
        Cast(yLocal.ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, vectorCount);

        yQueue_.EnQue(yLocal);
        x1Queue_.FreeTensor(x1Primary);
        x2Queue_.FreeTensor(x2Primary);
        x1AuxQueue_.FreeTensor(x1Aux);
        x2AuxQueue_.FreeTensor(x2Aux);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset, uint32_t count)
    {
        LocalTensor<uint8_t> yLocal = yQueue_.DeQue<uint8_t>();
        const DataCopyExtParams copy(1, count * sizeof(uint8_t), 0, 0, 0);
        DataCopyPad(yGlobal_[outputOffset], yLocal, copy);
        yQueue_.FreeTensor(yLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kBufferNum> x1Queue_;
    TQue<QuePosition::VECIN, kBufferNum> x2Queue_;
    TQue<QuePosition::VECIN, kBufferNum> x1AuxQueue_;
    TQue<QuePosition::VECIN, kBufferNum> x2AuxQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> yQueue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectedBuffer_;
    TBuf<TPosition::VECCALC> convertedX1Buffer_;
    TBuf<TPosition::VECCALC> convertedX2Buffer_;
    TBuf<TPosition::VECCALC> broadcastScratchBuffer_;
    GlobalTensor<T> x1Global_;
    GlobalTensor<T> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterTilingData* tiling_;
    uint32_t tileElements_;
};

template <typename T>
// 缓存可复用的行向量，按条带处理稠密输入。
class KernelGreaterRowStripe {
public:
    using ComputeT = typename RowStripePolicy<T>::ComputeType;
    static constexpr uint32_t kChunkElements =
        RowStripePolicy<T>::kChunkElements;
    static constexpr uint32_t kMaxRows =
        S9_GREATER_ROW_STRIPE_MAX_ROWS;
    static constexpr uint32_t kStripeElements =
        kMaxRows * kChunkElements;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        pipe_.InitBuffer(rowBuffer_,
                         BoxTilePolicy<T>::kElements * sizeof(T));
        pipe_.InitBuffer(denseQueue_, kBufferNum,
                         kStripeElements * sizeof(T));
        pipe_.InitBuffer(yQueue_, kBufferNum,
                         kStripeElements * sizeof(uint8_t));
        pipe_.InitBuffer(maskBuffer_, kStripeElements / 8U);
        pipe_.InitBuffer(selectionValueBuffer_,
                         kStripeElements * sizeof(half));
        pipe_.InitBuffer(selectedBuffer_,
                         kStripeElements * sizeof(half));
        pipe_.InitBuffer(denseConvertedBuffer_,
                         RowStripeUbBudget<T>::kDenseConvertedBytes);
        pipe_.InitBuffer(rowConvertedBuffer_,
                         RowStripeUbBudget<T>::kRowConvertedBytes);
        pipe_.InitBuffer(maximumBuffer_,
                         RowStripeMaximumStorage<T>::kBytes);
    }

    __aicore__ inline void Process()
    {
        if (tiling_->matrixX1Pattern ==
            S9_GREATER_MATRIX_ROW_BROADCAST) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ inline uint64_t MatrixInputOffset(
        uint64_t outerIndex, uint32_t rowOffset, uint32_t pattern,
        const uint64_t* inputDims, const uint64_t* inputStrides) const
    {
        uint64_t inputOffset = 0U;
        for (uint32_t reverse = tiling_->matrixRowStart;
             reverse > 0U; --reverse) {
            const uint32_t axis = reverse - 1U;
            const uint64_t coordinate =
                outerIndex % tiling_->outputDims[axis];
            outerIndex /= tiling_->outputDims[axis];
            if (inputDims[axis] != 1U) {
                inputOffset += coordinate * inputStrides[axis];
            }
        }
        if ((pattern & S9_GREATER_MATRIX_ROW_BROADCAST) == 0U) {
            inputOffset += static_cast<uint64_t>(rowOffset) *
                inputStrides[tiling_->outerRank - 1U];
        }
        return inputOffset;
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyCachedRow(uint64_t outerIndex,
                                         LocalTensor<T> rowLocal)
    {
        const uint32_t cols =
            static_cast<uint32_t>(tiling_->matrixCols);
        constexpr uint32_t kDataBlockBytes = 32U;
        const uint32_t rowBytes = cols * sizeof(T);
        const uint32_t alignedRowBytes =
            ((rowBytes + kDataBlockBytes - 1U) / kDataBlockBytes) *
            kDataBlockBytes;
        const uint8_t rightPadding = static_cast<uint8_t>(
            (alignedRowBytes - rowBytes) / sizeof(T));
        const DataCopyExtParams copy(
            1U, rowBytes, 0U, 0U, 0U);
        const DataCopyPadExtParams<T> pad(
            rightPadding != 0U, 0U, rightPadding, static_cast<T>(0));
        if constexpr (RowIsX1) {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, 0U, tiling_->matrixX1Pattern,
                tiling_->x1Dims, tiling_->x1Strides);
            DataCopyPad(rowLocal, x1Global_[offset], copy, pad);
        } else {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, 0U, tiling_->matrixX2Pattern,
                tiling_->x2Dims, tiling_->x2Strides);
            DataCopyPad(rowLocal, x2Global_[offset], copy, pad);
        }
        const event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyDenseStripe(
        uint64_t outerIndex, uint32_t rowOffset, uint32_t rows,
        uint32_t columnOffset, uint32_t validCols)
    {
        LocalTensor<T> denseLocal = denseQueue_.AllocTensor<T>();
        const uint32_t cols =
            static_cast<uint32_t>(tiling_->matrixCols);
        constexpr uint32_t kDataBlockBytes = 32U;
        const uint32_t validBytes = validCols * sizeof(T);
        const uint32_t alignedBytes =
            ((validBytes + kDataBlockBytes - 1U) /
                kDataBlockBytes) * kDataBlockBytes;
        const uint8_t rightPadding = static_cast<uint8_t>(
            (alignedBytes - validBytes) / sizeof(T));

        const uint32_t destinationStride =
            (kChunkElements * sizeof(T) - alignedBytes) /
            kDataBlockBytes;
        const DataCopyExtParams copy(
            static_cast<uint16_t>(rows), validBytes,
            (cols - validCols) * sizeof(T), destinationStride, 0U);
        const DataCopyPadExtParams<T> pad(
            rightPadding != 0U, 0U, rightPadding, static_cast<T>(0));
        if constexpr (RowIsX1) {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, rowOffset, tiling_->matrixX2Pattern,
                tiling_->x2Dims, tiling_->x2Strides) + columnOffset;
            DataCopyPad(denseLocal, x2Global_[offset], copy, pad);
        } else {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, rowOffset, tiling_->matrixX1Pattern,
                tiling_->x1Dims, tiling_->x1Strides) + columnOffset;
            DataCopyPad(denseLocal, x1Global_[offset], copy, pad);
        }
        denseQueue_.EnQue(denseLocal);
    }

    template <bool RowIsX1>
    __aicore__ inline void CompareStripe(
        LocalTensor<uint8_t> mask, LocalTensor<ComputeT> row,
        LocalTensor<ComputeT> dense, uint32_t rows)
    {
        const uint64_t maskCount = static_cast<uint64_t>(kChunkElements);
        const uint8_t repeatTime = static_cast<uint8_t>(rows);
        if constexpr (IsInt32<T>::kValue) {
            LocalTensor<int32_t> maximum =
                maximumBuffer_.Get<int32_t>();
            if constexpr (RowIsX1) {
                const BinaryRepeatParams maximumParams(
                    1U, 1U, 1U, 8U, 0U, 8U);
                Max(maximum, row, dense, maskCount, repeatTime,
                    maximumParams);
            } else {
                const BinaryRepeatParams maximumParams(
                    1U, 1U, 1U, 8U, 8U, 0U);
                Max(maximum, dense, row, maskCount, repeatTime,
                    maximumParams);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (RowIsX1) {
                const BinaryRepeatParams compareParams(
                    1U, 1U, 1U, 8U, 8U, 8U);
                Compare(mask, maximum, dense, CMPMODE::EQ,
                        maskCount, repeatTime, compareParams);
            } else {
                const BinaryRepeatParams compareParams(
                    1U, 1U, 1U, 8U, 8U, 0U);
                Compare(mask, maximum, row, CMPMODE::EQ,
                        maskCount, repeatTime, compareParams);
            }
        } else if constexpr (RowIsX1) {
            const BinaryRepeatParams compareParams(
                1U, 1U, 1U, 8U, 0U, 8U);
            Compare(mask, row, dense, CMPMODE::GT,
                    maskCount, repeatTime, compareParams);
        } else {
            const BinaryRepeatParams compareParams(
                1U, 1U, 1U, 8U, 8U, 0U);
            Compare(mask, dense, row, CMPMODE::GT,
                    maskCount, repeatTime, compareParams);
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeStripe(
        LocalTensor<T> rowInput, LocalTensor<T> denseInput,
        uint32_t rows)
    {
        const uint32_t count = rows * kChunkElements;
        LocalTensor<ComputeT> row;
        LocalTensor<ComputeT> dense;
        if constexpr (ConversionStorage<T>::kBytesPerElement != 0U) {
            row = rowConvertedBuffer_.Get<ComputeT>();
            dense = denseConvertedBuffer_.Get<ComputeT>();
            Cast(row, rowInput, RoundMode::CAST_NONE, kChunkElements);
            Cast(dense, denseInput, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
        } else {
            row = rowInput;
            dense = denseInput;
        }

        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        CompareStripe<RowIsX1>(mask, row, dense, rows);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> yLocal =
            yQueue_.AllocTensor<uint8_t>();
        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected = selectedBuffer_.Get<half>();
        OutputSelector<T>::Run(
            selected, mask, selectionValues, count);
        PipeBarrier<PIPE_V>();
        Cast(yLocal.ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, count);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void CopyOutStripe(
        uint64_t outputOffset, uint32_t rows,
        uint32_t columnOffset, uint32_t validCols)
    {
        LocalTensor<uint8_t> yLocal =
            yQueue_.DeQue<uint8_t>();
        const uint32_t cols =
            static_cast<uint32_t>(tiling_->matrixCols);
        constexpr uint32_t kDataBlockBytes = 32U;
        const uint32_t alignedBytes =
            ((validCols + kDataBlockBytes - 1U) /
                kDataBlockBytes) * kDataBlockBytes;

        const uint32_t sourceStride =
            (kChunkElements - alignedBytes) / kDataBlockBytes;
        const DataCopyExtParams copy(
            static_cast<uint16_t>(rows), validCols,
            sourceStride, cols - validCols, 0U);
        DataCopyPad(yGlobal_[outputOffset + columnOffset], yLocal, copy);
        yQueue_.FreeTensor(yLocal);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        OutputSelector<T>::InitTensor(selectionValues, kStripeElements);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> cachedRow = rowBuffer_.Get<T>();
        uint64_t cachedOuter = ~static_cast<uint64_t>(0U);
        const uint32_t cols =
            static_cast<uint32_t>(tiling_->matrixCols);
        for (uint64_t relative = 0U; relative < tileCount; ++relative) {
            const uint64_t tile = beginTile + relative;
            const uint64_t outerIndex =
                tile / tiling_->matrixRowTiles;
            const uint32_t rowTile = static_cast<uint32_t>(
                tile - outerIndex * tiling_->matrixRowTiles);
            const uint32_t rowOffset =
                rowTile * tiling_->matrixRowsPerTile;
            const uint64_t remainingRows =
                tiling_->matrixRows - rowOffset;
            const uint32_t rows = static_cast<uint32_t>(
                remainingRows < tiling_->matrixRowsPerTile ?
                    remainingRows : tiling_->matrixRowsPerTile);
            const uint64_t outputOffset =
                outerIndex * tiling_->matrixRows * tiling_->matrixCols +
                static_cast<uint64_t>(rowOffset) * tiling_->matrixCols;

            if (outerIndex != cachedOuter) {
                CopyCachedRow<RowIsX1>(outerIndex, cachedRow);
                cachedOuter = outerIndex;
            }
            for (uint32_t columnOffset = 0U;
                 columnOffset < cols;
                 columnOffset += kChunkElements) {
                const uint32_t remainingCols = cols - columnOffset;
                const uint32_t validCols =
                    remainingCols < kChunkElements ?
                        remainingCols : kChunkElements;
                CopyDenseStripe<RowIsX1>(
                    outerIndex, rowOffset, rows,
                    columnOffset, validCols);
                LocalTensor<T> denseLocal =
                    denseQueue_.DeQue<T>();
                ComputeStripe<RowIsX1>(
                    cachedRow[columnOffset], denseLocal, rows);
                denseQueue_.FreeTensor(denseLocal);
                CopyOutStripe(
                    outputOffset, rows, columnOffset, validCols);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TQue<QuePosition::VECIN, kBufferNum> denseQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> yQueue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectedBuffer_;
    TBuf<TPosition::VECCALC> denseConvertedBuffer_;
    TBuf<TPosition::VECCALC> rowConvertedBuffer_;
    TBuf<TPosition::VECCALC> maximumBuffer_;
    GlobalTensor<T> x1Global_;
    GlobalTensor<T> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterTilingData* tiling_;
};


template <uint32_t SlotElements = 19200U>
class KernelGreaterAlignedFp32RowSplitPipeline {
public:
    static constexpr uint32_t kMaximumCols =
        S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32;
    static constexpr uint32_t kSelectRepeatElements = 128U;
    static constexpr uint32_t kUserUbBytes =
        kMaximumCols * sizeof(float) +
        2U * SlotElements * sizeof(float) +
        SlotElements * sizeof(uint8_t) +
        SlotElements / 8U +
        kSelectRepeatElements * sizeof(half) + 32U;
    static_assert(SlotElements % 256U == 0U,
                  "split FP32 slot must preserve aligned repeats");
    static_assert(SlotElements / kSelectRepeatElements <= 255U,
                  "split FP32 slot must fit one vector command");
    static_assert(kUserUbBytes <= 192U * 1024U - 256U,
                  "split FP32 pipeline exceeds Arch 2201 UB");

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const GreaterRowSingleTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        pipe_.InitBuffer(rowBuffer_, kMaximumCols * sizeof(float));
        pipe_.InitBuffer(selectionValueBuffer_,
                         kSelectRepeatElements * sizeof(half));
        pipe_.InitBuffer(inputQueue0_, 1U,
                         SlotElements * sizeof(float));
        pipe_.InitBuffer(inputQueue1_, 1U,
                         SlotElements * sizeof(float));
        pipe_.InitBuffer(selectionScalarBuffer_, 32U);
        pipe_.InitBuffer(outputQueue_, 1U,
                         SlotElements * sizeof(uint8_t));
        pipe_.InitBuffer(maskBuffer_, SlotElements / 8U);
    }

    __aicore__ inline void Process()
    {
        if (tiling_->matrixX1Pattern ==
            S9_GREATER_MATRIX_ROW_BROADCAST) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ inline void InitializeSelectionState()
    {
        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        LocalTensor<half> scalarValue =
            selectionScalarBuffer_.Get<half>();
        Duplicate(selectionValues, static_cast<half>(1),
                  kSelectRepeatElements);
        Duplicate(scalarValue, static_cast<half>(0), 16U);
        PipeBarrier<PIPE_V>();
        SetCmpMask(scalarValue);
        SetVectorMask<half>(
            static_cast<int32_t>(kSelectRepeatElements));
        PipeBarrier<PIPE_V>();
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyCachedRow(LocalTensor<float> row)
    {
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(cols * sizeof(float) / 32U),
            0U, 0U);
        if constexpr (RowIsX1) {
            DataCopy(row, x1Global_, copy);
        } else {
            DataCopy(row, x2Global_, copy);
        }
        const event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
    }

    template <bool QueueZero, bool RowIsX1>
    __aicore__ inline void Prefetch(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        const uint32_t inputBytes = rows * cols * sizeof(float);
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(inputBytes / 32U), 0U, 0U);
        if constexpr (QueueZero) {
            LocalTensor<float> dense =
                inputQueue0_.AllocTensor<float>();
            if constexpr (RowIsX1) {
                DataCopy(dense, x2Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            } else {
                DataCopy(dense, x1Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            }
            inputQueue0_.EnQue(dense);
        } else {
            LocalTensor<float> dense =
                inputQueue1_.AllocTensor<float>();
            if constexpr (RowIsX1) {
                DataCopy(dense, x2Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            } else {
                DataCopy(dense, x1Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            }
            inputQueue1_.EnQue(dense);
        }
    }

    template <bool QueueZero>
    __aicore__ inline LocalTensor<float> DeQueInput()
    {
        if constexpr (QueueZero) {
            return inputQueue0_.DeQue<float>();
        } else {
            return inputQueue1_.DeQue<float>();
        }
    }

    template <bool QueueZero>
    __aicore__ inline void FreeInput(LocalTensor<float> dense)
    {
        if constexpr (QueueZero) {
            inputQueue0_.FreeTensor(dense);
        } else {
            inputQueue1_.FreeTensor(dense);
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeRows(
        LocalTensor<float> row, LocalTensor<float> dense,
        uint32_t rows, uint32_t cols)
    {
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        const uint8_t compareRepeats = static_cast<uint8_t>(cols / 64U);
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(
            mask.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ float*>(
            row.GetPhyAddr());
        auto* rawDense = reinterpret_cast<__ubuf__ float*>(
            dense.GetPhyAddr());
#endif
        const BinaryRepeatParams compareParams(
            1U, 1U, 1U, 8U, 8U, 8U);
        for (uint32_t rowIndex = 0U; rowIndex < rows; ++rowIndex) {
            const uint32_t elementOffset = rowIndex * cols;
            const uint32_t maskOffset = elementOffset / 8U;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            if constexpr (RowIsX1) {
                Compare(mask[maskOffset], row, dense[elementOffset],
                        CMPMODE::GT, static_cast<uint64_t>(64U),
                        compareRepeats, compareParams);
            } else {
                Compare(mask[maskOffset], dense[elementOffset], row,
                        CMPMODE::GT, static_cast<uint64_t>(64U),
                        compareRepeats, compareParams);
            }
#else
            if constexpr (RowIsX1) {
                vcmpv_gt(rawMask + maskOffset, rawRow,
                         rawDense + elementOffset, compareRepeats,
                         1U, 1U, 1U, 8U, 8U, 8U);
            } else {
                vcmpv_gt(rawMask + maskOffset,
                         rawDense + elementOffset, rawRow,
                         compareRepeats, 1U, 1U, 1U,
                         8U, 8U, 8U);
            }
#endif
        }
        PipeBarrier<PIPE_V>();

        const uint32_t count = rows * cols;
        const uint8_t repeats = static_cast<uint8_t>(
            count / kSelectRepeatElements);
        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected = dense.ReinterpretCast<half>();
        const BinaryRepeatParams selectParams(
            1U, 1U, 1U, 8U, 0U, 0U);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Select(selected, mask, selectionValues, repeats, selectParams);
#else
        auto* rawSelected = reinterpret_cast<__ubuf__ half*>(
            selected.GetPhyAddr());
        auto* rawSelectionValues = reinterpret_cast<__ubuf__ half*>(
            selectionValues.GetPhyAddr());
        vsel(rawSelected, rawSelectionValues, rawMask,
             repeats, 1U, 1U, 1U, 8U, 0U, 0U,
             static_cast<uint8_t>(
                 SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> output =
            outputQueue_.AllocTensor<uint8_t>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, count);
#else
        vconv_f162s8r(
            reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            rawSelected, repeats, 1U, 1U, 4U, 8U);
#endif
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        LocalTensor<uint8_t> output =
            outputQueue_.DeQue<uint8_t>();
        const uint32_t outputBytes = rows * cols;
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(outputBytes / 32U), 0U, 0U);
        DataCopy(yGlobal_[static_cast<uint64_t>(rowOffset) * cols],
                 output, copy);
        outputQueue_.FreeTensor(output);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        InitializeSelectionState();
        LocalTensor<float> cachedRow = rowBuffer_.Get<float>();
        CopyCachedRow<RowIsX1>(cachedRow);

        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t rowsPerBatch = SlotElements / cols;
        const uint64_t beginRow64 =
            beginTile * tiling_->matrixRowsPerTile;
        const uint64_t endRow64 =
            (beginTile + tileCount) * tiling_->matrixRowsPerTile <
                    tiling_->matrixRows ?
                (beginTile + tileCount) * tiling_->matrixRowsPerTile :
                tiling_->matrixRows;
        uint32_t rowOffset = static_cast<uint32_t>(beginRow64);
        const uint32_t endRow = static_cast<uint32_t>(endRow64);
        uint32_t rows = endRow - rowOffset < rowsPerBatch ?
            endRow - rowOffset : rowsPerBatch;
        Prefetch<true, RowIsX1>(rowOffset, rows, cols);

        while (rowOffset < endRow) {
            uint32_t nextOffset = rowOffset + rows;
            uint32_t nextRows = nextOffset < endRow ?
                (endRow - nextOffset < rowsPerBatch ?
                    endRow - nextOffset : rowsPerBatch) : 0U;
            LocalTensor<float> dense = DeQueInput<true>();
            if (nextRows != 0U) {
                Prefetch<false, RowIsX1>(
                    nextOffset, nextRows, cols);
            }
            ComputeRows<RowIsX1>(cachedRow, dense, rows, cols);
            FreeInput<true>(dense);
            CopyOut(rowOffset, rows, cols);
            rowOffset = nextOffset;
            rows = nextRows;
            if (rowOffset >= endRow) {
                break;
            }

            nextOffset = rowOffset + rows;
            nextRows = nextOffset < endRow ?
                (endRow - nextOffset < rowsPerBatch ?
                    endRow - nextOffset : rowsPerBatch) : 0U;
            LocalTensor<float> dense1 = DeQueInput<false>();
            if (nextRows != 0U) {
                Prefetch<true, RowIsX1>(
                    nextOffset, nextRows, cols);
            }
            ComputeRows<RowIsX1>(cachedRow, dense1, rows, cols);
            FreeInput<false>(dense1);
            CopyOut(rowOffset, rows, cols);
            rowOffset = nextOffset;
            rows = nextRows;
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TQue<QuePosition::VECIN, 1> inputQueue0_;
    TQue<QuePosition::VECIN, 1> inputQueue1_;
    TBuf<TPosition::VECCALC> selectionScalarBuffer_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    GlobalTensor<float> x1Global_;
    GlobalTensor<float> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterRowSingleTilingData* tiling_;
};

template <uint32_t SlotElements = 30720U>
class KernelGreaterAlignedFp16RowSplitExact {
public:
    static constexpr uint32_t kMaximumCols =
        S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32;
    static constexpr uint32_t kRepeatElements = 128U;
    static constexpr uint32_t kUserUbBytes =
        kMaximumCols * sizeof(half) + 256U +
        2U * SlotElements * sizeof(half) + 32U +
        2U * SlotElements * sizeof(uint8_t);
    static_assert(SlotElements % 256U == 0U,
                  "split FP16 slot must preserve aligned rows");
    static_assert(SlotElements / kRepeatElements <= 255U,
                  "FP16 batch must fit one VSEL/VCONV command");
    static_assert(kUserUbBytes <= 192U * 1024U - 256U,
                  "split exact FP16 pipeline exceeds Arch 2201 UB");

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const GreaterRowSingleTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        pipe_.InitBuffer(rowBuffer_, kMaximumCols * sizeof(half));
        pipe_.InitBuffer(selectionValueBuffer_,
                         kRepeatElements * sizeof(half));
        pipe_.InitBuffer(inputQueue0_, 1U,
                         SlotElements * sizeof(half));
        pipe_.InitBuffer(inputQueue1_, 1U,
                         SlotElements * sizeof(half));
        pipe_.InitBuffer(selectionScalarBuffer_, 32U);
        pipe_.InitBuffer(outputQueue_, 2U,
                         SlotElements * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        if (tiling_->matrixX1Pattern ==
            S9_GREATER_MATRIX_ROW_BROADCAST) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ inline void InitializeVectorState()
    {
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        LocalTensor<half> scalar = selectionScalarBuffer_.Get<half>();
        Duplicate(values, static_cast<half>(1), kRepeatElements);
        Duplicate(scalar, static_cast<half>(0), 16U);
        PipeBarrier<PIPE_V>();
        SetCmpMask(scalar);
        SetVectorMask<half>(static_cast<int32_t>(kRepeatElements));
        PipeBarrier<PIPE_V>();
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyCachedRow(LocalTensor<half> row)
    {
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(cols * sizeof(half) / 32U),
            0U, 0U);
        if constexpr (RowIsX1) {
            DataCopy(row, x1Global_, copy);
        } else {
            DataCopy(row, x2Global_, copy);
        }
        const event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
    }

    template <bool QueueZero, bool RowIsX1>
    __aicore__ inline void Prefetch(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        const uint32_t inputBytes = rows * cols * sizeof(half);
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(inputBytes / 32U), 0U, 0U);
        if constexpr (QueueZero) {
            LocalTensor<half> dense = inputQueue0_.AllocTensor<half>();
            if constexpr (RowIsX1) {
                DataCopy(dense, x2Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            } else {
                DataCopy(dense, x1Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            }
            inputQueue0_.EnQue(dense);
        } else {
            LocalTensor<half> dense = inputQueue1_.AllocTensor<half>();
            if constexpr (RowIsX1) {
                DataCopy(dense, x2Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            } else {
                DataCopy(dense, x1Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            }
            inputQueue1_.EnQue(dense);
        }
    }

    template <bool QueueZero>
    __aicore__ inline LocalTensor<half> DeQueInput()
    {
        if constexpr (QueueZero) {
            return inputQueue0_.DeQue<half>();
        } else {
            return inputQueue1_.DeQue<half>();
        }
    }

    template <bool QueueZero>
    __aicore__ inline void FreeInput(LocalTensor<half> dense)
    {
        if constexpr (QueueZero) {
            inputQueue0_.FreeTensor(dense);
        } else {
            inputQueue1_.FreeTensor(dense);
        }
    }

    __aicore__ inline void ConvertToByte(
        LocalTensor<uint8_t> output, LocalTensor<half> dense,
        uint32_t count)
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.ReinterpretCast<int8_t>(), dense,
             RoundMode::CAST_RINT, count);
#else
        auto* rawOutput = reinterpret_cast<__ubuf__ int8_t*>(
            output.GetPhyAddr());
        auto* rawDense = reinterpret_cast<__ubuf__ half*>(
            dense.GetPhyAddr());
        const uint8_t repeats = static_cast<uint8_t>(
            count / kRepeatElements);
        vconv_f162s8r(rawOutput, rawDense, repeats,
                      1U, 1U, 4U, 8U);
#endif
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeRows(
        LocalTensor<half> row, LocalTensor<half> dense,
        uint32_t rows, uint32_t cols)
    {

        LocalTensor<uint8_t> output =
            outputQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = output;
        const uint8_t compareRepeats = static_cast<uint8_t>(
            cols / kRepeatElements);
        const BinaryRepeatParams compareParams(
            1U, 1U, 1U, 8U, 8U, 8U);
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(
            mask.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ half*>(
            row.GetPhyAddr());
        auto* rawDense = reinterpret_cast<__ubuf__ half*>(
            dense.GetPhyAddr());
#endif
        for (uint32_t rowIndex = 0U; rowIndex < rows; ++rowIndex) {
            const uint32_t offset = rowIndex * cols;
            const uint32_t maskOffset = offset / 8U;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            if constexpr (RowIsX1) {
                Compare(mask[maskOffset], row, dense[offset],
                        CMPMODE::GT,
                        static_cast<uint64_t>(kRepeatElements),
                        compareRepeats, compareParams);
            } else {
                Compare(mask[maskOffset], dense[offset], row,
                        CMPMODE::GT,
                        static_cast<uint64_t>(kRepeatElements),
                        compareRepeats, compareParams);
            }
#else
            if constexpr (RowIsX1) {
                vcmpv_gt(rawMask + maskOffset, rawRow,
                         rawDense + offset, compareRepeats,
                         1U, 1U, 1U, 8U, 8U, 8U);
            } else {
                vcmpv_gt(rawMask + maskOffset,
                         rawDense + offset, rawRow,
                         compareRepeats, 1U, 1U, 1U,
                         8U, 8U, 8U);
            }
#endif
        }
        PipeBarrier<PIPE_V>();

        const uint32_t count = rows * cols;
        const uint8_t repeats = static_cast<uint8_t>(
            count / kRepeatElements);
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        const BinaryRepeatParams selectParams(
            1U, 1U, 1U, 8U, 0U, 0U);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Select(dense, mask, values, repeats, selectParams);
#else
        auto* rawValues = reinterpret_cast<__ubuf__ half*>(
            values.GetPhyAddr());
        vsel(rawDense, rawValues, rawMask, repeats,
             1U, 1U, 1U, 8U, 0U, 0U,
             static_cast<uint8_t>(
                 SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
        PipeBarrier<PIPE_V>();

        ConvertToByte(output, dense, count);
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        LocalTensor<uint8_t> output = outputQueue_.DeQue<uint8_t>();
        const uint32_t outputBytes = rows * cols;
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(outputBytes / 32U), 0U, 0U);
        DataCopy(yGlobal_[static_cast<uint64_t>(rowOffset) * cols],
                 output, copy);
        outputQueue_.FreeTensor(output);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        InitializeVectorState();
        LocalTensor<half> cachedRow = rowBuffer_.Get<half>();
        CopyCachedRow<RowIsX1>(cachedRow);

        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t rowsPerBatch = SlotElements / cols;
        const uint64_t beginRow64 =
            beginTile * tiling_->matrixRowsPerTile;
        const uint64_t endRow64 =
            (beginTile + tileCount) * tiling_->matrixRowsPerTile <
                    tiling_->matrixRows ?
                (beginTile + tileCount) * tiling_->matrixRowsPerTile :
                tiling_->matrixRows;
        uint32_t rowOffset = static_cast<uint32_t>(beginRow64);
        const uint32_t endRow = static_cast<uint32_t>(endRow64);
        uint32_t rows = endRow - rowOffset < rowsPerBatch ?
            endRow - rowOffset : rowsPerBatch;
        Prefetch<true, RowIsX1>(rowOffset, rows, cols);

        while (rowOffset < endRow) {
            uint32_t nextOffset = rowOffset + rows;
            uint32_t nextRows = nextOffset < endRow ?
                (endRow - nextOffset < rowsPerBatch ?
                    endRow - nextOffset : rowsPerBatch) : 0U;
            LocalTensor<half> dense = DeQueInput<true>();
            if (nextRows != 0U) {
                Prefetch<false, RowIsX1>(nextOffset, nextRows, cols);
            }
            ComputeRows<RowIsX1>(cachedRow, dense, rows, cols);
            FreeInput<true>(dense);
            CopyOut(rowOffset, rows, cols);
            rowOffset = nextOffset;
            rows = nextRows;
            if (rowOffset >= endRow) {
                break;
            }

            nextOffset = rowOffset + rows;
            nextRows = nextOffset < endRow ?
                (endRow - nextOffset < rowsPerBatch ?
                    endRow - nextOffset : rowsPerBatch) : 0U;
            LocalTensor<half> dense1 = DeQueInput<false>();
            if (nextRows != 0U) {
                Prefetch<true, RowIsX1>(nextOffset, nextRows, cols);
            }
            ComputeRows<RowIsX1>(cachedRow, dense1, rows, cols);
            FreeInput<false>(dense1);
            CopyOut(rowOffset, rows, cols);
            rowOffset = nextOffset;
            rows = nextRows;
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TQue<QuePosition::VECIN, 1> inputQueue0_;
    TQue<QuePosition::VECIN, 1> inputQueue1_;
    TBuf<TPosition::VECCALC> selectionScalarBuffer_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    GlobalTensor<half> x1Global_;
    GlobalTensor<half> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterRowSingleTilingData* tiling_;
};

template <uint32_t SlotElements = 29952U>
class KernelGreaterAlignedInt8RowStagedPipeline {
public:
    static constexpr uint32_t kMaximumCols =
        S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32;
    static constexpr uint32_t kRepeatElements = 128U;
    static constexpr uint32_t kUserUbBytes =
        kMaximumCols * (sizeof(int8_t) + sizeof(half)) +
        2U * SlotElements * sizeof(int8_t) + 32U +
        SlotElements * sizeof(half) + 32U +
        2U * SlotElements * sizeof(uint8_t);
    static_assert(SlotElements % 256U == 0U,
                  "staged INT8 slot must preserve aligned rows");
    static_assert(SlotElements / kRepeatElements <= 255U,
                  "staged INT8 batch must fit one dense command");
    static_assert(kUserUbBytes <= 192U * 1024U - 256U,
                  "staged INT8 pipeline exceeds Arch 2201 UB");

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const GreaterRowSingleTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        pipe_.InitBuffer(rowInputBuffer_,
                         kMaximumCols * sizeof(int8_t));
        pipe_.InitBuffer(rowComputeBuffer_,
                         kMaximumCols * sizeof(half));
        pipe_.InitBuffer(inputQueue_, 2U,
                         SlotElements * sizeof(int8_t));
        pipe_.InitBuffer(computeBankPad_, 32U);
        pipe_.InitBuffer(denseComputeBuffer_,
                         SlotElements * sizeof(half));
        pipe_.InitBuffer(outputBankPad_, 32U);
        pipe_.InitBuffer(outputQueue_, 2U,
                         SlotElements * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        if (tiling_->matrixX1Pattern ==
            S9_GREATER_MATRIX_ROW_BROADCAST) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ inline void InitializeVectorState()
    {
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
        SetVectorMask<half>(static_cast<int32_t>(kRepeatElements));
#endif
    }

    template <bool RowIsX1>
    __aicore__ inline LocalTensor<half> CopyAndConvertCachedRow()
    {
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        LocalTensor<int8_t> rowInput = rowInputBuffer_.Get<int8_t>();
        LocalTensor<half> rowCompute = rowComputeBuffer_.Get<half>();
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(cols * sizeof(int8_t) / 32U),
            0U, 0U);
        if constexpr (RowIsX1) {
            DataCopy(rowInput, x1Global_, copy);
        } else {
            DataCopy(rowInput, x2Global_, copy);
        }
        const event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(rowCompute, rowInput, RoundMode::CAST_NONE, cols);
#else
        const uint8_t repeats = static_cast<uint8_t>(
            cols / kRepeatElements);
        vconv_s82f16(
            reinterpret_cast<__ubuf__ half*>(rowCompute.GetPhyAddr()),
            reinterpret_cast<__ubuf__ int8_t*>(rowInput.GetPhyAddr()),
            repeats, 1U, 1U, 8U, 4U);
#endif
        PipeBarrier<PIPE_V>();
        return rowCompute;
    }

    template <bool RowIsX1>
    __aicore__ inline void Prefetch(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        LocalTensor<int8_t> input = inputQueue_.AllocTensor<int8_t>();
        const uint32_t bytes = rows * cols * sizeof(int8_t);
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(bytes / 32U), 0U, 0U);
        if constexpr (RowIsX1) {
            DataCopy(input, x2Global_[
                static_cast<uint64_t>(rowOffset) * cols], copy);
        } else {
            DataCopy(input, x1Global_[
                static_cast<uint64_t>(rowOffset) * cols], copy);
        }
        inputQueue_.EnQue(input);
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeRows(
        LocalTensor<half> row, LocalTensor<int8_t> input,
        uint32_t rows, uint32_t cols)
    {
        const uint32_t count = rows * cols;
        const uint8_t repeats = static_cast<uint8_t>(
            count / kRepeatElements);
        LocalTensor<half> dense = denseComputeBuffer_.Get<half>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(dense, input, RoundMode::CAST_NONE, count);
#else
        auto* rawDense = reinterpret_cast<__ubuf__ half*>(
            dense.GetPhyAddr());
        vconv_s82f16(
            rawDense,
            reinterpret_cast<__ubuf__ int8_t*>(input.GetPhyAddr()),
            repeats, 1U, 1U, 8U, 4U);
#endif
        PipeBarrier<PIPE_V>();

        const uint8_t rowRepeats = static_cast<uint8_t>(
            cols / kRepeatElements);
        const BinaryRepeatParams subtractParams(
            1U, 1U, 1U, 8U, 8U, 8U);
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
        auto* rawRow = reinterpret_cast<__ubuf__ half*>(
            row.GetPhyAddr());
#endif
        for (uint32_t rowIndex = 0U; rowIndex < rows; ++rowIndex) {
            const uint32_t offset = rowIndex * cols;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            if constexpr (RowIsX1) {
                SubRelu(dense[offset], row, dense[offset],
                        static_cast<uint64_t>(kRepeatElements),
                        rowRepeats, subtractParams);
            } else {
                SubRelu(dense[offset], dense[offset], row,
                        static_cast<uint64_t>(kRepeatElements),
                        rowRepeats, subtractParams);
            }
#else
            if constexpr (RowIsX1) {
                vsubrelu(rawDense + offset, rawRow,
                         rawDense + offset, rowRepeats,
                         1U, 1U, 1U, 8U, 8U, 8U);
            } else {
                vsubrelu(rawDense + offset, rawDense + offset,
                         rawRow, rowRepeats,
                         1U, 1U, 1U, 8U, 8U, 8U);
            }
#endif
        }
        PipeBarrier<PIPE_V>();

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        const UnaryRepeatParams minimumParams(1U, 1U, 8U, 8U);
        Mins(dense, dense, static_cast<half>(1),
             static_cast<uint64_t>(kRepeatElements),
             repeats, minimumParams);
#else
        vmins(rawDense, rawDense, static_cast<half>(1),
              repeats, 1U, 1U, 8U, 8U, false, false);
#endif
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> output =
            outputQueue_.AllocTensor<uint8_t>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.ReinterpretCast<int8_t>(), dense,
             RoundMode::CAST_CEIL, count);
#else
        vconv_f162s8c(
            reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            rawDense, repeats, 1U, 1U, 4U, 8U);
#endif
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        LocalTensor<uint8_t> output = outputQueue_.DeQue<uint8_t>();
        const uint32_t bytes = rows * cols;
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(bytes / 32U), 0U, 0U);
        DataCopy(yGlobal_[static_cast<uint64_t>(rowOffset) * cols],
                 output, copy);
        outputQueue_.FreeTensor(output);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        InitializeVectorState();
        LocalTensor<half> cachedRow =
            CopyAndConvertCachedRow<RowIsX1>();
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t rowsPerBatch = SlotElements / cols;
        const uint64_t beginRow64 =
            beginTile * tiling_->matrixRowsPerTile;
        const uint64_t endRow64 =
            (beginTile + tileCount) * tiling_->matrixRowsPerTile <
                    tiling_->matrixRows ?
                (beginTile + tileCount) * tiling_->matrixRowsPerTile :
                tiling_->matrixRows;
        uint32_t rowOffset = static_cast<uint32_t>(beginRow64);
        const uint32_t endRow = static_cast<uint32_t>(endRow64);
        uint32_t rows = endRow - rowOffset < rowsPerBatch ?
            endRow - rowOffset : rowsPerBatch;
        Prefetch<RowIsX1>(rowOffset, rows, cols);

        while (rowOffset < endRow) {
            const uint32_t nextOffset = rowOffset + rows;
            const uint32_t nextRows = nextOffset < endRow ?
                (endRow - nextOffset < rowsPerBatch ?
                    endRow - nextOffset : rowsPerBatch) : 0U;
            LocalTensor<int8_t> input = inputQueue_.DeQue<int8_t>();
            if (nextRows != 0U) {
                Prefetch<RowIsX1>(nextOffset, nextRows, cols);
            }
            ComputeRows<RowIsX1>(cachedRow, input, rows, cols);
            inputQueue_.FreeTensor(input);
            CopyOut(rowOffset, rows, cols);
            rowOffset = nextOffset;
            rows = nextRows;
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowInputBuffer_;
    TBuf<TPosition::VECCALC> rowComputeBuffer_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TBuf<TPosition::VECCALC> computeBankPad_;
    TBuf<TPosition::VECCALC> denseComputeBuffer_;
    TBuf<TPosition::VECCALC> outputBankPad_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    GlobalTensor<int8_t> x1Global_;
    GlobalTensor<int8_t> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterRowSingleTilingData* tiling_;
};

template <uint32_t SlotElements = 16384U>
class KernelGreaterAlignedBf16RowStagedPipeline {
public:
    static constexpr uint32_t kMaximumCols =
        S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32;
    static constexpr uint32_t kFloatRepeatElements = 64U;
    static constexpr uint32_t kSelectRepeatElements = 128U;
    static constexpr uint32_t kMaximumConvertElements =
        255U * kFloatRepeatElements;
    static constexpr uint32_t kUserUbBytes =
        kMaximumCols * (sizeof(bfloat16_t) + sizeof(float)) +
        kSelectRepeatElements * sizeof(half) +
        2U * SlotElements * sizeof(bfloat16_t) + 32U +
        SlotElements * sizeof(float) + 32U +
        2U * SlotElements * sizeof(uint8_t);
    static_assert(SlotElements % 256U == 0U,
                  "staged BF16 slot must preserve aligned rows");
    static_assert(SlotElements == kMaximumConvertElements + 64U,
                  "BF16 staging expects one 64-element conversion tail");
    static_assert(SlotElements / kSelectRepeatElements <= 255U,
                  "staged BF16 Select must fit one command");
    static_assert(kUserUbBytes <= 192U * 1024U - 256U,
                  "staged BF16 pipeline exceeds Arch 2201 UB");

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const GreaterRowSingleTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(x1));
        x2Global_.SetGlobalBuffer(
            reinterpret_cast<__gm__ bfloat16_t*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        pipe_.InitBuffer(rowInputBuffer_,
                         kMaximumCols * sizeof(bfloat16_t));
        pipe_.InitBuffer(rowComputeBuffer_,
                         kMaximumCols * sizeof(float));
        pipe_.InitBuffer(selectionValueBuffer_,
                         kSelectRepeatElements * sizeof(half));
        pipe_.InitBuffer(inputQueue_, 2U,
                         SlotElements * sizeof(bfloat16_t));
        pipe_.InitBuffer(computeBankPad_, 32U);
        pipe_.InitBuffer(denseComputeBuffer_,
                         SlotElements * sizeof(float));
        pipe_.InitBuffer(selectionScalarBuffer_, 32U);
        pipe_.InitBuffer(outputQueue_, 2U,
                         SlotElements * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        if (tiling_->matrixX1Pattern ==
            S9_GREATER_MATRIX_ROW_BROADCAST) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ inline void InitializeVectorState()
    {
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        LocalTensor<half> scalar = selectionScalarBuffer_.Get<half>();
        Duplicate(values, static_cast<half>(1), kSelectRepeatElements);
        Duplicate(scalar, static_cast<half>(0), 16U);
        PipeBarrier<PIPE_V>();
        SetCmpMask(scalar);
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1

        SetVectorMask<half>(
            static_cast<int32_t>(kSelectRepeatElements));
#endif
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ConvertBf16ToFloat(
        LocalTensor<float> dst, LocalTensor<bfloat16_t> src,
        uint32_t count)
    {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(dst, src, RoundMode::CAST_NONE, count);
#else
        auto* rawDst = reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr());
        auto* rawSrc = reinterpret_cast<__ubuf__ bfloat16_t*>(
            src.GetPhyAddr());
        const uint32_t totalRepeats = count / kFloatRepeatElements;
        const uint8_t firstRepeats = static_cast<uint8_t>(
            totalRepeats > 255U ? 255U : totalRepeats);
        vconv_bf162f32(rawDst, rawSrc, firstRepeats,
                       1U, 1U, 8U, 4U);
        if (totalRepeats > 255U) {
            vconv_bf162f32(
                rawDst + kMaximumConvertElements,
                rawSrc + kMaximumConvertElements,
                static_cast<uint8_t>(totalRepeats - 255U),
                1U, 1U, 8U, 4U);
        }
#endif
    }

    template <bool RowIsX1>
    __aicore__ inline LocalTensor<float> CopyAndConvertCachedRow()
    {
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        LocalTensor<bfloat16_t> rowInput =
            rowInputBuffer_.Get<bfloat16_t>();
        LocalTensor<float> rowCompute = rowComputeBuffer_.Get<float>();
        const DataCopyParams copy(
            1U,
            static_cast<uint16_t>(cols * sizeof(bfloat16_t) / 32U),
            0U, 0U);
        if constexpr (RowIsX1) {
            DataCopy(rowInput, x1Global_, copy);
        } else {
            DataCopy(rowInput, x2Global_, copy);
        }
        const event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
        ConvertBf16ToFloat(rowCompute, rowInput, cols);
        PipeBarrier<PIPE_V>();
        return rowCompute;
    }

    template <bool RowIsX1>
    __aicore__ inline void Prefetch(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        LocalTensor<bfloat16_t> input =
            inputQueue_.AllocTensor<bfloat16_t>();
        const uint32_t bytes = rows * cols * sizeof(bfloat16_t);
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(bytes / 32U), 0U, 0U);
        if constexpr (RowIsX1) {
            DataCopy(input, x2Global_[
                static_cast<uint64_t>(rowOffset) * cols], copy);
        } else {
            DataCopy(input, x1Global_[
                static_cast<uint64_t>(rowOffset) * cols], copy);
        }
        inputQueue_.EnQue(input);
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeRows(
        LocalTensor<float> row, LocalTensor<bfloat16_t> input,
        uint32_t rows, uint32_t cols)
    {
        const uint32_t count = rows * cols;
        LocalTensor<float> dense = denseComputeBuffer_.Get<float>();
        ConvertBf16ToFloat(dense, input, count);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> output =
            outputQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = output;
        const uint8_t compareRepeats = static_cast<uint8_t>(
            cols / kFloatRepeatElements);
        const BinaryRepeatParams compareParams(
            1U, 1U, 1U, 8U, 8U, 8U);
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(
            mask.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ float*>(
            row.GetPhyAddr());
        auto* rawDense = reinterpret_cast<__ubuf__ float*>(
            dense.GetPhyAddr());
#endif
        for (uint32_t rowIndex = 0U; rowIndex < rows; ++rowIndex) {
            const uint32_t offset = rowIndex * cols;
            const uint32_t maskOffset = offset / 8U;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            if constexpr (RowIsX1) {
                Compare(mask[maskOffset], row, dense[offset],
                        CMPMODE::GT,
                        static_cast<uint64_t>(kFloatRepeatElements),
                        compareRepeats, compareParams);
            } else {
                Compare(mask[maskOffset], dense[offset], row,
                        CMPMODE::GT,
                        static_cast<uint64_t>(kFloatRepeatElements),
                        compareRepeats, compareParams);
            }
#else
            if constexpr (RowIsX1) {
                vcmpv_gt(rawMask + maskOffset, rawRow,
                         rawDense + offset, compareRepeats,
                         1U, 1U, 1U, 8U, 8U, 8U);
            } else {
                vcmpv_gt(rawMask + maskOffset,
                         rawDense + offset, rawRow,
                         compareRepeats, 1U, 1U, 1U,
                         8U, 8U, 8U);
            }
#endif
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<half> selected = input.ReinterpretCast<half>();
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        const uint8_t selectRepeats = static_cast<uint8_t>(
            count / kSelectRepeatElements);
        const BinaryRepeatParams selectParams(
            1U, 1U, 1U, 8U, 0U, 0U);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Select(selected, mask, values, selectRepeats, selectParams);
#else
        auto* rawSelected = reinterpret_cast<__ubuf__ half*>(
            selected.GetPhyAddr());
        auto* rawValues = reinterpret_cast<__ubuf__ half*>(
            values.GetPhyAddr());
        vsel(rawSelected, rawValues, rawMask, selectRepeats,
             1U, 1U, 1U, 8U, 0U, 0U,
             static_cast<uint8_t>(
                 SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
        PipeBarrier<PIPE_V>();

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, count);
#else
        vconv_f162s8r(
            reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            rawSelected, selectRepeats, 1U, 1U, 4U, 8U);
#endif
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        LocalTensor<uint8_t> output = outputQueue_.DeQue<uint8_t>();
        const uint32_t bytes = rows * cols;
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(bytes / 32U), 0U, 0U);
        DataCopy(yGlobal_[static_cast<uint64_t>(rowOffset) * cols],
                 output, copy);
        outputQueue_.FreeTensor(output);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        InitializeVectorState();
        LocalTensor<float> cachedRow =
            CopyAndConvertCachedRow<RowIsX1>();
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t rowsPerBatch = SlotElements / cols;
        const uint64_t beginRow64 =
            beginTile * tiling_->matrixRowsPerTile;
        const uint64_t endRow64 =
            (beginTile + tileCount) * tiling_->matrixRowsPerTile <
                    tiling_->matrixRows ?
                (beginTile + tileCount) * tiling_->matrixRowsPerTile :
                tiling_->matrixRows;
        uint32_t rowOffset = static_cast<uint32_t>(beginRow64);
        const uint32_t endRow = static_cast<uint32_t>(endRow64);
        uint32_t rows = endRow - rowOffset < rowsPerBatch ?
            endRow - rowOffset : rowsPerBatch;
        Prefetch<RowIsX1>(rowOffset, rows, cols);

        while (rowOffset < endRow) {
            const uint32_t nextOffset = rowOffset + rows;
            const uint32_t nextRows = nextOffset < endRow ?
                (endRow - nextOffset < rowsPerBatch ?
                    endRow - nextOffset : rowsPerBatch) : 0U;
            LocalTensor<bfloat16_t> input =
                inputQueue_.DeQue<bfloat16_t>();
            if (nextRows != 0U) {
                Prefetch<RowIsX1>(nextOffset, nextRows, cols);
            }
            ComputeRows<RowIsX1>(cachedRow, input, rows, cols);
            inputQueue_.FreeTensor(input);
            CopyOut(rowOffset, rows, cols);
            rowOffset = nextOffset;
            rows = nextRows;
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowInputBuffer_;
    TBuf<TPosition::VECCALC> rowComputeBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TBuf<TPosition::VECCALC> computeBankPad_;
    TBuf<TPosition::VECCALC> denseComputeBuffer_;
    TBuf<TPosition::VECCALC> selectionScalarBuffer_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    GlobalTensor<bfloat16_t> x1Global_;
    GlobalTensor<bfloat16_t> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterRowSingleTilingData* tiling_;
};

template <uint32_t SlotElements = 17408U>
class KernelGreaterAlignedInt32RowDestructivePipeline {
public:
    static constexpr uint32_t kMaximumCols =
        S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32;
    static constexpr uint32_t kComputeRepeatElements = 64U;
    static constexpr uint32_t kSelectRepeatElements = 128U;
    static constexpr uint32_t kInputSlotBytes =
        SlotElements * sizeof(int32_t) + 32U;
    static constexpr uint32_t kOutputSlotBytes =
        SlotElements * sizeof(uint8_t) + 32U;
    static constexpr uint32_t kUserUbBytes =
        kMaximumCols * sizeof(int32_t) +
        kSelectRepeatElements * sizeof(half) + 32U +
        2U * kInputSlotBytes + 2U * kOutputSlotBytes;
    static_assert(SlotElements % 256U == 0U,
                  "destructive INT32 slot must preserve aligned rows");
    static_assert(SlotElements / kSelectRepeatElements <= 255U,
                  "destructive INT32 Select must fit one command");
    static_assert(kUserUbBytes <= 192U * 1024U - 256U,
                  "destructive INT32 pipeline exceeds Arch 2201 UB");

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const GreaterRowSingleTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        pipe_.InitBuffer(rowBuffer_,
                         kMaximumCols * sizeof(int32_t));
        pipe_.InitBuffer(selectionValueBuffer_,
                         kSelectRepeatElements * sizeof(half));
        pipe_.InitBuffer(selectionScalarBuffer_, 32U);
        pipe_.InitBuffer(inputQueue_, 2U, kInputSlotBytes);
        pipe_.InitBuffer(outputQueue_, 2U, kOutputSlotBytes);
    }

    __aicore__ inline void Process()
    {
        if (tiling_->matrixX1Pattern ==
            S9_GREATER_MATRIX_ROW_BROADCAST) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ inline void InitializeVectorState()
    {
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        LocalTensor<half> scalar = selectionScalarBuffer_.Get<half>();
        Duplicate(values, static_cast<half>(0), kSelectRepeatElements);
        Duplicate(scalar, static_cast<half>(1), 16U);
        PipeBarrier<PIPE_V>();
        SetCmpMask(scalar);
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1

        SetVectorMask<half>(
            static_cast<int32_t>(kSelectRepeatElements));
#endif
        PipeBarrier<PIPE_V>();
    }

    template <bool RowIsX1>
    __aicore__ inline LocalTensor<int32_t> CopyCachedRow()
    {
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        LocalTensor<int32_t> row = rowBuffer_.Get<int32_t>();
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(cols * sizeof(int32_t) / 32U),
            0U, 0U);
        if constexpr (RowIsX1) {
            DataCopy(row, x1Global_, copy);
        } else {
            DataCopy(row, x2Global_, copy);
        }
        const event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
        return row;
    }

    template <bool RowIsX1>
    __aicore__ inline void Prefetch(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        LocalTensor<int32_t> input = inputQueue_.AllocTensor<int32_t>();
        const uint32_t bytes = rows * cols * sizeof(int32_t);
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(bytes / 32U), 0U, 0U);
        if constexpr (RowIsX1) {
            DataCopy(input, x2Global_[
                static_cast<uint64_t>(rowOffset) * cols], copy);
        } else {
            DataCopy(input, x1Global_[
                static_cast<uint64_t>(rowOffset) * cols], copy);
        }
        inputQueue_.EnQue(input);
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeRows(
        LocalTensor<int32_t> row, LocalTensor<int32_t> dense,
        uint32_t rows, uint32_t cols)
    {
        const uint32_t count = rows * cols;
        const uint8_t compareRepeats = static_cast<uint8_t>(
            cols / kComputeRepeatElements);
        const BinaryRepeatParams binaryParams(
            1U, 1U, 1U, 8U, 8U, 8U);
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
        auto* rawDense = reinterpret_cast<__ubuf__ int32_t*>(
            dense.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ int32_t*>(
            row.GetPhyAddr());
#endif
        for (uint32_t rowIndex = 0U; rowIndex < rows; ++rowIndex) {
            const uint32_t offset = rowIndex * cols;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            if constexpr (RowIsX1) {
                Min(dense[offset], dense[offset], row,
                    static_cast<uint64_t>(kComputeRepeatElements),
                    compareRepeats, binaryParams);
            } else {
                Max(dense[offset], dense[offset], row,
                    static_cast<uint64_t>(kComputeRepeatElements),
                    compareRepeats, binaryParams);
            }
#else
            if constexpr (RowIsX1) {
                vmin(rawDense + offset, rawDense + offset, rawRow,
                     compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
            } else {
                vmax(rawDense + offset, rawDense + offset, rawRow,
                     compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
            }
#endif
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> output =
            outputQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = output;
        const BinaryRepeatParams compareParams(
            1U, 1U, 1U, 8U, 8U, 8U);
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(
            mask.GetPhyAddr());
#endif
        for (uint32_t rowIndex = 0U; rowIndex < rows; ++rowIndex) {
            const uint32_t offset = rowIndex * cols;
            const uint32_t maskOffset = offset / 8U;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            Compare(mask[maskOffset], dense[offset], row, CMPMODE::EQ,
                    static_cast<uint64_t>(kComputeRepeatElements),
                    compareRepeats, compareParams);
#else
            vcmpv_eq(rawMask + maskOffset, rawDense + offset, rawRow,
                     compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
#endif
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<half> selected = dense.ReinterpretCast<half>();
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        const uint8_t selectRepeats = static_cast<uint8_t>(
            count / kSelectRepeatElements);
        const BinaryRepeatParams selectParams(
            1U, 1U, 1U, 8U, 0U, 0U);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1

        SetVectorMask<half>(
            static_cast<int32_t>(kSelectRepeatElements));
        Select(selected, mask, values, selectRepeats, selectParams);
#else
        auto* rawSelected = reinterpret_cast<__ubuf__ half*>(
            selected.GetPhyAddr());
        auto* rawValues = reinterpret_cast<__ubuf__ half*>(
            values.GetPhyAddr());
        vsel(rawSelected, rawValues, rawMask, selectRepeats,
             1U, 1U, 1U, 8U, 0U, 0U,
             static_cast<uint8_t>(
                 SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
        PipeBarrier<PIPE_V>();

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, count);
#else
        vconv_f162s8r(
            reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            rawSelected, selectRepeats, 1U, 1U, 4U, 8U);
#endif
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(
        uint32_t rowOffset, uint32_t rows, uint32_t cols)
    {
        LocalTensor<uint8_t> output = outputQueue_.DeQue<uint8_t>();
        const uint32_t bytes = rows * cols;
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(bytes / 32U), 0U, 0U);
        DataCopy(yGlobal_[static_cast<uint64_t>(rowOffset) * cols],
                 output, copy);
        outputQueue_.FreeTensor(output);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        InitializeVectorState();
        LocalTensor<int32_t> cachedRow = CopyCachedRow<RowIsX1>();
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t rowsPerBatch = SlotElements / cols;
        const uint64_t beginRow64 =
            beginTile * tiling_->matrixRowsPerTile;
        const uint64_t endRow64 =
            (beginTile + tileCount) * tiling_->matrixRowsPerTile <
                    tiling_->matrixRows ?
                (beginTile + tileCount) * tiling_->matrixRowsPerTile :
                tiling_->matrixRows;
        uint32_t rowOffset = static_cast<uint32_t>(beginRow64);
        const uint32_t endRow = static_cast<uint32_t>(endRow64);
        uint32_t rows = endRow - rowOffset < rowsPerBatch ?
            endRow - rowOffset : rowsPerBatch;
        Prefetch<RowIsX1>(rowOffset, rows, cols);

        while (rowOffset < endRow) {
            const uint32_t nextOffset = rowOffset + rows;
            const uint32_t nextRows = nextOffset < endRow ?
                (endRow - nextOffset < rowsPerBatch ?
                    endRow - nextOffset : rowsPerBatch) : 0U;
            LocalTensor<int32_t> input =
                inputQueue_.DeQue<int32_t>();
            if (nextRows != 0U) {
                Prefetch<RowIsX1>(nextOffset, nextRows, cols);
            }
            ComputeRows<RowIsX1>(cachedRow, input, rows, cols);
            inputQueue_.FreeTensor(input);
            CopyOut(rowOffset, rows, cols);
            rowOffset = nextOffset;
            rows = nextRows;
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectionScalarBuffer_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    GlobalTensor<int32_t> x1Global_;
    GlobalTensor<int32_t> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterRowSingleTilingData* tiling_;
};

template <typename T>
struct BoundRowPipelinePolicy;

template <>
struct BoundRowPipelinePolicy<half> {

    static constexpr uint32_t kSlotElements = 28928U;
    static constexpr uint32_t kSlotBytesPerElement = 3U;
};

template <>
struct BoundRowPipelinePolicy<float> {
    static constexpr uint32_t kSlotElements = 20480U;
    static constexpr uint32_t kSlotBytesPerElement = 4U;
};

template <>
struct BoundRowPipelinePolicy<bfloat16_t> {
    static constexpr uint32_t kSlotElements = 15360U;
    static constexpr uint32_t kSlotBytesPerElement = 3U;
};

template <>
struct BoundRowPipelinePolicy<int32_t> {
    static constexpr uint32_t kSlotElements = 13824U;
    static constexpr uint32_t kSlotBytesPerElement = 4U;
};

template <>
struct BoundRowPipelinePolicy<int8_t> {
    static constexpr uint32_t kSlotElements = 28160U;
    static constexpr uint32_t kSlotBytesPerElement = 3U;
};

template <typename T>
struct BoundRowExpandedPipelinePolicy;

template <>
struct BoundRowExpandedPipelinePolicy<half> {
    static constexpr uint32_t kSlotElements = 30208U;
};

template <>
struct BoundRowExpandedPipelinePolicy<float> {
    static constexpr uint32_t kSlotElements = 21504U;
};

template <>
struct BoundRowExpandedPipelinePolicy<bfloat16_t> {
    static constexpr uint32_t kSlotElements = 16128U;
};

template <>
struct BoundRowExpandedPipelinePolicy<int32_t> {
    static constexpr uint32_t kSlotElements = 14336U;
};

template <>
struct BoundRowExpandedPipelinePolicy<int8_t> {
    static constexpr uint32_t kSlotElements = 29440U;
};

template <typename T>
struct BoundRowCompactPipelinePolicy;

template <>
struct BoundRowCompactPipelinePolicy<half> {
    static constexpr uint32_t kSlotElements = 28672U;
    static constexpr uint32_t kMaskBytes = 4736U;
};

template <>
struct BoundRowCompactPipelinePolicy<float> {
    static constexpr uint32_t kSlotElements = 20480U;
    static constexpr uint32_t kMaskBytes = 3392U;
};

template <>
struct BoundRowCompactPipelinePolicy<bfloat16_t> {
    static constexpr uint32_t kSlotElements = 15360U;
    static constexpr uint32_t kMaskBytes = 2560U;
};

template <>
struct BoundRowCompactPipelinePolicy<int32_t> {
    static constexpr uint32_t kSlotElements = 13568U;
    static constexpr uint32_t kMaskBytes = 2240U;
};

template <>
struct BoundRowCompactPipelinePolicy<int8_t> {
    static constexpr uint32_t kSlotElements = 27904U;
    static constexpr uint32_t kMaskBytes = 4608U;
};

template <typename T, bool MultipleOuter = false, bool CompactPitch = false,
          bool ContiguousOutput = false, bool ManualSelect = false,
          bool ExpandedSlots = false, bool StaticPingPong = false,
          bool CompactTiling = false, bool ArithmeticBoolean = false>
class KernelGreaterAlignedRowBoundPipeline {
public:
    using ComputeT = typename RowStripePolicy<T>::ComputeType;
    using TilingDataT = typename std::conditional<
        CompactTiling, GreaterRowSingleTilingData,
        GreaterTilingData>::type;
    static constexpr uint32_t kSlotElements =
        ExpandedSlots ?
            BoundRowExpandedPipelinePolicy<T>::kSlotElements :
        (CompactPitch ? BoundRowCompactPipelinePolicy<T>::kSlotElements :
            BoundRowPipelinePolicy<T>::kSlotElements);

    static constexpr uint32_t kOutputGapBytes = 32U;
    static constexpr uint32_t kSlotTailBytes =
        BoundRowPipelinePolicy<T>::kSlotBytesPerElement == 3U ?
            kOutputGapBytes : 0U;
    static constexpr uint32_t kSlotBytes = kSlotElements *
        BoundRowPipelinePolicy<T>::kSlotBytesPerElement + kSlotTailBytes;
    static constexpr uint32_t kOutputByteOffset =
        2U * kSlotElements + kOutputGapBytes;
    static constexpr uint32_t kComputeRepeatElements =
        256U / sizeof(ComputeT);
    static constexpr uint32_t kMaskRepeatStride =
        kComputeRepeatElements / 8U;
    static constexpr uint32_t kSelectRepeatElements = 128U;

    static constexpr uint32_t kRowAlignmentElements =
        CompactPitch ? 128U : 256U;
    static constexpr uint32_t kMaskRowAlignmentElements = 256U;
    static constexpr uint32_t kMaskBytes = CompactPitch ?
        BoundRowCompactPipelinePolicy<T>::kMaskBytes :
        kSlotElements / 8U;

    static constexpr bool kUseArithmeticBoolean = ArithmeticBoolean &&
        (std::is_same<T, half>::value || std::is_same<T, int8_t>::value);
    static constexpr uint32_t kMaximumCols =
        S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32;
    static constexpr uint32_t kUserUbBytes =
        kMaximumCols * sizeof(T) +
        2U * kSlotBytes +
        kMaskBytes +
        kSelectRepeatElements * sizeof(half) +
        (IsBfloat16<T>::kValue ?
            kSlotElements * sizeof(float) : 32U) +
        (ConversionStorage<T>::kBytesPerElement == 0U ? 32U :
            kMaximumCols * ConversionStorage<T>::kBytesPerElement) +
        (IsInt32<T>::kValue ?
            kSlotElements * sizeof(int32_t) : 32U);
    static constexpr uint32_t kUserUbLimit = ExpandedSlots ?
        192U * 1024U - 256U : 184U * 1024U;
    static_assert(kSlotElements % 256U == 0U,
                  "bound row slot must preserve 256-element alignment");
    static_assert(kSlotElements / kSelectRepeatElements <= 255U,
                  "one Select command must cover a complete bound slot");
    static_assert(kOutputByteOffset + kSlotElements <= kSlotBytes,
                  "byte result must remain inside the bound IO slot");
    static_assert(!CompactPitch || kRowAlignmentElements %
                      kComputeRepeatElements == 0U,
                  "compact row pitch must cover complete Compare repeats");
    static_assert(!CompactPitch || kRowAlignmentElements %
                      kSelectRepeatElements == 0U,
                  "compact row pitch must cover complete Select repeats");
    static_assert(!ContiguousOutput || !CompactPitch,
                  "contiguous output requires an unpadded aligned row");
    static_assert(!ContiguousOutput ||
                      kRowAlignmentElements % 32U == 0U,
                  "contiguous byte output must contain complete DMA blocks");
    static_assert(!ContiguousOutput ||
                      kSlotElements / 32U <= 65535U,
                  "contiguous output must fit one DMA blockLen field");
    static_assert(!ContiguousOutput ||
                      kSlotElements * sizeof(T) / 32U <= 65535U,
                  "contiguous input must fit one DMA blockLen field");
    static_assert(!ExpandedSlots || ManualSelect,
                  "expanded slots require the temp-free Select path");
    static_assert(!ExpandedSlots || !CompactPitch,
                  "expanded slots currently use the 256-element row pitch");
    static_assert(!CompactTiling || !MultipleOuter,
                  "compact tiling is valid only for a single outer slice");
    static_assert(!ArithmeticBoolean ||
                      (ManualSelect && ContiguousOutput && ExpandedSlots &&
                       StaticPingPong && CompactTiling && !CompactPitch &&
                       !MultipleOuter),
                  "arithmetic Greater is isolated to the aligned key-65 A/B");
    static_assert(kUserUbBytes <= kUserUbLimit,
                  "bound pipeline exceeds its Arch 2201 UB window");

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const TilingDataT& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        InitBankSeparatedBuffers();
    }

    __aicore__ inline void Process()
    {
        if constexpr (MultipleOuter) {
            if (tiling_->matrixX1Pattern ==
                S9_GREATER_MATRIX_ROW_BROADCAST) {
                ProcessOuterRows<true>();
            } else {
                ProcessOuterRows<false>();
            }
        } else {
            if (tiling_->matrixX1Pattern ==
                S9_GREATER_MATRIX_ROW_BROADCAST) {
                ProcessRows<true>();
            } else {
                ProcessRows<false>();
            }
        }
    }

private:
    __aicore__ inline void InitBankSeparatedBuffers()
    {
        constexpr uint32_t kRowBytes = kMaximumCols * sizeof(T);
        constexpr uint32_t kAllocatedMaskBytes = kMaskBytes;
        constexpr uint32_t kSelectionBytes =
            kSelectRepeatElements * sizeof(half);
        constexpr uint32_t kDenseConvertedBytes =
            IsBfloat16<T>::kValue ?
                kSlotElements * sizeof(float) : 32U;
        constexpr uint32_t kRowConvertedBytes =
            ConversionStorage<T>::kBytesPerElement == 0U ? 32U :
                kMaximumCols * ConversionStorage<T>::kBytesPerElement;
        constexpr uint32_t kMaximumBytes = IsInt32<T>::kValue ?
            kSlotElements * sizeof(int32_t) : 32U;

        if constexpr (CompactPitch && std::is_same<T, half>::value) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
        } else if constexpr (
            CompactPitch && std::is_same<T, float>::value) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
        } else if constexpr (CompactPitch && IsBfloat16<T>::kValue) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
        } else if constexpr (CompactPitch && IsInt32<T>::kValue) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
        } else if constexpr (CompactPitch) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
        } else if constexpr (ExpandedSlots && IsBfloat16<T>::kValue) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
        } else if constexpr (std::is_same<T, half>::value) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
        } else if constexpr (std::is_same<T, float>::value) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
        } else if constexpr (IsBfloat16<T>::kValue) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
        } else if constexpr (IsInt32<T>::kValue) {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
        } else {

            pipe_.InitBuffer(rowBuffer_, kRowBytes);
            pipe_.InitBuffer(ioQueue0_, 1U, kSlotBytes);
            pipe_.InitBuffer(ioQueue1_, 1U, kSlotBytes);
            pipe_.InitBuffer(selectionValueBuffer_, kSelectionBytes);
            pipe_.InitBuffer(maskBuffer_, kAllocatedMaskBytes);
            pipe_.InitBuffer(rowConvertedBuffer_, kRowConvertedBytes);
            pipe_.InitBuffer(denseConvertedBuffer_, kDenseConvertedBytes);
            pipe_.InitBuffer(maximumBuffer_, kMaximumBytes);
        }
    }

    __aicore__ inline uint64_t MatrixInputOffset(
        uint64_t outerIndex, uint32_t rowOffset, uint32_t pattern,
        const uint64_t* inputDims, const uint64_t* inputStrides) const
    {
        uint64_t inputOffset = 0U;
        for (uint32_t reverse = tiling_->matrixRowStart;
             reverse > 0U; --reverse) {
            const uint32_t axis = reverse - 1U;
            const uint64_t coordinate =
                outerIndex % tiling_->outputDims[axis];
            outerIndex /= tiling_->outputDims[axis];
            if (inputDims[axis] != 1U) {
                inputOffset += coordinate * inputStrides[axis];
            }
        }
        if ((pattern & S9_GREATER_MATRIX_ROW_BROADCAST) == 0U) {
            inputOffset += static_cast<uint64_t>(rowOffset) *
                inputStrides[tiling_->outerRank - 1U];
        }
        return inputOffset;
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyCachedRow(LocalTensor<T> rowInput)
    {
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t validBytes = cols * sizeof(T);
        const uint32_t alignedBytes =
            (validBytes + 31U) & ~31U;
        const DataCopyExtParams copy(
            1U, validBytes, 0U, 0U, 0U);
        const DataCopyPadExtParams<T> pad(
            alignedBytes != validBytes, 0U,
            static_cast<uint8_t>(
                (alignedBytes - validBytes) / sizeof(T)),
            static_cast<T>(0));
        if constexpr (RowIsX1) {
            DataCopyPad(rowInput, x1Global_[0U], copy, pad);
        } else {
            DataCopyPad(rowInput, x2Global_[0U], copy, pad);
        }
        const event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);

        if constexpr (ConversionStorage<T>::kBytesPerElement != 0U) {
            LocalTensor<ComputeT> converted =
                rowConvertedBuffer_.Get<ComputeT>();
            Cast(converted, rowInput, RoundMode::CAST_NONE, cols);
            PipeBarrier<PIPE_V>();
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyCachedRowAtOuter(
        uint64_t outerIndex, LocalTensor<T> rowInput)
    {
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t validBytes = cols * sizeof(T);
        const uint32_t alignedBytes =
            (validBytes + 31U) & ~31U;
        const DataCopyExtParams copy(
            1U, validBytes, 0U, 0U, 0U);
        const DataCopyPadExtParams<T> pad(
            alignedBytes != validBytes, 0U,
            static_cast<uint8_t>(
                (alignedBytes - validBytes) / sizeof(T)),
            static_cast<T>(0));
        if constexpr (RowIsX1) {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, 0U, tiling_->matrixX1Pattern,
                tiling_->x1Dims, tiling_->x1Strides);
            DataCopyPad(rowInput, x1Global_[offset], copy, pad);
        } else {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, 0U, tiling_->matrixX2Pattern,
                tiling_->x2Dims, tiling_->x2Strides);
            DataCopyPad(rowInput, x2Global_[offset], copy, pad);
        }
        const event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyDenseRowsTo(
        LocalTensor<T> ioLocal, uint32_t rowOffset,
        uint32_t rows, uint32_t cols, uint32_t paddedCols)
    {
        if constexpr (ContiguousOutput) {

            const uint32_t inputBytes = rows * cols * sizeof(T);
            const DataCopyParams copy(
                1U, static_cast<uint16_t>(inputBytes / 32U), 0U, 0U);
            if constexpr (RowIsX1) {
                DataCopy(ioLocal, x2Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            } else {
                DataCopy(ioLocal, x1Global_[
                    static_cast<uint64_t>(rowOffset) * cols], copy);
            }
            return;
        }
        const uint32_t validBytes = cols * sizeof(T);
        const uint32_t alignedBytes =
            (validBytes + 31U) & ~31U;
        const uint32_t destinationStride =
            (paddedCols * sizeof(T) - alignedBytes) / 32U;
        const DataCopyExtParams copy(
            static_cast<uint16_t>(rows), validBytes,
            0U, destinationStride, 0U);
        const DataCopyPadExtParams<T> pad(
            alignedBytes != validBytes, 0U,
            static_cast<uint8_t>(
                (alignedBytes - validBytes) / sizeof(T)),
            static_cast<T>(0));
        if constexpr (RowIsX1) {
            DataCopyPad(ioLocal, x2Global_[
                static_cast<uint64_t>(rowOffset) * cols], copy, pad);
        } else {
            DataCopyPad(ioLocal, x1Global_[
                static_cast<uint64_t>(rowOffset) * cols], copy, pad);
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyDenseRowsAtOuterTo(
        uint64_t outerIndex, LocalTensor<T> ioLocal,
        uint32_t rowOffset, uint32_t rows,
        uint32_t cols, uint32_t paddedCols)
    {
        if constexpr (ContiguousOutput) {
            const uint32_t inputBytes = rows * cols * sizeof(T);
            const DataCopyParams copy(
                1U, static_cast<uint16_t>(inputBytes / 32U), 0U, 0U);
            if constexpr (RowIsX1) {
                const uint64_t offset = MatrixInputOffset(
                    outerIndex, rowOffset, tiling_->matrixX2Pattern,
                    tiling_->x2Dims, tiling_->x2Strides);
                DataCopy(ioLocal, x2Global_[offset], copy);
            } else {
                const uint64_t offset = MatrixInputOffset(
                    outerIndex, rowOffset, tiling_->matrixX1Pattern,
                    tiling_->x1Dims, tiling_->x1Strides);
                DataCopy(ioLocal, x1Global_[offset], copy);
            }
            return;
        }
        const uint32_t validBytes = cols * sizeof(T);
        const uint32_t alignedBytes =
            (validBytes + 31U) & ~31U;
        const uint32_t destinationStride =
            (paddedCols * sizeof(T) - alignedBytes) / 32U;
        const DataCopyExtParams copy(
            static_cast<uint16_t>(rows), validBytes,
            0U, destinationStride, 0U);
        const DataCopyPadExtParams<T> pad(
            alignedBytes != validBytes, 0U,
            static_cast<uint8_t>(
                (alignedBytes - validBytes) / sizeof(T)),
            static_cast<T>(0));
        if constexpr (RowIsX1) {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, rowOffset, tiling_->matrixX2Pattern,
                tiling_->x2Dims, tiling_->x2Strides);
            DataCopyPad(ioLocal, x2Global_[offset], copy, pad);
        } else {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, rowOffset, tiling_->matrixX1Pattern,
                tiling_->x1Dims, tiling_->x1Strides);
            DataCopyPad(ioLocal, x1Global_[offset], copy, pad);
        }
    }

    template <bool QueueZero, bool RowIsX1>
    __aicore__ inline void Prefetch(
        uint32_t rowOffset, uint32_t rows,
        uint32_t cols, uint32_t paddedCols)
    {
        if constexpr (QueueZero) {
            LocalTensor<T> ioLocal = ioQueue0_.template AllocTensor<T>();
            CopyDenseRowsTo<RowIsX1>(
                ioLocal, rowOffset, rows, cols, paddedCols);
            ioQueue0_.template EnQue<
                TPosition::GM, TPosition::VECIN, T>(ioLocal);
        } else {
            LocalTensor<T> ioLocal = ioQueue1_.template AllocTensor<T>();
            CopyDenseRowsTo<RowIsX1>(
                ioLocal, rowOffset, rows, cols, paddedCols);
            ioQueue1_.template EnQue<
                TPosition::GM, TPosition::VECIN, T>(ioLocal);
        }
    }

    template <bool QueueZero, bool RowIsX1>
    __aicore__ inline void PrefetchAtOuter(
        uint64_t outerIndex, uint32_t rowOffset, uint32_t rows,
        uint32_t cols, uint32_t paddedCols)
    {
        if constexpr (QueueZero) {
            LocalTensor<T> ioLocal = ioQueue0_.template AllocTensor<T>();
            CopyDenseRowsAtOuterTo<RowIsX1>(
                outerIndex, ioLocal, rowOffset, rows, cols, paddedCols);
            ioQueue0_.template EnQue<
                TPosition::GM, TPosition::VECIN, T>(ioLocal);
        } else {
            LocalTensor<T> ioLocal = ioQueue1_.template AllocTensor<T>();
            CopyDenseRowsAtOuterTo<RowIsX1>(
                outerIndex, ioLocal, rowOffset, rows, cols, paddedCols);
            ioQueue1_.template EnQue<
                TPosition::GM, TPosition::VECIN, T>(ioLocal);
        }
    }

    template <bool QueueZero>
    __aicore__ inline LocalTensor<T> DeQueInput()
    {
        if constexpr (QueueZero) {
            return ioQueue0_.template DeQue<
                TPosition::GM, TPosition::VECIN, T>();
        } else {
            return ioQueue1_.template DeQue<
                TPosition::GM, TPosition::VECIN, T>();
        }
    }

    __aicore__ inline LocalTensor<ComputeT> DenseForCompare(
        LocalTensor<T> ioLocal, uint32_t count)
    {
        if constexpr (IsBfloat16<T>::kValue) {
            LocalTensor<ComputeT> dense =
                denseConvertedBuffer_.Get<ComputeT>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            Cast(dense, ioLocal, RoundMode::CAST_NONE, count);
#else
            if constexpr (ManualSelect) {

                constexpr uint32_t kConvertElements =
                    256U / sizeof(ComputeT);
                static_assert(kSlotElements % kConvertElements == 0U,
                              "BF16 slot must contain whole VCONV repeats");
                const uint8_t repeats = static_cast<uint8_t>(
                    count / kConvertElements);
                SetVectorMask<ComputeT>(
                    static_cast<int32_t>(kConvertElements));
                vconv_bf162f32(
                    reinterpret_cast<__ubuf__ float*>(
                        dense.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ bfloat16_t*>(
                        ioLocal.GetPhyAddr()),
                    repeats, 1U, 1U, 8U, 4U);
            } else {
                Cast(dense, ioLocal, RoundMode::CAST_NONE, count);
            }
#endif
            PipeBarrier<PIPE_V>();
            return dense;
        } else if constexpr (std::is_same<T, int8_t>::value) {
            LocalTensor<uint8_t> bytes =
                ioLocal.template ReinterpretCast<uint8_t>();
            LocalTensor<ComputeT> dense =
                bytes[kSlotElements].template ReinterpretCast<ComputeT>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            Cast(dense, ioLocal, RoundMode::CAST_NONE, count);
#else
            if constexpr (ManualSelect) {

                constexpr uint32_t kConvertElements =
                    256U / sizeof(ComputeT);
                static_assert(kSlotElements % kConvertElements == 0U,
                              "INT8 slot must contain whole VCONV repeats");
                const uint8_t repeats = static_cast<uint8_t>(
                    count / kConvertElements);
                vconv_s82f16(
                    reinterpret_cast<__ubuf__ half*>(
                        dense.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ int8_t*>(
                        ioLocal.GetPhyAddr()),
                    repeats, 1U, 1U, 8U, 4U);
            } else {
                Cast(dense, ioLocal, RoundMode::CAST_NONE, count);
            }
#endif
            PipeBarrier<PIPE_V>();
            return dense;
        } else {
            return ioLocal.template ReinterpretCast<ComputeT>();
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void CompareRows(
        LocalTensor<ComputeT> row, LocalTensor<ComputeT> dense,
        LocalTensor<uint8_t> mask, uint32_t rows, uint32_t cols)
    {
        const uint8_t compareRepeats = static_cast<uint8_t>(
            cols / kComputeRepeatElements);
        const uint32_t maskPitchElements = CompactPitch ?
            (cols + kMaskRowAlignmentElements - 1U) /
                kMaskRowAlignmentElements * kMaskRowAlignmentElements :
            cols;
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1

        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(
            mask.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ ComputeT*>(
            row.GetPhyAddr());
        auto* rawDense = reinterpret_cast<__ubuf__ ComputeT*>(
            dense.GetPhyAddr());
#endif
        if constexpr (IsInt32<T>::kValue) {
            LocalTensor<int32_t> maximum =
                maximumBuffer_.Get<int32_t>();
            const BinaryRepeatParams maximumParams(
                1U, 1U, 1U, 8U, 8U, 8U);
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
            auto* rawMaximum = reinterpret_cast<__ubuf__ int32_t*>(
                maximum.GetPhyAddr());
#endif
            for (uint32_t rowIndex = 0U;
                 rowIndex < rows; ++rowIndex) {
                const uint32_t elementOffset = rowIndex * cols;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
                if constexpr (RowIsX1) {
                    Max(maximum[elementOffset], row,
                        dense[elementOffset],
                        static_cast<uint64_t>(kComputeRepeatElements),
                        compareRepeats, maximumParams);
                } else {
                    Max(maximum[elementOffset], dense[elementOffset],
                        row,
                        static_cast<uint64_t>(kComputeRepeatElements),
                        compareRepeats, maximumParams);
                }
#else
                if constexpr (ManualSelect) {
                    if constexpr (RowIsX1) {
                        vmax(rawMaximum + elementOffset, rawRow,
                             rawDense + elementOffset, compareRepeats,
                             1U, 1U, 1U, 8U, 8U, 8U);
                    } else {
                        vmax(rawMaximum + elementOffset,
                             rawDense + elementOffset, rawRow,
                             compareRepeats,
                             1U, 1U, 1U, 8U, 8U, 8U);
                    }
                } else {
                    if constexpr (RowIsX1) {
                        Max(maximum[elementOffset], row,
                            dense[elementOffset],
                            static_cast<uint64_t>(
                                kComputeRepeatElements),
                            compareRepeats, maximumParams);
                    } else {
                        Max(maximum[elementOffset], dense[elementOffset],
                            row, static_cast<uint64_t>(
                                kComputeRepeatElements),
                            compareRepeats, maximumParams);
                    }
                }
#endif
            }
            PipeBarrier<PIPE_V>();
            const BinaryRepeatParams compareParams(
                1U, 1U, 1U, kMaskRepeatStride, 8U, 8U);
            for (uint32_t rowIndex = 0U;
                rowIndex < rows; ++rowIndex) {
                const uint32_t elementOffset = rowIndex * cols;
                const uint32_t maskOffset =
                    rowIndex * maskPitchElements / 8U;
                if constexpr (RowIsX1) {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
                    Compare(mask[maskOffset], maximum[elementOffset],
                            dense[elementOffset], CMPMODE::EQ,
                            static_cast<uint64_t>(kComputeRepeatElements),
                            compareRepeats, compareParams);
#else
                    if constexpr (ManualSelect) {
                        auto* rawMaximum =
                            reinterpret_cast<__ubuf__ int32_t*>(
                                maximum.GetPhyAddr());
                        vcmpv_eq(rawMask + maskOffset,
                                 rawMaximum + elementOffset,
                                 rawDense + elementOffset,
                                 compareRepeats, 1U, 1U, 1U,
                                 kMaskRepeatStride, 8U, 8U);
                    } else {
                        Compare(mask[maskOffset], maximum[elementOffset],
                                dense[elementOffset], CMPMODE::EQ,
                                static_cast<uint64_t>(kComputeRepeatElements),
                                compareRepeats, compareParams);
                    }
#endif
                } else {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
                    Compare(mask[maskOffset], maximum[elementOffset],
                            row, CMPMODE::EQ,
                            static_cast<uint64_t>(kComputeRepeatElements),
                            compareRepeats, compareParams);
#else
                    if constexpr (ManualSelect) {
                        auto* rawMaximum =
                            reinterpret_cast<__ubuf__ int32_t*>(
                                maximum.GetPhyAddr());
                        vcmpv_eq(rawMask + maskOffset,
                                 rawMaximum + elementOffset, rawRow,
                                 compareRepeats, 1U, 1U, 1U,
                                 kMaskRepeatStride, 8U, 8U);
                    } else {
                        Compare(mask[maskOffset], maximum[elementOffset],
                                row, CMPMODE::EQ,
                                static_cast<uint64_t>(kComputeRepeatElements),
                                compareRepeats, compareParams);
                    }
#endif
                }
            }
        } else {
            const BinaryRepeatParams compareParams(
                1U, 1U, 1U, kMaskRepeatStride, 8U, 8U);
            for (uint32_t rowIndex = 0U;
                 rowIndex < rows; ++rowIndex) {
                const uint32_t elementOffset = rowIndex * cols;
                const uint32_t maskOffset =
                    rowIndex * maskPitchElements / 8U;
                if constexpr (RowIsX1) {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
                    Compare(mask[maskOffset], row,
                            dense[elementOffset], CMPMODE::GT,
                            static_cast<uint64_t>(kComputeRepeatElements),
                            compareRepeats, compareParams);
#else
                    if constexpr (ManualSelect) {
                        vcmpv_gt(rawMask + maskOffset, rawRow,
                                 rawDense + elementOffset,
                                 compareRepeats, 1U, 1U, 1U,
                                 kMaskRepeatStride, 8U, 8U);
                    } else {
                        Compare(mask[maskOffset], row,
                                dense[elementOffset], CMPMODE::GT,
                                static_cast<uint64_t>(
                                    kComputeRepeatElements),
                                compareRepeats, compareParams);
                    }
#endif
                } else {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
                    Compare(mask[maskOffset], dense[elementOffset],
                            row, CMPMODE::GT,
                            static_cast<uint64_t>(kComputeRepeatElements),
                            compareRepeats, compareParams);
#else
                    if constexpr (ManualSelect) {
                        vcmpv_gt(rawMask + maskOffset,
                                 rawDense + elementOffset, rawRow,
                                 compareRepeats, 1U, 1U, 1U,
                                 kMaskRepeatStride, 8U, 8U);
                    } else {
                        Compare(mask[maskOffset], dense[elementOffset],
                                row, CMPMODE::GT,
                                static_cast<uint64_t>(
                                    kComputeRepeatElements),
                                compareRepeats, compareParams);
                    }
#endif
                }
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline LocalTensor<half> SelectionScalarTensor()
    {

        if constexpr (IsInt32<T>::kValue) {
            return denseConvertedBuffer_.Get<half>();
        } else {
            return maximumBuffer_.Get<half>();
        }
    }

    __aicore__ inline void InitializeSelectionState()
    {
        if constexpr (kUseArithmeticBoolean) {
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1

            SetVectorMask<half>(
                static_cast<int32_t>(kSelectRepeatElements));
#endif
            return;
        }
        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        Duplicate(selectionValues,
                  static_cast<half>(IsInt32<T>::kValue ? 0 : 1),
                  kSelectRepeatElements);
        if constexpr (ManualSelect) {
            LocalTensor<half> scalarValue = SelectionScalarTensor();
            Duplicate(scalarValue,
                      static_cast<half>(IsInt32<T>::kValue ? 1 : 0),
                      16U);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (ManualSelect) {

            SetCmpMask(SelectionScalarTensor());
            if constexpr (IsInt32<T>::kValue) {

                SetVectorMask<half>(
                    static_cast<int32_t>(kSelectRepeatElements));
            }
            PipeBarrier<PIPE_V>();
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeRowsArithmetic(
        LocalTensor<ComputeT> row, LocalTensor<T> ioLocal,
        uint32_t rows, uint32_t cols)
    {
        static_assert(std::is_same<ComputeT, half>::value,
                      "arithmetic Greater currently requires half compute");
        const uint32_t count = rows * cols;
        LocalTensor<ComputeT> dense = DenseForCompare(ioLocal, count);
        LocalTensor<half> selected;
        if constexpr (std::is_same<T, int8_t>::value) {

            selected = dense;
        } else {
            selected = ioLocal.template ReinterpretCast<half>();
        }
        const uint8_t rowRepeats = static_cast<uint8_t>(
            cols / kSelectRepeatElements);
        const BinaryRepeatParams subtractParams(
            1U, 1U, 1U, 8U, 8U, 8U);

#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
        auto* rawSelected = reinterpret_cast<__ubuf__ half*>(
            selected.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ half*>(
            row.GetPhyAddr());
        auto* rawDense = reinterpret_cast<__ubuf__ half*>(
            dense.GetPhyAddr());
#endif
        for (uint32_t rowIndex = 0U; rowIndex < rows; ++rowIndex) {
            const uint32_t elementOffset = rowIndex * cols;
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            if constexpr (RowIsX1) {
                SubRelu(selected[elementOffset], row,
                        dense[elementOffset],
                        static_cast<uint64_t>(kSelectRepeatElements),
                        rowRepeats, subtractParams);
            } else {
                SubRelu(selected[elementOffset], dense[elementOffset],
                        row,
                        static_cast<uint64_t>(kSelectRepeatElements),
                        rowRepeats, subtractParams);
            }
#else
            if constexpr (RowIsX1) {
                vsubrelu(rawSelected + elementOffset, rawRow,
                         rawDense + elementOffset, rowRepeats,
                         1U, 1U, 1U, 8U, 8U, 8U);
            } else {
                vsubrelu(rawSelected + elementOffset,
                         rawDense + elementOffset, rawRow, rowRepeats,
                         1U, 1U, 1U, 8U, 8U, 8U);
            }
#endif
        }
        PipeBarrier<PIPE_V>();

        const uint8_t denseRepeats = static_cast<uint8_t>(
            count / kSelectRepeatElements);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        const UnaryRepeatParams minimumParams(1U, 1U, 8U, 8U);
        Mins(selected, selected, static_cast<half>(1),
             static_cast<uint64_t>(kSelectRepeatElements),
             denseRepeats, minimumParams);
#else
        vmins(rawSelected, rawSelected, static_cast<half>(1),
              denseRepeats, 1U, 1U, 8U, 8U, false, false);
#endif
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> output;
        if constexpr (std::is_same<T, int8_t>::value) {

            output = ioLocal.template ReinterpretCast<uint8_t>();
        } else {
            output = ioLocal.template ReinterpretCast<uint8_t>()[
                kOutputByteOffset];
        }
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.template ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_CEIL, count);
#else
        vconv_f162s8c(
            reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            rawSelected, denseRepeats, 1U, 1U, 4U, 8U);
#endif
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeRows(
        LocalTensor<ComputeT> row, LocalTensor<T> ioLocal,
        uint32_t rows, uint32_t cols)
    {
        if constexpr (kUseArithmeticBoolean) {
            ComputeRowsArithmetic<RowIsX1>(row, ioLocal, rows, cols);
            return;
        }
        const uint32_t count = rows * cols;
        LocalTensor<ComputeT> dense = DenseForCompare(ioLocal, count);
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        CompareRows<RowIsX1>(row, dense, mask, rows, cols);

        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected =
            ioLocal.template ReinterpretCast<half>();
        const uint8_t selectRepeats = static_cast<uint8_t>(
            (CompactPitch ? cols : count) / kSelectRepeatElements);
        const BinaryRepeatParams selectParams(
            1U, 1U, 1U, 8U, 0U, 0U);
        if constexpr (ManualSelect) {

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1

            SetVectorMask<half>(
                static_cast<int32_t>(kSelectRepeatElements));
#else
            if constexpr (!IsInt32<T>::kValue) {
                SetVectorMask<half>(
                    static_cast<int32_t>(kSelectRepeatElements));
            }
#endif
#if !defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1
            auto* rawSelected = reinterpret_cast<__ubuf__ half*>(
                selected.GetPhyAddr());
            auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(
                mask.GetPhyAddr());
            auto* rawSelectionValues = reinterpret_cast<__ubuf__ half*>(
                selectionValues.GetPhyAddr());
#endif
            if constexpr (CompactPitch) {
                const uint32_t maskPitchElements =
                    (cols + kMaskRowAlignmentElements - 1U) /
                    kMaskRowAlignmentElements * kMaskRowAlignmentElements;
                for (uint32_t rowIndex = 0U;
                     rowIndex < rows; ++rowIndex) {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
                    Select(selected[rowIndex * cols],
                           mask[rowIndex * maskPitchElements / 8U],
                           selectionValues, selectRepeats, selectParams);
#else
                    vsel(rawSelected + rowIndex * cols,
                         rawSelectionValues,
                         rawMask + rowIndex * maskPitchElements / 8U,
                         selectRepeats, 1U, 1U, 1U, 8U, 0U, 0U,
                         static_cast<uint8_t>(
                             SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
                }
            } else {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
                Select(selected, mask, selectionValues,
                       selectRepeats, selectParams);
#else
                vsel(rawSelected, rawSelectionValues, rawMask,
                     selectRepeats, 1U, 1U, 1U, 8U, 0U, 0U,
                     static_cast<uint8_t>(
                         SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
            }
        } else {
            if constexpr (CompactPitch) {
                const uint32_t maskPitchElements =
                    (cols + kMaskRowAlignmentElements - 1U) /
                    kMaskRowAlignmentElements * kMaskRowAlignmentElements;
                for (uint32_t rowIndex = 0U;
                     rowIndex < rows; ++rowIndex) {
                    Select(selected[rowIndex * cols],
                           mask[rowIndex * maskPitchElements / 8U],
                           selectionValues,
                           static_cast<half>(IsInt32<T>::kValue ? 1 : 0),
                           SELMODE::VSEL_TENSOR_SCALAR_MODE,
                           static_cast<uint64_t>(kSelectRepeatElements),
                           selectRepeats, selectParams);
                }
            } else {
                Select(selected, mask, selectionValues,
                       static_cast<half>(IsInt32<T>::kValue ? 1 : 0),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE,
                       static_cast<uint64_t>(kSelectRepeatElements),
                       selectRepeats, selectParams);
            }
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<uint8_t> output;
        if constexpr (kUseArithmeticBoolean &&
                      std::is_same<T, int8_t>::value) {
            output = ioLocal.template ReinterpretCast<uint8_t>();
        } else {
            output = ioLocal.template ReinterpretCast<uint8_t>()[
                kOutputByteOffset];
        }
        if constexpr (ManualSelect) {
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1

            Cast(output.template ReinterpretCast<int8_t>(), selected,
                 RoundMode::CAST_RINT, count);
#else

            const uint8_t conversionRepeats = static_cast<uint8_t>(
                count / kSelectRepeatElements);
            vconv_f162s8r(
                reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
                reinterpret_cast<__ubuf__ half*>(selected.GetPhyAddr()),
                conversionRepeats, 1U, 1U, 4U, 8U);
#endif
        } else {
            Cast(output.template ReinterpretCast<int8_t>(), selected,
                 RoundMode::CAST_RINT, count);
        }
    }

    template <bool QueueZero>
    __aicore__ inline void CopyOutAndFree(
        LocalTensor<T> ioLocal, uint64_t outputRowOffset,
        uint32_t rows, uint32_t cols, uint32_t paddedCols)
    {
        if constexpr (QueueZero) {
            ioQueue0_.template EnQue<
                TPosition::VECOUT, TPosition::GM, T>(ioLocal);
            ioLocal = ioQueue0_.template DeQue<
                TPosition::VECOUT, TPosition::GM, T>();
        } else {
            ioQueue1_.template EnQue<
                TPosition::VECOUT, TPosition::GM, T>(ioLocal);
            ioLocal = ioQueue1_.template DeQue<
                TPosition::VECOUT, TPosition::GM, T>();
        }
        LocalTensor<uint8_t> output;
        if constexpr (kUseArithmeticBoolean &&
                      std::is_same<T, int8_t>::value) {
            output = ioLocal.template ReinterpretCast<uint8_t>();
        } else {
            output = ioLocal.template ReinterpretCast<uint8_t>()[
                kOutputByteOffset];
        }
        if constexpr (ContiguousOutput) {

            const uint32_t outputBytes = rows * cols;
            const DataCopyParams copy(
                1U, static_cast<uint16_t>(outputBytes / 32U), 0U, 0U);
            DataCopy(yGlobal_[outputRowOffset * cols], output, copy);
        } else {
            const uint32_t alignedOutputBytes =
                (cols + 31U) & ~31U;
            const uint32_t sourceStride =
                (paddedCols - alignedOutputBytes) / 32U;
            const DataCopyExtParams copy(
                static_cast<uint16_t>(rows), cols,
                sourceStride, 0U, 0U);
            DataCopyPad(yGlobal_[outputRowOffset * cols],
                        output, copy);
        }
        if constexpr (QueueZero) {
            ioQueue0_.FreeTensor(ioLocal);
        } else {
            ioQueue1_.FreeTensor(ioLocal);
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        InitializeSelectionState();

        LocalTensor<T> cachedRow = rowBuffer_.Get<T>();
        CopyCachedRow<RowIsX1>(cachedRow);

        LocalTensor<ComputeT> cachedRowCompute;
        if constexpr (ConversionStorage<T>::kBytesPerElement != 0U) {
            cachedRowCompute = rowConvertedBuffer_.Get<ComputeT>();
            const event_t vectorToVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_V));
            SetFlag<HardEvent::V_V>(vectorToVector);
            WaitFlag<HardEvent::V_V>(vectorToVector);
        } else {
            cachedRowCompute =
                cachedRow.template ReinterpretCast<ComputeT>();
        }

        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t paddedCols =
            (cols + kRowAlignmentElements - 1U) /
            kRowAlignmentElements * kRowAlignmentElements;
        const uint32_t rowsPerBatch = kSlotElements / paddedCols;
        const uint64_t beginRow64 =
            beginTile * tiling_->matrixRowsPerTile;
        const uint64_t endRow64 =
            (beginTile + tileCount) * tiling_->matrixRowsPerTile <
                    tiling_->matrixRows ?
                (beginTile + tileCount) * tiling_->matrixRowsPerTile :
                tiling_->matrixRows;
        uint32_t rowOffset = static_cast<uint32_t>(beginRow64);
        const uint32_t endRow = static_cast<uint32_t>(endRow64);
        uint32_t rows = endRow - rowOffset < rowsPerBatch ?
            endRow - rowOffset : rowsPerBatch;
        Prefetch<true, RowIsX1>(
            rowOffset, rows, cols, paddedCols);

        if constexpr (StaticPingPong) {

            while (rowOffset < endRow) {
                uint32_t nextOffset = rowOffset + rows;
                uint32_t nextRows = nextOffset < endRow ?
                    (endRow - nextOffset < rowsPerBatch ?
                        endRow - nextOffset : rowsPerBatch) : 0U;
                LocalTensor<T> ioLocal = DeQueInput<true>();
                if (nextRows != 0U) {
                    Prefetch<false, RowIsX1>(
                        nextOffset, nextRows, cols, paddedCols);
                }
                ComputeRows<RowIsX1>(
                    cachedRowCompute, ioLocal, rows, paddedCols);
                CopyOutAndFree<true>(
                    ioLocal, rowOffset, rows, cols, paddedCols);
                rowOffset = nextOffset;
                rows = nextRows;
                if (rowOffset >= endRow) {
                    break;
                }

                nextOffset = rowOffset + rows;
                nextRows = nextOffset < endRow ?
                    (endRow - nextOffset < rowsPerBatch ?
                        endRow - nextOffset : rowsPerBatch) : 0U;
                LocalTensor<T> ioLocal1 = DeQueInput<false>();
                if (nextRows != 0U) {
                    Prefetch<true, RowIsX1>(
                        nextOffset, nextRows, cols, paddedCols);
                }
                ComputeRows<RowIsX1>(
                    cachedRowCompute, ioLocal1, rows, paddedCols);
                CopyOutAndFree<false>(
                    ioLocal1, rowOffset, rows, cols, paddedCols);
                rowOffset = nextOffset;
                rows = nextRows;
            }
        } else {
            bool queueZero = true;
            while (rowOffset < endRow) {
                const uint32_t nextOffset = rowOffset + rows;
                const uint32_t nextRows = nextOffset < endRow ?
                    (endRow - nextOffset < rowsPerBatch ?
                        endRow - nextOffset : rowsPerBatch) : 0U;

                if (queueZero) {
                    LocalTensor<T> ioLocal = DeQueInput<true>();
                    if (nextRows != 0U) {
                        Prefetch<false, RowIsX1>(
                            nextOffset, nextRows, cols, paddedCols);
                    }
                    ComputeRows<RowIsX1>(
                        cachedRowCompute, ioLocal, rows, paddedCols);
                    CopyOutAndFree<true>(
                        ioLocal, rowOffset, rows, cols, paddedCols);
                } else {
                    LocalTensor<T> ioLocal = DeQueInput<false>();
                    if (nextRows != 0U) {
                        Prefetch<true, RowIsX1>(
                            nextOffset, nextRows, cols, paddedCols);
                    }
                    ComputeRows<RowIsX1>(
                        cachedRowCompute, ioLocal, rows, paddedCols);
                    CopyOutAndFree<false>(
                        ioLocal, rowOffset, rows, cols, paddedCols);
                }
                rowOffset = nextOffset;
                rows = nextRows;
                queueZero = !queueZero;
            }
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessOuterTile(
        uint64_t outerIndex, uint32_t tileRowOffset,
        uint32_t tileRows, LocalTensor<ComputeT> cachedRowCompute,
        uint32_t cols, uint32_t paddedCols)
    {
        const uint32_t rowsPerBatch = kSlotElements / paddedCols;
        const uint32_t endRow = tileRowOffset + tileRows;
        uint32_t rowOffset = tileRowOffset;
        uint32_t rows = tileRows < rowsPerBatch ?
            tileRows : rowsPerBatch;
        PrefetchAtOuter<true, RowIsX1>(
            outerIndex, rowOffset, rows, cols, paddedCols);

        if constexpr (StaticPingPong) {
            while (rowOffset < endRow) {
                uint32_t nextOffset = rowOffset + rows;
                uint32_t nextRows = nextOffset < endRow ?
                    (endRow - nextOffset < rowsPerBatch ?
                        endRow - nextOffset : rowsPerBatch) : 0U;
                uint64_t outputRowOffset =
                    outerIndex * tiling_->matrixRows + rowOffset;
                LocalTensor<T> ioLocal = DeQueInput<true>();
                if (nextRows != 0U) {
                    PrefetchAtOuter<false, RowIsX1>(
                        outerIndex, nextOffset, nextRows,
                        cols, paddedCols);
                }
                ComputeRows<RowIsX1>(
                    cachedRowCompute, ioLocal, rows, paddedCols);
                CopyOutAndFree<true>(
                    ioLocal, outputRowOffset,
                    rows, cols, paddedCols);
                rowOffset = nextOffset;
                rows = nextRows;
                if (rowOffset >= endRow) {
                    break;
                }

                nextOffset = rowOffset + rows;
                nextRows = nextOffset < endRow ?
                    (endRow - nextOffset < rowsPerBatch ?
                        endRow - nextOffset : rowsPerBatch) : 0U;
                outputRowOffset =
                    outerIndex * tiling_->matrixRows + rowOffset;
                LocalTensor<T> ioLocal1 = DeQueInput<false>();
                if (nextRows != 0U) {
                    PrefetchAtOuter<true, RowIsX1>(
                        outerIndex, nextOffset, nextRows,
                        cols, paddedCols);
                }
                ComputeRows<RowIsX1>(
                    cachedRowCompute, ioLocal1, rows, paddedCols);
                CopyOutAndFree<false>(
                    ioLocal1, outputRowOffset,
                    rows, cols, paddedCols);
                rowOffset = nextOffset;
                rows = nextRows;
            }
        } else {
            bool queueZero = true;
            while (rowOffset < endRow) {
                const uint32_t nextOffset = rowOffset + rows;
                const uint32_t nextRows = nextOffset < endRow ?
                    (endRow - nextOffset < rowsPerBatch ?
                        endRow - nextOffset : rowsPerBatch) : 0U;
                const uint64_t outputRowOffset =
                    outerIndex * tiling_->matrixRows + rowOffset;

                if (queueZero) {
                    LocalTensor<T> ioLocal = DeQueInput<true>();
                    if (nextRows != 0U) {
                        PrefetchAtOuter<false, RowIsX1>(
                            outerIndex, nextOffset, nextRows,
                            cols, paddedCols);
                    }
                    ComputeRows<RowIsX1>(
                        cachedRowCompute, ioLocal, rows, paddedCols);
                    CopyOutAndFree<true>(
                        ioLocal, outputRowOffset,
                        rows, cols, paddedCols);
                } else {
                    LocalTensor<T> ioLocal = DeQueInput<false>();
                    if (nextRows != 0U) {
                        PrefetchAtOuter<true, RowIsX1>(
                            outerIndex, nextOffset, nextRows,
                            cols, paddedCols);
                    }
                    ComputeRows<RowIsX1>(
                        cachedRowCompute, ioLocal, rows, paddedCols);
                    CopyOutAndFree<false>(
                        ioLocal, outputRowOffset,
                        rows, cols, paddedCols);
                }
                rowOffset = nextOffset;
                rows = nextRows;
                queueZero = !queueZero;
            }
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessOuterRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        InitializeSelectionState();

        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t paddedCols =
            (cols + kRowAlignmentElements - 1U) /
            kRowAlignmentElements * kRowAlignmentElements;
        LocalTensor<T> cachedRowInput = rowBuffer_.Get<T>();
        LocalTensor<ComputeT> cachedRowCompute;
        if constexpr (ConversionStorage<T>::kBytesPerElement != 0U) {
            cachedRowCompute = rowConvertedBuffer_.Get<ComputeT>();
        } else {
            cachedRowCompute =
                cachedRowInput.template ReinterpretCast<ComputeT>();
        }

        uint64_t cachedOuter = ~static_cast<uint64_t>(0U);
        for (uint64_t relative = 0U; relative < tileCount; ++relative) {
            const uint64_t tile = beginTile + relative;
            const uint64_t outerIndex =
                tile / tiling_->matrixRowTiles;
            const uint32_t rowTile = static_cast<uint32_t>(
                tile - outerIndex * tiling_->matrixRowTiles);
            const uint32_t rowOffset =
                rowTile * tiling_->matrixRowsPerTile;
            const uint64_t remainingRows =
                tiling_->matrixRows - rowOffset;
            const uint32_t rows = static_cast<uint32_t>(
                remainingRows < tiling_->matrixRowsPerTile ?
                    remainingRows : tiling_->matrixRowsPerTile);

            if (outerIndex != cachedOuter) {
                if (cachedOuter != ~static_cast<uint64_t>(0U)) {
                    PipeBarrier<PIPE_ALL>();
                }
                CopyCachedRowAtOuter<RowIsX1>(
                    outerIndex, cachedRowInput);
                if constexpr (
                    ConversionStorage<T>::kBytesPerElement != 0U) {
                    Cast(cachedRowCompute, cachedRowInput,
                         RoundMode::CAST_NONE, cols);
                    PipeBarrier<PIPE_V>();
                    const event_t vectorToVector = static_cast<event_t>(
                        GetTPipePtr()->FetchEventID(HardEvent::V_V));
                    SetFlag<HardEvent::V_V>(vectorToVector);
                    WaitFlag<HardEvent::V_V>(vectorToVector);
                }
                cachedOuter = outerIndex;
            }

            ProcessOuterTile<RowIsX1>(
                outerIndex, rowOffset, rows, cachedRowCompute,
                cols, paddedCols);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> ioQueue0_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> ioQueue1_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> denseConvertedBuffer_;
    TBuf<TPosition::VECCALC> rowConvertedBuffer_;
    TBuf<TPosition::VECCALC> maximumBuffer_;
    GlobalTensor<T> x1Global_;
    GlobalTensor<T> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const TilingDataT* tiling_;
};

template <typename T, uint32_t PipelineBufferNum>
class KernelGreaterRowSlab {
public:
    using ComputeT = typename RowStripePolicy<T>::ComputeType;
    using SelectionT = typename std::conditional<
        (sizeof(ComputeT) == sizeof(float)), float, half>::type;
    static constexpr uint32_t kChunkElements =
        RowStripePolicy<T>::kChunkElements;
    static constexpr uint8_t kCompareMaskRepeatStride =
        static_cast<uint8_t>(kChunkElements / 8U);

    static_assert(PipelineBufferNum == 1U || PipelineBufferNum == 2U,
                  "row slab supports one or two physical buffers");
    static constexpr uint32_t kPipelineBufferNum = PipelineBufferNum;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;
        slabCols_ = tiling.rowSlabCols;
        maxRows_ = tiling.rowSlabRows;
        const uint32_t cols = static_cast<uint32_t>(tiling.matrixCols);
        rowStorageCols_ =
            ((cols + kChunkElements - 1U) / kChunkElements) *
            kChunkElements;
        const uint32_t stripeElements = maxRows_ * kChunkElements;

        pipe_.InitBuffer(rowBuffer_, rowStorageCols_ * sizeof(T));
        pipe_.InitBuffer(denseQueue_, kPipelineBufferNum,
                         maxRows_ * slabCols_ * sizeof(T));

        constexpr uint32_t outputStorageBytes = sizeof(uint8_t);
        pipe_.InitBuffer(yQueue_, kPipelineBufferNum,
                         maxRows_ * slabCols_ * outputStorageBytes);
        pipe_.InitBuffer(maskBuffer_,
                         AlignBufferBytes(maxRows_ *
                                          kCompareMaskRepeatStride));
        pipe_.InitBuffer(selectionValueBuffer_,
                         AlignBufferBytes(kChunkElements *
                                          sizeof(SelectionT)));
        const uint32_t denseConvertedBytes =
            ConversionStorage<T>::kBytesPerElement == 0U ? 32U :
            AlignBufferBytes(maxRows_ * slabCols_ *
                             ConversionStorage<T>::kBytesPerElement);
        const uint32_t rowConvertedBytes =
            ConversionStorage<T>::kBytesPerElement == 0U ? 32U :
            AlignBufferBytes(rowStorageCols_ *
                             ConversionStorage<T>::kBytesPerElement);
        const uint32_t maximumBytes = IsInt32<T>::kValue ?
            AlignBufferBytes(stripeElements * sizeof(int32_t)) : 32U;
        pipe_.InitBuffer(denseConvertedBuffer_, denseConvertedBytes);
        pipe_.InitBuffer(rowConvertedBuffer_, rowConvertedBytes);
        pipe_.InitBuffer(maximumBuffer_, maximumBytes);
    }

    __aicore__ inline void Process()
    {
        if (tiling_->matrixX1Pattern ==
            S9_GREATER_MATRIX_ROW_BROADCAST) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ static inline uint32_t AlignBufferBytes(uint32_t bytes)
    {
        return (bytes + 31U) & ~31U;
    }

    __aicore__ inline uint64_t MatrixInputOffset(
        uint64_t outerIndex, uint32_t rowOffset, uint32_t pattern,
        const uint64_t* inputDims, const uint64_t* inputStrides) const
    {
        uint64_t inputOffset = 0U;
        for (uint32_t reverse = tiling_->matrixRowStart;
             reverse > 0U; --reverse) {
            const uint32_t axis = reverse - 1U;
            const uint64_t coordinate =
                outerIndex % tiling_->outputDims[axis];
            outerIndex /= tiling_->outputDims[axis];
            if (inputDims[axis] != 1U) {
                inputOffset += coordinate * inputStrides[axis];
            }
        }
        if ((pattern & S9_GREATER_MATRIX_ROW_BROADCAST) == 0U) {
            inputOffset += static_cast<uint64_t>(rowOffset) *
                inputStrides[tiling_->outerRank - 1U];
        }
        return inputOffset;
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyCachedRow(uint64_t outerIndex,
                                         LocalTensor<T> rowLocal)
    {
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        constexpr uint32_t kDataBlockBytes = 32U;
        const uint32_t rowBytes = cols * sizeof(T);
        const uint32_t alignedBytes =
            AlignBufferBytes(rowBytes);
        const uint8_t rightPadding = static_cast<uint8_t>(
            (alignedBytes - rowBytes) / sizeof(T));
        const DataCopyExtParams copy(1U, rowBytes, 0U, 0U, 0U);
        const DataCopyPadExtParams<T> pad(
            rightPadding != 0U, 0U, rightPadding, static_cast<T>(0));
        if constexpr (RowIsX1) {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, 0U, tiling_->matrixX1Pattern,
                tiling_->x1Dims, tiling_->x1Strides);
            DataCopyPad(rowLocal, x1Global_[offset], copy, pad);
        } else {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, 0U, tiling_->matrixX2Pattern,
                tiling_->x2Dims, tiling_->x2Strides);
            DataCopyPad(rowLocal, x2Global_[offset], copy, pad);
        }
        const event_t mte2ToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);
    }

    template <bool RowIsX1>
    __aicore__ inline void CopyDenseSlab(
        uint64_t outerIndex, uint32_t rowOffset, uint32_t rows,
        uint32_t columnOffset, uint32_t validCols)
    {
        LocalTensor<T> denseLocal = denseQueue_.AllocTensor<T>();
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        constexpr uint32_t kDataBlockBytes = 32U;
        const uint32_t validBytes = validCols * sizeof(T);
        const uint32_t alignedBytes = AlignBufferBytes(validBytes);
        const uint8_t rightPadding = static_cast<uint8_t>(
            (alignedBytes - validBytes) / sizeof(T));
        const uint32_t destinationStride =
            (slabCols_ * sizeof(T) - alignedBytes) / kDataBlockBytes;
        const DataCopyExtParams copy(
            static_cast<uint16_t>(rows), validBytes,
            (cols - validCols) * sizeof(T), destinationStride, 0U);
        const DataCopyPadExtParams<T> pad(
            rightPadding != 0U, 0U, rightPadding, static_cast<T>(0));
        if constexpr (RowIsX1) {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, rowOffset, tiling_->matrixX2Pattern,
                tiling_->x2Dims, tiling_->x2Strides) + columnOffset;
            DataCopyPad(denseLocal, x2Global_[offset], copy, pad);
        } else {
            const uint64_t offset = MatrixInputOffset(
                outerIndex, rowOffset, tiling_->matrixX1Pattern,
                tiling_->x1Dims, tiling_->x1Strides) + columnOffset;
            DataCopyPad(denseLocal, x1Global_[offset], copy, pad);
        }
        denseQueue_.EnQue(denseLocal);
    }

    template <bool RowIsX1>
    __aicore__ inline void CompareChunk(
        LocalTensor<uint8_t> mask, LocalTensor<ComputeT> row,
        LocalTensor<ComputeT> dense, uint32_t rows)
    {
        const uint64_t maskCount = static_cast<uint64_t>(kChunkElements);
        const uint8_t repeatTime = static_cast<uint8_t>(rows);
        const uint8_t denseRepeatStride = static_cast<uint8_t>(
            slabCols_ * sizeof(ComputeT) / 32U);
        if constexpr (IsInt32<T>::kValue) {
            LocalTensor<int32_t> maximum =
                maximumBuffer_.Get<int32_t>();
            if constexpr (RowIsX1) {
                const BinaryRepeatParams maximumParams(
                    1U, 1U, 1U, 8U, 0U, denseRepeatStride);
                Max(maximum, row, dense, maskCount, repeatTime,
                    maximumParams);
            } else {
                const BinaryRepeatParams maximumParams(
                    1U, 1U, 1U, 8U, denseRepeatStride, 0U);
                Max(maximum, dense, row, maskCount, repeatTime,
                    maximumParams);
            }
            PipeBarrier<PIPE_V>();
            if constexpr (RowIsX1) {
                const BinaryRepeatParams compareParams(
                    1U, 1U, 1U, kCompareMaskRepeatStride,
                    8U, denseRepeatStride);
                Compare(mask, maximum, dense, CMPMODE::EQ,
                        maskCount, repeatTime, compareParams);
            } else {
                const BinaryRepeatParams compareParams(
                    1U, 1U, 1U, kCompareMaskRepeatStride, 8U, 0U);
                Compare(mask, maximum, row, CMPMODE::EQ,
                        maskCount, repeatTime, compareParams);
            }
        } else if constexpr (RowIsX1) {
            const BinaryRepeatParams compareParams(
                1U, 1U, 1U, kCompareMaskRepeatStride,
                0U, denseRepeatStride);
            Compare(mask, row, dense, CMPMODE::GT,
                    maskCount, repeatTime, compareParams);
        } else {
            const BinaryRepeatParams compareParams(
                1U, 1U, 1U, kCompareMaskRepeatStride,
                denseRepeatStride, 0U);
            Compare(mask, dense, row, CMPMODE::GT,
                    maskCount, repeatTime, compareParams);
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeSlab(
        LocalTensor<ComputeT> cachedRow, LocalTensor<T> denseInput,
        LocalTensor<uint8_t> yLocal, uint32_t rows,
        uint32_t columnOffset, uint32_t validCols)
    {
        LocalTensor<ComputeT> dense;
        if constexpr (ConversionStorage<T>::kBytesPerElement != 0U) {
            dense = denseConvertedBuffer_.Get<ComputeT>();
            Cast(dense, denseInput, RoundMode::CAST_NONE,
                 rows * slabCols_);
            PipeBarrier<PIPE_V>();
        } else {
            dense = denseInput;
        }

        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        LocalTensor<SelectionT> selectionValues =
            selectionValueBuffer_.Get<SelectionT>();

        LocalTensor<SelectionT> selectedSlab =
            dense.template ReinterpretCast<SelectionT>();
        const uint8_t selectedRepeatStride = static_cast<uint8_t>(
            slabCols_ * sizeof(SelectionT) / 32U);
        const BinaryRepeatParams selectParams(
            1U, 1U, 1U, selectedRepeatStride, 0U, 0U);

        for (uint32_t chunkOffset = 0U; chunkOffset < validCols;
             chunkOffset += kChunkElements) {
            CompareChunk<RowIsX1>(
                mask, cachedRow[columnOffset + chunkOffset],
                dense[chunkOffset], rows);
            PipeBarrier<PIPE_V>();
            if constexpr (IsInt32<T>::kValue) {
                Select(selectedSlab[chunkOffset], mask,
                       selectionValues, static_cast<SelectionT>(1),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE,
                       static_cast<uint64_t>(kChunkElements),
                       static_cast<uint8_t>(rows), selectParams);
            } else {
                Select(selectedSlab[chunkOffset], mask,
                       selectionValues, static_cast<SelectionT>(0),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE,
                       static_cast<uint64_t>(kChunkElements),
                       static_cast<uint8_t>(rows), selectParams);
            }
            PipeBarrier<PIPE_V>();
        }
        if constexpr (sizeof(SelectionT) == sizeof(float)) {
            LocalTensor<half> selectedHalf =
                selectedSlab.template ReinterpretCast<half>();

            Cast(selectedHalf, selectedSlab, RoundMode::CAST_NONE,
                 rows * slabCols_);
            PipeBarrier<PIPE_V>();
            Cast(yLocal.ReinterpretCast<int8_t>(), selectedHalf,
                 RoundMode::CAST_RINT, rows * slabCols_);
        } else {
            Cast(yLocal.ReinterpretCast<int8_t>(), selectedSlab,
                 RoundMode::CAST_RINT, rows * slabCols_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutSlab(
        uint64_t outputOffset, uint32_t rows,
        uint32_t columnOffset, uint32_t validCols)
    {
        LocalTensor<uint8_t> yLocal = yQueue_.DeQue<uint8_t>();
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const uint32_t alignedBytes = AlignBufferBytes(validCols);
        const uint32_t sourceStride =
            (slabCols_ - alignedBytes) / 32U;
        const DataCopyExtParams copy(
            static_cast<uint16_t>(rows), validCols,
            sourceStride, cols - validCols, 0U);
        DataCopyPad(yGlobal_[outputOffset + columnOffset], yLocal, copy);
        yQueue_.FreeTensor(yLocal);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        LocalTensor<SelectionT> selectionValues =
            selectionValueBuffer_.Get<SelectionT>();
        if constexpr (IsInt32<T>::kValue) {
            Duplicate(selectionValues, static_cast<SelectionT>(0),
                      kChunkElements);
        } else {
            Duplicate(selectionValues, static_cast<SelectionT>(1),
                      kChunkElements);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<T> cachedRowInput = rowBuffer_.Get<T>();
        LocalTensor<ComputeT> cachedRow;
        if constexpr (ConversionStorage<T>::kBytesPerElement != 0U) {
            cachedRow = rowConvertedBuffer_.Get<ComputeT>();
        } else {
            cachedRow = cachedRowInput;
        }
        uint64_t cachedOuter = ~static_cast<uint64_t>(0U);
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        for (uint64_t relative = 0U; relative < tileCount; ++relative) {
            const uint64_t tile = beginTile + relative;
            const uint64_t outerIndex = tile / tiling_->matrixRowTiles;
            const uint32_t rowTile = static_cast<uint32_t>(
                tile - outerIndex * tiling_->matrixRowTiles);
            const uint32_t rowOffset =
                rowTile * tiling_->matrixRowsPerTile;
            const uint64_t remainingRows =
                tiling_->matrixRows - rowOffset;
            const uint32_t rows = static_cast<uint32_t>(
                remainingRows < tiling_->matrixRowsPerTile ?
                    remainingRows : tiling_->matrixRowsPerTile);
            const uint64_t outputOffset =
                outerIndex * tiling_->matrixRows * tiling_->matrixCols +
                static_cast<uint64_t>(rowOffset) * tiling_->matrixCols;

            if (outerIndex != cachedOuter) {
                if (cachedOuter != ~static_cast<uint64_t>(0U)) {
                    PipeBarrier<PIPE_ALL>();
                }
                CopyCachedRow<RowIsX1>(outerIndex, cachedRowInput);
                if constexpr (
                    ConversionStorage<T>::kBytesPerElement != 0U) {
                    Cast(cachedRow, cachedRowInput, RoundMode::CAST_NONE,
                         rowStorageCols_);
                    PipeBarrier<PIPE_V>();
                }
                cachedOuter = outerIndex;
            }

            for (uint32_t columnOffset = 0U; columnOffset < cols;
                 columnOffset += slabCols_) {
                const uint32_t remainingCols = cols - columnOffset;
                const uint32_t validCols = remainingCols < slabCols_ ?
                    remainingCols : slabCols_;
                CopyDenseSlab<RowIsX1>(
                    outerIndex, rowOffset, rows,
                    columnOffset, validCols);
                LocalTensor<T> denseLocal = denseQueue_.DeQue<T>();
                LocalTensor<uint8_t> yLocal =
                    yQueue_.AllocTensor<uint8_t>();
                ComputeSlab<RowIsX1>(
                    cachedRow, denseLocal, yLocal, rows,
                    columnOffset, validCols);
                denseQueue_.FreeTensor(denseLocal);
                yQueue_.EnQue(yLocal);
                CopyOutSlab(
                    outputOffset, rows, columnOffset, validCols);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowBuffer_;
    TQue<QuePosition::VECIN, kBufferNum> denseQueue_;
    TQue<QuePosition::VECOUT, kBufferNum> yQueue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> denseConvertedBuffer_;
    TBuf<TPosition::VECCALC> rowConvertedBuffer_;
    TBuf<TPosition::VECCALC> maximumBuffer_;
    GlobalTensor<T> x1Global_;
    GlobalTensor<T> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterTilingData* tiling_;
    uint32_t slabCols_;
    uint32_t maxRows_;
    uint32_t rowStorageCols_;
};

template <typename T>
class KernelGreaterCrossScalar {
public:
    static constexpr bool kSupported =
        std::is_same<T, half>::value || std::is_same<T, float>::value;
    static constexpr uint32_t kOutputBatchRows = 8U;

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterRowSingleTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        constexpr uint32_t kRepeatBytes = 256U;
        constexpr uint32_t kDataBlockBytes = 32U;
        const uint32_t cols = static_cast<uint32_t>(tiling.matrixCols);
        const uint32_t repeatElements = kRepeatBytes / sizeof(T);
        alignedCols_ =
            (cols + repeatElements - 1U) / repeatElements * repeatElements;
        const uint32_t outputBufferCount =
            alignedCols_ <= 7168U ? 2U : 1U;
        const uint32_t columnBytes =
            (tiling.matrixRowsPerTile * sizeof(T) + kDataBlockBytes - 1U) /
            kDataBlockBytes * kDataBlockBytes;
        const uint32_t maskBytes =
            (alignedCols_ / 8U + kDataBlockBytes - 1U) /
            kDataBlockBytes * kDataBlockBytes;

        pipe_.InitBuffer(rowQueue_, 1U, alignedCols_ * sizeof(T));
        pipe_.InitBuffer(columnQueue_, 1U, columnBytes);

        pipe_.InitBuffer(yQueue_, outputBufferCount,
                         kOutputBatchRows * alignedCols_ * sizeof(uint8_t));
        pipe_.InitBuffer(maskBuffer_, maskBytes);
        pipe_.InitBuffer(selectionValueBuffer_,
                         alignedCols_ * sizeof(half));
        pipe_.InitBuffer(selectedBuffer_,
                         alignedCols_ * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        if constexpr (!kSupported) {
            return;
        } else {
            if (tiling_->matrixX1Pattern ==
                S9_GREATER_MATRIX_ROW_BROADCAST) {
                ProcessRows<true>();
            } else {
                ProcessRows<false>();
            }
        }
    }

private:
    template <bool RowIsX1>
    __aicore__ inline LocalTensor<T> CopyCachedRow()
    {
        LocalTensor<T> row = rowQueue_.AllocTensor<T>();
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const DataCopyExtParams copy(
            1U, cols * sizeof(T), 0U, 0U, 0U);
        const DataCopyPadExtParams<T> pad(
            false, 0U, 0U, static_cast<T>(0));
        if constexpr (RowIsX1) {
            DataCopyPad(row, x1Global_, copy, pad);
        } else {
            DataCopyPad(row, x2Global_, copy, pad);
        }
        rowQueue_.EnQue(row);
        return rowQueue_.DeQue<T>();
    }

    template <bool RowIsX1>
    __aicore__ inline LocalTensor<T> CopyColumnScalars(
        uint32_t rowOffset, uint32_t rows)
    {
        LocalTensor<T> column = columnQueue_.AllocTensor<T>();
        const DataCopyExtParams copy(
            1U, rows * sizeof(T), 0U, 0U, 0U);
        const DataCopyPadExtParams<T> pad(
            false, 0U, 0U, static_cast<T>(0));
        if constexpr (RowIsX1) {
            DataCopyPad(column, x2Global_[rowOffset], copy, pad);
        } else {
            DataCopyPad(column, x1Global_[rowOffset], copy, pad);
        }
        columnQueue_.EnQue(column);
        return columnQueue_.DeQue<T>();
    }

    template <bool RowIsX1>
    __aicore__ inline void ComputeRow(
        LocalTensor<T> row, T scalar, LocalTensor<uint8_t> output)
    {
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        if constexpr (RowIsX1) {
            CompareScalar(mask, row, scalar, CMPMODE::GT, alignedCols_);
        } else {

            CompareScalar(mask, row, scalar, CMPMODE::LT, alignedCols_);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected = selectedBuffer_.Get<half>();
        OutputSelector<T>::Run(
            selected, mask, selectionValues, alignedCols_);
        PipeBarrier<PIPE_V>();

        Cast(output.template ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, alignedCols_);
    }

    __aicore__ inline void CopyOutBatch(
        LocalTensor<uint8_t> output, uint64_t outputOffset,
        uint32_t rows)
    {
        yQueue_.EnQue(output);
        output = yQueue_.DeQue<uint8_t>();
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        if (alignedCols_ == cols && (cols & 31U) == 0U) {
            const DataCopyParams copy(
                1U, static_cast<uint16_t>(rows * cols / 32U), 0U, 0U);
            DataCopy(yGlobal_[outputOffset], output, copy);
        } else {
            const uint32_t alignedOutputBytes = (cols + 31U) & ~31U;
            const uint32_t sourceStride =
                (alignedCols_ - alignedOutputBytes) / 32U;
            const DataCopyExtParams copy(
                static_cast<uint16_t>(rows), cols,
                sourceStride, 0U, 0U);
            DataCopyPad(yGlobal_[outputOffset], output, copy);
        }
        yQueue_.FreeTensor(output);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t baseTiles = tiling_->matrixTotalTiles / blocks;
        const uint64_t remainder = tiling_->matrixTotalTiles % blocks;
        const uint64_t beginTile = baseTiles * block +
            (block < remainder ? block : remainder);
        const uint64_t tileCount = baseTiles +
            (block < remainder ? 1U : 0U);
        if (tileCount == 0U) {
            return;
        }

        LocalTensor<half> selectionValues =
            selectionValueBuffer_.Get<half>();
        OutputSelector<T>::InitTensor(selectionValues, alignedCols_);
        PipeBarrier<PIPE_V>();

        LocalTensor<T> row = CopyCachedRow<RowIsX1>();
        const uint32_t cols = static_cast<uint32_t>(tiling_->matrixCols);
        const event_t mte2ToScalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        const event_t scalarToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        for (uint64_t relative = 0U; relative < tileCount; ++relative) {
            const uint64_t tile = beginTile + relative;
            const uint32_t rowOffset = static_cast<uint32_t>(
                tile * tiling_->matrixRowsPerTile);
            const uint64_t remaining = tiling_->matrixRows - rowOffset;
            const uint32_t rows = static_cast<uint32_t>(
                remaining < tiling_->matrixRowsPerTile ?
                    remaining : tiling_->matrixRowsPerTile);
            LocalTensor<T> column =
                CopyColumnScalars<RowIsX1>(rowOffset, rows);

            SetFlag<HardEvent::MTE2_S>(mte2ToScalar);
            WaitFlag<HardEvent::MTE2_S>(mte2ToScalar);
            for (uint32_t relativeRow = 0U;
                 relativeRow < rows;) {
                const uint32_t batchRows =
                    rows - relativeRow < kOutputBatchRows ?
                        rows - relativeRow : kOutputBatchRows;
                LocalTensor<uint8_t> output =
                    yQueue_.AllocTensor<uint8_t>();
                T scalars[kOutputBatchRows];
                for (uint32_t batchRow = 0U;
                     batchRow < batchRows; ++batchRow) {
                    scalars[batchRow] = column.GetValue(
                        relativeRow + batchRow);
                }

                SetFlag<HardEvent::S_V>(scalarToVector);
                WaitFlag<HardEvent::S_V>(scalarToVector);
                for (uint32_t batchRow = 0U;
                     batchRow < batchRows; ++batchRow) {
                    ComputeRow<RowIsX1>(
                        row, scalars[batchRow],
                        output[batchRow * alignedCols_]);
                }
                const uint64_t outputOffset = static_cast<uint64_t>(
                    rowOffset + relativeRow) * cols;
                CopyOutBatch(output, outputOffset, batchRows);
                relativeRow += batchRows;
            }
            columnQueue_.FreeTensor(column);
        }
        rowQueue_.FreeTensor(row);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> rowQueue_;
    TQue<QuePosition::VECIN, 1> columnQueue_;
    TQue<QuePosition::VECOUT, 2> yQueue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectedBuffer_;
    GlobalTensor<T> x1Global_;
    GlobalTensor<T> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterRowSingleTilingData* tiling_;
    uint32_t alignedCols_;
};

template <typename T>
// 处理连续稠密输入，使用独立缓冲组织搬入、比较和写回。
class KernelGreaterDense {
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;

        constexpr uint64_t kLargeBytes = 32ULL * 1024ULL * 1024ULL;
        constexpr uint64_t kBytesPerElement = 2ULL * sizeof(T) + 1ULL;
        if (tiling.totalElements >=
            (kLargeBytes + kBytesPerElement - 1U) / kBytesPerElement) {
            x1Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            x2Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }

        constexpr uint32_t tileElements = DenseTilePolicy<T>::kElements;
        constexpr uint32_t tileBytes = tileElements * sizeof(T);
        pipe_.InitBuffer(x1Queue_, kDenseBufferNum, tileBytes);
        pipe_.InitBuffer(x2Queue_, kDenseBufferNum, tileBytes);
        pipe_.InitBuffer(yQueue_, kDenseBufferNum,
                         tileElements * sizeof(uint8_t));
        pipe_.InitBuffer(maskBuffer_, tileElements / 8U);
        pipe_.InitBuffer(selectionValueBuffer_, tileElements * sizeof(half));
        pipe_.InitBuffer(selectedBuffer_, tileElements * sizeof(half));
        constexpr uint32_t convertedBytes =
            ConversionStorage<T>::kBytesPerElement == 0U ?
                32U :
                tileElements * ConversionStorage<T>::kBytesPerElement;
        pipe_.InitBuffer(convertedX1Buffer_, convertedBytes);
        pipe_.InitBuffer(convertedX2Buffer_, convertedBytes);
    }

    __aicore__ inline void Process()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        constexpr uint64_t kBlockAlignment = 256U;
        const uint64_t average = tiling_->totalElements / blocks +
            (tiling_->totalElements % blocks != 0U ? 1U : 0U);
        const uint64_t blockElements =
            (average + kBlockAlignment - 1U) & ~(kBlockAlignment - 1U);
        const uint64_t start = blockElements * block;
        const uint64_t blockEnd = start + blockElements;
        const uint64_t end = blockEnd < tiling_->totalElements ?
            blockEnd : tiling_->totalElements;
        if (start >= end) {
            return;
        }

        constexpr uint32_t tileElements = DenseTilePolicy<T>::kElements;
        LocalTensor<half> selectionValues = selectionValueBuffer_.Get<half>();
        OutputSelector<T>::InitTensor(selectionValues, tileElements);
        PipeBarrier<PIPE_V>();

        uint64_t current = start;
        while (current < end) {
            const uint64_t remaining = end - current;
            const uint32_t count = static_cast<uint32_t>(
                remaining < tileElements ? remaining : tileElements);
            CopyIn(current, count);
            Compute(count);
            CopyOut(current, count);
            current += count;
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> x1Local = x1Queue_.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue_.AllocTensor<T>();
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
        DataCopyPad(x1Local, x1Global_[offset], copy, pad);
        DataCopyPad(x2Local, x2Global_[offset], copy, pad);
        x1Queue_.EnQue(x1Local);
        x2Queue_.EnQue(x2Local);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        constexpr uint32_t kVectorAlignment = 256U;
        const uint32_t vectorCount =
            (count + kVectorAlignment - 1U) & ~(kVectorAlignment - 1U);
        LocalTensor<T> x1Local = x1Queue_.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue_.DeQue<T>();
        LocalTensor<uint8_t> yLocal = yQueue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        LocalTensor<half> selectionValues = selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected = selectedBuffer_.Get<half>();
        LocalTensor<float> convertedX1 = convertedX1Buffer_.Get<float>();
        LocalTensor<float> convertedX2 = convertedX2Buffer_.Get<float>();

        CompareRunner<T>::Run(mask, x1Local, x2Local, convertedX1,
                              convertedX2, vectorCount);
        PipeBarrier<PIPE_V>();
        OutputSelector<T>::Run(selected, mask, selectionValues, vectorCount);
        PipeBarrier<PIPE_V>();
        Cast(yLocal.ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, vectorCount);

        yQueue_.EnQue(yLocal);
        x1Queue_.FreeTensor(x1Local);
        x2Queue_.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<uint8_t> yLocal = yQueue_.DeQue<uint8_t>();
        const DataCopyExtParams copy(1, count * sizeof(uint8_t), 0, 0, 0);
        DataCopyPad(yGlobal_[offset], yLocal, copy);
        yQueue_.FreeTensor(yLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, kDenseBufferNum> x1Queue_;
    TQue<QuePosition::VECIN, kDenseBufferNum> x2Queue_;
    TQue<QuePosition::VECOUT, kDenseBufferNum> yQueue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectedBuffer_;
    TBuf<TPosition::VECCALC> convertedX1Buffer_;
    TBuf<TPosition::VECCALC> convertedX2Buffer_;
    GlobalTensor<T> x1Global_;
    GlobalTensor<T> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterTilingData* tiling_;
};

template <uint32_t OutputGroup>
class KernelGreaterInt32Periodic {
    static constexpr uint32_t kTile = 8192U;
    static constexpr uint32_t kMaxCols = 1024U;
    static constexpr uint32_t kPeriodicElements = kTile + 2U * kMaxCols;
    static constexpr uint32_t kScratchBytes = 48U * 1024U;
    static constexpr uint32_t kOutputElements = OutputGroup * kTile;
    static constexpr uint32_t kUbBytes = kMaxCols * 4U +
        kPeriodicElements * 4U + kScratchBytes + 2U * kTile * 4U +
        kOutputElements + kTile / 8U + 256U + 32U;
    static_assert(OutputGroup == 1U || OutputGroup == 2U);
    static_assert(kUbBytes <= 188160U);

    struct Transaction {
        uint64_t outer;
        uint64_t localOffset;
        uint64_t outputOffset;
        uint32_t count;
    };

public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                               const GreaterTilingData& tiling)
    {
        tiling_ = &tiling;
        rowIsX1_ = tiling.matrixX1Pattern == S9_GREATER_MATRIX_ROW_BROADCAST;
        rowGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            rowIsX1_ ? x1 : x2));
        denseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            rowIsX1_ ? x2 : x1));

        if (tiling.totalElements >= (32ULL * 1024ULL * 1024ULL + 4ULL) / 5ULL) {
            denseGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        cols_ = static_cast<uint32_t>(tiling.matrixCols);
        span_ = tiling.matrixRows * tiling.matrixCols;
        slotsPerOuter_ = 1U + span_ / kOutputElements +
            (span_ % kOutputElements != 0U ? 1U : 0U);
        pipe_.InitBuffer(compactRowBuffer_, kMaxCols * sizeof(int32_t));
        pipe_.InitBuffer(periodicRowBuffer_, kPeriodicElements * sizeof(int32_t));
        pipe_.InitBuffer(scratchBuffer_, kScratchBytes);
        pipe_.InitBuffer(denseQueue_, 2U, kTile * sizeof(int32_t));
        pipe_.InitBuffer(outputQueue_, 1U, kOutputElements);
        pipe_.InitBuffer(maskBuffer_, kTile / 8U);
        pipe_.InitBuffer(selectionValueBuffer_, 256U);
        pipe_.InitBuffer(selectionScalarBuffer_, 32U);
    }

    __aicore__ inline void Process()
    {
        if (rowIsX1_) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ inline uint64_t RowInputOffset(uint64_t outer) const
    {
        const uint64_t* dims = rowIsX1_ ? tiling_->x1Dims : tiling_->x2Dims;
        const uint64_t* strides = rowIsX1_ ? tiling_->x1Strides : tiling_->x2Strides;
        uint64_t result = 0U;
        for (uint32_t reverse = tiling_->matrixRowStart; reverse > 0U; --reverse) {
            const uint32_t axis = reverse - 1U;
            const uint64_t coordinate = outer % tiling_->outputDims[axis];
            outer /= tiling_->outputDims[axis];
            if (dims[axis] != 1U) {
                result += coordinate * strides[axis];
            }
        }
        return result;
    }

    __aicore__ inline bool NextTransaction(
        uint64_t& cursor, uint64_t end, Transaction& transaction) const
    {
        while (cursor < end) {
            const uint64_t slot = cursor++;
            const uint64_t outer = slot / slotsPerOuter_;
            const uint64_t index = slot % slotsPerOuter_;
            const uint64_t base = outer * span_;
            const uint32_t head = static_cast<uint32_t>(
                (512U - (base & 511U)) & 511U);
            const uint64_t local = index == 0U ? 0U :
                head + (index - 1U) * kOutputElements;
            const uint64_t remaining = local < span_ ? span_ - local : 0U;
            const uint32_t count = index == 0U ? head :
                static_cast<uint32_t>(remaining < kOutputElements ?
                    remaining : kOutputElements);
            if (count == 0U) {
                continue;
            }
            transaction.outer = outer;
            transaction.localOffset = local;
            transaction.outputOffset = base + local;
            transaction.count = count;
            return true;
        }
        return false;
    }

    __aicore__ inline void LoadAndExpandRow(uint64_t outer)
    {

        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> compact = compactRowBuffer_.Get<int32_t>();
        LocalTensor<int32_t> padded = scratchBuffer_.Get<int32_t>();
        LocalTensor<int32_t> periodic = periodicRowBuffer_.Get<int32_t>();
        LocalTensor<uint16_t> zeros = selectionValueBuffer_.Get<uint16_t>();
        const uint32_t alignedCols = (cols_ + 7U) & ~7U;
        Duplicate(compact, static_cast<int32_t>(0), alignedCols);
        const event_t vectorToMte2 = static_cast<event_t>(
            pipe_.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vectorToMte2);
        WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
        const DataCopyExtParams copy(1U, cols_ * sizeof(int32_t), 0U, 0U, 0U);
        const DataCopyPadExtParams<int32_t> pad(
            true, 0U, static_cast<uint8_t>(alignedCols - cols_), 0);
        DataCopyPad(compact, rowGlobal_[RowInputOffset(outer)], copy, pad);
        const event_t mte2ToVector = static_cast<event_t>(
            pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);

        const uint32_t cacheRows = (kTile + 2U * cols_ - 2U) / cols_;
        const uint32_t blocksPerRow = alignedCols / 8U;
        const BinaryRepeatParams repeatParams(
            static_cast<uint8_t>(blocksPerRow), 0U, 0U, 1U, 1U, 0U);
        SetMaskNorm();
        SetVectorMask<uint16_t, MaskMode::NORMAL>(128U);
        const uint32_t groups = cacheRows / 8U;
        uint32_t destinationOffset = 0U;
        for (uint32_t group = 0U; group < groups; ++group) {
            Or<uint16_t, false>(padded[destinationOffset].ReinterpretCast<uint16_t>(),
                compact.ReinterpretCast<uint16_t>(), zeros, MASK_PLACEHOLDER,
                static_cast<uint8_t>(blocksPerRow), repeatParams);
            destinationOffset += 8U * alignedCols;
        }
        const uint32_t tailRows = cacheRows - groups * 8U;
        if (tailRows != 0U) {
            SetVectorMask<uint16_t, MaskMode::NORMAL>(tailRows * 16U);
            Or<uint16_t, false>(padded[destinationOffset].ReinterpretCast<uint16_t>(),
                compact.ReinterpretCast<uint16_t>(), zeros, MASK_PLACEHOLDER,
                static_cast<uint8_t>(blocksPerRow), repeatParams);
        }
        PipeBarrier<PIPE_V>();
        const GatherMaskParams gatherParams = {
            1U, static_cast<uint16_t>(cacheRows),
            static_cast<uint16_t>(blocksPerRow), 0U};
        uint64_t reservedCount = 0U;
        GatherMask(periodic, padded, 7U, true, cols_, gatherParams, reservedCount);
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        SetCmpMask(selectionScalarBuffer_.Get<half>());
        SetVectorMask<half>(128);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrefetchDense(
        const Transaction& transaction, uint32_t offset)
    {
        const uint32_t remaining = transaction.count - offset;
        const uint32_t count = remaining < kTile ? remaining : kTile;
        LocalTensor<int32_t> dense = denseQueue_.AllocTensor<int32_t>();
        if ((count & 255U) != 0U) {
            Duplicate(dense[count & ~255U], static_cast<int32_t>(0), 256U);
            const event_t vectorToMte2 = static_cast<event_t>(
                pipe_.FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(vectorToMte2);
            WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
        }
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(count / 8U), 0U, 0U);
        DataCopy(dense, denseGlobal_[transaction.outputOffset + offset], copy);
        denseQueue_.EnQue(dense);
    }

    template <bool RowIsX1>
    __aicore__ inline void Compute(LocalTensor<int32_t> dense,
                                   LocalTensor<uint8_t> output,
                                   uint32_t phase, uint32_t count)
    {
        const uint32_t vectorCount = (count + 255U) & ~255U;
        const uint8_t compareRepeats = static_cast<uint8_t>(vectorCount / 64U);
        const uint8_t selectRepeats = static_cast<uint8_t>(vectorCount / 128U);
        LocalTensor<int32_t> row = periodicRowBuffer_.Get<int32_t>()[phase];
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected = dense.ReinterpretCast<half>();
        SetMaskNorm();
        SetVectorMask<half>(128);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        const BinaryRepeatParams binaryParams(1U, 1U, 1U, 8U, 8U, 8U);
        if constexpr (RowIsX1) {
            Min(dense, dense, row, 64ULL, compareRepeats, binaryParams);
        } else {
            Max(dense, dense, row, 64ULL, compareRepeats, binaryParams);
        }
#else
        auto* rawDense = reinterpret_cast<__ubuf__ int32_t*>(dense.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ int32_t*>(row.GetPhyAddr());
        if constexpr (RowIsX1) {
            vmin(rawDense, rawDense, rawRow,
                compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        } else {
            vmax(rawDense, rawDense, rawRow,
                compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        }
#endif
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Compare(mask, dense, row, CMPMODE::EQ, 64ULL, compareRepeats, binaryParams);
#else
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr());
        vcmpv_eq(rawMask, rawDense, rawRow,
            compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
#endif
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        SetVectorMask<half>(128);
        const BinaryRepeatParams selectParams(1U, 1U, 1U, 8U, 0U, 0U);
        Select(selected, mask, values, selectRepeats, selectParams);
#else
        auto* rawSelected = reinterpret_cast<__ubuf__ half*>(selected.GetPhyAddr());
        auto* rawValues = reinterpret_cast<__ubuf__ half*>(values.GetPhyAddr());
        vsel(rawSelected, rawValues, rawMask, selectRepeats,
            1U, 1U, 1U, 8U, 0U, 0U,
            static_cast<uint8_t>(SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.ReinterpretCast<int8_t>(), selected, RoundMode::CAST_RINT, vectorCount);
#else
        vconv_f162s8r(reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            rawSelected, selectRepeats, 1U, 1U, 4U, 8U);
#endif
    }

    __aicore__ inline void CopyOut(const Transaction& transaction)
    {
        LocalTensor<uint8_t> output = outputQueue_.DeQue<uint8_t>();
        if ((transaction.count & 31U) == 0U) {
            const DataCopyParams copy(
                1U, static_cast<uint16_t>(transaction.count / 32U), 0U, 0U);
            DataCopy(yGlobal_[transaction.outputOffset], output, copy);
        } else {
            const DataCopyExtParams copy(1U, transaction.count, 0U, 0U, 0U);
            DataCopyPad(yGlobal_[transaction.outputOffset], output, copy);
        }
        outputQueue_.FreeTensor(output);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t slots = tiling_->matrixOuterCount * slotsPerOuter_;
        const uint64_t baseCount = slots / blocks;
        const uint64_t remainder = slots % blocks;
        uint64_t cursor = baseCount * block + (block < remainder ? block : remainder);
        const uint64_t end = cursor + baseCount + (block < remainder ? 1U : 0U);
        Transaction current;
        if (!NextTransaction(cursor, end, current)) {
            return;
        }
        Duplicate(selectionValueBuffer_.Get<half>(), static_cast<half>(0), 128U);
        Duplicate(selectionScalarBuffer_.Get<half>(), static_cast<half>(1), 16U);
        PipeBarrier<PIPE_V>();
        LoadAndExpandRow(current.outer);
        uint64_t cachedOuter = current.outer;
        PrefetchDense(current, 0U);

        while (true) {
            if (current.outer != cachedOuter) {
                LoadAndExpandRow(current.outer);
                cachedOuter = current.outer;
            }
            LocalTensor<uint8_t> output = outputQueue_.AllocTensor<uint8_t>();
            Transaction next;
            bool hasNext = false;
            for (uint32_t offset = 0U; offset < current.count; offset += kTile) {
                LocalTensor<int32_t> dense = denseQueue_.DeQue<int32_t>();
                const uint32_t remaining = current.count - offset;
                const uint32_t count = remaining < kTile ? remaining : kTile;
                if (remaining > kTile) {
                    PrefetchDense(current, offset + kTile);
                } else {
                    hasNext = NextTransaction(cursor, end, next);
                    if (hasNext) {
                        PrefetchDense(next, 0U);
                    }
                }
                const uint32_t phase = static_cast<uint32_t>(
                    (current.localOffset + offset) % cols_);
                Compute<RowIsX1>(dense, output[offset], phase, count);
                denseQueue_.FreeTensor(dense);
            }
            outputQueue_.EnQue(output);
            CopyOut(current);
            if (!hasNext) {
                break;
            }
            current = next;
        }
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2U> denseQueue_;
    TQue<QuePosition::VECOUT, 1U> outputQueue_;
    TBuf<TPosition::VECCALC> compactRowBuffer_;
    TBuf<TPosition::VECCALC> periodicRowBuffer_;
    TBuf<TPosition::VECCALC> scratchBuffer_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectionScalarBuffer_;
    GlobalTensor<int32_t> rowGlobal_;
    GlobalTensor<int32_t> denseGlobal_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterTilingData* tiling_;
    uint64_t span_;
    uint64_t slotsPerOuter_;
    uint32_t cols_;
    bool rowIsX1_;
};


class KernelGreaterSmallHalf {
    static constexpr uint32_t kMax = 4096U;
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                               const GreaterSmallDenseTilingData& tiling)
    {
        count_ = static_cast<uint32_t>(tiling.totalElements);
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        pipe_.InitBuffer(storage_, 4U * kMax * sizeof(half) + kMax / 8U + kMax);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> x1 = storage_.Get<half>();
        LocalTensor<half> x2 = x1[kMax];
        LocalTensor<half> selected = x1[2U * kMax];
        LocalTensor<half> ones = x1[3U * kMax];
        LocalTensor<uint8_t> mask = x1[4U * kMax].ReinterpretCast<uint8_t>();
        LocalTensor<uint8_t> output = mask[kMax / 8U];
        const uint32_t rounded = (count_ + 127U) & ~127U;

        if (rounded != count_) {
            Duplicate(x1[count_ & ~127U], static_cast<half>(0), 128U);
            Duplicate(x2[count_ & ~127U], static_cast<half>(0), 128U);
            const event_t event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(event);
            WaitFlag<HardEvent::V_MTE2>(event);
        }
        const DataCopyExtParams inputCopy(1U, count_ * sizeof(half), 0U, 0U, 0U);
        const DataCopyPadExtParams<half> pad(true, 0U,
            static_cast<uint8_t>((16U - (count_ & 15U)) & 15U), static_cast<half>(0));
        DataCopyPad(x1, x1Global_, inputCopy, pad);
        DataCopyPad(x2, x2Global_, inputCopy, pad);
        Duplicate(ones, static_cast<half>(1), rounded);
        const event_t ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);
        Compare(mask, x1, x2, CMPMODE::GT, rounded);
        PipeBarrier<PIPE_V>();
        Select(selected, mask, ones, static_cast<half>(0),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, rounded);
        PipeBarrier<PIPE_V>();
        Cast(output.ReinterpretCast<int8_t>(), selected, RoundMode::CAST_RINT, rounded);
        const event_t written = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(written);
        WaitFlag<HardEvent::V_MTE3>(written);
        const DataCopyExtParams outputCopy(1U, count_, 0U, 0U, 0U);
        DataCopyPad(yGlobal_, output, outputCopy);
    }
private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> storage_;
    GlobalTensor<half> x1Global_;
    GlobalTensor<half> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    uint32_t count_;
};


class KernelGreaterFp32ResidentPredicates {
public:
    static constexpr uint32_t TileElements = 8192U;
    static constexpr uint32_t MaskBytes = 49152U;
    static_assert(4U * TileElements * sizeof(float) + MaskBytes + 288U <= 188160U);
    __aicore__ inline static bool CanStage(uint64_t n, uint64_t cores)
    {
        const uint64_t tiles = n / TileElements + (n % TileElements != 0U);
        const uint64_t maximum = tiles / cores + (tiles % cores != 0U);
        return maximum <= MaskBytes / (TileElements / 8U);
    }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                               const GreaterSmallDenseTilingData& tiling)
    {
        tiling_ = &tiling;
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if (tiling.totalElements >= (32ULL * 1024ULL * 1024ULL + 8ULL) / 9ULL) {
            x1Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            x2Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        pipe_.InitBuffer(x1Queue_, 2U, TileElements * sizeof(float));
        pipe_.InitBuffer(x2Queue_, 2U, TileElements * sizeof(float));
        pipe_.InitBuffer(maskBuffer_, MaskBytes);
        pipe_.InitBuffer(valuesBuffer_, 256U);
        pipe_.InitBuffer(scalarBuffer_, 32U);
    }
    __aicore__ inline void Process()
    {
        const uint64_t core = GetBlockIdx(), cores = GetBlockNum();
        const uint64_t stride = cores * TileElements;
        const uint64_t n = tiling_->totalElements;
        if (core * TileElements >= n) return;
        PrepareValues();
        uint32_t sequence = 0U;
        const uint64_t first = core * TileElements;
        CopyIn(first, static_cast<uint32_t>(n - first < TileElements ? n - first : TileElements));
        for (uint64_t offset = first; offset < n; offset += stride) {
            const uint32_t count = static_cast<uint32_t>(n - offset < TileElements ? n - offset : TileElements);
            const uint64_t next = offset + stride;
            if (next < n) CopyIn(next, static_cast<uint32_t>(n - next < TileElements ? n - next : TileElements));
            CompareAndCache(sequence++, count);
        }
        StartExpansion();
        sequence = 0U;
        for (uint64_t offset = core * TileElements; offset < n; offset += stride) {
            const uint32_t count = static_cast<uint32_t>(n - offset < TileElements ? n - offset : TileElements);
            ExpandAndStore(offset, sequence++, count);
        }
        FinishExpansion();
    }
private:
    __aicore__ inline void PrepareValues()
    {
        Duplicate(valuesBuffer_.Get<half>(), static_cast<half>(1), 128U);
        Duplicate(scalarBuffer_.Get<half>(), static_cast<half>(0), 16U);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<float> a = x1Queue_.AllocTensor<float>();
        LocalTensor<float> b = x2Queue_.AllocTensor<float>();
        if ((count & 255U) != 0U) {
            Duplicate(a[count & ~255U], 0.0f, 256U);
            Duplicate(b[count & ~255U], 0.0f, 256U);
            const event_t ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(ready); WaitFlag<HardEvent::V_MTE2>(ready);
        }
        if ((count & 7U) == 0U) {
            const DataCopyParams copy(1U, static_cast<uint16_t>(count / 8U), 0U, 0U);
            DataCopy(a, x1Global_[offset], copy); DataCopy(b, x2Global_[offset], copy);
        } else {
            const DataCopyExtParams copy(1U, count * sizeof(float), 0U, 0U, 0U);
            const DataCopyPadExtParams<float> pad(false, 0U, 0U, 0.0f);
            DataCopyPad(a, x1Global_[offset], copy, pad); DataCopyPad(b, x2Global_[offset], copy, pad);
        }
        x1Queue_.EnQue(a); x2Queue_.EnQue(b);
    }
    __aicore__ inline void CompareAndCache(uint32_t sequence, uint32_t count)
    {
        LocalTensor<float> a = x1Queue_.DeQue<float>();
        LocalTensor<float> b = x2Queue_.DeQue<float>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>()[sequence * (TileElements / 8U)];
        const uint32_t vectorCount = (count + 255U) & ~255U;
        for (uint32_t offset = 0U; offset < vectorCount; offset += 8192U) {
            const uint32_t segment = vectorCount - offset < 8192U ? vectorCount - offset : 8192U;
            Compare(mask[offset / 8U], a[offset], b[offset], CMPMODE::GT, segment);
        }
        PipeBarrier<PIPE_V>();
        x1Queue_.FreeTensor(a); x2Queue_.FreeTensor(b);
    }
    __aicore__ inline void StartExpansion()
    {

        selectedStorage_ = x1Queue_.AllocTensor<float>();
        outputStorage_ = x2Queue_.AllocTensor<float>();
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            outputEvent_[slot] = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
            outputPending_[slot] = false;
        }
        SetCmpMask(scalarBuffer_.Get<half>());
        SetMaskNorm(); SetVectorMask<half>(128U); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void WaitOutput(uint32_t slot)
    {
        if (outputPending_[slot]) {
            WaitFlag<HardEvent::MTE3_V>(outputEvent_[slot]);
            outputPending_[slot] = false;
        }
    }
    __aicore__ inline void ExpandAndStore(uint64_t offset, uint32_t sequence, uint32_t count)
    {
        const uint32_t slot = sequence & 1U;
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>()[sequence * (TileElements / 8U)];
        LocalTensor<half> selected = selectedStorage_.ReinterpretCast<half>();
        LocalTensor<uint8_t> output = outputStorage_.ReinterpretCast<uint8_t>()[slot * TileElements];
        const uint32_t vectorCount = (count + 255U) & ~255U;
        const uint8_t repeats = static_cast<uint8_t>(vectorCount / 128U);
        const BinaryRepeatParams params(1U, 1U, 1U, 8U, 0U, 0U);
        Select(selected, mask, valuesBuffer_.Get<half>(), repeats, params);
        PipeBarrier<PIPE_V>();
        WaitOutput(slot);
        Cast(output.ReinterpretCast<int8_t>(), selected, RoundMode::CAST_RINT, vectorCount);
        const event_t storeReady = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(storeReady); WaitFlag<HardEvent::V_MTE3>(storeReady);
        const DataCopyExtParams copy(1U, count, 0U, 0U, 0U);
        DataCopyPad(yGlobal_[offset], output, copy);
        SetFlag<HardEvent::MTE3_V>(outputEvent_[slot]); outputPending_[slot] = true;
    }
    __aicore__ inline void FinishExpansion()
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            WaitOutput(slot);
            pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputEvent_[slot]);
        }
        x1Queue_.FreeTensor(selectedStorage_); x2Queue_.FreeTensor(outputStorage_);
    }
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2U> x1Queue_, x2Queue_;
    TBuf<TPosition::VECCALC> maskBuffer_, valuesBuffer_, scalarBuffer_;
    LocalTensor<float> selectedStorage_, outputStorage_;
    GlobalTensor<float> x1Global_, x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterSmallDenseTilingData* tiling_;
    event_t outputEvent_[2]; bool outputPending_[2];
};

template <uint32_t TileElements, bool FastCopy>
class KernelGreaterDenseFp32ManualCompact {
public:
    static constexpr uint32_t kSelectRepeatElements = 128U;
    static constexpr uint32_t kUserUbBytes =
        4U * TileElements * sizeof(float) +
        2U * TileElements * sizeof(uint8_t) + TileElements / 8U +
        kSelectRepeatElements * sizeof(half) + 32U;
    static_assert(TileElements % 256U == 0U,
                  "FP32 dense tile must be vector aligned");
    static_assert(TileElements / 64U <= 255U,
                  "FP32 compare repeat must fit one instruction");
    static_assert(kUserUbBytes <= 188160U,
                  "FP32 dense user UB exceeds the audited safe limit");

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                const GreaterSmallDenseTilingData& tiling)
    {
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        tiling_ = &tiling;
        if (tiling.totalElements >= (32ULL * 1024ULL * 1024ULL + 8ULL) / 9ULL) {
            x1Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            x2Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        pipe_.InitBuffer(x1Queue_, 2U, TileElements * sizeof(float));
        pipe_.InitBuffer(x2Queue_, 2U, TileElements * sizeof(float));
        pipe_.InitBuffer(yQueue_, 2U, TileElements * sizeof(uint8_t));
        pipe_.InitBuffer(maskBuffer_, TileElements / 8U);
        pipe_.InitBuffer(selectionValueBuffer_,
                         kSelectRepeatElements * sizeof(half));
        pipe_.InitBuffer(selectionScalarBuffer_, 32U);
    }

    __aicore__ inline void Process()
    {
        const uint64_t n = tiling_->totalElements;
        const uint64_t tiles = n / TileElements + (n % TileElements != 0U);
        const uint64_t core = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t cores = static_cast<uint64_t>(GetBlockNum());
        if (core >= tiles) return;
        ownedTiles_ = (tiles - 1U - core) / cores + 1U;
        constexpr uint64_t capacity = TileElements * sizeof(float) / (TileElements / 8U);
        if (ownedTiles_ > capacity) {
            ProcessBaseline();
            return;
        }
        packedSequence_ = 0U;
        PackStage();
        PipeBarrier<PIPE_ALL>();
        CacheOwnMasks();
        ExpandStage();
    }

    __aicore__ inline void ProcessBaseline()
    {
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t elements = tiling_->totalElements;
        const uint64_t tiles = elements / TileElements + (elements % TileElements != 0U);
        if (block >= tiles) return;
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        LocalTensor<half> scalar = selectionScalarBuffer_.Get<half>();
        Duplicate(values, static_cast<half>(1), kSelectRepeatElements);
        Duplicate(scalar, static_cast<half>(0), 16U);
        PipeBarrier<PIPE_V>();
        uint64_t tile = block;
        uint64_t current = tile * TileElements;
        uint32_t count = static_cast<uint32_t>(elements - current < TileElements ? elements - current : TileElements);
        CopyIn(current, count);
        while (true) {
            const bool hasNext = tiles - tile > blocks;
            uint64_t next = 0U;
            uint32_t nextCount = 0U;
            if (hasNext) {
                next = (tile + blocks) * TileElements;
                nextCount = static_cast<uint32_t>(elements - next < TileElements ? elements - next : TileElements);
                CopyIn(next, nextCount);
            }
            Compute(count);
            CopyOut(current, count);
            if (!hasNext) break;
            tile += blocks;
            current = next;
            count = nextCount;
        }
    }

    __aicore__ inline void PackStage()
    {
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t elements = tiling_->totalElements;
        const uint64_t tiles = elements / TileElements + (elements % TileElements != 0U);
        if (block >= tiles) return;
        uint64_t tile = block;
        uint64_t current = tile * TileElements;
        uint32_t count = static_cast<uint32_t>(elements - current < TileElements ? elements - current : TileElements);
        CopyIn(current, count);
        while (true) {
            const bool hasNext = tiles - tile > blocks;
            uint64_t next = 0U;
            uint32_t nextCount = 0U;
            if (hasNext) {
                next = (tile + blocks) * TileElements;
                nextCount = static_cast<uint32_t>(elements - next < TileElements ? elements - next : TileElements);
                CopyIn(next, nextCount);
            }
            ComputePacked(count);
            CopyOutPacked(current, count);
            if (!hasNext) break;
            tile += blocks;
            current = next;
            count = nextCount;
        }
    }

private:
    __aicore__ inline uint64_t PoolOffset(uint64_t sequence) const
    {

        const uint64_t poolTile = (sequence / 8U) * static_cast<uint64_t>(GetBlockNum()) +
            static_cast<uint64_t>(GetBlockIdx());
        return poolTile * TileElements + (sequence % 8U) * (TileElements / 8U);
    }

    __aicore__ inline void ComputePacked(uint32_t count)
    {
        LocalTensor<float> x1 = x1Queue_.template DeQue<float>();
        LocalTensor<float> x2 = x2Queue_.template DeQue<float>();
        LocalTensor<uint8_t> mask = yQueue_.template AllocTensor<uint8_t>();
        const uint32_t n = (count + 255U) & ~255U;
        SetMaskNorm();
        ResetMask();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Compare(mask, x1, x2, CMPMODE::GT, n);
#else
        vcmpv_gt(reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(x1.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(x2.GetPhyAddr()),
            static_cast<uint8_t>(n / 64U), 1U, 1U, 1U, 1U, 8U, 8U);
#endif
        yQueue_.EnQue(mask);
        x1Queue_.FreeTensor(x1);
        x2Queue_.FreeTensor(x2);
    }

    __aicore__ inline void CopyOutPacked(uint64_t, uint32_t count)
    {
        LocalTensor<uint8_t> mask = yQueue_.template DeQue<uint8_t>();
        const DataCopyExtParams copy(1U, (count + 7U) / 8U, 0U, 0U, 0U);
        DataCopyPad(yGlobal_[PoolOffset(packedSequence_)], mask, copy);
        ++packedSequence_;
        yQueue_.FreeTensor(mask);
    }

    __aicore__ inline void CacheOwnMasks()
    {
        const uint64_t lastTile = static_cast<uint64_t>(GetBlockIdx()) +
            (ownedTiles_ - 1U) * static_cast<uint64_t>(GetBlockNum());
        const uint64_t remaining = tiling_->totalElements - lastTile * TileElements;
        const uint64_t lastCount = remaining < TileElements ? remaining : TileElements;
        maskBytes_ = (ownedTiles_ - 1U) * (TileElements / 8U) + (lastCount + 7U) / 8U;
        LocalTensor<float> storage = x1Queue_.template AllocTensor<float>();
        for (uint64_t offset = 0U; offset < maskBytes_; offset += TileElements) {
            const uint32_t bytes = static_cast<uint32_t>(maskBytes_ - offset < TileElements ? maskBytes_ - offset : TileElements);
            const DataCopyExtParams copy(1U, bytes, 0U, 0U, 0U);
            const DataCopyPadExtParams<uint8_t> pad(true, 0U,
                static_cast<uint8_t>((32U - (bytes & 31U)) & 31U), static_cast<uint8_t>(0));
            DataCopyPad(storage.ReinterpretCast<uint8_t>()[offset],
                yGlobal_[PoolOffset((offset / TileElements) * 8U)], copy, pad);
        }
        x1Queue_.EnQue(storage);
        ownedMaskStorage_ = x1Queue_.template DeQue<float>();
    }

    __aicore__ inline void ExpandStage()
    {
        LocalTensor<float> scratch = x2Queue_.template AllocTensor<float>();
        LocalTensor<half> selected = scratch.ReinterpretCast<half>();
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        Duplicate(values, static_cast<half>(1), 128U);
        Duplicate(selectionScalarBuffer_.Get<half>(), static_cast<half>(0), 16U);
        PipeBarrier<PIPE_V>();
        for (uint64_t sequence = 0U; sequence < ownedTiles_; ++sequence) {
            const uint64_t tile = static_cast<uint64_t>(GetBlockIdx()) + sequence * static_cast<uint64_t>(GetBlockNum());
            const uint64_t offset = tile * TileElements;
            const uint32_t count = static_cast<uint32_t>(tiling_->totalElements - offset < TileElements ? tiling_->totalElements - offset : TileElements);
            const uint32_t n = (count + 255U) & ~255U;
            LocalTensor<uint8_t> mask = ownedMaskStorage_.ReinterpretCast<uint8_t>()[sequence * (TileElements / 8U)];
            LocalTensor<uint8_t> output = yQueue_.template AllocTensor<uint8_t>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
            const BinaryRepeatParams selectParams(1U, 1U, 1U, 8U, 0U, 0U);
            Select(selected, mask, values, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   128ULL, static_cast<uint8_t>(n / 128U), selectParams);
            PipeBarrier<PIPE_V>();
            Cast(output.ReinterpretCast<int8_t>(), selected, RoundMode::CAST_RINT, n);
#else
            SetMaskNorm();
            SetCmpMask(selectionScalarBuffer_.Get<half>());
            SetVectorMask<half>(128);
            PipeBarrier<PIPE_V>();
            vsel(reinterpret_cast<__ubuf__ half*>(selected.GetPhyAddr()),
                reinterpret_cast<__ubuf__ half*>(values.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr()),
                static_cast<uint8_t>(n / 128U), 1U, 1U, 1U, 8U, 0U, 0U,
                static_cast<uint8_t>(SELMODE::VSEL_TENSOR_SCALAR_MODE));
            PipeBarrier<PIPE_V>();
            vconv_f162s8r(reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
                reinterpret_cast<__ubuf__ half*>(selected.GetPhyAddr()),
                static_cast<uint8_t>(n / 128U), 1U, 1U, 4U, 8U);
#endif
            yQueue_.EnQue(output);
            CopyOut(offset, count);
        }
        x2Queue_.FreeTensor(scratch);
        x1Queue_.FreeTensor(ownedMaskStorage_);
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<float> x1 = x1Queue_.template AllocTensor<float>();
        LocalTensor<float> x2 = x2Queue_.template AllocTensor<float>();
        if constexpr (FastCopy) {
            if ((count & 7U) == 0U) {
                const DataCopyParams copy(
                    1U, static_cast<uint16_t>(count / 8U), 0U, 0U);
                DataCopy(x1, x1Global_[offset], copy);
                DataCopy(x2, x2Global_[offset], copy);
            } else {
                CopyInPadded(x1, x2, offset, count);
            }
        } else {
            CopyInPadded(x1, x2, offset, count);
        }
        x1Queue_.EnQue(x1);
        x2Queue_.EnQue(x2);
    }

    __aicore__ inline void CopyInPadded(
        LocalTensor<float> x1, LocalTensor<float> x2,
        uint64_t offset, uint32_t count)
    {
        const DataCopyExtParams copy(
            1U, count * sizeof(float), 0U, 0U, 0U);
        const DataCopyPadExtParams<float> pad(
            false, 0U, 0U, static_cast<float>(0));
        DataCopyPad(x1, x1Global_[offset], copy, pad);
        DataCopyPad(x2, x2Global_[offset], copy, pad);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        const uint32_t vectorCount = (count + 255U) & ~255U;
        LocalTensor<float> x1 = x1Queue_.template DeQue<float>();
        LocalTensor<float> x2 = x2Queue_.template DeQue<float>();
        LocalTensor<uint8_t> output =
            yQueue_.template AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        LocalTensor<half> selected = x1.ReinterpretCast<half>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Compare(mask, x1, x2, CMPMODE::GT, vectorCount);
        PipeBarrier<PIPE_V>();
        const uint8_t repeats = static_cast<uint8_t>(
            vectorCount / kSelectRepeatElements);
        const BinaryRepeatParams selectParams(
            1U, 1U, 1U, 8U, 0U, 0U);
        Select(selected, mask, values, static_cast<half>(0),
               SELMODE::VSEL_TENSOR_SCALAR_MODE,
               static_cast<uint64_t>(kSelectRepeatElements), repeats,
               selectParams);
        PipeBarrier<PIPE_V>();
        Cast(output.ReinterpretCast<int8_t>(), selected,
             RoundMode::CAST_RINT, vectorCount);
#else
        const uint8_t compareRepeats = static_cast<uint8_t>(
            vectorCount / 64U);
        vcmpv_gt(
            reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(x1.GetPhyAddr()),
            reinterpret_cast<__ubuf__ float*>(x2.GetPhyAddr()),
            compareRepeats, 1U, 1U, 1U, 1U, 8U, 8U);
        PipeBarrier<PIPE_V>();
        SetCmpMask(selectionScalarBuffer_.Get<half>());
        SetVectorMask<half>(static_cast<int32_t>(kSelectRepeatElements));
        PipeBarrier<PIPE_V>();
        const uint8_t selectRepeats = static_cast<uint8_t>(
            vectorCount / kSelectRepeatElements);
        vsel(reinterpret_cast<__ubuf__ half*>(selected.GetPhyAddr()),
             reinterpret_cast<__ubuf__ half*>(values.GetPhyAddr()),
             reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr()),
             selectRepeats, 1U, 1U, 1U, 8U, 0U, 0U,
             static_cast<uint8_t>(SELMODE::VSEL_TENSOR_SCALAR_MODE));
        PipeBarrier<PIPE_V>();
        vconv_f162s8r(
            reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            reinterpret_cast<__ubuf__ half*>(selected.GetPhyAddr()),
            selectRepeats, 1U, 1U, 4U, 8U);
#endif
        yQueue_.EnQue(output);
        x1Queue_.FreeTensor(x1);
        x2Queue_.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<uint8_t> output =
            yQueue_.template DeQue<uint8_t>();
        if constexpr (FastCopy) {
            if ((count & 31U) == 0U) {
                const DataCopyParams copy(
                    1U, static_cast<uint16_t>(count / 32U), 0U, 0U);
                DataCopy(yGlobal_[offset], output, copy);
            } else {
                CopyOutPadded(output, offset, count);
            }
        } else {
            CopyOutPadded(output, offset, count);
        }
        yQueue_.FreeTensor(output);
    }

    __aicore__ inline void CopyOutPadded(
        LocalTensor<uint8_t> output, uint64_t offset, uint32_t count)
    {
        const DataCopyExtParams copy(1U, count, 0U, 0U, 0U);
        DataCopyPad(yGlobal_[offset], output, copy);
    }

    uint64_t ownedTiles_, packedSequence_, maskBytes_;
    LocalTensor<float> ownedMaskStorage_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2U> x1Queue_;
    TQue<QuePosition::VECIN, 2U> x2Queue_;
    TQue<QuePosition::VECOUT, 2U> yQueue_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectionScalarBuffer_;
    GlobalTensor<float> x1Global_;
    GlobalTensor<float> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterSmallDenseTilingData* tiling_;
};

template <uint32_t TileElements>
class KernelGreaterBfloat16EarlyRelease {
    static constexpr uint32_t kMaxCols = 2048U;
    static constexpr uint32_t kUbBytes = 11U * TileElements + TileElements / 8U +
        kMaxCols * 2U + 256U + 32U;
    static_assert(TileElements % 256U == 0U && TileElements / 64U <= 255U);
    static_assert(kUbBytes <= 188160U);
    struct Transaction { uint64_t outer, offset; uint32_t count; };
    struct Cursor { uint64_t tile, end, outer, index, base, local; };
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                               const GreaterCompactPeriodicTilingData& tiling)
    {
        tiling_ = &tiling;
        rowIsX1_ = tiling.matrixX1Pattern == S9_GREATER_MATRIX_ROW_BROADCAST;
        rowGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rowIsX1_ ? x1 : x2));
        denseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(rowIsX1_ ? x2 : x1));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if (tiling.totalElements >= (1024ULL * 1024ULL + 2ULL) / 3ULL) {
            denseGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        cols_ = static_cast<uint32_t>(tiling.matrixCols);
        alignedCols_ = (cols_ + 7U) & ~7U;
        rowsPerTile_ = TileElements / alignedCols_;
        if (tiling.matrixRows < rowsPerTile_) {
            rowsPerTile_ = static_cast<uint32_t>(tiling.matrixRows);
        }
        span_ = tiling.matrixRows * cols_;
        fullCount_ = rowsPerTile_ * cols_;
        tilesPerOuter_ = tiling.matrixRows / rowsPerTile_ +
            (tiling.matrixRows % rowsPerTile_ != 0U ? 1U : 0U);
        pipe_.InitBuffer(rawRowBuffer_, kMaxCols * sizeof(bfloat16_t));
        pipe_.InitBuffer(rowCacheBuffer_, TileElements * sizeof(float));
        pipe_.InitBuffer(convertedBuffer_, TileElements * sizeof(float));
        pipe_.InitBuffer(inputQueue_, 1U, TileElements * sizeof(bfloat16_t));
        pipe_.InitBuffer(outputQueue_, 1U, TileElements);
        pipe_.InitBuffer(maskBuffer_, TileElements / 8U);
        pipe_.InitBuffer(valuesBuffer_, 256U);
        pipe_.InitBuffer(scalarBuffer_, 32U);
    }

    __aicore__ inline void Process()
    {
        if (rowIsX1_) ProcessRows<true>();
        else ProcessRows<false>();
    }
private:
    __aicore__ inline uint64_t RowOffset(uint64_t outer) const
    {
        return outer * cols_;
    }

    __aicore__ inline bool Next(Cursor& cursor, Transaction& transaction) const
    {
        if (cursor.tile >= cursor.end) return false;
        transaction.outer = cursor.outer;
        transaction.offset = cursor.base + cursor.local;
        const uint64_t remaining = span_ - cursor.local;
        transaction.count = static_cast<uint32_t>(remaining < fullCount_ ? remaining : fullCount_);
        ++cursor.tile;
        ++cursor.index;
        cursor.local += fullCount_;
        if (cursor.index == tilesPerOuter_) {
            cursor.index = 0U;
            ++cursor.outer;
            cursor.base += span_;
            cursor.local = 0U;
        }
        return true;
    }

    __aicore__ inline void PrepareRow(uint64_t outer)
    {
        PipeBarrier<PIPE_V>();
        LocalTensor<bfloat16_t> raw = rawRowBuffer_.Get<bfloat16_t>();
        const uint32_t rawAligned = (cols_ + 15U) & ~15U;
        const DataCopyExtParams copy(1U, cols_ * sizeof(bfloat16_t), 0U, 0U, 0U);
        const DataCopyPadExtParams<bfloat16_t> pad(true, 0U,
            static_cast<uint8_t>(rawAligned - cols_), static_cast<bfloat16_t>(0));
        DataCopyPad(raw, rowGlobal_[RowOffset(outer)], copy, pad);
        const event_t ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready);
        WaitFlag<HardEvent::MTE2_V>(ready);
        LocalTensor<float> row = rowCacheBuffer_.Get<float>();
        LocalTensor<float> scratch = convertedBuffer_.Get<float>();
        Cast(row, raw, RoundMode::CAST_NONE, alignedCols_);
        PipeBarrier<PIPE_V>();
        const uint32_t blocks = alignedCols_ / 8U;
        const BinaryRepeatParams repeatParams(static_cast<uint8_t>(blocks), 0U, 0U, 1U, 1U, 0U);
        LocalTensor<uint16_t> zeros = scalarBuffer_.Get<uint16_t>();
        SetMaskNorm();
        SetVectorMask<uint16_t, MaskMode::NORMAL>(128U);
        uint32_t destination = 0U;
        for (uint32_t group = 0U; group < rowsPerTile_ / 8U; ++group) {
            Or<uint16_t, false>(scratch[destination].ReinterpretCast<uint16_t>(),
                row.ReinterpretCast<uint16_t>(), zeros, MASK_PLACEHOLDER,
                static_cast<uint8_t>(blocks), repeatParams);
            destination += 8U * alignedCols_;
        }
        const uint32_t tail = rowsPerTile_ % 8U;
        if (tail != 0U) {
            SetVectorMask<uint16_t, MaskMode::NORMAL>(tail * 16U);
            Or<uint16_t, false>(scratch[destination].ReinterpretCast<uint16_t>(),
                row.ReinterpretCast<uint16_t>(), zeros, MASK_PLACEHOLDER,
                static_cast<uint8_t>(blocks), repeatParams);
        }
        PipeBarrier<PIPE_V>();
        const GatherMaskParams gather = {1U, static_cast<uint16_t>(rowsPerTile_),
            static_cast<uint16_t>(blocks), 0U};
        uint64_t reserved = 0U;
        GatherMask(row, scratch, 7U, true, cols_, gather, reserved);
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        SetCmpMask(scalarBuffer_.Get<half>());
        SetVectorMask<half>(128);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Prefetch(const Transaction& transaction)
    {
        LocalTensor<bfloat16_t> input = inputQueue_.AllocTensor<bfloat16_t>();
        const DataCopyExtParams copy(1U, transaction.count * sizeof(bfloat16_t), 0U, 0U, 0U);
        const DataCopyPadExtParams<bfloat16_t> pad(false, 0U, 0U, static_cast<bfloat16_t>(0));
        DataCopyPad(input, denseGlobal_[transaction.offset], copy, pad);
        inputQueue_.EnQue(input);
    }

    template <bool RowIsX1>
    __aicore__ inline void Compute(LocalTensor<bfloat16_t> input, uint32_t count,
                                   const Transaction& next, bool hasNext)
    {
        const uint32_t vectorCount = (count + 255U) & ~255U;
        LocalTensor<float> dense = convertedBuffer_.Get<float>();
        LocalTensor<float> row = rowCacheBuffer_.Get<float>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        LocalTensor<half> values = valuesBuffer_.Get<half>();
        LocalTensor<half> selected = dense.ReinterpretCast<half>();
        LocalTensor<uint8_t> output = outputQueue_.AllocTensor<uint8_t>();
        SetMaskNorm();
        SetVectorMask<half>(128);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(dense, input, RoundMode::CAST_NONE, vectorCount);
#else
        auto* rawDense = reinterpret_cast<__ubuf__ float*>(dense.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ float*>(row.GetPhyAddr());
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr());
        const uint8_t repeats = static_cast<uint8_t>(vectorCount / 64U);
        vconv_bf162f32(rawDense, reinterpret_cast<__ubuf__ bfloat16_t*>(input.GetPhyAddr()),
            repeats, 1U, 1U, 8U, 4U);
#endif
        PipeBarrier<PIPE_V>();

        inputQueue_.FreeTensor(input);
        if (hasNext) Prefetch(next);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        if constexpr (RowIsX1) Compare(mask, row, dense, CMPMODE::GT, vectorCount);
        else Compare(mask, dense, row, CMPMODE::GT, vectorCount);
#else
        if constexpr (RowIsX1) vcmpv_gt(rawMask, rawRow, rawDense, repeats, 1U, 1U, 1U, 1U, 8U, 8U);
        else vcmpv_gt(rawMask, rawDense, rawRow, repeats, 1U, 1U, 1U, 1U, 8U, 8U);
#endif
        PipeBarrier<PIPE_V>();
        const uint8_t selectRepeats = static_cast<uint8_t>(vectorCount / 128U);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        SetVectorMask<half>(128);
        const BinaryRepeatParams selectParams(1U, 1U, 1U, 8U, 0U, 0U);
        Select(selected, mask, values, selectRepeats, selectParams);
#else
        auto* rawSelected = reinterpret_cast<__ubuf__ half*>(selected.GetPhyAddr());
        vsel(rawSelected, reinterpret_cast<__ubuf__ half*>(values.GetPhyAddr()), rawMask,
            selectRepeats, 1U, 1U, 1U, 8U, 0U, 0U,
            static_cast<uint8_t>(SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.ReinterpretCast<int8_t>(), selected, RoundMode::CAST_RINT, vectorCount);
#else
        vconv_f162s8r(reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            rawSelected, selectRepeats, 1U, 1U, 4U, 8U);
#endif
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(const Transaction& transaction)
    {
        LocalTensor<uint8_t> output = outputQueue_.DeQue<uint8_t>();
        const DataCopyExtParams copy(1U, transaction.count, 0U, 0U, 0U);
        DataCopyPad(yGlobal_[transaction.offset], output, copy);
        outputQueue_.FreeTensor(output);
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t core = GetBlockIdx(), cores = GetBlockNum();
        const uint64_t total = tiling_->matrixOuterCount * tilesPerOuter_;
        const uint64_t begin = (total / cores) * core + (core < total % cores ? core : total % cores);
        const uint64_t end = begin + total / cores + (core < total % cores ? 1U : 0U);
        Cursor cursor{begin, end, begin / tilesPerOuter_, begin % tilesPerOuter_, 0U, 0U};
        cursor.base = cursor.outer * span_;
        cursor.local = cursor.index * fullCount_;
        Transaction current;
        if (!Next(cursor, current)) return;
        Duplicate(valuesBuffer_.Get<half>(), static_cast<half>(1), 128U);
        Duplicate(scalarBuffer_.Get<half>(), static_cast<half>(0), 16U);
        Duplicate(rowCacheBuffer_.Get<float>(), static_cast<float>(0), TileElements);
        PipeBarrier<PIPE_V>();
        PrepareRow(current.outer);
        uint64_t cachedOuter = current.outer;
        Prefetch(current);
        while (true) {
            if (current.outer != cachedOuter) {
                PrepareRow(current.outer);
                cachedOuter = current.outer;
            }
            LocalTensor<bfloat16_t> input = inputQueue_.DeQue<bfloat16_t>();
            Transaction next;
            const bool hasNext = Next(cursor, next);
            Compute<RowIsX1>(input, current.count, next, hasNext);
            CopyOut(current);
            if (!hasNext) break;
            current = next;
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawRowBuffer_, rowCacheBuffer_, convertedBuffer_;
    TBuf<TPosition::VECCALC> maskBuffer_, valuesBuffer_, scalarBuffer_;
    TQue<QuePosition::VECIN, 1U> inputQueue_;
    TQue<QuePosition::VECOUT, 1U> outputQueue_;
    GlobalTensor<bfloat16_t> rowGlobal_, denseGlobal_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterCompactPeriodicTilingData* tiling_;
    uint64_t span_, tilesPerOuter_;
    uint32_t cols_, alignedCols_, rowsPerTile_, fullCount_;
    bool rowIsX1_;
};

template <uint32_t TileElements>
class KernelGreaterInt8DenseManual {
    static_assert(TileElements % 256U == 0U);
    static_assert(TileElements / 128U <= 255U);
    static_assert(10U * TileElements + TileElements / 8U + 256U <= 188160U);
public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                               const GreaterSmallDenseTilingData& tiling)
    {
        tiling_ = &tiling;
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if (tiling.totalElements >= (32ULL * 1024ULL * 1024ULL + 2ULL) / 3ULL) {
            x1Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            x2Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        pipe_.InitBuffer(x1Queue_, 2U, TileElements);
        pipe_.InitBuffer(x2Queue_, 2U, TileElements);
        pipe_.InitBuffer(yQueue_, 2U, TileElements);
        pipe_.InitBuffer(halfX1Buffer_, 2U * TileElements);
        pipe_.InitBuffer(halfX2Buffer_, 2U * TileElements);
        pipe_.InitBuffer(maskBuffer_, TileElements / 8U);
        pipe_.InitBuffer(valuesBuffer_, 256U);
    }

    __aicore__ inline void Process()
    {
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t average = tiling_->totalElements / blocks +
            (tiling_->totalElements % blocks != 0U ? 1U : 0U);

        const uint64_t blockElements = (average + 255U) & ~255ULL;
        const uint64_t start = block * blockElements;
        const uint64_t limit = start + blockElements;
        const uint64_t end = limit < tiling_->totalElements ? limit : tiling_->totalElements;
        if (start >= end) {
            return;
        }
        Duplicate(valuesBuffer_.Get<half>(), static_cast<half>(1), 128U);
        PipeBarrier<PIPE_V>();
        uint64_t current = start;
        uint32_t count = static_cast<uint32_t>(
            end - current < TileElements ? end - current : TileElements);
        CopyIn(current, count);
        while (current < end) {
            const uint64_t next = current + count;
            uint32_t nextCount = 0U;
            if (next < end) {
                nextCount = static_cast<uint32_t>(
                    end - next < TileElements ? end - next : TileElements);
                CopyIn(next, nextCount);
            }
            Compute(count);
            CopyOut(current, count);
            current = next;
            count = nextCount;
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<int8_t> x1 = x1Queue_.AllocTensor<int8_t>();
        LocalTensor<int8_t> x2 = x2Queue_.AllocTensor<int8_t>();
        if ((count & 255U) != 0U) {

            PipeBarrier<PIPE_V>();
            const uint32_t tailBlock = count & ~255U;
            Duplicate(x1[tailBlock].ReinterpretCast<int32_t>(), 0, 64U);
            Duplicate(x2[tailBlock].ReinterpretCast<int32_t>(), 0, 64U);
            const event_t event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(event);
            WaitFlag<HardEvent::V_MTE2>(event);
        }
        if ((count & 31U) == 0U) {
            const DataCopyParams copy(1U, static_cast<uint16_t>(count / 32U), 0U, 0U);
            DataCopy(x1, x1Global_[offset], copy);
            DataCopy(x2, x2Global_[offset], copy);
        } else {
            const DataCopyExtParams copy(1U, count, 0U, 0U, 0U);
            const DataCopyPadExtParams<int8_t> pad(false, 0U, 0U, 0);
            DataCopyPad(x1, x1Global_[offset], copy, pad);
            DataCopyPad(x2, x2Global_[offset], copy, pad);
        }
        x1Queue_.EnQue(x1);
        x2Queue_.EnQue(x2);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        const uint32_t vectorCount = (count + 255U) & ~255U;
        const uint8_t repeats = static_cast<uint8_t>(vectorCount / 128U);
        LocalTensor<int8_t> x1 = x1Queue_.DeQue<int8_t>();
        LocalTensor<int8_t> x2 = x2Queue_.DeQue<int8_t>();
        LocalTensor<half> halfX1 = halfX1Buffer_.Get<half>();
        LocalTensor<half> halfX2 = halfX2Buffer_.Get<half>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>();
        LocalTensor<half> values = valuesBuffer_.Get<half>();
        LocalTensor<uint8_t> output = yQueue_.AllocTensor<uint8_t>();

        Cast(halfX1, x1, RoundMode::CAST_NONE, vectorCount);
        Cast(halfX2, x2, RoundMode::CAST_NONE, vectorCount);
        PipeBarrier<PIPE_V>();
        Compare(mask, halfX1, halfX2, CMPMODE::GT, vectorCount);
        PipeBarrier<PIPE_V>();

        const BinaryRepeatParams selectParams(1U, 1U, 1U, 8U, 0U, 0U);
        Select(halfX1, mask, values, static_cast<half>(0),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, 128ULL, repeats, selectParams);
        PipeBarrier<PIPE_V>();
        Cast(output.ReinterpretCast<int8_t>(), halfX1, RoundMode::CAST_RINT, vectorCount);
        yQueue_.EnQue(output);
        x1Queue_.FreeTensor(x1);
        x2Queue_.FreeTensor(x2);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        LocalTensor<uint8_t> output = yQueue_.DeQue<uint8_t>();
        if ((count & 31U) == 0U) {
            const DataCopyParams copy(1U, static_cast<uint16_t>(count / 32U), 0U, 0U);
            DataCopy(yGlobal_[offset], output, copy);
        } else {
            const DataCopyExtParams copy(1U, count, 0U, 0U, 0U);
            DataCopyPad(yGlobal_[offset], output, copy);
        }
        yQueue_.FreeTensor(output);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2U> x1Queue_;
    TQue<QuePosition::VECIN, 2U> x2Queue_;
    TQue<QuePosition::VECOUT, 2U> yQueue_;
    TBuf<TPosition::VECCALC> halfX1Buffer_;
    TBuf<TPosition::VECCALC> halfX2Buffer_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> valuesBuffer_;
    GlobalTensor<int8_t> x1Global_;
    GlobalTensor<int8_t> x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterSmallDenseTilingData* tiling_;
};

class KernelGreaterInt32ChunkPredicates {
    static constexpr uint32_t kTile = 8192U;
    static constexpr uint32_t kDenseElements = kTile + 64U;
    static constexpr uint32_t kMaxCols = 1024U;
    static constexpr uint32_t kPeriodicElements = kTile + 2U * kMaxCols;
    static constexpr uint32_t kOutputElements = kTile;
    static constexpr uint32_t kCachedTiles = 72U;
    static constexpr uint32_t kOutputGroup = 4U;
    static constexpr uint32_t kUbBytes = kMaxCols * 4U + kPeriodicElements * 4U +
        2U * kDenseElements * 4U + kCachedTiles * (kTile / 8U) + 256U + 32U;
    static_assert(kUbBytes == 185120U && kUbBytes <= 188160U);
    static_assert(kOutputGroup <= 4U);

    struct Transaction {
        uint64_t outer;
        uint64_t localOffset;
        uint64_t outputOffset;
        uint32_t count;
    };

public:
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                               const GreaterCompactPeriodicTilingData& tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        tiling_ = &tiling;
        rowIsX1_ = tiling.matrixX1Pattern == S9_GREATER_MATRIX_ROW_BROADCAST;
        rowGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            rowIsX1_ ? x1 : x2));
        denseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            rowIsX1_ ? x2 : x1));

        if (tiling.totalElements >= (32ULL * 1024ULL * 1024ULL + 4ULL) / 5ULL) {
            denseGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        cols_ = static_cast<uint32_t>(tiling.matrixCols);
        span_ = tiling.matrixRows * tiling.matrixCols;
        slotsPerOuter_ = 1U + span_ / kOutputElements +
            (span_ % kOutputElements != 0U ? 1U : 0U);
        pipe_->InitBuffer(compactRowBuffer_, kMaxCols * sizeof(int32_t));
        pipe_->InitBuffer(periodicRowBuffer_, kPeriodicElements * sizeof(int32_t));
        pipe_->InitBuffer(denseQueue_, 2U, kDenseElements * sizeof(int32_t));
        pipe_->InitBuffer(maskBuffer_, kCachedTiles * (kTile / 8U));
        pipe_->InitBuffer(selectionValueBuffer_, 256U);
        pipe_->InitBuffer(selectionScalarBuffer_, 32U);
    }

    __aicore__ inline void Process()
    {
        if (rowIsX1_) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ inline uint64_t RowInputOffset(uint64_t outer) const
    {

        return outer * cols_;
    }

    __aicore__ inline bool NextTransaction(
        uint64_t& cursor, uint64_t end, Transaction& transaction) const
    {
        while (cursor < end) {
            const uint64_t slot = cursor++;
            const uint64_t outer = slot / slotsPerOuter_;
            const uint64_t index = slot % slotsPerOuter_;
            const uint64_t base = outer * span_;
            const uint32_t head = static_cast<uint32_t>(
                (512U - (base & 511U)) & 511U);
            const uint64_t local = index == 0U ? 0U :
                head + (index - 1U) * kOutputElements;
            const uint64_t remaining = local < span_ ? span_ - local : 0U;
            const uint32_t count = index == 0U ? head :
                static_cast<uint32_t>(remaining < kOutputElements ?
                    remaining : kOutputElements);
            if (count == 0U) {
                continue;
            }
            transaction.outer = outer;
            transaction.localOffset = local;
            transaction.outputOffset = base + local;
            transaction.count = count;
            return true;
        }
        return false;
    }

    __aicore__ inline void LoadAndExpandRow(uint64_t outer)
    {

        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> compact = compactRowBuffer_.Get<int32_t>();
        LocalTensor<int32_t> padded = denseQueue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> periodic = periodicRowBuffer_.Get<int32_t>();
        LocalTensor<uint16_t> zeros = selectionValueBuffer_.Get<uint16_t>();
        const uint32_t alignedCols = (cols_ + 7U) & ~7U;
        Duplicate(compact, static_cast<int32_t>(0), alignedCols);
        const event_t vectorToMte2 = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vectorToMte2);
        WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
        const DataCopyExtParams copy(1U, cols_ * sizeof(int32_t), 0U, 0U, 0U);
        const DataCopyPadExtParams<int32_t> pad(
            true, 0U, static_cast<uint8_t>(alignedCols - cols_), 0);
        DataCopyPad(compact, rowGlobal_[RowInputOffset(outer)], copy, pad);
        const event_t mte2ToVector = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);

        const uint32_t cacheRows = (kTile + 2U * cols_ - 2U) / cols_;
        const uint32_t blocksPerRow = alignedCols / 8U;
        const uint32_t paddedBlocksPerRow = blocksPerRow + ((blocksPerRow & 15U) == 0U ? 1U : 0U);
        const BinaryRepeatParams repeatParams(
            static_cast<uint8_t>(paddedBlocksPerRow), 0U, 0U, 1U, 1U, 0U);
        uint32_t remainingRows = cacheRows;
        uint32_t outputOffset = 0U;
        while (remainingRows != 0U) {
            const uint32_t groupRows = remainingRows < 8U ? remainingRows : 8U;
            SetMaskNorm();
            SetVectorMask<uint16_t, MaskMode::NORMAL>(groupRows * 16U);
            Or<uint16_t, false>(padded.ReinterpretCast<uint16_t>(),
                compact.ReinterpretCast<uint16_t>(), zeros, MASK_PLACEHOLDER,
                static_cast<uint8_t>(blocksPerRow), repeatParams);
            PipeBarrier<PIPE_V>();
            const GatherMaskParams gatherParams = {1U, static_cast<uint16_t>(groupRows),
                static_cast<uint16_t>(paddedBlocksPerRow), 0U};
            uint64_t reservedCount = 0U;

            GatherMask(periodic[outputOffset], padded, 7U, true, cols_, gatherParams, reservedCount);
            PipeBarrier<PIPE_V>();
            outputOffset += groupRows * cols_;
            remainingRows -= groupRows;
        }
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        SetCmpMask(selectionScalarBuffer_.Get<half>());
        SetVectorMask<half>(128);
        PipeBarrier<PIPE_V>();
        denseQueue_.FreeTensor(padded);
    }

    __aicore__ inline void PrefetchDense(
        const Transaction& transaction, uint32_t offset)
    {
        const uint32_t remaining = transaction.count - offset;
        const uint32_t count = remaining < kTile ? remaining : kTile;
        LocalTensor<int32_t> dense = denseQueue_.AllocTensor<int32_t>();
        if ((count & 255U) != 0U) {
            Duplicate(dense[count & ~255U], static_cast<int32_t>(0), 256U);
            const event_t vectorToMte2 = static_cast<event_t>(
                pipe_->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(vectorToMte2);
            WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
        }
        const DataCopyParams copy(
            1U, static_cast<uint16_t>(count / 8U), 0U, 0U);
        DataCopy(dense, denseGlobal_[transaction.outputOffset + offset], copy);
        denseQueue_.EnQue(dense);
    }

    template <bool RowIsX1>
    __aicore__ inline void MaskPart(LocalTensor<int32_t> dense,
                                   LocalTensor<uint8_t> mask,
                                   uint32_t phase, uint32_t count)
    {
        const uint32_t vectorCount = (count + 255U) & ~255U;
        const uint8_t compareRepeats = static_cast<uint8_t>(vectorCount / 64U);
        LocalTensor<int32_t> row = periodicRowBuffer_.Get<int32_t>()[phase];
        SetMaskNorm();
        SetVectorMask<half>(128);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        const BinaryRepeatParams binaryParams(1U, 1U, 1U, 8U, 8U, 8U);
        if constexpr (RowIsX1) {
            Min(dense, dense, row, 64ULL, compareRepeats, binaryParams);
        } else {
            Max(dense, dense, row, 64ULL, compareRepeats, binaryParams);
        }
#else
        auto* rawDense = reinterpret_cast<__ubuf__ int32_t*>(dense.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ int32_t*>(row.GetPhyAddr());
        if constexpr (RowIsX1) {
            vmin(rawDense, rawDense, rawRow,
                compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        } else {
            vmax(rawDense, rawDense, rawRow,
                compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        }
#endif
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Compare(mask, dense, row, CMPMODE::EQ, 64ULL, compareRepeats, binaryParams);
#else
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr());
        vcmpv_eq(rawMask, rawDense, rawRow,
            compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
#endif
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SelectPart(LocalTensor<uint8_t> mask,
                                       LocalTensor<uint8_t> output, uint32_t count)
    {
        const uint32_t vectorCount = (count + 255U) & ~255U;
        const uint8_t selectRepeats = static_cast<uint8_t>(vectorCount / 128U);
        LocalTensor<half> selected = periodicRowBuffer_.Get<half>();
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        SetMaskNorm();
        SetCmpMask(selectionScalarBuffer_.Get<half>());
        SetVectorMask<half>(128);
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        const BinaryRepeatParams selectParams(1U, 1U, 1U, 8U, 0U, 0U);
        Select(selected, mask, values, selectRepeats, selectParams);
#else
        auto* rawSelected = reinterpret_cast<__ubuf__ half*>(selected.GetPhyAddr());
        auto* rawValues = reinterpret_cast<__ubuf__ half*>(values.GetPhyAddr());
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr());
        vsel(rawSelected, rawValues, rawMask, selectRepeats,
            1U, 1U, 1U, 8U, 0U, 0U,
            static_cast<uint8_t>(SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.ReinterpretCast<int8_t>(), selected, RoundMode::CAST_RINT, vectorCount);
#else
        vconv_f162s8r(reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            rawSelected, selectRepeats, 1U, 1U, 4U, 8U);
#endif

        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StartExpansion()
    {
        outputStorage_[0] = denseQueue_.AllocTensor<int32_t>();
        outputStorage_[1] = denseQueue_.AllocTensor<int32_t>();
        outputSequence_ = 0U; groupParts_ = 0U; groupCount_ = 0U;
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            outputEvent_[slot] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE3_V>());
            outputPending_[slot] = false;
        }
    }

    __aicore__ inline void WaitOutput(uint32_t slot)
    {
        if (outputPending_[slot]) {
            WaitFlag<HardEvent::MTE3_V>(outputEvent_[slot]);
            outputPending_[slot] = false;
        }
    }

    __aicore__ inline void ExpandAndStore(const Transaction& transaction, uint32_t sequence, bool last)
    {
        const uint32_t slot = outputSequence_ & 1U;
        if (groupParts_ == 0U) {
            WaitOutput(slot);
            groupOffset_ = transaction.outputOffset;
            groupCount_ = 0U;
        }
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>()[sequence * (kTile / 8U)];
        LocalTensor<uint8_t> output = outputStorage_[slot].ReinterpretCast<uint8_t>()[groupParts_ * kTile];
        SelectPart(mask, output, transaction.count);
        groupCount_ += transaction.count;
        ++groupParts_;

        if (groupParts_ == kOutputGroup || transaction.count < kTile || last) {
            const event_t ready = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready); WaitFlag<HardEvent::V_MTE3>(ready);
            const DataCopyExtParams copy(1U, groupCount_, 0U, 0U, 0U);
            DataCopyPad(yGlobal_[groupOffset_], outputStorage_[slot].ReinterpretCast<uint8_t>(), copy);
            SetFlag<HardEvent::MTE3_V>(outputEvent_[slot]); outputPending_[slot] = true;
            ++outputSequence_; groupParts_ = 0U;
        }
    }

    __aicore__ inline void FinishExpansion()
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            WaitOutput(slot);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(outputEvent_[slot]);
            denseQueue_.FreeTensor(outputStorage_[slot]);
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t core = GetBlockIdx(), cores = GetBlockNum();
        const uint64_t slots = tiling_->matrixOuterCount * slotsPerOuter_;
        const uint64_t base = slots / cores, remainder = slots % cores;
        uint64_t cursor = base * core + (core < remainder ? core : remainder);
        const uint64_t end = cursor + base + (core < remainder ? 1U : 0U);
        Duplicate(selectionValueBuffer_.Get<half>(), static_cast<half>(0), 128U);
        Duplicate(selectionScalarBuffer_.Get<half>(), static_cast<half>(1), 16U);
        PipeBarrier<PIPE_V>();
        while (cursor < end) {
            const uint64_t batchBegin = cursor;
            Transaction current;
            if (!NextTransaction(cursor, end, current)) break;
            LoadAndExpandRow(current.outer);
            uint64_t cachedOuter = current.outer;
            PrefetchDense(current, 0U);
            uint32_t cached = 0U;
            while (true) {
                if (current.outer != cachedOuter) {

                    LoadAndExpandRow(current.outer);
                    cachedOuter = current.outer;
                }
                LocalTensor<int32_t> dense = denseQueue_.DeQue<int32_t>();
                Transaction next;
                const bool hasNext = cached + 1U < kCachedTiles && NextTransaction(cursor, end, next);
                if (hasNext) PrefetchDense(next, 0U);
                LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>()[cached * (kTile / 8U)];
                MaskPart<RowIsX1>(dense, mask, static_cast<uint32_t>(current.localOffset % cols_), current.count);
                denseQueue_.FreeTensor(dense);
                ++cached;
                if (!hasNext) break;
                current = next;
            }
            const uint64_t batchEnd = cursor;
            StartExpansion();
            uint64_t replay = batchBegin;
            for (uint32_t sequence = 0U; sequence < cached; ++sequence) {
                Transaction output;
                const bool valid = NextTransaction(replay, batchEnd, output);
                if (!valid) break;
                ExpandAndStore(output, sequence, sequence + 1U == cached);
            }
            FinishExpansion();

        }
    }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 2U> denseQueue_;
    TBuf<TPosition::VECCALC> compactRowBuffer_;
    TBuf<TPosition::VECCALC> periodicRowBuffer_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectionScalarBuffer_;
    GlobalTensor<int32_t> rowGlobal_;
    GlobalTensor<int32_t> denseGlobal_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterCompactPeriodicTilingData* tiling_;
    uint64_t span_;
    uint64_t slotsPerOuter_;
    uint32_t cols_;
    bool rowIsX1_;
    LocalTensor<int32_t> outputStorage_[2];
    event_t outputEvent_[2]; bool outputPending_[2];
    uint64_t groupOffset_; uint32_t groupCount_, groupParts_, outputSequence_;
};
class KernelGreaterInt32BitAlignedResident {
    static constexpr uint32_t kTile = 4096U;
    static constexpr uint32_t kDenseElements = kTile + 64U;
    static constexpr uint32_t kMaxCols = 1024U;
    static constexpr uint32_t kPeriodicElements = 5120U;
    static constexpr uint32_t kOutputElements = kTile;
    static constexpr uint32_t kMaskBytes = 131072U;
    static constexpr uint32_t kOutputGroup = 2U;
    static constexpr uint32_t kRawMaskBytes = 544U;
    static constexpr uint32_t kUbBytes = 2U*kDenseElements*4U +
        kPeriodicElements*4U + kMaskBytes + kRawMaskBytes + 512U + 288U;
    static_assert(kUbBytes == 186176U && kUbBytes <= 188160U);
    static_assert(kOutputGroup*(2U*kTile) <= kDenseElements*sizeof(int32_t));
    static_assert(4U*kTile <= kPeriodicElements*sizeof(int32_t));

    struct Transaction {
        uint64_t outer;
        uint64_t localOffset;
        uint64_t outputOffset;
        uint32_t count;
        uint32_t phase;
    };

    struct Cursor {
        uint64_t slot;
        uint64_t outer;
        uint64_t index;
        uint64_t base;
        uint64_t local;
        uint32_t head;
        uint32_t phase;
    };

public:
    __aicore__ inline static bool CanStage(const GreaterCompactPeriodicTilingData& t, uint64_t cores)
    {
        if (cores == 0U || t.matrixCols < 512U || t.matrixCols > kMaxCols ||
            t.matrixRows == 0U || t.matrixOuterCount == 0U) return false;
        const uint64_t cols=t.matrixCols, span=t.matrixRows*cols;
        const uint64_t body=span/kTile+(span%kTile!=0U), perOuter=1U+body;

        const uint64_t slots=t.matrixOuterCount*perOuter;
        const uint64_t owned=slots/cores+(slots%cores!=0U);
        if (owned > UINT64_MAX/512U || perOuter > UINT64_MAX/512U ||
            span > UINT64_MAX-255U) return false;
        const uint64_t gross=owned*512U;

        const uint64_t partial=2U*(perOuter-1U);
        const uint64_t full=owned>partial?(owned-partial)/perOuter:0U;
        const uint64_t outerBytes=((span+255U)/256U)*32U+32U;
        const uint64_t outerGross=perOuter*512U;
        const uint64_t saving=outerGross>outerBytes?outerGross-outerBytes:0U;
        if (gross-full*saving>kMaskBytes) return false;
        uint64_t gcd=span, rhs=512U;
        while(rhs!=0U) { const uint64_t r=gcd%rhs; gcd=rhs; rhs=r; }
        const uint64_t head=512U-gcd, step=kTile%cols;
        if (head>=cols || (step!=0U && body-1U>(cols-1U-head)/step)) return false;
        const uint64_t phase=head+(body-1U)*step;
        const uint64_t rows=kPeriodicElements/cols;

        return (phase&~7ULL)+kDenseElements<=rows*cols;
    }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                               const GreaterCompactPeriodicTilingData& tiling, TPipe* pipe)
    {
        pipe_ = pipe;
        tiling_ = &tiling;
        rowIsX1_ = tiling.matrixX1Pattern == S9_GREATER_MATRIX_ROW_BROADCAST;
        rowGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            rowIsX1_ ? x1 : x2));
        denseGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(
            rowIsX1_ ? x2 : x1));

        if (tiling.totalElements >= (32ULL * 1024ULL * 1024ULL + 4ULL) / 5ULL) {
            denseGlobal_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        cols_ = static_cast<uint32_t>(tiling.matrixCols);
        span_ = tiling.matrixRows * tiling.matrixCols;
        tileStep_=kTile%cols_;
        slotsPerOuter_ = 1U + span_ / kOutputElements +
            (span_ % kOutputElements != 0U ? 1U : 0U);
        pipe_->InitBuffer(periodicRowBuffer_, kPeriodicElements * sizeof(int32_t));
        pipe_->InitBuffer(denseQueue_, 2U, kDenseElements * sizeof(int32_t));
        pipe_->InitBuffer(maskBuffer_, kMaskBytes);
        pipe_->InitBuffer(rawMaskBuffer_, kRawMaskBytes);
        pipe_->InitBuffer(indexBuffer_, 512U);
        pipe_->InitBuffer(selectionValueBuffer_, 256U);
        pipe_->InitBuffer(selectionScalarBuffer_, 32U);
    }

    __aicore__ inline void Process()
    {
        if (rowIsX1_) {
            ProcessRows<true>();
        } else {
            ProcessRows<false>();
        }
    }

private:
    __aicore__ inline uint64_t RowInputOffset(uint64_t outer) const
    {

        return outer * cols_;
    }

    __aicore__ inline Cursor MakeCursor(uint64_t begin) const
    {
        Cursor cursor;
        cursor.slot = begin;
        cursor.outer = begin / slotsPerOuter_;
        cursor.index = begin % slotsPerOuter_;
        cursor.base = cursor.outer * span_;
        cursor.head = static_cast<uint32_t>((512U - (cursor.base & 511U)) & 511U);
        cursor.local = cursor.index == 0U ? 0U :
            cursor.head + (cursor.index - 1U) * kOutputElements;
        cursor.phase = static_cast<uint32_t>(cursor.local % cols_);
        return cursor;
    }

    __aicore__ inline bool NextTransaction(
        Cursor& cursor, uint64_t end, Transaction& transaction) const
    {
        while (cursor.slot < end) {
            const uint64_t outer = cursor.outer;
            const uint64_t local = cursor.local;
            const uint64_t output = cursor.base + local;
            const uint64_t remaining = local < span_ ? span_ - local : 0U;
            const uint32_t count = cursor.index == 0U ? cursor.head :
                static_cast<uint32_t>(remaining < kOutputElements ? remaining : kOutputElements);
            const uint32_t phase = cursor.phase;
            ++cursor.slot;
            ++cursor.index;
            if (cursor.index == slotsPerOuter_) {
                cursor.index = 0U;
                ++cursor.outer;
                cursor.base += span_;
                cursor.local = 0U;
                cursor.head = static_cast<uint32_t>((512U - (cursor.base & 511U)) & 511U);
                cursor.phase = 0U;
            } else if (cursor.index == 1U) {
                cursor.local = cursor.head;

                cursor.phase = cursor.head;
            } else {
                cursor.local += kOutputElements;
                cursor.phase += tileStep_;
                if (cursor.phase >= cols_) cursor.phase -= cols_;
            }
            if (count == 0U) continue;
            transaction.outer = outer;
            transaction.localOffset = local;
            transaction.outputOffset = output;
            transaction.count = count;
            transaction.phase = phase;
            return true;
        }
        return false;
    }

    __aicore__ inline void LoadAndExpandRow(uint64_t outer)
    {

        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> compact = denseQueue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> periodic = periodicRowBuffer_.Get<int32_t>();
        const uint32_t alignedCols = (cols_ + 7U) & ~7U;
        Duplicate(compact, static_cast<int32_t>(0), alignedCols);
        const event_t vectorToMte2 = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vectorToMte2);
        WaitFlag<HardEvent::V_MTE2>(vectorToMte2);
        const DataCopyExtParams copy(1U, cols_ * sizeof(int32_t), 0U, 0U, 0U);
        const DataCopyPadExtParams<int32_t> pad(
            true, 0U, static_cast<uint8_t>(alignedCols - cols_), 0);
        DataCopyPad(compact, rowGlobal_[RowInputOffset(outer)], copy, pad);
        const event_t mte2ToVector = static_cast<event_t>(
            pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(mte2ToVector);
        WaitFlag<HardEvent::MTE2_V>(mte2ToVector);

        const uint32_t cacheRows = kPeriodicElements / cols_;
        const GatherMaskParams p={1U,static_cast<uint16_t>(cacheRows),0U,0U};
        uint64_t selected=0U;
        GatherMask(periodic,compact,7U,true,cols_,p,selected);
        PipeBarrier<PIPE_V>();
        SetMaskNorm();
        ResetMask();
        PipeBarrier<PIPE_V>();
        SetCmpMask(selectionScalarBuffer_.Get<half>());
        SetVectorMask<half>(128);
        PipeBarrier<PIPE_V>();
        denseQueue_.FreeTensor(compact);
    }

    __aicore__ inline void PrefetchDense(
        const Transaction& transaction, uint32_t offset)
    {
        const uint32_t count=transaction.count-offset;
        const uint32_t prefix=transaction.phase&7U;
        LocalTensor<int32_t> dense=denseQueue_.AllocTensor<int32_t>();

        const DataCopyExtParams copy(1U,count*sizeof(int32_t),0U,0U,0U);
        const DataCopyPadExtParams<int32_t> pad(true,static_cast<uint8_t>(prefix),0U,0);
        DataCopyPad(dense,denseGlobal_[transaction.outputOffset+offset],copy,pad);
        denseQueue_.EnQue(dense);
    }

    template <bool RowIsX1>
    __aicore__ inline void MaskPart(LocalTensor<int32_t> dense,
                                   LocalTensor<uint8_t> mask,
                                   uint32_t phase, uint32_t count)
    {
        const uint32_t prefix=phase&7U;
        const uint32_t vectorCount=(count+prefix+63U)&~63U;
        LocalTensor<uint8_t> destination=mask;
        mask=rawMaskBuffer_.Get<uint8_t>();
        const uint8_t compareRepeats = static_cast<uint8_t>(vectorCount / 64U);
        LocalTensor<int32_t> row = periodicRowBuffer_.Get<int32_t>()[phase&~7U];
        SetMaskNorm();
        SetVectorMask<half>(128);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        const BinaryRepeatParams binaryParams(1U, 1U, 1U, 8U, 8U, 8U);
        if constexpr (RowIsX1) {
            Min(dense, dense, row, 64ULL, compareRepeats, binaryParams);
        } else {
            Max(dense, dense, row, 64ULL, compareRepeats, binaryParams);
        }
#else
        auto* rawDense = reinterpret_cast<__ubuf__ int32_t*>(dense.GetPhyAddr());
        auto* rawRow = reinterpret_cast<__ubuf__ int32_t*>(row.GetPhyAddr());
        if constexpr (RowIsX1) {
            vmin(rawDense, rawDense, rawRow,
                compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        } else {
            vmax(rawDense, rawDense, rawRow,
                compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
        }
#endif
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Compare(mask, dense, row, CMPMODE::EQ, 64ULL, compareRepeats, binaryParams);
#else
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr());
        vcmpv_eq(rawMask, rawDense, rawRow,
            compareRepeats, 1U, 1U, 1U, 8U, 8U, 8U);
#endif
        PipeBarrier<PIPE_V>();
        const uint32_t words=((count+255U)&~255U)/32U;
        LocalTensor<uint32_t> raw=mask.ReinterpretCast<uint32_t>();
        LocalTensor<uint32_t> out=destination.ReinterpretCast<uint32_t>();
        if(prefix==0U) {
            Or(destination.ReinterpretCast<uint16_t>(),mask.ReinterpretCast<uint16_t>(),
               mask.ReinterpretCast<uint16_t>(),words*2U);
        } else {

            LocalTensor<uint32_t> next=dense.ReinterpretCast<uint32_t>();
            Gather(next,raw,indexBuffer_.Get<uint32_t>(),0U,words);
            PipeBarrier<PIPE_V>();
            ShiftLeft(next,next,static_cast<uint32_t>(32U-prefix),words);
            ShiftRight(out,raw,prefix,words);
            PipeBarrier<PIPE_V>();
            Or(out.ReinterpretCast<uint16_t>(),out.ReinterpretCast<uint16_t>(),
               next.ReinterpretCast<uint16_t>(),words*2U);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SelectPart(LocalTensor<uint8_t> mask,
                                       LocalTensor<uint8_t> output, uint32_t count, uint32_t outputSlot)
    {
        const uint32_t vectorCount = (count + 255U) & ~255U;
        const uint8_t selectRepeats = static_cast<uint8_t>(vectorCount / 128U);
        LocalTensor<half> selected = periodicRowBuffer_.Get<half>();
        LocalTensor<half> values = selectionValueBuffer_.Get<half>();
        SetMaskNorm();
        SetCmpMask(selectionScalarBuffer_.Get<half>());
        SetVectorMask<half>(128);
        PipeBarrier<PIPE_V>();
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        const BinaryRepeatParams selectParams(1U, 1U, 1U, 8U, 0U, 0U);
        Select(selected, mask, values, selectRepeats, selectParams);
#else
        auto* rawSelected = reinterpret_cast<__ubuf__ half*>(selected.GetPhyAddr());
        auto* rawValues = reinterpret_cast<__ubuf__ half*>(values.GetPhyAddr());
        auto* rawMask = reinterpret_cast<__ubuf__ uint8_t*>(mask.GetPhyAddr());
        vsel(rawSelected, rawValues, rawMask, selectRepeats,
            1U, 1U, 1U, 8U, 0U, 0U,
            static_cast<uint8_t>(SELMODE::VSEL_TENSOR_SCALAR_MODE));
#endif
        PipeBarrier<PIPE_V>();

        if (groupParts_ == 0U) WaitOutput(outputSlot);
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
        Cast(output.ReinterpretCast<int8_t>(), selected, RoundMode::CAST_RINT, vectorCount);
#else
        vconv_f162s8r(reinterpret_cast<__ubuf__ int8_t*>(output.GetPhyAddr()),
            rawSelected, selectRepeats, 1U, 1U, 4U, 8U);
#endif

        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StartExpansion()
    {
        outputStorage_[0] = denseQueue_.AllocTensor<int32_t>();
        outputStorage_[1] = denseQueue_.AllocTensor<int32_t>();
        outputSequence_ = 0U; groupParts_ = 0U; groupCount_ = 0U;
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            outputEvent_[slot] = static_cast<event_t>(pipe_->AllocEventID<HardEvent::MTE3_V>());
            outputPending_[slot] = false;
        }
    }

    __aicore__ inline void WaitOutput(uint32_t slot)
    {
        if (outputPending_[slot]) {
            WaitFlag<HardEvent::MTE3_V>(outputEvent_[slot]);
            outputPending_[slot] = false;
        }
    }

    __aicore__ inline void ExpandAndStore(const Transaction& transaction, uint32_t maskOffset, bool last)
    {
        const uint32_t slot = outputSequence_ & 1U;
        if (groupParts_ == 0U) {
            groupOffset_ = transaction.outputOffset;
            groupCount_ = 0U;
        }
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>()[maskOffset];
        LocalTensor<uint8_t> output = outputStorage_[slot].ReinterpretCast<uint8_t>()[groupParts_ * (2U*kTile)];
        SelectPart(mask, output, transaction.count, slot);
        groupCount_ += transaction.count;
        ++groupParts_;

        if (groupParts_ == kOutputGroup || transaction.count < 2U*kTile || last) {
            const event_t ready = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready); WaitFlag<HardEvent::V_MTE3>(ready);
            const DataCopyExtParams copy(1U, groupCount_, 0U, 0U, 0U);
            DataCopyPad(yGlobal_[groupOffset_], outputStorage_[slot].ReinterpretCast<uint8_t>(), copy);
            SetFlag<HardEvent::MTE3_V>(outputEvent_[slot]); outputPending_[slot] = true;
            ++outputSequence_; groupParts_ = 0U;
        }
    }

    __aicore__ inline void FinishExpansion()
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            WaitOutput(slot);
            pipe_->ReleaseEventID<HardEvent::MTE3_V>(outputEvent_[slot]);
            denseQueue_.FreeTensor(outputStorage_[slot]);
        }
    }

    template <bool RowIsX1>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t core=GetBlockIdx(), cores=GetBlockNum();
        const uint64_t slots=tiling_->matrixOuterCount*slotsPerOuter_;
        const uint64_t base=slots/cores, remainder=slots%cores;
        uint64_t cursor=base*core+(core<remainder?core:remainder);
        const uint64_t begin=cursor, end=cursor+base+(core<remainder?1U:0U);
        Cursor iterator=MakeCursor(begin);
        Duplicate(selectionValueBuffer_.Get<half>(),static_cast<half>(0),128U);
        Duplicate(selectionScalarBuffer_.Get<half>(),static_cast<half>(1),16U);
        Duplicate(rawMaskBuffer_.Get<uint32_t>(),static_cast<uint32_t>(0),kRawMaskBytes/4U);
        CreateVecIndex(indexBuffer_.Get<int32_t>(),static_cast<int32_t>(1),128U);
        PipeBarrier<PIPE_V>();
        Muls(indexBuffer_.Get<int32_t>(),indexBuffer_.Get<int32_t>(),static_cast<int32_t>(4),128U);
        PipeBarrier<PIPE_V>();
        Transaction current;
        if(!NextTransaction(iterator,end,current)) return;

        LocalTensor<int32_t> initial0=denseQueue_.AllocTensor<int32_t>();
        LocalTensor<int32_t> initial1=denseQueue_.AllocTensor<int32_t>();
        Duplicate(initial0,static_cast<int32_t>(0),kDenseElements);
        Duplicate(initial1,static_cast<int32_t>(0),kDenseElements);
        PipeBarrier<PIPE_V>();
        denseQueue_.FreeTensor(initial0);
        denseQueue_.FreeTensor(initial1);
        LoadAndExpandRow(current.outer);
        uint64_t cachedOuter=current.outer;
        PrefetchDense(current,0U);
        uint32_t cached=0U, bytes=0U;
        while(true) {
            if(current.outer!=cachedOuter) {
                LoadAndExpandRow(current.outer); cachedOuter=current.outer;
            }
            LocalTensor<int32_t> dense=denseQueue_.DeQue<int32_t>();
            Transaction next;
            const bool hasNext=NextTransaction(iterator,end,next);
            if(hasNext) PrefetchDense(next,0U);
            MaskPart<RowIsX1>(dense,maskBuffer_.Get<uint8_t>()[bytes],
                current.phase,current.count);
            bytes+=((current.count+255U)&~255U)/8U;
            ++cached;
            denseQueue_.FreeTensor(dense);
            if(!hasNext) break;
            current=next;
        }
        StartExpansion();
        Cursor replay=MakeCursor(begin); bytes=0U;
        for(uint32_t sequence=0U;sequence<cached;) {
            Transaction output;
            if(!NextTransaction(replay,end,output)) break;
            ++sequence;

            if(output.count==kTile && sequence<cached) {
                Transaction second;
                if(NextTransaction(replay,end,second)) {
                    output.count+=second.count;
                    ++sequence;
                }
            }
            ExpandAndStore(output,bytes,sequence==cached);
            bytes+=((output.count+255U)&~255U)/8U;
        }
        FinishExpansion();
    }

    TPipe* pipe_;
    TQue<QuePosition::VECIN, 2U> denseQueue_;
    TBuf<TPosition::VECCALC> rawMaskBuffer_, indexBuffer_;
    TBuf<TPosition::VECCALC> periodicRowBuffer_;
    TBuf<TPosition::VECCALC> maskBuffer_;
    TBuf<TPosition::VECCALC> selectionValueBuffer_;
    TBuf<TPosition::VECCALC> selectionScalarBuffer_;
    GlobalTensor<int32_t> rowGlobal_;
    GlobalTensor<int32_t> denseGlobal_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterCompactPeriodicTilingData* tiling_;
    uint64_t span_;
    uint64_t slotsPerOuter_;
    uint32_t cols_, tileStep_;
    bool rowIsX1_;
    LocalTensor<int32_t> outputStorage_[2];
    event_t outputEvent_[2]; bool outputPending_[2];
    uint64_t groupOffset_; uint32_t groupCount_, groupParts_, outputSequence_;
};
class KernelGreaterInt8ResidentPredicates {
public:
    static constexpr uint32_t TileElements = 16384U;
    static constexpr uint32_t MaskBytes = 49152U;
    static_assert(8U * TileElements + MaskBytes + 288U <= 188160U);
    __aicore__ inline static bool CanStage(uint64_t n, uint64_t cores)
    {
        if (cores == 0U) return false;
        const uint64_t average = n / cores + (n % cores != 0U);

        if (average > 8ULL * MaskBytes) return false;
        const uint64_t blockElements = (average + 255U) & ~255ULL;
        return blockElements <= 8ULL * MaskBytes;
    }
    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                               const GreaterSmallDenseTilingData& tiling)
    {
        tiling_ = &tiling;
        x1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1));
        x2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2));
        yGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if (tiling.totalElements >= (32ULL * 1024ULL * 1024ULL + 2ULL) / 3ULL) {
            x1Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            x2Global_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        pipe_.InitBuffer(x1Queue_, 2U, TileElements);
        pipe_.InitBuffer(x2Queue_, 2U, TileElements);
        pipe_.InitBuffer(halfX1Buffer_, 2U * TileElements);
        pipe_.InitBuffer(halfX2Buffer_, 2U * TileElements);
        pipe_.InitBuffer(maskBuffer_, MaskBytes);
        pipe_.InitBuffer(valuesBuffer_, 256U);
        pipe_.InitBuffer(scalarBuffer_, 32U);
    }
    __aicore__ inline void Process()
    {
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t average = tiling_->totalElements / blocks +
            (tiling_->totalElements % blocks != 0U ? 1U : 0U);

        const uint64_t blockElements = (average + 255U) & ~255ULL;
        const uint64_t start = block * blockElements;
        const uint64_t limit = start + blockElements;
        const uint64_t end = limit < tiling_->totalElements ? limit : tiling_->totalElements;
        if (start >= end) return;
        Duplicate(valuesBuffer_.Get<half>(), static_cast<half>(1), 128U);
        Duplicate(scalarBuffer_.Get<half>(), static_cast<half>(0), 16U);
        PipeBarrier<PIPE_V>();
        uint64_t current = start;
        uint32_t count = static_cast<uint32_t>(
            end - current < TileElements ? end - current : TileElements);
        uint32_t sequence = 0U;
        CopyIn(current, count);
        while (current < end) {
            const uint64_t next = current + count;
            uint32_t nextCount = 0U;
            if (next < end) {
                nextCount = static_cast<uint32_t>(
                    end - next < TileElements ? end - next : TileElements);
                CopyIn(next, nextCount);
            }
            CompareAndCache(sequence++, count);
            current = next;
            count = nextCount;
        }
        StartExpansion();
        sequence = 0U;
        for (current = start; current < end; current += TileElements) {
            count = static_cast<uint32_t>(
                end - current < TileElements ? end - current : TileElements);
            ExpandAndStore(current, sequence++, count);
        }
        FinishExpansion();
    }
private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        LocalTensor<int8_t> x1 = x1Queue_.AllocTensor<int8_t>();
        LocalTensor<int8_t> x2 = x2Queue_.AllocTensor<int8_t>();
        if ((count & 255U) != 0U) {

            PipeBarrier<PIPE_V>();
            const uint32_t tailBlock = count & ~255U;
            Duplicate(x1[tailBlock].ReinterpretCast<int32_t>(), 0, 64U);
            Duplicate(x2[tailBlock].ReinterpretCast<int32_t>(), 0, 64U);
            const event_t event = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(event);
            WaitFlag<HardEvent::V_MTE2>(event);
        }
        if ((count & 31U) == 0U) {
            const DataCopyParams copy(1U, static_cast<uint16_t>(count / 32U), 0U, 0U);
            DataCopy(x1, x1Global_[offset], copy);
            DataCopy(x2, x2Global_[offset], copy);
        } else {
            const DataCopyExtParams copy(1U, count, 0U, 0U, 0U);
            const DataCopyPadExtParams<int8_t> pad(false, 0U, 0U, 0);
            DataCopyPad(x1, x1Global_[offset], copy, pad);
            DataCopyPad(x2, x2Global_[offset], copy, pad);
        }
        x1Queue_.EnQue(x1);
        x2Queue_.EnQue(x2);
    }
    __aicore__ inline void CompareAndCache(uint32_t sequence, uint32_t count)
    {
        const uint32_t vectorCount = (count + 255U) & ~255U;
        LocalTensor<int8_t> x1 = x1Queue_.DeQue<int8_t>();
        LocalTensor<int8_t> x2 = x2Queue_.DeQue<int8_t>();
        LocalTensor<half> halfX1 = halfX1Buffer_.Get<half>();
        LocalTensor<half> halfX2 = halfX2Buffer_.Get<half>();
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>()[sequence * (TileElements / 8U)];

        Cast(halfX1, x1, RoundMode::CAST_NONE, vectorCount);
        Cast(halfX2, x2, RoundMode::CAST_NONE, vectorCount);
        PipeBarrier<PIPE_V>();
        Compare(mask, halfX1, halfX2, CMPMODE::GT, vectorCount);
        PipeBarrier<PIPE_V>();
        x1Queue_.FreeTensor(x1);
        x2Queue_.FreeTensor(x2);
    }
    __aicore__ inline void StartExpansion()
    {

        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            outputEvent_[slot] = static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_V>());
            outputPending_[slot] = false;
        }
        SetCmpMask(scalarBuffer_.Get<half>());
        SetMaskNorm(); SetVectorMask<half>(128U); PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void WaitOutput(uint32_t slot)
    {
        if (outputPending_[slot]) {
            WaitFlag<HardEvent::MTE3_V>(outputEvent_[slot]);
            outputPending_[slot] = false;
        }
    }
    __aicore__ inline void ExpandAndStore(uint64_t offset, uint32_t sequence, uint32_t count)
    {
        const uint32_t slot = sequence & 1U;
        LocalTensor<uint8_t> mask = maskBuffer_.Get<uint8_t>()[sequence * (TileElements / 8U)];
        LocalTensor<half> selected = halfX1Buffer_.Get<half>();
        LocalTensor<uint8_t> output = halfX2Buffer_.Get<uint8_t>()[slot * TileElements];
        const uint32_t vectorCount = (count + 255U) & ~255U;
        const uint8_t repeats = static_cast<uint8_t>(vectorCount / 128U);
        const BinaryRepeatParams params(1U, 1U, 1U, 8U, 0U, 0U);
        Select(selected, mask, valuesBuffer_.Get<half>(), repeats, params);
        PipeBarrier<PIPE_V>();

        WaitOutput(slot);
        Cast(output.ReinterpretCast<int8_t>(), selected, RoundMode::CAST_RINT, vectorCount);
        const event_t storeReady = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(storeReady); WaitFlag<HardEvent::V_MTE3>(storeReady);
        const DataCopyExtParams copy(1U, count, 0U, 0U, 0U);
        DataCopyPad(yGlobal_[offset], output, copy);
        SetFlag<HardEvent::MTE3_V>(outputEvent_[slot]); outputPending_[slot] = true;
    }
    __aicore__ inline void FinishExpansion()
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            WaitOutput(slot);
            pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputEvent_[slot]);
        }
    }
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2U> x1Queue_, x2Queue_;
    TBuf<TPosition::VECCALC> halfX1Buffer_, halfX2Buffer_;
    TBuf<TPosition::VECCALC> maskBuffer_, valuesBuffer_, scalarBuffer_;
    GlobalTensor<int8_t> x1Global_, x2Global_;
    GlobalTensor<uint8_t> yGlobal_;
    const GreaterSmallDenseTilingData* tiling_;
    event_t outputEvent_[2]; bool outputPending_[2];
};

}

extern "C" __global__ __aicore__ void greater(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 131", GreaterSmallDenseTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 121", GreaterCompactPeriodicTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 115", GreaterCompactPeriodicTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 117", GreaterSmallDenseTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 109", GreaterSmallDenseTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 101", GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2", GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 3", GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 5", GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 7", GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 9", GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 11", GreaterTilingData);
    REGISTER_TILING_FOR_TILINGKEY(
        "TILING_KEY_VAR == 63", GreaterRowSingleTilingData);
    REGISTER_TILING_FOR_TILINGKEY(
        "TILING_KEY_VAR == 65", GreaterRowSingleTilingData);
    REGISTER_TILING_FOR_TILINGKEY(
        "TILING_KEY_VAR == 67", GreaterRowSingleTilingData);
    REGISTER_TILING_FOR_TILINGKEY(
        "TILING_KEY_VAR == 69", GreaterRowSingleTilingData);
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
        KernelGreater<DTYPE_X1> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
        KernelGreaterDense<DTYPE_X1> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
        KernelGreaterBox<DTYPE_X1> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
        KernelGreaterBox<DTYPE_X1> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
        KernelGreaterRowStripe<DTYPE_X1> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(9)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
        KernelGreaterRowSlab<DTYPE_X1, 2U> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
        KernelGreaterRowSlab<DTYPE_X1, 1U> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(63)) {
        GET_TILING_DATA_WITH_STRUCT(
            GreaterRowSingleTilingData, tilingData, tiling);

        KernelGreaterAlignedRowBoundPipeline<
            DTYPE_X1, false, false, false, true, true, true, true> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(65)) {
        GET_TILING_DATA_WITH_STRUCT(
            GreaterRowSingleTilingData, tilingData, tiling);
        if constexpr (std::is_same<DTYPE_X1, float>::value) {

            KernelGreaterAlignedFp32RowSplitPipeline<> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        } else if constexpr (std::is_same<DTYPE_X1, half>::value) {

            KernelGreaterAlignedFp16RowSplitExact<> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        } else if constexpr (std::is_same<DTYPE_X1, int8_t>::value) {

            KernelGreaterAlignedInt8RowStagedPipeline<> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        } else if constexpr (IsBfloat16<DTYPE_X1>::kValue) {

            KernelGreaterAlignedBf16RowStagedPipeline<> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        } else {

            KernelGreaterAlignedInt32RowDestructivePipeline<> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(67)) {
        GET_TILING_DATA_WITH_STRUCT(
            GreaterRowSingleTilingData, tilingData, tiling);

        KernelGreaterAlignedRowBoundPipeline<
            DTYPE_X1, false, true, false, true, false, true, true> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(69)) {
        GET_TILING_DATA_WITH_STRUCT(
            GreaterRowSingleTilingData, tilingData, tiling);
        KernelGreaterCrossScalar<DTYPE_X1> op;
        op.Init(x1, x2, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(101)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
        if constexpr (IsInt32<DTYPE_X1>::kValue) {
            KernelGreaterInt32Periodic<1U> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        } else {
            KernelGreaterBox<DTYPE_X1> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        }
    } else if (TILING_KEY_IS(109)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterSmallDenseTilingData, tilingData, tiling);
        if constexpr (std::is_same<DTYPE_X1, half>::value) {
            KernelGreaterSmallHalf op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        } else {

            GreaterTilingData denseTiling = {};
            denseTiling.totalElements = tilingData.totalElements;
            KernelGreaterDense<DTYPE_X1> op;
            op.Init(x1, x2, y, denseTiling);
            op.Process();
        }
    } else if (TILING_KEY_IS(115)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterCompactPeriodicTilingData, tilingData, tiling);
        if constexpr (IsInt32<DTYPE_X1>::kValue) {
            TPipe pipe;
            if(KernelGreaterInt32BitAlignedResident::CanStage(tilingData,GetBlockNum())) {
                KernelGreaterInt32BitAlignedResident op;
                op.Init(x1,x2,y,tilingData,&pipe); op.Process();
            } else {
            KernelGreaterInt32ChunkPredicates op;
            op.Init(x1, x2, y, tilingData, &pipe);
            op.Process();
            }
        } else {

            GreaterTilingData full = {};
            full.matrixRows = tilingData.matrixRows;
            full.matrixCols = tilingData.matrixCols;
            full.totalElements = tilingData.totalElements;
            full.matrixOuterCount = tilingData.matrixOuterCount;
            full.matrixX1Pattern = tilingData.matrixX1Pattern;
            full.matrixX2Pattern = tilingData.matrixX1Pattern == 2U ? 0U : 2U;
            full.rank = 3U;
            full.outerRank = 2U;
            full.rowElements = full.matrixCols;
            full.boxStart = 2U;
            full.boxElements = full.matrixCols;
            full.boxCount = full.matrixOuterCount * full.matrixRows;
            full.outputDims[1] = full.matrixRows;
            full.outputDims[2] = full.matrixCols;
            full.x1Dims[1] = full.matrixX1Pattern == 2U ? 1U : full.matrixRows;
            full.x2Dims[1] = full.matrixX1Pattern == 2U ? full.matrixRows : 1U;
            full.x1Dims[2] = full.matrixCols;
            full.x2Dims[2] = full.matrixCols;
            full.x1Strides[1] = full.matrixCols;
            full.x2Strides[1] = full.matrixCols;
            full.x1Strides[2] = 1U;
            full.x2Strides[2] = 1U;
            full.matrixEnabled = 1U;
            full.matrixRowsPerTile = 1U;
            full.matrixRowTiles = static_cast<uint32_t>(full.matrixRows);
            full.matrixTotalTiles = full.matrixRows * full.matrixOuterCount;
            full.matrixRowStart = 1U;
            full.outputDims[0] = full.matrixOuterCount;
            full.x1Dims[0] = full.matrixOuterCount;
            full.x2Dims[0] = full.matrixOuterCount;
            full.x1Strides[0] = full.matrixX1Pattern == 2U ? full.matrixCols : full.matrixRows * full.matrixCols;
            full.x2Strides[0] = full.matrixX1Pattern == 2U ? full.matrixRows * full.matrixCols : full.matrixCols;
            KernelGreaterBox<DTYPE_X1> op;
            op.Init(x1, x2, y, full);
            op.Process();
        }
    } else if (TILING_KEY_IS(117)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterSmallDenseTilingData, tilingData, tiling);
        if constexpr (std::is_same<DTYPE_X1, float>::value) {
            if (KernelGreaterFp32ResidentPredicates::CanStage(tilingData.totalElements, GetBlockNum())) {
                KernelGreaterFp32ResidentPredicates op;
                op.Init(x1, x2, y, tilingData);
                op.Process();
            } else {
            KernelGreaterDenseFp32ManualCompact<10240U, true> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
            }
        } else {
            GreaterTilingData full = {};
            full.totalElements = tilingData.totalElements;
            KernelGreaterDense<DTYPE_X1> op;
            op.Init(x1, x2, y, full);
            op.Process();
        }
    } else if (TILING_KEY_IS(121)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterCompactPeriodicTilingData, tilingData, tiling);
        if constexpr (IsBfloat16<DTYPE_X1>::kValue) {
            KernelGreaterBfloat16EarlyRelease<16128U> op;
            op.Init(x1, x2, y, tilingData);
            op.Process();
        } else {

            GreaterTilingData full = {};
            full.matrixRows = tilingData.matrixRows;
            full.matrixCols = tilingData.matrixCols;
            full.totalElements = tilingData.totalElements;
            full.matrixOuterCount = tilingData.matrixOuterCount;
            full.matrixX1Pattern = tilingData.matrixX1Pattern;
            full.matrixX2Pattern = tilingData.matrixX1Pattern == 2U ? 0U : 2U;
            full.rank = 3U;
            full.outerRank = 2U;
            full.rowElements = full.matrixCols;
            full.boxStart = 2U;
            full.boxElements = full.matrixCols;
            full.boxCount = full.matrixOuterCount * full.matrixRows;
            full.outputDims[1] = full.matrixRows;
            full.outputDims[2] = full.matrixCols;
            full.x1Dims[1] = full.matrixX1Pattern == 2U ? 1U : full.matrixRows;
            full.x2Dims[1] = full.matrixX1Pattern == 2U ? full.matrixRows : 1U;
            full.x1Dims[2] = full.matrixCols;
            full.x2Dims[2] = full.matrixCols;
            full.x1Strides[1] = full.matrixCols;
            full.x2Strides[1] = full.matrixCols;
            full.x1Strides[2] = 1U;
            full.x2Strides[2] = 1U;
            full.matrixEnabled = 1U;
            full.matrixRowsPerTile = 1U;
            full.matrixRowTiles = static_cast<uint32_t>(full.matrixRows);
            full.matrixTotalTiles = full.matrixRows * full.matrixOuterCount;
            full.matrixRowStart = 1U;
            full.outputDims[0] = full.matrixOuterCount;
            full.x1Dims[0] = full.matrixOuterCount;
            full.x2Dims[0] = full.matrixOuterCount;
            full.x1Strides[0] = full.matrixX1Pattern == 2U ? full.matrixCols : full.matrixRows * full.matrixCols;
            full.x2Strides[0] = full.matrixX1Pattern == 2U ? full.matrixRows * full.matrixCols : full.matrixCols;
            KernelGreaterBox<DTYPE_X1> op;
            op.Init(x1, x2, y, full);
            op.Process();
        }
    } else if (TILING_KEY_IS(131)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterSmallDenseTilingData, tilingData, tiling);
        if constexpr (IsInt8<DTYPE_X1>::kValue) {
            if (KernelGreaterInt8ResidentPredicates::CanStage(tilingData.totalElements, GetBlockNum())) {
                KernelGreaterInt8ResidentPredicates op;
                op.Init(x1, x2, y, tilingData);
                op.Process();
            } else {
                KernelGreaterInt8DenseManual<18432U> op;
                op.Init(x1, x2, y, tilingData);
                op.Process();
            }
        } else {
            GreaterTilingData full = {};
            full.totalElements = tilingData.totalElements;
            KernelGreaterDense<DTYPE_X1> op;
            op.Init(x1, x2, y, full);
            op.Process();
        }
    }
}
