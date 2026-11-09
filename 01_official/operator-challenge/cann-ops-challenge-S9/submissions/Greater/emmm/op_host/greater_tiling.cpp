/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file greater_tiling.cpp
 * \brief
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "greater_tiling.h"
#include "../op_kernel/greater_tiling_key.h"

namespace optiling {

constexpr uint64_t OUTPUT_BLOCK_ELEMENTS = 32;
constexpr uint64_t UB_RESERVED_BYTES = 8 * 1024;
constexpr uint64_t INT32_TILE_ELEMENTS = 8 * 1024;
constexpr uint64_t INT32_PADDED_ROW_TILE_ELEMENTS = 16 * 1024;
constexpr uint64_t INT32_LARGE_TILE_BYTES_PER_ELEMENT =
    2 * sizeof(int32_t) + sizeof(int8_t);
constexpr uint64_t INT32_LARGE_TILE_FIXED_BYTES = 256;
constexpr uint64_t INT8_REUSABLE_PREDICATE_MIN_CORE_ELEMENTS = 64 * 1024;
constexpr uint64_t INT8_REUSABLE_PREDICATE_MAX_CORE_ELEMENTS = 2 * 1024 * 1024;
constexpr uint64_t INT8_MACRO_TILE_ALIGNMENT_ELEMENTS = 512;
constexpr uint64_t INT8_MACRO_TILE_FIXED_UB_BYTES = 1888;
constexpr uint64_t INT8_MACRO_TILE_BYTES_PER_ELEMENT = 7;
constexpr uint64_t ADJACENT_BROADCAST_SOURCE_BYTES = 2 * 1024;
constexpr uint64_t BROADCAST_ROW_BUFFER_BYTES = 6 * 1024;

struct GreaterCompileInfo {};

struct BroadcastInfo {
    uint64_t totalLength = 1;
    uint64_t x1Length = 1;
    uint64_t x2Length = 1;
    uint32_t rank = 0;
    uint32_t mode = GREATER_BROADCAST_CONTIGUOUS;
    std::array<uint64_t, GREATER_MAX_BROADCAST_DIM> outShape {};
    std::array<uint64_t, GREATER_MAX_BROADCAST_DIM> x1Stride {};
    std::array<uint64_t, GREATER_MAX_BROADCAST_DIM> x2Stride {};
};

static bool MultiplyChecked(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool ReadBenchmarkUint64(const char* name, bool& present, uint64_t& value)
{
    const char* text = std::getenv(name);
    present = text != nullptr;
    value = 0;
    if (!present) {
        return true;
    }
    if (*text == '\0') {
        return false;
    }
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            return false;
        }
        const uint64_t digit = static_cast<uint64_t>(*cursor - '0');
        if (value > (std::numeric_limits<uint64_t>::max() - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    return true;
}

static bool ReadAlignedShape(
    const gert::Shape& shape, uint32_t rank, std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& dims,
    uint64_t& elementCount)
{
    if (shape.GetDimNum() > rank || rank > GREATER_MAX_BROADCAST_DIM) {
        return false;
    }
    dims.fill(1);
    elementCount = 1;
    const uint32_t offset = rank - static_cast<uint32_t>(shape.GetDimNum());
    for (size_t index = 0; index < shape.GetDimNum(); ++index) {
        const int64_t dim = shape.GetDim(index);
        if (dim < 0) {
            return false;
        }
        dims[offset + index] = static_cast<uint64_t>(dim);
        if (!MultiplyChecked(elementCount, static_cast<uint64_t>(dim), elementCount)) {
            return false;
        }
    }
    return true;
}

static void CalcCompressedStride(
    const std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& inputShape,
    const std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& outputShape, uint32_t rank,
    std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& stride)
{
    stride.fill(0);
    uint64_t runningStride = 1;
    for (int32_t index = static_cast<int32_t>(rank) - 1; index >= 0; --index) {
        const uint32_t dimIndex = static_cast<uint32_t>(index);
        stride[dimIndex] = runningStride;
        runningStride *= inputShape[dimIndex];
        if (inputShape[dimIndex] == 1 && outputShape[dimIndex] != 1) {
            stride[dimIndex] = 0;
        }
    }
}

static uint32_t GetBroadcastState(uint64_t x1Dim, uint64_t x2Dim)
{
    if (x1Dim == x2Dim) {
        return 0;
    }
    return x1Dim == 1 ? 1 : 2;
}

static void CompressBroadcastInfo(
    const BroadcastInfo& rawInfo, const std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& rawX1Shape,
    const std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& rawX2Shape, BroadcastInfo& info)
{
    info.totalLength = rawInfo.totalLength;
    info.x1Length = rawInfo.x1Length;
    info.x2Length = rawInfo.x2Length;
    std::array<uint64_t, GREATER_MAX_BROADCAST_DIM> x1Shape {};
    std::array<uint64_t, GREATER_MAX_BROADCAST_DIM> x2Shape {};
    uint32_t lastState = std::numeric_limits<uint32_t>::max();
    for (uint32_t index = 0; index < rawInfo.rank; ++index) {
        if (rawInfo.rank > 1 && rawInfo.outShape[index] == 1) {
            continue;
        }
        const uint32_t state = GetBroadcastState(rawX1Shape[index], rawX2Shape[index]);
        if (info.rank > 0 && state == lastState) {
            const uint32_t destination = info.rank - 1;
            info.outShape[destination] *= rawInfo.outShape[index];
            x1Shape[destination] *= rawX1Shape[index];
            x2Shape[destination] *= rawX2Shape[index];
        } else {
            const uint32_t destination = info.rank++;
            info.outShape[destination] = rawInfo.outShape[index];
            x1Shape[destination] = rawX1Shape[index];
            x2Shape[destination] = rawX2Shape[index];
            lastState = state;
        }
    }
    if (rawInfo.rank > 0 && info.rank == 0) {
        info.rank = 1;
        info.outShape[0] = 1;
        x1Shape[0] = 1;
        x2Shape[0] = 1;
    }
    CalcCompressedStride(x1Shape, info.outShape, info.rank, info.x1Stride);
    CalcCompressedStride(x2Shape, info.outShape, info.rank, info.x2Stride);
}

static bool OutputShapeMatches(
    const gert::Shape& shape, const std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& dims, uint32_t rank)
{
    if (shape.GetDimNum() != rank) {
        return false;
    }
    for (uint32_t index = 0; index < rank; ++index) {
        if (shape.GetDim(index) < 0 || static_cast<uint64_t>(shape.GetDim(index)) != dims[index]) {
            return false;
        }
    }
    return true;
}

static bool GetBroadcastInfo(gert::TilingContext* context, BroadcastInfo& info)
{
    const auto* x1ShapeInfo = context->GetInputShape(0);
    const auto* x2ShapeInfo = context->GetInputShape(1);
    const auto* outputShapeInfo = context->GetOutputShape(0);
    if (x1ShapeInfo == nullptr || x2ShapeInfo == nullptr || outputShapeInfo == nullptr) {
        return false;
    }
    const auto& x1Shape = x1ShapeInfo->GetStorageShape();
    const auto& x2Shape = x2ShapeInfo->GetStorageShape();
    const auto& outputShape = outputShapeInfo->GetStorageShape();
    const uint32_t rank = static_cast<uint32_t>(std::max(x1Shape.GetDimNum(), x2Shape.GetDimNum()));
    if (rank > GREATER_MAX_BROADCAST_DIM) {
        return false;
    }

    std::array<uint64_t, GREATER_MAX_BROADCAST_DIM> x1Dims {};
    std::array<uint64_t, GREATER_MAX_BROADCAST_DIM> x2Dims {};
    BroadcastInfo rawInfo {};
    rawInfo.rank = rank;
    if (!ReadAlignedShape(x1Shape, rank, x1Dims, rawInfo.x1Length) ||
        !ReadAlignedShape(x2Shape, rank, x2Dims, rawInfo.x2Length)) {
        return false;
    }
    for (uint32_t index = 0; index < rank; ++index) {
        const uint64_t x1Dim = x1Dims[index];
        const uint64_t x2Dim = x2Dims[index];
        if (x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1) {
            return false;
        }
        rawInfo.outShape[index] = (x1Dim == 0 || x2Dim == 0) ? 0 : std::max(x1Dim, x2Dim);
        if (!MultiplyChecked(rawInfo.totalLength, rawInfo.outShape[index], rawInfo.totalLength)) {
            return false;
        }
    }
    if (!OutputShapeMatches(outputShape, rawInfo.outShape, rank)) {
        return false;
    }

    CompressBroadcastInfo(rawInfo, x1Dims, x2Dims, info);
    if (info.x1Length == info.totalLength && info.x2Length == info.totalLength) {
        info.mode = GREATER_BROADCAST_CONTIGUOUS;
    } else if (info.x1Length == 1 && info.x2Length == info.totalLength) {
        info.mode = GREATER_BROADCAST_X1_SCALAR;
    } else if (info.x2Length == 1 && info.x1Length == info.totalLength) {
        info.mode = GREATER_BROADCAST_X2_SCALAR;
    } else {
        info.mode = GREATER_BROADCAST_GENERAL;
    }
    return true;
}

static uint64_t GetUbPartNum(ge::DataType dataType)
{
    if (dataType == ge::DT_BF16 || dataType == ge::DT_INT8 || dataType == ge::DT_UINT8) {
        return 10;
    }
    return 5;
}

static uint64_t GetInt8MacroTileDataNum(uint64_t ubLength)
{
    if (ubLength <= INT8_MACRO_TILE_FIXED_UB_BYTES) {
        return 0;
    }
    const uint64_t rawElements =
        (ubLength - INT8_MACRO_TILE_FIXED_UB_BYTES) /
        INT8_MACRO_TILE_BYTES_PER_ELEMENT;
    return rawElements / INT8_MACRO_TILE_ALIGNMENT_ELEMENTS *
        INT8_MACRO_TILE_ALIGNMENT_ELEMENTS;
}

static uint64_t GetAdjacentUbPartDataNum(
    uint64_t ubLength, ge::DataType dataType, uint32_t dataTypeLength, const BroadcastInfo& info,
    bool reserveScalarScratch)
{
    uint64_t bytesPerElement = 0;
    if (dataType == ge::DT_FLOAT16) {
        bytesPerElement = 2 * sizeof(uint16_t) + sizeof(uint8_t);
    } else if (dataType == ge::DT_FLOAT) {
        bytesPerElement = 2 * sizeof(float) + sizeof(uint8_t) + sizeof(uint16_t);
    } else if (dataType == ge::DT_BF16) {
        bytesPerElement = 2 * sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint16_t) + 2 * sizeof(float);
    } else if (dataType == ge::DT_INT32) {
        bytesPerElement = 2 * sizeof(int32_t) + sizeof(uint8_t) + sizeof(uint16_t);
    }
    if (bytesPerElement == 0) {
        return 0;
    }
    uint64_t ubPartDataNum =
        ubLength / bytesPerElement / OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
    if (reserveScalarScratch &&
        (dataType == ge::DT_BF16 || dataType == ge::DT_INT32) &&
        info.rank == 4 && dataTypeLength != 0) {
        uint64_t planeLength = 0;
        uint64_t scratchElements = 0;
        if (!MultiplyChecked(info.outShape[2], info.outShape[3], planeLength) ||
            !MultiplyChecked(info.outShape[2], OUTPUT_BLOCK_ELEMENTS / dataTypeLength, scratchElements) ||
            planeLength == 0) {
            return 0;
        }
        const uint64_t fixedTailElements = 256 / dataTypeLength;
        while (ubPartDataNum >= planeLength &&
               ubPartDataNum % planeLength + fixedTailElements < scratchElements) {
            ubPartDataNum -= OUTPUT_BLOCK_ELEMENTS;
        }
    }
    return ubPartDataNum;
}

static uint64_t GetAdjacentParallelSplitAlignment(
    const BroadcastInfo& info, ge::DataType dataType, uint32_t dataTypeLength)
{
    if (info.mode != GREATER_BROADCAST_GENERAL || info.rank < 2 ||
        (dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16 && dataType != ge::DT_FLOAT &&
         dataType != ge::DT_INT8 && dataType != ge::DT_INT32)) {
        return 0;
    }
    const uint32_t tailIndex = info.rank - 1;
    const uint32_t rowIndex = info.rank - 2;
    uint64_t tailBytes = 0;
    uint64_t scalarBytes = 0;
    if (!MultiplyChecked(info.outShape[tailIndex], dataTypeLength, tailBytes) ||
        !MultiplyChecked(info.outShape[rowIndex], dataTypeLength, scalarBytes)) {
        return 0;
    }
    const bool x1ScalarsX2Row = info.x1Stride[tailIndex] == 0 && info.x1Stride[rowIndex] == 1 &&
                                info.x2Stride[tailIndex] == 1 && info.x2Stride[rowIndex] == 0;
    const bool x2ScalarsX1Row = info.x2Stride[tailIndex] == 0 && info.x2Stride[rowIndex] == 1 &&
                                info.x1Stride[tailIndex] == 1 && info.x1Stride[rowIndex] == 0;
    if (tailBytes == 0 || scalarBytes == 0 || tailBytes > ADJACENT_BROADCAST_SOURCE_BYTES ||
        scalarBytes > ADJACENT_BROADCAST_SOURCE_BYTES || (!x1ScalarsX2Row && !x2ScalarsX1Row)) {
        return 0;
    }
    uint64_t splitAlignment = info.outShape[tailIndex];
    if (info.rank > 2 &&
        !MultiplyChecked(splitAlignment, info.outShape[rowIndex], splitAlignment)) {
        return 0;
    }
    return splitAlignment;
}

static bool IsOutputContiguous(
    const BroadcastInfo& info,
    const std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& stride)
{
    uint64_t expected = 1;
    for (int32_t index = static_cast<int32_t>(info.rank) - 1; index >= 0; --index) {
        const uint32_t dimIndex = static_cast<uint32_t>(index);
        if (stride[dimIndex] != expected) {
            return false;
        }
        expected *= info.outShape[dimIndex];
    }
    return info.rank != 0;
}

static bool IsRepeatedRowBroadcast(
    const BroadcastInfo& info,
    const std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& repeatedStride,
    const std::array<uint64_t, GREATER_MAX_BROADCAST_DIM>& contiguousStride)
{
    if (info.rank < 2 || repeatedStride[info.rank - 1] != 1 ||
        repeatedStride[info.rank - 2] != 0 || !IsOutputContiguous(info, contiguousStride)) {
        return false;
    }
    uint64_t expected = info.outShape[info.rank - 1];
    for (int32_t index = static_cast<int32_t>(info.rank) - 3; index >= 0; --index) {
        const uint32_t dimIndex = static_cast<uint32_t>(index);
        if (repeatedStride[dimIndex] != expected) {
            return false;
        }
        expected *= info.outShape[dimIndex];
    }
    return true;
}

static bool CanUseBf16RepeatedRowPath(const BroadcastInfo& info, ge::DataType dataType)
{
    return dataType == ge::DT_BF16 && info.mode == GREATER_BROADCAST_GENERAL &&
           (IsRepeatedRowBroadcast(info, info.x1Stride, info.x2Stride) ||
            IsRepeatedRowBroadcast(info, info.x2Stride, info.x1Stride));
}

static bool CanUseAdjacentScalarBlockPath(
    const BroadcastInfo& info, ge::DataType dataType, uint32_t dataTypeLength, uint64_t ubPartDataNum,
    uint64_t splitAlignment)
{
    if (info.mode != GREATER_BROADCAST_GENERAL || info.rank != 4 || splitAlignment == 0 ||
        splitAlignment > ubPartDataNum ||
        (dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16 && dataType != ge::DT_FLOAT &&
         dataType != ge::DT_INT32)) {
        return false;
    }
    const uint64_t rowCount = info.outShape[2];
    const uint64_t rowLength = info.outShape[3];
    uint64_t tailBytes = 0;
    uint64_t scalarBytes = 0;
    uint64_t planeRowGapBytes = 0;
    if (!MultiplyChecked(rowLength, dataTypeLength, tailBytes) ||
        !MultiplyChecked(rowCount, dataTypeLength, scalarBytes) ||
        !MultiplyChecked(rowCount - 1, tailBytes, planeRowGapBytes) ||
        tailBytes % OUTPUT_BLOCK_ELEMENTS != 0 || scalarBytes > 256 || info.outShape[1] > 4095 ||
        planeRowGapBytes % OUTPUT_BLOCK_ELEMENTS != 0 ||
        planeRowGapBytes / OUTPUT_BLOCK_ELEMENTS > 65535) {
        return false;
    }
    const bool x1ScalarsX2Row = info.x1Stride[3] == 0 && info.x1Stride[2] == 1 &&
                                info.x2Stride[3] == 1 && info.x2Stride[2] == 0;
    const bool scalarInvariantAcrossPlanes =
        (x1ScalarsX2Row && info.x1Stride[1] == 0) || (!x1ScalarsX2Row && info.x2Stride[1] == 0);
    const uint64_t rowPlaneStride = x1ScalarsX2Row ? info.x2Stride[1] : info.x1Stride[1];
    const uint64_t spareElements = ubPartDataNum % splitAlignment + 256 / dataTypeLength;
    return scalarInvariantAcrossPlanes &&
           (info.outShape[1] == 1 || rowPlaneStride == rowLength) &&
           spareElements >= rowCount * OUTPUT_BLOCK_ELEMENTS / dataTypeLength;
}

static bool CanUseAdjacentPaddedPlanePath(
    const BroadcastInfo& info, ge::DataType dataType, uint32_t dataTypeLength,
    uint64_t ubPartDataNum, uint64_t splitAlignment)
{
    if (info.mode != GREATER_BROADCAST_GENERAL || info.rank < 4 || splitAlignment == 0 ||
        splitAlignment > ubPartDataNum || dataTypeLength == 0 ||
        (dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16 && dataType != ge::DT_FLOAT &&
         dataType != ge::DT_INT32 && dataType != ge::DT_INT8)) {
        return false;
    }
    if (info.rank > 4) {
        const bool integerPath = dataType == ge::DT_INT32 || dataType == ge::DT_INT8;
        const bool rank5HalfPath = info.rank == 5 &&
            (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16);
        if (!integerPath && !rank5HalfPath) {
            return false;
        }
    }
    const uint32_t tailIndex = info.rank - 1;
    const uint32_t rowIndex = info.rank - 2;
    const uint32_t planeGroupIndex = info.rank - 3;
    const uint64_t rowCount = info.outShape[rowIndex];
    const uint64_t rowLength = info.outShape[tailIndex];
    uint64_t tailBytes = 0;
    uint64_t scalarBytes = 0;
    uint64_t planeLength = 0;
    uint64_t paddedPlaneLength = 0;
    const uint64_t paddedRowLength =
        (rowLength + OUTPUT_BLOCK_ELEMENTS - 1) /
        OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
    if (!MultiplyChecked(rowLength, dataTypeLength, tailBytes) ||
        !MultiplyChecked(rowCount, dataTypeLength, scalarBytes) ||
        !MultiplyChecked(rowCount, rowLength, planeLength) ||
        !MultiplyChecked(rowCount, paddedRowLength, paddedPlaneLength) ||
        (info.rank == 4 && tailBytes % OUTPUT_BLOCK_ELEMENTS == 0) ||
        scalarBytes > ADJACENT_BROADCAST_SOURCE_BYTES ||
        planeLength == 0 || paddedPlaneLength > ubPartDataNum) {
        return false;
    }
    const bool x1ScalarsX2Row = info.x1Stride[tailIndex] == 0 && info.x1Stride[rowIndex] == 1 &&
                                info.x2Stride[tailIndex] == 1 && info.x2Stride[rowIndex] == 0;
    const bool x2ScalarsX1Row = info.x2Stride[tailIndex] == 0 && info.x2Stride[rowIndex] == 1 &&
                                info.x1Stride[tailIndex] == 1 && info.x1Stride[rowIndex] == 0;
    const bool scalarInvariantAcrossPlanes =
        (x1ScalarsX2Row && info.x1Stride[planeGroupIndex] == 0) ||
        (x2ScalarsX1Row && info.x2Stride[planeGroupIndex] == 0);
    const uint64_t rowPlaneStride = x1ScalarsX2Row ?
        info.x2Stride[planeGroupIndex] : info.x1Stride[planeGroupIndex];
    return scalarInvariantAcrossPlanes &&
           (info.outShape[planeGroupIndex] == 1 || rowPlaneStride == rowLength);
}

static bool CanUseInt8AdjacentPlaneSplit(
    const BroadcastInfo& info, ge::DataType dataType, uint32_t dataTypeLength,
    uint64_t ubPartDataNum, uint64_t splitAlignment)
{
    if (dataType != ge::DT_INT8 || info.mode != GREATER_BROADCAST_GENERAL || info.rank != 4 ||
        splitAlignment == 0 || splitAlignment > ubPartDataNum) {
        return false;
    }
    uint64_t scalarBytes = 0;
    if (!MultiplyChecked(info.outShape[2], dataTypeLength, scalarBytes) || scalarBytes > 256) {
        return false;
    }
    const bool x1ScalarsX2Row = info.x1Stride[3] == 0 && info.x1Stride[2] == 1 &&
                                info.x2Stride[3] == 1 && info.x2Stride[2] == 0;
    const bool x2ScalarsX1Row = info.x2Stride[3] == 0 && info.x2Stride[2] == 1 &&
                                info.x1Stride[3] == 1 && info.x1Stride[2] == 0;
    return (x1ScalarsX2Row && info.x1Stride[1] == 0) ||
           (x2ScalarsX1Row && info.x2Stride[1] == 0);
}

static bool CanUseInt32PaddedRowPath(
    const BroadcastInfo& info, ge::DataType dataType, uint32_t dataTypeLength,
    uint64_t ubPartDataNum, uint64_t& splitAlignment)
{
    splitAlignment = 0;
    if (dataType != ge::DT_INT32 || dataTypeLength == 0 ||
        OUTPUT_BLOCK_ELEMENTS % dataTypeLength != 0 ||
        info.mode != GREATER_BROADCAST_GENERAL || info.rank < 2) {
        return false;
    }
    const uint32_t tailIndex = info.rank - 1;
    const uint32_t rowIndex = info.rank - 2;
    const uint64_t rowLength = info.outShape[tailIndex];
    const uint64_t rowCount = info.outShape[rowIndex];
    const uint64_t paddedRowLength =
        (rowLength + OUTPUT_BLOCK_ELEMENTS - 1) /
        OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
    uint64_t rowBytes = 0;
    uint64_t planeLength = 0;
    if (!MultiplyChecked(rowLength, dataTypeLength, rowBytes) ||
        !MultiplyChecked(rowCount, rowLength, planeLength) ||
        rowLength == 0 || rowCount == 0 || paddedRowLength > ubPartDataNum ||
        rowBytes > BROADCAST_ROW_BUFFER_BYTES ||
        rowBytes > std::numeric_limits<uint16_t>::max()) {
        return false;
    }
    const bool x1RowsX2Contiguous =
        info.x1Stride[tailIndex] == 1 && info.x1Stride[rowIndex] == 0 &&
        info.x2Stride[tailIndex] == 1 && info.x2Stride[rowIndex] == rowLength;
    const bool x2RowsX1Contiguous =
        info.x2Stride[tailIndex] == 1 && info.x2Stride[rowIndex] == 0 &&
        info.x1Stride[tailIndex] == 1 && info.x1Stride[rowIndex] == rowLength;
    if (!x1RowsX2Contiguous && !x2RowsX1Contiguous) {
        return false;
    }
    constexpr uint64_t minRowsForParallelSplit = OUTPUT_BLOCK_ELEMENTS * 8;
    splitAlignment = rowCount >= minRowsForParallelSplit ? rowLength : planeLength;
    return true;
}

static bool HasLinearInt32GroupOffsets(const BroadcastInfo& info)
{
    if (info.rank < 2) {
        return false;
    }
    const uint32_t tailIndex = info.rank - 1;
    const uint32_t rowIndex = info.rank - 2;
    const uint64_t rowLength = info.outShape[tailIndex];
    const uint64_t rowCount = info.outShape[rowIndex];
    const bool x1Rows =
        info.x1Stride[tailIndex] == 1 &&
        info.x1Stride[rowIndex] == 0;
    const auto& broadcastStride = x1Rows ?
        info.x1Stride : info.x2Stride;
    const auto& contiguousStride = x1Rows ?
        info.x2Stride : info.x1Stride;
    uint64_t expectedBroadcastStride = rowLength;
    uint64_t expectedContiguousStride = 0;
    if (!MultiplyChecked(
            rowCount, rowLength, expectedContiguousStride) ||
        broadcastStride[tailIndex] != 1 ||
        broadcastStride[rowIndex] != 0 ||
        contiguousStride[tailIndex] != 1 ||
        contiguousStride[rowIndex] != rowLength) {
        return false;
    }
    for (int32_t index = static_cast<int32_t>(rowIndex) - 1;
         index >= 0; --index) {
        const uint32_t dimIndex = static_cast<uint32_t>(index);
        if (broadcastStride[dimIndex] != expectedBroadcastStride ||
            contiguousStride[dimIndex] != expectedContiguousStride) {
            return false;
        }
        if (index > 0 &&
            (!MultiplyChecked(
                expectedBroadcastStride, info.outShape[dimIndex],
                expectedBroadcastStride) ||
             !MultiplyChecked(
                expectedContiguousStride, info.outShape[dimIndex],
                expectedContiguousStride))) {
            return false;
        }
    }
    return true;
}

static uint64_t GetInt32LargeTileElements(
    uint64_t ubLength, const BroadcastInfo& info)
{
    if (info.rank < 2) {
        return 0;
    }
    const uint32_t tailIndex = info.rank - 1;
    const uint32_t rowIndex = info.rank - 2;
    const uint64_t rowLength = info.outShape[tailIndex];
    const uint64_t rowCount = info.outShape[rowIndex];
    const uint64_t paddedRowLength =
        (rowLength + OUTPUT_BLOCK_ELEMENTS - 1) /
        OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
    const uint64_t vectorAlignmentElements = 256 / sizeof(int32_t);
    uint64_t rowBufferBytes = 0;
    if (rowLength == 0 || rowCount < 4 || paddedRowLength == rowLength ||
        paddedRowLength % vectorAlignmentElements != 0 ||
        !MultiplyChecked(
            paddedRowLength, sizeof(int32_t), rowBufferBytes) ||
        rowBufferBytes > ubLength ||
        INT32_LARGE_TILE_FIXED_BYTES > ubLength - rowBufferBytes) {
        return 0;
    }
    const uint64_t variableBytes =
        ubLength - rowBufferBytes - INT32_LARGE_TILE_FIXED_BYTES;
    uint64_t tileElements =
        variableBytes / INT32_LARGE_TILE_BYTES_PER_ELEMENT;
    tileElements = tileElements / paddedRowLength * paddedRowLength;
    return tileElements >= paddedRowLength ? tileElements : 0;
}

static void SetBroadcastFields(GreaterTilingData& tiling, const BroadcastInfo& info)
{
    tiling.rank = info.rank;
    tiling.reserved = 0;
    for (uint32_t index = 0; index < GREATER_MAX_BROADCAST_DIM; ++index) {
        tiling.outShape[index] = info.outShape[index];
        tiling.x1Stride[index] = info.x1Stride[index];
        tiling.x2Stride[index] = info.x2Stride[index];
    }
}

static bool SetLegacyAlignedTiling(
    gert::TilingContext* context, GreaterTilingData& tiling, uint64_t inputDataNum, uint32_t dataTypeLength,
    uint64_t ubPartDataNum, uint64_t coreNum, uint64_t splitAlignment)
{
    const uint64_t blockSize = OUTPUT_BLOCK_ELEMENTS * dataTypeLength;
    const uint64_t inputLength = inputDataNum * dataTypeLength;
    const uint64_t inputLengthAlign32 = ((inputLength + blockSize - 1) / blockSize) * blockSize;
    const uint64_t ubPartBlockNum = ubPartDataNum / OUTPUT_BLOCK_ELEMENTS;
    if (ubPartBlockNum == 0 || coreNum == 0 || splitAlignment == 0 ||
        splitAlignment % OUTPUT_BLOCK_ELEMENTS != 0) {
        return false;
    }
    if (ubPartDataNum >= inputDataNum) {
        coreNum = 1;
    } else {
        const uint64_t totalBlockNum = inputLengthAlign32 / blockSize;
        const uint64_t splitAlignmentBlocks = splitAlignment / OUTPUT_BLOCK_ELEMENTS;
        if (splitAlignment == OUTPUT_BLOCK_ELEMENTS) {
            coreNum = std::min(coreNum, totalBlockNum);
        } else {
            coreNum = std::min(coreNum, (totalBlockNum + splitAlignmentBlocks - 1) / splitAlignmentBlocks);
            while (coreNum > 1) {
                const uint64_t averageBlockNum = (totalBlockNum + coreNum - 1) / coreNum;
                const uint64_t alignedBlockNum =
                    (averageBlockNum + splitAlignmentBlocks - 1) / splitAlignmentBlocks * splitAlignmentBlocks;
                if (alignedBlockNum <= (totalBlockNum - 1) / (coreNum - 1)) {
                    break;
                }
                --coreNum;
            }
        }
    }

    const uint64_t totalBlockNum = inputLengthAlign32 / blockSize;
    const uint64_t splitAlignmentBlocks = splitAlignment / OUTPUT_BLOCK_ELEMENTS;
    uint64_t everyCoreInputBlockNum = totalBlockNum;
    uint64_t tailBlockNum = 0;
    if (coreNum > 1) {
        if (splitAlignment == OUTPUT_BLOCK_ELEMENTS) {
            everyCoreInputBlockNum = totalBlockNum / coreNum;
            tailBlockNum = totalBlockNum % coreNum;
        } else {
            everyCoreInputBlockNum =
                ((totalBlockNum + coreNum - 1) / coreNum + splitAlignmentBlocks - 1) /
                splitAlignmentBlocks * splitAlignmentBlocks;
            tailBlockNum = coreNum - 1;
        }
    }
    const uint64_t smallCoreInputBlockNum =
        splitAlignment == OUTPUT_BLOCK_ELEMENTS || tailBlockNum == 0 ?
        everyCoreInputBlockNum : totalBlockNum - everyCoreInputBlockNum * tailBlockNum;
    const uint64_t smallCoreDataNum = smallCoreInputBlockNum * OUTPUT_BLOCK_ELEMENTS;
    uint64_t smallCoreLoopNum = smallCoreInputBlockNum / ubPartBlockNum;
    smallCoreLoopNum += (smallCoreInputBlockNum % ubPartBlockNum) == 0 ? 0 : 1;
    uint64_t smallCoreTailDataNum = smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
    smallCoreTailDataNum = smallCoreTailDataNum == 0 ? ubPartDataNum : smallCoreTailDataNum;

    uint64_t bigCoreDataNum = 0;
    uint64_t bigCoreLoopNum = 0;
    uint64_t bigCoreTailDataNum = 0;
    if (tailBlockNum != 0) {
        const uint64_t bigCoreInputBlockNum = splitAlignment == OUTPUT_BLOCK_ELEMENTS ?
            everyCoreInputBlockNum + 1 : everyCoreInputBlockNum;
        bigCoreDataNum = bigCoreInputBlockNum * OUTPUT_BLOCK_ELEMENTS;
        bigCoreLoopNum = bigCoreInputBlockNum / ubPartBlockNum;
        bigCoreLoopNum += (bigCoreInputBlockNum % ubPartBlockNum) == 0 ? 0 : 1;
        bigCoreTailDataNum = bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum - 1);
        bigCoreTailDataNum = bigCoreTailDataNum == 0 ? ubPartDataNum : bigCoreTailDataNum;
        context->SetTilingKey(1);
    } else {
        context->SetTilingKey(0);
    }
    tiling.smallCoreDataNum = smallCoreDataNum;
    tiling.bigCoreDataNum = bigCoreDataNum;
    tiling.ubPartDataNum = ubPartDataNum;
    tiling.smallCoreTailDataNum = smallCoreTailDataNum;
    tiling.bigCoreTailDataNum = bigCoreTailDataNum;
    tiling.smallCoreLoopNum = smallCoreLoopNum;
    tiling.bigCoreLoopNum = bigCoreLoopNum;
    tiling.tailBlockNum = tailBlockNum;
    context->SetBlockDim(coreNum);
    return true;
}

static uint64_t SetExactTiling(
    gert::TilingContext* context, GreaterTilingData& tiling, uint64_t totalLength, uint64_t ubPartDataNum,
    uint64_t coreNum, uint32_t mode, uint64_t parallelSplitAlignment, bool preferAlignedLargeSplit,
    bool alignAllButFinalSplit)
{
    uint64_t usedCoreNum = 1;
    uint64_t alignedUnits = 0;
    bool useAlignedLargeSplit = false;
    if (preferAlignedLargeSplit && parallelSplitAlignment != 0 &&
        totalLength % parallelSplitAlignment == 0) {
        alignedUnits = totalLength / parallelSplitAlignment;
        usedCoreNum = std::min(coreNum, alignedUnits);
        useAlignedLargeSplit = usedCoreNum != 0;
    } else if (parallelSplitAlignment != 0 && totalLength <= ubPartDataNum) {
        alignedUnits = totalLength / parallelSplitAlignment;
        usedCoreNum = std::min(coreNum, alignedUnits);
        while (usedCoreNum > 1 && alignedUnits % usedCoreNum != 0) {
            --usedCoreNum;
        }
    } else if (parallelSplitAlignment != 0 && totalLength % parallelSplitAlignment == 0) {
        alignedUnits = totalLength / parallelSplitAlignment;
        usedCoreNum = std::min(coreNum, alignedUnits);
        useAlignedLargeSplit = usedCoreNum != 0;
    } else if (totalLength > ubPartDataNum) {
        usedCoreNum = std::min(coreNum, (totalLength + OUTPUT_BLOCK_ELEMENTS - 1) / OUTPUT_BLOCK_ELEMENTS);
    }
    uint64_t smallCoreDataNum = usedCoreNum == 0 ? 0 : totalLength / usedCoreNum;
    uint64_t tailBlockNum = usedCoreNum == 0 ? 0 : totalLength % usedCoreNum;
    uint64_t bigCoreDataNum = tailBlockNum == 0 ? 0 : smallCoreDataNum + 1;
    if (useAlignedLargeSplit) {
        const uint64_t smallCoreUnits = alignedUnits / usedCoreNum;
        tailBlockNum = alignedUnits % usedCoreNum;
        smallCoreDataNum = smallCoreUnits * parallelSplitAlignment;
        bigCoreDataNum = tailBlockNum == 0 ? 0 : (smallCoreUnits + 1) * parallelSplitAlignment;
    } else if (alignAllButFinalSplit && parallelSplitAlignment != 0 && usedCoreNum > 1) {
        // Exact contiguous tensors may end with a sub-alignment tail. Keep the
        // first N-1 partitions aligned and place that global tail on the final
        // core, but only when the resulting work split remains well balanced.
        const uint64_t alignedCoreDataNum =
            totalLength / usedCoreNum / parallelSplitAlignment * parallelSplitAlignment;
        const uint64_t finalCoreDataNum =
            totalLength - alignedCoreDataNum * (usedCoreNum - 1);
        if (alignedCoreDataNum != 0 &&
            finalCoreDataNum <= alignedCoreDataNum + alignedCoreDataNum / 16) {
            tailBlockNum = usedCoreNum - 1;
            bigCoreDataNum = alignedCoreDataNum;
            smallCoreDataNum = finalCoreDataNum;
        }
    }
    const uint64_t smallCoreLoopNum =
        smallCoreDataNum == 0 ? 0 : (smallCoreDataNum + ubPartDataNum - 1) / ubPartDataNum;
    const uint64_t bigCoreLoopNum =
        bigCoreDataNum == 0 ? 0 : (bigCoreDataNum + ubPartDataNum - 1) / ubPartDataNum;

    tiling.smallCoreDataNum = smallCoreDataNum;
    tiling.bigCoreDataNum = bigCoreDataNum;
    tiling.ubPartDataNum = ubPartDataNum;
    tiling.smallCoreTailDataNum =
        smallCoreLoopNum == 0 ? 0 : smallCoreDataNum - ubPartDataNum * (smallCoreLoopNum - 1);
    tiling.bigCoreTailDataNum =
        bigCoreLoopNum == 0 ? 0 : bigCoreDataNum - ubPartDataNum * (bigCoreLoopNum - 1);
    tiling.smallCoreLoopNum = smallCoreLoopNum;
    tiling.bigCoreLoopNum = bigCoreLoopNum;
    tiling.tailBlockNum = tailBlockNum;
    context->SetBlockDim(std::max<uint64_t>(usedCoreNum, 1));
    context->SetTilingKey(2 + mode);
    return std::max<uint64_t>(usedCoreNum, 1);
}

static bool CanUseFp32SteadyWaveMajor(
    const GreaterTilingData& tiling, uint64_t totalLength,
    uint64_t selectedCoreNum)
{
    constexpr uint64_t waveAlignment = 256;
    constexpr uint64_t maxUint32 = std::numeric_limits<uint32_t>::max();
    constexpr uint64_t maxUint64 = std::numeric_limits<uint64_t>::max();
    if (tiling.tailBlockNum < 7 || tiling.tailBlockNum >= 64) {
        return false;
    }
    const uint64_t blockCount = tiling.tailBlockNum + 1;
    if (blockCount != selectedCoreNum ||
        tiling.bigCoreLoopNum < 2 ||
        tiling.bigCoreLoopNum != tiling.smallCoreLoopNum ||
        tiling.bigCoreLoopNum > maxUint32 ||
        tiling.ubPartDataNum == 0 || tiling.ubPartDataNum > maxUint32 ||
        tiling.ubPartDataNum % waveAlignment != 0 ||
        tiling.bigCoreDataNum == 0 ||
        tiling.bigCoreDataNum % waveAlignment != 0 ||
        tiling.bigCoreDataNum > maxUint64 / 64 ||
        tiling.smallCoreDataNum < tiling.bigCoreDataNum ||
        tiling.bigCoreTailDataNum == 0 ||
        tiling.bigCoreTailDataNum > tiling.ubPartDataNum ||
        tiling.bigCoreTailDataNum % waveAlignment != 0 ||
        tiling.smallCoreTailDataNum == 0 ||
        tiling.smallCoreTailDataNum > tiling.ubPartDataNum) {
        return false;
    }

    const uint64_t fullWaveCount = tiling.bigCoreLoopNum - 1;
    if (fullWaveCount > maxUint64 / tiling.ubPartDataNum) {
        return false;
    }
    const uint64_t corePrefix = fullWaveCount * tiling.ubPartDataNum;
    if (tiling.smallCoreTailDataNum > maxUint64 - corePrefix ||
        tiling.bigCoreDataNum != corePrefix + tiling.bigCoreTailDataNum ||
        tiling.smallCoreDataNum != corePrefix + tiling.smallCoreTailDataNum ||
        tiling.smallCoreTailDataNum < tiling.bigCoreTailDataNum ||
        tiling.smallCoreTailDataNum - tiling.bigCoreTailDataNum !=
            tiling.smallCoreDataNum - tiling.bigCoreDataNum) {
        return false;
    }

    const uint64_t regularTotal = blockCount * tiling.bigCoreDataNum;
    const uint64_t finalDelta =
        tiling.smallCoreDataNum - tiling.bigCoreDataNum;
    return finalDelta <= maxUint64 - regularTotal &&
        regularTotal + finalDelta == totalLength;
}

// tiling 分发入口
static ge::graphStatus GreaterTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubLength = 0;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubLength);
    const uint64_t coreNum = static_cast<uint64_t>(ascendcPlatform.GetCoreNum());

    const auto* inputDesc = context->GetInputDesc(0);
    const auto* inputDesc2 = context->GetInputDesc(1);
    if (inputDesc == nullptr || inputDesc2 == nullptr || inputDesc->GetDataType() != inputDesc2->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(inputDesc->GetDataType(), dataTypeLength);
    if (coreNum == 0 || dataTypeLength == 0 || ubLength <= UB_RESERVED_BYTES) {
        return ge::GRAPH_FAILED;
    }
    ubLength -= UB_RESERVED_BYTES;
    const uint64_t blockSize = OUTPUT_BLOCK_ELEMENTS * dataTypeLength;
    const uint64_t ubPartLength = ubLength / GetUbPartNum(inputDesc->GetDataType());
    const uint64_t ubPartBlockNum = ubPartLength / blockSize;
    if (ubPartBlockNum == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubPartDataNum = ubPartBlockNum * OUTPUT_BLOCK_ELEMENTS;
    if (inputDesc->GetDataType() == ge::DT_INT32) {
        ubPartDataNum = std::min(ubPartDataNum, INT32_TILE_ELEMENTS);
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT && ubPartDataNum >= 256) {
        ubPartDataNum = ubPartDataNum / 256 * 256;
    }
    constexpr uint64_t fp32SingleBufferBytesPerElement =
        2 * sizeof(float) + sizeof(int8_t) + sizeof(uint16_t);
    const uint64_t fp32SingleBufferUbPartDataNum =
        ubLength / fp32SingleBufferBytesPerElement / 256 * 256;
    constexpr uint64_t fp32InplaceBytesPerElement =
        4 * sizeof(float) + 2 * sizeof(int8_t);
    // Two independent input generations and one paired-stage output slot.
    // Both stages remain compact in GM, so their byte outputs share one DMA.
    // The existing UB reserve covers queue metadata and alignment padding.
    constexpr uint64_t fp32InplacePreferredTileElements = 18944;
    const uint64_t fp32InplaceUbPartDataNum =
        std::min(
            ubLength / fp32InplaceBytesPerElement / 256 * 256,
            fp32InplacePreferredTileElements);

    BroadcastInfo broadcastInfo {};
    if (!GetBroadcastInfo(context, broadcastInfo)) {
        return ge::GRAPH_FAILED;
    }
    bool useInt32DirectLinearKey = false;
    GreaterInt32DirectTilingData int32DirectTiling {};
    uint64_t requiredWorkspaceBytes = 0;
    const bool int8BenchmarkScope =
        inputDesc->GetDataType() == ge::DT_INT8 &&
        broadcastInfo.mode == GREATER_BROADCAST_CONTIGUOUS;
    bool benchmarkTilePresent = false;
    bool benchmarkAivPresent = false;
    uint64_t benchmarkTileElements = 0;
    uint64_t benchmarkAivCap = 0;
    if (int8BenchmarkScope &&
        (!ReadBenchmarkUint64(
             "GREATER_BENCH_INT8_TILE_ELEMENTS", benchmarkTilePresent,
             benchmarkTileElements) ||
         !ReadBenchmarkUint64(
             "GREATER_BENCH_INT8_AIV_CAP", benchmarkAivPresent,
             benchmarkAivCap))) {
        std::fprintf(stderr, "GREATER_INT8_BENCH invalid_unsigned_value\n");
        return ge::GRAPH_FAILED;
    }
    const bool int8BenchmarkRequested =
        benchmarkTilePresent || benchmarkAivPresent;
    bool int32CohortPresent = false;
    uint64_t requestedInt32Cohort = 1;
    bool int32LargeTilePresent = false;
    uint64_t requestedInt32LargeTile = 0;
    const bool int32BenchmarkScope =
        inputDesc->GetDataType() == ge::DT_INT32 &&
        broadcastInfo.mode == GREATER_BROADCAST_GENERAL;
    if (int32BenchmarkScope &&
        (!ReadBenchmarkUint64(
             "GREATER_BENCH_INT32_COHORT_SIZE", int32CohortPresent,
             requestedInt32Cohort) ||
         !ReadBenchmarkUint64(
             "GREATER_BENCH_INT32_LARGE_TILE", int32LargeTilePresent,
             requestedInt32LargeTile))) {
        std::fprintf(stderr, "GREATER_INT32_BENCH invalid_unsigned_value\n");
        return ge::GRAPH_FAILED;
    }
    if (int32CohortPresent && requestedInt32Cohort != 1 &&
        requestedInt32Cohort != 2 && requestedInt32Cohort != 4 &&
        requestedInt32Cohort != 5) {
        std::fprintf(
            stderr, "GREATER_INT32_BENCH invalid_cohort requested=%llu\n",
            static_cast<unsigned long long>(requestedInt32Cohort));
        return ge::GRAPH_FAILED;
    }
    if (int32LargeTilePresent && requestedInt32LargeTile > 1) {
        std::fprintf(
            stderr,
            "GREATER_INT32_BENCH invalid_large_tile requested=%llu\n",
            static_cast<unsigned long long>(requestedInt32LargeTile));
        return ge::GRAPH_FAILED;
    }
    const bool canUseLegacyPath = broadcastInfo.mode == GREATER_BROADCAST_CONTIGUOUS &&
                                  broadcastInfo.totalLength != 0 &&
                                  broadcastInfo.totalLength % OUTPUT_BLOCK_ELEMENTS == 0 &&
                                  !int8BenchmarkRequested;
    if (canUseLegacyPath) {
        GreaterTilingData* tiling = context->GetTilingData<GreaterTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t legacyGmAlignmentElements =
            inputDesc->GetDataType() == ge::DT_BF16 ? OUTPUT_BLOCK_ELEMENTS : 256;
        const uint64_t legacyUbPartDataNum = inputDesc->GetDataType() == ge::DT_FLOAT ?
            fp32SingleBufferUbPartDataNum : ubPartDataNum;
        if (!SetLegacyAlignedTiling(
                context, *tiling, broadcastInfo.totalLength, dataTypeLength, legacyUbPartDataNum, coreNum,
                legacyGmAlignmentElements)) {
            return ge::GRAPH_FAILED;
        }
    } else {
        GreaterTilingData* tiling = context->GetTilingData<GreaterTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        SetBroadcastFields(*tiling, broadcastInfo);
        const uint64_t parallelSplitAlignment = GetAdjacentParallelSplitAlignment(
            broadcastInfo, inputDesc->GetDataType(), dataTypeLength);
        const bool useBf16RepeatedRowPath =
            CanUseBf16RepeatedRowPath(broadcastInfo, inputDesc->GetDataType());
        const uint64_t adjacentUbPartDataNum = GetAdjacentUbPartDataNum(
            ubLength, inputDesc->GetDataType(), dataTypeLength, broadcastInfo, true);
        const bool useAdjacentScalarBlockPath = CanUseAdjacentScalarBlockPath(
            broadcastInfo, inputDesc->GetDataType(), dataTypeLength, adjacentUbPartDataNum,
            parallelSplitAlignment);
        const uint64_t rawAdjacentUbPartDataNum = GetAdjacentUbPartDataNum(
            ubLength, inputDesc->GetDataType(), dataTypeLength, broadcastInfo, false);
        const uint64_t paddedPlaneUbPartDataNum = rawAdjacentUbPartDataNum == 0 ?
            ubPartDataNum : rawAdjacentUbPartDataNum;
        const bool useAdjacentPaddedPlanePath = CanUseAdjacentPaddedPlanePath(
            broadcastInfo, inputDesc->GetDataType(), dataTypeLength, paddedPlaneUbPartDataNum,
            parallelSplitAlignment);
        const bool useInt8AdjacentPlaneSplit = CanUseInt8AdjacentPlaneSplit(
            broadcastInfo, inputDesc->GetDataType(), dataTypeLength, ubPartDataNum,
            parallelSplitAlignment);
        uint64_t int32PaddedRowSplitAlignment = 0;
        const uint64_t int32PaddedRowUbPartDataNum = std::min(
            rawAdjacentUbPartDataNum == 0 ? ubPartDataNum : rawAdjacentUbPartDataNum,
            INT32_PADDED_ROW_TILE_ELEMENTS);
        const bool useInt32PaddedRowPath = CanUseInt32PaddedRowPath(
            broadcastInfo, inputDesc->GetDataType(), dataTypeLength, int32PaddedRowUbPartDataNum,
            int32PaddedRowSplitAlignment);
        const uint64_t int32LargeTileElements =
            useInt32PaddedRowPath && requestedInt32LargeTile != 0 ?
            GetInt32LargeTileElements(ubLength, broadcastInfo) : 0;
        const bool useInt32LargeTile =
            int32LargeTileElements > int32PaddedRowUbPartDataNum;
        uint64_t reusedInputUbPartDataNum = ubPartDataNum;
        bool canUseBf16RepeatedRowBatch = false;
        if (inputDesc->GetDataType() == ge::DT_BF16 &&
            parallelSplitAlignment == 0 &&
            !useAdjacentScalarBlockPath && !useAdjacentPaddedPlanePath) {
            uint64_t availableUbLength = ubLength;
            bool repeatedRowCacheReserved = false;
            if (useBf16RepeatedRowPath) {
                const uint64_t rowLength =
                    broadcastInfo.outShape[broadcastInfo.rank - 1];
                const uint64_t rowComputeDataNum =
                    (rowLength * dataTypeLength + 255) / 256 * 256 /
                    dataTypeLength;
                const uint64_t rowCacheBytes =
                    rowComputeDataNum * sizeof(float) + 2 * 256;
                if (rowCacheBytes < availableUbLength) {
                    availableUbLength -= rowCacheBytes;
                    repeatedRowCacheReserved = true;
                }
            }
            const bool scalarInputUsesOneBlock =
                broadcastInfo.mode == GREATER_BROADCAST_X1_SCALAR ||
                broadcastInfo.mode == GREATER_BROADCAST_X2_SCALAR;
            if (useBf16RepeatedRowPath && repeatedRowCacheReserved) {
                // Two raw BF16 slots, two independent byte-output slots,
                // and two FP32 work buffers: 2+2+1+1+4+4 = 14 bytes/element.
                // The idle streamed FP32 buffer holds the compact row until
                // expansion, so only 256 bytes of Broadcast scratch remain.
                // 128-element alignment also bounds the rounded row Cast.
                constexpr uint64_t broadcastScratchBytes = 256;
                constexpr uint64_t bf16VectorElements = 128;
                reusedInputUbPartDataNum =
                    (ubLength - broadcastScratchBytes) / 14 /
                    bf16VectorElements * bf16VectorElements;
            } else {
                const uint64_t reusedInputUbPartLength =
                    availableUbLength / (scalarInputUsesOneBlock ? 7 : 9);
                reusedInputUbPartDataNum =
                    reusedInputUbPartLength / blockSize * OUTPUT_BLOCK_ELEMENTS;
            }
            if (useBf16RepeatedRowPath) {
                const uint64_t rowLength =
                    broadcastInfo.outShape[broadcastInfo.rank - 1];
                const uint64_t paddedRowLength =
                    (rowLength + OUTPUT_BLOCK_ELEMENTS - 1) /
                    OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
                canUseBf16RepeatedRowBatch = repeatedRowCacheReserved &&
                    paddedRowLength <= reusedInputUbPartDataNum;
                if (!canUseBf16RepeatedRowBatch) {
                    const uint64_t fallbackUbPartLength =
                        ubLength / (scalarInputUsesOneBlock ? 7 : 9);
                    reusedInputUbPartDataNum =
                        fallbackUbPartLength / blockSize *
                        OUTPUT_BLOCK_ELEMENTS;
                }
            }
        }
        uint64_t paddedPlaneLength = 0;
        if (useAdjacentPaddedPlanePath) {
            const uint32_t tailIndex = broadcastInfo.rank - 1;
            const uint32_t rowIndex = broadcastInfo.rank - 2;
            const uint64_t paddedRowLength =
                (broadcastInfo.outShape[tailIndex] + OUTPUT_BLOCK_ELEMENTS - 1) /
                OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
            if (!MultiplyChecked(broadcastInfo.outShape[rowIndex], paddedRowLength, paddedPlaneLength)) {
                return ge::GRAPH_FAILED;
            }
        }
        uint64_t exactUbPartDataNum = useAdjacentScalarBlockPath ?
            adjacentUbPartDataNum : reusedInputUbPartDataNum;
        if (inputDesc->GetDataType() == ge::DT_FLOAT &&
            broadcastInfo.mode == GREATER_BROADCAST_CONTIGUOUS) {
            exactUbPartDataNum = fp32InplaceUbPartDataNum;
        }
        if (useAdjacentPaddedPlanePath && exactUbPartDataNum < paddedPlaneLength) {
            exactUbPartDataNum = paddedPlaneLength;
        }
        if (useInt32PaddedRowPath) {
            exactUbPartDataNum = useInt32LargeTile ?
                int32LargeTileElements : int32PaddedRowUbPartDataNum;
        }
        if (inputDesc->GetDataType() == ge::DT_INT8 && !useAdjacentPaddedPlanePath &&
            parallelSplitAlignment != 0 &&
            parallelSplitAlignment <= exactUbPartDataNum) {
            exactUbPartDataNum = exactUbPartDataNum / parallelSplitAlignment * parallelSplitAlignment;
        }
        const bool alignFp32ContiguousSplit =
            inputDesc->GetDataType() == ge::DT_FLOAT &&
            broadcastInfo.mode == GREATER_BROADCAST_CONTIGUOUS &&
            broadcastInfo.totalLength > exactUbPartDataNum;
        const bool alignInt8ContiguousSplit =
            inputDesc->GetDataType() == ge::DT_INT8 &&
            broadcastInfo.mode == GREATER_BROADCAST_CONTIGUOUS &&
            broadcastInfo.totalLength > exactUbPartDataNum;
        if (alignInt8ContiguousSplit && exactUbPartDataNum >= 256) {
            exactUbPartDataNum = exactUbPartDataNum / 256 * 256;
        }
        uint64_t exactCoreNum = coreNum;
        if (benchmarkAivPresent) {
            if (benchmarkAivCap == 0 || benchmarkAivCap > coreNum) {
                std::fprintf(
                    stderr,
                    "GREATER_INT8_BENCH invalid_aiv requested=%llu available=%llu\n",
                    static_cast<unsigned long long>(benchmarkAivCap),
                    static_cast<unsigned long long>(coreNum));
                return ge::GRAPH_FAILED;
            }
            exactCoreNum = benchmarkAivCap;
        }
        const bool alignExactContiguousSplit =
            alignFp32ContiguousSplit || alignInt8ContiguousSplit;
        const uint64_t selectedSplitAlignment = useInt32PaddedRowPath ?
            int32PaddedRowSplitAlignment :
            (useBf16RepeatedRowPath ? broadcastInfo.outShape[broadcastInfo.rank - 1] :
             (alignExactContiguousSplit ? 256 : parallelSplitAlignment));

        // Determine the key-7 eligibility with the original conservative UB
        // capacity.  In the eligible range every core owns more than either
        // tile capacity, so recomputing with the segmented macro tile can
        // only change loop/tail counts, not the work partition.
        SetExactTiling(
            context, *tiling, broadcastInfo.totalLength, exactUbPartDataNum,
            exactCoreNum, broadcastInfo.mode, selectedSplitAlignment,
            useInt8AdjacentPlaneSplit, alignExactContiguousSplit);
        const bool provisionalInt8MacroTileKey =
            inputDesc->GetDataType() == ge::DT_INT8 &&
            broadcastInfo.mode == GREATER_BROADCAST_CONTIGUOUS &&
            std::max(tiling->smallCoreDataNum, tiling->bigCoreDataNum) >=
                INT8_REUSABLE_PREDICATE_MIN_CORE_ELEMENTS &&
            std::max(tiling->smallCoreDataNum, tiling->bigCoreDataNum) <=
                INT8_REUSABLE_PREDICATE_MAX_CORE_ELEMENTS;
        if (provisionalInt8MacroTileKey) {
            const uint64_t macroTileDataNum =
                GetInt8MacroTileDataNum(ubLength);
            if (macroTileDataNum <= exactUbPartDataNum) {
                return ge::GRAPH_FAILED;
            }
            exactUbPartDataNum = macroTileDataNum;
        }
        const uint64_t nativeExactUbPartDataNum = exactUbPartDataNum;
        if (benchmarkTilePresent) {
            const uint64_t benchmarkTileAlignment =
                provisionalInt8MacroTileKey ?
                INT8_MACRO_TILE_ALIGNMENT_ELEMENTS : 256;
            if (benchmarkTileElements < 256 ||
                benchmarkTileElements > exactUbPartDataNum ||
                benchmarkTileElements % benchmarkTileAlignment != 0) {
                std::fprintf(
                    stderr,
                    "GREATER_INT8_BENCH invalid_tile requested=%llu native=%llu\n",
                    static_cast<unsigned long long>(benchmarkTileElements),
                    static_cast<unsigned long long>(exactUbPartDataNum));
                return ge::GRAPH_FAILED;
            }
            exactUbPartDataNum = benchmarkTileElements;
        }
        const uint64_t selectedCoreNum = SetExactTiling(
            context, *tiling, broadcastInfo.totalLength, exactUbPartDataNum, exactCoreNum, broadcastInfo.mode,
            selectedSplitAlignment, useInt8AdjacentPlaneSplit, alignExactContiguousSplit);
        if (useInt32PaddedRowPath) {
            const uint32_t tailIndex = broadcastInfo.rank - 1;
            const uint32_t rowIndex = broadcastInfo.rank - 2;
            const uint64_t rowLength = broadcastInfo.outShape[tailIndex];
            const uint64_t rowCount = broadcastInfo.outShape[rowIndex];
            const uint64_t paddedRowLength =
                (rowLength + OUTPUT_BLOCK_ELEMENTS - 1) /
                OUTPUT_BLOCK_ELEMENTS * OUTPUT_BLOCK_ELEMENTS;
            const uint64_t rowsPerTile =
                paddedRowLength == 0 ? 0 :
                std::min(rowCount, exactUbPartDataNum / paddedRowLength);
            const uint64_t tilesPerGroup =
                rowsPerTile == 0 ? 0 :
                (rowCount + rowsPerTile - 1) / rowsPerTile;
            uint64_t naturalGroupCount = 1;
            for (uint32_t index = 0; index < rowIndex; ++index) {
                if (!MultiplyChecked(
                        naturalGroupCount, broadcastInfo.outShape[index],
                        naturalGroupCount)) {
                    return ge::GRAPH_FAILED;
                }
            }

            constexpr uint64_t defaultCohortSize = 4;
            constexpr uint64_t minCohortTilesPerGroup = 8;
            constexpr uint64_t minCohortGroupsPerCore = 6;
            const bool denseCohortWorkload =
                tilesPerGroup >= minCohortTilesPerGroup &&
                naturalGroupCount >=
                    selectedCoreNum * minCohortGroupsPerCore;
            // Adjacent AIV pairs keep each cohort's row-cache and input
            // service interval short while dense natural groups retain enough
            // independent work to hide the row DMA.  Sparse and uneven
            // workloads remain guarded by the generic eligibility checks.
            const uint64_t productionCohortSize =
                denseCohortWorkload ? 2 : defaultCohortSize;
            uint64_t cohortSize = int32CohortPresent ?
                requestedInt32Cohort :
                (denseCohortWorkload ? productionCohortSize : 1);
            const bool compactRows =
                paddedRowLength != rowLength && rowCount >= 4;
            const uint64_t groupsPerWave = cohortSize == 0 ? 0 :
                selectedCoreNum / cohortSize;
            const bool cohortEligible =
                cohortSize > 1 && compactRows && rowCount < 256 &&
                selectedCoreNum % cohortSize == 0 &&
                tilesPerGroup >= cohortSize && groupsPerWave != 0 &&
                naturalGroupCount / groupsPerWave >= 2;
            if (cohortSize > 1 && !cohortEligible) {
                cohortSize = 1;
            }
            tiling->reserved = static_cast<uint32_t>(cohortSize);
            if (useInt32LargeTile) {
                tiling->reserved |= GREATER_INT32_LARGE_TILE_FLAG;
            }
            // Two half-sized int32 input batches occupy one exact tile.  One
            // output slot reserves nine bytes per half-tile: eight compact
            // batches plus one overlapping fp16 materialization window.  The
            // packed predicate covers one half-tile.  A row cache and one DMA
            // row complete the fixed allocation.
            constexpr uint64_t directRowPipelineBytesNumerator = 17;
            constexpr uint64_t directRowPipelineBytesDenominator = 2;
            constexpr uint64_t directRowPipelineFixedBytes = 256;
            constexpr uint64_t directRowPipelineBroadcastRows = 9;
            uint64_t directRowPipelineVariableBytes = 0;
            const bool directRowPipelineWholeBytesFit =
                MultiplyChecked(
                    exactUbPartDataNum,
                    directRowPipelineBytesNumerator,
                    directRowPipelineVariableBytes);
            if (directRowPipelineWholeBytesFit) {
                directRowPipelineVariableBytes /=
                    directRowPipelineBytesDenominator;
            }
            const uint64_t directRowPipelinePredicateBytes =
                exactUbPartDataNum / 16;
            const bool directRowPipelineVariableBytesFit =
                directRowPipelineWholeBytesFit &&
                directRowPipelineVariableBytes <=
                    std::numeric_limits<uint64_t>::max() -
                    directRowPipelinePredicateBytes;
            if (directRowPipelineVariableBytesFit) {
                directRowPipelineVariableBytes +=
                    directRowPipelinePredicateBytes;
            }
            const bool directRowPipelineUbFits =
                directRowPipelineVariableBytesFit &&
                ubLength >= directRowPipelineFixedBytes &&
                paddedRowLength <=
                    (ubLength - directRowPipelineFixedBytes) /
                    (directRowPipelineBroadcastRows * sizeof(int32_t)) &&
                directRowPipelineVariableBytes <=
                    ubLength - directRowPipelineFixedBytes -
                    paddedRowLength * directRowPipelineBroadcastRows *
                        sizeof(int32_t);
            constexpr uint64_t directRowGroupRows = 8;
            constexpr uint64_t minDirectRowDmaBytes = 16 * 1024;
            constexpr uint64_t maxDirectRowDmaBytes =
                std::numeric_limits<uint16_t>::max();
            const bool directRowDmaSizeIsRepresentable =
                rowLength <=
                    std::numeric_limits<uint64_t>::max() /
                    (directRowGroupRows * sizeof(int32_t));
            const uint64_t directRowDmaBytes =
                directRowDmaSizeIsRepresentable ?
                rowLength * directRowGroupRows * sizeof(int32_t) : 0;
            const bool directRowDmaIsLarge =
                directRowDmaSizeIsRepresentable &&
                directRowDmaBytes >= minDirectRowDmaBytes &&
                directRowDmaBytes <= maxDirectRowDmaBytes;
            const uint64_t directRowBatchCount =
                (rowCount + directRowGroupRows - 1) /
                directRowGroupRows;
            const bool directRowOutputCommandIsLarge =
                cohortSize != 0 &&
                directRowBatchCount / cohortSize >= 2;
            const bool useDirectRowPipeline =
                cohortEligible && !useInt32LargeTile &&
                directRowDmaIsLarge &&
                directRowOutputCommandIsLarge &&
                paddedRowLength % 64 == 0 &&
                rowsPerTile % directRowGroupRows == 0 &&
                exactUbPartDataNum % 128 == 0 &&
                directRowPipelineUbFits;
            if (useDirectRowPipeline) {
                tiling->reserved |=
                    GREATER_INT32_DIRECT_ROW_PIPELINE_FLAG;
                if (HasLinearInt32GroupOffsets(broadcastInfo)) {
                    tiling->reserved |=
                        GREATER_INT32_LINEAR_GROUP_OFFSETS_FLAG;
                    useInt32DirectLinearKey = true;
                    int32DirectTiling.rowLength = rowLength;
                    int32DirectTiling.rowCount = rowCount;
                    int32DirectTiling.naturalGroupCount =
                        naturalGroupCount;
                    int32DirectTiling.cohortSize =
                        static_cast<uint32_t>(cohortSize);
                    int32DirectTiling.x1Rows =
                        broadcastInfo.x1Stride[tailIndex] == 1 &&
                        broadcastInfo.x1Stride[rowIndex] == 0 ? 1U : 0U;
                    // One private packed slot per complete eight-row piece.
                    // Two phases own the same flat partition; no global barrier.
                    uint64_t packedUbBytes = 0;
                    uint64_t packedPieceCount = 0;
                    uint64_t packedWorkspaceBytes = 0;
                    const bool packedSizesFit =
                        MultiplyChecked(paddedRowLength, 182, packedUbBytes) &&
                        MultiplyChecked(naturalGroupCount, rowCount / 8,
                            packedPieceCount) &&
                        MultiplyChecked(packedPieceCount, paddedRowLength,
                            packedWorkspaceBytes);
                    if (!int32CohortPresent && !int32LargeTilePresent &&
                        rowCount % 8 == 0 && selectedCoreNum != 0 &&
                        packedSizesFit && ubLength >= 256 &&
                        packedUbBytes <= ubLength - 256 &&
                        paddedRowLength / 16 <= 255 &&
                        paddedRowLength * 8 / 32 <=
                            std::numeric_limits<uint16_t>::max() &&
                        packedPieceCount / selectedCoreNum >= 32) {
                        const uint64_t systemWorkspaceBytes =
                            ascendcPlatform.GetLibApiWorkSpaceSize();
                        if (packedWorkspaceBytes <=
                            std::numeric_limits<size_t>::max() -
                                systemWorkspaceBytes) {
                            requiredWorkspaceBytes =
                                systemWorkspaceBytes + packedWorkspaceBytes;
                            int32DirectTiling.cohortSize |=
                                GREATER_INT32_PACKED_GM_FLAG;
                        }
                    }
                }
            }
            if (int32CohortPresent || int32LargeTilePresent) {
                std::fprintf(
                    stderr,
                    "GREATER_INT32_BENCH requested=%llu cohort=%llu large=%u "
                    "block_dim=%llu groups=%llu tiles_per_group=%llu "
                    "rows_per_tile=%llu\n",
                    static_cast<unsigned long long>(requestedInt32Cohort),
                    static_cast<unsigned long long>(cohortSize),
                    useInt32LargeTile ? 1U : 0U,
                    static_cast<unsigned long long>(selectedCoreNum),
                    static_cast<unsigned long long>(naturalGroupCount),
                    static_cast<unsigned long long>(tilesPerGroup),
                    static_cast<unsigned long long>(rowsPerTile));
            }
        }
        const bool useInt8ReusablePredicateKey =
            inputDesc->GetDataType() == ge::DT_INT8 &&
            broadcastInfo.mode == GREATER_BROADCAST_CONTIGUOUS &&
            std::max(tiling->smallCoreDataNum, tiling->bigCoreDataNum) >=
                INT8_REUSABLE_PREDICATE_MIN_CORE_ELEMENTS &&
            std::max(tiling->smallCoreDataNum, tiling->bigCoreDataNum) <=
                INT8_REUSABLE_PREDICATE_MAX_CORE_ELEMENTS;
        if (useInt8ReusablePredicateKey != provisionalInt8MacroTileKey) {
            return ge::GRAPH_FAILED;
        }
        if (useInt32DirectLinearKey) {
            // INT32 key 8 is reserved for the compact direct-row route.
            // Adjacent padded-plane INT32 layouts remain correct through
            // the default general key instead of sharing this ABI.
            context->SetTilingKey(8);
        } else if (canUseBf16RepeatedRowBatch) {
            context->SetTilingKey(9);
        } else if (useAdjacentScalarBlockPath) {
            const bool singleTile = tiling->smallCoreLoopNum <= 1 && tiling->bigCoreLoopNum <= 1;
            context->SetTilingKey(singleTile ? 7 : 6);
        } else if (
            useAdjacentPaddedPlanePath &&
            inputDesc->GetDataType() != ge::DT_INT32) {
            context->SetTilingKey(8);
        } else if (useInt32PaddedRowPath) {
            context->SetTilingKey(9);
        } else if (
            inputDesc->GetDataType() == ge::DT_FLOAT &&
            broadcastInfo.mode == GREATER_BROADCAST_CONTIGUOUS &&
            CanUseFp32SteadyWaveMajor(
                *tiling, broadcastInfo.totalLength, selectedCoreNum)) {
            // Key 9 is shared only by disjoint dtype routes.  Selecting the
            // FP32 specialization here removes the runtime address-mode
            // branch while preserving the exact per-AIV tile-length sequence.
            context->SetTilingKey(9);
        } else if (
            useInt8ReusablePredicateKey) {
            // INT8 never uses the adjacent-scalar keys. Key 6 selects the
            // compile-time wave-major layout; key 7 retains the per-core
            // layout for controlled comparisons.
            context->SetTilingKey(6);
        }
        static bool benchmarkRecordPrinted = false;
        if (int8BenchmarkRequested && !benchmarkRecordPrinted) {
            std::fprintf(
                stderr,
                "GREATER_INT8_BENCH key=%u block_dim=%llu native_tile=%llu tile=%llu small=%llu big=%llu "
                "small_loops=%llu big_loops=%llu tail_blocks=%llu\n",
                useInt8ReusablePredicateKey ? 6U : 2U,
                static_cast<unsigned long long>(selectedCoreNum),
                static_cast<unsigned long long>(nativeExactUbPartDataNum),
                static_cast<unsigned long long>(tiling->ubPartDataNum),
                static_cast<unsigned long long>(tiling->smallCoreDataNum),
                static_cast<unsigned long long>(tiling->bigCoreDataNum),
                static_cast<unsigned long long>(tiling->smallCoreLoopNum),
                static_cast<unsigned long long>(tiling->bigCoreLoopNum),
                static_cast<unsigned long long>(tiling->tailBlockNum));
            benchmarkRecordPrinted = true;
        }
    }

    if (useInt32DirectLinearKey) {
        auto* rawTiling = context->GetRawTilingData();
        if (rawTiling == nullptr ||
            rawTiling->GetCapacity() < sizeof(int32DirectTiling)) {
            return ge::GRAPH_FAILED;
        }
        std::memcpy(
            rawTiling->GetData(), &int32DirectTiling,
            sizeof(int32DirectTiling));
        rawTiling->SetDataSize(sizeof(int32DirectTiling));
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = static_cast<size_t>(requiredWorkspaceBytes);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGreater([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(Greater).Tiling(GreaterTilingFunc).TilingParse<GreaterCompileInfo>(TilingParseForGreater);
} // namespace optiling
