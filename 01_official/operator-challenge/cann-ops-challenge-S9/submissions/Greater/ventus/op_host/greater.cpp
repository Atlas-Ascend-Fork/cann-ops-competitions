/**
 * Greater 算子注册、广播形状推导与主机侧分块规划。
 * 根据数据类型、广播模式和 UB 预算选择连续比较或行列广播处理路径。
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

#include "greater_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
ge::graphStatus InferBroadcastShape(const gert::Shape& x1, const gert::Shape& x2,
                                    gert::Shape& output)
{
    const size_t rank1 = x1.GetDimNum();
    const size_t rank2 = x2.GetDimNum();
    const size_t outputRank = std::max(rank1, rank2);
    output = rank1 >= rank2 ? x1 : x2;
    for (size_t outputAxis = 0; outputAxis < outputRank; ++outputAxis) {
        const int64_t axis1 = static_cast<int64_t>(outputAxis) -
            static_cast<int64_t>(outputRank - rank1);
        const int64_t axis2 = static_cast<int64_t>(outputAxis) -
            static_cast<int64_t>(outputRank - rank2);
        const int64_t dim1 = axis1 < 0 ? 1 : x1.GetDim(static_cast<size_t>(axis1));
        const int64_t dim2 = axis2 < 0 ? 1 : x2.GetDim(static_cast<size_t>(axis2));
        if (dim1 <= 0 || dim2 <= 0 || (dim1 != dim2 && dim1 != 1 && dim2 != 1)) {
            return ge::GRAPH_FAILED;
        }
        output.SetDim(outputAxis, std::max(dim1, dim2));
    }
    return ge::GRAPH_SUCCESS;
}

uint32_t GetAivCoreCount(const gert::TilingContext* context)
{
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    return std::max<uint32_t>(1U, platform.GetCoreNumAiv());
}

bool MultiplyChecked(uint64_t lhs, uint64_t rhs, uint64_t& output)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    output = lhs * rhs;
    return true;
}

uint64_t GetBoxTileElements(ge::DataType dataType)
{

    switch (dataType) {
        case ge::DT_FLOAT16:
            return S9_GREATER_BOX_ELEMENTS_FLOAT16;
        case ge::DT_BF16:
            return S9_GREATER_BOX_ELEMENTS_BFLOAT16;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32;
        case ge::DT_INT8:
            return S9_GREATER_BOX_ELEMENTS_INT8;
        default:
            return 0U;
    }
}

uint64_t GetDensePipelineMinimumElements(ge::DataType dataType,
                                         uint32_t aivCoreCount)
{

    uint64_t tileElements = 0U;
    uint64_t wavesPerCore = 1U;
    switch (dataType) {
        case ge::DT_FLOAT16:
        case ge::DT_INT8:
            tileElements = 11264U;
            break;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            tileElements = 7168U;
            break;
        case ge::DT_BF16:
            tileElements = 7168U;
            wavesPerCore = 8U;
            break;
        default:
            return std::numeric_limits<uint64_t>::max();
    }
    uint64_t minimumElements = 0U;
    if (!MultiplyChecked(tileElements, wavesPerCore, minimumElements) ||
        !MultiplyChecked(minimumElements, aivCoreCount, minimumElements)) {
        return std::numeric_limits<uint64_t>::max();
    }
    return minimumElements;
}

uint64_t AlignUp32(uint64_t value)
{
    return (value + 31U) & ~31ULL;
}

struct RowSlabTypePolicy {
    uint32_t sourceBytes;
    uint32_t computeBytes;
    uint32_t chunkElements;
    bool needsConversion;
    bool needsMaximum;
};

bool GetRowSlabTypePolicy(ge::DataType dataType,
                          RowSlabTypePolicy& policy)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            policy = {2U, 2U, 128U, false, false};
            return true;
        case ge::DT_FLOAT:
            policy = {4U, 4U, 64U, false, false};
            return true;
        case ge::DT_BF16:
            policy = {2U, 4U, 64U, true, false};
            return true;
        case ge::DT_INT32:
            policy = {4U, 4U, 64U, false, true};
            return true;
        case ge::DT_INT8:
            policy = {1U, 2U, 128U, true, false};
            return true;
        default:
            return false;
    }
}

uint64_t GetRowSlabUbBytes(const RowSlabTypePolicy& policy,
                           uint64_t rows, uint64_t cols,
                           uint64_t rowStorageCols,
                           uint32_t pipelineBufferNum)
{
    const uint64_t stripeElements = rows * policy.chunkElements;
    uint64_t bytes = 0U;
    bytes += AlignUp32(rowStorageCols * policy.sourceBytes);
    bytes += pipelineBufferNum *
        AlignUp32(rows * cols * policy.sourceBytes);

    constexpr uint64_t outputStorageBytes = sizeof(uint8_t);
    bytes += pipelineBufferNum *
        AlignUp32(rows * cols * outputStorageBytes);
    bytes += AlignUp32(rows * (policy.chunkElements / 8U));
    bytes += AlignUp32(policy.chunkElements * policy.computeBytes);
    bytes += policy.needsConversion ?
        AlignUp32(rows * cols * policy.computeBytes) : 32U;
    bytes += policy.needsConversion ?
        AlignUp32(rowStorageCols * policy.computeBytes) : 32U;
    bytes += policy.needsMaximum ?
        AlignUp32(stripeElements * sizeof(int32_t)) : 32U;
    return bytes;
}

bool FindRowSlab(const RowSlabTypePolicy& policy, uint64_t rows,
                 uint64_t matrixCols, uint32_t pipelineBufferNum,
                 uint64_t& selectedCols,
                 uint64_t& selectedUbBytes)
{
    const uint64_t chunksInRow =
        matrixCols / policy.chunkElements +
        (matrixCols % policy.chunkElements != 0U ? 1U : 0U);
    if (chunksInRow < 2U) {
        return false;
    }
    const uint64_t rowStorageCols =
        chunksInRow * policy.chunkElements;

    constexpr uint64_t kUbBytes = 192ULL * 1024ULL;
    const uint64_t ubHeadroomBytes =
        pipelineBufferNum == 2U &&
        policy.sourceBytes == sizeof(uint16_t) ? 8ULL * 1024ULL :
        (pipelineBufferNum == 1U && policy.needsConversion &&
         policy.computeBytes == sizeof(float) ? 4ULL * 1024ULL :
                                                2ULL * 1024ULL);
    const uint64_t maxPitchCols =
        255ULL * 32ULL / policy.computeBytes;
    uint64_t slabChunks = std::min<uint64_t>(
        chunksInRow, maxPitchCols / policy.chunkElements);
    while (slabChunks >= 2U) {
        const uint64_t slabCols = slabChunks * policy.chunkElements;
        const uint64_t ubBytes = GetRowSlabUbBytes(
            policy, rows, slabCols, rowStorageCols, pipelineBufferNum);
        if (ubBytes + ubHeadroomBytes <= kUbBytes) {
            selectedCols = slabCols;
            selectedUbBytes = ubBytes;
            return true;
        }
        --slabChunks;
    }
    return false;
}

bool FindAdaptiveRowSlab(const RowSlabTypePolicy& policy, uint64_t rows,
                         uint64_t matrixCols, uint64_t& selectedCols,
                         uint64_t& selectedUbBytes,
                         uint32_t& selectedBufferNum)
{
    selectedBufferNum = 2U;
    if (FindRowSlab(policy, rows, matrixCols, selectedBufferNum,
                    selectedCols, selectedUbBytes)) {
        return true;
    }
    selectedBufferNum = 1U;
    return FindRowSlab(policy, rows, matrixCols, selectedBufferNum,
                       selectedCols, selectedUbBytes);
}

void ConfigureRowSlab(GreaterTilingData& computed, ge::DataType dataType)
{
    if (computed.rowStripeEnabled == 0U ||
        computed.matrixRowsPerTile == 0U ||
        computed.matrixCols == 0U ||
        computed.matrixCols > std::numeric_limits<uint32_t>::max()) {
        return;
    }

    RowSlabTypePolicy policy = {};
    if (!GetRowSlabTypePolicy(dataType, policy)) {
        return;
    }

    uint64_t slabCols = 0U;
    uint64_t ubBytes = 0U;
    uint32_t bufferNum = 0U;
    if (FindAdaptiveRowSlab(policy, computed.matrixRowsPerTile,
                            computed.matrixCols, slabCols, ubBytes,
                            bufferNum) &&
        slabCols <= std::numeric_limits<uint32_t>::max() &&
        ubBytes <= std::numeric_limits<uint32_t>::max()) {
        computed.rowSlabEnabled = 1U;
        computed.rowSlabCols = static_cast<uint32_t>(slabCols);
        computed.rowSlabRows = computed.matrixRowsPerTile;
        computed.rowSlabUbBytes = static_cast<uint32_t>(ubBytes);
        computed.rowSlabBufferNum = bufferNum;
    }
}

uint64_t DivideRoundUp(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0U ? 1U : 0U);
}

uint64_t RowPrefixLoad(uint64_t tilePrefix, uint64_t matrixRows,
                       uint64_t rowsPerTile, uint64_t rowTiles)
{
    const uint64_t completeMatrices = tilePrefix / rowTiles;
    const uint64_t partialTiles = tilePrefix % rowTiles;
    return completeMatrices * matrixRows + partialTiles * rowsPerTile;
}

uint64_t CriticalRowLoad(uint64_t matrixRows, uint64_t matrixOuterCount,
                         uint64_t rowsPerTile, uint64_t blockLimit)
{
    const uint64_t rowTiles = DivideRoundUp(matrixRows, rowsPerTile);
    const uint64_t totalTiles = matrixOuterCount * rowTiles;
    const uint64_t blocks = std::min(blockLimit, totalTiles);
    const uint64_t baseTiles = totalTiles / blocks;
    const uint64_t remainder = totalTiles % blocks;
    uint64_t criticalRows = 0U;
    for (uint64_t block = 0U; block < blocks; ++block) {
        const uint64_t begin = baseTiles * block +
            std::min(block, remainder);
        const uint64_t count = baseTiles +
            (block < remainder ? 1U : 0U);
        const uint64_t rowLoad = RowPrefixLoad(
            begin + count, matrixRows, rowsPerTile, rowTiles) -
            RowPrefixLoad(begin, matrixRows, rowsPerTile, rowTiles);
        criticalRows = std::max(criticalRows, rowLoad);
    }
    return criticalRows;
}

struct RowSlabScheduleMetrics {
    uint64_t rowsPerTile;
    uint64_t rowTiles;
    uint64_t totalTiles;
    uint64_t criticalRows;
    uint64_t criticalTiles;
    uint64_t criticalDescriptors;
    uint64_t totalDescriptors;
    uint64_t criticalOuterLoads;
    uint64_t totalOuterLoads;
};

RowSlabScheduleMetrics EvaluateRowSlabSchedule(
    uint64_t matrixRows, uint64_t matrixOuterCount, uint64_t matrixCols,
    uint64_t rowsPerTile, uint64_t blockLimit,
    const RowSlabTypePolicy& policy)
{
    RowSlabScheduleMetrics metrics = {};
    metrics.rowsPerTile = rowsPerTile;
    metrics.rowTiles = DivideRoundUp(matrixRows, rowsPerTile);
    metrics.totalTiles = matrixOuterCount * metrics.rowTiles;
    const uint64_t blocks = std::min(blockLimit, metrics.totalTiles);
    metrics.criticalRows = CriticalRowLoad(
        matrixRows, matrixOuterCount, rowsPerTile, blockLimit);
    metrics.criticalTiles = DivideRoundUp(metrics.totalTiles, blocks);

    const uint64_t chunks = DivideRoundUp(
        matrixCols, policy.chunkElements);
    uint64_t slabCols = 0U;
    uint64_t ubBytes = 0U;
    uint32_t bufferNum = 0U;
    const uint64_t slabsPerTile = FindAdaptiveRowSlab(
        policy, rowsPerTile, matrixCols, slabCols, ubBytes, bufferNum) ?
            DivideRoundUp(matrixCols, slabCols) : chunks;
    metrics.criticalDescriptors =
        metrics.criticalTiles * slabsPerTile;
    metrics.totalDescriptors = metrics.totalTiles * slabsPerTile;

    const uint64_t baseTiles = metrics.totalTiles / blocks;
    const uint64_t remainder = metrics.totalTiles % blocks;
    for (uint64_t block = 0U; block < blocks; ++block) {
        const uint64_t begin = baseTiles * block +
            std::min(block, remainder);
        const uint64_t count = baseTiles +
            (block < remainder ? 1U : 0U);
        const uint64_t outerLoads =
            (begin + count - 1U) / metrics.rowTiles -
            begin / metrics.rowTiles + 1U;
        metrics.criticalOuterLoads = std::max(
            metrics.criticalOuterLoads, outerLoads);
        metrics.totalOuterLoads += outerLoads;
    }
    return metrics;
}

bool StrictlyDominates(const RowSlabScheduleMetrics& candidate,
                       const RowSlabScheduleMetrics& current)
{
    const bool noWorse =
        candidate.criticalRows <= current.criticalRows &&
        candidate.criticalTiles <= current.criticalTiles &&
        candidate.totalTiles <= current.totalTiles &&
        candidate.criticalDescriptors <= current.criticalDescriptors &&
        candidate.totalDescriptors <= current.totalDescriptors &&
        candidate.criticalOuterLoads <= current.criticalOuterLoads &&
        candidate.totalOuterLoads <= current.totalOuterLoads;
    const bool strictlyBetter =
        candidate.criticalRows < current.criticalRows ||
        candidate.criticalTiles < current.criticalTiles ||
        candidate.totalTiles < current.totalTiles ||
        candidate.criticalDescriptors < current.criticalDescriptors ||
        candidate.totalDescriptors < current.totalDescriptors ||
        candidate.criticalOuterLoads < current.criticalOuterLoads ||
        candidate.totalOuterLoads < current.totalOuterLoads;
    return noWorse && strictlyBetter;
}

uint64_t SelectDominantRowSlabRows(
    uint64_t matrixRows, uint64_t matrixOuterCount, uint64_t matrixCols,
    uint64_t blockLimit, ge::DataType dataType, uint64_t initialRows)
{
    RowSlabTypePolicy policy = {};
    if (!GetRowSlabTypePolicy(dataType, policy)) {
        return initialRows;
    }
    RowSlabScheduleMetrics selected = EvaluateRowSlabSchedule(
        matrixRows, matrixOuterCount, matrixCols,
        initialRows, blockLimit, policy);
    for (uint64_t candidateRows = 1U;
         candidateRows <= S9_GREATER_ROW_STRIPE_MAX_ROWS;
         ++candidateRows) {
        const RowSlabScheduleMetrics candidate = EvaluateRowSlabSchedule(
            matrixRows, matrixOuterCount, matrixCols,
            candidateRows, blockLimit, policy);
        if (StrictlyDominates(candidate, selected)) {
            selected = candidate;
        }
    }
    return selected.rowsPerTile;
}
}

namespace optiling {
// 根据广播模式、数据类型和 UB 预算选择连续比较或行列广播处理路径。
static ge::graphStatus GreaterTiling(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* x1Storage = context->GetInputShape(0);
    const auto* x2Storage = context->GetInputShape(1);
    const auto* x1Desc = context->GetInputDesc(0);
    const auto* x2Desc = context->GetInputDesc(1);
    if (x1Storage == nullptr || x2Storage == nullptr || x1Desc == nullptr || x2Desc == nullptr ||
        x1Desc->GetDataType() != x2Desc->GetDataType()) {
        return ge::GRAPH_FAILED;
    }

    const int elementBytes = ge::GetSizeByDataType(x1Desc->GetDataType());
    if (elementBytes <= 0) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& x1 = x1Storage->GetStorageShape();
    const gert::Shape& x2 = x2Storage->GetStorageShape();
    const size_t sourceRank1 = x1.GetDimNum();
    const size_t sourceRank2 = x2.GetDimNum();
    const size_t normalizedRank = std::max<size_t>(1, std::max(sourceRank1, sourceRank2));
    if (normalizedRank > S9_GREATER_MAX_RANK) {
        return ge::GRAPH_FAILED;
    }

    GreaterTilingData computed = {};
    computed.rank = static_cast<uint32_t>(normalizedRank);
    computed.totalElements = 1;
    bool denseInputs = true;
    for (size_t axis = 0; axis < normalizedRank; ++axis) {
        const int64_t axis1 = static_cast<int64_t>(axis) -
            static_cast<int64_t>(normalizedRank - sourceRank1);
        const int64_t axis2 = static_cast<int64_t>(axis) -
            static_cast<int64_t>(normalizedRank - sourceRank2);
        const int64_t signedDim1 = axis1 < 0 ? 1 : x1.GetDim(static_cast<size_t>(axis1));
        const int64_t signedDim2 = axis2 < 0 ? 1 : x2.GetDim(static_cast<size_t>(axis2));
        if (signedDim1 <= 0 || signedDim2 <= 0 ||
            (signedDim1 != signedDim2 && signedDim1 != 1 && signedDim2 != 1)) {
            return ge::GRAPH_FAILED;
        }
        const uint64_t dim1 = static_cast<uint64_t>(signedDim1);
        const uint64_t dim2 = static_cast<uint64_t>(signedDim2);
        const uint64_t outputDim = std::max(dim1, dim2);
        computed.x1Dims[axis] = dim1;
        computed.x2Dims[axis] = dim2;
        computed.outputDims[axis] = outputDim;
        denseInputs = denseInputs && dim1 == outputDim && dim2 == outputDim;
        if (!MultiplyChecked(computed.totalElements, outputDim, computed.totalElements)) {
            return ge::GRAPH_FAILED;
        }
    }

    uint64_t x1Stride = 1;
    uint64_t x2Stride = 1;
    for (size_t reverse = normalizedRank; reverse > 0; --reverse) {
        const size_t axis = reverse - 1;
        computed.x1Strides[axis] = x1Stride;
        computed.x2Strides[axis] = x2Stride;
        if (!MultiplyChecked(x1Stride, computed.x1Dims[axis], x1Stride) ||
            !MultiplyChecked(x2Stride, computed.x2Dims[axis], x2Stride)) {
            return ge::GRAPH_FAILED;
        }
    }

    int x1Mode = -1;
    int x2Mode = -1;
    size_t innerStart = normalizedRank;
    for (size_t reverse = normalizedRank; reverse > 0; --reverse) {
        const size_t axis = reverse - 1;
        if (computed.outputDims[axis] == 1) {
            innerStart = axis;
            continue;
        }
        const int currentX1Mode = computed.x1Dims[axis] == 1 ? 1 : 0;
        const int currentX2Mode = computed.x2Dims[axis] == 1 ? 1 : 0;
        if (x1Mode < 0) {
            x1Mode = currentX1Mode;
            x2Mode = currentX2Mode;
        } else if (x1Mode != currentX1Mode || x2Mode != currentX2Mode) {
            break;
        }
        innerStart = axis;
    }
    if (innerStart == normalizedRank) {
        innerStart = normalizedRank - 1;
    }
    computed.outerRank = static_cast<uint32_t>(innerStart);
    computed.x1InnerBroadcast = x1Mode == 1 ? 1U : 0U;
    computed.x2InnerBroadcast = x2Mode == 1 ? 1U : 0U;
    computed.rowElements = 1;
    for (size_t axis = innerStart; axis < normalizedRank; ++axis) {
        if (!MultiplyChecked(computed.rowElements, computed.outputDims[axis], computed.rowElements)) {
            return ge::GRAPH_FAILED;
        }
    }

    const uint64_t tileElements = GetBoxTileElements(x1Desc->GetDataType());
    if (tileElements == 0U) {
        return ge::GRAPH_FAILED;
    }
    size_t boxStart = normalizedRank;
    uint64_t boxElements = 1U;
    bool boxHasBroadcast = false;
    for (size_t reverse = normalizedRank; reverse > 0U; --reverse) {
        const size_t axis = reverse - 1U;
        if (computed.outputDims[axis] > tileElements / boxElements) {
            break;
        }
        boxElements *= computed.outputDims[axis];
        boxStart = axis;
        boxHasBroadcast = boxHasBroadcast ||
            computed.x1Dims[axis] != computed.outputDims[axis] ||
            computed.x2Dims[axis] != computed.outputDims[axis];
    }
    computed.boxStart = static_cast<uint32_t>(boxStart);
    computed.boxElements = boxElements;
    computed.boxCount = computed.totalElements / boxElements;
    computed.boxEnabled = boxStart < normalizedRank && boxHasBroadcast ? 1U : 0U;

    if (innerStart > 0U && computed.rowElements <= tileElements) {
        int rowX1Mode = -1;
        int rowX2Mode = -1;
        size_t rowStart = innerStart;
        for (size_t reverse = innerStart; reverse > 0U; --reverse) {
            const size_t axis = reverse - 1U;
            if (computed.outputDims[axis] == 1U) {
                rowStart = axis;
                continue;
            }
            const int currentX1Mode = computed.x1Dims[axis] == 1U ? 1 : 0;
            const int currentX2Mode = computed.x2Dims[axis] == 1U ? 1 : 0;
            if (rowX1Mode < 0) {
                rowX1Mode = currentX1Mode;
                rowX2Mode = currentX2Mode;
            } else if (rowX1Mode != currentX1Mode ||
                       rowX2Mode != currentX2Mode) {
                break;
            }
            rowStart = axis;
        }

        uint64_t matrixRows = 1U;
        for (size_t axis = rowStart; axis < innerStart; ++axis) {
            if (!MultiplyChecked(matrixRows, computed.outputDims[axis],
                                 matrixRows)) {
                return ge::GRAPH_FAILED;
            }
        }
        const uint64_t matrixCols = computed.rowElements;
        const uint64_t rowsPerTile =
            std::min(matrixRows, tileElements / matrixCols);
        const uint32_t x1Pattern =
            (rowX1Mode == 1 ? S9_GREATER_MATRIX_ROW_BROADCAST : 0U) |
            (computed.x1InnerBroadcast != 0U ?
                S9_GREATER_MATRIX_COL_BROADCAST : 0U);
        const uint32_t x2Pattern =
            (rowX2Mode == 1 ? S9_GREATER_MATRIX_ROW_BROADCAST : 0U) |
            (computed.x2InnerBroadcast != 0U ?
                S9_GREATER_MATRIX_COL_BROADCAST : 0U);

        if (matrixRows > 1U && rowsPerTile > 0U &&
            matrixRows <= std::numeric_limits<uint32_t>::max() &&
            rowsPerTile <= std::numeric_limits<uint32_t>::max()) {
            const uint64_t rowTiles = matrixRows / rowsPerTile +
                (matrixRows % rowsPerTile != 0U ? 1U : 0U);
            const uint64_t outerCount =
                computed.totalElements / matrixRows / matrixCols;
            uint64_t totalTiles = 0U;
            if (!MultiplyChecked(outerCount, rowTiles, totalTiles)) {
                return ge::GRAPH_FAILED;
            }
            if (rowTiles <= std::numeric_limits<uint32_t>::max()) {
                computed.matrixEnabled = 1U;
                computed.matrixRowStart = static_cast<uint32_t>(rowStart);
                computed.matrixRows = matrixRows;
                computed.matrixCols = matrixCols;
                computed.matrixOuterCount = outerCount;
                computed.matrixRowsPerTile =
                    static_cast<uint32_t>(rowsPerTile);
                computed.matrixRowTiles = static_cast<uint32_t>(rowTiles);
                computed.matrixTotalTiles = totalTiles;
                computed.matrixX1Pattern = x1Pattern;
                computed.matrixX2Pattern = x2Pattern;
            }
        }
    }

    constexpr uint64_t kTargetBytesPerCore = 32ULL * 1024ULL;
    const uint64_t trafficBytesPerElement = static_cast<uint64_t>(2 * elementBytes + 1);
    uint64_t totalTrafficBytes = std::numeric_limits<uint64_t>::max();
    if (computed.totalElements <= std::numeric_limits<uint64_t>::max() / trafficBytesPerElement) {
        totalTrafficBytes = computed.totalElements * trafficBytesPerElement;
    }
    uint64_t requestedBlocks = totalTrafficBytes / kTargetBytesPerCore;
    requestedBlocks += totalTrafficBytes % kTargetBytesPerCore != 0 ? 1U : 0U;
    requestedBlocks = std::max<uint64_t>(1, requestedBlocks);
    const uint32_t aivCoreCount = GetAivCoreCount(context);
    computed.blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(aivCoreCount, requestedBlocks));
    if (computed.boxEnabled != 0U) {
        computed.blockDim = static_cast<uint32_t>(
            std::min<uint64_t>(computed.blockDim, computed.boxCount));
    }
    if (computed.matrixEnabled != 0U) {
        const bool x1IsPureRow =
            computed.matrixX1Pattern == S9_GREATER_MATRIX_ROW_BROADCAST &&
            computed.matrixX2Pattern == 0U;
        const bool x2IsPureRow =
            computed.matrixX2Pattern == S9_GREATER_MATRIX_ROW_BROADCAST &&
            computed.matrixX1Pattern == 0U;

        const bool rowSpecializationSafe =
            computed.matrixOuterCount == 1U;
        if ((x1IsPureRow || x2IsPureRow) && rowSpecializationSafe) {

            const uint64_t targetTilesPerOuter = std::max<uint64_t>(
                1U, (aivCoreCount + computed.matrixOuterCount - 1U) /
                    computed.matrixOuterCount);
            const uint64_t floorRows = std::max<uint64_t>(
                1U, computed.matrixRows / targetTilesPerOuter);
            const uint64_t ceilRows = std::max<uint64_t>(
                1U, DivideRoundUp(
                    computed.matrixRows, targetTilesPerOuter));
            const uint64_t boundedFloorRows = std::min<uint64_t>(
                S9_GREATER_ROW_STRIPE_MAX_ROWS, floorRows);
            const uint64_t boundedCeilRows = std::min<uint64_t>(
                S9_GREATER_ROW_STRIPE_MAX_ROWS, ceilRows);
            const uint64_t rowBlockLimit = std::min<uint64_t>(
                aivCoreCount, requestedBlocks);
            const uint64_t floorCriticalRows = CriticalRowLoad(
                computed.matrixRows, computed.matrixOuterCount,
                boundedFloorRows, rowBlockLimit);
            const uint64_t ceilCriticalRows = CriticalRowLoad(
                computed.matrixRows, computed.matrixOuterCount,
                boundedCeilRows, rowBlockLimit);

            const uint64_t occupancyRows =
                ceilCriticalRows <= floorCriticalRows ?
                    boundedCeilRows : boundedFloorRows;
            const uint64_t stripeRows = SelectDominantRowSlabRows(
                computed.matrixRows, computed.matrixOuterCount,
                computed.matrixCols, rowBlockLimit,
                x1Desc->GetDataType(), occupancyRows);
            const uint64_t stripeTiles = computed.matrixRows / stripeRows +
                (computed.matrixRows % stripeRows != 0U ? 1U : 0U);
            uint64_t stripeTotalTiles = 0U;
            if (!MultiplyChecked(computed.matrixOuterCount, stripeTiles,
                                 stripeTotalTiles) ||
                stripeTiles > std::numeric_limits<uint32_t>::max()) {
                return ge::GRAPH_FAILED;
            }
            computed.matrixRowsPerTile = static_cast<uint32_t>(stripeRows);
            computed.matrixRowTiles = static_cast<uint32_t>(stripeTiles);
            computed.matrixTotalTiles = stripeTotalTiles;
            computed.rowStripeEnabled = 1U;

            computed.blockDim = static_cast<uint32_t>(
                std::min<uint64_t>(aivCoreCount, requestedBlocks));
        }

        const bool x1RowX2Column =
            computed.matrixX1Pattern ==
                S9_GREATER_MATRIX_ROW_BROADCAST &&
            computed.matrixX2Pattern ==
                S9_GREATER_MATRIX_COL_BROADCAST;
        const bool x1ColumnX2Row =
            computed.matrixX1Pattern ==
                S9_GREATER_MATRIX_COL_BROADCAST &&
            computed.matrixX2Pattern ==
                S9_GREATER_MATRIX_ROW_BROADCAST;
        const bool scalarCompareType =
            x1Desc->GetDataType() == ge::DT_FLOAT16 ||
            x1Desc->GetDataType() == ge::DT_FLOAT;

        const bool crossScalarShape =
            (x1RowX2Column || x1ColumnX2Row) &&
            scalarCompareType && computed.matrixOuterCount == 1U &&
            computed.matrixCols >= 1024U &&
            computed.matrixRows >= computed.blockDim;
        if (crossScalarShape) {
            const uint64_t crossRows = std::min<uint64_t>(
                S9_GREATER_ROW_STRIPE_MAX_ROWS,
                DivideRoundUp(computed.matrixRows, computed.blockDim));
            const uint64_t crossTiles =
                DivideRoundUp(computed.matrixRows, crossRows);
            if (crossTiles > std::numeric_limits<uint32_t>::max()) {
                return ge::GRAPH_FAILED;
            }
            computed.matrixRowsPerTile =
                static_cast<uint32_t>(crossRows);
            computed.matrixRowTiles =
                static_cast<uint32_t>(crossTiles);
            computed.matrixTotalTiles = crossTiles;
        }
        computed.blockDim = static_cast<uint32_t>(
            std::min<uint64_t>(computed.blockDim,
                               computed.matrixTotalTiles));
    }

    ConfigureRowSlab(computed, x1Desc->GetDataType());

    const bool densePipelineReady = denseInputs &&
        computed.totalElements >= GetDensePipelineMinimumElements(
            x1Desc->GetDataType(), aivCoreCount);
    const uint64_t rowSlabKey = computed.rowSlabBufferNum == 2U ? 9U : 11U;
    const bool x1IsPureRowBulk =
        computed.matrixX1Pattern == S9_GREATER_MATRIX_ROW_BROADCAST &&
        computed.matrixX2Pattern == 0U;
    const bool x2IsPureRowBulk =
        computed.matrixX2Pattern == S9_GREATER_MATRIX_ROW_BROADCAST &&
        computed.matrixX1Pattern == 0U;

    const bool boundRowPipelineShape =
        computed.rowStripeEnabled != 0U &&
        (x1IsPureRowBulk || x2IsPureRowBulk) &&
        computed.matrixCols >= 256U &&
        computed.matrixCols <= S9_GREATER_BOX_ELEMENTS_FLOAT32_INT32;
    const bool alignedRowPipelineReady =
        boundRowPipelineShape &&
        computed.matrixOuterCount == 1U &&
        computed.matrixRows >= aivCoreCount;
    const bool crossScalarReady =
        ((computed.matrixX1Pattern ==
              S9_GREATER_MATRIX_ROW_BROADCAST &&
          computed.matrixX2Pattern ==
              S9_GREATER_MATRIX_COL_BROADCAST) ||
         (computed.matrixX1Pattern ==
              S9_GREATER_MATRIX_COL_BROADCAST &&
          computed.matrixX2Pattern ==
              S9_GREATER_MATRIX_ROW_BROADCAST)) &&
        (x1Desc->GetDataType() == ge::DT_FLOAT16 ||
         x1Desc->GetDataType() == ge::DT_FLOAT) &&
        computed.matrixOuterCount == 1U &&
        computed.matrixCols >= 1024U &&
        computed.matrixRows >= computed.blockDim;
    const uint64_t rowPitchRemainder = computed.matrixCols & 255U;

    const bool compactRowPitchReady =
        rowPitchRemainder != 0U && rowPitchRemainder <= 128U;

    const bool contiguousOutputReady = rowPitchRemainder == 0U;
    uint64_t selectedTilingKey = 1U;
    if (computed.boxEnabled != 0U) {
        selectedTilingKey = 3U;
    }
    if (computed.matrixEnabled != 0U) {
        selectedTilingKey = 5U;
    }
    if (computed.rowStripeEnabled != 0U) {
        selectedTilingKey = 7U;
    }
    if (computed.rowSlabEnabled != 0U) {
        selectedTilingKey = rowSlabKey;
    }
    if (alignedRowPipelineReady) {
        selectedTilingKey = 63U;
    }
    if (alignedRowPipelineReady && contiguousOutputReady) {
        selectedTilingKey = 65U;
    }
    if (alignedRowPipelineReady && compactRowPitchReady) {
        selectedTilingKey = 67U;
    }
    if (densePipelineReady) {
        selectedTilingKey = 2U;
    }
    if (crossScalarReady) {
        selectedTilingKey = 69U;
    }

    const bool periodicRowLeft = computed.matrixX1Pattern ==
        S9_GREATER_MATRIX_ROW_BROADCAST && computed.matrixX2Pattern == 0U;
    const bool periodicRowRight = computed.matrixX2Pattern ==
        S9_GREATER_MATRIX_ROW_BROADCAST && computed.matrixX1Pattern == 0U;
    if (selectedTilingKey == 5U && x1Desc->GetDataType() == ge::DT_INT32 &&
        (periodicRowLeft || periodicRowRight) &&
        computed.matrixCols >= 512U && computed.matrixCols <= 1024U) {
        const uint64_t* denseDims = periodicRowLeft ? computed.x2Dims : computed.x1Dims;
        bool denseHasFullPrefix = true;
        for (uint32_t axis = 0U; axis < computed.matrixRowStart; ++axis) {
            if (denseDims[axis] != computed.outputDims[axis]) {
                denseHasFullPrefix = false;
            }
        }
        uint64_t span = 0U;
        if (!MultiplyChecked(computed.matrixRows, computed.matrixCols, span)) {
            return ge::GRAPH_FAILED;
        }
        if (denseHasFullPrefix && span != 0U && (span & 7U) == 0U) {
            uint64_t lhs = span;
            uint64_t rhs = 512U;
            while (rhs != 0U) {
                const uint64_t rem = lhs % rhs;
                lhs = rhs;
                rhs = rem;
            }
            const uint64_t largestHead = 512U - lhs;
            const uint64_t step = 8192U % computed.matrixCols;
            const uint64_t bodyTiles = DivideRoundUp(span, 8192U);

            const bool noPhaseWrap = step == 0U ||
                (largestHead < computed.matrixCols &&
                 bodyTiles - 1U <=
                    (computed.matrixCols - 1U - largestHead) / step);
            const bool alignedWindow = (computed.matrixCols & 7U) == 0U ||
                ((step & 7U) == 0U && noPhaseWrap);
            const uint64_t slotsPerOuter = 1U + DivideRoundUp(span, 8192ULL);
            uint64_t totalSlots = 0U;
            if (!MultiplyChecked(computed.matrixOuterCount, slotsPerOuter, totalSlots)) {
                return ge::GRAPH_FAILED;
            }
            if (alignedWindow && totalSlots >= 2ULL * computed.blockDim) {
                selectedTilingKey = 101U;
            }
        }
    }
    if (selectedTilingKey == 2U && x1Desc->GetDataType() == ge::DT_FLOAT) {
        selectedTilingKey = 117U;
    }
    if (denseInputs && x1Desc->GetDataType() == ge::DT_FLOAT16 &&
        computed.totalElements >= 1U && computed.totalElements <= 4096U) {
        selectedTilingKey = 109U;
        computed.blockDim = 1U;
    }
    if (selectedTilingKey == 101U) {
        const bool left = computed.matrixX1Pattern == S9_GREATER_MATRIX_ROW_BROADCAST;
        const uint64_t* rowDims = left ? computed.x1Dims : computed.x2Dims;
        bool rowHasFullPrefix = true;
        for (uint32_t axis = 0U; axis < computed.matrixRowStart; ++axis) {
            rowHasFullPrefix = rowHasFullPrefix && rowDims[axis] == computed.outputDims[axis];
        }
        if (rowHasFullPrefix) {
            selectedTilingKey = 115U;
        }
    }

    const bool bfRowLeft = computed.matrixX1Pattern == S9_GREATER_MATRIX_ROW_BROADCAST &&
        computed.matrixX2Pattern == 0U;
    const bool bfRowRight = computed.matrixX2Pattern == S9_GREATER_MATRIX_ROW_BROADCAST &&
        computed.matrixX1Pattern == 0U;
    if (selectedTilingKey == 5U && x1Desc->GetDataType() == ge::DT_BF16 &&
        (bfRowLeft || bfRowRight) && computed.matrixCols >= 512U &&
        computed.matrixCols <= 2040U && computed.matrixRows >= 2U) {
        bool fullPrefix = true;
        for (uint32_t axis = 0U; axis < computed.matrixRowStart; ++axis) {
            if (computed.x1Dims[axis] != computed.outputDims[axis] ||
                computed.x2Dims[axis] != computed.outputDims[axis]) fullPrefix = false;
        }
        const uint64_t alignedCols = (computed.matrixCols + 7U) & ~7ULL;
        const uint64_t rowsPerTile = std::min<uint64_t>(computed.matrixRows, 16128U / alignedCols);
        uint64_t newTiles = 0U;
        if (!MultiplyChecked(computed.matrixOuterCount,
                DivideRoundUp(computed.matrixRows, rowsPerTile), newTiles)) return ge::GRAPH_FAILED;
        if (fullPrefix && rowsPerTile >= 2U && newTiles >= 2ULL * computed.blockDim) {
            selectedTilingKey = 121U;
        }
    }
    if (selectedTilingKey == 2U && x1Desc->GetDataType() == ge::DT_INT8) selectedTilingKey = 131U;
    if (selectedTilingKey == 117U &&
        computed.totalElements >= (32ULL * 1024ULL * 1024ULL + 8ULL) / 9ULL) {
        computed.blockDim = std::max<uint32_t>(1U, (4U * computed.blockDim + 4U) / 5U);
    }
    context->SetTilingKey(selectedTilingKey);
    if (selectedTilingKey == 115U || selectedTilingKey == 121U) {
        auto* output = context->GetTilingData<GreaterCompactPeriodicTilingData>();
        if (output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        output->matrixRows = computed.matrixRows;
        output->matrixCols = computed.matrixCols;
        output->matrixOuterCount = computed.matrixOuterCount;
        output->totalElements = computed.totalElements;
        output->matrixX1Pattern = computed.matrixX1Pattern;
        output->reserved = 0U;
    } else if (selectedTilingKey == 109U || selectedTilingKey == 117U || selectedTilingKey == 131U) {
        auto* output = context->GetTilingData<GreaterSmallDenseTilingData>();
        if (output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        output->totalElements = computed.totalElements;
    } else if (selectedTilingKey == 63U || selectedTilingKey == 65U ||
        selectedTilingKey == 67U || selectedTilingKey == 69U) {

        auto* output = context->GetTilingData<GreaterRowSingleTilingData>();
        if (output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        output->matrixRows = computed.matrixRows;
        output->matrixCols = computed.matrixCols;
        output->matrixTotalTiles = computed.matrixTotalTiles;
        output->matrixRowsPerTile = computed.matrixRowsPerTile;
        output->matrixX1Pattern = computed.matrixX1Pattern;
        output->matrixX2Pattern = computed.matrixX2Pattern;
    } else {
        auto* output = context->GetTilingData<GreaterTilingData>();
        if (output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *output = computed;
    }
    context->SetBlockDim(computed.blockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus GreaterInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1 = context->GetInputShape(0);
    const gert::Shape* x2 = context->GetInputShape(1);
    gert::Shape* output = context->GetOutputShape(0);
    if (x1 == nullptr || x2 == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    return InferBroadcastShape(*x1, *x2, *output);
}

static ge::graphStatus GreaterInferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BOOL);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> inputTypes = {
            ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8};
        const std::initializer_list<ge::Format> inputFormats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        const std::initializer_list<ge::DataType> outputTypes = {
            ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL};
        const std::initializer_list<ge::Format> outputFormats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

        this->Input("x1").ParamType(REQUIRED).DataType(inputTypes)
            .Format(inputFormats).UnknownShapeFormat(inputFormats);
        this->Input("x2").ParamType(REQUIRED).DataType(inputTypes)
            .Format(inputFormats).UnknownShapeFormat(inputFormats);
        this->Output("y").ParamType(REQUIRED).DataType(outputTypes)
            .Format(outputFormats).UnknownShapeFormat(outputFormats);
        this->SetInferShape(ge::GreaterInferShape)
            .SetInferDataType(ge::GreaterInferDataType);
        this->AICore().SetTiling(optiling::GreaterTiling).AddConfig("ascend910b");
    }
};

OP_ADD(Greater);
}
