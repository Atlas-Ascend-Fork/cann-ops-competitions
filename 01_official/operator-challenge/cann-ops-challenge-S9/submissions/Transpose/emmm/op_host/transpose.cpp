/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2025 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Liang Yanglin <@liang-yanglin>
 * - Liu Jun <@kbryantttt>
 * - Zhou Jianhua <@LePenseur>
 * - Tu Yuanhang <@TuYHAAAAAA>
 * - Li Xing <@li-xingHIT>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software: you can redistribute it and/or modify it.
 * Licensed under the CANN Open Software License Agreement Version 2.0 (the "License").
 * You may not use this file except in compliance with the License.
 * See the LICENSE file at the root of the repository for the full text of the License.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file transpose.cpp
 * \brief Transpose operator definition.
 */
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;

ge::graphStatus InferShapeTranspose(gert::InferShapeContext* context)
{
    // get input shapes
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    if (xShape == nullptr) {
        return GRAPH_FAILED;
    }

    // get output shapes
    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    if (yShape == nullptr) {
        return GRAPH_FAILED;
    }

    const int64_t rank = static_cast<int64_t>(xShape->GetDimNum());
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr || rank < 1 || rank > 8) {
        return GRAPH_FAILED;
    }
    const gert::TypedContinuousVector<int64_t>* dims = attrs->GetListInt(0);
    if (dims == nullptr || static_cast<int64_t>(dims->GetSize()) != rank || dims->GetData() == nullptr) {
        return GRAPH_FAILED;
    }

    bool seen[8] = {};
    yShape->SetDimNum(static_cast<size_t>(rank));
    const int64_t* values = dims->GetData();
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t normalized = values[i] < 0 ? values[i] + rank : values[i];
        if (normalized < 0 || normalized >= rank || seen[normalized]) {
            return GRAPH_FAILED;
        }
        seen[normalized] = true;
        yShape->SetDim(static_cast<size_t>(i), xShape->GetDim(static_cast<size_t>(normalized)));
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeTranspose(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(IDX_0, context->GetInputDataType(IDX_0));
    return GRAPH_SUCCESS;
}
} // namespace ops

#include <cstdint>
#include <set>

#include "register/op_impl_registry.h"
#include <graph/utils/type_utils.h>
#include "tiling/platform/platform_ascendc.h"
#include "transpose_tiling.h"

namespace optiling {

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_DIM = 8;
constexpr uint32_t WS_SYS_SIZE = 512U;
constexpr uint32_t RESERVED_BYTES = 512U;
constexpr uint64_t VA_LANE_REPLICATE = 0x0001000100010001ULL;
constexpr uint64_t SRC_LANE_OFFSETS = 0x000c000800040000ULL;
constexpr uint64_t DST_LANE_OFFSETS = 0x0018001000080000ULL;
constexpr uint64_t SRC_GROUP_DELTA = 0x0040004000400040ULL;
constexpr uint64_t DST_GROUP_DELTA = 0x0001000100010001ULL;
constexpr uint64_t SRC_CONFIG_DELTA = 0x0010001000100010ULL;
constexpr uint64_t DST_CONFIG_DELTA = 0x0020002000200020ULL;
constexpr uint64_t FAST_OUTPUT_UB_BASE = 16384ULL >> 5U;
struct TransposeCompileInfo {};

static void FillFastVaConfig(TransposeTilingData* tiling)
{
    const uint64_t srcConfigBase = SRC_LANE_OFFSETS;
    const uint64_t dstConfigBase = FAST_OUTPUT_UB_BASE * VA_LANE_REPLICATE + DST_LANE_OFFSETS;
    for (uint32_t rowGroup = 0; rowGroup < 8U; ++rowGroup) {
        const uint64_t srcGroupBase = srcConfigBase + rowGroup * SRC_GROUP_DELTA;
        const uint64_t dstGroupBase = dstConfigBase + rowGroup * DST_GROUP_DELTA;
        for (uint32_t config = 0; config < 4U; ++config) {
            const uint32_t index = rowGroup * 4U + config;
            tiling->srcVaConfig[index] = srcGroupBase + config * SRC_CONFIG_DELTA;
            tiling->dstVaConfig[index] = dstGroupBase + config * DST_CONFIG_DELTA;
        }
    }
}

static bool ParsePermutation(const gert::RuntimeAttrs* attrs, int64_t rank, int64_t* perm)
{
    if (attrs == nullptr || perm == nullptr || rank < 1 || rank > static_cast<int64_t>(MAX_DIM)) {
        return false;
    }
    const gert::TypedContinuousVector<int64_t>* dims = attrs->GetListInt(0);
    if (dims == nullptr || static_cast<int64_t>(dims->GetSize()) != rank || dims->GetData() == nullptr) {
        return false;
    }
    bool seen[MAX_DIM] = {};
    const int64_t* values = dims->GetData();
    for (int64_t i = 0; i < rank; ++i) {
        int64_t normalized = values[i] < 0 ? values[i] + rank : values[i];
        if (normalized < 0 || normalized >= rank || seen[normalized]) {
            return false;
        }
        seen[normalized] = true;
        perm[i] = normalized;
    }
    return true;
}

static bool IsFast2DFp16(
    gert::TilingContext* context, const int64_t* perm, int64_t rank, int64_t coreNum)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape == nullptr || perm == nullptr || rank != 2 ||
        context->GetInputDesc(0)->GetDataType() != ge::DT_FLOAT16) {
        return false;
    }
    const auto& xStorage = xShape->GetStorageShape();
    if (xStorage.GetDimNum() != 2 || xStorage.GetDim(0) <= 0 || xStorage.GetDim(1) <= 0 ||
        perm[0] != 1 || perm[1] != 0 || coreNum <= 0) {
        return false;
    }
    constexpr int64_t macroRows = 128;
    constexpr int64_t macroCols = 64;
    const int64_t height = xStorage.GetDim(0);
    const int64_t width = xStorage.GetDim(1);
    if (height > macroRows || height % macroRows != 0 || width % macroCols != 0) {
        return false;
    }
    const int64_t macroTasks = (height / macroRows) * (width / macroCols);
    return macroTasks <= coreNum;
}

static bool IsTiled2DFp16(gert::TilingContext* context, const int64_t* perm, int64_t rank)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const auto* xDesc = context->GetInputDesc(0);
    if (xShape == nullptr || xDesc == nullptr || perm == nullptr || rank != 2 ||
        xDesc->GetDataType() != ge::DT_FLOAT16 || perm[0] != 1 || perm[1] != 0) {
        return false;
    }
    const auto& xStorage = xShape->GetStorageShape();
    return xStorage.GetDimNum() == 2 && xStorage.GetDim(0) > 0 && xStorage.GetDim(1) > 0;
}

static bool IsTiled2DGeneric(gert::TilingContext* context, const int64_t* perm, int64_t rank)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const auto* xDesc = context->GetInputDesc(0);
    if (xShape == nullptr || xDesc == nullptr || perm == nullptr || rank != 2 || perm[0] != 1 || perm[1] != 0) {
        return false;
    }
    const ge::DataType dtype = xDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_INT32 && dtype != ge::DT_INT8) {
        return false;
    }
    const auto& xStorage = xShape->GetStorageShape();
    return xStorage.GetDimNum() == 2 && xStorage.GetDim(0) > 0 && xStorage.GetDim(1) > 0;
}

// Detect a rotation of two contiguous axis groups after removing fixed outer
// axes.  With a unit-sized fixed suffix this is exactly a batch of ordinary
// dense matrix transposes:
//   [batch, left, right] -> [batch, right, left].
static bool GetBatchedRotationConfig(gert::TilingContext* context, const int64_t* perm, int64_t rank,
                                     uint64_t& batch, uint64_t& height, uint64_t& width, uint64_t& inner)
{
    if (context == nullptr || perm == nullptr || rank <= 2) {
        return false;
    }
    const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return false;
    }
    const auto& shape = xShape->GetStorageShape();
    int64_t prefix = 0;
    while (prefix < rank && perm[prefix] == prefix) {
        ++prefix;
    }
    int64_t end = rank;
    while (end > prefix && perm[end - 1] == end - 1) {
        --end;
    }
    if (end - prefix < 2) {
        return false;
    }

    const int64_t split = perm[prefix];
    if (split <= prefix || split >= end) {
        return false;
    }
    const int64_t middleLength = end - prefix;
    for (int64_t i = prefix; i < end; ++i) {
        int64_t expected = split + (i - prefix);
        if (expected >= end) {
            expected -= middleLength;
        }
        if (perm[i] != expected) {
            return false;
        }
    }

    inner = 1U;
    for (int64_t i = end; i < rank; ++i) {
        inner *= static_cast<uint64_t>(shape.GetDim(i));
    }
    batch = 1U;
    height = 1U;
    width = 1U;
    for (int64_t i = 0; i < prefix; ++i) {
        batch *= static_cast<uint64_t>(shape.GetDim(i));
    }
    for (int64_t i = prefix; i < split; ++i) {
        height *= static_cast<uint64_t>(shape.GetDim(i));
    }
    for (int64_t i = split; i < end; ++i) {
        width *= static_cast<uint64_t>(shape.GetDim(i));
    }
    return height > 1U && width > 1U;
}

// If the output last axis differs from the contiguous input last axis, fixing
// every other coordinate leaves one strided dense matrix:
//   input [..., height, ..., width] -> output [..., width, ..., height]
// where width is the original input last axis and height is the output last
// axis.  The axes need not be adjacent in output: a two-dimensional MTE3 copy
// can skip the fixed intervening coordinates.  This covers arbitrary prefix
// permutations without reconstructing an input address for every element.
static bool GetInputLastAxisMatrixConfig(gert::TilingContext* context, const int64_t* perm, int64_t rank,
                                         uint64_t& batch, uint64_t& height, uint64_t& width)
{
    if (context == nullptr || perm == nullptr || rank <= 2 || perm[rank - 1] == rank - 1) {
        return false;
    }
    const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape == nullptr) {
        return false;
    }
    const auto& shape = xShape->GetStorageShape();
    height = static_cast<uint64_t>(shape.GetDim(perm[rank - 1]));
    width = static_cast<uint64_t>(shape.GetDim(rank - 1));
    const uint64_t matrixElements = height * width;
    batch = matrixElements == 0U ? 0U : static_cast<uint64_t>(shape.GetShapeSize()) / matrixElements;
    return batch > 0U && height > 1U && width > 1U;
}

// 获取平台信息如ubSize, coreNum
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    // 获取ubSize coreNum
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 获取属性，shape信息
static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalIdx, ge::DataType& dataType,
                                         int64_t& rank, int64_t* perm)
{
    auto inputX = context->GetInputShape(0);
    if (inputX == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto& inputShape = inputX->GetStorageShape();
    totalIdx = inputShape.GetShapeSize();
    rank = static_cast<int64_t>(inputShape.GetDimNum());
    if (!ParsePermutation(context->GetAttrs(), rank, perm)) {
        return ge::GRAPH_FAILED;
    }

    // dtype校验
    const std::set<ge::DataType> supportedDtype = {
        ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8};
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dataType = inputDesc->GetDataType();
    if (supportedDtype.count(dataType) == 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = WS_SYS_SIZE + sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

// tiling 分发入口
ge::graphStatus TransposeTilingFunc(gert::TilingContext* context)
{
    // 1. platform
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    ubSize = ubSize - RESERVED_BYTES;
    // 2. shapes & dtype
    int64_t totalIdx = 0;
    ge::DataType dataType;
    int64_t rank = 0;
    int64_t perm[MAX_DIM] = {};
    if (GetShapeAttrsInfo(context, totalIdx, dataType, rank, perm) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // 3. workspace
    if (GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    TransposeTilingData* tiling = context->GetTilingData<TransposeTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *tiling = {};
    context->GetRawTilingData()->SetDataSize(sizeof(TransposeTilingData));
    tiling->dims = rank;
    tiling->totalnumber = totalIdx;
    const auto& inputShape = context->GetInputShape(0)->GetStorageShape();
    for (int64_t i = 0; i < rank; ++i) {
        tiling->shape[i] = inputShape.GetDim(i);
        tiling->perm[i] = perm[i];
    }
    uint32_t typeLength = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
    if (typeLength == 0U) {
        return ge::GRAPH_FAILED;
    }

    if (IsFast2DFp16(context, perm, rank, coreNum)) {
        constexpr uint64_t macroRows = 128U;
        constexpr uint64_t macroCols = 64U;
        const uint64_t height = static_cast<uint64_t>(tiling->shape[0]);
        const uint64_t width = static_cast<uint64_t>(tiling->shape[1]);
        const uint64_t macroTasks = (height / macroRows) * (width / macroCols);
        tiling->fastPath = 1;
        FillFastVaConfig(tiling);
        context->SetBlockDim(static_cast<uint32_t>(macroTasks));
        context->SetTilingKey(1U);
        return ge::GRAPH_SUCCESS;
    }
    if (IsTiled2DFp16(context, perm, rank)) {
        constexpr uint64_t macroRows = 128U;
        constexpr uint64_t macroCols = 64U;
        constexpr uint64_t microRows = 16U;
        constexpr uint64_t microCols = 16U;
        const uint64_t height = static_cast<uint64_t>(tiling->shape[0]);
        const uint64_t width = static_cast<uint64_t>(tiling->shape[1]);
        const uint64_t fullRows = (height / macroRows) * macroRows;
        const uint64_t fullCols = (width / macroCols) * macroCols;
        const uint64_t macroTasks = (fullRows / macroRows) * (fullCols / macroCols);
        const uint64_t allColTiles = (width + microCols - 1U) / microCols;
        const uint64_t bottomTasks = ((height - fullRows + microRows - 1U) / microRows) * allColTiles;
        const uint64_t rightColTiles = (width - fullCols + microCols - 1U) / microCols;
        const uint64_t rightTasks = (fullRows / microRows) * rightColTiles;
        const uint64_t totalTasks = macroTasks + bottomTasks + rightTasks;
        const uint32_t tiledCoreNum = static_cast<uint32_t>(
            totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks : static_cast<uint64_t>(coreNum));

        tiling->fastPath = 2;
        FillFastVaConfig(tiling);
        context->SetBlockDim(tiledCoreNum);
        context->SetTilingKey(2U);
        return ge::GRAPH_SUCCESS;
    }
    if (IsTiled2DGeneric(context, perm, rank)) {
        const uint64_t tileExtent = typeLength == 1U ? 32U : 16U;
        const uint64_t height = static_cast<uint64_t>(tiling->shape[0]);
        const uint64_t width = static_cast<uint64_t>(tiling->shape[1]);
        const uint64_t totalTasks = ((height + tileExtent - 1U) / tileExtent) *
                                    ((width + tileExtent - 1U) / tileExtent);
        const uint32_t tiledCoreNum = static_cast<uint32_t>(
            totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks : static_cast<uint64_t>(coreNum));
        tiling->fastPath = 3;
        context->SetBlockDim(tiledCoreNum);
        context->SetTilingKey(3U);
        return ge::GRAPH_SUCCESS;
    }

    uint64_t rotationBatch = 0U;
    uint64_t rotationHeight = 0U;
    uint64_t rotationWidth = 0U;
    uint64_t rotationInner = 0U;
    if (totalIdx >= 4096 && GetBatchedRotationConfig(
            context, perm, rank, rotationBatch, rotationHeight, rotationWidth, rotationInner)) {
        // With no fixed suffix, every non-trivial rotation necessarily moves
        // the physical input-last axis.  If that axis is output-adjacent to
        // the last axis, let the input-last-axis recognizer below choose a
        // 4/16/64-block-wide tile instead of committing here to one native
        // tile per task.  A non-adjacent input-last axis needs general output
        // address reconstruction; retain the dense rotation tile there, where
        // it treats both contiguous groups as one matrix.
        bool preferInputLastWideTile = false;
        if (rotationInner == 1U) {
            const auto& rotationShape = context->GetInputShape(0)->GetStorageShape();
            const uint64_t matrixHeight = static_cast<uint64_t>(rotationShape.GetDim(perm[rank - 1]));
            const uint64_t matrixWidth = static_cast<uint64_t>(rotationShape.GetDim(rank - 1));
            const uint64_t matrixTileRows = typeLength == 1U ? 32U : 16U;
            if (perm[rank - 2] == rank - 1) {
                const uint64_t matrixTileCols = matrixTileRows * 4U;
                const uint64_t matrixTasks =
                    ((matrixHeight + matrixTileRows - 1U) / matrixTileRows) *
                    ((matrixWidth + matrixTileCols - 1U) / matrixTileCols);
                const uint64_t matrixMinimumElements =
                    (matrixTileRows * matrixTileCols) / 4U;
                preferInputLastWideTile = matrixHeight * matrixWidth >=
                                          matrixTasks * matrixMinimumElements;
            } else if (typeLength == 1U && matrixWidth <= matrixTileRows * 8U) {
                // For a general output stride, slicing the rotation into
                // compact input-last-axis matrices replaces many native
                // 32x32 descriptors with one 32x256 descriptor per slice.
                // Select it only when descriptor count falls by at least 4x,
                // all AIVs remain useful, and padded Vector work grows by no
                // more than 25 percent.
                const uint64_t compactTileCols = matrixTileRows * 8U;
                const uint64_t slicedBatch =
                    static_cast<uint64_t>(totalIdx) /
                    (matrixHeight * matrixWidth);
                const uint64_t slicedTasksPerBatch =
                    ((matrixHeight + matrixTileRows - 1U) / matrixTileRows) *
                    ((matrixWidth + compactTileCols - 1U) / compactTileCols);
                const uint64_t slicedTasks = slicedBatch * slicedTasksPerBatch;
                const uint64_t rotationTasks = rotationBatch *
                    ((rotationHeight + matrixTileRows - 1U) / matrixTileRows) *
                    ((rotationWidth + matrixTileRows - 1U) / matrixTileRows);
                const uint64_t slicedPaddedElements = slicedTasks *
                    matrixTileRows * compactTileCols;
                const uint64_t rotationPaddedElements = rotationTasks *
                    matrixTileRows * matrixTileRows;
                preferInputLastWideTile =
                    slicedTasks >= static_cast<uint64_t>(coreNum) &&
                    rotationTasks >= slicedTasks * 4U &&
                    slicedPaddedElements * 4U <=
                        rotationPaddedElements * 5U;
            }
        }
        // INT8 matrices only two or three columns wide fall below the native
        // 32x32 density gate.  Gathering one byte at a time through key 5
        // destroys locality; key 7 can instead load 16 complete short rows
        // and deinterleave their columns with one B8 VNCHW operation.
        if (rotationInner == 1U && typeLength == 1U && rotationWidth <= 3U) {
            constexpr uint64_t skinnyTileRows = 256U;
            const uint64_t tasksPerBatch =
                (rotationHeight + skinnyTileRows - 1U) / skinnyTileRows;
            const uint64_t totalTasks = rotationBatch * tasksPerBatch;
            const uint32_t tiledCoreNum = static_cast<uint32_t>(
                totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks : static_cast<uint64_t>(coreNum));
            tiling->fastPath = 7;
            context->SetBlockDim(tiledCoreNum);
            context->SetTilingKey(7U);
            return ge::GRAPH_SUCCESS;
        }
        if (rotationInner == 1U && !preferInputLastWideTile) {
            const uint64_t tileExtent = typeLength == 1U ? 32U : 16U;
            const uint64_t tasksPerBatch = ((rotationHeight + tileExtent - 1U) / tileExtent) *
                                           ((rotationWidth + tileExtent - 1U) / tileExtent);
            const uint64_t minimumElementsPerTask = (tileExtent * tileExtent) / 4U;
            if (rotationHeight * rotationWidth >= tasksPerBatch * minimumElementsPerTask) {
                const uint64_t totalTasks = rotationBatch * tasksPerBatch;
                const uint32_t tiledCoreNum = static_cast<uint32_t>(
                    totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks : static_cast<uint64_t>(coreNum));
                tiling->fastPath = 6;
                context->SetBlockDim(tiledCoreNum);
                context->SetTilingKey(6U);
                return ge::GRAPH_SUCCESS;
            }
        }

        // A small fixed suffix turns each matrix element into a short record.
        // Keep large records on the contiguous-suffix DMA path below; for
        // sub-block records, transpose 16x16 record tiles entirely in UB.
        constexpr uint64_t recordTileExtent = 16U;
        const uint64_t recordTasksPerBatch =
            ((rotationHeight + recordTileExtent - 1U) / recordTileExtent) *
            ((rotationWidth + recordTileExtent - 1U) / recordTileExtent);
        constexpr uint64_t minimumRecordsPerTask = (recordTileExtent * recordTileExtent) / 4U;
        if (rotationInner > 1U && rotationInner * typeLength < BLOCK_SIZE &&
            rotationHeight * rotationWidth >= recordTasksPerBatch * minimumRecordsPerTask) {
            const uint64_t totalTasks = rotationBatch * recordTasksPerBatch;
            const uint32_t tiledCoreNum = static_cast<uint32_t>(
                totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks : static_cast<uint64_t>(coreNum));
            tiling->fastPath = 7;
            context->SetBlockDim(tiledCoreNum);
            context->SetTilingKey(7U);
            return ge::GRAPH_SUCCESS;
        }
    }

    uint64_t lastTwoBatch = 0U;
    uint64_t lastTwoHeight = 0U;
    uint64_t lastTwoWidth = 0U;
    if (totalIdx >= 4096 &&
        GetInputLastAxisMatrixConfig(context, perm, rank, lastTwoBatch, lastTwoHeight, lastTwoWidth)) {
        const uint64_t tileRows = typeLength == 1U ? 32U : 16U;
        const uint64_t hugeTileCols = tileRows * 64U;
        const uint64_t hugeTasksPerBatch = ((lastTwoHeight + tileRows - 1U) / tileRows) *
                                           ((lastTwoWidth + hugeTileCols - 1U) / hugeTileCols);
        const uint64_t hugeTotalTasks = lastTwoBatch * hugeTasksPerBatch;
        const uint64_t hugeMinimumElementsPerTask = (tileRows * hugeTileCols) / 4U;
        const uint64_t hugePaddedElements = hugeTasksPerBatch * tileRows * hugeTileCols;
        const bool hugeTileDense = lastTwoHeight * lastTwoWidth * 4U >=
                                   hugePaddedElements * 3U;
        if (lastTwoHeight * lastTwoWidth >= hugeTasksPerBatch * hugeMinimumElementsPerTask &&
            hugeTileDense &&
            hugeTotalTasks >= static_cast<uint64_t>(coreNum)) {
            const uint32_t matrixKey = perm[rank - 2] == rank - 1 ? 10U : 13U;
            tiling->fastPath = matrixKey;
            context->SetBlockDim(static_cast<uint32_t>(coreNum));
            context->SetTilingKey(matrixKey);
            return ge::GRAPH_SUCCESS;
        }

        const bool contiguousMatrix = perm[rank - 2] == rank - 1;
        if (contiguousMatrix && typeLength == 4U && lastTwoBatch >= 64U &&
            totalIdx >= 16U * 1024U * 1024U) {
            // Wider input bursts win while the 128-column geometry stays compact.
            // Keep only well-filled tails and aligned matrix batches so the larger
            // MTE2 bursts do not amplify sparse or phase-rotating accesses.
            constexpr uint64_t balancedTileRows = 16U * 8U;
            constexpr uint64_t balancedTileCols = 16U * 8U;
            const uint64_t balancedRowTiles =
                (lastTwoHeight + balancedTileRows - 1U) / balancedTileRows;
            const uint64_t balancedColTiles =
                (lastTwoWidth + balancedTileCols - 1U) / balancedTileCols;
            const uint64_t balancedTasksPerBatch = balancedRowTiles * balancedColTiles;
            const uint64_t balancedTotalTasks = lastTwoBatch * balancedTasksPerBatch;
            const uint64_t balancedPaddedElements =
                balancedTasksPerBatch * balancedTileRows * balancedTileCols;
            const uint64_t balancedTailCols = lastTwoWidth % balancedTileCols;
            const bool balancedTailDense =
                balancedTailCols == 0U || balancedTailCols >= balancedTileCols / 2U;
            const bool balancedBatchAligned =
                (lastTwoHeight * lastTwoWidth * typeLength) % BLOCK_SIZE == 0U;
            const bool balancedInputRowsAligned =
                (lastTwoWidth * typeLength) % BLOCK_SIZE == 0U;
            const bool balancedInputSetupAmortized =
                balancedInputRowsAligned || balancedColTiles >= 5U;
            const bool balancedDense =
                lastTwoHeight * lastTwoWidth * 4U >= balancedPaddedElements * 3U;
            if (balancedColTiles <= 6U && balancedTailDense && balancedBatchAligned &&
                balancedInputSetupAmortized && balancedDense &&
                balancedTotalTasks >= static_cast<uint64_t>(coreNum)) {
                tiling->fastPath = 18U;
                context->SetBlockDim(static_cast<uint32_t>(coreNum));
                context->SetTilingKey(18U);
                return ge::GRAPH_SUCCESS;
            }

            constexpr uint64_t longStoreTileRows = 16U * 12U;
            constexpr uint64_t longStoreTileCols = 16U * 4U;
            constexpr uint64_t sharedOutputTileRows = 16U * 16U;
            constexpr uint64_t sharedOutputTileCols = 16U * 4U;
            constexpr uint64_t singleBufferTileRows = 16U * 24U;
            constexpr uint64_t singleBufferTileCols = 16U * 4U;
            constexpr uint64_t wideInputTileRows = 16U * 24U;
            constexpr uint64_t wideInputTileCols = 16U * 6U;
            constexpr uint64_t wideTailAlignmentCols = 16U * 3U;
            constexpr uint64_t inplaceWideTileRows = 16U * 24U;
            constexpr uint64_t inplaceWideTileCols = 16U * 7U;
            constexpr uint64_t inplaceWideTailAlignmentCols = 16U * 5U;
            const uint64_t singleBufferTasksPerBatch =
                ((lastTwoHeight + singleBufferTileRows - 1U) / singleBufferTileRows) *
                ((lastTwoWidth + singleBufferTileCols - 1U) / singleBufferTileCols);
            const uint64_t singleBufferTotalTasks =
                lastTwoBatch * singleBufferTasksPerBatch;
            const uint64_t singleBufferPaddedElements =
                singleBufferTasksPerBatch * singleBufferTileRows * singleBufferTileCols;
            const bool singleBufferDense =
                lastTwoHeight * lastTwoWidth * 4U >= singleBufferPaddedElements * 3U;
            const uint64_t wideInputRowTiles =
                (lastTwoHeight + wideInputTileRows - 1U) /
                wideInputTileRows;
            const uint64_t wideInputColTiles =
                (lastTwoWidth + wideInputTileCols - 1U) /
                wideInputTileCols;
            const uint64_t wideInputTasksPerBatch =
                wideInputRowTiles * wideInputColTiles;
            const uint64_t wideInputTotalTasks =
                lastTwoBatch * wideInputTasksPerBatch;
            const uint64_t wideInputPaddedElements =
                wideInputTasksPerBatch * wideInputTileRows *
                wideInputTileCols;
            const uint64_t wideInputTailRows =
                lastTwoHeight % wideInputTileRows;
            const uint64_t wideInputTailCols =
                lastTwoWidth % wideInputTileCols;
            const bool wideInputRowTailDense =
                wideInputTailRows == 0U ||
                wideInputTailRows * 4U >= wideInputTileRows * 3U;
            const uint64_t dmaBlockElements = 32U / typeLength;
            const uint64_t alignedWideInputTailCols =
                ((wideInputTailCols + dmaBlockElements - 1U) /
                 dmaBlockElements) * dmaBlockElements;
            const bool wideInputTailUsesPartialAlignmentBand =
                wideInputTailCols != 0U &&
                alignedWideInputTailCols == wideTailAlignmentCols &&
                wideInputTailCols < alignedWideInputTailCols;
            const bool wideInputDense =
                lastTwoHeight * lastTwoWidth * 4U >=
                wideInputPaddedElements * 3U;
            const uint64_t singleBufferInputRowSegments =
                lastTwoHeight *
                ((lastTwoWidth + singleBufferTileCols - 1U) /
                 singleBufferTileCols);
            const uint64_t wideInputRowSegments =
                lastTwoHeight * wideInputColTiles;
            const bool wideInputDescriptorReduction =
                wideInputRowSegments * 5U <=
                singleBufferInputRowSegments * 4U;
            constexpr uint64_t wideInputMinTasksPerCore = 8U;
            const bool wideInputSteady =
                wideInputTotalTasks >=
                static_cast<uint64_t>(coreNum) *
                    wideInputMinTasksPerCore;
            const uint64_t inplaceWideRowTiles =
                (lastTwoHeight + inplaceWideTileRows - 1U) /
                inplaceWideTileRows;
            const uint64_t inplaceWideColTiles =
                (lastTwoWidth + inplaceWideTileCols - 1U) /
                inplaceWideTileCols;
            const uint64_t inplaceWideTasksPerBatch =
                inplaceWideRowTiles * inplaceWideColTiles;
            const uint64_t inplaceWideTotalTasks =
                lastTwoBatch * inplaceWideTasksPerBatch;
            const uint64_t inplaceWidePaddedElements =
                inplaceWideTasksPerBatch * inplaceWideTileRows *
                inplaceWideTileCols;
            const uint64_t inplaceWideTailRows =
                lastTwoHeight % inplaceWideTileRows;
            const uint64_t inplaceWideTailCols =
                lastTwoWidth % inplaceWideTileCols;
            const bool inplaceWideRowTailSafe =
                inplaceWideTailRows == 0U ||
                inplaceWideTailRows > 3U * inplaceWideTileCols;
            const uint64_t alignedInplaceWideTailCols =
                ((inplaceWideTailCols + dmaBlockElements - 1U) /
                 dmaBlockElements) * dmaBlockElements;
            const bool inplaceWideTailUsesPartialAlignmentBand =
                inplaceWideTailCols != 0U &&
                alignedInplaceWideTailCols ==
                    inplaceWideTailAlignmentCols &&
                inplaceWideTailCols < alignedInplaceWideTailCols;
            const bool inplaceWideDense =
                lastTwoHeight * lastTwoWidth * 4U >=
                inplaceWidePaddedElements * 3U;
            const uint64_t inplaceWideInputRowSegments =
                lastTwoHeight * inplaceWideColTiles;
            const bool inplaceWideDescriptorReduction =
                inplaceWideInputRowSegments * 5U <=
                singleBufferInputRowSegments * 4U;
            constexpr uint64_t inplaceWideMinTasksPerCore = 8U;
            const bool inplaceWideSteady =
                inplaceWideTotalTasks >=
                static_cast<uint64_t>(coreNum) *
                    inplaceWideMinTasksPerCore;
            if (singleBufferDense &&
                singleBufferTotalTasks >= static_cast<uint64_t>(coreNum)) {
                // Split the 96 KiB input tile into two independently guarded
                // row generations only when every row tile has a non-empty
                // second half and enough steady-state tasks amortize fill and
                // drain.  This is a UB-lifetime and scheduler-density rule.
                const uint64_t singleBufferTailRows =
                    lastTwoHeight % singleBufferTileRows;
                const bool splitInputTailSafe =
                    singleBufferTailRows == 0U ||
                    singleBufferTailRows > singleBufferTileRows / 2U;
                constexpr uint64_t splitInputMinTasksPerCore = 8U;
                const bool splitInputSteady =
                    singleBufferTotalTasks >=
                        static_cast<uint64_t>(coreNum) *
                            splitInputMinTasksPerCore;
                // The 96-column accumulated-output schedule is profitable when the
                // final source row occupies a partially padded 48-column DMA
                // alignment band.  Exact 40- and 48-column block boundaries
                // have a different measured MTE service regime and stay on
                // the established 64-column path.
                const bool wideInputRange =
                    splitInputTailSafe && splitInputSteady &&
                    wideInputRowTailDense &&
                    wideInputTailUsesPartialAlignmentBand &&
                    wideInputDense && wideInputDescriptorReduction &&
                    wideInputSteady;
                // Merging the two row tiles is profitable when the second
                // 384-row tile is a dense, non-empty tail.  A complete
                // 768-row generation has a measured long-store resonance,
                // while additional row-tile pairs do not reduce the task
                // envelope.  Keep those topologies on the established key 27.
                const bool adaptiveTallTailPair =
                    wideInputRange && wideInputRowTiles == 2U &&
                    wideInputTailRows != 0U;
                const bool inplaceWideRange =
                    inplaceWideRowTailSafe &&
                    inplaceWideTailUsesPartialAlignmentBand &&
                    inplaceWideDense && inplaceWideDescriptorReduction &&
                    inplaceWideSteady;
                const uint32_t singleBufferKey =
                    wideInputRange ?
                    (adaptiveTallTailPair ? 33U : 27U) :
                    (splitInputTailSafe && splitInputSteady ? 25U : 22U);
                tiling->fastPath = singleBufferKey;
                uint32_t launchAivCount = static_cast<uint32_t>(coreNum);
                if (singleBufferKey == 33U && lastTwoWidth <= 768U) {
                    // Keep the original longest-lane macro count, but avoid
                    // launching extra AIVs that only add short final waves.
                    // Both quantities come from generic shape topology and
                    // the platform core count, not any fixed measurement case.
                    const uint64_t macroCount = lastTwoBatch * wideInputRowTiles;
                    const uint64_t maxAivs = static_cast<uint64_t>(coreNum);
                    const uint64_t macrosPerAiv =
                        (macroCount + maxAivs - 1U) / maxAivs;
                    launchAivCount = static_cast<uint32_t>(
                        (macroCount + macrosPerAiv - 1U) / macrosPerAiv);
                    // Two AIVs may share 768 staging rows, within the same
                    // 384-row-per-AIV budget. An odd extra AIV needs no slot.
                    // Preserve the library-owned workspace prefix.
                    const uint64_t paddedWidth =
                        ((lastTwoWidth + 15U) / 16U) * 16U;
                    size_t* workspace = context->GetWorkspaceSizes(1);
                    workspace[0] += static_cast<uint64_t>(launchAivCount) *
                        384U * paddedWidth * sizeof(uint32_t);
                }
                context->SetBlockDim(launchAivCount);
                context->SetTilingKey(singleBufferKey);
                return ge::GRAPH_SUCCESS;
            }

            const uint64_t sharedOutputTasksPerBatch =
                ((lastTwoHeight + sharedOutputTileRows - 1U) / sharedOutputTileRows) *
                ((lastTwoWidth + sharedOutputTileCols - 1U) / sharedOutputTileCols);
            const uint64_t sharedOutputTotalTasks =
                lastTwoBatch * sharedOutputTasksPerBatch;
            const uint64_t sharedOutputPaddedElements =
                sharedOutputTasksPerBatch * sharedOutputTileRows * sharedOutputTileCols;
            const bool sharedOutputDense =
                lastTwoHeight * lastTwoWidth * 4U >= sharedOutputPaddedElements * 3U;
            if (sharedOutputDense &&
                sharedOutputTotalTasks >= static_cast<uint64_t>(coreNum)) {
                tiling->fastPath = 21U;
                context->SetBlockDim(static_cast<uint32_t>(coreNum));
                context->SetTilingKey(21U);
                return ge::GRAPH_SUCCESS;
            }

            const uint64_t longStoreTasksPerBatch =
                ((lastTwoHeight + longStoreTileRows - 1U) / longStoreTileRows) *
                ((lastTwoWidth + longStoreTileCols - 1U) / longStoreTileCols);
            const uint64_t longStoreTotalTasks = lastTwoBatch * longStoreTasksPerBatch;
            const uint64_t longStorePaddedElements =
                longStoreTasksPerBatch * longStoreTileRows * longStoreTileCols;
            const bool longStoreDense =
                lastTwoHeight * lastTwoWidth * 4U >= longStorePaddedElements * 3U;
            if (longStoreDense && longStoreTotalTasks >= static_cast<uint64_t>(coreNum)) {
                tiling->fastPath = 17U;
                context->SetBlockDim(static_cast<uint32_t>(coreNum));
                context->SetTilingKey(17U);
                return ge::GRAPH_SUCCESS;
            }
        }

        const uint64_t wideTileRows = contiguousMatrix ?
            tileRows * (typeLength == 4U ? 3U : 2U) : tileRows;
        const uint64_t wideTileCols = contiguousMatrix ?
            tileRows * (typeLength == 4U ? 15U : 16U) : tileRows * 16U;
        const uint64_t wideTasksPerBatch = ((lastTwoHeight + wideTileRows - 1U) / wideTileRows) *
                                           ((lastTwoWidth + wideTileCols - 1U) / wideTileCols);
        const uint64_t wideTotalTasks = lastTwoBatch * wideTasksPerBatch;
        const uint64_t wideMinimumElementsPerTask = (wideTileRows * wideTileCols) / 4U;
        if (lastTwoHeight * lastTwoWidth >= wideTasksPerBatch * wideMinimumElementsPerTask &&
            wideTotalTasks >= static_cast<uint64_t>(coreNum)) {
            // A general-stride INT8 matrix that fits within eight native
            // column blocks needs the same task count and moves the same GM
            // bytes with a 32x256 tile.  Avoid the otherwise unused upper half
            // of the 32x512 UB tile and its eight extra B8 VNCHW groups.
            const bool compactInt8GeneralTile =
                !contiguousMatrix && typeLength == 1U &&
                lastTwoWidth <= tileRows * 8U;
            const uint32_t matrixKey = contiguousMatrix ? 9U :
                (compactInt8GeneralTile ? 26U : 12U);
            tiling->fastPath = matrixKey;
            context->SetBlockDim(static_cast<uint32_t>(coreNum));
            context->SetTilingKey(matrixKey);
            return ge::GRAPH_SUCCESS;
        }

        const bool generalOutputStride = perm[rank - 2] != rank - 1;
        const uint64_t tileBlocks = generalOutputStride ? 2U : 4U;
        const uint64_t tileCols = tileRows * tileBlocks;
        if (generalOutputStride && typeLength == 4U && lastTwoBatch >= 1024U &&
            lastTwoHeight > tileRows) {
            const uint64_t mergedTileRows = tileRows * 2U;
            const uint64_t mergedTasksPerBatch =
                ((lastTwoHeight + mergedTileRows - 1U) / mergedTileRows) *
                ((lastTwoWidth + tileCols - 1U) / tileCols);
            const uint64_t mergedMinimumElementsPerTask =
                (mergedTileRows * tileCols) / 2U;
            if (lastTwoHeight * lastTwoWidth >=
                mergedTasksPerBatch * mergedMinimumElementsPerTask) {
                const uint64_t mergedTotalTasks = lastTwoBatch * mergedTasksPerBatch;
                const uint32_t mergedCoreNum = static_cast<uint32_t>(
                    mergedTotalTasks < static_cast<uint64_t>(coreNum) ? mergedTotalTasks :
                                                                       static_cast<uint64_t>(coreNum));
                // When the fastest remaining input-batch axis is physically
                // adjacent to each matrix row but lies outside a non-trivial
                // output-batch suffix, two matrices can share one wider 2-D
                // load.  Bound the combined tile by the native 32x64 B32 UB
                // geometry; this is a layout/capacity gate, not a shape match.
                const auto& matrixShape =
                    context->GetInputShape(0)->GetStorageShape();
                const int64_t rowAxis = perm[rank - 1];
                int64_t inputFastBatchAxis = -1;
                for (int64_t axis = rank - 2; axis >= 0; --axis) {
                    if (axis != rowAxis && matrixShape.GetDim(axis) > 1) {
                        inputFastBatchAxis = axis;
                        break;
                    }
                }
                uint64_t inputFastStride = 1U;
                if (inputFastBatchAxis >= 0) {
                    for (int64_t axis = inputFastBatchAxis + 1; axis < rank; ++axis) {
                        inputFastStride *=
                            static_cast<uint64_t>(matrixShape.GetDim(axis));
                    }
                }
                int64_t inputFastOutputPos = -1;
                for (int64_t i = 0; i < rank; ++i) {
                    if (perm[i] == inputFastBatchAxis) {
                        inputFastOutputPos = i;
                        break;
                    }
                }
                uint64_t innerBatchExtent = 1U;
                if (inputFastOutputPos >= 0) {
                    for (int64_t i = inputFastOutputPos + 1; i < rank; ++i) {
                        if (perm[i] != rowAxis && perm[i] != rank - 1) {
                            innerBatchExtent *= static_cast<uint64_t>(
                                matrixShape.GetDim(perm[i]));
                        }
                    }
                }
                const uint64_t inputFastExtent = inputFastBatchAxis < 0 ? 0U :
                    static_cast<uint64_t>(matrixShape.GetDim(inputFastBatchAxis));
                const bool adjacentPairRange =
                    inputFastBatchAxis >= 0 && inputFastOutputPos >= 0 &&
                    inputFastStride == lastTwoWidth && inputFastExtent >= 2U &&
                    innerBatchExtent >= 4U && lastTwoHeight <= tileRows * 2U &&
                    lastTwoWidth * 2U <= tileRows * 4U;
                if (adjacentPairRange) {
                    int64_t outputFastBatchAxis = -1;
                    int64_t outputFastBatchPos = -1;
                    for (int64_t i = rank - 1; i >= 0; --i) {
                        if (perm[i] != rowAxis && perm[i] != rank - 1 &&
                            matrixShape.GetDim(perm[i]) > 1) {
                            outputFastBatchAxis = perm[i];
                            outputFastBatchPos = i;
                            break;
                        }
                    }
                    uint64_t outputFastStride = 1U;
                    if (outputFastBatchPos >= 0) {
                        for (int64_t i = outputFastBatchPos + 1; i < rank; ++i) {
                            outputFastStride *= static_cast<uint64_t>(
                                matrixShape.GetDim(perm[i]));
                        }
                    }
                    const uint64_t outputFastExtent =
                        outputFastBatchAxis < 0 ? 0U : static_cast<uint64_t>(
                            matrixShape.GetDim(outputFastBatchAxis));
                    uint64_t rectangleTotalGroups = 0U;
                    uint64_t quadRectangleTotalGroups = 0U;
                    uint64_t tripleQuadRectangleTotalGroups = 0U;
                    uint64_t fourByFourRectangleTotalGroups = 0U;
                    uint64_t sixByThreeRectangleTotalGroups = 0U;
                    uint64_t sixBySixRectangleTotalGroups = 0U;
                    uint64_t sevenBySixRectangleTotalGroups = 0U;
                    if (outputFastExtent > 0U && inputFastExtent > 0U) {
                        const uint64_t rectangleMiddleExtent =
                            innerBatchExtent / outputFastExtent;
                        const uint64_t rectangleOuterExtent =
                            lastTwoBatch / (inputFastExtent * innerBatchExtent);
                        rectangleTotalGroups = rectangleOuterExtent *
                            ((inputFastExtent + 1U) / 2U) *
                            rectangleMiddleExtent *
                            ((outputFastExtent + 1U) / 2U);
                        quadRectangleTotalGroups = rectangleOuterExtent *
                            ((inputFastExtent + 3U) / 4U) *
                            rectangleMiddleExtent *
                            ((outputFastExtent + 1U) / 2U);
                        tripleQuadRectangleTotalGroups = rectangleOuterExtent *
                            ((inputFastExtent + 2U) / 3U) *
                            rectangleMiddleExtent *
                            ((outputFastExtent + 3U) / 4U);
                        fourByFourRectangleTotalGroups = rectangleOuterExtent *
                            ((inputFastExtent + 3U) / 4U) *
                            rectangleMiddleExtent *
                            ((outputFastExtent + 3U) / 4U);
                        sixByThreeRectangleTotalGroups = rectangleOuterExtent *
                            ((inputFastExtent + 5U) / 6U) *
                            rectangleMiddleExtent *
                            ((outputFastExtent + 2U) / 3U);
                        sixBySixRectangleTotalGroups = rectangleOuterExtent *
                            ((inputFastExtent + 5U) / 6U) *
                            rectangleMiddleExtent *
                            ((outputFastExtent + 5U) / 6U);
                        sevenBySixRectangleTotalGroups = rectangleOuterExtent *
                            ((inputFastExtent + 6U) / 7U) *
                            rectangleMiddleExtent *
                            ((outputFastExtent + 5U) / 6U);
                    }
                    // One offset table is built per active core.  Too few
                    // rectangles cannot amortize it, while an overlong range
                    // makes the extra Gather pipeline dominate.  Bound work
                    // per core using scheduler density rather than any axis
                    // or complete-shape fingerprint.
                    constexpr uint64_t rectangleMinGroupsPerCore = 60U;
                    constexpr uint64_t rectangleMaxGroupsPerCore = 96U;
                    const bool rectangleCoreDense =
                        rectangleTotalGroups >= static_cast<uint64_t>(coreNum) *
                                                    rectangleMinGroupsPerCore &&
                        rectangleTotalGroups <= static_cast<uint64_t>(coreNum) *
                                                    rectangleMaxGroupsPerCore;
                    const bool adjacentRectangleTopology =
                        outputFastBatchAxis >= 0 &&
                        outputFastBatchAxis != inputFastBatchAxis &&
                        outputFastBatchPos > inputFastOutputPos &&
                        outputFastStride == lastTwoHeight &&
                        outputFastExtent >= 2U &&
                        lastTwoBatch >= 8192U &&
                        lastTwoHeight * 2U <= tileRows * 4U;
                    // Seven full input coordinates reduce the rectangle count
                    // by one seventh.  A 176-element source row and two
                    // Coordinate-aligned 50- and 78-row output banks fit in
                    // 192 KiB; key 34 combines them into one 128-row VNCHW
                    // range to
                    // amortize VA programming while retaining independent
                    // stores.  Divisibility makes every selected rectangle
                    // use the same full 7-by-6 topology.
                    constexpr uint64_t sevenBySixMinGroupsPerCore = 5U;
                    constexpr uint64_t sevenBySixMaxGroupsPerCore = 12U;
                    const uint64_t sevenBySixInputRowCapacity =
                        tileRows * 11U;
                    constexpr uint64_t sevenBySixOutputRowCapacity = 152U;
                    const bool sevenBySixCoreDense =
                        sevenBySixRectangleTotalGroups >=
                            static_cast<uint64_t>(coreNum) *
                                sevenBySixMinGroupsPerCore &&
                        sevenBySixRectangleTotalGroups <=
                            static_cast<uint64_t>(coreNum) *
                                sevenBySixMaxGroupsPerCore;
                    const bool sevenBySixRectangleRange =
                        adjacentRectangleTopology &&
                        inputFastExtent >= 7U &&
                        inputFastExtent % 7U == 0U &&
                        outputFastExtent >= 6U &&
                        outputFastExtent % 6U == 0U &&
                        lastTwoWidth * 7U <=
                            sevenBySixInputRowCapacity &&
                        lastTwoWidth * 7U * 4U >=
                            sevenBySixInputRowCapacity * 3U &&
                        lastTwoHeight * 6U <=
                            sevenBySixOutputRowCapacity &&
                        lastTwoHeight * 6U * 4U >=
                            sevenBySixOutputRowCapacity * 3U &&
                        sevenBySixCoreDense;
                    if (sevenBySixRectangleRange) {
                        tiling->fastPath = 34U;
                        context->SetBlockDim(mergedCoreNum);
                        context->SetTilingKey(34U);
                        return ge::GRAPH_SUCCESS;
                    }
                    // Six output coordinates make each output row twice as
                    // long as the retained 6x3 route.  A streamed layout
                    // keeps only one output coordinate in each input bank,
                    // so the complete six-coordinate intermediate and its
                    // bounded Gather table still fit in 192 KiB.
                    constexpr uint64_t sixBySixMinGroupsPerCore = 5U;
                    constexpr uint64_t sixBySixMaxGroupsPerCore = 12U;
                    constexpr uint64_t sixBySixPackedRowBlocks = 19U;
                    const uint64_t sixBySixPackedRowCapacity =
                        sixBySixPackedRowBlocks * (32U / typeLength);
                    const uint64_t sixBySixInputTail = inputFastExtent % 6U;
                    const bool sixBySixCoreDense =
                        sixBySixRectangleTotalGroups >=
                            static_cast<uint64_t>(coreNum) *
                                sixBySixMinGroupsPerCore &&
                        sixBySixRectangleTotalGroups <=
                            static_cast<uint64_t>(coreNum) *
                                sixBySixMaxGroupsPerCore;
                    const bool sixBySixRectangleRange =
                        adjacentRectangleTopology &&
                        inputFastExtent >= 6U &&
                        outputFastExtent >= 6U &&
                        outputFastExtent % 6U == 0U &&
                        (sixBySixInputTail == 0U ||
                         sixBySixInputTail >= 4U) &&
                        lastTwoWidth * 6U <= tileRows * 10U &&
                        lastTwoHeight * 6U <= sixBySixPackedRowCapacity &&
                        sixBySixCoreDense;
                    if (sixBySixRectangleRange) {
                        tiling->fastPath = 32U;
                        context->SetBlockDim(mergedCoreNum);
                        context->SetTilingKey(32U);
                        return ge::GRAPH_SUCCESS;
                    }
                    // A 6x3 rectangle trades one third more output
                    // descriptors for one third fewer input descriptors and
                    // fewer total Vector rectangles.  The shared output and
                    // consumed input bank fit with a bounded Gather table.
                    constexpr uint64_t sixByThreeMinGroupsPerCore = 10U;
                    constexpr uint64_t sixByThreeMaxGroupsPerCore = 22U;
                    constexpr uint64_t sixByThreePackedRowBlocks = 10U;
                    const uint64_t sixByThreePackedRowCapacity =
                        sixByThreePackedRowBlocks * (32U / typeLength);
                    const uint64_t sixByThreeInputTail = inputFastExtent % 6U;
                    const bool sixByThreeCoreDense =
                        sixByThreeRectangleTotalGroups >=
                            static_cast<uint64_t>(coreNum) *
                                sixByThreeMinGroupsPerCore &&
                        sixByThreeRectangleTotalGroups <=
                            static_cast<uint64_t>(coreNum) *
                                sixByThreeMaxGroupsPerCore;
                    const bool sixByThreeRectangleRange =
                        adjacentRectangleTopology &&
                        inputFastExtent >= 6U &&
                        outputFastExtent >= 3U &&
                        outputFastExtent % 3U == 0U &&
                        (sixByThreeInputTail == 0U ||
                         sixByThreeInputTail >= 4U) &&
                        lastTwoWidth * 6U <= tileRows * 10U &&
                        lastTwoHeight * 3U <= sixByThreePackedRowCapacity &&
                        sixByThreeCoreDense;
                    if (sixByThreeRectangleRange) {
                        tiling->fastPath = 29U;
                        context->SetBlockDim(mergedCoreNum);
                        context->SetTilingKey(29U);
                        return ge::GRAPH_SUCCESS;
                    }
                    // Four input coordinates reduce the number of wide MTE2
                    // rectangles while two Gather phases reuse the consumed
                    // input bank as packed output.  The range is expressed
                    // only through topology, fill, UB capacity, divisibility,
                    // and scheduler density.
                    constexpr uint64_t fourByFourMinGroupsPerCore = 12U;
                    constexpr uint64_t fourByFourMaxGroupsPerCore = 24U;
                    constexpr uint64_t fourByFourPackedRowBlocks = 13U;
                    const uint64_t fourByFourPackedRowCapacity =
                        fourByFourPackedRowBlocks * (32U / typeLength);
                    const uint64_t fourByFourInputTail =
                        inputFastExtent % 4U;
                    const bool fourByFourCoreDense =
                        fourByFourRectangleTotalGroups >=
                            static_cast<uint64_t>(coreNum) *
                                fourByFourMinGroupsPerCore &&
                        fourByFourRectangleTotalGroups <=
                            static_cast<uint64_t>(coreNum) *
                                fourByFourMaxGroupsPerCore;
                    const bool fourByFourRectangleRange =
                        adjacentRectangleTopology &&
                        inputFastExtent >= 4U &&
                        outputFastExtent >= 4U &&
                        outputFastExtent % 4U == 0U &&
                        (fourByFourInputTail == 0U ||
                         fourByFourInputTail >= 3U) &&
                        lastTwoWidth * 4U <= tileRows * 7U &&
                        lastTwoHeight * 4U <= fourByFourPackedRowCapacity &&
                        fourByFourCoreDense;
                    if (fourByFourRectangleRange) {
                        tiling->fastPath = 28U;
                        context->SetBlockDim(mergedCoreNum);
                        context->SetTilingKey(28U);
                        return ge::GRAPH_SUCCESS;
                    }
                    // A 3x4 rectangle favors the output-bound side of the
                    // pipeline.  Shared vector-output and packed banks keep
                    // its UB footprint below 192 KiB, while double-buffered
                    // inputs preserve MTE2 overlap.  Selection depends only
                    // on adjacency, capacity, axis extents, and scheduler
                    // density.
                    constexpr uint64_t tripleQuadMinGroupsPerCore = 18U;
                    constexpr uint64_t tripleQuadMaxGroupsPerCore = 32U;
                    constexpr uint64_t tripleQuadPackedRowBlocks = 13U;
                    const uint64_t tripleQuadPackedRowCapacity =
                        tripleQuadPackedRowBlocks * (32U / typeLength);
                    const bool tripleQuadCoreDense =
                        tripleQuadRectangleTotalGroups >=
                            static_cast<uint64_t>(coreNum) *
                                tripleQuadMinGroupsPerCore &&
                        tripleQuadRectangleTotalGroups <=
                            static_cast<uint64_t>(coreNum) *
                                tripleQuadMaxGroupsPerCore;
                    const bool tripleQuadRectangleRange =
                        adjacentRectangleTopology &&
                        inputFastExtent >= 3U &&
                        outputFastExtent >= 4U &&
                        outputFastExtent % 4U == 0U &&
                        lastTwoWidth * 3U <= tileRows * 5U &&
                        lastTwoHeight * 4U <= tripleQuadPackedRowCapacity &&
                        tripleQuadCoreDense;
                    if (tripleQuadRectangleRange) {
                        tiling->fastPath = 24U;
                        context->SetBlockDim(mergedCoreNum);
                        context->SetTilingKey(24U);
                        return ge::GRAPH_SUCCESS;
                    }
                    // A 4x2 rectangle halves the number of wide MTE2
                    // descriptors while preserving two output batches.  Its
                    // compact Gather rows consume seven DMA blocks, and the
                    // work-density interval keeps offset construction and
                    // tail handling amortized without identifying a shape.
                    constexpr uint64_t quadRectangleMinGroupsPerCore = 30U;
                    constexpr uint64_t quadRectangleMaxGroupsPerCore = 48U;
                    constexpr uint64_t quadPackedRowBlocks = 7U;
                    const uint64_t quadPackedRowCapacity =
                        quadPackedRowBlocks * (32U / typeLength);
                    const bool quadRectangleCoreDense =
                        quadRectangleTotalGroups >=
                            static_cast<uint64_t>(coreNum) *
                                quadRectangleMinGroupsPerCore &&
                        quadRectangleTotalGroups <=
                            static_cast<uint64_t>(coreNum) *
                                quadRectangleMaxGroupsPerCore;
                    const bool quadRectangleRange =
                        adjacentRectangleTopology &&
                        inputFastExtent >= 4U &&
                        lastTwoWidth * 4U <= tileRows * 7U &&
                        lastTwoHeight * 2U <= quadPackedRowCapacity &&
                        quadRectangleCoreDense;
                    if (quadRectangleRange) {
                        tiling->fastPath = 23U;
                        context->SetBlockDim(mergedCoreNum);
                        context->SetTilingKey(23U);
                        return ge::GRAPH_SUCCESS;
                    }
                    const bool adjacentRectangleRange =
                        adjacentRectangleTopology && rectangleCoreDense;
                    if (adjacentRectangleRange) {
                        tiling->fastPath = 20U;
                        context->SetBlockDim(mergedCoreNum);
                        context->SetTilingKey(20U);
                        return ge::GRAPH_SUCCESS;
                    }
                    tiling->fastPath = 19U;
                    context->SetBlockDim(mergedCoreNum);
                    context->SetTilingKey(19U);
                    return ge::GRAPH_SUCCESS;
                }
                tiling->fastPath = 16U;
                context->SetBlockDim(mergedCoreNum);
                context->SetTilingKey(16U);
                return ge::GRAPH_SUCCESS;
            }
        }
        const uint64_t tasksPerBatch = ((lastTwoHeight + tileRows - 1U) / tileRows) *
                                       ((lastTwoWidth + tileCols - 1U) / tileCols);
        const uint64_t minimumElementsPerTask = (tileRows * tileCols) / 4U;
        if (lastTwoHeight * lastTwoWidth >= tasksPerBatch * minimumElementsPerTask) {
            const uint64_t totalTasks = lastTwoBatch * tasksPerBatch;
            const uint32_t tiledCoreNum = static_cast<uint32_t>(
                totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks : static_cast<uint64_t>(coreNum));
            const uint32_t matrixKey = generalOutputStride ? 11U : 8U;
            tiling->fastPath = matrixKey;
            context->SetBlockDim(tiledCoreNum);
            context->SetTilingKey(matrixKey);
            return ge::GRAPH_SUCCESS;
        }

        // Repeated small matrices cannot amortize a four-block-wide tile, but
        // one native VNCHW tile still removes per-element division and scalar
        // GM gathers.  Keep at least one eighth of the native tile occupied
        // so sparse unit-dimension cases stay on the lightweight fallback.
        const uint64_t compactTileCols = tileRows;
        const uint64_t compactTasksPerBatch = ((lastTwoHeight + tileRows - 1U) / tileRows) *
                                              ((lastTwoWidth + compactTileCols - 1U) / compactTileCols);
        const uint64_t compactTotalTasks = lastTwoBatch * compactTasksPerBatch;
        const uint64_t compactMinimumElementsPerTask = (tileRows * compactTileCols) / 8U;
        const uint64_t compactDenseRows = lastTwoHeight < tileRows ? lastTwoHeight : tileRows;
        const uint64_t compactDenseCols = lastTwoWidth < compactTileCols ? lastTwoWidth : compactTileCols;
        const bool compactAverageDense =
            lastTwoHeight * lastTwoWidth >= compactTasksPerBatch * compactMinimumElementsPerTask;
        // Do not let one unavoidable tail tile send a long output row to the
        // much slower per-element strided-DMA path.  Below that path's
        // evidence-based 512-element crossover, retain the stricter average
        // density gate because the lightweight fallback is faster.
        const bool compactFullTileDense =
            lastTwoHeight >= 512U &&
            compactDenseRows * compactDenseCols >= compactMinimumElementsPerTask;
        if (compactAverageDense || compactFullTileDense) {
            const uint32_t compactCoreNum = static_cast<uint32_t>(
                compactTotalTasks < static_cast<uint64_t>(coreNum) ? compactTotalTasks :
                                                                    static_cast<uint64_t>(coreNum));
            tiling->fastPath = 14;
            context->SetBlockDim(compactCoreNum);
            context->SetTilingKey(14U);
            return ge::GRAPH_SUCCESS;
        }
    }

    int64_t suffixStart = rank - 1;
    if (perm[rank - 1] == rank - 1) {
        while (suffixStart > 0 && perm[suffixStart - 1] + 1 == perm[suffixStart]) {
            --suffixStart;
        }
        uint64_t blockElements = 1U;
        for (int64_t i = suffixStart; i < rank; ++i) {
            blockElements *= static_cast<uint64_t>(tiling->shape[perm[i]]);
        }
        if (blockElements * typeLength >= BLOCK_SIZE) {
            constexpr uint64_t chunkBytes = 32U * 1024U;
            const uint64_t chunkElements = chunkBytes / typeLength;
            const uint64_t outerBlocks = static_cast<uint64_t>(totalIdx) / blockElements;
            const uint64_t chunksPerBlock = (blockElements + chunkElements - 1U) / chunkElements;
            const uint64_t totalTasks = outerBlocks * chunksPerBlock;
            const uint32_t tiledCoreNum = static_cast<uint32_t>(
                totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks : static_cast<uint64_t>(coreNum));
            tiling->fastPath = 4;
            context->SetBlockDim(tiledCoreNum);
            context->SetTilingKey(4U);
            return ge::GRAPH_SUCCESS;
        }

        // A short contiguous suffix is a record attached to every element of
        // the remaining permutation.  Pair the physical axis immediately
        // before that suffix with the last non-suffix output axis, move record
        // tiles by 2-D DMA, and reconstruct addresses once per tile instead of
        // once per element.  The maximal suffix guarantees these are distinct
        // axes.
        const int64_t suffixInputAxis = perm[suffixStart];
        if (totalIdx >= 4096 && suffixStart > 0 && suffixInputAxis > 0) {
            const int64_t inputContiguousAxis = suffixInputAxis - 1;
            const int64_t outputLastAxis = perm[suffixStart - 1];
            const uint64_t recordHeight = static_cast<uint64_t>(tiling->shape[outputLastAxis]);
            const uint64_t recordWidth = static_cast<uint64_t>(tiling->shape[inputContiguousAxis]);
            constexpr uint64_t recordTileExtent = 16U;
            const uint64_t recordTasksPerBatch =
                ((recordHeight + recordTileExtent - 1U) / recordTileExtent) *
                ((recordWidth + recordTileExtent - 1U) / recordTileExtent);
            const uint64_t recordMatrixElements = recordHeight * recordWidth * blockElements;
            const uint64_t recordBatch = static_cast<uint64_t>(totalIdx) / recordMatrixElements;
            const uint64_t recordTotalTasks = recordBatch * recordTasksPerBatch;
            constexpr uint64_t minimumRecordsPerTask =
                (recordTileExtent * recordTileExtent) / 16U;
            if (inputContiguousAxis != outputLastAxis && recordHeight > 1U && recordWidth > 1U &&
                recordHeight * recordWidth >= recordTasksPerBatch * minimumRecordsPerTask) {
                const uint32_t recordCoreNum = static_cast<uint32_t>(
                    recordTotalTasks < static_cast<uint64_t>(coreNum) ? recordTotalTasks :
                                                                       static_cast<uint64_t>(coreNum));
                tiling->fastPath = 15;
                context->SetBlockDim(recordCoreNum);
                context->SetTilingKey(15U);
                return ge::GRAPH_SUCCESS;
            }
        }
    }
    const uint64_t rowElements = static_cast<uint64_t>(tiling->shape[perm[rank - 1]]);

    // Unit dimensions do not change the flattened memory layout.  If the
    // original-rank recognizers miss a case because a moved unit axis is at an
    // edge, squeeze those axes only in the private tiling payload and reuse
    // the same dense-matrix or contiguous-suffix kernels.  Shape inference has
    // already consumed the original rank, so public output semantics remain
    // unchanged.
    if (totalIdx >= 4096) {
        int64_t squeezedShape[MAX_DIM] = {};
        int64_t squeezedPerm[MAX_DIM] = {};
        int64_t originalToSqueezed[MAX_DIM] = {};
        int64_t squeezedRank = 0;
        for (int64_t axis = 0; axis < rank; ++axis) {
            if (tiling->shape[axis] != 1) {
                originalToSqueezed[axis] = squeezedRank;
                squeezedShape[squeezedRank++] = tiling->shape[axis];
            }
        }
        int64_t squeezedOutput = 0;
        for (int64_t position = 0; position < rank; ++position) {
            if (tiling->shape[perm[position]] != 1) {
                squeezedPerm[squeezedOutput++] = originalToSqueezed[perm[position]];
            }
        }

        const auto commitSqueezed = [&]() {
            tiling->dims = squeezedRank;
            for (uint32_t i = 0U; i < MAX_DIM; ++i) {
                tiling->shape[i] = i < static_cast<uint32_t>(squeezedRank) ? squeezedShape[i] : 0;
                tiling->perm[i] = i < static_cast<uint32_t>(squeezedRank) ? squeezedPerm[i] : 0;
            }
        };

        if (squeezedRank < rank && squeezedRank > 0) {
            bool squeezedIdentity = true;
            for (int64_t i = 0; i < squeezedRank; ++i) {
                squeezedIdentity = squeezedIdentity && squeezedPerm[i] == i;
            }
            if (squeezedIdentity) {
                constexpr uint64_t chunkBytes = 32U * 1024U;
                const uint64_t chunkElements = chunkBytes / typeLength;
                const uint64_t totalTasks =
                    (static_cast<uint64_t>(totalIdx) + chunkElements - 1U) / chunkElements;
                const uint32_t squeezedCoreNum = static_cast<uint32_t>(
                    totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks : static_cast<uint64_t>(coreNum));
                commitSqueezed();
                tiling->fastPath = 4;
                context->SetBlockDim(squeezedCoreNum);
                context->SetTilingKey(4U);
                return ge::GRAPH_SUCCESS;
            }

            if (squeezedRank >= 2 && squeezedPerm[squeezedRank - 1] != squeezedRank - 1) {
                const uint64_t squeezedHeight =
                    static_cast<uint64_t>(squeezedShape[squeezedPerm[squeezedRank - 1]]);
                const uint64_t squeezedWidth = static_cast<uint64_t>(squeezedShape[squeezedRank - 1]);
                const uint64_t squeezedMatrixElements = squeezedHeight * squeezedWidth;
                const uint64_t squeezedBatch = static_cast<uint64_t>(totalIdx) / squeezedMatrixElements;
                const uint64_t tileRows = typeLength == 1U ? 32U : 16U;

                const uint64_t tileBlocks[4] = {64U, 16U, 2U, 1U};
                const uint32_t matrixKeys[4] = {13U, 12U, 11U, 14U};
                const uint64_t densityDivisors[4] = {4U, 4U, 4U, 8U};
                if (typeLength == 4U && squeezedBatch >= 1024U &&
                    squeezedHeight > tileRows) {
                    const uint64_t mergedTileRows = tileRows * 2U;
                    const uint64_t mergedTileCols = tileRows * 2U;
                    const uint64_t mergedTasksPerBatch =
                        ((squeezedHeight + mergedTileRows - 1U) / mergedTileRows) *
                        ((squeezedWidth + mergedTileCols - 1U) / mergedTileCols);
                    const uint64_t mergedMinimumElementsPerTask =
                        (mergedTileRows * mergedTileCols) / 2U;
                    if (squeezedHeight * squeezedWidth >=
                        mergedTasksPerBatch * mergedMinimumElementsPerTask) {
                        const uint64_t mergedTotalTasks = squeezedBatch * mergedTasksPerBatch;
                        const uint32_t mergedCoreNum = static_cast<uint32_t>(
                            mergedTotalTasks < static_cast<uint64_t>(coreNum) ? mergedTotalTasks :
                                                                              static_cast<uint64_t>(coreNum));
                        commitSqueezed();
                        tiling->fastPath = 16U;
                        context->SetBlockDim(mergedCoreNum);
                        context->SetTilingKey(16U);
                        return ge::GRAPH_SUCCESS;
                    }
                }
                for (uint32_t candidate = 0U; candidate < 4U; ++candidate) {
                    const uint64_t tileCols = tileRows * tileBlocks[candidate];
                    const uint64_t tasksPerBatch =
                        ((squeezedHeight + tileRows - 1U) / tileRows) *
                        ((squeezedWidth + tileCols - 1U) / tileCols);
                    const uint64_t totalTasks = squeezedBatch * tasksPerBatch;
                    const uint64_t minimumElementsPerTask =
                        (tileRows * tileCols) / densityDivisors[candidate];
                    const uint64_t denseRows = squeezedHeight < tileRows ? squeezedHeight : tileRows;
                    const uint64_t denseCols = squeezedWidth < tileCols ? squeezedWidth : tileCols;
                    const bool averageDense = squeezedHeight * squeezedWidth >=
                                              tasksPerBatch * minimumElementsPerTask;
                    const bool compactFullTileDense = candidate == 3U && squeezedHeight >= 512U &&
                                                      denseRows * denseCols >= minimumElementsPerTask;
                    const bool enoughDensity = averageDense || compactFullTileDense;
                    const bool enoughParallelism = candidate >= 2U ||
                                                   totalTasks >= static_cast<uint64_t>(coreNum);
                    if (enoughDensity && enoughParallelism) {
                        const uint32_t squeezedCoreNum = static_cast<uint32_t>(
                            totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks :
                                                                        static_cast<uint64_t>(coreNum));
                        commitSqueezed();
                        tiling->fastPath = matrixKeys[candidate];
                        context->SetBlockDim(squeezedCoreNum);
                        context->SetTilingKey(matrixKeys[candidate]);
                        return ge::GRAPH_SUCCESS;
                    }
                }
            } else if (squeezedRank >= 2) {
                int64_t squeezedSuffixStart = squeezedRank - 1;
                while (squeezedSuffixStart > 0 &&
                       squeezedPerm[squeezedSuffixStart - 1] + 1 == squeezedPerm[squeezedSuffixStart]) {
                    --squeezedSuffixStart;
                }
                uint64_t squeezedBlockElements = 1U;
                for (int64_t i = squeezedSuffixStart; i < squeezedRank; ++i) {
                    squeezedBlockElements *= static_cast<uint64_t>(squeezedShape[squeezedPerm[i]]);
                }
                if (squeezedBlockElements * typeLength >= BLOCK_SIZE) {
                    constexpr uint64_t chunkBytes = 32U * 1024U;
                    const uint64_t chunkElements = chunkBytes / typeLength;
                    const uint64_t outerBlocks = static_cast<uint64_t>(totalIdx) / squeezedBlockElements;
                    const uint64_t chunksPerBlock =
                        (squeezedBlockElements + chunkElements - 1U) / chunkElements;
                    const uint64_t totalTasks = outerBlocks * chunksPerBlock;
                    const uint32_t squeezedCoreNum = static_cast<uint32_t>(
                        totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks :
                                                                    static_cast<uint64_t>(coreNum));
                    commitSqueezed();
                    tiling->fastPath = 4;
                    context->SetBlockDim(squeezedCoreNum);
                    context->SetTilingKey(4U);
                    return ge::GRAPH_SUCCESS;
                }

                const int64_t suffixInputAxis = squeezedPerm[squeezedSuffixStart];
                if (squeezedSuffixStart > 0 && suffixInputAxis > 0) {
                    const int64_t inputContiguousAxis = suffixInputAxis - 1;
                    const int64_t outputLastAxis = squeezedPerm[squeezedSuffixStart - 1];
                    const uint64_t recordHeight =
                        static_cast<uint64_t>(squeezedShape[outputLastAxis]);
                    const uint64_t recordWidth =
                        static_cast<uint64_t>(squeezedShape[inputContiguousAxis]);
                    constexpr uint64_t tileExtent = 16U;
                    const uint64_t tasksPerBatch =
                        ((recordHeight + tileExtent - 1U) / tileExtent) *
                        ((recordWidth + tileExtent - 1U) / tileExtent);
                    const uint64_t matrixElements = recordHeight * recordWidth * squeezedBlockElements;
                    const uint64_t batch = static_cast<uint64_t>(totalIdx) / matrixElements;
                    const uint64_t totalTasks = batch * tasksPerBatch;
                    constexpr uint64_t minimumRecordsPerTask =
                        (tileExtent * tileExtent) / 16U;
                    if (inputContiguousAxis != outputLastAxis && recordHeight > 1U && recordWidth > 1U &&
                        recordHeight * recordWidth >= tasksPerBatch * minimumRecordsPerTask) {
                        const uint32_t squeezedCoreNum = static_cast<uint32_t>(
                            totalTasks < static_cast<uint64_t>(coreNum) ? totalTasks :
                                                                        static_cast<uint64_t>(coreNum));
                        commitSqueezed();
                        tiling->fastPath = 15;
                        context->SetBlockDim(squeezedCoreNum);
                        context->SetTilingKey(15U);
                        return ge::GRAPH_SUCCESS;
                    }
                }
            }
        }
    }

    if (rowElements >= 512U) {
        constexpr uint64_t stridedChunkRows = 2048U;
        const uint64_t outerRows = static_cast<uint64_t>(totalIdx) / rowElements;
        const uint64_t chunksPerRow = (rowElements + stridedChunkRows - 1U) / stridedChunkRows;
        const uint64_t stridedTasks = outerRows * chunksPerRow;
        const uint32_t stridedCoreNum = static_cast<uint32_t>(
            stridedTasks < static_cast<uint64_t>(coreNum) ? stridedTasks : static_cast<uint64_t>(coreNum));
        tiling->fastPath = 5;
        context->SetBlockDim(stridedCoreNum);
        context->SetTilingKey(5U);
        return ge::GRAPH_SUCCESS;
    }
    context->SetTilingKey(0U);

    // --- safer numeric types ---
    uint64_t inputBytes = static_cast<uint64_t>(typeLength);
    if (totalIdx <= 0) {
        return ge::GRAPH_FAILED;
    }

    // ub-based tileBlockNum guard (避免为0)
    uint32_t ubDataNumber = 0;
    switch (inputBytes) {
        // 核内每次处理的元素个数为n，中间计算需要使用的临时tensor为6*n个元素（每个元素大小4字节）
        // 数值的计算方式为：6*n*4B / inputBytes + 1
        case 1:
            ubDataNumber = 25U;
            break; // 输入元素的数据类型长度为1B
        case 2:
            ubDataNumber = 13U;
            break; // 输入元素的数据类型长度为2B
        case 4:
            ubDataNumber = 7U;
            break; // 输入元素的数据类型长度为4B
        case 8:
            ubDataNumber = 4U;
            break; // 输入元素的数据类型长度为8B
        default:
            ubDataNumber = 25U;
            break; // 未知类型按 1 B 处理
    }
    uint64_t tmp = (ubSize / BLOCK_SIZE / BUFFER_NUM);
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
        uint64_t tb = tmp / ubDataNumber;
        tileBlockNum = (tb == 0) ? 1U : static_cast<uint32_t>(tb);
    }

    // 每个 tile 包含的元素数（至少 1）
    uint32_t tileDataNum = static_cast<uint32_t>((static_cast<uint64_t>(tileBlockNum) * BLOCK_SIZE) / inputBytes);
    if (tileDataNum == 0U)
        tileDataNum = 1U;

    // 按真实元素数分核。不能按向上取整后的 32B block 数分配，否则最后一个核
    // 会处理并写回逻辑张量末尾之外的填充元素。
    const uint64_t totalElements = static_cast<uint64_t>(totalIdx);
    uint64_t coreNum64 = static_cast<uint64_t>(coreNum);
    if (coreNum64 > totalElements) {
        coreNum64 = totalElements;
    }
    uint32_t finalCoreNum = static_cast<uint32_t>(coreNum64);

    const uint64_t smallCoreDataNum = totalElements / coreNum64;
    const uint32_t tailBlockNum = static_cast<uint32_t>(totalElements % coreNum64);
    const uint64_t bigCoreDataNum = smallCoreDataNum + (tailBlockNum == 0U ? 0ULL : 1ULL);

    const auto tileCount = [tileDataNum](uint64_t elements) -> uint64_t {
        return (elements + static_cast<uint64_t>(tileDataNum) - 1ULL) / static_cast<uint64_t>(tileDataNum);
    };
    const auto tailCount = [tileDataNum](uint64_t elements, uint64_t tiles) -> uint64_t {
        return elements - static_cast<uint64_t>(tileDataNum) * (tiles - 1ULL);
    };
    const uint64_t finalSmallTileNum = tileCount(smallCoreDataNum);
    const uint64_t smallTailDataNum = tailCount(smallCoreDataNum, finalSmallTileNum);
    const uint64_t finalBigTileNum = tileCount(bigCoreDataNum);
    const uint64_t bigTailDataNum = tailCount(bigCoreDataNum, finalBigTileNum);

    // write back
    tiling->smallCoreDataNum = static_cast<int64_t>(smallCoreDataNum);
    tiling->bigCoreDataNum = static_cast<int64_t>(bigCoreDataNum);
    tiling->tileDataNum = static_cast<int64_t>(tileDataNum);
    tiling->smallTailDataNum = static_cast<int64_t>(smallTailDataNum);
    tiling->bigTailDataNum = static_cast<int64_t>(bigTailDataNum);
    tiling->finalSmallTileNum = static_cast<int64_t>(finalSmallTileNum);
    tiling->finalBigTileNum = static_cast<int64_t>(finalBigTileNum);
    tiling->tailBlockNum = static_cast<int64_t>(tailBlockNum);
    context->SetBlockDim(finalCoreNum);
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

#include "register/op_def_registry.h"

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dims").ListInt();

        this->SetInferShape(InferShapeTranspose).SetInferDataType(InferDataTypeTranspose);
        this->AICore().SetTiling(optiling::TransposeTilingFunc);

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Transpose);
} // namespace ops
