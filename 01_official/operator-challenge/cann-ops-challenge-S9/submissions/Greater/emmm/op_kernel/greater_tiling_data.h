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
 * \file greater_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __GREATER_TILLING_DATA_H__
#define __GREATER_TILLING_DATA_H__

constexpr uint32_t GREATER_MAX_BROADCAST_DIM = 8;
constexpr uint32_t GREATER_BROADCAST_CONTIGUOUS = 0;
constexpr uint32_t GREATER_BROADCAST_X1_SCALAR = 1;
constexpr uint32_t GREATER_BROADCAST_X2_SCALAR = 2;
constexpr uint32_t GREATER_BROADCAST_GENERAL = 3;
constexpr uint32_t GREATER_INT32_COHORT_MASK = 0xFFU;
constexpr uint32_t GREATER_INT32_LARGE_TILE_FLAG = 1U << 8;
constexpr uint32_t GREATER_INT32_DIRECT_ROW_PIPELINE_FLAG = 1U << 9;
constexpr uint32_t GREATER_INT32_LINEAR_GROUP_OFFSETS_FLAG = 1U << 10;
constexpr uint32_t GREATER_INT32_PACKED_GM_FLAG = 1U << 31;

struct GreaterCoreTilingData {
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint64_t ubPartDataNum;
    uint64_t smallCoreTailDataNum;
    uint64_t bigCoreTailDataNum;
    uint64_t smallCoreLoopNum;
    uint64_t bigCoreLoopNum;
    uint64_t tailBlockNum;
};

// Compact ABI for the structurally proven linear INT32 repeated-row path.
// Unlike GreaterTilingData, this record contains no rank-sized shape/stride
// tables for the AIV to copy or decode at launch time.
struct GreaterInt32DirectTilingData {
    uint64_t rowLength;
    uint64_t rowCount;
    uint64_t naturalGroupCount;
    uint32_t cohortSize;
    uint32_t x1Rows;
};

struct GreaterTilingData {
    uint64_t smallCoreDataNum;
    uint64_t bigCoreDataNum;
    uint64_t ubPartDataNum;
    uint64_t smallCoreTailDataNum;
    uint64_t bigCoreTailDataNum;
    uint64_t smallCoreLoopNum;
    uint64_t bigCoreLoopNum;
    uint64_t tailBlockNum;
    uint32_t rank;
    uint32_t reserved;
    uint64_t outShape[GREATER_MAX_BROADCAST_DIM];
    uint64_t x1Stride[GREATER_MAX_BROADCAST_DIM];
    uint64_t x2Stride[GREATER_MAX_BROADCAST_DIM];
};
#endif
