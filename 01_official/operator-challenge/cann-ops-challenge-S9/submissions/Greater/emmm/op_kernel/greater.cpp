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
 * \file greater.cpp
 * \brief
 */

#include "greater.h"
#include "greater_fp32_ring.h"
#include "greater_int32_cached_rows.h"
#include "greater_int32_direct_row_pipeline.h"
#include "greater_int32_packed_gm.h"

enum class GreaterTilingKey : uint32_t {
    TILING_KEY_EXAMPLE_FLOAT = 0,
    TILING_KEY_EXAMPLE_INT32 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void greater(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GreaterTilingData);

    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterCoreTilingData, tilingData, tiling);
        NsGreater::Greater<DTYPE_X1, DTYPE_X2, DTYPE_Y, true> op;
        op.Init(
            x1, x2, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
            tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
            tilingData.bigCoreTailDataNum, tilingData.tailBlockNum);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterCoreTilingData, tilingData, tiling);
        NsGreater::Greater<DTYPE_X1, DTYPE_X2, DTYPE_Y, false> op;
        op.Init(
            x1, x2, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.bigCoreLoopNum,
            tilingData.smallCoreLoopNum, tilingData.ubPartDataNum, tilingData.smallCoreTailDataNum,
            tilingData.bigCoreTailDataNum, tilingData.tailBlockNum);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterCoreTilingData, tilingData, tiling);
        NsGreater::GreaterFlexible<DTYPE_X1, DTYPE_X2, DTYPE_Y, GREATER_BROADCAST_CONTIGUOUS> op;
        op.Init(
            x1, x2, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
            tilingData.smallCoreTailDataNum, tilingData.bigCoreTailDataNum, tilingData.smallCoreLoopNum,
            tilingData.bigCoreLoopNum, tilingData.tailBlockNum, 0, nullptr, nullptr, nullptr);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterCoreTilingData, tilingData, tiling);
        NsGreater::GreaterFlexible<DTYPE_X1, DTYPE_X2, DTYPE_Y, GREATER_BROADCAST_X1_SCALAR> op;
        op.Init(
            x1, x2, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
            tilingData.smallCoreTailDataNum, tilingData.bigCoreTailDataNum, tilingData.smallCoreLoopNum,
            tilingData.bigCoreLoopNum, tilingData.tailBlockNum, 0, nullptr, nullptr, nullptr);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterCoreTilingData, tilingData, tiling);
        NsGreater::GreaterFlexible<DTYPE_X1, DTYPE_X2, DTYPE_Y, GREATER_BROADCAST_X2_SCALAR> op;
        op.Init(
            x1, x2, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
            tilingData.smallCoreTailDataNum, tilingData.bigCoreTailDataNum, tilingData.smallCoreLoopNum,
            tilingData.bigCoreLoopNum, tilingData.tailBlockNum, 0, nullptr, nullptr, nullptr);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
        NsGreater::GreaterFlexible<DTYPE_X1, DTYPE_X2, DTYPE_Y, GREATER_BROADCAST_GENERAL> op;
        op.Init(
            x1, x2, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
            tilingData.smallCoreTailDataNum, tilingData.bigCoreTailDataNum, tilingData.smallCoreLoopNum,
            tilingData.bigCoreLoopNum, tilingData.tailBlockNum, tilingData.rank, tilingData.outShape,
            tilingData.x1Stride, tilingData.x2Stride);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        if constexpr (std::is_same_v<DTYPE_X1, int8_t>) {
            GET_TILING_DATA_WITH_STRUCT(
                GreaterCoreTilingData, tilingData, tiling);
            NsGreater::GreaterFlexible<
                DTYPE_X1, DTYPE_X2, DTYPE_Y,
                GREATER_BROADCAST_CONTIGUOUS, false, false, false, false,
                true, true> op;
            op.Init(
                x1, x2, y, tilingData.smallCoreDataNum,
                tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                tilingData.smallCoreTailDataNum,
                tilingData.bigCoreTailDataNum,
                tilingData.smallCoreLoopNum, tilingData.bigCoreLoopNum,
                tilingData.tailBlockNum, 0, nullptr, nullptr, nullptr);
            op.Process();
        } else {
            GET_TILING_DATA_WITH_STRUCT(
                GreaterTilingData, tilingData, tiling);
            NsGreater::GreaterFlexible<
                DTYPE_X1, DTYPE_X2, DTYPE_Y,
                GREATER_BROADCAST_GENERAL, true> op;
            op.Init(
                x1, x2, y, tilingData.smallCoreDataNum,
                tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                tilingData.smallCoreTailDataNum,
                tilingData.bigCoreTailDataNum,
                tilingData.smallCoreLoopNum, tilingData.bigCoreLoopNum,
                tilingData.tailBlockNum, tilingData.rank,
                tilingData.outShape, tilingData.x1Stride,
                tilingData.x2Stride);
            op.Process();
        }
    } else if (TILING_KEY_IS(7)) {
        if constexpr (
            std::is_same_v<DTYPE_X1, int8_t> ||
            std::is_same_v<DTYPE_X1, uint8_t>) {
            GET_TILING_DATA_WITH_STRUCT(
                GreaterCoreTilingData, tilingData, tiling);
            NsGreater::GreaterFlexible<
                DTYPE_X1, DTYPE_X2, DTYPE_Y,
                GREATER_BROADCAST_CONTIGUOUS, false, false, false, false,
                true> op;
            op.Init(
                x1, x2, y, tilingData.smallCoreDataNum,
                tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                tilingData.smallCoreTailDataNum,
                tilingData.bigCoreTailDataNum,
                tilingData.smallCoreLoopNum, tilingData.bigCoreLoopNum,
                tilingData.tailBlockNum, 0, nullptr, nullptr, nullptr);
            op.Process();
        } else {
            GET_TILING_DATA_WITH_STRUCT(
                GreaterTilingData, tilingData, tiling);
            NsGreater::GreaterFlexible<
                DTYPE_X1, DTYPE_X2, DTYPE_Y,
                GREATER_BROADCAST_GENERAL, true, true> op;
            op.Init(
                x1, x2, y, tilingData.smallCoreDataNum,
                tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                tilingData.smallCoreTailDataNum,
                tilingData.bigCoreTailDataNum,
                tilingData.smallCoreLoopNum, tilingData.bigCoreLoopNum,
                tilingData.tailBlockNum, tilingData.rank,
                tilingData.outShape, tilingData.x1Stride,
                tilingData.x2Stride);
            op.Process();
        }
    } else if (TILING_KEY_IS(8)) {
        if constexpr (std::is_same_v<DTYPE_X1, int32_t>) {
            GET_TILING_DATA_WITH_STRUCT(
                GreaterInt32DirectTilingData, tilingData, tiling);
            if ((tilingData.cohortSize & GREATER_INT32_PACKED_GM_FLAG) != 0) {
                NsGreater::GreaterInt32PackedGm op;
                op.Init(x1, x2, y, workspace, tilingData);
                op.Process();
            } else {
                NsGreater::GreaterInt32DirectRowPipeline<true> op;
                op.InitLinear(x1, x2, y, tilingData);
                op.Process();
            }
        } else {
            GET_TILING_DATA_WITH_STRUCT(GreaterTilingData, tilingData, tiling);
            NsGreater::GreaterFlexible<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, GREATER_BROADCAST_GENERAL, false, false, true> op;
            op.Init(
                x1, x2, y, tilingData.smallCoreDataNum, tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                tilingData.smallCoreTailDataNum, tilingData.bigCoreTailDataNum, tilingData.smallCoreLoopNum,
                tilingData.bigCoreLoopNum, tilingData.tailBlockNum, tilingData.rank, tilingData.outShape,
                tilingData.x1Stride, tilingData.x2Stride);
            op.Process();
        }
    } else if (TILING_KEY_IS(9)) {
        if constexpr (std::is_same_v<DTYPE_X1, float>) {
            GET_TILING_DATA_WITH_STRUCT(GreaterCoreTilingData, tilingData, tiling);
            NsGreater::GreaterFlexible<
                DTYPE_X1, DTYPE_X2, DTYPE_Y,
                GREATER_BROADCAST_CONTIGUOUS, false, false, false, false,
                false, false, true> op;
            op.Init(
                x1, x2, y, tilingData.smallCoreDataNum,
                tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                tilingData.smallCoreTailDataNum,
                tilingData.bigCoreTailDataNum,
                tilingData.smallCoreLoopNum, tilingData.bigCoreLoopNum,
                tilingData.tailBlockNum, 0, nullptr, nullptr, nullptr);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_X1, bfloat16_t>) {
            GET_TILING_DATA_WITH_STRUCT(
                GreaterTilingData, tilingData, tiling);
            NsGreater::GreaterFlexible<
                DTYPE_X1, DTYPE_X2, DTYPE_Y,
                GREATER_BROADCAST_GENERAL, false, false, false, false,
                false, false, true> op;
            op.Init(
                x1, x2, y, tilingData.smallCoreDataNum,
                tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                tilingData.smallCoreTailDataNum,
                tilingData.bigCoreTailDataNum,
                tilingData.smallCoreLoopNum,
                tilingData.bigCoreLoopNum, tilingData.tailBlockNum,
                tilingData.rank, tilingData.outShape,
                tilingData.x1Stride, tilingData.x2Stride);
            op.Process();
        } else if constexpr (std::is_same_v<DTYPE_X1, int32_t>) {
            GET_TILING_DATA_WITH_STRUCT(
                GreaterTilingData, tilingData, tiling);
            if ((tilingData.reserved &
                 GREATER_INT32_DIRECT_ROW_PIPELINE_FLAG) != 0) {
                NsGreater::GreaterInt32DirectRowPipeline<> op;
                op.Init(
                    x1, x2, y, tilingData.smallCoreDataNum,
                    tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                    tilingData.smallCoreTailDataNum,
                    tilingData.bigCoreTailDataNum,
                    tilingData.smallCoreLoopNum,
                    tilingData.bigCoreLoopNum, tilingData.tailBlockNum,
                    tilingData.rank, tilingData.outShape,
                    tilingData.x1Stride, tilingData.x2Stride,
                    tilingData.reserved);
                op.Process();
            } else {
                NsGreater::GreaterInt32CachedRows op;
                op.Init(
                    x1, x2, y, tilingData.smallCoreDataNum,
                    tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                    tilingData.smallCoreTailDataNum,
                    tilingData.bigCoreTailDataNum,
                    tilingData.smallCoreLoopNum,
                    tilingData.bigCoreLoopNum, tilingData.tailBlockNum,
                    tilingData.rank, tilingData.outShape,
                    tilingData.x1Stride, tilingData.x2Stride,
                    tilingData.reserved);
                op.Process();
            }
        } else {
            GET_TILING_DATA_WITH_STRUCT(
                GreaterTilingData, tilingData, tiling);
            NsGreater::GreaterFlexible<
                DTYPE_X1, DTYPE_X2, DTYPE_Y,
                GREATER_BROADCAST_GENERAL, false, false, false, true> op;
            op.Init(
                x1, x2, y, tilingData.smallCoreDataNum,
                tilingData.bigCoreDataNum, tilingData.ubPartDataNum,
                tilingData.smallCoreTailDataNum,
                tilingData.bigCoreTailDataNum,
                tilingData.smallCoreLoopNum,
                tilingData.bigCoreLoopNum, tilingData.tailBlockNum,
                tilingData.rank, tilingData.outShape,
                tilingData.x1Stride, tilingData.x2Stride);
            op.Process();
        }
    }
}
