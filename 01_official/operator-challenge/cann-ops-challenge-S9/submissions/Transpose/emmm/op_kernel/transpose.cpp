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
 * \file transpose.h
 * \brief
 */
#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
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
 * \file transpose_tiling_data.h
 * \brief tiling data struct
 */
#ifndef _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_
#define _ROTARY_POSITION_EMBEDDING_GRAD_TILING_DATA_H_

struct TransposeTilingData {
    int64_t fastPath;
    int64_t smallCoreDataNum;
    int64_t bigCoreDataNum;
    int64_t finalBigTileNum;
    int64_t finalSmallTileNum;
    int64_t tileDataNum;
    int64_t smallTailDataNum;
    int64_t bigTailDataNum;
    int64_t tailBlockNum;
    int64_t dims;
    int64_t totalnumber;
    int64_t shape[8];
    uint64_t vaConfigPadding;
    uint64_t dstVaConfig[32];
    uint64_t srcVaConfig[32];
    int64_t perm[8];
};
#endif
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
 * \file transpose_tiling_key.h
 * \brief transpose tiling key declare
 */
#ifndef __TRANSPOSE_TILING_KEY_H__
#define __TRANSPOSE_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

#define ELEMENTWISE_TPL_SCH_MODE_0 0
#define ELEMENTWISE_TPL_SCH_MODE_1 1
#define ELEMENTWISE_TPL_SCH_MODE_2 2
#define ELEMENTWISE_TPL_SCH_MODE_3 3
#define ELEMENTWISE_TPL_SCH_MODE_4 4
#define ELEMENTWISE_TPL_SCH_MODE_5 5
#define ELEMENTWISE_TPL_SCH_MODE_6 6
#define ELEMENTWISE_TPL_SCH_MODE_7 7
#define ELEMENTWISE_TPL_SCH_MODE_8 8
#define ELEMENTWISE_TPL_SCH_MODE_9 9
#define ELEMENTWISE_TPL_SCH_MODE_10 10
#define ELEMENTWISE_TPL_SCH_MODE_11 11
#define ELEMENTWISE_TPL_SCH_MODE_12 12
#define ELEMENTWISE_TPL_SCH_MODE_13 13
#define ELEMENTWISE_TPL_SCH_MODE_14 14
#define ELEMENTWISE_TPL_SCH_MODE_15 15
#define ELEMENTWISE_TPL_SCH_MODE_16 16
#define ELEMENTWISE_TPL_SCH_MODE_17 17
#define ELEMENTWISE_TPL_SCH_MODE_18 18
#define ELEMENTWISE_TPL_SCH_MODE_19 19
#define ELEMENTWISE_TPL_SCH_MODE_20 20
#define ELEMENTWISE_TPL_SCH_MODE_21 21
#define ELEMENTWISE_TPL_SCH_MODE_22 22
#define ELEMENTWISE_TPL_SCH_MODE_23 23
#define ELEMENTWISE_TPL_SCH_MODE_24 24
#define ELEMENTWISE_TPL_SCH_MODE_25 25
#define ELEMENTWISE_TPL_SCH_MODE_26 26
#define ELEMENTWISE_TPL_SCH_MODE_27 27
#define ELEMENTWISE_TPL_SCH_MODE_28 28
#define ELEMENTWISE_TPL_SCH_MODE_29 29
#define ELEMENTWISE_TPL_SCH_MODE_30 30
#define ELEMENTWISE_TPL_SCH_MODE_31 31
#define ELEMENTWISE_TPL_SCH_MODE_32 32
#define ELEMENTWISE_TPL_SCH_MODE_33 33
#define ELEMENTWISE_TPL_SCH_MODE_34 34

ASCENDC_TPL_ARGS_DECL(
    Transpose,
    ASCENDC_TPL_UINT_DECL(schMode, 6, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1,
        ELEMENTWISE_TPL_SCH_MODE_2, ELEMENTWISE_TPL_SCH_MODE_3, ELEMENTWISE_TPL_SCH_MODE_4,
        ELEMENTWISE_TPL_SCH_MODE_5, ELEMENTWISE_TPL_SCH_MODE_6, ELEMENTWISE_TPL_SCH_MODE_7,
        ELEMENTWISE_TPL_SCH_MODE_8, ELEMENTWISE_TPL_SCH_MODE_9, ELEMENTWISE_TPL_SCH_MODE_10,
        ELEMENTWISE_TPL_SCH_MODE_11, ELEMENTWISE_TPL_SCH_MODE_12, ELEMENTWISE_TPL_SCH_MODE_13,
        ELEMENTWISE_TPL_SCH_MODE_14, ELEMENTWISE_TPL_SCH_MODE_15, ELEMENTWISE_TPL_SCH_MODE_16,
        ELEMENTWISE_TPL_SCH_MODE_17, ELEMENTWISE_TPL_SCH_MODE_18,
        ELEMENTWISE_TPL_SCH_MODE_19, ELEMENTWISE_TPL_SCH_MODE_20,
        ELEMENTWISE_TPL_SCH_MODE_21, ELEMENTWISE_TPL_SCH_MODE_22,
        ELEMENTWISE_TPL_SCH_MODE_23, ELEMENTWISE_TPL_SCH_MODE_24,
        ELEMENTWISE_TPL_SCH_MODE_25, ELEMENTWISE_TPL_SCH_MODE_26,
        ELEMENTWISE_TPL_SCH_MODE_27, ELEMENTWISE_TPL_SCH_MODE_28,
        ELEMENTWISE_TPL_SCH_MODE_29, ELEMENTWISE_TPL_SCH_MODE_30,
        ELEMENTWISE_TPL_SCH_MODE_31, ELEMENTWISE_TPL_SCH_MODE_32,
        ELEMENTWISE_TPL_SCH_MODE_33, ELEMENTWISE_TPL_SCH_MODE_34), );

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, ELEMENTWISE_TPL_SCH_MODE_0, ELEMENTWISE_TPL_SCH_MODE_1,
            ELEMENTWISE_TPL_SCH_MODE_2, ELEMENTWISE_TPL_SCH_MODE_3, ELEMENTWISE_TPL_SCH_MODE_4,
            ELEMENTWISE_TPL_SCH_MODE_5, ELEMENTWISE_TPL_SCH_MODE_6, ELEMENTWISE_TPL_SCH_MODE_7,
            ELEMENTWISE_TPL_SCH_MODE_8, ELEMENTWISE_TPL_SCH_MODE_9, ELEMENTWISE_TPL_SCH_MODE_10,
            ELEMENTWISE_TPL_SCH_MODE_11, ELEMENTWISE_TPL_SCH_MODE_12, ELEMENTWISE_TPL_SCH_MODE_13,
            ELEMENTWISE_TPL_SCH_MODE_14, ELEMENTWISE_TPL_SCH_MODE_15, ELEMENTWISE_TPL_SCH_MODE_16,
            ELEMENTWISE_TPL_SCH_MODE_17, ELEMENTWISE_TPL_SCH_MODE_18,
            ELEMENTWISE_TPL_SCH_MODE_19, ELEMENTWISE_TPL_SCH_MODE_20,
            ELEMENTWISE_TPL_SCH_MODE_21, ELEMENTWISE_TPL_SCH_MODE_22,
            ELEMENTWISE_TPL_SCH_MODE_23, ELEMENTWISE_TPL_SCH_MODE_24,
            ELEMENTWISE_TPL_SCH_MODE_25, ELEMENTWISE_TPL_SCH_MODE_26,
            ELEMENTWISE_TPL_SCH_MODE_27, ELEMENTWISE_TPL_SCH_MODE_28,
            ELEMENTWISE_TPL_SCH_MODE_29, ELEMENTWISE_TPL_SCH_MODE_30,
            ELEMENTWISE_TPL_SCH_MODE_31, ELEMENTWISE_TPL_SCH_MODE_32,
            ELEMENTWISE_TPL_SCH_MODE_33, ELEMENTWISE_TPL_SCH_MODE_34)), );

#endif

namespace NsTranspose {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t MAXDIM = 8;

class FastTranspose2DFp16 {
public:
    __aicore__ inline FastTranspose2DFp16(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, GM_ADDR tiling)
    {
        width = AscendC::GetBlockNum() * TILE_COLS;
        const uint32_t inputElements = TILE_ROWS * width;
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t wStart = blockIdx * TILE_COLS;
        inputOffset = wStart;
        outputOffset = wStart * TILE_ROWS;
        xGm.SetGlobalBuffer((__gm__ half*)x, inputElements);
        zGm.SetGlobalBuffer((__gm__ half*)z, inputElements);
        (void)tiling;
    }

    __aicore__ inline void Process()
    {
        CopyInFirstStage();
        CopyInSecondStage();
        CopyInThirdStage();
        CopyInFourthStage();
        BuildVaConfig();
        LoadVaRowGroup<0U, false>();
        LoadVaRowGroup<1U, true>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(1));
        ComputePreloadedRowGroup<true>();
        ComputePreloadedRowGroup<false>();
        LoadVaRowGroup<2U, false>();
        LoadVaRowGroup<3U, true>();
        ComputePreloadedRowGroup<true>();
        ComputePreloadedRowGroup<false>();
        LoadVaRowGroup<4U, false>();
        LoadVaRowGroup<5U, true>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(2));
        ComputePreloadedRowGroup<true>();
        ComputePreloadedRowGroup<false>();
        LoadVaRowGroup<6U, false>();
        LoadVaRowGroup<7U, true>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(3));
        ComputePreloadedRowGroup<false>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(4));
        ComputePreloadedRowGroup<true>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(0));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(0));
        CopyOut();
    }

private:
    __aicore__ inline void BuildVaConfig()
    {
        AscendC::LocalTensor<uint64_t> dstVaConfig(
            AscendC::TPosition::VECIN, DST_VA_CONFIG_UB_OFFSET, VA_CONFIG_ELEMENTS);
        AscendC::LocalTensor<uint64_t> srcVaConfig(
            AscendC::TPosition::VECIN, SRC_VA_CONFIG_UB_OFFSET, VA_CONFIG_ELEMENTS);
        constexpr uint64_t srcBase = 0x000c000800040000ULL;
        constexpr uint64_t srcStep = 0x0010001000100010ULL;
        constexpr uint64_t dstLaneReplicate = 0x0001000100010001ULL;
        constexpr uint64_t dstLaneOffsets = 0x0018001000080000ULL;
        constexpr uint64_t dstBase =
            (OUTPUT_UB_OFFSET >> 5U) * dstLaneReplicate + dstLaneOffsets;
        constexpr uint64_t dstRowStep = 0x0001000100010001ULL;
        constexpr uint64_t dstConfigStep = 0x0020002000200020ULL;
        constexpr uint64_t srcGroupStep = srcStep << 2U;
        uint64_t src0 = srcBase;
        uint64_t src1 = srcBase + srcStep;
        uint64_t src2 = src1 + srcStep;
        uint64_t src3 = src2 + srcStep;
        uint64_t dst0 = dstBase;
        uint64_t dst1 = dstBase + dstConfigStep;
        uint64_t dst2 = dst1 + dstConfigStep;
        uint64_t dst3 = dst2 + dstConfigStep;
        uint32_t index = 0U;
        for (uint32_t group = 0U; group < VA_CONFIG_ELEMENTS / 4U; ++group) {
            srcVaConfig.SetValue(index, src0);
            srcVaConfig.SetValue(index + 1U, src1);
            srcVaConfig.SetValue(index + 2U, src2);
            srcVaConfig.SetValue(index + 3U, src3);
            dstVaConfig.SetValue(index, dst0);
            dstVaConfig.SetValue(index + 1U, dst1);
            dstVaConfig.SetValue(index + 2U, dst2);
            dstVaConfig.SetValue(index + 3U, dst3);
            src0 += srcGroupStep;
            src1 += srcGroupStep;
            src2 += srcGroupStep;
            src3 += srcGroupStep;
            dst0 += dstRowStep;
            dst1 += dstRowStep;
            dst2 += dstRowStep;
            dst3 += dstRowStep;
            index += 4U;
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(static_cast<event_t>(0));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(static_cast<event_t>(0));
    }

    __aicore__ inline void CopyInFirstStage()
    {
        AscendC::LocalTensor<half> input(
            AscendC::TPosition::VECIN, INPUT_UB_OFFSET, HALF_TILE_ELEMENTS);
        AscendC::DataCopyExtParams params{
            TILE_COLS, TILE_COLS * sizeof(half),
            static_cast<uint32_t>((width - TILE_COLS) * sizeof(half)), 0, 0};
        AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(input, xGm[inputOffset], params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(1));
    }

    __aicore__ inline void CopyInSecondStage()
    {
        AscendC::LocalTensor<half> input(
            AscendC::TPosition::VECIN, INPUT_UB_OFFSET + HALF_TILE_ELEMENTS * sizeof(half), QUARTER_TILE_ELEMENTS);
        AscendC::DataCopyExtParams params{
            QUARTER_TILE_ROWS, TILE_COLS * sizeof(half),
            static_cast<uint32_t>((width - TILE_COLS) * sizeof(half)), 0, 0};
        AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(input, xGm[inputOffset + HALF_TILE_ROWS * width], params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(2));
    }

    __aicore__ inline void CopyInThirdStage()
    {
        AscendC::LocalTensor<half> input(
            AscendC::TPosition::VECIN,
            INPUT_UB_OFFSET + THREE_QUARTER_TILE_ELEMENTS * sizeof(half),
            EIGHTH_TILE_ELEMENTS);
        AscendC::DataCopyExtParams params{
            EIGHTH_TILE_ROWS, TILE_COLS * sizeof(half),
            static_cast<uint32_t>((width - TILE_COLS) * sizeof(half)), 0, 0};
        AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(
            input,
            xGm[inputOffset + THREE_QUARTER_TILE_ROWS * width],
            params,
            pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(3));
    }

    __aicore__ inline void CopyInFourthStage()
    {
        AscendC::LocalTensor<half> input(
            AscendC::TPosition::VECIN,
            INPUT_UB_OFFSET + SEVEN_EIGHTH_TILE_ELEMENTS * sizeof(half),
            EIGHTH_TILE_ELEMENTS);
        AscendC::DataCopyExtParams params{
            EIGHTH_TILE_ROWS, TILE_COLS * sizeof(half),
            static_cast<uint32_t>((width - TILE_COLS) * sizeof(half)), 0, 0};
        AscendC::DataCopyPadExtParams<half> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(
            input,
            xGm[inputOffset + SEVEN_EIGHTH_TILE_ROWS * width],
            params,
            pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(4));
    }

    template <uint32_t rowGroup, bool alternateSet>
    __aicore__ inline void LoadVaRowGroup()
    {
        AscendC::LocalTensor<uint64_t> dstVaConfig(
            AscendC::TPosition::VECIN, DST_VA_CONFIG_UB_OFFSET, VA_CONFIG_ELEMENTS);
        AscendC::LocalTensor<uint64_t> srcVaConfig(
            AscendC::TPosition::VECIN, SRC_VA_CONFIG_UB_OFFSET, VA_CONFIG_ELEMENTS);
        constexpr uint32_t configOffset = rowGroup * 4U;
        __ubuf__ uint64_t* dst = reinterpret_cast<__ubuf__ uint64_t*>(dstVaConfig[configOffset].GetPhyAddr());
        __ubuf__ uint64_t* src = reinterpret_cast<__ubuf__ uint64_t*>(srcVaConfig[configOffset].GetPhyAddr());
        if constexpr (alternateSet) {
            vld_va_reg(VA4, dst, L128);
            vld_va_reg(VA5, dst, H128);
            vld_va_reg(VA6, src, L128);
            vld_va_reg(VA7, src, H128);
        } else {
            vld_va_reg(VA0, dst, L128);
            vld_va_reg(VA1, dst, H128);
            vld_va_reg(VA2, src, L128);
            vld_va_reg(VA3, src, H128);
        }
    }

    template <bool alternateSet>
    __aicore__ inline void ComputePreloadedRowGroup()
    {
        if constexpr (alternateSet) {
            scatter_vnchwconv_b16(VA4, VA6, 4, 128, 1);
        } else {
            scatter_vnchwconv_b16(VA0, VA2, 4, 128, 1);
        }
    }

    template <uint32_t rowGroup, bool alternateSet>
    __aicore__ inline void ComputeRowGroup()
    {
        LoadVaRowGroup<rowGroup, alternateSet>();
        ComputePreloadedRowGroup<alternateSet>();
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> output(
            AscendC::TPosition::VECOUT, OUTPUT_UB_OFFSET, TILE_ELEMENTS);
        AscendC::DataCopyExtParams params{
            TILE_COLS, TILE_ROWS * sizeof(half), 0, 0, 0};
        AscendC::DataCopyPad(zGm[outputOffset], output, params);
    }

private:
    static constexpr uint32_t TILE_ROWS = 128U;
    static constexpr uint32_t TILE_COLS = 64U;
    static constexpr uint32_t TILE_ELEMENTS = TILE_ROWS * TILE_COLS;
    static constexpr uint32_t HALF_TILE_ELEMENTS = TILE_ELEMENTS / 2U;
    static constexpr uint32_t HALF_TILE_ROWS = TILE_ROWS / 2U;
    static constexpr uint32_t QUARTER_TILE_ELEMENTS = TILE_ELEMENTS / 4U;
    static constexpr uint32_t QUARTER_TILE_ROWS = TILE_ROWS / 4U;
    static constexpr uint32_t THREE_QUARTER_TILE_ELEMENTS = 3U * QUARTER_TILE_ELEMENTS;
    static constexpr uint32_t THREE_QUARTER_TILE_ROWS = 3U * QUARTER_TILE_ROWS;
    static constexpr uint32_t EIGHTH_TILE_ELEMENTS = TILE_ELEMENTS / 8U;
    static constexpr uint32_t EIGHTH_TILE_ROWS = TILE_ROWS / 8U;
    static constexpr uint32_t SEVEN_EIGHTH_TILE_ELEMENTS = 7U * EIGHTH_TILE_ELEMENTS;
    static constexpr uint32_t SEVEN_EIGHTH_TILE_ROWS = 7U * EIGHTH_TILE_ROWS;
    static constexpr uint32_t INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t OUTPUT_UB_OFFSET = TILE_ELEMENTS * sizeof(half);
    static constexpr uint32_t VA_CONFIG_ELEMENTS = 8U * 4U;
    static constexpr uint32_t DST_VA_CONFIG_UB_OFFSET = OUTPUT_UB_OFFSET + TILE_ELEMENTS * sizeof(half);
    static constexpr uint32_t SRC_VA_CONFIG_UB_OFFSET =
        DST_VA_CONFIG_UB_OFFSET + VA_CONFIG_ELEMENTS * sizeof(uint64_t);
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> zGm;
    uint32_t width{0};
    uint32_t inputOffset{0};
    uint32_t outputOffset{0};
};

// General FP16 matrix transpose. Full 128x64 macro tiles use the tuned
// scatter_vnchwconv schedule above; only the matrix edges fall back to 16x16
// vtranspose tiles. Work is distributed by tiles instead of output elements,
// so all GM traffic is issued as two-dimensional DMA.
class TiledTranspose2DFp16 {
public:
    __aicore__ inline TiledTranspose2DFp16(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, GM_ADDR tiling, const TransposeTilingData* tilingData)
    {
        height = static_cast<uint32_t>(tilingData->shape[0]);
        width = static_cast<uint32_t>(tilingData->shape[1]);
        totalElements = static_cast<uint32_t>(tilingData->totalnumber);
        blockIdx = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();

        macroRowCount = height / MACRO_ROWS;
        macroColCount = width / MACRO_COLS;
        macroTaskCount = macroRowCount * macroColCount;
        fullRows = macroRowCount * MACRO_ROWS;
        fullCols = macroColCount * MACRO_COLS;
        allColTileCount = CeilDiv(width, MICRO_COLS);
        bottomRowTileCount = CeilDiv(height - fullRows, MICRO_ROWS);
        bottomTaskCount = bottomRowTileCount * allColTileCount;
        rightColTileCount = CeilDiv(width - fullCols, MICRO_COLS);
        rightTaskCount = (fullRows / MICRO_ROWS) * rightColTileCount;
        totalTaskCount = macroTaskCount + bottomTaskCount + rightTaskCount;

        xGm.SetGlobalBuffer((__gm__ half*)x, totalElements);
        zGm.SetGlobalBuffer((__gm__ half*)z, totalElements);
        vaConfigGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint64_t*>(tiling + VA_CONFIG_TILING_OFFSET), 2U * VA_CONFIG_ELEMENTS);
    }

    __aicore__ inline void Process()
    {
        bool vaConfigReady = false;
        if (blockIdx < macroTaskCount) {
            CopyVaConfig();
        }

        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            if (task < macroTaskCount) {
                if (!vaConfigReady) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(CONFIG_EVENT));
                    vaConfigReady = true;
                }
                const uint32_t macroRow = task / macroColCount;
                const uint32_t macroCol = task - macroRow * macroColCount;
                ProcessMacroTile(macroRow * MACRO_ROWS, macroCol * MACRO_COLS);
                continue;
            }

            uint32_t edgeTask = task - macroTaskCount;
            if (edgeTask < bottomTaskCount) {
                const uint32_t rowTile = edgeTask / allColTileCount;
                const uint32_t colTile = edgeTask - rowTile * allColTileCount;
                ProcessMicroTile(fullRows + rowTile * MICRO_ROWS, colTile * MICRO_COLS);
                continue;
            }

            edgeTask -= bottomTaskCount;
            const uint32_t rowTile = edgeTask / rightColTileCount;
            const uint32_t colTile = edgeTask - rowTile * rightColTileCount;
            ProcessMicroTile(rowTile * MICRO_ROWS, fullCols + colTile * MICRO_COLS);
        }
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) const
    {
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline void CopyVaConfig()
    {
        AscendC::LocalTensor<uint64_t> vaConfig(
            AscendC::TPosition::VECIN, DST_VA_CONFIG_UB_OFFSET, 2U * VA_CONFIG_ELEMENTS);
        AscendC::DataCopy(vaConfig, vaConfigGm, 2U * VA_CONFIG_ELEMENTS);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(CONFIG_EVENT));
    }

    __aicore__ inline void CopyInMacro(uint32_t rowStart, uint32_t colStart)
    {
        const uint32_t inputOffset = rowStart * width + colStart;
        AscendC::LocalTensor<half> input(
            AscendC::TPosition::VECIN, INPUT_UB_OFFSET, HALF_MACRO_ELEMENTS);
        AscendC::DataCopyExtParams params{
            64U, 128U, (width - MACRO_COLS) * 2U, 0U, 0U};
        AscendC::DataCopyPadExtParams<half> pad{false, 0U, 0U, 0};
        AscendC::DataCopyPad(input, xGm[inputOffset], params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(FIRST_INPUT_EVENT));

        input = AscendC::LocalTensor<half>(
            AscendC::TPosition::VECIN, INPUT_UB_OFFSET + HALF_MACRO_ELEMENTS * sizeof(half), HALF_MACRO_ELEMENTS);
        AscendC::DataCopyPad(input, xGm[inputOffset + 64U * width], params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(SECOND_INPUT_EVENT));
    }

    template <uint32_t rowGroup, bool alternateSet>
    __aicore__ inline void LoadVaRowGroup()
    {
        AscendC::LocalTensor<uint64_t> dstVaConfig(
            AscendC::TPosition::VECIN, DST_VA_CONFIG_UB_OFFSET, VA_CONFIG_ELEMENTS);
        AscendC::LocalTensor<uint64_t> srcVaConfig(
            AscendC::TPosition::VECIN, SRC_VA_CONFIG_UB_OFFSET, VA_CONFIG_ELEMENTS);
        constexpr uint32_t configOffset = rowGroup * 4U;
        __ubuf__ uint64_t* dst = reinterpret_cast<__ubuf__ uint64_t*>(dstVaConfig[configOffset].GetPhyAddr());
        __ubuf__ uint64_t* src = reinterpret_cast<__ubuf__ uint64_t*>(srcVaConfig[configOffset].GetPhyAddr());
        if constexpr (alternateSet) {
            vld_va_reg(VA4, dst, L128);
            vld_va_reg(VA5, dst, H128);
            vld_va_reg(VA6, src, L128);
            vld_va_reg(VA7, src, H128);
        } else {
            vld_va_reg(VA0, dst, L128);
            vld_va_reg(VA1, dst, H128);
            vld_va_reg(VA2, src, L128);
            vld_va_reg(VA3, src, H128);
        }
    }

    template <bool alternateSet>
    __aicore__ inline void ComputePreloadedRowGroup()
    {
        if constexpr (alternateSet) {
            scatter_vnchwconv_b16(VA4, VA6, 4, 128, 1);
        } else {
            scatter_vnchwconv_b16(VA0, VA2, 4, 128, 1);
        }
    }

    __aicore__ inline void ComputeMacro()
    {
        LoadVaRowGroup<0U, false>();
        LoadVaRowGroup<1U, true>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(FIRST_INPUT_EVENT));
        ComputePreloadedRowGroup<true>();
        ComputePreloadedRowGroup<false>();
        LoadVaRowGroup<2U, false>();
        LoadVaRowGroup<3U, true>();
        ComputePreloadedRowGroup<true>();
        ComputePreloadedRowGroup<false>();
        LoadVaRowGroup<4U, false>();
        LoadVaRowGroup<5U, true>();
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(SECOND_INPUT_EVENT));
        ComputePreloadedRowGroup<true>();
        ComputePreloadedRowGroup<false>();
        LoadVaRowGroup<6U, false>();
        LoadVaRowGroup<7U, true>();
        ComputePreloadedRowGroup<true>();
        ComputePreloadedRowGroup<false>();
    }

    __aicore__ inline void ProcessMacroTile(uint32_t rowStart, uint32_t colStart)
    {
        CopyInMacro(rowStart, colStart);
        ComputeMacro();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));

        AscendC::LocalTensor<half> output(
            AscendC::TPosition::VECOUT, OUTPUT_UB_OFFSET, MACRO_ELEMENTS);
        const uint32_t outputOffset = colStart * height + rowStart;
        AscendC::DataCopyExtParams params{
            64U, 256U, 0U, (height - MACRO_ROWS) * 2U, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
    }

    __aicore__ inline void ProcessMicroTile(uint32_t rowStart, uint32_t colStart)
    {
        const uint32_t rows = (height - rowStart < MICRO_ROWS) ? height - rowStart : MICRO_ROWS;
        const uint32_t cols = (width - colStart < MICRO_COLS) ? width - colStart : MICRO_COLS;
        const uint32_t inputOffset = rowStart * width + colStart;
        AscendC::LocalTensor<half> input(
            AscendC::TPosition::VECIN, INPUT_UB_OFFSET, MICRO_ELEMENTS);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * 2U, (width - cols) * 2U, 0U, 0U};
        AscendC::DataCopyPadExtParams<half> pad{
            true, 0U, static_cast<uint8_t>(MICRO_COLS - cols), static_cast<half>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(MICRO_INPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(MICRO_INPUT_EVENT));

        AscendC::LocalTensor<half> output(
            AscendC::TPosition::VECOUT, OUTPUT_UB_OFFSET, MICRO_ELEMENTS);
        AscendC::Transpose(output, input);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));

        const uint32_t outputOffset = colStart * height + rowStart;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * 2U, 0U, (height - rows) * 2U, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
    }

private:
    static constexpr uint32_t MACRO_ROWS = 128U;
    static constexpr uint32_t MACRO_COLS = 64U;
    static constexpr uint32_t MACRO_ELEMENTS = MACRO_ROWS * MACRO_COLS;
    static constexpr uint32_t HALF_MACRO_ELEMENTS = MACRO_ELEMENTS / 2U;
    static constexpr uint32_t MICRO_ROWS = 16U;
    static constexpr uint32_t MICRO_COLS = 16U;
    static constexpr uint32_t MICRO_ELEMENTS = MICRO_ROWS * MICRO_COLS;
    static constexpr uint32_t INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t OUTPUT_UB_OFFSET = MACRO_ELEMENTS * sizeof(half);
    static constexpr uint32_t VA_CONFIG_ELEMENTS = 8U * 4U;
    static constexpr uint32_t DST_VA_CONFIG_UB_OFFSET = OUTPUT_UB_OFFSET + MACRO_ELEMENTS * sizeof(half);
    static constexpr uint32_t SRC_VA_CONFIG_UB_OFFSET =
        DST_VA_CONFIG_UB_OFFSET + VA_CONFIG_ELEMENTS * sizeof(uint64_t);
    static constexpr uint32_t VA_CONFIG_TILING_OFFSET = 20U * sizeof(uint64_t);
    static constexpr int32_t CONFIG_EVENT = 0;
    static constexpr int32_t FIRST_INPUT_EVENT = 1;
    static constexpr int32_t SECOND_INPUT_EVENT = 2;
    static constexpr int32_t MICRO_INPUT_EVENT = 3;
    static constexpr int32_t OUTPUT_EVENT = 0;

    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> zGm;
    AscendC::GlobalTensor<uint64_t> vaConfigGm;
    uint32_t height{0};
    uint32_t width{0};
    uint32_t totalElements{0};
    uint32_t blockIdx{0};
    uint32_t blockNum{1};
    uint32_t macroRowCount{0};
    uint32_t macroColCount{0};
    uint32_t macroTaskCount{0};
    uint32_t fullRows{0};
    uint32_t fullCols{0};
    uint32_t allColTileCount{0};
    uint32_t bottomRowTileCount{0};
    uint32_t bottomTaskCount{0};
    uint32_t rightColTileCount{0};
    uint32_t rightTaskCount{0};
    uint32_t totalTaskCount{0};
};

// Matrix transpose for B8/B32 data. B32 is decomposed into two 16x8
// scatter_vnchwconv operations; B8 uses the four low/high-half combinations
// required for a 32x32 tile. The same fixed tile extents also make all edge
// DMA rows 32-byte aligned after padding.
template <typename T, uint32_t LAST_TILE_BLOCK_COUNT = 4U, bool GENERAL_OUTPUT_STRIDE = false,
          uint32_t LAST_TILE_ROW_BLOCK_COUNT = 1U, bool SHARED_OUTPUT_BUFFER = false,
          bool SINGLE_INPUT_OUTPUT_BUFFER = false,
          uint32_t INPUT_BATCH_GROUP_COUNT = 2U,
          uint32_t OUTPUT_BATCH_GROUP_COUNT = 2U,
          uint32_t RECTANGLE_PACKED_ROW_BLOCK_COUNT = 0U,
          bool SHARED_RECTANGLE_BUFFERS = false,
          bool SPLIT_INPUT_PREFETCH = false,
          bool WIDE_INPUT_ACCUMULATED_OUTPUT = false,
          bool TWO_PHASE_RECTANGLE_PACKED_OUTPUT = false,
          bool HYBRID_WIDE_PHASE = false,
          bool ADAPTIVE_TALL_TAIL = false,
          bool QUEUED_INPUT_PHASE_PIPELINE = false>
class TiledTranspose2DGeneric {
public:
    __aicore__ inline TiledTranspose2DGeneric(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const TransposeTilingData* tilingData)
    {
        height = static_cast<uint32_t>(tilingData->shape[0]);
        width = static_cast<uint32_t>(tilingData->shape[1]);
        totalElements = static_cast<uint32_t>(tilingData->totalnumber);
        batchCount = 1U;
        matrixElements = height * width;
        rowTileCount = CeilDiv(height, TILE_ROWS);
        colTileCount = CeilDiv(width, TILE_COLS);
        tasksPerBatch = rowTileCount * colTileCount;
        totalTaskCount = tasksPerBatch;
        blockIdx = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();
        xGm.SetGlobalBuffer((__gm__ T*)x, totalElements);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalElements);
    }

    __aicore__ inline void InitBatched(GM_ADDR x, GM_ADDR z, const TransposeTilingData* tilingData)
    {
        const uint32_t dims = static_cast<uint32_t>(tilingData->dims);
        uint32_t prefix = 0U;
        while (prefix < dims && tilingData->perm[prefix] == static_cast<int64_t>(prefix)) {
            ++prefix;
        }
        uint32_t end = dims;
        while (end > prefix && tilingData->perm[end - 1U] == static_cast<int64_t>(end - 1U)) {
            --end;
        }
        const uint32_t split = static_cast<uint32_t>(tilingData->perm[prefix]);

        batchCount = 1U;
        height = 1U;
        width = 1U;
        for (uint32_t i = 0U; i < prefix; ++i) {
            batchCount *= static_cast<uint32_t>(tilingData->shape[i]);
        }
        for (uint32_t i = prefix; i < split; ++i) {
            height *= static_cast<uint32_t>(tilingData->shape[i]);
        }
        for (uint32_t i = split; i < end; ++i) {
            width *= static_cast<uint32_t>(tilingData->shape[i]);
        }

        totalElements = static_cast<uint32_t>(tilingData->totalnumber);
        matrixElements = height * width;
        rowTileCount = CeilDiv(height, TILE_ROWS);
        colTileCount = CeilDiv(width, TILE_COLS);
        tasksPerBatch = rowTileCount * colTileCount;
        totalTaskCount = batchCount * tasksPerBatch;
        blockIdx = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();
        xGm.SetGlobalBuffer((__gm__ T*)x, totalElements);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalElements);
    }

    __aicore__ inline void InitLastTwoAxes(GM_ADDR x, GM_ADDR z, const TransposeTilingData* tilingData)
    {
        dims = static_cast<uint32_t>(tilingData->dims);
        for (uint32_t i = 0U; i < dims; ++i) {
            shape[i] = static_cast<uint32_t>(tilingData->shape[i]);
            perm[i] = static_cast<uint32_t>(tilingData->perm[i]);
            if constexpr (GENERAL_OUTPUT_STRIDE) {
                if (perm[i] == dims - 1U) {
                    inputLastOutputPos = i;
                }
            }
        }
        if constexpr (!GENERAL_OUTPUT_STRIDE) {
            inputLastOutputPos = dims - 2U;
        }
        inputStride[dims - 1U] = 1U;
        for (int32_t i = static_cast<int32_t>(dims) - 2; i >= 0; --i) {
            inputStride[i] = inputStride[i + 1] * shape[i + 1U];
        }
        if constexpr (GENERAL_OUTPUT_STRIDE) {
            outputStride[dims - 1U] = 1U;
            for (int32_t i = static_cast<int32_t>(dims) - 2; i >= 0; --i) {
                outputStride[i] = outputStride[i + 1] * shape[perm[i + 1U]];
            }
        }

        height = shape[perm[dims - 1U]];
        width = shape[dims - 1U];
        inputRowStride = inputStride[perm[dims - 1U]];
        if constexpr (GENERAL_OUTPUT_STRIDE) {
            outputRowStride = outputStride[inputLastOutputPos];
        } else {
            outputRowStride = height;
        }
        totalElements = static_cast<uint32_t>(tilingData->totalnumber);
        matrixElements = height * width;
        batchCount = totalElements / matrixElements;
        rowTileCount = CeilDiv(height, LAST_TILE_ROWS);
        colTileCount = CeilDiv(width, LAST_TILE_COLS);
        tasksPerBatch = rowTileCount * colTileCount;
        if constexpr (ADAPTIVE_TALL_TAIL) {
            const uint32_t fullColumnTileCount = width / LAST_TILE_COLS;
            tasksPerBatch = rowTileCount * fullColumnTileCount +
                CeilDiv(height, ADAPTIVE_TAIL_ROWS);
        }
        totalTaskCount = batchCount * tasksPerBatch;
        blockIdx = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();
        xGm.SetGlobalBuffer((__gm__ T*)x, totalElements);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalElements);
    }

    __aicore__ inline void Process()
    {
        if (totalTaskCount > blockNum) {
            ProcessTilesPipelined();
            return;
        }
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t rowTile = task / colTileCount;
            const uint32_t colTile = task - rowTile * colTileCount;
            ProcessTile(0U, rowTile * TILE_ROWS, colTile * TILE_COLS);
        }
    }

    __aicore__ inline void ProcessBatched()
    {
        if (totalTaskCount > blockNum) {
            ProcessTilesPipelined();
            return;
        }
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t batch = task / tasksPerBatch;
            const uint32_t matrixTask = task - batch * tasksPerBatch;
            const uint32_t rowTile = matrixTask / colTileCount;
            const uint32_t colTile = matrixTask - rowTile * colTileCount;
            ProcessTile(batch, rowTile * TILE_ROWS, colTile * TILE_COLS);
        }
    }

    __aicore__ inline void ProcessLastTwoAxes()
    {
        if constexpr (QUEUED_INPUT_PHASE_PIPELINE) {
            ProcessContiguousLastTwoAxesQueuedInputPhases();
            return;
        }
        if constexpr (ADAPTIVE_TALL_TAIL) {
            ProcessContiguousLastTwoAxesAdaptiveTallTail();
            return;
        }
        if constexpr (HYBRID_WIDE_PHASE) {
            ProcessContiguousLastTwoAxesHybridWidePhase();
            return;
        }
        if constexpr (WIDE_INPUT_ACCUMULATED_OUTPUT) {
            ProcessContiguousLastTwoAxesWideInputAccumulatedOutput();
            return;
        }
        if constexpr (SPLIT_INPUT_PREFETCH) {
            ProcessContiguousLastTwoAxesSplitInputPrefetch();
            return;
        }
        if constexpr (SINGLE_INPUT_OUTPUT_BUFFER) {
            ProcessContiguousLastTwoAxesSingleInputOutput();
            return;
        }
        if constexpr (!GENERAL_OUTPUT_STRIDE && TYPE_BYTES == 4U &&
                      LAST_TILE_ROW_BLOCKS == 8U && LAST_TILE_BLOCKS == 8U) {
            ProcessContiguousLastTwoAxesOneInputTwoOutputs();
            return;
        }
        if constexpr (!GENERAL_OUTPUT_STRIDE && LAST_TILE_BLOCKS <= 16U) {
            ProcessContiguousLastTwoAxesPipelined();
            return;
        }
        if constexpr (GENERAL_OUTPUT_STRIDE) {
            if (batchCount >= blockNum) {
                if constexpr (TYPE_BYTES == 4U && LAST_TILE_ROW_BLOCKS == 2U &&
                              LAST_TILE_BLOCKS == 2U) {
                    if (tasksPerBatch == 1U) {
                        ProcessGeneralSingleTilePrefetched();
                        return;
                    }
                }
                ProcessGeneralLastTwoAxesPipelined();
                return;
            }
        }
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t batch = task / tasksPerBatch;
            const uint32_t matrixTask = task - batch * tasksPerBatch;
            const uint32_t rowTile = matrixTask / colTileCount;
            const uint32_t colTile = matrixTask - rowTile * colTileCount;
            const uint32_t inputBatchOffset = InputBatchOffset(batch);
            uint32_t outputBatchOffset = batch * matrixElements;
            if constexpr (GENERAL_OUTPUT_STRIDE) {
                outputBatchOffset = OutputBatchOffset(batch);
            }
            ProcessLastTwoAxesTile(inputBatchOffset, outputBatchOffset,
                                   rowTile * LAST_TILE_ROWS, colTile * LAST_TILE_COLS);
        }
    }

    // Merge two matrices that are physically adjacent along the fastest
    // remaining input-batch axis.  Their rows form one wider dense matrix in
    // GM, so one 2-D load and one wider VNCHW sequence replace two independent
    // loads without assuming any particular rank or shape.
    __aicore__ inline void ProcessAdjacentInputBatchPairs()
    {
        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        InitAdjacentInputBatchLayout();
        const uint32_t groupsPerBlock = pairGroupCount / blockNum;
        const uint32_t extraBlocks = pairGroupCount - groupsPerBlock * blockNum;
        const uint32_t groupBegin = blockIdx * groupsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        const uint32_t groupEnd = groupBegin + groupsPerBlock +
            (blockIdx < extraBlocks ? 1U : 0U);

        if (groupBegin < groupEnd) {
            uint32_t currentInputOffset = 0U;
            uint32_t currentOutputOffset = 0U;
            uint32_t currentPairCount = 0U;
            DecodePairGroup(groupBegin, currentInputOffset,
                            currentOutputOffset, currentPairCount);
            LoadAdjacentInputBatchPair(
                0U, currentInputOffset, currentPairCount);

            for (uint32_t group = groupBegin; group < groupEnd; ++group) {
                const uint32_t buffer = (group - groupBegin) & 1U;
                uint32_t nextInputOffset = 0U;
                uint32_t nextOutputOffset = 0U;
                uint32_t nextPairCount = 0U;
                if (group + 1U < groupEnd) {
                    DecodePairGroup(group + 1U, nextInputOffset,
                                    nextOutputOffset, nextPairCount);
                    LoadAdjacentInputBatchPair(
                        buffer ^ 1U, nextInputOffset, nextPairCount);
                }
                ComputeStoreAdjacentInputBatchPair(
                    buffer, currentOutputOffset, currentPairCount);
                currentInputOffset = nextInputOffset;
                currentOutputOffset = nextOutputOffset;
                currentPairCount = nextPairCount;
            }
        }

        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    // Group one physically adjacent input-batch axis and one physically
    // adjacent output-batch axis.  A rectangle replaces per-matrix transfers
    // with one wide input transfer per output group member and one wide output
    // transfer per input group member.
    __aicore__ inline void ProcessAdjacentBatchRectangles()
    {
        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            if constexpr (TWO_PHASE_RECTANGLE_PACKED_OUTPUT) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            }
            if constexpr (!SHARED_RECTANGLE_BUFFERS) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            }
        }

        InitAdjacentBatchRectangleLayout();
        const uint32_t groupsPerBlock = rectangleGroupCount / blockNum;
        const uint32_t extraBlocks =
            rectangleGroupCount - groupsPerBlock * blockNum;
        const uint32_t groupBegin = blockIdx * groupsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        const uint32_t groupEnd = groupBegin + groupsPerBlock +
            (blockIdx < extraBlocks ? 1U : 0U);
        uint32_t currentInputOffset = 0U;
        uint32_t currentOutputOffset = 0U;
        uint32_t currentInputPairCount = 0U;
        uint32_t currentOutputPairCount = 0U;

        if (groupBegin < groupEnd) {
            DecodeRectangleGroup(
                groupBegin, currentInputOffset, currentOutputOffset,
                currentInputPairCount, currentOutputPairCount);
            LoadAdjacentBatchRectangle(
                0U, currentInputOffset,
                currentInputPairCount, currentOutputPairCount);
        }

        // The first input generation and the per-core Gather offset table
        // occupy disjoint UB intervals and have no producer/consumer edge.
        // Submit MTE2 first so its fill overlaps the one-time Scalar/Vector
        // table synthesis instead of extending the task's startup prefix.
        BuildRectangleGatherOffsets();
        if constexpr (SHARED_RECTANGLE_BUFFERS &&
                      !TWO_PHASE_RECTANGLE_PACKED_OUTPUT) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(SHARED_RECTANGLE_EVENT));
        }

        if (groupBegin < groupEnd) {
            for (uint32_t group = groupBegin; group < groupEnd; ++group) {
                const uint32_t buffer = (group - groupBegin) & 1U;
                uint32_t nextInputOffset = 0U;
                uint32_t nextOutputOffset = 0U;
                uint32_t nextInputPairCount = 0U;
                uint32_t nextOutputPairCount = 0U;
                if (group + 1U < groupEnd) {
                    DecodeRectangleGroup(
                        group + 1U, nextInputOffset, nextOutputOffset,
                        nextInputPairCount, nextOutputPairCount);
                    LoadAdjacentBatchRectangle(
                        buffer ^ 1U, nextInputOffset,
                        nextInputPairCount, nextOutputPairCount);
                }
                ComputeStoreAdjacentBatchRectangle(
                    buffer, currentOutputOffset,
                    currentInputPairCount, currentOutputPairCount);
                currentInputOffset = nextInputOffset;
                currentOutputOffset = nextOutputOffset;
                currentInputPairCount = nextInputPairCount;
                currentOutputPairCount = nextOutputPairCount;
            }
        }

        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            if constexpr (TWO_PHASE_RECTANGLE_PACKED_OUTPUT) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            }
            if constexpr (!SHARED_RECTANGLE_BUFFERS) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            }
        }
        if constexpr (SHARED_RECTANGLE_BUFFERS &&
                      !TWO_PHASE_RECTANGLE_PACKED_OUTPUT) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(SHARED_RECTANGLE_EVENT));
        }
    }

    __aicore__ inline void ProcessSixOutputRectangles()
    {
        if (width * INPUT_BATCH_GROUP_COUNT <=
            COMBINED_SIX_INPUT_ROW_STRIDE) {
            ProcessCombinedSixOutputRectangles();
            return;
        }
        ProcessStreamedSixOutputRectangles();
    }

    // Keep the complete compact 7-by-6 source generation, but treat
    // 50- and 78-row output banks as one contiguous 128-row VNCHW destination
    // for the head.  The bank boundary coincides with two complete input
    // coordinates on the target geometry, reducing split MTE3 descriptors.
    // After bank zero drains, the residual at-most-48-row phase reuses it
    // while MTE3 continues reading bank one.
    __aicore__ inline void ProcessSevenInputSixOutputRectanglesPaired()
    {
        static_assert(TYPE_BYTES == 4U);
        static_assert(INPUT_BATCH_GROUP_COUNT == 7U);
        static_assert(OUTPUT_BATCH_GROUP_COUNT == 6U);
        static_assert(SEVEN_BY_SIX_UB_END <= UB_CAPACITY_BYTES);

        constexpr event_t inputEvent = static_cast<event_t>(0);
        constexpr event_t outputEvent0 = static_cast<event_t>(1);
        constexpr event_t outputEvent1 = static_cast<event_t>(2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent1);

        InitAdjacentBatchRectangleLayout();
        const uint32_t groupsPerBlock = rectangleGroupCount / blockNum;
        const uint32_t extraBlocks =
            rectangleGroupCount - groupsPerBlock * blockNum;
        const uint32_t groupBegin = blockIdx * groupsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        const uint32_t groupEnd = groupBegin + groupsPerBlock +
            (blockIdx < extraBlocks ? 1U : 0U);

        if (groupBegin < groupEnd) {
            uint32_t currentInputOffset = 0U;
            uint32_t currentOutputOffset = 0U;
            uint32_t currentInputPairCount = 0U;
            uint32_t currentOutputPairCount = 0U;
            DecodeRectangleGroup(
                groupBegin, currentInputOffset, currentOutputOffset,
                currentInputPairCount, currentOutputPairCount);
            LoadSevenInputSixOutputRectangleFull(currentInputOffset);

            for (uint32_t group = groupBegin; group < groupEnd; ++group) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent1);

                uint32_t nextInputOffset = currentInputOffset;
                uint32_t nextOutputOffset = currentOutputOffset;
                uint32_t nextInputPairCount = currentInputPairCount;
                uint32_t nextOutputPairCount = currentOutputPairCount;
                const bool hasNext = group + 1U < groupEnd;
                if (hasNext) {
                    DecodeRectangleGroup(
                        group + 1U, nextInputOffset, nextOutputOffset,
                        nextInputPairCount, nextOutputPairCount);
                }

                ComputeSevenInputSixOutputPairedHead();
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent0);
                StoreSevenInputSixOutputPhaseFull<
                    0U, 0U, SEVEN_BY_SIX_OUTPUT_BANK0_ROWS>(
                        currentOutputOffset);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent1);
                StoreSevenInputSixOutputPhaseFull<
                    1U, SEVEN_BY_SIX_OUTPUT_BANK0_ROWS,
                    SEVEN_BY_SIX_OUTPUT_BANK1_ROWS>(currentOutputOffset);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent1);

                const bool hasTail =
                    INPUT_BATCH_GROUP_COUNT * width >
                        SEVEN_BY_SIX_OUTPUT_HEAD_ROWS;
                if (hasTail) {
                    // Bank zero ends at an input-coordinate boundary for the
                    // target geometry.  Reuse it for the residual phase as
                    // soon as MTE3 releases it; bank one's store remains in
                    // flight over the disjoint compact interval.
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent0);
                    ComputeSevenInputSixOutputTail();
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
                if (hasNext) {
                    LoadSevenInputSixOutputRectangleFull(nextInputOffset);
                }

                if (hasTail) {
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent0);
                    StoreSevenInputSixOutputPhaseFull<
                        0U, SEVEN_BY_SIX_OUTPUT_HEAD_ROWS,
                        SEVEN_BY_SIX_OUTPUT_TAIL_ROWS>(currentOutputOffset);
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent0);
                }

                currentInputOffset = nextInputOffset;
                currentOutputOffset = nextOutputOffset;
                currentInputPairCount = nextInputPairCount;
                currentOutputPairCount = nextOutputPairCount;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent1);
    }

    // When all six adjacent input coordinates fit in the 152-element compact
    // row capacity, concatenate the six output coordinates along the row
    // axis and transpose one logical rectangle.  This removes the padded
    // 32-row gap between coordinates, so the result is already laid out as
    // six compact height-element sections and needs no Gather pass.
    __aicore__ inline void ProcessCombinedSixOutputRectangles()
    {
        static_assert(TYPE_BYTES == 4U);
        static_assert(INPUT_BATCH_GROUP_COUNT == 6U);
        static_assert(OUTPUT_BATCH_GROUP_COUNT == 6U);
        static_assert(COMBINED_SIX_OUTPUT_UB_END <= UB_CAPACITY_BYTES);

        constexpr event_t inputEvent = static_cast<event_t>(0);
        constexpr event_t outputEvent = static_cast<event_t>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

        InitAdjacentBatchRectangleLayout();
        const uint32_t groupsPerBlock = rectangleGroupCount / blockNum;
        const uint32_t extraBlocks =
            rectangleGroupCount - groupsPerBlock * blockNum;
        const uint32_t groupBegin = blockIdx * groupsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        const uint32_t groupEnd = groupBegin + groupsPerBlock +
            (blockIdx < extraBlocks ? 1U : 0U);

        if (groupBegin < groupEnd) {
            uint32_t currentInputOffset = 0U;
            uint32_t currentOutputOffset = 0U;
            uint32_t currentInputPairCount = 0U;
            uint32_t currentOutputPairCount = 0U;
            DecodeRectangleGroup(
                groupBegin, currentInputOffset, currentOutputOffset,
                currentInputPairCount, currentOutputPairCount);
            LoadCombinedSixOutputRectangle(
                currentInputOffset, currentInputPairCount,
                currentOutputPairCount);

            for (uint32_t group = groupBegin; group < groupEnd; ++group) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEvent);
                ComputeCombinedSixOutputRectangle(
                    currentInputPairCount, currentOutputPairCount);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);

                uint32_t nextInputOffset = currentInputOffset;
                uint32_t nextOutputOffset = currentOutputOffset;
                uint32_t nextInputPairCount = currentInputPairCount;
                uint32_t nextOutputPairCount = currentOutputPairCount;
                if (group + 1U < groupEnd) {
                    DecodeRectangleGroup(
                        group + 1U, nextInputOffset, nextOutputOffset,
                        nextInputPairCount, nextOutputPairCount);
                    LoadCombinedSixOutputRectangle(
                        nextInputOffset, nextInputPairCount,
                        nextOutputPairCount);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
                StoreCombinedSixOutputRectangle(
                    currentOutputOffset, currentInputPairCount,
                    currentOutputPairCount);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

                currentInputOffset = nextInputOffset;
                currentOutputOffset = nextOutputOffset;
                currentInputPairCount = nextInputPairCount;
                currentOutputPairCount = nextOutputPairCount;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    // Accumulate all six physically adjacent output coordinates in one
    // intermediate.  One output coordinate is loaded into each of two small
    // alternating input banks, preserving the retained 600-byte MTE2 burst
    // while avoiding the 120-KiB generations a conventional 6x6 rectangle
    // would require.  Each compact 600-byte output row is then streamed
    // through the consumed input banks, halving the short-row MTE3 work of
    // two independent 6x3 rectangles.
    __aicore__ inline void ProcessStreamedSixOutputRectangles()
    {
        static_assert(TYPE_BYTES == 4U);
        static_assert(INPUT_BATCH_GROUP_COUNT == 6U);
        static_assert(OUTPUT_BATCH_GROUP_COUNT == 6U);
        static_assert(SIX_OUTPUT_OFFSETS_UB_OFFSET +
                          SIX_OUTPUT_OFFSET_ELEMENTS * sizeof(uint32_t) <=
                      UB_CAPACITY_BYTES);

        for (uint32_t bank = 0U; bank < PIPELINE_BUFFER_COUNT; ++bank) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(bank));
        }

        InitAdjacentBatchRectangleLayout();
        const uint32_t groupsPerBlock = rectangleGroupCount / blockNum;
        const uint32_t extraBlocks =
            rectangleGroupCount - groupsPerBlock * blockNum;
        const uint32_t groupBegin = blockIdx * groupsPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        const uint32_t groupEnd = groupBegin + groupsPerBlock +
            (blockIdx < extraBlocks ? 1U : 0U);

        if (groupBegin < groupEnd) {
            uint32_t currentInputOffset = 0U;
            uint32_t currentOutputOffset = 0U;
            uint32_t currentInputPairCount = 0U;
            uint32_t currentOutputPairCount = 0U;
            DecodeRectangleGroup(
                groupBegin, currentInputOffset, currentOutputOffset,
                currentInputPairCount, currentOutputPairCount);

            // Bank zero can start filling while the per-core Gather table is
            // synthesized in a disjoint UB interval.
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(0));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                static_cast<event_t>(0));
            LoadStreamedSixOutputPair(
                0U, currentInputOffset, currentInputPairCount, 0U);
            BuildStreamedSixOutputOffsets();

            for (uint32_t group = groupBegin; group < groupEnd; ++group) {
                ComputeStreamedSixOutputRectangle(
                    currentInputOffset, currentInputPairCount,
                    currentOutputPairCount);
                StoreStreamedSixOutputRectangle(
                    currentOutputOffset, currentInputPairCount,
                    currentOutputPairCount);

                if (group + 1U < groupEnd) {
                    DecodeRectangleGroup(
                        group + 1U, currentInputOffset,
                        currentOutputOffset, currentInputPairCount,
                        currentOutputPairCount);
                    // The penultimate packed row uses bank zero.  Refill it
                    // for the next rectangle while MTE3 is still draining
                    // the final row from bank one.
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                        static_cast<event_t>(0));
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                        static_cast<event_t>(0));
                    LoadStreamedSixOutputPair(
                        0U, currentInputOffset,
                        currentInputPairCount, 0U);
                }
            }
        }

        for (uint32_t bank = 0U; bank < PIPELINE_BUFFER_COUNT; ++bank) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(bank));
        }
    }

    __aicore__ inline void ProcessContiguousLastTwoAxesPipelined()
    {
        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            if constexpr (!SHARED_OUTPUT_BUFFER) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            }
        }
        if constexpr (SHARED_OUTPUT_BUFFER) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(SHARED_OUTPUT_EVENT));
        }

        uint32_t progress = 0U;
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t batch = task / tasksPerBatch;
            const uint32_t matrixTask = task - batch * tasksPerBatch;
            const uint32_t rowTile = matrixTask / colTileCount;
            const uint32_t colTile = matrixTask - rowTile * colTileCount;
            ProcessLastTwoAxesTilePipelined(
                progress & 1U, InputBatchOffset(batch), batch * matrixElements,
                rowTile * LAST_TILE_ROWS, colTile * LAST_TILE_COLS);
            ++progress;
        }

        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            if constexpr (!SHARED_OUTPUT_BUFFER) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            }
        }
        if constexpr (SHARED_OUTPUT_BUFFER) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(SHARED_OUTPUT_EVENT));
        }
    }

    __aicore__ inline void ProcessContiguousLastTwoAxesSingleInputOutput()
    {
        constexpr event_t inputEvent = static_cast<event_t>(0);
        constexpr event_t outputEvent = static_cast<event_t>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t batch = task / tasksPerBatch;
            const uint32_t matrixTask = task - batch * tasksPerBatch;
            const uint32_t rowTile = matrixTask / colTileCount;
            const uint32_t colTile = matrixTask - rowTile * colTileCount;
            ProcessLastTwoAxesTileSingleInputOutput(
                InputBatchOffset(batch), batch * matrixElements,
                rowTile * LAST_TILE_ROWS, colTile * LAST_TILE_COLS);
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    // Split a single 96 KiB input bank into two independent row generations.
    // Once Vector has consumed the first half of the current tile, MTE2 may
    // prefetch the first half of the next tile while Vector consumes the
    // disjoint second half.  The tile geometry, GM bytes, task order, and
    // single output generation remain unchanged.
    __aicore__ inline void ProcessContiguousLastTwoAxesSplitInputPrefetch()
    {
        constexpr event_t firstInputEvent = static_cast<event_t>(0);
        constexpr event_t outputEvent = static_cast<event_t>(1);
        constexpr event_t secondInputEvent = static_cast<event_t>(2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(firstInputEvent);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(secondInputEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

        if (blockIdx < totalTaskCount) {
            uint32_t currentInputBatchOffset = 0U;
            uint32_t currentOutputBatchOffset = 0U;
            uint32_t currentRowStart = 0U;
            uint32_t currentColStart = 0U;
            DecodeContiguousLastTwoTask(
                blockIdx, currentInputBatchOffset, currentOutputBatchOffset,
                currentRowStart, currentColStart);
            LoadLastTwoAxesInputHalf(
                firstInputEvent, currentInputBatchOffset,
                currentRowStart, currentColStart, 0U);
            LoadLastTwoAxesInputHalf(
                secondInputEvent, currentInputBatchOffset,
                currentRowStart, currentColStart, 1U);

            for (uint32_t task = blockIdx; task < totalTaskCount;
                 task += blockNum) {
                const uint32_t nextTask = task + blockNum;
                const bool hasNext = nextTask < totalTaskCount;
                uint32_t nextInputBatchOffset = 0U;
                uint32_t nextOutputBatchOffset = 0U;
                uint32_t nextRowStart = 0U;
                uint32_t nextColStart = 0U;
                if (hasNext) {
                    DecodeContiguousLastTwoTask(
                        nextTask, nextInputBatchOffset, nextOutputBatchOffset,
                        nextRowStart, nextColStart);
                }

                AscendC::LocalTensor<T> input(
                    AscendC::TPosition::VECIN,
                    INPUT_UB_OFFSET, LAST_TILE_ELEMENTS);
                AscendC::LocalTensor<T> output(
                    AscendC::TPosition::VECOUT,
                    LAST_OUTPUT_UB_OFFSET, LAST_TILE_ELEMENTS);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                    firstInputEvent);
                ComputeLastTwoB32Half(output, input, 0U);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    firstInputEvent);
                if (hasNext) {
                    LoadLastTwoAxesInputHalf(
                        firstInputEvent, nextInputBatchOffset,
                        nextRowStart, nextColStart, 0U);
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                    secondInputEvent);
                ComputeLastTwoB32Half(
                    output, input, LAST_TILE_ROW_BLOCKS / 2U);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    secondInputEvent);
                if (hasNext) {
                    LoadLastTwoAxesInputHalf(
                        secondInputEvent, nextInputBatchOffset,
                        nextRowStart, nextColStart, 1U);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
                StoreLastTwoAxesOutput(
                    output, currentOutputBatchOffset,
                    currentRowStart, currentColStart);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

                currentInputBatchOffset = nextInputBatchOffset;
                currentOutputBatchOffset = nextOutputBatchOffset;
                currentRowStart = nextRowStart;
                currentColStart = nextColStart;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(firstInputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(secondInputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    // A 384-by-112 output consumes 168 KiB and leaves only 24 KiB.  Reuse the
    // unwritten third 112-column output slice for the first two 112-row input
    // phases.  Once those phases have populated their disjoint destinations,
    // the slice becomes output-only; stage the remaining rows through the
    // 48-row compact bank.  This retains long output rows and wide input DMA
    // while avoiding an in-place square transpose.
    __aicore__ inline void ProcessContiguousLastTwoAxesHybridWidePhase()
    {
        constexpr event_t compactInputEvent = static_cast<event_t>(0);
        constexpr event_t scratchInputEvent = static_cast<event_t>(2);
        constexpr event_t outputEvent = static_cast<event_t>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(compactInputEvent);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(scratchInputEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(outputEvent);

        const uint32_t firstTask = static_cast<uint32_t>(
            static_cast<uint64_t>(totalTaskCount) * blockIdx / blockNum);
        const uint32_t taskLimit = static_cast<uint32_t>(
            static_cast<uint64_t>(totalTaskCount) * (blockIdx + 1U) /
            blockNum);
        if (firstTask < taskLimit) {
            uint32_t currentInputBatchOffset = 0U;
            uint32_t currentOutputBatchOffset = 0U;
            uint32_t currentRowStart = 0U;
            uint32_t currentColStart = 0U;
            DecodeContiguousLastTwoTask(
                firstTask, currentInputBatchOffset, currentOutputBatchOffset,
                currentRowStart, currentColStart);
            LoadLastTwoAxesHybridCompactPhase(
                compactInputEvent, currentInputBatchOffset,
                currentRowStart, currentColStart,
                HYBRID_COMPACT_FIRST_ROW_BLOCK);

            for (uint32_t task = firstTask; task < taskLimit; ++task) {
                const uint32_t nextTask = task + 1U;
                const bool hasNext = nextTask < taskLimit;
                uint32_t nextInputBatchOffset = 0U;
                uint32_t nextOutputBatchOffset = 0U;
                uint32_t nextRowStart = 0U;
                uint32_t nextColStart = 0U;
                if (hasNext) {
                    DecodeContiguousLastTwoTask(
                        nextTask, nextInputBatchOffset,
                        nextOutputBatchOffset, nextRowStart, nextColStart);
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
                AscendC::LocalTensor<T> output(
                    AscendC::TPosition::VECOUT,
                    WIDE_ACCUMULATED_OUTPUT_UB_OFFSET,
                    LAST_TILE_ELEMENTS);
                AscendC::LocalTensor<T> compactInput(
                    AscendC::TPosition::VECIN,
                    WIDE_PHASE_INPUT_UB_OFFSET,
                    WIDE_PHASE_INPUT_ELEMENTS);
                const uint32_t rows =
                    height - currentRowStart < LAST_TILE_ROWS ?
                        height - currentRowStart : LAST_TILE_ROWS;
                const uint32_t cols =
                    width - currentColStart < LAST_TILE_COLS ?
                        width - currentColStart : LAST_TILE_COLS;
                const uint32_t activeRowBlocks = CeilDiv(rows, TILE_ROWS);
                const uint32_t activeColumnBlocks = CeilDiv(cols, TILE_COLS);
                // The first compact phase was prefetched while the preceding
                // output generation was active.  Start scratch phase zero
                // only after MTE3 has released the aliased output slice.
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                    outputEvent);
                LoadLastTwoAxesInplaceScratchPhase(
                    scratchInputEvent, currentInputBatchOffset,
                    currentRowStart, currentColStart, 0U);

                for (uint32_t phase = 0U; phase < 2U; ++phase) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                        scratchInputEvent);
                    AscendC::LocalTensor<T> scratchInput(
                        AscendC::TPosition::VECIN,
                        INPLACE_SCRATCH_UB_OFFSET,
                        LAST_TILE_ELEMENTS - INPLACE_SCRATCH_UB_OFFSET /
                            TYPE_BYTES);
                    ComputeLastTwoB32RowPhase(
                        output, scratchInput,
                        phase * INPLACE_PHASE_ROW_BLOCKS,
                        INPLACE_PHASE_ROW_BLOCKS, activeColumnBlocks,
                        LAST_TILE_ROWS);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                        scratchInputEvent);
                    if (phase + 1U < 2U) {
                        LoadLastTwoAxesInplaceScratchPhase(
                            scratchInputEvent, currentInputBatchOffset,
                            currentRowStart, currentColStart, phase + 1U);
                    }
                }

                uint32_t compactFirstRowBlock =
                    HYBRID_COMPACT_FIRST_ROW_BLOCK;
                while (compactFirstRowBlock < activeRowBlocks) {
                    const uint32_t remainingRowBlocks =
                        activeRowBlocks - compactFirstRowBlock;
                    const uint32_t phaseRowBlocks =
                        remainingRowBlocks < WIDE_PHASE_ROW_BLOCKS ?
                            remainingRowBlocks : WIDE_PHASE_ROW_BLOCKS;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                        compactInputEvent);
                    ComputeLastTwoB32RowPhase(
                        output, compactInput, compactFirstRowBlock,
                        phaseRowBlocks, activeColumnBlocks, LAST_TILE_COLS);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                        compactInputEvent);
                    compactFirstRowBlock += phaseRowBlocks;
                    if (compactFirstRowBlock < activeRowBlocks) {
                        LoadLastTwoAxesHybridCompactPhase(
                            compactInputEvent, currentInputBatchOffset,
                            currentRowStart, currentColStart,
                            compactFirstRowBlock);
                    }
                }

                if (hasNext) {
                    LoadLastTwoAxesHybridCompactPhase(
                        compactInputEvent, nextInputBatchOffset,
                        nextRowStart, nextColStart,
                        HYBRID_COMPACT_FIRST_ROW_BLOCK);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
                StoreLastTwoAxesOutput(
                    output, currentOutputBatchOffset,
                    currentRowStart, currentColStart);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(outputEvent);

                currentInputBatchOffset = nextInputBatchOffset;
                currentOutputBatchOffset = nextOutputBatchOffset;
                currentRowStart = nextRowStart;
                currentColStart = nextColStart;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(compactInputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(scratchInputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(outputEvent);
    }

    // Keep the complete 80-by-384 transposed tile in a 120 KiB output bank.
    // The remaining 70 KiB input bank stages 224 and 160 source rows in turn.
    // Each source row is still loaded at the full 80-column width, while all
    // five output column blocks are published by one long-row MTE3 transfer.
    __aicore__ inline void ProcessContiguousLastTwoAxesWideInputAccumulatedOutput()
    {
        constexpr event_t inputEvent = static_cast<event_t>(0);
        constexpr event_t outputEvent = static_cast<event_t>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

        if (blockIdx < totalTaskCount) {
            uint32_t currentInputBatchOffset = 0U;
            uint32_t currentOutputBatchOffset = 0U;
            uint32_t currentRowStart = 0U;
            uint32_t currentColStart = 0U;
            DecodeContiguousLastTwoTask(
                blockIdx, currentInputBatchOffset, currentOutputBatchOffset,
                currentRowStart, currentColStart);
            LoadLastTwoAxesWideInputRowPhase(
                inputEvent, currentInputBatchOffset,
                currentRowStart, currentColStart, 0U);

            for (uint32_t task = blockIdx; task < totalTaskCount;
                 task += blockNum) {
                const uint32_t nextTask = task + blockNum;
                const bool hasNext = nextTask < totalTaskCount;
                uint32_t nextInputBatchOffset = 0U;
                uint32_t nextOutputBatchOffset = 0U;
                uint32_t nextRowStart = 0U;
                uint32_t nextColStart = 0U;
                if (hasNext) {
                    DecodeContiguousLastTwoTask(
                        nextTask, nextInputBatchOffset, nextOutputBatchOffset,
                        nextRowStart, nextColStart);
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
                AscendC::LocalTensor<T> input(
                    AscendC::TPosition::VECIN,
                    WIDE_PHASE_INPUT_UB_OFFSET,
                    WIDE_PHASE_INPUT_ELEMENTS);
                AscendC::LocalTensor<T> output(
                    AscendC::TPosition::VECOUT,
                    WIDE_ACCUMULATED_OUTPUT_UB_OFFSET,
                    LAST_TILE_ELEMENTS);
                const uint32_t rows =
                    height - currentRowStart < LAST_TILE_ROWS ?
                        height - currentRowStart : LAST_TILE_ROWS;
                const uint32_t cols =
                    width - currentColStart < LAST_TILE_COLS ?
                        width - currentColStart : LAST_TILE_COLS;
                const uint32_t activeRowBlocks =
                    CeilDiv(rows, TILE_ROWS);
                const uint32_t activeColumnBlocks =
                    CeilDiv(cols, TILE_COLS);

                const uint32_t activePhaseCount =
                    CeilDiv(activeRowBlocks, WIDE_PHASE_ROW_BLOCKS);
                for (uint32_t phase = 0U;
                     phase < activePhaseCount; ++phase) {
                    if (phase > 0U) {
                        LoadLastTwoAxesWideInputRowPhase(
                            inputEvent, currentInputBatchOffset,
                            currentRowStart, currentColStart, phase);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEvent);
                    const uint32_t firstRowBlock =
                        phase * WIDE_PHASE_ROW_BLOCKS;
                    const uint32_t remainingRowBlocks =
                        activeRowBlocks - firstRowBlock;
                    const uint32_t phaseRowBlocks =
                        remainingRowBlocks < WIDE_PHASE_ROW_BLOCKS ?
                            remainingRowBlocks : WIDE_PHASE_ROW_BLOCKS;
                    ComputeLastTwoB32RowPhase(
                        output, input, firstRowBlock,
                        phaseRowBlocks, activeColumnBlocks,
                        LAST_TILE_COLS);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
                }

                if (hasNext) {
                    LoadLastTwoAxesWideInputRowPhase(
                        inputEvent, nextInputBatchOffset,
                        nextRowStart, nextColStart, 0U);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
                StoreLastTwoAxesOutput(
                    output, currentOutputBatchOffset,
                    currentRowStart, currentColStart);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

                currentInputBatchOffset = nextInputBatchOffset;
                currentOutputBatchOffset = nextOutputBatchOffset;
                currentRowStart = nextRowStart;
                currentColStart = nextColStart;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    __aicore__ inline uint32_t PackedFilterMaskWord(
        uint32_t begin, uint32_t end, uint32_t word)
    {
        const uint32_t base = word * 32U;
        const uint32_t first = begin > base ? begin - base : 0U;
        const uint32_t last = end > base ? end - base : 0U;
        const uint32_t low = first < 32U ? first : 32U;
        const uint32_t high = last < 32U ? last : 32U;
        const uint32_t prefix = low == 32U ? 0xffffffffU : ((1U << low) - 1U);
        const uint32_t suffix = high == 32U ? 0xffffffffU : ((1U << high) - 1U);
        return suffix & ~prefix;
    }

    __aicore__ inline void BuildPackedRowFilterPatterns(uint32_t paddedWidth)
    {
        AscendC::LocalTensor<uint32_t> patterns(
            AscendC::TPosition::VECCALC, 110U * 1024U, 192U);
        // Previous output-bank MTE3 drains precede this Vector producer.
        AscendC::Duplicate(patterns, 0xffffffffU, 192U);
        // Protect both the old output generation and these new all-one bits.
        AscendC::SetFlag<AscendC::HardEvent::V_S>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(2);
        const uint32_t drift = width & 7U;
        const uint32_t countWords = CeilDiv(paddedWidth + 7U * drift, 32U);
        for (uint32_t row = 0U; row < 8U; ++row) {
            const uint32_t begin = row * drift;
            const uint32_t end = begin + paddedWidth;
            for (uint32_t word = 0U; word <= begin / 32U; ++word) {
                patterns.SetValue(row * 24U + word,
                    PackedFilterMaskWord(begin, end, word));
            }
            for (uint32_t word = end / 32U; word < countWords; ++word) {
                patterns.SetValue(row * 24U + word,
                    PackedFilterMaskWord(begin, end, word));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(2);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(2);
    }

    __aicore__ inline void FilterPackedRowHalf(
        const AscendC::LocalTensor<T>& packed,
        const AscendC::LocalTensor<T>& regular,
        uint32_t firstRow, uint32_t liveRows, uint32_t paddedWidth)
    {
        const auto packedBits = packed.template ReinterpretCast<uint32_t>();
        const auto regularBits = regular.template ReinterpretCast<uint32_t>();
        AscendC::LocalTensor<uint32_t> patterns(
            AscendC::TPosition::VECCALC, 110U * 1024U, 192U);
        AscendC::SetMaskCount();
        set_vector_mask(0, paddedWidth + 7U * (width & 7U));
        for (uint32_t row = 0U; row < liveRows; row += 8U) {
            const uint32_t source = (firstRow + row) * width;
            const uint32_t remaining = liveRows - row;
            // Each repeat uses its own drift mask and emits one padded row.
            AscendC::GatherMaskImpl<uint32_t>(
                reinterpret_cast<__ubuf__ uint32_t*>(
                    regularBits[row * paddedWidth].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(
                    packedBits[source].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(
                    patterns.GetPhyAddr()), 0U,
                AscendC::GatherMaskParams{1U,
                    static_cast<uint16_t>(remaining < 8U ? remaining : 8U),
                    static_cast<uint16_t>(width / 8U), 3U});
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

    // Read complete source rows, stage transposed 32-row records contiguously
    // in AIV-private GM, then assemble long destination rows using aligned
    // DMA into two UB banks. No inter-core publication or global barrier.
    __aicore__ inline void ProcessFullRowWorkspace(GM_ADDR workspace)
    {
        static_assert(TYPE_BYTES == 4U);
        if (width > 768U) {
            ProcessContiguousLastTwoAxesManualDoubleInput();
            return;
        }
        constexpr uint32_t phaseRows = 32U;
        constexpr uint32_t macroRows = 384U;
        constexpr uint32_t outputCols = 32U;
        constexpr uint32_t bankBytes = 96U * 1024U;
        constexpr uint32_t compactBankBytes = 48U * 1024U;
        constexpr uint32_t retainedColumnCapacity =
            compactBankBytes / (phaseRows * TYPE_BYTES);
        const uint32_t paddedWidth = CeilDiv(width, 16U) * 16U;
        const bool packedRows = width <= 576U && (width & 7U) != 0U &&
            inputRowStride == width;
        const uint32_t retainedColumns = paddedWidth < retainedColumnCapacity ?
            paddedWidth : retainedColumnCapacity;
        const uint32_t macroCount = batchCount * rowTileCount;
        const uint32_t workspaceElements = macroRows * paddedWidth;
        AscendC::GlobalTensor<T> staged;
        staged.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(workspace) +
                static_cast<uint64_t>(blockIdx) * workspaceElements,
            workspaceElements);

        for (uint32_t task = blockIdx; task < macroCount; task += blockNum) {
            const uint32_t batch = task / rowTileCount;
            const uint32_t rowStart = (task % rowTileCount) * macroRows;
            const uint32_t rows = height - rowStart < macroRows ?
                height - rowStart : macroRows;
            const uint32_t phases = CeilDiv(rows, phaseRows);
            const uint32_t inputBase = InputBatchOffset(batch) +
                rowStart * inputRowStride;
            const uint32_t outputBase = batch * matrixElements + rowStart;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            for (uint32_t phase = 0U; phase < phases; ++phase) {
                const uint32_t remaining = rows - phase * phaseRows;
                const uint32_t liveRows = remaining < phaseRows ?
                    remaining : phaseRows;
                AscendC::LocalTensor<T> input(
                    AscendC::TPosition::VECIN, 0U,
                    (packedRows ? 73U * 1024U : bankBytes) / TYPE_BYTES);
                AscendC::LocalTensor<T> transposed(
                    AscendC::TPosition::VECOUT,
                    packedRows ? 120U * 1024U : bankBytes,
                    (packedRows ? 72U * 1024U : bankBytes) / TYPE_BYTES);
                const uint32_t alignedWidth = CeilDiv(width, 8U) * 8U;
                AscendC::DataCopyExtParams load{
                    static_cast<uint16_t>(liveRows), width * TYPE_BYTES,
                    (inputRowStride - width) * TYPE_BYTES,
                    (paddedWidth - alignedWidth) * TYPE_BYTES / 32U, 0U};
                AscendC::DataCopyPadExtParams<T> pad{
                    true, 0U, static_cast<uint8_t>(alignedWidth - width),
                    static_cast<T>(0)};
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                if (packedRows) {
                    const uint32_t elements = liveRows * width;
                    AscendC::DataCopyExtParams slab{1U,
                        elements * TYPE_BYTES, 0U, 0U, 0U};
                    AscendC::DataCopyPadExtParams<T> slabPad{true, 0U,
                        static_cast<uint8_t>((8U - (elements & 7U)) & 7U),
                        static_cast<T>(0)};
                    AscendC::DataCopyPad(input,
                        xGm[inputBase + phase * phaseRows * inputRowStride],
                        slab, slabPad);
                } else {
                    AscendC::DataCopyPad(input,
                        xGm[inputBase + phase * phaseRows * inputRowStride],
                        load, pad);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                if (packedRows && phase == 0U) {
                    BuildPackedRowFilterPatterns(paddedWidth);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
                if (packedRows) {
                    AscendC::LocalTensor<T> regular(
                        AscendC::TPosition::VECCALC, 73U * 1024U,
                        36U * 1024U / TYPE_BYTES);
                    for (uint32_t half = 0U; half < liveRows; half += 16U) {
                        const uint32_t halfRows = liveRows - half < 16U ?
                            liveRows - half : 16U;
                        FilterPackedRowHalf(input, regular, half,
                            halfRows, paddedWidth);
                        ComputeLastTwoB32RowPhase(transposed, regular,
                            half / TILE_ROWS, 1U, paddedWidth / TILE_COLS,
                            paddedWidth, phaseRows);
                    }
                } else {
                    ComputeLastTwoB32RowPhase(
                        transposed, input, 0U, CeilDiv(liveRows, TILE_ROWS),
                        paddedWidth / TILE_COLS, paddedWidth, phaseRows);
                }
                uint32_t firstStoredColumn = 0U;
                if (phase + 1U == phases) {
                    // The final VNCHW's PIPE_V has retired all input reads.
                    // Reuse only the half-bank not needed by stage2 loads.
                    AscendC::LocalTensor<T> retained(
                        AscendC::TPosition::VECIN, compactBankBytes,
                        compactBankBytes / TYPE_BYTES);
                    const auto retainedBits = retained.template ReinterpretCast<uint16_t>();
                    const auto transposedBits = transposed.template ReinterpretCast<uint16_t>();
                    AscendC::BinaryRepeatParams prefixCopy{1U, 1U, 1U, 8U, 8U, 8U};
                    AscendC::Or(retainedBits, transposedBits, transposedBits,
                        static_cast<uint64_t>(128U),
                        static_cast<uint8_t>(retainedColumns * phaseRows * TYPE_BYTES / 256U),
                        prefixCopy);
                    // Explicit RAW edge to later retained-record consumers.
                    AscendC::PipeBarrier<PIPE_V>();
                    firstStoredColumn = retainedColumns;
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
                if (firstStoredColumn < paddedWidth) {
                    AscendC::DataCopy(
                        staged[(phase * paddedWidth + firstStoredColumn) * phaseRows],
                        transposed[firstStoredColumn * phaseRows],
                        (paddedWidth - firstStoredColumn) * phaseRows);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);

            // Keep DMA transfers block-contiguous. Vector integer moves
            // perform the phase/column record permutation entirely in UB.
            // This edge publishes stage one's GM before any consumer load.
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            for (uint32_t colStart = 0U; colStart < width;
                 colStart += outputCols) {
                const uint32_t bank = (colStart / outputCols) & 1U;
                const event_t event = static_cast<event_t>(bank);
                const uint32_t cols = width - colStart < outputCols ?
                    width - colStart : outputCols;
                const bool hasRetainedPhase = colStart + cols <= retainedColumns;
                const uint32_t gmPhases = phases - (hasRetainedPhase ? 1U : 0U);
                // One input generation suffices only if its current load
                // remains covered by the independently banked output stores.
                LoadWorkspaceCompactGroup(staged, colStart, 0U,
                    cols, gmPhases, paddedWidth);
                AscendC::LocalTensor<T> compact(
                    AscendC::TPosition::VECIN, 0U,
                    compactBankBytes / TYPE_BYTES);
                AscendC::LocalTensor<T> assembled(
                    AscendC::TPosition::VECOUT, bankBytes + bank * compactBankBytes,
                    compactBankBytes / TYPE_BYTES);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
                const auto inputBits = compact.template ReinterpretCast<uint16_t>();
                const auto outputBits = assembled.template ReinterpretCast<uint16_t>();
                AscendC::BinaryRepeatParams recordMove{
                    1U, 1U, 1U,
                    static_cast<uint8_t>(macroRows * TYPE_BYTES / 32U),
                    static_cast<uint8_t>(phaseRows * TYPE_BYTES / 32U),
                    static_cast<uint8_t>(phaseRows * TYPE_BYTES / 32U)};
                for (uint32_t phase = 0U; phase < gmPhases; ++phase) {
                    const uint32_t sourceOffset = phase * cols * phaseRows * 2U;
                    AscendC::Or(outputBits[phase * phaseRows * 2U],
                        inputBits[sourceOffset], inputBits[sourceOffset],
                        static_cast<uint64_t>(phaseRows * 2U),
                        static_cast<uint8_t>(cols), recordMove);
                }
                if (hasRetainedPhase) {
                    AscendC::LocalTensor<T> retained(
                        AscendC::TPosition::VECIN, compactBankBytes,
                        compactBankBytes / TYPE_BYTES);
                    const auto retainedBits = retained.template ReinterpretCast<uint16_t>();
                    const uint32_t retainedOffset = colStart * phaseRows * 2U;
                    AscendC::Or(outputBits[gmPhases * phaseRows * 2U],
                        retainedBits[retainedOffset], retainedBits[retainedOffset],
                        static_cast<uint64_t>(phaseRows * 2U),
                        static_cast<uint8_t>(cols), recordMove);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
                const uint32_t alignedRows = CeilDiv(rows, 8U) * 8U;
                AscendC::DataCopyExtParams publish{
                    static_cast<uint16_t>(cols), rows * TYPE_BYTES,
                    (macroRows - alignedRows) * TYPE_BYTES / 32U,
                    (outputRowStride - rows) * TYPE_BYTES, 0U};
                AscendC::DataCopyPad(zGm[outputBase + colStart * outputRowStride],
                    assembled, publish);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        }
    }

    // Two logical AIVs share one 768-row workspace generation. This does not
    // assume that the runtime places them on a physical core pair. All blocks,
    // including an odd unused block and idle tail groups, join both barriers.
    __aicore__ inline void ProcessPairedFullRowWorkspace(GM_ADDR workspace)
    {
        static_assert(TYPE_BYTES == 4U);
        if (width > 768U || blockNum < 2U) {
            ProcessFullRowWorkspace(workspace);
            return;
        }
        constexpr uint32_t phaseRows = 32U;
        constexpr uint32_t macroRows = 768U;
        constexpr uint32_t outputCols = 16U;
        constexpr uint32_t bankBytes = 96U * 1024U;
        constexpr uint32_t consumerBankBytes = 48U * 1024U;
        const uint32_t paddedWidth = CeilDiv(width, 16U) * 16U;
        const bool packedRows = width <= 576U && (width & 7U) != 0U &&
            inputRowStride == width;
        const uint32_t groups = blockNum / 2U;
        const uint32_t group = blockIdx / 2U;
        const uint32_t lane = blockIdx & 1U;
        const uint32_t rowMacros = CeilDiv(height, macroRows);
        const uint32_t macroCount = batchCount * rowMacros;
        const uint32_t workspaceElements = macroRows * paddedWidth;
        AscendC::GlobalTensor<T> staged;
        staged.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(workspace) +
                static_cast<uint64_t>(group < groups ? group : 0U) * workspaceElements,
            workspaceElements);

        for (uint32_t cohort = 0U; cohort < macroCount; cohort += groups) {
            const uint32_t task = cohort + group;
            const bool active = group < groups && task < macroCount;
            const uint32_t batch = active ? task / rowMacros : 0U;
            const uint32_t rowStart = active ? (task % rowMacros) * macroRows : 0U;
            const uint32_t rows = active ?
                (height - rowStart < macroRows ? height - rowStart : macroRows) : 0U;
            const uint32_t phases = CeilDiv(rows, phaseRows);
            const uint32_t inputBase = InputBatchOffset(batch) + rowStart * inputRowStride;
            const uint32_t outputBase = batch * matrixElements + rowStart;

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            for (uint32_t phase = lane; phase < phases; phase += 2U) {
                const uint32_t remaining = rows - phase * phaseRows;
                const uint32_t liveRows = remaining < phaseRows ? remaining : phaseRows;
                AscendC::LocalTensor<T> input(
                    AscendC::TPosition::VECIN, 0U,
                    (packedRows ? 73U * 1024U : bankBytes) / TYPE_BYTES);
                AscendC::LocalTensor<T> transposed(
                    AscendC::TPosition::VECOUT,
                    packedRows ? 120U * 1024U : bankBytes,
                    (packedRows ? 72U * 1024U : bankBytes) / TYPE_BYTES);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
                if (packedRows) {
                    const uint32_t elements = liveRows * width;
                    AscendC::DataCopyExtParams slab{1U, elements * TYPE_BYTES, 0U, 0U, 0U};
                    AscendC::DataCopyPadExtParams<T> slabPad{true, 0U,
                        static_cast<uint8_t>((8U - (elements & 7U)) & 7U), static_cast<T>(0)};
                    AscendC::DataCopyPad(input,
                        xGm[inputBase + phase * phaseRows * inputRowStride], slab, slabPad);
                } else {
                    const uint32_t alignedWidth = CeilDiv(width, 8U) * 8U;
                    AscendC::DataCopyExtParams load{
                        static_cast<uint16_t>(liveRows), width * TYPE_BYTES,
                        (inputRowStride - width) * TYPE_BYTES,
                        (paddedWidth - alignedWidth) * TYPE_BYTES / 32U, 0U};
                    AscendC::DataCopyPadExtParams<T> pad{true, 0U,
                        static_cast<uint8_t>(alignedWidth - width), static_cast<T>(0)};
                    AscendC::DataCopyPad(input,
                        xGm[inputBase + phase * phaseRows * inputRowStride], load, pad);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                if (packedRows && phase == lane) {
                    BuildPackedRowFilterPatterns(paddedWidth);
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
                if (packedRows) {
                    AscendC::LocalTensor<T> regular(
                        AscendC::TPosition::VECCALC, 73U * 1024U, 36U * 1024U / TYPE_BYTES);
                    for (uint32_t half = 0U; half < liveRows; half += 16U) {
                        const uint32_t halfRows = liveRows - half < 16U ? liveRows - half : 16U;
                        FilterPackedRowHalf(input, regular, half, halfRows, paddedWidth);
                        ComputeLastTwoB32RowPhase(transposed, regular,
                            half / TILE_ROWS, 1U, paddedWidth / TILE_COLS, paddedWidth, phaseRows);
                    }
                } else {
                    ComputeLastTwoB32RowPhase(transposed, input, 0U,
                        CeilDiv(liveRows, TILE_ROWS), paddedWidth / TILE_COLS, paddedWidth, phaseRows);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1);
                AscendC::DataCopy(staged[phase * paddedWidth * phaseRows],
                    transposed, paddedWidth * phaseRows);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            // Publish every producer's workspace records before peer reads.
            AscendC::SyncAll<true>();

            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
            const uint32_t firstColumn = lane * outputCols;
            if (active && firstColumn < width) {
                const uint32_t firstCols = width - firstColumn < outputCols ?
                    width - firstColumn : outputCols;
                LoadWorkspaceCompactGroup(staged, firstColumn, 0U,
                    firstCols, phases, paddedWidth);
            }
            uint32_t consumerGeneration = 0U;
            for (uint32_t colStart = lane * outputCols;
                 active && colStart < width;
                 colStart += 2U * outputCols, ++consumerGeneration) {
                const uint32_t bank = consumerGeneration & 1U;
                const event_t event = static_cast<event_t>(bank);
                const uint32_t cols = width - colStart < outputCols ? width - colStart : outputCols;
                const uint32_t nextColumn = colStart + 2U * outputCols;
                if (nextColumn < width) {
                    const uint32_t nextCols = width - nextColumn < outputCols ?
                        width - nextColumn : outputCols;
                    // The next input bank has no live Vector reader after its
                    // own V_MTE2 token. It can fill while this group computes.
                    LoadWorkspaceCompactGroup(staged, nextColumn, bank ^ 1U,
                        nextCols, phases, paddedWidth);
                }
                AscendC::LocalTensor<T> compact(
                    AscendC::TPosition::VECIN,
                    bank * consumerBankBytes, consumerBankBytes / TYPE_BYTES);
                AscendC::LocalTensor<T> assembled(
                    AscendC::TPosition::VECOUT,
                    bankBytes + bank * consumerBankBytes, consumerBankBytes / TYPE_BYTES);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
                const auto inputBits = compact.template ReinterpretCast<uint16_t>();
                const auto outputBits = assembled.template ReinterpretCast<uint16_t>();
                AscendC::BinaryRepeatParams recordMove{1U, 1U, 1U,
                    static_cast<uint8_t>(macroRows * TYPE_BYTES / 32U),
                    static_cast<uint8_t>(phaseRows * TYPE_BYTES / 32U),
                    static_cast<uint8_t>(phaseRows * TYPE_BYTES / 32U)};
                for (uint32_t phase = 0U; phase < phases; ++phase) {
                    const uint32_t sourceOffset = phase * cols * phaseRows * 2U;
                    AscendC::Or(outputBits[phase * phaseRows * 2U],
                        inputBits[sourceOffset], inputBits[sourceOffset],
                        static_cast<uint64_t>(phaseRows * 2U), static_cast<uint8_t>(cols), recordMove);
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
                const uint32_t alignedRows = CeilDiv(rows, 8U) * 8U;
                AscendC::DataCopyExtParams publish{
                    static_cast<uint16_t>(cols), rows * TYPE_BYTES,
                    (macroRows - alignedRows) * TYPE_BYTES / 32U,
                    (outputRowStride - rows) * TYPE_BYTES, 0U};
                AscendC::DataCopyPad(zGm[outputBase + colStart * outputRowStride], assembled, publish);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            }
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
            // Retire all readers before either producer reuses the slot.
            AscendC::SyncAll<true>();
        }
    }

    __aicore__ inline void LoadWorkspaceCompactGroup(
        const AscendC::GlobalTensor<T>& staged, uint32_t colStart,
        uint32_t bank, uint32_t cols, uint32_t phases, uint32_t paddedWidth)
    {
        constexpr uint32_t phaseRows = 32U;
        constexpr uint32_t compactBankBytes = 48U * 1024U;
        const event_t event = static_cast<event_t>(bank);
        AscendC::LocalTensor<T> compact(AscendC::TPosition::VECIN,
            bank * compactBankBytes, compactBankBytes / TYPE_BYTES);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyParams load{
            static_cast<uint16_t>(phases),
            static_cast<uint16_t>(cols * phaseRows * TYPE_BYTES / 32U),
            static_cast<uint16_t>((paddedWidth - cols) * phaseRows *
                TYPE_BYTES / 32U),
            0U};
        if (phases != 0U) {
            AscendC::DataCopy(compact, staged[colStart * phaseRows], load);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    // Reproduce the queue-managed 64-row overlap with two explicit 24-KiB
    // input banks.  Issuing phase p + 1 before waiting on phase p lets MTE2
    // run beside VNCHW without per-phase TQue allocation bookkeeping.  After
    // the short-start task, prefetch the next task's first adjacent bank pair
    // with one 48-KiB descriptor while the current output is published.
    __aicore__ inline void ProcessContiguousLastTwoAxesManualDoubleInput()
    {
        constexpr event_t outputEvent = static_cast<event_t>(2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            static_cast<event_t>(0));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
            static_cast<event_t>(1));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

        if (blockIdx < totalTaskCount) {
            uint32_t currentInputBatchOffset = 0U;
            uint32_t currentOutputBatchOffset = 0U;
            uint32_t currentRowStart = 0U;
            uint32_t currentColStart = 0U;
            bool currentTallTail = false;
            DecodeAdaptiveTallTailTask(
                blockIdx, currentInputBatchOffset,
                currentOutputBatchOffset, currentRowStart,
                currentColStart, currentTallTail);
            LoadLastTwoAxesManualInputPhase(
                static_cast<event_t>(0), 0U,
                currentInputBatchOffset, currentRowStart,
                currentColStart, currentTallTail, 0U);
            bool currentPairPrefetched = false;

            for (uint32_t task = blockIdx; task < totalTaskCount;
                 task += blockNum) {
                const uint32_t nextTask = task + blockNum;
                const bool hasNext = nextTask < totalTaskCount;
                uint32_t nextInputBatchOffset = 0U;
                uint32_t nextOutputBatchOffset = 0U;
                uint32_t nextRowStart = 0U;
                uint32_t nextColStart = 0U;
                bool nextTallTail = false;
                if (hasNext) {
                    DecodeAdaptiveTallTailTask(
                        nextTask, nextInputBatchOffset,
                        nextOutputBatchOffset, nextRowStart,
                        nextColStart, nextTallTail);
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                    outputEvent);
                AscendC::LocalTensor<T> output(
                    AscendC::TPosition::VECOUT,
                    WIDE_ACCUMULATED_OUTPUT_UB_OFFSET,
                    LAST_TILE_ELEMENTS);
                const uint32_t tileRows = currentTallTail ?
                    ADAPTIVE_TAIL_ROWS : LAST_TILE_ROWS;
                const uint32_t tileCols = currentTallTail ?
                    ADAPTIVE_TAIL_COLS : LAST_TILE_COLS;
                const uint32_t phaseRowBlocks = currentTallTail ?
                    QUEUED_TAIL_PHASE_ROW_BLOCKS :
                    QUEUED_PHASE_ROW_BLOCKS;
                const uint32_t rows =
                    height - currentRowStart < tileRows ?
                        height - currentRowStart : tileRows;
                const uint32_t cols =
                    width - currentColStart < tileCols ?
                        width - currentColStart : tileCols;
                const uint32_t activeRowBlocks =
                    CeilDiv(rows, TILE_ROWS);
                const uint32_t activeColumnBlocks =
                    CeilDiv(cols, TILE_COLS);
                const uint32_t activePhaseCount =
                    CeilDiv(activeRowBlocks, phaseRowBlocks);

                for (uint32_t phase = 0U;
                     phase < activePhaseCount; ++phase) {
                    const uint32_t bank = phase & 1U;
                    const event_t inputEvent =
                        static_cast<event_t>(bank);
                    if (!(currentPairPrefetched && phase == 0U) &&
                        phase + 1U < activePhaseCount) {
                        const uint32_t nextBank = bank ^ 1U;
                        LoadLastTwoAxesManualInputPhase(
                            static_cast<event_t>(nextBank), nextBank,
                            currentInputBatchOffset, currentRowStart,
                            currentColStart, currentTallTail, phase + 1U);
                    }
                    if (!(currentPairPrefetched && phase == 1U)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                            inputEvent);
                    }
                    AscendC::LocalTensor<T> input(
                        AscendC::TPosition::VECIN,
                        WIDE_PHASE_INPUT_UB_OFFSET +
                            bank * QUEUED_PHASE_INPUT_ELEMENTS * TYPE_BYTES,
                        QUEUED_PHASE_INPUT_ELEMENTS);
                    const uint32_t firstRowBlock =
                        phase * phaseRowBlocks;
                    const uint32_t remainingRowBlocks =
                        activeRowBlocks - firstRowBlock;
                    const uint32_t currentPhaseRowBlocks =
                        remainingRowBlocks < phaseRowBlocks ?
                            remainingRowBlocks : phaseRowBlocks;
                    if (currentTallTail) {
                        ComputeLastTwoB32RawAffineRowPhase<true>(
                            output, input, firstRowBlock,
                            currentPhaseRowBlocks);
                    } else {
                        ComputeLastTwoB32RawAffineRowPhase<false>(
                            output, input, firstRowBlock,
                            currentPhaseRowBlocks);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                        inputEvent);
                }

                if (hasNext) {
                    LoadLastTwoAxesManualInputPair(
                        nextInputBatchOffset, nextRowStart,
                        nextColStart, nextTallTail);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                    outputEvent);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                    outputEvent);
                StoreLastTwoAxesAdaptiveOutput(
                    output, currentOutputBatchOffset,
                    currentRowStart, currentColStart,
                    tileRows, tileCols);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                    outputEvent);

                currentInputBatchOffset = nextInputBatchOffset;
                currentOutputBatchOffset = nextOutputBatchOffset;
                currentRowStart = nextRowStart;
                currentColStart = nextColStart;
                currentTallTail = nextTallTail;
                currentPairPrefetched = hasNext;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            static_cast<event_t>(0));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            static_cast<event_t>(1));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    // Keep D28's 384-by-96 accumulated output and useful GM traffic, but
    // divide the remaining 48 KiB into two queue-managed 24-KiB input
    // generations.  CANN's TQue dependencies make phase p + 1 MTE2 visible
    // to the scheduler while Vector consumes phase p, unlike the manually
    // paired event prototype whose compiled object did not change.  The
    // adaptive 768-by-48 tail uses the same 6,144-element banks, exchanging
    // width for phase height without a second implementation.
    __aicore__ inline void ProcessContiguousLastTwoAxesQueuedInputPhases()
    {
        constexpr event_t outputEvent = static_cast<event_t>(0);
        queuedPipe.InitBuffer(
            queuedOutputBuffer, LAST_TILE_ELEMENTS * TYPE_BYTES);
        queuedPipe.InitBuffer(
            queuedInputQueue, 2U,
            QUEUED_PHASE_INPUT_ELEMENTS * TYPE_BYTES);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

        if (blockIdx < totalTaskCount) {
            uint32_t currentInputBatchOffset = 0U;
            uint32_t currentOutputBatchOffset = 0U;
            uint32_t currentRowStart = 0U;
            uint32_t currentColStart = 0U;
            bool currentTallTail = false;
            DecodeAdaptiveTallTailTask(
                blockIdx, currentInputBatchOffset,
                currentOutputBatchOffset, currentRowStart,
                currentColStart, currentTallTail);
            LoadLastTwoAxesQueuedInputPhase(
                currentInputBatchOffset, currentRowStart,
                currentColStart, currentTallTail, 0U);

            for (uint32_t task = blockIdx; task < totalTaskCount;
                 task += blockNum) {
                const uint32_t nextTask = task + blockNum;
                const bool hasNext = nextTask < totalTaskCount;
                uint32_t nextInputBatchOffset = 0U;
                uint32_t nextOutputBatchOffset = 0U;
                uint32_t nextRowStart = 0U;
                uint32_t nextColStart = 0U;
                bool nextTallTail = false;
                if (hasNext) {
                    DecodeAdaptiveTallTailTask(
                        nextTask, nextInputBatchOffset,
                        nextOutputBatchOffset, nextRowStart,
                        nextColStart, nextTallTail);
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                    outputEvent);
                AscendC::LocalTensor<T> output =
                    queuedOutputBuffer.Get<T>();
                const uint32_t tileRows = currentTallTail ?
                    ADAPTIVE_TAIL_ROWS : LAST_TILE_ROWS;
                const uint32_t tileCols = currentTallTail ?
                    ADAPTIVE_TAIL_COLS : LAST_TILE_COLS;
                const uint32_t phaseRowBlocks = currentTallTail ?
                    QUEUED_TAIL_PHASE_ROW_BLOCKS :
                    QUEUED_PHASE_ROW_BLOCKS;
                const uint32_t rows =
                    height - currentRowStart < tileRows ?
                        height - currentRowStart : tileRows;
                const uint32_t cols =
                    width - currentColStart < tileCols ?
                        width - currentColStart : tileCols;
                const uint32_t activeRowBlocks =
                    CeilDiv(rows, TILE_ROWS);
                const uint32_t activeColumnBlocks =
                    CeilDiv(cols, TILE_COLS);
                const uint32_t activePhaseCount =
                    CeilDiv(activeRowBlocks, phaseRowBlocks);

                for (uint32_t phase = 0U;
                     phase < activePhaseCount; ++phase) {
                    if (phase + 1U < activePhaseCount) {
                        LoadLastTwoAxesQueuedInputPhase(
                            currentInputBatchOffset,
                            currentRowStart, currentColStart,
                            currentTallTail, phase + 1U);
                    }
                    AscendC::LocalTensor<T> input =
                        queuedInputQueue.DeQue<T>();
                    const uint32_t firstRowBlock =
                        phase * phaseRowBlocks;
                    const uint32_t remainingRowBlocks =
                        activeRowBlocks - firstRowBlock;
                    const uint32_t currentPhaseRowBlocks =
                        remainingRowBlocks < phaseRowBlocks ?
                            remainingRowBlocks : phaseRowBlocks;
                    ComputeLastTwoB32RowPhase(
                        output, input, firstRowBlock,
                        currentPhaseRowBlocks, activeColumnBlocks,
                        tileCols, tileRows);
                    queuedInputQueue.FreeTensor(input);
                }

                if (hasNext) {
                    LoadLastTwoAxesQueuedInputPhase(
                        nextInputBatchOffset, nextRowStart,
                        nextColStart, nextTallTail, 0U);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                    outputEvent);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                    outputEvent);
                StoreLastTwoAxesAdaptiveOutput(
                    output, currentOutputBatchOffset,
                    currentRowStart, currentColStart,
                    tileRows, tileCols);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                    outputEvent);

                currentInputBatchOffset = nextInputBatchOffset;
                currentOutputBatchOffset = nextOutputBatchOffset;
                currentRowStart = nextRowStart;
                currentColStart = nextColStart;
                currentTallTail = nextTallTail;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    // The final at-most-half-width column tile can exchange width for height
    // without changing either UB bank size:
    //
    //   LAST_TILE_ROWS * LAST_TILE_COLS
    //     == (2 * LAST_TILE_ROWS) * (LAST_TILE_COLS / 2).
    //
    // Keep the established 384-by-96 schedule for complete column tiles, but
    // combine two tail row tiles into one 768-by-48 generation.  This halves
    // tail input phase calls and publishes twice-long output rows while the
    // useful input/output bytes and total VNCHW block work remain unchanged.
    __aicore__ inline void ProcessContiguousLastTwoAxesAdaptiveTallTail()
    {
        constexpr event_t inputEvent = static_cast<event_t>(0);
        constexpr event_t outputEvent = static_cast<event_t>(1);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);

        if (blockIdx < totalTaskCount) {
            uint32_t currentInputBatchOffset = 0U;
            uint32_t currentOutputBatchOffset = 0U;
            uint32_t currentRowStart = 0U;
            uint32_t currentColStart = 0U;
            bool currentTallTail = false;
            DecodeAdaptiveTallTailTask(
                blockIdx, currentInputBatchOffset,
                currentOutputBatchOffset, currentRowStart,
                currentColStart, currentTallTail);
            LoadLastTwoAxesAdaptiveRowPhase(
                inputEvent, currentInputBatchOffset,
                currentRowStart, currentColStart,
                currentTallTail, 0U);

            for (uint32_t task = blockIdx; task < totalTaskCount;
                 task += blockNum) {
                const uint32_t nextTask = task + blockNum;
                const bool hasNext = nextTask < totalTaskCount;
                uint32_t nextInputBatchOffset = 0U;
                uint32_t nextOutputBatchOffset = 0U;
                uint32_t nextRowStart = 0U;
                uint32_t nextColStart = 0U;
                bool nextTallTail = false;
                if (hasNext) {
                    DecodeAdaptiveTallTailTask(
                        nextTask, nextInputBatchOffset,
                        nextOutputBatchOffset, nextRowStart,
                        nextColStart, nextTallTail);
                }

                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                    outputEvent);
                AscendC::LocalTensor<T> input(
                    AscendC::TPosition::VECIN,
                    WIDE_PHASE_INPUT_UB_OFFSET,
                    WIDE_PHASE_INPUT_ELEMENTS);
                AscendC::LocalTensor<T> output(
                    AscendC::TPosition::VECOUT,
                    WIDE_ACCUMULATED_OUTPUT_UB_OFFSET,
                    LAST_TILE_ELEMENTS);
                const uint32_t tileRows = currentTallTail ?
                    ADAPTIVE_TAIL_ROWS : LAST_TILE_ROWS;
                const uint32_t tileCols = currentTallTail ?
                    ADAPTIVE_TAIL_COLS : LAST_TILE_COLS;
                const uint32_t phaseRowBlocks = currentTallTail ?
                    ADAPTIVE_TAIL_PHASE_ROW_BLOCKS :
                    WIDE_PHASE_ROW_BLOCKS;
                const uint32_t rows =
                    height - currentRowStart < tileRows ?
                        height - currentRowStart : tileRows;
                const uint32_t cols =
                    width - currentColStart < tileCols ?
                        width - currentColStart : tileCols;
                const uint32_t activeRowBlocks =
                    CeilDiv(rows, TILE_ROWS);
                const uint32_t activeColumnBlocks =
                    CeilDiv(cols, TILE_COLS);
                const uint32_t activePhaseCount =
                    CeilDiv(activeRowBlocks, phaseRowBlocks);

                for (uint32_t phase = 0U;
                     phase < activePhaseCount; ++phase) {
                    if (phase > 0U) {
                        LoadLastTwoAxesAdaptiveRowPhase(
                            inputEvent, currentInputBatchOffset,
                            currentRowStart, currentColStart,
                            currentTallTail, phase);
                    }
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                        inputEvent);
                    const uint32_t firstRowBlock =
                        phase * phaseRowBlocks;
                    const uint32_t remainingRowBlocks =
                        activeRowBlocks - firstRowBlock;
                    const uint32_t currentPhaseRowBlocks =
                        remainingRowBlocks < phaseRowBlocks ?
                            remainingRowBlocks : phaseRowBlocks;
                    ComputeLastTwoB32RowPhase(
                        output, input, firstRowBlock,
                        currentPhaseRowBlocks, activeColumnBlocks,
                        tileCols, tileRows);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                        inputEvent);
                }

                if (hasNext) {
                    LoadLastTwoAxesAdaptiveRowPhase(
                        inputEvent, nextInputBatchOffset,
                        nextRowStart, nextColStart,
                        nextTallTail, 0U);
                }

                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                    outputEvent);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                    outputEvent);
                StoreLastTwoAxesAdaptiveOutput(
                    output, currentOutputBatchOffset,
                    currentRowStart, currentColStart,
                    tileRows, tileCols);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                    outputEvent);

                currentInputBatchOffset = nextInputBatchOffset;
                currentOutputBatchOffset = nextOutputBatchOffset;
                currentRowStart = nextRowStart;
                currentColStart = nextColStart;
                currentTallTail = nextTallTail;
            }
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    __aicore__ inline void ProcessContiguousLastTwoAxesOneInputTwoOutputs()
    {
        constexpr event_t inputEvent = static_cast<event_t>(2);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(buffer));
        }

        uint32_t progress = 0U;
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t batch = task / tasksPerBatch;
            const uint32_t matrixTask = task - batch * tasksPerBatch;
            const uint32_t rowTile = matrixTask / colTileCount;
            const uint32_t colTile = matrixTask - rowTile * colTileCount;
            ProcessLastTwoAxesTileOneInputTwoOutputs(
                progress & 1U,
                InputBatchOffset(batch), batch * matrixElements,
                rowTile * LAST_TILE_ROWS, colTile * LAST_TILE_COLS);
            ++progress;
        }

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(buffer));
        }
    }

private:
    __aicore__ inline void DecodeAdaptiveTallTailTask(
        uint32_t task, uint32_t& inputBatchOffset,
        uint32_t& outputBatchOffset, uint32_t& rowStart,
        uint32_t& colStart, bool& tallTail) const
    {
        const uint32_t batch = task / tasksPerBatch;
        const uint32_t matrixTask = task - batch * tasksPerBatch;
        const uint32_t fullColumnTileCount = width / LAST_TILE_COLS;
        const uint32_t mainTaskCount =
            rowTileCount * fullColumnTileCount;
        tallTail = matrixTask >= mainTaskCount;
        if (tallTail) {
            rowStart = (matrixTask - mainTaskCount) *
                ADAPTIVE_TAIL_ROWS;
            colStart = fullColumnTileCount * LAST_TILE_COLS;
        } else {
            const uint32_t rowTile =
                matrixTask / fullColumnTileCount;
            const uint32_t colTile =
                matrixTask - rowTile * fullColumnTileCount;
            rowStart = rowTile * LAST_TILE_ROWS;
            colStart = colTile * LAST_TILE_COLS;
        }
        inputBatchOffset = InputBatchOffset(batch);
        outputBatchOffset = batch * matrixElements;
    }

    __aicore__ inline void DecodeContiguousLastTwoTask(
        uint32_t task, uint32_t& inputBatchOffset,
        uint32_t& outputBatchOffset, uint32_t& rowStart,
        uint32_t& colStart) const
    {
        const uint32_t batch = task / tasksPerBatch;
        const uint32_t matrixTask = task - batch * tasksPerBatch;
        const uint32_t rowTile = matrixTask / colTileCount;
        const uint32_t colTile = matrixTask - rowTile * colTileCount;
        inputBatchOffset = InputBatchOffset(batch);
        outputBatchOffset = batch * matrixElements;
        rowStart = rowTile * LAST_TILE_ROWS;
        colStart = colTile * LAST_TILE_COLS;
    }

    __aicore__ inline void LoadLastTwoAxesWideInput(
        event_t event, uint32_t inputBatchOffset,
        uint32_t rowStart, uint32_t colStart)
    {
        const uint32_t rows =
            height - rowStart < LAST_TILE_ROWS ?
                height - rowStart : LAST_TILE_ROWS;
        const uint32_t cols =
            width - colStart < LAST_TILE_COLS ?
                width - colStart : LAST_TILE_COLS;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset +
            rowStart * inputRowStride + colStart;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            INPUT_UB_OFFSET, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES, inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void LoadLastTwoAxesWideInputRowPhase(
        event_t event, uint32_t inputBatchOffset,
        uint32_t rowStart, uint32_t colStart, uint32_t phase)
    {
        const uint32_t tileRows =
            height - rowStart < LAST_TILE_ROWS ?
                height - rowStart : LAST_TILE_ROWS;
        const uint32_t phaseRowOffset =
            phase * WIDE_PHASE_ROWS;
        const uint32_t rows =
            tileRows - phaseRowOffset < WIDE_PHASE_ROWS ?
                tileRows - phaseRowOffset : WIDE_PHASE_ROWS;
        const uint32_t cols =
            width - colStart < LAST_TILE_COLS ?
                width - colStart : LAST_TILE_COLS;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset +
            (rowStart + phaseRowOffset) * inputRowStride + colStart;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            WIDE_PHASE_INPUT_UB_OFFSET,
            WIDE_PHASE_INPUT_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void LoadLastTwoAxesAdaptiveRowPhase(
        event_t event, uint32_t inputBatchOffset,
        uint32_t rowStart, uint32_t colStart,
        bool tallTail, uint32_t phase)
    {
        const uint32_t tileRows = tallTail ?
            ADAPTIVE_TAIL_ROWS : LAST_TILE_ROWS;
        const uint32_t tileCols = tallTail ?
            ADAPTIVE_TAIL_COLS : LAST_TILE_COLS;
        const uint32_t phaseRows = tallTail ?
            ADAPTIVE_TAIL_PHASE_ROWS : WIDE_PHASE_ROWS;
        const uint32_t activeTileRows =
            height - rowStart < tileRows ?
                height - rowStart : tileRows;
        const uint32_t phaseRowOffset = phase * phaseRows;
        const uint32_t rows =
            activeTileRows - phaseRowOffset < phaseRows ?
                activeTileRows - phaseRowOffset : phaseRows;
        const uint32_t cols =
            width - colStart < tileCols ?
                width - colStart : tileCols;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((tileCols - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset +
            (rowStart + phaseRowOffset) * inputRowStride + colStart;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            WIDE_PHASE_INPUT_UB_OFFSET,
            WIDE_PHASE_INPUT_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void LoadLastTwoAxesManualInputPhase(
        event_t event, uint32_t bank,
        uint32_t inputBatchOffset, uint32_t rowStart,
        uint32_t colStart, bool tallTail, uint32_t phase)
    {
        const uint32_t tileRows = tallTail ?
            ADAPTIVE_TAIL_ROWS : LAST_TILE_ROWS;
        const uint32_t tileCols = tallTail ?
            ADAPTIVE_TAIL_COLS : LAST_TILE_COLS;
        const uint32_t phaseRows = tallTail ?
            QUEUED_TAIL_PHASE_ROWS : QUEUED_PHASE_ROWS;
        const uint32_t activeTileRows =
            height - rowStart < tileRows ?
                height - rowStart : tileRows;
        const uint32_t phaseRowOffset = phase * phaseRows;
        const uint32_t rows =
            activeTileRows - phaseRowOffset < phaseRows ?
                activeTileRows - phaseRowOffset : phaseRows;
        const uint32_t cols =
            width - colStart < tileCols ?
                width - colStart : tileCols;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((tileCols - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset +
            (rowStart + phaseRowOffset) * inputRowStride + colStart;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            WIDE_PHASE_INPUT_UB_OFFSET +
                bank * QUEUED_PHASE_INPUT_ELEMENTS * TYPE_BYTES,
            QUEUED_PHASE_INPUT_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void LoadLastTwoAxesManualInputPair(
        uint32_t inputBatchOffset, uint32_t rowStart,
        uint32_t colStart, bool tallTail)
    {
        const uint32_t tileRows = tallTail ?
            ADAPTIVE_TAIL_ROWS : LAST_TILE_ROWS;
        const uint32_t tileCols = tallTail ?
            ADAPTIVE_TAIL_COLS : LAST_TILE_COLS;
        const uint32_t phaseRows = tallTail ?
            QUEUED_TAIL_PHASE_ROWS : QUEUED_PHASE_ROWS;
        const uint32_t activeTileRows =
            height - rowStart < tileRows ?
                height - rowStart : tileRows;
        const uint32_t pairRows = 2U * phaseRows;
        const uint32_t rows = activeTileRows < pairRows ?
            activeTileRows : pairRows;
        const uint32_t cols =
            width - colStart < tileCols ?
                width - colStart : tileCols;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((tileCols - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset +
            rowStart * inputRowStride + colStart;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            WIDE_PHASE_INPUT_UB_OFFSET,
            2U * QUEUED_PHASE_INPUT_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            static_cast<event_t>(0));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
            static_cast<event_t>(1));
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            static_cast<event_t>(0));
    }

    __aicore__ inline void LoadLastTwoAxesQueuedInputPhase(
        uint32_t inputBatchOffset, uint32_t rowStart,
        uint32_t colStart, bool tallTail, uint32_t phase)
    {
        const uint32_t tileRows = tallTail ?
            ADAPTIVE_TAIL_ROWS : LAST_TILE_ROWS;
        const uint32_t tileCols = tallTail ?
            ADAPTIVE_TAIL_COLS : LAST_TILE_COLS;
        const uint32_t phaseRows = tallTail ?
            QUEUED_TAIL_PHASE_ROWS : QUEUED_PHASE_ROWS;
        const uint32_t activeTileRows =
            height - rowStart < tileRows ?
                height - rowStart : tileRows;
        const uint32_t phaseRowOffset = phase * phaseRows;
        const uint32_t rows =
            activeTileRows - phaseRowOffset < phaseRows ?
                activeTileRows - phaseRowOffset : phaseRows;
        const uint32_t cols =
            width - colStart < tileCols ?
                width - colStart : tileCols;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((tileCols - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset +
            (rowStart + phaseRowOffset) * inputRowStride + colStart;
        AscendC::LocalTensor<T> input =
            queuedInputQueue.AllocTensor<T>();
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        queuedInputQueue.EnQue(input);
    }

    __aicore__ inline void LoadLastTwoAxesInplaceScratchPhase(
        event_t event, uint32_t inputBatchOffset,
        uint32_t rowStart, uint32_t colStart, uint32_t phase)
    {
        const uint32_t cols =
            width - colStart < LAST_TILE_COLS ?
                width - colStart : LAST_TILE_COLS;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_ROWS - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        const uint32_t phaseRowOffset = phase * INPLACE_PHASE_ROWS;
        const uint32_t inputOffset = inputBatchOffset +
            (rowStart + phaseRowOffset) * inputRowStride + colStart;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            INPLACE_SCRATCH_UB_OFFSET,
            LAST_TILE_ELEMENTS - INPLACE_SCRATCH_UB_OFFSET / TYPE_BYTES);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(INPLACE_PHASE_ROWS), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void LoadLastTwoAxesHybridCompactPhase(
        event_t event, uint32_t inputBatchOffset,
        uint32_t rowStart, uint32_t colStart,
        uint32_t firstRowBlock)
    {
        const uint32_t tileRows =
            height - rowStart < LAST_TILE_ROWS ?
                height - rowStart : LAST_TILE_ROWS;
        const uint32_t phaseRowOffset = firstRowBlock * TILE_ROWS;
        const uint32_t remainingRows = tileRows - phaseRowOffset;
        const uint32_t rows =
            remainingRows < WIDE_PHASE_ROWS ?
                remainingRows : WIDE_PHASE_ROWS;
        const uint32_t cols =
            width - colStart < LAST_TILE_COLS ?
                width - colStart : LAST_TILE_COLS;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset +
            (rowStart + phaseRowOffset) * inputRowStride + colStart;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            WIDE_PHASE_INPUT_UB_OFFSET,
            WIDE_PHASE_INPUT_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void CopyInplaceBlock(
        const AscendC::LocalTensor<T>& output,
        const AscendC::LocalTensor<T>& compact,
        uint32_t sourceRowBlock, uint32_t sourceColumnBlock,
        uint32_t compactBlock)
    {
        const uint32_t sourceBase =
            sourceRowBlock * TILE_ROWS * LAST_TILE_ROWS +
            INPLACE_SCRATCH_FIRST_ELEMENT +
            sourceColumnBlock * TILE_COLS;
        constexpr uint16_t blockLen =
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        constexpr uint16_t sourceGap =
            (LAST_TILE_ROWS - TILE_COLS) * TYPE_BYTES /
            DMA_BLOCK_BYTES;
        AscendC::DataCopyParams params{
            static_cast<uint16_t>(TILE_ROWS), blockLen, sourceGap, 0U};
        AscendC::DataCopy(
            compact[compactBlock * TILE_ELEMENTS], output[sourceBase], params);
    }

    __aicore__ inline void CopyInplaceLowerBlockRow(
        const AscendC::LocalTensor<T>& output,
        const AscendC::LocalTensor<T>& compact,
        uint32_t rowBlock, uint32_t compactOffset)
    {
        const uint32_t sourceBase =
            rowBlock * TILE_ROWS * LAST_TILE_ROWS +
            INPLACE_SCRATCH_FIRST_ELEMENT;
        const uint16_t blockLen = static_cast<uint16_t>(
            rowBlock * TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES);
        const uint16_t sourceGap = static_cast<uint16_t>(
            (LAST_TILE_ROWS - rowBlock * TILE_COLS) * TYPE_BYTES /
            DMA_BLOCK_BYTES);
        AscendC::DataCopyParams params{
            static_cast<uint16_t>(TILE_ROWS), blockLen, sourceGap, 0U};
        AscendC::DataCopy(compact[compactOffset], output[sourceBase], params);
    }

    __aicore__ inline void TransposeStridedB32BlockRun(
        const AscendC::LocalTensor<T>& output,
        uint32_t sourceBase, uint32_t destinationBase,
        uint32_t repeatCount)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        constexpr uint16_t destinationRepeatStride =
            TILE_ROWS * LAST_TILE_ROWS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(repeatCount),
            repeatCount == 1U ? 0U : destinationRepeatStride,
            repeatCount == 1U ? 0U :
                TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES);
        for (uint32_t halfColumn = 0U; halfColumn < 2U; ++halfColumn) {
            for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                srcList[row] = reinterpret_cast<uint64_t>(
                    output[sourceBase + row * LAST_TILE_ROWS +
                           halfColumn * 8U].GetPhyAddr());
            }
            for (uint32_t outputRow = 0U; outputRow < 8U; ++outputRow) {
                const uint32_t row = halfColumn * 8U + outputRow;
                dstList[2U * outputRow] = reinterpret_cast<uint64_t>(
                    output[destinationBase +
                           row * LAST_TILE_ROWS].GetPhyAddr());
                dstList[2U * outputRow + 1U] =
                    reinterpret_cast<uint64_t>(
                        output[destinationBase + row * LAST_TILE_ROWS +
                               8U].GetPhyAddr());
            }
            // The stock scalar-address overload does not establish the
            // Scalar-to-Vector VA-register dependency.  Program the four VA
            // registers directly, then make that update visible before
            // issuing VNCHW.  V_S protects the registers from the next run.
            AscendC::SetVaReg(dstList, srcList);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(
                static_cast<event_t>(0));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(
                static_cast<event_t>(0));
            AscendC::TransDataTo5HDIntrinsicsImpl<T>(
                dstList, srcList, params);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(
                static_cast<event_t>(0));
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(
                static_cast<event_t>(0));
        }
    }

    __aicore__ inline void TransposeCompactB32BlockRun(
        const AscendC::LocalTensor<T>& output,
        const AscendC::LocalTensor<T>& compact,
        uint32_t compactBase, uint32_t compactRowElements,
        uint32_t destinationBase, uint32_t repeatCount,
        uint16_t sourceRepeatStride,
        uint16_t destinationRepeatStride)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(repeatCount),
            repeatCount == 1U ? 0U : destinationRepeatStride,
            repeatCount == 1U ? 0U : sourceRepeatStride);
        for (uint32_t halfColumn = 0U; halfColumn < 2U; ++halfColumn) {
            for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                srcList[row] = reinterpret_cast<uint64_t>(
                    compact[compactBase + row * compactRowElements +
                            halfColumn * 8U].GetPhyAddr());
            }
            for (uint32_t outputRow = 0U; outputRow < 8U; ++outputRow) {
                const uint32_t row = halfColumn * 8U + outputRow;
                dstList[2U * outputRow] = reinterpret_cast<uint64_t>(
                    output[destinationBase +
                           row * LAST_TILE_ROWS].GetPhyAddr());
                dstList[2U * outputRow + 1U] =
                    reinterpret_cast<uint64_t>(
                        output[destinationBase + row * LAST_TILE_ROWS +
                               8U].GetPhyAddr());
            }
            AscendC::SetVaReg(dstList, srcList);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(
                static_cast<event_t>(0));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(
                static_cast<event_t>(0));
            AscendC::TransDataTo5HDIntrinsicsImpl<T>(
                dstList, srcList, params);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(
                static_cast<event_t>(0));
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(
                static_cast<event_t>(0));
        }
    }

    __aicore__ inline void TransposeLastTwoAxesInplaceSquarePhase(
        const AscendC::LocalTensor<T>& output, event_t compactEvent,
        uint32_t columnBlockCount)
    {
        const uint32_t squareBlocks =
            columnBlockCount < INPLACE_PHASE_ROW_BLOCKS ?
                columnBlockCount : INPLACE_PHASE_ROW_BLOCKS;
        AscendC::LocalTensor<T> compact(
            AscendC::TPosition::VECCALC,
            WIDE_PHASE_INPUT_UB_OFFSET,
            WIDE_PHASE_INPUT_ELEMENTS);

        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(compactEvent);
        // LocalTensor-to-LocalTensor DataCopy is emitted on PIPE_V for C220,
        // not on MTE2.  Retire the preceding tail/phase Vector generation
        // before overwriting the compact bank.
        AscendC::PipeBarrier<PIPE_V>();

        // Save the diagonal blocks as a compact sequence and transpose all of
        // them with one repeated descriptor.  The address-table helpers retire
        // each VNCHW before the compact bank is reused below.
        for (uint32_t block = 0U; block < squareBlocks; ++block) {
            CopyInplaceBlock(output, compact, block, block, block);
        }
        AscendC::PipeBarrier<PIPE_V>();
        constexpr uint16_t diagonalDestinationRepeatStride =
            (TILE_ROWS * LAST_TILE_ROWS + TILE_COLS) * TYPE_BYTES /
            DMA_BLOCK_BYTES;
        TransposeCompactB32BlockRun(
            output, compact, 0U, TILE_COLS,
            INPLACE_SCRATCH_FIRST_ELEMENT, squareBlocks,
            TILE_ELEMENTS * TYPE_BYTES / DMA_BLOCK_BYTES,
            diagonalDestinationRepeatStride);

        // Evacuate the complete lower triangle.  Row r is copied as one
        // aligned 16-by-(16*r) rectangle, so the six UB copies and subsequent
        // repeated VNCHW runs replace 42 independent block exchanges.
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t compactOffset = 0U;
        for (uint32_t rowBlock = 1U;
             rowBlock < squareBlocks; ++rowBlock) {
            CopyInplaceLowerBlockRow(
                output, compact, rowBlock, compactOffset);
            compactOffset += rowBlock * TILE_ELEMENTS;
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t rowBlock = 0U;
             rowBlock + 1U < squareBlocks; ++rowBlock) {
            const uint32_t repeatCount = squareBlocks - rowBlock - 1U;
            const uint32_t sourceBase =
                rowBlock * TILE_ROWS * LAST_TILE_ROWS +
                INPLACE_SCRATCH_FIRST_ELEMENT +
                (rowBlock + 1U) * TILE_COLS;
            const uint32_t destinationBase =
                (rowBlock + 1U) * TILE_ROWS * LAST_TILE_ROWS +
                INPLACE_SCRATCH_FIRST_ELEMENT + rowBlock * TILE_COLS;
            TransposeStridedB32BlockRun(
                output, sourceBase, destinationBase, repeatCount);
        }

        compactOffset = 0U;
        constexpr uint16_t rowDestinationRepeatStride =
            TILE_ROWS * LAST_TILE_ROWS * TYPE_BYTES / DMA_BLOCK_BYTES;
        for (uint32_t rowBlock = 1U;
             rowBlock < squareBlocks; ++rowBlock) {
            const uint32_t destinationBase =
                INPLACE_SCRATCH_FIRST_ELEMENT + rowBlock * TILE_COLS;
            TransposeCompactB32BlockRun(
                output, compact, compactOffset,
                rowBlock * TILE_COLS, destinationBase, rowBlock,
                TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES,
                rowDestinationRepeatStride);
            compactOffset += rowBlock * TILE_ELEMENTS;
        }

        // A partially padded final column tile has fewer source columns than
        // phase rows.  Those extra source rows map into blocks outside the
        // source rectangle and can therefore be transposed directly.
        for (uint32_t rowBlock = squareBlocks;
             rowBlock < INPLACE_PHASE_ROW_BLOCKS; ++rowBlock) {
            const uint32_t sourceBase =
                rowBlock * TILE_ROWS * LAST_TILE_ROWS +
                INPLACE_SCRATCH_FIRST_ELEMENT;
            const uint32_t destinationBase =
                INPLACE_SCRATCH_FIRST_ELEMENT + rowBlock * TILE_COLS;
            TransposeStridedB32BlockRun(
                output, sourceBase, destinationBase, columnBlockCount);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(compactEvent);
    }

    __aicore__ inline void LoadLastTwoAxesInputHalf(
        event_t event, uint32_t inputBatchOffset,
        uint32_t rowStart, uint32_t colStart, uint32_t half)
    {
        constexpr uint32_t halfRows = LAST_TILE_ROWS / 2U;
        constexpr uint32_t halfElements = LAST_TILE_ELEMENTS / 2U;
        const uint32_t rowOffset = half * halfRows;
        const uint32_t tileRows =
            height - rowStart < LAST_TILE_ROWS ? height - rowStart : LAST_TILE_ROWS;
        const uint32_t rows = tileRows - rowOffset < halfRows ?
            tileRows - rowOffset : halfRows;
        const uint32_t cols =
            width - colStart < LAST_TILE_COLS ? width - colStart : LAST_TILE_COLS;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset +
            (rowStart + rowOffset) * inputRowStride + colStart;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            INPUT_UB_OFFSET + half * halfElements * TYPE_BYTES,
            halfElements);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES, inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void StoreLastTwoAxesOutput(
        const AscendC::LocalTensor<T>& output,
        uint32_t outputBatchOffset, uint32_t rowStart,
        uint32_t colStart)
    {
        const uint32_t rows =
            height - rowStart < LAST_TILE_ROWS ? height - rowStart : LAST_TILE_ROWS;
        const uint32_t cols =
            width - colStart < LAST_TILE_COLS ? width - colStart : LAST_TILE_COLS;
        const uint32_t outputOffset =
            outputBatchOffset + colStart * outputRowStride + rowStart;
        const uint32_t alignedRows =
            CeilDiv(rows, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((LAST_TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * TYPE_BYTES,
            outputSrcGapBlocks, (outputRowStride - rows) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
    }

    __aicore__ inline void StoreLastTwoAxesAdaptiveOutput(
        const AscendC::LocalTensor<T>& output,
        uint32_t outputBatchOffset, uint32_t rowStart,
        uint32_t colStart, uint32_t tileRows,
        uint32_t tileCols)
    {
        const uint32_t rows =
            height - rowStart < tileRows ?
                height - rowStart : tileRows;
        const uint32_t cols =
            width - colStart < tileCols ?
                width - colStart : tileCols;
        const uint32_t outputOffset =
            outputBatchOffset + colStart * outputRowStride + rowStart;
        const uint32_t alignedRows =
            CeilDiv(rows, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((tileRows - alignedRows) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * TYPE_BYTES,
            outputSrcGapBlocks,
            (outputRowStride - rows) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
    }

    __aicore__ inline void StoreLastTwoAxesOutputColumnChunk(
        const AscendC::LocalTensor<T>& output,
        uint32_t outputBatchOffset, uint32_t rowStart,
        uint32_t colStart, uint32_t firstColumn,
        uint32_t maximumColumns)
    {
        const uint32_t rows =
            height - rowStart < LAST_TILE_ROWS ?
                height - rowStart : LAST_TILE_ROWS;
        const uint32_t tileCols =
            width - colStart < LAST_TILE_COLS ?
                width - colStart : LAST_TILE_COLS;
        const uint32_t remainingCols = tileCols - firstColumn;
        const uint32_t cols =
            remainingCols < maximumColumns ?
                remainingCols : maximumColumns;
        const uint32_t outputOffset = outputBatchOffset +
            (colStart + firstColumn) * outputRowStride + rowStart;
        const uint32_t alignedRows =
            CeilDiv(rows, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((LAST_TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * TYPE_BYTES,
            outputSrcGapBlocks, (outputRowStride - rows) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
    }

    __aicore__ inline void InitAdjacentBatchRectangleLayout()
    {
        InitAdjacentInputBatchLayout();
        rectangleOutputBatchAxis = 0U;
        rectangleOutputBatchPos = 0U;
        for (int32_t i = static_cast<int32_t>(dims) - 1; i >= 0; --i) {
            if (static_cast<uint32_t>(i) == inputLastOutputPos ||
                static_cast<uint32_t>(i) == dims - 1U) {
                continue;
            }
            const uint32_t axis = perm[i];
            if (shape[axis] > 1U) {
                rectangleOutputBatchAxis = axis;
                rectangleOutputBatchPos = static_cast<uint32_t>(i);
                break;
            }
        }
        rectangleOutputAxisExtent = shape[rectangleOutputBatchAxis];
        rectangleOutputAxisTileCount =
            CeilDiv(rectangleOutputAxisExtent, OUTPUT_BATCH_GROUP_COUNT);
        rectangleMiddleExtent = 1U;
        for (uint32_t i = pairBatchOutputPos + 1U;
             i < rectangleOutputBatchPos; ++i) {
            if (i == inputLastOutputPos || i == dims - 1U) {
                continue;
            }
            rectangleMiddleExtent *= shape[perm[i]];
        }
        const uint32_t outerExtent = batchCount /
            (pairAxisExtent * rectangleMiddleExtent *
             rectangleOutputAxisExtent);
        rectangleGroupCount =
            outerExtent * pairAxisTileCount * rectangleMiddleExtent *
            rectangleOutputAxisTileCount;
        rectangleOutputAxisInputStride =
            inputStride[rectangleOutputBatchAxis];
        rectanglePackedRowStride =
            CeilDiv(OUTPUT_BATCH_GROUP_COUNT * height, DMA_BLOCK_ELEMENTS) *
            DMA_BLOCK_ELEMENTS;
        rectanglePackedElements =
            INPUT_BATCH_GROUP_COUNT * width * rectanglePackedRowStride;
    }

    __aicore__ inline void BuildRectangleGatherOffsets()
    {
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            RECTANGLE_OFFSETS_UB_OFFSET, RECTANGLE_OFFSET_TILE_ELEMENTS);
        const uint32_t rows = TWO_PHASE_RECTANGLE_PACKED_OUTPUT ?
            RECTANGLE_GATHER_CHUNK_ROWS :
            INPUT_BATCH_GROUP_COUNT * width;
        for (uint32_t outputPair = 0U;
             outputPair < OUTPUT_BATCH_GROUP_COUNT; ++outputPair) {
            for (uint32_t element = 0U; element < height; ++element) {
                offsets.SetValue(
                    outputPair * height + element,
                    (outputPair * LAST_TILE_ROWS + element) * TYPE_BYTES);
            }
        }
        for (uint32_t element = OUTPUT_BATCH_GROUP_COUNT * height;
             element < rectanglePackedRowStride; ++element) {
            offsets.SetValue(element, 0U);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(RECTANGLE_PATTERN_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(RECTANGLE_PATTERN_EVENT));

        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.template ReinterpretCast<int32_t>();
        uint32_t generatedRows = 1U;
        while (generatedRows < rows) {
            const uint32_t remainingRows = rows - generatedRows;
            const uint32_t copiedRows = remainingRows < generatedRows ?
                remainingRows : generatedRows;
            AscendC::Adds(
                signedOffsets[generatedRows * rectanglePackedRowStride],
                signedOffsets,
                static_cast<int32_t>(generatedRows *
                                     RECTANGLE_OUTPUT_COLS * TYPE_BYTES),
                static_cast<int32_t>(copiedRows * rectanglePackedRowStride));
            AscendC::PipeBarrier<PIPE_V>();
            generatedRows += copiedRows;
        }
    }

    __aicore__ inline void LoadCombinedSixOutputRectangle(
        uint32_t inputBatchOffset, uint32_t inputPairCount,
        uint32_t outputPairCount)
    {
        constexpr event_t inputEvent = static_cast<event_t>(0);
        const uint32_t cols = inputPairCount * width;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((COMBINED_SIX_INPUT_ROW_STRIDE - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            COMBINED_SIX_INPUT_UB_OFFSET,
            COMBINED_SIX_INPUT_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(height), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        for (uint32_t outputPair = 0U;
             outputPair < outputPairCount; ++outputPair) {
            const uint32_t localOffset =
                outputPair * height * COMBINED_SIX_INPUT_ROW_STRIDE;
            const uint32_t globalOffset = inputBatchOffset +
                outputPair * rectangleOutputAxisInputStride;
            AscendC::DataCopyPad(
                input[localOffset], xGm[globalOffset], inputParams, pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEvent);
    }

    __aicore__ inline void ComputeCombinedSixOutputRectangle(
        uint32_t inputPairCount, uint32_t outputPairCount)
    {
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            COMBINED_SIX_INPUT_UB_OFFSET,
            COMBINED_SIX_INPUT_ELEMENTS);
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            COMBINED_SIX_OUTPUT_UB_OFFSET,
            COMBINED_SIX_OUTPUT_ELEMENTS);
        const uint32_t inputRows = outputPairCount * height;
        const uint32_t inputCols = inputPairCount * width;
        const uint32_t rowBlockCount = CeilDiv(inputRows, TILE_ROWS);
        const uint32_t columnBlockCount = CeilDiv(inputCols, TILE_COLS);

        uint64_t dstList[16];
        uint64_t srcList[16];
        constexpr uint16_t dstRepeatStride =
            TILE_COLS * COMBINED_SIX_OUTPUT_ROW_STRIDE * TYPE_BYTES /
            DMA_BLOCK_BYTES;
        constexpr uint16_t srcRepeatStride =
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(columnBlockCount),
            dstRepeatStride, srcRepeatStride);
        for (uint32_t rowBlock = 0U;
             rowBlock < rowBlockCount; ++rowBlock) {
            const uint32_t inputRow = rowBlock * TILE_ROWS;
            const uint32_t outputBase = rowBlock * TILE_ROWS;
            for (uint32_t halfColumn = 0U;
                 halfColumn < 2U; ++halfColumn) {
                for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                    const uint32_t sourceRow = inputRow + row < inputRows ?
                        inputRow + row : 0U;
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[sourceRow * COMBINED_SIX_INPUT_ROW_STRIDE +
                              halfColumn * 8U].GetPhyAddr());
                }
                for (uint32_t outputRow = 0U;
                     outputRow < 8U; ++outputRow) {
                    const uint32_t row = halfColumn * 8U + outputRow;
                    dstList[2U * outputRow] = reinterpret_cast<uint64_t>(
                        output[outputBase +
                               row * COMBINED_SIX_OUTPUT_ROW_STRIDE]
                            .GetPhyAddr());
                    dstList[2U * outputRow + 1U] =
                        reinterpret_cast<uint64_t>(
                            output[outputBase +
                                   row * COMBINED_SIX_OUTPUT_ROW_STRIDE + 8U]
                                .GetPhyAddr());
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void StoreCombinedSixOutputRectangle(
        uint32_t outputBatchOffset, uint32_t inputPairCount,
        uint32_t outputPairCount)
    {
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            COMBINED_SIX_OUTPUT_UB_OFFSET,
            COMBINED_SIX_OUTPUT_ELEMENTS);
        const uint32_t outputCols = outputPairCount * height;
        const uint32_t alignedOutputCols =
            CeilDiv(outputCols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((COMBINED_SIX_OUTPUT_ROW_STRIDE - alignedOutputCols) *
             TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(width), outputCols * TYPE_BYTES,
            outputSrcGapBlocks,
            (outputRowStride - outputCols) * TYPE_BYTES, 0U};
        for (uint32_t inputPair = 0U;
             inputPair < inputPairCount; ++inputPair) {
            const uint32_t localOffset =
                inputPair * width * COMBINED_SIX_OUTPUT_ROW_STRIDE;
            const uint32_t globalOffset =
                outputBatchOffset + inputPair * pairOutputStride;
            AscendC::DataCopyPad(
                zGm[globalOffset], output[localOffset], outputParams);
        }
    }

    __aicore__ inline void LoadSevenInputSixOutputRectangleFull(
        uint32_t inputBatchOffset)
    {
        constexpr event_t inputEvent = static_cast<event_t>(0);
        const uint32_t cols = INPUT_BATCH_GROUP_COUNT * width;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((SEVEN_BY_SIX_INPUT_ROW_STRIDE - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            SEVEN_BY_SIX_INPUT_UB_OFFSET,
            SEVEN_BY_SIX_INPUT_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(height), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
#pragma unroll
        for (uint32_t outputPair = 0U;
             outputPair < OUTPUT_BATCH_GROUP_COUNT; ++outputPair) {
            const uint32_t localOffset = outputPair * height *
                SEVEN_BY_SIX_INPUT_ROW_STRIDE;
            const uint32_t globalOffset = inputBatchOffset +
                outputPair * rectangleOutputAxisInputStride;
            AscendC::DataCopyPad(
                input[localOffset], xGm[globalOffset], inputParams, pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEvent);
    }

    template <uint32_t PHASE_START>
    __aicore__ inline void ComputeSevenInputSixOutputRaw(
        uint8_t repeatTimes)
    {
        static_assert(TYPE_BYTES == 4U);
        constexpr uint64_t inputRowBytes =
            SEVEN_BY_SIX_INPUT_ROW_STRIDE * TYPE_BYTES;
        constexpr uint64_t outputRowBytes =
            SEVEN_BY_SIX_OUTPUT_ROW_STRIDE * TYPE_BYTES;
        constexpr uint16_t destinationRepeatStride =
            TILE_COLS * SEVEN_BY_SIX_OUTPUT_ROW_STRIDE * TYPE_BYTES /
            DMA_BLOCK_BYTES;
        constexpr uint16_t sourceRepeatStride =
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, repeatTimes,
            destinationRepeatStride, sourceRepeatStride);
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            SEVEN_BY_SIX_INPUT_UB_OFFSET,
            SEVEN_BY_SIX_INPUT_ELEMENTS);
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            SEVEN_BY_SIX_OUTPUT_UB_OFFSET,
            PHASE_START == 0U ? SEVEN_BY_SIX_OUTPUT_HEAD_ELEMENTS :
                SEVEN_BY_SIX_OUTPUT_BANK0_ELEMENTS);
        const uint32_t inputRows = OUTPUT_BATCH_GROUP_COUNT * height;
        const uint32_t rowBlockCount = CeilDiv(inputRows, TILE_ROWS);
        const uint64_t inputBase =
            reinterpret_cast<uint64_t>(input.GetPhyAddr());
        const uint64_t outputBase =
            reinterpret_cast<uint64_t>(output.GetPhyAddr());
        uint64_t dstList[16];
        uint64_t srcList[16];

        for (uint32_t rowBlock = 0U;
             rowBlock < rowBlockCount; ++rowBlock) {
            const uint32_t inputRow = rowBlock * TILE_ROWS;
            const uint64_t sourceBlock = inputBase +
                inputRow * inputRowBytes + PHASE_START * TYPE_BYTES;
            const uint64_t destinationBlock = outputBase +
                inputRow * TYPE_BYTES;
            const bool completeRowBlock =
                inputRow + TILE_ROWS <= inputRows;
            for (uint32_t halfColumn = 0U;
                 halfColumn < 2U; ++halfColumn) {
                const uint64_t source = sourceBlock +
                    halfColumn * 8U * TYPE_BYTES;
                const uint64_t safeSource = inputBase +
                    PHASE_START * TYPE_BYTES +
                    halfColumn * 8U * TYPE_BYTES;
                if (completeRowBlock) {
                    srcList[0] = source;
                    srcList[1] = source + inputRowBytes;
                    srcList[2] = source + 2U * inputRowBytes;
                    srcList[3] = source + 3U * inputRowBytes;
                    srcList[4] = source + 4U * inputRowBytes;
                    srcList[5] = source + 5U * inputRowBytes;
                    srcList[6] = source + 6U * inputRowBytes;
                    srcList[7] = source + 7U * inputRowBytes;
                    srcList[8] = source + 8U * inputRowBytes;
                    srcList[9] = source + 9U * inputRowBytes;
                    srcList[10] = source + 10U * inputRowBytes;
                    srcList[11] = source + 11U * inputRowBytes;
                    srcList[12] = source + 12U * inputRowBytes;
                    srcList[13] = source + 13U * inputRowBytes;
                    srcList[14] = source + 14U * inputRowBytes;
                    srcList[15] = source + 15U * inputRowBytes;
                } else {
#pragma unroll
                    for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                        srcList[row] = inputRow + row < inputRows ?
                            source + row * inputRowBytes : safeSource;
                    }
                }

                const uint64_t destination = destinationBlock +
                    halfColumn * 8U * outputRowBytes;
                dstList[0] = destination;
                dstList[1] = destination + 8U * TYPE_BYTES;
                dstList[2] = destination + outputRowBytes;
                dstList[3] = destination + outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[4] = destination + 2U * outputRowBytes;
                dstList[5] = destination + 2U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[6] = destination + 3U * outputRowBytes;
                dstList[7] = destination + 3U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[8] = destination + 4U * outputRowBytes;
                dstList[9] = destination + 4U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[10] = destination + 5U * outputRowBytes;
                dstList[11] = destination + 5U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[12] = destination + 6U * outputRowBytes;
                dstList[13] = destination + 6U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[14] = destination + 7U * outputRowBytes;
                dstList[15] = destination + 7U * outputRowBytes +
                    8U * TYPE_BYTES;
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeSevenInputSixOutputPairedHead()
    {
        ComputeSevenInputSixOutputRaw<0U>(
            static_cast<uint8_t>(
                SEVEN_BY_SIX_OUTPUT_HEAD_ROWS / TILE_COLS));
    }

    __aicore__ inline void ComputeSevenInputSixOutputTail()
    {
        const uint32_t remainingColumns =
            INPUT_BATCH_GROUP_COUNT * width -
                SEVEN_BY_SIX_OUTPUT_HEAD_ROWS;
        const uint8_t repeats = static_cast<uint8_t>(
            CeilDiv(remainingColumns, TILE_COLS));
        ComputeSevenInputSixOutputRaw<SEVEN_BY_SIX_OUTPUT_HEAD_ROWS>(
            repeats);
    }

    template <uint32_t BANK, uint32_t PHASE_START,
              uint32_t PHASE_CAPACITY>
    __aicore__ inline void StoreSevenInputSixOutputPhaseFull(
        uint32_t outputBatchOffset)
    {
        static_assert(BANK < PIPELINE_BUFFER_COUNT);
        constexpr uint32_t bankOffset =
            BANK == 0U ? SEVEN_BY_SIX_OUTPUT_UB_OFFSET :
                SEVEN_BY_SIX_OUTPUT_UB_OFFSET +
                    SEVEN_BY_SIX_OUTPUT_BANK0_BYTES;
        constexpr uint32_t bankElements =
            BANK == 0U ? SEVEN_BY_SIX_OUTPUT_BANK0_ELEMENTS :
                SEVEN_BY_SIX_OUTPUT_BANK1_ELEMENTS;
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            bankOffset, bankElements);
        const uint32_t outputCols = OUTPUT_BATCH_GROUP_COUNT * height;
        const uint32_t alignedOutputCols =
            CeilDiv(outputCols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputCols = INPUT_BATCH_GROUP_COUNT * width;
        const uint32_t phaseEnd =
            PHASE_START + PHASE_CAPACITY < inputCols ?
                PHASE_START + PHASE_CAPACITY : inputCols;
        const uint32_t outputSrcGapBlocks =
            ((SEVEN_BY_SIX_OUTPUT_ROW_STRIDE - alignedOutputCols) *
             TYPE_BYTES) / DMA_BLOCK_BYTES;

#pragma unroll
        for (uint32_t inputPair = 0U;
             inputPair < INPUT_BATCH_GROUP_COUNT; ++inputPair) {
            const uint32_t pairStart = inputPair * width;
            const uint32_t pairEnd = pairStart + width;
            const uint32_t copyStart = PHASE_START > pairStart ?
                PHASE_START : pairStart;
            const uint32_t copyEnd = phaseEnd < pairEnd ?
                phaseEnd : pairEnd;
            if (copyStart >= copyEnd) {
                continue;
            }
            const uint32_t rows = copyEnd - copyStart;
            const uint32_t localRow = copyStart - PHASE_START;
            const uint32_t outputRow = copyStart - pairStart;
            AscendC::DataCopyExtParams outputParams{
                static_cast<uint16_t>(rows), outputCols * TYPE_BYTES,
                outputSrcGapBlocks,
                (outputRowStride - outputCols) * TYPE_BYTES, 0U};
            const uint32_t globalOffset = outputBatchOffset +
                inputPair * pairOutputStride +
                outputRow * outputRowStride;
            AscendC::DataCopyPad(
                zGm[globalOffset],
                output[localRow * SEVEN_BY_SIX_OUTPUT_ROW_STRIDE],
                outputParams);
        }
    }

    __aicore__ inline void BuildStreamedSixOutputOffsets()
    {
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            SIX_OUTPUT_OFFSETS_UB_OFFSET, SIX_OUTPUT_OFFSET_ELEMENTS);
        for (uint32_t outputPair = 0U;
             outputPair < OUTPUT_BATCH_GROUP_COUNT; ++outputPair) {
            for (uint32_t element = 0U; element < height; ++element) {
                offsets.SetValue(
                    outputPair * height + element,
                    (outputPair * LAST_TILE_ROWS + element) * TYPE_BYTES);
            }
        }
        for (uint32_t element = OUTPUT_BATCH_GROUP_COUNT * height;
             element < SIX_OUTPUT_PACKED_ROW_STRIDE; ++element) {
            offsets.SetValue(element, 0U);
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(RECTANGLE_PATTERN_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(RECTANGLE_PATTERN_EVENT));

        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.template ReinterpretCast<int32_t>();
        uint32_t generatedRows = 1U;
        while (generatedRows < SIX_OUTPUT_GATHER_CHUNK_ROWS) {
            const uint32_t remainingRows =
                SIX_OUTPUT_GATHER_CHUNK_ROWS - generatedRows;
            const uint32_t copiedRows = remainingRows < generatedRows ?
                remainingRows : generatedRows;
            AscendC::Adds(
                signedOffsets[
                    generatedRows * SIX_OUTPUT_PACKED_ROW_STRIDE],
                signedOffsets,
                static_cast<int32_t>(
                    generatedRows * SIX_OUTPUT_OUTPUT_COLS * TYPE_BYTES),
                static_cast<int32_t>(
                    copiedRows * SIX_OUTPUT_PACKED_ROW_STRIDE));
            AscendC::PipeBarrier<PIPE_V>();
            generatedRows += copiedRows;
        }
    }

    __aicore__ inline void LoadStreamedSixOutputPair(
        uint32_t bank, uint32_t inputBatchOffset,
        uint32_t inputPairCount, uint32_t outputPair)
    {
        const event_t event = static_cast<event_t>(bank);
        const uint32_t cols = inputPairCount * width;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            bank * SIX_OUTPUT_INPUT_BUFFER_BYTES, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(height), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols),
            static_cast<T>(0)};
        AscendC::DataCopyPad(
            input,
            xGm[inputBatchOffset +
                outputPair * rectangleOutputAxisInputStride],
            inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void ComputeStreamedSixOutputPair(
        uint32_t bank, uint32_t outputPair)
    {
        const event_t event = static_cast<event_t>(bank);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            bank * SIX_OUTPUT_INPUT_BUFFER_BYTES, LAST_TILE_ELEMENTS);
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            SIX_OUTPUT_OUTPUT_UB_OFFSET, SIX_OUTPUT_OUTPUT_ELEMENTS);

        uint64_t dstList[16];
        uint64_t srcList[16];
        constexpr uint16_t dstRepeatStride =
            TILE_COLS * SIX_OUTPUT_OUTPUT_COLS * TYPE_BYTES /
            DMA_BLOCK_BYTES;
        constexpr uint16_t srcRepeatStride =
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(LAST_TILE_BLOCKS),
            dstRepeatStride, srcRepeatStride);
        for (uint32_t rowBlock = 0U;
             rowBlock < LAST_TILE_ROW_BLOCKS; ++rowBlock) {
            const uint32_t inputRow = rowBlock * TILE_ROWS;
            const uint32_t outputBase =
                outputPair * LAST_TILE_ROWS + rowBlock * TILE_ROWS;
            for (uint32_t halfColumn = 0U;
                 halfColumn < 2U; ++halfColumn) {
                for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(inputRow + row) * LAST_TILE_COLS +
                              halfColumn * 8U].GetPhyAddr());
                }
                for (uint32_t outputRow = 0U;
                     outputRow < 8U; ++outputRow) {
                    const uint32_t row = halfColumn * 8U + outputRow;
                    dstList[2U * outputRow] = reinterpret_cast<uint64_t>(
                        output[outputBase +
                               row * SIX_OUTPUT_OUTPUT_COLS].GetPhyAddr());
                    dstList[2U * outputRow + 1U] =
                        reinterpret_cast<uint64_t>(
                            output[outputBase +
                                   row * SIX_OUTPUT_OUTPUT_COLS + 8U]
                                .GetPhyAddr());
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeStreamedSixOutputRectangle(
        uint32_t inputBatchOffset, uint32_t inputPairCount,
        uint32_t outputPairCount)
    {
        ComputeStreamedSixOutputPair(0U, 0U);
        if (outputPairCount > 2U) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                static_cast<event_t>(0));
        }

        if (outputPairCount > 1U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(
                static_cast<event_t>(1));
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                static_cast<event_t>(1));
            LoadStreamedSixOutputPair(
                1U, inputBatchOffset, inputPairCount, 1U);
        }
        for (uint32_t outputPair = 1U;
             outputPair < outputPairCount; ++outputPair) {
            const uint32_t bank = outputPair & 1U;
            if (outputPair + 1U < outputPairCount) {
                LoadStreamedSixOutputPair(
                    bank ^ 1U, inputBatchOffset, inputPairCount,
                    outputPair + 1U);
            }
            ComputeStreamedSixOutputPair(bank, outputPair);
            if (outputPair + 2U < outputPairCount) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                    static_cast<event_t>(bank));
            }
        }
    }

    __aicore__ inline void StoreStreamedSixOutputRectangle(
        uint32_t outputBatchOffset, uint32_t inputPairCount,
        uint32_t outputPairCount)
    {
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            SIX_OUTPUT_OUTPUT_UB_OFFSET, SIX_OUTPUT_OUTPUT_ELEMENTS);
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            SIX_OUTPUT_OFFSETS_UB_OFFSET, SIX_OUTPUT_OFFSET_ELEMENTS);
        const uint32_t outputCols = outputPairCount * height;
        const uint32_t alignedOutputCols =
            CeilDiv(outputCols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((SIX_OUTPUT_PACKED_ROW_STRIDE - alignedOutputCols) *
             TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(width), outputCols * TYPE_BYTES,
            outputSrcGapBlocks,
            (outputRowStride - outputCols) * TYPE_BYTES, 0U};

        for (uint32_t inputPair = 0U;
             inputPair < inputPairCount; ++inputPair) {
            const uint32_t bank = inputPair & 1U;
            const event_t event = static_cast<event_t>(bank);
            if (inputPair >= PIPELINE_BUFFER_COUNT) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            }
            AscendC::LocalTensor<T> packed(
                AscendC::TPosition::VECOUT,
                bank * SIX_OUTPUT_INPUT_BUFFER_BYTES,
                LAST_TILE_ELEMENTS);
            for (uint32_t rowStart = 0U; rowStart < width;
                 rowStart += SIX_OUTPUT_GATHER_CHUNK_ROWS) {
                const uint32_t remainingRows = width - rowStart;
                const uint32_t chunkRows =
                    remainingRows < SIX_OUTPUT_GATHER_CHUNK_ROWS ?
                        remainingRows : SIX_OUTPUT_GATHER_CHUNK_ROWS;
                AscendC::Gather(
                    packed[rowStart * SIX_OUTPUT_PACKED_ROW_STRIDE],
                    output[(inputPair * width + rowStart) *
                           SIX_OUTPUT_OUTPUT_COLS],
                    offsets, 0U,
                    chunkRows * SIX_OUTPUT_PACKED_ROW_STRIDE);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::DataCopyPad(
                zGm[outputBatchOffset + inputPair * pairOutputStride],
                packed, outputParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline void DecodeRectangleGroup(
        uint32_t group, uint32_t& inputOffset, uint32_t& outputOffset,
        uint32_t& inputPairCount, uint32_t& outputPairCount) const
    {
        const uint32_t outputAxisTile =
            group % rectangleOutputAxisTileCount;
        uint32_t remainder = group / rectangleOutputAxisTileCount;
        const uint32_t middleCoordinate =
            remainder % rectangleMiddleExtent;
        remainder /= rectangleMiddleExtent;
        const uint32_t inputAxisTile = remainder % pairAxisTileCount;
        const uint32_t outerCoordinate = remainder / pairAxisTileCount;
        const uint32_t inputAxisCoordinate =
            inputAxisTile * INPUT_BATCH_GROUP_COUNT;
        const uint32_t outputAxisCoordinate =
            outputAxisTile * OUTPUT_BATCH_GROUP_COUNT;
        const uint32_t outputBatch =
            (((outerCoordinate * pairAxisExtent + inputAxisCoordinate) *
               rectangleMiddleExtent + middleCoordinate) *
              rectangleOutputAxisExtent) + outputAxisCoordinate;
        uint32_t coordinates[MAXDIM] = {};
        DecodeBatchOffsets(outputBatch, coordinates, inputOffset, outputOffset);
        const uint32_t inputRemaining =
            pairAxisExtent - inputAxisCoordinate;
        inputPairCount = inputRemaining < INPUT_BATCH_GROUP_COUNT ?
            inputRemaining : INPUT_BATCH_GROUP_COUNT;
        const uint32_t outputRemaining =
            rectangleOutputAxisExtent - outputAxisCoordinate;
        outputPairCount = outputRemaining < OUTPUT_BATCH_GROUP_COUNT ?
            outputRemaining : OUTPUT_BATCH_GROUP_COUNT;
    }

    __aicore__ inline void LoadAdjacentBatchRectangle(
        uint32_t buffer, uint32_t inputBatchOffset,
        uint32_t inputPairCount, uint32_t outputPairCount)
    {
        const event_t event = static_cast<event_t>(buffer);
        const uint32_t cols = inputPairCount * width;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(height), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES, inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        if constexpr (TWO_PHASE_RECTANGLE_PACKED_OUTPUT) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        }
        for (uint32_t outputPair = 0U;
             outputPair < outputPairCount; ++outputPair) {
            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                buffer * RECTANGLE_INPUT_BUFFER_BYTES +
                    outputPair * LAST_TILE_ELEMENTS * TYPE_BYTES,
                LAST_TILE_ELEMENTS);
            const uint32_t offset = inputBatchOffset +
                outputPair * rectangleOutputAxisInputStride;
            AscendC::DataCopyPad(input, xGm[offset], inputParams, pad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void ComputeLastTwoB32Rectangle(
        const AscendC::LocalTensor<T>& output,
        const AscendC::LocalTensor<T>& input,
        uint32_t outputColumnOffset)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        constexpr uint16_t dstRepeatStride =
            TILE_COLS * RECTANGLE_OUTPUT_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        constexpr uint16_t srcRepeatStride =
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(LAST_TILE_BLOCKS),
            dstRepeatStride, srcRepeatStride);
        for (uint32_t rowBlock = 0U;
             rowBlock < LAST_TILE_ROW_BLOCKS; ++rowBlock) {
            const uint32_t inputRow = rowBlock * TILE_ROWS;
            const uint32_t outputBase =
                outputColumnOffset + rowBlock * TILE_ROWS;
            for (uint32_t halfColumn = 0U; halfColumn < 2U; ++halfColumn) {
                for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(inputRow + row) * LAST_TILE_COLS +
                              halfColumn * 8U].GetPhyAddr());
                }
                for (uint32_t outputRow = 0U;
                     outputRow < 8U; ++outputRow) {
                    const uint32_t row = halfColumn * 8U + outputRow;
                    dstList[2U * outputRow] = reinterpret_cast<uint64_t>(
                        output[outputBase +
                               row * RECTANGLE_OUTPUT_COLS].GetPhyAddr());
                    dstList[2U * outputRow + 1U] =
                        reinterpret_cast<uint64_t>(
                            output[outputBase +
                                   row * RECTANGLE_OUTPUT_COLS + 8U].GetPhyAddr());
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeStoreAdjacentBatchRectangle(
        uint32_t buffer, uint32_t outputBatchOffset,
        uint32_t inputPairCount, uint32_t outputPairCount)
    {
        const event_t event = static_cast<event_t>(buffer);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            RECTANGLE_OUTPUT_UB_OFFSET +
                (SHARED_RECTANGLE_BUFFERS ? 0U : buffer) *
                    RECTANGLE_OUTPUT_TILE_BYTES,
            RECTANGLE_OUTPUT_ELEMENTS);
        if constexpr (!SHARED_RECTANGLE_BUFFERS) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
        for (uint32_t outputPair = 0U;
             outputPair < outputPairCount; ++outputPair) {
            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                buffer * RECTANGLE_INPUT_BUFFER_BYTES +
                    outputPair * LAST_TILE_ELEMENTS * TYPE_BYTES,
                LAST_TILE_ELEMENTS);
            ComputeLastTwoB32Rectangle(
                output, input, outputPair * LAST_TILE_ROWS);
        }
        AscendC::LocalTensor<T> packed(
            AscendC::TPosition::VECOUT,
            TWO_PHASE_RECTANGLE_PACKED_OUTPUT ?
                buffer * RECTANGLE_INPUT_BUFFER_BYTES :
                RECTANGLE_PACKED_UB_OFFSET +
                    (SHARED_RECTANGLE_BUFFERS ? 0U : buffer) *
                        RECTANGLE_PACKED_TILE_BYTES,
            RECTANGLE_PACKED_TILE_ELEMENTS);
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            RECTANGLE_OFFSETS_UB_OFFSET, RECTANGLE_OFFSET_TILE_ELEMENTS);
        const event_t outputEvent = SHARED_RECTANGLE_BUFFERS ?
            static_cast<event_t>(SHARED_RECTANGLE_EVENT) : event;
        if constexpr (SHARED_RECTANGLE_BUFFERS &&
                      !TWO_PHASE_RECTANGLE_PACKED_OUTPUT) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
        }
        const uint32_t outputCols = outputPairCount * height;
        const uint32_t alignedOutputCols =
            CeilDiv(outputCols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((rectanglePackedRowStride - alignedOutputCols) * TYPE_BYTES) /
            DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(width), outputCols * TYPE_BYTES,
            outputSrcGapBlocks,
            (outputRowStride - outputCols) * TYPE_BYTES, 0U};
        if constexpr (TWO_PHASE_RECTANGLE_PACKED_OUTPUT) {
            // The packed output aliases the consumed input bank.  Gather it
            // in bounded source-address windows, then publish all four input
            // coordinates through one uninterrupted MTE3 phase.  This keeps
            // the 4x4 MTE2 reduction without draining MTE3 halfway through
            // the rectangle.
            const uint32_t packedRows = inputPairCount * width;
            for (uint32_t packedRowStart = 0U;
                 packedRowStart < packedRows;
                 packedRowStart += RECTANGLE_GATHER_CHUNK_ROWS) {
                const uint32_t remainingRows = packedRows - packedRowStart;
                const uint32_t chunkRows =
                    remainingRows < RECTANGLE_GATHER_CHUNK_ROWS ?
                        remainingRows : RECTANGLE_GATHER_CHUNK_ROWS;
                AscendC::Gather(
                    packed[packedRowStart * rectanglePackedRowStride],
                    output[packedRowStart * RECTANGLE_OUTPUT_COLS],
                    offsets, 0U,
                    chunkRows * rectanglePackedRowStride);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
            for (uint32_t inputPair = 0U;
                 inputPair < inputPairCount; ++inputPair) {
                const uint32_t localOffset =
                    inputPair * width * rectanglePackedRowStride;
                const uint32_t globalOffset =
                    outputBatchOffset + inputPair * pairOutputStride;
                AscendC::DataCopyPad(
                    zGm[globalOffset], packed[localOffset], outputParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            if constexpr (!SHARED_RECTANGLE_BUFFERS) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
            }
        } else {
            AscendC::Gather(
                packed, output, offsets, 0U, rectanglePackedElements);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
            for (uint32_t inputPair = 0U;
                 inputPair < inputPairCount; ++inputPair) {
                const uint32_t localOffset =
                    inputPair * width * rectanglePackedRowStride;
                const uint32_t globalOffset =
                    outputBatchOffset + inputPair * pairOutputStride;
                AscendC::DataCopyPad(
                    zGm[globalOffset], packed[localOffset], outputParams);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
        }
    }

    __aicore__ inline void InitAdjacentInputBatchLayout()
    {
        const uint32_t rowAxis = perm[dims - 1U];
        pairBatchAxis = 0U;
        for (int32_t axis = static_cast<int32_t>(dims) - 2; axis >= 0; --axis) {
            if (static_cast<uint32_t>(axis) != rowAxis && shape[axis] > 1U) {
                pairBatchAxis = static_cast<uint32_t>(axis);
                break;
            }
        }
        pairBatchOutputPos = 0U;
        for (uint32_t i = 0U; i < dims; ++i) {
            if (perm[i] == pairBatchAxis) {
                pairBatchOutputPos = i;
                break;
            }
        }
        pairAxisExtent = shape[pairBatchAxis];
        pairAxisTileCount =
            CeilDiv(pairAxisExtent, INPUT_BATCH_GROUP_COUNT);
        pairInnerExtent = 1U;
        for (uint32_t i = pairBatchOutputPos + 1U; i < dims; ++i) {
            if (i == inputLastOutputPos || i == dims - 1U) {
                continue;
            }
            pairInnerExtent *= shape[perm[i]];
        }
        const uint32_t outerExtent =
            batchCount / (pairAxisExtent * pairInnerExtent);
        pairGroupCount = outerExtent * pairAxisTileCount * pairInnerExtent;
        pairOutputStride = outputStride[pairBatchOutputPos];
    }

    __aicore__ inline void DecodePairGroup(
        uint32_t group, uint32_t& inputOffset, uint32_t& outputOffset,
        uint32_t& pairCount) const
    {
        const uint32_t inner = group % pairInnerExtent;
        const uint32_t axisGroup = group / pairInnerExtent;
        const uint32_t axisTile = axisGroup % pairAxisTileCount;
        const uint32_t outer = axisGroup / pairAxisTileCount;
        const uint32_t axisCoordinate =
            axisTile * INPUT_BATCH_GROUP_COUNT;
        const uint32_t outputBatch =
            outer * pairAxisExtent * pairInnerExtent +
            axisCoordinate * pairInnerExtent + inner;
        uint32_t coordinates[MAXDIM] = {};
        DecodeBatchOffsets(outputBatch, coordinates, inputOffset, outputOffset);
        const uint32_t remaining = pairAxisExtent - axisCoordinate;
        pairCount = remaining < INPUT_BATCH_GROUP_COUNT ?
            remaining : INPUT_BATCH_GROUP_COUNT;
    }

    __aicore__ inline void LoadAdjacentInputBatchPair(
        uint32_t buffer, uint32_t inputBatchOffset, uint32_t pairCount)
    {
        const event_t event = static_cast<event_t>(buffer);
        const uint32_t cols = pairCount * width;
        const uint32_t alignedCols =
            CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            buffer * LAST_TILE_ELEMENTS * TYPE_BYTES, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(height), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES, inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputBatchOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void ComputeStoreAdjacentInputBatchPair(
        uint32_t buffer, uint32_t outputBatchOffset, uint32_t pairCount)
    {
        const event_t event = static_cast<event_t>(buffer);
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            buffer * LAST_TILE_ELEMENTS * TYPE_BYTES, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            PIPELINED_OUTPUT_UB_OFFSET +
                buffer * LAST_TILE_ELEMENTS * TYPE_BYTES,
            LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        ComputeLastTwoB32(output, input);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

        const uint32_t alignedRows =
            CeilDiv(height, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((LAST_TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(width), height * TYPE_BYTES,
            outputSrcGapBlocks, (outputRowStride - height) * TYPE_BYTES, 0U};
        for (uint32_t pair = 0U; pair < pairCount; ++pair) {
            const uint32_t localOffset = pair * width * LAST_TILE_ROWS;
            const uint32_t globalOffset =
                outputBatchOffset + pair * pairOutputStride;
            AscendC::DataCopyPad(
                zGm[globalOffset], output[localOffset], outputParams);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
    }

    __aicore__ inline void ProcessGeneralSingleTilePrefetched()
    {
        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        const uint32_t batchesPerBlock = batchCount / blockNum;
        const uint32_t extraBlocks = batchCount - batchesPerBlock * blockNum;
        const uint32_t batchBegin = blockIdx * batchesPerBlock +
            (blockIdx < extraBlocks ? blockIdx : extraBlocks);
        const uint32_t batchEnd = batchBegin + batchesPerBlock +
            (blockIdx < extraBlocks ? 1U : 0U);
        uint32_t coordinates[MAXDIM] = {};
        uint32_t currentInputOffset = 0U;
        uint32_t currentOutputOffset = 0U;
        DecodeBatchOffsets(batchBegin, coordinates,
                           currentInputOffset, currentOutputOffset);
        LoadGeneralSingleTile(0U, currentInputOffset);

        for (uint32_t batch = batchBegin; batch < batchEnd; ++batch) {
            const uint32_t buffer = (batch - batchBegin) & 1U;
            uint32_t nextInputOffset = currentInputOffset;
            uint32_t nextOutputOffset = currentOutputOffset;
            if (batch + 1U < batchEnd) {
                AdvanceBatchOffsets(coordinates, nextInputOffset, nextOutputOffset);
                LoadGeneralSingleTile(buffer ^ 1U, nextInputOffset);
            }
            ComputeStoreGeneralSingleTile(buffer, currentOutputOffset);
            currentInputOffset = nextInputOffset;
            currentOutputOffset = nextOutputOffset;
        }

        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline void ProcessGeneralLastTwoAxesPipelined()
    {
        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        uint32_t progress = 0U;
        uint32_t batchBegin = blockIdx;
        uint32_t batchEnd = batchCount;
        uint32_t batchStep = blockNum;
        if constexpr (TYPE_BYTES == 4U && LAST_TILE_ROW_BLOCKS == 2U &&
                      LAST_TILE_BLOCKS == 2U) {
            const uint32_t batchesPerBlock = batchCount / blockNum;
            const uint32_t extraBlocks = batchCount - batchesPerBlock * blockNum;
            batchBegin = blockIdx * batchesPerBlock +
                         (blockIdx < extraBlocks ? blockIdx : extraBlocks);
            batchEnd = batchBegin + batchesPerBlock + (blockIdx < extraBlocks ? 1U : 0U);
            batchStep = 1U;
        }
        uint32_t batchCoordinates[MAXDIM] = {};
        uint32_t sequentialInputBatchOffset = 0U;
        uint32_t sequentialOutputBatchOffset = 0U;
        if constexpr (TYPE_BYTES == 4U && LAST_TILE_ROW_BLOCKS == 2U &&
                      LAST_TILE_BLOCKS == 2U) {
            DecodeBatchOffsets(batchBegin, batchCoordinates,
                               sequentialInputBatchOffset,
                               sequentialOutputBatchOffset);
        }
        for (uint32_t batch = batchBegin; batch < batchEnd; batch += batchStep) {
            uint32_t inputBatchOffset = 0U;
            uint32_t outputBatchOffset = 0U;
            if constexpr (TYPE_BYTES == 4U && LAST_TILE_ROW_BLOCKS == 2U &&
                          LAST_TILE_BLOCKS == 2U) {
                inputBatchOffset = sequentialInputBatchOffset;
                outputBatchOffset = sequentialOutputBatchOffset;
            } else {
                inputBatchOffset = InputBatchOffset(batch);
                outputBatchOffset = OutputBatchOffset(batch);
            }
            for (uint32_t matrixTask = 0U; matrixTask < tasksPerBatch; ++matrixTask) {
                const uint32_t rowTile = matrixTask / colTileCount;
                const uint32_t colTile = matrixTask - rowTile * colTileCount;
                ProcessLastTwoAxesTilePipelined(
                    progress & 1U, inputBatchOffset, outputBatchOffset,
                    rowTile * LAST_TILE_ROWS, colTile * LAST_TILE_COLS);
                ++progress;
            }
            if constexpr (TYPE_BYTES == 4U && LAST_TILE_ROW_BLOCKS == 2U &&
                          LAST_TILE_BLOCKS == 2U) {
                AdvanceBatchOffsets(batchCoordinates,
                                    sequentialInputBatchOffset,
                                    sequentialOutputBatchOffset);
            }
        }

        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline void ProcessTilesPipelined()
    {
        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        uint32_t progress = 0U;
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t batch = task / tasksPerBatch;
            const uint32_t matrixTask = task - batch * tasksPerBatch;
            const uint32_t rowTile = matrixTask / colTileCount;
            const uint32_t colTile = matrixTask - rowTile * colTileCount;
            ProcessTilePipelined(progress & 1U, batch,
                                 rowTile * TILE_ROWS, colTile * TILE_COLS);
            ++progress;
        }

        for (uint32_t buffer = 0U; buffer < PIPELINE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) const
    {
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline uint32_t InputBatchOffset(uint32_t outputBatch) const
    {
        uint32_t offset = 0U;
        uint32_t remainder = outputBatch;
        if constexpr (!GENERAL_OUTPUT_STRIDE) {
            for (int32_t i = static_cast<int32_t>(dims) - 3; i >= 0; --i) {
                const uint32_t axis = perm[i];
                const uint32_t coordinate = remainder % shape[axis];
                remainder /= shape[axis];
                offset += coordinate * inputStride[axis];
            }
            return offset;
        }
        for (int32_t i = static_cast<int32_t>(dims) - 1; i >= 0; --i) {
            if (static_cast<uint32_t>(i) == inputLastOutputPos ||
                static_cast<uint32_t>(i) == dims - 1U) {
                continue;
            }
            const uint32_t axis = perm[i];
            const uint32_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            offset += coordinate * inputStride[axis];
        }
        return offset;
    }

    __aicore__ inline uint32_t OutputBatchOffset(uint32_t outputBatch) const
    {
        uint32_t offset = 0U;
        uint32_t remainder = outputBatch;
        for (int32_t i = static_cast<int32_t>(dims) - 1; i >= 0; --i) {
            if (static_cast<uint32_t>(i) == inputLastOutputPos ||
                static_cast<uint32_t>(i) == dims - 1U) {
                continue;
            }
            const uint32_t axis = perm[i];
            const uint32_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            offset += coordinate * outputStride[i];
        }
        return offset;
    }

    __aicore__ inline void DecodeBatchOffsets(
        uint32_t outputBatch, uint32_t (&coordinates)[MAXDIM],
        uint32_t& inputOffset, uint32_t& outputOffset) const
    {
        inputOffset = 0U;
        outputOffset = 0U;
        uint32_t remainder = outputBatch;
        for (int32_t i = static_cast<int32_t>(dims) - 1; i >= 0; --i) {
            if (static_cast<uint32_t>(i) == inputLastOutputPos ||
                static_cast<uint32_t>(i) == dims - 1U) {
                continue;
            }
            const uint32_t axis = perm[i];
            const uint32_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            coordinates[i] = coordinate;
            inputOffset += coordinate * inputStride[axis];
            outputOffset += coordinate * outputStride[i];
        }
    }

    __aicore__ inline void AdvanceBatchOffsets(
        uint32_t (&coordinates)[MAXDIM], uint32_t& inputOffset,
        uint32_t& outputOffset) const
    {
        for (int32_t i = static_cast<int32_t>(dims) - 1; i >= 0; --i) {
            if (static_cast<uint32_t>(i) == inputLastOutputPos ||
                static_cast<uint32_t>(i) == dims - 1U) {
                continue;
            }
            const uint32_t axis = perm[i];
            ++coordinates[i];
            inputOffset += inputStride[axis];
            outputOffset += outputStride[i];
            if (coordinates[i] < shape[axis]) {
                return;
            }
            coordinates[i] = 0U;
            inputOffset -= shape[axis] * inputStride[axis];
            outputOffset -= shape[axis] * outputStride[i];
        }
    }

    __aicore__ inline void ComputeB32(const AscendC::LocalTensor<T>& output,
                                     const AscendC::LocalTensor<T>& input)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        AscendC::TransDataTo5HDParams params(false, false, 1U, 0U, 0U);
        for (uint32_t halfColumn = 0U; halfColumn < 2U; ++halfColumn) {
            for (uint32_t row = 0U; row < 16U; ++row) {
                srcList[row] = reinterpret_cast<uint64_t>(input[row * 16U + halfColumn * 8U].GetPhyAddr());
            }
            for (uint32_t outputRow = 0U; outputRow < 8U; ++outputRow) {
                const uint32_t row = halfColumn * 8U + outputRow;
                dstList[2U * outputRow] = reinterpret_cast<uint64_t>(output[row * 16U].GetPhyAddr());
                dstList[2U * outputRow + 1U] =
                    reinterpret_cast<uint64_t>(output[row * 16U + 8U].GetPhyAddr());
            }
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ComputeB8(const AscendC::LocalTensor<T>& output,
                                    const AscendC::LocalTensor<T>& input)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        AscendC::TransDataTo5HDParams params(false, false, 1U, 0U, 0U);

        for (uint32_t row = 0U; row < 16U; ++row) {
            dstList[row] = reinterpret_cast<uint64_t>(output[row * 32U].GetPhyAddr());
            srcList[row] = reinterpret_cast<uint64_t>(input[row * 32U].GetPhyAddr());
        }
        AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0U; row < 16U; ++row) {
            dstList[row] = reinterpret_cast<uint64_t>(output[512U + row * 32U].GetPhyAddr());
        }
        params.dstHighHalf = false;
        params.srcHighHalf = true;
        AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0U; row < 16U; ++row) {
            dstList[row] = reinterpret_cast<uint64_t>(output[row * 32U].GetPhyAddr());
            srcList[row] = reinterpret_cast<uint64_t>(input[512U + row * 32U].GetPhyAddr());
        }
        params.dstHighHalf = true;
        params.srcHighHalf = false;
        AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t row = 0U; row < 16U; ++row) {
            dstList[row] = reinterpret_cast<uint64_t>(output[512U + row * 32U].GetPhyAddr());
        }
        params.dstHighHalf = true;
        params.srcHighHalf = true;
        AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeLastTwoB16(const AscendC::LocalTensor<T>& output,
                                            const AscendC::LocalTensor<T>& input)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        AscendC::TransDataTo5HDParams params(false, false, 1U, 0U, 0U);
        for (uint32_t rowBlock = 0U; rowBlock < LAST_TILE_ROW_BLOCKS; ++rowBlock) {
            const uint32_t inputRow = rowBlock * TILE_ROWS;
            for (uint32_t block = 0U; block < LAST_TILE_BLOCKS; ++block) {
                const uint32_t inputColumn = block * TILE_COLS;
                const uint32_t outputBase =
                    block * TILE_COLS * LAST_TILE_ROWS + rowBlock * TILE_ROWS;
                for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(inputRow + row) * LAST_TILE_COLS + inputColumn].GetPhyAddr());
                    dstList[row] = reinterpret_cast<uint64_t>(
                        output[outputBase + row * LAST_TILE_ROWS].GetPhyAddr());
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeLastTwoB32(const AscendC::LocalTensor<T>& output,
                                            const AscendC::LocalTensor<T>& input)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        constexpr uint16_t dstRepeatStride = LAST_TILE_BLOCKS == 1U ? 0U :
            TILE_COLS * LAST_TILE_ROWS * TYPE_BYTES / DMA_BLOCK_BYTES;
        constexpr uint16_t srcRepeatStride = LAST_TILE_BLOCKS == 1U ? 0U :
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(LAST_TILE_BLOCKS),
            dstRepeatStride, srcRepeatStride);
        for (uint32_t rowBlock = 0U; rowBlock < LAST_TILE_ROW_BLOCKS; ++rowBlock) {
            const uint32_t inputRow = rowBlock * TILE_ROWS;
            const uint32_t outputBase = rowBlock * TILE_ROWS;
            for (uint32_t halfColumn = 0U; halfColumn < 2U; ++halfColumn) {
                for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(inputRow + row) * LAST_TILE_COLS +
                              halfColumn * 8U].GetPhyAddr());
                }
                for (uint32_t outputRow = 0U; outputRow < 8U; ++outputRow) {
                    const uint32_t row = halfColumn * 8U + outputRow;
                    dstList[2U * outputRow] = reinterpret_cast<uint64_t>(
                        output[outputBase + row * LAST_TILE_ROWS].GetPhyAddr());
                    dstList[2U * outputRow + 1U] = reinterpret_cast<uint64_t>(
                        output[outputBase + row * LAST_TILE_ROWS + 8U].GetPhyAddr());
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeLastTwoB32ColumnChunk(
        const AscendC::LocalTensor<T>& output,
        const AscendC::LocalTensor<T>& input,
        uint32_t firstColumnBlock, uint32_t columnBlockCount)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        constexpr uint16_t dstRepeatStride =
            TILE_COLS * LAST_TILE_ROWS * TYPE_BYTES / DMA_BLOCK_BYTES;
        constexpr uint16_t srcRepeatStride =
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(columnBlockCount),
            dstRepeatStride, srcRepeatStride);
        const uint32_t firstColumn = firstColumnBlock * TILE_COLS;
        for (uint32_t rowBlock = 0U;
             rowBlock < LAST_TILE_ROW_BLOCKS; ++rowBlock) {
            const uint32_t inputRow = rowBlock * TILE_ROWS;
            const uint32_t outputBase = rowBlock * TILE_ROWS;
            for (uint32_t halfColumn = 0U;
                 halfColumn < 2U; ++halfColumn) {
                for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(inputRow + row) * LAST_TILE_COLS +
                              firstColumn + halfColumn * 8U].GetPhyAddr());
                }
                for (uint32_t outputRow = 0U;
                     outputRow < 8U; ++outputRow) {
                    const uint32_t row = halfColumn * 8U + outputRow;
                    dstList[2U * outputRow] = reinterpret_cast<uint64_t>(
                        output[outputBase +
                               row * LAST_TILE_ROWS].GetPhyAddr());
                    dstList[2U * outputRow + 1U] =
                        reinterpret_cast<uint64_t>(
                            output[outputBase +
                                   row * LAST_TILE_ROWS + 8U].GetPhyAddr());
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeLastTwoB32RowPhase(
        const AscendC::LocalTensor<T>& output,
        const AscendC::LocalTensor<T>& input,
        uint32_t firstRowBlock, uint32_t rowBlockCount,
        uint32_t columnBlockCount, uint32_t inputLocalRowStride,
        uint32_t outputLocalRowStride = LAST_TILE_ROWS)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        const uint16_t dstRepeatStride = static_cast<uint16_t>(
            TILE_COLS * outputLocalRowStride * TYPE_BYTES /
            DMA_BLOCK_BYTES);
        constexpr uint16_t srcRepeatStride =
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(columnBlockCount),
            dstRepeatStride, srcRepeatStride);
        for (uint32_t localRowBlock = 0U;
             localRowBlock < rowBlockCount; ++localRowBlock) {
            const uint32_t inputRow = localRowBlock * TILE_ROWS;
            const uint32_t outputBase =
                (firstRowBlock + localRowBlock) * TILE_ROWS;
            for (uint32_t halfColumn = 0U;
                 halfColumn < 2U; ++halfColumn) {
                for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(inputRow + row) * inputLocalRowStride +
                              halfColumn * 8U].GetPhyAddr());
                }
                for (uint32_t outputRow = 0U;
                     outputRow < 8U; ++outputRow) {
                    const uint32_t row = halfColumn * 8U + outputRow;
                    dstList[2U * outputRow] = reinterpret_cast<uint64_t>(
                        output[outputBase +
                               row * outputLocalRowStride].GetPhyAddr());
                    dstList[2U * outputRow + 1U] =
                        reinterpret_cast<uint64_t>(
                            output[outputBase +
                                   row * outputLocalRowStride + 8U].GetPhyAddr());
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    // Key 33 has two compile-time geometries.  Construct their affine UB
    // addresses from raw byte bases so the compiler does not repeatedly
    // lower LocalTensor indexing through ZEROEXT/SHL chains.  The VA
    // programming, VNCHW order, repeat geometry, and per-instruction Vector
    // barrier remain identical to the stock row-phase implementation.
    template <bool TALL_TAIL>
    __aicore__ inline void ComputeLastTwoB32RawAffineRowPhase(
        const AscendC::LocalTensor<T>& output,
        const AscendC::LocalTensor<T>& input,
        uint32_t firstRowBlock, uint32_t rowBlockCount)
    {
        static_assert(TYPE_BYTES == 4U);
        constexpr uint32_t outputRows =
            TALL_TAIL ? ADAPTIVE_TAIL_ROWS : LAST_TILE_ROWS;
        constexpr uint32_t inputCols =
            TALL_TAIL ? ADAPTIVE_TAIL_COLS : LAST_TILE_COLS;
        constexpr uint8_t columnBlockCount =
            TALL_TAIL ? ADAPTIVE_TAIL_COLS / TILE_COLS :
                        LAST_TILE_BLOCKS;
        constexpr uint64_t inputRowBytes = inputCols * TYPE_BYTES;
        constexpr uint64_t outputRowBytes = outputRows * TYPE_BYTES;
        constexpr uint16_t destinationRepeatStride =
            TILE_COLS * outputRows * TYPE_BYTES /
            DMA_BLOCK_BYTES;
        constexpr uint16_t sourceRepeatStride =
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, columnBlockCount,
            destinationRepeatStride, sourceRepeatStride);
        const uint64_t outputBase =
            reinterpret_cast<uint64_t>(output.GetPhyAddr());
        const uint64_t inputBase =
            reinterpret_cast<uint64_t>(input.GetPhyAddr());
        uint64_t dstList[16];
        uint64_t srcList[16];

        for (uint32_t localRowBlock = 0U;
             localRowBlock < rowBlockCount; ++localRowBlock) {
            const uint64_t sourceBlock = inputBase +
                localRowBlock * TILE_ROWS * inputRowBytes;
            const uint64_t destinationBlock = outputBase +
                (firstRowBlock + localRowBlock) *
                    TILE_ROWS * TYPE_BYTES;
            for (uint32_t halfColumn = 0U;
                 halfColumn < 2U; ++halfColumn) {
                const uint64_t source = sourceBlock +
                    halfColumn * 8U * TYPE_BYTES;
                const uint64_t destination = destinationBlock +
                    halfColumn * 8U * outputRowBytes;
                srcList[0] = source;
                srcList[1] = source + inputRowBytes;
                srcList[2] = source + 2U * inputRowBytes;
                srcList[3] = source + 3U * inputRowBytes;
                srcList[4] = source + 4U * inputRowBytes;
                srcList[5] = source + 5U * inputRowBytes;
                srcList[6] = source + 6U * inputRowBytes;
                srcList[7] = source + 7U * inputRowBytes;
                srcList[8] = source + 8U * inputRowBytes;
                srcList[9] = source + 9U * inputRowBytes;
                srcList[10] = source + 10U * inputRowBytes;
                srcList[11] = source + 11U * inputRowBytes;
                srcList[12] = source + 12U * inputRowBytes;
                srcList[13] = source + 13U * inputRowBytes;
                srcList[14] = source + 14U * inputRowBytes;
                srcList[15] = source + 15U * inputRowBytes;

                dstList[0] = destination;
                dstList[1] = destination + 8U * TYPE_BYTES;
                dstList[2] = destination + outputRowBytes;
                dstList[3] = destination + outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[4] = destination + 2U * outputRowBytes;
                dstList[5] = destination + 2U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[6] = destination + 3U * outputRowBytes;
                dstList[7] = destination + 3U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[8] = destination + 4U * outputRowBytes;
                dstList[9] = destination + 4U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[10] = destination + 5U * outputRowBytes;
                dstList[11] = destination + 5U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[12] = destination + 6U * outputRowBytes;
                dstList[13] = destination + 6U * outputRowBytes +
                    8U * TYPE_BYTES;
                dstList[14] = destination + 7U * outputRowBytes;
                dstList[15] = destination + 7U * outputRowBytes +
                    8U * TYPE_BYTES;

                AscendC::SetVaReg(dstList, srcList);
                AscendC::TransDataTo5HDIntrinsicsImpl<T>(
                    dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeLastTwoB32Half(
        const AscendC::LocalTensor<T>& output,
        const AscendC::LocalTensor<T>& input,
        uint32_t firstRowBlock)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        constexpr uint16_t dstRepeatStride =
            TILE_COLS * LAST_TILE_ROWS * TYPE_BYTES / DMA_BLOCK_BYTES;
        constexpr uint16_t srcRepeatStride =
            TILE_COLS * TYPE_BYTES / DMA_BLOCK_BYTES;
        AscendC::TransDataTo5HDParams params(
            false, false, static_cast<uint8_t>(LAST_TILE_BLOCKS),
            dstRepeatStride, srcRepeatStride);
        for (uint32_t localRowBlock = 0U;
             localRowBlock < LAST_TILE_ROW_BLOCKS / 2U;
             ++localRowBlock) {
            const uint32_t rowBlock = firstRowBlock + localRowBlock;
            const uint32_t inputRow = rowBlock * TILE_ROWS;
            const uint32_t outputBase = rowBlock * TILE_ROWS;
            for (uint32_t halfColumn = 0U; halfColumn < 2U; ++halfColumn) {
                for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(inputRow + row) * LAST_TILE_COLS +
                              halfColumn * 8U].GetPhyAddr());
                }
                for (uint32_t outputRow = 0U;
                     outputRow < 8U; ++outputRow) {
                    const uint32_t row = halfColumn * 8U + outputRow;
                    dstList[2U * outputRow] = reinterpret_cast<uint64_t>(
                        output[outputBase +
                               row * LAST_TILE_ROWS].GetPhyAddr());
                    dstList[2U * outputRow + 1U] =
                        reinterpret_cast<uint64_t>(
                            output[outputBase +
                                   row * LAST_TILE_ROWS + 8U].GetPhyAddr());
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ComputeLastTwoB8(const AscendC::LocalTensor<T>& output,
                                           const AscendC::LocalTensor<T>& input)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        for (uint32_t rowBlock = 0U; rowBlock < LAST_TILE_ROW_BLOCKS; ++rowBlock) {
            const uint32_t inputRow = rowBlock * TILE_ROWS;
            for (uint32_t block = 0U; block < LAST_TILE_BLOCKS; ++block) {
                const uint32_t inputColumn = block * TILE_COLS;
                const uint32_t outputBase =
                    block * TILE_COLS * LAST_TILE_ROWS + rowBlock * TILE_ROWS;
                AscendC::TransDataTo5HDParams params(false, false, 1U, 0U, 0U);
                for (uint32_t row = 0U; row < 16U; ++row) {
                    dstList[row] = reinterpret_cast<uint64_t>(
                        output[outputBase + row * LAST_TILE_ROWS].GetPhyAddr());
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(inputRow + row) * LAST_TILE_COLS + inputColumn].GetPhyAddr());
                }
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();

                for (uint32_t row = 0U; row < 16U; ++row) {
                    dstList[row] = reinterpret_cast<uint64_t>(
                        output[outputBase + 16U * LAST_TILE_ROWS + row * LAST_TILE_ROWS].GetPhyAddr());
                }
                params.dstHighHalf = false;
                params.srcHighHalf = true;
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();

                for (uint32_t row = 0U; row < 16U; ++row) {
                    dstList[row] = reinterpret_cast<uint64_t>(
                        output[outputBase + row * LAST_TILE_ROWS].GetPhyAddr());
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(inputRow + 16U + row) * LAST_TILE_COLS + inputColumn].GetPhyAddr());
                }
                params.dstHighHalf = true;
                params.srcHighHalf = false;
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();

                for (uint32_t row = 0U; row < 16U; ++row) {
                    dstList[row] = reinterpret_cast<uint64_t>(
                        output[outputBase + 16U * LAST_TILE_ROWS + row * LAST_TILE_ROWS].GetPhyAddr());
                }
                params.dstHighHalf = true;
                params.srcHighHalf = true;
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
    }

    __aicore__ inline void ProcessTile(uint32_t batch, uint32_t rowStart, uint32_t colStart)
    {
        const uint32_t rows = (height - rowStart < TILE_ROWS) ? height - rowStart : TILE_ROWS;
        const uint32_t cols = (width - colStart < TILE_COLS) ? width - colStart : TILE_COLS;
        const uint32_t alignedCols = CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        const uint32_t inputOffset = batch * matrixElements + rowStart * width + colStart;

        AscendC::LocalTensor<T> input(AscendC::TPosition::VECIN, INPUT_UB_OFFSET, TILE_ELEMENTS);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES, (width - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));

        AscendC::LocalTensor<T> output(AscendC::TPosition::VECOUT, OUTPUT_UB_OFFSET, TILE_ELEMENTS);
        if constexpr (TYPE_BYTES == 4U) {
            ComputeB32(output, input);
        } else if constexpr (TYPE_BYTES == 2U) {
            AscendC::Transpose(output, input);
        } else {
            ComputeB8(output, input);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));

        const uint32_t outputOffset = batch * matrixElements + colStart * height + rowStart;
        const uint32_t alignedRows = CeilDiv(rows, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * TYPE_BYTES, outputSrcGapBlocks,
            (height - rows) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
    }

    __aicore__ inline void ProcessTilePipelined(
        uint32_t buffer, uint32_t batch, uint32_t rowStart, uint32_t colStart)
    {
        const event_t event = static_cast<event_t>(buffer);
        const uint32_t rows = (height - rowStart < TILE_ROWS) ? height - rowStart : TILE_ROWS;
        const uint32_t cols = (width - colStart < TILE_COLS) ? width - colStart : TILE_COLS;
        const uint32_t alignedCols = CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        const uint32_t inputOffset = batch * matrixElements + rowStart * width + colStart;

        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN, buffer * TILE_BYTES, TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES, (width - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            PIPELINED_TILE_OUTPUT_UB_OFFSET + buffer * TILE_BYTES, TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        if constexpr (TYPE_BYTES == 4U) {
            ComputeB32(output, input);
        } else if constexpr (TYPE_BYTES == 2U) {
            AscendC::Transpose(output, input);
        } else {
            ComputeB8(output, input);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

        const uint32_t outputOffset = batch * matrixElements + colStart * height + rowStart;
        const uint32_t alignedRows = CeilDiv(rows, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * TYPE_BYTES, outputSrcGapBlocks,
            (height - rows) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
    }

    __aicore__ inline void ProcessLastTwoAxesTile(uint32_t inputBatchOffset, uint32_t outputBatchOffset,
                                                 uint32_t rowStart, uint32_t colStart)
    {
        const uint32_t rows = (height - rowStart < LAST_TILE_ROWS) ? height - rowStart : LAST_TILE_ROWS;
        const uint32_t cols = (width - colStart < LAST_TILE_COLS) ? width - colStart : LAST_TILE_COLS;
        const uint32_t alignedCols = CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset + rowStart * inputRowStride + colStart;

        AscendC::LocalTensor<T> input(AscendC::TPosition::VECIN, INPUT_UB_OFFSET, LAST_TILE_ELEMENTS);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES, (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));

        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT, LAST_OUTPUT_UB_OFFSET, LAST_TILE_ELEMENTS);
        if constexpr (TYPE_BYTES == 4U) {
            ComputeLastTwoB32(output, input);
        } else if constexpr (TYPE_BYTES == 2U) {
            ComputeLastTwoB16(output, input);
        } else {
            ComputeLastTwoB8(output, input);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));

        const uint32_t outputOffset = outputBatchOffset + colStart * outputRowStride + rowStart;
        const uint32_t alignedRows = CeilDiv(rows, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((LAST_TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * TYPE_BYTES, outputSrcGapBlocks,
            (outputRowStride - rows) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
    }

    __aicore__ inline void ProcessLastTwoAxesTilePipelined(
        uint32_t buffer, uint32_t inputBatchOffset, uint32_t outputBatchOffset,
        uint32_t rowStart, uint32_t colStart)
    {
        const event_t inputEvent = static_cast<event_t>(buffer);
        const event_t outputEvent = static_cast<event_t>(
            SHARED_OUTPUT_BUFFER ? SHARED_OUTPUT_EVENT : buffer);
        const uint32_t rows = (height - rowStart < LAST_TILE_ROWS) ? height - rowStart : LAST_TILE_ROWS;
        const uint32_t cols = (width - colStart < LAST_TILE_COLS) ? width - colStart : LAST_TILE_COLS;
        const uint32_t alignedCols = CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        const uint32_t inputOffset = inputBatchOffset + rowStart * inputRowStride + colStart;

        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN, buffer * LAST_TILE_ELEMENTS * TYPE_BYTES, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES, (inputRowStride - cols) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEvent);

        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            PIPELINED_OUTPUT_UB_OFFSET +
                (SHARED_OUTPUT_BUFFER ? 0U : buffer * LAST_TILE_ELEMENTS * TYPE_BYTES),
            LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
        if constexpr (TYPE_BYTES == 4U) {
            ComputeLastTwoB32(output, input);
        } else if constexpr (TYPE_BYTES == 2U) {
            ComputeLastTwoB16(output, input);
        } else {
            ComputeLastTwoB8(output, input);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent);

        const uint32_t outputOffset = outputBatchOffset + colStart * outputRowStride + rowStart;
        const uint32_t alignedRows = CeilDiv(rows, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((LAST_TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * TYPE_BYTES, outputSrcGapBlocks,
            (outputRowStride - rows) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    __aicore__ inline void ProcessLastTwoAxesTileSingleInputOutput(
        uint32_t inputBatchOffset, uint32_t outputBatchOffset,
        uint32_t rowStart, uint32_t colStart)
    {
        constexpr event_t inputEvent = static_cast<event_t>(0);
        constexpr event_t outputEvent = static_cast<event_t>(1);
        const uint32_t rows =
            height - rowStart < LAST_TILE_ROWS ? height - rowStart : LAST_TILE_ROWS;
        const uint32_t cols =
            width - colStart < LAST_TILE_COLS ? width - colStart : LAST_TILE_COLS;
        const uint32_t alignedCols = CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        const uint32_t inputOffset =
            inputBatchOffset + rowStart * inputRowStride + colStart;

        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN, INPUT_UB_OFFSET, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES, inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEvent);

        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT, LAST_OUTPUT_UB_OFFSET, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
        ComputeLastTwoB32(output, input);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent);

        const uint32_t outputOffset =
            outputBatchOffset + colStart * outputRowStride + rowStart;
        const uint32_t alignedRows = CeilDiv(rows, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((LAST_TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * TYPE_BYTES, outputSrcGapBlocks,
            (outputRowStride - rows) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    __aicore__ inline void ProcessLastTwoAxesTileOneInputTwoOutputs(
        uint32_t outputBuffer, uint32_t inputBatchOffset, uint32_t outputBatchOffset,
        uint32_t rowStart, uint32_t colStart)
    {
        constexpr event_t inputEvent = static_cast<event_t>(2);
        const event_t outputEvent = static_cast<event_t>(outputBuffer);
        const uint32_t rows =
            height - rowStart < LAST_TILE_ROWS ? height - rowStart : LAST_TILE_ROWS;
        const uint32_t cols =
            width - colStart < LAST_TILE_COLS ? width - colStart : LAST_TILE_COLS;
        const uint32_t alignedCols = CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        const uint32_t inputOffset =
            inputBatchOffset + rowStart * inputRowStride + colStart;

        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN, INPUT_UB_OFFSET, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES, inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputEvent);

        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            LAST_OUTPUT_UB_OFFSET + outputBuffer * LAST_TILE_ELEMENTS * TYPE_BYTES,
            LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
        ComputeLastTwoB32(output, input);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(inputEvent);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(outputEvent);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(outputEvent);

        const uint32_t outputOffset =
            outputBatchOffset + colStart * outputRowStride + rowStart;
        const uint32_t alignedRows = CeilDiv(rows, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((LAST_TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), rows * TYPE_BYTES, outputSrcGapBlocks,
            (outputRowStride - rows) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(outputEvent);
    }

    __aicore__ inline void LoadGeneralSingleTile(
        uint32_t buffer, uint32_t inputBatchOffset)
    {
        const event_t event = static_cast<event_t>(buffer);
        const uint32_t rows = height;
        const uint32_t cols = width;
        const uint32_t alignedCols = CeilDiv(cols, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            ((LAST_TILE_COLS - alignedCols) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            buffer * LAST_TILE_ELEMENTS * TYPE_BYTES, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), cols * TYPE_BYTES,
            (inputRowStride - cols) * TYPE_BYTES, inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedCols - cols), static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputBatchOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void ComputeStoreGeneralSingleTile(
        uint32_t buffer, uint32_t outputBatchOffset)
    {
        const event_t event = static_cast<event_t>(buffer);
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            buffer * LAST_TILE_ELEMENTS * TYPE_BYTES, LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT,
            PIPELINED_OUTPUT_UB_OFFSET +
                buffer * LAST_TILE_ELEMENTS * TYPE_BYTES,
            LAST_TILE_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        ComputeLastTwoB32(output, input);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

        const uint32_t alignedRows =
            CeilDiv(height, DMA_BLOCK_ELEMENTS) * DMA_BLOCK_ELEMENTS;
        const uint32_t outputSrcGapBlocks =
            ((LAST_TILE_ROWS - alignedRows) * TYPE_BYTES) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(width), height * TYPE_BYTES,
            outputSrcGapBlocks, (outputRowStride - height) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputBatchOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
    }

private:
    static constexpr uint32_t TYPE_BYTES = sizeof(T);
    static constexpr uint32_t DMA_BLOCK_BYTES = 32U;
    static constexpr uint32_t DMA_BLOCK_ELEMENTS = DMA_BLOCK_BYTES / TYPE_BYTES;
    static constexpr uint32_t TILE_ROWS = TYPE_BYTES == 1U ? 32U : 16U;
    static constexpr uint32_t TILE_COLS = TYPE_BYTES == 1U ? 32U : 16U;
    static constexpr uint32_t TILE_ELEMENTS = TILE_ROWS * TILE_COLS;
    static constexpr uint32_t PIPELINE_BUFFER_COUNT = 2U;
    static constexpr uint32_t TILE_BYTES = TILE_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t INPLACE_COMPACT_BLOCK_CAPACITY =
        LAST_TILE_BLOCK_COUNT * (LAST_TILE_BLOCK_COUNT - 1U) / 2U;
    static constexpr uint32_t PIPELINED_TILE_OUTPUT_UB_OFFSET =
        PIPELINE_BUFFER_COUNT * TILE_BYTES;
    static constexpr uint32_t LAST_TILE_ROW_BLOCKS = LAST_TILE_ROW_BLOCK_COUNT;
    static constexpr uint32_t LAST_TILE_ROWS = TILE_ROWS * LAST_TILE_ROW_BLOCKS;
    static constexpr uint32_t LAST_TILE_BLOCKS = LAST_TILE_BLOCK_COUNT;
    static constexpr uint32_t LAST_TILE_COLS = TILE_COLS * LAST_TILE_BLOCKS;
    static constexpr uint32_t LAST_TILE_ELEMENTS = LAST_TILE_ROWS * LAST_TILE_COLS;
    static constexpr uint32_t WIDE_OUTPUT_BLOCKS = 3U;
    static constexpr uint32_t WIDE_OUTPUT_COLS =
        TILE_COLS * WIDE_OUTPUT_BLOCKS;
    static constexpr uint32_t WIDE_OUTPUT_ELEMENTS =
        LAST_TILE_ROWS * WIDE_OUTPUT_COLS;
    static constexpr uint32_t WIDE_OUTPUT_UB_OFFSET =
        LAST_TILE_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t UB_CAPACITY_BYTES = 192U * 1024U;
    static constexpr uint32_t WIDE_PHASE_ROW_BLOCKS =
        (UB_CAPACITY_BYTES - LAST_TILE_ELEMENTS * TYPE_BYTES) /
        (TILE_ROWS * LAST_TILE_COLS * TYPE_BYTES);
    static constexpr uint32_t WIDE_PHASE_ROWS =
        TILE_ROWS * WIDE_PHASE_ROW_BLOCKS;
    static constexpr uint32_t WIDE_PHASE_INPUT_ELEMENTS =
        WIDE_PHASE_ROWS * LAST_TILE_COLS;
    static constexpr uint32_t WIDE_ACCUMULATED_OUTPUT_UB_OFFSET = 0U;
    static constexpr uint32_t WIDE_PHASE_INPUT_UB_OFFSET =
        LAST_TILE_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t ADAPTIVE_TAIL_ROWS =
        2U * LAST_TILE_ROWS;
    static constexpr uint32_t ADAPTIVE_TAIL_COLS =
        LAST_TILE_COLS / 2U;
    static constexpr uint32_t ADAPTIVE_TAIL_PHASE_ROW_BLOCKS =
        WIDE_PHASE_INPUT_ELEMENTS /
        (TILE_ROWS * ADAPTIVE_TAIL_COLS);
    static constexpr uint32_t ADAPTIVE_TAIL_PHASE_ROWS =
        TILE_ROWS * ADAPTIVE_TAIL_PHASE_ROW_BLOCKS;
    static constexpr uint32_t QUEUED_PHASE_ROW_BLOCKS =
        WIDE_PHASE_ROW_BLOCKS / 2U;
    static constexpr uint32_t QUEUED_PHASE_ROWS =
        TILE_ROWS * QUEUED_PHASE_ROW_BLOCKS;
    static constexpr uint32_t QUEUED_PHASE_INPUT_ELEMENTS =
        QUEUED_PHASE_ROWS * LAST_TILE_COLS;
    static constexpr uint32_t QUEUED_TAIL_PHASE_ROW_BLOCKS =
        QUEUED_PHASE_INPUT_ELEMENTS /
        (TILE_ROWS * ADAPTIVE_TAIL_COLS);
    static constexpr uint32_t QUEUED_TAIL_PHASE_ROWS =
        TILE_ROWS * QUEUED_TAIL_PHASE_ROW_BLOCKS;
    static constexpr uint32_t INPLACE_PHASE_ROW_BLOCKS =
        LAST_TILE_BLOCKS;
    static constexpr uint32_t INPLACE_PHASE_ROWS =
        TILE_ROWS * INPLACE_PHASE_ROW_BLOCKS;
    static constexpr uint32_t INPLACE_SCRATCH_OUTPUT_PHASE = 2U;
    static constexpr uint32_t INPLACE_SCRATCH_FIRST_ELEMENT =
        INPLACE_SCRATCH_OUTPUT_PHASE * INPLACE_PHASE_ROWS;
    static constexpr uint32_t INPLACE_SCRATCH_UB_OFFSET =
        INPLACE_SCRATCH_FIRST_ELEMENT * TYPE_BYTES;
    static constexpr uint32_t HYBRID_COMPACT_FIRST_ROW_BLOCK =
        2U * INPLACE_PHASE_ROW_BLOCKS;
    static constexpr uint32_t RECTANGLE_OUTPUT_COLS =
        LAST_TILE_ROWS * OUTPUT_BATCH_GROUP_COUNT;
    static constexpr uint32_t RECTANGLE_OUTPUT_ELEMENTS =
        LAST_TILE_COLS * RECTANGLE_OUTPUT_COLS;
    static constexpr uint32_t RECTANGLE_INPUT_BUFFER_BYTES =
        OUTPUT_BATCH_GROUP_COUNT * LAST_TILE_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t RECTANGLE_OUTPUT_TILE_BYTES =
        RECTANGLE_OUTPUT_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t RECTANGLE_PACKED_ROW_CAPACITY =
        RECTANGLE_PACKED_ROW_BLOCK_COUNT == 0U ?
            RECTANGLE_OUTPUT_COLS :
            RECTANGLE_PACKED_ROW_BLOCK_COUNT * DMA_BLOCK_ELEMENTS;
    static constexpr uint32_t RECTANGLE_GATHER_CHUNK_ROWS = 24U;
    static constexpr uint32_t RECTANGLE_PACKED_SOURCE_ROWS = LAST_TILE_COLS;
    static constexpr uint32_t RECTANGLE_PACKED_TILE_ELEMENTS =
        RECTANGLE_PACKED_SOURCE_ROWS * RECTANGLE_PACKED_ROW_CAPACITY;
    static constexpr uint32_t RECTANGLE_PACKED_TILE_BYTES =
        RECTANGLE_PACKED_TILE_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t RECTANGLE_OFFSET_TILE_ELEMENTS =
        TWO_PHASE_RECTANGLE_PACKED_OUTPUT ?
            RECTANGLE_GATHER_CHUNK_ROWS * RECTANGLE_PACKED_ROW_CAPACITY :
            RECTANGLE_PACKED_TILE_ELEMENTS;
    static constexpr uint32_t RECTANGLE_BUFFER_COUNT =
        SHARED_RECTANGLE_BUFFERS ? 1U : PIPELINE_BUFFER_COUNT;
    static constexpr uint32_t RECTANGLE_OUTPUT_UB_OFFSET =
        PIPELINE_BUFFER_COUNT * RECTANGLE_INPUT_BUFFER_BYTES;
    static constexpr uint32_t RECTANGLE_PACKED_UB_OFFSET =
        RECTANGLE_OUTPUT_UB_OFFSET +
        RECTANGLE_BUFFER_COUNT * RECTANGLE_OUTPUT_TILE_BYTES;
    static constexpr uint32_t RECTANGLE_OFFSETS_UB_OFFSET =
        TWO_PHASE_RECTANGLE_PACKED_OUTPUT ?
            RECTANGLE_OUTPUT_UB_OFFSET +
                RECTANGLE_BUFFER_COUNT * RECTANGLE_OUTPUT_TILE_BYTES :
            RECTANGLE_PACKED_UB_OFFSET +
                RECTANGLE_BUFFER_COUNT * RECTANGLE_PACKED_TILE_BYTES;
    static constexpr uint32_t SIX_OUTPUT_INPUT_BUFFER_BYTES =
        LAST_TILE_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t SIX_OUTPUT_OUTPUT_COLS =
        OUTPUT_BATCH_GROUP_COUNT * LAST_TILE_ROWS;
    static constexpr uint32_t SIX_OUTPUT_OUTPUT_ELEMENTS =
        LAST_TILE_COLS * SIX_OUTPUT_OUTPUT_COLS;
    static constexpr uint32_t SIX_OUTPUT_OUTPUT_UB_OFFSET =
        PIPELINE_BUFFER_COUNT * SIX_OUTPUT_INPUT_BUFFER_BYTES;
    static constexpr uint32_t SIX_OUTPUT_PACKED_ROW_STRIDE =
        RECTANGLE_PACKED_ROW_CAPACITY;
    static constexpr uint32_t SIX_OUTPUT_GATHER_CHUNK_ROWS = 24U;
    static constexpr uint32_t SIX_OUTPUT_OFFSET_ELEMENTS =
        SIX_OUTPUT_GATHER_CHUNK_ROWS * SIX_OUTPUT_PACKED_ROW_STRIDE;
    static constexpr uint32_t SIX_OUTPUT_OFFSETS_UB_OFFSET =
        SIX_OUTPUT_OUTPUT_UB_OFFSET +
        SIX_OUTPUT_OUTPUT_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t COMBINED_SIX_INPUT_ROW_STRIDE =
        SIX_OUTPUT_PACKED_ROW_STRIDE;
    static constexpr uint32_t COMBINED_SIX_INPUT_ROWS =
        (SIX_OUTPUT_PACKED_ROW_STRIDE / OUTPUT_BATCH_GROUP_COUNT) *
        OUTPUT_BATCH_GROUP_COUNT;
    // The final B32 half reads eight padding elements past a 152-element
    // compact row.  Only discarded output rows consume those values; reserve
    // one physical tail so the last source row remains inside UB.
    static constexpr uint32_t COMBINED_SIX_INPUT_ELEMENTS =
        COMBINED_SIX_INPUT_ROWS * COMBINED_SIX_INPUT_ROW_STRIDE +
        TILE_COLS / 2U;
    static constexpr uint32_t COMBINED_SIX_INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t COMBINED_SIX_OUTPUT_ROW_STRIDE =
        LAST_TILE_COLS;
    static constexpr uint32_t COMBINED_SIX_OUTPUT_ROWS = LAST_TILE_COLS;
    static constexpr uint32_t COMBINED_SIX_OUTPUT_ELEMENTS =
        COMBINED_SIX_OUTPUT_ROWS * COMBINED_SIX_OUTPUT_ROW_STRIDE;
    static constexpr uint32_t COMBINED_SIX_OUTPUT_UB_OFFSET =
        COMBINED_SIX_INPUT_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t COMBINED_SIX_OUTPUT_UB_END =
        COMBINED_SIX_OUTPUT_UB_OFFSET +
        COMBINED_SIX_OUTPUT_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t SEVEN_BY_SIX_INPUT_ROW_STRIDE =
        11U * TILE_COLS;
    static constexpr uint32_t SEVEN_BY_SIX_INPUT_ROWS =
        COMBINED_SIX_INPUT_ROWS;
    static constexpr uint32_t SEVEN_BY_SIX_INPUT_ELEMENTS =
        SEVEN_BY_SIX_INPUT_ROWS * SEVEN_BY_SIX_INPUT_ROW_STRIDE +
        TILE_COLS / 2U;
    static constexpr uint32_t SEVEN_BY_SIX_INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_ROW_STRIDE =
        LAST_TILE_COLS;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_HEAD_ROWS =
        8U * TILE_COLS;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_BANK0_ROWS = 50U;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_BANK1_ROWS =
        SEVEN_BY_SIX_OUTPUT_HEAD_ROWS -
            SEVEN_BY_SIX_OUTPUT_BANK0_ROWS;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_TAIL_ROWS =
        3U * TILE_COLS;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_BANK0_ELEMENTS =
        SEVEN_BY_SIX_OUTPUT_BANK0_ROWS *
            SEVEN_BY_SIX_OUTPUT_ROW_STRIDE;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_BANK1_ELEMENTS =
        SEVEN_BY_SIX_OUTPUT_BANK1_ROWS *
            SEVEN_BY_SIX_OUTPUT_ROW_STRIDE;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_HEAD_ELEMENTS =
        SEVEN_BY_SIX_OUTPUT_HEAD_ROWS *
            SEVEN_BY_SIX_OUTPUT_ROW_STRIDE;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_BANK0_BYTES =
        SEVEN_BY_SIX_OUTPUT_BANK0_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_BANK1_BYTES =
        SEVEN_BY_SIX_OUTPUT_BANK1_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t SEVEN_BY_SIX_OUTPUT_UB_OFFSET =
        SEVEN_BY_SIX_INPUT_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t SEVEN_BY_SIX_UB_END =
        SEVEN_BY_SIX_OUTPUT_UB_OFFSET +
        SEVEN_BY_SIX_OUTPUT_BANK0_BYTES +
        SEVEN_BY_SIX_OUTPUT_BANK1_BYTES;
    static constexpr uint32_t PIPELINED_OUTPUT_UB_OFFSET =
        PIPELINE_BUFFER_COUNT * LAST_TILE_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t OUTPUT_UB_OFFSET = TILE_ELEMENTS * TYPE_BYTES;
    static constexpr uint32_t LAST_OUTPUT_UB_OFFSET = LAST_TILE_ELEMENTS * TYPE_BYTES;
    static constexpr int32_t INPUT_EVENT = 0;
    static constexpr int32_t OUTPUT_EVENT = 0;
    static constexpr int32_t SHARED_OUTPUT_EVENT = 2;
    static constexpr int32_t RECTANGLE_PATTERN_EVENT = 2;
    static constexpr int32_t SHARED_RECTANGLE_EVENT = 2;

    AscendC::TPipe queuedPipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> queuedInputQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> queuedOutputBuffer;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    uint32_t height{0};
    uint32_t width{0};
    uint32_t totalElements{0};
    uint32_t batchCount{0};
    uint32_t matrixElements{0};
    uint32_t rowTileCount{0};
    uint32_t colTileCount{0};
    uint32_t tasksPerBatch{0};
    uint32_t totalTaskCount{0};
    uint32_t blockIdx{0};
    uint32_t blockNum{1};
    uint32_t dims{0};
    uint32_t shape[MAXDIM]{};
    uint32_t perm[MAXDIM]{};
    uint32_t inputStride[MAXDIM]{};
    uint32_t outputStride[MAXDIM]{};
    uint32_t inputRowStride{0};
    uint32_t outputRowStride{0};
    uint32_t inputLastOutputPos{0};
    uint32_t pairBatchAxis{0};
    uint32_t pairBatchOutputPos{0};
    uint32_t pairAxisExtent{0};
    uint32_t pairAxisTileCount{0};
    uint32_t pairInnerExtent{0};
    uint32_t pairGroupCount{0};
    uint32_t pairOutputStride{0};
    uint32_t rectangleOutputBatchAxis{0};
    uint32_t rectangleOutputBatchPos{0};
    uint32_t rectangleOutputAxisExtent{0};
    uint32_t rectangleOutputAxisTileCount{0};
    uint32_t rectangleMiddleExtent{0};
    uint32_t rectangleGroupCount{0};
    uint32_t rectangleOutputAxisInputStride{0};
    uint32_t rectanglePackedRowStride{0};
    uint32_t rectanglePackedElements{0};
};

// A fixed suffix smaller than one DMA block makes each logical matrix element
// a short interleaved record.  Move 16x16 record tiles with two-dimensional
// DMA and perform the record transpose with scalar UB accesses.  This avoids
// both per-element GM gathers and rank-dependent index reconstruction.
template <typename T>
class TiledRecordTranspose {
public:
    __aicore__ inline TiledRecordTranspose(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const TransposeTilingData* tilingData)
    {
        const uint32_t dims = static_cast<uint32_t>(tilingData->dims);
        uint32_t prefix = 0U;
        while (prefix < dims && tilingData->perm[prefix] == static_cast<int64_t>(prefix)) {
            ++prefix;
        }
        uint32_t end = dims;
        while (end > prefix && tilingData->perm[end - 1U] == static_cast<int64_t>(end - 1U)) {
            --end;
        }
        const uint32_t split = static_cast<uint32_t>(tilingData->perm[prefix]);

        batchCount = 1U;
        height = 1U;
        width = 1U;
        inner = 1U;
        for (uint32_t i = 0U; i < prefix; ++i) {
            batchCount *= static_cast<uint32_t>(tilingData->shape[i]);
        }
        for (uint32_t i = prefix; i < split; ++i) {
            height *= static_cast<uint32_t>(tilingData->shape[i]);
        }
        for (uint32_t i = split; i < end; ++i) {
            width *= static_cast<uint32_t>(tilingData->shape[i]);
        }
        for (uint32_t i = end; i < dims; ++i) {
            inner *= static_cast<uint32_t>(tilingData->shape[i]);
        }

        totalElements = static_cast<uint32_t>(tilingData->totalnumber);
        matrixElements = height * width * inner;
        activeTileCols = UseGatherMask() ?
            ((VECTOR_REPEAT_BYTES / (inner * TYPE_BYTES) < TILE_COLS) ?
                 VECTOR_REPEAT_BYTES / (inner * TYPE_BYTES) : TILE_COLS) : TILE_COLS;
        const bool useSkinnyByteMatrix =
            TYPE_BYTES == 1U && inner == 1U && width <= MAX_SKINNY_BYTE_WIDTH;
        rowTileCount = CeilDiv(height, useSkinnyByteMatrix ? SKINNY_TILE_ROWS : TILE_ROWS);
        colTileCount = CeilDiv(width, activeTileCols);
        tasksPerBatch = rowTileCount * colTileCount;
        totalTaskCount = batchCount * tasksPerBatch;
        blockIdx = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();
        xGm.SetGlobalBuffer((__gm__ T*)x, totalElements);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalElements);
    }

    __aicore__ inline void Process()
    {
        if constexpr (TYPE_BYTES == 1U) {
            if (inner == 1U && width <= MAX_SKINNY_BYTE_WIDTH) {
                ProcessSkinnyByteMatrix();
                return;
            }
        }
        if (UseGatherMask()) {
            if constexpr (TYPE_BYTES == 4U) {
                BuildGatherMaskPatterns<uint32_t>();
            } else {
                BuildGatherMaskPatterns<uint16_t>();
            }
        }
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t batch = task / tasksPerBatch;
            const uint32_t matrixTask = task - batch * tasksPerBatch;
            const uint32_t rowTile = matrixTask / colTileCount;
            const uint32_t colTile = matrixTask - rowTile * colTileCount;
            ProcessTile(batch, rowTile * TILE_ROWS, colTile * activeTileCols);
        }
    }

private:
    __aicore__ inline void ProcessSkinnyByteMatrix()
    {
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t batch = task / tasksPerBatch;
            const uint32_t matrixTask = task - batch * tasksPerBatch;
            ProcessSkinnyByteMatrixTile(batch, matrixTask * SKINNY_TILE_ROWS);
        }
    }

    __aicore__ inline void ProcessSkinnyByteMatrixTile(uint32_t batch, uint32_t rowStart)
    {
        const uint32_t rows =
            (height - rowStart < SKINNY_TILE_ROWS) ? height - rowStart : SKINNY_TILE_ROWS;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN, SKINNY_INPUT_UB_OFFSET, SKINNY_INPUT_BUFFER_BYTES);
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT, SKINNY_OUTPUT_UB_OFFSET, SKINNY_OUTPUT_BUFFER_BYTES);

        const uint32_t inputOffset = batch * matrixElements + rowStart * width;
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), width, 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(DMA_BLOCK_BYTES - width), static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));

        for (uint32_t rowGroup = 0U; rowGroup < rows; rowGroup += SKINNY_VECTOR_ROWS) {
            uint64_t dstList[16];
            uint64_t srcList[16];
            for (uint32_t row = 0U; row < 16U; ++row) {
                srcList[row] = reinterpret_cast<uint64_t>(
                    input[(rowGroup + row) * DMA_BLOCK_BYTES].GetPhyAddr());
                dstList[row] = reinterpret_cast<uint64_t>(
                    output[row * SKINNY_TILE_ROWS + rowGroup].GetPhyAddr());
            }
            AscendC::TransDataTo5HDParams params(false, false, 1U, 0U, 0U);
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
            AscendC::PipeBarrier<PIPE_V>();
            if (rows - rowGroup > 16U) {
                for (uint32_t row = 0U; row < 16U; ++row) {
                    srcList[row] = reinterpret_cast<uint64_t>(
                        input[(rowGroup + 16U + row) * DMA_BLOCK_BYTES].GetPhyAddr());
                }
                params.dstHighHalf = true;
                AscendC::TransDataTo5HD<T>(dstList, srcList, params);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));

        const uint32_t outputOffset = batch * matrixElements + rowStart;
        const uint32_t alignedRows = AlignUp(rows, DMA_BLOCK_BYTES);
        const uint32_t outputSrcGapBlocks =
            (SKINNY_TILE_ROWS - alignedRows) / DMA_BLOCK_BYTES;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(width), rows, outputSrcGapBlocks, height - rows, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) const
    {
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t alignment) const
    {
        return CeilDiv(value, alignment) * alignment;
    }

    template <typename CopyT>
    __aicore__ inline void CopyRecords(const AscendC::LocalTensor<T>& input,
                                       const AscendC::LocalTensor<T>& output,
                                       uint32_t rows, uint32_t cols,
                                       uint32_t inputRowStride, uint32_t outputRowStride)
    {
        AscendC::LocalTensor<CopyT> inputWide = input.template ReinterpretCast<CopyT>();
        AscendC::LocalTensor<CopyT> outputWide = output.template ReinterpretCast<CopyT>();
        const uint32_t unitsPerRecord = inner * TYPE_BYTES / sizeof(CopyT);
        const uint32_t inputRowUnits = inputRowStride * TYPE_BYTES / sizeof(CopyT);
        const uint32_t outputRowUnits = outputRowStride * TYPE_BYTES / sizeof(CopyT);
        for (uint32_t col = 0U; col < cols; ++col) {
            for (uint32_t row = 0U; row < rows; ++row) {
                const uint32_t inputRecord = row * inputRowUnits + col * unitsPerRecord;
                const uint32_t outputRecord = col * outputRowUnits + row * unitsPerRecord;
                for (uint32_t unit = 0U; unit < unitsPerRecord; ++unit) {
                    outputWide.SetValue(outputRecord + unit, inputWide.GetValue(inputRecord + unit));
                }
            }
        }
    }

    template <uint32_t UNITS_PER_RECORD>
    __aicore__ inline void CopyFixedByteRecords(const AscendC::LocalTensor<T>& input,
                                                const AscendC::LocalTensor<T>& output,
                                                uint32_t rows, uint32_t cols,
                                                uint32_t inputRowStride,
                                                uint32_t outputRowStride)
    {
        for (uint32_t col = 0U; col < cols; ++col) {
            for (uint32_t row = 0U; row < rows; ++row) {
                const uint32_t inputRecord = row * inputRowStride + col * UNITS_PER_RECORD;
                const uint32_t outputRecord = col * outputRowStride + row * UNITS_PER_RECORD;
                for (uint32_t unit = 0U; unit < UNITS_PER_RECORD; ++unit) {
                    output.SetValue(outputRecord + unit, input.GetValue(inputRecord + unit));
                }
            }
        }
    }

    __aicore__ inline bool UseGatherMask() const
    {
        return (TYPE_BYTES != 1U || (inner & 1U) == 0U) &&
               inner * TYPE_BYTES <= VECTOR_REPEAT_BYTES;
    }

    template <typename UnitT>
    __aicore__ inline void BuildGatherMaskPatterns()
    {
        AscendC::LocalTensor<UnitT> patterns(
            AscendC::TPosition::VECCALC, PATTERN_UB_OFFSET,
            PATTERN_BUFFER_BYTES / sizeof(UnitT));
        const uint32_t unitsPerRecord = inner * TYPE_BYTES / sizeof(UnitT);
        constexpr uint32_t bitsPerWord = sizeof(UnitT) * 8U;
        constexpr uint32_t wordsPerPattern = PATTERN_STRIDE_BYTES / sizeof(UnitT);
        for (uint32_t col = 0U; col < activeTileCols; ++col) {
            const uint32_t firstUnit = col * unitsPerRecord;
            const uint32_t lastUnit = firstUnit + unitsPerRecord;
            for (uint32_t word = 0U; word < wordsPerPattern; ++word) {
                uint32_t value = 0U;
                const uint32_t firstBit = word * bitsPerWord;
                for (uint32_t bit = 0U; bit < bitsPerWord; ++bit) {
                    const uint32_t unit = firstBit + bit;
                    if (unit >= firstUnit && unit < lastUnit) {
                        value |= 1U << bit;
                    }
                }
                patterns.SetValue(col * wordsPerPattern + word, static_cast<UnitT>(value));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(static_cast<event_t>(PATTERN_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(static_cast<event_t>(PATTERN_EVENT));
    }

    template <typename UnitT>
    __aicore__ inline void GatherRecords(const AscendC::LocalTensor<T>& input,
                                         const AscendC::LocalTensor<T>& output,
                                         uint32_t rows, uint32_t cols,
                                         uint32_t outputRowStride)
    {
        AscendC::LocalTensor<UnitT> inputWide = input.template ReinterpretCast<UnitT>();
        AscendC::LocalTensor<UnitT> outputWide = output.template ReinterpretCast<UnitT>();
        AscendC::LocalTensor<UnitT> patterns(
            AscendC::TPosition::VECCALC, PATTERN_UB_OFFSET,
            PATTERN_BUFFER_BYTES / sizeof(UnitT));
        const uint32_t outputRowUnits = outputRowStride * TYPE_BYTES / sizeof(UnitT);
        constexpr uint32_t wordsPerPattern = PATTERN_STRIDE_BYTES / sizeof(UnitT);
        uint64_t selected = 0U;
        const AscendC::GatherMaskParams params{
            1U, static_cast<uint16_t>(rows), VECTOR_REPEAT_BYTES / DMA_BLOCK_BYTES, 0U};
        for (uint32_t col = 0U; col < cols; ++col) {
            AscendC::GatherMask(outputWide[col * outputRowUnits], inputWide,
                                patterns[col * wordsPerPattern], false, 0U, params, selected);
        }
    }

    __aicore__ inline void TransposeByteRecord(const AscendC::LocalTensor<T>& input,
                                               const AscendC::LocalTensor<T>& temporary,
                                               const AscendC::LocalTensor<T>& output)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        AscendC::TransDataTo5HDParams params(false, false, 1U, 0U, 0U);

        // Transpose the 16 strided records into byte planes.  DataCopyPad
        // gives every source record a full 32-byte row, so all VA addresses
        // satisfy the VNCHW 32-byte alignment requirement.
        for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
            srcList[row] = reinterpret_cast<uint64_t>(
                input[row * DMA_BLOCK_BYTES].GetPhyAddr());
            dstList[row] = reinterpret_cast<uint64_t>(
                temporary[row * DMA_BLOCK_BYTES].GetPhyAddr());
        }
        AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        if (inner > 16U) {
            for (uint32_t byte = 0U; byte < 16U; ++byte) {
                dstList[byte] = reinterpret_cast<uint64_t>(
                    temporary[(16U + byte) * DMA_BLOCK_BYTES].GetPhyAddr());
            }
            params.srcHighHalf = true;
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Transpose the byte planes back into row-major records, now packed
        // in aligned UB rows.  MTE3 removes the final row padding while
        // publishing the exact odd-byte records to GM.
        for (uint32_t byte = 0U; byte < 16U; ++byte) {
            srcList[byte] = reinterpret_cast<uint64_t>(
                temporary[byte * DMA_BLOCK_BYTES].GetPhyAddr());
            dstList[byte] = reinterpret_cast<uint64_t>(
                output[byte * DMA_BLOCK_BYTES].GetPhyAddr());
        }
        params.srcHighHalf = false;
        AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        if (inner > 16U) {
            for (uint32_t byte = 0U; byte < 16U; ++byte) {
                srcList[byte] = reinterpret_cast<uint64_t>(
                    temporary[(16U + byte) * DMA_BLOCK_BYTES].GetPhyAddr());
            }
            params.dstHighHalf = true;
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessOddByteRecordTileSerial(uint32_t batch, uint32_t rowStart,
                                                          uint32_t colStart, uint32_t rows,
                                                          uint32_t cols)
    {
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN, BYTE_INPUT_UB_OFFSET, BYTE_INPUT_BUFFER_BYTES);
        AscendC::LocalTensor<T> temporary(
            AscendC::TPosition::VECCALC, BYTE_SERIAL_TEMP_UB_OFFSET, BYTE_TEMP_BUFFER_BYTES);
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT, BYTE_SERIAL_OUTPUT_UB_OFFSET, BYTE_OUTPUT_BUFFER_BYTES);

        for (uint32_t col = 0U; col < cols; ++col) {
            const uint32_t inputOffset = batch * matrixElements +
                (rowStart * width + colStart + col) * inner;
            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(rows), inner,
                (width - 1U) * inner, 0U, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0U, static_cast<uint8_t>(DMA_BLOCK_BYTES - inner), static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                static_cast<event_t>(INPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                static_cast<event_t>(INPUT_EVENT));

            TransposeByteRecord(input, temporary, output);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                static_cast<event_t>(OUTPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                static_cast<event_t>(OUTPUT_EVENT));

            const uint32_t outputOffset = batch * matrixElements +
                ((colStart + col) * height + rowStart) * inner;
            AscendC::DataCopyExtParams outputParams{
                static_cast<uint16_t>(rows), inner, 0U, 0U, 0U};
            AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                static_cast<event_t>(OUTPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                static_cast<event_t>(OUTPUT_EVENT));
        }
    }

    __aicore__ inline void ProcessOddByteRecordTilePipelined(uint32_t batch, uint32_t rowStart,
                                                             uint32_t colStart, uint32_t rows,
                                                             uint32_t cols)
    {
        // Ping-pong all three stages.  V_MTE2 protects an input/temporary
        // bank until VNCHW has consumed it, while MTE3_V protects an output
        // bank until its strided store has completed.  This lets MTE2 for
        // column N+1 and MTE3 for column N-1 run beside VNCHW for column N.
        for (uint32_t buffer = 0U; buffer < BYTE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        for (uint32_t col = 0U; col < cols; ++col) {
            const uint32_t buffer = col & 1U;
            const event_t event = static_cast<event_t>(buffer);
            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                BYTE_INPUT_UB_OFFSET + buffer * BYTE_INPUT_BUFFER_BYTES,
                BYTE_INPUT_BUFFER_BYTES);
            AscendC::LocalTensor<T> temporary(
                AscendC::TPosition::VECCALC,
                BYTE_TEMP_UB_OFFSET + buffer * BYTE_TEMP_BUFFER_BYTES,
                BYTE_TEMP_BUFFER_BYTES);
            AscendC::LocalTensor<T> output(
                AscendC::TPosition::VECOUT,
                BYTE_OUTPUT_UB_OFFSET + buffer * BYTE_OUTPUT_BUFFER_BYTES,
                BYTE_OUTPUT_BUFFER_BYTES);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            const uint32_t inputOffset = batch * matrixElements +
                (rowStart * width + colStart + col) * inner;
            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(rows), inner,
                (width - 1U) * inner, 0U, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0U, static_cast<uint8_t>(DMA_BLOCK_BYTES - inner), static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            TransposeByteRecord(input, temporary, output);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

            const uint32_t outputOffset = batch * matrixElements +
                ((colStart + col) * height + rowStart) * inner;
            AscendC::DataCopyExtParams outputParams{
                static_cast<uint16_t>(rows), inner, 0U, 0U, 0U};
            AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        for (uint32_t buffer = 0U; buffer < BYTE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline void ProcessTile(uint32_t batch, uint32_t rowStart, uint32_t colStart)
    {
        const uint32_t rows = (height - rowStart < TILE_ROWS) ? height - rowStart : TILE_ROWS;
        const uint32_t cols =
            (width - colStart < activeTileCols) ? width - colStart : activeTileCols;
        if constexpr (TYPE_BYTES == 1U) {
            if ((inner & 1U) != 0U && inner >= MIN_VNCHW_BYTE_RECORD) {
                if (rows < TILE_ROWS || inner >= MIN_PIPELINED_BYTE_RECORD) {
                    ProcessOddByteRecordTilePipelined(batch, rowStart, colStart, rows, cols);
                } else {
                    ProcessOddByteRecordTileSerial(batch, rowStart, colStart, rows, cols);
                }
                return;
            }
        }
        const uint32_t inputRowElements = cols * inner;
        const bool useGatherMask = UseGatherMask();
        const uint32_t inputRowStride = useGatherMask ?
            VECTOR_REPEAT_BYTES / TYPE_BYTES : AlignUp(inputRowElements, DMA_BLOCK_ELEMENTS);
        const uint32_t outputRowElements = rows * inner;
        const uint32_t outputRowStride = AlignUp(outputRowElements, DMA_BLOCK_ELEMENTS);
        const uint32_t inputOffset =
            batch * matrixElements + (rowStart * width + colStart) * inner;

        AscendC::LocalTensor<T> input(AscendC::TPosition::VECIN, INPUT_UB_OFFSET, MAX_TILE_ELEMENTS);
        const uint32_t alignedInputRowBytes = AlignUp(inputRowElements * TYPE_BYTES, DMA_BLOCK_BYTES);
        const uint32_t inputDstGapBlocks = useGatherMask ?
            (VECTOR_REPEAT_BYTES - alignedInputRowBytes) / DMA_BLOCK_BYTES : 0U;
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), inputRowElements * TYPE_BYTES,
            (width - cols) * inner * TYPE_BYTES, inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedInputRowBytes / TYPE_BYTES - inputRowElements),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        if (useGatherMask) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(static_cast<event_t>(INPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(static_cast<event_t>(INPUT_EVENT));
        }

        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT, OUTPUT_UB_OFFSET, MAX_TILE_ELEMENTS);
        if (useGatherMask) {
            if constexpr (TYPE_BYTES == 4U) {
                GatherRecords<uint32_t>(input, output, rows, cols, outputRowStride);
            } else {
                GatherRecords<uint16_t>(input, output, rows, cols, outputRowStride);
            }
        } else {
            const uint32_t recordBytes = inner * TYPE_BYTES;
            if constexpr (TYPE_BYTES == 1U) {
                if (inner == 3U) {
                    CopyFixedByteRecords<3U>(
                        input, output, rows, cols, inputRowStride, outputRowStride);
                } else if (inner == 5U) {
                    CopyFixedByteRecords<5U>(
                        input, output, rows, cols, inputRowStride, outputRowStride);
                } else {
                    CopyRecords<uint8_t>(
                        input, output, rows, cols, inputRowStride, outputRowStride);
                }
            } else if ((recordBytes & 7U) == 0U) {
                CopyRecords<uint64_t>(input, output, rows, cols, inputRowStride, outputRowStride);
            } else if ((recordBytes & 3U) == 0U) {
                CopyRecords<uint32_t>(input, output, rows, cols, inputRowStride, outputRowStride);
            } else if ((recordBytes & 1U) == 0U) {
                CopyRecords<uint16_t>(input, output, rows, cols, inputRowStride, outputRowStride);
            } else {
                CopyRecords<uint8_t>(input, output, rows, cols, inputRowStride, outputRowStride);
            }
        }
        if (useGatherMask) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
        } else {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
        }

        const uint32_t outputOffset =
            batch * matrixElements + (colStart * height + rowStart) * inner;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), outputRowElements * TYPE_BYTES, 0U,
            (height - rows) * inner * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
    }

private:
    static constexpr uint32_t TYPE_BYTES = sizeof(T);
    static constexpr uint32_t DMA_BLOCK_BYTES = 32U;
    static constexpr uint32_t DMA_BLOCK_ELEMENTS = DMA_BLOCK_BYTES / TYPE_BYTES;
    static constexpr uint32_t TILE_ROWS = 16U;
    static constexpr uint32_t TILE_COLS = 16U;
    static constexpr uint32_t MAX_TILE_BYTES = 8U * 1024U;
    static constexpr uint32_t MAX_TILE_ELEMENTS = MAX_TILE_BYTES / TYPE_BYTES;
    static constexpr uint32_t INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t OUTPUT_UB_OFFSET = MAX_TILE_BYTES;
    static constexpr uint32_t VECTOR_REPEAT_BYTES = 256U;
    static constexpr uint32_t PATTERN_STRIDE_BYTES = 32U;
    static constexpr uint32_t PATTERN_BUFFER_BYTES = TILE_COLS * PATTERN_STRIDE_BYTES;
    static constexpr uint32_t PATTERN_UB_OFFSET = 2U * MAX_TILE_BYTES;
    static constexpr uint32_t BYTE_BUFFER_COUNT = 2U;
    static constexpr uint32_t BYTE_INPUT_BUFFER_BYTES = TILE_ROWS * DMA_BLOCK_BYTES;
    static constexpr uint32_t BYTE_TEMP_BUFFER_BYTES = DMA_BLOCK_BYTES * DMA_BLOCK_BYTES;
    static constexpr uint32_t BYTE_OUTPUT_BUFFER_BYTES = TILE_ROWS * DMA_BLOCK_BYTES;
    static constexpr uint32_t BYTE_INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t BYTE_TEMP_UB_OFFSET =
        BYTE_INPUT_UB_OFFSET + BYTE_BUFFER_COUNT * BYTE_INPUT_BUFFER_BYTES;
    static constexpr uint32_t BYTE_OUTPUT_UB_OFFSET =
        BYTE_TEMP_UB_OFFSET + BYTE_BUFFER_COUNT * BYTE_TEMP_BUFFER_BYTES;
    static constexpr uint32_t BYTE_SERIAL_TEMP_UB_OFFSET = MAX_TILE_BYTES;
    static constexpr uint32_t BYTE_SERIAL_OUTPUT_UB_OFFSET =
        BYTE_SERIAL_TEMP_UB_OFFSET + BYTE_TEMP_BUFFER_BYTES;
    static constexpr uint32_t MIN_VNCHW_BYTE_RECORD = 7U;
    static constexpr uint32_t MIN_PIPELINED_BYTE_RECORD = 15U;
    static constexpr uint32_t MAX_SKINNY_BYTE_WIDTH = 3U;
    static constexpr uint32_t SKINNY_TILE_ROWS = 256U;
    static constexpr uint32_t SKINNY_VECTOR_ROWS = 32U;
    static constexpr uint32_t SKINNY_INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t SKINNY_INPUT_BUFFER_BYTES =
        SKINNY_TILE_ROWS * DMA_BLOCK_BYTES;
    static constexpr uint32_t SKINNY_OUTPUT_UB_OFFSET = SKINNY_INPUT_BUFFER_BYTES;
    static constexpr uint32_t SKINNY_OUTPUT_BUFFER_BYTES =
        TILE_COLS * SKINNY_TILE_ROWS;
    static constexpr int32_t INPUT_EVENT = 0;
    static constexpr int32_t OUTPUT_EVENT = 0;
    static constexpr int32_t PATTERN_EVENT = 1;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    uint32_t height{0};
    uint32_t width{0};
    uint32_t inner{0};
    uint32_t batchCount{0};
    uint32_t matrixElements{0};
    uint32_t totalElements{0};
    uint32_t rowTileCount{0};
    uint32_t colTileCount{0};
    uint32_t activeTileCols{0};
    uint32_t tasksPerBatch{0};
    uint32_t totalTaskCount{0};
    uint32_t blockIdx{0};
    uint32_t blockNum{1};
};

// General permutations that retain a sub-block contiguous suffix are
// transposes of short records.  Pair the physical axis immediately before the
// suffix with the last non-suffix output axis.  All other coordinates are
// decoded once per tile; GM traffic remains two-dimensional and local.
template <typename T>
class GeneralRecordTranspose {
public:
    __aicore__ inline GeneralRecordTranspose(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const TransposeTilingData* tilingData)
    {
        dims = static_cast<uint32_t>(tilingData->dims);
        totalElements = static_cast<uint32_t>(tilingData->totalnumber);
        for (uint32_t i = 0U; i < dims; ++i) {
            shape[i] = static_cast<uint32_t>(tilingData->shape[i]);
            perm[i] = static_cast<uint32_t>(tilingData->perm[i]);
        }
        inputStride[dims - 1U] = 1U;
        outputStride[dims - 1U] = 1U;
        for (int32_t i = static_cast<int32_t>(dims) - 2; i >= 0; --i) {
            inputStride[i] = inputStride[i + 1] * shape[i + 1U];
            outputStride[i] = outputStride[i + 1] * shape[perm[i + 1U]];
        }

        suffixStart = dims - 1U;
        while (suffixStart > 0U && perm[suffixStart - 1U] + 1U == perm[suffixStart]) {
            --suffixStart;
        }
        suffixInputAxis = perm[suffixStart];
        inputContiguousAxis = suffixInputAxis - 1U;
        outputLastPos = suffixStart - 1U;
        outputLastAxis = perm[outputLastPos];
        for (uint32_t i = 0U; i < suffixStart; ++i) {
            if (perm[i] == inputContiguousAxis) {
                inputContiguousOutputPos = i;
            }
        }

        inner = 1U;
        for (uint32_t i = suffixStart; i < dims; ++i) {
            inner *= shape[perm[i]];
        }
        height = shape[outputLastAxis];
        width = shape[inputContiguousAxis];
        inputRowStride = inputStride[outputLastAxis];
        outputRowStride = outputStride[inputContiguousOutputPos];
        matrixElements = height * width * inner;
        batchCount = totalElements / matrixElements;
        activeTileCols = UseGatherMask() ?
            ((VECTOR_REPEAT_BYTES / (inner * TYPE_BYTES) < TILE_COLS) ?
                 VECTOR_REPEAT_BYTES / (inner * TYPE_BYTES) : TILE_COLS) : TILE_COLS;
        rowTileCount = CeilDiv(height, TILE_ROWS);
        colTileCount = CeilDiv(width, activeTileCols);
        tasksPerBatch = rowTileCount * colTileCount;
        totalTaskCount = batchCount * tasksPerBatch;
        blockIdx = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();
        xGm.SetGlobalBuffer((__gm__ T*)x, totalElements);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalElements);
    }

    __aicore__ inline void Process()
    {
        if (UseGatherMask()) {
            if constexpr (TYPE_BYTES == 4U) {
                BuildGatherMaskPatterns<uint32_t>();
            } else {
                BuildGatherMaskPatterns<uint16_t>();
            }
        }
        if (tasksPerBatch > 1U && batchCount >= blockNum) {
            for (uint32_t batch = blockIdx; batch < batchCount; batch += blockNum) {
                const uint32_t inputBatchOffset = InputBatchOffset(batch);
                const uint32_t outputBatchOffset = OutputBatchOffset(batch);
                for (uint32_t matrixTask = 0U; matrixTask < tasksPerBatch; ++matrixTask) {
                    const uint32_t rowTile = matrixTask / colTileCount;
                    const uint32_t colTile = matrixTask - rowTile * colTileCount;
                    ProcessTile(inputBatchOffset, outputBatchOffset,
                                rowTile * TILE_ROWS, colTile * activeTileCols);
                }
            }
            return;
        }
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t batch = task / tasksPerBatch;
            const uint32_t matrixTask = task - batch * tasksPerBatch;
            const uint32_t rowTile = matrixTask / colTileCount;
            const uint32_t colTile = matrixTask - rowTile * colTileCount;
            ProcessTile(InputBatchOffset(batch), OutputBatchOffset(batch),
                        rowTile * TILE_ROWS, colTile * activeTileCols);
        }
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) const
    {
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t alignment) const
    {
        return CeilDiv(value, alignment) * alignment;
    }

    __aicore__ inline uint32_t InputBatchOffset(uint32_t outputBatch) const
    {
        uint32_t offset = 0U;
        uint32_t remainder = outputBatch;
        for (int32_t i = static_cast<int32_t>(suffixStart) - 1; i >= 0; --i) {
            if (static_cast<uint32_t>(i) == inputContiguousOutputPos ||
                static_cast<uint32_t>(i) == outputLastPos) {
                continue;
            }
            const uint32_t axis = perm[i];
            const uint32_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            offset += coordinate * inputStride[axis];
        }
        return offset;
    }

    __aicore__ inline uint32_t OutputBatchOffset(uint32_t outputBatch) const
    {
        uint32_t offset = 0U;
        uint32_t remainder = outputBatch;
        for (int32_t i = static_cast<int32_t>(suffixStart) - 1; i >= 0; --i) {
            if (static_cast<uint32_t>(i) == inputContiguousOutputPos ||
                static_cast<uint32_t>(i) == outputLastPos) {
                continue;
            }
            const uint32_t axis = perm[i];
            const uint32_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            offset += coordinate * outputStride[i];
        }
        return offset;
    }

    template <typename CopyT>
    __aicore__ inline void CopyRecords(const AscendC::LocalTensor<T>& input,
                                       const AscendC::LocalTensor<T>& output,
                                       uint32_t rows, uint32_t cols,
                                       uint32_t inputLocalRowStride,
                                       uint32_t outputLocalRowStride)
    {
        AscendC::LocalTensor<CopyT> inputWide = input.template ReinterpretCast<CopyT>();
        AscendC::LocalTensor<CopyT> outputWide = output.template ReinterpretCast<CopyT>();
        const uint32_t unitsPerRecord = inner * TYPE_BYTES / sizeof(CopyT);
        const uint32_t inputRowUnits = inputLocalRowStride * TYPE_BYTES / sizeof(CopyT);
        const uint32_t outputRowUnits = outputLocalRowStride * TYPE_BYTES / sizeof(CopyT);
        for (uint32_t col = 0U; col < cols; ++col) {
            for (uint32_t row = 0U; row < rows; ++row) {
                const uint32_t inputRecord = row * inputRowUnits + col * unitsPerRecord;
                const uint32_t outputRecord = col * outputRowUnits + row * unitsPerRecord;
                for (uint32_t unit = 0U; unit < unitsPerRecord; ++unit) {
                    outputWide.SetValue(outputRecord + unit, inputWide.GetValue(inputRecord + unit));
                }
            }
        }
    }

    template <uint32_t UNITS_PER_RECORD>
    __aicore__ inline void CopyFixedByteRecords(const AscendC::LocalTensor<T>& input,
                                                const AscendC::LocalTensor<T>& output,
                                                uint32_t rows, uint32_t cols,
                                                uint32_t inputLocalRowStride,
                                                uint32_t outputLocalRowStride)
    {
        for (uint32_t col = 0U; col < cols; ++col) {
            for (uint32_t row = 0U; row < rows; ++row) {
                const uint32_t inputRecord =
                    row * inputLocalRowStride + col * UNITS_PER_RECORD;
                const uint32_t outputRecord =
                    col * outputLocalRowStride + row * UNITS_PER_RECORD;
                for (uint32_t unit = 0U; unit < UNITS_PER_RECORD; ++unit) {
                    output.SetValue(outputRecord + unit, input.GetValue(inputRecord + unit));
                }
            }
        }
    }

    __aicore__ inline void TransposeByteRecord(const AscendC::LocalTensor<T>& input,
                                               const AscendC::LocalTensor<T>& temporary,
                                               const AscendC::LocalTensor<T>& output)
    {
        uint64_t dstList[16];
        uint64_t srcList[16];
        AscendC::TransDataTo5HDParams params(false, false, 1U, 0U, 0U);

        for (uint32_t row = 0U; row < TILE_ROWS; ++row) {
            srcList[row] = reinterpret_cast<uint64_t>(
                input[row * DMA_BLOCK_BYTES].GetPhyAddr());
            dstList[row] = reinterpret_cast<uint64_t>(
                temporary[row * DMA_BLOCK_BYTES].GetPhyAddr());
        }
        AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        if (inner > 16U) {
            for (uint32_t byte = 0U; byte < 16U; ++byte) {
                dstList[byte] = reinterpret_cast<uint64_t>(
                    temporary[(16U + byte) * DMA_BLOCK_BYTES].GetPhyAddr());
            }
            params.srcHighHalf = true;
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
            AscendC::PipeBarrier<PIPE_V>();
        }

        for (uint32_t byte = 0U; byte < 16U; ++byte) {
            srcList[byte] = reinterpret_cast<uint64_t>(
                temporary[byte * DMA_BLOCK_BYTES].GetPhyAddr());
            dstList[byte] = reinterpret_cast<uint64_t>(
                output[byte * DMA_BLOCK_BYTES].GetPhyAddr());
        }
        params.srcHighHalf = false;
        AscendC::TransDataTo5HD<T>(dstList, srcList, params);
        AscendC::PipeBarrier<PIPE_V>();
        if (inner > 16U) {
            for (uint32_t byte = 0U; byte < 16U; ++byte) {
                srcList[byte] = reinterpret_cast<uint64_t>(
                    temporary[(16U + byte) * DMA_BLOCK_BYTES].GetPhyAddr());
            }
            params.dstHighHalf = true;
            AscendC::TransDataTo5HD<T>(dstList, srcList, params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessOddByteRecordTileSerial(
        uint32_t inputBatchOffset, uint32_t outputBatchOffset,
        uint32_t rowStart, uint32_t colStart, uint32_t rows, uint32_t cols)
    {
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN, BYTE_INPUT_UB_OFFSET, BYTE_INPUT_BUFFER_BYTES);
        AscendC::LocalTensor<T> temporary(
            AscendC::TPosition::VECCALC, BYTE_SERIAL_TEMP_UB_OFFSET, BYTE_TEMP_BUFFER_BYTES);
        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT, BYTE_SERIAL_OUTPUT_UB_OFFSET, BYTE_OUTPUT_BUFFER_BYTES);

        for (uint32_t col = 0U; col < cols; ++col) {
            const uint32_t inputOffset = inputBatchOffset +
                rowStart * inputRowStride + (colStart + col) * inner;
            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(rows), inner,
                (inputRowStride - inner) * TYPE_BYTES, 0U, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0U, static_cast<uint8_t>(DMA_BLOCK_BYTES - inner), static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
                static_cast<event_t>(INPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
                static_cast<event_t>(INPUT_EVENT));

            TransposeByteRecord(input, temporary, output);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(
                static_cast<event_t>(OUTPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(
                static_cast<event_t>(OUTPUT_EVENT));

            const uint32_t outputOffset = outputBatchOffset +
                (colStart + col) * outputRowStride + rowStart * inner;
            AscendC::DataCopyExtParams outputParams{
                static_cast<uint16_t>(rows), inner, 0U, 0U, 0U};
            AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                static_cast<event_t>(OUTPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                static_cast<event_t>(OUTPUT_EVENT));
        }
    }

    __aicore__ inline void ProcessOddByteRecordTilePipelined(
        uint32_t inputBatchOffset, uint32_t outputBatchOffset,
        uint32_t rowStart, uint32_t colStart, uint32_t rows, uint32_t cols)
    {
        for (uint32_t buffer = 0U; buffer < BYTE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        for (uint32_t col = 0U; col < cols; ++col) {
            const uint32_t buffer = col & 1U;
            const event_t event = static_cast<event_t>(buffer);
            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                BYTE_INPUT_UB_OFFSET + buffer * BYTE_INPUT_BUFFER_BYTES,
                BYTE_INPUT_BUFFER_BYTES);
            AscendC::LocalTensor<T> temporary(
                AscendC::TPosition::VECCALC,
                BYTE_TEMP_UB_OFFSET + buffer * BYTE_TEMP_BUFFER_BYTES,
                BYTE_TEMP_BUFFER_BYTES);
            AscendC::LocalTensor<T> output(
                AscendC::TPosition::VECOUT,
                BYTE_OUTPUT_UB_OFFSET + buffer * BYTE_OUTPUT_BUFFER_BYTES,
                BYTE_OUTPUT_BUFFER_BYTES);

            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            const uint32_t inputOffset = inputBatchOffset +
                rowStart * inputRowStride + (colStart + col) * inner;
            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(rows), inner,
                (inputRowStride - inner) * TYPE_BYTES, 0U, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0U, static_cast<uint8_t>(DMA_BLOCK_BYTES - inner), static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            TransposeByteRecord(input, temporary, output);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

            const uint32_t outputOffset = outputBatchOffset +
                (colStart + col) * outputRowStride + rowStart * inner;
            AscendC::DataCopyExtParams outputParams{
                static_cast<uint16_t>(rows), inner, 0U, 0U, 0U};
            AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        for (uint32_t buffer = 0U; buffer < BYTE_BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline bool UseGatherMask() const
    {
        // GatherMask removes the scalar record shuffle once a logical matrix
        // is large enough to amortize pattern setup.  Output ownership and the
        // row/column task grid stay identical to the scalar path; only the UB
        // rearrangement changes.  C220 GatherMask supports 16- and 32-bit
        // units, so byte records must contain an even number of bytes.
        const uint32_t recordBytes = inner * TYPE_BYTES;
        const bool supportedUnit = TYPE_BYTES != 1U || (inner & 1U) == 0U;
        return supportedUnit && recordBytes <= VECTOR_REPEAT_BYTES &&
               height * width >= MIN_GATHER_MATRIX_RECORDS;
    }

    template <typename UnitT>
    __aicore__ inline void BuildGatherMaskPatterns()
    {
        AscendC::LocalTensor<UnitT> patterns(
            AscendC::TPosition::VECCALC, PATTERN_UB_OFFSET,
            PATTERN_BUFFER_BYTES / sizeof(UnitT));
        const uint32_t unitsPerRecord = inner * TYPE_BYTES / sizeof(UnitT);
        constexpr uint32_t bitsPerWord = sizeof(UnitT) * 8U;
        constexpr uint32_t wordsPerPattern = PATTERN_STRIDE_BYTES / sizeof(UnitT);
        for (uint32_t col = 0U; col < activeTileCols; ++col) {
            const uint32_t firstUnit = col * unitsPerRecord;
            const uint32_t lastUnit = firstUnit + unitsPerRecord;
            for (uint32_t word = 0U; word < wordsPerPattern; ++word) {
                uint32_t value = 0U;
                const uint32_t firstBit = word * bitsPerWord;
                for (uint32_t bit = 0U; bit < bitsPerWord; ++bit) {
                    const uint32_t unit = firstBit + bit;
                    if (unit >= firstUnit && unit < lastUnit) {
                        value |= 1U << bit;
                    }
                }
                patterns.SetValue(col * wordsPerPattern + word, static_cast<UnitT>(value));
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::S_V>(static_cast<event_t>(PATTERN_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(static_cast<event_t>(PATTERN_EVENT));
    }

    template <typename UnitT>
    __aicore__ inline void GatherRecords(const AscendC::LocalTensor<T>& input,
                                         const AscendC::LocalTensor<T>& output,
                                         uint32_t rows, uint32_t cols,
                                         uint32_t outputLocalRowStride)
    {
        AscendC::LocalTensor<UnitT> inputWide = input.template ReinterpretCast<UnitT>();
        AscendC::LocalTensor<UnitT> outputWide = output.template ReinterpretCast<UnitT>();
        AscendC::LocalTensor<UnitT> patterns(
            AscendC::TPosition::VECCALC, PATTERN_UB_OFFSET,
            PATTERN_BUFFER_BYTES / sizeof(UnitT));
        const uint32_t outputRowUnits = outputLocalRowStride * TYPE_BYTES / sizeof(UnitT);
        constexpr uint32_t wordsPerPattern = PATTERN_STRIDE_BYTES / sizeof(UnitT);
        uint64_t selected = 0U;
        const AscendC::GatherMaskParams params{
            1U, static_cast<uint16_t>(rows), VECTOR_REPEAT_BYTES / DMA_BLOCK_BYTES, 0U};
        for (uint32_t col = 0U; col < cols; ++col) {
            AscendC::GatherMask(outputWide[col * outputRowUnits], inputWide,
                                patterns[col * wordsPerPattern], false, 0U, params, selected);
        }
    }

    __aicore__ inline void ProcessTile(uint32_t inputBatchOffset, uint32_t outputBatchOffset,
                                       uint32_t rowStart, uint32_t colStart)
    {
        const uint32_t rows = (height - rowStart < TILE_ROWS) ? height - rowStart : TILE_ROWS;
        const uint32_t cols =
            (width - colStart < activeTileCols) ? width - colStart : activeTileCols;
        if constexpr (TYPE_BYTES == 1U) {
            if ((inner & 1U) != 0U && inner >= MIN_VNCHW_BYTE_RECORD) {
                if (rows < TILE_ROWS || inner >= MIN_PIPELINED_BYTE_RECORD) {
                    ProcessOddByteRecordTilePipelined(
                        inputBatchOffset, outputBatchOffset,
                        rowStart, colStart, rows, cols);
                } else {
                    ProcessOddByteRecordTileSerial(
                        inputBatchOffset, outputBatchOffset,
                        rowStart, colStart, rows, cols);
                }
                return;
            }
        }
        const uint32_t inputRowElements = cols * inner;
        const bool useGatherMask = UseGatherMask();
        const uint32_t inputLocalRowStride = useGatherMask ?
            VECTOR_REPEAT_BYTES / TYPE_BYTES : AlignUp(inputRowElements, DMA_BLOCK_ELEMENTS);
        const uint32_t outputRowElements = rows * inner;
        const uint32_t outputLocalRowStride = AlignUp(outputRowElements, DMA_BLOCK_ELEMENTS);
        const uint32_t inputOffset =
            inputBatchOffset + rowStart * inputRowStride + colStart * inner;

        AscendC::LocalTensor<T> input(AscendC::TPosition::VECIN, INPUT_UB_OFFSET, MAX_TILE_ELEMENTS);
        const uint32_t alignedInputRowBytes = AlignUp(inputRowElements * TYPE_BYTES, DMA_BLOCK_BYTES);
        const uint32_t inputDstGapBlocks = useGatherMask ?
            (VECTOR_REPEAT_BYTES - alignedInputRowBytes) / DMA_BLOCK_BYTES : 0U;
        AscendC::DataCopyExtParams inputParams{
            static_cast<uint16_t>(rows), inputRowElements * TYPE_BYTES,
            (inputRowStride - inputRowElements) * TYPE_BYTES, inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U, static_cast<uint8_t>(alignedInputRowBytes / TYPE_BYTES - inputRowElements),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputOffset], inputParams, pad);
        if (useGatherMask) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(static_cast<event_t>(INPUT_EVENT));
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(static_cast<event_t>(INPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(static_cast<event_t>(INPUT_EVENT));
        }

        AscendC::LocalTensor<T> output(
            AscendC::TPosition::VECOUT, OUTPUT_UB_OFFSET, MAX_TILE_ELEMENTS);
        if (useGatherMask) {
            if constexpr (TYPE_BYTES == 4U) {
                GatherRecords<uint32_t>(input, output, rows, cols, outputLocalRowStride);
            } else {
                GatherRecords<uint16_t>(input, output, rows, cols, outputLocalRowStride);
            }
        } else {
            const uint32_t recordBytes = inner * TYPE_BYTES;
            if constexpr (TYPE_BYTES == 1U) {
                if (inner == 3U) {
                    CopyFixedByteRecords<3U>(input, output, rows, cols,
                                            inputLocalRowStride, outputLocalRowStride);
                } else if (inner == 5U) {
                    CopyFixedByteRecords<5U>(input, output, rows, cols,
                                            inputLocalRowStride, outputLocalRowStride);
                } else {
                    CopyRecords<uint8_t>(input, output, rows, cols,
                                         inputLocalRowStride, outputLocalRowStride);
                }
            } else if ((recordBytes & 7U) == 0U) {
                CopyRecords<uint64_t>(input, output, rows, cols, inputLocalRowStride, outputLocalRowStride);
            } else if ((recordBytes & 3U) == 0U) {
                CopyRecords<uint32_t>(input, output, rows, cols, inputLocalRowStride, outputLocalRowStride);
            } else if ((recordBytes & 1U) == 0U) {
                CopyRecords<uint16_t>(input, output, rows, cols, inputLocalRowStride, outputLocalRowStride);
            } else {
                CopyRecords<uint8_t>(input, output, rows, cols, inputLocalRowStride, outputLocalRowStride);
            }
        }
        if (useGatherMask) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
        } else {
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(static_cast<event_t>(OUTPUT_EVENT));
        }

        const uint32_t outputOffset =
            outputBatchOffset + colStart * outputRowStride + rowStart * inner;
        AscendC::DataCopyExtParams outputParams{
            static_cast<uint16_t>(cols), outputRowElements * TYPE_BYTES, 0U,
            (outputRowStride - outputRowElements) * TYPE_BYTES, 0U};
        AscendC::DataCopyPad(zGm[outputOffset], output, outputParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(OUTPUT_EVENT));
    }

private:
    static constexpr uint32_t TYPE_BYTES = sizeof(T);
    static constexpr uint32_t DMA_BLOCK_BYTES = 32U;
    static constexpr uint32_t DMA_BLOCK_ELEMENTS = DMA_BLOCK_BYTES / TYPE_BYTES;
    static constexpr uint32_t TILE_ROWS = 16U;
    static constexpr uint32_t TILE_COLS = 16U;
    static constexpr uint32_t MIN_GATHER_MATRIX_RECORDS = 512U;
    static constexpr uint32_t MAX_TILE_BYTES = 8U * 1024U;
    static constexpr uint32_t MAX_TILE_ELEMENTS = MAX_TILE_BYTES / TYPE_BYTES;
    static constexpr uint32_t INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t OUTPUT_UB_OFFSET = MAX_TILE_BYTES;
    static constexpr uint32_t VECTOR_REPEAT_BYTES = 256U;
    static constexpr uint32_t PATTERN_STRIDE_BYTES = 32U;
    static constexpr uint32_t PATTERN_BUFFER_BYTES = TILE_COLS * PATTERN_STRIDE_BYTES;
    static constexpr uint32_t PATTERN_UB_OFFSET = 2U * MAX_TILE_BYTES;
    static constexpr uint32_t BYTE_BUFFER_COUNT = 2U;
    static constexpr uint32_t BYTE_INPUT_BUFFER_BYTES = TILE_ROWS * DMA_BLOCK_BYTES;
    static constexpr uint32_t BYTE_TEMP_BUFFER_BYTES = DMA_BLOCK_BYTES * DMA_BLOCK_BYTES;
    static constexpr uint32_t BYTE_OUTPUT_BUFFER_BYTES = TILE_ROWS * DMA_BLOCK_BYTES;
    static constexpr uint32_t BYTE_INPUT_UB_OFFSET = 0U;
    static constexpr uint32_t BYTE_TEMP_UB_OFFSET =
        BYTE_INPUT_UB_OFFSET + BYTE_BUFFER_COUNT * BYTE_INPUT_BUFFER_BYTES;
    static constexpr uint32_t BYTE_OUTPUT_UB_OFFSET =
        BYTE_TEMP_UB_OFFSET + BYTE_BUFFER_COUNT * BYTE_TEMP_BUFFER_BYTES;
    static constexpr uint32_t BYTE_SERIAL_TEMP_UB_OFFSET = MAX_TILE_BYTES;
    static constexpr uint32_t BYTE_SERIAL_OUTPUT_UB_OFFSET =
        BYTE_SERIAL_TEMP_UB_OFFSET + BYTE_TEMP_BUFFER_BYTES;
    static constexpr uint32_t MIN_VNCHW_BYTE_RECORD = 7U;
    static constexpr uint32_t MIN_PIPELINED_BYTE_RECORD = 15U;
    static constexpr int32_t INPUT_EVENT = 0;
    static constexpr int32_t OUTPUT_EVENT = 0;
    static constexpr int32_t PATTERN_EVENT = 1;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    uint32_t dims{0};
    uint32_t totalElements{0};
    uint32_t shape[MAXDIM]{};
    uint32_t perm[MAXDIM]{};
    uint32_t inputStride[MAXDIM]{};
    uint32_t outputStride[MAXDIM]{};
    uint32_t suffixStart{0};
    uint32_t suffixInputAxis{0};
    uint32_t inputContiguousAxis{0};
    uint32_t outputLastPos{0};
    uint32_t outputLastAxis{0};
    uint32_t inputContiguousOutputPos{0};
    uint32_t inner{0};
    uint32_t height{0};
    uint32_t width{0};
    uint32_t inputRowStride{0};
    uint32_t outputRowStride{0};
    uint32_t matrixElements{0};
    uint32_t batchCount{0};
    uint32_t rowTileCount{0};
    uint32_t colTileCount{0};
    uint32_t activeTileCols{0};
    uint32_t tasksPerBatch{0};
    uint32_t totalTaskCount{0};
    uint32_t blockIdx{0};
    uint32_t blockNum{1};
};

// Permutations that preserve a contiguous input suffix are copies of larger
// contiguous blocks. Compute one base address per block and move 32 KiB chunks
// with DMA instead of recomputing an input address for every element.
template <typename T>
class ContiguousSuffixTranspose {
public:
    __aicore__ inline ContiguousSuffixTranspose(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const TransposeTilingData* tilingData)
    {
        dims = static_cast<uint32_t>(tilingData->dims);
        totalElements = static_cast<uint32_t>(tilingData->totalnumber);
        blockIdx = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();
        for (uint32_t i = 0; i < dims; ++i) {
            shape[i] = static_cast<uint32_t>(tilingData->shape[i]);
            perm[i] = static_cast<uint32_t>(tilingData->perm[i]);
        }
        inputStride[dims - 1U] = 1U;
        for (int32_t i = static_cast<int32_t>(dims) - 2; i >= 0; --i) {
            inputStride[i] = inputStride[i + 1] * shape[i + 1];
        }

        suffixStart = dims - 1U;
        while (suffixStart > 0U && perm[suffixStart - 1U] + 1U == perm[suffixStart]) {
            --suffixStart;
        }
        blockElements = 1U;
        for (uint32_t i = suffixStart; i < dims; ++i) {
            blockElements *= shape[perm[i]];
        }
        outerBlockCount = totalElements / blockElements;
        useRecordGroups = suffixStart > 0U && blockElements <= CHUNK_ELEMENTS;
        if (useRecordGroups) {
            groupSize = shape[perm[suffixStart - 1U]];
            groupInputStride = inputStride[perm[suffixStart - 1U]];
            alignedRecordElements =
                CeilDiv(blockElements * TYPE_BYTES, DMA_BLOCK_BYTES) * DMA_BLOCK_ELEMENTS;
            recordsPerTask = CHUNK_ELEMENTS / alignedRecordElements;
            if (recordsPerTask > groupSize) {
                recordsPerTask = groupSize;
            }
            recordTasksPerGroup = CeilDiv(groupSize, recordsPerTask);
            outerGroupCount = outerBlockCount / groupSize;
            totalTaskCount = outerGroupCount * recordTasksPerGroup;

            // When an input-contiguous axis remains outside the copied suffix,
            // two adjacent coordinates can share one wider record chain.  The
            // compact output groups remain separated by their derived output
            // stride and are published by one two-dimensional descriptor.
            if (recordTasksPerGroup == 1U && suffixStart >= 2U &&
                perm[suffixStart] > 0U) {
                const uint32_t inputContiguousAxis =
                    perm[suffixStart] - 1U;
                uint32_t outputStride = 1U;
                bool foundContiguousAxis = false;
                for (int32_t i = static_cast<int32_t>(suffixStart) - 2;
                     i >= 0; --i) {
                    const uint32_t axis = perm[i];
                    if (axis == inputContiguousAxis) {
                        foundContiguousAxis = true;
                        break;
                    }
                    outputStride *= shape[axis];
                }
                if (foundContiguousAxis) {
                    mergedAxisExtent = shape[inputContiguousAxis];
                    mergedAxisOutputStride = outputStride;
                    mergedAxisTileCount =
                        CeilDiv(mergedAxisExtent, MERGED_GROUP_COUNT);
                    mergedTaskCount =
                        (outerGroupCount / mergedAxisExtent) *
                        mergedAxisTileCount;
                    mergedAlignedRecordElements =
                        CeilDiv(MERGED_GROUP_COUNT * blockElements *
                                    TYPE_BYTES,
                                DMA_BLOCK_BYTES) *
                        DMA_BLOCK_ELEMENTS;
                    mergedAlignedGroupElements =
                        CeilDiv(groupSize * blockElements * TYPE_BYTES,
                                DMA_BLOCK_BYTES) *
                        DMA_BLOCK_ELEMENTS;
                    const uint32_t mergedInputElements =
                        groupSize * mergedAlignedRecordElements;
                    const uint32_t mergedOutputElements =
                        MERGED_GROUP_COUNT * mergedAlignedGroupElements;
                    const uint64_t maximumGatherOffsetBytes =
                        (static_cast<uint64_t>(groupSize - 1U) *
                             mergedAlignedRecordElements +
                         MERGED_GROUP_COUNT * blockElements - 1U) *
                        TYPE_BYTES;
                    useMergedRecordGroups =
                        mergedAxisExtent >= MERGED_GROUP_COUNT &&
                        groupInputStride >=
                            MERGED_GROUP_COUNT * blockElements &&
                        mergedAlignedRecordElements -
                                MERGED_GROUP_COUNT * blockElements <=
                            255U &&
                        mergedInputElements <= CHUNK_ELEMENTS &&
                        mergedOutputElements <= CHUNK_ELEMENTS &&
                        mergedOutputElements <= OFFSET_ELEMENTS &&
                        maximumGatherOffsetBytes <
                            MAXIMUM_GATHER_OFFSET_BYTES;
                    if (useMergedRecordGroups &&
                        (groupSize & 1U) == 0U) {
                        wideRecordsPerTask = groupSize / 2U;
                        wideMergedAxisTileCount =
                            CeilDiv(mergedAxisExtent,
                                    WIDE_MERGED_GROUP_COUNT);
                        wideMergedTaskCount =
                            (outerGroupCount / mergedAxisExtent) *
                            wideMergedAxisTileCount *
                            WIDE_RECORD_TASK_COUNT;
                        wideMergedAlignedRecordElements =
                            CeilDiv(WIDE_MERGED_GROUP_COUNT * blockElements *
                                        TYPE_BYTES,
                                    DMA_BLOCK_BYTES) *
                            DMA_BLOCK_ELEMENTS;
                        wideMergedAlignedGroupElements =
                            CeilDiv(wideRecordsPerTask * blockElements *
                                        TYPE_BYTES,
                                    DMA_BLOCK_BYTES) *
                            DMA_BLOCK_ELEMENTS;
                        const uint32_t wideInputElements =
                            wideRecordsPerTask *
                            wideMergedAlignedRecordElements;
                        const uint32_t wideOutputElements =
                            WIDE_MERGED_GROUP_COUNT *
                            wideMergedAlignedGroupElements;
                        const uint64_t wideMaximumGatherOffsetBytes =
                            (static_cast<uint64_t>(
                                 wideRecordsPerTask - 1U) *
                                 wideMergedAlignedRecordElements +
                             WIDE_MERGED_GROUP_COUNT * blockElements - 1U) *
                            TYPE_BYTES;
                        useWideMergedRecordGroups =
                            mergedAxisExtent >= WIDE_MERGED_GROUP_COUNT &&
                            wideMergedTaskCount == mergedTaskCount &&
                            groupInputStride >=
                                WIDE_MERGED_GROUP_COUNT * blockElements &&
                            wideMergedAlignedRecordElements -
                                    WIDE_MERGED_GROUP_COUNT * blockElements <=
                                255U &&
                            wideInputElements <= CHUNK_ELEMENTS &&
                            wideOutputElements <= CHUNK_ELEMENTS &&
                            wideOutputElements <= OFFSET_ELEMENTS &&
                            wideMaximumGatherOffsetBytes <
                                MAXIMUM_GATHER_OFFSET_BYTES;
                        if (useWideMergedRecordGroups &&
                            wideMergedTaskCount >=
                                blockNum *
                                    MIN_FIVE_MERGED_TASKS_PER_CORE) {
                            fiveMergedAxisTileCount =
                                CeilDiv(mergedAxisExtent,
                                        FIVE_MERGED_GROUP_COUNT);
                            fiveMergedTaskCount =
                                (outerGroupCount / mergedAxisExtent) *
                                fiveMergedAxisTileCount *
                                WIDE_RECORD_TASK_COUNT;
                            // Five-coordinate offset synthesis is fixed per
                            // AIV.  Require enough DMA work to amortize it;
                            // the lower-bound reach test avoids constructing
                            // tables that cannot satisfy Gather's 8-KiB span.
                            const uint64_t fiveMinimumGatherReachBytes =
                                static_cast<uint64_t>(wideRecordsPerTask) *
                                FIVE_MERGED_GROUP_COUNT * blockElements *
                                TYPE_BYTES;
                            if (fiveMergedTaskCount >=
                                    blockNum *
                                        MIN_FIVE_MERGED_TASKS_PER_CORE &&
                                fiveMinimumGatherReachBytes <
                                    MAXIMUM_GATHER_OFFSET_BYTES) {
                                fiveMergedAlignedRecordElements =
                                    CeilDiv(
                                        FIVE_MERGED_GROUP_COUNT *
                                            blockElements * TYPE_BYTES,
                                        DMA_BLOCK_BYTES) *
                                    DMA_BLOCK_ELEMENTS;
                                const uint32_t fiveInputElements =
                                    wideRecordsPerTask *
                                    fiveMergedAlignedRecordElements;
                                const uint32_t fiveOutputElements =
                                    FIVE_MERGED_GROUP_COUNT *
                                    wideMergedAlignedGroupElements;
                                const uint64_t
                                    fiveMaximumGatherOffsetBytes =
                                        (static_cast<uint64_t>(
                                             wideRecordsPerTask - 1U) *
                                             fiveMergedAlignedRecordElements +
                                         FIVE_MERGED_GROUP_COUNT *
                                                 blockElements -
                                             1U) *
                                        TYPE_BYTES;
                                useFiveMergedRecordGroups =
                                    mergedAxisExtent >=
                                        FIVE_MERGED_GROUP_COUNT &&
                                    fiveMergedTaskCount <=
                                        wideMergedTaskCount &&
                                    groupInputStride >=
                                        FIVE_MERGED_GROUP_COUNT *
                                            blockElements &&
                                    fiveMergedAlignedRecordElements -
                                            FIVE_MERGED_GROUP_COUNT *
                                                blockElements <=
                                        255U &&
                                    fiveInputElements <= CHUNK_ELEMENTS &&
                                    fiveOutputElements <= CHUNK_ELEMENTS &&
                                    fiveOutputElements <= OFFSET_ELEMENTS &&
                                    fiveMaximumGatherOffsetBytes <
                                        MAXIMUM_GATHER_OFFSET_BYTES;
                                if (useFiveMergedRecordGroups) {
                                    tenMergedAxisTileCount =
                                        CeilDiv(mergedAxisExtent,
                                                TEN_MERGED_GROUP_COUNT);
                                    tenMergedTaskCount =
                                        (outerGroupCount / mergedAxisExtent) *
                                        tenMergedAxisTileCount *
                                        WIDE_RECORD_TASK_COUNT;
                                    tenMergedAlignedRecordElements =
                                        CeilDiv(
                                            TEN_MERGED_GROUP_COUNT *
                                                blockElements * TYPE_BYTES,
                                            DMA_BLOCK_BYTES) *
                                        DMA_BLOCK_ELEMENTS;
                                    const uint32_t tenInputElements =
                                        wideRecordsPerTask *
                                        tenMergedAlignedRecordElements;
                                    const uint32_t tenOutputElements =
                                        TEN_MERGED_GROUP_COUNT *
                                        wideMergedAlignedGroupElements;
                                    // Use the widest coordinate group whose
                                    // input and compact output both fit their
                                    // 32-KiB generations, while retaining
                                    // enough tasks to amortize the shared
                                    // offset table.
                                    useTenMergedRecordGroups =
                                        mergedAxisExtent >=
                                            TEN_MERGED_GROUP_COUNT &&
                                        groupInputStride >=
                                            TEN_MERGED_GROUP_COUNT *
                                                blockElements &&
                                        tenMergedAlignedRecordElements -
                                                TEN_MERGED_GROUP_COUNT *
                                                    blockElements <=
                                            255U &&
                                        tenInputElements <= CHUNK_ELEMENTS &&
                                        tenOutputElements <= CHUNK_ELEMENTS &&
                                        tenMergedTaskCount >=
                                            blockNum *
                                                MIN_TEN_MERGED_TASKS_PER_CORE;
                                    if (useTenMergedRecordGroups) {
                                        twentyFiveMergedAxisTileCount =
                                            CeilDiv(
                                                mergedAxisExtent,
                                                TWENTY_FIVE_MERGED_GROUP_COUNT);
                                        twentyFiveMergedTaskCount =
                                            (outerGroupCount /
                                             mergedAxisExtent) *
                                            twentyFiveMergedAxisTileCount *
                                            WIDE_RECORD_TASK_COUNT;
                                        twentyFiveMergedAlignedRecordElements =
                                            CeilDiv(
                                                TWENTY_FIVE_MERGED_GROUP_COUNT *
                                                    blockElements * TYPE_BYTES,
                                                DMA_BLOCK_BYTES) *
                                            DMA_BLOCK_ELEMENTS;
                                        const uint32_t twentyFiveInputBytes =
                                            wideRecordsPerTask *
                                            twentyFiveMergedAlignedRecordElements *
                                            TYPE_BYTES;
                                        const uint32_t twentyFiveOutputBytes =
                                            TWENTY_FIVE_MERGED_GROUP_COUNT *
                                            wideMergedAlignedGroupElements *
                                            TYPE_BYTES;
                                        const uint32_t thirteenOffsetBytes =
                                            THIRTEEN_MERGED_GROUP_COUNT *
                                            wideMergedAlignedGroupElements *
                                            sizeof(uint32_t);
                                        const uint32_t thirteenGatherReachElements =
                                            (wideRecordsPerTask - 1U) *
                                                twentyFiveMergedAlignedRecordElements +
                                            THIRTEEN_MERGED_GROUP_COUNT *
                                                blockElements;
                                        useTwentyFiveMergedRecordGroups =
                                            mergedAxisExtent >=
                                                TWENTY_FIVE_MERGED_GROUP_COUNT &&
                                            groupInputStride >=
                                                TWENTY_FIVE_MERGED_GROUP_COUNT *
                                                    blockElements &&
                                            twentyFiveMergedAlignedRecordElements -
                                                    TWENTY_FIVE_MERGED_GROUP_COUNT *
                                                        blockElements <=
                                                255U &&
                                            twentyFiveInputBytes <=
                                                TWENTY_FIVE_BANK_BYTES &&
                                            twentyFiveOutputBytes <=
                                                TWENTY_FIVE_BANK_BYTES &&
                                            thirteenOffsetBytes <=
                                                TWENTY_FIVE_OFFSET_BYTES &&
                                            thirteenGatherReachElements *
                                                    TYPE_BYTES <=
                                                TWENTY_FIVE_BANK_BYTES &&
                                            twentyFiveMergedTaskCount <=
                                                tenMergedTaskCount &&
                                            twentyFiveMergedTaskCount >=
                                                blockNum *
                                                    MIN_TWENTY_FIVE_MERGED_TASKS_PER_CORE;

                                        // The 27-coordinate bank geometry is
                                        // specialized for FP16.  Keeping it
                                        // out of wider instantiations avoids
                                        // adding an unreachable selector and
                                        // Gather body to their hot kernels.
                                        if constexpr (TYPE_BYTES == 2U) {
                                            twentySevenMergedAxisTileCount =
                                                CeilDiv(
                                                    mergedAxisExtent,
                                                    TWENTY_SEVEN_MERGED_GROUP_COUNT);
                                            twentySevenMergedTaskCount =
                                                (outerGroupCount /
                                                 mergedAxisExtent) *
                                                twentySevenMergedAxisTileCount *
                                                WIDE_RECORD_TASK_COUNT;
                                            twentySevenMergedAlignedRecordElements =
                                                CeilDiv(
                                                    TWENTY_SEVEN_MERGED_GROUP_COUNT *
                                                        blockElements * TYPE_BYTES,
                                                    DMA_BLOCK_BYTES) *
                                                DMA_BLOCK_ELEMENTS;
                                            const uint32_t twentySevenInputBytes =
                                                wideRecordsPerTask *
                                                twentySevenMergedAlignedRecordElements *
                                                TYPE_BYTES;
                                            const uint32_t twentySevenOutputBytes =
                                                TWENTY_SEVEN_MERGED_GROUP_COUNT *
                                                wideMergedAlignedGroupElements *
                                                TYPE_BYTES;
                                            const uint32_t nineOffsetBytes =
                                                NINE_MERGED_GROUP_COUNT *
                                                wideMergedAlignedGroupElements *
                                                sizeof(uint32_t);
                                            const uint32_t
                                                twentySevenGatherReachElements =
                                                (wideRecordsPerTask - 1U) *
                                                    twentySevenMergedAlignedRecordElements +
                                                TWENTY_SEVEN_MERGED_GROUP_COUNT *
                                                    blockElements;
                                            useTwentySevenMergedRecordGroups =
                                                mergedAxisExtent >=
                                                    TWENTY_SEVEN_MERGED_GROUP_COUNT &&
                                                groupInputStride >=
                                                    TWENTY_SEVEN_MERGED_GROUP_COUNT *
                                                        blockElements &&
                                                twentySevenMergedAlignedRecordElements -
                                                        TWENTY_SEVEN_MERGED_GROUP_COUNT *
                                                            blockElements <=
                                                    255U &&
                                                twentySevenInputBytes <=
                                                    TWENTY_SEVEN_BANK_BYTES &&
                                                twentySevenOutputBytes <=
                                                    TWENTY_SEVEN_BANK_BYTES &&
                                                nineOffsetBytes <=
                                                    TWENTY_SEVEN_OFFSET_BYTES &&
                                                twentySevenGatherReachElements *
                                                        TYPE_BYTES <=
                                                    TWENTY_SEVEN_BANK_BYTES &&
                                                twentySevenMergedTaskCount <=
                                                    tenMergedTaskCount &&
                                                twentySevenMergedTaskCount >=
                                                    blockNum *
                                                        MIN_TWENTY_SEVEN_MERGED_TASKS_PER_CORE;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            chunksPerBlock = CeilDiv(blockElements, CHUNK_ELEMENTS);
            totalTaskCount = outerBlockCount * chunksPerBlock;
        }
        xGm.SetGlobalBuffer((__gm__ T*)x, totalElements);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalElements);
    }

    __aicore__ inline void Process()
    {
        if constexpr (TYPE_BYTES == 2U || TYPE_BYTES == 4U) {
            if (useRecordGroups && recordTasksPerGroup == 1U && groupSize >= 8U &&
                alignedRecordElements != blockElements &&
                groupSize * blockElements <= OFFSET_ELEMENTS) {
                const uint64_t totalBytes =
                    static_cast<uint64_t>(totalElements) * TYPE_BYTES;
                if (useMergedRecordGroups &&
                    mergedTaskCount >= blockNum * 4U &&
                    totalBytes >= static_cast<uint64_t>(blockNum) *
                                      MIN_MERGED_BYTES_PER_CORE) {
                    if (useWideMergedRecordGroups) {
                        if (useFiveMergedRecordGroups) {
                            if (useTenMergedRecordGroups) {
                                if constexpr (TYPE_BYTES == 2U) {
                                    if (useTwentySevenMergedRecordGroups) {
                                        ProcessTwentySevenMergedGatheredRecordGroups();
                                    } else if (useTwentyFiveMergedRecordGroups) {
                                        ProcessTwentyFiveMergedGatheredRecordGroups();
                                    } else {
                                        ProcessTenMergedGatheredRecordGroups();
                                    }
                                } else if (useTwentyFiveMergedRecordGroups) {
                                    ProcessTwentyFiveMergedGatheredRecordGroups();
                                } else {
                                    ProcessTenMergedGatheredRecordGroups();
                                }
                            } else {
                                ProcessFiveMergedGatheredRecordGroups();
                            }
                        } else {
                            ProcessWideMergedGatheredRecordGroups();
                        }
                        return;
                    }
                    ProcessMergedGatheredRecordGroups();
                    return;
                }
                ProcessGatheredRecordGroups();
                return;
            }
        }
        // Each bank is owned by MTE2 only after MTE3 has finished consuming
        // its previous generation.  The matching MTE2_MTE3 event makes the
        // store wait for the current load without draining either engine in
        // Scalar, so loads for task N+1 can overlap stores for task N.
        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(
                static_cast<event_t>(buffer));
        }

        uint32_t progress = 0U;
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t buffer = progress & 1U;
            const event_t event = static_cast<event_t>(buffer);
            uint32_t inputBase = 0U;
            uint32_t outputBase = 0U;
            uint32_t copyBytes = 0U;
            uint32_t inputSrcStride = 0U;
            uint16_t copyBlocks = 1U;
            uint8_t rightPadding = 0U;
            if (useRecordGroups) {
                const uint32_t outerGroup = task / recordTasksPerGroup;
                const uint32_t recordTask = task - outerGroup * recordTasksPerGroup;
                const uint32_t recordStart = recordTask * recordsPerTask;
                const uint32_t processRecords =
                    (groupSize - recordStart < recordsPerTask) ? groupSize - recordStart : recordsPerTask;
                const uint32_t firstOutputBlock = outerGroup * groupSize + recordStart;
                inputBase = InputBlockOffset(firstOutputBlock);
                outputBase = firstOutputBlock * blockElements;
                copyBytes = blockElements * TYPE_BYTES;
                inputSrcStride = (groupInputStride - blockElements) * TYPE_BYTES;
                copyBlocks = static_cast<uint16_t>(processRecords);
                rightPadding = static_cast<uint8_t>(alignedRecordElements - blockElements);
            } else {
                const uint32_t outerBlock = task / chunksPerBlock;
                const uint32_t chunk = task - outerBlock * chunksPerBlock;
                const uint32_t chunkOffset = chunk * CHUNK_ELEMENTS;
                const uint32_t processElements =
                    (blockElements - chunkOffset < CHUNK_ELEMENTS) ? blockElements - chunkOffset : CHUNK_ELEMENTS;
                inputBase = InputBlockOffset(outerBlock) + chunkOffset;
                outputBase = outerBlock * blockElements + chunkOffset;
                copyBytes = processElements * TYPE_BYTES;
            }

            AscendC::LocalTensor<T> local(
                AscendC::TPosition::VECIN, buffer * CHUNK_BYTES, CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            AscendC::DataCopyExtParams copyIn{
                copyBlocks, copyBytes, inputSrcStride, 0U, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                useRecordGroups, 0U, rightPadding, static_cast<T>(0)};
            AscendC::DataCopyPad(local, xGm[inputBase], copyIn, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);
            AscendC::DataCopyExtParams copyOut{
                copyBlocks, copyBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(zGm[outputBase], local, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
            ++progress;
        }

        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(
                static_cast<event_t>(buffer));
        }
    }

private:
    __aicore__ inline void BuildMergedCoordinateOffsets(
        const AscendC::LocalTensor<uint32_t>& offsets,
        uint32_t halfGroupElements,
        uint32_t inputAlignedRecordElements,
        uint32_t coordinateCount)
    {
        constexpr uint32_t offsetBlockElements =
            DMA_BLOCK_BYTES / sizeof(uint32_t);
        if (blockElements >= offsetBlockElements * 4U &&
            (blockElements & 1U) != 0U) {
            for (uint32_t element = 0U;
                 element < blockElements; ++element) {
                offsets.SetValue(element, element * TYPE_BYTES);
            }
            for (uint32_t record = 1U;
                 record < wideRecordsPerTask; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements -
                     recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) /
                     offsetBlockElements) * offsetBlockElements;
                for (uint32_t element = 0U;
                     element < prefixElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * inputAlignedRecordElements +
                         element) * TYPE_BYTES);
                }
                for (uint32_t element =
                         prefixElements + vectorElements;
                     element < blockElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * inputAlignedRecordElements +
                         element) * TYPE_BYTES);
                }
            }
        } else {
            uint32_t offsetSeedRecords = 1U;
            while (offsetSeedRecords < wideRecordsPerTask &&
                   (offsetSeedRecords * blockElements * sizeof(uint32_t)) %
                           DMA_BLOCK_BYTES !=
                       0U) {
                ++offsetSeedRecords;
            }
            for (uint32_t record = 0U;
                 record < offsetSeedRecords; ++record) {
                for (uint32_t element = 0U;
                     element < blockElements; ++element) {
                    offsets.SetValue(
                        record * blockElements + element,
                        (record * inputAlignedRecordElements +
                         element) * TYPE_BYTES);
                }
            }
        }
        for (uint32_t coordinate = 0U;
             coordinate < coordinateCount; ++coordinate) {
            for (uint32_t element = halfGroupElements;
                 element < wideMergedAlignedGroupElements; ++element) {
                offsets.SetValue(
                    coordinate * wideMergedAlignedGroupElements + element,
                    0U);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(0U));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(0U));
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.template ReinterpretCast<int32_t>();
        if (blockElements >= offsetBlockElements * 4U &&
            (blockElements & 1U) != 0U) {
            for (uint32_t record = 1U;
                 record < wideRecordsPerTask; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements -
                     recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) /
                     offsetBlockElements) * offsetBlockElements;
                AscendC::Adds(
                    signedOffsets[recordBase + prefixElements],
                    signedOffsets,
                    static_cast<int32_t>(
                        (record * inputAlignedRecordElements +
                         prefixElements) * TYPE_BYTES),
                    static_cast<int32_t>(vectorElements));
            }
        } else {
            uint32_t offsetSeedRecords = 1U;
            while (offsetSeedRecords < wideRecordsPerTask &&
                   (offsetSeedRecords * blockElements * sizeof(uint32_t)) %
                           DMA_BLOCK_BYTES !=
                       0U) {
                ++offsetSeedRecords;
            }
            uint32_t generatedRecords = offsetSeedRecords;
            while (generatedRecords < wideRecordsPerTask) {
                const uint32_t remainingRecords =
                    wideRecordsPerTask - generatedRecords;
                const uint32_t copiedRecords =
                    remainingRecords < generatedRecords ?
                        remainingRecords : generatedRecords;
                AscendC::Adds(
                    signedOffsets[generatedRecords * blockElements],
                    signedOffsets,
                    static_cast<int32_t>(
                        generatedRecords *
                        inputAlignedRecordElements * TYPE_BYTES),
                    static_cast<int32_t>(
                        copiedRecords * blockElements));
                AscendC::PipeBarrier<PIPE_V>();
                generatedRecords += copiedRecords;
            }
        }
        for (uint32_t coordinate = 1U;
             coordinate < coordinateCount; ++coordinate) {
            AscendC::Adds(
                signedOffsets[
                    coordinate * wideMergedAlignedGroupElements],
                signedOffsets,
                static_cast<int32_t>(
                    coordinate * blockElements * TYPE_BYTES),
                static_cast<int32_t>(halfGroupElements));
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void DecodeTwentyFiveMergedTask(
        uint32_t task, uint32_t tasksPerOuter,
        uint32_t& inputBase, uint32_t& outputBase,
        uint32_t& processCoordinates) const
    {
        const uint32_t innerCoordinate =
            task % mergedAxisOutputStride;
        uint32_t remainder = task / mergedAxisOutputStride;
        const uint32_t combinedTile =
            remainder % tasksPerOuter;
        const uint32_t outerCoordinate =
            remainder / tasksPerOuter;
        const uint32_t axisTile =
            combinedTile / WIDE_RECORD_TASK_COUNT;
        const uint32_t recordTask =
            combinedTile % WIDE_RECORD_TASK_COUNT;
        const uint32_t axisCoordinate =
            axisTile * TWENTY_FIVE_MERGED_GROUP_COUNT;
        const uint32_t outerGroup =
            (outerCoordinate * mergedAxisExtent + axisCoordinate) *
                mergedAxisOutputStride +
            innerCoordinate;
        const uint32_t recordStart =
            recordTask * wideRecordsPerTask;
        const uint32_t remainingCoordinates =
            mergedAxisExtent - axisCoordinate;
        processCoordinates =
            remainingCoordinates < TWENTY_FIVE_MERGED_GROUP_COUNT ?
                remainingCoordinates : TWENTY_FIVE_MERGED_GROUP_COUNT;
        const uint32_t firstOutputBlock =
            outerGroup * groupSize + recordStart;
        inputBase = InputBlockOffset(firstOutputBlock);
        outputBase = firstOutputBlock * blockElements;
    }

    __aicore__ inline void PrefetchTwentyFiveMergedInput(
        uint32_t bank, uint32_t inputBase,
        uint32_t processCoordinates)
    {
        const event_t event = static_cast<event_t>(bank);
        const uint32_t inputRecordElements =
            processCoordinates * blockElements;
        const uint32_t alignedInputRecordElements =
            CeilDiv(inputRecordElements * TYPE_BYTES,
                    DMA_BLOCK_BYTES) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            (twentyFiveMergedAlignedRecordElements -
             alignedInputRecordElements) * TYPE_BYTES /
            DMA_BLOCK_BYTES;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            bank * TWENTY_FIVE_BANK_BYTES,
            TWENTY_FIVE_BANK_BYTES / TYPE_BYTES);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams copyIn{
            static_cast<uint16_t>(wideRecordsPerTask),
            inputRecordElements * TYPE_BYTES,
            (groupInputStride - inputRecordElements) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U,
            static_cast<uint8_t>(
                alignedInputRecordElements - inputRecordElements),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputBase], copyIn, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    __aicore__ inline void DecodeTwentySevenMergedTask(
        uint32_t task, uint32_t tasksPerOuter,
        uint32_t& inputBase, uint32_t& outputBase,
        uint32_t& processCoordinates) const
    {
        const uint32_t innerCoordinate =
            task % mergedAxisOutputStride;
        uint32_t remainder = task / mergedAxisOutputStride;
        const uint32_t combinedTile =
            remainder % tasksPerOuter;
        const uint32_t outerCoordinate =
            remainder / tasksPerOuter;
        const uint32_t axisTile =
            combinedTile / WIDE_RECORD_TASK_COUNT;
        const uint32_t recordTask =
            combinedTile % WIDE_RECORD_TASK_COUNT;
        const uint32_t axisCoordinate =
            axisTile * TWENTY_SEVEN_MERGED_GROUP_COUNT;
        const uint32_t outerGroup =
            (outerCoordinate * mergedAxisExtent + axisCoordinate) *
                mergedAxisOutputStride +
            innerCoordinate;
        const uint32_t recordStart =
            recordTask * wideRecordsPerTask;
        const uint32_t remainingCoordinates =
            mergedAxisExtent - axisCoordinate;
        processCoordinates =
            remainingCoordinates < TWENTY_SEVEN_MERGED_GROUP_COUNT ?
                remainingCoordinates : TWENTY_SEVEN_MERGED_GROUP_COUNT;
        const uint32_t firstOutputBlock =
            outerGroup * groupSize + recordStart;
        inputBase = InputBlockOffset(firstOutputBlock);
        outputBase = firstOutputBlock * blockElements;
    }

    __aicore__ inline void PrefetchTwentySevenMergedInput(
        uint32_t bank, uint32_t inputBase,
        uint32_t processCoordinates)
    {
        const event_t event = static_cast<event_t>(bank);
        const uint32_t inputRecordElements =
            processCoordinates * blockElements;
        const uint32_t alignedInputRecordElements =
            CeilDiv(inputRecordElements * TYPE_BYTES,
                    DMA_BLOCK_BYTES) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            (twentySevenMergedAlignedRecordElements -
             alignedInputRecordElements) * TYPE_BYTES /
            DMA_BLOCK_BYTES;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            bank * TWENTY_SEVEN_BANK_BYTES,
            TWENTY_SEVEN_BANK_BYTES / TYPE_BYTES);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams copyIn{
            static_cast<uint16_t>(wideRecordsPerTask),
            inputRecordElements * TYPE_BYTES,
            (groupInputStride - inputRecordElements) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U,
            static_cast<uint8_t>(
                alignedInputRecordElements - inputRecordElements),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputBase], copyIn, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    // A 27-coordinate task retains two input and two output generations.
    // Nine-coordinate offsets are reused at source origins 0, 9, and 18,
    // so four 41-KiB data banks plus one 27-KiB table occupy 191 KiB.
    // Compared with the 25-coordinate route this reduces whole-task DMA and
    // decoding work without weakening MTE2/Gather/MTE3 overlap.
    __aicore__ inline void ProcessTwentySevenMergedGatheredRecordGroups()
    {
        const uint32_t halfGroupElements =
            wideRecordsPerTask * blockElements;
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            TWENTY_SEVEN_OFFSET_UB_OFFSET,
            TWENTY_SEVEN_OFFSET_BYTES / sizeof(uint32_t));

        for (uint32_t bank = 0U; bank < BUFFER_COUNT; ++bank) {
            const event_t event = static_cast<event_t>(bank);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        const uint32_t tasksPerOuter =
            twentySevenMergedAxisTileCount * WIDE_RECORD_TASK_COUNT;
        uint32_t task = blockIdx;
        uint32_t inputBase = 0U;
        uint32_t outputBase = 0U;
        uint32_t processCoordinates = 0U;
        if (task < twentySevenMergedTaskCount) {
            DecodeTwentySevenMergedTask(
                task, tasksPerOuter, inputBase, outputBase,
                processCoordinates);
            PrefetchTwentySevenMergedInput(
                0U, inputBase, processCoordinates);
        }

        BuildMergedCoordinateOffsets(
            offsets, halfGroupElements,
            twentySevenMergedAlignedRecordElements,
            NINE_MERGED_GROUP_COUNT);

        uint32_t progress = 0U;
        for (; task < twentySevenMergedTaskCount; task += blockNum) {
            const uint32_t bank = progress & 1U;
            const event_t event = static_cast<event_t>(bank);
            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                bank * TWENTY_SEVEN_BANK_BYTES,
                TWENTY_SEVEN_BANK_BYTES / TYPE_BYTES);
            AscendC::LocalTensor<T> output(
                AscendC::TPosition::VECOUT,
                BUFFER_COUNT * TWENTY_SEVEN_BANK_BYTES +
                    bank * TWENTY_SEVEN_BANK_BYTES,
                TWENTY_SEVEN_BANK_BYTES / TYPE_BYTES);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            const uint32_t nextTask = task + blockNum;
            uint32_t nextInputBase = 0U;
            uint32_t nextOutputBase = 0U;
            uint32_t nextProcessCoordinates = 0U;
            if (nextTask < twentySevenMergedTaskCount) {
                DecodeTwentySevenMergedTask(
                    nextTask, tasksPerOuter,
                    nextInputBase, nextOutputBase,
                    nextProcessCoordinates);
                PrefetchTwentySevenMergedInput(
                    bank ^ 1U, nextInputBase,
                    nextProcessCoordinates);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            const uint32_t firstCoordinates =
                processCoordinates < NINE_MERGED_GROUP_COUNT ?
                    processCoordinates : NINE_MERGED_GROUP_COUNT;
            AscendC::Gather(
                output, input, offsets, 0U,
                firstCoordinates * wideMergedAlignedGroupElements);
            if (processCoordinates > NINE_MERGED_GROUP_COUNT) {
                const uint32_t secondRemaining =
                    processCoordinates - NINE_MERGED_GROUP_COUNT;
                const uint32_t secondCoordinates =
                    secondRemaining < NINE_MERGED_GROUP_COUNT ?
                        secondRemaining : NINE_MERGED_GROUP_COUNT;
                AscendC::Gather(
                    output[NINE_MERGED_GROUP_COUNT *
                           wideMergedAlignedGroupElements],
                    input[NINE_MERGED_GROUP_COUNT * blockElements],
                    offsets, 0U,
                    secondCoordinates * wideMergedAlignedGroupElements);
            }
            if (processCoordinates > 2U * NINE_MERGED_GROUP_COUNT) {
                const uint32_t thirdCoordinates =
                    processCoordinates -
                    2U * NINE_MERGED_GROUP_COUNT;
                AscendC::Gather(
                    output[2U * NINE_MERGED_GROUP_COUNT *
                           wideMergedAlignedGroupElements],
                    input[2U * NINE_MERGED_GROUP_COUNT * blockElements],
                    offsets, 0U,
                    thirdCoordinates * wideMergedAlignedGroupElements);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

            const uint32_t outputCoordinateStrideElements =
                mergedAxisOutputStride * groupSize * blockElements;
            AscendC::DataCopyExtParams copyOut{
                static_cast<uint16_t>(processCoordinates),
                halfGroupElements * TYPE_BYTES, 0U,
                (outputCoordinateStrideElements - halfGroupElements) *
                    TYPE_BYTES,
                0U};
            AscendC::DataCopyPad(zGm[outputBase], output, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);

            inputBase = nextInputBase;
            outputBase = nextOutputBase;
            processCoordinates = nextProcessCoordinates;
            ++progress;
        }

        for (uint32_t bank = 0U; bank < BUFFER_COUNT; ++bank) {
            const event_t event = static_cast<event_t>(bank);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    // A 25-coordinate task uses four 38-KiB data banks and one 40-KiB
    // offset bank.  The offset table describes at most thirteen coordinates
    // and is reused for the remaining twelve by shifting the Gather source.
    // This keeps the total UB footprint at 192 KiB while reducing task and
    // DMA descriptor count relative to the retained 21-coordinate path.
    __aicore__ inline void ProcessTwentyFiveMergedGatheredRecordGroups()
    {
        const uint32_t halfGroupElements =
            wideRecordsPerTask * blockElements;
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            TWENTY_FIVE_OFFSET_UB_OFFSET,
            TWENTY_FIVE_OFFSET_BYTES / sizeof(uint32_t));

        for (uint32_t bank = 0U; bank < BUFFER_COUNT; ++bank) {
            const event_t event = static_cast<event_t>(bank);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        const uint32_t tasksPerOuter =
            twentyFiveMergedAxisTileCount * WIDE_RECORD_TASK_COUNT;
        uint32_t task = blockIdx;
        uint32_t inputBase = 0U;
        uint32_t outputBase = 0U;
        uint32_t processCoordinates = 0U;
        if (task < twentyFiveMergedTaskCount) {
            DecodeTwentyFiveMergedTask(
                task, tasksPerOuter, inputBase, outputBase,
                processCoordinates);
            PrefetchTwentyFiveMergedInput(
                0U, inputBase, processCoordinates);
        }

        BuildMergedCoordinateOffsets(
            offsets, halfGroupElements,
            twentyFiveMergedAlignedRecordElements,
            THIRTEEN_MERGED_GROUP_COUNT);

        uint32_t progress = 0U;
        for (; task < twentyFiveMergedTaskCount; task += blockNum) {
            const uint32_t bank = progress & 1U;
            const event_t event = static_cast<event_t>(bank);
            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                bank * TWENTY_FIVE_BANK_BYTES,
                TWENTY_FIVE_BANK_BYTES / TYPE_BYTES);
            AscendC::LocalTensor<T> output(
                AscendC::TPosition::VECOUT,
                BUFFER_COUNT * TWENTY_FIVE_BANK_BYTES +
                    bank * TWENTY_FIVE_BANK_BYTES,
                TWENTY_FIVE_BANK_BYTES / TYPE_BYTES);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            const uint32_t nextTask = task + blockNum;
            uint32_t nextInputBase = 0U;
            uint32_t nextOutputBase = 0U;
            uint32_t nextProcessCoordinates = 0U;
            if (nextTask < twentyFiveMergedTaskCount) {
                DecodeTwentyFiveMergedTask(
                    nextTask, tasksPerOuter,
                    nextInputBase, nextOutputBase,
                    nextProcessCoordinates);
                PrefetchTwentyFiveMergedInput(
                    bank ^ 1U, nextInputBase,
                    nextProcessCoordinates);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            const uint32_t firstCoordinates =
                processCoordinates < THIRTEEN_MERGED_GROUP_COUNT ?
                    processCoordinates : THIRTEEN_MERGED_GROUP_COUNT;
            AscendC::Gather(
                output, input, offsets, 0U,
                firstCoordinates * wideMergedAlignedGroupElements);
            if (processCoordinates > THIRTEEN_MERGED_GROUP_COUNT) {
                AscendC::Gather(
                    output[THIRTEEN_MERGED_GROUP_COUNT *
                           wideMergedAlignedGroupElements],
                    input[THIRTEEN_MERGED_GROUP_COUNT * blockElements],
                    offsets, 0U,
                    (processCoordinates - THIRTEEN_MERGED_GROUP_COUNT) *
                        wideMergedAlignedGroupElements);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

            const uint32_t outputCoordinateStrideElements =
                mergedAxisOutputStride * groupSize * blockElements;
            AscendC::DataCopyExtParams copyOut{
                static_cast<uint16_t>(processCoordinates),
                halfGroupElements * TYPE_BYTES, 0U,
                (outputCoordinateStrideElements - halfGroupElements) *
                    TYPE_BYTES,
                0U};
            AscendC::DataCopyPad(zGm[outputBase], output, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);

            inputBase = nextInputBase;
            outputBase = nextOutputBase;
            processCoordinates = nextProcessCoordinates;
            ++progress;
        }

        for (uint32_t bank = 0U; bank < BUFFER_COUNT; ++bank) {
            const event_t event = static_cast<event_t>(bank);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline void DecodeTenMergedTask(
        uint32_t task, uint32_t tenTasksPerOuter,
        uint32_t& inputBase, uint32_t& outputBase,
        uint32_t& processCoordinates) const
    {
        const uint32_t innerCoordinate =
            task % mergedAxisOutputStride;
        uint32_t remainder = task / mergedAxisOutputStride;
        const uint32_t combinedTile =
            remainder % tenTasksPerOuter;
        const uint32_t outerCoordinate =
            remainder / tenTasksPerOuter;
        const uint32_t tenAxisTile =
            combinedTile / WIDE_RECORD_TASK_COUNT;
        const uint32_t recordTask =
            combinedTile % WIDE_RECORD_TASK_COUNT;
        const uint32_t axisCoordinate =
            tenAxisTile * TEN_MERGED_GROUP_COUNT;
        const uint32_t outerGroup =
            (outerCoordinate * mergedAxisExtent + axisCoordinate) *
                mergedAxisOutputStride +
            innerCoordinate;
        const uint32_t recordStart =
            recordTask * wideRecordsPerTask;
        const uint32_t remainingCoordinates =
            mergedAxisExtent - axisCoordinate;
        processCoordinates =
            remainingCoordinates < TEN_MERGED_GROUP_COUNT ?
                remainingCoordinates : TEN_MERGED_GROUP_COUNT;
        const uint32_t firstOutputBlock =
            outerGroup * groupSize + recordStart;
        inputBase = InputBlockOffset(firstOutputBlock);
        outputBase = firstOutputBlock * blockElements;
    }

    __aicore__ inline void PrefetchTenMergedInput(
        uint32_t bank, uint32_t inputBase,
        uint32_t processCoordinates)
    {
        const event_t event = static_cast<event_t>(bank);
        const uint32_t inputRecordElements =
            processCoordinates * blockElements;
        const uint32_t alignedInputRecordElements =
            CeilDiv(inputRecordElements * TYPE_BYTES,
                    DMA_BLOCK_BYTES) * DMA_BLOCK_ELEMENTS;
        const uint32_t inputDstGapBlocks =
            (tenMergedAlignedRecordElements -
             alignedInputRecordElements) * TYPE_BYTES /
            DMA_BLOCK_BYTES;
        AscendC::LocalTensor<T> input(
            AscendC::TPosition::VECIN,
            bank * CHUNK_BYTES, CHUNK_ELEMENTS);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::DataCopyExtParams copyIn{
            static_cast<uint16_t>(wideRecordsPerTask),
            inputRecordElements * TYPE_BYTES,
            (groupInputStride - inputRecordElements) * TYPE_BYTES,
            inputDstGapBlocks, 0U};
        AscendC::DataCopyPadExtParams<T> pad{
            true, 0U,
            static_cast<uint8_t>(
                alignedInputRecordElements - inputRecordElements),
            static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm[inputBase], copyIn, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
    }

    // Fill the two 32-KiB data banks with the widest complete record group
    // whose offset table fits in the remaining 64 KiB.  One wide input
    // descriptor, one Gather and one long store replace several narrow tasks.
    __aicore__ inline void ProcessTenMergedGatheredRecordGroups()
    {
        const uint32_t halfGroupElements =
            wideRecordsPerTask * blockElements;
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            4U * CHUNK_BYTES, 2U * CHUNK_BYTES / sizeof(uint32_t));

        for (uint32_t bank = 0U; bank < BUFFER_COUNT; ++bank) {
            const event_t event = static_cast<event_t>(bank);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        const uint32_t tenTasksPerOuter =
            tenMergedAxisTileCount * WIDE_RECORD_TASK_COUNT;
        uint32_t task = blockIdx;
        uint32_t inputBase = 0U;
        uint32_t outputBase = 0U;
        uint32_t processCoordinates = 0U;
        if (task < tenMergedTaskCount) {
            DecodeTenMergedTask(
                task, tenTasksPerOuter, inputBase, outputBase,
                processCoordinates);
            PrefetchTenMergedInput(
                0U, inputBase, processCoordinates);
        }

        // Offset synthesis is independent of the first input generation.
        // Submit that MTE2 work before the one-time Scalar/Vector table build
        // so kernel fill covers the first wide record load without changing
        // the steady-state two-bank ownership chain.
        BuildMergedCoordinateOffsets(
            offsets, halfGroupElements,
            tenMergedAlignedRecordElements,
            TEN_MERGED_GROUP_COUNT);

        uint32_t progress = 0U;
        for (; task < tenMergedTaskCount; task += blockNum) {
            const uint32_t bank = progress & 1U;
            const event_t event = static_cast<event_t>(bank);
            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                bank * CHUNK_BYTES, CHUNK_ELEMENTS);
            AscendC::LocalTensor<T> output(
                AscendC::TPosition::VECOUT,
                BUFFER_COUNT * CHUNK_BYTES +
                    bank * CHUNK_BYTES,
                CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            const uint32_t nextTask = task + blockNum;
            uint32_t nextInputBase = 0U;
            uint32_t nextOutputBase = 0U;
            uint32_t nextProcessCoordinates = 0U;
            if (nextTask < tenMergedTaskCount) {
                DecodeTenMergedTask(
                    nextTask, tenTasksPerOuter,
                    nextInputBase, nextOutputBase,
                    nextProcessCoordinates);
                PrefetchTenMergedInput(
                    bank ^ 1U, nextInputBase,
                    nextProcessCoordinates);
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            AscendC::Gather(
                output, input, offsets, 0U,
                processCoordinates * wideMergedAlignedGroupElements);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

            const uint32_t outputCoordinateStrideElements =
                mergedAxisOutputStride * groupSize * blockElements;
            AscendC::DataCopyExtParams copyOut{
                static_cast<uint16_t>(processCoordinates),
                halfGroupElements * TYPE_BYTES, 0U,
                (outputCoordinateStrideElements - halfGroupElements) *
                    TYPE_BYTES,
                0U};
            AscendC::DataCopyPad(zGm[outputBase], output, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);

            inputBase = nextInputBase;
            outputBase = nextOutputBase;
            processCoordinates = nextProcessCoordinates;
            ++progress;
        }

        for (uint32_t bank = 0U; bank < BUFFER_COUNT; ++bank) {
            const event_t event = static_cast<event_t>(bank);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline void ProcessFiveMergedGatheredRecordGroups()
    {
        const uint32_t halfGroupElements =
            wideRecordsPerTask * blockElements;
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            4U * CHUNK_BYTES, CHUNK_BYTES / sizeof(uint32_t));
        constexpr uint32_t offsetBlockElements =
            DMA_BLOCK_BYTES / sizeof(uint32_t);

        if (blockElements >= offsetBlockElements * 4U &&
            (blockElements & 1U) != 0U) {
            for (uint32_t element = 0U; element < blockElements; ++element) {
                offsets.SetValue(element, element * TYPE_BYTES);
            }
            for (uint32_t record = 1U;
                 record < wideRecordsPerTask; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements -
                     recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) /
                     offsetBlockElements) *
                    offsetBlockElements;
                for (uint32_t element = 0U;
                     element < prefixElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * fiveMergedAlignedRecordElements +
                         element) * TYPE_BYTES);
                }
                for (uint32_t element =
                         prefixElements + vectorElements;
                     element < blockElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * fiveMergedAlignedRecordElements +
                         element) * TYPE_BYTES);
                }
            }
        } else {
            uint32_t offsetSeedRecords = 1U;
            while (offsetSeedRecords < wideRecordsPerTask &&
                   (offsetSeedRecords * blockElements * sizeof(uint32_t)) %
                           DMA_BLOCK_BYTES !=
                       0U) {
                ++offsetSeedRecords;
            }
            for (uint32_t record = 0U;
                 record < offsetSeedRecords; ++record) {
                for (uint32_t element = 0U;
                     element < blockElements; ++element) {
                    offsets.SetValue(
                        record * blockElements + element,
                        (record * fiveMergedAlignedRecordElements +
                         element) * TYPE_BYTES);
                }
            }
        }
        for (uint32_t coordinate = 0U;
             coordinate < FIVE_MERGED_GROUP_COUNT; ++coordinate) {
            for (uint32_t element = halfGroupElements;
                 element < wideMergedAlignedGroupElements; ++element) {
                offsets.SetValue(
                    coordinate * wideMergedAlignedGroupElements + element,
                    0U);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(0U));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(0U));
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.template ReinterpretCast<int32_t>();
        if (blockElements >= offsetBlockElements * 4U &&
            (blockElements & 1U) != 0U) {
            for (uint32_t record = 1U;
                 record < wideRecordsPerTask; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements -
                     recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) /
                     offsetBlockElements) *
                    offsetBlockElements;
                AscendC::Adds(
                    signedOffsets[recordBase + prefixElements],
                    signedOffsets,
                    static_cast<int32_t>(
                        (record * fiveMergedAlignedRecordElements +
                         prefixElements) * TYPE_BYTES),
                    static_cast<int32_t>(vectorElements));
            }
        } else {
            uint32_t offsetSeedRecords = 1U;
            while (offsetSeedRecords < wideRecordsPerTask &&
                   (offsetSeedRecords * blockElements * sizeof(uint32_t)) %
                           DMA_BLOCK_BYTES !=
                       0U) {
                ++offsetSeedRecords;
            }
            uint32_t generatedRecords = offsetSeedRecords;
            while (generatedRecords < wideRecordsPerTask) {
                const uint32_t remainingRecords =
                    wideRecordsPerTask - generatedRecords;
                const uint32_t copiedRecords =
                    remainingRecords < generatedRecords ?
                        remainingRecords : generatedRecords;
                AscendC::Adds(
                    signedOffsets[generatedRecords * blockElements],
                    signedOffsets,
                    static_cast<int32_t>(
                        generatedRecords *
                        fiveMergedAlignedRecordElements * TYPE_BYTES),
                    static_cast<int32_t>(
                        copiedRecords * blockElements));
                AscendC::PipeBarrier<PIPE_V>();
                generatedRecords += copiedRecords;
            }
        }
        for (uint32_t coordinate = 1U;
             coordinate < FIVE_MERGED_GROUP_COUNT; ++coordinate) {
            AscendC::Adds(
                signedOffsets[
                    coordinate * wideMergedAlignedGroupElements],
                signedOffsets,
                static_cast<int32_t>(
                    coordinate * blockElements * TYPE_BYTES),
                static_cast<int32_t>(halfGroupElements));
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        const uint32_t fiveTasksPerOuter =
            fiveMergedAxisTileCount * WIDE_RECORD_TASK_COUNT;
        uint32_t progress = 0U;
        for (uint32_t task = blockIdx;
             task < fiveMergedTaskCount; task += blockNum) {
            const uint32_t buffer = progress & 1U;
            const event_t event = static_cast<event_t>(buffer);
            const uint32_t innerCoordinate =
                task % mergedAxisOutputStride;
            uint32_t remainder = task / mergedAxisOutputStride;
            const uint32_t combinedTile =
                remainder % fiveTasksPerOuter;
            const uint32_t outerCoordinate =
                remainder / fiveTasksPerOuter;
            const uint32_t fiveAxisTile =
                combinedTile / WIDE_RECORD_TASK_COUNT;
            const uint32_t recordTask =
                combinedTile % WIDE_RECORD_TASK_COUNT;
            const uint32_t axisCoordinate =
                fiveAxisTile * FIVE_MERGED_GROUP_COUNT;
            const uint32_t outerGroup =
                (outerCoordinate * mergedAxisExtent + axisCoordinate) *
                    mergedAxisOutputStride +
                innerCoordinate;
            const uint32_t recordStart =
                recordTask * wideRecordsPerTask;
            const uint32_t remainingCoordinates =
                mergedAxisExtent - axisCoordinate;
            const uint32_t processCoordinates =
                remainingCoordinates < FIVE_MERGED_GROUP_COUNT ?
                    remainingCoordinates : FIVE_MERGED_GROUP_COUNT;
            const uint32_t firstOutputBlock =
                outerGroup * groupSize + recordStart;
            const uint32_t inputBase =
                InputBlockOffset(firstOutputBlock);
            const uint32_t outputBase =
                firstOutputBlock * blockElements;
            const uint32_t inputRecordElements =
                processCoordinates * blockElements;
            const uint32_t alignedInputRecordElements =
                CeilDiv(inputRecordElements * TYPE_BYTES,
                        DMA_BLOCK_BYTES) *
                DMA_BLOCK_ELEMENTS;
            const uint32_t inputDstGapBlocks =
                (fiveMergedAlignedRecordElements -
                 alignedInputRecordElements) *
                TYPE_BYTES / DMA_BLOCK_BYTES;

            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                buffer * CHUNK_BYTES, CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::DataCopyExtParams copyIn{
                static_cast<uint16_t>(wideRecordsPerTask),
                inputRecordElements * TYPE_BYTES,
                (groupInputStride - inputRecordElements) * TYPE_BYTES,
                inputDstGapBlocks, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0U,
                static_cast<uint8_t>(
                    alignedInputRecordElements - inputRecordElements),
                static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm[inputBase], copyIn, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            AscendC::LocalTensor<T> output(
                AscendC::TPosition::VECOUT,
                BUFFER_COUNT * CHUNK_BYTES + buffer * CHUNK_BYTES,
                CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            AscendC::Gather(
                output, input, offsets, 0U,
                processCoordinates *
                    wideMergedAlignedGroupElements);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

            const uint32_t outputCoordinateStrideElements =
                mergedAxisOutputStride * groupSize * blockElements;
            AscendC::DataCopyExtParams copyOut{
                static_cast<uint16_t>(processCoordinates),
                halfGroupElements * TYPE_BYTES, 0U,
                (outputCoordinateStrideElements - halfGroupElements) *
                    TYPE_BYTES,
                0U};
            AscendC::DataCopyPad(zGm[outputBase], output, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            ++progress;
        }

        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline void ProcessWideMergedGatheredRecordGroups()
    {
        const uint32_t wideGroupElements =
            wideRecordsPerTask * blockElements;
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            4U * CHUNK_BYTES, CHUNK_BYTES / sizeof(uint32_t));
        constexpr uint32_t offsetBlockElements =
            DMA_BLOCK_BYTES / sizeof(uint32_t);

        // Build offsets for one half-record group.  Three vector Adds derive
        // the other adjacent input coordinates without changing the number
        // of gathered elements relative to two full-record pair tasks.
        if (blockElements >= offsetBlockElements * 4U &&
            (blockElements & 1U) != 0U) {
            for (uint32_t element = 0U; element < blockElements; ++element) {
                offsets.SetValue(element, element * TYPE_BYTES);
            }
            for (uint32_t record = 1U;
                 record < wideRecordsPerTask; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements -
                     recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) /
                     offsetBlockElements) *
                    offsetBlockElements;
                for (uint32_t element = 0U;
                     element < prefixElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * wideMergedAlignedRecordElements +
                         element) * TYPE_BYTES);
                }
                for (uint32_t element =
                         prefixElements + vectorElements;
                     element < blockElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * wideMergedAlignedRecordElements +
                         element) * TYPE_BYTES);
                }
            }
        } else {
            uint32_t offsetSeedRecords = 1U;
            while (offsetSeedRecords < wideRecordsPerTask &&
                   (offsetSeedRecords * blockElements * sizeof(uint32_t)) %
                           DMA_BLOCK_BYTES !=
                       0U) {
                ++offsetSeedRecords;
            }
            for (uint32_t record = 0U;
                 record < offsetSeedRecords; ++record) {
                for (uint32_t element = 0U;
                     element < blockElements; ++element) {
                    offsets.SetValue(
                        record * blockElements + element,
                        (record * wideMergedAlignedRecordElements +
                         element) * TYPE_BYTES);
                }
            }
        }
        for (uint32_t coordinate = 0U;
             coordinate < WIDE_MERGED_GROUP_COUNT; ++coordinate) {
            for (uint32_t element = wideGroupElements;
                 element < wideMergedAlignedGroupElements; ++element) {
                offsets.SetValue(
                    coordinate * wideMergedAlignedGroupElements + element,
                    0U);
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(0U));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(0U));
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.template ReinterpretCast<int32_t>();
        if (blockElements >= offsetBlockElements * 4U &&
            (blockElements & 1U) != 0U) {
            for (uint32_t record = 1U;
                 record < wideRecordsPerTask; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements -
                     recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) /
                     offsetBlockElements) *
                    offsetBlockElements;
                AscendC::Adds(
                    signedOffsets[recordBase + prefixElements],
                    signedOffsets,
                    static_cast<int32_t>(
                        (record * wideMergedAlignedRecordElements +
                         prefixElements) * TYPE_BYTES),
                    static_cast<int32_t>(vectorElements));
            }
        } else {
            uint32_t offsetSeedRecords = 1U;
            while (offsetSeedRecords < wideRecordsPerTask &&
                   (offsetSeedRecords * blockElements * sizeof(uint32_t)) %
                           DMA_BLOCK_BYTES !=
                       0U) {
                ++offsetSeedRecords;
            }
            uint32_t generatedRecords = offsetSeedRecords;
            while (generatedRecords < wideRecordsPerTask) {
                const uint32_t remainingRecords =
                    wideRecordsPerTask - generatedRecords;
                const uint32_t copiedRecords =
                    remainingRecords < generatedRecords ?
                        remainingRecords : generatedRecords;
                AscendC::Adds(
                    signedOffsets[generatedRecords * blockElements],
                    signedOffsets,
                    static_cast<int32_t>(
                        generatedRecords *
                        wideMergedAlignedRecordElements * TYPE_BYTES),
                    static_cast<int32_t>(
                        copiedRecords * blockElements));
                AscendC::PipeBarrier<PIPE_V>();
                generatedRecords += copiedRecords;
            }
        }
        for (uint32_t coordinate = 1U;
             coordinate < WIDE_MERGED_GROUP_COUNT; ++coordinate) {
            AscendC::Adds(
                signedOffsets[
                    coordinate * wideMergedAlignedGroupElements],
                signedOffsets,
                static_cast<int32_t>(
                    coordinate * blockElements * TYPE_BYTES),
                static_cast<int32_t>(wideGroupElements));
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        uint32_t progress = 0U;
        for (uint32_t task = blockIdx;
             task < wideMergedTaskCount; task += blockNum) {
            const uint32_t buffer = progress & 1U;
            const event_t event = static_cast<event_t>(buffer);
            const uint32_t innerCoordinate =
                task % mergedAxisOutputStride;
            uint32_t remainder = task / mergedAxisOutputStride;
            const uint32_t pairTile =
                remainder % mergedAxisTileCount;
            const uint32_t outerCoordinate =
                remainder / mergedAxisTileCount;
            const uint32_t wideAxisTile =
                pairTile / WIDE_RECORD_TASK_COUNT;
            const uint32_t recordTask =
                pairTile % WIDE_RECORD_TASK_COUNT;
            const uint32_t axisCoordinate =
                wideAxisTile * WIDE_MERGED_GROUP_COUNT;
            const uint32_t outerGroup =
                (outerCoordinate * mergedAxisExtent + axisCoordinate) *
                    mergedAxisOutputStride +
                innerCoordinate;
            const uint32_t recordStart =
                recordTask * wideRecordsPerTask;
            const uint32_t remainingCoordinates =
                mergedAxisExtent - axisCoordinate;
            const uint32_t processCoordinates =
                remainingCoordinates < WIDE_MERGED_GROUP_COUNT ?
                    remainingCoordinates : WIDE_MERGED_GROUP_COUNT;
            const uint32_t firstOutputBlock =
                outerGroup * groupSize + recordStart;
            const uint32_t inputBase =
                InputBlockOffset(firstOutputBlock);
            const uint32_t outputBase =
                firstOutputBlock * blockElements;
            const uint32_t inputRecordElements =
                processCoordinates * blockElements;
            const uint32_t alignedInputRecordElements =
                CeilDiv(inputRecordElements * TYPE_BYTES,
                        DMA_BLOCK_BYTES) *
                DMA_BLOCK_ELEMENTS;
            const uint32_t inputDstGapBlocks =
                (wideMergedAlignedRecordElements -
                 alignedInputRecordElements) *
                TYPE_BYTES / DMA_BLOCK_BYTES;

            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                buffer * CHUNK_BYTES, CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::DataCopyExtParams copyIn{
                static_cast<uint16_t>(wideRecordsPerTask),
                inputRecordElements * TYPE_BYTES,
                (groupInputStride - inputRecordElements) * TYPE_BYTES,
                inputDstGapBlocks, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0U,
                static_cast<uint8_t>(
                    alignedInputRecordElements - inputRecordElements),
                static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm[inputBase], copyIn, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            AscendC::LocalTensor<T> output(
                AscendC::TPosition::VECOUT,
                BUFFER_COUNT * CHUNK_BYTES + buffer * CHUNK_BYTES,
                CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            AscendC::Gather(
                output, input, offsets, 0U,
                processCoordinates *
                    wideMergedAlignedGroupElements);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

            const uint32_t outputCoordinateStrideElements =
                mergedAxisOutputStride * groupSize * blockElements;
            AscendC::DataCopyExtParams copyOut{
                static_cast<uint16_t>(processCoordinates),
                wideGroupElements * TYPE_BYTES, 0U,
                (outputCoordinateStrideElements - wideGroupElements) *
                    TYPE_BYTES,
                0U};
            AscendC::DataCopyPad(zGm[outputBase], output, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            ++progress;
        }

        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline void ProcessMergedGatheredRecordGroups()
    {
        const uint32_t groupElements = groupSize * blockElements;
        const uint32_t mergedOffsetElements =
            MERGED_GROUP_COUNT * mergedAlignedGroupElements;
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            4U * CHUNK_BYTES, CHUNK_BYTES / sizeof(uint32_t));
        constexpr uint32_t offsetBlockElements =
            DMA_BLOCK_BYTES / sizeof(uint32_t);

        // Build the first coordinate with the same range-dependent algorithm
        // as the retained one-coordinate path.  The second coordinate starts
        // at an aligned output-group boundary and differs only by its compact
        // input displacement, so one vector Adds derives the entire table.
        if (blockElements >= offsetBlockElements * 4U &&
            (blockElements & 1U) != 0U) {
            for (uint32_t element = 0U; element < blockElements; ++element) {
                offsets.SetValue(element, element * TYPE_BYTES);
            }
            for (uint32_t record = 1U; record < groupSize; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements - recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) / offsetBlockElements) *
                    offsetBlockElements;
                for (uint32_t element = 0U;
                     element < prefixElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * mergedAlignedRecordElements + element) *
                            TYPE_BYTES);
                }
                for (uint32_t element = prefixElements + vectorElements;
                     element < blockElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * mergedAlignedRecordElements + element) *
                            TYPE_BYTES);
                }
            }
        } else {
            uint32_t offsetSeedRecords = 1U;
            while (offsetSeedRecords < groupSize &&
                   (offsetSeedRecords * blockElements * sizeof(uint32_t)) %
                           DMA_BLOCK_BYTES !=
                       0U) {
                ++offsetSeedRecords;
            }
            for (uint32_t record = 0U; record < offsetSeedRecords; ++record) {
                for (uint32_t element = 0U; element < blockElements;
                     ++element) {
                    offsets.SetValue(
                        record * blockElements + element,
                        (record * mergedAlignedRecordElements + element) *
                            TYPE_BYTES);
                }
            }
        }
        for (uint32_t element = groupElements;
             element < mergedAlignedGroupElements; ++element) {
            offsets.SetValue(element, 0U);
        }
        for (uint32_t element = mergedAlignedGroupElements + groupElements;
             element < mergedOffsetElements; ++element) {
            offsets.SetValue(element, 0U);
        }

        AscendC::SetFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(0U));
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(
            static_cast<event_t>(0U));
        AscendC::LocalTensor<int32_t> signedOffsets =
            offsets.template ReinterpretCast<int32_t>();
        if (blockElements >= offsetBlockElements * 4U &&
            (blockElements & 1U) != 0U) {
            for (uint32_t record = 1U; record < groupSize; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements - recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) / offsetBlockElements) *
                    offsetBlockElements;
                AscendC::Adds(
                    signedOffsets[recordBase + prefixElements],
                    signedOffsets,
                    static_cast<int32_t>(
                        (record * mergedAlignedRecordElements +
                         prefixElements) * TYPE_BYTES),
                    static_cast<int32_t>(vectorElements));
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            uint32_t offsetSeedRecords = 1U;
            while (offsetSeedRecords < groupSize &&
                   (offsetSeedRecords * blockElements * sizeof(uint32_t)) %
                           DMA_BLOCK_BYTES !=
                       0U) {
                ++offsetSeedRecords;
            }
            uint32_t generatedRecords = offsetSeedRecords;
            while (generatedRecords < groupSize) {
                const uint32_t remainingRecords =
                    groupSize - generatedRecords;
                const uint32_t copiedRecords =
                    remainingRecords < generatedRecords ?
                        remainingRecords : generatedRecords;
                AscendC::Adds(
                    signedOffsets[generatedRecords * blockElements],
                    signedOffsets,
                    static_cast<int32_t>(generatedRecords *
                                         mergedAlignedRecordElements *
                                         TYPE_BYTES),
                    static_cast<int32_t>(copiedRecords * blockElements));
                AscendC::PipeBarrier<PIPE_V>();
                generatedRecords += copiedRecords;
            }
        }
        AscendC::Adds(
            signedOffsets[mergedAlignedGroupElements], signedOffsets,
            static_cast<int32_t>(blockElements * TYPE_BYTES),
            static_cast<int32_t>(groupElements));
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        uint32_t progress = 0U;
        for (uint32_t task = blockIdx; task < mergedTaskCount;
             task += blockNum) {
            const uint32_t buffer = progress & 1U;
            const event_t event = static_cast<event_t>(buffer);
            const uint32_t innerCoordinate =
                task % mergedAxisOutputStride;
            uint32_t remainder = task / mergedAxisOutputStride;
            const uint32_t axisTile = remainder % mergedAxisTileCount;
            const uint32_t outerCoordinate =
                remainder / mergedAxisTileCount;
            const uint32_t axisCoordinate =
                axisTile * MERGED_GROUP_COUNT;
            const uint32_t outerGroup =
                (outerCoordinate * mergedAxisExtent + axisCoordinate) *
                    mergedAxisOutputStride +
                innerCoordinate;
            const uint32_t remainingCoordinates =
                mergedAxisExtent - axisCoordinate;
            const uint32_t processCoordinates =
                remainingCoordinates < MERGED_GROUP_COUNT ?
                    remainingCoordinates : MERGED_GROUP_COUNT;
            const uint32_t firstOutputBlock = outerGroup * groupSize;
            const uint32_t inputBase = InputBlockOffset(firstOutputBlock);
            const uint32_t outputBase = firstOutputBlock * blockElements;
            const uint32_t inputRecordElements =
                processCoordinates * blockElements;
            const uint32_t alignedInputRecordElements =
                CeilDiv(inputRecordElements * TYPE_BYTES, DMA_BLOCK_BYTES) *
                DMA_BLOCK_ELEMENTS;
            const uint32_t inputDstGapBlocks =
                (mergedAlignedRecordElements - alignedInputRecordElements) *
                TYPE_BYTES / DMA_BLOCK_BYTES;

            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN,
                buffer * CHUNK_BYTES, CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::DataCopyExtParams copyIn{
                static_cast<uint16_t>(groupSize),
                inputRecordElements * TYPE_BYTES,
                (groupInputStride - inputRecordElements) * TYPE_BYTES,
                inputDstGapBlocks, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0U,
                static_cast<uint8_t>(alignedInputRecordElements -
                                     inputRecordElements),
                static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm[inputBase], copyIn, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            AscendC::LocalTensor<T> output(
                AscendC::TPosition::VECOUT,
                BUFFER_COUNT * CHUNK_BYTES + buffer * CHUNK_BYTES,
                CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            AscendC::Gather(
                output, input, offsets, 0U,
                processCoordinates * mergedAlignedGroupElements);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

            AscendC::DataCopyExtParams copyOut{
                static_cast<uint16_t>(processCoordinates),
                groupElements * TYPE_BYTES, 0U,
                (mergedAxisOutputStride - 1U) * groupElements * TYPE_BYTES,
                0U};
            AscendC::DataCopyPad(zGm[outputBase], output, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            ++progress;
        }

        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline void ProcessGatheredRecordGroups()
    {
        const uint32_t compactElements = groupSize * blockElements;
        AscendC::LocalTensor<uint32_t> offsets(
            AscendC::TPosition::VECCALC,
            4U * CHUNK_BYTES, CHUNK_BYTES / sizeof(uint32_t));
        constexpr uint32_t offsetBlockElements =
            DMA_BLOCK_BYTES / sizeof(uint32_t);
        if (blockElements >= offsetBlockElements * 4U &&
            (blockElements & 1U) != 0U) {
            // Every record has one aligned interior even when its logical row
            // begins unaligned.  Seed the first record, write only the short
            // unaligned boundaries of later records in Scalar, then copy each
            // aligned interior from the first record with one vector Adds.
            for (uint32_t element = 0U; element < blockElements; ++element) {
                offsets.SetValue(element, element * TYPE_BYTES);
            }
            for (uint32_t record = 1U; record < groupSize; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements - recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) / offsetBlockElements) *
                    offsetBlockElements;
                for (uint32_t element = 0U;
                     element < prefixElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * alignedRecordElements + element) * TYPE_BYTES);
                }
                for (uint32_t element = prefixElements + vectorElements;
                     element < blockElements; ++element) {
                    offsets.SetValue(
                        recordBase + element,
                        (record * alignedRecordElements + element) * TYPE_BYTES);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_V>(
                static_cast<event_t>(0U));
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(
                static_cast<event_t>(0U));
            AscendC::LocalTensor<int32_t> signedOffsets =
                offsets.template ReinterpretCast<int32_t>();
            for (uint32_t record = 1U; record < groupSize; ++record) {
                const uint32_t recordBase = record * blockElements;
                const uint32_t prefixElements =
                    (offsetBlockElements - recordBase % offsetBlockElements) %
                    offsetBlockElements;
                const uint32_t vectorElements =
                    ((blockElements - prefixElements) / offsetBlockElements) *
                    offsetBlockElements;
                AscendC::Adds(
                    signedOffsets[recordBase + prefixElements],
                    signedOffsets,
                    static_cast<int32_t>(
                        (record * alignedRecordElements + prefixElements) *
                        TYPE_BYTES),
                    static_cast<int32_t>(vectorElements));
            }
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            // Short records amortize best by seeding the shortest whole-row
            // period whose byte size is vector aligned.
            uint32_t offsetSeedRecords = 1U;
            while (offsetSeedRecords < groupSize &&
                   (offsetSeedRecords * blockElements * sizeof(uint32_t)) %
                           DMA_BLOCK_BYTES !=
                       0U) {
                ++offsetSeedRecords;
            }
            for (uint32_t record = 0U; record < offsetSeedRecords; ++record) {
                for (uint32_t element = 0U; element < blockElements; ++element) {
                    offsets.SetValue(
                        record * blockElements + element,
                        (record * alignedRecordElements + element) * TYPE_BYTES);
                }
            }
            if (offsetSeedRecords < groupSize) {
                AscendC::SetFlag<AscendC::HardEvent::S_V>(
                    static_cast<event_t>(0U));
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(
                    static_cast<event_t>(0U));
                AscendC::LocalTensor<int32_t> signedOffsets =
                    offsets.template ReinterpretCast<int32_t>();
                uint32_t generatedRecords = offsetSeedRecords;
                while (generatedRecords < groupSize) {
                    const uint32_t remainingRecords =
                        groupSize - generatedRecords;
                    const uint32_t copiedRecords =
                        remainingRecords < generatedRecords ?
                            remainingRecords : generatedRecords;
                    AscendC::Adds(
                        signedOffsets[generatedRecords * blockElements],
                        signedOffsets,
                        static_cast<int32_t>(generatedRecords *
                                             alignedRecordElements * TYPE_BYTES),
                        static_cast<int32_t>(copiedRecords * blockElements));
                    AscendC::PipeBarrier<PIPE_V>();
                    generatedRecords += copiedRecords;
                }
            }
        }

        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        }

        uint32_t progress = 0U;
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t buffer = progress & 1U;
            const event_t event = static_cast<event_t>(buffer);
            const uint32_t firstOutputBlock = task * groupSize;
            const uint32_t inputBase = InputBlockOffset(firstOutputBlock);
            const uint32_t outputBase = firstOutputBlock * blockElements;
            const uint32_t recordBytes = blockElements * TYPE_BYTES;

            AscendC::LocalTensor<T> input(
                AscendC::TPosition::VECIN, buffer * CHUNK_BYTES, CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::DataCopyExtParams copyIn{
                static_cast<uint16_t>(groupSize), recordBytes,
                (groupInputStride - blockElements) * TYPE_BYTES, 0U, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0U,
                static_cast<uint8_t>(alignedRecordElements - blockElements),
                static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm[inputBase], copyIn, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(event);

            AscendC::LocalTensor<T> output(
                AscendC::TPosition::VECOUT,
                BUFFER_COUNT * CHUNK_BYTES + buffer * CHUNK_BYTES,
                CHUNK_ELEMENTS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
            AscendC::Gather(output, input, offsets, 0U, compactElements);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(event);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(event);

            AscendC::DataCopyExtParams copyOut{
                1U, compactElements * TYPE_BYTES, 0U, 0U, 0U};
            AscendC::DataCopyPad(zGm[outputBase], output, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
            ++progress;
        }

        for (uint32_t buffer = 0U; buffer < BUFFER_COUNT; ++buffer) {
            const event_t event = static_cast<event_t>(buffer);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
        }
    }

    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) const
    {
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline uint32_t InputBlockOffset(uint32_t outputBlock) const
    {
        uint32_t offset = 0U;
        uint32_t remainder = outputBlock;
        for (int32_t i = static_cast<int32_t>(suffixStart) - 1; i >= 0; --i) {
            const uint32_t axis = perm[i];
            const uint32_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            offset += coordinate * inputStride[axis];
        }
        return offset;
    }

private:
    static constexpr uint32_t TYPE_BYTES = sizeof(T);
    static constexpr uint32_t DMA_BLOCK_BYTES = 32U;
    static constexpr uint32_t DMA_BLOCK_ELEMENTS = DMA_BLOCK_BYTES / TYPE_BYTES;
    static constexpr uint32_t CHUNK_BYTES = 32U * 1024U;
    static constexpr uint32_t CHUNK_ELEMENTS = CHUNK_BYTES / TYPE_BYTES;
    static constexpr uint32_t BUFFER_COUNT = 2U;
    static constexpr uint32_t MERGED_GROUP_COUNT = 2U;
    static constexpr uint32_t WIDE_MERGED_GROUP_COUNT = 4U;
    static constexpr uint32_t WIDE_RECORD_TASK_COUNT = 2U;
    static constexpr uint32_t FIVE_MERGED_GROUP_COUNT = 5U;
    static constexpr uint32_t MIN_FIVE_MERGED_TASKS_PER_CORE = 80U;
    static constexpr uint32_t TEN_MERGED_GROUP_COUNT = 21U;
    static constexpr uint32_t MIN_TEN_MERGED_TASKS_PER_CORE = 24U;
    static constexpr uint32_t THIRTEEN_MERGED_GROUP_COUNT = 13U;
    static constexpr uint32_t TWENTY_FIVE_MERGED_GROUP_COUNT = 25U;
    static constexpr uint32_t MIN_TWENTY_FIVE_MERGED_TASKS_PER_CORE = 20U;
    static constexpr uint32_t NINE_MERGED_GROUP_COUNT = 9U;
    static constexpr uint32_t TWENTY_SEVEN_MERGED_GROUP_COUNT = 27U;
    static constexpr uint32_t MIN_TWENTY_SEVEN_MERGED_TASKS_PER_CORE = 20U;
    static constexpr uint32_t TWENTY_FIVE_BANK_BYTES = 38U * 1024U;
    static constexpr uint32_t TWENTY_FIVE_OFFSET_BYTES = 40U * 1024U;
    static constexpr uint32_t TWENTY_FIVE_OFFSET_UB_OFFSET =
        4U * TWENTY_FIVE_BANK_BYTES;
    static constexpr uint32_t TWENTY_SEVEN_BANK_BYTES = 41U * 1024U;
    static constexpr uint32_t TWENTY_SEVEN_OFFSET_BYTES = 27U * 1024U;
    static constexpr uint32_t TWENTY_SEVEN_OFFSET_UB_OFFSET =
        4U * TWENTY_SEVEN_BANK_BYTES;
    static_assert(
        TWENTY_SEVEN_OFFSET_UB_OFFSET + TWENTY_SEVEN_OFFSET_BYTES <=
        192U * 1024U);
    static constexpr uint32_t MIN_MERGED_BYTES_PER_CORE =
        CHUNK_BYTES * 5U / 2U;
    static constexpr uint64_t MAXIMUM_GATHER_OFFSET_BYTES = 8U * 1024U;
    static constexpr uint32_t OFFSET_ELEMENTS = CHUNK_BYTES / sizeof(uint32_t);
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    uint32_t dims{0};
    uint32_t totalElements{0};
    uint32_t shape[MAXDIM] = {};
    uint32_t perm[MAXDIM] = {};
    uint32_t inputStride[MAXDIM] = {};
    uint32_t suffixStart{0};
    uint32_t blockElements{0};
    uint32_t outerBlockCount{0};
    uint32_t chunksPerBlock{0};
    bool useRecordGroups{false};
    uint32_t groupSize{0};
    uint32_t groupInputStride{0};
    uint32_t alignedRecordElements{0};
    uint32_t recordsPerTask{0};
    uint32_t recordTasksPerGroup{0};
    uint32_t outerGroupCount{0};
    uint32_t totalTaskCount{0};
    bool useMergedRecordGroups{false};
    uint32_t mergedAxisExtent{0};
    uint32_t mergedAxisOutputStride{0};
    uint32_t mergedAxisTileCount{0};
    uint32_t mergedTaskCount{0};
    uint32_t mergedAlignedRecordElements{0};
    uint32_t mergedAlignedGroupElements{0};
    bool useWideMergedRecordGroups{false};
    uint32_t wideRecordsPerTask{0};
    uint32_t wideMergedAxisTileCount{0};
    uint32_t wideMergedTaskCount{0};
    uint32_t wideMergedAlignedRecordElements{0};
    uint32_t wideMergedAlignedGroupElements{0};
    bool useFiveMergedRecordGroups{false};
    uint32_t fiveMergedAxisTileCount{0};
    uint32_t fiveMergedTaskCount{0};
    uint32_t fiveMergedAlignedRecordElements{0};
    bool useTenMergedRecordGroups{false};
    uint32_t tenMergedAxisTileCount{0};
    uint32_t tenMergedTaskCount{0};
    uint32_t tenMergedAlignedRecordElements{0};
    bool useTwentyFiveMergedRecordGroups{false};
    uint32_t twentyFiveMergedAxisTileCount{0};
    uint32_t twentyFiveMergedTaskCount{0};
    uint32_t twentyFiveMergedAlignedRecordElements{0};
    bool useTwentySevenMergedRecordGroups{false};
    uint32_t twentySevenMergedAxisTileCount{0};
    uint32_t twentySevenMergedTaskCount{0};
    uint32_t twentySevenMergedAlignedRecordElements{0};
    uint32_t blockIdx{0};
    uint32_t blockNum{1};
};

// General permutations are a batch of output-contiguous rows. Elements in
// each row have a constant source stride, so aligned 2-D DMA can gather them
// into 32-byte UB rows and pack those rows directly into contiguous output.
// This removes the per-element vector division and scalar GM gather fallback.
template <typename T>
class StridedLastAxisTranspose {
public:
    __aicore__ inline StridedLastAxisTranspose(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, const TransposeTilingData* tilingData)
    {
        dims = static_cast<uint32_t>(tilingData->dims);
        totalElements = static_cast<uint32_t>(tilingData->totalnumber);
        for (uint32_t i = 0U; i < dims; ++i) {
            shape[i] = static_cast<uint32_t>(tilingData->shape[i]);
            perm[i] = static_cast<uint32_t>(tilingData->perm[i]);
        }
        inputStride[dims - 1U] = 1U;
        for (int32_t i = static_cast<int32_t>(dims) - 2; i >= 0; --i) {
            inputStride[i] = inputStride[i + 1] * shape[i + 1];
        }

        lastAxis = perm[dims - 1U];
        rowElements = shape[lastAxis];
        outerRows = totalElements / rowElements;
        chunksPerRow = CeilDiv(rowElements, CHUNK_ROWS);
        totalTaskCount = outerRows * chunksPerRow;
        blockIdx = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();
        xGm.SetGlobalBuffer((__gm__ T*)x, totalElements);
        zGm.SetGlobalBuffer((__gm__ T*)z, totalElements);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t task = blockIdx; task < totalTaskCount; task += blockNum) {
            const uint32_t outputRow = task / chunksPerRow;
            const uint32_t chunk = task - outputRow * chunksPerRow;
            const uint32_t chunkOffset = chunk * CHUNK_ROWS;
            const uint32_t processElements =
                (rowElements - chunkOffset < CHUNK_ROWS) ? rowElements - chunkOffset : CHUNK_ROWS;
            const uint32_t inputBase = InputRowOffset(outputRow) + chunkOffset * inputStride[lastAxis];
            const uint32_t outputBase = outputRow * rowElements + chunkOffset;

            AscendC::LocalTensor<T> local(
                AscendC::TPosition::VECIN, 0U, CHUNK_ROWS * DMA_BLOCK_ELEMENTS);
            AscendC::DataCopyExtParams copyIn{
                static_cast<uint16_t>(processElements), TYPE_BYTES,
                (inputStride[lastAxis] - 1U) * TYPE_BYTES, 0U, 0U};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0U, static_cast<uint8_t>(DMA_BLOCK_ELEMENTS - 1U), static_cast<T>(0)};
            AscendC::DataCopyPad(local, xGm[inputBase], copyIn, pad);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(static_cast<event_t>(COPY_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(static_cast<event_t>(COPY_EVENT));

            AscendC::DataCopyExtParams copyOut{
                static_cast<uint16_t>(processElements), TYPE_BYTES, 0U, 0U, 0U};
            AscendC::DataCopyPad(zGm[outputBase], local, copyOut);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(COPY_EVENT));
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(static_cast<event_t>(COPY_EVENT));
        }
    }

private:
    __aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t divisor) const
    {
        return (value + divisor - 1U) / divisor;
    }

    __aicore__ inline uint32_t InputRowOffset(uint32_t outputRow) const
    {
        uint32_t offset = 0U;
        uint32_t remainder = outputRow;
        for (int32_t i = static_cast<int32_t>(dims) - 2; i >= 0; --i) {
            const uint32_t axis = perm[i];
            const uint32_t coordinate = remainder % shape[axis];
            remainder /= shape[axis];
            offset += coordinate * inputStride[axis];
        }
        return offset;
    }

private:
    static constexpr uint32_t TYPE_BYTES = sizeof(T);
    static constexpr uint32_t DMA_BLOCK_ELEMENTS = 32U / TYPE_BYTES;
    static constexpr uint32_t CHUNK_ROWS = 2048U;
    static constexpr int32_t COPY_EVENT = 0;

    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;
    uint32_t dims{0};
    uint32_t totalElements{0};
    uint32_t shape[MAXDIM]{};
    uint32_t perm[MAXDIM]{};
    uint32_t inputStride[MAXDIM]{};
    uint32_t lastAxis{0};
    uint32_t rowElements{0};
    uint32_t outerRows{0};
    uint32_t chunksPerRow{0};
    uint32_t totalTaskCount{0};
    uint32_t blockIdx{0};
    uint32_t blockNum{1};
};

template <typename T>
class Transpose {
public:
    __aicore__ inline Transpose(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> zGm;

    int64_t coreDataNum;
    int64_t tileNum;
    int64_t tileDataNum;
    int64_t tailDataNum;
    int64_t processDataNum;
    int64_t globalBufferIndex;
    int64_t perm[MAXDIM];
    int64_t dims = 8;
    int64_t Stride[MAXDIM];
    int64_t NewStride[MAXDIM];
};

template <typename T>
__aicore__ inline void Transpose<T>::Init(GM_ADDR x, GM_ADDR y, const TransposeTilingData* tilingData)
{
    this->dims = tilingData->dims;
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

    uint32_t coreNum = AscendC::GetBlockIdx();
    this->globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        this->globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                                   (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }
    xGm.SetGlobalBuffer((__gm__ T*)x, tilingData->totalnumber);
    zGm.SetGlobalBuffer((__gm__ T*)y + this->globalBufferIndex, this->coreDataNum);
    pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(int32_t));
    pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmp3, this->tileDataNum * sizeof(float));
    pipe.InitBuffer(tmp4, this->tileDataNum * sizeof(int32_t));
    pipe.InitBuffer(tmp5, this->tileDataNum * sizeof(int32_t));
    pipe.InitBuffer(tmp6, this->tileDataNum * sizeof(int32_t));

    for (int i = 0; i < dims; i++) {
        perm[i] = tilingData->perm[i];
    }
    Stride[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; --i)
        Stride[i] = Stride[i + 1] * tilingData->shape[i + 1];

    NewStride[dims - 1] = 1;
    for (int i = dims - 2; i >= 0; --i)
        NewStride[i] = NewStride[i + 1] * tilingData->shape[perm[i + 1]];
}

template <typename T>
__aicore__ inline void Transpose<T>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    AscendC::DataCopyExtParams copyParams{
        1U, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0U, 0U, 0U};
    AscendC::DataCopyPad(zGm[progress * this->tileDataNum], zLocal, copyParams);
    outQueueZ.FreeTensor(zLocal);
}

template <typename T>
__aicore__ inline void Transpose<T>::Compute(int32_t progress)
{
    AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
    auto p1 = tmp1.Get<int32_t>();
    auto p2 = tmp2.Get<float>();
    auto p3 = tmp3.Get<float>();
    auto p4 = tmp4.Get<int32_t>();
    auto p5 = tmp5.Get<int32_t>();
    auto p6 = tmp6.Get<int32_t>();
    AscendC::ArithProgression<int32_t>(
        p1, static_cast<int32_t>(this->globalBufferIndex + progress * this->tileDataNum), static_cast<int32_t>(1),
        static_cast<int32_t>(this->processDataNum));

    AscendC::Sub(p6, p6, p6, this->processDataNum);
    for (int j = 0; j < dims; j++) {
        AscendC::Duplicate<float>(p3, static_cast<float>(NewStride[j]), this->processDataNum);
        AscendC::Cast(p2, p1, AscendC::RoundMode::CAST_TRUNC, this->processDataNum);
        AscendC::Div(p2, p2, p3, this->processDataNum);
        AscendC::Cast(p4, p2, AscendC::RoundMode::CAST_FLOOR, this->processDataNum);
        AscendC::Muls(p5, p4, static_cast<int32_t>(NewStride[j]), this->processDataNum);
        AscendC::Sub(p1, p1, p5, this->processDataNum);
        AscendC::Muls(p4, p4, static_cast<int32_t>(Stride[perm[j]]), this->processDataNum);
        AscendC::Add(p6, p6, p4, this->processDataNum);
    }

    for (int total = 0; total < this->processDataNum; total++) {
        zLocal.SetValue(total, xGm.GetValue(p6.GetValue(total)));
    }
    outQueueZ.EnQue<T>(zLocal);
}

template <typename T>
__aicore__ inline void Transpose<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;

    for (int32_t i = 0; i < loopCount; i++) {
        if (i == this->tileNum - 1) {
            this->processDataNum = this->tailDataNum;
        }
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsTranspose
#endif
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
 * \brief
 */


template <uint32_t schMode>
__global__ __aicore__ void transpose(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TransposeTilingData);
    if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_1) {
        NsTranspose::FastTranspose2DFp16 op;
        op.Init(input, output, tiling);
        op.Process();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_2) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DFp16 op;
        op.Init(input, output, tiling, &tilingData);
        op.Process();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_3) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT> op;
        op.Init(input, output, &tilingData);
        op.Process();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_4) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::ContiguousSuffixTranspose<DTYPE_INPUT> op;
        op.Init(input, output, &tilingData);
        op.Process();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_5) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::StridedLastAxisTranspose<DTYPE_INPUT> op;
        op.Init(input, output, &tilingData);
        op.Process();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_6) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT> op;
        op.InitBatched(input, output, &tilingData);
        op.ProcessBatched();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_7) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledRecordTranspose<DTYPE_INPUT> op;
        op.Init(input, output, &tilingData);
        op.Process();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_8) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_9) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_INPUT) == 4U) {
            NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 15U, false, 3U> op;
            op.InitLastTwoAxes(input, output, &tilingData);
            op.ProcessLastTwoAxes();
        } else {
            NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 16U, false, 2U> op;
            op.InitLastTwoAxes(input, output, &tilingData);
            op.ProcessLastTwoAxes();
        }
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_10) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 64U> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_11) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 2U, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_12) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 16U, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_13) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 64U, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_14) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 1U, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_15) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::GeneralRecordTranspose<DTYPE_INPUT> op;
        op.Init(input, output, &tilingData);
        op.Process();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_16) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 2U, true, 2U> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_17) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 4U, false, 12U> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_18) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 8U, false, 8U> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_19) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 4U, true, 2U> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessAdjacentInputBatchPairs();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_20) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 4U, true, 2U> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessAdjacentBatchRectangles();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_21) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 4U, false, 16U, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_22) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 4U, false, 24U, false, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_23) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<
            DTYPE_INPUT, 7U, true, 2U, false, false, 4U, 2U, 7U> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessAdjacentBatchRectangles();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_24) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<
            DTYPE_INPUT, 5U, true, 2U, false, false, 3U, 4U, 13U, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessAdjacentBatchRectangles();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_25) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<
            DTYPE_INPUT, 4U, false, 24U, false, true,
            2U, 2U, 0U, false, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_26) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<DTYPE_INPUT, 8U, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_27) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<
            DTYPE_INPUT, 6U, false, 24U, false, false,
            2U, 2U, 0U, false, false, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_28) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<
            DTYPE_INPUT, 7U, true, 2U, false, false,
            4U, 4U, 13U, true, false, false, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessAdjacentBatchRectangles();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_29) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<
            DTYPE_INPUT, 10U, true, 2U, false, false,
            6U, 3U, 10U, true, false, false, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessAdjacentBatchRectangles();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_30) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<
            DTYPE_INPUT, 6U, false, 24U, false, false,
            2U, 2U, 0U, false, false, true,
            false, false, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_31) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::TiledTranspose2DGeneric<
            DTYPE_INPUT, 6U, false, 24U, false, false,
            2U, 2U, 0U, false, false, true,
            false, false, true, true> op;
        op.InitLastTwoAxes(input, output, &tilingData);
        op.ProcessLastTwoAxes();
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_32) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_INPUT) == 4U) {
            NsTranspose::TiledTranspose2DGeneric<
                DTYPE_INPUT, 10U, true, 2U, false, false,
                6U, 6U, 19U> op;
            op.InitLastTwoAxes(input, output, &tilingData);
            op.ProcessSixOutputRectangles();
        } else {
            NsTranspose::Transpose<DTYPE_INPUT> op;
            op.Init(input, output, &tilingData);
            op.Process();
        }
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_33) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_INPUT) == 4U) {
            NsTranspose::TiledTranspose2DGeneric<
                DTYPE_INPUT, 6U, false, 24U, false, false,
                2U, 2U, 0U, false, false, true,
                false, false, true> op;
            op.InitLastTwoAxes(input, output, &tilingData);
            op.ProcessPairedFullRowWorkspace(AscendC::GetUserWorkspace(workspace));
        } else {
            NsTranspose::Transpose<DTYPE_INPUT> op;
            op.Init(input, output, &tilingData);
            op.Process();
        }
    } else if constexpr (schMode == ELEMENTWISE_TPL_SCH_MODE_34) {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        if constexpr (sizeof(DTYPE_INPUT) == 4U) {
            NsTranspose::TiledTranspose2DGeneric<
                DTYPE_INPUT, 10U, true, 2U, false, false,
                7U, 6U, 19U> op;
            op.InitLastTwoAxes(input, output, &tilingData);
            op.ProcessSevenInputSixOutputRectanglesPaired();
        } else {
            NsTranspose::Transpose<DTYPE_INPUT> op;
            op.Init(input, output, &tilingData);
            op.Process();
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(TransposeTilingData, tilingData, tiling);
        NsTranspose::Transpose<DTYPE_INPUT> op;
        op.Init(input, output, &tilingData);
        op.Process();
    }
}
