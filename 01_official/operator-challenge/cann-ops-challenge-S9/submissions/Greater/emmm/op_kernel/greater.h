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
 * \file greater.h
 * \brief
 */
#ifndef __GREATER_H__
#define __GREATER_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "greater_tiling_data.h"
#include "greater_tiling_key.h"

namespace NsGreater {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr uint64_t VECTOR_ALIGN_BYTES = 256;
constexpr uint64_t DATA_BLOCK_BYTES = 32;
constexpr uint64_t BROADCAST_ROW_BUFFER_BYTES = 6 * 1024;
constexpr uint64_t ADJACENT_BROADCAST_SOURCE_BYTES = 2 * 1024;
constexpr uint64_t INT32_SCALAR_TILE_ELEMENTS = 4096;
constexpr uint64_t INT32_COMPACT_MIN_ROW_COUNT = 4;
constexpr uint64_t DATA_COPY_PAD_MAX_BYTES = 65535;

__aicore__ inline void MaterializeInt32Predicate(
    AscendC::LocalTensor<int8_t> destination, AscendC::LocalTensor<half> result,
    AscendC::LocalTensor<half> zeroBlock, uint64_t count)
{
    constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
    AscendC::Duplicate<half>(zeroBlock, (half)0, halfRepeatElements);
    const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
    const uint8_t repeatTimes = static_cast<uint8_t>(
        (count + halfRepeatElements - 1) / halfRepeatElements);
    AscendC::Select(
        result, destination, zeroBlock, (half)1,
        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, halfRepeatElements,
        repeatTimes, repeatParams);
    AscendC::Cast(destination, result, AscendC::RoundMode::CAST_NONE, count);
}

__aicore__ inline void MaterializeInt8Predicate(
    AscendC::LocalTensor<int8_t> destination,
    AscendC::LocalTensor<int8_t> predicate,
    AscendC::LocalTensor<half> result, AscendC::LocalTensor<half> oneBlock,
    uint64_t count)
{
    constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
    AscendC::Duplicate<half>(oneBlock, (half)1, halfRepeatElements);
    const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
    const uint8_t repeatTimes = static_cast<uint8_t>(
        (count + halfRepeatElements - 1) / halfRepeatElements);
    AscendC::Select(
        result, predicate, oneBlock, (half)0,
        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, halfRepeatElements,
        repeatTimes, repeatParams);
    AscendC::Cast(destination, result, AscendC::RoundMode::CAST_NONE, count);
}

__aicore__ inline void MaterializePreloadedInt8Predicate(
    AscendC::LocalTensor<int8_t> destination,
    AscendC::LocalTensor<int8_t> predicate,
    AscendC::LocalTensor<half> result, AscendC::LocalTensor<half> oneBlock,
    uint64_t count)
{
    constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
    const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
    const uint8_t repeatTimes = static_cast<uint8_t>(
        (count + halfRepeatElements - 1) / halfRepeatElements);
    AscendC::Select(
        result, predicate, oneBlock, repeatTimes, repeatParams);
    AscendC::Cast(destination, result, AscendC::RoundMode::CAST_NONE, count);
}

template <typename TYPE_X1, typename TYPE_X2, typename TYPE_Y, bool IsExistBigCore>
class Greater {
    using T = TYPE_X1;
    static constexpr int32_t INPUT_QUEUE_BUFFER_NUM =
        std::is_same_v<TYPE_X1, float> ? 1 : BUFFER_NUM;
    static constexpr int32_t OUTPUT_QUEUE_BUFFER_NUM =
        (std::is_same_v<TYPE_X1, half> || std::is_same_v<TYPE_X1, float>) ? 1 : BUFFER_NUM;

public:
    __aicore__ inline Greater() {}
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum, uint64_t bigCoreLoopNum,
        uint64_t smallCoreLoopNum, uint64_t ubPartDataNum, uint64_t smallCoreTailDataNum, uint64_t bigCoreTailDataNum,
        uint64_t tailBlockNum)
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint64_t coreNum = AscendC::GetBlockIdx();
        uint64_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;

        if constexpr (IsExistBigCore) {
            if (coreNum < tailBlockNum) {
                this->coreDataNum = bigCoreDataNum;
                this->tileNum = bigCoreLoopNum;
                this->tailDataNum = bigCoreTailDataNum;
            } else {
                this->coreDataNum = smallCoreDataNum;
                this->tileNum = smallCoreLoopNum;
                this->tailDataNum = smallCoreTailDataNum;
                globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
            }
        } else {
            this->coreDataNum = smallCoreDataNum;
            this->tileNum = smallCoreLoopNum;
            this->tailDataNum = smallCoreTailDataNum;
            globalBufferIndex = smallCoreDataNum * AscendC::GetBlockIdx();
        }

        x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x1 + globalBufferIndex, this->coreDataNum);
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X2*)x2 + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(
            inQueueX1, INPUT_QUEUE_BUFFER_NUM,
            this->ubPartDataNum * sizeof(TYPE_X1) + VECTOR_ALIGN_BYTES);
        pipe.InitBuffer(
            inQueueX2, INPUT_QUEUE_BUFFER_NUM,
            this->ubPartDataNum * sizeof(TYPE_X2) + VECTOR_ALIGN_BYTES);
        pipe.InitBuffer(
            outQueueY, OUTPUT_QUEUE_BUFFER_NUM,
            this->ubPartDataNum * sizeof(int8_t) + VECTOR_ALIGN_BYTES);
        if (std::is_same_v<TYPE_X1, float>) {
            pipe.InitBuffer(
                calc_buf_1,
                this->ubPartDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
        } else if (std::is_same_v<TYPE_X1, bfloat16_t>) {
            pipe.InitBuffer(calc_buf_1, this->ubPartDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
            pipe.InitBuffer(calc_buf_2, this->ubPartDataNum * sizeof(float) + VECTOR_ALIGN_BYTES);
            pipe.InitBuffer(calc_buf_3, this->ubPartDataNum * sizeof(float) + VECTOR_ALIGN_BYTES);
        } else if (std::is_same_v<TYPE_X1, int8_t> || std::is_same_v<TYPE_X1, uint8_t>) {
            pipe.InitBuffer(calc_buf_1, this->ubPartDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
            pipe.InitBuffer(calc_buf_2, this->ubPartDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
        } else if (std::is_same_v<TYPE_X1, int32_t>) {
            pipe.InitBuffer(calc_buf_1, this->ubPartDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
        }
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->ubPartDataNum;
        for (int32_t i = 0; i < loopCount - 1; i++) {
            SetComputeDataNum();
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        this->processDataNum = this->tailDataNum;
        SetComputeDataNum();
        CopyIn(loopCount - 1);
        Compute(loopCount - 1);
        CopyOut(loopCount - 1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template AllocTensor<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template AllocTensor<TYPE_X2>();
        AscendC::DataCopy(x1Local, x1Gm[progress * this->ubPartDataNum], this->processDataNum);
        AscendC::DataCopy(x2Local, x2Gm[progress * this->ubPartDataNum], this->processDataNum);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        if constexpr (std::is_same_v<TYPE_X1, half>) {
            AscendC::LocalTensor<half> x1Local = inQueueX1.template DeQue<half>();
            AscendC::LocalTensor<half> x2Local = inQueueX2.template DeQue<half>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::GT, this->processDataNumComputes);
            AscendC::Duplicate<half>(x1Local, (half)1, this->processDataNumComputes);
            AscendC::Select(
                x1Local, yLocal, x1Local, (half)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                this->processDataNumComputes);
            AscendC::Cast(yLocal, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<TYPE_X1, bfloat16_t>) {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::LocalTensor<half> yFp16 = calc_buf_1.Get<half>();
            AscendC::LocalTensor<float> x1Fp32 = calc_buf_2.Get<float>();
            AscendC::LocalTensor<float> x2Fp32 = calc_buf_3.Get<float>();
            AscendC::Cast(x1Fp32, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
            AscendC::Cast(x2Fp32, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
            AscendC::Compare(yLocal, x1Fp32, x2Fp32, AscendC::CMPMODE::GT, this->processDataNumComputes);
            constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
            AscendC::LocalTensor<half> oneBlock = x1Fp32.template ReinterpretCast<half>();
            AscendC::Duplicate<half>(oneBlock, (half)1, halfRepeatElements);
            const uint8_t repeatTimes = static_cast<uint8_t>(
                this->processDataNumComputes / halfRepeatElements);
            const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
            AscendC::Select(
                yFp16, yLocal, oneBlock, (half)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                halfRepeatElements, repeatTimes, repeatParams);
            AscendC::Cast(yLocal, yFp16, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<TYPE_X1, float>) {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::LocalTensor<half> yFp16 = calc_buf_1.Get<half>();
            AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::GT, this->processDataNumComputes);
            AscendC::Duplicate<float>(x1Local, 1.0F, this->processDataNumComputes);
            AscendC::Select(
                x1Local, yLocal, x1Local, 0.0F, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                this->processDataNumComputes);
            AscendC::Cast(yFp16, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
            AscendC::Cast(yLocal, yFp16, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<TYPE_X1, int8_t> || std::is_same_v<TYPE_X1, uint8_t>) {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::LocalTensor<half> x1Fp16 = calc_buf_1.Get<half>();
            AscendC::LocalTensor<half> x2Fp16 = calc_buf_2.Get<half>();
            AscendC::Cast(x1Fp16, x1Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
            AscendC::Cast(x2Fp16, x2Local, AscendC::RoundMode::CAST_NONE, this->processDataNumComputes);
            AscendC::Compare(yLocal, x1Fp16, x2Fp16, AscendC::CMPMODE::GT, this->processDataNumComputes);
            AscendC::Duplicate<half>(x1Fp16, (half)1, this->processDataNumComputes);
            AscendC::Select(
                x1Fp16, yLocal, x1Fp16, (half)0,
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                this->processDataNumComputes);
            AscendC::Cast(
                yLocal, x1Fp16, AscendC::RoundMode::CAST_NONE,
                this->processDataNumComputes);
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<TYPE_X1, int32_t>) {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::LocalTensor<half> yFp16 = calc_buf_1.Get<half>();
            AscendC::LocalTensor<half> zeroBlock = x2Local.template ReinterpretCast<half>();
            AscendC::Max(x1Local, x1Local, x2Local, this->processDataNumComputes);
            AscendC::Compare(yLocal, x2Local, x1Local, AscendC::CMPMODE::EQ, this->processDataNumComputes);
            MaterializeInt32Predicate(
                yLocal, yFp16, zeroBlock, this->processDataNumComputes);
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
    }

    __aicore__ inline void SetComputeDataNum()
    {
        uint64_t processBytes = this->processDataNum * sizeof(TYPE_X1);
        this->processDataNumComputes =
            ((processBytes + VECTOR_ALIGN_BYTES - 1) / VECTOR_ALIGN_BYTES) * VECTOR_ALIGN_BYTES / sizeof(TYPE_X1);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.template DeQue<int8_t>();
        AscendC::DataCopy(yGm[progress * this->ubPartDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, INPUT_QUEUE_BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN, INPUT_QUEUE_BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECOUT, OUTPUT_QUEUE_BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calc_buf_1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calc_buf_2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calc_buf_3;
    AscendC::GlobalTensor<TYPE_X1> x1Gm;
    AscendC::GlobalTensor<TYPE_X2> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint64_t coreDataNum;
    uint64_t tileNum;
    uint64_t ubPartDataNum;
    uint64_t tailDataNum;
    uint64_t processDataNum;
    uint64_t processDataNumComputes;
};

template <
    typename TYPE_X1, typename TYPE_X2, typename TYPE_Y, uint32_t BroadcastMode,
    bool AdjacentScalarBlockPath = false, bool AdjacentSingleTile = false,
    bool AdjacentPaddedPlanePath = false, bool Int32PaddedRowPath = false,
    bool ReuseInt8Predicate = false, bool Int8WaveMajor = false,
    bool Fp32SteadyWaveMajor = false>
class GreaterFlexible {
    using T = TYPE_X1;
    static constexpr bool INT8_SEGMENTED_MACRO_TILE =
        ReuseInt8Predicate && std::is_same_v<TYPE_X1, int8_t> &&
        BroadcastMode == GREATER_BROADCAST_CONTIGUOUS;
    static constexpr bool INT8_WAVE_MAJOR =
        INT8_SEGMENTED_MACRO_TILE && Int8WaveMajor;
    static constexpr bool FP32_STEADY_WAVE_MAJOR =
        Fp32SteadyWaveMajor && std::is_same_v<TYPE_X1, float> &&
        BroadcastMode == GREATER_BROADCAST_CONTIGUOUS;
    static constexpr bool BF16_REPEATED_ROW_BATCH =
        Fp32SteadyWaveMajor && std::is_same_v<TYPE_X1, bfloat16_t> &&
        BroadcastMode == GREATER_BROADCAST_GENERAL;
    static constexpr int32_t QUEUE_BUFFER_NUM =
        FP32_STEADY_WAVE_MAJOR ? BUFFER_NUM :
        (AdjacentScalarBlockPath || AdjacentPaddedPlanePath || Int32PaddedRowPath ||
         BF16_REPEATED_ROW_BATCH ||
         (std::is_same_v<TYPE_X1, float> && BroadcastMode == GREATER_BROADCAST_CONTIGUOUS)) ?
        1 : BUFFER_NUM;
    static constexpr int32_t OUTPUT_QUEUE_BUFFER_NUM =
        FP32_STEADY_WAVE_MAJOR ? 1 :
        BF16_REPEATED_ROW_BATCH ? BUFFER_NUM :
        (INT8_SEGMENTED_MACRO_TILE ? 1 : QUEUE_BUFFER_NUM);

public:
    __aicore__ inline GreaterFlexible() {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y, uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
        uint64_t ubPartDataNum, uint64_t smallCoreTailDataNum, uint64_t bigCoreTailDataNum,
        uint64_t smallCoreLoopNum, uint64_t bigCoreLoopNum, uint64_t tailBlockNum, uint32_t rank,
        const uint64_t* outShape, const uint64_t* x1Stride, const uint64_t* x2Stride)
    {
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        this->ubPartDataNum = ubPartDataNum;
        if constexpr (INT8_WAVE_MAJOR) {
            this->outShape[0] = smallCoreDataNum;
            this->outShape[1] = bigCoreDataNum;
            this->outShape[2] = smallCoreLoopNum;
            this->outShape[3] = bigCoreLoopNum;
            this->outShape[4] = tailBlockNum;
        }
        if (tailBlockNum != 0 && blockIdx < tailBlockNum) {
            coreDataNum = bigCoreDataNum;
            tileNum = bigCoreLoopNum;
            tailDataNum = bigCoreTailDataNum;
            blockOffset = blockIdx * bigCoreDataNum;
        } else {
            coreDataNum = smallCoreDataNum;
            tileNum = smallCoreLoopNum;
            tailDataNum = smallCoreTailDataNum;
            blockOffset = tailBlockNum * bigCoreDataNum + (blockIdx - tailBlockNum) * smallCoreDataNum;
        }
        if constexpr (FP32_STEADY_WAVE_MAJOR) {
            waveTailStride = bigCoreTailDataNum;
        }
        this->rank = 0;
        if constexpr (BroadcastMode == GREATER_BROADCAST_GENERAL) {
            this->rank = AdjacentScalarBlockPath ? 4 : rank;
            constexpr uint32_t copiedRank =
                AdjacentScalarBlockPath ? 4 : GREATER_MAX_BROADCAST_DIM;
            for (uint32_t index = 0; index < copiedRank; ++index) {
                this->outShape[index] = outShape[index];
                this->x1Stride[index] = x1Stride[index];
                this->x2Stride[index] = x2Stride[index];
            }
            if constexpr (std::is_same_v<TYPE_X1, bfloat16_t>) {
                if (IsRepeatedRowBroadcast(this->x1Stride, this->x2Stride)) {
                    tailIndex = 1;
                } else if (IsRepeatedRowBroadcast(this->x2Stride, this->x1Stride)) {
                    tailIndex = 2;
                }
            }
        }
        if constexpr (AdjacentPaddedPlanePath) {
            tailIndex = this->rank - 1;
            rowIndex = this->rank - 2;
            planeGroupIndex = this->rank - 3;
            paddedRowLength =
                (this->outShape[tailIndex] + DATA_BLOCK_BYTES - 1) /
                DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            paddedPlaneLength = this->outShape[rowIndex] * paddedRowLength;
            const uint64_t maxPlanesByUb = this->ubPartDataNum / paddedPlaneLength;
            const uint64_t rowBytes = paddedRowLength * sizeof(TYPE_X1);
            const uint64_t maxPlanesByRowBuffer = BROADCAST_ROW_BUFFER_BYTES / rowBytes;
            maxPaddedPlaneCount = maxPlanesByUb < maxPlanesByRowBuffer ?
                maxPlanesByUb : maxPlanesByRowBuffer;
            maxPaddedPlaneCount = maxPaddedPlaneCount < this->outShape[planeGroupIndex] ?
                maxPaddedPlaneCount : this->outShape[planeGroupIndex];
            if constexpr (std::is_same_v<TYPE_X1, int8_t>) {
                const uint64_t rowLength = this->outShape[tailIndex];
                const uint64_t planeOutputBytes = this->outShape[rowIndex] * rowLength;
                const bool compactAdjacentOutput =
                    paddedRowLength != rowLength && rowLength % sizeof(uint16_t) == 0 &&
                    planeOutputBytes != 0 && planeOutputBytes <= DATA_COPY_PAD_MAX_BYTES;
                if (compactAdjacentOutput) {
                    const uint64_t maxPlanesByCopy = DATA_COPY_PAD_MAX_BYTES / planeOutputBytes;
                    maxPaddedPlaneCount = maxPaddedPlaneCount < maxPlanesByCopy ?
                        maxPaddedPlaneCount : maxPlanesByCopy;
                }
            }
            this->ubPartDataNum = maxPaddedPlaneCount * paddedPlaneLength;
        }
        if constexpr (Int32PaddedRowPath) {
            tailIndex = this->rank - 1;
            rowIndex = this->rank - 2;
            paddedRowLength =
                (this->outShape[tailIndex] + DATA_BLOCK_BYTES - 1) /
                DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            maxPaddedRowCount = this->ubPartDataNum / paddedRowLength;
            maxPaddedRowCount = maxPaddedRowCount < this->outShape[rowIndex] ?
                maxPaddedRowCount : this->outShape[rowIndex];
            this->ubPartDataNum = maxPaddedRowCount * paddedRowLength;
        }

        x1Gm.SetGlobalBuffer((__gm__ TYPE_X1*)x1);
        x2Gm.SetGlobalBuffer((__gm__ TYPE_X2*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);
        if constexpr (
            (std::is_same_v<TYPE_X1, float> &&
             BroadcastMode == GREATER_BROADCAST_CONTIGUOUS) ||
            INT8_SEGMENTED_MACRO_TILE) {
            x1Gm.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
            x2Gm.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        if constexpr (std::is_same_v<TYPE_X1, bfloat16_t> &&
            BroadcastMode == GREATER_BROADCAST_X1_SCALAR) {
            pipe.InitBuffer(inQueueX1, QUEUE_BUFFER_NUM, DATA_BLOCK_BYTES);
        } else {
            pipe.InitBuffer(
                inQueueX1, QUEUE_BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X1) + VECTOR_ALIGN_BYTES);
        }
        if constexpr (std::is_same_v<TYPE_X2, bfloat16_t> &&
            BroadcastMode == GREATER_BROADCAST_X2_SCALAR) {
            pipe.InitBuffer(inQueueX2, QUEUE_BUFFER_NUM, DATA_BLOCK_BYTES);
        } else {
            pipe.InitBuffer(
                inQueueX2, QUEUE_BUFFER_NUM, this->ubPartDataNum * sizeof(TYPE_X2) + VECTOR_ALIGN_BYTES);
        }
        if constexpr (!AdjacentScalarBlockPath && !AdjacentPaddedPlanePath && !Int32PaddedRowPath) {
            pipe.InitBuffer(x1ScalarQueue, 1, 32);
            pipe.InitBuffer(x2ScalarQueue, 1, 32);
        }
        uint64_t rowBufferBytes =
            (AdjacentPaddedPlanePath || Int32PaddedRowPath) ? BROADCAST_ROW_BUFFER_BYTES : DATA_BLOCK_BYTES;
        if constexpr (
            BroadcastMode == GREATER_BROADCAST_GENERAL && !AdjacentScalarBlockPath &&
            !AdjacentPaddedPlanePath && !Int32PaddedRowPath) {
            if (rank > 1) {
                const uint64_t tailBytes = this->outShape[rank - 1] * sizeof(TYPE_X1);
                useRowBroadcastBuffer = !std::is_same_v<TYPE_X1, int32_t> && tailBytes != 0 &&
                                        tailBytes <= BROADCAST_ROW_BUFFER_BYTES &&
                                        tailBytes % DATA_BLOCK_BYTES == 0 &&
                                        (IsOuterBroadcast(this->x1Stride, this->x2Stride) ||
                                         IsOuterBroadcast(this->x2Stride, this->x1Stride));
                if (useRowBroadcastBuffer) {
                    rowBufferBytes = tailBytes;
                }
                if constexpr (
                    std::is_same_v<TYPE_X1, half> || std::is_same_v<TYPE_X1, bfloat16_t> ||
                    std::is_same_v<TYPE_X1, float> || std::is_same_v<TYPE_X1, int8_t>) {
                    const uint64_t rowCount = this->outShape[rank - 2];
                    const uint64_t scalarBytes = rowCount * sizeof(TYPE_X1);
                    useAdjacentBroadcast = tailBytes != 0 && scalarBytes != 0 &&
                                           tailBytes <= ADJACENT_BROADCAST_SOURCE_BYTES &&
                                           scalarBytes <= ADJACENT_BROADCAST_SOURCE_BYTES &&
                                           ((this->x1Stride[rank - 1] == 0 && this->x1Stride[rank - 2] == 1 &&
                                             this->x2Stride[rank - 1] == 1 && this->x2Stride[rank - 2] == 0) ||
                                            (this->x2Stride[rank - 1] == 0 && this->x2Stride[rank - 2] == 1 &&
                                             this->x1Stride[rank - 1] == 1 && this->x1Stride[rank - 2] == 0));
                    if (useAdjacentBroadcast) {
                        const uint64_t sourceBytes = tailBytes > scalarBytes ? tailBytes : scalarBytes;
                        rowBufferBytes = rowBufferBytes > sourceBytes ? rowBufferBytes : sourceBytes;
                        const bool scalarInvariantAcrossPlanes = rank == 4 &&
                            ((this->x1Stride[rank - 1] == 0 && this->x1Stride[1] == 0) ||
                             (this->x2Stride[rank - 1] == 0 && this->x2Stride[1] == 0));
                        useAdjacentPlaneCache = scalarBytes <= VECTOR_ALIGN_BYTES && scalarInvariantAcrossPlanes;
                        const uint64_t planeRowGapBytes = rowCount > 0 ? (rowCount - 1) * tailBytes : 0;
                        const uint64_t rowPlaneStride = this->x1Stride[rank - 1] == 0 ?
                            this->x2Stride[1] : this->x1Stride[1];
                        useAdjacentStridedRows = useAdjacentPlaneCache && this->outShape[1] <= 4095 &&
                                                 planeRowGapBytes % DATA_BLOCK_BYTES == 0 &&
                                                 planeRowGapBytes / DATA_BLOCK_BYTES <= 65535 &&
                                                 (this->outShape[1] == 1 || rowPlaneStride == this->outShape[3]);
                        const uint64_t planeLength = rowCount * this->outShape[rank - 1];
                        if constexpr (std::is_same_v<TYPE_X1, int8_t>) {
                            useAdjacentBroadcast = useAdjacentPlaneCache && planeLength <= this->ubPartDataNum;
                        }
                        const uint64_t spareElements = planeLength == 0 ? 0 :
                            this->ubPartDataNum % planeLength + VECTOR_ALIGN_BYTES / sizeof(TYPE_X1);
                        useAdjacentScalarBlocks = !std::is_same_v<TYPE_X1, int8_t> && useAdjacentStridedRows &&
                            spareElements >= rowCount * DATA_BLOCK_BYTES / sizeof(TYPE_X1);
                    }
                }
            }
        }
        if constexpr (!AdjacentScalarBlockPath) {
            pipe.InitBuffer(broadcastRowQueue, 1, rowBufferBytes);
        }
        pipe.InitBuffer(
            outQueueY, OUTPUT_QUEUE_BUFFER_NUM,
            this->ubPartDataNum * (FP32_STEADY_WAVE_MAJOR ? 2 : 1) *
                sizeof(int8_t) + VECTOR_ALIGN_BYTES);
        if (std::is_same_v<TYPE_X1, float>) {
            if constexpr (BroadcastMode != GREATER_BROADCAST_CONTIGUOUS) {
                pipe.InitBuffer(
                    calc_buf_1,
                    this->ubPartDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
            }
        } else if (std::is_same_v<TYPE_X1, bfloat16_t>) {
            if constexpr (AdjacentScalarBlockPath || AdjacentPaddedPlanePath) {
                pipe.InitBuffer(calc_buf_1, this->ubPartDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
            }
            pipe.InitBuffer(calc_buf_2, this->ubPartDataNum * sizeof(float) + VECTOR_ALIGN_BYTES);
            pipe.InitBuffer(calc_buf_3, this->ubPartDataNum * sizeof(float) + VECTOR_ALIGN_BYTES);
            if constexpr (BF16_REPEATED_ROW_BATCH) {
                // The compact row lives temporarily in the streamed FP32
                // buffer. Only Broadcast's disjoint scratch stays dedicated.
                pipe.InitBuffer(calc_buf_1, VECTOR_ALIGN_BYTES);
            }
        } else if (std::is_same_v<TYPE_X1, int8_t> || std::is_same_v<TYPE_X1, uint8_t>) {
            const uint64_t calcDataNum = INT8_SEGMENTED_MACRO_TILE ?
                this->ubPartDataNum / 2 : this->ubPartDataNum;
            pipe.InitBuffer(calc_buf_1, calcDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
            pipe.InitBuffer(calc_buf_2, calcDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
        } else if (std::is_same_v<TYPE_X1, int32_t>) {
            pipe.InitBuffer(calc_buf_1, this->ubPartDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum == 0 || tileNum == 0 || ubPartDataNum == 0) {
            return;
        }
        if constexpr (INT8_SEGMENTED_MACRO_TILE) {
            InitInt8PredicateConstants();
        }
        if constexpr (INT8_WAVE_MAJOR) {
            ProcessInt8WaveMajor();
            return;
        }
        if constexpr (FP32_STEADY_WAVE_MAJOR) {
            ProcessFp32SteadyWaveMajor();
            return;
        }
        if constexpr (AdjacentPaddedPlanePath) {
            ProcessAdjacentPaddedPlanes();
            return;
        }
        if constexpr (Int32PaddedRowPath) {
            ProcessInt32PaddedRows();
            return;
        }
        if constexpr (AdjacentScalarBlockPath) {
            ProcessAdjacentScalarBlocks();
            return;
        }
        if constexpr (
            std::is_same_v<TYPE_X1, bfloat16_t> &&
            BroadcastMode == GREATER_BROADCAST_GENERAL) {
            if (tailIndex != 0) {
                if constexpr (BF16_REPEATED_ROW_BATCH) {
                    ProcessBf16RepeatedRowBatches();
                } else {
                    ProcessBf16RepeatedRows();
                }
                return;
            }
        }
        uint64_t outputOffset = blockOffset;
        uint64_t remaining = coreDataNum;
        while (remaining > 0) {
            uint64_t validLength = remaining > ubPartDataNum ? ubPartDataNum : remaining;
            if constexpr (BroadcastMode == GREATER_BROADCAST_GENERAL) {
                if (useAdjacentBroadcast && rank > 2 && remaining > validLength) {
                    const uint64_t planeLength = outShape[rank - 2] * outShape[rank - 1];
                    if (planeLength != 0 && outputOffset % planeLength == 0) {
                        const uint64_t alignedLength = validLength / planeLength * planeLength;
                        if (alignedLength != 0) {
                            validLength = alignedLength;
                        }
                    }
                }
            }
            if constexpr (
                std::is_same_v<TYPE_X1, int32_t> &&
                (BroadcastMode == GREATER_BROADCAST_X1_SCALAR ||
                 BroadcastMode == GREATER_BROADCAST_X2_SCALAR)) {
                if (validLength > INT32_SCALAR_TILE_ELEMENTS) {
                    validLength = INT32_SCALAR_TILE_ELEMENTS;
                }
            }
            bool packed = false;
            if constexpr (BroadcastMode == GREATER_BROADCAST_GENERAL) {
                if (rank > 0 && outShape[rank - 1] != 0) {
                    const uint64_t tailLength = outShape[rank - 1];
                    const bool alignedStart = outputOffset % tailLength == 0;
                    const bool alignedRow = tailLength * sizeof(TYPE_X1) % DATA_BLOCK_BYTES == 0;
                    if (!std::is_same_v<TYPE_X1, int32_t> && alignedStart && alignedRow &&
                        tailLength <= ubPartDataNum) {
                        const uint64_t packedLength = validLength / tailLength * tailLength;
                        if (packedLength != 0) {
                            validLength = packedLength;
                            packed = true;
                        }
                    }
                    if (!packed) {
                        const uint64_t tailRemaining = tailLength - outputOffset % tailLength;
                        validLength = validLength > tailRemaining ? tailRemaining : validLength;
                    }
                }
            }
            if (validLength == 0) {
                return;
            }
            processDataNum = validLength;
            SetComputeDataNum();
            if (packed) {
                CopyInPacked(outputOffset);
            } else {
                CopyIn(outputOffset);
            }
            Compute();
            CopyOut(outputOffset);
            outputOffset += validLength;
            remaining -= validLength;
        }
    }

private:
    __aicore__ inline uint64_t Fp32SteadyWaveOffset(
        uint64_t wave, uint64_t fullWaveCount)
    {
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t blockCount = AscendC::GetBlockNum();
        const uint64_t fullWaveAdvance = blockCount * ubPartDataNum;
        const uint64_t pairedWaveCount = fullWaveCount / 2 * 2;
        if (wave < pairedWaveCount) {
            return (wave / 2) * (2 * fullWaveAdvance) +
                blockIdx * (2 * ubPartDataNum) + (wave % 2) * ubPartDataNum;
        }
        if (wave < fullWaveCount) {
            return wave * fullWaveAdvance + blockIdx * ubPartDataNum;
        }
        return fullWaveCount * fullWaveAdvance +
            blockIdx * waveTailStride;
    }

    __aicore__ inline void ComputeFp32SteadyTo(
        AscendC::LocalTensor<int8_t> yLocal)
    {
        AscendC::LocalTensor<float> x1Local = inQueueX1.template DeQue<float>();
        AscendC::LocalTensor<float> x2Local = inQueueX2.template DeQue<float>();
        AscendC::LocalTensor<half> yFp16 = x1Local.template ReinterpretCast<half>();
        AscendC::LocalTensor<half> oneBlock = x2Local.template ReinterpretCast<half>();
        AscendC::Compare(
            yLocal, x1Local, x2Local, AscendC::CMPMODE::GT, processDataNumComputes);
        constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
        AscendC::Duplicate<half>(oneBlock, (half)1, halfRepeatElements);
        const uint8_t repeatTimes = static_cast<uint8_t>(
            (processDataNumComputes + halfRepeatElements - 1) / halfRepeatElements);
        const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
        AscendC::Select(
            yFp16, yLocal, oneBlock, (half)0,
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, halfRepeatElements,
            repeatTimes, repeatParams);
        AscendC::Cast(yLocal, yFp16, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void ProcessFp32SteadyWaveMajor()
    {
        const uint64_t fullWaveCount = tileNum - 1;
        const uint64_t pairedWaveCount = fullWaveCount / 2 * 2;
        processDataNum = ubPartDataNum;
        processDataNumComputes = ubPartDataNum;
        CopyIn(Fp32SteadyWaveOffset(0, fullWaveCount));
        AscendC::LocalTensor<int8_t> outputPair;
        uint64_t outputPairOffset = 0;
        for (uint64_t wave = 0; wave < tileNum; ++wave) {
            if (wave + 1 < tileNum) {
                processDataNum = wave + 1 == fullWaveCount ? tailDataNum : ubPartDataNum;
                SetComputeDataNum();
                CopyIn(Fp32SteadyWaveOffset(wave + 1, fullWaveCount));
            }
            processDataNum = wave == fullWaveCount ? tailDataNum : ubPartDataNum;
            SetComputeDataNum();
            const bool paired = wave < pairedWaveCount;
            const uint64_t slot = paired ? wave % 2 : 0;
            if (slot == 0) {
                outputPair = outQueueY.template AllocTensor<int8_t>();
                outputPairOffset = Fp32SteadyWaveOffset(wave, fullWaveCount);
            }
            ComputeFp32SteadyTo(outputPair[slot * ubPartDataNum]);
            if (!paired || slot == 1) {
                // First-stage bytes remain live until the second stage is
                // complete; paired GM ownership is contiguous by construction.
                if (paired) {
                    processDataNum = 2 * ubPartDataNum;
                }
                outQueueY.template EnQue<int8_t>(outputPair);
                CopyOut(outputPairOffset);
            }
        }
    }

    __aicore__ inline void ProcessInt8WaveMajor()
    {
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t blockCount = AscendC::GetBlockNum();
        const uint64_t bigClassCount = outShape[4];
        const uint64_t commonFullWaves = bigClassCount == 0 ?
            outShape[2] - 1 :
            (outShape[3] < outShape[2] ? outShape[3] : outShape[2]) - 1;
        const uint64_t fullWaveAdvance = blockCount * ubPartDataNum;
        uint64_t outputOffset = blockIdx * ubPartDataNum;
        for (uint64_t wave = 0; wave < commonFullWaves; ++wave) {
            processDataNum = ubPartDataNum;
            SetComputeDataNum();
            CopyIn(outputOffset);
            Compute();
            CopyOut(outputOffset);
            outputOffset += fullWaveAdvance;
        }

        const uint64_t bigResidualWaves = bigClassCount == 0 ?
            0 : outShape[3] - commonFullWaves;
        const uint64_t smallResidualWaves =
            outShape[2] - commonFullWaves;
        const uint64_t bigTailLength = bigResidualWaves == 0 ?
            0 : outShape[1] -
                (commonFullWaves + bigResidualWaves - 1) * ubPartDataNum;
        const uint64_t smallTailLength = outShape[0] -
            (commonFullWaves + smallResidualWaves - 1) * ubPartDataNum;
        const uint64_t residualWaveCount =
            bigResidualWaves > smallResidualWaves ?
            bigResidualWaves : smallResidualWaves;
        uint64_t waveBase = commonFullWaves * fullWaveAdvance;
        for (uint64_t wave = 0; wave < residualWaveCount; ++wave) {
            const uint64_t bigLength = wave >= bigResidualWaves ? 0 :
                (wave == 0 ? bigTailLength : ubPartDataNum);
            const uint64_t smallLength = wave >= smallResidualWaves ? 0 :
                (wave == 0 ? smallTailLength : ubPartDataNum);
            uint64_t validLength = 0;
            if (blockIdx < bigClassCount) {
                validLength = bigLength;
                outputOffset = waveBase + blockIdx * bigLength;
            } else {
                validLength = smallLength;
                outputOffset = waveBase + bigClassCount * bigLength +
                    (blockIdx - bigClassCount) * smallLength;
            }
            if (validLength != 0) {
                processDataNum = validLength;
                SetComputeDataNum();
                CopyIn(outputOffset);
                Compute();
                CopyOut(outputOffset);
            }
            waveBase += bigClassCount * bigLength +
                (blockCount - bigClassCount) * smallLength;
        }
    }

    __aicore__ inline void ProcessBf16RepeatedRows()
    {
        const uint64_t rowLength = outShape[rank - 1];
        const uint64_t rowCount = outShape[rank - 2];
        const uint64_t planeLength = rowCount * rowLength;
        uint64_t outputOffset = blockOffset;
        uint64_t repeatedOffset = outputOffset / planeLength * rowLength;
        uint64_t currentRow = outputOffset / rowLength % rowCount;
        uint64_t remaining = coreDataNum;
        while (remaining >= rowLength) {
            processDataNum = rowLength;
            SetComputeDataNum();
            CacheBf16RepeatedRow(repeatedOffset);
            const uint64_t remainingRows = remaining / rowLength;
            const uint64_t rowsUntilPlaneEnd = rowCount - currentRow;
            const uint64_t groupRows = remainingRows < rowsUntilPlaneEnd ?
                remainingRows : rowsUntilPlaneEnd;
            for (uint64_t row = 0; row < groupRows; ++row) {
                CopyInBf16ContiguousRow(outputOffset);
                ComputeBf16WithCachedRow();
                CopyOut(outputOffset);
                outputOffset += rowLength;
                remaining -= rowLength;
            }
            currentRow += groupRows;
            if (currentRow != rowCount) {
                continue;
            }
            currentRow = 0;
            repeatedOffset += rowLength;
        }
    }

    __aicore__ inline void ProcessBf16RepeatedRowBatches()
    {
        const uint64_t rowLength = outShape[rank - 1];
        const uint64_t paddedRowLength =
            (rowLength + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES *
            DATA_BLOCK_BYTES;
        const uint64_t rowCount = outShape[rank - 2];
        const uint64_t planeLength = rowCount * rowLength;
        const bool compactBatch =
            rowLength <= DATA_COPY_PAD_MAX_BYTES / sizeof(TYPE_X1) &&
            paddedRowLength * sizeof(float) / DATA_BLOCK_BYTES <= 65535;
        uint64_t maxBatchRows = ubPartDataNum / paddedRowLength;
        if (compactBatch) {
            const uint64_t maxDmaRows =
                DATA_COPY_PAD_MAX_BYTES / (rowLength * sizeof(TYPE_X1));
            maxBatchRows = maxBatchRows < maxDmaRows ? maxBatchRows : maxDmaRows;
            maxBatchRows = maxBatchRows < 255 ? maxBatchRows : 255;
        }
        uint64_t outputOffset = blockOffset;
        uint64_t repeatedOffset = outputOffset / planeLength * rowLength;
        uint64_t currentRow = outputOffset / rowLength % rowCount;
        uint64_t remaining = coreDataNum;
        while (remaining >= rowLength) {
            const uint64_t remainingRows = remaining / rowLength;
            const uint64_t rowsUntilPlaneEnd = rowCount - currentRow;
            uint64_t groupRows = remainingRows < rowsUntilPlaneEnd ?
                remainingRows : rowsUntilPlaneEnd;
            // The expanded operand is read-only throughout this natural
            // broadcast group. Only the streaming operand's raw/FP32 buffers
            // are reused by predicate materialization. Expand the largest
            // batch once; every later batch consumes a prefix of these rows.
            const uint64_t cachedRows = groupRows < maxBatchRows ?
                groupRows : maxBatchRows;
            processDataNum = rowLength;
            SetComputeDataNum();
            CacheBf16RepeatedRow(repeatedOffset);
            // Both raw queues are free after the row-cache Cast. Borrow the
            // otherwise idle repeated-side queue as the second streaming slot.
            // Raw input is disjoint from cache preparation. Issue it before
            // Gather's count-return dependency can hold back the first DMA.
            bool inputInX1Queue = true;
            CopyInBf16BorrowedRows<true>(
                outputOffset, rowLength, paddedRowLength, cachedRows, compactBatch);
            ExpandBf16RepeatedRow(rowLength, paddedRowLength, cachedRows, compactBatch);
            AscendC::PipeBarrier<PIPE_V>();  // expanded row -> Compare RAW
            while (groupRows != 0) {
                const uint64_t batchRows = groupRows < maxBatchRows ?
                    groupRows : maxBatchRows;
                const uint64_t logicalLength = batchRows * rowLength;
                const uint64_t rowsAfterBatch = groupRows - batchRows;
                const uint64_t nextBatchRows = rowsAfterBatch < maxBatchRows ?
                    rowsAfterBatch : maxBatchRows;
                if (nextBatchRows != 0) {
                    if (inputInX1Queue) {
                        CopyInBf16BorrowedRows<false>(
                            outputOffset + logicalLength, rowLength,
                            paddedRowLength, nextBatchRows, compactBatch);
                    } else {
                        CopyInBf16BorrowedRows<true>(
                            outputOffset + logicalLength, rowLength,
                            paddedRowLength, nextBatchRows, compactBatch);
                    }
                }
                processDataNum = compactBatch ? logicalLength : batchRows * paddedRowLength;
                SetComputeDataNum();
                if (inputInX1Queue) {
                    ComputeBf16BorrowedRows<true>();
                } else {
                    ComputeBf16BorrowedRows<false>();
                }
                if (compactBatch) {
                    CopyOut(outputOffset);
                } else {
                    CopyOutBf16PaddedRows(outputOffset, rowLength, batchRows);
                }
                outputOffset += logicalLength;
                remaining -= logicalLength;
                currentRow += batchRows;
                groupRows -= batchRows;
                inputInX1Queue = !inputInX1Queue;
            }
            // The next group's Broadcast may overwrite the retained operand
            // only after the last Compare has consumed it (Vector WAR).
            AscendC::PipeBarrier<PIPE_V>();
            if (currentRow != rowCount) {
                continue;
            }
            currentRow = 0;
            repeatedOffset += rowLength;
        }
    }

    template <bool UseX1Queue>
    __aicore__ inline void CopyInBf16BorrowedRows(
        uint64_t contiguousOffset, uint64_t rowLength,
        uint64_t paddedRowLength, uint64_t rowCount, bool compactBatch)
    {
        auto& queue = UseX1Queue ? inQueueX1 : inQueueX2;
        AscendC::LocalTensor<TYPE_X1> raw =
            queue.template AllocTensor<TYPE_X1>();
        if (compactBatch) {
            if (tailIndex == 1) {
                CopyContiguousLength(raw, x2Gm, contiguousOffset, rowCount * rowLength);
            } else {
                CopyContiguousLength(raw, x1Gm, contiguousOffset, rowCount * rowLength);
            }
            queue.EnQue(raw);
            return;
        }
        const uint64_t rowBytes = rowLength * sizeof(TYPE_X1);
        const uint64_t alignedRowBytes =
            (rowBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(rowCount);
        params.blockLen = static_cast<uint16_t>(rowBytes);
        params.srcGap = 0;
        params.dstGap = static_cast<uint16_t>(
            (paddedRowLength * sizeof(TYPE_X1) - alignedRowBytes) / DATA_BLOCK_BYTES);
        if (tailIndex == 1) {
            AscendC::DataCopyPad(raw, x2Gm[contiguousOffset], params, {false, 0, 0, 0});
        } else {
            AscendC::DataCopyPad(raw, x1Gm[contiguousOffset], params, {false, 0, 0, 0});
        }
        queue.EnQue(raw);
    }

    template <bool UseX1Queue>
    __aicore__ inline void ComputeBf16BorrowedRows()
    {
        auto& queue = UseX1Queue ? inQueueX1 : inQueueX2;
        AscendC::LocalTensor<TYPE_X1> raw = queue.template DeQue<TYPE_X1>();
        AscendC::LocalTensor<float> x1Fp32 = calc_buf_2.Get<float>();
        AscendC::LocalTensor<float> x2Fp32 = calc_buf_3.Get<float>();
        AscendC::LocalTensor<float> streamed = tailIndex == 1 ? x2Fp32 : x1Fp32;
        AscendC::Cast(streamed, raw, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
        // BF16 input has no consumers after Cast. Its V_MTE2 completion
        // token can now release the next DMA while Vector materializes y.
        queue.FreeTensor(raw);
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
        AscendC::PipeBarrier<PIPE_V>();  // streamed Cast -> Compare RAW
        AscendC::Compare(yLocal, x1Fp32, x2Fp32, AscendC::CMPMODE::GT, processDataNumComputes);
        AscendC::PipeBarrier<PIPE_V>();  // Compare retires FP32; predicate RAW
        // The expanded operand remains read-only. Reuse only the retired
        // streamed plane: N half results followed by a disjoint 128-half one.
        constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
        AscendC::LocalTensor<half> result = streamed.template ReinterpretCast<half>();
        AscendC::LocalTensor<half> oneBlock = result[processDataNumComputes];
        AscendC::Duplicate<half>(oneBlock, (half)1, halfRepeatElements);
        AscendC::PipeBarrier<PIPE_V>();  // oneBlock -> Select RAW
        const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
        AscendC::Select(
            result, yLocal, oneBlock, (half)0,
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, halfRepeatElements,
            static_cast<uint8_t>(processDataNumComputes / halfRepeatElements), repeatParams);
        AscendC::PipeBarrier<PIPE_V>();  // Select -> Cast RAW; predicate WAR
        AscendC::Cast(yLocal, result, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
        outQueueY.template EnQue<int8_t>(yLocal);
    }

    __aicore__ inline void CacheBf16RepeatedRow(uint64_t repeatedOffset)
    {
        if (tailIndex == 1) {
            AscendC::LocalTensor<TYPE_X1> x1Local =
                inQueueX1.template AllocTensor<TYPE_X1>();
            CopyContiguousLength(x1Local, x1Gm, repeatedOffset, processDataNum);
            inQueueX1.EnQue(x1Local);
            x1Local = inQueueX1.template DeQue<TYPE_X1>();
            AscendC::LocalTensor<float> x1Fp32 = BF16_REPEATED_ROW_BATCH ?
                calc_buf_3.Get<float>() : calc_buf_2.Get<float>();
            AscendC::Cast(
                x1Fp32, x1Local, AscendC::RoundMode::CAST_NONE,
                processDataNumComputes);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueX1.FreeTensor(x1Local);
        } else {
            AscendC::LocalTensor<TYPE_X2> x2Local =
                inQueueX2.template AllocTensor<TYPE_X2>();
            CopyContiguousLength(x2Local, x2Gm, repeatedOffset, processDataNum);
            inQueueX2.EnQue(x2Local);
            x2Local = inQueueX2.template DeQue<TYPE_X2>();
            AscendC::LocalTensor<float> x2Fp32 = BF16_REPEATED_ROW_BATCH ?
                calc_buf_2.Get<float>() : calc_buf_3.Get<float>();
            AscendC::Cast(
                x2Fp32, x2Local, AscendC::RoundMode::CAST_NONE,
                processDataNumComputes);
            AscendC::PipeBarrier<PIPE_V>();
            inQueueX2.FreeTensor(x2Local);
        }
    }

    __aicore__ inline void ExpandBf16RepeatedRow(
        uint64_t rowLength, uint64_t paddedRowLength, uint64_t rowCount, bool compactBatch)
    {
        AscendC::LocalTensor<float> repeated = tailIndex == 1 ?
            calc_buf_2.Get<float>() : calc_buf_3.Get<float>();
        // Source/destination remain disjoint. The caller's PIPE_V retires
        // this temporary source before streaming Cast overwrites it.
        AscendC::LocalTensor<float> cached = tailIndex == 1 ?
            calc_buf_3.Get<float>() : calc_buf_2.Get<float>();
        AscendC::LocalTensor<uint8_t> scratch = calc_buf_1.Get<uint8_t>();
        BroadcastAdjacentRowLocal(
            repeated, cached, static_cast<uint32_t>(paddedRowLength),
            static_cast<uint32_t>(rowCount), scratch);
        if (compactBatch && rowLength != paddedRowLength) {
            AscendC::PipeBarrier<PIPE_V>();  // Broadcast -> compaction RAW
            const AscendC::GatherMaskParams gatherParams(
                1, static_cast<uint16_t>(rowCount),
                static_cast<uint16_t>(paddedRowLength * sizeof(float) / DATA_BLOCK_BYTES), 0);
            uint64_t gatheredCount = 0;
            AscendC::GatherMask<float>(
                repeated, repeated, 7, true, static_cast<uint32_t>(rowLength),
                gatherParams, gatheredCount);
        }
    }

    __aicore__ inline void CopyInBf16ContiguousRowsPadded(
        uint64_t contiguousOffset, uint64_t rowLength,
        uint64_t paddedRowLength, uint64_t rowCount)
    {
        const uint64_t rowBytes = rowLength * sizeof(TYPE_X1);
        const uint64_t alignedRowBytes =
            (rowBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES *
            DATA_BLOCK_BYTES;
        const uint64_t paddedRowBytes = paddedRowLength * sizeof(TYPE_X1);
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(rowCount);
        copyParams.blockLen = static_cast<uint16_t>(rowBytes);
        copyParams.srcGap = 0;
        copyParams.dstGap = static_cast<uint16_t>(
            (paddedRowBytes - alignedRowBytes) / DATA_BLOCK_BYTES);
        if (tailIndex == 1) {
            AscendC::LocalTensor<TYPE_X2> x2Local =
                inQueueX2.template AllocTensor<TYPE_X2>();
            AscendC::DataCopyPad(
                x2Local, x2Gm[contiguousOffset], copyParams,
                {false, 0, 0, 0});
            inQueueX2.EnQue(x2Local);
        } else {
            AscendC::LocalTensor<TYPE_X1> x1Local =
                inQueueX1.template AllocTensor<TYPE_X1>();
            AscendC::DataCopyPad(
                x1Local, x1Gm[contiguousOffset], copyParams,
                {false, 0, 0, 0});
            inQueueX1.EnQue(x1Local);
        }
    }

    __aicore__ inline void CopyOutBf16PaddedRows(
        uint64_t outputOffset, uint64_t rowLength, uint64_t rowCount)
    {
        AscendC::LocalTensor<int8_t> yLocal =
            outQueueY.template DeQue<int8_t>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(rowCount);
        copyParams.blockLen = static_cast<uint16_t>(rowLength);
        copyParams.srcGap = 0;
        copyParams.dstGap = 0;
        AscendC::DataCopyPad(yGm[outputOffset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyInBf16ContiguousRow(uint64_t contiguousOffset)
    {
        if (tailIndex == 1) {
            AscendC::LocalTensor<TYPE_X2> x2Local =
                inQueueX2.template AllocTensor<TYPE_X2>();
            CopyContiguousLength(
                x2Local, x2Gm, contiguousOffset, processDataNum);
            inQueueX2.EnQue(x2Local);
        } else {
            AscendC::LocalTensor<TYPE_X1> x1Local =
                inQueueX1.template AllocTensor<TYPE_X1>();
            CopyContiguousLength(
                x1Local, x1Gm, contiguousOffset, processDataNum);
            inQueueX1.EnQue(x1Local);
        }
    }

    __aicore__ inline void ComputeBf16WithCachedRow()
    {
        AscendC::LocalTensor<int8_t> yLocal =
            outQueueY.template AllocTensor<int8_t>();
        AscendC::LocalTensor<float> x1Fp32 = calc_buf_2.Get<float>();
        AscendC::LocalTensor<float> x2Fp32 = calc_buf_3.Get<float>();
        AscendC::LocalTensor<half> yFp16;
        AscendC::LocalTensor<half> oneBlock;
        if (tailIndex == 1) {
            AscendC::LocalTensor<TYPE_X2> x2Local =
                inQueueX2.template DeQue<TYPE_X2>();
            AscendC::Cast(
                x2Fp32, x2Local, AscendC::RoundMode::CAST_NONE,
                processDataNumComputes);
            AscendC::Compare(
                yLocal, x1Fp32, x2Fp32, AscendC::CMPMODE::GT,
                processDataNumComputes);
            yFp16 = x2Local.template ReinterpretCast<half>();
            oneBlock = x2Fp32.template ReinterpretCast<half>();
            MaterializeBf16CachedPredicate(yLocal, yFp16, oneBlock);
            inQueueX2.FreeTensor(x2Local);
        } else {
            AscendC::LocalTensor<TYPE_X1> x1Local =
                inQueueX1.template DeQue<TYPE_X1>();
            AscendC::Cast(
                x1Fp32, x1Local, AscendC::RoundMode::CAST_NONE,
                processDataNumComputes);
            AscendC::Compare(
                yLocal, x1Fp32, x2Fp32, AscendC::CMPMODE::GT,
                processDataNumComputes);
            yFp16 = x1Local.template ReinterpretCast<half>();
            oneBlock = x1Fp32.template ReinterpretCast<half>();
            MaterializeBf16CachedPredicate(yLocal, yFp16, oneBlock);
            inQueueX1.FreeTensor(x1Local);
        }
        outQueueY.template EnQue<int8_t>(yLocal);
    }

    __aicore__ inline void MaterializeBf16CachedPredicate(
        AscendC::LocalTensor<int8_t> predicate,
        AscendC::LocalTensor<half> result,
        AscendC::LocalTensor<half> oneBlock)
    {
        constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
        AscendC::Duplicate<half>(oneBlock, (half)1, halfRepeatElements);
        const uint8_t repeatTimes = static_cast<uint8_t>(
            processDataNumComputes / halfRepeatElements);
        const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
        AscendC::Select(
            result, predicate, oneBlock, (half)0,
            AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
            halfRepeatElements, repeatTimes, repeatParams);
        AscendC::Cast(
            predicate, result, AscendC::RoundMode::CAST_NONE,
            processDataNumComputes);
    }

    template <bool NarrowRowCacheCompletion>
    __aicore__ inline void ProcessInt32PaddedRowsImpl()
    {
        const uint64_t rowLength = outShape[tailIndex];
        const bool x1Rows = x1Stride[tailIndex] == 1 && x1Stride[rowIndex] == 0;
        uint64_t outputOffset = blockOffset;
        uint64_t remaining = coreDataNum;
        while (remaining >= rowLength) {
            const uint64_t currentRow = outputOffset / rowLength % outShape[rowIndex];
            uint64_t planeRows = outShape[rowIndex] - currentRow;
            const uint64_t remainingRows = remaining / rowLength;
            planeRows = planeRows < remainingRows ? planeRows : remainingRows;
            const uint64_t broadcastOffset = CalcInputOffset(
                outputOffset, x1Rows ? x1Stride : x2Stride);
            AscendC::LocalTensor<TYPE_X1> broadcastRow =
                broadcastRowQueue.template AllocTensor<TYPE_X1>();
            if (x1Rows) {
                CopyContiguousLength(
                    broadcastRow, x1Gm, broadcastOffset, outShape[tailIndex]);
            } else {
                CopyContiguousLength(
                    broadcastRow, x2Gm, broadcastOffset, outShape[tailIndex]);
            }
            broadcastRowQueue.EnQue(broadcastRow);
            broadcastRow = broadcastRowQueue.template DeQue<TYPE_X1>();
            while (planeRows > 0) {
                const uint64_t rowCount = planeRows < maxPaddedRowCount ?
                    planeRows : maxPaddedRowCount;
                processDataNum = rowCount * paddedRowLength;
                SetComputeDataNum();
                CopyInInt32PaddedRows(outputOffset, rowCount, broadcastRow, x1Rows);
                Compute();
                CopyOutInt32PaddedRows(outputOffset, rowCount);
                const uint64_t logicalLength = rowCount * rowLength;
                outputOffset += logicalLength;
                remaining -= logicalLength;
                planeRows -= rowCount;
            }
            if constexpr (
                NarrowRowCacheCompletion && std::is_same_v<TYPE_X1, int32_t>) {
                // The cached row is consumed only by the vector Broadcast
                // above. For workloads with several natural groups per AIV,
                // avoid serializing an unrelated result write before loading
                // the next group.
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            broadcastRowQueue.FreeTensor(broadcastRow);
        }
    }

    template <bool NarrowRowCacheCompletion>
    __aicore__ inline void ProcessInt32CompactRowsImpl()
    {
        const uint64_t rowLength = outShape[tailIndex];
        const uint64_t maxCompactRowCount = maxPaddedRowCount;
        const bool x1Rows = x1Stride[tailIndex] == 1 && x1Stride[rowIndex] == 0;
        uint64_t outputOffset = blockOffset;
        uint64_t remaining = coreDataNum;
        while (remaining >= rowLength) {
            const uint64_t currentRow = outputOffset / rowLength % outShape[rowIndex];
            uint64_t planeRows = outShape[rowIndex] - currentRow;
            const uint64_t remainingRows = remaining / rowLength;
            planeRows = planeRows < remainingRows ? planeRows : remainingRows;
            const uint64_t broadcastOffset = CalcInputOffset(
                outputOffset, x1Rows ? x1Stride : x2Stride);
            AscendC::LocalTensor<TYPE_X1> broadcastRow =
                broadcastRowQueue.template AllocTensor<TYPE_X1>();
            if (x1Rows) {
                CopyContiguousLength(
                    broadcastRow, x1Gm, broadcastOffset, rowLength);
            } else {
                CopyContiguousLength(
                    broadcastRow, x2Gm, broadcastOffset, rowLength);
            }
            broadcastRowQueue.EnQue(broadcastRow);
            broadcastRow = broadcastRowQueue.template DeQue<TYPE_X1>();
            while (planeRows > 0) {
                const uint64_t rowCount = planeRows < maxCompactRowCount ?
                    planeRows : maxCompactRowCount;
                processDataNum = rowCount * rowLength;
                SetComputeDataNum();
                CopyInInt32CompactRows(
                    outputOffset, rowCount, broadcastRow, x1Rows);
                Compute();
                CopyOut(outputOffset);
                outputOffset += processDataNum;
                remaining -= processDataNum;
                planeRows -= rowCount;
            }
            if constexpr (NarrowRowCacheCompletion) {
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                AscendC::PipeBarrier<PIPE_ALL>();
            }
            broadcastRowQueue.FreeTensor(broadcastRow);
        }
    }

    __aicore__ inline void ProcessInt32PaddedRows()
    {
        if constexpr (std::is_same_v<TYPE_X1, int32_t>) {
            InitInt32PaddedPredicateConstants();
            uint64_t naturalGroupCount = 1;
            for (uint32_t index = 0; index < rowIndex; ++index) {
                naturalGroupCount *= outShape[index];
            }
            if (paddedRowLength != outShape[tailIndex] &&
                outShape[rowIndex] >= INT32_COMPACT_MIN_ROW_COUNT) {
                if (naturalGroupCount >=
                    static_cast<uint64_t>(AscendC::GetBlockNum()) * 2) {
                    ProcessInt32CompactRowsImpl<true>();
                } else {
                    ProcessInt32CompactRowsImpl<false>();
                }
                return;
            }
            if (naturalGroupCount >=
                static_cast<uint64_t>(AscendC::GetBlockNum()) * 2) {
                ProcessInt32PaddedRowsImpl<true>();
                return;
            }
        }
        ProcessInt32PaddedRowsImpl<false>();
    }

    __aicore__ inline void InitInt32PaddedPredicateConstants()
    {
        constexpr uint32_t halfRepeatElements =
            VECTOR_ALIGN_BYTES / sizeof(half);
        AscendC::LocalTensor<half> zeroBlock =
            calc_buf_1.Get<half>()[ubPartDataNum];
        AscendC::Duplicate<half>(
            zeroBlock, (half)1, halfRepeatElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetCmpMask(zeroBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<half>(
            zeroBlock, (half)0, halfRepeatElements);
    }

    __aicore__ inline void InitInt8PredicateConstants()
    {
        constexpr uint32_t halfRepeatElements =
            VECTOR_ALIGN_BYTES / sizeof(half);
        const uint64_t segmentCapacity = ubPartDataNum / 2;
        AscendC::LocalTensor<half> oneBlock =
            calc_buf_2.Get<half>()[segmentCapacity];
        AscendC::Duplicate<half>(
            oneBlock, (half)0, halfRepeatElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetCmpMask(oneBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<half>(
            oneBlock, (half)1, halfRepeatElements);
    }

    __aicore__ inline void ProcessAdjacentPaddedPlanes()
    {
        const uint64_t planeLength = outShape[rowIndex] * outShape[tailIndex];
        uint64_t outputOffset = blockOffset;
        uint64_t remaining = coreDataNum;
        while (remaining >= planeLength) {
            const uint64_t planeIndex = outputOffset / planeLength;
            const uint64_t planesToWrap =
                outShape[planeGroupIndex] - planeIndex % outShape[planeGroupIndex];
            const uint64_t remainingPlanes = remaining / planeLength;
            uint64_t planeCount = remainingPlanes < maxPaddedPlaneCount ?
                remainingPlanes : maxPaddedPlaneCount;
            planeCount = planeCount < planesToWrap ? planeCount : planesToWrap;
            processDataNum = planeCount * paddedPlaneLength;
            SetComputeDataNum();
            CopyInAdjacentPaddedPlanes(outputOffset, planeCount);
            if constexpr (
                std::is_same_v<TYPE_X1, half> || std::is_same_v<TYPE_X1, bfloat16_t>) {
                processDataNum = planeCount * planeLength;
                SetComputeDataNum();
            }
            Compute();
            CopyOutAdjacentPaddedPlanes(outputOffset, planeCount);
            outputOffset += planeCount * planeLength;
            remaining -= planeCount * planeLength;
        }
    }

    __aicore__ inline void ProcessAdjacentScalarBlocks()
    {
        const uint64_t planeLength = outShape[2] * outShape[3];
        if constexpr (AdjacentSingleTile) {
            processDataNum = coreDataNum;
            SetComputeDataNum();
            CopyInAdjacentScalarBlocks(blockOffset, planeLength);
            Compute();
            CopyOut(blockOffset);
            return;
        }
        uint64_t outputOffset = blockOffset;
        uint64_t remaining = coreDataNum;
        while (remaining > 0) {
            uint64_t validLength = remaining > ubPartDataNum ? ubPartDataNum : remaining;
            if (remaining > validLength) {
                validLength = validLength / planeLength * planeLength;
            }
            if (validLength == 0) {
                return;
            }
            processDataNum = validLength;
            SetComputeDataNum();
            CopyInAdjacentScalarBlocks(outputOffset, planeLength);
            Compute();
            CopyOut(outputOffset);
            outputOffset += validLength;
            remaining -= validLength;
        }
    }

    __aicore__ inline uint64_t CalcInputOffset(uint64_t outputIndex, const uint64_t* stride)
    {
        uint64_t inputOffset = 0;
        uint64_t remaining = outputIndex;
        for (int32_t index = static_cast<int32_t>(rank) - 1; index >= 0; --index) {
            const uint32_t dimIndex = static_cast<uint32_t>(index);
            const uint64_t dim = outShape[dimIndex];
            const uint64_t coordinate = dim == 0 ? 0 : remaining % dim;
            remaining = dim == 0 ? 0 : remaining / dim;
            inputOffset += coordinate * stride[dimIndex];
        }
        return inputOffset;
    }

    __aicore__ inline void InitOutputCoord(uint64_t outputIndex, uint64_t* coordinate)
    {
        uint64_t remaining = outputIndex;
        for (int32_t index = static_cast<int32_t>(rank) - 1; index >= 0; --index) {
            const uint32_t dimIndex = static_cast<uint32_t>(index);
            const uint64_t dim = outShape[dimIndex];
            coordinate[dimIndex] = dim == 0 ? 0 : remaining % dim;
            remaining = dim == 0 ? 0 : remaining / dim;
        }
    }

    __aicore__ inline void AdvanceTailSegment(uint64_t* coordinate, uint64_t& x1Offset, uint64_t& x2Offset)
    {
        for (int32_t index = static_cast<int32_t>(rank) - 2; index >= 0; --index) {
            const uint32_t dimIndex = static_cast<uint32_t>(index);
            ++coordinate[dimIndex];
            x1Offset += x1Stride[dimIndex];
            x2Offset += x2Stride[dimIndex];
            if (coordinate[dimIndex] < outShape[dimIndex]) {
                return;
            }
            coordinate[dimIndex] = 0;
            x1Offset -= outShape[dimIndex] * x1Stride[dimIndex];
            x2Offset -= outShape[dimIndex] * x2Stride[dimIndex];
        }
    }

    __aicore__ inline void AdvanceAdjacentPlane4(
        uint64_t* coordinate, uint64_t& x1Offset, uint64_t& x2Offset)
    {
        ++coordinate[1];
        x1Offset += x1Stride[1];
        x2Offset += x2Stride[1];
        if (coordinate[1] < outShape[1]) {
            return;
        }
        coordinate[1] = 0;
        x1Offset -= outShape[1] * x1Stride[1];
        x2Offset -= outShape[1] * x2Stride[1];
        ++coordinate[0];
        x1Offset += x1Stride[0];
        x2Offset += x2Stride[0];
        if (coordinate[0] == outShape[0]) {
            coordinate[0] = 0;
            x1Offset -= outShape[0] * x1Stride[0];
            x2Offset -= outShape[0] * x2Stride[0];
        }
    }

    __aicore__ inline void WrapAdjacentPlaneGroup4(
        uint64_t* coordinate, uint64_t& x1Offset, uint64_t& x2Offset)
    {
        const uint64_t firstPlane = coordinate[1];
        coordinate[1] = 0;
        x1Offset += x1Stride[0] - firstPlane * x1Stride[1];
        x2Offset += x2Stride[0] - firstPlane * x2Stride[1];
        ++coordinate[0];
        if (coordinate[0] == outShape[0]) {
            coordinate[0] = 0;
            x1Offset -= outShape[0] * x1Stride[0];
            x2Offset -= outShape[0] * x2Stride[0];
        }
    }

    __aicore__ inline bool IsOuterRowBroadcast(const uint64_t* stride)
    {
        if (rank == 0 || stride[rank - 1] != 1) {
            return false;
        }
        for (uint32_t index = 0; index + 1 < rank; ++index) {
            if (stride[index] != 0) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool IsOutputContiguous(const uint64_t* stride)
    {
        uint64_t expected = 1;
        for (int32_t index = static_cast<int32_t>(rank) - 1; index >= 0; --index) {
            const uint32_t dimIndex = static_cast<uint32_t>(index);
            if (stride[dimIndex] != expected) {
                return false;
            }
            expected *= outShape[dimIndex];
        }
        return rank != 0;
    }

    __aicore__ inline bool IsRepeatedRowBroadcast(
        const uint64_t* repeatedStride, const uint64_t* contiguousStride)
    {
        if (rank < 2 || repeatedStride[rank - 1] != 1 ||
            repeatedStride[rank - 2] != 0 || !IsOutputContiguous(contiguousStride)) {
            return false;
        }
        uint64_t expected = outShape[rank - 1];
        for (int32_t index = static_cast<int32_t>(rank) - 3; index >= 0; --index) {
            const uint32_t dimIndex = static_cast<uint32_t>(index);
            if (repeatedStride[dimIndex] != expected) {
                return false;
            }
            expected *= outShape[dimIndex];
        }
        return true;
    }

    __aicore__ inline bool IsOuterBroadcast(const uint64_t* broadcastStride, const uint64_t* contiguousStride)
    {
        return IsOuterRowBroadcast(broadcastStride) && IsOutputContiguous(contiguousStride);
    }

    template <typename U>
    __aicore__ inline void CopyContiguousLength(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source, uint64_t sourceOffset,
        uint64_t length)
    {
        if constexpr (
            std::is_same_v<U, float> &&
            BroadcastMode == GREATER_BROADCAST_CONTIGUOUS) {
            constexpr uint64_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(U);
            const uint64_t alignedLength = length / elementsPerBlock * elementsPerBlock;
            if (alignedLength != 0) {
                AscendC::DataCopy(destination, source[sourceOffset], alignedLength);
            }
            if (alignedLength == length) {
                return;
            }
            AscendC::DataCopyParams tailCopyParams;
            tailCopyParams.blockCount = 1;
            tailCopyParams.blockLen = (length - alignedLength) * sizeof(U);
            tailCopyParams.srcStride = 0;
            tailCopyParams.dstStride = 0;
            AscendC::DataCopyPad(
                destination[alignedLength], source[sourceOffset + alignedLength],
                tailCopyParams, {false, 0, 0, 0});
            return;
        }
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = length * sizeof(U);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(destination, source[sourceOffset], copyParams, {false, 0, 0, 0});
    }

    template <typename U>
    __aicore__ inline void CopyContiguous(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source, uint64_t sourceOffset)
    {
        CopyContiguousLength(destination, source, sourceOffset, processDataNum);
    }

    template <typename U>
    __aicore__ inline void BroadcastScalar(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source, uint64_t sourceOffset,
        uint64_t fillLength, AscendC::TQue<AscendC::QuePosition::VECIN, 1>& scalarQueue)
    {
        if constexpr (std::is_same_v<U, bfloat16_t> &&
            (BroadcastMode == GREATER_BROADCAST_X1_SCALAR || BroadcastMode == GREATER_BROADCAST_X2_SCALAR)) {
            CopyContiguousLength(destination, source, sourceOffset, 1);
            return;
        }
        AscendC::LocalTensor<U> scalar = scalarQueue.AllocTensor<U>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = sizeof(U);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(scalar, source[sourceOffset], copyParams, {false, 0, 0, 0});
        scalarQueue.EnQue(scalar);
        scalar = scalarQueue.DeQue<U>();
        const U scalarValue = scalar.GetValue(0);
        if constexpr (BroadcastMode == GREATER_BROADCAST_X2_SCALAR && std::is_same_v<U, int32_t>) {
            x2ScalarValue = scalarValue;
        } else if constexpr (std::is_same_v<U, int8_t> || std::is_same_v<U, uint8_t>) {
            AscendC::LocalTensor<half> scalarHalf = calc_buf_1.Get<half>();
            AscendC::Duplicate(scalarHalf, static_cast<half>(scalarValue), fillLength);
            AscendC::Cast(destination, scalarHalf, AscendC::RoundMode::CAST_NONE, fillLength);
        } else {
            AscendC::Duplicate(destination, scalarValue, fillLength);
        }
        scalarQueue.FreeTensor(scalar);
    }

    __aicore__ inline void CopyIn(uint64_t outputOffset)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template AllocTensor<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template AllocTensor<TYPE_X2>();
        uint64_t x1Offset = outputOffset;
        uint64_t x2Offset = outputOffset;
        bool x1Scalar = false;
        bool x2Scalar = false;
        if constexpr (BroadcastMode == GREATER_BROADCAST_X1_SCALAR) {
            x1Offset = 0;
            x1Scalar = true;
        } else if constexpr (BroadcastMode == GREATER_BROADCAST_X2_SCALAR) {
            x2Offset = 0;
            x2Scalar = true;
        } else if constexpr (BroadcastMode == GREATER_BROADCAST_GENERAL) {
            if constexpr (std::is_same_v<TYPE_X1, bfloat16_t>) {
                if (tailIndex != 0) {
                    const uint64_t rowLength = outShape[rank - 1];
                    const uint64_t planeLength = outShape[rank - 2] * rowLength;
                    const uint64_t repeatedOffset = outputOffset / planeLength * rowLength +
                        outputOffset % rowLength;
                    x1Offset = tailIndex == 1 ? repeatedOffset : outputOffset;
                    x2Offset = tailIndex == 2 ? repeatedOffset : outputOffset;
                } else {
                    x1Offset = CalcInputOffset(outputOffset, x1Stride);
                    x2Offset = CalcInputOffset(outputOffset, x2Stride);
                }
            } else {
                x1Offset = CalcInputOffset(outputOffset, x1Stride);
                x2Offset = CalcInputOffset(outputOffset, x2Stride);
            }
            x1Scalar = rank == 0 || x1Stride[rank - 1] == 0;
            x2Scalar = rank == 0 || x2Stride[rank - 1] == 0;
        }

        if (x1Scalar) {
            BroadcastScalar(x1Local, x1Gm, x1Offset, processDataNumComputes, x1ScalarQueue);
        } else {
            CopyContiguous(x1Local, x1Gm, x1Offset);
        }
        if (x2Scalar) {
            BroadcastScalar(x2Local, x2Gm, x2Offset, processDataNumComputes, x2ScalarQueue);
        } else {
            CopyContiguous(x2Local, x2Gm, x2Offset);
        }
        if constexpr (!((
            ReuseInt8Predicate &&
            BroadcastMode == GREATER_BROADCAST_CONTIGUOUS) ||
            FP32_STEADY_WAVE_MAJOR)) {
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    template <typename U>
    __aicore__ inline void CopyRepeatedRows(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source, uint64_t sourceOffset,
        uint64_t rowLength, uint64_t rowCount)
    {
        AscendC::LocalTensor<U> row = broadcastRowQueue.AllocTensor<U>();
        CopyContiguousLength(row, source, sourceOffset, rowLength);
        broadcastRowQueue.EnQue(row);
        row = broadcastRowQueue.DeQue<U>();
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint64_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
            AscendC::DataCopy(destination[rowIndex * rowLength], row, rowLength);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        broadcastRowQueue.FreeTensor(row);
    }

    template <typename U>
    __aicore__ inline void BroadcastAdjacentScalarsLocal(
        AscendC::LocalTensor<U> destination, AscendC::LocalTensor<U> scalars,
        uint32_t rowLength, uint32_t rowCount)
    {
        const uint32_t destinationShape[2] = {rowCount, rowLength};
        const uint32_t sourceShape[2] = {rowCount, 1};
        AscendC::Broadcast<U, 2, 1>(destination, scalars, destinationShape, sourceShape);
    }

    template <typename U>
    __aicore__ inline void ExpandAdjacentScalarsToBlocks(
        AscendC::LocalTensor<U> expanded, AscendC::LocalTensor<U> scalars,
        AscendC::LocalTensor<U> zeroBlock, uint32_t rowCount)
    {
        constexpr uint32_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(U);
        if constexpr (!std::is_same_v<U, half>) {
            if (rowCount > 1) {
                AscendC::Duplicate(
                    zeroBlock.template ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0),
                    DATA_BLOCK_BYTES / sizeof(uint16_t));
            }
        }
        AscendC::BrcbToOneBlock(scalars, rowCount, elementsPerBlock, expanded);
    }

    template <typename U>
    __aicore__ inline void BroadcastAdjacentScalarBlockPlanes(
        AscendC::LocalTensor<U> destination, AscendC::LocalTensor<U> expanded,
        uint32_t rowLength, uint32_t rowCount, uint64_t planeLength, uint64_t planeCount,
        bool complete)
    {
        constexpr uint32_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(U);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<U, AscendC::MaskMode::COUNTER>(rowLength);
        const AscendC::CopyRepeatParams repeatParams = {
            1, 0, static_cast<uint16_t>(rowLength / elementsPerBlock), 1};
        for (uint64_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
            AscendC::Copy<U, false>(
                destination[planeIndex * planeLength], expanded, MASK_PLACEHOLDER,
                static_cast<uint8_t>(rowCount), repeatParams);
        }
        if (complete) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

    template <typename U>
    __aicore__ inline void BroadcastAdjacentRowPlanesLocal(
        AscendC::LocalTensor<U> planes, AscendC::LocalTensor<U> zeroBlock,
        uint32_t rowLength, uint32_t rowCount, uint64_t planeLength, uint64_t planeCount,
        bool complete)
    {
        constexpr uint32_t rowsPerOr = 8;
        constexpr uint32_t uint16ElementsPerBlock = DATA_BLOCK_BYTES / sizeof(uint16_t);
        constexpr uint32_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(U);
        constexpr uint32_t dtypeCount = sizeof(U) == sizeof(float) ? 2 : 1;
        const uint32_t broadcastRows = rowCount - 1;
        const uint32_t fullOrCount = broadcastRows / rowsPerOr;
        const uint32_t tailRows = broadcastRows - fullOrCount * rowsPerOr;
        const uint8_t repeatTimes = static_cast<uint8_t>(rowLength / elementsPerBlock);
        const uint8_t destinationBlockStride = static_cast<uint8_t>(
            rowLength * dtypeCount / uint16ElementsPerBlock);
        const AscendC::CopyRepeatParams repeatParams = {
            destinationBlockStride, 0, 1, 1};
        const AscendC::BinaryRepeatParams binaryRepeatParams(
            destinationBlockStride, 0, 0, 1, 1, 0);
        if (fullOrCount > 0) {
            AscendC::SetMaskNorm();
            AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::NORMAL>(
                rowsPerOr * uint16ElementsPerBlock);
            for (uint64_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
                const uint64_t planeOffset = planeIndex * planeLength;
                for (uint32_t chunkIndex = 0; chunkIndex < fullOrCount; ++chunkIndex) {
                    const uint64_t destinationOffset = planeOffset + rowLength +
                        static_cast<uint64_t>(chunkIndex) * rowsPerOr * rowLength;
                    if constexpr (std::is_same_v<U, half>) {
                        AscendC::Copy<uint16_t, false>(
                            planes[destinationOffset].template ReinterpretCast<uint16_t>(),
                            planes[planeOffset].template ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER,
                            repeatTimes, repeatParams);
                    } else {
                        AscendC::Or<uint16_t, false>(
                            planes[destinationOffset].template ReinterpretCast<uint16_t>(),
                            planes[planeOffset].template ReinterpretCast<uint16_t>(),
                            zeroBlock.template ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER,
                            repeatTimes, binaryRepeatParams);
                    }
                }
            }
        }
        if (tailRows > 0) {
            AscendC::SetMaskNorm();
            AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::NORMAL>(
                tailRows * uint16ElementsPerBlock);
            const uint64_t tailOffset = rowLength +
                static_cast<uint64_t>(fullOrCount) * rowsPerOr * rowLength;
            for (uint64_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
                const uint64_t planeOffset = planeIndex * planeLength;
                if constexpr (std::is_same_v<U, half>) {
                    AscendC::Copy<uint16_t, false>(
                        planes[planeOffset + tailOffset].template ReinterpretCast<uint16_t>(),
                        planes[planeOffset].template ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER,
                        repeatTimes, repeatParams);
                } else {
                    AscendC::Or<uint16_t, false>(
                        planes[planeOffset + tailOffset].template ReinterpretCast<uint16_t>(),
                        planes[planeOffset].template ReinterpretCast<uint16_t>(),
                        zeroBlock.template ReinterpretCast<uint16_t>(), MASK_PLACEHOLDER,
                        repeatTimes, binaryRepeatParams);
                }
            }
        }
        if (complete) {
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

    template <typename U>
    __aicore__ inline void BroadcastAdjacentScalars(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source, uint64_t sourceOffset,
        uint32_t rowLength, uint32_t rowCount)
    {
        AscendC::LocalTensor<U> scalars = broadcastRowQueue.AllocTensor<U>();
        CopyContiguousLength(scalars, source, sourceOffset, rowCount);
        broadcastRowQueue.EnQue(scalars);
        scalars = broadcastRowQueue.DeQue<U>();
        BroadcastAdjacentScalarsLocal(destination, scalars, rowLength, rowCount);
        AscendC::PipeBarrier<PIPE_ALL>();
        broadcastRowQueue.FreeTensor(scalars);
    }

    template <typename U>
    __aicore__ inline void BroadcastAdjacentRow(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source, uint64_t sourceOffset,
        uint32_t rowLength, uint32_t rowCount)
    {
        AscendC::LocalTensor<U> row = broadcastRowQueue.AllocTensor<U>();
        CopyContiguousLength(row, source, sourceOffset, rowLength);
        broadcastRowQueue.EnQue(row);
        row = broadcastRowQueue.DeQue<U>();
        const uint32_t destinationShape[2] = {rowCount, rowLength};
        const uint32_t sourceShape[2] = {1, rowLength};
        AscendC::Broadcast<U, 2, 0>(destination, row, destinationShape, sourceShape);
        AscendC::PipeBarrier<PIPE_ALL>();
        broadcastRowQueue.FreeTensor(row);
    }

    template <typename U>
    __aicore__ inline void BroadcastAdjacentRowLocal(
        AscendC::LocalTensor<U> destination, AscendC::LocalTensor<U> row,
        uint32_t rowLength, uint32_t rowCount)
    {
        const uint32_t destinationShape[2] = {rowCount, rowLength};
        const uint32_t sourceShape[2] = {1, rowLength};
        AscendC::Broadcast<U, 2, 0>(destination, row, destinationShape, sourceShape);
    }

    template <typename U>
    __aicore__ inline void BroadcastAdjacentRowLocal(
        AscendC::LocalTensor<U> destination, AscendC::LocalTensor<U> row,
        uint32_t rowLength, uint32_t rowCount,
        AscendC::LocalTensor<uint8_t>& scratch)
    {
        const uint32_t destinationShape[2] = {rowCount, rowLength};
        const uint32_t sourceShape[2] = {1, rowLength};
        AscendC::Broadcast<U, 2, 0>(
            destination, row, destinationShape, sourceShape, scratch);
    }

    template <typename U>
    __aicore__ inline void CopyInt32PaddedContiguousRows(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source,
        uint64_t sourceOffset, uint64_t rowCount)
    {
        const uint64_t rowLength = outShape[tailIndex];
        const uint64_t rowBytes = rowLength * sizeof(U);
        const uint64_t alignedRowBytes =
            (rowBytes + DATA_BLOCK_BYTES - 1) / DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        const uint64_t paddedRowBytes = paddedRowLength * sizeof(U);
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(rowCount);
        copyParams.blockLen = static_cast<uint16_t>(rowBytes);
        copyParams.srcGap = 0;
        copyParams.dstGap = static_cast<uint16_t>(
            (paddedRowBytes - alignedRowBytes) / DATA_BLOCK_BYTES);
        AscendC::DataCopyPad(
            destination, source[sourceOffset], copyParams, {false, 0, 0, 0});
    }

    __aicore__ inline void CopyInInt32PaddedRows(
        uint64_t outputOffset, uint64_t rowCount,
        AscendC::LocalTensor<TYPE_X1> broadcastRow, bool x1Rows)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template AllocTensor<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template AllocTensor<TYPE_X2>();
        if (x1Rows) {
            const uint64_t contiguousOffset = CalcInputOffset(outputOffset, x2Stride);
            CopyInt32PaddedContiguousRows(x2Local, x2Gm, contiguousOffset, rowCount);
            BroadcastAdjacentRowLocal(
                x1Local, broadcastRow, static_cast<uint32_t>(paddedRowLength),
                static_cast<uint32_t>(rowCount));
        } else {
            const uint64_t contiguousOffset = CalcInputOffset(outputOffset, x1Stride);
            CopyInt32PaddedContiguousRows(x1Local, x1Gm, contiguousOffset, rowCount);
            BroadcastAdjacentRowLocal(
                x2Local, broadcastRow, static_cast<uint32_t>(paddedRowLength),
                static_cast<uint32_t>(rowCount));
        }
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void CopyInInt32CompactRows(
        uint64_t outputOffset, uint64_t rowCount,
        AscendC::LocalTensor<TYPE_X1> broadcastRow, bool x1Rows)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template AllocTensor<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template AllocTensor<TYPE_X2>();
        const uint64_t rowLength = outShape[tailIndex];
        const uint64_t logicalLength = rowCount * rowLength;
        if (x1Rows) {
            const uint64_t contiguousOffset = CalcInputOffset(outputOffset, x2Stride);
            CopyContiguousLength(x2Local, x2Gm, contiguousOffset, logicalLength);
            BroadcastAdjacentRowLocal(
                x1Local, broadcastRow, static_cast<uint32_t>(paddedRowLength),
                static_cast<uint32_t>(rowCount));
        } else {
            const uint64_t contiguousOffset = CalcInputOffset(outputOffset, x1Stride);
            CopyContiguousLength(x1Local, x1Gm, contiguousOffset, logicalLength);
            BroadcastAdjacentRowLocal(
                x2Local, broadcastRow, static_cast<uint32_t>(paddedRowLength),
                static_cast<uint32_t>(rowCount));
        }
        AscendC::LocalTensor<TYPE_X1> compactBroadcast =
            x1Rows ? x1Local : x2Local;
        AscendC::GatherMaskParams gatherParams(
            1, static_cast<uint16_t>(rowCount),
            static_cast<uint16_t>(
                paddedRowLength * sizeof(TYPE_X1) / DATA_BLOCK_BYTES),
            0);
        uint64_t gatheredCount = 0;
        AscendC::GatherMask<TYPE_X1>(
            compactBroadcast, compactBroadcast, 7, true,
            static_cast<uint32_t>(rowLength),
            gatherParams, gatheredCount);
        AscendC::PipeBarrier<PIPE_V>();
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void CopyOutInt32PaddedRows(uint64_t outputOffset, uint64_t rowCount)
    {
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.template DeQue<int8_t>();
        const uint64_t rowLength = outShape[tailIndex];
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(rowCount);
        copyParams.blockLen = static_cast<uint16_t>(rowLength);
        copyParams.srcGap = 0;
        copyParams.dstGap = 0;
        AscendC::DataCopyPad(yGm[outputOffset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void BroadcastAdjacentHalfScalarsToInt8(
        AscendC::LocalTensor<int8_t> destination, AscendC::LocalTensor<half> scalars,
        uint32_t rowLength, uint32_t rowCount)
    {
        AscendC::LocalTensor<half> expanded = calc_buf_1.Get<half>();
        BroadcastAdjacentScalarsLocal(expanded, scalars, rowLength, rowCount);
        AscendC::Cast(
            destination, expanded, AscendC::RoundMode::CAST_NONE,
            static_cast<uint64_t>(rowLength) * rowCount);
    }

    template <typename U>
    __aicore__ inline void CopyAdjacentScalarPlaneCopies(
        AscendC::LocalTensor<U> planes, uint64_t planeLength, uint64_t planeCount)
    {
        for (uint64_t planeIndex = 1; planeIndex < planeCount; ++planeIndex) {
            AscendC::DataCopy(planes[planeIndex * planeLength], planes, planeLength);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <typename U>
    __aicore__ inline void CopyAdjacentRowsToPlanes(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source, uint64_t sourceOffset,
        uint64_t rowLength, uint64_t planeLength, uint64_t planeCount)
    {
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(planeCount);
        copyParams.blockLen = static_cast<uint16_t>(rowLength * sizeof(U));
        copyParams.srcStride = 0;
        copyParams.dstStride = static_cast<uint16_t>(
            (planeLength - rowLength) * sizeof(U) / DATA_BLOCK_BYTES);
        AscendC::DataCopyPad(destination, source[sourceOffset], copyParams, {false, 0, 0, 0});
    }

    __aicore__ inline void CopyAdjacentBroadcastRows(
        AscendC::LocalTensor<TYPE_X1> x1Local, AscendC::LocalTensor<TYPE_X2> x2Local,
        uint64_t& x1Offset, uint64_t& x2Offset, uint64_t* coordinate, uint64_t& copiedLength)
    {
        const uint64_t rowLength = outShape[rank - 1];
        const uint64_t availableRows = (processDataNum - copiedLength) / rowLength;
        const uint64_t rowsToWrap = outShape[rank - 2] - coordinate[rank - 2];
        const uint32_t rowCount = static_cast<uint32_t>(
            availableRows < rowsToWrap ? availableRows : rowsToWrap);
        const uint32_t rowLength32 = static_cast<uint32_t>(rowLength);
        if (x1Stride[rank - 1] == 0) {
            BroadcastAdjacentScalars(
                x1Local[copiedLength], x1Gm, x1Offset, rowLength32, rowCount);
            BroadcastAdjacentRow(
                x2Local[copiedLength], x2Gm, x2Offset, rowLength32, rowCount);
        } else {
            BroadcastAdjacentRow(
                x1Local[copiedLength], x1Gm, x1Offset, rowLength32, rowCount);
            BroadcastAdjacentScalars(
                x2Local[copiedLength], x2Gm, x2Offset, rowLength32, rowCount);
        }
        copiedLength += static_cast<uint64_t>(rowCount) * rowLength;
        if (copiedLength < processDataNum) {
            const uint32_t adjacentRowIndex = rank - 2;
            if (rank == 4 && coordinate[adjacentRowIndex] == 0 &&
                rowCount == outShape[adjacentRowIndex]) {
                AdvanceAdjacentPlane4(coordinate, x1Offset, x2Offset);
            } else {
                for (uint32_t rowIndex = 0; rowIndex < rowCount; ++rowIndex) {
                    AdvanceTailSegment(coordinate, x1Offset, x2Offset);
                }
            }
        }
    }

    __aicore__ inline void CopyAdjacentBroadcastPlanes4(
        AscendC::LocalTensor<TYPE_X1> x1Local, AscendC::LocalTensor<TYPE_X2> x2Local,
        uint64_t& x1Offset, uint64_t& x2Offset, uint64_t* coordinate, uint64_t& copiedLength)
    {
        const bool adjacentScalarBlocks = AdjacentScalarBlockPath || useAdjacentScalarBlocks;
        const bool adjacentStridedRows = AdjacentScalarBlockPath || useAdjacentStridedRows;
        const uint64_t rowLength = outShape[3];
        const uint64_t rowCount = outShape[2];
        const uint64_t planeLength = rowCount * rowLength;
        const uint32_t rowLength32 = static_cast<uint32_t>(rowLength);
        const uint32_t rowCount32 = static_cast<uint32_t>(rowCount);
        while (copiedLength < processDataNum) {
            const uint64_t availablePlanes = (processDataNum - copiedLength) / planeLength;
            const uint64_t planesToWrap = outShape[1] - coordinate[1];
            const uint64_t groupPlanes = availablePlanes < planesToWrap ? availablePlanes : planesToWrap;
            if (x1Stride[3] == 0) {
                AscendC::LocalTensor<TYPE_X1> scalars = adjacentScalarBlocks ?
                    x2Local[processDataNum] : x1Local[processDataNum];
                CopyContiguousLength(scalars, x1Gm, x1Offset, rowCount);
                if (adjacentStridedRows) {
                    CopyAdjacentRowsToPlanes(
                        x2Local[copiedLength], x2Gm, x2Offset, rowLength, planeLength, groupPlanes);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::LocalTensor<TYPE_X1> expandedScalars = x1Local[processDataNum];
                if (adjacentScalarBlocks) {
                    constexpr uint32_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(TYPE_X1);
                    const uint32_t scalarStorageElements =
                        (rowCount32 + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
                    AscendC::LocalTensor<TYPE_X1> zeroBlock = scalars[scalarStorageElements];
                    ExpandAdjacentScalarsToBlocks(
                        expandedScalars, scalars, zeroBlock, rowCount32);
                    BroadcastAdjacentScalarBlockPlanes(
                        x1Local[copiedLength], expandedScalars, rowLength32, rowCount32,
                        planeLength, groupPlanes, rowCount32 <= 1);
                    if (rowCount32 > 1) {
                        BroadcastAdjacentRowPlanesLocal(
                            x2Local[copiedLength], zeroBlock, rowLength32, rowCount32,
                            planeLength, groupPlanes, true);
                    }
                    copiedLength += groupPlanes * planeLength;
                    if (copiedLength < processDataNum) {
                        WrapAdjacentPlaneGroup4(coordinate, x1Offset, x2Offset);
                    }
                    continue;
                }
                if constexpr (std::is_same_v<TYPE_X1, int8_t>) {
                    BroadcastAdjacentScalarsLocal(
                        x1Local[copiedLength], scalars, rowLength32, rowCount32);
                    CopyAdjacentScalarPlaneCopies(
                        x1Local[copiedLength], planeLength, groupPlanes);
                }
                for (uint64_t planeIndex = 0; planeIndex < groupPlanes; ++planeIndex) {
                    if constexpr (!std::is_same_v<TYPE_X1, int8_t>) {
                        BroadcastAdjacentScalarsLocal(
                            x1Local[copiedLength], scalars, rowLength32, rowCount32);
                    }
                    if (adjacentStridedRows) {
                        if (rowCount32 > 1) {
                            BroadcastAdjacentRowLocal(
                                x2Local[copiedLength + rowLength], x2Local[copiedLength],
                                rowLength32, rowCount32 - 1);
                        }
                    } else {
                        BroadcastAdjacentRow(
                            x2Local[copiedLength], x2Gm, x2Offset, rowLength32, rowCount32);
                    }
                    copiedLength += planeLength;
                    if (copiedLength < processDataNum) {
                        AdvanceAdjacentPlane4(coordinate, x1Offset, x2Offset);
                    }
                }
            } else {
                AscendC::LocalTensor<TYPE_X2> scalars = adjacentScalarBlocks ?
                    x1Local[processDataNum] : x2Local[processDataNum];
                CopyContiguousLength(scalars, x2Gm, x2Offset, rowCount);
                if (adjacentStridedRows) {
                    CopyAdjacentRowsToPlanes(
                        x1Local[copiedLength], x1Gm, x1Offset, rowLength, planeLength, groupPlanes);
                }
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::LocalTensor<TYPE_X2> expandedScalars = x2Local[processDataNum];
                if (adjacentScalarBlocks) {
                    constexpr uint32_t elementsPerBlock = DATA_BLOCK_BYTES / sizeof(TYPE_X2);
                    const uint32_t scalarStorageElements =
                        (rowCount32 + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
                    AscendC::LocalTensor<TYPE_X2> zeroBlock = scalars[scalarStorageElements];
                    ExpandAdjacentScalarsToBlocks(
                        expandedScalars, scalars, zeroBlock, rowCount32);
                    if (rowCount32 > 1) {
                        BroadcastAdjacentRowPlanesLocal(
                            x1Local[copiedLength], zeroBlock, rowLength32, rowCount32,
                            planeLength, groupPlanes, false);
                    }
                    BroadcastAdjacentScalarBlockPlanes(
                        x2Local[copiedLength], expandedScalars, rowLength32, rowCount32,
                        planeLength, groupPlanes, true);
                    copiedLength += groupPlanes * planeLength;
                    if (copiedLength < processDataNum) {
                        WrapAdjacentPlaneGroup4(coordinate, x1Offset, x2Offset);
                    }
                    continue;
                }
                AscendC::LocalTensor<half> int8ScalarsHalf;
                if constexpr (std::is_same_v<TYPE_X2, int8_t>) {
                    int8ScalarsHalf = calc_buf_2.Get<half>();
                    AscendC::Cast(
                        int8ScalarsHalf, scalars, AscendC::RoundMode::CAST_NONE, rowCount32);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                for (uint64_t planeIndex = 0; planeIndex < groupPlanes; ++planeIndex) {
                    if constexpr (std::is_same_v<TYPE_X2, int8_t>) {
                        BroadcastAdjacentHalfScalarsToInt8(
                            x2Local[copiedLength], int8ScalarsHalf, rowLength32, rowCount32);
                    }
                    if (adjacentStridedRows) {
                        if (rowCount32 > 1) {
                            BroadcastAdjacentRowLocal(
                                x1Local[copiedLength + rowLength], x1Local[copiedLength],
                                rowLength32, rowCount32 - 1);
                        }
                    } else {
                        BroadcastAdjacentRow(
                            x1Local[copiedLength], x1Gm, x1Offset, rowLength32, rowCount32);
                    }
                    if constexpr (!std::is_same_v<TYPE_X2, int8_t>) {
                        BroadcastAdjacentScalarsLocal(
                            x2Local[copiedLength], scalars, rowLength32, rowCount32);
                    }
                    copiedLength += planeLength;
                    if (copiedLength < processDataNum) {
                        AdvanceAdjacentPlane4(coordinate, x1Offset, x2Offset);
                    }
                }
            }
        }
    }

    template <typename U>
    __aicore__ inline void BroadcastAdjacentPaddedScalars(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source,
        uint64_t sourceOffset, uint32_t rowCount, uint64_t planeCount)
    {
        AscendC::LocalTensor<U> scalars = broadcastRowQueue.AllocTensor<U>();
        CopyContiguousLength(scalars, source, sourceOffset, rowCount);
        broadcastRowQueue.EnQue(scalars);
        scalars = broadcastRowQueue.DeQue<U>();
        for (uint64_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
            BroadcastAdjacentScalarsLocal(
                destination[planeIndex * paddedPlaneLength], scalars,
                static_cast<uint32_t>(paddedRowLength), rowCount);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        broadcastRowQueue.FreeTensor(scalars);
    }

    template <typename U>
    __aicore__ inline void BroadcastAdjacentPaddedRows(
        AscendC::LocalTensor<U> destination, AscendC::GlobalTensor<U>& source,
        uint64_t sourceOffset, uint32_t rowCount, uint64_t planeCount)
    {
        AscendC::LocalTensor<U> rows = broadcastRowQueue.AllocTensor<U>();
        for (uint64_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
            CopyContiguousLength(
                rows[planeIndex * paddedRowLength], source,
                sourceOffset + planeIndex * outShape[tailIndex], outShape[tailIndex]);
        }
        broadcastRowQueue.EnQue(rows);
        rows = broadcastRowQueue.DeQue<U>();
        for (uint64_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
            BroadcastAdjacentRowLocal(
                destination[planeIndex * paddedPlaneLength], rows[planeIndex * paddedRowLength],
                static_cast<uint32_t>(paddedRowLength), rowCount);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        broadcastRowQueue.FreeTensor(rows);
    }

    __aicore__ inline void CopyInAdjacentPaddedPlanes(uint64_t outputOffset, uint64_t planeCount)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template AllocTensor<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template AllocTensor<TYPE_X2>();
        const uint32_t rowCount = static_cast<uint32_t>(outShape[rowIndex]);
        const uint64_t x1Offset = CalcInputOffset(outputOffset, x1Stride);
        const uint64_t x2Offset = CalcInputOffset(outputOffset, x2Stride);
        if (x1Stride[tailIndex] == 0) {
            BroadcastAdjacentPaddedScalars(x1Local, x1Gm, x1Offset, rowCount, planeCount);
            BroadcastAdjacentPaddedRows(x2Local, x2Gm, x2Offset, rowCount, planeCount);
        } else {
            BroadcastAdjacentPaddedRows(x1Local, x1Gm, x1Offset, rowCount, planeCount);
            BroadcastAdjacentPaddedScalars(x2Local, x2Gm, x2Offset, rowCount, planeCount);
        }
        if constexpr (
            std::is_same_v<TYPE_X1, half> || std::is_same_v<TYPE_X1, bfloat16_t>) {
            const uint64_t rowLength = outShape[tailIndex];
            if (paddedRowLength != rowLength) {
                const uint64_t rowTotal = planeCount * outShape[rowIndex];
                AscendC::GatherMaskParams gatherParams(
                    1, static_cast<uint16_t>(rowTotal),
                    static_cast<uint16_t>(paddedRowLength * sizeof(TYPE_X1) / DATA_BLOCK_BYTES), 0);
                uint64_t gatheredCount = 0;
                AscendC::GatherMask<TYPE_X1>(
                    x1Local, x1Local, 7, true, static_cast<uint32_t>(rowLength),
                    gatherParams, gatheredCount);
                AscendC::GatherMask<TYPE_X2>(
                    x2Local, x2Local, 7, true, static_cast<uint32_t>(rowLength),
                    gatherParams, gatheredCount);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void CopyOutAdjacentPaddedPlanes(uint64_t outputOffset, uint64_t planeCount)
    {
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.template DeQue<int8_t>();
        const uint64_t rowLength = outShape[tailIndex];
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint16_t>(rowLength);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        if constexpr (
            std::is_same_v<TYPE_X1, half> || std::is_same_v<TYPE_X1, bfloat16_t>) {
            copyParams.blockLen = static_cast<uint16_t>(planeCount * outShape[rowIndex] * rowLength);
            AscendC::DataCopyPad(yGm[outputOffset], yLocal, copyParams);
            outQueueY.FreeTensor(yLocal);
            return;
        }
        if constexpr (std::is_same_v<TYPE_X1, int8_t>) {
            const uint64_t planeOutputBytes = outShape[rowIndex] * rowLength;
            const bool compactAdjacentOutput =
                paddedRowLength != rowLength && rowLength % sizeof(uint16_t) == 0 &&
                planeOutputBytes != 0 && planeOutputBytes <= DATA_COPY_PAD_MAX_BYTES;
            if (compactAdjacentOutput) {
                const uint64_t rowTotal = planeCount * outShape[rowIndex];
                AscendC::LocalTensor<uint16_t> packedOutput = yLocal.template ReinterpretCast<uint16_t>();
                AscendC::GatherMaskParams gatherParams(
                    1, static_cast<uint16_t>(rowTotal),
                    static_cast<uint16_t>(paddedRowLength / DATA_BLOCK_BYTES), 0);
                uint64_t gatheredCount = 0;
                AscendC::GatherMask<uint16_t>(
                    packedOutput, packedOutput, 7, true,
                    static_cast<uint32_t>(rowLength / sizeof(uint16_t)), gatherParams, gatheredCount);
                AscendC::PipeBarrier<PIPE_V>();
                copyParams.blockLen = static_cast<uint16_t>(rowTotal * rowLength);
                AscendC::DataCopyPad(yGm[outputOffset], yLocal, copyParams);
                outQueueY.FreeTensor(yLocal);
                return;
            }
        }
        for (uint64_t planeIndex = 0; planeIndex < planeCount; ++planeIndex) {
            for (uint64_t currentRow = 0; currentRow < outShape[rowIndex]; ++currentRow) {
                AscendC::DataCopyPad(
                    yGm[outputOffset + (planeIndex * outShape[rowIndex] + currentRow) * rowLength],
                    yLocal[planeIndex * paddedPlaneLength + currentRow * paddedRowLength], copyParams);
            }
        }
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyInAdjacentScalarBlocks(uint64_t outputOffset, uint64_t planeLength)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template AllocTensor<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template AllocTensor<TYPE_X2>();
        uint64_t coordinate[2] = {};
        const uint64_t planeIndex = outputOffset / planeLength;
        coordinate[1] = planeIndex % outShape[1];
        coordinate[0] = planeIndex / outShape[1];
        uint64_t x1Offset = coordinate[0] * x1Stride[0] + coordinate[1] * x1Stride[1];
        uint64_t x2Offset = coordinate[0] * x2Stride[0] + coordinate[1] * x2Stride[1];
        uint64_t copiedLength = 0;
        CopyAdjacentBroadcastPlanes4(
            x1Local, x2Local, x1Offset, x2Offset, coordinate, copiedLength);
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void CopyInPacked(uint64_t outputOffset)
    {
        AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template AllocTensor<TYPE_X1>();
        AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template AllocTensor<TYPE_X2>();
        const uint64_t rowLength = outShape[rank - 1];
        const uint64_t rowCount = processDataNum / rowLength;
        uint64_t x1Offset = CalcInputOffset(outputOffset, x1Stride);
        uint64_t x2Offset = CalcInputOffset(outputOffset, x2Stride);
        const bool x1OuterBroadcast = useRowBroadcastBuffer && IsOuterBroadcast(x1Stride, x2Stride);
        const bool x2OuterBroadcast = useRowBroadcastBuffer && IsOuterBroadcast(x2Stride, x1Stride);
        if (x1OuterBroadcast) {
            CopyRepeatedRows(x1Local, x1Gm, x1Offset, rowLength, rowCount);
            CopyContiguousLength(x2Local, x2Gm, x2Offset, processDataNum);
        } else if (x2OuterBroadcast) {
            CopyRepeatedRows(x2Local, x2Gm, x2Offset, rowLength, rowCount);
            CopyContiguousLength(x1Local, x1Gm, x1Offset, processDataNum);
        } else {
            uint64_t coordinate[GREATER_MAX_BROADCAST_DIM] = {};
            InitOutputCoord(outputOffset, coordinate);
            uint64_t copiedLength = 0;
            const uint64_t adjacentPlaneLength = rank > 1 ? outShape[rank - 2] * outShape[rank - 1] : 0;
            if (useAdjacentPlaneCache && adjacentPlaneLength != 0 &&
                outputOffset % adjacentPlaneLength == 0 && processDataNum % adjacentPlaneLength == 0) {
                CopyAdjacentBroadcastPlanes4(
                    x1Local, x2Local, x1Offset, x2Offset, coordinate, copiedLength);
                inQueueX1.EnQue(x1Local);
                inQueueX2.EnQue(x2Local);
                return;
            }
            while (copiedLength < processDataNum) {
                if (useAdjacentBroadcast) {
                    CopyAdjacentBroadcastRows(
                        x1Local, x2Local, x1Offset, x2Offset, coordinate, copiedLength);
                    continue;
                }
                if (x1Stride[rank - 1] == 0) {
                    BroadcastScalar(
                        x1Local[copiedLength], x1Gm, x1Offset, rowLength, x1ScalarQueue);
                } else {
                    CopyContiguousLength(x1Local[copiedLength], x1Gm, x1Offset, rowLength);
                }
                if (x2Stride[rank - 1] == 0) {
                    BroadcastScalar(
                        x2Local[copiedLength], x2Gm, x2Offset, rowLength, x2ScalarQueue);
                } else {
                    CopyContiguousLength(x2Local[copiedLength], x2Gm, x2Offset, rowLength);
                }
                copiedLength += rowLength;
                if (copiedLength < processDataNum) {
                    AdvanceTailSegment(coordinate, x1Offset, x2Offset);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute()
    {
        if constexpr (std::is_same_v<TYPE_X1, half>) {
            AscendC::LocalTensor<half> x1Local = inQueueX1.template DeQue<half>();
            AscendC::LocalTensor<half> x2Local = inQueueX2.template DeQue<half>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::GT, processDataNumComputes);
            if constexpr (AdjacentScalarBlockPath) {
                constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
                constexpr uint32_t maxAlignedRepeatTimes = 254;
                constexpr uint32_t firstBatchElements =
                    maxAlignedRepeatTimes * halfRepeatElements;
                AscendC::LocalTensor<half> oneBlock = x2Local[ubPartDataNum];
                AscendC::Duplicate<half>(oneBlock, (half)1, halfRepeatElements);
                const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
                const uint32_t repeatTimes = processDataNumComputes / halfRepeatElements;
                const uint8_t firstRepeatTimes = static_cast<uint8_t>(
                    repeatTimes > maxAlignedRepeatTimes ? maxAlignedRepeatTimes : repeatTimes);
                AscendC::Select(
                    x1Local, yLocal, oneBlock, (half)0,
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, halfRepeatElements,
                    firstRepeatTimes, repeatParams);
                if (repeatTimes > maxAlignedRepeatTimes) {
                    AscendC::Select(
                        x1Local[firstBatchElements], yLocal[firstBatchElements / 8], oneBlock,
                        (half)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        halfRepeatElements,
                        static_cast<uint8_t>(repeatTimes - maxAlignedRepeatTimes), repeatParams);
                }
            } else {
                AscendC::Duplicate<half>(x1Local, (half)1, processDataNumComputes);
                AscendC::Select(
                    x1Local, yLocal, x1Local, (half)0,
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, processDataNumComputes);
            }
            AscendC::Cast(yLocal, x1Local, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<TYPE_X1, bfloat16_t>) {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::LocalTensor<half> yFp16;
            if constexpr (AdjacentScalarBlockPath || AdjacentPaddedPlanePath) {
                yFp16 = calc_buf_1.Get<half>();
            } else if constexpr (BroadcastMode == GREATER_BROADCAST_X1_SCALAR) {
                yFp16 = x2Local.template ReinterpretCast<half>();
            } else {
                yFp16 = x1Local.template ReinterpretCast<half>();
            }
            AscendC::LocalTensor<float> x1Fp32 = calc_buf_2.Get<float>();
            AscendC::LocalTensor<float> x2Fp32 = calc_buf_3.Get<float>();
            if constexpr (BroadcastMode == GREATER_BROADCAST_X1_SCALAR) {
                constexpr uint32_t scalarBlockElements = DATA_BLOCK_BYTES / sizeof(bfloat16_t);
                AscendC::Cast(x1Fp32, x1Local, AscendC::RoundMode::CAST_NONE, scalarBlockElements);
                AscendC::Cast(x2Fp32, x2Local, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
                const auto eventId = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
                AscendC::CompareScalar(
                    yLocal, x2Fp32, x1Fp32.GetValue(0), AscendC::CMPMODE::LT,
                    processDataNumComputes);
            } else if constexpr (BroadcastMode == GREATER_BROADCAST_X2_SCALAR) {
                constexpr uint32_t scalarBlockElements = DATA_BLOCK_BYTES / sizeof(bfloat16_t);
                AscendC::Cast(x2Fp32, x2Local, AscendC::RoundMode::CAST_NONE, scalarBlockElements);
                AscendC::Cast(x1Fp32, x1Local, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
                const auto eventId = GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S);
                AscendC::SetFlag<AscendC::HardEvent::V_S>(eventId);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(eventId);
                AscendC::CompareScalar(
                    yLocal, x1Fp32, x2Fp32.GetValue(0), AscendC::CMPMODE::GT,
                    processDataNumComputes);
            } else {
                AscendC::Cast(x1Fp32, x1Local, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
                AscendC::Cast(x2Fp32, x2Local, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
                AscendC::Compare(yLocal, x1Fp32, x2Fp32, AscendC::CMPMODE::GT, processDataNumComputes);
            }
            constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
            AscendC::LocalTensor<half> oneBlock = x1Fp32.template ReinterpretCast<half>();
            AscendC::Duplicate<half>(oneBlock, (half)1, halfRepeatElements);
            const uint8_t repeatTimes = static_cast<uint8_t>(
                processDataNumComputes / halfRepeatElements);
            const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
            AscendC::Select(
                yFp16, yLocal, oneBlock, (half)0, AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                halfRepeatElements, repeatTimes, repeatParams);
            AscendC::Cast(yLocal, yFp16, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<TYPE_X1, float>) {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::LocalTensor<half> yFp16;
            AscendC::LocalTensor<half> oneBlock;
            if constexpr (BroadcastMode == GREATER_BROADCAST_CONTIGUOUS) {
                // Compare is the final consumer of both raw inputs. Reuse
                // their dead storage for predicate materialization and the
                // constant block, allowing a larger wave-local DMA tile.
                yFp16 = x1Local.template ReinterpretCast<half>();
                oneBlock = x2Local.template ReinterpretCast<half>();
            } else {
                yFp16 = calc_buf_1.Get<half>();
                oneBlock = x1Local.template ReinterpretCast<half>();
            }
            AscendC::Compare(yLocal, x1Local, x2Local, AscendC::CMPMODE::GT, processDataNumComputes);
            constexpr uint32_t halfRepeatElements = VECTOR_ALIGN_BYTES / sizeof(half);
            AscendC::Duplicate<half>(oneBlock, (half)1, halfRepeatElements);
            const uint8_t repeatTimes = static_cast<uint8_t>(
                (processDataNumComputes + halfRepeatElements - 1) / halfRepeatElements);
            const AscendC::BinaryRepeatParams repeatParams(1, 1, 0, 8, 0, 0);
            AscendC::Select(
                yFp16, yLocal, oneBlock, (half)0,
                AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, halfRepeatElements,
                repeatTimes, repeatParams);
            AscendC::Cast(yLocal, yFp16, AscendC::RoundMode::CAST_NONE, processDataNumComputes);
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<TYPE_X1, int8_t> || std::is_same_v<TYPE_X1, uint8_t>) {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::LocalTensor<half> x1Fp16 = calc_buf_1.Get<half>();
            AscendC::LocalTensor<half> x2Fp16 = calc_buf_2.Get<half>();
            if constexpr (INT8_SEGMENTED_MACRO_TILE) {
                const uint64_t segmentCapacity = ubPartDataNum / 2;
                if (processDataNum == ubPartDataNum) {
                    ComputeInt8MacroSegment(
                        yLocal, x1Local, x2Local, x1Fp16, x2Fp16,
                        0, segmentCapacity);
                    ComputeInt8MacroSegment(
                        yLocal, x1Local, x2Local, x1Fp16, x2Fp16,
                        segmentCapacity, segmentCapacity);
                } else if (processDataNumComputes <= segmentCapacity) {
                    ComputeInt8MacroSegment(
                        yLocal, x1Local, x2Local, x1Fp16, x2Fp16,
                        0, processDataNumComputes);
                } else {
                    ComputeInt8MacroSegment(
                        yLocal, x1Local, x2Local, x1Fp16, x2Fp16,
                        0, segmentCapacity);
                    ComputeInt8MacroSegment(
                        yLocal, x1Local, x2Local, x1Fp16, x2Fp16,
                        segmentCapacity,
                        processDataNumComputes - segmentCapacity);
                }
            } else {
                AscendC::Cast(
                    x1Fp16, x1Local, AscendC::RoundMode::CAST_NONE,
                    processDataNumComputes);
                AscendC::Cast(
                    x2Fp16, x2Local, AscendC::RoundMode::CAST_NONE,
                    processDataNumComputes);
                AscendC::Compare(
                    yLocal, x1Fp16, x2Fp16,
                    AscendC::CMPMODE::GT, processDataNumComputes);
                if constexpr (ReuseInt8Predicate) {
                    MaterializeInt8Predicate(
                        yLocal, yLocal, x1Fp16, x2Fp16,
                        processDataNumComputes);
                } else {
                    AscendC::Duplicate<half>(
                        x1Fp16, (half)1, processDataNumComputes);
                    AscendC::Select(
                        x1Fp16, yLocal, x1Fp16, (half)0,
                        AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE,
                        processDataNumComputes);
                    AscendC::Cast(
                        yLocal, x1Fp16, AscendC::RoundMode::CAST_NONE,
                        processDataNumComputes);
                }
            }
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        } else if constexpr (std::is_same_v<TYPE_X1, int32_t>) {
            AscendC::LocalTensor<TYPE_X1> x1Local = inQueueX1.template DeQue<TYPE_X1>();
            AscendC::LocalTensor<TYPE_X2> x2Local = inQueueX2.template DeQue<TYPE_X2>();
            AscendC::LocalTensor<int8_t> yLocal = outQueueY.template AllocTensor<int8_t>();
            AscendC::LocalTensor<half> yFp16 = calc_buf_1.Get<half>();
            AscendC::LocalTensor<half> zeroBlock = x2Local.template ReinterpretCast<half>();
            if constexpr (BroadcastMode == GREATER_BROADCAST_X2_SCALAR) {
                AscendC::Maxs(x1Local, x1Local, x2ScalarValue, processDataNumComputes);
                AscendC::CompareScalar(
                    yLocal, x1Local, x2ScalarValue, AscendC::CMPMODE::EQ, processDataNumComputes);
            } else {
                AscendC::Max(x1Local, x1Local, x2Local, processDataNumComputes);
                AscendC::Compare(yLocal, x2Local, x1Local, AscendC::CMPMODE::EQ, processDataNumComputes);
            }
            if constexpr (Int32PaddedRowPath) {
                MaterializeInt32PaddedPredicate(yLocal, yFp16);
            } else {
                MaterializeInt32Predicate(
                    yLocal, yFp16, zeroBlock, processDataNumComputes);
            }
            outQueueY.template EnQue<int8_t>(yLocal);
            inQueueX1.FreeTensor(x1Local);
            inQueueX2.FreeTensor(x2Local);
        }
    }

    __aicore__ inline void ComputeInt8MacroSegment(
        AscendC::LocalTensor<int8_t> yLocal,
        AscendC::LocalTensor<TYPE_X1> x1Local,
        AscendC::LocalTensor<TYPE_X2> x2Local,
        AscendC::LocalTensor<half> x1Fp16,
        AscendC::LocalTensor<half> x2Fp16,
        uint64_t segmentOffset, uint64_t segmentComputeNum)
    {
        AscendC::Cast(
            x1Fp16, x1Local[segmentOffset],
            AscendC::RoundMode::CAST_NONE, segmentComputeNum);
        AscendC::Cast(
            x2Fp16, x2Local[segmentOffset],
            AscendC::RoundMode::CAST_NONE, segmentComputeNum);
        AscendC::Compare(
            yLocal[segmentOffset], x1Fp16, x2Fp16,
            AscendC::CMPMODE::GT, segmentComputeNum);
        MaterializePreloadedInt8Predicate(
            yLocal[segmentOffset], yLocal[segmentOffset],
            x1Fp16, x2Fp16[ubPartDataNum / 2], segmentComputeNum);
    }

    __aicore__ inline void MaterializeInt32PaddedPredicate(
        AscendC::LocalTensor<int8_t> predicate,
        AscendC::LocalTensor<half> result)
    {
        constexpr uint32_t halfRepeatElements =
            VECTOR_ALIGN_BYTES / sizeof(half);
        AscendC::LocalTensor<half> zeroBlock =
            calc_buf_1.Get<half>()[ubPartDataNum];
        const uint8_t repeatTimes = static_cast<uint8_t>(
            (processDataNumComputes + halfRepeatElements - 1) /
            halfRepeatElements);
        const AscendC::BinaryRepeatParams repeatParams(
            1, 1, 0, 8, 0, 0);
        AscendC::Select(
            result, predicate, zeroBlock, repeatTimes, repeatParams);
        AscendC::Cast(
            predicate, result, AscendC::RoundMode::CAST_NONE,
            processDataNumComputes);
    }

    __aicore__ inline void SetComputeDataNum()
    {
        const uint64_t processBytes = processDataNum * sizeof(TYPE_X1);
        processDataNumComputes =
            ((processBytes + VECTOR_ALIGN_BYTES - 1) / VECTOR_ALIGN_BYTES) * VECTOR_ALIGN_BYTES / sizeof(TYPE_X1);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset)
    {
        AscendC::LocalTensor<int8_t> yLocal = outQueueY.template DeQue<int8_t>();
        if constexpr (
            std::is_same_v<TYPE_X1, float> &&
            BroadcastMode == GREATER_BROADCAST_CONTIGUOUS) {
            if (processDataNum % DATA_BLOCK_BYTES == 0) {
                AscendC::DataCopy(yGm[outputOffset], yLocal, processDataNum);
                outQueueY.FreeTensor(yLocal);
                return;
            }
        }
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = processDataNum;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[outputOffset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_BUFFER_NUM> inQueueX1;
    AscendC::TQue<AscendC::QuePosition::VECIN, QUEUE_BUFFER_NUM> inQueueX2;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x1ScalarQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> x2ScalarQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> broadcastRowQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, OUTPUT_QUEUE_BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calc_buf_1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calc_buf_2;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calc_buf_3;
    AscendC::GlobalTensor<TYPE_X1> x1Gm;
    AscendC::GlobalTensor<TYPE_X2> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t ubPartDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t waveTailStride = 0;
    uint64_t blockOffset = 0;
    uint64_t processDataNum = 0;
    uint64_t processDataNumComputes = 0;
    uint64_t paddedRowLength = 0;
    uint64_t paddedPlaneLength = 0;
    uint64_t maxPaddedPlaneCount = 0;
    uint64_t maxPaddedRowCount = 0;
    uint32_t tailIndex = 0;
    uint32_t rowIndex = 0;
    uint32_t planeGroupIndex = 0;
    uint32_t rank = 0;
    bool useRowBroadcastBuffer = false;
    uint64_t outShape[GREATER_MAX_BROADCAST_DIM] = {};
    uint64_t x1Stride[GREATER_MAX_BROADCAST_DIM] = {};
    uint64_t x2Stride[GREATER_MAX_BROADCAST_DIM] = {};
    int32_t x2ScalarValue = 0;
    bool useAdjacentBroadcast = false;
    bool useAdjacentPlaneCache = false;
    bool useAdjacentStridedRows = false;
    bool useAdjacentScalarBlocks = false;
};
} // namespace NsGreater
#endif // GREATER_H
