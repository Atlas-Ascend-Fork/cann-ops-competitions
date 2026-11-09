/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __GREATER_INT32_DIRECT_ROW_PIPELINE_H__
#define __GREATER_INT32_DIRECT_ROW_PIPELINE_H__

#include "greater.h"

namespace NsGreater {

template <bool LinearGroupLayout = false>
class GreaterInt32DirectRowPipeline {
public:
    static constexpr uint64_t ROW_BATCH = 8;
    static constexpr uint64_t BROADCAST_SEED_ROWS = 8;
    static constexpr uint64_t MAX_OUTPUT_BATCHES = 8;

    __aicore__ inline GreaterInt32DirectRowPipeline() {}

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        uint64_t smallCoreDataNum, uint64_t bigCoreDataNum,
        uint64_t ubPartDataNum, uint64_t smallCoreTailDataNum,
        uint64_t bigCoreTailDataNum, uint64_t smallCoreLoopNum,
        uint64_t bigCoreLoopNum, uint64_t tailBlockNum, uint32_t rank,
        const uint64_t* outShape, const uint64_t* x1Stride,
        const uint64_t* x2Stride, uint32_t int32CohortSize)
    {
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        if (tailBlockNum != 0 && blockIdx < tailBlockNum) {
            coreDataNum = bigCoreDataNum;
        } else {
            coreDataNum = smallCoreDataNum;
        }

        if constexpr (!LinearGroupLayout) {
            this->rank = rank;
            for (uint32_t index = 0;
                 index < GREATER_MAX_BROADCAST_DIM; ++index) {
                this->outShape[index] = outShape[index];
                this->x1Stride[index] = x1Stride[index];
                this->x2Stride[index] = x2Stride[index];
            }
            tailIndex = rank - 1;
            rowIndex = rank - 2;
            rowLength = this->outShape[tailIndex];
            rowCount = this->outShape[rowIndex];
            x1Rows =
                this->x1Stride[tailIndex] == 1 &&
                this->x1Stride[rowIndex] == 0;
        }
        this->int32CohortSize =
            int32CohortSize & GREATER_INT32_COHORT_MASK;
        paddedRowLength =
            (rowLength + DATA_BLOCK_BYTES - 1) /
            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        (void)ubPartDataNum;
        batchDataNum = ROW_BATCH * paddedRowLength;

        x1Gm.SetGlobalBuffer((__gm__ int32_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int32_t*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);
        if (x1Rows) {
            x2Gm.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        } else {
            x1Gm.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        pipe.InitBuffer(
            inputQueue, 2,
            batchDataNum * sizeof(int32_t));
        pipe.InitBuffer(
            broadcastRowQueue, 1,
            paddedRowLength * sizeof(int32_t));
        pipe.InitBuffer(
            broadcastTileCache,
            ROW_BATCH * paddedRowLength * sizeof(int32_t));
        // One lane-sized output wave turns the 7/8 adjacent vector batches
        // into one 56/64 KiB MTE3 command.  The next natural group's row,
        // input fill, and broadcast materialization provide the independent
        // work that overlaps this single-buffer drain.
        pipe.InitBuffer(
            outputQueue, 1,
            (MAX_OUTPUT_BATCHES + 1) * batchDataNum *
                sizeof(int8_t));
        pipe.InitBuffer(
            predicateScratch,
            VECTOR_ALIGN_BYTES + batchDataNum / 8);
    }

    __aicore__ inline void InitLinear(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
        const GreaterInt32DirectTilingData& tilingData)
    {
        static_assert(LinearGroupLayout,
            "InitLinear requires the compact linear instantiation");
        rowLength = tilingData.rowLength;
        rowCount = tilingData.rowCount;
        naturalGroupCount = tilingData.naturalGroupCount;
        int32CohortSize = tilingData.cohortSize;
        x1Rows = tilingData.x1Rows != 0;
        coreDataNum = naturalGroupCount;
        paddedRowLength =
            (rowLength + DATA_BLOCK_BYTES - 1) /
            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        batchDataNum = ROW_BATCH * paddedRowLength;

        x1Gm.SetGlobalBuffer((__gm__ int32_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int32_t*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);
        if (x1Rows) {
            x2Gm.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        } else {
            x1Gm.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        }

        pipe.InitBuffer(
            inputQueue, 2,
            batchDataNum * sizeof(int32_t));
        pipe.InitBuffer(
            broadcastRowQueue, 1,
            paddedRowLength * sizeof(int32_t));
        pipe.InitBuffer(
            broadcastTileCache,
            ROW_BATCH * paddedRowLength * sizeof(int32_t));
        pipe.InitBuffer(
            outputQueue, 1,
            (MAX_OUTPUT_BATCHES + 1) * batchDataNum *
                sizeof(int8_t));
        pipe.InitBuffer(
            predicateScratch,
            VECTOR_ALIGN_BYTES + batchDataNum / 8);
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum == 0 || batchDataNum == 0 ||
            int32CohortSize == 0) {
            return;
        }
        InitPredicateConstants();
        ProcessCohortGroups();
    }

private:
    __aicore__ inline uint64_t NaturalGroupCount()
    {
        if constexpr (LinearGroupLayout) {
            return naturalGroupCount;
        } else {
            uint64_t count = 1;
            for (uint32_t index = 0; index < rowIndex; ++index) {
                count *= outShape[index];
            }
            return count;
        }
    }

    __aicore__ inline void ProcessCohortGroups()
    {
        const uint64_t naturalGroupLength = rowCount * rowLength;
        const uint64_t batchCount =
            (rowCount + ROW_BATCH - 1) / ROW_BATCH;
        const uint64_t blockCount = AscendC::GetBlockNum();
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t cohortSize = int32CohortSize;
        const uint64_t groupsPerWave = blockCount / cohortSize;
        const uint64_t groupSlot = blockIdx / cohortSize;
        const uint64_t physicalLane = blockIdx % cohortSize;
        const uint64_t groupCount = NaturalGroupCount();
        const uint64_t groupLogicalBytes = ROW_BATCH * rowLength;
        // Let the cohort issue adjacent input batches together.  Each lane
        // keeps its compact results consecutive in UB and uses one strided
        // MTE3 command, so neither the number of DMA descriptors nor the GM
        // traffic changes.  Fall back to contiguous lane ranges whenever the
        // byte-stride form cannot represent the geometry exactly.
        const bool interleaveLaneBatches =
            cohortSize > 1 && rowCount % ROW_BATCH == 0 &&
            batchCount >= cohortSize &&
            (batchCount + cohortSize - 1) / cohortSize <=
                MAX_OUTPUT_BATCHES &&
            groupLogicalBytes <=
                DATA_COPY_PAD_MAX_BYTES /
                    (cohortSize - 1);

        for (uint64_t groupIndex = groupSlot;
             groupIndex < groupCount;
             groupIndex += groupsPerWave) {
            const uint64_t waveIndex =
                groupIndex / groupsPerWave;
            const uint64_t logicalLane =
                (physicalLane + waveIndex % cohortSize) %
                cohortSize;
            const uint64_t groupOffset =
                groupIndex * naturalGroupLength;
            uint64_t broadcastGroupOffset;
            uint64_t contiguousGroupOffset;
            if constexpr (LinearGroupLayout) {
                broadcastGroupOffset = groupIndex * rowLength;
                contiguousGroupOffset = groupOffset;
            } else {
                broadcastGroupOffset = 0;
                contiguousGroupOffset = 0;
                CalcInputOffsets(
                    groupOffset, x1Rows,
                    broadcastGroupOffset, contiguousGroupOffset);
            }

            const uint64_t baseBatchesPerLane =
                batchCount / cohortSize;
            const uint64_t extraBatchLanes =
                batchCount % cohortSize;
            const uint64_t laneBatchCount = interleaveLaneBatches ?
                (logicalLane < batchCount ?
                    (batchCount - 1 - logicalLane) / cohortSize + 1 : 0) :
                baseBatchesPerLane +
                    (logicalLane < extraBatchLanes ? 1 : 0);
            if (laneBatchCount == 0) {
                continue;
            }
            const uint64_t firstBatch = interleaveLaneBatches ?
                logicalLane :
                logicalLane * baseBatchesPerLane +
                    (logicalLane < extraBatchLanes ?
                        logicalLane : extraBatchLanes);
            const uint64_t batchAdvance =
                interleaveLaneBatches ? cohortSize : 1;
            uint64_t nextBatchToPrefetch = 0;

            PrefetchBroadcastRow(
                broadcastGroupOffset, x1Rows);
            const uint64_t initialPrefetchCount =
                laneBatchCount < 2 ? laneBatchCount : 2;
            for (uint64_t index = 0;
                 index < initialPrefetchCount; ++index) {
                const uint64_t batch =
                    firstBatch + nextBatchToPrefetch++ * batchAdvance;
                const uint64_t firstRow = batch * ROW_BATCH;
                const uint64_t rowsRemaining = rowCount - firstRow;
                const uint64_t currentRowCount =
                    rowsRemaining < ROW_BATCH ?
                    rowsRemaining : ROW_BATCH;
                CopyInput(
                    contiguousGroupOffset + firstRow * rowLength,
                    currentRowCount, x1Rows);
            }
            MaterializeBroadcastTile();
            uint64_t completedBatchCount = 0;
            while (completedBatchCount < laneBatchCount) {
                const uint64_t outputBatchCount =
                    laneBatchCount - completedBatchCount <
                            MAX_OUTPUT_BATCHES ?
                        laneBatchCount - completedBatchCount :
                        MAX_OUTPUT_BATCHES;
                const uint64_t outputFirstBatch =
                    firstBatch + completedBatchCount * batchAdvance;
                const uint64_t outputFirstRow =
                    outputFirstBatch * ROW_BATCH;
                const uint64_t outputFirstOffset =
                    groupOffset + outputFirstRow * rowLength;
                uint64_t outputRowCount = 0;
                AscendC::LocalTensor<int8_t> output =
                    outputQueue.AllocTensor<int8_t>();
                for (uint64_t outputBatch = 0;
                     outputBatch < outputBatchCount;
                     ++outputBatch) {
                    const uint64_t currentBatch =
                        outputFirstBatch + outputBatch * batchAdvance;
                    const uint64_t currentFirstRow =
                        currentBatch * ROW_BATCH;
                    const uint64_t rowsRemaining =
                        rowCount - currentFirstRow;
                    const uint64_t currentRowCount =
                        rowsRemaining < ROW_BATCH ?
                        rowsRemaining : ROW_BATCH;
                    ComputePredicate(currentRowCount, x1Rows);
                    if (nextBatchToPrefetch < laneBatchCount) {
                        const uint64_t prefetchBatch =
                            firstBatch +
                            nextBatchToPrefetch++ * batchAdvance;
                        const uint64_t prefetchFirstRow =
                            prefetchBatch * ROW_BATCH;
                        const uint64_t prefetchRowsRemaining =
                            rowCount - prefetchFirstRow;
                        const uint64_t prefetchRowCount =
                            prefetchRowsRemaining < ROW_BATCH ?
                            prefetchRowsRemaining : ROW_BATCH;
                        CopyInput(
                            contiguousGroupOffset +
                                prefetchFirstRow * rowLength,
                            prefetchRowCount, x1Rows);
                    }
                    MaterializePredicate(
                        output, outputBatch, currentRowCount);
                    outputRowCount += currentRowCount;
                }
                outputQueue.EnQue<int8_t>(output);
                CopyOutput(
                    outputFirstOffset, outputRowCount, batchAdvance);
                completedBatchCount += outputBatchCount;
            }
        }
    }

    __aicore__ inline void PrefetchBroadcastRow(
        uint64_t broadcastOffset, bool x1Rows)
    {
        AscendC::LocalTensor<int32_t> row =
            broadcastRowQueue.AllocTensor<int32_t>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint16_t>(
            rowLength * sizeof(int32_t));
        copyParams.srcGap = 0;
        copyParams.dstGap = 0;
        AscendC::GlobalTensor<int32_t>& broadcastGm =
            x1Rows ? x1Gm : x2Gm;
        AscendC::DataCopyPad(
            row, broadcastGm[broadcastOffset],
            copyParams, {false, 0, 0, 0});
        broadcastRowQueue.EnQue(row);
    }

    __aicore__ inline void MaterializeBroadcastTile()
    {
        AscendC::LocalTensor<int32_t> row =
            broadcastRowQueue.DeQue<int32_t>();

        AscendC::LocalTensor<int32_t> broadcastTile =
            broadcastTileCache.Get<int32_t>();
        const uint32_t destinationShape[2] = {
            static_cast<uint32_t>(BROADCAST_SEED_ROWS),
            static_cast<uint32_t>(paddedRowLength)};
        const uint32_t sourceShape[2] = {
            1, static_cast<uint32_t>(paddedRowLength)};
        AscendC::Broadcast<int32_t, 2, 0>(
            broadcastTile, row, destinationShape, sourceShape);
        AscendC::GatherMaskParams gatherParams(
            1, static_cast<uint16_t>(BROADCAST_SEED_ROWS),
            static_cast<uint16_t>(
                paddedRowLength * sizeof(int32_t) /
                DATA_BLOCK_BYTES),
            0);
        uint64_t gatheredCount = 0;
        AscendC::GatherMask<int32_t>(
            broadcastTile, broadcastTile, 7, true,
            static_cast<uint32_t>(rowLength),
            gatherParams, gatheredCount);
        AscendC::PipeBarrier<PIPE_V>();
        broadcastRowQueue.FreeTensor(row);
    }

    __aicore__ inline void CopyInput(
        uint64_t contiguousOffset,
        uint64_t rowCount, bool x1Rows)
    {
        AscendC::GlobalTensor<int32_t>& contiguousGm =
            x1Rows ? x2Gm : x1Gm;
        AscendC::LocalTensor<int32_t> input =
            inputQueue.AllocTensor<int32_t>();
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint16_t>(
            rowCount * rowLength * sizeof(int32_t));
        copyParams.srcGap = 0;
        copyParams.dstGap = 0;
        AscendC::DataCopyPad(
            input, contiguousGm[contiguousOffset], copyParams,
            {false, 0, 0, 0});
        inputQueue.EnQue<int32_t>(input);
    }

    __aicore__ inline void ComputePredicate(
        uint64_t rowCount, bool x1Rows)
    {
        (void)rowCount;
        AscendC::LocalTensor<int32_t> input =
            inputQueue.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> broadcastTile =
            broadcastTileCache.Get<int32_t>();
        AscendC::LocalTensor<int8_t> predicate =
            predicateScratch.Get<int8_t>()[VECTOR_ALIGN_BYTES];
        if (x1Rows) {
            AscendC::Min(
                input, broadcastTile, input,
                static_cast<int32_t>(batchDataNum));
        } else {
            AscendC::Max(
                input, input, broadcastTile,
                static_cast<int32_t>(batchDataNum));
        }
        AscendC::Compare(
            predicate, broadcastTile, input,
            AscendC::CMPMODE::EQ,
            static_cast<uint32_t>(batchDataNum));
        inputQueue.FreeTensor(input);
    }

    __aicore__ inline void MaterializePredicate(
        AscendC::LocalTensor<int8_t> output,
        uint64_t outputSlot, uint64_t rowCount)
    {
        (void)rowCount;
        const uint64_t processDataNum = batchDataNum;
        AscendC::LocalTensor<int8_t> predicate =
            predicateScratch.Get<int8_t>()[VECTOR_ALIGN_BYTES];
        AscendC::LocalTensor<int8_t> outputChunk =
            output[outputSlot * batchDataNum];
        AscendC::LocalTensor<half> result =
            outputChunk.ReinterpretCast<half>();
        AscendC::LocalTensor<half> zeroBlock =
            predicateScratch.Get<half>();
        const AscendC::BinaryRepeatParams repeatParams(
            1, 1, 0, 8, 0, 0);
        const uint8_t repeatTimes = static_cast<uint8_t>(
            processDataNum /
            (VECTOR_ALIGN_BYTES / sizeof(half)));
        AscendC::Select(
            result, predicate, zeroBlock,
            repeatTimes, repeatParams);
        // Narrow in place.  Each slot begins one compact batch behind its
        // fp16 window, so forward Cast cannot overwrite a later result.
        AscendC::Cast(
            outputChunk, result, AscendC::RoundMode::CAST_NONE,
            processDataNum);
    }

    __aicore__ inline void CopyOutput(
        uint64_t outputOffset, uint64_t rowCount,
        uint64_t batchAdvance)
    {
        AscendC::LocalTensor<int8_t> output =
            outputQueue.DeQue<int8_t>();
        const uint64_t fullGroupCount =
            rowCount / ROW_BATCH;
        const uint64_t tailRows =
            rowCount % ROW_BATCH;
        const uint64_t groupLogicalBytes =
            ROW_BATCH * rowLength;
        const uint64_t groupPaddedBytes =
            (groupLogicalBytes + DATA_BLOCK_BYTES - 1) /
            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        const uint64_t groupVectorElements =
            batchDataNum;
        if (fullGroupCount != 0) {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount =
                static_cast<uint16_t>(fullGroupCount);
            copyParams.blockLen = static_cast<uint16_t>(
                groupLogicalBytes);
            copyParams.srcGap = static_cast<uint16_t>(
                (groupVectorElements - groupPaddedBytes) /
                DATA_BLOCK_BYTES);
            copyParams.dstGap = static_cast<uint16_t>(
                (batchAdvance - 1) * groupLogicalBytes);
            AscendC::DataCopyPad(
                yGm[outputOffset], output, copyParams);
        }
        if (tailRows != 0) {
            // Interleaved lanes are admitted only for whole ROW_BATCH
            // groups, so the tail form remains the original contiguous path.
            AscendC::DataCopyParams tailParams;
            tailParams.blockCount = 1;
            tailParams.blockLen = static_cast<uint16_t>(
                tailRows * rowLength);
            tailParams.srcGap = 0;
            tailParams.dstGap = 0;
            AscendC::DataCopyPad(
                yGm[
                    outputOffset +
                    fullGroupCount * ROW_BATCH *
                        rowLength],
                output[fullGroupCount * groupVectorElements],
                tailParams);
        }
        outputQueue.FreeTensor(output);
    }

    __aicore__ inline void InitPredicateConstants()
    {
        constexpr uint32_t halfRepeatElements =
            VECTOR_ALIGN_BYTES / sizeof(half);
        AscendC::LocalTensor<half> zeroBlock =
            predicateScratch.Get<half>();
        AscendC::Duplicate<half>(
            zeroBlock, (half)1, halfRepeatElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetCmpMask(zeroBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<half>(
            zeroBlock, (half)0, halfRepeatElements);
    }

    __aicore__ inline void CalcInputOffsets(
        uint64_t outputIndex, bool x1Rows,
        uint64_t& broadcastOffset, uint64_t& contiguousOffset)
    {
        const uint64_t* broadcastStride =
            x1Rows ? x1Stride : x2Stride;
        const uint64_t* contiguousStride =
            x1Rows ? x2Stride : x1Stride;
        uint64_t remaining = outputIndex;
        for (int32_t index = static_cast<int32_t>(rank) - 1;
             index >= 0; --index) {
            const uint32_t dimIndex =
                static_cast<uint32_t>(index);
            const uint64_t dim = outShape[dimIndex];
            const uint64_t coordinate = remaining % dim;
            remaining /= dim;
            broadcastOffset +=
                coordinate * broadcastStride[dimIndex];
            contiguousOffset +=
                coordinate * contiguousStride[dimIndex];
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 2> inputQueue;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1>
        broadcastRowQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC>
        broadcastTileCache;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC>
        predicateScratch;
    AscendC::GlobalTensor<int32_t> x1Gm;
    AscendC::GlobalTensor<int32_t> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint64_t coreDataNum = 0;
    uint64_t batchDataNum = 0;
    uint64_t paddedRowLength = 0;
    uint64_t rowLength = 0;
    uint64_t rowCount = 0;
    uint64_t naturalGroupCount = 0;
    uint32_t tailIndex = 0;
    uint32_t rowIndex = 0;
    uint32_t rank = 0;
    uint32_t int32CohortSize = 1;
    bool x1Rows = false;
    uint64_t outShape[GREATER_MAX_BROADCAST_DIM] = {};
    uint64_t x1Stride[GREATER_MAX_BROADCAST_DIM] = {};
    uint64_t x2Stride[GREATER_MAX_BROADCAST_DIM] = {};
};

} // namespace NsGreater

#endif // GREATER_INT32_DIRECT_ROW_PIPELINE_H__
