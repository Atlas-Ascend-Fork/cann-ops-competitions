/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __GREATER_INT32_CACHED_ROWS_H__
#define __GREATER_INT32_CACHED_ROWS_H__

#include "greater.h"

namespace NsGreater {

class GreaterInt32CachedRows {
public:
    __aicore__ inline GreaterInt32CachedRows() {}

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
            blockOffset = blockIdx * bigCoreDataNum;
        } else {
            coreDataNum = smallCoreDataNum;
            blockOffset = tailBlockNum * bigCoreDataNum +
                (blockIdx - tailBlockNum) * smallCoreDataNum;
        }

        this->rank = rank;
        for (uint32_t index = 0; index < GREATER_MAX_BROADCAST_DIM; ++index) {
            this->outShape[index] = outShape[index];
            this->x1Stride[index] = x1Stride[index];
            this->x2Stride[index] = x2Stride[index];
        }
        tailIndex = rank - 1;
        rowIndex = rank - 2;
        this->int32CohortSize =
            int32CohortSize & GREATER_INT32_COHORT_MASK;
        useLargeTile =
            (int32CohortSize & GREATER_INT32_LARGE_TILE_FLAG) != 0;
        const uint64_t rowLength = this->outShape[tailIndex];
        paddedRowLength =
            (rowLength + DATA_BLOCK_BYTES - 1) /
            DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
        maxPaddedRowCount = ubPartDataNum / paddedRowLength;
        maxPaddedRowCount = maxPaddedRowCount < this->outShape[rowIndex] ?
            maxPaddedRowCount : this->outShape[rowIndex];
        this->ubPartDataNum = maxPaddedRowCount * paddedRowLength;

        x1Gm.SetGlobalBuffer((__gm__ int32_t*)x1);
        x2Gm.SetGlobalBuffer((__gm__ int32_t*)x2);
        yGm.SetGlobalBuffer((__gm__ int8_t*)y);
        const bool x1Rows =
            this->x1Stride[tailIndex] == 1 &&
            this->x1Stride[rowIndex] == 0;
        if (x1Rows) {
            x2Gm.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        } else {
            x1Gm.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        const uint64_t alignedTailBytes = useLargeTile ?
            0 : VECTOR_ALIGN_BYTES;
        pipe.InitBuffer(
            inputBuf,
            this->ubPartDataNum * sizeof(int32_t) + alignedTailBytes);
        pipe.InitBuffer(
            broadcastRowQueue, 1,
            paddedRowLength * sizeof(int32_t));
        pipe.InitBuffer(
            broadcastTileCache,
            this->ubPartDataNum * sizeof(int32_t) + alignedTailBytes);
        pipe.InitBuffer(
            outputQueue, 1,
            this->ubPartDataNum * sizeof(int8_t) + alignedTailBytes);
        pipe.InitBuffer(
            predicateScratch,
            useLargeTile ? VECTOR_ALIGN_BYTES :
            this->ubPartDataNum * sizeof(half) + VECTOR_ALIGN_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (coreDataNum == 0 || ubPartDataNum == 0) {
            return;
        }
        inputReady = pipe.AllocEventID<AscendC::HardEvent::MTE2_V>();
        inputReusable = pipe.AllocEventID<AscendC::HardEvent::V_MTE2>();
        InitPredicateConstants();

        if (paddedRowLength != outShape[tailIndex] &&
            outShape[rowIndex] >= INT32_COMPACT_MIN_ROW_COUNT) {
            if (CanUseCohortGroups()) {
                ProcessCohortGroups<true>();
            } else if (CanUseGridStrideGroups()) {
                ProcessGridStrideGroups<true>();
            } else {
                ProcessRows<true>();
            }
        } else {
            ProcessRows<false>();
        }

        if (inputIssued) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                static_cast<event_t>(inputReusable));
        }
        pipe.ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReady);
        pipe.ReleaseEventID<AscendC::HardEvent::V_MTE2>(inputReusable);
    }

private:
    __aicore__ inline uint64_t NaturalGroupCount()
    {
        uint64_t naturalGroupCount = 1;
        for (uint32_t index = 0; index < rowIndex; ++index) {
            naturalGroupCount *= outShape[index];
        }
        return naturalGroupCount;
    }

    __aicore__ inline bool CanUseCohortGroups()
    {
        if (int32CohortSize != 2 && int32CohortSize != 4 &&
            int32CohortSize != 5) {
            return false;
        }

        constexpr uint64_t completeGroupSplitRowLimit =
            DATA_BLOCK_BYTES * 8;
        const uint64_t rowCount = outShape[rowIndex];
        const uint64_t rowLength = outShape[tailIndex];
        const uint64_t blockCount = AscendC::GetBlockNum();
        if (rowCount >= completeGroupSplitRowLimit || rowLength == 0 ||
            maxPaddedRowCount == 0 || blockCount == 0 ||
            blockCount % int32CohortSize != 0) {
            return false;
        }

        const uint64_t tileCount =
            (rowCount + maxPaddedRowCount - 1) / maxPaddedRowCount;
        const uint64_t groupsPerWave = blockCount / int32CohortSize;
        const uint64_t naturalGroupCount = NaturalGroupCount();
        const uint64_t naturalGroupLength = rowCount * rowLength;
        return tileCount >= int32CohortSize && groupsPerWave != 0 &&
            naturalGroupCount / groupsPerWave >= 2 &&
            naturalGroupLength != 0 &&
            coreDataNum % naturalGroupLength == 0;
    }

    __aicore__ inline bool CanUseGridStrideGroups()
    {
        constexpr uint64_t completeGroupSplitRowLimit =
            DATA_BLOCK_BYTES * 8;
        if (outShape[rowIndex] >= completeGroupSplitRowLimit) {
            return false;
        }

        const uint64_t naturalGroupCount = NaturalGroupCount();
        const uint64_t blockCount = AscendC::GetBlockNum();
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        if (blockCount == 0 || blockIdx >= naturalGroupCount ||
            naturalGroupCount / blockCount < 2) {
            return false;
        }

        const uint64_t naturalGroupLength =
            outShape[rowIndex] * outShape[tailIndex];
        const uint64_t assignedGroupCount =
            (naturalGroupCount - 1 - blockIdx) / blockCount + 1;
        return naturalGroupLength != 0 &&
            coreDataNum % naturalGroupLength == 0 &&
            coreDataNum / naturalGroupLength == assignedGroupCount;
    }

    template <bool CompactRows>
    __aicore__ inline void ProcessCohortGroup(
        uint64_t groupIndex, uint64_t logicalLane)
    {
        const uint64_t rowLength = outShape[tailIndex];
        const uint64_t rowCount = outShape[rowIndex];
        const uint64_t naturalGroupLength = rowCount * rowLength;
        const uint64_t groupOffset = groupIndex * naturalGroupLength;
        const bool x1Rows =
            x1Stride[tailIndex] == 1 && x1Stride[rowIndex] == 0;
        LoadBroadcastTile<CompactRows>(groupOffset, x1Rows);

        const uint64_t tileCount =
            (rowCount + maxPaddedRowCount - 1) / maxPaddedRowCount;
        for (uint64_t tile = logicalLane; tile < tileCount;
             tile += int32CohortSize) {
            const uint64_t firstRow = tile * maxPaddedRowCount;
            const uint64_t remainingRows = rowCount - firstRow;
            const uint64_t currentRowCount =
                remainingRows < maxPaddedRowCount ?
                remainingRows : maxPaddedRowCount;
            const uint64_t logicalLength = currentRowCount * rowLength;
            const uint64_t outputOffset =
                groupOffset + firstRow * rowLength;
            processDataNum = CompactRows ?
                logicalLength : currentRowCount * paddedRowLength;
            processDataNumComputes =
                ((processDataNum * sizeof(int32_t) +
                  VECTOR_ALIGN_BYTES - 1) /
                 VECTOR_ALIGN_BYTES) *
                VECTOR_ALIGN_BYTES / sizeof(int32_t);

            LoadContiguousInput<CompactRows>(
                outputOffset, currentRowCount, logicalLength, x1Rows);
            Compute(x1Rows);
            StoreOutput<CompactRows>(
                outputOffset, currentRowCount, logicalLength);
        }
    }

    template <bool CompactRows>
    __aicore__ inline void ProcessCohortGroups()
    {
        const uint64_t blockCount = AscendC::GetBlockNum();
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t cohortSize = int32CohortSize;
        const uint64_t groupsPerWave = blockCount / cohortSize;
        const uint64_t groupSlot = blockIdx / cohortSize;
        const uint64_t physicalLane = blockIdx % cohortSize;
        const uint64_t naturalGroupCount = NaturalGroupCount();

        for (uint64_t groupIndex = groupSlot;
             groupIndex < naturalGroupCount;
             groupIndex += groupsPerWave) {
            const uint64_t logicalLane =
                (physicalLane + groupIndex % cohortSize) % cohortSize;
            ProcessCohortGroup<CompactRows>(groupIndex, logicalLane);
        }
    }

    template <bool CompactRows>
    __aicore__ inline void ProcessGridStrideGroups()
    {
        const uint64_t rowLength = outShape[tailIndex];
        const uint64_t naturalGroupLength =
            outShape[rowIndex] * rowLength;
        const uint64_t blockCount = AscendC::GetBlockNum();
        const uint64_t blockIdx = AscendC::GetBlockIdx();
        const uint64_t assignedGroupCount =
            coreDataNum / naturalGroupLength;
        const uint64_t groupAdvance = blockCount * naturalGroupLength;
        const bool x1Rows =
            x1Stride[tailIndex] == 1 && x1Stride[rowIndex] == 0;
        uint64_t groupOffset = blockIdx * naturalGroupLength;

        for (uint64_t group = 0; group < assignedGroupCount; ++group) {
            uint64_t outputOffset = groupOffset;
            uint64_t planeRows = outShape[rowIndex];
            LoadBroadcastTile<CompactRows>(outputOffset, x1Rows);

            uint64_t rowCount = planeRows < maxPaddedRowCount ?
                planeRows : maxPaddedRowCount;
            uint64_t logicalLength = rowCount * rowLength;
            processDataNum = CompactRows ?
                logicalLength : rowCount * paddedRowLength;
            processDataNumComputes =
                ((processDataNum * sizeof(int32_t) +
                  VECTOR_ALIGN_BYTES - 1) /
                 VECTOR_ALIGN_BYTES) *
                VECTOR_ALIGN_BYTES / sizeof(int32_t);
            LoadContiguousInput<CompactRows>(
                outputOffset, rowCount, logicalLength, x1Rows);

            while (planeRows > 0) {
                const uint64_t nextPlaneRows = planeRows - rowCount;
                const uint64_t nextRowCount =
                    nextPlaneRows < maxPaddedRowCount ?
                    nextPlaneRows : maxPaddedRowCount;
                const uint64_t nextLogicalLength =
                    nextRowCount * rowLength;
                Compute<CompactRows>(
                    x1Rows, nextPlaneRows != 0,
                    outputOffset + logicalLength,
                    nextRowCount, nextLogicalLength);
                StoreOutput<CompactRows>(
                    outputOffset, rowCount, logicalLength);

                outputOffset += logicalLength;
                planeRows = nextPlaneRows;
                rowCount = nextRowCount;
                logicalLength = nextLogicalLength;
                if (planeRows != 0) {
                    processDataNum = CompactRows ?
                        logicalLength : rowCount * paddedRowLength;
                    processDataNumComputes =
                        ((processDataNum * sizeof(int32_t) +
                          VECTOR_ALIGN_BYTES - 1) /
                         VECTOR_ALIGN_BYTES) *
                        VECTOR_ALIGN_BYTES / sizeof(int32_t);
                }
            }
            groupOffset += groupAdvance;
        }
    }

    template <bool CompactRows>
    __aicore__ inline void ProcessRows()
    {
        const uint64_t rowLength = outShape[tailIndex];
        const bool x1Rows =
            x1Stride[tailIndex] == 1 && x1Stride[rowIndex] == 0;
        uint64_t outputOffset = blockOffset;
        uint64_t remaining = coreDataNum;
        while (remaining >= rowLength) {
            const uint64_t currentRow =
                outputOffset / rowLength % outShape[rowIndex];
            uint64_t planeRows = outShape[rowIndex] - currentRow;
            const uint64_t remainingRows = remaining / rowLength;
            planeRows = planeRows < remainingRows ?
                planeRows : remainingRows;

            LoadBroadcastTile<CompactRows>(outputOffset, x1Rows);

            while (planeRows > 0) {
                const uint64_t rowCount = planeRows < maxPaddedRowCount ?
                    planeRows : maxPaddedRowCount;
                const uint64_t logicalLength = rowCount * rowLength;
                processDataNum = CompactRows ?
                    logicalLength : rowCount * paddedRowLength;
                processDataNumComputes =
                    ((processDataNum * sizeof(int32_t) +
                      VECTOR_ALIGN_BYTES - 1) /
                     VECTOR_ALIGN_BYTES) *
                    VECTOR_ALIGN_BYTES / sizeof(int32_t);

                LoadContiguousInput<CompactRows>(
                    outputOffset, rowCount, logicalLength, x1Rows);
                Compute(x1Rows);
                StoreOutput<CompactRows>(
                    outputOffset, rowCount, logicalLength);

                outputOffset += logicalLength;
                remaining -= logicalLength;
                planeRows -= rowCount;
            }
        }
    }

    template <bool CompactRows>
    __aicore__ inline void LoadBroadcastTile(
        uint64_t outputOffset, bool x1Rows)
    {
        const uint64_t rowLength = outShape[tailIndex];
        const uint64_t broadcastOffset = CalcInputOffset(
            outputOffset, x1Rows ? x1Stride : x2Stride);
        AscendC::LocalTensor<int32_t> row =
            broadcastRowQueue.AllocTensor<int32_t>();
        CopyContiguousLength(
            row, x1Rows ? x1Gm : x2Gm,
            broadcastOffset, rowLength);
        broadcastRowQueue.EnQue(row);
        row = broadcastRowQueue.DeQue<int32_t>();

        AscendC::LocalTensor<int32_t> broadcastTile =
            broadcastTileCache.Get<int32_t>();
        const uint32_t destinationShape[2] = {
            static_cast<uint32_t>(maxPaddedRowCount),
            static_cast<uint32_t>(paddedRowLength)};
        const uint32_t sourceShape[2] = {
            1, static_cast<uint32_t>(paddedRowLength)};
        AscendC::Broadcast<int32_t, 2, 0>(
            broadcastTile, row, destinationShape, sourceShape);

        if constexpr (CompactRows) {
            AscendC::GatherMaskParams gatherParams(
                1, static_cast<uint16_t>(maxPaddedRowCount),
                static_cast<uint16_t>(
                    paddedRowLength * sizeof(int32_t) /
                    DATA_BLOCK_BYTES),
                0);
            uint64_t gatheredCount = 0;
            AscendC::GatherMask<int32_t>(
                broadcastTile, broadcastTile, 7, true,
                static_cast<uint32_t>(rowLength),
                gatherParams, gatheredCount);
        }
        AscendC::PipeBarrier<PIPE_V>();
        broadcastRowQueue.FreeTensor(row);
    }

    template <bool CompactRows>
    __aicore__ inline void LoadContiguousInput(
        uint64_t outputOffset, uint64_t rowCount,
        uint64_t logicalLength, bool x1Rows)
    {
        if (inputIssued) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(
                static_cast<event_t>(inputReusable));
        }
        AscendC::LocalTensor<int32_t> input = inputBuf.Get<int32_t>();
        const uint64_t contiguousOffset = CalcInputOffset(
            outputOffset, x1Rows ? x2Stride : x1Stride);
        AscendC::GlobalTensor<int32_t>& contiguousGm =
            x1Rows ? x2Gm : x1Gm;

        if constexpr (CompactRows) {
            CopyContiguousLength(
                input, contiguousGm, contiguousOffset, logicalLength);
        } else {
            const uint64_t rowLength = outShape[tailIndex];
            const uint64_t rowBytes = rowLength * sizeof(int32_t);
            const uint64_t alignedRowBytes =
                (rowBytes + DATA_BLOCK_BYTES - 1) /
                DATA_BLOCK_BYTES * DATA_BLOCK_BYTES;
            const uint64_t paddedRowBytes =
                paddedRowLength * sizeof(int32_t);
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(rowCount);
            copyParams.blockLen = static_cast<uint16_t>(rowBytes);
            copyParams.srcGap = 0;
            copyParams.dstGap = static_cast<uint16_t>(
                (paddedRowBytes - alignedRowBytes) /
                DATA_BLOCK_BYTES);
            AscendC::DataCopyPad(
                input, contiguousGm[contiguousOffset], copyParams,
                {false, 0, 0, 0});
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(
            static_cast<event_t>(inputReady));
        inputIssued = true;
    }

    template <bool CompactRows = false>
    __aicore__ inline void Compute(
        bool x1Rows, bool prefetchNext = false,
        uint64_t nextOutputOffset = 0, uint64_t nextRowCount = 0,
        uint64_t nextLogicalLength = 0)
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(
            static_cast<event_t>(inputReady));
        AscendC::LocalTensor<int32_t> input = inputBuf.Get<int32_t>();
        AscendC::LocalTensor<int32_t> broadcastTile =
            broadcastTileCache.Get<int32_t>();
        AscendC::LocalTensor<int8_t> output =
            outputQueue.AllocTensor<int8_t>();

        if (x1Rows) {
            AscendC::Min(
                input, broadcastTile, input, processDataNumComputes);
        } else {
            AscendC::Max(
                input, input, broadcastTile, processDataNumComputes);
        }
        AscendC::Compare(
            output, broadcastTile, input,
            AscendC::CMPMODE::EQ, processDataNumComputes);
        if (useLargeTile) {
            MaterializeAliasedPredicate(output, input);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                static_cast<event_t>(inputReusable));
        } else {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(
                static_cast<event_t>(inputReusable));
            // Compare is the final consumer of the raw input TBuf.  Start the
            // next transfer immediately after that dependency instead of
            // draining current Select/Cast and MTE3 first.  The next MTE2 and
            // current predicate materialization touch disjoint UB ranges.
            if (prefetchNext) {
                LoadContiguousInput<CompactRows>(
                    nextOutputOffset, nextRowCount,
                    nextLogicalLength, x1Rows);
            }
            MaterializePredicate(output);
        }
        outputQueue.EnQue<int8_t>(output);
    }

    __aicore__ inline void MaterializePredicate(
        AscendC::LocalTensor<int8_t> output)
    {
        constexpr uint64_t halfRepeatElements =
            VECTOR_ALIGN_BYTES / sizeof(half);
        AscendC::LocalTensor<half> result =
            predicateScratch.Get<half>();
        AscendC::LocalTensor<half> zeroBlock = result[ubPartDataNum];
        const AscendC::BinaryRepeatParams repeatParams(
            1, 1, 0, 8, 0, 0);
        const uint8_t repeatTimes = static_cast<uint8_t>(
            (processDataNumComputes + halfRepeatElements - 1) /
            halfRepeatElements);
        AscendC::Select(
            result, output, zeroBlock, repeatTimes, repeatParams);
        AscendC::Cast(
            output, result, AscendC::RoundMode::CAST_NONE,
            processDataNumComputes);
    }

    __aicore__ inline void MaterializeAliasedPredicate(
        AscendC::LocalTensor<int8_t> output,
        AscendC::LocalTensor<int32_t> input)
    {
        constexpr uint64_t halfRepeatElements =
            VECTOR_ALIGN_BYTES / sizeof(half);
        AscendC::LocalTensor<half> result =
            input.template ReinterpretCast<half>();
        AscendC::LocalTensor<half> zeroBlock =
            predicateScratch.Get<half>();
        const AscendC::BinaryRepeatParams repeatParams(
            1, 1, 0, 8, 0, 0);
        const uint8_t repeatTimes = static_cast<uint8_t>(
            (processDataNumComputes + halfRepeatElements - 1) /
            halfRepeatElements);
        AscendC::Select(
            result, output, zeroBlock, repeatTimes, repeatParams);
        AscendC::Cast(
            output, result, AscendC::RoundMode::CAST_NONE,
            processDataNumComputes);
    }

    template <bool CompactRows>
    __aicore__ inline void StoreOutput(
        uint64_t outputOffset, uint64_t rowCount,
        uint64_t logicalLength)
    {
        AscendC::LocalTensor<int8_t> output =
            outputQueue.DeQue<int8_t>();
        AscendC::DataCopyParams copyParams;
        if constexpr (CompactRows) {
            copyParams.blockCount = 1;
            copyParams.blockLen =
                static_cast<uint16_t>(logicalLength);
        } else {
            copyParams.blockCount = static_cast<uint16_t>(rowCount);
            copyParams.blockLen = static_cast<uint16_t>(
                outShape[tailIndex]);
        }
        copyParams.srcGap = 0;
        copyParams.dstGap = 0;
        AscendC::DataCopyPad(
            yGm[outputOffset], output, copyParams);
        outputQueue.FreeTensor(output);
    }

    __aicore__ inline void InitPredicateConstants()
    {
        constexpr uint32_t halfRepeatElements =
            VECTOR_ALIGN_BYTES / sizeof(half);
        AscendC::LocalTensor<half> zeroBlock =
            useLargeTile ? predicateScratch.Get<half>() :
            predicateScratch.Get<half>()[ubPartDataNum];
        AscendC::Duplicate<half>(
            zeroBlock, (half)1, halfRepeatElements);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetCmpMask(zeroBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate<half>(
            zeroBlock, (half)0, halfRepeatElements);
    }

    __aicore__ inline uint64_t CalcInputOffset(
        uint64_t outputIndex, const uint64_t* stride)
    {
        uint64_t inputOffset = 0;
        uint64_t remaining = outputIndex;
        for (int32_t index = static_cast<int32_t>(rank) - 1;
             index >= 0; --index) {
            const uint32_t dimIndex = static_cast<uint32_t>(index);
            const uint64_t dim = outShape[dimIndex];
            const uint64_t coordinate = remaining % dim;
            remaining /= dim;
            inputOffset += coordinate * stride[dimIndex];
        }
        return inputOffset;
    }

    __aicore__ inline void CopyContiguousLength(
        AscendC::LocalTensor<int32_t> destination,
        AscendC::GlobalTensor<int32_t>& source,
        uint64_t sourceOffset, uint64_t length)
    {
        if (useLargeTile) {
            AscendC::DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(
                length * sizeof(int32_t));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(
                destination, source[sourceOffset], copyParams,
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        } else {
            AscendC::DataCopyParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint16_t>(
                length * sizeof(int32_t));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            AscendC::DataCopyPad(
                destination, source[sourceOffset], copyParams,
                {false, 0, 0, 0});
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> inputBuf;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> broadcastRowQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> broadcastTileCache;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outputQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> predicateScratch;
    AscendC::GlobalTensor<int32_t> x1Gm;
    AscendC::GlobalTensor<int32_t> x2Gm;
    AscendC::GlobalTensor<int8_t> yGm;
    uint64_t coreDataNum = 0;
    uint64_t blockOffset = 0;
    uint64_t ubPartDataNum = 0;
    uint64_t processDataNum = 0;
    uint64_t processDataNumComputes = 0;
    uint64_t paddedRowLength = 0;
    uint64_t maxPaddedRowCount = 0;
    uint32_t tailIndex = 0;
    uint32_t rowIndex = 0;
    uint32_t rank = 0;
    uint32_t int32CohortSize = 1;
    uint64_t outShape[GREATER_MAX_BROADCAST_DIM] = {};
    uint64_t x1Stride[GREATER_MAX_BROADCAST_DIM] = {};
    uint64_t x2Stride[GREATER_MAX_BROADCAST_DIM] = {};
    AscendC::TEventID inputReady;
    AscendC::TEventID inputReusable;
    bool inputIssued = false;
    bool useLargeTile = false;
};

} // namespace NsGreater

#endif // GREATER_INT32_CACHED_ROWS_H__
