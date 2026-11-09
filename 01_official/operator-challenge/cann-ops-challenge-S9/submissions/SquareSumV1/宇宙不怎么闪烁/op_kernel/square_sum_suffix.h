#pragma once
// Private SquareSumKernel<T> member definitions; included inside the class.

    // Shared with the full-row BF16 microbatch; callers provide aligned FP32
    // rows, <=255 repeats/stride blocks, and a completed square operation.
    __aicore__ inline void FoldAlignedSuffixRows(LocalTensor<float> sumsLocal,
                                                LocalTensor<float> calcLocal,
                                                uint32_t reduceCount, uint32_t rowCount,
                                                uint32_t rowPitch)
    {
        const uint8_t rowStrideBlocks = static_cast<uint8_t>(
            rowPitch * sizeof(float) / 32U);
        uint32_t active = reduceCount;
        while (active > kPlaneReduceChunkElems) {
            // Round the half UP to 64 floats: all UB operand starts
            // stay 32B aligned, and an odd tail (e.g. 129) is kept.
            const uint32_t half = ((active + 127U) / 128U) * 64U;
            const uint32_t remaining = active - half;
            for (uint32_t col = 0U; col < remaining; col += kPlaneReduceChunkElems) {
                Add<float>(
                    calcLocal[col], calcLocal[col], calcLocal[col + half],
                    static_cast<uint64_t>(MinU32(kPlaneReduceChunkElems,
                                                 remaining - col)),
                    static_cast<uint8_t>(rowCount),
                    {1U, 1U, 1U, rowStrideBlocks, rowStrideBlocks,
                     rowStrideBlocks});
            }
            PipeBarrier<PIPE_V>();
            active = half;
        }
        WholeReduceSum<float>(
            sumsLocal, calcLocal, static_cast<int32_t>(active),
            static_cast<int32_t>(rowCount), 1, 1,
            static_cast<int32_t>(rowStrideBlocks));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitSuffixBuffers()
    {
        if constexpr (std::is_same_v<T, float>) {
            const uint32_t alignedTileF = AlignElems32<float>(tileElems);
            const uint32_t alignedOutF = AlignElems32<float>(kSuffixOutTileElems);
            pipe->InitBuffer(inQueue, kBufferNum, alignedTileF * sizeof(float));
            pipe->InitBuffer(outQueue, kBufferNum, alignedOutF * sizeof(float));
            pipe->InitBuffer(workBuf, alignedTileF * sizeof(float));
            pipe->InitBuffer(sumBuf, AlignElems32<float>(1U) * sizeof(float));
            return;
        }
        const uint32_t alignedTileT = AlignElems32<T>(tileElems);
        const uint32_t alignedTileF = AlignElems32<float>(tileElems);
        const uint32_t alignedOutT = AlignElems32<T>(kSuffixOutTileElems);
        const uint32_t alignedOutF = AlignElems32<float>(kSuffixOutTileElems);
        pipe->InitBuffer(inQueue, kBufferNum, alignedTileT * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, alignedOutT * sizeof(T));
        pipe->InitBuffer(calcBuf, alignedTileF * sizeof(float));
        pipe->InitBuffer(workBuf, alignedTileF * sizeof(float));
        pipe->InitBuffer(sumBuf, AlignElems32<float>(kSuffixBatchRows) * sizeof(float));
        pipe->InitBuffer(valueFloatBuf, alignedOutF * sizeof(float));
    }

    __aicore__ inline void InitContiguousBlockBuffers()
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceSize);
        const uint32_t batchElems = accTileElems * rowPitch;
        pipe->InitBuffer(inQueue, 2U, batchElems * sizeof(T));
        pipe->InitBuffer(outQueue, 2U, AlignElems32<T>(accTileElems) * sizeof(T));
        pipe->InitBuffer(calcBuf, batchElems * sizeof(float));
        pipe->InitBuffer(squareBuf, batchElems * sizeof(float));
        pipe->InitBuffer(sumBuf, AlignElems32<float>(accTileElems) * sizeof(float));
        pipe->InitBuffer(accBuf, AlignElems32<float>(accTileElems) * sizeof(float));
    }

    __aicore__ inline bool CanProcessLargeSuffixChunked() const
    {
        if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, half> ||
                        std::is_same_v<T, bfloat16_t>)) {
            return false;
        } else {
            // The source-row gap in the 2D DMA is expressed in bytes.  Leave
            // unusually large logical rows on the old tiled fallback rather
            // than risking a narrowed stride field.
            return reduceSize > kSuffixBatchMaxReduce &&
                   reduceSize <= (0xFFFFFFFFU / sizeof(T));
        }
    }

    __aicore__ inline bool UseFusedSuffix() const
    {
        return std::is_same_v<T, bfloat16_t>;
    }

    __aicore__ inline void InitLargeSuffixChunkedBuffers()
    {
        const bool fusedSuffix = UseFusedSuffix();
        const uint32_t chunkElems = fusedSuffix
            ? square_sum_policy::kFusedSuffixCols : kLargeSuffixChunkElems;
        const uint32_t batchRows = fusedSuffix
            ? fusedSuffixRows : kLargeSuffixBatchRows;
        const uint32_t inputRowPitch = AlignElems32<T>(chunkElems);
        const uint32_t inputElems = batchRows * inputRowPitch;
        const uint32_t floatElems = batchRows * inputRowPitch;
        const uint32_t outputElems = AlignElems32<T>(batchRows);
        // Final BF16 Cast rounds its count to sixteen elements, so the
        // FP32 source must cover that count too (e.g. 23 rows need 32 floats).
        const uint32_t sumElems = AlignElems32<T>(batchRows);

        // The two VECIN slots retain the current stripe while MTE2 fetches
        // the next one.  VECOUT is also double-buffered so MTE3 can retire a
        // previous output-row batch while the next batch is reduced.
        const uint32_t inputPadBytes = fusedSuffix
            ? square_sum_policy::kFusedSuffixInputPadBytes : 0U;
        // Physical per-slot padding shifts the second BF16 slot by 32B; DMA
        // and Cast still access only the logical stripe, never this padding.
        pipe->InitBuffer(suffixChunkInQueue, 2U, inputElems * sizeof(T) + inputPadBytes);
        pipe->InitBuffer(suffixChunkOutQueue, 2U, outputElems * sizeof(T));
        pipe->InitBuffer(suffixChunkCalcBuf, floatElems * sizeof(float));
        if (!fusedSuffix) {
            pipe->InitBuffer(suffixChunkSquareBuf, floatElems * sizeof(float));
        }
        pipe->InitBuffer(suffixChunkSumBuf, sumElems * sizeof(float));
        pipe->InitBuffer(suffixChunkAccBuf, sumElems * sizeof(float));
    }

    __aicore__ inline void CopyLargeSuffixChunk(uint32_t outputBase, uint32_t rowCount,
                                                uint32_t reduceBase, uint32_t reduceCount)
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceCount);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceCount);
        LocalTensor<T> inputLocal = suffixChunkInQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rowCount),
            static_cast<uint32_t>(reduceCount * sizeof(T)),
            static_cast<uint32_t>((reduceSize - reduceCount) * sizeof(T)), 0U, 0U};
        DataCopyPadExtParams<T> padParams{
            rightPadding != 0U, 0U, rightPadding, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[outputBase * reduceSize + reduceBase],
                    copyParams, padParams);
        suffixChunkInQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessLargeSuffixChunked()
    {
        const bool fusedSuffix = UseFusedSuffix();
        const uint32_t chunkElems = fusedSuffix
            ? square_sum_policy::kFusedSuffixCols : kLargeSuffixChunkElems;
        const uint32_t batchRows = fusedSuffix
            ? fusedSuffixRows : kLargeSuffixBatchRows;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputRows = 0U;
        bool hasPendingOutput = false;

        for (uint32_t outputBase = start; outputBase < end;
             outputBase += batchRows) {
            const uint32_t rowCount = MinU32(batchRows, end - outputBase);
            uint32_t reduceBase = 0U;
            uint32_t reduceCount = MinU32(chunkElems, reduceSize);
            CopyLargeSuffixChunk(outputBase, rowCount, reduceBase, reduceCount);

            // MTE3 starts before this batch's first DeQue, so it can overlap
            // both the first MTE2 stripe and all following vector reductions.
            if (hasPendingOutput) {
                LocalTensor<T> readyOut = suffixChunkOutQueue.DeQue<T>();
                CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
                suffixChunkOutQueue.FreeTensor(readyOut);
            }

            LocalTensor<float> accLocal = suffixChunkAccBuf.Get<float>();
            LocalTensor<float> sumsLocal = suffixChunkSumBuf.Get<float>();
            Duplicate(accLocal, 0.0F, AlignElems32<T>(rowCount));

            while (true) {
                LocalTensor<T> ready = suffixChunkInQueue.DeQue<T>();
                const uint32_t nextReduceBase = reduceBase + reduceCount;
                const bool hasNext = nextReduceBase < reduceSize;
                uint32_t nextReduceCount = 0U;
                if (hasNext) {
                    nextReduceCount = MinU32(chunkElems, reduceSize - nextReduceBase);
                    CopyLargeSuffixChunk(outputBase, rowCount, nextReduceBase,
                                         nextReduceCount);
                }

                const uint32_t rowPitch = AlignElems32<T>(reduceCount);
                const uint32_t batchElems = rowCount * rowPitch;
                LocalTensor<float> calcLocal = suffixChunkCalcBuf.Get<float>();
                if (fusedSuffix) {
                    Cast(calcLocal, ready, RoundMode::CAST_NONE, batchElems);
                    PipeBarrier<PIPE_V>();
                    Mul(calcLocal, calcLocal, calcLocal, batchElems);
                    PipeBarrier<PIPE_V>();
                    FoldAlignedSuffixRows(sumsLocal, calcLocal, reduceCount,
                                          rowCount, rowPitch);
                } else {
                    LocalTensor<float> squareLocal = suffixChunkSquareBuf.Get<float>();
                    SquarePlaneBatch(squareLocal, ready, calcLocal, batchElems);
                    WholeReduceSum<float>(
                        sumsLocal, squareLocal,
                        static_cast<int32_t>(reduceCount),
                        static_cast<int32_t>(rowCount), 1, 1,
                        static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
                }
                Add(accLocal, sumsLocal, accLocal, rowCount);
                if (fusedSuffix) PipeBarrier<PIPE_V>();  // finish scratch reads before reuse
                suffixChunkInQueue.FreeTensor(ready);

                if (!hasNext) {
                    break;
                }
                reduceBase = nextReduceBase;
                reduceCount = nextReduceCount;
            }

            LocalTensor<T> outputLocal = suffixChunkOutQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, accLocal, rowCount);
            suffixChunkOutQueue.EnQue(outputLocal);
            pendingOutputOffset = outputBase;
            pendingOutputRows = rowCount;
            hasPendingOutput = true;
        }

        if (hasPendingOutput) {
            LocalTensor<T> readyOut = suffixChunkOutQueue.DeQue<T>();
            CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
            suffixChunkOutQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void ProcessContiguousBlockReduce()
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceSize);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceSize);
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        if (start >= end) {
            return;
        }

        uint32_t prefetchedTask = start;
        uint32_t prefetchedOffset = prefetchedTask * accTileElems;
        uint32_t prefetchedRows = MinU32(accTileElems, outputSize - prefetchedOffset);
        {
            LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
            CopyContiguousBlockInput(inputLocal, prefetchedOffset * reduceSize,
                                     prefetchedRows, rowPitch, rightPadding);
            inQueue.EnQue(inputLocal);
        }
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputRows = 0U;
        bool hasPendingOutput = false;

        for (uint32_t task = start; task < end; ++task) {
            const uint32_t outputOffset = prefetchedOffset;
            const uint32_t rowCount = prefetchedRows;
            const uint32_t batchElems = rowCount * rowPitch;

            LocalTensor<T> ready = inQueue.DeQue<T>();
            if (task + 1U < end) {
                prefetchedTask = task + 1U;
                prefetchedOffset = prefetchedTask * accTileElems;
                prefetchedRows = MinU32(accTileElems, outputSize - prefetchedOffset);
                LocalTensor<T> nextInput = inQueue.AllocTensor<T>();
                CopyContiguousBlockInput(nextInput, prefetchedOffset * reduceSize,
                                         prefetchedRows, rowPitch, rightPadding);
                inQueue.EnQue(nextInput);
            }
            // The two VECOUT buffers allow the preceding result to retire on
            // MTE3 while this task squares and reduces its prefetched rows.
            // Queue events retain the ownership dependency for each buffer.
            if (hasPendingOutput) {
                LocalTensor<T> readyOut = outQueue.DeQue<T>();
                CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
                outQueue.FreeTensor(readyOut);
            }
            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            LocalTensor<float> squareLocal = squareBuf.Get<float>();
            LocalTensor<float> sumsLocal = sumBuf.Get<float>();
            SquarePlaneBatch(squareLocal, ready, calcLocal, batchElems);
            const bool usedPackedChunks =
                TryReduceContiguousFloatChunks(sumsLocal, squareLocal, rowCount, rowPitch);
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            if (!usedPackedChunks) {
                LocalTensor<float> accLocal = accBuf.Get<float>();
                Duplicate(accLocal, 0.0F, AlignElems32<float>(rowCount));
                if (reduceSize <= kPlaneReduceChunkElems) {
                    WholeReduceSum<float>(sumsLocal, squareLocal,
                                          static_cast<int32_t>(reduceSize),
                                          static_cast<int32_t>(rowCount), 1, 1,
                                          static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
                    Add(accLocal, sumsLocal, accLocal, rowCount);
                } else {
                    for (uint32_t reduceBase = 0U; reduceBase < reduceSize;
                         reduceBase += kPlaneReduceChunkElems) {
                        const uint32_t reduceCount =
                            MinU32(kPlaneReduceChunkElems, reduceSize - reduceBase);
                        WholeReduceSum<float>(sumsLocal, squareLocal[reduceBase],
                                              static_cast<int32_t>(reduceCount),
                                              static_cast<int32_t>(rowCount), 1, 1,
                                          static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
                        Add(accLocal, sumsLocal, accLocal, rowCount);
                    }
                }
                ConvertPlaneOutput(outputLocal, accLocal, rowCount);
            } else {
                ConvertPlaneOutput(outputLocal, sumsLocal, rowCount);
            }
            inQueue.FreeTensor(ready);

            outQueue.EnQue(outputLocal);
            pendingOutputOffset = outputOffset;
            pendingOutputRows = rowCount;
            hasPendingOutput = true;
        }
        if (hasPendingOutput) {
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            CopyContiguousBlockOutput(pendingOutputOffset, readyOut, pendingOutputRows);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline bool TryReduceContiguousFloatChunks(LocalTensor<float> sumsLocal,
                                                           LocalTensor<float> squareLocal,
                                                           uint32_t rowCount,
                                                           uint32_t rowPitch)
    {
        if constexpr (!std::is_same_v<T, float>) {
            (void)sumsLocal;
            (void)squareLocal;
            (void)rowCount;
            (void)rowPitch;
            return false;
        } else {
            constexpr uint32_t kChunkElems = kPlaneReduceChunkElems;
            constexpr uint32_t kMaxRepeat = 255U;
            if (rowPitch != reduceSize || reduceSize < kChunkElems * 8U ||
                (reduceSize % kChunkElems) != 0U) {
                return false;
            }
            const uint32_t chunkCount = reduceSize / kChunkElems;
            // The second stage reads one aligned partial row per output row.
            if (chunkCount > kChunkElems || (chunkCount & 7U) != 0U) {
                return false;
            }
            const uint32_t rowsPerCall = kMaxRepeat / chunkCount;
            if (rowsPerCall == 0U) {
                return false;
            }

            LocalTensor<float> partialLocal = calcBuf.Get<float>();
            for (uint32_t rowBase = 0U; rowBase < rowCount; rowBase += rowsPerCall) {
                const uint32_t rows = MinU32(rowsPerCall, rowCount - rowBase);
                RepeatReduceSum<float>(partialLocal, squareLocal[rowBase * rowPitch],
                                       static_cast<int32_t>(rows * chunkCount),
                                       static_cast<int32_t>(kChunkElems), 0, 1, 1,
                                       static_cast<int32_t>(kChunkElems / 8U));
                WholeReduceSum<float>(sumsLocal[rowBase], partialLocal,
                                      static_cast<int32_t>(chunkCount),
                                      static_cast<int32_t>(rows), 1, 1,
                                      static_cast<int32_t>(chunkCount / 8U));
            }
            return true;
        }
    }

    __aicore__ inline void CopyContiguousBlockInput(LocalTensor<T>& inputLocal,
                                                    uint32_t gmOffset, uint32_t rows,
                                                    uint32_t rowPitch, uint8_t rightPadding)
    {
        if (rightPadding == 0U) {
            DataCopy(inputLocal, inputGm[gmOffset], rows * rowPitch);
            return;
        }
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows),
                                     static_cast<uint32_t>(reduceSize * sizeof(T)),
                                     0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{true, 0U, rightPadding, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
    }

    __aicore__ inline void CopyContiguousBlockOutput(uint32_t outputOffset,
                                                     LocalTensor<T>& outputLocal,
                                                     uint32_t rows)
    {
        if (rows == AlignElems32<T>(rows)) {
            DataCopy(outputGm[outputOffset], outputLocal, rows);
            return;
        }
        DataCopyExtParams outParams{1U, static_cast<uint32_t>(rows * sizeof(T)),
                                     0U, 0U, 0U};
        DataCopyPad(outputGm[outputOffset], outputLocal, outParams);
    }

    __aicore__ inline void ProcessSmallSuffixRows()
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceSize);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceSize);
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        for (uint32_t outBase = start; outBase < end; outBase += kSuffixBatchRows) {
            const uint32_t outCount = MinU32(kSuffixBatchRows, end - outBase);
            const uint32_t batchElems = outCount * rowPitch;
            LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
            DataCopyExtParams copyParams{static_cast<uint16_t>(outCount),
                                         static_cast<uint32_t>(reduceSize * sizeof(T)), 0U, 0U, 0U};
            DataCopyPadExtParams<T> padParams{rightPadding != 0U, 0U, rightPadding, static_cast<T>(0)};
            DataCopyPad(inputLocal, inputGm[outBase * reduceSize], copyParams, padParams);
            inQueue.EnQue(inputLocal);

            LocalTensor<T> ready = inQueue.DeQue<T>();
            LocalTensor<T> outputLocal;
            LocalTensor<float> sumsLocal;
            LocalTensor<float> squareLocal;
            if constexpr (std::is_same_v<T, float>) {
                outputLocal = outQueue.AllocTensor<T>();
                sumsLocal = outputLocal;
                squareLocal = workBuf.Get<float>();
                Mul(squareLocal, ready, ready, batchElems);
            } else {
                LocalTensor<float> calcLocal = calcBuf.Get<float>();
                Cast(calcLocal, ready, RoundMode::CAST_NONE, batchElems);
                if constexpr (std::is_same_v<T, bfloat16_t>) {
                    squareLocal = calcLocal;
                    sumsLocal = sumBuf.Get<float>();
                } else {
                    squareLocal = workBuf.Get<float>();
                    sumsLocal = valueFloatBuf.Get<float>();
                }
                Mul(squareLocal, calcLocal, calcLocal, batchElems);
            }
            WholeReduceSum<float>(sumsLocal, squareLocal, static_cast<int32_t>(reduceSize),
                                  static_cast<int32_t>(outCount), 1, 1,
                                  static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
            inQueue.FreeTensor(ready);
            if constexpr (!std::is_same_v<T, float>) {
                outputLocal = outQueue.AllocTensor<T>();
                constexpr RoundMode rounding = std::is_same_v<T, bfloat16_t>
                    ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
                Cast(outputLocal, sumsLocal, rounding, AlignElems32<T>(outCount));
            }
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(outCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outBase], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void ProcessSuffixReduce()
    {
        if (reduceSize != 0U && reduceSize <= kSuffixBatchMaxReduce) {
            ProcessSmallSuffixRows();
        } else {
            ProcessScalarReduction(true);
        }
    }
