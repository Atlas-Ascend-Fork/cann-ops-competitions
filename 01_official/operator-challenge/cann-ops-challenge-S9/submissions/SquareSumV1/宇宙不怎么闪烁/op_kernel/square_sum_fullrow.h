#pragma once
// Private SquareSumKernel<T> member definitions; BF16 suffix only.

    __aicore__ inline void InitFullRowSuffixBuffers()
    {
        const uint32_t outputPitch = AlignElems32<T>(fullRowSuffixRows);
        // Reuse the suffix queues, but allocate no stripe, square or acc buffer.
        pipe->InitBuffer(suffixChunkInQueue, 2U,
                         fullRowSuffixRows * fullRowSuffixPitch * sizeof(T));
        pipe->InitBuffer(suffixChunkOutQueue, 2U, outputPitch * sizeof(T));
        pipe->InitBuffer(suffixChunkCalcBuf,
                         square_sum_policy::kFullRowSuffixMicroRows * fullRowSuffixPitch * sizeof(float));
        pipe->InitBuffer(suffixChunkSumBuf, outputPitch * sizeof(float));
    }

    __aicore__ inline void CopyFullRowSuffixInput(uint32_t outputBase, uint32_t rowCount)
    {
        LocalTensor<T> inputLocal = suffixChunkInQueue.AllocTensor<T>();
        const uint8_t rightPadding = static_cast<uint8_t>(fullRowSuffixPitch - reduceSize);
        DataCopyExtParams copyParams{static_cast<uint16_t>(rowCount),
                                     static_cast<uint32_t>(reduceSize * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{
            rightPadding != 0U, 0U, rightPadding, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[outputBase * reduceSize], copyParams, padParams);
        suffixChunkInQueue.EnQue(inputLocal);
    }

    __aicore__ inline void CopyFullRowSuffixOutput(uint32_t outputBase, uint32_t rowCount)
    {
        LocalTensor<T> readyOut = suffixChunkOutQueue.DeQue<T>();
        // B need not be a multiple of 16, so even a 16-row tail can have an
        // unaligned GM start. Always use the exact-byte padded DMA overload.
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(rowCount * sizeof(T)),
                                     0U, 0U, 0U};
        DataCopyPad(outputGm[outputBase], readyOut, copyParams);
        suffixChunkOutQueue.FreeTensor(readyOut);
    }

    __aicore__ inline void ProcessFullRowSuffix()
    {
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        if (start >= end) return;
        CopyFullRowSuffixInput(start, MinU32(fullRowSuffixRows, end - start));
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputRows = 0U;
        bool hasPendingOutput = false;

        for (uint32_t outputBase = start; outputBase < end;) {
            const uint32_t rowCount = MinU32(fullRowSuffixRows, end - outputBase);
            LocalTensor<T> ready = suffixChunkInQueue.DeQue<T>();
            const uint32_t nextBase = outputBase + rowCount;
            if (nextBase < end) {
                CopyFullRowSuffixInput(nextBase, MinU32(fullRowSuffixRows, end - nextBase));
            }
            if (hasPendingOutput) {
                CopyFullRowSuffixOutput(pendingOutputOffset, pendingOutputRows);
            }

            LocalTensor<float> sumsLocal = suffixChunkSumBuf.Get<float>();
            LocalTensor<float> calcLocal = suffixChunkCalcBuf.Get<float>();
            // This includes every FP32 source element read by the final BF16 Cast.
            Duplicate(sumsLocal, 0.0F, AlignElems32<T>(rowCount));
            PipeBarrier<PIPE_V>();
            for (uint32_t localRow = 0U; localRow < rowCount;
                 localRow += square_sum_policy::kFullRowSuffixMicroRows) {
                const uint32_t subRows = MinU32(square_sum_policy::kFullRowSuffixMicroRows,
                                                rowCount - localRow);
                const uint32_t count = subRows * fullRowSuffixPitch;
                Cast(calcLocal, ready[localRow * fullRowSuffixPitch], RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                Mul(calcLocal, calcLocal, calcLocal, count);
                PipeBarrier<PIPE_V>();
                FoldAlignedSuffixRows(sumsLocal[localRow], calcLocal, reduceSize,
                                      subRows, fullRowSuffixPitch);
            }
            // The fold's PIPE_V has completed all microbatch reads before reuse.
            suffixChunkInQueue.FreeTensor(ready);
            LocalTensor<T> outputLocal = suffixChunkOutQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, sumsLocal, rowCount);
            PipeBarrier<PIPE_V>();  // finish Cast's sum reads before the next Duplicate
            suffixChunkOutQueue.EnQue(outputLocal);
            pendingOutputOffset = outputBase;
            pendingOutputRows = rowCount;
            hasPendingOutput = true;
            outputBase = nextBase;
        }
        if (hasPendingOutput) {
            CopyFullRowSuffixOutput(pendingOutputOffset, pendingOutputRows);
        }
    }
