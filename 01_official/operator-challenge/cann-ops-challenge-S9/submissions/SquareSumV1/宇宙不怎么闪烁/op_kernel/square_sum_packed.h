#pragma once
// BF16 suffix: linear GM reads, FP32 block sums, exact masked row boundaries.

    __aicore__ inline void InitPackedSuffixBuffers()
    {
        const uint32_t dense = square_sum_policy::Align(packedSuffixRows * reduceSize, 64U);
        const uint32_t pitch = square_sum_policy::PackedSuffixBlockPitch(reduceSize);
        const uint32_t outputPitch = AlignElems32<T>(packedSuffixRows);
        pipe->InitBuffer(suffixChunkInQueue, 2U,
                         AlignElems32<T>(packedSuffixRows * reduceSize) * sizeof(T));
        pipe->InitBuffer(suffixChunkOutQueue, 2U, outputPitch * sizeof(T));
        pipe->InitBuffer(suffixChunkCalcBuf, (dense + 64U) * sizeof(float));
        pipe->InitBuffer(suffixChunkSquareBuf, (dense / 8U + 8U) * sizeof(float));
        pipe->InitBuffer(accBuf, (packedSuffixRows * (pitch + 16U) + 64U) * sizeof(float));
        pipe->InitBuffer(workBuf, packedSuffixRows * (pitch + 16U) * sizeof(uint32_t));
        pipe->InitBuffer(suffixChunkSumBuf, outputPitch * sizeof(float));
        pipe->InitBuffer(planeSecondSumBuf, outputPitch * sizeof(float));
        LocalTensor<float> blockLocal = suffixChunkSquareBuf.Get<float>();
        Duplicate(blockLocal[dense / 8U], 0.0F, 8U);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildPackedSuffixIndex()
    {
        const uint32_t pitch = square_sum_policy::PackedSuffixBlockPitch(reduceSize);
        const uint32_t blockCount = packedSuffixRows * pitch;
        const uint32_t count = packedSuffixRows * (pitch + 16U);
        const uint32_t zero = square_sum_policy::Align(packedSuffixRows * reduceSize, 64U) / 8U;
        LocalTensor<uint32_t> indexLocal = workBuf.Get<uint32_t>();
        LocalTensor<int32_t> indices = indexLocal.template ReinterpretCast<int32_t>();
        CreateVecIndex(indices, 0, count);
        PipeBarrier<PIPE_V>();
        Muls(indices, indices, static_cast<int32_t>(sizeof(float)), count);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0U; row < packedSuffixRows; ++row) {
            const uint32_t firstBlock = (row * reduceSize + 7U) / 8U;
            Adds(indices[row * pitch], indices[row * pitch],
                 (static_cast<int32_t>(firstBlock) - static_cast<int32_t>(row * pitch)) *
                 static_cast<int32_t>(sizeof(float)), pitch);
        }
        // Appended indices address original squares, not the block-sum table.
        for (uint32_t row = 0U; row < packedSuffixRows; ++row) {
            const uint32_t first = row * reduceSize;
            const uint32_t end = first + reduceSize;
            const uint32_t left = blockCount + row * 16U;
            const uint32_t right = left + 8U;
            Adds(indices[left], indices[left],
                 (static_cast<int32_t>(first & ~7U) - static_cast<int32_t>(left)) *
                 static_cast<int32_t>(sizeof(float)), 8U);
            Adds(indices[right], indices[right],
                 (static_cast<int32_t>(end & ~7U) - static_cast<int32_t>(right)) *
                 static_cast<int32_t>(sizeof(float)), 8U);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0U; row < packedSuffixRows; ++row) {
            const uint32_t firstBlock = (row * reduceSize + 7U) / 8U;
            const uint32_t endBlock = (row + 1U) * reduceSize / 8U;
            const uint32_t valid = endBlock - firstBlock;
            if (valid < pitch) {
                const uint32_t base = valid / 8U * 8U;
                // Padding occupies at most two 32B blocks. Mask out the valid
                // prefix of the first block and never write the following row.
                uint64_t mask[2] = {
                    ((1ULL << (pitch - base)) - 1ULL) & (~0ULL << (valid - base)), 0ULL};
                Duplicate(indexLocal[row * pitch + base],
                          static_cast<uint32_t>(zero * sizeof(float)), mask, 1U, 1U, 8U);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyPackedSuffixInput(uint32_t outputBase, uint32_t rowCount)
    {
        const uint32_t count = rowCount * reduceSize;
        LocalTensor<T> inputLocal = suffixChunkInQueue.AllocTensor<T>();
        const uint32_t gmOffset = outputBase * reduceSize;
        if ((gmOffset & 15U) == 0U && (count & 15U) == 0U && count / 16U <= 65535U) {
            DataCopyParams copyParams{1U, static_cast<uint16_t>(count / 16U), 0U, 0U};
            DataCopy(inputLocal, inputGm[gmOffset], copyParams);
            suffixChunkInQueue.EnQue(inputLocal);
            return;
        }
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{
            true, 0U, static_cast<uint8_t>(AlignElems32<T>(count) - count), static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[outputBase * reduceSize], copyParams, padParams);
        suffixChunkInQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ReducePackedSuffix(LocalTensor<float> calcLocal, uint32_t rowCount)
    {
        const uint32_t dense = square_sum_policy::Align(rowCount * reduceSize, 64U);
        const uint32_t pitch = square_sum_policy::PackedSuffixBlockPitch(reduceSize);
        LocalTensor<float> blockLocal = suffixChunkSquareBuf.Get<float>();
        for (uint32_t offset = 0U; offset < dense; offset += 255U * 64U) {
            const uint32_t repeats = MinU32(255U, (dense - offset) / 64U);
            // Float destination stride is in 8-element (32B) result blocks.
            BlockReduceSum<float>(blockLocal[offset / 8U], calcLocal[offset],
                                  static_cast<int32_t>(repeats), 64, 1, 1, 8);
        }
        PipeBarrier<PIPE_V>();
        LocalTensor<float> gatheredLocal = accBuf.Get<float>();
        LocalTensor<uint32_t> indexLocal = workBuf.Get<uint32_t>();
        Gather(gatheredLocal, blockLocal, indexLocal, 0U, rowCount * pitch);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> sumsLocal = suffixChunkSumBuf.Get<float>();
        LocalTensor<float> edgeSumsLocal = planeSecondSumBuf.Get<float>();
        const uint32_t outputPitch = AlignElems32<T>(rowCount);
        Duplicate(sumsLocal, 0.0F, outputPitch);
        Duplicate(edgeSumsLocal, 0.0F, outputPitch);
        PipeBarrier<PIPE_V>();
        // A real row has at most floor(R/8) complete blocks. The remaining
        // gathered columns are sentinel zeros, not input/reduction tails.
        FoldAlignedSuffixRows(sumsLocal, gatheredLocal, reduceSize / 8U, rowCount, pitch);
        const uint32_t edgeBase = packedSuffixRows * pitch;
        LocalTensor<float> edgesLocal = gatheredLocal[edgeBase];
        Gather(edgesLocal, calcLocal, indexLocal[edgeBase], 0U, rowCount * 16U);
        Duplicate(edgesLocal[rowCount * 16U], 0.0F, 64U);
        PipeBarrier<PIPE_V>();
        const uint32_t groups = square_sum_policy::TightAtomicGroupRows(reduceSize);
        for (uint32_t phase = 0U; phase < groups && phase < rowCount; ++phase) {
            const uint32_t firstLane = phase * reduceSize & 7U;
            const uint32_t lastLane = (phase + 1U) * reduceSize & 7U;
            const uint32_t repeats = (rowCount - 1U - phase) / groups + 1U;
            const uint64_t leftMask = firstLane != 0U ? 0xFFULL & (~0ULL << firstLane) : 0ULL;
            const uint64_t rightMask = ((1ULL << lastLane) - 1ULL) << 8U;
            if ((leftMask | rightMask) != 0ULL) {
                uint64_t mask[2] = {leftMask | rightMask, 0ULL};
                WholeReduceSum<float>(edgeSumsLocal[phase], edgesLocal[phase * 16U], mask,
                                      repeats, groups, 1, 2U * groups);
            }
        }
        PipeBarrier<PIPE_V>();
        Add(sumsLocal, sumsLocal, edgeSumsLocal, rowCount);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessPackedSuffix()
    {
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        if (start >= end) return;
        uint32_t prefetchBase = start;
        for (uint32_t slot = 0U; slot < 2U && prefetchBase < end; ++slot) {
            const uint32_t count = MinU32(packedSuffixRows, end - prefetchBase);
            CopyPackedSuffixInput(prefetchBase, count);
            prefetchBase += count;
        }
        // Both DMA requests may run while vector/scalar index setup executes.
        BuildPackedSuffixIndex();
        uint32_t pendingBase = 0U;
        uint32_t pendingRows = 0U;
        for (uint32_t outputBase = start; outputBase < end;) {
            const uint32_t rowCount = MinU32(packedSuffixRows, end - outputBase);
            LocalTensor<T> ready = suffixChunkInQueue.DeQue<T>();
            const uint32_t nextBase = outputBase + rowCount;
            if (pendingRows != 0U) CopyFullRowSuffixOutput(pendingBase, pendingRows);
            const uint32_t dmaElems = AlignElems32<T>(rowCount * reduceSize);
            const uint32_t dense = square_sum_policy::Align(dmaElems, 64U);
            LocalTensor<float> calcLocal = suffixChunkCalcBuf.Get<float>();
            Cast(calcLocal, ready, RoundMode::CAST_NONE, dmaElems);
            PipeBarrier<PIPE_V>();
            suffixChunkInQueue.FreeTensor(ready);
            if (prefetchBase < end) {
                const uint32_t count = MinU32(packedSuffixRows, end - prefetchBase);
                CopyPackedSuffixInput(prefetchBase, count);
                prefetchBase += count;
            }
            // Bound even masked-off physical lanes in the last boundary WRS.
            Duplicate(calcLocal[dmaElems], 0.0F, dense - dmaElems + 64U);
            PipeBarrier<PIPE_V>();
            Mul(calcLocal, calcLocal, calcLocal, dense);
            PipeBarrier<PIPE_V>();
            ReducePackedSuffix(calcLocal, rowCount);
            LocalTensor<T> outputLocal = suffixChunkOutQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, suffixChunkSumBuf.Get<float>(), rowCount);
            PipeBarrier<PIPE_V>();
            suffixChunkOutQueue.EnQue(outputLocal);
            pendingBase = outputBase;
            pendingRows = rowCount;
            outputBase = nextBase;
        }
        if (pendingRows != 0U) CopyFullRowSuffixOutput(pendingBase, pendingRows);
    }
