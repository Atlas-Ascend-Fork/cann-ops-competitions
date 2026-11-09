#pragma once
// Private SquareSumKernel<T> members: FP32 atomic inner, tight-DMA split-R.

    __aicore__ inline void InitTightAtomicInnerBuffers()
    {
        const uint32_t groupRows = square_sum_policy::TightAtomicGroupRows(innerSize);
        const uint32_t groupElems = groupRows * innerSize;
        const uint32_t gatherElems = groupRows * square_sum_policy::kTightAtomicGatherCols;
        // Dispatch checks the complete 64-bit budget before any allocation.
        pipe->InitBuffer(tightInQueue, 2U,
                         square_sum_policy::kTightAtomicBatchRows * innerSize * sizeof(float));
        pipe->InitBuffer(accBuf, (groupElems + 8U) * sizeof(float));
        pipe->InitBuffer(sumBuf, gatherElems * sizeof(float));
        pipe->InitBuffer(workBuf, gatherElems * sizeof(uint32_t));
        pipe->InitBuffer(calcBuf, AlignElems32<float>(innerSize) * sizeof(float));
    }

    __aicore__ inline void BuildTightAtomicIndex()
    {
        const uint32_t groupRows = square_sum_policy::TightAtomicGroupRows(innerSize);
        constexpr uint32_t pitch = square_sum_policy::kTightAtomicGatherCols;
        const uint32_t gatherElems = groupRows * pitch;
        LocalTensor<uint32_t> indexLocal = workBuf.Get<uint32_t>();
        LocalTensor<int32_t> indexSigned = indexLocal.template ReinterpretCast<int32_t>();
        // At least 64 lanes; every row base is 32B aligned. Indices are bytes,
        // not elements, and describe the first 64 columns of each acc row.
        CreateVecIndex(indexSigned, 0, gatherElems);
        PipeBarrier<PIPE_V>();
        Muls(indexSigned, indexSigned, static_cast<int32_t>(sizeof(float)), gatherElems);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 1U; row < groupRows; ++row) {
            Adds(indexSigned[row * pitch], indexSigned[row * pitch],
                 (static_cast<int32_t>(row * innerSize) -
                  static_cast<int32_t>(row * pitch)) * static_cast<int32_t>(sizeof(float)),
                 pitch);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyTightAtomicInput(uint32_t rowBase, uint32_t rowCount)
    {
        // Called only for nonempty batches. Read v real floats, zero [v,e)
        // by DMA; the consumer subsequently zeros only the aligned [e,N).
        const uint32_t validElems = rowCount * innerSize;
        const uint32_t dmaElems = AlignElems32<float>(validElems);
        LocalTensor<float> inputLocal = tightInQueue.AllocTensor<float>();
        const uint32_t gmOffset = rowBase * innerSize;
        // Whole aligned blocks need no padding. Every unaligned start/count
        // keeps the byte-exact padded path, including the global final tile.
        if ((gmOffset & 7U) == 0U && (validElems & 7U) == 0U &&
            validElems / 8U <= 65535U) {
            DataCopyParams copyParams{
                1U, static_cast<uint16_t>(validElems / 8U), 0U, 0U};
            DataCopy(inputLocal, inputGm[gmOffset], copyParams);
            tightInQueue.EnQue(inputLocal);
            return;
        }
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(validElems * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> padParams{
            true, 0U, static_cast<uint8_t>(dmaElems - validElems), 0.0F};
        DataCopyPad(inputLocal, inputGm[rowBase * innerSize], copyParams, padParams);
        tightInQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ReduceTightAtomicGroups(LocalTensor<float> values,
                                                   uint32_t groups, uint32_t width)
    {
        // All bases are 32B aligned. Use continuous count overloads, never
        // uint8 repeat/stride fields (the official width/8 is already 521).
        while (groups > 1U) {
            const uint32_t pairs = groups / 2U;
            Add(values, values, values[pairs * width], pairs * width);
            PipeBarrier<PIPE_V>();
            if ((groups & 1U) != 0U) {
                Add(values, values, values[(groups - 1U) * width], width);
                PipeBarrier<PIPE_V>();
            }
            groups = pairs;
        }
    }

    __aicore__ inline void ProcessTightAtomicInner()
    {
        if (GetBlockIdx() == 0U) {
            InitGlobalMemory(outputGm, outputSize, 0.0F);
        }
        SyncAll();

        const uint32_t groupRows = square_sum_policy::TightAtomicGroupRows(innerSize);
        const uint32_t groupElems = groupRows * innerSize;
        constexpr uint32_t pitch = square_sum_policy::kTightAtomicGatherCols;
        LocalTensor<float> accLocal = accBuf.Get<float>();
        Duplicate(accLocal, 0.0F, groupElems + 8U);
        PipeBarrier<PIPE_V>();
        BuildTightAtomicIndex();

        // Interleave whole B-row tiles, not per-core contiguous row ranges.
        // B is divisible by G: every full tile GM start is 32B aligned.
        const uint64_t rowStride = static_cast<uint64_t>(GetBlockNum()) *
                                   square_sum_policy::kTightAtomicBatchRows;
        const uint64_t firstRow = static_cast<uint64_t>(GetBlockIdx()) *
                                  square_sum_policy::kTightAtomicBatchRows;
        uint32_t rowBase = firstRow < reduceSize ? static_cast<uint32_t>(firstRow) : reduceSize;
        uint32_t rowCount = MinU32(square_sum_policy::kTightAtomicBatchRows, reduceSize - rowBase);
        uint64_t prefetchRow = firstRow;
        for (uint32_t slot = 0U; slot < 2U && prefetchRow < reduceSize; ++slot) {
            CopyTightAtomicInput(static_cast<uint32_t>(prefetchRow), MinU32(
                square_sum_policy::kTightAtomicBatchRows, reduceSize - static_cast<uint32_t>(prefetchRow)));
            prefetchRow += rowStride;
        }
        while (rowCount != 0U) {
            LocalTensor<float> ready = tightInQueue.DeQue<float>();
            const uint64_t nextRow = static_cast<uint64_t>(rowBase) + rowStride;
            const uint32_t nextRowBase = nextRow < reduceSize
                ? static_cast<uint32_t>(nextRow) : reduceSize;
            const uint32_t nextRowCount = MinU32(
                square_sum_policy::kTightAtomicBatchRows, reduceSize - nextRowBase);

            const uint32_t validElems = rowCount * innerSize;
            const uint32_t dmaElems = AlignElems32<float>(validElems);
            const uint32_t groups = (rowCount + groupRows - 1U) / groupRows;
            const uint32_t activeElems = groups * groupElems;
            if (activeElems > dmaElems) {
                Duplicate(ready[dmaElems], 0.0F, activeElems - dmaElems);
                PipeBarrier<PIPE_V>();
            }
            Mul(ready, ready, ready, activeElems);
            PipeBarrier<PIPE_V>();
            ReduceTightAtomicGroups(ready, groups, groupElems);
            Add(accLocal, accLocal, ready, groupElems);
            PipeBarrier<PIPE_V>();
            // Standard VECIN consumer lifetime protects in-place reads/writes
            // from the next MTE2 reuse. No access to ready after FreeTensor.
            tightInQueue.FreeTensor(ready);
            // Submit the next DMA before waiting for the already queued tile.
            // Depth two is logical as well as physical; no third UB allocation.
            if (prefetchRow < reduceSize) {
                CopyTightAtomicInput(static_cast<uint32_t>(prefetchRow), MinU32(
                    square_sum_policy::kTightAtomicBatchRows, reduceSize - static_cast<uint32_t>(prefetchRow)));
                prefetchRow += rowStride;
            }
            rowBase = nextRowBase;
            rowCount = nextRowCount;
        }

        // acc is [G,I], with an untouched zero sentinel at G*I. Gather only
        // 64 columns at a time and reduce G rows, never the column axis.
        LocalTensor<float> gatheredLocal = sumBuf.Get<float>();
        LocalTensor<uint32_t> indexLocal = workBuf.Get<uint32_t>();
        LocalTensor<float> outputLocal = calcBuf.Get<float>();
        for (uint32_t colBase = 0U; colBase < innerSize; colBase += pitch) {
            const uint32_t count = MinU32(pitch, innerSize - colBase);
            if (count < pitch) {
                // One masked vector fill repairs all G rows; real columns keep
                // their byte offsets. count is 1..63 here, so the shift is defined.
                uint64_t mask[2] = {~0ULL << count, 0ULL};
                Duplicate(indexLocal, static_cast<uint32_t>(groupElems * sizeof(float)),
                          mask, static_cast<uint8_t>(groupRows), 1U,
                          static_cast<uint8_t>(pitch / 8U));
                PipeBarrier<PIPE_V>();
            }
            Gather(gatheredLocal, accLocal, indexLocal, 0U, groupRows * pitch);
            PipeBarrier<PIPE_V>();
            ReduceTightAtomicGroups(gatheredLocal, groupRows, pitch);
            Adds(outputLocal[colBase], gatheredLocal, 0.0F, AlignElems32<float>(count));
            // Finish scratch reads before another Gather overwrites it.
            PipeBarrier<PIPE_V>();
            if (colBase + pitch < innerSize) {
                LocalTensor<int32_t> indexSigned = indexLocal.template ReinterpretCast<int32_t>();
                Adds(indexSigned, indexSigned, static_cast<int32_t>(pitch * sizeof(float)),
                     groupRows * pitch);
                PipeBarrier<PIPE_V>();
            }
        }

        // One exact atomic transfer from the separate output row. It is never
        // modified/reallocated afterwards; empty cores also participate.
        event_t vecToMte3 = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(vecToMte3);
        WaitFlag<HardEvent::V_MTE3>(vecToMte3);
        SetAtomicAdd<float>();
        DataCopyExtParams outParams{
            1U, static_cast<uint32_t>(innerSize * sizeof(float)), 0U, 0U, 0U};
        DataCopyPad(outputGm, outputLocal, outParams);
        SetAtomicNone();
    }
