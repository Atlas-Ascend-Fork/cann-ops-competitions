#pragma once
// Private SquareSumKernel<T> members: BF16 row ownership, per-core FP32 partials.

    __aicore__ inline void InitWorkspaceBf16Buffers()
    {
        // This row-owned path streams each real input once. Keep partials
        // cached in workspace, but do not allocate input lines in L2.
        inputGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t groupRows = square_sum_policy::TightAtomicGroupRows(innerSize);
        const uint32_t batchElems = AlignElems32<T>(
            square_sum_policy::kTightBf16BatchRows * innerSize);
        // Host checks the full WorkspaceBf16Bytes budget before selecting mode 7.
        pipe->InitBuffer(workspaceInQueue, 2U, batchElems * sizeof(T));
        pipe->InitBuffer(calcBuf, batchElems * sizeof(float));
        pipe->InitBuffer(accBuf, (groupRows * innerSize + 8U) * sizeof(float));
        pipe->InitBuffer(sumBuf, groupRows * workspacePitch * sizeof(float));
        pipe->InitBuffer(workBuf, groupRows * workspacePitch * sizeof(uint32_t));
        pipe->InitBuffer(outQueue, 1U, workspacePitch * sizeof(float));
    }

    __aicore__ inline void PrefetchWorkspaceBf16()
    {
        if (workspacePrefetchRow >= workspaceRowEnd) return;
        // A queued tile never crosses an outer boundary. The next tile may
        // belong to a different segment, but remains FIFO until that segment.
        const uint32_t outer = workspacePrefetchRow / reduceSize;
        const uint32_t end = MinU32(workspaceRowEnd, (outer + 1U) * reduceSize);
        const uint32_t rows = MinU32(square_sum_policy::kTightBf16BatchRows,
                                     end - workspacePrefetchRow);
        const uint32_t valid = rows * innerSize;
        LocalTensor<T> inputLocal = workspaceInQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(valid * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{
            true, 0U, static_cast<uint8_t>(AlignElems32<T>(valid) - valid), static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[workspacePrefetchRow * innerSize], copyParams, padParams);
        workspaceInQueue.EnQue(inputLocal);
        workspacePrefetchRow += rows;
    }

    __aicore__ inline void AccumulateWorkspaceBf16Segment(uint32_t rowBase, uint32_t rowEnd,
                                                          uint32_t slot)
    {
        const uint32_t groupRows = square_sum_policy::TightAtomicGroupRows(innerSize);
        const uint32_t groupElems = groupRows * innerSize;
        LocalTensor<float> accLocal = accBuf.Get<float>();
        Duplicate(accLocal, 0.0F, groupElems + 8U);
        PipeBarrier<PIPE_V>();
        uint32_t rowCount = MinU32(square_sum_policy::kTightBf16BatchRows, rowEnd - rowBase);
        while (rowCount != 0U) {
            LocalTensor<T> ready = workspaceInQueue.DeQue<T>();
            const uint32_t nextRowBase = rowBase + rowCount;
            const uint32_t nextRowCount = nextRowBase < rowEnd
                ? MinU32(square_sum_policy::kTightBf16BatchRows, rowEnd - nextRowBase) : 0U;

            const uint32_t dmaElems = AlignElems32<T>(rowCount * innerSize);
            const uint32_t groups = (rowCount + groupRows - 1U) / groupRows;
            const uint32_t activeElems = groups * groupElems;
            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            Cast(calcLocal, ready, RoundMode::CAST_NONE, dmaElems);
            PipeBarrier<PIPE_V>();
            workspaceInQueue.FreeTensor(ready);
            PrefetchWorkspaceBf16();
            if (activeElems > dmaElems) {
                Duplicate(calcLocal[dmaElems], 0.0F, activeElems - dmaElems);
                PipeBarrier<PIPE_V>();
            }
            Mul(calcLocal, calcLocal, calcLocal, activeElems);
            PipeBarrier<PIPE_V>();
            ReduceTightBf16Groups(calcLocal, groups, groupElems);
            Add(accLocal, accLocal, calcLocal, groupElems);
            PipeBarrier<PIPE_V>();
            rowBase = nextRowBase;
            rowCount = nextRowCount;
        }

        LocalTensor<float> paddedLocal = sumBuf.Get<float>();
        LocalTensor<uint32_t> indexLocal = workBuf.Get<uint32_t>();
        Gather(paddedLocal, accLocal, indexLocal, 0U, groupRows * workspacePitch);
        PipeBarrier<PIPE_V>();
        ReduceTightBf16Groups(paddedLocal, groupRows, workspacePitch);
        // MTE3 must not read sumBuf while the next outer segment overwrites it.
        // Copy to the one VECOUT slot; its queue lifetime protects every reuse.
        LocalTensor<float> partialLocal = outQueue.AllocTensor<float>();
        Adds(partialLocal, paddedLocal, 0.0F, workspacePitch);
        PipeBarrier<PIPE_V>();
        outQueue.EnQue(partialLocal);
        LocalTensor<float> readyOut = outQueue.DeQue<float>();
        // This producer owns the entire aligned slot, including zero padding.
        DataCopy(workspaceGm[slot * workspacePitch], readyOut, workspacePitch);
        outQueue.FreeTensor(readyOut);
    }

    __aicore__ inline uint32_t WorkspaceBf16RowOwner(uint32_t row)
    {
        // Invert CalcRange's quotient/remainder partition. Mode 7 has N/C >= 80.
        const uint32_t rows = inputSize / innerSize;
        const uint32_t base = rows / GetBlockNum();
        const uint32_t remain = rows % GetBlockNum();
        const uint32_t cut = remain * (base + 1U);
        return row < cut ? row / (base + 1U) : remain + (row - cut) / base;
    }

    __aicore__ inline uint32_t WorkspaceBf16Slot(uint32_t core, uint32_t outer)
    {
        const uint32_t rows = inputSize / innerSize;
        const uint32_t base = rows / GetBlockNum();
        const uint32_t remain = rows % GetBlockNum();
        const uint32_t coreStart = core * base + MinU32(core, remain);
        return core * workspaceSlots + (outer - coreStart / reduceSize);
    }

    __aicore__ inline void CopyWorkspaceBf16Output()
    {
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcPlaneOutputRange(start, end);
        for (uint32_t outputOffset = start; outputOffset < end;) {
            const uint32_t outer = outputOffset / innerSize;
            const uint32_t innerOffset = outputOffset % innerSize;
            const uint32_t count = MinU32(end - outputOffset, innerSize - innerOffset);
            const uint32_t dmaElems = AlignElems32<float>(count);
            const uint32_t castElems = AlignElems32<T>(count);
            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            // Cast consumes e16 lanes from calc, not the e8-padded input queue.
            Duplicate(calcLocal, 0.0F, castElems);
            PipeBarrier<PIPE_V>();
            const uint32_t firstCore = WorkspaceBf16RowOwner(outer * reduceSize);
            const uint32_t lastCore = WorkspaceBf16RowOwner((outer + 1U) * reduceSize - 1U);
            for (uint32_t core = firstCore; core <= lastCore; ++core) {
                const uint32_t slot = WorkspaceBf16Slot(core, outer);
                // Read only a real slice of a slot written in this invocation.
                // The BF16 B80 input slot also fits one P-float output slice.
                LocalTensor<float> inputLocal = workspaceInQueue.AllocTensor<float>();
                DataCopyExtParams copyParams{
                    1U, static_cast<uint32_t>(count * sizeof(float)), 0U, 0U, 0U};
                DataCopyPadExtParams<float> padParams{
                    true, 0U, static_cast<uint8_t>(dmaElems - count), 0.0F};
                DataCopyPad(inputLocal, workspaceGm[slot * workspacePitch + innerOffset],
                            copyParams, padParams);
                workspaceInQueue.EnQue(inputLocal);
                LocalTensor<float> ready = workspaceInQueue.DeQue<float>();
                Add(calcLocal, calcLocal, ready, count);
                PipeBarrier<PIPE_V>();
                workspaceInQueue.FreeTensor(ready);
            }
            CopyTightBf16Output(calcLocal, outputOffset, count);
            outputOffset += count;
        }
    }

    __aicore__ inline void ProcessWorkspaceBf16Inner()
    {
        uint32_t rowStart = 0U;
        uint32_t rowEnd = 0U;
        CalcRange(inputSize / innerSize, rowStart, rowEnd);
        workspacePrefetchRow = rowStart;
        workspaceRowEnd = rowEnd;
        PrefetchWorkspaceBf16();
        PrefetchWorkspaceBf16();
        // One full-I byte index per core; first DMA may overlap its construction.
        BuildTightBf16Index(0U, innerSize);
        const uint32_t firstOuter = rowStart / reduceSize;
        for (uint32_t row = rowStart; row < rowEnd;) {
            const uint32_t outer = row / reduceSize;
            const uint32_t segmentEnd = MinU32(rowEnd, (outer + 1U) * reduceSize);
            const uint32_t slot = GetBlockIdx() * workspaceSlots + (outer - firstOuter);
            AccumulateWorkspaceBf16Segment(row, segmentEnd, slot);
            row = segmentEnd;
        }
        // Retire all private partial writes before readers access other cores.
        // Every core participates, including empty input/output owners. No clear
        // is needed: every consumed slot is fully overwritten in this call.
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        CopyWorkspaceBf16Output();
    }
