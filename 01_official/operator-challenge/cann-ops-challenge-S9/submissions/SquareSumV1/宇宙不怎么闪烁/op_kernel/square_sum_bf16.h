#pragma once
// Private SquareSumKernel<T> members: BF16 contiguous-inner, owned/full-row DMA.

    __aicore__ inline void InitTightBf16InnerBuffers()
    {
        const uint32_t groupRows = square_sum_policy::TightAtomicGroupRows(innerSize);
        const uint32_t pitch = AlignElems32<T>(innerSize);
        const uint32_t batchElems = AlignElems32<T>(
            square_sum_policy::kTightBf16BatchRows * innerSize);
        // Only called after balanced eligibility and the complete uint64 UB gate.
        pipe->InitBuffer(inQueue, 2U, batchElems * sizeof(T));
        pipe->InitBuffer(calcBuf, batchElems * sizeof(float));
        pipe->InitBuffer(accBuf, (groupRows * innerSize + 8U) * sizeof(float));
        pipe->InitBuffer(sumBuf, groupRows * pitch * sizeof(float));
        pipe->InitBuffer(workBuf, groupRows * pitch * sizeof(uint32_t));
        pipe->InitBuffer(outQueue, 1U, pitch * sizeof(T));
    }

    __aicore__ inline void BuildTightBf16Index(uint32_t innerOffset, uint32_t outputCount)
    {
        const uint32_t groupRows = square_sum_policy::TightAtomicGroupRows(innerSize);
        const uint32_t pitch = AlignElems32<T>(innerSize);
        LocalTensor<uint32_t> indexLocal = workBuf.Get<uint32_t>();
        LocalTensor<int32_t> indexSigned = indexLocal.template ReinterpretCast<int32_t>();
        Duplicate(indexLocal, static_cast<uint32_t>(groupRows * innerSize * sizeof(float)),
                  groupRows * pitch);
        // CreateVecIndex may seed lanes through scalar stores. Finish the
        // vector initialization before those seeds, not merely before VEC work.
        event_t vecToScalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vecToScalar);
        WaitFlag<HardEvent::V_S>(vecToScalar);
        for (uint32_t row = 0U; row < groupRows; ++row) {
            // Pack the owned slice at column zero of each aligned padded row.
            // Unowned/padded columns are restored to the zero sentinel below.
            CreateVecIndex(indexSigned[row * pitch],
                           static_cast<int32_t>(row * innerSize + innerOffset), outputCount);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0U; row < groupRows; ++row) {
            Muls(indexSigned[row * pitch], indexSigned[row * pitch],
                 static_cast<int32_t>(sizeof(float)), outputCount);
        }
        // For tiny slices (even count=1), do not assume CreateVecIndex's
        // scalar seeds preserve lanes outside count. Restore all padding only
        // after vector index construction, then publish it to Gather's VEC pipe.
        SetFlag<HardEvent::V_S>(vecToScalar);
        WaitFlag<HardEvent::V_S>(vecToScalar);
        const uint32_t zeroByteOffset = groupRows * innerSize * sizeof(float);
        for (uint32_t row = 0U; row < groupRows; ++row) {
            for (uint32_t col = outputCount; col < pitch; ++col) {
                indexLocal.SetValue(row * pitch + col, zeroByteOffset);
            }
        }
        event_t scalarToVec = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalarToVec);
        WaitFlag<HardEvent::S_V>(scalarToVec);
    }

    __aicore__ inline void CopyTightBf16Input(uint32_t inputBase,
                                               uint32_t rowBase, uint32_t rowCount)
    {
        const uint32_t validElems = rowCount * innerSize;
        const uint32_t dmaElems = AlignElems32<T>(validElems);
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{
            1U, static_cast<uint32_t>(validElems * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{
            true, 0U, static_cast<uint8_t>(dmaElems - validElems), static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[inputBase + rowBase * innerSize], copyParams, padParams);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ReduceTightBf16Groups(LocalTensor<float> values,
                                                 uint32_t groups, uint32_t width)
    {
        // Width is G*I or P, both multiples of eight floats. Continuous-count
        // Add avoids uint8 repeat/stride limits; the odd last group is never lost.
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

    __aicore__ inline void CopyTightBf16Output(LocalTensor<float> values,
                                               uint32_t outputOffset, uint32_t outputCount)
    {
        LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
        Cast(outputLocal, values, RoundMode::CAST_RINT, AlignElems32<T>(outputCount));
        PipeBarrier<PIPE_V>();
        outQueue.EnQue(outputLocal);
        LocalTensor<T> readyOut = outQueue.DeQue<T>();
        DataCopyExtParams outParams{
            1U, static_cast<uint32_t>(outputCount * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(outputGm[outputOffset], readyOut, outParams);
        // VECOUT protects reuse from MTE3, including the next outer row.
        outQueue.FreeTensor(readyOut);
    }

    __aicore__ inline void ProcessTightBf16Slice(uint32_t inputBase,
                                                uint32_t outputOffset,
                                                uint32_t outputCount, uint32_t batchRows)
    {
        const uint32_t rowPitch = AlignElems32<T>(outputCount);
        LocalTensor<float> accLocal = accBuf.Get<float>();
        // Nothing beyond this owned, padded row participates in reduction.
        Duplicate(accLocal, 0.0F, rowPitch);
        PipeBarrier<PIPE_V>();
        uint32_t rowBase = 0U;
        uint32_t rowCount = MinU32(batchRows, reduceSize);
        if (rowCount != 0U) CopyBalancedInnerBatch(inputBase, outputCount, rowBase, rowCount);
        while (rowCount != 0U) {
            LocalTensor<T> ready = inQueue.DeQue<T>();
            const uint32_t nextRowBase = rowBase + rowCount;
            const uint32_t nextRowCount = nextRowBase < reduceSize
                ? MinU32(batchRows, reduceSize - nextRowBase) : 0U;
            if (nextRowCount != 0U) {
                CopyBalancedInnerBatch(inputBase, outputCount, nextRowBase, nextRowCount);
            }
            const uint32_t batchElems = rowCount * rowPitch;
            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            Cast(calcLocal, ready, RoundMode::CAST_NONE, batchElems);
            PipeBarrier<PIPE_V>();
            inQueue.FreeTensor(ready);  // Cast has consumed this input slot.
            Mul(calcLocal, calcLocal, calcLocal, batchElems);
            PipeBarrier<PIPE_V>();
            AccumulateRows(accLocal, calcLocal, rowPitch, rowCount);
            PipeBarrier<PIPE_V>();  // finish the row tree before the next Cast
            rowBase = nextRowBase;
            rowCount = nextRowCount;
        }
        CopyTightBf16Output(accLocal, outputOffset, outputCount);
    }

    __aicore__ inline void ProcessTightBf16Inner()
    {
        const uint32_t groupRows = square_sum_policy::TightAtomicGroupRows(innerSize);
        const uint32_t groupElems = groupRows * innerSize;
        const uint32_t pitch = AlignElems32<T>(innerSize);
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcPlaneOutputRange(start, end);
        for (uint32_t outputOffset = start; outputOffset < end;) {
            const uint32_t innerOffset = outputOffset % innerSize;
            const uint32_t outputCount = MinU32(end - outputOffset, innerSize - innerOffset);
            const uint32_t inputBase = BaseInputOffset(outputOffset - innerOffset);
            const uint32_t sliceRows = square_sum_policy::TightBf16SliceRows(
                innerSize, outputCount);
            if (sliceRows != 0U) {
                ProcessTightBf16Slice(inputBase + innerOffset, outputOffset,
                                     outputCount, sliceRows);
                outputOffset += outputCount;
                continue;
            }
            // Wider ownership retains full-row DMA and the original Gather.
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, groupElems + 8U);
            PipeBarrier<PIPE_V>();
            BuildTightBf16Index(innerOffset, outputCount);

            uint32_t rowBase = 0U;
            uint32_t rowCount = MinU32(square_sum_policy::kTightBf16BatchRows, reduceSize);
            if (rowCount != 0U) CopyTightBf16Input(inputBase, rowBase, rowCount);
            while (rowCount != 0U) {
                LocalTensor<T> ready = inQueue.DeQue<T>();
                const uint32_t nextRowBase = rowBase + rowCount;
                const uint32_t nextRowCount = nextRowBase < reduceSize
                    ? MinU32(square_sum_policy::kTightBf16BatchRows, reduceSize - nextRowBase)
                    : 0U;
                if (nextRowCount != 0U) CopyTightBf16Input(inputBase, nextRowBase, nextRowCount);

                const uint32_t dmaElems = AlignElems32<T>(rowCount * innerSize);
                const uint32_t groups = (rowCount + groupRows - 1U) / groupRows;
                const uint32_t activeElems = groups * groupElems;
                LocalTensor<float> calcLocal = calcBuf.Get<float>();
                // Cast includes DMA's zero tail. N can be smaller than e (by
                // eight floats); allocation covers both, but only N is reduced.
                Cast(calcLocal, ready, RoundMode::CAST_NONE, dmaElems);
                PipeBarrier<PIPE_V>();
                inQueue.FreeTensor(ready);  // no input reads after Cast
                if (activeElems > dmaElems) {
                    Duplicate(calcLocal[dmaElems], 0.0F, activeElems - dmaElems);
                    PipeBarrier<PIPE_V>();
                }
                Mul(calcLocal, calcLocal, calcLocal, activeElems);
                PipeBarrier<PIPE_V>();
                ReduceTightBf16Groups(calcLocal, groups, groupElems);
                Add(accLocal, accLocal, calcLocal, groupElems);
                PipeBarrier<PIPE_V>();  // calc can now be overwritten by next Cast
                rowBase = nextRowBase;
                rowCount = nextRowCount;
            }

            LocalTensor<float> paddedLocal = sumBuf.Get<float>();
            LocalTensor<uint32_t> indexLocal = workBuf.Get<uint32_t>();
            Gather(paddedLocal, accLocal, indexLocal, 0U, groupRows * pitch);
            PipeBarrier<PIPE_V>();
            ReduceTightBf16Groups(paddedLocal, groupRows, pitch);
            CopyTightBf16Output(paddedLocal, outputOffset, outputCount);
            outputOffset += outputCount;
        }
    }
