    __aicore__ inline bool IsRank3CScalarOther() const
    {
        if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, half>)) {
            return false;
        } else {
            if (outDims[0] != 1U || outDims[1] != 1U || outDims[4] < 64U ||
                static_cast<uint64_t>(outDims[3]) * sizeof(T) > kCScalarCacheBytes) {
                return false;
            }
            return selfDims[0] == outDims[0] && selfDims[1] == outDims[1] &&
                   selfDims[2] == outDims[2] && selfDims[3] == outDims[3] &&
                   selfDims[4] == outDims[4] &&
                   otherDims[0] == 1U && otherDims[1] == 1U && otherDims[2] == 1U &&
                   otherDims[3] == outDims[3] && otherDims[4] == 1U;
        }
    }

    __aicore__ inline bool IsRank4CRowOther() const
    {
        if constexpr (!(std::is_same_v<T, float> || std::is_same_v<T, half>)) {
            return false;
        } else {
            if (outDims[0] != 1U || outDims[3] < 2U || outDims[4] < 2U) {
                return false;
            }
            return selfDims[0] == outDims[0] && selfDims[1] == outDims[1] &&
                   selfDims[2] == outDims[2] && selfDims[3] == outDims[3] &&
                   selfDims[4] == outDims[4] &&
                   otherDims[0] == 1U && otherDims[1] == 1U &&
                   otherDims[2] == outDims[2] && otherDims[3] == 1U &&
                   otherDims[4] == outDims[4];
        }
    }

    __aicore__ inline uint32_t AlignedUnitCount(uint32_t itemCount, uint32_t itemBytes) const
    {
        const uint32_t itemsPerUnit = 32U / GcdU32(itemBytes, 32U);
        return (itemCount + itemsPerUnit - 1U) / itemsPerUnit;
    }

    __aicore__ inline void DeriveAlignedItemRange(uint32_t itemCount, uint32_t itemBytes,
                                                   uint32_t& itemStart, uint32_t& itemEnd) const
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t itemsPerUnit = 32U / GcdU32(itemBytes, 32U);
        const uint32_t unitCount = (itemCount + itemsPerUnit - 1U) / itemsPerUnit;
        const uint32_t baseUnits = unitCount / blockNum;
        const uint32_t extraUnits = unitCount - baseUnits * blockNum;
        const uint32_t extraBefore = blockIdx < extraUnits ? blockIdx : extraUnits;
        const uint32_t myUnits = baseUnits + (blockIdx < extraUnits ? 1U : 0U);
        const uint32_t startUnit = blockIdx * baseUnits + extraBefore;
        itemStart = MinU32(itemCount, startUnit * itemsPerUnit);
        itemEnd = MinU32(itemCount, (startUnit + myUnits) * itemsPerUnit);
    }

    __aicore__ inline uint32_t CScalarRowGroupLimit() const
    {
        if constexpr (std::is_same_v<T, float>) {
            return kCScalarRowGroupFloatElems;
        } else if constexpr (std::is_same_v<T, half>) {
            return kCScalarRowGroupHalfElems;
        } else {
            return 0U;
        }
    }

    __aicore__ inline bool CanProcessCScalarOtherRowGroups(uint32_t cDim, uint32_t kDim) const
    {
        const uint32_t groupLimit = CScalarRowGroupLimit();
        const uint32_t nDim = outDims[kDimNum - 3U];
        const uint32_t blockNum = GetBlockNum();
        if (cDim < 2U || kDim == 0U || groupLimit / kDim < 2U || blockNum == 0U) {
            return false;
        }
        const uint32_t nSliceBytes = cDim * kDim;
        return AlignedUnitCount(nDim, nSliceBytes) >= blockNum;
    }

    __aicore__ inline void ProcessCScalarOtherRowGroups(uint32_t cDim, uint32_t kDim)
    {
        const uint32_t nDim = outDims[kDimNum - 3U];
        const uint32_t maxGroupRows = MinU32(cDim, CScalarRowGroupLimit() / kDim);
        const uint32_t maxGroupElems = maxGroupRows * kDim;
        const uint32_t inLocalElems = AlignElems256<T>(maxGroupElems);
        const uint32_t outLocalElems = AlignElems256<uint8_t>(maxGroupElems);

        uint32_t nStart = 0U;
        uint32_t nEnd = 0U;
        DeriveAlignedItemRange(nDim, cDim * kDim, nStart, nEnd);
        if (nStart >= nEnd) {
            return;
        }

        pipe->InitBuffer(cScalarPipelineInQueue, 2U, inLocalElems * sizeof(T));
        pipe->InitBuffer(otherQueue, kBufferNum, kCScalarCacheBytes);
        pipe->InitBuffer(cScalarPipelineOutQueue, 2U, outLocalElems * sizeof(uint8_t));
        pipe->InitBuffer(broadcastBuf, inLocalElems * sizeof(T));
        InitTempBuffers(maxGroupElems);
        pipe->InitBuffer(maskBuf, outLocalElems * sizeof(uint8_t));
        InitLoopConstants(maxGroupElems);

        LocalTensor<T> rhsTensor = otherQueue.AllocTensor<T>();
        CopyDense(rhsTensor, otherGm, 0U, cDim);
        otherQueue.EnQue(rhsTensor);
        LocalTensor<T> rhsLocal = otherQueue.DeQue<T>();
        PipeBarrier<PIPE_ALL>();

        LocalTensor<T> rhsExpanded = broadcastBuf.Get<T>();
        for (uint32_t cBase = 0U; cBase < cDim; cBase += maxGroupRows) {
            const uint32_t groupRows = MinU32(maxGroupRows, cDim - cBase);
            const uint32_t flatCount = groupRows * kDim;
            BuildCScalarBroadcast(rhsExpanded, rhsLocal, cBase, groupRows, kDim, flatCount);
            ProcessCScalarRowGroupRange(rhsExpanded, nStart, nEnd, cBase, groupRows, cDim, kDim);
        }
        otherQueue.FreeTensor(rhsLocal);
    }

    __aicore__ inline void BuildCScalarBroadcast(LocalTensor<T>& dst, LocalTensor<T>& rhsLocal,
                                                  uint32_t cBase, uint32_t groupRows,
                                                  uint32_t kDim, uint32_t flatCount)
    {
        const uint32_t alignElems = 32U / sizeof(T);
        const uint32_t computeElems = AlignElems256<T>(flatCount);
        if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint16_t> dstBits = dst.template ReinterpretCast<uint16_t>();
            LocalTensor<uint16_t> rhsBits = rhsLocal.template ReinterpretCast<uint16_t>();
            for (int32_t row = static_cast<int32_t>(groupRows) - 1; row >= 0; --row) {
                const uint32_t rowIndex = static_cast<uint32_t>(row);
                const uint32_t rowStart = rowIndex * kDim;
                const uint32_t alignedStart = (rowStart / alignElems) * alignElems;
                const uint32_t fillEnd = rowIndex + 1U == groupRows ? computeElems : rowStart + kDim;
                Duplicate(dstBits[alignedStart], rhsBits.GetValue(cBase + rowIndex), fillEnd - alignedStart);
            }
        } else {
            for (int32_t row = static_cast<int32_t>(groupRows) - 1; row >= 0; --row) {
                const uint32_t rowIndex = static_cast<uint32_t>(row);
                const uint32_t rowStart = rowIndex * kDim;
                const uint32_t alignedStart = (rowStart / alignElems) * alignElems;
                const uint32_t fillEnd = rowIndex + 1U == groupRows ? computeElems : rowStart + kDim;
                Duplicate(dst[alignedStart], rhsLocal.GetValue(cBase + rowIndex), fillEnd - alignedStart);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessCScalarRowGroupRange(LocalTensor<T>& rhsExpanded,
                                                        uint32_t nStart, uint32_t nEnd,
                                                        uint32_t cBase, uint32_t groupRows,
                                                        uint32_t cDim, uint32_t kDim)
    {
        const uint32_t flatCount = groupRows * kDim;
        uint32_t currentN = nStart;
        uint32_t currentOffset = (currentN * cDim + cBase) * kDim;
        PrefetchCScalarGroup(currentOffset, flatCount);
        uint32_t pendingOffset = 0U;
        bool hasPendingOutput = false;

        while (true) {
            LocalTensor<T> currentLocal = cScalarPipelineInQueue.DeQue<T>();
            const uint32_t nextN = currentN + 1U;
            const bool hasNext = nextN < nEnd;
            if (hasNext) {
                const uint32_t nextOffset = (nextN * cDim + cBase) * kDim;
                PrefetchCScalarGroup(nextOffset, flatCount);
            }
            if (hasPendingOutput) {
                CopyOutCScalarPipelined(pendingOffset, flatCount);
            }
            ComputeCScalarBroadcastReady(currentLocal, rhsExpanded, flatCount);
            pendingOffset = currentOffset;
            hasPendingOutput = true;
            if (!hasNext) {
                break;
            }
            currentN = nextN;
            currentOffset = (currentN * cDim + cBase) * kDim;
        }
        CopyOutCScalarPipelined(pendingOffset, flatCount);
    }

    __aicore__ inline void PrefetchCScalarGroup(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> selfLocal = cScalarPipelineInQueue.AllocTensor<T>();
        CopyDense(selfLocal, selfGm, offset, count);
        cScalarPipelineInQueue.EnQue(selfLocal);
    }

    __aicore__ inline void ComputeCScalarBroadcastReady(LocalTensor<T>& selfLocal,
                                                         LocalTensor<T>& rhsExpanded,
                                                         uint32_t count)
    {
        LocalTensor<uint8_t> outLocal = cScalarPipelineOutQueue.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        const uint32_t computeElems = AlignElems256<T>(count);
        Compare(mask, selfLocal, rhsExpanded, CMPMODE::GT, computeElems);
        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<half> oneHalf = selfFloatBuf.Get<half>();
            LocalTensor<half> resultHalf = otherFloatBuf.Get<half>();
            Select(resultHalf, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, computeElems);
            Cast(outLocal, resultHalf, RoundMode::CAST_NONE, computeElems);
        } else if constexpr (std::is_same_v<T, half>) {
            LocalTensor<half> oneHalf = otherFloatBuf.Get<half>();
            Duplicate(oneHalf, static_cast<half>(1.0F), 128U);
            const uint8_t repeatTime = static_cast<uint8_t>(computeElems / 128U);
            const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
            Select(selfLocal, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
                   repeatParams);
            Cast(outLocal, selfLocal, RoundMode::CAST_NONE, computeElems);
        }
        cScalarPipelineOutQueue.EnQue(outLocal);
        cScalarPipelineInQueue.FreeTensor(selfLocal);
    }

    __aicore__ inline void ProcessCScalarOther()
    {
        const uint32_t cDim = outDims[kDimNum - 2U];
        const uint32_t kDim = outDims[kDimNum - 1U];
        if (CanProcessCScalarOtherRowGroups(cDim, kDim)) {
            ProcessCScalarOtherRowGroups(cDim, kDim);
            return;
        }
        uint32_t pipelineLimit = 0U;
        if constexpr (std::is_same_v<T, float>) {
            pipelineLimit = kCScalarPipelineFloatElems;
        } else if constexpr (std::is_same_v<T, half>) {
            pipelineLimit = kCScalarPipelineHalfElems;
        }
        if (cDim != 0U && kDim != 0U && pipelineLimit != 0U) {
            // Keep each double-buffered transfer inside one C-scalar row. For
            // large K, segment the row instead of falling back to the
            // single-buffer path so MTE2, VEC, and MTE3 remain overlapped.
            const uint32_t pipelineElems = MinU32(kDim, pipelineLimit);
            ProcessCScalarOtherPipelined(cDim, kDim, pipelineElems);
            return;
        }

        const uint32_t inLocalElems = AlignElems256<T>(tileElems);
        const uint32_t outLocalElems = AlignElems256<uint8_t>(tileElems);
        pipe->InitBuffer(selfQueue, kBufferNum, inLocalElems * sizeof(T));
        pipe->InitBuffer(otherQueue, kBufferNum, kCScalarCacheBytes);
        pipe->InitBuffer(outQueue, kBufferNum, outLocalElems * sizeof(uint8_t));
        InitTempBuffers();
        pipe->InitBuffer(maskBuf, outLocalElems * sizeof(uint8_t));
        InitLoopConstants();

        LocalTensor<T> rhsTensor = otherQueue.AllocTensor<T>();
        CopyDense(rhsTensor, otherGm, 0U, cDim);
        otherQueue.EnQue(rhsTensor);
        LocalTensor<T> rhsLocal = otherQueue.DeQue<T>();
        PipeBarrier<PIPE_ALL>();

        for (uint32_t offset = startElem; offset < endElem;) {
            const uint32_t kIndex = offset % kDim;
            const uint32_t curElems = MinU32(tileElems, MinU32(endElem - offset, kDim - kIndex));
            const uint32_t cIndex = (offset / kDim) % cDim;
            const T rhsScalar = rhsLocal.GetValue(cIndex);

            LocalTensor<T> selfLocal = selfQueue.AllocTensor<T>();
            CopyDense(selfLocal, selfGm, offset, curElems);
            selfQueue.EnQue(selfLocal);

            ComputeScalarOther(rhsScalar, curElems);
            CopyOut(offset, curElems);
            offset += curElems;
        }
        otherQueue.FreeTensor(rhsLocal);
    }

    __aicore__ inline void ProcessCScalarOtherPipelined(uint32_t cDim, uint32_t kDim,
                                                         uint32_t pipelineElems)
    {
        const uint32_t inLocalElems = AlignElems256<T>(pipelineElems);
        const uint32_t outLocalElems = AlignElems256<uint8_t>(pipelineElems);
        pipe->InitBuffer(cScalarPipelineInQueue, 2U, inLocalElems * sizeof(T));
        pipe->InitBuffer(otherQueue, kBufferNum, kCScalarCacheBytes);
        pipe->InitBuffer(cScalarPipelineOutQueue, 2U, outLocalElems * sizeof(uint8_t));
        InitTempBuffers(pipelineElems);
        pipe->InitBuffer(maskBuf, outLocalElems * sizeof(uint8_t));
        InitLoopConstants(pipelineElems);

        LocalTensor<T> rhsTensor = otherQueue.AllocTensor<T>();
        CopyDense(rhsTensor, otherGm, 0U, cDim);
        otherQueue.EnQue(rhsTensor);
        LocalTensor<T> rhsLocal = otherQueue.DeQue<T>();
        PipeBarrier<PIPE_ALL>();

        uint32_t currentOffset = startElem;
        uint32_t currentElems = MinU32(endElem - currentOffset,
                                       MinU32(pipelineElems, kDim - (currentOffset % kDim)));
        PrefetchCScalarRow(currentOffset, currentElems);
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputElems = 0U;
        bool hasPendingOutput = false;
        while (true) {
            // DeQue releases the queue slot before allocating the next input
            // buffer. Keeping the current tensor live separately allows MTE2
            // to fill the second VECIN buffer while VEC processes this one.
            LocalTensor<T> currentLocal = cScalarPipelineInQueue.DeQue<T>();
            const uint32_t nextOffset = currentOffset + currentElems;
            const bool hasNext = nextOffset < endElem;
            uint32_t nextElems = 0U;
            if (hasNext) {
                nextElems = MinU32(endElem - nextOffset,
                                   MinU32(pipelineElems, kDim - (nextOffset % kDim)));
                PrefetchCScalarRow(nextOffset, nextElems);
            }

            // Retire the preceding output before issuing this compare.  The
            // two VECOUT buffers preserve MTE3->VEC ownership while allowing
            // MTE2(next), MTE3(previous), and VEC(current) to run together.
            if (hasPendingOutput) {
                CopyOutCScalarPipelined(pendingOutputOffset, pendingOutputElems);
            }
            const uint32_t cIndex = (currentOffset / kDim) % cDim;
            ComputeScalarOtherReady(currentLocal, rhsLocal.GetValue(cIndex), currentElems);
            pendingOutputOffset = currentOffset;
            pendingOutputElems = currentElems;
            hasPendingOutput = true;
            if (!hasNext) {
                break;
            }
            currentOffset = nextOffset;
            currentElems = nextElems;
        }
        if (hasPendingOutput) {
            CopyOutCScalarPipelined(pendingOutputOffset, pendingOutputElems);
        }
        otherQueue.FreeTensor(rhsLocal);
    }

    __aicore__ inline void PrefetchCScalarRow(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> selfLocal = cScalarPipelineInQueue.AllocTensor<T>();
        CopyDense(selfLocal, selfGm, offset, count);
        cScalarPipelineInQueue.EnQue(selfLocal);
    }

    __aicore__ inline void ComputeScalarOtherReady(LocalTensor<T>& selfLocal,
                                                    T rhsScalar, uint32_t count)
    {
        LocalTensor<uint8_t> outLocal = cScalarPipelineOutQueue.AllocTensor<uint8_t>();
        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            const uint32_t computeElems = AlignElems256<float>(count);
            CompareScalar(mask, selfLocal, rhsScalar, CMPMODE::GT, computeElems);
            LocalTensor<half> oneHalf = selfFloatBuf.Get<half>();
            LocalTensor<half> resultHalf = otherFloatBuf.Get<half>();
            Select(resultHalf, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, computeElems);
            Cast(outLocal, resultHalf, RoundMode::CAST_NONE, computeElems);
        } else if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            const uint32_t computeElems = AlignElems256<half>(count);
            CompareScalar(mask, selfLocal, rhsScalar, CMPMODE::GT, computeElems);
            LocalTensor<half> oneHalf = otherFloatBuf.Get<half>();
            Duplicate(oneHalf, static_cast<half>(1.0F), 128U);
            const uint8_t repeatTime = static_cast<uint8_t>(computeElems / 128U);
            const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
            Select(selfLocal, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
                   repeatParams);
            Cast(outLocal, selfLocal, RoundMode::CAST_NONE, computeElems);
        } else {
            for (uint32_t i = 0U; i < count; ++i) {
                outLocal.SetValue(i, selfLocal.GetValue(i) > rhsScalar ? 1U : 0U);
            }
        }
        cScalarPipelineOutQueue.EnQue(outLocal);
        cScalarPipelineInQueue.FreeTensor(selfLocal);
    }

    __aicore__ inline void CopyOutCScalarPipelined(uint32_t offset, uint32_t count)
    {
        LocalTensor<uint8_t> outLocal = cScalarPipelineOutQueue.DeQue<uint8_t>();
        if (((offset & 31U) == 0U) && ((count & 31U) == 0U)) {
            DataCopy(outGm[offset], outLocal, count);
        } else {
            DataCopyExtParams copyParams{1U, count, 0U, 0U, 0U};
            DataCopyPad(outGm[offset], outLocal, copyParams);
        }
        cScalarPipelineOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void ComputeScalarOther(T rhsScalar, uint32_t count)
    {
        LocalTensor<T> selfLocal = selfQueue.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueue.AllocTensor<uint8_t>();
        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            const uint32_t computeElems = AlignElems256<float>(count);
            CompareScalar(mask, selfLocal, rhsScalar, CMPMODE::GT, computeElems);
            LocalTensor<half> oneHalf = selfFloatBuf.Get<half>();
            LocalTensor<half> resultHalf = otherFloatBuf.Get<half>();
            Select(resultHalf, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, computeElems);
            Cast(outLocal, resultHalf, RoundMode::CAST_NONE, computeElems);
        } else if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            const uint32_t computeElems = AlignElems256<half>(count);
            CompareScalar(mask, selfLocal, rhsScalar, CMPMODE::GT, computeElems);
            LocalTensor<half> oneHalf = otherFloatBuf.Get<half>();
            Duplicate(oneHalf, static_cast<half>(1.0F), 128U);
            const uint8_t repeatTime = static_cast<uint8_t>(computeElems / 128U);
            const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
            Select(selfLocal, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
                   repeatParams);
            Cast(outLocal, selfLocal, RoundMode::CAST_NONE, computeElems);
        } else {
            for (uint32_t i = 0U; i < count; ++i) {
                outLocal.SetValue(i, selfLocal.GetValue(i) > rhsScalar ? 1U : 0U);
            }
        }
        outQueue.EnQue(outLocal);
        selfQueue.FreeTensor(selfLocal);
    }
