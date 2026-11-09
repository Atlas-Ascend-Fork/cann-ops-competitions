    __aicore__ inline uint32_t DensePipelineElems() const
    {
        if constexpr (std::is_same_v<T, float>) {
            return kDensePipelineFloatElems;
        } else if constexpr (std::is_same_v<T, half>) {
            return kDensePipelineHalfElems;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return kDensePipelineInt8Elems;
        } else {
            return 0U;
        }
    }

    __aicore__ inline bool CanProcessDenseSameShapePipelined() const
    {
        const uint32_t pipelineElems = DensePipelineElems();
        return sameShape && pipelineElems != 0U &&
               (pipelineElems % blockElems) == 0U;
    }

    __aicore__ inline void ProcessDenseSameShapePipelined()
    {
        selfGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        otherGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t pipelineElems = DensePipelineElems();
        const uint32_t inLocalElems = AlignElems256<T>(pipelineElems);
        const uint32_t outLocalElems = AlignElems256<uint8_t>(pipelineElems);
        uint32_t inSlotBytes = inLocalElems * sizeof(T);
        if constexpr (std::is_same_v<T, int8_t>) {
            inSlotBytes += kInt8InputSlotPadBytes;
        }
        pipe->InitBuffer(densePipelineSelfQueue, 2U, inSlotBytes);
        if constexpr (std::is_same_v<T, int8_t>) {
            pipe->InitBuffer(broadcastBuf, kInt8BankPadSelfToOtherBytes);
        }
        pipe->InitBuffer(densePipelineOtherQueue, 2U, inSlotBytes);
        if constexpr (std::is_same_v<T, int8_t>) {
            pipe->InitBuffer(bankPadBuf, kInt8BankPadOtherToOutBytes);
        }
        pipe->InitBuffer(densePipelineOutQueue, 2U,
                         outLocalElems * sizeof(uint8_t));
        if constexpr (std::is_same_v<T, int8_t>) {
            pipe->InitBuffer(bankTailPadBuf, kInt8BankPadOutToSelfHalfBytes);
        }
        InitTempBuffers(pipelineElems);
        // Compare emits one packed result bit per input element.  Dense fp32
        // does not need the byte-per-element mask used by the generic path;
        // reclaim that UB space for a larger double-buffered tile.
        const uint32_t maskBytes =
            AlignElems256<uint8_t>((pipelineElems + 7U) / 8U);
        if constexpr (!std::is_same_v<T, int8_t>) {
            pipe->InitBuffer(maskBuf, maskBytes);
        }
        InitLoopConstants(pipelineElems);

        // Large FP32/INT8 streams use interleaved complete tiles. Only the
        // global final tile is short, rather than a fragment on every core.
        // Small tensors and FP16 retain the original balanced range.
        const bool striped = (std::is_same_v<T, float> || std::is_same_v<T, int8_t>) &&
            totalSize / pipelineElems >= GetBlockNum();
        const uint32_t rangeEnd = striped ? totalSize : endElem;
        uint32_t currentOffset = striped ? GetBlockIdx() * pipelineElems : startElem;
        uint32_t currentElems = MinU32(pipelineElems, rangeEnd - currentOffset);
        PrefetchDenseSameShape(currentOffset, currentElems);
        uint32_t pendingOutputOffset = 0U;
        uint32_t pendingOutputElems = 0U;
        bool hasPendingOutput = false;

        while (true) {
            LocalTensor<T> currentSelf = densePipelineSelfQueue.DeQue<T>();
            LocalTensor<T> currentOther = densePipelineOtherQueue.DeQue<T>();
            const uint64_t next = static_cast<uint64_t>(currentOffset) +
                (striped ? static_cast<uint64_t>(pipelineElems) * GetBlockNum() : currentElems);
            const bool hasNext = next < rangeEnd;
            const uint32_t nextOffset = hasNext ? static_cast<uint32_t>(next) : 0U;
            uint32_t nextElems = 0U;
            if (hasNext) {
                nextElems = MinU32(pipelineElems, rangeEnd - nextOffset);
                PrefetchDenseSameShape(nextOffset, nextElems);
            }

            // Queue ownership lets this MTE3 transfer overlap the current
            // vector sequence and the next pair of MTE2 reads.
            if (hasPendingOutput) {
                CopyOutDenseSameShape(pendingOutputOffset, pendingOutputElems);
            }
            ComputeDenseSameShapeReady(currentSelf, currentOther, currentElems);
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
            CopyOutDenseSameShape(pendingOutputOffset, pendingOutputElems);
        }
    }

    __aicore__ inline void PrefetchDenseSameShape(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> selfLocal = densePipelineSelfQueue.AllocTensor<T>();
        LocalTensor<T> otherLocal = densePipelineOtherQueue.AllocTensor<T>();
        CopyDense(selfLocal, selfGm, offset, count);
        CopyDense(otherLocal, otherGm, offset, count);
        densePipelineSelfQueue.EnQue(selfLocal);
        densePipelineOtherQueue.EnQue(otherLocal);
    }

    __aicore__ inline void ComputeDenseSameShapeReady(LocalTensor<T>& selfLocal,
                                                       LocalTensor<T>& otherLocal,
                                                       uint32_t count)
    {
        LocalTensor<uint8_t> outLocal = densePipelineOutQueue.AllocTensor<uint8_t>();
        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            const uint32_t computeElems = ((count + 127U) / 128U) * 128U;
            Compare(mask, selfLocal, otherLocal, CMPMODE::GT, computeElems);
            LocalTensor<half> oneHalf = selfLocal.template ReinterpretCast<half>();
            LocalTensor<half> resultHalf = otherLocal.template ReinterpretCast<half>();
            Duplicate(oneHalf, static_cast<half>(1.0F), 128U);
            const uint8_t repeatTime = static_cast<uint8_t>(computeElems / 128U);
            const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
            Select(resultHalf, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
                   repeatParams);
            Cast(outLocal, resultHalf, RoundMode::CAST_NONE, computeElems);
        } else if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            const uint32_t computeElems = AlignElems256<half>(count);
            Compare(mask, selfLocal, otherLocal, CMPMODE::GT, computeElems);
            // Both inputs are dead after Compare.  Keep one 128-element
            // vector of ones in the rhs slot and broadcast it across Select
            // repeats; use the scalar operand for false instead of writing a
            // full zero tensor.  This removes two output-sized constant fills
            // from the dense fp16 vector chain.
            Duplicate(otherLocal, static_cast<half>(1.0F), 128U);
            const uint8_t repeatTime = static_cast<uint8_t>(computeElems / 128U);
            const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
            Select(selfLocal, mask, otherLocal, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
                   repeatParams);
            Cast(outLocal, selfLocal, RoundMode::CAST_NONE, computeElems);
        } else if constexpr (std::is_same_v<T, int8_t>) {
            LocalTensor<half> selfHalf = selfFloatBuf.Get<half>();
            LocalTensor<half> otherHalf = otherFloatBuf.Get<half>();
            LocalTensor<uint8_t> mask = outLocal;
            const uint32_t computeElems = AlignElems256<int8_t>(count);
            Cast(selfHalf, selfLocal, RoundMode::CAST_NONE, computeElems);
            Cast(otherHalf, otherLocal, RoundMode::CAST_NONE, computeElems);
            Compare(mask, selfHalf, otherHalf, CMPMODE::GT, computeElems);
            // The rhs cast buffer is dead after Compare.  Keep only one
            // 128-element vector of ones there and broadcast it across Select
            // repeats instead of writing a full-tile constant tensor.
            Duplicate(otherHalf, static_cast<half>(1.0F), 128U);
            const uint8_t repeatTime = static_cast<uint8_t>(computeElems / 128U);
            const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
            Select(selfHalf, mask, otherHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
                   repeatParams);
            Cast(outLocal, selfHalf, RoundMode::CAST_NONE, computeElems);
        }
        densePipelineOutQueue.EnQue(outLocal);
        densePipelineSelfQueue.FreeTensor(selfLocal);
        densePipelineOtherQueue.FreeTensor(otherLocal);
    }

    __aicore__ inline void CopyOutDenseSameShape(uint32_t offset, uint32_t count)
    {
        LocalTensor<uint8_t> outLocal = densePipelineOutQueue.DeQue<uint8_t>();
        if (((offset & 31U) == 0U) && ((count & 31U) == 0U)) {
            DataCopy(outGm[offset], outLocal, count);
        } else {
            DataCopyExtParams copyParams{1U, count, 0U, 0U, 0U};
            DataCopyPad(outGm[offset], outLocal, copyParams);
        }
        densePipelineOutQueue.FreeTensor(outLocal);
    }
