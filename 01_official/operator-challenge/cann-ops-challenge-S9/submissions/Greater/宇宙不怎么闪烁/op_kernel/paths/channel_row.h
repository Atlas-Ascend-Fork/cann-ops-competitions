    __aicore__ inline bool CanProcessCRowOtherBroadcast() const
    {
        const uint32_t nDim = outDims[1];
        const uint32_t cDim = outDims[2];
        const uint32_t hDim = outDims[3];
        const uint32_t wDim = outDims[4];
        const uint32_t blockNum = GetBlockNum();
        if (nDim == 0U || cDim == 0U || hDim < 16U || wDim == 0U || blockNum == 0U) {
            return false;
        }
        const uint32_t planeCount = nDim * cDim;
        const uint32_t planeElems = hDim * wDim;
        return AlignedUnitCount(planeCount, planeElems) >= blockNum;
    }

    __aicore__ inline void ProcessCRowOtherBroadcast()
    {
        const uint32_t nDim = outDims[1];
        const uint32_t cDim = outDims[2];
        const uint32_t hDim = outDims[3];
        const uint32_t wDim = outDims[4];
        const uint32_t rowPitchElems = 256U / sizeof(T);
        const uint32_t hBlock = MinU32(kCRowBroadcastMaxRows, hDim);
        const uint32_t maxTileElems = hBlock * rowPitchElems;
        const uint32_t outLocalElems = AlignElems256<uint8_t>(maxTileElems);
        const uint32_t planeElems = hDim * wDim;
        const uint32_t planeCount = nDim * cDim;

        uint32_t planeStart = 0U;
        uint32_t planeEnd = 0U;
        DeriveAlignedItemRange(planeCount, planeElems, planeStart, planeEnd);
        if (planeStart >= planeEnd) {
            return;
        }

        pipe->InitBuffer(cRowPipelineInQueue, 2U, maxTileElems * sizeof(T));
        pipe->InitBuffer(otherQueue, kBufferNum, rowPitchElems * sizeof(T));
        pipe->InitBuffer(cRowPipelineOutQueue, 2U, outLocalElems * sizeof(uint8_t));
        pipe->InitBuffer(broadcastBuf, maxTileElems * sizeof(T));
        InitTempBuffers(maxTileElems);
        pipe->InitBuffer(maskBuf, outLocalElems * sizeof(uint8_t));
        InitLoopConstants(maxTileElems);

        LocalTensor<T> rhsExpanded = broadcastBuf.Get<T>();
        for (uint32_t plane = planeStart; plane < planeEnd; ++plane) {
            const uint32_t n = plane / cDim;
            const uint32_t c = plane - n * cDim;
            for (uint32_t w0 = 0U; w0 < wDim; w0 += rowPitchElems) {
                const uint32_t wt = MinU32(rowPitchElems, wDim - w0);
                LocalTensor<T> rhsTensor = otherQueue.AllocTensor<T>();
                CopyDense(rhsTensor, otherGm, c * wDim + w0, wt);
                otherQueue.EnQue(rhsTensor);
                LocalTensor<T> rhsLocal = otherQueue.DeQue<T>();
                PipeBarrier<PIPE_ALL>();
                BuildCRowBroadcast(rhsExpanded, rhsLocal, hBlock, rowPitchElems);
                otherQueue.FreeTensor(rhsLocal);
                ProcessCRowPlaneTile(rhsExpanded, n, c, w0, wt, hBlock,
                                     cDim, hDim, wDim, rowPitchElems);
            }
        }
    }

    __aicore__ inline void BuildCRowBroadcast(LocalTensor<T>& dst, LocalTensor<T>& rhsLocal,
                                               uint32_t hBlock, uint32_t rowPitchElems)
    {
        const uint64_t mask = static_cast<uint64_t>(rowPitchElems);
        const uint8_t repeat = static_cast<uint8_t>(hBlock);
        const UnaryRepeatParams repeatParams{1U, 1U, 8U, 0U};
        Adds(dst, rhsLocal, static_cast<T>(0), mask, repeat, repeatParams);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessCRowPlaneTile(LocalTensor<T>& rhsExpanded,
                                                 uint32_t n, uint32_t c,
                                                 uint32_t w0, uint32_t wt,
                                                 uint32_t hBlock, uint32_t cDim,
                                                 uint32_t hDim, uint32_t wDim,
                                                 uint32_t rowPitchElems)
    {
        uint32_t currentH = 0U;
        uint32_t currentRows = MinU32(hBlock, hDim);
        uint32_t currentOffset = (((n * cDim + c) * hDim) * wDim) + w0;
        PrefetchCRowBlock(currentOffset, currentRows, wt, wDim, rowPitchElems);
        uint32_t pendingOffset = 0U;
        uint32_t pendingRows = 0U;
        bool hasPendingOutput = false;

        while (true) {
            LocalTensor<T> currentLocal = cRowPipelineInQueue.DeQue<T>();
            const uint32_t nextH = currentH + currentRows;
            const bool hasNext = nextH < hDim;
            uint32_t nextRows = 0U;
            if (hasNext) {
                nextRows = MinU32(hBlock, hDim - nextH);
                const uint32_t nextOffset = (((n * cDim + c) * hDim + nextH) * wDim) + w0;
                PrefetchCRowBlock(nextOffset, nextRows, wt, wDim, rowPitchElems);
            }
            if (hasPendingOutput) {
                CopyOutCRowPipelined(pendingOffset, pendingRows, wt, wDim, rowPitchElems);
            }
            ComputeCRowBroadcastReady(currentLocal, rhsExpanded, currentRows * rowPitchElems);
            pendingOffset = currentOffset;
            pendingRows = currentRows;
            hasPendingOutput = true;
            if (!hasNext) {
                break;
            }
            currentH = nextH;
            currentRows = nextRows;
            currentOffset = (((n * cDim + c) * hDim + currentH) * wDim) + w0;
        }
        CopyOutCRowPipelined(pendingOffset, pendingRows, wt, wDim, rowPitchElems);
    }

    __aicore__ inline void PrefetchCRowBlock(uint32_t offset, uint32_t rows,
                                              uint32_t width, uint32_t fullWidth,
                                              uint32_t rowPitchElems)
    {
        LocalTensor<T> selfLocal = cRowPipelineInQueue.AllocTensor<T>();
        const uint32_t blockBytes = width * sizeof(T);
        const uint32_t paddedBlockBytes = AlignUp32Bytes(blockBytes);
        const uint32_t rowPitchBytes = rowPitchElems * sizeof(T);
        const uint32_t dstStride = (rowPitchBytes - paddedBlockBytes) / 32U;
        const uint32_t srcStride = (fullWidth - width) * sizeof(T);
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows), blockBytes,
                                     srcStride, dstStride, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(selfLocal, selfGm[offset], copyParams, padParams);
        cRowPipelineInQueue.EnQue(selfLocal);
    }

    __aicore__ inline void ComputeCRowBroadcastReady(LocalTensor<T>& selfLocal,
                                                      LocalTensor<T>& rhsExpanded,
                                                      uint32_t count)
    {
        LocalTensor<uint8_t> outLocal = cRowPipelineOutQueue.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        Compare(mask, selfLocal, rhsExpanded, CMPMODE::GT, count);
        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<half> oneHalf = selfFloatBuf.Get<half>();
            LocalTensor<half> resultHalf = otherFloatBuf.Get<half>();
            Select(resultHalf, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
            Cast(outLocal, resultHalf, RoundMode::CAST_NONE, count);
        } else if constexpr (std::is_same_v<T, half>) {
            LocalTensor<half> oneHalf = otherFloatBuf.Get<half>();
            Duplicate(oneHalf, static_cast<half>(1.0F), 128U);
            const uint8_t repeatTime = static_cast<uint8_t>(count / 128U);
            const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
            Select(selfLocal, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
                   repeatParams);
            Cast(outLocal, selfLocal, RoundMode::CAST_NONE, count);
        }
        cRowPipelineOutQueue.EnQue(outLocal);
        cRowPipelineInQueue.FreeTensor(selfLocal);
    }

    __aicore__ inline void CopyOutCRowPipelined(uint32_t offset, uint32_t rows,
                                                 uint32_t width, uint32_t fullWidth,
                                                 uint32_t rowPitchElems)
    {
        LocalTensor<uint8_t> outLocal = cRowPipelineOutQueue.DeQue<uint8_t>();
        const uint32_t paddedBlockBytes = AlignUp32Bytes(width);
        const uint32_t srcStride = (rowPitchElems - paddedBlockBytes) / 32U;
        const uint32_t dstStride = fullWidth - width;
        DataCopyExtParams copyParams{static_cast<uint16_t>(rows), width,
                                     srcStride, dstStride, 0U};
        DataCopyPad(outGm[offset], outLocal, copyParams);
        cRowPipelineOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void ProcessCRowOther()
    {
        if (CanProcessCRowOtherBroadcast()) {
            ProcessCRowOtherBroadcast();
            return;
        }
        if constexpr (std::is_same_v<T, half>) {
            if (outDims[4] <= tileElems) {
                const uint32_t inLocalElems = AlignElems256<T>(tileElems);
                const uint32_t outLocalElems = AlignElems256<uint8_t>(tileElems);
                pipe->InitBuffer(selfQueue, kBufferNum, inLocalElems * sizeof(T));
                pipe->InitBuffer(otherQueue, kBufferNum, inLocalElems * sizeof(T));
                pipe->InitBuffer(outQueue, kBufferNum, outLocalElems * sizeof(uint8_t));
                pipe->InitBuffer(maskBuf, outLocalElems * sizeof(uint8_t));

                const uint32_t nDim = outDims[1];
                const uint32_t cDim = outDims[2];
                const uint32_t hDim = outDims[3];
                const uint32_t wDim = outDims[4];
                uint32_t maxRows = tileElems / wDim;
                if (maxRows == 0U) {
                    maxRows = 1U;
                }
                uint32_t hBlock = MinU32(8U, maxRows);
                hBlock = MinU32(hBlock, hDim);
                const uint32_t hBlocks = (hDim + hBlock - 1U) / hBlock;
                const uint32_t taskNum = nDim * cDim * hBlocks;
                const uint32_t blockNum = GetBlockNum();

                for (uint32_t task = GetBlockIdx(); task < taskNum; task += blockNum) {
                    uint32_t tmp = task;
                    const uint32_t hBlockId = tmp % hBlocks;
                    tmp /= hBlocks;
                    const uint32_t c = tmp % cDim;
                    const uint32_t n = tmp / cDim;

                    const uint32_t h0 = hBlockId * hBlock;
                    const uint32_t ht = MinU32(hBlock, hDim - h0);
                    const uint32_t elems = ht * wDim;
                    const uint32_t offset = (((n * cDim + c) * hDim + h0) * wDim);

                    LocalTensor<T> selfLocal = selfQueue.AllocTensor<T>();
                    CopyDense(selfLocal, selfGm, offset, elems);
                    selfQueue.EnQue(selfLocal);

                    LocalTensor<T> otherLocal = otherQueue.AllocTensor<T>();
                    CopyDense(otherLocal, otherGm, c * wDim, wDim);
                    PipeBarrier<PIPE_ALL>();
                    for (uint32_t h = 1U; h < ht; ++h) {
                        const uint32_t localBase = h * wDim;
                        for (uint32_t w = 0U; w < wDim; ++w) {
                            otherLocal.SetValue(localBase + w, otherLocal.GetValue(w));
                        }
                    }
                    otherQueue.EnQue(otherLocal);

                    Compute(elems);
                    CopyOut(offset, elems);
                }
                return;
            }
        }
        ProcessCRowOtherRowCache();
    }
