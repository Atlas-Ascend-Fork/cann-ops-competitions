    __aicore__ inline void ProcessCRowOtherRowCache()
    {
        const uint32_t inLocalElems = AlignElems256<T>(tileElems);
        const uint32_t outLocalElems = AlignElems256<uint8_t>(tileElems);
        pipe->InitBuffer(selfQueue, kBufferNum, inLocalElems * sizeof(T));
        pipe->InitBuffer(otherQueue, kBufferNum, inLocalElems * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, outLocalElems * sizeof(uint8_t));
        InitTempBuffers();
        pipe->InitBuffer(maskBuf, outLocalElems * sizeof(uint8_t));
        InitLoopConstants();

        const uint32_t nDim = outDims[1];
        const uint32_t cDim = outDims[2];
        const uint32_t hDim = outDims[3];
        const uint32_t wDim = outDims[4];
        const uint32_t wTile = MinU32(tileElems, wDim);
        const uint32_t wBlocks = (wDim + wTile - 1U) / wTile;
        const uint32_t nhTotal = nDim * hDim;
        const uint32_t blockNum = GetBlockNum();
        uint32_t nhBlock = 8U;
        if (cDim * ((nhTotal + nhBlock - 1U) / nhBlock) * wBlocks < blockNum * 2U) {
            nhBlock = 4U;
        }
        if (cDim * ((nhTotal + nhBlock - 1U) / nhBlock) * wBlocks < blockNum) {
            nhBlock = 2U;
        }
        if (cDim * ((nhTotal + nhBlock - 1U) / nhBlock) * wBlocks < blockNum) {
            nhBlock = 1U;
        }
        const uint32_t nhBlocks = (nhTotal + nhBlock - 1U) / nhBlock;
        const uint32_t taskNum = cDim * wBlocks * nhBlocks;

        for (uint32_t task = GetBlockIdx(); task < taskNum; task += blockNum) {
            uint32_t tmp = task;
            const uint32_t wBlock = tmp % wBlocks;
            tmp /= wBlocks;
            const uint32_t c = tmp % cDim;
            const uint32_t nhBlockId = tmp / cDim;

            const uint32_t w0 = wBlock * wTile;
            const uint32_t wt = MinU32(wTile, wDim - w0);
            const uint32_t nh0 = nhBlockId * nhBlock;
            const uint32_t nht = MinU32(nhBlock, nhTotal - nh0);

            LocalTensor<T> rowTensor = otherQueue.AllocTensor<T>();
            CopyDense(rowTensor, otherGm, c * wDim + w0, wt);
            otherQueue.EnQue(rowTensor);
            LocalTensor<T> rowLocal = otherQueue.DeQue<T>();
            PipeBarrier<PIPE_ALL>();

            for (uint32_t i = 0U; i < nht; ++i) {
                const uint32_t nh = nh0 + i;
                const uint32_t n = nh / hDim;
                const uint32_t h = nh - n * hDim;
                const uint32_t offset = (((n * cDim + c) * hDim + h) * wDim) + w0;

                LocalTensor<T> selfLocal = selfQueue.AllocTensor<T>();
                CopyDense(selfLocal, selfGm, offset, wt);
                selfQueue.EnQue(selfLocal);
                ComputeRowOther(rowLocal, wt);
                CopyOut(offset, wt);
            }
            otherQueue.FreeTensor(rowLocal);
        }
    }

    __aicore__ inline void ComputeRowOther(LocalTensor<T>& rhsLocal, uint32_t count)
    {
        LocalTensor<T> selfLocal = selfQueue.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueue.AllocTensor<uint8_t>();
        if constexpr (std::is_same_v<T, float>) {
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            const uint32_t computeElems = AlignElems256<float>(count);
            Compare(mask, selfLocal, rhsLocal, CMPMODE::GT, computeElems);
            LocalTensor<half> oneHalf = selfFloatBuf.Get<half>();
            LocalTensor<half> resultHalf = otherFloatBuf.Get<half>();
            Select(resultHalf, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, computeElems);
            Cast(outLocal, resultHalf, RoundMode::CAST_NONE, computeElems);
        } else if constexpr (std::is_same_v<T, half>) {
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            const uint32_t computeElems = AlignElems256<half>(count);
            Compare(mask, selfLocal, rhsLocal, CMPMODE::GT, computeElems);
            LocalTensor<half> oneHalf = otherFloatBuf.Get<half>();
            Duplicate(oneHalf, static_cast<half>(1.0F), 128U);
            const uint8_t repeatTime = static_cast<uint8_t>(computeElems / 128U);
            const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
            Select(selfLocal, mask, oneHalf, static_cast<half>(0.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
                   repeatParams);
            Cast(outLocal, selfLocal, RoundMode::CAST_NONE, computeElems);
        }
        outQueue.EnQue(outLocal);
        selfQueue.FreeTensor(selfLocal);
    }

    __aicore__ inline uint32_t InitInputOffset(uint32_t outputOffset, const uint32_t (&strides)[kDimNum],
                                               uint32_t (&coords)[kDimNum])
    {
        uint32_t rem = outputOffset;
        uint32_t inputOffset = 0U;
        for (uint32_t i = 0U; i < kDimNum; ++i) {
            uint32_t coord = 0U;
            if (outStrides[i] != 0U) {
                coord = rem / outStrides[i];
                rem -= coord * outStrides[i];
            }
            coords[i] = coord;
            inputOffset += coord * strides[i];
        }
        return inputOffset;
    }

    __aicore__ inline void AdvanceInputOffset(uint32_t (&coords)[kDimNum], uint32_t& inputOffset,
                                              const uint32_t (&strides)[kDimNum], uint32_t step)
    {
        constexpr uint32_t lastIdx = kDimNum - 1U;
        if (step == 0U) {
            return;
        }
        coords[lastIdx] += step;
        inputOffset += step * strides[lastIdx];
        if (coords[lastIdx] < outDims[lastIdx]) {
            return;
        }

        inputOffset -= outDims[lastIdx] * strides[lastIdx];
        coords[lastIdx] = 0U;
        for (int32_t dim = static_cast<int32_t>(lastIdx) - 1; dim >= 0; --dim) {
            const uint32_t dimIdx = static_cast<uint32_t>(dim);
            ++coords[dimIdx];
            inputOffset += strides[dimIdx];
            if (coords[dimIdx] < outDims[dimIdx]) {
                break;
            }
            inputOffset -= outDims[dimIdx] * strides[dimIdx];
            coords[dimIdx] = 0U;
        }
    }

    __aicore__ inline void FillBroadcast(LocalTensor<T>& local, GlobalTensor<T>& gm,
                                         const uint32_t (&strides)[kDimNum],
                                         uint32_t offset, uint32_t count)
    {
        if (FillBroadcastRuns(local, gm, strides, offset, count)) {
            return;
        }
        for (uint32_t i = 0U; i < count; ++i) {
            uint32_t coords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            local.SetValue(i, gm.GetValue(InitInputOffset(offset + i, strides, coords)));
        }
    }

    __aicore__ inline bool FillBroadcastRuns(LocalTensor<T>& local, GlobalTensor<T>& gm,
                                             const uint32_t (&strides)[kDimNum],
                                             uint32_t offset, uint32_t count)
    {
        const uint32_t lastDim = outDims[kDimNum - 1U];
        if (lastDim == 0U) {
            return false;
        }
        uint32_t done = 0U;
        if (strides[kDimNum - 1U] == 0U) {
            uint32_t coords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t inputOffset = InitInputOffset(offset, strides, coords);
            while (done < count) {
                const uint32_t inner = coords[kDimNum - 1U];
                const uint32_t run = MinU32(count - done, lastDim - inner);
                const T value = gm.GetValue(inputOffset);
                uint32_t localIdx = done;
                uint32_t remaining = run;
                if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> ||
                              std::is_same_v<T, int32_t>) {
                    const uint32_t prefix = PrefixToAlign32<T>(localIdx, remaining);
                    for (uint32_t j = 0U; j < prefix; ++j) {
                        local.SetValue(localIdx + j, value);
                    }
                    localIdx += prefix;
                    remaining -= prefix;

                    const uint32_t vecRun = AlignDownElems32<T>(remaining);
                    if (vecRun > 0U && IsLocalAligned32<T>(localIdx)) {
                        Duplicate(local[localIdx], value, vecRun);
                        localIdx += vecRun;
                        remaining -= vecRun;
                    }
                }
                for (uint32_t j = 0U; j < remaining; ++j) {
                    local.SetValue(localIdx + j, value);
                }
                done += run;
                AdvanceInputOffset(coords, inputOffset, strides, run);
            }
            return true;
        }
        if (strides[kDimNum - 1U] == 1U) {
            DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
            uint32_t coords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t inputOffset = InitInputOffset(offset, strides, coords);
            while (done < count) {
                const uint32_t inner = coords[kDimNum - 1U];
                const uint32_t run = MinU32(count - done, lastDim - inner);
                uint32_t localIdx = done;
                uint32_t gmIdx = inputOffset;
                uint32_t remaining = run;

                const uint32_t prefix = PrefixToAlign32<T>(localIdx, remaining);
                for (uint32_t j = 0U; j < prefix; ++j) {
                    local.SetValue(localIdx + j, gm.GetValue(gmIdx + j));
                }
                localIdx += prefix;
                gmIdx += prefix;
                remaining -= prefix;

                const uint32_t vecRun = AlignDownElems32<T>(remaining);
                if (vecRun > 0U && IsLocalAligned32<T>(localIdx)) {
                    DataCopyExtParams copyParams{1U, static_cast<uint32_t>(vecRun * sizeof(T)), 0U, 0U, 0U};
                    DataCopyPad(local[localIdx], gm[gmIdx], copyParams, padParams);
                    localIdx += vecRun;
                    gmIdx += vecRun;
                    remaining -= vecRun;
                }

                for (uint32_t j = 0U; j < remaining; ++j) {
                    local.SetValue(localIdx + j, gm.GetValue(gmIdx + j));
                }
                done += run;
                AdvanceInputOffset(coords, inputOffset, strides, run);
            }
            return true;
        }
        return false;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> selfLocal = selfQueue.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherQueue.AllocTensor<T>();

        if (sameShape) {
            CopyDense(selfLocal, selfGm, offset, count);
            CopyDense(otherLocal, otherGm, offset, count);
        } else if (selfScalar || otherScalar) {
            if (selfScalar) {
                FillScalar(selfLocal, selfGm.GetValue(0U), count);
            } else {
                CopyDense(selfLocal, selfGm, offset, count);
            }
            if (otherScalar) {
                FillScalar(otherLocal, otherGm.GetValue(0U), count);
            } else {
                CopyDense(otherLocal, otherGm, offset, count);
            }
        } else {
            if (selfDense) {
                CopyDense(selfLocal, selfGm, offset, count);
            } else {
                FillBroadcast(selfLocal, selfGm, selfStrides, offset, count);
            }
            if (otherDense) {
                CopyDense(otherLocal, otherGm, offset, count);
            } else {
                FillBroadcast(otherLocal, otherGm, otherStrides, offset, count);
            }
        }
        selfQueue.EnQue(selfLocal);
        otherQueue.EnQue(otherLocal);
    }

    __aicore__ inline void CopyDense(LocalTensor<T>& local, GlobalTensor<T>& gm,
                                     uint32_t offset, uint32_t count)
    {
        if (IsGmCopyAligned<T>(offset, count)) {
            DataCopy(local, gm[offset], count);
            return;
        }
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(local, gm[offset], copyParams, padParams);
    }

    __aicore__ inline void FillScalar(LocalTensor<T>& local, T value, uint32_t count)
    {
        if constexpr (std::is_same_v<T, half> || std::is_same_v<T, float> ||
                      std::is_same_v<T, int32_t>) {
            const uint32_t vecElems = AlignDownElems32<T>(count);
            if (vecElems > 0U) {
                Duplicate(local, value, vecElems);
            }
            for (uint32_t i = vecElems; i < count; ++i) {
                local.SetValue(i, value);
            }
        } else {
            for (uint32_t i = 0U; i < count; ++i) {
                local.SetValue(i, value);
            }
        }
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> selfLocal = selfQueue.DeQue<T>();
        LocalTensor<T> otherLocal = otherQueue.DeQue<T>();
        LocalTensor<uint8_t> outLocal = outQueue.AllocTensor<uint8_t>();
        for (uint32_t i = 0U; i < count; ++i) {
            outLocal.SetValue(i, selfLocal.GetValue(i) > otherLocal.GetValue(i) ? 1U : 0U);
        }
        outQueue.EnQue(outLocal);
        selfQueue.FreeTensor(selfLocal);
        otherQueue.FreeTensor(otherLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<uint8_t> outLocal = outQueue.DeQue<uint8_t>();
        if (((offset & 31U) == 0U) && ((count & 31U) == 0U)) {
            DataCopy(outGm[offset], outLocal, count);
            outQueue.FreeTensor(outLocal);
            return;
        }
        DataCopyExtParams copyParams{1U, count, 0U, 0U, 0U};
        DataCopyPad(outGm[offset], outLocal, copyParams);
        outQueue.FreeTensor(outLocal);
    }
