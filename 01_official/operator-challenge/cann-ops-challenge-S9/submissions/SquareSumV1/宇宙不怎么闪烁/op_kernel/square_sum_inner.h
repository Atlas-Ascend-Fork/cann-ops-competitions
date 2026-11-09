#pragma once
// Private SquareSumKernel<T> member definitions; included inside the class.

    __aicore__ inline void InitInnerBuffers()
    {
        const uint32_t alignedRowElems = AlignElems32<T>(accTileElems);
        pipe->InitBuffer(inQueue, 2U,
                         alignedRowElems * kInnerPipelinedBatchRows * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, alignedRowElems * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe->InitBuffer(calcBuf,
                             alignedRowElems * kInnerPipelinedBatchRows * sizeof(float));
        }
        pipe->InitBuffer(accBuf, alignedRowElems * sizeof(float));
    }

    __aicore__ inline bool CanProcessAtomicInner() const
    {
        return std::is_same_v<T, float> && outputSize == innerSize &&
               square_sum_policy::ContiguousInner(inDims, reduceMask) &&
               atomicBatchRows != 0U;
    }

    __aicore__ inline void InitAtomicInnerBuffers()
    {
        const uint32_t rowPitch = AlignElems32<float>(innerSize);
        pipe->InitBuffer(inQueue, 2U,
                         atomicBatchRows *
                             atomicMainCols * sizeof(float));
        if (atomicTailCols != 0U) {
            pipe->InitBuffer(rank4InQueue, 2U,
                             atomicBatchRows * AlignElems32<float>(atomicTailCols) * sizeof(float));
        }
        pipe->InitBuffer(accBuf, rowPitch * sizeof(float));
    }

    __aicore__ inline void CopyAtomicInnerInput(uint32_t rowBase, uint32_t rowCount)
    {
        const uint32_t rowElems = innerSize;
        LocalTensor<float> mainLocal = inQueue.AllocTensor<float>();
        DataCopyExtParams mainCopyParams{
            static_cast<uint16_t>(rowCount),
            static_cast<uint32_t>(atomicMainCols * sizeof(float)),
            static_cast<uint32_t>(atomicTailCols * sizeof(float)),
            0U, 0U};
        DataCopyPadExtParams<float> noPadParams{false, 0U, 0U, 0.0F};
        DataCopyPad(mainLocal, inputGm[rowBase * rowElems], mainCopyParams,
                    noPadParams);
        inQueue.EnQue(mainLocal);

        if (atomicTailCols == 0U) return;
        const uint32_t tailPitch =
            AlignElems32<float>(atomicTailCols);
        LocalTensor<float> tailLocal = rank4InQueue.AllocTensor<float>();
        DataCopyExtParams tailCopyParams{
            static_cast<uint16_t>(rowCount),
            static_cast<uint32_t>(atomicTailCols * sizeof(float)),
            static_cast<uint32_t>(atomicMainCols * sizeof(float)),
            0U, 0U};
        DataCopyPadExtParams<float> tailPadParams{
            true, 0U,
            static_cast<uint8_t>(tailPitch - atomicTailCols), 0.0F};
        DataCopyPad(tailLocal,
                    inputGm[rowBase * rowElems + atomicMainCols],
                    tailCopyParams, tailPadParams);
        rank4InQueue.EnQue(tailLocal);
    }

    __aicore__ inline void ProcessAtomicInner()
    {
        const uint32_t rowElems = innerSize;
        const uint32_t rowPitch = AlignElems32<float>(rowElems);
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx == 0U) {
            InitGlobalMemory(outputGm, outputSize, 0.0F);
        }
        SyncAll();

        uint32_t rowStart = 0U;
        uint32_t rowEnd = 0U;
        CalcRange(reduceSize, rowStart, rowEnd);
        LocalTensor<float> accLocal = accBuf.Get<float>();
        Duplicate(accLocal, 0.0F, rowPitch);

        uint32_t rowBase = rowStart;
        uint32_t rowCount = MinU32(atomicBatchRows, rowEnd - rowBase);
        if (rowCount != 0U) {
            CopyAtomicInnerInput(rowBase, rowCount);
        }
        while (rowCount != 0U) {
            LocalTensor<float> mainReady = inQueue.DeQue<float>();
            LocalTensor<float> tailReady;
            if (atomicTailCols != 0U) tailReady = rank4InQueue.DeQue<float>();
            const uint32_t nextRowBase = rowBase + rowCount;
            const uint32_t nextRowCount = nextRowBase < rowEnd
                ? MinU32(atomicBatchRows, rowEnd - nextRowBase)
                : 0U;
            if (nextRowCount != 0U) {
                CopyAtomicInnerInput(nextRowBase, nextRowCount);
            }
            AccumulateSquaredRows(accLocal, mainReady,
                                  atomicMainCols, rowCount);
            if (atomicTailCols != 0U) {
                AccumulateSquaredRows(accLocal[atomicMainCols], tailReady,
                                      AlignElems32<float>(atomicTailCols), rowCount);
            }
            inQueue.FreeTensor(mainReady);
            if (atomicTailCols != 0U) rank4InQueue.FreeTensor(tailReady);
            rowBase = nextRowBase;
            rowCount = nextRowCount;
        }

        // All cores initialize/synchronize, including empty row ranges. Sum
        // unscaled partials; no compensation for floating-point order.
        event_t eventID = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        SetAtomicAdd<float>();
        DataCopyExtParams outParams{1U, static_cast<uint32_t>(rowElems * sizeof(float)),
                                    0U, 0U, 0U};
        DataCopyPad(outputGm, accLocal, outParams);
        SetAtomicNone();
    }

    __aicore__ inline void InitRank4Reduce02Buffers()
    {
        const uint32_t alignedRowT = AlignElems32<T>(accTileElems);
        const uint32_t alignedRowF = AlignElems32<float>(accTileElems);
        if (CanProcessRank4Reduce02BatchB()) {
            const uint32_t inputElems = kRank4Reduce02BatchB * inDims[3] * alignedRowT;
            pipe->InitBuffer(rank4InQueue, 2U, inputElems * sizeof(T));
            pipe->InitBuffer(outQueue, kBufferNum,
                             kRank4Reduce02BatchB * alignedRowT * sizeof(T));
            pipe->InitBuffer(accBuf,
                             kRank4Reduce02BatchB * alignedRowF * sizeof(float));
            return;
        }
        const uint32_t batchElemsT = alignedRowT * kInnerReduceBatchRows;
        pipe->InitBuffer(rank4InQueue, 2U, batchElemsT * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, alignedRowT * sizeof(T));
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            const uint32_t batchElemsF = alignedRowF * kInnerReduceBatchRows;
            pipe->InitBuffer(calcBuf, batchElemsF * sizeof(float));
            pipe->InitBuffer(squareBuf, batchElemsF * sizeof(float));
        }
        // AccumulateSquaredRows also touches half/BF16 DMA padding. A short
        // tile can have alignedRowT=16 but alignedRowF=8; reserve the wider row.
        pipe->InitBuffer(accBuf, alignedRowT * sizeof(float));
    }

    __aicore__ inline void InitRank4Reduce02CoreRangeBuffers()
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t maxBCount = (inDims[2] + blockNum - 1U) / blockNum;
        const uint32_t alignedRowT = AlignElems32<T>(inDims[4]);
        const uint32_t alignedRowF = AlignElems32<float>(inDims[4]);
        pipe->InitBuffer(rank4InQueue, 2U,
                         maxBCount * inDims[3] * alignedRowT * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum,
                         maxBCount * alignedRowT * sizeof(T));
        pipe->InitBuffer(accBuf,
                         maxBCount * alignedRowF * sizeof(float));
    }

    __aicore__ inline bool CanProcessRank4Reduce02BatchB() const
    {
        const uint32_t dimB = inDims[2];
        const uint32_t dimD = inDims[4];
        return dimB != 0U && accTileElems == dimD &&
               taskNum == (dimB + kRank4Reduce02BatchB - 1U) /
                          kRank4Reduce02BatchB;
    }

    __aicore__ inline bool CanProcessBalancedInner() const
    {
        return std::is_same_v<T, bfloat16_t> &&
               square_sum_policy::ContiguousInner(inDims, reduceMask) &&
               balancedBatchRows != 0U;
    }

    __aicore__ inline void InitBalancedInnerBuffers()
    {
        const uint32_t rowPitch = AlignElems32<T>(innerSize);
        const uint32_t batchElems = balancedBatchRows * rowPitch;
        pipe->InitBuffer(inQueue, 2U, batchElems * sizeof(T));
        pipe->InitBuffer(outQueue, 1U, rowPitch * sizeof(T));
        pipe->InitBuffer(calcBuf, batchElems * sizeof(float));
        pipe->InitBuffer(accBuf, rowPitch * sizeof(float));
    }

    __aicore__ inline void CopyBalancedInnerBatch(uint32_t inputBase,
                                                  uint32_t outputCount,
                                                  uint32_t reduceBase,
                                                  uint32_t reduceCount)
    {
        const uint32_t rowPitch = AlignElems32<T>(outputCount);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - outputCount);
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(reduceCount),
            static_cast<uint32_t>(outputCount * sizeof(T)),
            static_cast<uint32_t>((innerSize - outputCount) * sizeof(T)),
            0U, 0U};
        DataCopyPadExtParams<T> padParams{
            rightPadding != 0U, 0U, rightPadding, static_cast<T>(0)};
        DataCopyPad(inputLocal,
                    inputGm[inputBase + reduceBase * innerSize],
                    copyParams, padParams);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessBalancedInner()
    {
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcPlaneOutputRange(start, end);
        for (uint32_t outputOffset = start; outputOffset < end;) {
            const uint32_t innerOffset = outputOffset % innerSize;
            const uint32_t outputCount = MinU32(
                end - outputOffset, innerSize - innerOffset);
            const uint32_t rowPitch = AlignElems32<T>(outputCount);
            const uint32_t inputBase = BaseInputOffset(outputOffset);
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, rowPitch);

            uint32_t reduceBase = 0U;
            uint32_t reduceCount =
                MinU32(balancedBatchRows, reduceSize);
            CopyBalancedInnerBatch(inputBase, outputCount, reduceBase, reduceCount);
            while (true) {
                LocalTensor<T> ready = inQueue.DeQue<T>();
                const uint32_t nextReduceBase = reduceBase + reduceCount;
                const bool hasNext = nextReduceBase < reduceSize;
                uint32_t nextReduceCount = 0U;
                if (hasNext) {
                    nextReduceCount = MinU32(
                        balancedBatchRows,
                        reduceSize - nextReduceBase);
                    CopyBalancedInnerBatch(inputBase, outputCount, nextReduceBase,
                                           nextReduceCount);
                }

                const uint32_t batchElems = reduceCount * rowPitch;
                LocalTensor<float> calcLocal = calcBuf.Get<float>();
                Cast(calcLocal, ready, RoundMode::CAST_NONE, batchElems);
                Mul(calcLocal, calcLocal, calcLocal, batchElems);
                AccumulateRows(accLocal, calcLocal, rowPitch, reduceCount);
                inQueue.FreeTensor(ready);

                if (!hasNext) {
                    break;
                }
                reduceBase = nextReduceBase;
                reduceCount = nextReduceCount;
            }

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, accLocal, outputCount);
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{
                1U, static_cast<uint32_t>(outputCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outputOffset], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
            outputOffset += outputCount;
        }
    }

    __aicore__ inline void ProcessInnerAccum()
    {
        const uint32_t lastReduceDim = LastReduceDim();
        if (lastReduceDim >= kDimNum) {
            return;
        }
        const uint32_t innerReduceSize = inDims[lastReduceDim];
        const uint32_t outerReduceSize = innerReduceSize == 0U ? 0U : reduceSize / innerReduceSize;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t outputOffset = task * accTileElems;
            if (outputOffset >= outputSize) {
                continue;
            }
            const uint32_t outputCount = MinU32(accTileElems, outputSize - outputOffset);
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            uint32_t localOutputOffset = 0U;
            while (localOutputOffset < outputCount) {
                const uint32_t globalOutputOffset = outputOffset + localOutputOffset;
                const uint32_t innerOffset = globalOutputOffset % innerSize;
                const uint32_t count = MinU32(outputCount - localOutputOffset, innerSize - innerOffset);
                const uint32_t rowPitch = AlignElems32<T>(count);
                const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - count);
                LocalTensor<float> accLocal = accBuf.Get<float>();
                Duplicate(accLocal, 0.0F, rowPitch);
                const uint32_t inputBase = BaseInputOffset(globalOutputOffset);
                uint32_t outerReduceCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
                uint32_t outerReduceOffset = 0U;
                for (uint32_t outerReduce = 0U; outerReduce < outerReduceSize; ++outerReduce) {
                    uint32_t reduceBase = 0U;
                    uint32_t reduceCount =
                        MinU32(kInnerPipelinedBatchRows, innerReduceSize);
                    LocalTensor<T> firstInput = inQueue.AllocTensor<T>();
                    DataCopyExtParams firstCopyParams{
                        static_cast<uint16_t>(reduceCount),
                        static_cast<uint32_t>(count * sizeof(T)),
                        static_cast<uint32_t>((innerSize - count) * sizeof(T)), 0U, 0U};
                    DataCopyPadExtParams<T> padParams{rightPadding != 0U, 0U, rightPadding,
                                                      static_cast<T>(0)};
                    DataCopyPad(firstInput, inputGm[inputBase + outerReduceOffset],
                                firstCopyParams, padParams);
                    inQueue.EnQue(firstInput);

                    while (true) {
                        LocalTensor<T> ready = inQueue.DeQue<T>();
                        const uint32_t nextReduceBase = reduceBase + reduceCount;
                        uint32_t nextReduceCount = 0U;
                        if (nextReduceBase < innerReduceSize) {
                            nextReduceCount = MinU32(kInnerPipelinedBatchRows,
                                                     innerReduceSize - nextReduceBase);
                            LocalTensor<T> nextInput = inQueue.AllocTensor<T>();
                            DataCopyExtParams nextCopyParams{
                                static_cast<uint16_t>(nextReduceCount),
                                static_cast<uint32_t>(count * sizeof(T)),
                                static_cast<uint32_t>((innerSize - count) * sizeof(T)),
                                0U, 0U};
                            DataCopyPad(nextInput,
                                        inputGm[inputBase + outerReduceOffset +
                                                nextReduceBase * innerSize],
                                        nextCopyParams, padParams);
                            inQueue.EnQue(nextInput);
                        }

                        AccumulateSquaredRows(accLocal, ready, rowPitch, reduceCount);
                        inQueue.FreeTensor(ready);

                        if (nextReduceBase >= innerReduceSize) {
                            break;
                        }
                        reduceBase = nextReduceBase;
                        reduceCount = nextReduceCount;
                    }
                    AdvanceReduceOffset(lastReduceDim, outerReduceCoords, outerReduceOffset);
                }
                for (uint32_t i = 0U; i < count; ++i) {
                    if constexpr (std::is_same_v<T, bfloat16_t>) {
                        outputLocal.SetValue(localOutputOffset + i, FloatToBf16(accLocal.GetValue(i)));
                    } else {
                        outputLocal.SetValue(localOutputOffset + i, static_cast<T>(accLocal.GetValue(i)));
                    }
                }
                localOutputOffset += count;
            }
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(outputCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outputOffset], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void CopyRank4Reduce02Input(uint32_t dimB, uint32_t dimC,
                                                   uint32_t dimD, uint32_t b,
                                                   uint32_t dBase, uint32_t count,
                                                   uint8_t rightPadding,
                                                   uint32_t a, uint32_t cBase,
                                                   uint32_t cCount)
    {
        const uint32_t gmOffset = ((a * dimB + b) * dimC + cBase) * dimD + dBase;
        LocalTensor<T> inputLocal = rank4InQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{static_cast<uint16_t>(cCount),
                                     static_cast<uint32_t>(count * sizeof(T)),
                                     static_cast<uint32_t>((dimD - count) * sizeof(T)),
                                     0U, 0U};
        DataCopyPadExtParams<T> padParams{rightPadding != 0U, 0U, rightPadding,
                                          static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
        rank4InQueue.EnQue(inputLocal);
    }

    __aicore__ inline void CopyRank4Reduce02BatchBInput(uint32_t dimB,
                                                        uint32_t dimC,
                                                        uint32_t dimD,
                                                        uint32_t a,
                                                        uint32_t bBase,
                                                        uint32_t bCount)
    {
        const uint32_t gmOffset = (a * dimB + bBase) * dimC * dimD;
        LocalTensor<T> inputLocal = rank4InQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(bCount * dimC),
            static_cast<uint32_t>(dimD * sizeof(T)), 0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
        rank4InQueue.EnQue(inputLocal);
    }

    __aicore__ inline void CopyRank4Reduce02CoreRangeInput(uint32_t dimB,
                                                           uint32_t dimC,
                                                           uint32_t dimD,
                                                           uint32_t a,
                                                           uint32_t bBase,
                                                           uint32_t bCount)
    {
        const uint32_t gmOffset = (a * dimB + bBase) * dimC * dimD;
        LocalTensor<T> inputLocal = rank4InQueue.AllocTensor<T>();
        DataCopy(inputLocal, inputGm[gmOffset], bCount * dimC * dimD);
        rank4InQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessRank4Reduce02CoreRange()
    {
        const uint32_t dimA = inDims[1];
        const uint32_t dimB = inDims[2];
        const uint32_t dimC = inDims[3];
        const uint32_t dimD = inDims[4];
        const uint32_t rowPitchT = AlignElems32<T>(dimD);
        const uint32_t rowPitchF = AlignElems32<float>(dimD);
        uint32_t bBase = 0U;
        uint32_t bEnd = 0U;
        CalcRange(dimB, bBase, bEnd);
        if (bBase >= bEnd) {
            return;
        }
        const uint32_t bCount = bEnd - bBase;
        LocalTensor<float> accLocal = accBuf.Get<float>();
        Duplicate(accLocal, 0.0F, bCount * rowPitchF);

        CopyRank4Reduce02CoreRangeInput(dimB, dimC, dimD, 0U, bBase, bCount);
        for (uint32_t a = 0U; a < dimA; ++a) {
            LocalTensor<T> ready = rank4InQueue.DeQue<T>();
            if (a + 1U < dimA) {
                CopyRank4Reduce02CoreRangeInput(dimB, dimC, dimD,
                                                a + 1U, bBase, bCount);
            }
            for (uint32_t localB = 0U; localB < bCount; ++localB) {
                AccumulateSquaredRows(accLocal[localB * rowPitchF],
                                      ready[localB * dimC * rowPitchT],
                                      rowPitchT, dimC);
            }
            rank4InQueue.FreeTensor(ready);
        }

        LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
        for (uint32_t localB = 0U; localB < bCount; ++localB) {
            ConvertPlaneOutput(outputLocal[localB * rowPitchT],
                               accLocal[localB * rowPitchF], dimD);
        }
        outQueue.EnQue(outputLocal);
        LocalTensor<T> readyOut = outQueue.DeQue<T>();
        DataCopy(outputGm[bBase * dimD], readyOut, bCount * dimD);
        outQueue.FreeTensor(readyOut);
    }

    __aicore__ inline void ProcessRank4Reduce02BatchB()
    {
        const uint32_t dimA = inDims[1];
        const uint32_t dimB = inDims[2];
        const uint32_t dimC = inDims[3];
        const uint32_t dimD = inDims[4];
        const uint32_t rowPitchT = AlignElems32<T>(dimD);
        const uint32_t rowPitchF = AlignElems32<float>(dimD);
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t bBase = task * kRank4Reduce02BatchB;
            if (bBase >= dimB) {
                continue;
            }
            const uint32_t bCount = MinU32(kRank4Reduce02BatchB, dimB - bBase);
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, bCount * rowPitchF);

            CopyRank4Reduce02BatchBInput(dimB, dimC, dimD, 0U, bBase, bCount);
            for (uint32_t a = 0U; a < dimA; ++a) {
                LocalTensor<T> ready = rank4InQueue.DeQue<T>();
                if (a + 1U < dimA) {
                    CopyRank4Reduce02BatchBInput(dimB, dimC, dimD,
                                                 a + 1U, bBase, bCount);
                }
                for (uint32_t localB = 0U; localB < bCount; ++localB) {
                    AccumulateSquaredRows(accLocal[localB * rowPitchF],
                                          ready[localB * dimC * rowPitchT],
                                          rowPitchT, dimC);
                }
                rank4InQueue.FreeTensor(ready);
            }

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            for (uint32_t localB = 0U; localB < bCount; ++localB) {
                ConvertPlaneOutput(outputLocal[localB * rowPitchT],
                                   accLocal[localB * rowPitchF], dimD);
            }
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopy(outputGm[bBase * dimD], readyOut, bCount * dimD);
            outQueue.FreeTensor(readyOut);
        }
    }

    __aicore__ inline void ProcessRank4Reduce02()
    {
        if (CanProcessRank4Reduce02BatchB()) {
            ProcessRank4Reduce02BatchB();
            return;
        }
        const uint32_t dimA = inDims[1];
        const uint32_t dimB = inDims[2];
        const uint32_t dimC = inDims[3];
        const uint32_t dimD = inDims[4];
        if (dimA == 0U || dimB == 0U || dimC == 0U || dimD == 0U ||
            accTileElems == 0U) {
            return;
        }
        const uint32_t dTiles = (dimD + accTileElems - 1U) / accTileElems;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcRange(taskNum, start, end);
        for (uint32_t task = start; task < end; ++task) {
            const uint32_t b = task / dTiles;
            const uint32_t dTile = task - b * dTiles;
            if (b >= dimB) {
                continue;
            }
            const uint32_t dBase = dTile * accTileElems;
            const uint32_t count = MinU32(accTileElems, dimD - dBase);
            const uint32_t rowPitch = AlignElems32<T>(count);
            const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - count);
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, AlignElems32<float>(count));

            uint32_t a = 0U;
            uint32_t cBase = 0U;
            uint32_t cCount = MinU32(kInnerReduceBatchRows, dimC);
            CopyRank4Reduce02Input(dimB, dimC, dimD, b, dBase, count,
                                   rightPadding, a, cBase, cCount);
            while (true) {
                LocalTensor<T> ready = rank4InQueue.DeQue<T>();
                uint32_t nextA = a;
                uint32_t nextCBase = cBase + cCount;
                if (nextCBase >= dimC) {
                    nextCBase = 0U;
                    ++nextA;
                }
                const bool hasNext = nextA < dimA;
                uint32_t nextCCount = 0U;
                if (hasNext) {
                    nextCCount = MinU32(kInnerReduceBatchRows, dimC - nextCBase);
                    CopyRank4Reduce02Input(dimB, dimC, dimD, b, dBase, count,
                                           rightPadding, nextA, nextCBase, nextCCount);
                }

                AccumulateSquaredRows(accLocal, ready, rowPitch, cCount);
                rank4InQueue.FreeTensor(ready);

                if (!hasNext) {
                    break;
                }
                a = nextA;
                cBase = nextCBase;
                cCount = nextCCount;
            }

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, accLocal, count);
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(count * sizeof(T)),
                                        0U, 0U, 0U};
            DataCopyPad(outputGm[b * dimD + dBase], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
    }
