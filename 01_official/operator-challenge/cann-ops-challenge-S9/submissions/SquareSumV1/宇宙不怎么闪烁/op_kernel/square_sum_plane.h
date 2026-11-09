#pragma once
// Private SquareSumKernel<T> member definitions; included inside the class.

    __aicore__ inline bool UseWidePlaneReduce() const
    {
        // Two full vector stripes; divisibility ensures no short second half.
        return std::is_same_v<T, half> && planeSourceCols != 0U &&
               planeSourceCols % kWidePlaneReduceChunkElems == 0U;
    }

    __aicore__ inline uint32_t PlaneReduceChunkElems() const
    {
        return UseWidePlaneReduce() ? kWidePlaneReduceChunkElems : kPlaneReduceChunkElems;
    }

    __aicore__ inline bool CanProcessDirectPlane() const
    {
        // A single outer group makes each plane output range contiguous even
        // across multidimensional prefix/suffix reductions.
        return std::is_same_v<T, half> && outputSize == innerSize &&
               directPlaneRows != 0U;
    }

    __aicore__ inline void InitDirectPlaneBuffers()
    {
        const uint32_t denseElems = directPlaneRows * planeSourceCols;
        const uint32_t sumBytes = AlignElems32<float>(directPlaneRows) * sizeof(float);
        pipe->InitBuffer(inQueue, 2U, AlignElems32<T>(denseElems) * sizeof(T));
        pipe->InitBuffer(outQueue, 1U, AlignElems32<T>(directPlaneRows) * sizeof(T));
        pipe->InitBuffer(sumBuf, sumBytes);
        pipe->InitBuffer(planeSecondSumBuf, sumBytes);
        pipe->InitBuffer(planeThirdSumBuf, sumBytes);
        pipe->InitBuffer(accBuf, (AlignElems32<float>(denseElems) + 64U) * sizeof(float));
    }

    __aicore__ inline void ReduceDirectPlaneRows(LocalTensor<float> accLocal,
                                                 LocalTensor<float> sumsLocal,
                                                 uint32_t rowCount)
    {
        LocalTensor<float> secondLocal = planeSecondSumBuf.Get<float>();
        LocalTensor<float> thirdLocal = planeThirdSumBuf.Get<float>();
        // Some phases have no third window. Initialize every consumed partial.
        Duplicate(thirdLocal, 0.0F, directPlaneRows);
        PipeBarrier<PIPE_V>();
        const uint32_t groups = square_sum_policy::DirectPlaneGroupRows(planeSourceCols);
        const uint32_t srcRepeatBlocks = groups * planeSourceCols / 8U;
        const bool hasThird = planeSourceCols + 7U > 128U && groups != 1U;
        for (uint32_t phase = 0U; phase < groups && phase < rowCount; ++phase) {
            const uint32_t misalign = phase * planeSourceCols % 8U;
            const uint32_t alignedBase = phase * planeSourceCols - misalign;
            const uint32_t repeats = (rowCount - 1U - phase) / groups + 1U;
            for (uint32_t window = 0U; window < 3U; ++window) {
                uint64_t mask[2] = {square_sum_policy::DirectPlaneWindowMask(
                    planeSourceCols, misalign, window), 0ULL};
                if (mask[0] == 0ULL) continue;
                LocalTensor<float> partial = window == 0U ? sumsLocal :
                    (window == 1U ? secondLocal : thirdLocal);
                for (uint32_t repeat = 0U; repeat < repeats;) {
                    const uint32_t count = MinU32(square_sum_policy::kMaxRepeat, repeats - repeat);
                    // CANN WRS float: source 32B aligned, scalar destination 4B
                    // aligned; dstRepStride counts FP32 elements, src counts blocks.
                    WholeReduceSum<float>(partial[phase + repeat * groups],
                        accLocal[alignedBase + repeat * groups * planeSourceCols + window * 64U],
                        mask, static_cast<int32_t>(count), static_cast<int32_t>(groups),
                        1, static_cast<int32_t>(srcRepeatBlocks));
                    repeat += count;
                }
            }
        }
        PipeBarrier<PIPE_V>();
        Add(sumsLocal, sumsLocal, secondLocal, rowCount);
        PipeBarrier<PIPE_V>();
        if (hasThird) {
            Add(sumsLocal, sumsLocal, thirdLocal, rowCount);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyDirectPlaneInput(uint32_t inputBase,
                                                  uint32_t prefixOffset,
                                                  uint32_t rowCount)
    {
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
        DataCopyExtParams copyParams{
            1U,
            static_cast<uint32_t>(rowCount * planeSourceCols * sizeof(T)),
            0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[inputBase + prefixOffset], copyParams,
                    padParams);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ProcessDirectPlane()
    {
        const uint32_t accElems = directPlaneRows * planeSourceCols;
        LocalTensor<float> accLocal = accBuf.Get<float>();
        // Full tiles are multiples of 16 rows. The aligned 64-float guard also
        // bounds the last masked WRS window; unused tail capacity is cleared below.
        Duplicate(accLocal[accElems], 0.0F, 64U);
        PipeBarrier<PIPE_V>();

        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcPlaneOutputRange(start, end);
        for (uint32_t outputOffset = start; outputOffset < end;) {
            const uint32_t rowCount =
                MinU32(directPlaneRows, end - outputOffset);
            const uint32_t tightElems = rowCount * planeSourceCols;
            const uint32_t inputBase = BaseInputOffset(outputOffset);
            for (uint32_t offset = 0U; offset < accElems;
                 offset += square_sum_policy::kDirectPlaneDenseChunkElems) {
                Duplicate(accLocal[offset], 0.0F, MinU32(
                    square_sum_policy::kDirectPlaneDenseChunkElems, accElems - offset));
            }
            PipeBarrier<PIPE_V>();

            uint32_t prefixCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t prefixReduce = 0U;
            uint32_t prefixOffset = 0U;
            CopyDirectPlaneInput(inputBase, prefixOffset, rowCount);
            while (true) {
                LocalTensor<T> ready = inQueue.DeQue<T>();
                const uint32_t nextPrefixReduce = prefixReduce + 1U;
                uint32_t nextPrefixOffset = prefixOffset;
                const bool hasNext = nextPrefixReduce < reduceSize / planeSourceCols;
                if (hasNext) {
                    AdvanceReduceOffset(PlaneStart(SuffixReduceStart()), prefixCoords, nextPrefixOffset);
                    CopyDirectPlaneInput(inputBase, nextPrefixOffset, rowCount);
                }

                for (uint32_t offset = 0U; offset < tightElems;
                     offset += square_sum_policy::kDirectPlaneDenseChunkElems) {
                    MulAddDst(accLocal[offset], ready[offset], ready[offset], MinU32(
                        square_sum_policy::kDirectPlaneDenseChunkElems, tightElems - offset));
                }
                // Complete the fused acc update and input reads before reuse.
                PipeBarrier<PIPE_V>();
                inQueue.FreeTensor(ready);

                if (!hasNext) {
                    break;
                }
                prefixReduce = nextPrefixReduce;
                prefixOffset = nextPrefixOffset;
            }

            LocalTensor<float> sumsLocal = sumBuf.Get<float>();
            ReduceDirectPlaneRows(accLocal, sumsLocal, rowCount);

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, sumsLocal, rowCount);
            PipeBarrier<PIPE_V>();
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{
                1U, static_cast<uint32_t>(rowCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outputOffset], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
            outputOffset += rowCount;
        }
    }

    __aicore__ inline void InitPlaneBuffers()
    {
        const uint32_t planeChunkElems = PlaneReduceChunkElems();
        const uint32_t inputElems = accTileElems * AlignElems32<T>(planeChunkElems);
        const uint32_t floatElems = accTileElems * AlignElems32<float>(planeChunkElems);
        pipe->InitBuffer(inQueue, 2U, inputElems * sizeof(T));
        pipe->InitBuffer(outQueue, kBufferNum, AlignElems32<T>(accTileElems) * sizeof(T));
        pipe->InitBuffer(calcBuf, floatElems * sizeof(float));
        if (!UseWidePlaneReduce()) {
            pipe->InitBuffer(squareBuf, floatElems * sizeof(float));
        }
        pipe->InitBuffer(sumBuf, AlignElems32<float>(accTileElems) * sizeof(float));
        pipe->InitBuffer(accBuf, AlignElems32<float>(accTileElems) * sizeof(float));
    }

    __aicore__ inline void CopyPlaneInput(uint32_t inputBase, uint32_t prefixOffset,
                                          uint32_t suffixReduceSize, uint32_t reduceBase,
                                          uint32_t reduceCount, uint32_t rowCount)
    {
        const uint32_t rowPitch = AlignElems32<T>(reduceCount);
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitch - reduceCount);
        const uint32_t gmOffset = inputBase + prefixOffset + reduceBase;
        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();

        // Fully aligned plane chunks can use the normal 2-D copy. Retain
        // DataCopyPad for every tail or unaligned row.
        const uint64_t blockBytes = static_cast<uint64_t>(reduceCount) * sizeof(T);
        const uint64_t srcGapBytes =
            static_cast<uint64_t>(suffixReduceSize - reduceCount) * sizeof(T);
        const uint64_t gmOffsetBytes = static_cast<uint64_t>(gmOffset) * sizeof(T);
        if (rightPadding == 0U && (blockBytes & 31U) == 0U &&
            (srcGapBytes & 31U) == 0U && (gmOffsetBytes & 31U) == 0U &&
            blockBytes / 32U <= 65535U && srcGapBytes / 32U <= 65535U) {
            DataCopyParams copyParams{static_cast<uint16_t>(rowCount),
                                      static_cast<uint16_t>(blockBytes / 32U),
                                      static_cast<uint16_t>(srcGapBytes / 32U), 0U};
            DataCopy(inputLocal, inputGm[gmOffset], copyParams);
            inQueue.EnQue(inputLocal);
            return;
        }
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rowCount),
            static_cast<uint32_t>(reduceCount * sizeof(T)),
            static_cast<uint32_t>((suffixReduceSize - reduceCount) * sizeof(T)),
            0U, 0U};
        DataCopyPadExtParams<T> padParams{rightPadding != 0U, 0U, rightPadding,
                                          static_cast<T>(0)};
        DataCopyPad(inputLocal, inputGm[gmOffset], copyParams, padParams);
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline bool UseBalancedPlaneOutputRange() const
    {
        return UseWidePlaneReduce();
    }

    __aicore__ inline void ProcessPlaneReduce()
    {
        const uint32_t suffixReduceStart = SuffixReduceStart();
        const uint32_t planeStart = PlaneStart(suffixReduceStart);
        if (suffixReduceStart == 0U || planeStart == suffixReduceStart || innerSize == 0U) {
            return;
        }

        uint32_t suffixReduceSize = 1U;
        for (uint32_t dim = suffixReduceStart; dim < kDimNum; ++dim) {
            suffixReduceSize *= inDims[dim];
        }
        if (suffixReduceSize == 0U) {
            return;
        }
        const uint32_t prefixReduceSize = reduceSize / suffixReduceSize;
        const uint32_t planeReduceChunkElems = PlaneReduceChunkElems();
        const uint32_t planeTiles = (innerSize + accTileElems - 1U) / accTileElems;
        uint32_t start = 0U;
        uint32_t end = 0U;
        if (UseBalancedPlaneOutputRange()) {
            CalcPlaneOutputRange(start, end);
        } else {
            uint32_t startTask = 0U;
            uint32_t endTask = 0U;
            CalcRange(taskNum, startTask, endTask);
            const uint32_t startOuter = startTask / planeTiles;
            const uint32_t startTile = startTask % planeTiles;
            start = MinU32(outputSize,
                           startOuter * innerSize + startTile * accTileElems);
            if (endTask >= taskNum) {
                end = outputSize;
            } else {
                const uint32_t endOuter = endTask / planeTiles;
                const uint32_t endTile = endTask % planeTiles;
                end = MinU32(outputSize,
                             endOuter * innerSize + endTile * accTileElems);
            }
        }
        for (uint32_t outputOffset = start; outputOffset < end;) {
            const uint32_t planeBase = outputOffset % innerSize;
            const uint32_t rowCount =
                MinU32(accTileElems,
                       MinU32(end - outputOffset, innerSize - planeBase));
            const uint32_t inputBase = BaseInputOffset(outputOffset);
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, 0.0F, AlignElems32<float>(rowCount));

            uint32_t prefixCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
            uint32_t prefixReduce = 0U;
            uint32_t prefixOffset = 0U;
            uint32_t reduceBase = 0U;
            uint32_t reduceCount = MinU32(planeReduceChunkElems, suffixReduceSize);
            CopyPlaneInput(inputBase, prefixOffset, suffixReduceSize, reduceBase,
                           reduceCount, rowCount);

            while (true) {
                LocalTensor<T> ready = inQueue.DeQue<T>();

                uint32_t nextPrefixReduce = prefixReduce;
                uint32_t nextPrefixOffset = prefixOffset;
                uint32_t nextReduceBase = reduceBase + reduceCount;
                if (nextReduceBase >= suffixReduceSize) {
                    nextReduceBase = 0U;
                    ++nextPrefixReduce;
                    if (nextPrefixReduce < prefixReduceSize) {
                        AdvanceReduceOffset(planeStart, prefixCoords, nextPrefixOffset);
                    }
                }
                const bool hasNext = nextPrefixReduce < prefixReduceSize;
                uint32_t nextReduceCount = 0U;
                if (hasNext) {
                    nextReduceCount =
                        MinU32(planeReduceChunkElems, suffixReduceSize - nextReduceBase);
                    CopyPlaneInput(inputBase, nextPrefixOffset, suffixReduceSize, nextReduceBase,
                                   nextReduceCount, rowCount);
                }

                const uint32_t rowPitch = AlignElems32<T>(reduceCount);
                LocalTensor<float> calcLocal = calcBuf.Get<float>();
                LocalTensor<float> sumsLocal = sumBuf.Get<float>();
                const uint32_t batchElems = rowCount * rowPitch;
                if (UseWidePlaneReduce()) {
                    Cast(calcLocal, ready, RoundMode::CAST_NONE, batchElems);
                    Mul(calcLocal, calcLocal, calcLocal, batchElems);
                    const uint8_t rowStrideBlocks = static_cast<uint8_t>(
                        rowPitch * sizeof(float) / 32U);
                    Add<float>(
                        calcLocal, calcLocal, calcLocal[kPlaneReduceChunkElems],
                        static_cast<uint64_t>(kPlaneReduceChunkElems),
                        static_cast<uint8_t>(rowCount),
                        {1U, 1U, 1U, rowStrideBlocks, rowStrideBlocks,
                         rowStrideBlocks});
                    WholeReduceSum<float>(
                        sumsLocal, calcLocal,
                        static_cast<int32_t>(kPlaneReduceChunkElems),
                        static_cast<int32_t>(rowCount), 1, 1,
                        static_cast<int32_t>(rowStrideBlocks));
                } else {
                    LocalTensor<float> squareLocal = squareBuf.Get<float>();
                    SquarePlaneBatch(squareLocal, ready, calcLocal, batchElems);
                    WholeReduceSum<float>(
                        sumsLocal, squareLocal, static_cast<int32_t>(reduceCount),
                        static_cast<int32_t>(rowCount), 1, 1,
                        static_cast<int32_t>(rowPitch * sizeof(float) / 32U));
                }
                Add(accLocal, sumsLocal, accLocal, rowCount);
                inQueue.FreeTensor(ready);

                if (!hasNext) {
                    break;
                }
                prefixReduce = nextPrefixReduce;
                prefixOffset = nextPrefixOffset;
                reduceBase = nextReduceBase;
                reduceCount = nextReduceCount;
            }

            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            ConvertPlaneOutput(outputLocal, accLocal, rowCount);
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(rowCount * sizeof(T)),
                                        0U, 0U, 0U};
            DataCopyPad(outputGm[outputOffset], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
            outputOffset += rowCount;
        }
    }
