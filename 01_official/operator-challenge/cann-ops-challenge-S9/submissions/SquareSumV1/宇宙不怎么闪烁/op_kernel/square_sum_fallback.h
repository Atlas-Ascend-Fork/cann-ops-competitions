#pragma once
// Private SquareSumKernel<T> member definitions; included inside the class.

    __aicore__ inline void ProcessGenericFallback()
    {
        if (LastReduceDim() == kDimNum - 1U) ProcessScalarReduction(false);
    }

    __aicore__ inline void ProcessScalarReduction(bool suffix)
    {
        const uint32_t lastReduceDim = suffix ? kDimNum : LastReduceDim();
        const uint32_t contiguousReduceSize = suffix ? reduceSize : inDims[lastReduceDim];
        const uint32_t outerReduceSize = contiguousReduceSize == 0U
            ? 0U : reduceSize / contiguousReduceSize;
        uint32_t start = 0U;
        uint32_t end = 0U;
        CalcOutputRange(start, end);
        for (uint32_t outBase = start; outBase < end; outBase += kSuffixOutTileElems) {
            const uint32_t outCount = MinU32(kSuffixOutTileElems, end - outBase);
            LocalTensor<T> outputLocal;
            LocalTensor<float> scalarFloat;
            if constexpr (std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t>) {
                outputLocal = outQueue.AllocTensor<T>();
            }
            if constexpr (std::is_same_v<T, float>) {
                scalarFloat = outputLocal;
            } else {
                scalarFloat = valueFloatBuf.Get<float>();
            }
            for (uint32_t localOut = 0U; localOut < outCount; ++localOut) {
                const uint32_t outIdx = outBase + localOut;
                const uint32_t inputBase = suffix ? outIdx * reduceSize : BaseInputOffset(outIdx);
                uint32_t outerReduceCoords[kDimNum] = {0U, 0U, 0U, 0U, 0U};
                uint32_t outerReduceOffset = 0U;
                float acc = 0.0F;
                for (uint32_t outerReduce = 0U; outerReduce < outerReduceSize; ++outerReduce) {
                    for (uint32_t r = 0U; r < contiguousReduceSize; r += tileElems) {
                        const uint32_t count = MinU32(tileElems, contiguousReduceSize - r);
                        LocalTensor<T> inputLocal = inQueue.AllocTensor<T>();
                        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(count * sizeof(T)),
                                                     0U, 0U, 0U};
                        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
                        DataCopyPad(inputLocal, inputGm[inputBase + outerReduceOffset + r],
                                    copyParams, padParams);
                        inQueue.EnQue(inputLocal);
                        LocalTensor<T> ready = inQueue.DeQue<T>();
                        LocalTensor<float> workLocal = workBuf.Get<float>();
                        LocalTensor<float> sumLocal = sumBuf.Get<float>();
                        LocalTensor<float> scratch;
                        if constexpr (std::is_same_v<T, float>) {
                            scratch = ready;
                            Mul(workLocal, ready, ready, count);
                        } else {
                            scratch = calcBuf.Get<float>();
                            Cast(scratch, ready, RoundMode::CAST_NONE, AlignElems32<T>(count));
                            Mul(workLocal, scratch, scratch, count);
                        }
                        ReduceSum(sumLocal, workLocal, scratch, count);
                        acc += sumLocal.GetValue(0U);
                        inQueue.FreeTensor(ready);
                    }
                    if (!suffix) {
                        AdvanceReduceOffset(lastReduceDim, outerReduceCoords, outerReduceOffset);
                    }
                }
                scalarFloat.SetValue(localOut, acc);
            }
            if constexpr (!std::is_same_v<T, float>) {
                if constexpr (std::is_same_v<T, half>) outputLocal = outQueue.AllocTensor<T>();
                constexpr RoundMode rounding = std::is_same_v<T, bfloat16_t>
                    ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
                Cast(outputLocal, scalarFloat, rounding, AlignElems32<T>(outCount));
            }
            outQueue.EnQue(outputLocal);
            LocalTensor<T> readyOut = outQueue.DeQue<T>();
            DataCopyExtParams outParams{1U, static_cast<uint32_t>(outCount * sizeof(T)), 0U, 0U, 0U};
            DataCopyPad(outputGm[outBase], readyOut, outParams);
            outQueue.FreeTensor(readyOut);
        }
    }
