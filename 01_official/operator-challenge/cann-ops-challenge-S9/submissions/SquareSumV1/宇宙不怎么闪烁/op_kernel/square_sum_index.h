#pragma once
// Private SquareSumKernel<T> member definitions; included inside the class.

    __aicore__ inline void CalcRange(uint32_t total, uint32_t& start, uint32_t& end)
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t base = total / blockNum;
        const uint32_t remain = total % blockNum;
        start = blockIdx * base + (blockIdx < remain ? blockIdx : remain);
        end = start + base + (blockIdx < remain ? 1U : 0U);
    }

    __aicore__ inline void CalcOutputRange(uint32_t& start, uint32_t& end)
    {
        uint32_t blockStart = 0U;
        uint32_t blockEnd = 0U;
        CalcRange(totalBlocks, blockStart, blockEnd);
        start = blockStart * blockElems;
        end = MinU32(outputSize, blockEnd * blockElems);
    }

    __aicore__ inline uint32_t BaseInputOffset(uint32_t outputOffset)
    {
        uint32_t rem = outputOffset;
        uint32_t inputOffset = 0U;
        for (uint32_t i = 0U; i < kDimNum; ++i) {
            uint32_t coord = rem / outStrides[i];
            rem -= coord * outStrides[i];
            if (reduceMask[i] == 0U) {
                inputOffset += coord * inStrides[i];
            }
        }
        return inputOffset;
    }

    __aicore__ inline uint32_t LastReduceDim()
    {
        for (int32_t dim = static_cast<int32_t>(kDimNum) - 1; dim >= 0; --dim) {
            if (reduceMask[static_cast<uint32_t>(dim)] != 0U) {
                return static_cast<uint32_t>(dim);
            }
        }
        return kDimNum;
    }

    __aicore__ inline uint32_t SuffixReduceStart()
    {
        uint32_t start = kDimNum - 1U;
        while (start > 0U && reduceMask[start - 1U] != 0U) {
            --start;
        }
        return start;
    }

    __aicore__ inline uint32_t PlaneStart(uint32_t suffixReduceStart)
    {
        uint32_t start = suffixReduceStart;
        while (start > 0U && reduceMask[start - 1U] == 0U) {
            --start;
        }
        return start;
    }

    __aicore__ inline void AdvanceReduceOffset(uint32_t lastReduceDim,
                                                    uint32_t (&coords)[kDimNum], uint32_t& inputOffset)
    {
        for (int32_t dim = static_cast<int32_t>(lastReduceDim) - 1; dim >= 0; --dim) {
            const uint32_t dimIdx = static_cast<uint32_t>(dim);
            if (reduceMask[dimIdx] == 0U) {
                continue;
            }
            const uint32_t stride = inStrides[dimIdx];
            ++coords[dimIdx];
            inputOffset += stride;
            if (coords[dimIdx] < inDims[dimIdx]) {
                return;
            }
            coords[dimIdx] = 0U;
            inputOffset -= inDims[dimIdx] * stride;
        }
    }

    __aicore__ inline void CalcPlaneOutputRange(uint32_t& start,
                                                uint32_t& end) const
    {
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t unitElems = 32U / sizeof(T);
        const uint32_t unitCount =
            (outputSize + unitElems - 1U) / unitElems;
        const uint32_t baseUnits = unitCount / blockNum;
        const uint32_t extraUnits = unitCount % blockNum;
        const uint32_t startUnit =
            blockIdx * baseUnits + MinU32(blockIdx, extraUnits);
        const uint32_t myUnits =
            baseUnits + (blockIdx < extraUnits ? 1U : 0U);
        start = MinU32(outputSize, startUnit * unitElems);
        end = MinU32(outputSize, (startUnit + myUnits) * unitElems);
    }
