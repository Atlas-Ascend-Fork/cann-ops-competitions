#pragma once
// Private SquareSumKernel<T> member definitions; included inside the class.

    __aicore__ inline void SquarePlaneBatch(LocalTensor<float> squareLocal,
                                            LocalTensor<T> inputLocal,
                                            LocalTensor<float> calcLocal,
                                            uint32_t count)
    {
        if constexpr (std::is_same_v<T, float>) {
            Mul(squareLocal, inputLocal, inputLocal, count);
        } else {
            Cast(calcLocal, inputLocal, RoundMode::CAST_NONE, count);
            Mul(squareLocal, calcLocal, calcLocal, count);
        }
    }

    __aicore__ inline void ConvertPlaneOutput(LocalTensor<T> outputLocal,
                                              LocalTensor<float> accLocal,
                                              uint32_t count)
    {
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(outputLocal, accLocal, AlignElems32<float>(count));
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            Cast(outputLocal, accLocal, RoundMode::CAST_RINT, AlignElems32<T>(count));
        } else {
            Cast(outputLocal, accLocal, RoundMode::CAST_RINT, count);
        }
    }

    __aicore__ inline void AccumulateRows(LocalTensor<float> accLocal, LocalTensor<float> rowsLocal,
                                          uint32_t rowPitch, uint32_t rowCount)
    {
        uint32_t activeRows = rowCount;
        while (activeRows > 1U) {
            const uint32_t pairedRows = activeRows / 2U;
            Add(rowsLocal, rowsLocal[pairedRows * rowPitch], rowsLocal, pairedRows * rowPitch);
            if ((activeRows & 1U) != 0U) {
                Add(rowsLocal, rowsLocal[(activeRows - 1U) * rowPitch], rowsLocal, rowPitch);
            }
            activeRows = pairedRows;
        }
        if (rowCount != 0U) {
            Add(accLocal, rowsLocal, accLocal, rowPitch);
        }
    }

    __aicore__ inline void AccumulateSquaredRows(LocalTensor<float> accLocal,
                                                  LocalTensor<T> rowsLocal,
                                                  uint32_t rowPitch, uint32_t rowCount)
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            const uint32_t batchElems = rowCount * rowPitch;
            LocalTensor<float> calcLocal = calcBuf.Get<float>();
            Cast(calcLocal, rowsLocal, RoundMode::CAST_NONE, batchElems);
            Mul(calcLocal, calcLocal, calcLocal, batchElems);
            AccumulateRows(accLocal, calcLocal, rowPitch, rowCount);
        } else {
            constexpr uint32_t kFusedCols = 64U;
            const uint32_t srcRowStrideBlocks = rowPitch * sizeof(T) / 32U;
            for (uint32_t col = 0U; col < rowPitch; col += kFusedCols) {
                const uint32_t colCount = MinU32(kFusedCols, rowPitch - col);
                // Preserve the sequential fused accumulation: a zero destination
                // repeat stride reuses this output slice for every input row.
                MulAddDst(accLocal[col], rowsLocal[col], rowsLocal[col],
                          static_cast<uint64_t>(colCount), static_cast<uint8_t>(rowCount),
                          {1U, 1U, 1U, 0U,
                           static_cast<uint8_t>(srcRowStrideBlocks),
                           static_cast<uint8_t>(srcRowStrideBlocks)});
            }
        }
    }
