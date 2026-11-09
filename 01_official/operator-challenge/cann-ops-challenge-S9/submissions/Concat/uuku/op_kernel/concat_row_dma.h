// Included inside ConcatKernel; not a standalone compilation unit.
// Window/row execution and shared strided DMA helpers.
// Units: row/axis/segment arguments are element counts; copy descriptors use bytes.
// Generic admission must complete before allocation or writes; no mid-write fallback.

    template <typename T>
    __aicore__ inline bool TryProcessWindowOwnedRows(uint32_t localElems,
                                                          GlobalTensor<T>& yGm)
    {
        constexpr uint32_t kWindowBytes = 512U;
        if (sizeof(T) != 4U || !IsManyInputLastAxisB32() || localElems == 0U ||
            outputAxisSize == 0U || outerSize == 0U) {
            return false;
        }

        constexpr uint32_t kWindowElems = kWindowBytes / sizeof(T);
        const uint32_t windowCount = CeilDivU32(outputAxisSize, kWindowElems);
        const uint32_t blockNum = GetBlockNum();
        const uint32_t blockIdx = GetBlockIdx();
        if (windowCount == 0U || blockNum == 0U || blockIdx >= blockNum ||
            windowCount < blockNum) {
            return false;
        }

        // Each core owns complete 512B output windows. This prevents adjacent
        // input segments assigned to different cores from serializing on one
        // GM 512B address region.
        for (uint32_t windowIdx = blockIdx; windowIdx < windowCount; windowIdx += blockNum) {
            const uint32_t windowStart = windowIdx * kWindowElems;
            const uint32_t windowEnd = MinU32(windowStart + kWindowElems, outputAxisSize);

            for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
                const uint32_t axisLen = AxisLen(inputIdx);
                if (axisLen == 0U) {
                    continue;
                }
                const uint32_t axisStart = AxisOffset(inputIdx);
                const uint32_t axisEnd = axisStart + axisLen;
                const uint32_t partStart = MaxU32(axisStart, windowStart);
                const uint32_t partEnd = MinU32(axisEnd, windowEnd);
                if (partStart >= partEnd) {
                    continue;
                }

                const uint32_t partElems = partEnd - partStart;
                const uint32_t partElemsAlign = AlignElems32<T>(partElems);
                if (partElemsAlign == 0U || partElemsAlign > localElems) {
                    return false;
                }
                uint32_t rowBatch = localElems / partElemsAlign;
                if (rowBatch == 0U) {
                    return false;
                }
                if (rowBatch > 4095U) {
                    rowBatch = 4095U;
                }

                GlobalTensor<T> xGm;
                xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(InputPtr(inputIdx)),
                                    static_cast<uint64_t>(outerSize) * axisLen);
                xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                const uint32_t inputAxisOffset = partStart - axisStart;
                uint32_t row = 0U;
                while (row < outerSize) {
                    const uint32_t rowCount = MinU32(rowBatch, outerSize - row);
                    CopyStridedRows<T>(xGm, yGm, row, rowCount, partStart,
                                            axisLen, inputAxisOffset, partElems,
                                            outputAxisSize);
                    row += rowCount;
                }
            }
        }
        return true;
    }

    template <typename T>
    __aicore__ inline void ProcessRows(uint32_t localElems, GlobalTensor<T>& yGm)
    {
        const uint32_t blockNum = GetBlockNum();
        // CanProcessRows preflights every input before any UB allocation/write.
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t rowsPerCore = outerSize / blockNum + (outerSize % blockNum != 0U ? 1U : 0U);
        const uint64_t begin64 = static_cast<uint64_t>(blockIdx) * rowsPerCore;
        if (begin64 >= outerSize) return;
        const uint32_t rowBegin = static_cast<uint32_t>(begin64);
        const uint32_t rowEnd = rowBegin + MinU32(rowsPerCore, outerSize - rowBegin);
        if (rowBegin >= rowEnd) {
            return;
        }

        const uint32_t outputRowElems = outputAxisSize * innerSize;
        uint32_t axisOffset = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
            const uint32_t axisLen = AxisLen(inputIdx);
            if (axisLen == 0U) {
                continue;
            }
            const uint32_t segmentElems = axisLen * innerSize;
            const uint32_t segmentElemsAlign = AlignElems32<T>(segmentElems);
            uint32_t rowBatch = localElems / segmentElemsAlign;
            if (rowBatch > 4095U) {
                rowBatch = 4095U;
            }

            GlobalTensor<T> xGm;
            const uint64_t inputElems = static_cast<uint64_t>(outerSize) * segmentElems;
            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(InputPtr(inputIdx)),
                                static_cast<uint64_t>(inputElems));
            // Keep short last-axis row copies on the default cache path.
            // Multi-inner row copies retain the validated input bypass.
            if (innerSize > 1U) {
                xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }

            uint32_t row = rowBegin;
            while (row < rowEnd) {
                const uint32_t currRows = MinU32(rowBatch, rowEnd - row);
                CopyRowsSegment<T>(xGm, yGm, row, currRows, axisOffset,
                                   segmentElems, outputRowElems);
                row += currRows;
            }
            axisOffset += axisLen;
        }
        return;
    }

    template <typename T>
    __aicore__ inline void CopyStridedRows(GlobalTensor<T>& xGm, GlobalTensor<T>& yGm,
                                                uint32_t rowStart, uint32_t rowCount,
                                                uint64_t outputAxisOffset,
                                                uint32_t inputRowElems,
                                                uint32_t inputAxisOffset,
                                                uint32_t segmentElems,
                                                uint32_t outputRowElems)
    {
        const uint32_t copyBytes = segmentElems * sizeof(T);
        const uint32_t inputGapBytes = (inputRowElems - segmentElems) * sizeof(T);
        const uint32_t outputGapBytes = (outputRowElems - segmentElems) * sizeof(T);
        DataCopyExtParams copyIn{static_cast<uint16_t>(rowCount), copyBytes,
                                 inputGapBytes, 0U, 0U};
        DataCopyExtParams copyOut{static_cast<uint16_t>(rowCount), copyBytes,
                                  0U, outputGapBytes, 0U};
        DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};

        const uint64_t inputOffset = static_cast<uint64_t>(rowStart) * inputRowElems +
                                     inputAxisOffset;
        const uint64_t outputOffset = static_cast<uint64_t>(rowStart) * outputRowElems +
                                      outputAxisOffset;
        LocalTensor<T> xLocal = copyQueue.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[inputOffset], copyIn, padParams);
        copyQueue.EnQue(xLocal);

        LocalTensor<T> yReady = copyQueue.DeQue<T>();
        DataCopyPad(yGm[outputOffset], yReady, copyOut);
        copyQueue.FreeTensor(yReady);
    }

    template <typename T>
    __aicore__ inline void CopyRowsSegment(GlobalTensor<T>& xGm, GlobalTensor<T>& yGm,
                                           uint32_t rowStart, uint32_t rowCount,
                                           uint32_t axisOffset, uint32_t segmentElems,
                                           uint32_t outputRowElems)
    {
        CopyStridedRows<T>(xGm, yGm, rowStart, rowCount,
                          static_cast<uint64_t>(axisOffset) * innerSize,
                          segmentElems, 0U, segmentElems, outputRowElems);
    }

    template <typename T>
    __aicore__ inline void CopyMteGroupedRows(uint32_t groupStart, uint32_t groupEnd,
                                              GlobalTensor<T>& yGm, uint32_t rowStart,
                                              uint32_t rowCount, uint32_t groupAxisOffset,
                                              uint32_t groupElems, uint32_t groupElemsAlign)
    {
        QueueMteGroupedRows<T>(groupStart, groupEnd, rowStart, rowCount, groupElemsAlign,
                               true);
        StoreMteGroupedRows<T>(yGm, rowStart, rowCount, groupAxisOffset, groupElems,
                               groupElemsAlign);
    }

    template <typename T>
    __aicore__ inline void QueueMteGroupedRows(uint32_t groupStart, uint32_t groupEnd,
                                               uint32_t rowStart, uint32_t rowCount,
                                               uint32_t groupElemsAlign,
                                               bool preserveInterCopyBarrier)
    {
        LocalTensor<T> yLocal = copyQueue.AllocTensor<T>();
        uint32_t localAxisOffset = 0U;
        for (uint32_t inputIdx = groupStart; inputIdx < groupEnd; ++inputIdx) {
            const uint32_t axisLen = AxisLen(inputIdx);
            if (axisLen == 0U) {
                continue;
            }

            GlobalTensor<T> xGm;
            const uint32_t inputElems = outerSize * axisLen;
            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(InputPtr(inputIdx)),
                                static_cast<uint64_t>(inputElems));
            xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            const uint32_t copyBytes = axisLen * sizeof(T);
            const uint32_t localGapBlocks = ((groupElemsAlign - axisLen) * sizeof(T)) / 32U;
            DataCopyExtParams copyIn{
                static_cast<uint16_t>(rowCount), copyBytes, 0U, localGapBlocks, 0U};
            DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};
            const uint64_t inputOffset = static_cast<uint64_t>(rowStart) * axisLen;

            DataCopyPad(yLocal[localAxisOffset], xGm[inputOffset], copyIn, padParams);
            if (preserveInterCopyBarrier) {
                PipeBarrier<PIPE_ALL>();
            }
            localAxisOffset += axisLen;
        }
        copyQueue.EnQue(yLocal);
    }

    template <typename T>
    __aicore__ inline void StoreMteGroupedRows(GlobalTensor<T>& yGm, uint32_t rowStart,
                                               uint32_t rowCount, uint32_t groupAxisOffset,
                                               uint32_t groupElems,
                                               uint32_t groupElemsAlign)
    {
        const uint32_t outputRowElems = outputAxisSize;
        const uint32_t copyBytes = groupElems * sizeof(T);
        const uint32_t outputGapBytes = (outputRowElems - groupElems) * sizeof(T);
        const uint32_t srcGapBlocks = ((groupElemsAlign - groupElems) * sizeof(T)) / 32U;
        const uint64_t outputOffset =
            static_cast<uint64_t>(rowStart) * outputRowElems + groupAxisOffset;
        DataCopyExtParams copyOut{
            static_cast<uint16_t>(rowCount), copyBytes, srcGapBlocks, outputGapBytes, 0U};
        LocalTensor<T> yReady = copyQueue.DeQue<T>();
        DataCopyPad(yGm[outputOffset], yReady, copyOut);
        copyQueue.FreeTensor(yReady);
    }
