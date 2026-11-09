// Included inside ConcatKernel; not a standalone compilation unit.
// Continuous small-width B32 rows. Each core owns a quotient/remainder row shard.
// Gather and direct-copy inputs share the existing host schedule and queue order.

    __aicore__ inline uint32_t ContinuousGatherPitch(uint32_t width) const
    {
        uint32_t pitch = 8U;
        while (pitch < width) {
            pitch <<= 1U;
        }
        return pitch;
    }

    __aicore__ inline uint32_t ContinuousGatherPitchShift(uint32_t pitch) const
    {
        uint32_t shift = 3U;
        while ((1U << shift) < pitch) {
            ++shift;
        }
        return shift;
    }

    __aicore__ inline void BuildContinuousGatherIndex(
        uint32_t width, uint32_t pitch, uint32_t rowCount,
        LocalTensor<uint32_t>& indexLocal)
    {
        const uint32_t count = rowCount * pitch;
        LocalTensor<int32_t> indexSigned =
            indexLocal.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> baseLocal = continuousBaseIndexBuf.Get<int32_t>();

        // pitch is a power of two. baseLocal contains byte offsets for the
        // padded row grid; shifting derives the row number without scalar
        // division, then rowDelta maps the grid back to the contiguous input.
        ShiftRight(indexSigned, baseLocal,
                   static_cast<int32_t>(ContinuousGatherPitchShift(pitch) + 2U),
                   count);
        const int32_t rowDelta =
            (static_cast<int32_t>(width) - static_cast<int32_t>(pitch)) *
            static_cast<int32_t>(sizeof(uint32_t));
        Muls(indexSigned, indexSigned, rowDelta, count);
        Add(indexSigned, indexSigned, baseLocal, count);
    }

    template <typename T>
    __aicore__ inline bool TryProcessContinuousSmallWidthRows(
        GlobalTensor<T>& yGm)
    {
        constexpr uint32_t kMaxPitch = 128U;
        constexpr uint32_t kGatherWidthMax = 82U;
        const uint32_t blockNum = GetBlockNum();
        if (sizeof(T) != 4U || routeKey != kRouteContinuousB32 ||
            blockNum != 40U) {
            return false;
        }

        const uint32_t baseRowsPerCore = outerSize / blockNum;
        const uint32_t extraRows = outerSize - baseRowsPerCore * blockNum;
        const uint32_t maxRowsPerCore =
            baseRowsPerCore + (extraRows != 0U ? 1U : 0U);
        const uint32_t blockIdx = GetBlockIdx();
        // Assign the quotient to every core and one remainder row to each
        // leading core.  This keeps intervals contiguous while eliminating a
        // short final-core tail from the transfer-bound row route.
        const uint32_t rowBegin =
            blockIdx * baseRowsPerCore + MinU32(blockIdx, extraRows);
        const uint32_t rowCount =
            baseRowsPerCore + (blockIdx < extraRows ? 1U : 0U);
        const uint32_t rowEnd = rowBegin + rowCount;
        if (rowBegin >= rowEnd) {
            return true;
        }

        const uint32_t localElems = maxRowsPerCore * kMaxPitch;
        pipe->InitBuffer(inQueue, 1U, localElems * sizeof(T));
        pipe->InitBuffer(outQueue, 2U, localElems * sizeof(T));
        pipe->InitBuffer(maskBuf, localElems * sizeof(uint32_t));
        pipe->InitBuffer(continuousBaseIndexBuf,
                         localElems * sizeof(int32_t));
        pipe->InitBuffer(copyQueue, kBufferNum, localElems * sizeof(T));
        InitMetaCache();

        LocalTensor<uint32_t> indexLocal = maskBuf.Get<uint32_t>();
        LocalTensor<int32_t> baseLocal = continuousBaseIndexBuf.Get<int32_t>();
        CreateVecIndex(baseLocal, 0, rowCount * kMaxPitch);
        Muls(baseLocal, baseLocal, static_cast<int32_t>(sizeof(uint32_t)),
             rowCount * kMaxPitch);

        for (uint32_t scheduleIdx = 0U;
             scheduleIdx < routeScheduleCount; ++scheduleIdx) {
            // Host tiling preserves the same generic nine-phase interleave,
            // while the kernel reads every selected input exactly once.
            const uint32_t inputIdx = ScheduleInput(scheduleIdx);
            if (inputIdx >= inputCount) {
                return false;
            }
            const uint32_t width = AxisLen(inputIdx);
            if (width == 0U || width > kMaxPitch) {
                return false;
            }
            GlobalTensor<T> xGm;
            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(InputPtr(inputIdx)),
                                static_cast<uint64_t>(outerSize) * width);
            xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            const bool useGather =
                (width & 7U) != 0U && width <= kGatherWidthMax;
            if (!useGather) {
                if ((width & 7U) == 0U) {
                    LocalTensor<T> staged = copyQueue.AllocTensor<T>();
                    DataCopyExtParams copyIn{
                        1U, static_cast<uint32_t>(rowCount * width * sizeof(T)),
                        0U, 0U, 0U};
                    DataCopyPadExtParams<T> padParams{
                        false, 0U, 0U, static_cast<T>(0)};
                    DataCopyPad(staged,
                                xGm[static_cast<uint64_t>(rowBegin) * width],
                                copyIn, padParams);
                    copyQueue.EnQue(staged);

                    LocalTensor<T> ready = copyQueue.DeQue<T>();
                    const uint32_t outputGapBytes =
                        (outputAxisSize - width) * sizeof(T);
                    DataCopyExtParams copyOut{
                        static_cast<uint16_t>(rowCount),
                        static_cast<uint32_t>(width * sizeof(T)),
                        0U, outputGapBytes, 0U};
                    const uint64_t outputOffset =
                        static_cast<uint64_t>(rowBegin) * outputAxisSize +
                        AxisOffset(inputIdx);
                    DataCopyPad(yGm[outputOffset], ready, copyOut);
                    copyQueue.FreeTensor(ready);
                    continue;
                }

                CopyRowsSegment<T>(xGm, yGm, rowBegin, rowCount,
                                   AxisOffset(inputIdx), width, outputAxisSize);
                continue;
            }

            const uint32_t pitch = ContinuousGatherPitch(width);
            LocalTensor<T> staged = inQueue.AllocTensor<T>();
            DataCopyExtParams copyIn{
                1U, static_cast<uint32_t>(rowCount * width * sizeof(T)),
                0U, 0U, 0U};
            DataCopyPadExtParams<T> padParams{
                false, 0U, 0U, static_cast<T>(0)};
            DataCopyPad(staged,
                        xGm[static_cast<uint64_t>(rowBegin) * width],
                        copyIn, padParams);
            inQueue.EnQue(staged);

            BuildContinuousGatherIndex(width, pitch, rowCount, indexLocal);
            LocalTensor<T> ready = inQueue.DeQue<T>();
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            Gather(outputLocal, ready, indexLocal, 0U, rowCount * pitch);
            outQueue.EnQue(outputLocal);
            inQueue.FreeTensor(ready);

            LocalTensor<T> outputReady = outQueue.DeQue<T>();
            const uint32_t alignedWidth = AlignElems32<T>(width);
            const uint32_t srcGapBlocks =
                ((pitch - alignedWidth) * sizeof(T)) / 32U;
            const uint32_t outputGapBytes =
                (outputAxisSize - width) * sizeof(T);
            DataCopyExtParams copyOut{
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(width * sizeof(T)),
                srcGapBlocks, outputGapBytes, 0U};
            const uint64_t outputOffset =
                static_cast<uint64_t>(rowBegin) * outputAxisSize +
                AxisOffset(inputIdx);
            DataCopyPad(yGm[outputOffset], outputReady, copyOut);
            outQueue.FreeTensor(outputReady);
        }
        return true;
    }
