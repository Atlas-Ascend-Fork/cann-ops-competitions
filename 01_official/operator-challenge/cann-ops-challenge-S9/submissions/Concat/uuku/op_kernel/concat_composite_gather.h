// Included inside ConcatKernel; not a standalone compilation unit.
// Legacy composite B32 groups: build metadata/index, prefetch, gather, then store.
// Kept as a complete route; do not delete solely because current host selection skips it.

    __aicore__ inline CompositeGatherGroup NextCompositeGatherGroup(
        uint32_t& cursor, uint32_t& axisCursor, uint32_t rowCount) const
    {
        while (cursor < inputCount && AxisLen(cursor) == 0U) {
            ++cursor;
        }
        if (cursor >= inputCount) {
            return CompositeGatherGroup{0U, 0U, 0U, 0U, 0U, 0U, false};
        }

        const uint32_t groupStart = cursor;
        const uint32_t groupAxisOffset = axisCursor;
        uint32_t groupElems = 0U;
        uint32_t sourceElems = 0U;
        uint32_t nonZeroMembers = 0U;
        while (cursor < inputCount) {
            const uint32_t width = AxisLen(cursor);
            if (width == 0U) {
                ++cursor;
                continue;
            }
            if (width > kContinuousMaxWidth ||
                (groupElems != 0U &&
                 groupElems + width > kContinuousMaxWidth)) {
                break;
            }
            groupElems += width;
            axisCursor += width;
            sourceElems += AlignElems32<uint32_t>(rowCount * width);
            ++nonZeroMembers;
            ++cursor;
            if (groupElems == kContinuousMaxWidth) {
                break;
            }
        }
        return CompositeGatherGroup{
            groupStart, cursor, groupAxisOffset, groupElems, sourceElems,
            nonZeroMembers, groupElems != 0U};
    }

    __aicore__ inline void ApplyCompositeGatherColumns(
        LocalTensor<int32_t>& indexLocal, LocalTensor<int32_t>& rowLocal,
        uint32_t columnBegin, uint32_t columnCount, int32_t rowDelta,
        int32_t baseDelta, uint32_t rowCount)
    {
        const UnaryRepeatParams repeatParams{1U, 1U, 16U, 16U};
        uint32_t column = columnBegin;
        uint32_t remaining = columnCount;
        while (remaining != 0U) {
            const uint32_t half = column >> 6U;
            const uint32_t withinHalf = column & 63U;
            const uint32_t chunk = MinU32(remaining, 64U - withinHalf);
            const uint64_t chunkBits =
                chunk == 64U ? ~0ULL : ((1ULL << chunk) - 1ULL);
            uint64_t mask[2] = {chunkBits << withinHalf, 0ULL};
            const uint32_t tensorOffset = half * 64U;
            Muls(rowLocal[tensorOffset], rowLocal[tensorOffset], rowDelta,
                 mask, static_cast<uint8_t>(rowCount), repeatParams);
            Adds(indexLocal[tensorOffset], indexLocal[tensorOffset], baseDelta,
                 mask, static_cast<uint8_t>(rowCount), repeatParams);
            column += chunk;
            remaining -= chunk;
        }
    }

    __aicore__ inline void BuildCompositeGatherIndex(
        const CompositeGatherGroup& group, uint32_t rowCount,
        LocalTensor<uint32_t>& indexLocal)
    {
        const uint32_t count = rowCount * kContinuousMaxWidth;
        LocalTensor<int32_t> indexSigned =
            indexLocal.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> baseLocal =
            continuousBaseIndexBuf.Get<int32_t>();
        LocalTensor<int32_t> rowLocal = periodicIndexWorkBuf.Get<int32_t>();

        // baseLocal is a byte-indexed hypothetical 128-pitch row grid.  Each
        // member mask replaces that stride with its real contiguous width and
        // adds the member's 32B-aligned source-slab base.
        ShiftRight(rowLocal, baseLocal, 9, count);
        Adds(indexSigned, baseLocal, 0, count);
        uint32_t groupColumn = 0U;
        uint32_t sourceBase = 0U;
        for (uint32_t inputIdx = group.groupStart;
             inputIdx < group.groupEnd; ++inputIdx) {
            const uint32_t width = AxisLen(inputIdx);
            if (width == 0U) {
                continue;
            }
            const int32_t rowDelta =
                (static_cast<int32_t>(width) -
                 static_cast<int32_t>(kContinuousMaxWidth)) *
                static_cast<int32_t>(sizeof(uint32_t));
            const int32_t baseDelta =
                (static_cast<int32_t>(sourceBase) -
                 static_cast<int32_t>(groupColumn)) *
                static_cast<int32_t>(sizeof(uint32_t));
            ApplyCompositeGatherColumns(
                indexSigned, rowLocal, groupColumn, width, rowDelta,
                baseDelta, rowCount);
            sourceBase += AlignElems32<uint32_t>(rowCount * width);
            groupColumn += width;
        }

        // Gather writes a full 128-pitch row.  Redirect the unstored tail to
        // the beginning of the source slot, keeping speculative reads bounded.
        if (groupColumn < kContinuousMaxWidth) {
            const int32_t tailRowDelta =
                -static_cast<int32_t>(kContinuousMaxWidth *
                                      sizeof(uint32_t));
            const int32_t tailBaseDelta =
                -static_cast<int32_t>(groupColumn * sizeof(uint32_t));
            ApplyCompositeGatherColumns(
                indexSigned, rowLocal, groupColumn,
                kContinuousMaxWidth - groupColumn, tailRowDelta,
                tailBaseDelta, rowCount);
        }
        Add(indexSigned, indexSigned, rowLocal, count);
    }

    template <typename T>
    __aicore__ inline void LoadCompositeGatherGroup(
        const CompositeGatherGroup& group, uint32_t rowBegin,
        uint32_t rowCount)
    {
        LocalTensor<T> staged = inQueue.AllocTensor<T>();
        uint32_t sourceBase = 0U;
        for (uint32_t inputIdx = group.groupStart;
             inputIdx < group.groupEnd; ++inputIdx) {
            const uint32_t width = AxisLen(inputIdx);
            if (width == 0U) {
                continue;
            }
            GlobalTensor<T> xGm;
            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(InputPtr(inputIdx)),
                                static_cast<uint64_t>(outerSize) * width);
            xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            DataCopyExtParams copyIn{
                1U, static_cast<uint32_t>(rowCount * width * sizeof(T)),
                0U, 0U, 0U};
            DataCopyPadExtParams<T> padParams{
                false, 0U, 0U, static_cast<T>(0)};
            DataCopyPad(staged[sourceBase],
                        xGm[static_cast<uint64_t>(rowBegin) * width],
                        copyIn, padParams);
            sourceBase += AlignElems32<T>(rowCount * width);
        }
        inQueue.EnQue(staged);
    }

    template <typename T>
    __aicore__ inline void StoreCompositeGatherGroup(
        const CompositeGatherGroup& group, uint32_t pitch,
        uint32_t rowBegin, uint32_t rowCount, GlobalTensor<T>& yGm)
    {
        LocalTensor<T> outputReady = outQueue.DeQue<T>();
        const uint32_t alignedWidth = AlignElems32<T>(group.groupElems);
        const uint32_t srcGapBlocks =
            ((pitch - alignedWidth) * sizeof(T)) / 32U;
        const uint32_t outputGapBytes =
            (outputAxisSize - group.groupElems) * sizeof(T);
        DataCopyExtParams copyOut{
            static_cast<uint16_t>(rowCount),
            static_cast<uint32_t>(group.groupElems * sizeof(T)),
            srcGapBlocks, outputGapBytes, 0U};
        const uint64_t outputOffset =
            static_cast<uint64_t>(rowBegin) * outputAxisSize +
            group.groupAxisOffset;
        DataCopyPad(yGm[outputOffset], outputReady, copyOut);
        outQueue.FreeTensor(outputReady);
    }

    template <typename T>
    __aicore__ inline void CopyCompositeGatherDirect(
        const CompositeGatherGroup& group, uint32_t rowBegin,
        uint32_t rowCount, GlobalTensor<T>& yGm)
    {
        GlobalTensor<T> xGm;
        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(InputPtr(group.groupStart)),
            static_cast<uint64_t>(outerSize) * group.groupElems);
        xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        LocalTensor<T> staged = copyQueue.AllocTensor<T>();
        DataCopyExtParams copyIn{
            1U,
            static_cast<uint32_t>(rowCount * group.groupElems * sizeof(T)),
            0U, 0U, 0U};
        DataCopyPadExtParams<T> padParams{
            false, 0U, 0U, static_cast<T>(0)};
        DataCopyPad(staged,
                    xGm[static_cast<uint64_t>(rowBegin) * group.groupElems],
                    copyIn, padParams);
        copyQueue.EnQue(staged);

        LocalTensor<T> ready = copyQueue.DeQue<T>();
        const uint32_t outputGapBytes =
            (outputAxisSize - group.groupElems) * sizeof(T);
        DataCopyExtParams copyOut{
            static_cast<uint16_t>(rowCount),
            static_cast<uint32_t>(group.groupElems * sizeof(T)),
            0U, outputGapBytes, 0U};
        const uint64_t outputOffset =
            static_cast<uint64_t>(rowBegin) * outputAxisSize +
            group.groupAxisOffset;
        DataCopyPad(yGm[outputOffset], ready, copyOut);
        copyQueue.FreeTensor(ready);
    }

    template <typename T>
    __aicore__ inline bool TryProcessCompositeGatherRows(
        GlobalTensor<T>& yGm)
    {
        const uint32_t blockNum = GetBlockNum();
        if (sizeof(T) != 4U || routeKey != kRouteCompositeGatherB32 ||
            blockNum != kAivCoreNum910B || phaseBoundaryCount == 0U ||
            phaseUnitCount == 0U || phaseOwnerWordCount == 0U ||
            phaseOwnerWordCount > 255U) {
            return false;
        }

        const uint32_t baseRowsPerCore = outerSize / blockNum;
        const uint32_t extraRows = outerSize - baseRowsPerCore * blockNum;
        const uint32_t maxRowsPerCore =
            baseRowsPerCore + (extraRows != 0U ? 1U : 0U);
        if (maxRowsPerCore != phaseOwnerWordCount) {
            return false;
        }

        // Reconstruct the host's complete class model before reserving UB.
        uint32_t modelCursor = 0U;
        uint32_t modelAxisCursor = 0U;
        uint32_t modelGroupCount = 0U;
        uint32_t modelMaxSourceElems = 0U;
        uint32_t modelRequiredGatherSourceElems = 0U;
        while (true) {
            const CompositeGatherGroup group =
                NextCompositeGatherGroup(
                    modelCursor, modelAxisCursor, maxRowsPerCore);
            if (!group.valid) {
                break;
            }
            modelMaxSourceElems =
                MaxU32(modelMaxSourceElems, group.sourceElems);
            if (group.nonZeroMembers > 1U) {
                modelRequiredGatherSourceElems = MaxU32(
                    modelRequiredGatherSourceElems,
                    kContinuousMaxWidth - group.groupElems);
            } else if ((group.groupElems & 7U) != 0U) {
                const uint32_t pitch =
                    ContinuousGatherPitch(group.groupElems);
                modelRequiredGatherSourceElems = MaxU32(
                    modelRequiredGatherSourceElems,
                    maxRowsPerCore * group.groupElems + pitch -
                        group.groupElems);
            }
            ++modelGroupCount;
        }
        if (modelCursor != inputCount || modelAxisCursor != outputAxisSize ||
            modelGroupCount != phaseBoundaryCount ||
            modelMaxSourceElems != phaseUnitCount ||
            modelMaxSourceElems < modelRequiredGatherSourceElems) {
            return false;
        }

        const uint32_t maxLocalElems =
            maxRowsPerCore * kContinuousMaxWidth;
        const uint32_t modelMetaStride = AlignElems32<uint32_t>(inputCount);
        const uint64_t metaBytes =
            static_cast<uint64_t>(4U) * modelMetaStride * sizeof(uint32_t);
        const uint64_t requiredBytes =
            static_cast<uint64_t>(modelMaxSourceElems) * sizeof(T) +
            6ULL * maxLocalElems * sizeof(T) + metaBytes + 8ULL * 1024ULL;
        if (requiredBytes > ubBytes) {
            return false;
        }

        pipe->InitBuffer(inQueue, 1U, modelMaxSourceElems * sizeof(T));
        pipe->InitBuffer(outQueue, 2U, maxLocalElems * sizeof(T));
        pipe->InitBuffer(maskBuf, maxLocalElems * sizeof(uint32_t));
        pipe->InitBuffer(periodicIndexWorkBuf,
                         maxLocalElems * sizeof(int32_t));
        pipe->InitBuffer(continuousBaseIndexBuf,
                         maxLocalElems * sizeof(int32_t));
        pipe->InitBuffer(copyQueue, 1U, maxLocalElems * sizeof(T));
        InitMetaCache();

        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t rowBegin =
            blockIdx * baseRowsPerCore + MinU32(blockIdx, extraRows);
        const uint32_t rowCount =
            baseRowsPerCore + (blockIdx < extraRows ? 1U : 0U);
        if (rowCount == 0U) {
            return true;
        }

        LocalTensor<uint32_t> indexLocal = maskBuf.Get<uint32_t>();
        LocalTensor<int32_t> baseLocal =
            continuousBaseIndexBuf.Get<int32_t>();
        CreateVecIndex(baseLocal, 0, maxLocalElems);
        Muls(baseLocal, baseLocal, static_cast<int32_t>(sizeof(uint32_t)),
             maxLocalElems);

        CompositeGatherGroup pendingGroup{
            0U, 0U, 0U, 0U, 0U, 0U, false};
        uint32_t pendingPitch = 0U;
        uint32_t cursor = 0U;
        uint32_t axisCursor = 0U;
        while (true) {
            const CompositeGatherGroup group =
                NextCompositeGatherGroup(cursor, axisCursor, rowCount);
            if (!group.valid) {
                break;
            }

            // Issue MTE3(previous) before the current continuous MTE2/index
            // work, retaining the proven one-outstanding-input queue rule.
            if (pendingGroup.valid) {
                StoreCompositeGatherGroup(
                    pendingGroup, pendingPitch, rowBegin, rowCount, yGm);
                pendingGroup.valid = false;
            }

            if (group.nonZeroMembers == 1U &&
                (group.groupElems & 7U) == 0U) {
                CopyCompositeGatherDirect(group, rowBegin, rowCount, yGm);
                continue;
            }

            LoadCompositeGatherGroup<T>(group, rowBegin, rowCount);
            const uint32_t pitch = group.nonZeroMembers == 1U
                ? ContinuousGatherPitch(group.groupElems)
                : kContinuousMaxWidth;
            if (group.nonZeroMembers == 1U) {
                BuildContinuousGatherIndex(
                    group.groupElems, pitch, rowCount, indexLocal);
            } else {
                BuildCompositeGatherIndex(group, rowCount, indexLocal);
            }

            LocalTensor<T> ready = inQueue.DeQue<T>();
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            Gather(outputLocal, ready, indexLocal, 0U, rowCount * pitch);
            outQueue.EnQue(outputLocal);
            inQueue.FreeTensor(ready);
            pendingGroup = group;
            pendingPitch = pitch;
        }
        if (pendingGroup.valid) {
            StoreCompositeGatherGroup(
                pendingGroup, pendingPitch, rowBegin, rowCount, yGm);
        }
        return true;
    }
