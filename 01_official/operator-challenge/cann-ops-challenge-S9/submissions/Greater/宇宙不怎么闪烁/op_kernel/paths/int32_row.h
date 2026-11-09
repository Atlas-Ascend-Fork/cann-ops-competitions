    struct Int32RowTile {
        uint32_t offset;
        uint32_t rows;
        uint32_t width;
        uint32_t column;
        uint32_t span;
    };

    __aicore__ inline bool CanProcessInt32RowSelfBroadcast() const
    {
        if constexpr (!std::is_same_v<T, int32_t>) {
            return false;
        } else {
            const uint32_t nDim = outDims[1];
            const uint32_t cDim = outDims[2];
            const uint32_t hDim = outDims[3];
            const uint32_t wDim = outDims[4];
            if (outDims[0] != 1U || nDim == 0U || cDim == 0U || hDim < 2U ||
                wDim == 0U || wDim > kInt32RowSelfPitchElems || GetBlockNum() == 0U) {
                return false;
            }
            const uint32_t groupElems = kInt32RowSelfMaxRows * wDim;
            const uint32_t groupCount = totalSize / groupElems;
            return hDim % kInt32RowSelfMaxRows == 0U &&
                   groupCount >= GetBlockNum() &&
                   selfDims[0] == outDims[0] && selfDims[1] == nDim &&
                   selfDims[2] == cDim && selfDims[3] == 1U &&
                   selfDims[4] == wDim && otherDense;
        }
    }

    __aicore__ inline void ProcessInt32RowSelfBroadcast()
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            const uint32_t hDim = outDims[3];
            const uint32_t wDim = outDims[4];
            const uint32_t planeElems = hDim * wDim;

            // A complete 12-row tile is the natural unit of the int32 fast
            // path.  Splitting by the generic 256-element work block leaves
            // every core with partial rows at both ends, which adds padded
            // DMA/vector work and makes otherwise identical cores finish at
            // different times.  When H is tile-aligned, distribute complete
            // tiles instead.  Case2 has 3360 tiles, so 40 AIVs receive exactly
            // 84 tiles each and no core executes a boundary fragment.
            uint32_t rangeStart = startElem;
            uint32_t rangeEnd = endElem;
            if (hDim % kInt32RowSelfMaxRows == 0U) {
                const uint32_t groupElems =
                    kInt32RowSelfMaxRows * wDim;
                const uint32_t groupCount = totalSize / groupElems;
                const uint32_t blockNum = GetBlockNum();
                if (groupCount >= blockNum) {
                    const uint32_t blockIdx = GetBlockIdx();
                    const uint32_t baseGroups = groupCount / blockNum;
                    const uint32_t remainGroups =
                        groupCount - baseGroups * blockNum;
                    const uint32_t extraBefore =
                        blockIdx < remainGroups ? blockIdx : remainGroups;
                    const uint32_t myGroups = baseGroups +
                        (blockIdx < remainGroups ? 1U : 0U);
                    const uint32_t startGroup =
                        blockIdx * baseGroups + extraBefore;
                    rangeStart = startGroup * groupElems;
                    rangeEnd = rangeStart + myGroups * groupElems;
                }
            }

            // Bypass L2 only for the one-pass dense RHS stream.
            otherGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            const bool compact = wDim == kInt32RowPackedWidthElems;
            InitInt32RowSelfBroadcastBuffers(compact);
            if (compact) {
                BuildInt32RowCompactGatherPattern();
            } else {
                BuildInt32RowGatherPattern();
            }
            const uint32_t stripeRows = 3U * kInt32RowSelfMaxRows;
            const uint32_t stripeElems = stripeRows * wDim;
            const uint32_t stripes = totalSize / stripeElems;
            if (compact && hDim % stripeRows == 0U && stripes >= GetBlockNum()) {
                // Consecutive cores stream neighboring multi-tile stripes.
                // Every stripe is wholly inside one broadcast plane; all
                // input/output addresses retain their 32-byte alignment.
                for (uint32_t stripe = GetBlockIdx(); stripe < stripes;
                     stripe += GetBlockNum()) {
                    const uint32_t offset = stripe * stripeElems;
                    ProcessInt32RowFlatRange(hDim, wDim, planeElems,
                                             offset, offset + stripeElems);
                }
                return;
            }
            ProcessInt32RowFlatRange(
                hDim, wDim, planeElems, rangeStart, rangeEnd);
        }
    }

    __aicore__ inline void InitInt32RowSelfBroadcastBuffers(bool compact)
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            // Relative UB starts are deliberately phased as follows:
            // self row g0, expanded g8, rhs slots g2/g10, and output slots
            // g4/g12.  The two streaming queues therefore use different bank
            // groups in both ping/pong phases.
            pipe->InitBuffer(selfQueue, kBufferNum, kInt32RowSelfBytes);
            pipe->InitBuffer(bankPadBuf, kInt32RowSelfToExpandedPadBytes);
            pipe->InitBuffer(broadcastBuf,
                             kInt32RowBatchElems * sizeof(int32_t));
            pipe->InitBuffer(bankTailPadBuf,
                             kInt32RowExpandedToOtherPadBytes);
            pipe->InitBuffer(int32RowOtherQueue, 2U, kInt32RowOtherSlotBytes);
            pipe->InitBuffer(bankScratchPadBuf,
                              kInt32RowOtherToOutPadBytes);
            pipe->InitBuffer(int32RowOutQueue, 2U, kInt32RowOutSlotBytes);
            pipe->InitBuffer(int32RowGatherPatternBuf,
                             compact ? kInt32RowCompactGatherPatternBytes :
                                       kInt32RowGatherPatternBytes);
            // The existing 256B bank-separation allocation now also holds an
            // invariant zero vector. No Select destination aliases this source.
            LocalTensor<half> selectZero = bankPadBuf.Get<half>();
            Duplicate(selectZero, static_cast<half>(0.0F), 128U);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void ProcessInt32RowFlatRange(
        uint32_t hDim, uint32_t wDim, uint32_t planeElems,
        uint32_t rangeStart, uint32_t rangeEnd)
    {
        if constexpr (std::is_same_v<T, int32_t>) {

            uint32_t rangeOffset = rangeStart;
            while (rangeOffset < rangeEnd) {
                const uint32_t plane = rangeOffset / planeElems;
                const uint32_t planeStart = plane * planeElems;
                const uint32_t planeEnd =
                    MinU32(rangeEnd, planeStart + planeElems);

                LocalTensor<int32_t> selfRow = selfQueue.AllocTensor<int32_t>();
                CopyInt32RowSelf(selfRow, plane * wDim, wDim);
                selfQueue.EnQue(selfRow);
                selfRow = selfQueue.DeQue<int32_t>();
                if (wDim == kInt32RowPackedWidthElems) {
                    ProcessInt32RowCompactPlaneRange(
                        rangeOffset, planeEnd, planeStart, wDim, selfRow);
                } else {
                    ProcessInt32RowPlaneRange(rangeOffset, planeEnd,
                                              planeStart, wDim, selfRow);
                }
                rangeOffset = planeEnd;
            }
        }
    }

    __aicore__ inline void CopyInt32RowSelf(LocalTensor<int32_t>& local,
                                             uint32_t offset, uint32_t width)
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            DataCopyExtParams copyParams{
                1U, static_cast<uint32_t>(width * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0};
            DataCopyPad(local, selfGm[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void BuildInt32RowSelfBroadcast(LocalTensor<int32_t>& dst,
                                                       LocalTensor<int32_t>& row,
                                                       uint32_t rows)
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            constexpr uint32_t repeatElems = 64U;
            constexpr uint32_t repeatsPerRow = kInt32RowSelfPitchElems / repeatElems;
            constexpr uint8_t dstRepeatStride =
                static_cast<uint8_t>(kInt32RowSelfPitchElems * sizeof(int32_t) / 32U);
            const UnaryRepeatParams repeatParams{1U, 1U, dstRepeatStride, 0U};
            for (uint32_t repeat = 0U; repeat < repeatsPerRow; ++repeat) {
                const uint32_t offset = repeat * repeatElems;
                Adds(dst[offset], row[offset], static_cast<int32_t>(0),
                     static_cast<uint64_t>(repeatElems),
                     static_cast<uint8_t>(rows), repeatParams);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void BuildInt32RowGatherPattern()
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            // GatherMask consumes 128 half predicates per 256-byte repeat.  A
            // custom 128-bit pattern occupies the first four uint32 words of
            // each 32-byte pattern block.  Every eighth repeat ends one
            // 1024-element UB row, so clear that repeat's final predicate.
            LocalTensor<uint32_t> words =
                int32RowGatherPatternBuf.Get<uint32_t>();
            constexpr uint32_t wordsPerPatternBlock =
                kInt32RowGatherPatternStrideBytes / sizeof(uint32_t);
            for (uint32_t repeat = 0U; repeat < kInt32RowGatherRepeats;
                 ++repeat) {
                const uint32_t base = repeat * wordsPerPatternBlock;
                words.SetValue(base, 0xFFFFFFFFU);
                words.SetValue(base + 1U, 0xFFFFFFFFU);
                words.SetValue(base + 2U, 0xFFFFFFFFU);
                words.SetValue(base + 3U,
                    ((repeat + 1U) & 7U) == 0U ? 0x7FFFFFFFU : 0xFFFFFFFFU);
            }
            const event_t eventId = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventId);
            WaitFlag<HardEvent::S_V>(eventId);
        }
    }

    __aicore__ inline void BuildInt32RowCompactGatherPattern()
    {
        if constexpr (std::is_same_v<T, int32_t>) {
            LocalTensor<uint32_t> words =
                int32RowGatherPatternBuf.Get<uint32_t>();
            constexpr uint32_t wordCount =
                kInt32RowCompactGatherPatternBytes / sizeof(uint32_t);
            constexpr uint32_t wordsPerPatternBlock =
                kInt32RowGatherPatternStrideBytes / sizeof(uint32_t);
            constexpr uint32_t repeatsPerRow =
                kInt32RowSelfPitchElems / (256U / sizeof(int32_t));
            Duplicate(words, 0xFFFFFFFFU, wordCount);
            const event_t vecToScalar = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(vecToScalar);
            WaitFlag<HardEvent::V_S>(vecToScalar);
            for (uint32_t row = 0U; row < kInt32RowSelfMaxRows; ++row) {
                const uint32_t repeat = (row + 1U) * repeatsPerRow - 1U;
                words.SetValue(repeat * wordsPerPatternBlock + 1U,
                               0x7FFFFFFFU);
            }
            const event_t scalarToVec = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(scalarToVec);
            WaitFlag<HardEvent::S_V>(scalarToVec);
        }
    }

    __aicore__ inline bool CanPackInt32RowTile(
        const Int32RowTile& tile, uint32_t fullWidth) const
    {
        return fullWidth == kInt32RowPackedWidthElems &&
               tile.column == 0U && tile.width == fullWidth &&
               tile.rows + 1U >= kInt32RowSelfMaxRows;
    }

    __aicore__ inline Int32RowTile MakeInt32RowTile(
        uint32_t offset, uint32_t rangeEnd,
        uint32_t planeStart, uint32_t fullWidth) const
    {
        const uint32_t localOffset = offset - planeStart;
        const uint32_t column = localOffset % fullWidth;
        const uint32_t remaining = rangeEnd - offset;
        if (column != 0U) {
            const uint32_t width = MinU32(fullWidth - column, remaining);
            return Int32RowTile{offset, 1U, width, column, width};
        }
        const uint32_t fullRows = remaining / fullWidth;
        if (fullRows != 0U) {
            const uint32_t rows = MinU32(kInt32RowSelfMaxRows, fullRows);
            return Int32RowTile{offset, rows, fullWidth, 0U, rows * fullWidth};
        }
        return Int32RowTile{offset, 1U, remaining, 0U, remaining};
    }

    __aicore__ inline void ProcessInt32RowCompactPlaneRange(
        uint32_t rangeStart, uint32_t rangeEnd,
        uint32_t planeStart, uint32_t fullWidth,
        LocalTensor<int32_t>& selfRow)
    {
        Int32RowTile current =
            MakeInt32RowTile(rangeStart, rangeEnd, planeStart, fullWidth);
        PrefetchInt32RowOtherCompact(current);
        BuildInt32RowCompactSelf(selfRow);
        Int32RowTile pending{};
        bool hasPending = false;

        while (true) {
            LocalTensor<int32_t> otherLocal =
                int32RowOtherQueue.DeQue<int32_t>();
            const uint32_t nextOffset = current.offset + current.span;
            const bool hasNext = nextOffset < rangeEnd;
            Int32RowTile next{};
            if (hasNext) {
                next = MakeInt32RowTile(nextOffset, rangeEnd,
                                       planeStart, fullWidth);
                PrefetchInt32RowOtherCompact(next);
            }
            if (hasPending) {
                CopyOutInt32RowCompactTile(pending);
            }
            ComputeInt32RowCompactTileReady(otherLocal, current);
            pending = current;
            hasPending = true;

            if (!hasNext) {
                break;
            }
            current = next;
        }
        CopyOutInt32RowCompactTile(pending);
    }

    __aicore__ inline void BuildInt32RowCompactSelf(
        LocalTensor<int32_t>& selfRow)
    {
        LocalTensor<int32_t> expanded =
            int32RowOtherQueue.AllocTensor<int32_t>();
        BuildInt32RowSelfBroadcast(expanded, selfRow,
                                   kInt32RowSelfMaxRows);
        LocalTensor<int32_t> compact = broadcastBuf.Get<int32_t>();
        LocalTensor<uint32_t> gatherPattern =
            int32RowGatherPatternBuf.Get<uint32_t>();
        uint64_t selectedCount = 0U;
        GatherMask(compact, expanded, gatherPattern, false, 0U,
            {1U, static_cast<uint16_t>(kInt32RowCompactGatherRepeats),
             8U, 1U}, selectedCount);
        selfQueue.FreeTensor(selfRow);
        int32RowOtherQueue.FreeTensor(expanded);
    }

    __aicore__ inline void PrefetchInt32RowOtherCompact(
        const Int32RowTile& tile)
    {
        LocalTensor<int32_t> otherLocal =
            int32RowOtherQueue.AllocTensor<int32_t>();
        if (IsGmCopyAligned<int32_t>(tile.offset, tile.span)) {
            DataCopy(otherLocal, otherGm[tile.offset], tile.span);
        } else {
            DataCopyExtParams copyParams{
                1U, static_cast<uint32_t>(tile.span * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0};
            DataCopyPad(otherLocal, otherGm[tile.offset], copyParams, padParams);
        }
        int32RowOtherQueue.EnQue(otherLocal);
    }

    __aicore__ inline void ComputeInt32RowCompactTileReady(
        LocalTensor<int32_t>& otherLocal, const Int32RowTile& tile)
    {
        LocalTensor<uint8_t> outLocal =
            int32RowOutQueue.AllocTensor<uint8_t>();
        LocalTensor<int32_t> selfCompact = broadcastBuf.Get<int32_t>();
        const uint32_t selfBase = tile.column;
        const uint32_t computeElems =
            ((tile.span + 127U) / 128U) * 128U;
        Min(otherLocal, selfCompact[selfBase], otherLocal, computeElems);
        LocalTensor<uint8_t> leMask = outLocal;
        Compare(leMask, selfCompact[selfBase], otherLocal,
                CMPMODE::EQ, computeElems);
        LocalTensor<half> resultHalf = otherLocal.ReinterpretCast<half>();
        LocalTensor<half> selectZero = bankPadBuf.Get<half>();
        const uint8_t repeatTime =
            static_cast<uint8_t>(computeElems / 128U);
        const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
        Select(resultHalf, leMask, selectZero, static_cast<half>(1.0F),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
               repeatParams);
        Cast(outLocal, resultHalf, RoundMode::CAST_NONE, computeElems);
        int32RowOutQueue.EnQue(outLocal);
        int32RowOtherQueue.FreeTensor(otherLocal);
    }

    __aicore__ inline void CopyOutInt32RowCompactTile(
        const Int32RowTile& tile)
    {
        LocalTensor<uint8_t> outLocal =
            int32RowOutQueue.DeQue<uint8_t>();
        // Queue FreeTensor/AllocTensor fences slot reuse after its own DMA.
        // Independent output slots need not serialize the entire MTE3 pipe.
        DataCopyExtParams copyParams{1U, tile.span, 0U, 0U, 0U};
        DataCopyPad(outGm[tile.offset], outLocal, copyParams);
        int32RowOutQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void ProcessInt32RowPlaneRange(
        uint32_t rangeStart, uint32_t rangeEnd,
        uint32_t planeStart, uint32_t fullWidth,
        LocalTensor<int32_t>& selfRow)
    {
        Int32RowTile current =
            MakeInt32RowTile(rangeStart, rangeEnd, planeStart, fullWidth);
        // The first dense RHS transfer is independent of self expansion.
        // Issue it first so MTE2 overlaps the plane's VEC template build.
        PrefetchInt32RowOther(current, fullWidth);
        LocalTensor<int32_t> selfExpanded = broadcastBuf.Get<int32_t>();
        BuildInt32RowSelfBroadcast(selfExpanded, selfRow,
                                   kInt32RowSelfMaxRows);
        selfQueue.FreeTensor(selfRow);
        Int32RowTile pending{};
        bool hasPending = false;

        while (true) {
            LocalTensor<int32_t> otherLocal =
                int32RowOtherQueue.DeQue<int32_t>();
            const uint32_t nextOffset = current.offset + current.span;
            const bool hasNext = nextOffset < rangeEnd;
            Int32RowTile next{};
            if (hasNext) {
                next = MakeInt32RowTile(nextOffset, rangeEnd,
                                       planeStart, fullWidth);
                PrefetchInt32RowOther(next, fullWidth);
            }
            if (hasPending) {
                CopyOutInt32RowTile(pending, fullWidth);
            }
            ComputeInt32RowTileReady(selfExpanded, otherLocal,
                                     current, fullWidth);
            pending = current;
            hasPending = true;

            if (!hasNext) {
                break;
            }
            current = next;
        }
        if (hasPending) {
            CopyOutInt32RowTile(pending, fullWidth);
        }
    }

    __aicore__ inline void PrefetchInt32RowOther(
        const Int32RowTile& tile, uint32_t fullWidth)
    {
        LocalTensor<int32_t> otherLocal =
            int32RowOtherQueue.AllocTensor<int32_t>();
        const uint32_t blockBytes = tile.width * sizeof(int32_t);
        const uint32_t rowPitchBytes =
            kInt32RowSelfPitchElems * sizeof(int32_t);
        const uint32_t paddedBlockBytes = AlignUp32Bytes(blockBytes);
        const uint32_t dstStride =
            (rowPitchBytes - paddedBlockBytes) / 32U;
        const uint32_t srcStride =
            (fullWidth - tile.width) * sizeof(int32_t);
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(tile.rows), blockBytes,
            srcStride, dstStride, 0U};
        DataCopyPadExtParams<int32_t> padParams{false, 0U, 0U, 0};
        DataCopyPad(otherLocal, otherGm[tile.offset], copyParams, padParams);
        int32RowOtherQueue.EnQue(otherLocal);
    }

    __aicore__ inline void ComputeInt32RowTileReady(
        LocalTensor<int32_t>& selfExpanded,
        LocalTensor<int32_t>& otherLocal,
        const Int32RowTile& tile, uint32_t fullWidth)
    {
        LocalTensor<uint8_t> outLocal =
            int32RowOutQueue.AllocTensor<uint8_t>();
        LocalTensor<int32_t> minLocal = otherLocal;
        const bool fullRows = tile.column == 0U && tile.width == fullWidth;
        const bool packedOutput = CanPackInt32RowTile(tile, fullWidth);

        for (uint32_t row0 = 0U; row0 < tile.rows;
             row0 += kInt32RowComputeRows) {
            const uint32_t chunkRows =
                MinU32(kInt32RowComputeRows, tile.rows - row0);
            const uint32_t logicalElems = fullRows ?
                chunkRows * kInt32RowSelfPitchElems : tile.width;
            const uint32_t computeElems =
                ((logicalElems + 127U) / 128U) * 128U;
            const uint32_t selfBase = fullRows ?
                row0 * kInt32RowSelfPitchElems : tile.column;
            const uint32_t localBase = fullRows ?
                row0 * kInt32RowSelfPitchElems : 0U;

            Min(minLocal[localBase], selfExpanded[selfBase],
                otherLocal[localBase], computeElems);
            // The packed EQ predicate occupies only one eighth of this output
            // chunk.  Select consumes it before Cast overwrites the same byte
            // range with the final bool materialization.
            LocalTensor<uint8_t> leMask = outLocal[localBase];
            Compare(leMask, selfExpanded[selfBase], minLocal[localBase],
                    CMPMODE::EQ, computeElems);
            LocalTensor<half> resultHalf =
                minLocal[localBase].ReinterpretCast<half>();
            LocalTensor<half> selectZero = bankPadBuf.Get<half>();
            const uint8_t repeatTime =
                static_cast<uint8_t>(computeElems / 128U);
            const BinaryRepeatParams repeatParams{1U, 1U, 1U, 8U, 0U, 0U};
            Select(resultHalf, leMask, selectZero, static_cast<half>(1.0F),
                   SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U, repeatTime,
                   repeatParams);
            if (packedOutput) {
                // The lower half of the consumed 49 KiB rhs slot holds the
                // aligned Select result.  Compact into its dead upper half,
                // shifted by 256 bytes (eight banks), before the narrowing
                // Cast.  This converts eleven or twelve 1023-byte MTE3 blocks
                // into one contiguous block without another UB allocation.
                LocalTensor<half> packedHalf =
                    minLocal.template ReinterpretCast<half>()[
                        kInt32RowPackedHalfOffsetBytes / sizeof(half)];
                LocalTensor<uint16_t> gatherPattern =
                    int32RowGatherPatternBuf.Get<uint16_t>();
                uint64_t selectedCount = 0U;
                const uint16_t gatherRepeats = static_cast<uint16_t>(
                    tile.rows * (kInt32RowSelfPitchElems /
                                 kInt32RowHalfRepeatElems));
                GatherMask(packedHalf, resultHalf, gatherPattern, false, 0U,
                    {1U, gatherRepeats, 8U, 1U}, selectedCount);
                Cast(outLocal, packedHalf, RoundMode::CAST_NONE, tile.span);
            } else {
                Cast(outLocal[localBase], resultHalf,
                     RoundMode::CAST_NONE, computeElems);
            }
        }
        int32RowOutQueue.EnQue(outLocal);
        int32RowOtherQueue.FreeTensor(otherLocal);
    }

    __aicore__ inline void CopyOutInt32RowTile(
        const Int32RowTile& tile, uint32_t fullWidth)
    {
        LocalTensor<uint8_t> outLocal =
            int32RowOutQueue.DeQue<uint8_t>();
        // Queue depth two lets VEC materialize the next result while MTE3 is
        // draining the previous slot.  Do not, however, let two 12 KiB GM
        // writes overlap on MTE3: the B4 simulator shows that the paired
        // transfers contend and each expands from about 2 us to 4--5 us.
        // This barrier preserves cross-pipe VEC/MTE3 overlap while serializing
        // only consecutive writes on the output pipe.
        PipeBarrier<PIPE_MTE3>();
        if (CanPackInt32RowTile(tile, fullWidth)) {
            DataCopyExtParams packedCopyParams{
                1U, tile.span, 0U, 0U, 0U};
            DataCopyPad(outGm[tile.offset], outLocal, packedCopyParams);
            int32RowOutQueue.FreeTensor(outLocal);
            return;
        }
        const uint32_t paddedBlockBytes = AlignUp32Bytes(tile.width);
        const uint32_t srcStride =
            (kInt32RowSelfPitchElems - paddedBlockBytes) / 32U;
        const uint32_t dstStride = fullWidth - tile.width;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(tile.rows), tile.width,
            srcStride, dstStride, 0U};
        DataCopyPad(outGm[tile.offset], outLocal, copyParams);
        int32RowOutQueue.FreeTensor(outLocal);
    }
