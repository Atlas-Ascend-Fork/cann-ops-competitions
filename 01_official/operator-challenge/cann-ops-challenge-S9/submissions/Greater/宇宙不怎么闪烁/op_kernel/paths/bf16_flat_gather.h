    // Experimental C-row path: contiguous self DMA, exact widened RHS Gather.
    static constexpr uint32_t kBf16FlatTileElems = 12032U;
    static constexpr uint32_t kBf16FlatRhsElems = 2048U;
    static constexpr uint32_t kBf16ExpandedCapacity = 12288U;
    static constexpr uint32_t kBf16FlatExplicitBytes =
        4U * kBf16FlatTileElems + 4U * kBf16FlatRhsElems +
        2U * kBf16FlatTileElems + 4U * kBf16FlatTileElems +
        4U * kBf16ExpandedCapacity + kBf16ExpandedCapacity / 2U +
        kBf16FlatTileElems / 8U + 4U * 256U;
    static_assert(kBf16FlatExplicitBytes == 186336U, "flat BF16 UB accounting");
    static_assert(kBf16FlatExplicitBytes + 8192U <= 192U * 1024U,
                  "flat BF16 overlaps Select scratch");

    struct Bf16FlatTile {
        uint32_t offset;
        uint32_t count;
        uint32_t group;
    };

    __aicore__ inline Bf16FlatTile MakeBf16FlatTile(
        uint32_t offset, uint32_t end, uint32_t width, uint32_t planeElems) const
    {
        const uint32_t group = offset / planeElems;
        const uint32_t groupEnd = (group + 1U) * planeElems;
        const uint32_t rowTileElems = (kBf16FlatTileElems / width) * width;
        return Bf16FlatTile{offset,
            MinU32(rowTileElems, MinU32(end, groupEnd) - offset), group};
    }

    __aicore__ inline void ProcessBf16CRowFlatGather()
    {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            const uint32_t width = outDims[4];
            // Original C-row eligibility remains the entry guard; explicitly
            // bound RHS capacity and address arithmetic for future callers.
            if (width < 1024U || width > kBf16FlatRhsElems ||
                (kBf16FlatTileElems / width) * AlignElems256<bfloat16_t>(width) >
                    kBf16ExpandedCapacity) {
                ProcessBf16CRowOther();
                return;
            }
            selfGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            pipe->InitBuffer(bf16CRowSelfQueue, 2U, kBf16FlatTileElems * 2U);
            pipe->InitBuffer(bankPadBuf, 256U);  // invariant half ones
            pipe->InitBuffer(bf16CRowOtherQueue, 2U, kBf16FlatRhsElems * 2U);
            pipe->InitBuffer(bankTailPadBuf, 256U);
            pipe->InitBuffer(bf16CRowOutQueue, 2U, kBf16FlatTileElems);
            pipe->InitBuffer(bankScratchPadBuf, 256U);
            pipe->InitBuffer(otherFloatBuf, kBf16FlatTileElems * 4U);
            // Separate compact RHS and the shared expansion/self buffer.
            // Previously both FP32 operands began at the same 512B phase.
            pipe->InitBuffer(bankMaskPadBuf, 256U);
            pipe->InitBuffer(broadcastBuf, kBf16ExpandedCapacity * 4U);
            pipe->InitBuffer(int32RowGatherPatternBuf, kBf16ExpandedCapacity / 2U);
            pipe->InitBuffer(maskBuf, kBf16FlatTileElems / 8U);
            LocalTensor<half> one = bankPadBuf.Get<half>();
            Duplicate(one, static_cast<half>(1.0F), 128U);
            PipeBarrier<PIPE_V>();

            uint32_t rowStart = 0U;
            uint32_t rowCount = 0U;
            Bf16CRowRange(totalSize / width, outDims[3], GetBlockNum(),
                          GetBlockIdx(), rowStart, rowCount);
            if (rowCount == 0U) {
                return;
            }
            const uint32_t end = (rowStart + rowCount) * width;
            const uint32_t planeElems = outDims[3] * width;
            Bf16FlatTile current = MakeBf16FlatTile(rowStart * width, end, width, planeElems);
            PrefetchBf16Flat(current, width);
            // Every tile starts at column zero: ownership, group ends
            // and tile capacity are all whole rows. Cache only the padding pattern,
            // never input values or comparison results.
            BuildBf16FlatPattern(width);
            Bf16FlatTile pending{};
            bool hasPending = false;
            while (true) {
                LocalTensor<bfloat16_t> self = bf16CRowSelfQueue.DeQue<bfloat16_t>();
                LocalTensor<bfloat16_t> rhs = bf16CRowOtherQueue.DeQue<bfloat16_t>();
                const uint32_t nextOffset = current.offset + current.count;
                const bool hasNext = nextOffset < end;
                Bf16FlatTile next{};
                if (hasNext) {
                    next = MakeBf16FlatTile(nextOffset, end, width, planeElems);
                    PrefetchBf16Flat(next, width);
                }
                if (hasPending) {
                    CopyOutBf16Flat(pending);
                }
                ComputeBf16Flat(self, rhs, current, width,
                    !hasPending || current.group != pending.group);
                pending = current;
                hasPending = true;
                if (!hasNext) {
                    break;
                }
                current = next;
            }
            CopyOutBf16Flat(pending);
        }
    }

    __aicore__ inline void PrefetchBf16Flat(const Bf16FlatTile& tile, uint32_t width)
    {
        LocalTensor<bfloat16_t> self = bf16CRowSelfQueue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> rhs = bf16CRowOtherQueue.AllocTensor<bfloat16_t>();
        DataCopyPadExtParams<bfloat16_t> pad{false, 0U, 0U, static_cast<bfloat16_t>(0.0F)};
        DataCopyExtParams selfCopy{1U, tile.count * 2U, 0U, 0U, 0U};
        DataCopyExtParams rhsCopy{1U, width * 2U, 0U, 0U, 0U};
        DataCopyPad(self, selfGm[tile.offset], selfCopy, pad);
        DataCopyPad(rhs, otherGm[tile.group * width], rhsCopy, pad);
        bf16CRowSelfQueue.EnQue(self);
        bf16CRowOtherQueue.EnQue(rhs);
    }

    __aicore__ inline void BuildBf16FlatPattern(uint32_t width)
    {
        const uint32_t pitch = AlignElems256<bfloat16_t>(width);
        const uint32_t rows = kBf16FlatTileElems / width;
        const uint32_t repeats = rows * pitch / 64U;
        LocalTensor<uint32_t> pattern = int32RowGatherPatternBuf.Get<uint32_t>();
        const TEventID vToS = pipe->AllocEventID<HardEvent::V_S>();
        const TEventID sToV = pipe->AllocEventID<HardEvent::S_V>();
        Duplicate(pattern, 0xFFFFFFFFU, repeats * 8U);
        SetFlag<HardEvent::V_S>(vToS);
        WaitFlag<HardEvent::V_S>(vToS);
        // Each 64-element GatherMask repeat consumes two words in a 32-byte
        // pattern block. Only the <=127 padding bits at each row end differ.
        for (uint32_t row = 0U; row < rows; ++row) {
            for (uint32_t word = width / 32U; word < pitch / 32U; ++word) {
                const uint32_t valid = width > word * 32U ? width - word * 32U : 0U;
                const uint32_t bits = valid == 0U ? 0U : ((1U << valid) - 1U);
                const uint32_t globalWord = row * (pitch / 32U) + word;
                pattern.SetValue((globalWord / 2U) * 8U + globalWord % 2U, bits);
            }
        }
        SetFlag<HardEvent::S_V>(sToV);
        WaitFlag<HardEvent::S_V>(sToV);
        pipe->ReleaseEventID<HardEvent::V_S>(vToS);
        pipe->ReleaseEventID<HardEvent::S_V>(sToV);
    }

    __aicore__ inline void ComputeBf16Flat(
        LocalTensor<bfloat16_t>& self, LocalTensor<bfloat16_t>& rhs,
        const Bf16FlatTile& tile, uint32_t width, bool refreshRhs)
    {
        const uint32_t compute = ((tile.count + 127U) / 128U) * 128U;
        // Short tiles consume the same compact RHS prefix.
        // The index buffer is disjoint from every payload scratch.
        // RHS expansion dies after GatherMask; its buffer then holds self.
        // The compact RHS stays in otherFloatBuf across the whole group.
        LocalTensor<float> selfFloat = broadcastBuf.Get<float>();
        LocalTensor<float> rhsGathered = otherFloatBuf.Get<float>();
        LocalTensor<float> rhsFloat = broadcastBuf.Get<float>();
        LocalTensor<uint32_t> index = int32RowGatherPatternBuf.Get<uint32_t>();
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<uint8_t> out = bf16CRowOutQueue.AllocTensor<uint8_t>();
        // The broadcast group fixes RHS values for this invocation.
        // Its first tile is full-sized whenever another tile follows;
        // all later tiles need only the already-expanded prefix.
        if (refreshRhs) {
            PipeBarrier<PIPE_V>();
            Cast(rhsFloat, rhs, RoundMode::CAST_NONE, AlignElems256<bfloat16_t>(width));
            PipeBarrier<PIPE_V>();
            const uint32_t pitch = AlignElems256<bfloat16_t>(width);
            const uint32_t rows = kBf16FlatTileElems / width;
            LocalTensor<int32_t> expanded = rhsFloat.ReinterpretCast<int32_t>();
            for (uint32_t row = 1U; row < rows; ++row) {
                Adds(expanded[row * pitch], expanded, static_cast<int32_t>(0), pitch);
            }
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> compact = rhsGathered.ReinterpretCast<int32_t>();
            uint64_t selected = 0U;
            GatherMask(compact, expanded, index, false, 0U,
                {1U, static_cast<uint16_t>(rows * pitch / 64U), 8U, 1U}, selected);
        }
        PipeBarrier<PIPE_V>();  // Finish expansion reads before reusing storage.
        Cast(selfFloat, self, RoundMode::CAST_NONE, compute);
        PipeBarrier<PIPE_V>();
        Compare(mask, selfFloat, rhsGathered, CMPMODE::GT, compute);
        LocalTensor<half> result = self.template ReinterpretCast<half>();
        LocalTensor<half> one = bankPadBuf.Get<half>();
        const BinaryRepeatParams repeats{1U, 1U, 1U, 8U, 0U, 0U};
        Select(result, mask, one, static_cast<half>(0.0F),
               SELMODE::VSEL_TENSOR_SCALAR_MODE, 128U,
               static_cast<uint8_t>(compute / 128U), repeats);
        Cast(out, result, RoundMode::CAST_NONE, compute);
        // Padded self lanes are unspecified but lane-local; their Gather
        // indices remain in range, and their output bytes are never written.
        bf16CRowOutQueue.EnQue(out);
        bf16CRowSelfQueue.FreeTensor(self);
        bf16CRowOtherQueue.FreeTensor(rhs);
    }

    __aicore__ inline void CopyOutBf16Flat(const Bf16FlatTile& tile)
    {
        LocalTensor<uint8_t> out = bf16CRowOutQueue.DeQue<uint8_t>();
        DataCopyExtParams copy{1U, tile.count, 0U, 0U, 0U};
        DataCopyPad(outGm[tile.offset], out, copy);
        bf16CRowOutQueue.FreeTensor(out);
    }
