// Included inside ConcatKernel. Metadata and queue members are shared with the
// route-5 family, but allocations/lifetimes are exclusive to this mode.
TBuf<TPosition::VECCALC> streamColumns;
TBuf<TPosition::VECCALC> streamGroupWidths, streamGroupBases;

__aicore__ inline uint32_t StreamInput(uint32_t n) {
    return AsymPlan(8U + n) & 0xffU;
}
__aicore__ inline uint32_t StreamInputWidth(uint32_t n) {
    // Matching host encodes all admitted mode3 plans, including ungrouped.
    return (AsymPlan(8U + n) >> 8U) & 0xffU;
}
__aicore__ inline uint32_t StreamMembers(uint32_t n) {
    const uint32_t members = (AsymPlan(8U + n) >> 16U) & 0xffU;
    return members ? members : 1U;
}
__aicore__ inline uint32_t StreamGroupWidth(uint32_t n) {
    return AsymPlan(8U + n) >> 24U;
}

__aicore__ inline uint64_t InitStreamInputBypassOffset() {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201) && defined(L2_CACHE_HINT) && \
    (!defined(ASCENDC_OOM) || ASCENDC_OOM == 0)
    // On CANN 8.5 / A2, the SDK adds one invocation-wide GM alias offset.
    // Obtain it through a valid input, once per active core; never hard-code
    // the offset, change metadata/output addresses, or retain it across calls.
    const uint32_t input = StreamInput(0U);
    auto base = reinterpret_cast<__gm__ uint32_t*>(InputPtr(input));
    GlobalTensor<uint32_t> anchor;
    anchor.SetGlobalBuffer(base, static_cast<uint64_t>(outerSize) * StreamInputWidth(0U));
    anchor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    return reinterpret_cast<uint64_t>(anchor.GetPhyAddr()) -
           reinterpret_cast<uint64_t>(base);
#else
    return 0U;
#endif
}

__aicore__ inline void PrefetchStreamInput(uint32_t n, uint32_t slot,
                                          uint32_t first, uint32_t rows,
                                          uint64_t bypassOffset) {
    auto src = slot == 0U ? asymInSmall.AllocTensor<uint32_t>() :
                            asymInLarge.AllocTensor<uint32_t>();
    const uint32_t input = StreamInput(n), width = StreamInputWidth(n), count = rows * width;
    auto base = reinterpret_cast<__gm__ uint32_t*>(InputPtr(input));
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201) && defined(L2_CACHE_HINT) && \
    (!defined(ASCENDC_OOM) || ASCENDC_OOM == 0)
    base = reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<uint64_t>(base) + bypassOffset);
#else
    (void)bypassOffset;
#endif
    GlobalTensor<uint32_t> gm;
    gm.SetGlobalBuffer(base, static_cast<uint64_t>(outerSize) * width);
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 2201) || !defined(L2_CACHE_HINT) || \
    (defined(ASCENDC_OOM) && ASCENDC_OOM != 0)
    // Preserve the SDK path for other architectures and OOM instrumentation.
    gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
    // Guards belong to UB only: no data outside the real input is read in GM.
    DataCopyPad(src[8U], gm[static_cast<uint64_t>(first) * width],
                DataCopyExtParams{1U, count * 4U, 0U, 0U, 0U},
                DataCopyPadExtParams<uint32_t>{true, 0U,
                    static_cast<uint8_t>((8U - count % 8U) % 8U), 0U});
    if (slot == 0U) asymInSmall.EnQue(src); else asymInLarge.EnQue(src);
}

__aicore__ inline void PrefetchStreamGroup(uint32_t n, uint32_t slot,
                                          uint32_t first, uint32_t rows,
                                          uint64_t bypassOffset) {
    const uint32_t members = StreamMembers(n);
    if (members == 1U) {
        PrefetchStreamInput(n, slot, first, rows, bypassOffset);
        return;
    }
    auto src = slot == 0U ? asymInSmall.AllocTensor<uint32_t>() :
                            asymInLarge.AllocTensor<uint32_t>();
    uint32_t slab = 8U;
    for (uint32_t m = 0U; m < members; ++m) {
        const uint32_t input = StreamInput(n + m), width = StreamInputWidth(n + m);
        const uint32_t count = rows * width;
        auto base = reinterpret_cast<__gm__ uint32_t*>(InputPtr(input));
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201) && defined(L2_CACHE_HINT) && \
    (!defined(ASCENDC_OOM) || ASCENDC_OOM == 0)
        base = reinterpret_cast<__gm__ uint32_t*>(reinterpret_cast<uint64_t>(base) + bypassOffset);
#else
        (void)bypassOffset;
#endif
        GlobalTensor<uint32_t> gm;
        gm.SetGlobalBuffer(base, static_cast<uint64_t>(outerSize) * width);
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 2201) || !defined(L2_CACHE_HINT) || \
    (defined(ASCENDC_OOM) && ASCENDC_OOM != 0)
        gm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif
        DataCopyPad(src[slab], gm[static_cast<uint64_t>(first) * width],
            DataCopyExtParams{1U, count * 4U, 0U, 0U, 0U},
            DataCopyPadExtParams<uint32_t>{true, 0U,
                static_cast<uint8_t>((8U - count % 8U) % 8U), 0U});
        slab += AlignElems32<uint32_t>(count);
    }
    if (slot == 0U) asymInSmall.EnQue(src); else asymInLarge.EnQue(src);
}

__aicore__ inline void BuildInputAffineIndex(uint32_t width, uint32_t rows) {
    auto index = packOffsets.Get<int32_t>();
    auto rowBytes = packDeltas.Get<int32_t>();
    auto columnBytes = streamColumns.Get<int32_t>();
    // The previous Gather/edge use of this mutable plane must finish first.
    PipeBarrier<PIPE_V>();
    Muls(index, rowBytes, static_cast<int32_t>(width), rows * 64U);
    PipeBarrier<PIPE_V>();
    Add(index, index, columnBytes, uint64_t(64U), static_cast<uint8_t>(rows),
        BinaryRepeatParams{1U, 1U, 1U, 8U, 8U, 0U});
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void BuildStreamGroupIndex(uint32_t n, uint32_t members,
                                             uint32_t consumed, uint32_t width,
                                             uint32_t bandColumn, uint32_t bandWidth,
                                             uint32_t rows) {
    const auto f = concat_input_stream_geometry::MakeFragment(
        consumed, width - consumed, bandColumn, bandWidth);
    auto widths = streamGroupWidths.Get<int32_t>();
    auto bases = streamGroupBases.Get<int32_t>();
    auto columns = streamColumns.Get<int32_t>();
    auto index = packOffsets.Get<int32_t>();
    // Scratch can have been borrowed by an edge store or the prior Gather.
    PipeBarrier<PIPE_V>();
    Duplicate(widths, int32_t(0), 64U);
    Duplicate(bases, int32_t(0), 64U);
    PipeBarrier<PIPE_V>();
    uint32_t start = 0U, slab = 8U;
    for (uint32_t m = 0U; m < members; ++m) {
        const uint32_t memberWidth = StreamInputWidth(n + m);
        const uint32_t a = consumed > start ? consumed : start;
        const uint32_t b = MinU32(consumed + f.count, start + memberWidth);
        if (a < b) {
            uint64_t mask[2] = {PackColumnMask(f.lane + a - consumed,
                                               f.lane + b - consumed), 0ULL};
            const int32_t byteBase = (static_cast<int32_t>(slab) +
                static_cast<int32_t>(consumed) - static_cast<int32_t>(f.lane) -
                static_cast<int32_t>(start)) * 4;
            Duplicate(widths, static_cast<int32_t>(memberWidth), mask, 1U, 1U, 8U);
            Adds(bases, columns, byteBase, mask, 1U, UnaryRepeatParams{1U, 1U, 8U, 8U});
            // Different masks can touch the same physical 32B UB block.
            PipeBarrier<PIPE_V>();
        }
        start += memberWidth;
        slab += AlignElems32<uint32_t>(rows * memberWidth);
    }
    Mul(index, packDeltas.Get<int32_t>(), widths, uint64_t(64U),
        static_cast<uint8_t>(rows), BinaryRepeatParams{1U, 1U, 1U, 8U, 8U, 0U});
    PipeBarrier<PIPE_V>();
    Add(index, index, bases, uint64_t(64U), static_cast<uint8_t>(rows),
        BinaryRepeatParams{1U, 1U, 1U, 8U, 8U, 0U});
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FillStreamFragment(LocalTensor<uint32_t>& dst,
                                          LocalTensor<uint32_t>& src,
                                          uint32_t inputColumn, uint32_t remaining,
                                          uint32_t bandColumn, uint32_t bandWidth,
                                          uint32_t rows, uint32_t pitch,
                                          uint32_t& count, bool mixed = false) {
    const auto f = concat_input_stream_geometry::MakeFragment(
        inputColumn, remaining, bandColumn, bandWidth);
    uint64_t mask[2] = {PackColumnMask(f.lane, f.lane + f.count), 0ULL};
    Gather(dst[f.alignedColumn], src, packOffsets.Get<uint32_t>(), mixed ? 0U : f.sourceByteBase,
           mask, static_cast<uint8_t>(rows), static_cast<uint16_t>(pitch / 8U));
    // Within a singleton, a non-final fragment ends on a 32B boundary;
    // consecutive Gathers read the same immutable index/source but write
    // disjoint blocks. The next group's index builder fences shared blocks
    // and index reuse. Band completion has its own fence before edge scratch.
    count = f.count;
}

__aicore__ inline bool StoreStreamBand(uint32_t slot, uint32_t left,
                                       uint32_t width, uint32_t pitch) {
    auto ready = slot == 0U ? asymOutSmall.DeQue<uint32_t>() :
                              asymOutLarge.DeQue<uint32_t>();
    const auto s = concat_input_stream_geometry::MakeSpan(
        sliceFirstRow, sliceLastRow, sliceFirstCol, sliceLastCol, left, width);
    bool borrowedIndex = false;
    if (s.rows == 1U) {
        StoreSliceEdge(ready, s.ubFirst * pitch + s.head,
                        s.first * outputAxisSize + left + s.head, s.tail - s.head);
        borrowedIndex = true;
    } else {
        const uint32_t fullFirst = s.head ? 1U : 0U;
        const uint32_t fullLast = s.rows - (s.tail < width ? 1U : 0U);
        if (fullLast > fullFirst) {
            GlobalTensor<uint32_t> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputAddr), totalElems);
            DataCopyPad(gm[static_cast<uint64_t>(s.first + fullFirst) * outputAxisSize + left],
                        ready[(s.ubFirst + fullFirst) * pitch],
                        DataCopyExtParams{static_cast<uint16_t>(fullLast - fullFirst), width * 4U,
                            (pitch - AlignElems32<uint32_t>(width)) / 8U,
                            (outputAxisSize - width) * 4U, 0U});
        }
        if (s.head) {
            StoreSliceEdge(ready, s.ubFirst * pitch + s.head,
                s.first * outputAxisSize + left + s.head, width - s.head);
            borrowedIndex = true;
        }
        if (s.tail < width) {
            StoreSliceEdge(ready, (s.ubFirst + s.rows - 1U) * pitch,
                (s.first + s.rows - 1U) * outputAxisSize + left, s.tail);
            borrowedIndex = true;
        }
    }
    // StoreSliceEdge's V->MTE3 dequeue completes its V read of ready before
    // this output slot can be released and eventually reused by Vector.
    if (slot == 0U) asymOutSmall.FreeTensor(ready); else asymOutLarge.FreeTensor(ready);
    return borrowedIndex;
}

__aicore__ inline bool TryProcessInputStream() {
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    const uint64_t blocks = (static_cast<uint64_t>(totalElems) + 7U) / 8U;
    const uint32_t lo = MinU32(totalElems, static_cast<uint32_t>(blocks * core / cores * 8U));
    const uint32_t hi = core + 1U == cores ? totalElems :
        static_cast<uint32_t>(blocks * (core + 1U) / cores * 8U);
    if (lo == hi) return true;
    const uint32_t firstRow = lo / outputAxisSize, firstCol = lo % outputAxisSize;
    const uint32_t lastRow = (hi - 1U) / outputAxisSize, lastCol = (hi - 1U) % outputAxisSize + 1U;
    const uint32_t rowCapacity = AsymPlan(3U);
    const uint32_t pitch = AsymPlan(5U), inputs = AsymPlan(6U);
    pipe->InitBuffer(asymInSmall, 1U, AsymPlan(0U) * 4U);
    pipe->InitBuffer(asymInLarge, 1U, AsymPlan(1U) * 4U);
    pipe->InitBuffer(asymOutSmall, 1U, AsymPlan(2U) * 4U);
    pipe->InitBuffer(packOffsets, AsymPlan(4U) * 4U);
    pipe->InitBuffer(packDeltas, AsymPlan(4U) * 4U);
    pipe->InitBuffer(streamColumns, 64U * 4U);
    // Only the members byte marks mixed scratch, not the encoded total width.
    if ((AsymPlan(8U) >> 16U) & 0xffU) {
        pipe->InitBuffer(streamGroupWidths, 64U * 4U);
        pipe->InitBuffer(streamGroupBases, 64U * 4U);
    }
    pipe->InitBuffer(sliceEdgeOutput, 1U, (pitch + 64U) * 4U);
    auto rowBytes = packDeltas.Get<int32_t>();
    auto columnBytes = streamColumns.Get<int32_t>();
    CreateVecIndex(columnBytes, int32_t(0), 64U);
    PipeBarrier<PIPE_V>();
    Muls(columnBytes, columnBytes, int32_t(4), 64U);
    PipeBarrier<PIPE_V>();

    const uint64_t inputBypassOffset = InitStreamInputBypassOffset();
    for (uint32_t batchFirst = firstRow; batchFirst <= lastRow; batchFirst += rowCapacity) {
        sliceFirstRow = batchFirst;
        sliceLastRow = MinU32(batchFirst + rowCapacity - 1U, lastRow);
        sliceFirstCol = batchFirst == firstRow ? firstCol : 0U;
        sliceLastCol = sliceLastRow == lastRow ? lastCol : outputAxisSize;
        const uint32_t rows = sliceLastRow - sliceFirstRow + 1U;
        // Only the first/last batch clips the core's physical GM ownership.
        // Intermediate batch boundaries belong to this same core/ordered MTE3.
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 0; r < rows; ++r)
            Duplicate(rowBytes[r * 64U], static_cast<int32_t>(r * 4U), 64U);
        PipeBarrier<PIPE_V>();
        uint32_t left = 0U, bandColumn = 0U;
        uint32_t bandWidth = MinU32(pitch, outputAxisSize);
        auto dst = asymOutSmall.AllocTensor<uint32_t>();
        PrefetchStreamGroup(0U, 0U, sliceFirstRow, rows, inputBypassOffset);
        for (uint32_t n = 0U, group = 0U; n < inputs; ++group) {
            const uint32_t inputSlot = group % 2U, members = StreamMembers(n);
            const uint32_t width = StreamGroupWidth(n);
            // Independent producers precede this input's MTE2->V dequeue:
            // the alternate slot is free, and indices read only row/column
            // templates. Keep the dequeue immediately before payload Gather.
            if (n + members < inputs)
                PrefetchStreamGroup(n + members, 1U - inputSlot, sliceFirstRow, rows,
                                    inputBypassOffset);
            if (members == 1U) BuildInputAffineIndex(width, rows);
            else BuildStreamGroupIndex(n, members, 0U, width, bandColumn, bandWidth, rows);
            auto src = inputSlot == 0U ? asymInSmall.DeQue<uint32_t>() : asymInLarge.DeQue<uint32_t>();
            uint32_t consumed = 0U;
            while (consumed < width) {
                if (members > 1U && consumed)
                    BuildStreamGroupIndex(n, members, consumed, width, bandColumn, bandWidth, rows);
                uint32_t count = 0U;
                FillStreamFragment(dst, src, consumed, width - consumed,
                                    bandColumn, bandWidth, rows, pitch, count, members > 1U);
                consumed += count; bandColumn += count;
                if (bandColumn == bandWidth) {
                    // Edge compaction overwrites packOffsets before its first
                    // internal fence. Complete all payload Gather reads here;
                    // the output TQue still supplies the V->MTE3 dependency.
                    PipeBarrier<PIPE_V>();
                    asymOutSmall.EnQue(dst);
                    const bool borrowed = StoreStreamBand(0U, left, bandWidth, pitch);
                    left += bandWidth; bandColumn = 0U;
                    if (left < outputAxisSize) {
                        // AllocTensor waits for this single slot's prior MTE3.
                        dst = asymOutSmall.AllocTensor<uint32_t>();
                        bandWidth = MinU32(pitch, outputAxisSize - left);
                        if (borrowed && consumed < width && members == 1U)
                            BuildInputAffineIndex(width, rows);
                    }
                }
            }
            // Free only after all fragments using this input row slab finish.
            if (inputSlot == 0U) asymInSmall.FreeTensor(src); else asymInLarge.FreeTensor(src);
            n += members;
        }
    }
    return true;
}
