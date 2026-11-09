// Included inside ConcatKernel. Route 4 reuses the route-exclusive pack buffers.
struct RowPhasePrefixSpan { uint32_t first, last; };

__aicore__ inline RowPhasePrefixSpan RowPhaseSpan(
    uint32_t row, uint32_t column, uint32_t width, uint32_t rowWidth,
    uint32_t lo, uint32_t hi) const
{
    const uint64_t base = static_cast<uint64_t>(row) * rowWidth + column;
    const uint32_t first = lo > base ? MinU32(width, static_cast<uint32_t>(lo - base)) : 0U;
    const uint32_t last = hi > base ? MinU32(width, static_cast<uint32_t>(hi - base)) : 0U;
    return {first, last};
}

__aicore__ inline void RowPhaseClippedStore(
    GlobalTensor<uint32_t>& dst, const LocalTensor<uint32_t>& ready,
    const LocalTensor<uint32_t>& edge, uint32_t sourceRow, uint32_t outputRow,
    uint32_t column, uint32_t width, uint32_t rowWidth, RowPhasePrefixSpan span)
{
    const uint32_t start = sourceRow * width + span.first;
    const uint32_t length = span.last - span.first;
    const uint32_t prefix = MinU32((0U - start) & 7U, length);
    const uint64_t output = static_cast<uint64_t>(outputRow) * rowWidth + column + span.first;
    if (length > prefix)
        DataCopyPad(dst[output + prefix], ready[start + prefix],
                    DataCopyExtParams{1U, (length - prefix) * 4U, 0U, 0U, 0U});
    if (prefix != 0U)
        DataCopyPad(dst[output], edge[sourceRow * 8U],
                    DataCopyExtParams{1U, prefix * 4U, 0U, 0U, 0U});
}

__aicore__ inline bool TryProcessSlabRows()
{
    if (elemBytes != 4U || routeKey != 4U) return false;
    const uint32_t capacity = phaseBoundaryCount, indexWords = phaseUnitCount;
    const uint32_t metaReserve = 4U * ((inputCount + 7U) / 8U * 8U) * 4U;
    if (GetBlockNum() != 40U || capacity < 8192U || capacity > 44992U ||
        capacity % 8U != 0U || indexWords < 128U || indexWords > 768U ||
        indexWords % 64U != 0U ||
        4ULL * (static_cast<uint64_t>(capacity) + 64U) + 8ULL * indexWords +
            256U + metaReserve + 8192U > ubBytes)
        return false;
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    const uint64_t blocks = (static_cast<uint64_t>(totalElems) + 7U) / 8U;
    const uint32_t lo = static_cast<uint32_t>(blocks * core / cores * 8U);
    const uint32_t hi = core + 1U == cores ? totalElems :
        static_cast<uint32_t>(blocks * (core + 1U) / cores * 8U);
    if (lo >= hi) return true;
    const uint32_t rowWidth = outputAxisSize * innerSize;
    const uint32_t firstRow = lo / rowWidth, firstColumn = lo % rowWidth;
    const uint32_t lastRow = (hi - 1U) / rowWidth, lastColumn = (hi - 1U) % rowWidth + 1U;
    pipe->InitBuffer(packInput, 1U, (capacity + 64U) * 4U);
    pipe->InitBuffer(packOutput, 1U, indexWords * 4U);
    pipe->InitBuffer(packOffsets, indexWords * 4U);
    pipe->InitBuffer(packDeltas, 64U * 4U);
    // Nonzero-depth queues allocate their free IDs dynamically. Keep this
    // bridge ID reserved, distinct from the following output FreeTensor flag.
    const event_t done = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_V>());
    auto index = packOffsets.Get<int32_t>();
    auto columns = packDeltas.Get<int32_t>();
    CreateVecIndex(columns, int32_t(0), 64U);
    PipeBarrier<PIPE_V>();
    Muls(columns, columns, int32_t(4), 64U);
    PipeBarrier<PIPE_V>();
    GlobalTensor<uint32_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputAddr), totalElems);
    for (uint32_t i = 0U; i < inputCount; ++i) {
        const uint32_t width = AxisLen(i) * innerSize;
        if (width == 0U) continue;
        const uint32_t column = AxisOffset(i) * innerSize;
        const uint32_t begin = firstRow + (column + width <= firstColumn ? 1U : 0U);
        const uint32_t end = lastRow + 1U - (column >= lastColumn ? 1U : 0U);
        if (begin >= end) continue;
        const uint32_t batch = MinU32(MinU32(capacity / width, (indexWords - 64U) / 8U), end - begin);
        const uint32_t shift = (width & 1U) ? 3U : (width & 2U) ? 2U : (width & 4U) ? 1U : 0U;
        const uint32_t period = 1U << shift;
        GlobalTensor<uint32_t> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(InputPtr(i)),
                            static_cast<uint64_t>(outerSize) * width);
        src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        for (uint32_t row = begin; row < end; row += batch) {
            const uint32_t nr = MinU32(batch, end - row);
            // Reuse only after the prior slab's two readers retired. This load
            // may overlap index preparation, but not prior-slab Gather/MTE3.
            auto input = packInput.AllocTensor<uint32_t>();
            DataCopyPad(input, src[static_cast<uint64_t>(row) * width],
                        DataCopyExtParams{1U, nr * width * 4U, 0U, 0U, 0U},
                        DataCopyPadExtParams<uint32_t>{false, 0U, 0U, 0U});
            packInput.EnQue(input);
            // Shared indices cannot be rewritten before the prior Gather reads.
            PipeBarrier<PIPE_V>();
            Duplicate(index, int32_t(0), indexWords);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0U; r < nr; ++r) {
                const auto span = RowPhaseSpan(row + r, column, width, rowWidth, lo, hi);
                const uint32_t start = r * width + span.first;
                const uint32_t prefix = MinU32((0U - start) & 7U, span.last - span.first);
                // Full rows all stage eight words so their prefixes share one
                // DMA. The body-overlapping words are bit-identical, same-core.
                const uint32_t firstWords = span.first == 0U && span.last == width ? 8U : prefix;
                if (firstWords != 0U)
                    Duplicate(index[r * 8U], static_cast<int32_t>(start * 4U), firstWords);
            }
            PipeBarrier<PIPE_V>();
            Add(index, index, columns, uint64_t(8), static_cast<uint8_t>(nr),
                BinaryRepeatParams{1U, 1U, 1U, 1U, 1U, 0U});
            PipeBarrier<PIPE_V>();
            auto ready = packInput.DeQue<uint32_t>(); // MTE2 -> Vector
            auto edge = packOutput.AllocTensor<uint32_t>();
            // Inactive lanes use source words [0,8), never uninitialized padding.
            Gather(edge, ready, index.ReinterpretCast<uint32_t>(), 0U, nr * 8U);
            packOutput.EnQue(edge);
            auto result = packOutput.DeQue<uint32_t>(); // Vector -> MTE3 (both input readers)
            const auto head = RowPhaseSpan(row, column, width, rowWidth, lo, hi);
            const auto tail = RowPhaseSpan(row + nr - 1U, column, width, rowWidth, lo, hi);
            const bool clippedHead = head.first != 0U || head.last != width;
            const bool clippedTail = tail.first != 0U || tail.last != width;
            const uint32_t fullFirst = clippedHead ? 1U : 0U;
            const uint32_t fullEnd = nr - (clippedTail ? 1U : 0U);
            for (uint32_t phase = 0U; phase < period; ++phase) {
                const uint32_t r = fullFirst + ((phase - fullFirst) & (period - 1U));
                if (r >= fullEnd) continue;
                const uint32_t count = 1U + ((fullEnd - 1U - r) >> shift);
                const uint32_t prefix = (0U - r * width) & 7U;
                const uint32_t length = width - prefix;
                const uint64_t output = static_cast<uint64_t>(row + r) * rowWidth + column;
                DataCopyPad(dst[output + prefix], ready[r * width + prefix],
                            DataCopyExtParams{static_cast<uint16_t>(count), length * 4U,
                                (period * width - AlignElems32<uint32_t>(length)) / 8U,
                                (period * rowWidth - length) * 4U, 0U});
            }
            // Unlike the phase body, all edge rows have a fixed 32B UB pitch.
            // Each overlapped body word has the same source bits; physical core
            // clipping keeps both stores inside this core's exclusive interval.
            if (fullFirst < fullEnd)
                DataCopyPad(dst[static_cast<uint64_t>(row + fullFirst) * rowWidth + column],
                            result[fullFirst * 8U],
                            DataCopyExtParams{static_cast<uint16_t>(fullEnd - fullFirst),
                                32U, 0U, (rowWidth - 8U) * 4U, 0U});
            if (clippedHead)
                RowPhaseClippedStore(dst, ready, result, 0U, row, column, width, rowWidth, head);
            if (nr > 1U && clippedTail)
                RowPhaseClippedStore(dst, ready, result, nr - 1U, row + nr - 1U,
                                     column, width, rowWidth, tail);
            // packInput.FreeTensor only supplies V -> MTE2. Its raw payload was
            // also read directly by MTE3, so bridge that retirement first.
            SetFlag<HardEvent::MTE3_V>(done);
            WaitFlag<HardEvent::MTE3_V>(done);
            packInput.FreeTensor(ready);
            packOutput.FreeTensor(result);
        }
    }
    pipe->ReleaseEventID<HardEvent::MTE3_V>(static_cast<TEventID>(done));
    return true;
}
