// Data flow per tile: exact GM fragments -> 32B-padded source pool -> Gather
// and raw byte packing -> exact GM output. Only the metadata walk is scalar.
// Index units are source bytes. B8 index/mask lanes address destination byte pairs.
// UB lifetimes: input queue protects MTE2->V; output queue protects V->MTE3;
// PIPE_V barriers protect reuse of index, masks and logical packing scratch.
// Included inside ConcatKernel. Route-exclusive, single-slot vector fallback.
// Raw B16/B32 Gather; raw B8 uses two B16 Gathers and logical byte extraction.
static constexpr uint32_t kFallbackTileBytes = 2048U;
static constexpr uint32_t kFallbackSourceBytes = 32U * kFallbackTileBytes;
TQue<QuePosition::VECIN, 1> fallbackInput;
TQue<QuePosition::VECOUT, 1> fallbackOutput;
TBuf<TPosition::VECCALC> fallbackEvenIndex, fallbackOddIndex;
TBuf<TPosition::VECCALC> fallbackEvenMask, fallbackOddMask;
TBuf<TPosition::VECCALC> fallbackEvenWork, fallbackOddWork, fallbackColumns;

__aicore__ inline uint64_t FallbackLaneMask(uint32_t first, uint32_t last) const
{
    const uint64_t top = last == 64U ? UINT64_MAX : (1ULL << last) - 1ULL;
    const uint64_t bottom = first == 0U ? 0ULL : (1ULL << first) - 1ULL;
    return top & ~bottom;
}

__aicore__ inline void FillFallbackIndices(LocalTensor<int32_t>& index,
                                           uint32_t first, uint32_t count,
                                           uint32_t source, uint32_t step)
{
    auto columns = fallbackColumns.Get<int32_t>();
    const uint32_t end = first + count;
    for (uint32_t block = first / 64U * 64U; block < end; block += 64U) {
        const uint32_t beginLane = first > block ? first - block : 0U;
        const uint32_t endLane = MinU32(64U, end - block);
        uint64_t mask[2] = {FallbackLaneMask(beginLane, endLane), 0ULL};
        const int32_t base = static_cast<int32_t>(source) +
            (static_cast<int32_t>(block) - static_cast<int32_t>(first)) * static_cast<int32_t>(step);
        // Both tensor bases are 32B aligned, including fragments starting at
        // arbitrary lanes. Inactive indices retain the initialized guard.
        Adds(index[block], columns, base, mask, 1U, UnaryRepeatParams{1U, 1U, 8U, 8U});
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FillFallbackByteMask(LocalTensor<uint16_t>& values,
                                            uint32_t first, uint32_t count,
                                            uint16_t value)
{
    const uint32_t end = first + count;
    for (uint32_t block = first / 64U * 64U; block < end; block += 64U) {
        const uint32_t beginLane = first > block ? first - block : 0U;
        const uint32_t endLane = MinU32(64U, end - block);
        uint64_t mask[2] = {FallbackLaneMask(beginLane, endLane), 0ULL};
        Duplicate(values[block], value, mask, 1U, 1U, 8U);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void FillFallbackByteIndices(uint32_t destination,
                                               uint32_t count, uint32_t source)
{
    for (uint32_t lane = 0U; lane < 2U; ++lane) {
        const uint32_t firstByte = destination + ((lane - destination) & 1U);
        if (firstByte >= destination + count) continue;
        const uint32_t localByte = source + firstByte - destination;
        const uint32_t pairs = (destination + count - firstByte + 1U) / 2U;
        auto index = lane == 0U ? fallbackEvenIndex.Get<int32_t>() : fallbackOddIndex.Get<int32_t>();
        auto mask = lane == 0U ? fallbackEvenMask.Get<uint16_t>() : fallbackOddMask.Get<uint16_t>();
        FillFallbackIndices(index, firstByte / 2U, pairs, localByte & ~1U, 2U);
        FillFallbackByteMask(mask, firstByte / 2U, pairs,
            static_cast<uint16_t>((localByte & 1U) ? 0xff00U : 0x00ffU));
    }
}

__aicore__ inline void PackFallbackBytes(LocalTensor<uint8_t>& source,
                                        LocalTensor<uint8_t>& output, uint32_t bytes)
{
    const uint32_t pairs = (bytes + 1U) / 2U;
    auto even = fallbackEvenWork.Get<uint16_t>(), odd = fallbackOddWork.Get<uint16_t>();
    auto evenMask = fallbackEvenMask.Get<uint16_t>(), oddMask = fallbackOddMask.Get<uint16_t>();
    auto packed = output.ReinterpretCast<uint16_t>();
    auto raw = source.ReinterpretCast<uint16_t>();
    Gather(even, raw, fallbackEvenIndex.Get<uint32_t>(), 0U, pairs);
    Gather(odd, raw, fallbackOddIndex.Get<uint32_t>(), 0U, pairs);
    PipeBarrier<PIPE_V>();
    And(even, even, evenMask, pairs);
    And(odd, odd, oddMask, pairs);
    PipeBarrier<PIPE_V>();
    // Source parity masks are consumed before reusing them as constant masks.
    Duplicate(evenMask, uint16_t(0x00ffU), pairs);
    Duplicate(oddMask, uint16_t(0xff00U), pairs);
    ShiftRight(packed, even, uint16_t(8U), pairs);
    PipeBarrier<PIPE_V>();
    Or(even, even, packed, pairs);
    PipeBarrier<PIPE_V>();
    And(even, even, evenMask, pairs);
    ShiftLeft(packed, odd, uint16_t(8U), pairs);
    PipeBarrier<PIPE_V>();
    Or(odd, odd, packed, pairs);
    PipeBarrier<PIPE_V>();
    And(odd, odd, oddMask, pairs);
    PipeBarrier<PIPE_V>();
    Or(packed, even, odd, pairs);
}

template <typename T>
__aicore__ inline void ProcessVectorPackedFallback()
{
    static_assert(sizeof(T) == 1U || sizeof(T) == 2U || sizeof(T) == 4U,
                  "Raw fallback supports B8, B16 and B32");
    constexpr uint32_t fixedBytes = kFallbackSourceBytes + 32U + kFallbackTileBytes +
        8192U + 4096U + 4096U + 256U;
    static_assert(fixedBytes + 8192U + 4U * 256U * 4U <= 192U * 1024U,
                  "Fallback buffers and optional metadata cache must fit UB");
    const uint64_t bytes = totalElems64 * sizeof(T);
    const uint64_t begin = FallbackByteBoundary(bytes, GetBlockIdx(), GetBlockNum());
    const uint64_t end = FallbackByteBoundary(bytes, GetBlockIdx() + 1U, GetBlockNum());
    if (begin >= end) return;

    InitMetaCache();
    pipe->InitBuffer(fallbackInput, 1U, kFallbackSourceBytes + 32U);
    pipe->InitBuffer(fallbackOutput, 1U, kFallbackTileBytes);
    pipe->InitBuffer(fallbackEvenIndex, 4096U);
    pipe->InitBuffer(fallbackColumns, 256U);
    if constexpr (sizeof(T) == 1U) {
        pipe->InitBuffer(fallbackOddIndex, 4096U);
        pipe->InitBuffer(fallbackEvenMask, 2048U);
        pipe->InitBuffer(fallbackOddMask, 2048U);
        pipe->InitBuffer(fallbackEvenWork, 2048U);
        pipe->InitBuffer(fallbackOddWork, 2048U);
    }
    auto columns = fallbackColumns.Get<int32_t>();
    CreateVecIndex(columns, int32_t(0), 64U);
    PipeBarrier<PIPE_V>();
    Muls(columns, columns, static_cast<int32_t>(sizeof(T) == 1U ? 2U : sizeof(T)), 64U);
    PipeBarrier<PIPE_V>();
    GlobalTensor<uint8_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(outputAddr), bytes);

    for (uint64_t tile = begin; tile < end;) {
        const uint32_t tileBytes = end - tile < kFallbackTileBytes
            ? static_cast<uint32_t>(end - tile) : kFallbackTileBytes;
        const uint32_t lanes = sizeof(T) == 1U ? (tileBytes + 1U) / 2U : tileBytes / sizeof(T);
        const uint32_t roundedLanes = AlignUpU32(lanes, 64U);
        // Retire previous Gather/index/mask reads before reinitializing them.
        PipeBarrier<PIPE_V>();
        auto index = fallbackEvenIndex.Get<int32_t>();
        Duplicate(index, static_cast<int32_t>(kFallbackSourceBytes), roundedLanes);
        if constexpr (sizeof(T) == 1U) {
            Duplicate(fallbackOddIndex.Get<int32_t>(), static_cast<int32_t>(kFallbackSourceBytes), roundedLanes);
            Duplicate(fallbackEvenMask.Get<uint16_t>(), uint16_t(0), roundedLanes);
            Duplicate(fallbackOddMask.Get<uint16_t>(), uint16_t(0), roundedLanes);
        }
        auto staged = fallbackInput.AllocTensor<uint8_t>();
        auto guard = staged.template ReinterpretCast<uint16_t>();
        Duplicate(guard[kFallbackSourceBytes / 2U], uint16_t(0), 16U);
        PipeBarrier<PIPE_V>();
        uint32_t destination = 0U, source = 0U;
        while (destination < tileBytes) {
            const auto segment = LocateSegment64((tile + destination) / sizeof(T));
            const uint64_t available = static_cast<uint64_t>(tileBytes - destination) / sizeof(T);
            const uint32_t count = static_cast<uint32_t>(
                segment.count < available ? segment.count : available) * sizeof(T);
            const uint32_t padded = AlignUpU32(count, 32U);
            GlobalTensor<uint8_t> src;
            const uint64_t inputBytes = outerSize64 * segment.axisLen * innerSize64 * sizeof(T);
            src.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(InputPtr(segment.inputIdx)), inputBytes);
            // Exact GM length; explicitly initialized UB padding. At most
            // tileBytes nonempty fragments, each consuming <=32*count bytes.
            DataCopyPad(staged[source], src[segment.inputOffset * sizeof(T)],
                DataCopyExtParams{1U, count, 0U, 0U, 0U},
                DataCopyPadExtParams<uint8_t>{true, 0U, static_cast<uint8_t>(padded - count), 0U});
            if constexpr (sizeof(T) == 1U)
                FillFallbackByteIndices(destination, count, source);
            else
                FillFallbackIndices(index, destination / sizeof(T), count / sizeof(T), source, sizeof(T));
            source += padded;
            destination += count;
        }
        fallbackInput.EnQue(staged);
        auto ready = fallbackInput.DeQue<uint8_t>(); // MTE2 -> V
        auto output = fallbackOutput.AllocTensor<uint8_t>(); // retire prior MTE3
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == 1U) {
            PackFallbackBytes(ready, output, tileBytes);
        } else {
            auto raw = ready.template ReinterpretCast<T>();
            auto packed = output.template ReinterpretCast<T>();
            Gather(packed, raw, index.template ReinterpretCast<uint32_t>(), 0U, lanes);
        }
        fallbackOutput.EnQue(output);
        auto complete = fallbackOutput.DeQue<uint8_t>(); // V -> MTE3
        // Source is read only by V, never by output DMA; release after Gather.
        fallbackInput.FreeTensor(ready);
        DataCopyPad(dst[tile], complete, DataCopyExtParams{1U, tileBytes, 0U, 0U, 0U});
        fallbackOutput.FreeTensor(complete);
        tile += tileBytes;
    }
}
