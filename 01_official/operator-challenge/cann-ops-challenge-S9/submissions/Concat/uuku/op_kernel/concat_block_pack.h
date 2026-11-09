// Included inside ConcatKernel. All allocations and route guards precede stores.
struct BlockPackGroup {
    uint32_t begin, end, axis, width, source;
};
TQue<QuePosition::VECIN, 2> packInput;
TQue<QuePosition::VECOUT, 2> packOutput;
TBuf<TPosition::VECCALC> packScratch;
TBuf<TPosition::VECCALC> packOffsets;
TBuf<TPosition::VECCALC> packDeltas;
TBuf<TPosition::VECCALC> packMasks;

#include "concat_row_phase_prefix.h"

__aicore__ inline BlockPackGroup NextBlockPackGroup(uint32_t cursor,
                                                   uint32_t axis, uint32_t rows)
{
    BlockPackGroup g{cursor, cursor, axis, 0U, 0U};
    while (g.end < inputCount) {
        const uint32_t w = AxisLen(g.end);
        if (g.width + w > 192U) break;
        g.width += w;
        g.source += AlignElems32<uint32_t>(rows * w);
        ++g.end;
    }
    return g;
}

__aicore__ inline void PrefetchBlockPack(const BlockPackGroup& g,
                                        uint32_t firstRow, uint32_t rows)
{
    auto slab = packInput.AllocTensor<uint32_t>();
    uint32_t base = 0U;
    for (uint32_t i = g.begin; i < g.end; ++i) {
        const uint32_t w = AxisLen(i);
        if (w == 0U) continue;
        GlobalTensor<uint32_t> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(InputPtr(i)),
                            static_cast<uint64_t>(outerSize) * w);
        src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyPad(slab[base], src[static_cast<uint64_t>(firstRow) * w],
                    DataCopyExtParams{1U, rows * w * 4U, 0U, 0U, 0U},
                    DataCopyPadExtParams<uint32_t>{false, 0U, 0U, 0U});
        base += AlignElems32<uint32_t>(rows * w);
    }
    // Each DMA writes a disjoint aligned allocation including its rounded tail.
    packInput.EnQue(slab);
}

__aicore__ inline void StoreBlockPack(const BlockPackGroup& g,
                                     uint32_t firstRow, uint32_t rows)
{
    GlobalTensor<uint32_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputAddr), totalElems);
    auto ready = packOutput.DeQue<uint32_t>();
    DataCopyPad(dst[static_cast<uint64_t>(firstRow) * outputAxisSize + g.axis], ready,
                DataCopyExtParams{static_cast<uint16_t>(rows), g.width * 4U,
                                  0U, (outputAxisSize - g.width) * 4U, 0U});
    packOutput.FreeTensor(ready);
}

__aicore__ inline uint64_t PackColumnMask(uint32_t lo, uint32_t hi)
{
    const uint64_t top = hi == 64U ? ~0ULL : (1ULL << hi) - 1ULL;
    return top & ~((1ULL << lo) - 1ULL);
}

__aicore__ inline void ComputeElementPackInto(const BlockPackGroup& g, uint32_t rows,
                                         LocalTensor<uint32_t>& src,
                                         LocalTensor<uint32_t>& dst)
{
    auto indices = packOffsets.Get<uint32_t>();
    auto deltas = packDeltas.Get<uint32_t>();
    auto work = packScratch.Get<int32_t>();
    auto signedIndex = indices.ReinterpretCast<int32_t>();
    auto rowIds = dst.ReinterpretCast<int32_t>();
    const uint32_t pitch = AlignElems32<uint32_t>(g.width);
    const uint32_t nr = MinU32(4U, rows);
    const uint32_t count = nr * pitch;
    const uint8_t stride = static_cast<uint8_t>(pitch / 8U);
    const UnaryRepeatParams unary{1U, 1U, stride, stride};
    const BinaryRepeatParams binary{1U, 1U, 1U, stride, stride, stride};
    for (uint32_t r = 0; r < nr; ++r)
        Duplicate(rowIds[r * pitch], static_cast<int32_t>(r), pitch);
    CreateVecIndex(signedIndex, int32_t(0), count);
    Duplicate(deltas, uint32_t(0), count);
    Duplicate(src[g.source], uint32_t(0), 8U);
    PipeBarrier<PIPE_V>();
    uint32_t base = 0U, col = 0U;
    for (uint32_t i = g.begin; i < g.end; ++i) {
        const uint32_t w = AxisLen(i);
        if (!w) continue;
        const uint32_t stop = col + w;
        for (uint32_t part = col / 64U * 64U; part < stop; part += 64U) {
            // A scalar uint64_t selects the CONTIGUOUS-count overload, not a
            // bit mask. B32 bitwise mode requires {low64, high64=0}.
            uint64_t mask[2] = {PackColumnMask(col > part ? col - part : 0U,
                                              MinU32(64U, stop - part)), 0ULL};
            static_assert(sizeof(mask) == 2U * sizeof(uint64_t), "bitwise mask must be an array");
            Muls(work[part], rowIds[part], static_cast<int32_t>(w) - static_cast<int32_t>(pitch),
                 mask, static_cast<uint8_t>(nr), unary);
            PipeBarrier<PIPE_V>();
            Adds(work[part], work[part], static_cast<int32_t>(base) - static_cast<int32_t>(col),
                 mask, static_cast<uint8_t>(nr), unary);
            PipeBarrier<PIPE_V>();
            Add(signedIndex[part], signedIndex[part], work[part],
                mask, static_cast<uint8_t>(nr), binary);
            Duplicate(deltas[part], 16U * w, mask, static_cast<uint8_t>(nr), 1U, stride);
            PipeBarrier<PIPE_V>();
        }
        col = stop;
        base += AlignElems32<uint32_t>(rows * w);
    }
    Muls(signedIndex, signedIndex, int32_t(4), count);
    PipeBarrier<PIPE_V>();
    if (g.width != pitch) {
        const uint32_t part = g.width / 64U * 64U;
        uint64_t mask[2] = {PackColumnMask(g.width - part, pitch - part), 0ULL};
        static_assert(sizeof(mask) == 2U * sizeof(uint64_t), "bitwise mask must be an array");
        Duplicate(indices[part], g.source * 4U, mask, static_cast<uint8_t>(nr), 1U, stride);
        PipeBarrier<PIPE_V>();
    }
    // rowIds aliases dst only during template creation. Gather starts after
    // every use of rowIds and work has completed on PIPE_V.
    for (uint32_t r = 0; r < rows; r += 4U) {
        Gather(dst[r * pitch], src, indices, 0U, MinU32(4U, rows - r) * pitch);
        PipeBarrier<PIPE_V>();
        if (r + 4U < rows) {
            Add(signedIndex, signedIndex, deltas.ReinterpretCast<int32_t>(), count);
            PipeBarrier<PIPE_V>();
        }
    }
}

__aicore__ inline void ComputeElementPack(const BlockPackGroup& g, uint32_t rows,
                                         LocalTensor<uint32_t>& src)
{
    auto dst = packOutput.AllocTensor<uint32_t>();
    ComputeElementPackInto(g, rows, src, dst);
    packOutput.EnQue(dst);
}

__aicore__ inline bool TryProcessBlockPack()
{
    if (elemBytes != 4U || routeKey != 3U) return false;
    const uint32_t cores = GetBlockNum();
    const uint32_t core = GetBlockIdx();
    // Every inter-core row boundary is a multiple of 8: all B32 GM output
    // boundaries therefore align to 32 bytes even when the row width does not.
    const uint32_t first = static_cast<uint64_t>(outerSize) * core / cores / 8U * 8U;
    const uint32_t last = core + 1U == cores ? outerSize :
        static_cast<uint64_t>(outerSize) * (core + 1U) / cores / 8U * 8U;
    if (first == last) return true;
    const uint32_t rows = last - first;
    pipe->InitBuffer(packInput, 2U, phaseBoundaryCount * 4U);
    pipe->InitBuffer(packOutput, 2U, phaseUnitCount * 4U);
    pipe->InitBuffer(packScratch, phaseOwnerWordCount * 4U);
    pipe->InitBuffer(packOffsets, phaseOwnerWordCount * 4U);
    pipe->InitBuffer(packDeltas, phaseOwnerWordCount * 4U);
    InitMetaCache();
    BlockPackGroup g = NextBlockPackGroup(0U, 0U, rows);
    PrefetchBlockPack(g, first, rows);
    BlockPackGroup pending{};
    bool hasPending = false;
    while (g.width != 0U) {
        auto src = packInput.DeQue<uint32_t>();
        const BlockPackGroup next = NextBlockPackGroup(g.end, g.axis + g.width, rows);
        if (next.width != 0U) PrefetchBlockPack(next, first, rows);
        if (hasPending) StoreBlockPack(pending, first, rows);
        ComputeElementPack(g, rows, src);
        packInput.FreeTensor(src);
        pending = g;
        hasPending = true;
        g = next;
    }
    if (hasPending) StoreBlockPack(pending, first, rows);
    return true;
}
