// Each core owns one disjoint, 32-byte-aligned output interval. Only the
// groups intersecting an interval boundary need a short partial-row copy.
TQue<QuePosition::VECOUT, 1> sliceEdgeOutput;
uint32_t sliceFirstRow = 0U, sliceFirstCol = 0U;
uint32_t sliceLastRow = 0U, sliceLastCol = 0U;
struct SlicePackGroup {
    BlockPackGroup pack;
    uint32_t first, rows, headSkip, tailCount;
};
__aicore__ inline SlicePackGroup NextSlicePackGroup(uint32_t group, uint32_t begin,
                                                  uint32_t axis) {
    SlicePackGroup s{};
    s.pack.begin = begin; s.pack.end = AsymPlan(8U + group); s.pack.axis = axis;
    const uint32_t right = s.pack.end == inputCount ? outputAxisSize : AxisOffset(s.pack.end);
    s.pack.width = right - axis;
    s.first = sliceFirstRow + (right <= sliceFirstCol ? 1U : 0U);
    const uint32_t last = sliceLastRow + 1U - (axis >= sliceLastCol ? 1U : 0U);
    s.rows = last - s.first;
    s.headSkip = s.first == sliceFirstRow && sliceFirstCol > axis ? sliceFirstCol - axis : 0U;
    s.tailCount = last - 1U == sliceLastRow && sliceLastCol < right ? sliceLastCol - axis : s.pack.width;
    for (uint32_t i = begin; i < s.pack.end; ++i)
        s.pack.source += AlignElems32<uint32_t>(s.rows * AxisLen(i));
    return s;
}
__aicore__ inline void StoreSliceEdge(LocalTensor<uint32_t>& packed, uint32_t sourceOffset,
                                     uint32_t outputOffset, uint32_t count) {
    if (!count) return;
    auto index = packOffsets.Get<int32_t>();
    CreateVecIndex(index, static_cast<int32_t>(sourceOffset), count);
    PipeBarrier<PIPE_V>();
    Muls(index, index, int32_t(4), count);
    PipeBarrier<PIPE_V>();
    auto edge = sliceEdgeOutput.AllocTensor<uint32_t>();
    Gather(edge, packed, index.ReinterpretCast<uint32_t>(), 0U, count);
    sliceEdgeOutput.EnQue(edge);
    auto ready = sliceEdgeOutput.DeQue<uint32_t>();
    GlobalTensor<uint32_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputAddr), totalElems);
    DataCopyPad(dst[outputOffset], ready, DataCopyExtParams{1U, count * 4U, 0U, 0U, 0U});
    sliceEdgeOutput.FreeTensor(ready);
}
__aicore__ inline void StoreSlicePack(const SlicePackGroup& s, uint32_t slot) {
    auto ready = slot == 0U ? asymOutSmall.DeQue<uint32_t>() : asymOutLarge.DeQue<uint32_t>();
    const uint32_t pitch = AlignElems32<uint32_t>(s.pack.width);
    if (s.rows == 1U) {
        StoreSliceEdge(ready, s.headSkip, s.first * outputAxisSize + s.pack.axis + s.headSkip,
                        s.tailCount - s.headSkip);
    } else {
        const uint32_t fullFirst = s.headSkip ? 1U : 0U;
        const uint32_t fullLast = s.rows - (s.tailCount < s.pack.width ? 1U : 0U);
        if (fullLast > fullFirst) {
            GlobalTensor<uint32_t> dst;
            dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputAddr), totalElems);
            DataCopyPad(dst[uint64_t(s.first + fullFirst) * outputAxisSize + s.pack.axis],
                        ready[fullFirst * pitch],
                        DataCopyExtParams{static_cast<uint16_t>(fullLast - fullFirst), s.pack.width * 4U,
                                          0U, (outputAxisSize - s.pack.width) * 4U, 0U});
        }
        if (s.headSkip) StoreSliceEdge(ready, s.headSkip,
            s.first * outputAxisSize + s.pack.axis + s.headSkip, s.pack.width - s.headSkip);
        if (s.tailCount < s.pack.width) StoreSliceEdge(ready, (s.rows - 1U) * pitch,
            (s.first + s.rows - 1U) * outputAxisSize + s.pack.axis, s.tailCount);
    }
    // Every edge Gather has completed before this source tensor is released.
    if (slot == 0U) asymOutSmall.FreeTensor(ready); else asymOutLarge.FreeTensor(ready);
}
__aicore__ inline bool TryProcessSlicePack() {
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    const uint64_t blocks = (static_cast<uint64_t>(totalElems) + 7U) / 8U;
    const uint32_t lo = MinU32(totalElems, static_cast<uint32_t>(blocks * core / cores * 8U));
    const uint32_t hi = core + 1U == cores ? totalElems : static_cast<uint32_t>(blocks * (core + 1U) / cores * 8U);
    if (lo == hi) return true;
    sliceFirstRow = lo / outputAxisSize; sliceFirstCol = lo % outputAxisSize;
    sliceLastRow = (hi - 1U) / outputAxisSize; sliceLastCol = (hi - 1U) % outputAxisSize + 1U;
    pipe->InitBuffer(asymInSmall, 1U, AsymPlan(0U) * 4U);
    pipe->InitBuffer(asymInLarge, 1U, AsymPlan(1U) * 4U);
    pipe->InitBuffer(asymOutSmall, 1U, AsymPlan(2U) * 4U);
    pipe->InitBuffer(asymOutLarge, 1U, AsymPlan(3U) * 4U);
    pipe->InitBuffer(packScratch, AsymPlan(4U) * 4U);
    pipe->InitBuffer(packOffsets, AsymPlan(4U) * 4U);
    pipe->InitBuffer(packDeltas, AsymPlan(4U) * 4U);
    pipe->InitBuffer(sliceEdgeOutput, 1U, AsymPlan(4U));
    const uint32_t groups = AsymPlan(6U);
    uint32_t slot = AsymPlan(5U), pendingSlot = 0U;
    SlicePackGroup s = NextSlicePackGroup(0U, 0U, 0U), pending{};
    AsymPrefetch(s.pack, slot, s.first, s.rows);
    for (uint32_t group = 0U; group < groups; ++group) {
        auto src = slot == 0U ? asymInSmall.DeQue<uint32_t>() : asymInLarge.DeQue<uint32_t>();
        SlicePackGroup next{};
        if (group + 1U < groups) {
            next = NextSlicePackGroup(group + 1U, s.pack.end, s.pack.axis + s.pack.width);
            AsymPrefetch(next.pack, 1U - slot, next.first, next.rows);
        }
        if (group != 0U) StoreSlicePack(pending, pendingSlot);
        auto dst = slot == 0U ? asymOutSmall.AllocTensor<uint32_t>() : asymOutLarge.AllocTensor<uint32_t>();
        ComputeElementPackInto(s.pack, s.rows, src, dst);
        if (slot == 0U) { asymOutSmall.EnQue(dst); asymInSmall.FreeTensor(src); }
        else { asymOutLarge.EnQue(dst); asymInLarge.FreeTensor(src); }
        pending = s; pendingSlot = slot; slot = 1U - slot; s = next;
    }
    StoreSlicePack(pending, pendingSlot);
    return true;
}
