// Unequal ping/pong allocations: each bank retains its own queue dependencies.
TQue<QuePosition::VECIN, 1> asymInSmall, asymInLarge;
TQue<QuePosition::VECOUT, 1> asymOutSmall, asymOutLarge;

__aicore__ inline uint32_t AsymPlan(uint32_t word) {
    return metaLocal.GetValue(4U * metaStride + word);
}
__aicore__ inline BlockPackGroup AsymGroup(uint32_t group, uint32_t begin,
                                          uint32_t axis, uint32_t rows) {
    BlockPackGroup g{begin, AsymPlan(8U + group), axis, 0U, 0U};
    for (uint32_t i = begin; i < g.end; ++i) {
        const uint32_t w = AxisLen(i);
        g.width += w;
        g.source += AlignElems32<uint32_t>(rows * w);
    }
    return g;
}
__aicore__ inline void AsymPrefetch(const BlockPackGroup& g, uint32_t slot,
                                   uint32_t first, uint32_t rows) {
    auto slab = slot == 0U ? asymInSmall.AllocTensor<uint32_t>() :
                             asymInLarge.AllocTensor<uint32_t>();
    uint32_t base = 0U;
    for (uint32_t i = g.begin; i < g.end; ++i) {
        const uint32_t w = AxisLen(i);
        if (!w) continue;
        GlobalTensor<uint32_t> src;
        src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(InputPtr(i)),
                            static_cast<uint64_t>(outerSize) * w);
        src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        DataCopyPad(slab[base], src[static_cast<uint64_t>(first) * w],
                    DataCopyExtParams{1U, rows * w * 4U, 0U, 0U, 0U},
                    DataCopyPadExtParams<uint32_t>{false, 0U, 0U, 0U});
        base += AlignElems32<uint32_t>(rows * w);
    }
    if (slot == 0U) asymInSmall.EnQue(slab); else asymInLarge.EnQue(slab);
}
__aicore__ inline void AsymStore(const BlockPackGroup& g, uint32_t slot,
                                uint32_t first, uint32_t rows) {
    auto ready = slot == 0U ? asymOutSmall.DeQue<uint32_t>() :
                              asymOutLarge.DeQue<uint32_t>();
    GlobalTensor<uint32_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(outputAddr), totalElems);
    DataCopyPad(dst[static_cast<uint64_t>(first) * outputAxisSize + g.axis], ready,
                DataCopyExtParams{static_cast<uint16_t>(rows), g.width * 4U,
                                  0U, (outputAxisSize - g.width) * 4U, 0U});
    if (slot == 0U) asymOutSmall.FreeTensor(ready); else asymOutLarge.FreeTensor(ready);
}
__aicore__ inline bool TryProcessAsymPack() {
    if (elemBytes != 4U || routeKey != 5U) return false;
    InitMetaCache();
    if (AsymPlan(7U) == 3U) return TryProcessInputStream();
    if (AsymPlan(7U) == 1U) return TryProcessSlicePack();
    const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
    const uint32_t first = static_cast<uint64_t>(outerSize) * core / cores / 8U * 8U;
    const uint32_t last = core + 1U == cores ? outerSize :
        static_cast<uint64_t>(outerSize) * (core + 1U) / cores / 8U * 8U;
    if (first == last) return true;
    const uint32_t rows = last - first;
    pipe->InitBuffer(asymInSmall, 1U, AsymPlan(0U) * 4U);
    pipe->InitBuffer(asymInLarge, 1U, AsymPlan(1U) * 4U);
    pipe->InitBuffer(asymOutSmall, 1U, AsymPlan(2U) * 4U);
    pipe->InitBuffer(asymOutLarge, 1U, AsymPlan(3U) * 4U);
    pipe->InitBuffer(packScratch, AsymPlan(4U) * 4U);
    pipe->InitBuffer(packOffsets, AsymPlan(4U) * 4U);
    pipe->InitBuffer(packDeltas, AsymPlan(4U) * 4U);
    const uint32_t groups = AsymPlan(6U);
    uint32_t slot = AsymPlan(5U);
    BlockPackGroup g = AsymGroup(0U, 0U, 0U, rows), pending{};
    AsymPrefetch(g, slot, first, rows);
    uint32_t pendingSlot = 0U;
    for (uint32_t group = 0U; group < groups; ++group) {
        auto src = slot == 0U ? asymInSmall.DeQue<uint32_t>() : asymInLarge.DeQue<uint32_t>();
        BlockPackGroup next{};
        if (group + 1U < groups) {
            next = AsymGroup(group + 1U, g.end, g.axis + g.width, rows);
            AsymPrefetch(next, 1U - slot, first, rows);
        }
        if (group != 0U) AsymStore(pending, pendingSlot, first, rows);
        auto dst = slot == 0U ? asymOutSmall.AllocTensor<uint32_t>() : asymOutLarge.AllocTensor<uint32_t>();
        ComputeElementPackInto(g, rows, src, dst);
        if (slot == 0U) {
            asymOutSmall.EnQue(dst); asymInSmall.FreeTensor(src);
        } else {
            asymOutLarge.EnQue(dst); asymInLarge.FreeTensor(src);
        }
        pending = g; pendingSlot = slot;
        slot = 1U - slot; g = next;
    }
    AsymStore(pending, pendingSlot, first, rows);
    return true;
}
