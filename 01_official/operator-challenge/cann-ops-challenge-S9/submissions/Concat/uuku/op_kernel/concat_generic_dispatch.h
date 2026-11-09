// Read this file as admission first, execution second. All fallback buffers
// are route-exclusive. Large contiguous fragments keep the linear DMA path;
// fragmented layouts use the terminal vector packer. Width thresholds are
// transfer-size policies, never comparisons against benchmark shape tuples.
// Included inside ConcatKernel. All admission checks precede UB allocation
// and output writes; a route cannot partially write and then change ownership.
__aicore__ inline uint64_t FallbackByteBoundary(uint64_t bytes, uint32_t core,
                                               uint32_t cores) const
{
    const uint64_t blocks = bytes / 32U + (bytes % 32U != 0U ? 1U : 0U);
    // Quotient/remainder form avoids overflowing blocks * core.
    const uint64_t owned = blocks / cores * core + blocks % cores * core / cores;
    return owned > bytes / 32U ? bytes : owned * 32U;
}

__aicore__ inline SegmentInfo LocateSegment64(uint64_t outputOffset) const
{
    const uint64_t rowWidth = static_cast<uint64_t>(outputAxisSize) * innerSize64;
    const uint64_t row = outputOffset / rowWidth;
    const uint64_t column = outputOffset % rowWidth;
    const uint32_t axis = static_cast<uint32_t>(column / innerSize64);
    uint32_t first = 0U, last = inputCount;
    while (first < last) {
        const uint32_t mid = first + (last - first) / 2U;
        const uint32_t end = AxisOffset(mid) + AxisLen(mid);
        if (axis >= end)
            first = mid + 1U;
        else
            last = mid;
    }
    // Validated host prefixes cover every nonempty output position. Searching
    // for the first prefix end above axis also skips arbitrary zero-width runs.
    const uint32_t width = AxisLen(first);
    const uint64_t inputWidth = static_cast<uint64_t>(width) * innerSize64;
    const uint64_t begin = static_cast<uint64_t>(AxisOffset(first)) * innerSize64;
    return {first, width, row * inputWidth + column - begin,
            begin + inputWidth - column};
}

template <typename T>
__aicore__ inline bool CanProcessRows(uint32_t localElems) const
{
    const uint32_t cores = GetBlockNum();
    if (wideGeometry || cores == 0U || outerSize < cores ||
        innerSize == 0U || outputAxisSize == 0U || localElems == 0U) return false;
    const uint64_t rowElems = static_cast<uint64_t>(outputAxisSize) * innerSize;
    const uint64_t rowBytes = rowElems * sizeof(T);
    if (rowElems > UINT32_MAX || rowBytes > UINT32_MAX) return false;
    const uint32_t rows = outerSize / cores + (outerSize % cores != 0U ? 1U : 0U);
    // Row shards may touch only at physical 32B boundaries. Individual input
    // pieces inside one shard retain the parent's ordered MTE3 writes.
    if (outerSize > rows && (static_cast<uint64_t>(rows) * rowBytes) % 32U != 0U)
        return false;
    for (uint32_t i = 0U; i < inputCount; ++i) {
        const uint64_t width = static_cast<uint64_t>(AxisLen(i)) * innerSize;
        if (width == 0U) continue;
        if (width > localElems || AlignElems32<T>(static_cast<uint32_t>(width)) > localElems)
            return false;
    }
    return true;
}

template <typename T>
__aicore__ inline bool CanProcessAlignedGroups(uint32_t localElems) const
{
    if (sizeof(T) != 4U || wideGeometry || innerSize != 1U ||
        outerSize == 0U || outerSize > 4095U || outputAxisSize % 8U != 0U ||
        static_cast<uint64_t>(outputAxisSize) * sizeof(T) > UINT32_MAX ||
        GetBlockNum() == 0U || alignedGroupCount < GetBlockNum()) return false;
    for (uint32_t i = 0U; i < inputCount; ++i) {
        const uint32_t width = AxisLen(i);
        if (width == 0U || width > localElems ||
            static_cast<uint64_t>(AlignElems32<T>(width)) * outerSize > localElems)
            return false;
    }
    return true;
}

template <typename T>
__aicore__ inline bool CanProcessWindows(uint32_t localElems) const
{
    return sizeof(T) == 4U && !wideGeometry && IsManyInputLastAxisB32() &&
           outputAxisSize % 8U == 0U && outputAxisSize <= UINT32_MAX / 4U &&
           localElems >= 128U && GetBlockNum() != 0U &&
           CeilDivU32(outputAxisSize, 128U) >= GetBlockNum();
}

template <typename T>
__aicore__ inline void ProcessLinearDma64(uint32_t singleInput, uint32_t anchorInput)
{
    const uint64_t bytes = totalElems64 * sizeof(T);
    const uint64_t lo = FallbackByteBoundary(bytes, GetBlockIdx(), GetBlockNum()) / sizeof(T);
    const uint64_t hi = FallbackByteBoundary(bytes, GetBlockIdx() + 1U, GetBlockNum()) / sizeof(T);
    if (lo >= hi) return;
    pipe->InitBuffer(copyQueue, kBufferNum, AlignElems32<T>(tileElems) * sizeof(T));
    InitMetaCache();
    const uint64_t bypass = InitLinearInputBypassOffset<T>(anchorInput);
    GlobalTensor<T> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(outputAddr), totalElems64);
    const uint32_t singleWidth = singleInput < inputCount ? AxisLen(singleInput) : 0U;
    for (uint64_t progress = lo; progress < hi;) {
        const SegmentInfo part = singleInput < inputCount
            ? SegmentInfo{singleInput, singleWidth, progress, hi - progress}
            : LocateSegment64(progress);
        const uint64_t remaining = part.count < hi - progress ? part.count : hi - progress;
        const uint32_t count = remaining < tileElems ? static_cast<uint32_t>(remaining) : tileElems;
        CopyOneSegment<T>(part.inputIdx, part.axisLen, part.inputOffset, progress, count, dst, bypass);
        progress += count;
    }
}

template <typename T>
__aicore__ inline void ProcessGenericFallback(GlobalTensor<T>& dst)
{
    const uint32_t localElems = AlignElems32<T>(tileElems);
    const bool grouped = CanProcessAlignedGroups<T>(localElems);
    const bool windowed = !grouped && CanProcessWindows<T>(localElems);
    const bool rows = !grouped && !windowed && CanProcessRows<T>(localElems);
    if (grouped || windowed || rows) {
        pipe->InitBuffer(copyQueue, kBufferNum, localElems * sizeof(T));
        InitMetaCache();
        if (grouped) {
            if (!TryProcessCoalescedAlignedGroups<T>(localElems, dst))
                TryProcessAlignedInputGroups<T>(localElems, dst);
        } else if (windowed) {
            TryProcessWindowOwnedRows<T>(localElems, dst);
        } else {
            ProcessRows<T>(localElems, dst);
        }
        return;
    }
    // Host already inspected every descriptor during shape validation. This
    // one word replaces a full per-core scan; earlier route priority is intact.
    __gm__ uint32_t* td = reinterpret_cast<__gm__ uint32_t*>(tilingAddr);
    const uint32_t linearPlan = td[kTilingLinearPlanOffset];
    if ((linearPlan & concat_linear_route::kBulkDmaFlag) != 0U) {
        const uint32_t anchor = linearPlan & concat_linear_route::kInputIndexMask;
        const uint32_t single = (linearPlan & concat_linear_route::kSingleInputFlag) != 0U
            ? anchor : inputCount;
        if (wideGeometry) ProcessLinearDma64<T>(single, anchor);
        else ProcessLinearDma32<T>(single, anchor, dst);
        return;
    }
    ProcessVectorPackedFallback<T>();
}
