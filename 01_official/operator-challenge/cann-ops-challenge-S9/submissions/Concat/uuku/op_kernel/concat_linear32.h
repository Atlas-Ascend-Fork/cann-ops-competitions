// Lightweight linear DMA for validated U32 geometry. Logical positions and
// counts stay U32; GM addresses remain U64. Included inside ConcatKernel.
struct LinearSegment32 {
    uint32_t inputIdx, axisLen;
    uint64_t inputOffset;
    uint32_t count;
};

__aicore__ inline LinearSegment32 LocateSegment32(uint32_t outputOffset,
                                                 uint32_t startInputIdx)
{
    LinearSegment32 info{0U, 0U, 0U, 0U};
    const uint32_t rowStride = outputAxisSize * innerSize;
    const uint32_t row = outputOffset / rowStride;
    const uint32_t remInRow = outputOffset - row * rowStride;
    const uint32_t axis = remInRow / innerSize;
    const uint32_t inner = remInRow - axis * innerSize;
    for (uint32_t i = startInputIdx; i < inputCount; ++i) {
        const uint32_t axisStart = AxisOffset(i);
        const uint32_t axisLen = AxisLen(i);
        const uint32_t axisEnd = axisStart + axisLen;
        if (axis >= axisStart && axis < axisEnd) {
            const uint32_t localAxis = axis - axisStart;
            const uint32_t inputRowElems = axisLen * innerSize;
            info.inputIdx = i;
            info.axisLen = axisLen;
            info.inputOffset = static_cast<uint64_t>(row) * inputRowElems +
                static_cast<uint64_t>(localAxis) * innerSize + inner;
            info.count = axisEnd * innerSize - remInRow;
            return info;
        }
    }
    return info;
}

template <typename T>
__aicore__ inline void ProcessLinearDma32(uint32_t singleInput, uint32_t anchorInput,
                                        GlobalTensor<T>& dst)
{
    constexpr uint32_t blockElems = concat_linear_route::kWorkBlockBytes / sizeof(T);
    const uint32_t cores = GetBlockNum();
    if (cores == 0U) return;
    // Quotient/remainder avoids overflow when totalElems approaches UINT32_MAX.
    const uint32_t blocks = totalElems / blockElems + (totalElems % blockElems != 0U);
    const uint32_t blocksPerCore = CeilDivU32(blocks, cores);
    const uint32_t firstBlock = GetBlockIdx() * blocksPerCore;
    if (firstBlock >= blocks) return;
    const uint32_t lastBlock = MinU32(firstBlock + blocksPerCore, blocks);
    const uint32_t begin = firstBlock * blockElems;
    // The rounded final block may end at 2^32; clamp before multiplying it.
    const uint32_t end = lastBlock == blocks ? totalElems : lastBlock * blockElems;
    pipe->InitBuffer(copyQueue, kBufferNum, AlignElems32<T>(tileElems) * sizeof(T));
    InitMetaCache();
    const uint64_t bypass = InitLinearInputBypassOffset<T, false>(anchorInput);
    uint32_t progress = begin;
    if (singleInput < inputCount) {
        const uint32_t width = AxisLen(singleInput);
        while (progress < end) {
            const uint32_t count = MinU32(tileElems, end - progress);
            CopyOneSegment<T, false>(singleInput, width, progress, progress,
                                      count, dst, bypass);
            progress += count;
        }
        return;
    }
    uint32_t hintInputIdx = 0U, lastRow = 0xFFFFFFFFU;
    const uint32_t rowStride = outputAxisSize * innerSize;
    while (progress < end) {
        const uint32_t row = progress / rowStride;
        if (row != lastRow) {
            hintInputIdx = 0U;
            lastRow = row;
        }
        const LinearSegment32 part = LocateSegment32(progress, hintInputIdx);
        if (part.count == 0U) break;
        hintInputIdx = part.inputIdx;
        const uint32_t count = MinU32(tileElems, MinU32(part.count, end - progress));
        CopyOneSegment<T, false>(part.inputIdx, part.axisLen, part.inputOffset,
                                  progress, count, dst, bypass);
        progress += count;
    }
}
