// One launch-local queue lifetime for an entire small B16 row shard.
// Included inside ConcatKernel; no extra class members or dtype ABI changes.
template <typename T>
__aicore__ inline bool TryProcessSmallB16RowGroup(GlobalTensor<T>& yGm)
{
    if constexpr (sizeof(T) != 2U) {
        return false;
    } else {
        constexpr uint32_t elementBytes = sizeof(T);
        constexpr uint32_t elemsPerBlock = 32U / elementBytes;
        constexpr uint32_t reservedBytes = 8U * 1024U;
        const uint32_t blockNum = GetBlockNum();
        if (blockNum == 0U || outerSize < blockNum || innerSize != 1U ||
            inputCount < 2U || inputCount > kMaxConcatInputs ||
            outputAxisSize == 0U || outputAxisSize % elemsPerBlock != 0U) {
            return false;
        }
        const uint32_t rowsPerCore = CeilDivU32(outerSize, blockNum);
        if (rowsPerCore > 4095U) {
            return false;
        }
        // Sum(align(width,16)) <= axis + 15*inputCount. This conservative
        // bound avoids a separate per-core GM metadata scan before staging.
        const uint64_t paddedRowBound =
            (static_cast<uint64_t>(outputAxisSize) +
             static_cast<uint64_t>(inputCount) * (elemsPerBlock - 1U) +
             elemsPerBlock - 1U) / elemsPerBlock * elemsPerBlock;
        const uint64_t stageBytes = paddedRowBound * rowsPerCore * sizeof(T);
        // At most one existing generic copy slot plus the unchanged metadata
        // reserve. Reject before allocating or modifying any shared state.
        if (stageBytes > static_cast<uint64_t>(tileElems) * sizeof(T) ||
            stageBytes + reservedBytes > ubBytes) {
            return false;
        }
        const uint32_t rowBegin = GetBlockIdx() * rowsPerCore;
        const uint32_t rowEnd = MinU32(rowBegin + rowsPerCore, outerSize);
        if (rowBegin >= rowEnd) {
            return true;
        }
        const uint32_t rows = rowEnd - rowBegin;
        pipe->InitBuffer(copyQueue, 1U, static_cast<uint32_t>(stageBytes));
        InitMetaCache();
        LocalTensor<T> staged = copyQueue.AllocTensor<T>();
        uint32_t stageOffset = 0U;
        const DataCopyPadExtParams<T> pad{false, 0U, 0U, static_cast<T>(0)};
        for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
            const uint32_t width = AxisLen(inputIdx);
            if (width == 0U) {
                continue;
            }
            GlobalTensor<T> xGm;
            xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(InputPtr(inputIdx)),
                                static_cast<uint64_t>(outerSize) * width);
            // Keep the parent's default last-axis input cache policy.
            const DataCopyExtParams copyIn{
                static_cast<uint16_t>(rows), width * elementBytes, 0U, 0U, 0U};
            DataCopyPad(staged[stageOffset],
                        xGm[static_cast<uint64_t>(rowBegin) * width], copyIn, pad);
            stageOffset += rows * AlignElems32<T>(width);
        }
        // This single queue event makes every preceding MTE2 slab available
        // to MTE3. No slab is overwritten, and there is no Vector packing.
        copyQueue.EnQue(staged);
        LocalTensor<T> ready = copyQueue.DeQue<T>();
        stageOffset = 0U;
        uint32_t axisOffset = 0U;
        for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
            const uint32_t width = AxisLen(inputIdx);
            if (width == 0U) {
                continue;
            }
            const DataCopyExtParams copyOut{
                static_cast<uint16_t>(rows), width * elementBytes, 0U,
                (outputAxisSize - width) * elementBytes, 0U};
            const uint64_t outputOffset =
                static_cast<uint64_t>(rowBegin) * outputAxisSize + axisOffset;
            DataCopyPad(yGm[outputOffset], ready[stageOffset], copyOut);
            stageOffset += rows * AlignElems32<T>(width);
            axisOffset += width;
        }
        // Free records completion after all nonempty input writes;
        // no cross-launch state exists.
        copyQueue.FreeTensor(ready);
        return true;
    }
}
