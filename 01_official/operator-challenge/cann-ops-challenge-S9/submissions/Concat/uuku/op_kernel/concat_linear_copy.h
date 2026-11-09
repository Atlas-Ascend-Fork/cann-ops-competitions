// Included inside ConcatKernel. Only the bulk linear DMA fast path uses this
// invocation-local alias; row, gather and stream routes keep their own policy.
template <typename T, bool WideGeometry = true>
__aicore__ inline uint64_t InitLinearInputBypassOffset(uint32_t anchorInput)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201) && defined(L2_CACHE_HINT) && \
    (!defined(ASCENDC_OOM) || ASCENDC_OOM == 0)
    // Host supplies a validated nonempty anchor. Prepare the SDK-derived A2
    // alias once per invocation, without another per-core descriptor scan.
    const uint32_t width = AxisLen(anchorInput);
    auto base = reinterpret_cast<__gm__ T*>(InputPtr(anchorInput));
    GlobalTensor<T> anchor;
    const uint64_t inputElems = WideGeometry ? outerSize64 * width * innerSize64
        : static_cast<uint64_t>(outerSize * width * innerSize);
    anchor.SetGlobalBuffer(base, inputElems);
    anchor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    return reinterpret_cast<uint64_t>(anchor.GetPhyAddr()) -
           reinterpret_cast<uint64_t>(base);
#else
    (void)anchorInput;
#endif
    return 0U;
}

template <typename T, bool WideGeometry = true>
__aicore__ inline void CopyOneSegment(uint32_t inputIdx, uint32_t axisLen,
                                      uint64_t inputOffset, uint64_t outputOffset,
                                      uint32_t count, GlobalTensor<T>& yGm,
                                      uint64_t bypassOffset)
{
    // The input and size still change at segment boundaries. Keep the original
    // binding/copy lifetime; only repeated SDK hint preparation is removed.
    auto base = reinterpret_cast<__gm__ T*>(InputPtr(inputIdx));
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201) && defined(L2_CACHE_HINT) && \
    (!defined(ASCENDC_OOM) || ASCENDC_OOM == 0)
    base = reinterpret_cast<__gm__ T*>(reinterpret_cast<uint64_t>(base) + bypassOffset);
#else
    (void)bypassOffset;
#endif
    GlobalTensor<T> xGm;
    // Narrow mode is admitted only when all geometry and total elements fit
    // U32. For nonempty output, each intermediate input product is <= total.
    const uint64_t inputElems = WideGeometry ? outerSize64 * axisLen * innerSize64
        : static_cast<uint64_t>(outerSize * axisLen * innerSize);
    xGm.SetGlobalBuffer(base, static_cast<uint64_t>(inputElems));
#if !defined(__NPU_ARCH__) || (__NPU_ARCH__ != 2201) || !defined(L2_CACHE_HINT) || \
    (defined(ASCENDC_OOM) && ASCENDC_OOM != 0)
    xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
#endif

    const uint32_t copyBytes = count * sizeof(T);
    DataCopyExtParams copyParams{1U, copyBytes, 0U, 0U, 0U};
    DataCopyPadExtParams<T> padParams{false, 0U, 0U, static_cast<T>(0)};

    LocalTensor<T> xLocal = copyQueue.AllocTensor<T>();
    DataCopyPad(xLocal, xGm[inputOffset], copyParams, padParams);
    copyQueue.EnQue(xLocal);

    LocalTensor<T> yReady = copyQueue.DeQue<T>();
    DataCopyPad(yGm[outputOffset], yReady, copyParams);
    copyQueue.FreeTensor(yReady);
}
