// Included inside ConcatKernel; not a standalone compilation unit.
// Periodic repeated-width B32 groups. Host admission proves pattern and UB limits.
// Units: widths/pitches are B32 elements; DMA lengths and Gather offsets are bytes.

    template <typename T>
    __aicore__ inline void BuildPeriodicGatherIndexScalar(
        uint32_t patternInputs, LocalTensor<uint32_t>& indexLocal)
    {
        uint32_t axisOffset = 0U;
        uint32_t stageOffset = 0U;
        for (uint32_t member = 0U; member < patternInputs; ++member) {
            const uint32_t width = AxisLen(member);
            for (uint32_t row = 0U; row < outerSize; ++row) {
                const uint32_t dstBase = row * kPeriodicGatherAxisElems + axisOffset;
                const uint32_t srcBase = stageOffset + row * width;
                for (uint32_t col = 0U; col < width; ++col) {
                    indexLocal.SetValue(dstBase + col, (srcBase + col) * sizeof(T));
                }
            }
            axisOffset += width;
            stageOffset += outerSize * width;
        }
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventId);
        WaitFlag<HardEvent::S_V>(eventId);
    }

    __aicore__ inline uint64_t PeriodicMemberMask(uint32_t axisOffset,
                                                  uint32_t width) const
    {
        const uint64_t rowMask = width == kPeriodicGatherAxisElems
            ? 0xFFFFFFFFULL
            : ((1ULL << width) - 1ULL) << axisOffset;
        return rowMask | (rowMask << kPeriodicGatherAxisElems);
    }

    template <typename T>
    __aicore__ inline void BuildPeriodicGatherIndexVector(
        uint32_t patternInputs, LocalTensor<uint32_t>& indexLocal)
    {
        LocalTensor<int32_t> indexSigned =
            indexLocal.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> workLocal = periodicIndexWorkBuf.Get<int32_t>();
        const uint32_t totalPairs = outerSize / 2U;
        constexpr uint32_t kElemsPerPair = 2U * kPeriodicGatherAxisElems;
        constexpr uint32_t kMaxPairRepeats = 255U;
        const UnaryRepeatParams repeatParams{1U, 1U, 8U, 8U};

        for (uint32_t pairBase = 0U; pairBase < totalPairs;
             pairBase += kMaxPairRepeats) {
            const uint32_t pairCount =
                MinU32(kMaxPairRepeats, totalPairs - pairBase);
            const uint32_t elementBase = pairBase * kElemsPerPair;
            const uint32_t elementCount = pairCount * kElemsPerPair;
            LocalTensor<int32_t> indexChunk = indexSigned[elementBase];
            LocalTensor<int32_t> workChunk = workLocal[elementBase];

            CreateVecIndex(indexChunk, static_cast<int32_t>(elementBase),
                           elementCount);
            Muls(indexChunk, indexChunk, static_cast<int32_t>(sizeof(T)),
                 elementCount);
            CreateVecIndex(workChunk, static_cast<int32_t>(elementBase),
                           elementCount);
            ShiftRight(workChunk, workChunk, static_cast<int32_t>(5),
                       elementCount);

            uint32_t axisOffset = 0U;
            uint32_t stageOffset = 0U;
            for (uint32_t member = 0U; member < patternInputs; ++member) {
                const uint32_t width = AxisLen(member);
                uint64_t memberMask[2] = {
                    PeriodicMemberMask(axisOffset, width), 0ULL};
                const int32_t rowDelta =
                    (static_cast<int32_t>(width) -
                     static_cast<int32_t>(kPeriodicGatherAxisElems)) *
                    static_cast<int32_t>(sizeof(T));
                const int32_t baseDelta =
                    (static_cast<int32_t>(stageOffset) -
                     static_cast<int32_t>(axisOffset)) *
                    static_cast<int32_t>(sizeof(T));
                Muls(workChunk, workChunk, rowDelta, memberMask,
                     static_cast<uint8_t>(pairCount), repeatParams);
                Adds(indexChunk, indexChunk, baseDelta, memberMask,
                     static_cast<uint8_t>(pairCount), repeatParams);
                axisOffset += width;
                stageOffset += outerSize * width;
            }
            Add(indexChunk, indexChunk, workChunk, elementCount);
        }
    }

    template <typename T>
    __aicore__ inline void BuildPeriodicGatherIndex(
        uint32_t patternInputs, LocalTensor<uint32_t>& indexLocal)
    {
        if ((outerSize & 1U) == 0U) {
            BuildPeriodicGatherIndexVector<T>(patternInputs, indexLocal);
            return;
        }
        BuildPeriodicGatherIndexScalar<T>(patternInputs, indexLocal);
    }

    template <typename T>
    __aicore__ inline bool TryProcessPeriodicGather(GlobalTensor<T>& yGm)
    {
        if (periodicPatternInputs == 0U) {
            return false;
        }
        const uint32_t patternInputs = periodicPatternInputs;
        const uint32_t groupCount = inputCount / patternInputs;
        const uint32_t periodDataElems = outerSize * kPeriodicGatherAxisElems;
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        if (blockIdx >= groupCount) {
            return true;
        }

        const uint32_t periodBytes = periodDataElems * sizeof(T);
        pipe->InitBuffer(inQueue, 1U, periodBytes);
        pipe->InitBuffer(outQueue, 1U, periodBytes);
        pipe->InitBuffer(maskBuf, periodDataElems * sizeof(uint32_t));
        pipe->InitBuffer(periodicIndexWorkBuf,
                         periodDataElems * sizeof(int32_t));
        InitMetaCache();
        LocalTensor<uint32_t> indexLocal = maskBuf.Get<uint32_t>();
        BuildPeriodicGatherIndex<T>(patternInputs, indexLocal);

        DataCopyExtParams copyOut{
            static_cast<uint16_t>(outerSize),
            static_cast<uint32_t>(kPeriodicGatherAxisElems * sizeof(T)),
            0U,
            static_cast<uint32_t>((outputAxisSize - kPeriodicGatherAxisElems) * sizeof(T)),
            0U};

        for (uint32_t group = blockIdx; group < groupCount; group += blockNum) {
            LocalTensor<T> staged = inQueue.AllocTensor<T>();
            uint32_t stageOffset = 0U;
            const uint32_t groupStart = group * patternInputs;
            for (uint32_t member = 0U; member < patternInputs; ++member) {
                const uint32_t inputIdx = groupStart + member;
                const uint32_t inputElems = outerSize * AxisLen(inputIdx);
                GlobalTensor<T> xGm;
                xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(InputPtr(inputIdx)),
                                    static_cast<uint64_t>(inputElems));
                xGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
                DataCopy(staged[stageOffset], xGm, inputElems);
                stageOffset += inputElems;
            }
            inQueue.EnQue(staged);

            LocalTensor<T> ready = inQueue.DeQue<T>();
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();
            Gather(outputLocal, ready, indexLocal, 0U, periodDataElems);
            outQueue.EnQue(outputLocal);
            inQueue.FreeTensor(ready);

            LocalTensor<T> outputReady = outQueue.DeQue<T>();
            DataCopyPad(yGm[group * kPeriodicGatherAxisElems], outputReady, copyOut);
            outQueue.FreeTensor(outputReady);
        }
        return true;
    }
