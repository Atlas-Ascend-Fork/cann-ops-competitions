// Vector-selected local lists with disjoint 32-byte output ownership.
// This header is included inside the kernel's anonymous namespace.
template <typename T, bool Ping = false>
class KernelIndexAddFiltered {
public:
    using AccT = typename AccumulatorType<T>::Type;
    static constexpr bool kBf16 = IsSame<T, bfloat16_t>::value;
    static constexpr bool kInt8 = IsSame<T, int8_t>::value;

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
                                const IndexAddTilingData* t, TPipe* pipe)
    {
        t_ = t; pipe_ = pipe;
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), t_->selfElements);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source), t_->sourceElements);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), t_->selfElements);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(index), t_->indexCount);
        pipe_->InitBuffer(indexBuffer_, t_->indexBufferBytes);
        pipe_->InitBuffer(headBuffer_, t_->csrHeadBytes);
        pipe_->InitBuffer(prefixBuffer_, t_->csrPrefixBytes);
        pipe_->InitBuffer(positionBuffer_, t_->indexBufferBytes);
        pipe_->InitBuffer(nodeBuffer_, t_->indexBufferBytes);
        pipe_->InitBuffer(expandedBuffer_, t_->csrPrefixBytes);
        pipe_->InitBuffer(maskBuffer_, 2 * t_->matchMaskBufferBytes);
        pipe_->InitBuffer(inputQueue_, 2, t_->csrSourceSlotBytes);
        pipe_->InitBuffer(accBuffer_, t_->accumulatorBufferBytes);
        pipe_->InitBuffer(rawBuffer_, t_->rawBufferBytes);
        if constexpr (kBf16 || kInt8) { pipe_->InitBuffer(convertBuffer_, t_->castBufferBytes); }
        next_ = indexBuffer_.Get<uint32_t>();
        head_ = headBuffer_.Get<uint32_t>();
        prefix_ = prefixBuffer_.Get<uint32_t>();
        positions_ = positionBuffer_.Get<int32_t>();
        nodes_ = nodeBuffer_.Get<uint32_t>();
        expanded_ = expandedBuffer_.Get<uint32_t>();
        mask_ = maskBuffer_.Get<uint8_t>();
        mask2_ = mask_[t_->matchMaskBufferBytes];
        raw_ = rawBuffer_.Get<T>();
        acc_ = accBuffer_.Get<AccT>();
        if constexpr (Ping) {
            for (uint32_t b = 0; b < 2; ++b) {
                rawFree_[b] = pipe_->AllocEventID<HardEvent::MTE3_MTE2>();
                rawVectorFree_[b] = pipe_->AllocEventID<HardEvent::MTE3_V>();
                rawPending_[b] = false;
            }
            rawSlot_ = 0;
        }
    }

    __aicore__ inline void Process()
    {
        BuildLists();
        // Filtering already counted this core's selected updates. Retain caching
        // when a small set of owners receives a disproportionate input stream.
        if (t_->rowPresenceBufferBytes && static_cast<uint64_t>(selectedCount_) * t_->destinationAxis <=
            4 * (t_->indexCount * rows_ + t_->destinationAxis)) {
            self_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            source_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        if constexpr (Ping) { PipeBarrier<PIPE_ALL>(); }
        uint64_t begin = begin_;
        while (begin < end_) {
            const uint64_t row = begin / t_->inner;
            const uint64_t outer = row / t_->destinationAxis;
            const uint32_t d = row - outer * t_->destinationAxis;
            const uint32_t localD = d >= startD_ ? d - startD_ : d + t_->destinationAxis - startD_;
            const uint32_t count = head_.GetValue(localD) >> 16;
            if (count == 0) {
                uint64_t stop = (row + 1) * t_->inner;
                uint32_t nextD = localD + 1;
                if (nextD == t_->destinationAxis) { nextD = 0; }
                while (stop < end_ && (head_.GetValue(nextD) >> 16) == 0) {
                    stop += t_->inner;
                    if (++nextD == t_->destinationAxis) { nextD = 0; }
                }
                if (stop > end_) { stop = end_; }
                CopySpan(begin, stop); begin = stop;
            } else {
                const uint64_t offset = begin - row * t_->inner;
                uint64_t n = t_->inner - offset;
                if (n > end_ - begin) { n = end_ - begin; }
                if (n > t_->tileInner) { n = t_->tileInner; }
                UpdateTile(begin, outer, localD, offset, n, count); begin += n;
            }
        }
        if constexpr (Ping) {
            for (uint32_t b = 0; b < 2; ++b) {
                WaitRaw(b);
                pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(rawFree_[b]);
                pipe_->ReleaseEventID<HardEvent::MTE3_V>(rawVectorFree_[b]);
            }
        }
    }

private:
    // Index values are dead after BuildLists; the compact head/node lists remain live.
    // Reuse that storage without reducing the source batch or the output tile.
    __aicore__ inline void WaitRaw(uint32_t b)
    {
        if (rawPending_[b]) {
            WaitFlag<HardEvent::MTE3_MTE2>(rawFree_[b]);
            WaitFlag<HardEvent::MTE3_V>(rawVectorFree_[b]);
            rawPending_[b] = false;
        }
    }
    __aicore__ inline void BeginRaw()
    {
        WaitRaw(rawSlot_);
        raw_ = rawSlot_ == 0 ? rawBuffer_.Get<T>() : indexBuffer_.Get<T>();
    }
    __aicore__ inline void EndRaw()
    {
        SetFlag<HardEvent::MTE3_MTE2>(rawFree_[rawSlot_]);
        SetFlag<HardEvent::MTE3_V>(rawVectorFree_[rawSlot_]);
        rawPending_[rawSlot_] = true;
        rawSlot_ ^= 1;
    }
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        const event_t event = static_cast<event_t>(pipe_->FetchEventID(E));
        SetFlag<E>(event); WaitFlag<E>(event);
    }

    __aicore__ inline void BuildLists()
    {
        const uint64_t unit = 32 / sizeof(T);
        const uint64_t blocks = (t_->selfElements + unit - 1) / unit;
        const uint64_t perCore = blocks / t_->usedCoreNum;
        const uint64_t extra = blocks % t_->usedCoreNum;
        const uint64_t core = GetBlockIdx();
        begin_ = (core * perCore + (core < extra ? core : extra)) * unit;
        end_ = begin_ + (perCore + (core < extra ? 1 : 0)) * unit;
        if (end_ > t_->selfElements) { end_ = t_->selfElements; }
        const uint64_t firstRow = begin_ / t_->inner;
        const uint64_t rowCount = (end_ - 1) / t_->inner - firstRow + 1;
        startD_ = firstRow % t_->destinationAxis;
        rows_ = rowCount < t_->destinationAxis ? rowCount : t_->destinationAxis;
        const uint32_t n = t_->indexVectorElements;
        Duplicate(next_, static_cast<uint32_t>(0xffffffff), n);
        Duplicate(head_, static_cast<uint32_t>(0xffff), t_->csrHeadBytes / 4);
        Sync<HardEvent::V_MTE2>();
        DataCopyExtParams p{1, static_cast<uint32_t>(t_->indexCount * 4), 0, 0, 0};
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        DataCopyPad(next_, index_, p, pad);

        // Compare packs 64 bits per 64 INT32 lanes. GatherMask requires a
        // 32-byte stride between those 64-bit masks, so expand by Gather.
        auto offsets = prefix_.template ReinterpretCast<int32_t>();
        auto tmp = nodes_.template ReinterpretCast<int32_t>();
        auto expanded = expanded_.template ReinterpretCast<int32_t>();
        const uint32_t expandedCount = t_->csrPrefixBytes / 4;
        CreateVecIndex(offsets, static_cast<int32_t>(0), expandedCount); PipeBarrier<PIPE_V>();
        ShiftRight(tmp, offsets, static_cast<int32_t>(3), expandedCount); PipeBarrier<PIPE_V>();
        ShiftLeft(tmp, tmp, static_cast<int32_t>(3), expandedCount); PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, static_cast<int32_t>(2), expandedCount); PipeBarrier<PIPE_V>();
        Duplicate(expanded, static_cast<int32_t>(4), expandedCount); PipeBarrier<PIPE_V>();
        And(offsets.template ReinterpretCast<uint16_t>(), offsets.template ReinterpretCast<uint16_t>(),
            expanded.template ReinterpretCast<uint16_t>(), expandedCount * 2); PipeBarrier<PIPE_V>();
        Add(offsets, offsets, tmp, expandedCount); PipeBarrier<PIPE_V>();
        Sync<HardEvent::MTE2_V>();
        // A2 INT32 CompareScalar supports EQ only. Valid indices are below
        // 65535, so FP32 conversion is exact and enables GE/LT comparison.
        Cast(positions_.template ReinterpretCast<float>(), next_.template ReinterpretCast<int32_t>(),
            RoundMode::CAST_RINT, n); PipeBarrier<PIPE_V>();
        if (rows_ == t_->destinationAxis) {
            Duplicate(mask_.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0xffffffff),
                t_->matchMaskBufferBytes / 4);
        } else {
            CompareScalar(mask_, positions_.template ReinterpretCast<float>(), static_cast<float>(static_cast<int32_t>(startD_)), CMPMODE::GE, n);
            const uint32_t stop = startD_ + rows_;
            CompareScalar(mask2_, positions_.template ReinterpretCast<float>(),
                static_cast<float>(static_cast<int32_t>(stop <= t_->destinationAxis ? stop : stop - t_->destinationAxis)), CMPMODE::LT, n);
            PipeBarrier<PIPE_V>();
            if (stop <= t_->destinationAxis) {
                And(mask_.template ReinterpretCast<uint16_t>(), mask_.template ReinterpretCast<uint16_t>(),
                    mask2_.template ReinterpretCast<uint16_t>(), t_->matchMaskBufferBytes / 2);
            } else {
                Or(mask_.template ReinterpretCast<uint16_t>(), mask_.template ReinterpretCast<uint16_t>(),
                    mask2_.template ReinterpretCast<uint16_t>(), t_->matchMaskBufferBytes / 2);
            }
        }
        PipeBarrier<PIPE_V>();
        Gather(expanded_, mask_.template ReinterpretCast<uint32_t>(), prefix_, 0, expandedCount);
        PipeBarrier<PIPE_V>();
        CreateVecIndex(positions_, static_cast<int32_t>(0), n); PipeBarrier<PIPE_V>();
        uint64_t selected = 0;
        GatherMask(nodes_, positions_.template ReinterpretCast<uint32_t>(), expanded_, false, 0,
            {1, static_cast<uint16_t>(n / 64), 8, 1}, selected);
        Sync<HardEvent::V_S>(); Sync<HardEvent::MTE2_S>();
        selectedCount_ = static_cast<uint32_t>(selected);
        for (uint32_t r = selected; r > 0; --r) {
            const uint32_t k = r - 1;
            const uint32_t m = nodes_.GetValue(k);
            if (m >= t_->indexCount) { continue; }
            const uint32_t d = next_.GetValue(m);
            if (d >= t_->destinationAxis) { continue; }
            const uint32_t localD = d >= startD_ ? d - startD_ : d + t_->destinationAxis - startD_;
            if (localD >= rows_) { continue; }
            const uint32_t old = head_.GetValue(localD);
            nodes_.SetValue(k, ((old & 0xffff) << 16) | m);
            head_.SetValue(localD, ((old >> 16) + 1) * 65536 + k);
        }
    }

    __aicore__ inline void CopySpan(uint64_t first, uint64_t end)
    {
        const uint64_t cap = t_->csrSourceSlotBytes / sizeof(T);
        uint64_t offset = first;
        uint32_t count = end - offset < cap ? end - offset : cap;
        auto input = inputQueue_.AllocTensor<T>();
        DataCopyExtParams p{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(input, self_[offset], p, pad); inputQueue_.EnQue(input);
        uint64_t nextOffset = offset + count;
        for (;;) {
            auto current = inputQueue_.DeQue<T>();
            uint32_t nextCount = 0;
            if (nextOffset < end) {
                nextCount = end - nextOffset < cap ? end - nextOffset : cap;
                auto nextInput = inputQueue_.AllocTensor<T>();
                DataCopyExtParams nextParams{1, static_cast<uint32_t>(nextCount * sizeof(T)), 0, 0, 0};
                DataCopyPad(nextInput, self_[nextOffset], nextParams, pad);
                inputQueue_.EnQue(nextInput);
            }
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams outParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(out_[offset], current, outParams);
            Sync<HardEvent::MTE3_V>(); inputQueue_.FreeTensor(current);
            if (nextCount == 0) { break; }
            offset = nextOffset; count = nextCount; nextOffset += nextCount;
        }
    }

    __aicore__ inline uint32_t Aligned(uint32_t n)
    {
        return AlignUp(n, kInt8 ? 128 : 32 / sizeof(T));
    }

    __aicore__ inline void ReadSelf(uint64_t offset, uint32_t n, uint32_t aligned)
    {
        if constexpr (kInt8) {
            Duplicate(raw_.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), aligned / 4);
            Sync<HardEvent::V_MTE2>();
        }
        DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(AlignUp(n, 32 / sizeof(T)) - n), static_cast<T>(0)};
        DataCopyPad(raw_, self_[offset], p, pad);
        Sync<HardEvent::MTE2_V>();
        if constexpr (kInt8) {
            auto bridge = convertBuffer_.Get<half>();
            Cast(bridge, raw_, RoundMode::CAST_NONE, aligned); PipeBarrier<PIPE_V>();
            Cast(acc_, bridge, RoundMode::CAST_RINT, aligned);
        } else if constexpr (kBf16) { Cast(acc_, raw_, RoundMode::CAST_NONE, aligned); }
        else { DataCopy(acc_, raw_, aligned); }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyFlat(uint64_t offset, uint32_t n)
    {
        DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(raw_, self_[offset], p, pad);
        Sync<HardEvent::MTE2_MTE3>();
        DataCopyPad(out_[offset], raw_, p);
        Sync<HardEvent::MTE3_MTE2>();
        if constexpr (kInt8) { Sync<HardEvent::MTE3_V>(); }
    }

    __aicore__ inline void Prefetch(uint64_t outer, uint64_t offset, uint32_t valid,
                                    uint32_t aligned, uint32_t stride,
                                    uint32_t& cursor, uint32_t batch)
    {
        auto input = inputQueue_.AllocTensor<T>();
        if constexpr (kInt8) {
            Duplicate(input.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), batch * aligned / 4);
            Sync<HardEvent::V_MTE2>();
        }
        uint32_t positions[16];
        for (uint32_t j = 0; j < batch; ++j) {
            const uint32_t packed = nodes_.GetValue(cursor);
            positions[j] = packed & 0xffff;
            cursor = packed >> 16;
        }
        uint32_t j = 0;
        while (j < batch) {
            uint32_t run = 1, gap = 0;
            if (j + 1 < batch) {
                gap = positions[j + 1] - positions[j]; run = 2;
                while (j + run < batch && positions[j + run] - positions[j + run - 1] == gap) { ++run; }
            }
            uint64_t sourceStride = run > 1 ? (static_cast<uint64_t>(gap) * t_->inner - valid) * sizeof(T) : 0;
            if (sourceStride > 0xffffffffULL) { run = 1; sourceStride = 0; }
            DataCopyExtParams p{static_cast<uint16_t>(run), static_cast<uint32_t>(valid * sizeof(T)),
                static_cast<uint32_t>(sourceStride),
                static_cast<uint32_t>((aligned * sizeof(T) - AlignUp(valid * sizeof(T), 32)) / 32), 0};
            DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(AlignUp(valid, 32 / sizeof(T)) - valid), static_cast<T>(0)};
            DataCopyPad(input[j * aligned], source_[(outer * t_->indexCount + positions[j]) * t_->inner + offset], p, pad);
            j += run;
        }
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void Consume(LocalTensor<T> input, uint32_t n, uint32_t batch)
    {
        LocalTensor<AccT> wide;
        if constexpr (kInt8) {
            auto bridge = convertBuffer_.Get<half>();
            wide = input.template ReinterpretCast<int16_t>();
            Cast(bridge, input, RoundMode::CAST_NONE, n * batch); PipeBarrier<PIPE_V>();
            Cast(wide, bridge, RoundMode::CAST_RINT, n * batch); PipeBarrier<PIPE_V>();
        } else if constexpr (kBf16) {
            wide = convertBuffer_.Get<float>();
            Cast(wide, input, RoundMode::CAST_NONE, n * batch); PipeBarrier<PIPE_V>();
        } else { wide = input; }
        SetMaskCount(); SetVectorMask<AccT, MaskMode::COUNTER>(0, n);
        const BinaryRepeatParams addParams{1, 1, 1, 8, 8, 8};
        for (uint32_t j = 0; j < batch; ++j) {
            Add<AccT, false>(acc_, acc_, wide[j * n], MASK_PLACEHOLDER, 1, addParams);
            PipeBarrier<PIPE_V>();
            if constexpr (kBf16) {
                const UnaryRepeatParams narrow{1, 1, 4, 8};
                const UnaryRepeatParams widen{1, 1, 8, 4};
                Cast<bfloat16_t, float, false>(raw_, acc_, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, narrow);
                PipeBarrier<PIPE_V>();
                Cast<float, bfloat16_t, false>(acc_, raw_, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, widen);
                PipeBarrier<PIPE_V>();
            }
        }
        ResetMask(); SetMaskNorm();
    }

    __aicore__ inline void WriteResult(uint64_t offset, uint32_t n, uint32_t aligned)
    {
        if constexpr (kInt8) {
            auto scratch = convertBuffer_.Get<int16_t>();
            auto bridge = convertBuffer_.Get<half>();
            ShiftLeft(scratch, acc_, static_cast<int16_t>(8), aligned); PipeBarrier<PIPE_V>();
            ShiftRight(acc_, scratch, static_cast<int16_t>(8), aligned); PipeBarrier<PIPE_V>();
            Cast(bridge, acc_, RoundMode::CAST_NONE, aligned); PipeBarrier<PIPE_V>();
            Cast(raw_, bridge, RoundMode::CAST_NONE, aligned);
        } else if constexpr (kBf16) { Cast(raw_, acc_, RoundMode::CAST_RINT, aligned); }
        else { DataCopy(raw_, acc_, aligned); }
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
        DataCopyPad(out_[offset], raw_, p);
        if constexpr (Ping) { EndRaw(); }
        else { Sync<HardEvent::MTE3_MTE2>(); Sync<HardEvent::MTE3_V>(); }
    }

    __aicore__ inline void UpdateTile(uint64_t outputOffset, uint64_t outer, uint32_t d,
                                     uint64_t offset, uint32_t valid, uint32_t count)
    {
        const uint32_t aligned = Aligned(valid);
        if constexpr (Ping) { BeginRaw(); }
        ReadSelf(outputOffset, valid, aligned);
        uint32_t cursor = head_.GetValue(d) & 0xffff;
        const uint32_t stride = 0;
        uint32_t remaining = count;
        uint32_t batch = remaining < t_->csrBatchSize ? remaining : t_->csrBatchSize;
        Prefetch(outer, offset, valid, aligned, stride, cursor, batch);
        remaining -= batch;
        for (;;) {
            auto input = inputQueue_.DeQue<T>();
            uint32_t nextBatch = remaining < t_->csrBatchSize ? remaining : t_->csrBatchSize;
            if (nextBatch != 0) {
                Prefetch(outer, offset, valid, aligned, stride, cursor, nextBatch);
                remaining -= nextBatch;
            }
            Consume(input, aligned, batch);
            inputQueue_.FreeTensor(input);
            if (nextBatch == 0) { break; }
            batch = nextBatch;
        }
        WriteResult(outputOffset, valid, aligned);
    }

    const IndexAddTilingData* t_;
    TPipe* pipe_;
    GlobalTensor<T> self_, source_, out_;
    GlobalTensor<uint32_t> index_;
    TBuf<QuePosition::VECCALC> indexBuffer_, headBuffer_, prefixBuffer_, accBuffer_, rawBuffer_, convertBuffer_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    LocalTensor<uint32_t> next_, head_, prefix_;
    LocalTensor<T> raw_;
    LocalTensor<AccT> acc_;
    TBuf<QuePosition::VECCALC> positionBuffer_, nodeBuffer_, expandedBuffer_, maskBuffer_;
    LocalTensor<int32_t> positions_;
    LocalTensor<uint32_t> nodes_, expanded_;
    LocalTensor<uint8_t> mask_, mask2_;
    uint64_t begin_, end_;
    uint32_t startD_, rows_, selectedCount_;
    TEventID rawFree_[2], rawVectorFree_[2];
    bool rawPending_[2];
    uint32_t rawSlot_;
};
