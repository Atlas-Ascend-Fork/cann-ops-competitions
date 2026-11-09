// Stable per-destination lists with weighted, disjoint output ownership.
// This header is included inside the kernel's anonymous namespace.
template <typename T>
class KernelIndexAddOrdered {
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
        pipe_->InitBuffer(inputQueue_, 2, t_->csrSourceSlotBytes);
        pipe_->InitBuffer(accBuffer_, t_->accumulatorBufferBytes);
        pipe_->InitBuffer(rawBuffer_, t_->rawBufferBytes);
        if constexpr (kBf16 || kInt8) { pipe_->InitBuffer(convertBuffer_, t_->castBufferBytes); }
        next_ = indexBuffer_.Get<uint32_t>();
        head_ = headBuffer_.Get<uint32_t>();
        prefix_ = prefixBuffer_.Get<uint32_t>();
        raw_ = rawBuffer_.Get<T>();
        acc_ = accBuffer_.Get<AccT>();
    }

    __aicore__ inline void Process()
    {
        BuildLists();
        const uint64_t total = static_cast<uint64_t>(totalWeight_) * t_->inner * t_->outer;
        const uint64_t unit = 32 / sizeof(T);
        const uint32_t core = GetBlockIdx();
        uint64_t begin = Locate(total * core / t_->usedCoreNum) / unit * unit;
        uint64_t end = core + 1 == t_->usedCoreNum ? t_->selfElements
            : Locate(total * (core + 1) / t_->usedCoreNum) / unit * unit;
        while (begin < end) {
            const uint64_t row = begin / t_->inner;
            const uint64_t outer = row / t_->destinationAxis;
            const uint32_t d = row - outer * t_->destinationAxis;
            const uint32_t count = (head_.GetValue(d) >> 16) & 0x7fff;
            if (count == 0) {
                uint64_t stop = (row + 1) * t_->inner;
                uint32_t e = d + 1;
                while (e < t_->destinationAxis && stop < end &&
                       ((head_.GetValue(e) >> 16) & 0x7fff) == 0) {
                    ++e; stop += t_->inner;
                }
                if (stop > end) { stop = end; }
                while (begin < stop) {
                    uint32_t n = stop - begin < t_->tileInner ? stop - begin : t_->tileInner;
                    CopyFlat(begin, n); begin += n;
                }
            } else {
                const uint64_t offset = begin - row * t_->inner;
                uint64_t n = t_->inner - offset;
                if (n > end - begin) { n = end - begin; }
                if (n > t_->tileInner) { n = t_->tileInner; }
                UpdateTile(begin, outer, d, offset, n, count);
                begin += n;
            }
        }
    }

private:
    template <HardEvent E>
    __aicore__ inline void Sync()
    {
        const event_t event = static_cast<event_t>(pipe_->FetchEventID(E));
        SetFlag<E>(event); WaitFlag<E>(event);
    }

    __aicore__ inline void BuildLists()
    {
        Duplicate(head_, static_cast<uint32_t>(0xffff), t_->csrHeadBytes / 4);
        Duplicate(prefix_, static_cast<uint32_t>(0), t_->csrPrefixBytes / 4);
        DataCopyExtParams p{1, static_cast<uint32_t>(t_->indexCount * 4), 0, 0, 0};
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        DataCopyPad(next_, index_, p, pad);
        Sync<HardEvent::MTE2_S>(); Sync<HardEvent::V_S>();
        for (uint32_t r = t_->indexCount; r > 0; --r) {
            const uint32_t m = r - 1;
            const uint32_t d = next_.GetValue(m);
            if (d >= t_->destinationAxis) { next_.SetValue(m, 0xffff); continue; }
            const uint32_t old = head_.GetValue(d);
            const uint32_t count = (old >> 16) & 0x7fff;
            const uint32_t oldHead = old & 0xffff;
            uint32_t flag = old & 0x80000000;
            next_.SetValue(m, oldHead);
            if (count == 1) { prefix_.SetValue(d, oldHead - m); }
            else if (count >= 2 && flag == 0 && prefix_.GetValue(d) != oldHead - m) { flag = 0x80000000; }
            head_.SetValue(d, flag | ((count + 1) << 16) | m);
        }
        uint32_t sum = 0;
        for (uint32_t d = 0; d < t_->destinationAxis; ++d) {
            const uint32_t h = head_.GetValue(d);
            const uint32_t count = (h >> 16) & 0x7fff;
            const uint32_t stride = (h & 0x80000000) != 0 ? 0xffff : prefix_.GetValue(d);
            prefix_.SetValue(d, (stride << 16) | sum);
            sum += (kBf16 ? 4 : kInt8 ? 2 : 1) * count + 2;
        }
        prefix_.SetValue(t_->destinationAxis, sum);
        totalWeight_ = sum;
    }

    __aicore__ inline uint64_t Locate(uint64_t target)
    {
        const uint64_t outerWeight = static_cast<uint64_t>(totalWeight_) * t_->inner;
        const uint64_t o = target / outerWeight;
        const uint64_t remainder = target - o * outerWeight;
        const uint32_t bucket = remainder / t_->inner;
        uint32_t lo = 0, hi = t_->destinationAxis;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if ((prefix_.GetValue(mid) & 0xffff) <= bucket) { lo = mid; } else { hi = mid; }
        }
        const uint32_t before = prefix_.GetValue(lo) & 0xffff;
        const uint32_t weight = (prefix_.GetValue(lo + 1) & 0xffff) - before;
        return (o * t_->destinationAxis + lo) * t_->inner +
            (remainder - static_cast<uint64_t>(before) * t_->inner) / weight;
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
            positions[j] = cursor;
            cursor = stride == 0xffff ? next_.GetValue(cursor) : cursor + stride;
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
        Sync<HardEvent::MTE3_MTE2>(); Sync<HardEvent::MTE3_V>();
    }

    __aicore__ inline void UpdateTile(uint64_t outputOffset, uint64_t outer, uint32_t d,
                                     uint64_t offset, uint32_t valid, uint32_t count)
    {
        if (t_->rowPresenceBufferBytes) {
            const CacheMode mode = valid * sizeof(T) >= 512 ? CacheMode::CACHE_MODE_DISABLE : CacheMode::CACHE_MODE_NORMAL;
            self_.SetL2CacheHint(mode); source_.SetL2CacheHint(mode);
        }
        const uint32_t aligned = Aligned(valid);
        ReadSelf(outputOffset, valid, aligned);
        uint32_t cursor = head_.GetValue(d) & 0xffff;
        const uint32_t stride = prefix_.GetValue(d) >> 16;
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
    uint32_t totalWeight_;
};
