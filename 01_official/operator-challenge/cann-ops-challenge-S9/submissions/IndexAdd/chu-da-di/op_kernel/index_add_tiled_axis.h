// FP16/BF16/FP32/INT32 destination-axis tiles preserve ordered waves across UB cliffs.
template <typename T, bool ClipWaves = false>
class KernelIndexAddTiledAxis {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
        const IndexAddTilingData* t, TPipe* pipe)
    {
        local_ = *t;
        local_.inverseAlignedDestinations = t->csrHeadBytes;
        local_.inverseWaveCapacity = t->csrPrefixBytes;
        local_.inverseBatchRows = t->csrBatchSize;
        local_.matchMaskBufferBytes = AlignUp(t->indexVectorElements / 8, 32);
        local_.sourceCacheBufferBytes = t->csrSourceSlotBytes;
        local_.innerTileCount = (t->destinationAxis + t->csrHeadBytes - 1) / t->csrHeadBytes;
        local_.inverseSourceRowElements = (AlignUp(t->indexCount * sizeof(T), 32) + 32) / sizeof(T);
        local_.inverseSourceSentinel = AlignUp(t->indexCount * sizeof(T), 32) / sizeof(T);
        local_.accumulatorBufferBytes = t->csrBatchSize * t->csrHeadBytes * sizeof(T);
        local_.rawBufferBytes = local_.accumulatorBufferBytes > t->indexVectorElements
            ? local_.accumulatorBufferBytes : t->indexVectorElements;
        local_.inverseConvertBufferBytes = IsSame<T, bfloat16_t>::value ? 2 * t->csrBatchSize * t->csrHeadBytes * 4 : 0;
        local_.rowPresenceBufferBytes = t->csrHeadBytes * 4;
        t_ = &local_; p_ = pipe;
        self_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self), t_->selfElements);
        source_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source), t_->sourceElements);
        out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), t_->selfElements);
        index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index), t_->indexCount);
        p_->InitBuffer(indexBuffer_, t_->indexBufferBytes);
        p_->InitBuffer(sourceBuffer_, t_->sourceCacheBufferBytes);
        p_->InitBuffer(destinationBuffer_, t_->accumulatorBufferBytes);
        p_->InitBuffer(gatherBuffer_, t_->rawBufferBytes);
        p_->InitBuffer(countBuffer_, t_->rowPresenceBufferBytes);
        if constexpr (IsSame<T, bfloat16_t>::value) { p_->InitBuffer(convertBuffer_, t_->inverseConvertBufferBytes); }
        p_->InitBuffer(waveBuffer_, t_->inverseWaveCapacity * t_->inverseAlignedDestinations * 4);
        indices_ = indexBuffer_.Get<int32_t>();
        src_ = sourceBuffer_.Get<T>(); dst_ = destinationBuffer_.Get<T>();
        gathered_ = gatherBuffer_.Get<T>(); counts_ = countBuffer_.Get<uint32_t>();
        waves_ = waveBuffer_.Get<uint32_t>();
        startD_ = (GetBlockIdx() % t_->innerTileCount) * t_->inverseAlignedDestinations;
        validD_ = t_->destinationAxis - startD_ < t_->inverseAlignedDestinations
            ? t_->destinationAxis - startD_ : t_->inverseAlignedDestinations;
    }
    __aicore__ inline void Process()
    {
        if constexpr (!IsSame<T, int8_t>::value) {
            BuildWaves();
            const uint32_t group = GetBlockIdx() / t_->innerTileCount;
            const uint32_t tile = GetBlockIdx() % t_->innerTileCount;
            const uint32_t groups = (t_->usedCoreNum + t_->innerTileCount - 1 - tile) / t_->innerTileCount;
            for (uint64_t first = group * t_->inverseBatchRows; first < t_->outer;
                 first += groups * t_->inverseBatchRows) {
                const uint32_t rows = t_->outer - first < t_->inverseBatchRows
                    ? t_->outer - first : t_->inverseBatchRows;
                Batch(first, rows);
            }
        }
    }
private:
    template <HardEvent E> __aicore__ inline void Sync()
    {
        const event_t e = static_cast<event_t>(p_->FetchEventID(E)); SetFlag<E>(e); WaitFlag<E>(e);
    }
    __aicore__ inline void BuildWaves()
    {
        const uint32_t n = t_->indexVectorElements;
        const uint32_t lanes = t_->inverseAlignedDestinations;
        Duplicate(indices_, static_cast<int32_t>(-1), n); Sync<HardEvent::V_MTE2>();
        DataCopyExtParams ip{1, static_cast<uint32_t>(t_->indexCount * 4), 0, 0, 0};
        DataCopyPadExtParams<int32_t> ipad{false,0,0,0};
        DataCopyPad(indices_, index_, ip, ipad);
        auto scratch = waveBuffer_.Get<uint8_t>();
        auto lowMask = scratch;
        auto highMask = scratch[t_->matchMaskBufferBytes];
        auto offsets = gatherBuffer_.Get<int32_t>();
        const uint32_t expandedBytes = n / 2;
        const uint32_t expandedN = expandedBytes / 4;
        auto temp = offsets[expandedBytes / 4];
        auto expanded = scratch[2 * t_->matchMaskBufferBytes].template ReinterpretCast<int32_t>();
        CreateVecIndex(offsets, static_cast<int32_t>(0), expandedN); PipeBarrier<PIPE_V>();
        ShiftRight(temp, offsets, static_cast<int32_t>(3), expandedN); PipeBarrier<PIPE_V>();
        ShiftLeft(temp, temp, static_cast<int32_t>(3), expandedN); PipeBarrier<PIPE_V>();
        ShiftLeft(offsets, offsets, static_cast<int32_t>(2), expandedN); PipeBarrier<PIPE_V>();
        Duplicate(expanded, static_cast<int32_t>(4), expandedN); PipeBarrier<PIPE_V>();
        And(offsets.template ReinterpretCast<uint16_t>(), offsets.template ReinterpretCast<uint16_t>(),
            expanded.template ReinterpretCast<uint16_t>(), expandedN * 2); PipeBarrier<PIPE_V>();
        Add(offsets, offsets, temp, expandedN); PipeBarrier<PIPE_V>();
        auto floatingIndex = sourceBuffer_.Get<float>();
        auto positions = sourceBuffer_.Get<int32_t>()[n];
        Sync<HardEvent::MTE2_V>();
        Cast(floatingIndex, indices_, RoundMode::CAST_RINT, n); PipeBarrier<PIPE_V>();
        CompareScalar(lowMask, floatingIndex, static_cast<float>(static_cast<int32_t>(startD_)), CMPMODE::GE, n);
        CompareScalar(highMask, floatingIndex, static_cast<float>(static_cast<int32_t>(startD_ + validD_)), CMPMODE::LT, n);
        PipeBarrier<PIPE_V>();
        And(lowMask.template ReinterpretCast<uint16_t>(), lowMask.template ReinterpretCast<uint16_t>(),
            highMask.template ReinterpretCast<uint16_t>(), t_->matchMaskBufferBytes / 2); PipeBarrier<PIPE_V>();
        Gather(expanded.template ReinterpretCast<uint32_t>(), lowMask.template ReinterpretCast<uint32_t>(),
            offsets.template ReinterpretCast<uint32_t>(), 0, expandedN); PipeBarrier<PIPE_V>();
        CreateVecIndex(positions, static_cast<int32_t>(0), n); PipeBarrier<PIPE_V>();
        uint64_t selected = 0;
        GatherMask(positions, positions, expanded.template ReinterpretCast<uint32_t>(), false, 0,
            {1, static_cast<uint16_t>(n / 64), 8, 1}, selected);
        PipeBarrier<PIPE_V>();
        Duplicate(counts_, static_cast<uint32_t>(0), lanes);
        Duplicate(waves_, static_cast<uint32_t>(t_->inverseSourceSentinel * sizeof(T)), lanes * t_->inverseWaveCapacity);
        Sync<HardEvent::V_S>();
        auto tails = indices_.template ReinterpretCast<uint32_t>();
        if constexpr (ClipWaves) {
            for (uint32_t w = 0; w < t_->inverseWaveCapacity; ++w) { waveBegin_[w] = lanes; waveEnd_[w] = 0; }
        }
        for (uint32_t k = 0; k < selected; ++k) {
            const uint32_t m = positions.GetValue(k);
            if (m >= t_->indexCount) { continue; }
            const int32_t d = indices_.GetValue(m) - static_cast<int32_t>(startD_);
            if (d < 0 || static_cast<uint32_t>(d) >= validD_) { continue; }
            const uint32_t depth = counts_.GetValue(d);
            counts_.SetValue(d, depth + 1);
            if (depth < t_->inverseWaveCapacity) {
                waves_.SetValue(depth * lanes + d, m * sizeof(T));
                if constexpr (ClipWaves) {
                    const uint32_t col = static_cast<uint32_t>(d);
                    if (col < waveBegin_[depth]) { waveBegin_[depth] = col; }
                    if (col + 1 > waveEnd_[depth]) { waveEnd_[depth] = col + 1; }
                }
                if (depth + 1 > waveCount_) { waveCount_ = depth + 1; }
            } else { tails.SetValue(tailCount_++, (static_cast<uint32_t>(d) << 16) | m); }
        }
        for (uint32_t r = 0; r < t_->inverseBatchRows; ++r) {
            src_.SetValue(r * t_->inverseSourceRowElements + t_->inverseSourceSentinel, static_cast<T>(0));
        }
        Sync<HardEvent::S_V>(); Sync<HardEvent::S_MTE2>();
    }
    __aicore__ inline void Batch(uint64_t first, uint32_t rows)
    {
        const uint32_t lanes = t_->inverseAlignedDestinations;
        const uint32_t pitch = t_->inverseSourceRowElements;
        const uint32_t n = rows * lanes;
        DataCopyExtParams dp{static_cast<uint16_t>(rows), static_cast<uint32_t>(validD_ * sizeof(T)),
            static_cast<uint32_t>((t_->destinationAxis - validD_) * sizeof(T)),
            static_cast<uint32_t>((lanes - validD_) * sizeof(T) / 32), 0};
        DataCopyExtParams sp{static_cast<uint16_t>(rows), static_cast<uint32_t>(t_->indexCount * sizeof(T)), 0,
            static_cast<uint32_t>((pitch * sizeof(T) - AlignUp(t_->indexCount * sizeof(T), 32)) / 32), 0};
        DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
        DataCopyPad(dst_, self_[first * t_->destinationAxis + startD_], dp, pad);
        DataCopyPad(src_, source_[first * t_->indexCount], sp, pad);
        Sync<HardEvent::MTE2_V>();
        if constexpr (ClipWaves) {
            if constexpr (IsSame<T, bfloat16_t>::value) {
                auto wide = convertBuffer_.Get<float>();
                Cast(wide, dst_, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
                for (uint32_t w = 0; w < waveCount_; ++w) {
                    const uint32_t begin = waveBegin_[w] / 64 * 64;
                    const uint32_t span = AlignUp(waveEnd_[w], 64) - begin;
                    for (uint32_t r = 0; r < rows; ++r) {
                        Gather(gathered_[r * lanes + begin], src_[r * pitch], waves_[w * lanes + begin], 0, span);
                    }
                    PipeBarrier<PIPE_V>();
                    if (span == lanes) {
                        Cast(wide[n], gathered_, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
                        Add(wide, wide, wide[n], n); PipeBarrier<PIPE_V>();
                        Cast(dst_, wide, RoundMode::CAST_RINT, n); PipeBarrier<PIPE_V>();
                        Cast(wide, dst_, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
                    } else {
                        for (uint32_t r = 0; r < rows; ++r) { Cast(wide[n + r * lanes + begin], gathered_[r * lanes + begin], RoundMode::CAST_NONE, span); }
                        PipeBarrier<PIPE_V>();
                        for (uint32_t r = 0; r < rows; ++r) { Add(wide[r * lanes + begin], wide[r * lanes + begin], wide[n + r * lanes + begin], span); }
                        PipeBarrier<PIPE_V>();
                        for (uint32_t r = 0; r < rows; ++r) { Cast(dst_[r * lanes + begin], wide[r * lanes + begin], RoundMode::CAST_RINT, span); }
                        PipeBarrier<PIPE_V>();
                        for (uint32_t r = 0; r < rows; ++r) { Cast(wide[r * lanes + begin], dst_[r * lanes + begin], RoundMode::CAST_NONE, span); }
                        PipeBarrier<PIPE_V>();
                    }
                }
            } else {
                for (uint32_t w = 0; w < waveCount_; ++w) {
                    const uint32_t begin = waveBegin_[w] / 64 * 64;
                    const uint32_t span = AlignUp(waveEnd_[w], 64) - begin;
                    for (uint32_t r = 0; r < rows; ++r) {
                        Gather(gathered_[r * lanes + begin], src_[r * pitch], waves_[w * lanes + begin], 0, span);
                    }
                    PipeBarrier<PIPE_V>();
                    if (span == lanes) { Add(dst_, dst_, gathered_, n); }
                    else { for (uint32_t r = 0; r < rows; ++r) { Add(dst_[r * lanes + begin], dst_[r * lanes + begin], gathered_[r * lanes + begin], span); } }
                    PipeBarrier<PIPE_V>();
                }
            }
        } else if constexpr (IsSame<T, bfloat16_t>::value) {
            auto wide = convertBuffer_.Get<float>();
            Cast(wide, dst_, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
            for (uint32_t w = 0; w < waveCount_; ++w) {
                for (uint32_t r = 0; r < rows; ++r) {
                    Gather(gathered_[r * lanes], src_[r * pitch], waves_[w * lanes], 0, lanes);
                }
                PipeBarrier<PIPE_V>();
                Cast(wide[n], gathered_, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
                Add(wide, wide, wide[n], n); PipeBarrier<PIPE_V>();
                Cast(dst_, wide, RoundMode::CAST_RINT, n); PipeBarrier<PIPE_V>();
                Cast(wide, dst_, RoundMode::CAST_NONE, n); PipeBarrier<PIPE_V>();
            }
        } else {
            for (uint32_t w = 0; w < waveCount_; ++w) {
                for (uint32_t r = 0; r < rows; ++r) {
                    Gather(gathered_[r * lanes], src_[r * pitch], waves_[w * lanes], 0, lanes);
                }
                PipeBarrier<PIPE_V>();
                Add(dst_, dst_, gathered_, n); PipeBarrier<PIPE_V>();
            }
        }
        if (tailCount_ != 0) {
            Sync<HardEvent::V_S>();
            auto tails = indices_.template ReinterpretCast<uint32_t>();
            for (uint32_t r = 0; r < rows; ++r) {
                for (uint32_t k = 0; k < tailCount_; ++k) {
                    const uint32_t packed = tails.GetValue(k);
                    const uint32_t d = r * lanes + (packed >> 16);
                    const uint32_t s = r * pitch + (packed & 0xffff);
                    if constexpr (IsSame<T, int32_t>::value) {
                        auto du = dst_.template ReinterpretCast<uint32_t>();
                        auto su = src_.template ReinterpretCast<uint32_t>();
                        du.SetValue(d, du.GetValue(d) + su.GetValue(s));
                    } else if constexpr (IsSame<T, bfloat16_t>::value) {
                        auto du = dst_.template ReinterpretCast<uint16_t>();
                        auto su = src_.template ReinterpretCast<uint16_t>();
                        union FloatBits { uint32_t bits; float value; };
                        FloatBits left, right, sum;
                        left.bits = static_cast<uint32_t>(du.GetValue(d)) << 16;
                        right.bits = static_cast<uint32_t>(su.GetValue(s)) << 16;
                        sum.value = left.value + right.value;
                        if ((sum.bits & 0x7fffffffu) > 0x7f800000u) { sum.bits |= 0x00400000u; }
                        else { sum.bits += 0x7fffu + ((sum.bits >> 16) & 1u); }
                        du.SetValue(d, static_cast<uint16_t>(sum.bits >> 16));
                    } else {
                        const float a = static_cast<float>(dst_.GetValue(d));
                        const float b = static_cast<float>(src_.GetValue(s));
                        dst_.SetValue(d, static_cast<T>(a + b));
                    }
                }
            }
            Sync<HardEvent::S_MTE3>();
        } else { Sync<HardEvent::V_MTE3>(); }
        DataCopyExtParams op{static_cast<uint16_t>(rows), static_cast<uint32_t>(validD_ * sizeof(T)),
            static_cast<uint32_t>((lanes - validD_) * sizeof(T) / 32),
            static_cast<uint32_t>((t_->destinationAxis - validD_) * sizeof(T)), 0};
        DataCopyPad(out_[first * t_->destinationAxis + startD_], dst_, op);
        Sync<HardEvent::MTE3_MTE2>();
    }
    IndexAddTilingData local_;
    const IndexAddTilingData* t_; TPipe* p_;
    GlobalTensor<T> self_, source_, out_; GlobalTensor<int32_t> index_;
    TBuf<QuePosition::VECCALC> indexBuffer_, sourceBuffer_, destinationBuffer_, gatherBuffer_, countBuffer_, waveBuffer_, convertBuffer_;
    LocalTensor<int32_t> indices_; LocalTensor<T> src_, dst_, gathered_; LocalTensor<uint32_t> counts_, waves_;
    uint32_t startD_, validD_, waveCount_=0, tailCount_=0;
    uint32_t waveBegin_[16], waveEnd_[16];
};

__aicore__ inline bool IndexAddDiffuseAxis(GM_ADDR index, uint64_t m, uint64_t d, TPipe* pipe)
{
    // Eight 64-element strata distinguish diffuse indices from common hot
    // subsets without constructing an M-element histogram on every core.
    const uint32_t samples = 512;
    const uint32_t bitmapWords = AlignUp((d + 31) / 32, 8);
    TBuf<QuePosition::VECCALC> sampleBuffer;
    pipe->InitBuffer(sampleBuffer, samples * 4 + bitmapWords * 4);
    auto sample = sampleBuffer.Get<int32_t>();
    auto bitmap = sampleBuffer.Get<uint32_t>()[samples];
    Duplicate(bitmap, static_cast<uint32_t>(0), bitmapWords);
    GlobalTensor<int32_t> gm; gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index), m);
    const uint32_t step = m / 8;
    DataCopyExtParams p{8,256,static_cast<uint32_t>((step - 64) * 4),0,0};
    DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(sample, gm, p, pad);
    const event_t input = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE2_S));
    const event_t zero = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::MTE2_S>(input); WaitFlag<HardEvent::MTE2_S>(input);
    SetFlag<HardEvent::V_S>(zero); WaitFlag<HardEvent::V_S>(zero);
    uint32_t unique = 0;
    for (uint32_t i = 0; i < samples; ++i) {
        const uint32_t v = sample.GetValue(i);
        if (v >= d) { return false; }
        const uint32_t word = v >> 5;
        const uint32_t bit = static_cast<uint32_t>(1) << (v & 31);
        const uint32_t previous = bitmap.GetValue(word);
        if ((previous & bit) == 0) { ++unique; bitmap.SetValue(word, previous | bit); }
        if (i >= 63 && ((i + 1) & i) == 0 && unique * 8 < (i + 1) * 7) { return false; }
    }
    return unique >= 448;
}
