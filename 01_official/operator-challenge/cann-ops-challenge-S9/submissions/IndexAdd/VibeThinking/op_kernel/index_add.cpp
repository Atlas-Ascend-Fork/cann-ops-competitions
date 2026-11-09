#include "kernel_operator.h"

#include <type_traits>

namespace {
using namespace AscendC;

template <HardEvent event>
__aicore__ inline void Sync()
{
    const TEventID id = GetTPipePtr()->FetchEventID<event>();
    SetFlag<event>(id);
    WaitFlag<event>(id);
}

__aicore__ inline uint64_t ReadShift(const IndexAddTilingData& t)
{
    return (static_cast<uint64_t>(t.l2High) << 32) | t.l2Low;
}

__aicore__ inline void OwnedSlices(const IndexAddTilingData& t, uint64_t& first, uint64_t& last)
{
    const uint64_t cores = GetBlockNum();
    const uint64_t core = GetBlockIdx();
    const uint64_t units = (static_cast<uint64_t>(t.outer) + t.unit - 1) / t.unit;
    const uint64_t base = units / cores;
    const uint64_t extra = units % cores;
    const uint64_t start = core * base + (core < extra ? core : extra);
    const uint64_t count = base + (core < extra ? 1 : 0);
    first = start * t.unit;
    last = (start + count) * t.unit;
    if (last > t.outer) {
        last = t.outer;
    }
    if (first > last) {
        first = last;
    }
}

union FloatBits {
    uint32_t u;
    float f;
};

__aicore__ inline float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h >> 15) << 31;
    int32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    FloatBits b;
    if (exp == 0x1f) {
        b.u = sign | 0x7f800000u | (mant << 13);
    } else if (exp == 0) {
        if (mant == 0) {
            b.u = sign;
        } else {
            int32_t e = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x3ff;
            b.u = sign | (static_cast<uint32_t>(e - 15 + 127) << 23) | (mant << 13);
        }
    } else {
        b.u = sign | (static_cast<uint32_t>(exp - 15 + 127) << 23) | (mant << 13);
    }
    return b.f;
}

__aicore__ inline uint32_t RoundShift(uint32_t mant, uint32_t shift)
{
    if (shift == 0) {
        return mant;
    }
    if (shift > 31) {
        return 0;
    }
    const uint32_t kept = mant >> shift;
    const uint32_t rest = mant & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rest > half || (rest == half && (kept & 1u))) {
        return kept + 1;
    }
    return kept;
}

__aicore__ inline uint16_t FloatToFp16(float v)
{
    FloatBits b;
    b.f = v;
    const uint16_t sign = static_cast<uint16_t>((b.u >> 16) & 0x8000);
    const int32_t exp = static_cast<int32_t>((b.u >> 23) & 0xff);
    const uint32_t mant = b.u & 0x7fffff;
    if (exp == 0xff) {
        return static_cast<uint16_t>(sign | 0x7c00 | (mant != 0 ? 0x200 : 0));
    }
    if (exp == 0) {
        return sign;
    }
    const int32_t e = exp - 127 + 15;
    if (e >= 0x1f) {
        return static_cast<uint16_t>(sign | 0x7c00);
    }
    if (e >= 1) {
        uint32_t m = RoundShift(mant, 13);
        int32_t ee = e;
        if (m == 0x400) {
            m = 0;
            ++ee;
            if (ee >= 0x1f) {
                return static_cast<uint16_t>(sign | 0x7c00);
            }
        }
        return static_cast<uint16_t>(sign | (ee << 10) | m);
    }

    const uint32_t shift = static_cast<uint32_t>(14 - e);
    const uint32_t m = RoundShift(mant | 0x800000, shift);
    return static_cast<uint16_t>(sign | m);
}

__aicore__ inline float Bf16ToFloat(uint16_t h)
{
    FloatBits b;
    b.u = (static_cast<uint32_t>(h >> 15) << 31) | (static_cast<uint32_t>((h >> 7) & 0xff) << 23) |
          (static_cast<uint32_t>(h & 0x7f) << 16);
    return b.f;
}

__aicore__ inline uint16_t FloatToBf16(float v)
{
    FloatBits b;
    b.f = v;
    const uint16_t sign = static_cast<uint16_t>((b.u >> 16) & 0x8000);
    uint32_t exp = (b.u >> 23) & 0xff;
    const uint32_t mant = b.u & 0x7fffff;
    if (exp == 0xff) {
        return static_cast<uint16_t>(sign | 0x7f80 | (mant != 0 ? 1 : 0));
    }
    uint32_t m = RoundShift(mant, 16);
    if (m == 0x80) {
        m = 0;
        ++exp;
        if (exp == 0xff) {
            return static_cast<uint16_t>(sign | 0x7f80);
        }
    }
    return static_cast<uint16_t>(sign | (exp << 7) | m);
}

enum class Kind { kInt8, kInt32, kFloat, kFp16, kBf16 };

// 通用路径先复制 self，再按 index 顺序累加，保留重复索引的更新次序。
template <typename E, Kind K>
class ScalarKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(source));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ E*>(out));
    }

    __aicore__ inline void Process()
    {
        uint64_t first = 0;
        uint64_t last = 0;
        OwnedSlices(tiling_, first, last);
        if (first >= last) {
            return;
        }
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t inner = tiling_.inner;
        const uint64_t indexLen = tiling_.indexLen;
        const uint64_t end = last * selfLen * inner;
        for (uint64_t k = first * selfLen * inner; k < end; ++k) {
            outGm_.SetValue(k, selfGm_.GetValue(k));
        }
        for (uint64_t i = 0; i < indexLen; ++i) {
            const uint64_t row = static_cast<uint64_t>(static_cast<uint32_t>(indexGm_.GetValue(i)));
            for (uint64_t o = first; o < last; ++o) {
                const uint64_t to = (o * selfLen + row) * inner;
                const uint64_t from = (o * indexLen + i) * inner;
                for (uint64_t k = 0; k < inner; ++k) {
                    outGm_.SetValue(to + k, Combine(outGm_.GetValue(to + k), sourceGm_.GetValue(from + k)));
                }
            }
        }
    }

private:
    __aicore__ inline E Combine(E a, E b) const
    {
        if constexpr (K == Kind::kInt8) {
            return static_cast<E>(static_cast<uint8_t>(static_cast<int32_t>(a) + static_cast<int32_t>(b)));
        } else if constexpr (K == Kind::kInt32) {
            return static_cast<E>(static_cast<uint32_t>(a) + static_cast<uint32_t>(b));
        } else if constexpr (K == Kind::kFloat) {
            return a + b;
        } else if constexpr (K == Kind::kFp16) {
            return FloatToFp16(Fp16ToFloat(a) + Fp16ToFloat(b));
        } else {
            return FloatToBf16(Bf16ToFloat(a) + Bf16ToFloat(b));
        }
    }

    IndexAddTilingData tiling_;
    GlobalTensor<E> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<E> sourceGm_;
    GlobalTensor<E> outGm_;
};

template <typename T>
class RowKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;
        const uint32_t lanes = 32 / sizeof(T);
        pitch_ = (tiling.inner + lanes - 1) / lanes * lanes;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));

        selfGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        sourceGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(rowsBuf_, tiling.selfLen * pitch_ * sizeof(T));
        pipe_.InitBuffer(rowBuf_, pitch_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        uint64_t first = 0;
        uint64_t last = 0;
        OwnedSlices(tiling_, first, last);
        if (first >= last) {
            return;
        }
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t inner = tiling_.inner;
        const uint64_t indexLen = tiling_.indexLen;
        const int32_t count = static_cast<int32_t>(inner);
        LocalTensor<T> rows = rowsBuf_.Get<T>();
        LocalTensor<T> row = rowBuf_.Get<T>();
        for (uint64_t o = first; o < last; ++o) {
            for (uint64_t r = 0; r < selfLen; ++r) {
                Load(rows[r * pitch_], selfGm_, (o * selfLen + r) * inner);
            }
            Sync<HardEvent::MTE2_V>();
            for (uint64_t i = 0; i < indexLen; ++i) {
                const uint64_t target = static_cast<uint64_t>(static_cast<uint32_t>(indexGm_.GetValue(i)));
                Load(row, sourceGm_, (o * indexLen + i) * inner);
                Sync<HardEvent::MTE2_V>();
                Add(rows[target * pitch_], rows[target * pitch_], row, count);

                Sync<HardEvent::V_MTE2>();
            }
            Sync<HardEvent::V_MTE3>();
            for (uint64_t r = 0; r < selfLen; ++r) {
                DataCopyExtParams store{1, static_cast<uint32_t>(inner * sizeof(T)), 0, 0, 0};
                DataCopyPad(outGm_[(o * selfLen + r) * inner], rows[r * pitch_], store);
            }

            Sync<HardEvent::MTE3_MTE2>();
        }
    }

private:
    __aicore__ inline void Load(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint64_t offset)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(tiling_.inner * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], params, pad);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rowsBuf_;
    TBuf<TPosition::VECCALC> rowBuf_;
    IndexAddTilingData tiling_;
    uint32_t pitch_ = 0;
    GlobalTensor<T> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outGm_;
};

constexpr uint32_t kSlotCap = 20;
constexpr uint32_t kSlotCols = 64;
constexpr uint32_t kLayerCap = 64;
constexpr uint32_t kHistGroup = 16;

template <typename T>
class LayerKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out, GM_ADDR workspace,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;
        const uint64_t shift = ReadShift(tiling);
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(reinterpret_cast<uint64_t>(source) + shift));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(GetUserWorkspace(workspace)));
        wsByteGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(GetUserWorkspace(workspace)));

        if (tiling.path != 8) {
            selfGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        if (shift == 0) {
            sourceGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        const uint32_t selfLen = tiling.selfLen;
        const uint32_t indexLen = tiling.indexLen;
        const uint32_t slab = tiling.slabRows;
        const uint32_t bytes = static_cast<uint32_t>(sizeof(T));
        align_ = 32 / bytes;
        pitch_ = (selfLen + align_ - 1) / align_ * align_;
        binSlots_ = (selfLen + 63) / 64 * 64;
        valueSlots_ = (indexLen + (align_ - 1) * kLayerCap + align_ + 7) / 8 * 8;

        stageHalf_ = (slab * indexLen * bytes + 31) / 32 * 32 / bytes;
        if (stageHalf_ * bytes < indexLen * 4) {
            stageHalf_ = (indexLen * 4 + 31) / 32 * 32 / bytes;
        }
        selfHalf_ = (slab * selfLen * bytes + 31) / 32 * 32 / bytes;
        if (Lanes() && tiling.path == 8) {

            lanePitch_ = (selfLen + 15) / 16 * 16;
            laneSrc_ = (indexLen + 15) / 16 * 16;
            stageHalf_ = align_ * laneSrc_;
            selfHalf_ = align_ * lanePitch_;
            pipe_.InitBuffer(stageBuf_, 2 * stageHalf_ * bytes);
            pipe_.InitBuffer(selfBuf_, 2 * selfHalf_ * bytes);
            pipe_.InitBuffer(transBuf_, laneSrc_ * 32 + 32);
            pipe_.InitBuffer(slotBuf_, (kSlotCap * lanePitch_ + 8 * 64) * 4 + 32);
            pipe_.InitBuffer(accBuf_, lanePitch_ * 32);
            pipe_.InitBuffer(permBuf_, lanePitch_ * 32);
            pipe_.InitBuffer(outBuf_, lanePitch_ * 32);
            pipe_.InitBuffer(packMaskBuf_, lanePitch_ * 4);
        } else {
            pipe_.InitBuffer(stageBuf_, 2 * stageHalf_ * bytes + 32);
            pipe_.InitBuffer(selfBuf_, 2 * selfHalf_ * bytes + 32);
            pipe_.InitBuffer(outBuf_, slab * pitch_ * bytes + 32);
            pipe_.InitBuffer(permBuf_, slab * pitch_ * bytes + 32);
            pipe_.InitBuffer(valueBuf_, slab * valueSlots_ * bytes + 32);
            pipe_.InitBuffer(permTableBuf_, slab * pitch_ * 4 + 32);
            pipe_.InitBuffer(outTableBuf_, slab * pitch_ * 4 + 32);
        }
        if (Lanes() && tiling.path == 8) {
            return;
        }
        pipe_.InitBuffer(countBuf_, binSlots_ * 4);
        pipe_.InitBuffer(prefixBuf_, binSlots_ * 4);
        pipe_.InitBuffer(histBuf_, kHistGroup * binSlots_);
        pipe_.InitBuffer(halfBuf_, 3 * binSlots_ * 2);
        pipe_.InitBuffer(floatBuf_, (4 * binSlots_ + 72) * 4);
        pipe_.InitBuffer(orderBuf_, binSlots_ * 4);
        pipe_.InitBuffer(inverseBuf_, binSlots_ * 4);
        pipe_.InitBuffer(layerBuf_, valueSlots_ * 4);
    }

    __aicore__ inline static bool Lanes()
    {
        return sizeof(T) == 2;
    }

    __aicore__ inline bool Plan()
    {
        const uint32_t selfLen = tiling_.selfLen;
        const uint32_t indexLen = tiling_.indexLen;
        const uint32_t bytes = static_cast<uint32_t>(sizeof(T));
        const uint32_t cores = static_cast<uint32_t>(GetBlockNum());
        const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t histBase = 0;

        const uint32_t orderBase = cores * binSlots_ / 4;
        const uint32_t inverseBase = orderBase + binSlots_;
        const uint32_t layerBase = inverseBase + binSlots_;
        const uint32_t first = Share(indexLen, cores, core);
        const uint32_t last = Share(indexLen, cores, core + 1);
        const uint32_t binFirst = Share(selfLen, cores, core);
        const uint32_t binLast = Share(selfLen, cores, core + 1);

        LocalTensor<int32_t> idxTensor = stageBuf_.Get<int32_t>();
        LocalTensor<int32_t> count = countBuf_.Get<int32_t>();
        LocalTensor<int32_t> layer = layerBuf_.Get<int32_t>();
        if (last > first) {
            LoadInt(idxTensor, indexGm_[first], last - first);
        }
        Duplicate(count, 0, static_cast<int32_t>(binSlots_));
        LocalTensor<int32_t> order = orderBuf_.Get<int32_t>();
        LocalTensor<int32_t> inverse = inverseBuf_.Get<int32_t>();
        if (core == 0) {
            Duplicate(order, 0, static_cast<int32_t>(binSlots_));
            Duplicate(layer, 0, static_cast<int32_t>(valueSlots_));
            Sync<HardEvent::V_MTE3>();
            DataCopy(wsGm_[orderBase], order, binSlots_);
            DataCopy(wsGm_[inverseBase], order, binSlots_);
            DataCopy(wsGm_[layerBase], layer, valueSlots_);
        }
        Sync<HardEvent::MTE2_S>();
        PipeBarrier<PIPE_V>();

        const __ubuf__ int32_t* idx = reinterpret_cast<__ubuf__ int32_t*>(idxTensor.GetPhyAddr());
        __ubuf__ int32_t* hits = reinterpret_cast<__ubuf__ int32_t*>(count.GetPhyAddr());
        __ubuf__ uint8_t* hitBytes = reinterpret_cast<__ubuf__ uint8_t*>(count.GetPhyAddr());
        for (uint32_t k = first; k < last; ++k) {

            const uint32_t v = static_cast<uint32_t>(idx[k - first]);
            if (hitBytes[v] <= kLayerCap) {
                ++hitBytes[v];
            }
        }
        DataCopy(wsByteGm_[histBase + core * binSlots_], countBuf_.Get<int8_t>(), binSlots_);
        Sync<HardEvent::MTE3_S>();
        SyncAll<true>();

        LocalTensor<int32_t> prefix = prefixBuf_.Get<int32_t>();
        LocalTensor<int8_t> hist = histBuf_.Get<int8_t>();
        LocalTensor<half> halves = halfBuf_.Get<half>();
        LocalTensor<half> totalHalf = halves[binSlots_];
        LocalTensor<half> prefixHalf = halves[2 * binSlots_];
        const int32_t binCount = static_cast<int32_t>(binSlots_);
        Duplicate(totalHalf, static_cast<half>(0), binCount);
        Duplicate(prefixHalf, static_cast<half>(0), binCount);
        for (uint32_t g = 0; g < cores; g += kHistGroup) {
            const uint32_t n = cores - g < kHistGroup ? cores - g : kHistGroup;
            DataCopy(hist, wsByteGm_[histBase + g * binSlots_], n * binSlots_);
            Sync<HardEvent::MTE2_V>();
            for (uint32_t j = 0; j < n; ++j) {
                Cast(halves, hist[j * binSlots_], RoundMode::CAST_NONE, binSlots_);
                Add(totalHalf, totalHalf, halves, binCount);
                if (g + j < core) {
                    Add(prefixHalf, prefixHalf, halves, binCount);
                }
            }
            Sync<HardEvent::V_MTE2>();
        }
        Cast(count, totalHalf, RoundMode::CAST_RINT, binSlots_);
        Cast(prefix, prefixHalf, RoundMode::CAST_RINT, binSlots_);
        LocalTensor<float> fl = floatBuf_.Get<float>();
        LocalTensor<float> total = fl;
        LocalTensor<float> range = fl[binSlots_];
        LocalTensor<float> work = fl[2 * binSlots_];
        LocalTensor<float> sums = fl[3 * binSlots_];
        const int32_t slots = static_cast<int32_t>(binSlots_);
        Cast(total, count, RoundMode::CAST_RINT, binSlots_);

        __ubuf__ float* ramp = reinterpret_cast<__ubuf__ float*>(range.GetPhyAddr());
        for (uint32_t v = 0; v < 8; ++v) {
            ramp[v] = static_cast<float>(static_cast<int32_t>(v));
        }
        for (uint32_t built = 8; built < binSlots_; built *= 2) {
            Adds(range[built], range, static_cast<float>(static_cast<int32_t>(built)), static_cast<int32_t>(built));
        }
        Muls(range, range, -1.0f, slots);
        Adds(range, range, static_cast<float>(static_cast<int32_t>(binFirst)), slots);
        Maxs(range, range, 0.0f, slots);
        Mins(range, range, 1.0f, slots);

        above_[0] = selfLen;
        before_[0] = binFirst;
        uint32_t layers = 0;
        while (true) {
            Adds(work, total, -static_cast<float>(static_cast<int32_t>(layers)), slots);
            Maxs(work, work, 0.0f, slots);
            Mins(work, work, 1.0f, slots);
            const uint32_t above = Sum(sums, work);
            Mul(work, work, range, slots);
            const uint32_t before = Sum(sums, work);
            above_[layers + 1] = above;
            before_[layers + 1] = before;
            if (above == 0) {
                break;
            }
            ++layers;
            if (layers > kLayerCap) {
                break;
            }
        }
        const bool fits = layers <= kLayerCap;
        Duplicate(order, 0, slots);
        Duplicate(inverse, 0, slots);
        PipeBarrier<PIPE_V>();
        if (fits) {
            __ubuf__ int32_t* totals = hits;
            __ubuf__ int32_t* ord = reinterpret_cast<__ubuf__ int32_t*>(order.GetPhyAddr());
            __ubuf__ int32_t* inv = reinterpret_cast<__ubuf__ int32_t*>(inverse.GetPhyAddr());
            for (uint32_t c = 0; c <= layers; ++c) {
                seen_[c] = 0;
            }
            for (uint32_t v = binFirst; v < binLast; ++v) {
                const uint32_t c = static_cast<uint32_t>(totals[v]);
                const uint32_t r = above_[c + 1] + (before_[c] - before_[c + 1]) + seen_[c]++;
                ord[r] = static_cast<int32_t>(v * bytes);
                inv[v] = static_cast<int32_t>(r);
            }
            AddInto(wsGm_[orderBase], order, binSlots_);
            AddInto(wsGm_[inverseBase], inverse, binSlots_);
        }
        Sync<HardEvent::MTE3_S>();
        SyncAll<true>();
        if (!fits) {
            return false;
        }

        uint32_t offset = 0;
        for (uint32_t l = 0; l < layers; ++l) {
            size_[l] = above_[l + 1];
            start_[l] = offset;
            offset += (size_[l] + align_ - 1) / align_ * align_;
        }
        layers_ = layers;
        values_ = offset;
        DataCopy(inverse, wsGm_[inverseBase], binSlots_);
        Duplicate(count, 0, slots);
        Duplicate(layer, 0, static_cast<int32_t>(valueSlots_));
        Sync<HardEvent::MTE2_S>();
        PipeBarrier<PIPE_V>();
        {
            const __ubuf__ int32_t* pre = reinterpret_cast<__ubuf__ int32_t*>(prefix.GetPhyAddr());
            const __ubuf__ int32_t* inv = reinterpret_cast<__ubuf__ int32_t*>(inverse.GetPhyAddr());
            __ubuf__ int32_t* seen = hits;
            __ubuf__ int32_t* slot = reinterpret_cast<__ubuf__ int32_t*>(layer.GetPhyAddr());
            for (uint32_t k = first; k < last; ++k) {
                const uint32_t v = static_cast<uint32_t>(idx[k - first]);
                const uint32_t l = static_cast<uint32_t>(pre[v] + seen[v]++);
                slot[start_[l] + static_cast<uint32_t>(inv[v])] = static_cast<int32_t>(k * bytes);
            }
        }
        AddInto(wsGm_[layerBase], layer, valueSlots_);
        Sync<HardEvent::MTE3_S>();
        SyncAll<true>();

        DataCopy(order, wsGm_[orderBase], binSlots_);
        DataCopy(inverse, wsGm_[inverseBase], binSlots_);
        DataCopy(layer, wsGm_[layerBase], valueSlots_);
        Sync<HardEvent::MTE2_V>();
        Muls(inverse, inverse, static_cast<int32_t>(bytes), slots);
        return true;
    }

    __aicore__ inline bool PlanSlots(uint64_t first, uint64_t last)
    {
        const uint32_t selfLen = tiling_.selfLen;
        const uint32_t indexLen = tiling_.indexLen;
        const uint32_t cores = static_cast<uint32_t>(GetBlockNum());
        const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t cols = lanePitch_;
        const uint32_t blockCount = cols / 8;
        const uint32_t producers = blockCount < cores ? blockCount : cores;
        const uint32_t headerAt = kSlotCap * cols;
        const uint32_t colFirst = Share(blockCount, cores, core) * 8;
        const uint32_t colLast = Share(blockCount, cores, core + 1) * 8;
        const uint32_t owned = colLast - colFirst;
        if (owned > 0) {
            const uint32_t n = (indexLen + 63) / 64 * 64;
            const int32_t laneCount = static_cast<int32_t>(n);
            const uint32_t unit = (n * 4 + 31) / 32 * 32;
            const uint32_t laneAt = 5 * unit;
            const LocalTensor<int32_t> idxInt = StageInts(0);
            const LocalTensor<float> idx = StageFloats(unit);
            const LocalTensor<float> record = StageFloats(2 * unit);
            const LocalTensor<float> kept = StageFloats(3 * unit);
            const LocalTensor<int32_t> found = StageInts(4 * unit);
            const LocalTensor<int32_t> laneTable = StageInts(laneAt);
            const LocalTensor<int32_t> header = StageInts(laneAt + kSlotCap * kSlotCols * 4);
            const LocalTensor<uint8_t> low = packMaskBuf_.Get<uint8_t>();
            const LocalTensor<uint8_t> high = low[(n / 8 + 31) / 32 * 32];
            Duplicate(idxInt, static_cast<int32_t>(1 << 24), laneCount);
            Duplicate(laneTable, static_cast<int32_t>(laneSrc_ * 32), static_cast<int32_t>(kSlotCap * kSlotCols));
            Duplicate(header, 0, 8);
            Sync<HardEvent::V_MTE2>();
            LoadInt(idxInt, indexGm_[0], indexLen);
            const TEventID indexReady = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
            SetFlag<HardEvent::MTE2_V>(indexReady);
            Preload(first, last, 5 * unit + kSlotCap * kSlotCols * 4 + 32);
            WaitFlag<HardEvent::MTE2_V>(indexReady);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(indexReady);
            Cast(idx, idxInt, RoundMode::CAST_RINT, n);
            Muls(record, idx, -1.0f, laneCount);
            Maxs(record, record, 0.0f, laneCount);
            Mins(record, record, 1.0f, laneCount);
            Muls(record, record, static_cast<float>(static_cast<int32_t>(selfLen)), laneCount);
            Add(idx, idx, record, laneCount);
            CompareScalar(low, idx, static_cast<float>(static_cast<int32_t>(colFirst)), CMPMODE::GE, n);
            CompareScalar(high, idx, static_cast<float>(static_cast<int32_t>(colLast)), CMPMODE::LT, n);
            And(low.template ReinterpretCast<uint16_t>(), low.template ReinterpretCast<uint16_t>(),
                high.template ReinterpretCast<uint16_t>(), n / 16);

            CreateVecIndex(record, 0.0f, n);
            PipeBarrier<PIPE_V>();
            CompareScalar(high, record, static_cast<float>(static_cast<int32_t>(indexLen)), CMPMODE::LT, n);
            And(low.template ReinterpretCast<uint16_t>(), low.template ReinterpretCast<uint16_t>(),
                high.template ReinterpretCast<uint16_t>(), n / 16);
            Muls(record, record, 1024.0f, laneCount);
            Add(record, record, idx, laneCount);
            Adds(record, record, -static_cast<float>(static_cast<int32_t>(colFirst)), laneCount);
            uint64_t selected = 0;
            GatherMask(kept, record, low.template ReinterpretCast<uint32_t>(), true, n, {1, 1, 8, 0}, selected);
            Sync<HardEvent::V_S>();
            const uint32_t hits = static_cast<uint32_t>(selected);
            if (hits > 0) {
                Cast(found, kept, RoundMode::CAST_RINT, hits);
                Sync<HardEvent::V_S>();
            }
            const __ubuf__ int32_t* rec = reinterpret_cast<__ubuf__ int32_t*>(found.GetPhyAddr());
            __ubuf__ int32_t* lane = reinterpret_cast<__ubuf__ int32_t*>(laneTable.GetPhyAddr());
            uint32_t seen[kSlotCols];
            for (uint32_t j = 0; j < owned; ++j) {
                seen[j] = 0;
            }
            uint32_t deepest = 0;
            for (uint32_t h = 0; h < hits; ++h) {
                const uint32_t r = static_cast<uint32_t>(rec[h]);
                const uint32_t j = r & 1023;
                const uint32_t c = seen[j];
                if (c < kSlotCap) {
                    lane[c * kSlotCols + j] = static_cast<int32_t>((r >> 10) * 32);
                }
                seen[j] = c + 1;
                deepest = c + 1 > deepest ? c + 1 : deepest;
            }
            reinterpret_cast<__ubuf__ int32_t*>(header.GetPhyAddr())[0] = static_cast<int32_t>(deepest);
            for (uint32_t w = 0; w < kSlotCap; ++w) {
                DataCopy(wsGm_[w * cols + colFirst], laneTable[w * kSlotCols], owned);
            }
            DataCopy(wsGm_[headerAt + core * 8], header, 8);
        } else {
            Preload(first, last, 0);
        }
        Sync<HardEvent::MTE3_S>();
        SyncAll<true>();
        const LocalTensor<int32_t> table = slotBuf_.Get<int32_t>();
        const LocalTensor<int32_t> headers = table[headerAt];
        DataCopy(table, wsGm_[0], kSlotCap * cols);
        DataCopy(headers, wsGm_[headerAt], producers * 8);
        Sync<HardEvent::MTE2_S>();
        uint32_t waves = 0;
        for (uint32_t p = 0; p < producers; ++p) {
            const uint32_t d = static_cast<uint32_t>(headers.GetValue(p * 8));
            waves = d > waves ? d : waves;
        }
        slotWaves_ = waves;
        return waves <= kSlotCap && producers <= 64;
    }

    __aicore__ inline void Preload(uint64_t first, uint64_t last, uint32_t scratchBytes)
    {
        const uint32_t bytes = static_cast<uint32_t>(sizeof(T));
        preloaded_ = first < last && scratchBytes <= stageHalf_ * bytes;
        if (!preloaded_) {
            return;
        }
        const uint64_t rows = (last - first) < align_ ? (last - first) : align_;
        const uint32_t selfGap = (lanePitch_ * bytes - (tiling_.selfLen * bytes + 31) / 32 * 32) / 32;
        const uint32_t srcGap = (laneSrc_ * bytes - (tiling_.indexLen * bytes + 31) / 32 * 32) / 32;
        LoadRows(selfBuf_.Get<T>()[selfHalf_], selfGm_, first * tiling_.selfLen, rows, tiling_.selfLen, selfGap);
        LoadRows(stageBuf_.Get<T>()[stageHalf_], sourceGm_, first * tiling_.indexLen, rows, tiling_.indexLen, srcGap);
    }

    __aicore__ inline LocalTensor<int32_t> StageInts(uint32_t at)
    {
        return stageBuf_.Get<uint8_t>()[at].template ReinterpretCast<int32_t>();
    }

    __aicore__ inline LocalTensor<float> StageFloats(uint32_t at)
    {
        return stageBuf_.Get<uint8_t>()[at].template ReinterpretCast<float>();
    }

    __aicore__ inline void Process(uint64_t first, uint64_t last)
    {
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t indexLen = tiling_.indexLen;
        const uint64_t slab = tiling_.slabRows;
        const uint32_t bytes = static_cast<uint32_t>(sizeof(T));
        LocalTensor<T> self = selfBuf_.Get<T>();
        LocalTensor<T> src = stageBuf_.Get<T>();
        LocalTensor<T> outRows = outBuf_.Get<T>();
        LocalTensor<T> perm = permBuf_.Get<T>();
        LocalTensor<T> value = valueBuf_.Get<T>();
        LocalTensor<uint32_t> order = orderBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> inverse = inverseBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> layer = layerBuf_.Get<uint32_t>();

        const uint32_t valueStride = values_ * bytes / 32;
        const bool batched = valueStride > 0 && valueStride <= 255 && pitch_ * bytes / 32 <= 255;
        LocalTensor<uint32_t> permTable = permTableBuf_.Get<uint32_t>();
        LocalTensor<uint32_t> outTable = outTableBuf_.Get<uint32_t>();
        if (batched) {

            LocalTensor<int32_t> permInt = permTableBuf_.Get<int32_t>();
            LocalTensor<int32_t> outInt = outTableBuf_.Get<int32_t>();
            LocalTensor<int32_t> orderInt = orderBuf_.Get<int32_t>();
            LocalTensor<int32_t> inverseInt = inverseBuf_.Get<int32_t>();
            Duplicate(permInt, 0, static_cast<int32_t>(pitch_));
            Duplicate(outInt, 0, static_cast<int32_t>(pitch_));
            Add(permInt, permInt, orderInt, static_cast<int32_t>(selfLen));
            Add(outInt, outInt, inverseInt, static_cast<int32_t>(selfLen));
            for (uint64_t g = 1; g < slab; ++g) {
                Adds(permInt[g * pitch_], permInt, static_cast<int32_t>(g * selfLen * bytes),
                     static_cast<int32_t>(pitch_));
                Adds(outInt[g * pitch_], outInt, static_cast<int32_t>(g * pitch_ * bytes),
                     static_cast<int32_t>(pitch_));
            }
        }
        TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        TEventID spent[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                             GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
        bool live[2] = {false, false};
        uint32_t cur = 0;
        if (first < last) {
            const uint64_t rows = (last - first) < slab ? (last - first) : slab;
            Load(self, selfGm_, first * selfLen, rows * selfLen);
            Load(src, sourceGm_, first * indexLen, rows * indexLen);
            SetFlag<HardEvent::MTE2_V>(full[0]);
        }
        for (uint64_t o = first; o < last;) {
            const uint64_t rows = (last - o) < slab ? (last - o) : slab;
            const uint64_t next = o + rows;
            const uint32_t nxt = 1 - cur;
            if (next < last) {
                const uint64_t nextRows = (last - next) < slab ? (last - next) : slab;
                if (live[nxt]) {
                    WaitFlag<HardEvent::V_MTE2>(spent[nxt]);
                    live[nxt] = false;
                }
                Load(self[nxt * selfHalf_], selfGm_, next * selfLen, nextRows * selfLen);
                Load(src[nxt * stageHalf_], sourceGm_, next * indexLen, nextRows * indexLen);
                SetFlag<HardEvent::MTE2_V>(full[nxt]);
            }
            WaitFlag<HardEvent::MTE2_V>(full[cur]);
            if (o != first) {

                Sync<HardEvent::MTE3_V>();
            }
            const LocalTensor<T> curSelf = self[cur * selfHalf_];
            const LocalTensor<T> curSrc = src[cur * stageHalf_];
            if (batched) {
                Gather(perm, curSelf, permTable, 0, static_cast<uint32_t>(rows * pitch_));
                for (uint64_t g = 0; g < rows; ++g) {
                    Gather(value[g * values_], curSrc, layer, static_cast<uint32_t>(g * indexLen) * bytes, values_);
                }
                const uint32_t window = 256 / bytes;
                const uint8_t permStride = static_cast<uint8_t>(pitch_ * bytes / 32);
                for (uint32_t l = 0; l < layers_; ++l) {
                    for (uint32_t w = 0; w < size_[l]; w += window) {
                        const uint32_t n = size_[l] - w < window ? size_[l] - w : window;
                        Add(perm[w], perm[w], value[start_[l] + w], static_cast<uint64_t>(n), static_cast<uint8_t>(rows),
                            {1, 1, 1, permStride, permStride, static_cast<uint8_t>(valueStride)});
                    }
                }
                Gather(outRows, perm, outTable, 0, static_cast<uint32_t>(rows * pitch_));
            } else {
                for (uint64_t g = 0; g < rows; ++g) {
                    Gather(perm, curSelf, order, static_cast<uint32_t>(g * selfLen) * bytes,
                           static_cast<uint32_t>(selfLen));
                    Gather(value, curSrc, layer, static_cast<uint32_t>(g * indexLen) * bytes, values_);
                    for (uint32_t l = 0; l < layers_; ++l) {
                        Add(perm, perm, value[start_[l]], static_cast<int32_t>(size_[l]));
                    }
                    Gather(outRows[g * pitch_], perm, inverse, 0, static_cast<uint32_t>(selfLen));
                }
            }
            SetFlag<HardEvent::V_MTE2>(spent[cur]);
            live[cur] = true;
            Sync<HardEvent::V_MTE3>();
            for (uint64_t g = 0; g < rows; ++g) {
                DataCopyExtParams store{1, static_cast<uint32_t>(selfLen) * bytes, 0, 0, 0};
                DataCopyPad(outGm_[(o + g) * selfLen], outRows[g * pitch_], store);
            }
            o = next;
            cur = nxt;
        }

        for (uint32_t b = 0; b < 2; ++b) {
            if (live[b]) {
                WaitFlag<HardEvent::V_MTE2>(spent[b]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(full[b]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(spent[b]);
        }
    }

    __aicore__ inline void ProcessLanes(uint64_t first, uint64_t last)
    {
        const uint32_t selfLen = tiling_.selfLen;
        const uint32_t indexLen = tiling_.indexLen;
        const uint32_t bytes = static_cast<uint32_t>(sizeof(T));
        const uint32_t lanes = align_;
        const int32_t slots = static_cast<int32_t>(binSlots_);

        const LocalTensor<int32_t> table = slotBuf_.Get<int32_t>();
        Mins(table, table, static_cast<int32_t>(laneSrc_ * 32), static_cast<int32_t>(kSlotCap * lanePitch_));
        Maxs(table, table, 0, static_cast<int32_t>(kSlotCap * lanePitch_));
        Duplicate(transBuf_.Get<T>()[laneSrc_ * lanes], static_cast<T>(0), static_cast<int32_t>(lanes));
        LocalTensor<T> self = selfBuf_.Get<T>();
        LocalTensor<T> src = stageBuf_.Get<T>();
        LocalTensor<T> trans = transBuf_.Get<T>();
        LocalTensor<T> acc = accBuf_.Get<T>();
        LocalTensor<T> perm = permBuf_.Get<T>();
        LocalTensor<T> out = outBuf_.Get<T>();
        LocalTensor<uint32_t> waveTable = slotBuf_.Get<uint32_t>();
        const uint32_t selfGap = (lanePitch_ * bytes - (selfLen * bytes + 31) / 32 * 32) / 32;
        const uint32_t srcGap = (laneSrc_ * bytes - (indexLen * bytes + 31) / 32 * 32) / 32;

        const bool pack = selfLen != lanePitch_;
        LocalTensor<uint16_t> packMask = packMaskBuf_.Get<uint16_t>();
        if (pack) {
            Duplicate(packMask, static_cast<uint16_t>(0xFFFF), lanePitch_ * 2);
            Sync<HardEvent::V_S>();
            for (uint32_t r = 0; r < lanes; ++r) {
                for (uint32_t c = selfLen; c < lanePitch_; ++c) {
                    const uint32_t at = r * lanePitch_ + c;
                    const uint32_t word = (at / 128) * 16 + (at % 128) / 16;
                    const uint16_t clear = static_cast<uint16_t>(~(1u << (at % 16)));
                    packMask.SetValue(word, static_cast<uint16_t>(packMask.GetValue(word) & clear));
                    packMask.SetValue(word + 8, static_cast<uint16_t>(packMask.GetValue(word + 8) & clear));
                }
            }
        }
        TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        TEventID spent[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                             GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
        bool live[2] = {false, false};

        uint32_t cur = preloaded_ ? 1 : 0;
        bool arrived = preloaded_;
        if (first < last && !preloaded_) {
            const uint64_t rows = (last - first) < lanes ? (last - first) : lanes;
            LoadRows(self, selfGm_, first * selfLen, rows, selfLen, selfGap);
            LoadRows(src, sourceGm_, first * indexLen, rows, indexLen, srcGap);
            SetFlag<HardEvent::MTE2_V>(full[0]);
        }
        for (uint64_t o = first; o < last;) {
            const uint64_t rows = (last - o) < lanes ? (last - o) : lanes;
            const uint64_t next = o + rows;
            const uint32_t nxt = 1 - cur;
            if (next < last) {
                const uint64_t nextRows = (last - next) < lanes ? (last - next) : lanes;
                if (live[nxt]) {
                    WaitFlag<HardEvent::V_MTE2>(spent[nxt]);
                    live[nxt] = false;
                }
                LoadRows(self[nxt * selfHalf_], selfGm_, next * selfLen, nextRows, selfLen, selfGap);
                LoadRows(src[nxt * stageHalf_], sourceGm_, next * indexLen, nextRows, indexLen, srcGap);
                SetFlag<HardEvent::MTE2_V>(full[nxt]);
            }
            if (!arrived) {
                WaitFlag<HardEvent::MTE2_V>(full[cur]);
            }
            arrived = false;
            if (o != first) {

                Sync<HardEvent::MTE3_V>();
            }

            ToBlocks(acc, self[cur * selfHalf_], lanePitch_);
            ToBlocks(trans, src[cur * stageHalf_], laneSrc_);

            for (uint32_t w = 0; w < slotWaves_; ++w) {
                GatherBlocks(perm, trans, waveTable[w * lanePitch_], lanePitch_);
                Add(acc, acc, perm, static_cast<int32_t>(lanePitch_ * lanes));
            }
            ToRows(out, acc, lanePitch_);
            if (pack) {
                uint64_t kept = 0;
                GatherMask(out.template ReinterpretCast<uint16_t>(), out.template ReinterpretCast<uint16_t>(), packMask,
                           false, 0, {1, static_cast<uint16_t>(lanePitch_ / 8), 8, 1}, kept);
            }
            SetFlag<HardEvent::V_MTE2>(spent[cur]);
            live[cur] = true;
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams store = pack
                ? DataCopyExtParams{1, static_cast<uint32_t>(rows) * selfLen * bytes, 0, 0, 0}
                : DataCopyExtParams{static_cast<uint16_t>(rows), selfLen * bytes, selfGap, 0, 0};
            DataCopyPad(outGm_[o * selfLen], out, store);
            o = next;
            cur = nxt;
        }
        for (uint32_t b = 0; b < 2; ++b) {
            if (live[b]) {
                WaitFlag<HardEvent::V_MTE2>(spent[b]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(full[b]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(spent[b]);
        }
    }

private:

    __aicore__ inline float SumOffsets(const LocalTensor<float>& work, const LocalTensor<int32_t>& table, uint32_t count)
    {
        const uint32_t groups = (count + 63) / 64;
        LocalTensor<float> sums = work[groups * 64];
        Duplicate(work, 0.0f, static_cast<int32_t>(groups * 64));
        Cast(work, table, RoundMode::CAST_RINT, count);
        WholeReduceSum(sums, work, 64, static_cast<int32_t>(groups), 1, 1, 8);
        WholeReduceSum(sums[64], sums, static_cast<int32_t>(groups), 1, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        return sums.GetValue(64);
    }

    __aicore__ inline void ToBlocks(const LocalTensor<T>& blocks, const LocalTensor<T>& rowsTensor, uint32_t columns)
    {
        uint64_t dst[16];
        uint64_t src[16];
        const uint8_t repeat = static_cast<uint8_t>(columns / 16);
        if constexpr (sizeof(T) == 2) {
            for (uint32_t c = 0; c < 16; ++c) {
                dst[c] = reinterpret_cast<uint64_t>(blocks[c * 16].GetPhyAddr());
                src[c] = reinterpret_cast<uint64_t>(rowsTensor[c * columns].GetPhyAddr());
            }
            TransDataTo5HD<uint16_t>(dst, src, TransDataTo5HDParams{false, false, repeat, 16, 1});
        } else {

            for (uint32_t r = 0; r < 8; ++r) {
                src[r] = reinterpret_cast<uint64_t>(rowsTensor[r * columns].GetPhyAddr());
                src[r + 8] = reinterpret_cast<uint64_t>(rowsTensor[r * columns + 8].GetPhyAddr());
                dst[2 * r] = reinterpret_cast<uint64_t>(blocks[r * 8].GetPhyAddr());
                dst[2 * r + 1] = reinterpret_cast<uint64_t>(blocks[(r + 8) * 8].GetPhyAddr());
            }
            TransDataTo5HD<uint32_t>(dst, src, TransDataTo5HDParams{false, false, repeat, 16, 2});
        }
    }

    __aicore__ inline void ToRows(const LocalTensor<T>& rowsTensor, const LocalTensor<T>& blocks, uint32_t columns)
    {
        uint64_t dst[16];
        uint64_t src[16];
        const uint8_t repeat = static_cast<uint8_t>(columns / 16);
        if constexpr (sizeof(T) == 2) {
            for (uint32_t c = 0; c < 16; ++c) {
                src[c] = reinterpret_cast<uint64_t>(blocks[c * 16].GetPhyAddr());
                dst[c] = reinterpret_cast<uint64_t>(rowsTensor[c * columns].GetPhyAddr());
            }
            TransDataTo5HD<uint16_t>(dst, src, TransDataTo5HDParams{false, false, repeat, 1, 16});
        } else {
            for (uint32_t j = 0; j < 16; ++j) {
                src[j] = reinterpret_cast<uint64_t>(blocks[j * 8].GetPhyAddr());
            }
            for (uint32_t r = 0; r < 8; ++r) {
                dst[2 * r] = reinterpret_cast<uint64_t>(rowsTensor[r * columns].GetPhyAddr());
                dst[2 * r + 1] = reinterpret_cast<uint64_t>(rowsTensor[r * columns + 8].GetPhyAddr());
            }
            TransDataTo5HD<uint32_t>(dst, src, TransDataTo5HDParams{false, false, repeat, 2, 16});
        }
    }

    __aicore__ inline void GatherBlocks(const LocalTensor<T>& dst, const LocalTensor<T>& src,
                                        const LocalTensor<uint32_t>& offsets, uint32_t count)
    {
        const uint32_t padded = (count + 7) / 8 * 8;
        for (uint32_t pos = 0; pos < padded;) {
            uint32_t calls = (padded - pos) / 8;
            if (calls > 255) {
                calls = 255;
            }
            if constexpr (sizeof(T) == 2) {
                Gatherb(dst[pos * align_].template ReinterpretCast<uint16_t>(), src.template ReinterpretCast<uint16_t>(),
                        offsets[pos], static_cast<uint8_t>(calls), GatherRepeatParams{1, 8});
            } else {
                Gatherb(dst[pos * align_].template ReinterpretCast<uint32_t>(), src.template ReinterpretCast<uint32_t>(),
                        offsets[pos], static_cast<uint8_t>(calls), GatherRepeatParams{1, 8});
            }
            pos += calls * 8;
        }
    }

    __aicore__ inline void LoadRows(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint64_t offset,
                                    uint64_t rows, uint32_t rowLen, uint32_t gap)
    {

        const uint32_t rowBytes = rowLen * static_cast<uint32_t>(sizeof(T));
        DataCopyExtParams params = gap == 0 && rowBytes % 32 == 0
            ? DataCopyExtParams{1, static_cast<uint32_t>(rows) * rowBytes, 0, 0, 0}
            : DataCopyExtParams{static_cast<uint16_t>(rows), rowBytes, 0, gap, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], params, pad);
    }

    __aicore__ inline static uint32_t Share(uint32_t total, uint32_t cores, uint32_t core)
    {
        const uint32_t base = total / cores;
        const uint32_t extra = total % cores;
        return core * base + (core < extra ? core : extra);
    }

    __aicore__ inline uint32_t Sum(const LocalTensor<float>& sums, const LocalTensor<float>& src)
    {
        const int32_t groups = static_cast<int32_t>(binSlots_ / 64);
        WholeReduceSum(sums, src, 64, groups, 1, 1, 8);
        WholeReduceSum(sums[64], sums, groups, 1, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        return static_cast<uint32_t>(static_cast<int32_t>(sums.GetValue(64) + 0.5f));
    }

    __aicore__ inline void AddInto(const GlobalTensor<int32_t>& dst, const LocalTensor<int32_t>& src, uint32_t count)
    {
        DataCopyExtParams params{1, count * 4, 0, 0, 0};
        DataCopyPadWithReduce<int32_t, ReduceType::REDUCE_ADD>(dst, src, params);
    }

    __aicore__ inline void LoadInt(const LocalTensor<int32_t>& dst, const GlobalTensor<int32_t>& src, uint32_t count)
    {
        DataCopyExtParams params{1, count * 4, 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(dst, src, params, pad);
    }

    __aicore__ inline void Load(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint64_t offset,
                                uint64_t count)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], params, pad);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> stageBuf_;
    TBuf<TPosition::VECCALC> selfBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> permBuf_;
    TBuf<TPosition::VECCALC> valueBuf_;
    TBuf<TPosition::VECCALC> countBuf_;
    TBuf<TPosition::VECCALC> prefixBuf_;
    TBuf<TPosition::VECCALC> histBuf_;
    TBuf<TPosition::VECCALC> halfBuf_;
    TBuf<TPosition::VECCALC> floatBuf_;
    TBuf<TPosition::VECCALC> orderBuf_;
    TBuf<TPosition::VECCALC> inverseBuf_;
    TBuf<TPosition::VECCALC> layerBuf_;
    TBuf<TPosition::VECCALC> permTableBuf_;
    TBuf<TPosition::VECCALC> outTableBuf_;
    TBuf<TPosition::VECCALC> transBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> packMaskBuf_;
    TBuf<TPosition::VECCALC> slotBuf_;
    IndexAddTilingData tiling_;
    uint32_t lanePitch_ = 0;
    uint32_t laneSrc_ = 0;
    uint32_t align_ = 1;
    uint32_t pitch_ = 0;
    uint32_t stageHalf_ = 0;
    bool preloaded_ = false;
    uint32_t selfHalf_ = 0;
    uint32_t binSlots_ = 0;
    uint32_t valueSlots_ = 0;
    uint32_t layers_ = 0;
    uint32_t slotWaves_ = 0;
    uint32_t values_ = 0;
    uint32_t above_[kLayerCap + 2];
    uint32_t before_[kLayerCap + 2];
    uint32_t seen_[kLayerCap + 2];
    uint32_t size_[kLayerCap];
    uint32_t start_[kLayerCap];
    GlobalTensor<T> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outGm_;
    GlobalTensor<int32_t> wsGm_;
    GlobalTensor<int8_t> wsByteGm_;
};

// 每核持有独立的目标行，source 分批读入后依次更新，避免跨核写冲突。
template <typename T>
class OrderedRowKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        selfGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        sourceGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t inner = tiling.inner;
        const uint32_t rowBytes = inner * static_cast<uint32_t>(sizeof(T));
        pitch_ = (rowBytes + 31) / 32 * 32;

        lanes_ = (rowBytes + 1) / 2;
                pipe_.InitBuffer(indexBuf_, (tiling.indexLen * 4 + 31) / 32 * 32);
        pipe_.InitBuffer(selfBuf_, (tiling.selfLen * rowBytes + 31) / 32 * 32 + 32);
        pipe_.InitBuffer(accBuf_, tiling.selfLen * pitch_ + 32);
        pipe_.InitBuffer(sourceBuf_, (tiling.slabRows * rowBytes + 31) / 32 * 32 + 32);
        pipe_.InitBuffer(rampBuf_, (lanes_ + 8) * 4);
        pipe_.InitBuffer(rowBuf_, pitch_ + 32);
        pipe_.InitBuffer(workBuf_, 4 * 2 * kWork + 64);
    }

    __aicore__ inline void Process()
    {
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t inner = tiling_.inner;
        const uint64_t indexLen = tiling_.indexLen;
        const uint64_t rowBytes = inner * sizeof(T);
        const uint64_t rows = static_cast<uint64_t>(tiling_.outer) * selfLen;
        const uint64_t units = (rows + tiling_.unit - 1) / tiling_.unit;
        const uint64_t cores = GetBlockNum();
        const uint64_t core = GetBlockIdx();
        const uint64_t base = units / cores;
        const uint64_t extra = units % cores;
        const uint64_t firstUnit = core * base + (core < extra ? core : extra);
        uint64_t first = firstUnit * tiling_.unit;
        uint64_t last = (firstUnit + base + (core < extra ? 1 : 0)) * tiling_.unit;
        last = last < rows ? last : rows;
        if (first >= last) {
            return;
        }
        LocalTensor<int32_t> idxTensor = indexBuf_.Get<int32_t>();
        LoadIndex(idxTensor);
        const __ubuf__ int32_t* idx = reinterpret_cast<__ubuf__ int32_t*>(idxTensor.GetPhyAddr());
        __ubuf__ uint32_t* ramp = reinterpret_cast<__ubuf__ uint32_t*>(rampBuf_.Get<uint32_t>().GetPhyAddr());
        const uint32_t laneBytes = IsInt8() ? 2 : static_cast<uint32_t>(sizeof(T));
        const uint32_t laneCount = IsInt8() ? lanes_ : static_cast<uint32_t>(inner);
        for (uint32_t j = 0; j < laneCount; ++j) {
            ramp[j] = j * laneBytes;
        }
        if (IsInt8()) {
            LocalTensor<int16_t> work = workBuf_.Get<int16_t>();
            Duplicate(work, static_cast<int16_t>(0x7F7F), static_cast<int32_t>(lanes_));
            Duplicate(work[kWork], static_cast<int16_t>(-32640), static_cast<int32_t>(lanes_));
        }
        const uint64_t chunkRows = tiling_.slabRows;
        for (uint64_t o = first / selfLen; o * selfLen < last; ++o) {
            const uint64_t r0 = first > o * selfLen ? first - o * selfLen : 0;
            const uint64_t r1 = last - o * selfLen < selfLen ? last - o * selfLen : selfLen;
            const uint64_t owned = r1 - r0;
            uint64_t lo = indexLen;
            uint64_t hi = 0;
            for (uint64_t i = 0; i < indexLen; ++i) {
                const uint64_t t = static_cast<uint32_t>(idx[i]);
                if (t >= r0 && t < r1) {
                    lo = i < lo ? i : lo;
                    hi = i + 1;
                }
            }
            Load(selfBuf_.Get<T>(), selfGm_, (o * selfLen + r0) * inner, owned * inner);
            Sync<HardEvent::MTE2_V>();
            for (uint64_t j = 0; j < owned; ++j) {
                GatherRow(accBuf_.Get<T>()[j * pitch_ / sizeof(T)], selfBuf_.Get<T>(), static_cast<uint32_t>(j * rowBytes));
            }
            for (uint64_t c = lo; c < hi; c += chunkRows) {
                const uint64_t end = hi - c < chunkRows ? hi : c + chunkRows;
                Load(sourceBuf_.Get<T>(), sourceGm_, (o * indexLen + c) * inner, (end - c) * inner);
                Sync<HardEvent::MTE2_V>();
                for (uint64_t i = c; i < end; ++i) {
                    const uint64_t t = static_cast<uint32_t>(idx[i]);
                    if (t >= r0 && t < r1) {
                        GatherRow(rowBuf_.Get<T>(), sourceBuf_.Get<T>(), static_cast<uint32_t>((i - c) * rowBytes));
                        AddRow((t - r0) * pitch_);
                    }
                }

                Sync<HardEvent::V_MTE2>();
            }
            Sync<HardEvent::V_MTE3>();
            for (uint64_t j = 0; j < owned; ++j) {
                DataCopyExtParams store{1, static_cast<uint32_t>(rowBytes), 0, 0, 0};
                DataCopyPad(outGm_[(o * selfLen + r0 + j) * inner], accBuf_.Get<T>()[j * pitch_ / sizeof(T)], store);
            }
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

private:
    static constexpr uint32_t kWork = 512;

    __aicore__ inline bool IsInt8() const
    {
        return std::is_same<T, int8_t>::value;
    }

    __aicore__ inline void LoadIndex(const LocalTensor<int32_t>& dst)
    {
        DataCopyExtParams params{1, tiling_.indexLen * 4, 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(dst, indexGm_[0], params, pad);
        Sync<HardEvent::MTE2_S>();
    }

    __aicore__ inline void Load(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint64_t offset, uint64_t count)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], params, pad);
    }

    __aicore__ inline void GatherRow(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t offset)
    {
        const LocalTensor<uint32_t> ramp = rampBuf_.Get<uint32_t>();
        if constexpr (std::is_same<T, int8_t>::value) {
            Gather(dst.template ReinterpretCast<uint16_t>(), src.template ReinterpretCast<uint16_t>(), ramp, offset, lanes_);
        } else {
            Gather(dst, src, ramp, offset, tiling_.inner);
        }
    }

    __aicore__ inline void AddRow(uint64_t at)
    {
        if constexpr (std::is_same<T, int8_t>::value) {

            const int32_t n = static_cast<int32_t>(lanes_);
            LocalTensor<int16_t> a = accBuf_.Get<int16_t>()[static_cast<uint32_t>(at / 2)];
            LocalTensor<int16_t> b = rowBuf_.Get<int16_t>();
            LocalTensor<int16_t> work = workBuf_.Get<int16_t>();
            LocalTensor<int16_t> low = work;
            LocalTensor<int16_t> top = work[kWork];
            LocalTensor<int16_t> t1 = work[2 * kWork];
            LocalTensor<int16_t> t2 = work[3 * kWork];
            And(t1, a, low, n);
            And(t2, b, low, n);
            Add(t1, t1, t2, n);
            Or(t2, a, b, n);
            And(a, a, b, n);
            Sub(t2, t2, a, n);
            And(t2, t2, top, n);
            Or(a, t1, t2, n);
            And(t1, t1, t2, n);
            Sub(a, a, t1, n);
        } else {
            const uint32_t inner = tiling_.inner;
            LocalTensor<T> a = accBuf_.Get<T>()[static_cast<uint32_t>(at / 2)];
            LocalTensor<float> work = workBuf_.Get<float>();

            Cast(work, a, RoundMode::CAST_NONE, inner);
            Cast(work[kWork], rowBuf_.Get<T>(), RoundMode::CAST_NONE, inner);
            Add(work, work, work[kWork], static_cast<int32_t>(inner));
            Cast(a, work, RoundMode::CAST_RINT, inner);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> selfBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    TBuf<TPosition::VECCALC> rampBuf_;
    TBuf<TPosition::VECCALC> rowBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    IndexAddTilingData tiling_;
    uint32_t pitch_ = 0;
    uint32_t lanes_ = 0;
    GlobalTensor<T> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outGm_;
};

template <typename T>
class SplitRowKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out));
        selfGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        sourceGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t lanes = 32 / sizeof(T);
        pitch_ = (tiling.inner + lanes - 1) / lanes * lanes;
        pipe_.InitBuffer(indexBuf_, (tiling.indexLen * 4 + 31) / 32 * 32);
        pipe_.InitBuffer(accBuf_, tiling.selfLen * pitch_ * sizeof(T) + 32);
        pipe_.InitBuffer(stageBuf_, tiling.slabRows * pitch_ * sizeof(T) + 32);
    }

    __aicore__ inline void Process()
    {
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t indexLen = tiling_.indexLen;
        const uint64_t total = static_cast<uint64_t>(tiling_.outer) * indexLen;
        const uint64_t cores = GetBlockNum();
        const uint64_t core = GetBlockIdx();
        const uint64_t base = total / cores;
        const uint64_t extra = total % cores;
        const uint64_t begin = core * base + (core < extra ? core : extra);
        const uint64_t end = begin + base + (core < extra ? 1 : 0);
        LocalTensor<T> stage = stageBuf_.Get<T>();
        LocalTensor<T> acc = accBuf_.Get<T>();

        LocalTensor<int32_t> idxTensor = indexBuf_.Get<int32_t>();
        DataCopyExtParams params{1, tiling_.indexLen * 4, 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(idxTensor, indexGm_[0], params, pad);
        const uint64_t rows = static_cast<uint64_t>(tiling_.outer) * selfLen;
        const uint64_t rowFirst = core * (rows / cores) + (core < rows % cores ? core : rows % cores);
        const uint64_t rowLast = rowFirst + rows / cores + (core < rows % cores ? 1 : 0);
        for (uint64_t r = rowFirst; r < rowLast; r += tiling_.slabRows) {
            const uint64_t n = rowLast - r < tiling_.slabRows ? rowLast - r : tiling_.slabRows;
            for (uint64_t j = 0; j < n; ++j) {
                Load(stage[j * pitch_], selfGm_, (r + j) * tiling_.inner);
            }
            Sync<HardEvent::MTE2_MTE3>();
            for (uint64_t j = 0; j < n; ++j) {
                Store(outGm_[(r + j) * tiling_.inner], stage[j * pitch_]);
            }
            Sync<HardEvent::MTE3_MTE2>();
        }
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::MTE3_S>();
        SyncAll<true>();
        if (begin >= end) {
            return;
        }

        const __ubuf__ int32_t* idx = reinterpret_cast<__ubuf__ int32_t*>(idxTensor.GetPhyAddr());
        const int32_t count = static_cast<int32_t>(tiling_.inner);
        Duplicate(acc, static_cast<T>(0), static_cast<int32_t>(selfLen * pitch_));

        const uint64_t half = tiling_.slabRows / 2 > 0 ? tiling_.slabRows / 2 : 1;
        TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        TEventID spent[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                             GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
        bool live[2] = {false, false};
        uint64_t slice = begin / indexLen;
        uint64_t g = begin;
        uint64_t n = Take(g, end, half);
        uint32_t cur = 0;
        LoadRows(stage, g, n);
        SetFlag<HardEvent::MTE2_V>(full[0]);
        while (g < end) {
            const uint64_t o = g / indexLen;
            const uint64_t next = g + n;
            const uint64_t nextN = next < end ? Take(next, end, half) : 0;
            const uint32_t nxt = 1 - cur;
            if (nextN > 0) {
                if (live[nxt]) {
                    WaitFlag<HardEvent::V_MTE2>(spent[nxt]);
                    live[nxt] = false;
                }
                LoadRows(stage[nxt * half * pitch_], next, nextN);
                SetFlag<HardEvent::MTE2_V>(full[nxt]);
            }
            if (o != slice) {
                Flush(slice);
                Sync<HardEvent::MTE3_V>();
                Duplicate(acc, static_cast<T>(0), static_cast<int32_t>(selfLen * pitch_));
                slice = o;
            }
            WaitFlag<HardEvent::MTE2_V>(full[cur]);
            const LocalTensor<T> rowsNow = stage[cur * half * pitch_];
            for (uint64_t j = 0; j < n; ++j) {
                const uint32_t t = static_cast<uint32_t>(idx[g + j - o * indexLen]);
                Add(acc[t * pitch_], acc[t * pitch_], rowsNow[j * pitch_], count);
            }
            SetFlag<HardEvent::V_MTE2>(spent[cur]);
            live[cur] = true;
            g = next;
            n = nextN;
            cur = nxt;
        }
        Flush(slice);
        for (uint32_t b = 0; b < 2; ++b) {
            if (live[b]) {
                WaitFlag<HardEvent::V_MTE2>(spent[b]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(full[b]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(spent[b]);
        }
    }

private:

    __aicore__ inline uint64_t Take(uint64_t g, uint64_t end, uint64_t limit) const
    {
        uint64_t n = end - g < limit ? end - g : limit;
        const uint64_t sliceEnd = (g / tiling_.indexLen + 1) * tiling_.indexLen;
        return sliceEnd - g < n ? sliceEnd - g : n;
    }

    __aicore__ inline void LoadRows(const LocalTensor<T>& dst, uint64_t g, uint64_t n)
    {
        for (uint64_t j = 0; j < n; ++j) {
            Load(dst[j * pitch_], sourceGm_, (g + j) * tiling_.inner);
        }
    }

    __aicore__ inline void Flush(uint64_t o)
    {
        const uint64_t selfLen = tiling_.selfLen;
        LocalTensor<T> acc = accBuf_.Get<T>();
        Sync<HardEvent::V_MTE3>();
        for (uint64_t r = 0; r < selfLen; ++r) {
            DataCopyExtParams store{1, static_cast<uint32_t>(tiling_.inner * sizeof(T)), 0, 0, 0};
            DataCopyPadWithReduce<T, ReduceType::REDUCE_ADD>(outGm_[(o * selfLen + r) * tiling_.inner],
                                                            acc[r * pitch_], store);
        }
    }

    __aicore__ inline void Load(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint64_t offset)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(tiling_.inner * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, src[offset], params, pad);
    }

    __aicore__ inline void Store(const GlobalTensor<T>& dst, const LocalTensor<T>& src)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(tiling_.inner * sizeof(T)), 0, 0, 0};
        DataCopyPad(dst, src, params);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> stageBuf_;
    IndexAddTilingData tiling_;
    uint32_t pitch_ = 0;
    GlobalTensor<T> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outGm_;
};

class WideInt8Kernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(source));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out));
        selfBytes_ = tiling.outer * tiling.selfLen * tiling.inner;
        sourceBytes_ = tiling.outer * tiling.indexLen * tiling.inner;
        pipe_.InitBuffer(indexBuf_, (tiling.indexLen * 4 + 31) / 32 * 32);
        pipe_.InitBuffer(selfBuf_, selfBytes_ + 32);
        pipe_.InitBuffer(sourceBuf_, sourceBytes_ + 32);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> idxTensor = indexBuf_.Get<int32_t>();
        LocalTensor<int8_t> selfRaw = selfBuf_.Get<int8_t>();
        LocalTensor<int8_t> sourceRaw = sourceBuf_.Get<int8_t>();
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        const uint32_t inner = tiling_.inner;
        const uint32_t selfLen = tiling_.selfLen;
        const uint32_t indexLen = tiling_.indexLen;
        const uint32_t rows = tiling_.outer * indexLen;
        const uint32_t begin = static_cast<uint32_t>(static_cast<uint64_t>(rows) * core / cores);
        const uint32_t end = static_cast<uint32_t>(static_cast<uint64_t>(rows) * (core + 1) / cores);
        DataCopyExtParams indexParams{1, tiling_.indexLen * 4, 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPad{false, 0, 0, 0};
        DataCopyPad(idxTensor, indexGm_[0], indexParams, indexPad);
        DataCopyPadExtParams<int8_t> bytePad{false, 0, 0, 0};
        DataCopyExtParams selfParams{1, selfBytes_, 0, 0, 0};
        if (core == 0) {
            DataCopyPad(selfRaw, selfGm_[0], selfParams, bytePad);
        }
        if (begin < end) {
            DataCopyExtParams sourceParams{1, (end - begin) * inner, 0, 0, 0};
            DataCopyPad(sourceRaw, sourceGm_[begin * inner], sourceParams, bytePad);
        }
        if (core == 0) {
            Sync<HardEvent::MTE2_MTE3>();
            DataCopyPad(outGm_[0], selfRaw, selfParams);
        }
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::MTE3_S>();

        SyncAll<true>();
        const __ubuf__ int32_t* idx = reinterpret_cast<__ubuf__ int32_t*>(idxTensor.GetPhyAddr());
        DataCopyExtParams rowParams{1, inner, 0, 0, 0};
        for (uint32_t g = begin; g < end; ++g) {
            const uint32_t o = g / indexLen;
            const uint32_t i = g - o * indexLen;
            const uint32_t to = (o * selfLen + static_cast<uint32_t>(idx[i])) * inner;
            DataCopyPadWithReduce<int8_t, ReduceType::REDUCE_ADD>(outGm_[to], sourceRaw[(g - begin) * inner], rowParams);
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> selfBuf_;
    TBuf<TPosition::VECCALC> sourceBuf_;
    IndexAddTilingData tiling_;
    uint32_t selfBytes_ = 0;
    uint32_t sourceBytes_ = 0;
    GlobalTensor<int8_t> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<int8_t> sourceGm_;
    GlobalTensor<int8_t> outGm_;
};

constexpr uint32_t kMaxBf16Rows = 1024;
constexpr uint32_t kMaxBf16Hits = 2048;

class Bf16LayerKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;

        const uint64_t selfElems = static_cast<uint64_t>(tiling.outer) * tiling.selfLen * tiling.inner;
        const uint64_t sourceElems = static_cast<uint64_t>(tiling.outer) * tiling.indexLen * tiling.inner;
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(self), selfElems);
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index), tiling.indexLen);
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(source), sourceElems);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(out), selfElems);
        if (tiling.path != 11) {
            selfGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            sourceGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }

        pitch_ = tiling.path == 11 ? 1 : (tiling.inner + 15) / 16 * 16;
        const uint32_t rows = tiling.slabRows;
        pipe_.InitBuffer(indexBuf_, (tiling.indexLen * 4 + 31) / 32 * 32);
        pipe_.InitBuffer(rowBuf_, rows * pitch_ * 2 + 32);

        pipe_.InitBuffer(deltaBuf_, (tiling.path == 11 ? 3 : 2) * rows * pitch_ * 2 + 32);
        pipe_.InitBuffer(accBuf_, rows * pitch_ * 4 + 32);
        pipe_.InitBuffer(wideBuf_, rows * pitch_ * 4 + 32);
        if (tiling.indexLen <= kSelectLanes) {
            pipe_.InitBuffer(selectBuf_, 4 * kSelectLanes * 4);
            pipe_.InitBuffer(maskBuf_, 2 * kSelectLanes / 8 + 32);
            pipe_.InitBuffer(hitBuf_, 2 * kSelectLanes * 4);
        }
    }

    __aicore__ inline void Process()
    {
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t inner = tiling_.inner;
        const uint64_t indexLen = tiling_.indexLen;
        const uint64_t rows = static_cast<uint64_t>(tiling_.outer) * selfLen;
        const uint64_t units = (rows + tiling_.unit - 1) / tiling_.unit;
        const uint64_t cores = GetBlockNum();
        const uint64_t core = GetBlockIdx();
        const uint64_t base = units / cores;
        const uint64_t extra = units % cores;
        const uint64_t firstUnit = core * base + (core < extra ? core : extra);
        const uint64_t first = firstUnit * tiling_.unit;
        uint64_t last = (firstUnit + base + (core < extra ? 1 : 0)) * tiling_.unit;
        last = last < rows ? last : rows;
        if (first >= last) {
            return;
        }
        LocalTensor<int32_t> idxTensor = indexBuf_.Get<int32_t>();
        DataCopyExtParams indexParams{1, tiling_.indexLen * 4, 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPad{false, 0, 0, 0};
        DataCopyPad(idxTensor, indexGm_[0], indexParams, indexPad);
        Sync<HardEvent::MTE2_S>();
        const __ubuf__ int32_t* idx = reinterpret_cast<__ubuf__ int32_t*>(idxTensor.GetPhyAddr());
        LocalTensor<bfloat16_t> row = rowBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> delta = deltaBuf_.Get<bfloat16_t>();
        LocalTensor<int16_t> deltaBits = deltaBuf_.Get<int16_t>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> wide = wideBuf_.Get<float>();
        const uint32_t rowBytes = static_cast<uint32_t>(inner * 2);

        const uint64_t firstSlice = first / selfLen;
        const uint64_t lastSlice = (last - 1) / selfLen;
        const bool allRows = lastSlice - firstSlice >= 2 || (last - first) >= selfLen;
        const uint32_t headRow = static_cast<uint32_t>(first - firstSlice * selfLen);
        const uint32_t tailRow = static_cast<uint32_t>(last - lastSlice * selfLen);
        for (uint32_t j = 0; j < static_cast<uint32_t>(selfLen); ++j) {
            head_[j] = kNoHit;
        }
        uint32_t hits = 0;

        const bool bySlot = !allRows && lastSlice - firstSlice <= 1 && indexLen <= kSelectLanes &&
                            last - first <= kMaxBf16Rows;
        if (bySlot) {
            hits = SelectHits(idxTensor, first, last);
        }
        for (uint32_t i = 0; i < static_cast<uint32_t>(bySlot ? 0 : indexLen); ++i) {
            const uint32_t t = static_cast<uint32_t>(idx[i]);
            const bool mine = allRows || (firstSlice == lastSlice ? (t >= headRow && t < tailRow)
                                                                  : (t >= headRow || t < tailRow));
            if (!mine) {
                continue;
            }
            hitPos_[hits] = static_cast<uint16_t>(i);
            hitNext_[hits] = kNoHit;
            if (head_[t] == kNoHit) {
                head_[t] = static_cast<uint16_t>(hits);
            } else {
                hitNext_[tail_[t]] = static_cast<uint16_t>(hits);
            }
            tail_[t] = static_cast<uint16_t>(hits);
            ++hits;
        }

        const uint32_t owned = static_cast<uint32_t>(last - first);
        const int32_t count = static_cast<int32_t>(owned * pitch_);
        bool any = false;
        for (uint32_t j = 0; j < owned; ++j) {
            cursor_[j] = bySlot ? head_[j] : head_[(first + j) % selfLen];
            any = any || cursor_[j] != kNoHit;
        }

        LocalTensor<bfloat16_t> packed = wideBuf_.Get<bfloat16_t>();
        DataCopyExtParams selfParams{1, owned * rowBytes, 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> selfPad{false, 0, 0, static_cast<bfloat16_t>(0)};
        DataCopyPad(packed, selfGm_[first * inner], selfParams, selfPad);
        __ubuf__ uint32_t* ramp = reinterpret_cast<__ubuf__ uint32_t*>(accBuf_.Get<uint32_t>().GetPhyAddr());
        for (uint32_t e = 0; e < static_cast<uint32_t>(inner); ++e) {
            ramp[e] = 2 * e;
        }
        Sync<HardEvent::MTE2_V>();
        LocalTensor<uint32_t> rampTable = accBuf_.Get<uint32_t>();
        for (uint32_t j = 0; j < owned; ++j) {
            Gather(row[j * pitch_], packed, rampTable, j * rowBytes, static_cast<uint32_t>(inner));
        }
        Cast(acc, row, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        const uint32_t half = static_cast<uint32_t>(count);
        TEventID zeroed[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                              GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
        TEventID loaded[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                              GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        bool pending[2] = {false, false};
        for (uint32_t b = 0; b < 2 && any; ++b) {
            any = IssueLayer(deltaBits[b * half], delta[b * half], zeroed[b], loaded[b], first, owned, count, rowBytes);
            pending[b] = true;
        }

        for (uint32_t b = 0; pending[b]; b = 1 - b) {
            WaitFlag<HardEvent::MTE2_V>(loaded[b]);
            const LocalTensor<bfloat16_t> cur = delta[b * half];
            Cast(wide, cur, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            Add(acc, acc, wide, count);
            Cast(cur, acc, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
            Cast(acc, cur, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            pending[b] = false;
            if (any) {
                any = IssueLayer(deltaBits[b * half], cur, zeroed[b], loaded[b], first, owned, count, rowBytes);
                pending[b] = true;
            }
        }
        for (uint32_t b = 0; b < 2; ++b) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(zeroed[b]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(loaded[b]);
        }
        Cast(row, acc, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
        if (pitch_ == 128 && inner != pitch_ && tiling_.indexLen <= kSelectLanes) {

            LocalTensor<uint16_t> keep = maskBuf_.Get<uint16_t>()[kSelectLanes / 8 / 2];
            for (uint32_t w = 0; w < 8; ++w) {
                const uint32_t lo = w * 16;
                const uint16_t bits = lo + 16 <= inner ? 0xFFFF
                    : (lo >= inner ? 0 : static_cast<uint16_t>((1u << (inner - lo)) - 1));
                keep.SetValue(w, bits);
                keep.SetValue(w + 8, bits);
            }
            uint64_t kept = 0;
            GatherMask(row.template ReinterpretCast<uint16_t>(), row.template ReinterpretCast<uint16_t>(), keep, false, 0,
                       {1, static_cast<uint16_t>(owned), 8, 0}, kept);
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams store{1, owned * rowBytes, 0, 0, 0};
            DataCopyPad(outGm_[first * inner], row, store);
        } else {
            Sync<HardEvent::V_MTE3>();
            for (uint32_t j = 0; j < owned; ++j) {
                DataCopyExtParams store{1, rowBytes, 0, 0, 0};
                DataCopyPad(outGm_[(first + j) * inner], row[j * pitch_], store);
            }
        }
    }

    __aicore__ inline void ProcessStacks()
    {
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t inner = tiling_.inner;
        const uint64_t indexLen = tiling_.indexLen;
        const uint64_t cores = GetBlockNum();
        const uint64_t core = GetBlockIdx();
        const uint64_t base = selfLen / cores;
        const uint64_t extra = selfLen % cores;
        const uint64_t first = core * base + (core < extra ? core : extra);
        const uint64_t last = first + base + (core < extra ? 1 : 0);
        if (first >= last) {
            return;
        }
        const uint32_t owned = static_cast<uint32_t>(last - first);
        const uint16_t stack = static_cast<uint16_t>(tiling_.outer);
        const uint32_t rowBytes = static_cast<uint32_t>(inner * 2);

        const uint32_t tile = static_cast<uint32_t>(tiling_.outer) * static_cast<uint32_t>((inner + 15) / 16 * 16);
        const int32_t count = static_cast<int32_t>(owned * tile);
        const uint32_t srcGap = static_cast<uint32_t>(indexLen - 1) * rowBytes;
        const uint32_t selfGap = static_cast<uint32_t>(selfLen - 1) * rowBytes;
        LocalTensor<int32_t> idxTensor = indexBuf_.Get<int32_t>();
        DataCopyExtParams indexParams{1, tiling_.indexLen * 4, 0, 0, 0};
        DataCopyPadExtParams<int32_t> indexPad{false, 0, 0, 0};
        DataCopyPad(idxTensor, indexGm_[0], indexParams, indexPad);
        Sync<HardEvent::MTE2_S>();
        LocalTensor<bfloat16_t> row = rowBuf_.Get<bfloat16_t>();
        DataCopyExtParams selfParams{stack, rowBytes, selfGap, 0, 0};
        DataCopyPadExtParams<bfloat16_t> selfPad{false, 0, 0, static_cast<bfloat16_t>(0)};
        for (uint32_t j = 0; j < owned; ++j) {
            DataCopyPad(row[j * tile], selfGm_[(first + j) * inner], selfParams, selfPad);
        }
        uint32_t hits = 0;
        if (indexLen <= kSelectLanes) {
            hits = SelectHits(idxTensor, first, last);
        } else {
            const __ubuf__ int32_t* idx = reinterpret_cast<__ubuf__ int32_t*>(idxTensor.GetPhyAddr());
            for (uint32_t j = 0; j < owned; ++j) {
                head_[j] = kNoHit;
            }
            for (uint32_t i = 0; i < static_cast<uint32_t>(indexLen); ++i) {
                const uint32_t j = static_cast<uint32_t>(idx[i]) - static_cast<uint32_t>(first);
                if (j >= owned) {
                    continue;
                }
                hitPos_[hits] = static_cast<uint16_t>(i);
                hitNext_[hits] = kNoHit;
                if (head_[j] == kNoHit) {
                    head_[j] = static_cast<uint16_t>(hits);
                } else {
                    hitNext_[tail_[j]] = static_cast<uint16_t>(hits);
                }
                tail_[j] = static_cast<uint16_t>(hits);
                ++hits;
            }
        }
        bool any = false;
        for (uint32_t j = 0; j < owned; ++j) {
            cursor_[j] = head_[j];
            any = any || head_[j] != kNoHit;
        }
        LocalTensor<bfloat16_t> delta = deltaBuf_.Get<bfloat16_t>();
        LocalTensor<int16_t> deltaBits = deltaBuf_.Get<int16_t>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> wide = wideBuf_.Get<float>();

        TEventID zeroed[3];
        TEventID loaded[3];
        for (uint32_t b = 0; b < 3; ++b) {
            zeroed[b] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
            loaded[b] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        }
        const uint32_t half = static_cast<uint32_t>(tiling_.slabRows);
        uint32_t queue[3] = {0, 0, 0};
        uint32_t head = 0;
        uint32_t queued = 0;
        uint32_t spare = 2;
        for (uint32_t b = 0; b < 2 && any; ++b) {
            any = IssueStacks(deltaBits[b * half], delta[b * half], zeroed[b], loaded[b], owned, count, rowBytes,
                              srcGap, tile);
            queue[queued++] = b;
        }
        Sync<HardEvent::MTE2_V>();
        Cast(acc, row, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        while (queued > 0) {
            const uint32_t cur = queue[head];
            head = (head + 1) % 3;
            --queued;
            if (any) {
                any = IssueStacks(deltaBits[spare * half], delta[spare * half], zeroed[spare], loaded[spare], owned,
                                  count, rowBytes, srcGap, tile);
                queue[(head + queued) % 3] = spare;
                ++queued;
            }
            WaitFlag<HardEvent::MTE2_V>(loaded[cur]);
            const LocalTensor<bfloat16_t> layer = delta[cur * half];
            Cast(wide, layer, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            Add(acc, acc, wide, count);
            Cast(layer, acc, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
            Cast(acc, layer, RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            spare = cur;
        }
        for (uint32_t b = 0; b < 3; ++b) {
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(zeroed[b]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(loaded[b]);
        }
        Cast(row, acc, RoundMode::CAST_RINT, static_cast<uint32_t>(count));
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams store{stack, rowBytes, 0, selfGap, 0};
        for (uint32_t j = 0; j < owned; ++j) {
            DataCopyPad(outGm_[(first + j) * inner], row[j * tile], store);
        }
    }

    __aicore__ inline void ProcessHits()
    {
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t inner = tiling_.inner;
        const uint64_t indexLen = tiling_.indexLen;
        const uint64_t cores = GetBlockNum();
        const uint64_t core = GetBlockIdx();
        const uint64_t base = selfLen / cores;
        const uint64_t extra = selfLen % cores;
        const uint64_t first = core * base + (core < extra ? core : extra);
        const uint64_t last = first + base + (core < extra ? 1 : 0);
        if (first >= last) {
            return;
        }
        const uint32_t owned = static_cast<uint32_t>(last - first);
        const uint16_t stack = static_cast<uint16_t>(tiling_.outer);
        const uint32_t rowBytes = static_cast<uint32_t>(inner * 2);
        const uint32_t tile = static_cast<uint32_t>(tiling_.outer) * static_cast<uint32_t>((inner + 15) / 16 * 16);
        const uint32_t srcGap = static_cast<uint32_t>(indexLen - 1) * rowBytes;
        const uint32_t selfGap = static_cast<uint32_t>(selfLen - 1) * rowBytes;
        LocalTensor<int32_t> idxTensor = indexBuf_.Get<int32_t>();
        DataCopyExtParams indexParams{1, tiling_.indexLen * 4, 0, 0, 0};
        DataCopyPad(idxTensor, indexGm_[0], indexParams, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        LocalTensor<bfloat16_t> row = rowBuf_.Get<bfloat16_t>();
        DataCopyExtParams selfParams{stack, rowBytes, selfGap, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, static_cast<bfloat16_t>(0)};
        for (uint32_t j = 0; j < owned; ++j) {
            DataCopyPad(row[j * tile], selfGm_[(first + j) * inner], selfParams, pad);
        }
        Sync<HardEvent::MTE2_V>();
        Stage(idxTensor);
        LocalTensor<int32_t> found = hitBuf_.Get<int32_t>();
        const uint32_t hits = Select(found, first, last, -static_cast<int32_t>(first));
        Sync<HardEvent::V_S>();
        const __ubuf__ int32_t* rec = reinterpret_cast<__ubuf__ int32_t*>(found.GetPhyAddr());
        LocalTensor<bfloat16_t> slots = deltaBuf_.Get<bfloat16_t>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> wide = wideBuf_.Get<float>();
        TEventID loaded[2];
        TEventID read[2];
        for (uint32_t b = 0; b < 2; ++b) {
            loaded[b] = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
            read[b] = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        }
        DataCopyExtParams srcParams{stack, rowBytes, srcGap, 0, 0};
        if (hits > 0) {
            DataCopyPad(slots, sourceGm_[(rec[0] >> 10) * inner], srcParams, pad);
            SetFlag<HardEvent::MTE2_V>(loaded[0]);
        }
        for (uint32_t h = 0; h < hits; ++h) {
            const uint32_t b = h & 1;
            const int32_t r = rec[h];
            if (h + 1 < hits) {
                if (h >= 1) {
                    WaitFlag<HardEvent::V_MTE2>(read[1 - b]);
                }
                DataCopyPad(slots[(1 - b) * tile], sourceGm_[(rec[h + 1] >> 10) * inner], srcParams, pad);
                SetFlag<HardEvent::MTE2_V>(loaded[1 - b]);
            }
            WaitFlag<HardEvent::MTE2_V>(loaded[b]);
            const LocalTensor<bfloat16_t> state = row[(r & 1023) * tile];
            Cast(wide, slots[b * tile], RoundMode::CAST_NONE, tile);
            SetFlag<HardEvent::V_MTE2>(read[b]);
            Cast(acc, state, RoundMode::CAST_NONE, tile);
            Add(acc, acc, wide, static_cast<int32_t>(tile));
            Cast(state, acc, RoundMode::CAST_RINT, tile);
        }
        for (uint32_t h = hits > 2 ? hits - 2 : 0; h < hits; ++h) {
            WaitFlag<HardEvent::V_MTE2>(read[h & 1]);
        }
        for (uint32_t b = 0; b < 2; ++b) {
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(loaded[b]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(read[b]);
        }
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams store{stack, rowBytes, 0, selfGap, 0};
        for (uint32_t j = 0; j < owned; ++j) {
            DataCopyPad(outGm_[(first + j) * inner], row[j * tile], store);
        }
    }

private:
    __aicore__ inline bool IssueStacks(const LocalTensor<int16_t>& bits, const LocalTensor<bfloat16_t>& dst,
                                       TEventID zeroed, TEventID loaded, uint32_t owned, int32_t count,
                                       uint32_t rowBytes, uint32_t srcGap, uint32_t tile)
    {
        Duplicate(bits, static_cast<int16_t>(0), count);
        SetFlag<HardEvent::V_MTE2>(zeroed);
        WaitFlag<HardEvent::V_MTE2>(zeroed);
        const uint16_t stack = static_cast<uint16_t>(tiling_.outer);
        bool more = false;
        for (uint32_t j = 0; j < owned; ++j) {
            const uint16_t h = cursor_[j];
            if (h != kNoHit) {
                DataCopyExtParams params{stack, rowBytes, srcGap, 0, 0};
                DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, static_cast<bfloat16_t>(0)};
                DataCopyPad(dst[j * tile], sourceGm_[hitPos_[h] * tiling_.inner], params, pad);
                cursor_[j] = hitNext_[h];
                more = more || hitNext_[h] != kNoHit;
            }
        }
        SetFlag<HardEvent::MTE2_V>(loaded);
        return more;
    }

    __aicore__ inline bool IssueLayer(const LocalTensor<int16_t>& bits, const LocalTensor<bfloat16_t>& dst,
                                      TEventID zeroed, TEventID loaded, uint64_t first, uint32_t owned,
                                      int32_t count, uint32_t rowBytes)
    {
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t indexLen = tiling_.indexLen;
        Duplicate(bits, static_cast<int16_t>(0), count);
        SetFlag<HardEvent::V_MTE2>(zeroed);
        WaitFlag<HardEvent::V_MTE2>(zeroed);
        bool more = false;
        for (uint32_t j = 0; j < owned; ++j) {
            const uint16_t h = cursor_[j];
            if (h != kNoHit) {
                const uint64_t slice = (first + j) / selfLen;
                Load(dst[j * pitch_], sourceGm_, (slice * indexLen + hitPos_[h]) * tiling_.inner, rowBytes);
                cursor_[j] = hitNext_[h];
                more = more || hitNext_[h] != kNoHit;
            }
        }
        SetFlag<HardEvent::MTE2_V>(loaded);
        return more;
    }

    __aicore__ inline void Stage(const LocalTensor<int32_t>& idxTensor)
    {
        const uint32_t n = kSelectLanes;
        LocalTensor<float> sel = selectBuf_.Get<float>();

        Duplicate(sel, -1.0f, n);
        CreateVecIndex(sel[n], 0.0f, n);
        PipeBarrier<PIPE_V>();
        Cast(sel, idxTensor, RoundMode::CAST_RINT, tiling_.indexLen);
        Muls(sel[n], sel[n], 1024.0f, n);
        Add(sel[n], sel[n], sel, n);
    }

    __aicore__ inline uint32_t Select(const LocalTensor<int32_t>& dst, uint64_t lo, uint64_t hi, int32_t shift)
    {
        const uint32_t n = kSelectLanes;
        LocalTensor<float> sel = selectBuf_.Get<float>();
        LocalTensor<uint8_t> low = maskBuf_.Get<uint8_t>();
        LocalTensor<uint8_t> high = low[n / 8];
        CompareScalar(low, sel, static_cast<float>(static_cast<int32_t>(lo)), CMPMODE::GE, n);
        CompareScalar(high, sel, static_cast<float>(static_cast<int32_t>(hi)), CMPMODE::LT, n);
        And(low.template ReinterpretCast<uint16_t>(), low.template ReinterpretCast<uint16_t>(),
            high.template ReinterpretCast<uint16_t>(), n / 16);
        Adds(sel[2 * n], sel[n], static_cast<float>(shift), n);
        uint64_t selected = 0;
        GatherMask(sel[3 * n], sel[2 * n], low.template ReinterpretCast<uint32_t>(), true, n, {1, 1, 8, 0}, selected);
        Sync<HardEvent::V_S>();
        const uint32_t count = static_cast<uint32_t>(selected);
        if (count > 0) {
            Cast(dst, sel[3 * n], RoundMode::CAST_RINT, count);
        }
        return count;
    }

    __aicore__ inline uint32_t SelectHits(const LocalTensor<int32_t>& idxTensor, uint64_t first, uint64_t last)
    {
        const uint64_t selfLen = tiling_.selfLen;
        const uint32_t n = kSelectLanes;
        LocalTensor<int32_t> found = hitBuf_.Get<int32_t>();
        Stage(idxTensor);
        const uint64_t firstSlice = first / selfLen;
        const uint64_t lastSlice = (last - 1) / selfLen;
        uint32_t counts[2] = {0, 0};
        for (uint64_t s = firstSlice; s <= lastSlice; ++s) {
            const uint64_t base = s * selfLen;
            const uint64_t lo = (first > base ? first : base) - base;
            const uint64_t end = last < base + selfLen ? last : base + selfLen;
            const uint32_t k = static_cast<uint32_t>(s - firstSlice);
            counts[k] = Select(found[k * n], lo, end - base, static_cast<int32_t>(base) - static_cast<int32_t>(first));
        }
        Sync<HardEvent::V_S>();
        const uint32_t owned = static_cast<uint32_t>(last - first);
        for (uint32_t j = 0; j < owned; ++j) {
            head_[j] = kNoHit;
        }
        const __ubuf__ int32_t* rec = reinterpret_cast<__ubuf__ int32_t*>(found.GetPhyAddr());
        uint32_t hits = 0;
        for (uint32_t k = 0; k < 2; ++k) {
            for (uint32_t h = 0; h < counts[k]; ++h) {
                const uint32_t r = static_cast<uint32_t>(rec[k * n + h]);
                const uint32_t j = r & 1023;
                hitPos_[hits] = static_cast<uint16_t>(r >> 10);
                hitNext_[hits] = kNoHit;
                if (head_[j] == kNoHit) {
                    head_[j] = static_cast<uint16_t>(hits);
                } else {
                    hitNext_[tail_[j]] = static_cast<uint16_t>(hits);
                }
                tail_[j] = static_cast<uint16_t>(hits);
                ++hits;
            }
        }
        return hits;
    }

    __aicore__ inline void Load(const LocalTensor<bfloat16_t>& dst, const GlobalTensor<bfloat16_t>& src,
                                uint64_t offset, uint32_t bytes)
    {
        DataCopyExtParams params{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad{false, 0, 0, static_cast<bfloat16_t>(0)};
        DataCopyPad(dst, src[offset], params, pad);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> rowBuf_;
    TBuf<TPosition::VECCALC> deltaBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> wideBuf_;
    TBuf<TPosition::VECCALC> selectBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> hitBuf_;
    IndexAddTilingData tiling_;
    uint32_t pitch_ = 0;
    static constexpr uint32_t kSelectLanes = 512;
    static constexpr uint16_t kNoHit = 0xFFFF;
    uint16_t head_[kMaxBf16Rows];
    uint16_t tail_[kMaxBf16Rows];
    uint16_t cursor_[kMaxBf16Rows];
    uint16_t hitPos_[kMaxBf16Hits];
    uint16_t hitNext_[kMaxBf16Hits];
    GlobalTensor<bfloat16_t> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<bfloat16_t> sourceGm_;
    GlobalTensor<bfloat16_t> outGm_;
};

constexpr uint32_t kAtomicChunk = 8;

class AtomicRowKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;
        const uint64_t shift = ReadShift(tiling);
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<uint64_t>(self) + shift));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(reinterpret_cast<uint64_t>(source) + shift));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(out));
        if (shift == 0) {
            selfGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            sourceGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        rowBytes_ = tiling.inner * 4;
        for (uint32_t b = 0; b < 2; ++b) {
            pipe_.InitBuffer(chunkBuf_[b], kAtomicChunk * rowBytes_);
        }
        pipe_.InitBuffer(indexBuf_, (tiling.indexLen * 4 + 31) / 32 * 32);
        pipe_.InitBuffer(headBuf_, 128);
    }

    __aicore__ inline void Process()
    {
        const uint64_t inner = tiling_.inner;
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t indexLen = tiling_.indexLen;
        const uint64_t cores = GetBlockNum();
        const uint64_t core = GetBlockIdx();
        LocalTensor<int32_t> idx = indexBuf_.Get<int32_t>();
        DataCopyExtParams idxCopy{1, tiling_.indexLen * 4, 0, 0, 0};
        DataCopyPad(idx, indexGm_[0], idxCopy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});

        LocalTensor<int32_t> first = chunkBuf_[0].Get<int32_t>();
        const uint64_t selfRows = static_cast<uint64_t>(tiling_.outer) * selfLen;
        const uint64_t selfFirst = selfRows * core / cores;
        const uint64_t selfLast = selfRows * (core + 1) / cores;
        for (uint64_t r = selfFirst; r < selfLast; r += kAtomicChunk) {
            const uint64_t n = selfLast - r < kAtomicChunk ? selfLast - r : kAtomicChunk;
            const DataCopyExtParams copy{1, static_cast<uint32_t>(n * rowBytes_), 0, 0, 0};
            DataCopyPad(first, selfGm_[r * inner], copy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            Sync<HardEvent::MTE2_MTE3>();
            DataCopyPad(outGm_[r * inner], first, copy);
            Sync<HardEvent::MTE3_MTE2>();
        }
        Sync<HardEvent::MTE2_S>();
        Sync<HardEvent::MTE3_S>();
        SyncAll<true>();

        const uint64_t rows = static_cast<uint64_t>(tiling_.outer) * indexLen;
        const uint64_t chunks = (rows + kAtomicChunk - 1) / kAtomicChunk;
        const uint64_t begin = chunks * core / cores * kAtomicChunk;
        uint64_t end = chunks * (core + 1) / cores * kAtomicChunk;
        if (end > rows) {
            end = rows;
        }
        if (begin >= end) {
            return;
        }
        TEventID loaded[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>(),
                              GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>()};
        TEventID free[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>()};
        bool busy[2] = {false, false};
        const TEventID evensDone = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        const TEventID zeroed = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        const bool shifted = inner % 8 == 4;
        LocalTensor<int32_t> head = headBuf_.Get<int32_t>();

        LocalTensor<uint32_t> headOffsets = head[kAtomicChunk / 2 * 8].template ReinterpretCast<uint32_t>();
        for (uint32_t k = 0; k < 4; ++k) {
            headOffsets.SetValue(k, 16 + 4 * k);
        }
        LoadChunk(first, begin, end);
        SetFlag<HardEvent::MTE2_MTE3>(loaded[0]);
        busy[0] = true;
        uint32_t cur = 0;
        const DataCopyExtParams rowCopy{1, rowBytes_, 0, 0, 0};
        const DataCopyExtParams shiftedCopy{1, rowBytes_ + 16, 0, 0, 0};
        uint64_t targets[kAtomicChunk];
        for (uint64_t r = begin; r < end; r += kAtomicChunk) {
            const uint64_t n = end - r < kAtomicChunk ? end - r : kAtomicChunk;
            const uint32_t nxt = 1 - cur;
            if (r + n < end) {
                if (busy[nxt]) {
                    WaitFlag<HardEvent::MTE3_MTE2>(free[nxt]);
                }
                LoadChunk(chunkBuf_[nxt].Get<int32_t>(), r + n, end);
                SetFlag<HardEvent::MTE2_MTE3>(loaded[nxt]);
                busy[nxt] = true;
            }
            WaitFlag<HardEvent::MTE2_MTE3>(loaded[cur]);
            LocalTensor<int32_t> rowsTensor = chunkBuf_[cur].Get<int32_t>();
            for (uint64_t j = 0; j < n; ++j) {
                const uint64_t g = r + j;
                const uint64_t outer = g / indexLen;
                int32_t t = idx.GetValue(static_cast<uint32_t>(g - outer * indexLen));
                if (t < 0) {
                    t += static_cast<int32_t>(selfLen);
                }
                targets[j] = (outer * selfLen + static_cast<uint64_t>(t)) * inner;
            }
            const uint64_t step = shifted ? 2 : 1;
            for (uint64_t j = 0; j < n; j += step) {
                DataCopyPadWithReduce<int32_t, ReduceType::REDUCE_ADD>(outGm_[targets[j]], rowsTensor[j * inner], rowCopy);
            }
            if (shifted && n > 1) {

                SetFlag<HardEvent::MTE3_V>(evensDone);
                WaitFlag<HardEvent::MTE3_V>(evensDone);
                for (uint64_t j = 1; j < n; j += 2) {
                    Duplicate(rowsTensor[j * inner - 4], 0, 4);
                    if (targets[j] == 0) {
                        Gather(head[j / 2 * 8].template ReinterpretCast<uint32_t>(),
                               rowsTensor[j * inner - 4].template ReinterpretCast<uint32_t>(), headOffsets, 0, 4);
                    }
                }
                SetFlag<HardEvent::V_MTE3>(zeroed);
                WaitFlag<HardEvent::V_MTE3>(zeroed);
                for (uint64_t j = 1; j < n; j += 2) {
                    if (targets[j] == 0) {
                        const DataCopyExtParams headCopy{1, 16, 0, 0, 0};
                        const DataCopyExtParams tailCopy{1, rowBytes_ - 16, 0, 0, 0};
                        DataCopyPadWithReduce<int32_t, ReduceType::REDUCE_ADD>(outGm_[0], head[j / 2 * 8], headCopy);
                        DataCopyPadWithReduce<int32_t, ReduceType::REDUCE_ADD>(outGm_[4], rowsTensor[j * inner + 4], tailCopy);
                    } else {
                        DataCopyPadWithReduce<int32_t, ReduceType::REDUCE_ADD>(outGm_[targets[j] - 4],
                                                                              rowsTensor[j * inner - 4], shiftedCopy);
                    }
                }
            }
            SetFlag<HardEvent::MTE3_MTE2>(free[cur]);
            cur = nxt;
        }
        for (uint32_t b = 0; b < 2; ++b) {
            if (busy[b]) {
                WaitFlag<HardEvent::MTE3_MTE2>(free[b]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE3>(loaded[b]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(free[b]);
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(evensDone);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(zeroed);
    }

private:
    __aicore__ inline void LoadChunk(const LocalTensor<int32_t>& dst, uint64_t row, uint64_t end)
    {
        const uint64_t n = end - row < kAtomicChunk ? end - row : kAtomicChunk;
        const DataCopyExtParams copy{1, static_cast<uint32_t>(n * rowBytes_), 0, 0, 0};
        DataCopyPad(dst, sourceGm_[row * tiling_.inner], copy, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> chunkBuf_[2];
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> headBuf_;
    IndexAddTilingData tiling_;
    uint32_t rowBytes_ = 0;
    GlobalTensor<int32_t> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<int32_t> sourceGm_;
    GlobalTensor<int32_t> outGm_;
};

constexpr uint32_t kWaveCap = 24;
constexpr uint32_t kEarlyWaves = 12;
constexpr uint32_t kWaveHist = 32;
constexpr uint32_t kWaveCols = 64;
constexpr uint32_t kRankKeys = 1024;

class PairWaveKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out, GM_ADDR workspace,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;
        const uint64_t shift = ReadShift(tiling);
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<uint64_t>(self) + shift));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<uint64_t>(source) + shift));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        outBits_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(out));
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(GetUserWorkspace(workspace)));
        if (shift == 0) {
            selfGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            sourceGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        const uint32_t selfLen = tiling.selfLen;
        const uint32_t indexLen = tiling.indexLen;
        cols_ = (selfLen + 7) / 8 * 8;
        lanes_ = (indexLen + 63) / 64 * 64;
        srcPitch_ = (indexLen * 4 + 31) / 32 * 32;
        selfAt_ = 2 * srcPitch_ + 32;
        slot_ = selfAt_ + (2 * selfLen * 4 + 31) / 32 * 32;
        valueLanes_ = (2 * cols_ + 2 * indexLen + 8 * kWaveCap + 7) / 8 * 8;
        valuesAt_ = 2 * slot_;
        packAt_ = valuesAt_ + valueLanes_ * 4;
        backAt_ = packAt_ + valueLanes_ * 4;
        sumsAt_ = backAt_ + 2 * cols_ * 4;
        pipe_.InitBuffer(arena_, sumsAt_ + 2 * cols_ * 4);
    }

    __aicore__ inline void Process()
    {
        const uint32_t cores = static_cast<uint32_t>(GetBlockNum());
        const uint32_t core = static_cast<uint32_t>(GetBlockIdx());
        const uint64_t outer = tiling_.outer;
        const uint32_t selfLen = tiling_.selfLen;
        const uint32_t indexLen = tiling_.indexLen;
        const uint32_t cols = cols_;
        const uint32_t n = lanes_;
        const LocalTensor<uint8_t> arena = arena_.Get<uint8_t>();
        const uint32_t tableAt = 0;
        const uint32_t countsAt = kWaveCap * cols;
        const uint32_t histAt = countsAt + cols;

        const uint64_t outBytes = outer * selfLen * 4;
        const uint64_t blocks = (outBytes + 31) / 32;
        const uint64_t zeroFirst = Share64(blocks, cores, core) * 32;
        uint64_t zeroLast = Share64(blocks, cores, core + 1) * 32;
        zeroLast = zeroLast < outBytes ? zeroLast : outBytes;
        const uint32_t chunk = valueLanes_ * 4 / 32 * 32;
        const LocalTensor<float> zeros = Floats(packAt_);
        if (zeroLast > zeroFirst) {
            Duplicate(zeros, 0.0f, static_cast<int32_t>(chunk / 4));
            Sync<HardEvent::V_MTE3>();
            for (uint64_t at = zeroFirst; at < zeroLast; at += chunk) {
                const uint64_t len = zeroLast - at < chunk ? zeroLast - at : chunk;
                DataCopyExtParams store{1, static_cast<uint32_t>(len), 0, 0, 0};
                DataCopyPad(outGm_[at / 4], zeros, store);
            }
        }

        const uint32_t blockCount = cols / 8;
        const uint32_t colFirst = Share(blockCount, cores, core) * 8;
        const uint32_t colLast = Share(blockCount, cores, core + 1) * 8;
        const uint32_t owned = colLast - colFirst;
        const LocalTensor<int32_t> idxInt = Ints(0);
        const LocalTensor<float> idx = Floats(n * 4);
        const LocalTensor<float> record = Floats(slot_);
        const LocalTensor<float> kept = Floats(slot_ + n * 4);
        const LocalTensor<int32_t> found = Ints(valuesAt_);
        const uint32_t laneAt = valuesAt_ + n * 4;
        const uint32_t ownHistAt = laneAt + kWaveCap * kWaveCols * 4;
        const LocalTensor<uint8_t> low = arena[backAt_];
        const LocalTensor<uint8_t> high = arena[backAt_ + (n / 8 + 31) / 32 * 32];
        Duplicate(idxInt, static_cast<int32_t>(1 << 24), static_cast<int32_t>(n));
        Sync<HardEvent::V_MTE2>();
        DataCopyExtParams indexParams{1, indexLen * 4, 0, 0, 0};
        DataCopyPad(idxInt, indexGm_[0], indexParams, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>();
        const int32_t laneCount = static_cast<int32_t>(n);
        Cast(idx, idxInt, RoundMode::CAST_RINT, n);

        Muls(record, idx, -1.0f, laneCount);
        Maxs(record, record, 0.0f, laneCount);
        Mins(record, record, 1.0f, laneCount);
        Muls(record, record, static_cast<float>(static_cast<int32_t>(selfLen)), laneCount);
        Add(idx, idx, record, laneCount);
        CompareScalar(low, idx, static_cast<float>(static_cast<int32_t>(colFirst)), CMPMODE::GE, n);
        CompareScalar(high, idx, static_cast<float>(static_cast<int32_t>(colLast)), CMPMODE::LT, n);
        And(low.template ReinterpretCast<uint16_t>(), low.template ReinterpretCast<uint16_t>(),
            high.template ReinterpretCast<uint16_t>(), n / 16);

        CreateVecIndex(record, 0.0f, n);
        PipeBarrier<PIPE_V>();
        CompareScalar(high, record, static_cast<float>(static_cast<int32_t>(indexLen)), CMPMODE::LT, n);
        And(low.template ReinterpretCast<uint16_t>(), low.template ReinterpretCast<uint16_t>(),
            high.template ReinterpretCast<uint16_t>(), n / 16);
        Muls(record, record, 1024.0f, laneCount);
        Add(record, record, idx, laneCount);
        Adds(record, record, -static_cast<float>(static_cast<int32_t>(colFirst)), laneCount);
        uint64_t selected = 0;
        GatherMask(kept, record, low.template ReinterpretCast<uint32_t>(), true, n, {1, 1, 8, 0}, selected);
        Sync<HardEvent::V_S>();
        const uint32_t hits = static_cast<uint32_t>(selected);
        if (hits > 0) {
            Cast(found, kept, RoundMode::CAST_RINT, hits);
            Sync<HardEvent::V_S>();
        }
        const __ubuf__ int32_t* rec = reinterpret_cast<__ubuf__ int32_t*>(found.GetPhyAddr());
        __ubuf__ int32_t* lane = reinterpret_cast<__ubuf__ int32_t*>(Ints(laneAt).GetPhyAddr());
        for (uint32_t j = 0; j < owned; ++j) {
            count_[j] = 0;
        }
        for (uint32_t h = 0; h < hits; ++h) {
            const uint32_t r = static_cast<uint32_t>(rec[h]);
            const uint32_t j = r & 1023;
            const uint32_t c = count_[j];
            if (c < kWaveCap) {
                lane[c * kWaveCols + j] = static_cast<int32_t>((r >> 10) * 4);
            }
            count_[j] = c + 1;
        }
        uint32_t deepest = 0;
        uint32_t hist[kWaveHist];
        for (uint32_t k = 0; k < kWaveHist; ++k) {
            hist[k] = 0;
        }
        __ubuf__ int32_t* counts = reinterpret_cast<__ubuf__ int32_t*>(found.GetPhyAddr());
        for (uint32_t j = 0; j < owned; ++j) {
            const uint32_t c = count_[j];
            deepest = c > deepest ? c : deepest;
            ++hist[c < kWaveHist ? c : kWaveHist - 1];
            counts[j] = static_cast<int32_t>(c);
        }
        __ubuf__ int32_t* ownHist = reinterpret_cast<__ubuf__ int32_t*>(Ints(ownHistAt).GetPhyAddr());
        for (uint32_t k = 0; k < kWaveHist; ++k) {
            ownHist[k] = static_cast<int32_t>(hist[k]);
        }
        const uint32_t stored = deepest < kWaveCap ? deepest : kWaveCap;
        for (uint32_t w = 0; w < stored; ++w) {
            DataCopy(wsGm_[tableAt + w * cols + colFirst], Ints(laneAt + w * kWaveCols * 4), owned);
        }
        if (owned > 0) {
            DataCopy(wsGm_[countsAt + colFirst], found, owned);
        }
        DataCopy(wsGm_[histAt + core * kWaveHist], Ints(ownHistAt), kWaveHist);
        Sync<HardEvent::MTE3_S>();
        SyncAll<true>();

        const LocalTensor<int32_t> histAll = Ints(backAt_);
        const LocalTensor<int32_t> allCounts = Ints(valuesAt_);
        DataCopy(histAll, wsGm_[histAt], cores * kWaveHist);
        DataCopy(allCounts, wsGm_[countsAt], cols);
        Sync<HardEvent::MTE2_V>();

        DataCopy(Ints(0), wsGm_[0], kEarlyWaves * cols);
        for (uint32_t c = 1; c < cores; ++c) {
            Add(histAll, histAll, histAll[c * kWaveHist], static_cast<int32_t>(kWaveHist));
        }
        Sync<HardEvent::V_S>();
        uint32_t waves = 0;
        uint32_t total = 0;
        uint32_t targets = 0;
        for (uint32_t k = 0; k < kWaveHist; ++k) {
            const uint32_t h = static_cast<uint32_t>(histAll.GetValue(k));
            above_[k] = h;
            total += h * k;
            targets += h;
            waves = h > 0 ? k : waves;
        }
        bool valid = total == indexLen && targets == cols && waves <= kWaveCap;
        const uint64_t pairs = (outer + 1) / 2;
        const uint64_t pairFirst = Share64(pairs, cores, core);
        const uint64_t pairLast = Share64(pairs, cores, core + 1);
        pairFirst_ = pairFirst;
        pairLast_ = pairLast;
        if (valid) {
            valid = Plan(waves);
        }
        if (!valid) {
            Fallback(pairFirst * 2, pairLast * 2 < outer ? pairLast * 2 : outer);
            return;
        }
        Compute(pairFirst, pairLast);
    }

private:
    __aicore__ inline static uint32_t Share(uint32_t total, uint32_t cores, uint32_t core)
    {
        return core * (total / cores) + (core < total % cores ? core : total % cores);
    }

    __aicore__ inline static uint64_t Share64(uint64_t total, uint64_t cores, uint64_t core)
    {
        return core * (total / cores) + (core < total % cores ? core : total % cores);
    }

    __aicore__ inline LocalTensor<float> Floats(uint32_t at)
    {
        return arena_.Get<uint8_t>()[at].template ReinterpretCast<float>();
    }

    __aicore__ inline LocalTensor<int32_t> Ints(uint32_t at)
    {
        return arena_.Get<uint8_t>()[at].template ReinterpretCast<int32_t>();
    }

    __aicore__ inline bool Plan(uint32_t waves)
    {
        const uint32_t selfLen = tiling_.selfLen;
        const uint32_t cols = cols_;
        const int32_t keyCount = static_cast<int32_t>(kRankKeys);
        const uint32_t k4 = kRankKeys * 4;
        uint32_t deeper = 0;
        for (int32_t k = static_cast<int32_t>(kWaveHist) - 1; k >= 0; --k) {
            const uint32_t h = above_[k];
            above_[k] = deeper;
            deeper += h;
        }
        waves_ = waves;
        if (waves > kEarlyWaves) {
            DataCopy(Ints(kEarlyWaves * cols * 4), wsGm_[kEarlyWaves * cols], (waves - kEarlyWaves) * cols);
        }
        const LocalTensor<int32_t> counts = Ints(valuesAt_);
        const LocalTensor<float> keys = Floats(valuesAt_ + k4);
        const LocalTensor<int32_t> ids = Ints(valuesAt_ + 2 * k4);
        LocalTensor<float> pairs = Floats(valuesAt_ + 3 * k4);
        LocalTensor<float> tmp = Floats(valuesAt_ + 5 * k4);
        const LocalTensor<float> sorted = Floats(valuesAt_ + 5 * k4);
        const LocalTensor<int32_t> order = Ints(valuesAt_ + 6 * k4);
        const LocalTensor<int32_t> rank = Ints(valuesAt_ + 7 * k4);
        const LocalTensor<int32_t> colOff = Ints(valuesAt_ + 8 * k4);

        const LocalTensor<int32_t> keyInts = Ints(valuesAt_ + k4);
        const LocalTensor<uint32_t> keyBits = keyInts.template ReinterpretCast<uint32_t>();
        const int32_t keyBase = static_cast<int32_t>(0x3f800000u + kRankKeys - 1);
        Duplicate(keyBits, static_cast<uint32_t>(0), keyCount);
        CreateVecIndex(ids, 0, keyCount);
        PipeBarrier<PIPE_V>();
        ShiftLeft(keyBits, counts.template ReinterpretCast<uint32_t>(), static_cast<uint32_t>(10), cols);
        PipeBarrier<PIPE_V>();
        Adds(keyInts, keyInts, keyBase, static_cast<int32_t>(cols));
        PipeBarrier<PIPE_V>();
        Sub(keyInts, keyInts, ids, static_cast<int32_t>(cols));
        PipeBarrier<PIPE_V>();
        Sort<float, true>(pairs, keys, ids.template ReinterpretCast<uint32_t>(), tmp, keyCount / 32);
        PipeBarrier<PIPE_V>();
        Extract(sorted, order.template ReinterpretCast<uint32_t>(), pairs, keyCount / 32);
        PipeBarrier<PIPE_V>();

        const int32_t laneCount = static_cast<int32_t>(2 * cols);
        const uint32_t span = 2 * cols * 4;
        const LocalTensor<uint32_t> laneIdx = Ints(packAt_).template ReinterpretCast<uint32_t>();
        const LocalTensor<uint32_t> halfIdx = Ints(packAt_ + span).template ReinterpretCast<uint32_t>();
        const LocalTensor<uint32_t> evenIdx = Ints(packAt_ + 2 * span).template ReinterpretCast<uint32_t>();
        const LocalTensor<int32_t> parity = Ints(backAt_);
        CreateVecIndex(laneIdx.template ReinterpretCast<int32_t>(), 0, 2 * cols);
        PipeBarrier<PIPE_V>();
        ShiftRight(halfIdx, laneIdx, static_cast<uint32_t>(1), 2 * cols);
        PipeBarrier<PIPE_V>();
        ShiftLeft(evenIdx, halfIdx, static_cast<uint32_t>(1), 2 * cols);
        ShiftLeft(halfIdx, halfIdx, static_cast<uint32_t>(2), 2 * cols);
        PipeBarrier<PIPE_V>();
        Sub(parity, laneIdx.template ReinterpretCast<int32_t>(), evenIdx.template ReinterpretCast<int32_t>(), laneCount);
        Gather(colOff.template ReinterpretCast<uint32_t>(), order.template ReinterpretCast<uint32_t>(), halfIdx, 0,
               2 * cols);
        PipeBarrier<PIPE_V>();
        ShiftLeft(colOff.template ReinterpretCast<uint32_t>(), colOff.template ReinterpretCast<uint32_t>(),
                  static_cast<uint32_t>(2), 2 * cols);

        const LocalTensor<int32_t> pack = Ints(packAt_);
        Duplicate(pack, 0, static_cast<int32_t>(valueLanes_));
        PipeBarrier<PIPE_V>();
        Muls(pack, parity, static_cast<int32_t>(selfLen * 4), laneCount);
        Muls(parity, parity, static_cast<int32_t>(srcPitch_), laneCount);
        PipeBarrier<PIPE_V>();
        Adds(pack, pack, static_cast<int32_t>(selfAt_), laneCount);
        PipeBarrier<PIPE_V>();
        Add(pack, pack, colOff, laneCount);
        Sync<HardEvent::MTE2_V>();
        uint32_t cursor = 2 * cols;
        for (uint32_t w = 0; w < waves; ++w) {
            const uint32_t lanes = 2 * above_[w];
            size_[w] = lanes;
            start_[w] = cursor;
            Gather(pack[cursor].template ReinterpretCast<uint32_t>(), Ints(0).template ReinterpretCast<uint32_t>(),
                   colOff.template ReinterpretCast<uint32_t>(), w * cols * 4, lanes);
            Add(pack[cursor], pack[cursor], parity, static_cast<int32_t>(lanes));
            cursor += (lanes + 7) / 8 * 8;
        }
        gathered_ = cursor;

        Sync<HardEvent::V_MTE2>();
        preloaded_ = pairFirst_ < pairLast_;
        if (preloaded_) {
            const uint64_t row = pairFirst_ * 2;
            Load(0, row, tiling_.outer - row < 2 ? 1 : 2);
        }

        Muls(keyInts, order, -1, keyCount);
        CreateVecIndex(ids, 0, keyCount);
        PipeBarrier<PIPE_V>();
        Adds(keyInts, keyInts, keyBase, keyCount);
        PipeBarrier<PIPE_V>();
        Sort<float, true>(pairs, keys, ids.template ReinterpretCast<uint32_t>(), tmp, keyCount / 32);
        PipeBarrier<PIPE_V>();
        Extract(sorted, rank.template ReinterpretCast<uint32_t>(), pairs, keyCount / 32);

        const int32_t outLen = static_cast<int32_t>(2 * selfLen);
        const LocalTensor<int32_t> pairRanks = Ints(slot_);
        const LocalTensor<int32_t> skip = Ints(slot_ + span);
        const LocalTensor<int32_t> skipTail = Ints(slot_ + 2 * span);

        const int32_t pad = static_cast<int32_t>((cols - selfLen) * 4);
        Muls(rank, rank, 8, static_cast<int32_t>(cols));
        Adds(pairRanks, rank, 0, static_cast<int32_t>(cols));
        Adds(pairRanks[cols], rank, 4, static_cast<int32_t>(cols));
        CreateVecIndex(skip, 0, 2 * selfLen);
        PipeBarrier<PIPE_V>();
        Adds(skipTail, skip, -static_cast<int32_t>(selfLen), outLen);
        ShiftLeft(skip.template ReinterpretCast<uint32_t>(), skip.template ReinterpretCast<uint32_t>(),
                  static_cast<uint32_t>(2), 2 * selfLen);
        PipeBarrier<PIPE_V>();
        ShiftRight(skipTail.template ReinterpretCast<uint32_t>(), skipTail.template ReinterpretCast<uint32_t>(),
                   static_cast<uint32_t>(31), 2 * selfLen);
        PipeBarrier<PIPE_V>();
        Muls(skipTail, skipTail, -pad, outLen);
        PipeBarrier<PIPE_V>();
        Adds(skipTail, skipTail, pad, outLen);
        PipeBarrier<PIPE_V>();
        Add(skip, skip, skipTail, outLen);
        Gather(Ints(backAt_).template ReinterpretCast<uint32_t>(), pairRanks.template ReinterpretCast<uint32_t>(),
               skip.template ReinterpretCast<uint32_t>(), 0, 2 * selfLen);

        PipeBarrier<PIPE_V>();
        Mins(pack, pack, static_cast<int32_t>(slot_ - 4), static_cast<int32_t>(cursor));
        Maxs(pack, pack, 0, static_cast<int32_t>(cursor));
        return true;
    }

    __aicore__ inline void Load(uint32_t slot, uint64_t row, uint64_t rows)
    {
        const uint32_t at = slot * slot_;
        const uint32_t indexBytes = tiling_.indexLen * 4;
        const DataCopyExtParams src = indexBytes % 32 == 0
            ? DataCopyExtParams{1, static_cast<uint32_t>(rows) * indexBytes, 0, 0, 0}
            : DataCopyExtParams{static_cast<uint16_t>(rows), indexBytes, 0, 0, 0};
        const DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(Floats(at), sourceGm_[row * tiling_.indexLen], src, pad);
        const DataCopyExtParams self{1, static_cast<uint32_t>(rows * tiling_.selfLen * 4), 0, 0, 0};
        DataCopyPad(Floats(at + selfAt_), selfGm_[row * tiling_.selfLen], self, pad);
    }

    __aicore__ inline void Compute(uint64_t pairFirst, uint64_t pairLast)
    {
        const uint64_t outer = tiling_.outer;
        const uint32_t selfLen = tiling_.selfLen;
        if (pairFirst >= pairLast) {
            return;
        }
        for (uint32_t s = 0; s < 2; ++s) {
            Duplicate(Floats(s * slot_ + selfAt_ - 32), 0.0f, 8);
        }

        Sync<HardEvent::V_MTE2>();
        const LocalTensor<float> values = Floats(valuesAt_);
        const LocalTensor<uint32_t> pack = arena_.Get<uint8_t>()[packAt_].template ReinterpretCast<uint32_t>();
        const LocalTensor<uint32_t> back = arena_.Get<uint8_t>()[backAt_].template ReinterpretCast<uint32_t>();
        TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        TEventID written[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>(),
                               GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>()};
        const TEventID ready = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        bool pending[2] = {false, false};
        if (!preloaded_) {
            Load(0, pairFirst * 2, outer - pairFirst * 2 < 2 ? 1 : 2);
        }
        SetFlag<HardEvent::MTE2_V>(full[0]);
        for (uint64_t p = pairFirst; p < pairLast; ++p) {
            const uint32_t s = static_cast<uint32_t>((p - pairFirst) & 1);
            const uint64_t row = p * 2;
            const uint64_t rows = outer - row < 2 ? 1 : 2;
            if (p + 1 < pairLast) {
                const uint32_t ns = 1 - s;
                if (pending[ns]) {
                    WaitFlag<HardEvent::MTE3_MTE2>(written[ns]);
                    pending[ns] = false;
                }
                Load(ns, row + 2, outer - row - 2 < 2 ? 1 : 2);
                SetFlag<HardEvent::MTE2_V>(full[ns]);
            }
            WaitFlag<HardEvent::MTE2_V>(full[s]);
            const LocalTensor<float> slot = Floats(s * slot_);
            Gather(values, slot, pack, 0, gathered_);
            for (uint32_t w = 0; w < waves_; ++w) {
                Add(values, values, values[start_[w]], static_cast<int32_t>(size_[w]));
            }
            Gather(Floats(s * slot_ + selfAt_), values, back, 0, static_cast<uint32_t>(rows * selfLen));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            const DataCopyExtParams store{1, static_cast<uint32_t>(rows * selfLen * 4), 0, 0, 0};
            DataCopyPadWithReduce<int32_t, ReduceType::REDUCE_ADD>(outBits_[row * selfLen],
                                                                  Ints(s * slot_ + selfAt_), store);
            SetFlag<HardEvent::MTE3_MTE2>(written[s]);
            pending[s] = true;
        }
        for (uint32_t s = 0; s < 2; ++s) {
            if (pending[s]) {
                WaitFlag<HardEvent::MTE3_MTE2>(written[s]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(full[s]);
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(written[s]);
        }
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(ready);
    }

    __aicore__ inline void Fallback(uint64_t first, uint64_t last)
    {
        const uint64_t selfLen = tiling_.selfLen;
        const uint64_t indexLen = tiling_.indexLen;
        for (uint64_t o = first; o < last; ++o) {
            for (uint64_t j = 0; j < selfLen; ++j) {
                outGm_.SetValue(o * selfLen + j, selfGm_.GetValue(o * selfLen + j));
            }
            for (uint64_t i = 0; i < indexLen; ++i) {
                int32_t t = indexGm_.GetValue(i);
                if (t < 0) {
                    t += static_cast<int32_t>(selfLen);
                }
                const uint64_t to = o * selfLen + static_cast<uint64_t>(t);
                outGm_.SetValue(to, outGm_.GetValue(to) + sourceGm_.GetValue(o * indexLen + i));
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
    IndexAddTilingData tiling_;
    uint32_t cols_ = 0;
    uint32_t lanes_ = 0;
    uint32_t srcPitch_ = 0;
    uint32_t selfAt_ = 0;
    uint32_t slot_ = 0;
    uint32_t valueLanes_ = 0;
    uint32_t valuesAt_ = 0;
    uint32_t packAt_ = 0;
    uint32_t backAt_ = 0;
    uint32_t waves_ = 0;
    uint32_t gathered_ = 0;
    uint32_t sumsAt_ = 0;
    uint64_t pairFirst_ = 0;
    uint64_t pairLast_ = 0;
    bool preloaded_ = false;
    uint32_t count_[kWaveCols];
    uint32_t above_[kWaveHist];
    uint32_t size_[kWaveCap];
    uint32_t start_[kWaveCap];
    GlobalTensor<float> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<float> sourceGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<int32_t> outBits_;
    GlobalTensor<int32_t> wsGm_;
};

// 按列块聚合多行更新，通过工作区整理结果，减少零散的全局内存访问。
class ColumnSlabKernel {
public:
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out, GM_ADDR workspace,
                                const IndexAddTilingData& tiling)
    {
        tiling_ = tiling;
        const uint64_t shift = ReadShift(tiling);
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<uint64_t>(self) + shift));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reinterpret_cast<uint64_t>(source) + shift));
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(out));
        slabGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace)));
        if (shift == 0) {
            selfGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            sourceGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        const uint32_t rows = tiling.unit;
        const uint32_t pool = 2 * tiling.selfTile * rows > 4 * tiling.sourceTile * rows
            ? 2 * tiling.selfTile * rows : 4 * tiling.sourceTile * rows;
        pipe_.InitBuffer(poolBuf_, pool * 4);
        pipe_.InitBuffer(indexBuf_, (tiling.indexLen * 2 + 31) / 32 * 32);
    }

    __aicore__ inline void Process()
    {
        uint64_t first = 0;
        uint64_t last = 0;
        OwnedSlices(tiling_, first, last);
        if (first >= last) {
            return;
        }
        const uint32_t rows = tiling_.unit;
        const uint32_t selfLen = tiling_.selfLen;
        const uint32_t indexLen = tiling_.indexLen;
        const LocalTensor<float> pool = poolBuf_.Get<float>();

        const LocalTensor<int32_t> positions = pool.template ReinterpretCast<int32_t>();
        const LocalTensor<int32_t> behind = pool[(indexLen + 7) / 8 * 8].template ReinterpretCast<int32_t>();
        const int32_t count = static_cast<int32_t>(indexLen);
        DataCopyExtParams indexParams{1, indexLen * 4, 0, 0, 0};
        DataCopyPad(positions, indexGm_[0], indexParams, DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>();
        Muls(behind, positions, static_cast<int32_t>(-1), count);
        Maxs(behind, behind, static_cast<int32_t>(0), count);
        Mins(behind, behind, static_cast<int32_t>(1), count);
        Muls(behind, behind, static_cast<int32_t>(selfLen), count);
        Add(positions, positions, behind, count);
        Cast(indexBuf_.Get<int16_t>(), positions, RoundMode::CAST_NONE, indexLen);
        Sync<HardEvent::V_S>();

        for (uint64_t group = first; group < last; group += rows) {
            const uint16_t live = static_cast<uint16_t>(last - group < rows ? last - group : rows);
            const uint64_t slab = group * selfLen;
            if (group > first) {
                Sync<HardEvent::MTE3_MTE2>();
            }
            LoadSelf(pool, group, live);
            Sync<HardEvent::MTE3_MTE2>();
            SetAtomicAdd<float>();
            AddSource(pool, group, live, slab);
            SetAtomicNone();
            Sync<HardEvent::MTE3_MTE2>();
            StoreOut(pool, group, live, slab);
        }
    }

private:

    __aicore__ inline static void Turn(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rows,
                                       uint32_t cols, uint32_t srcPitch, uint32_t dstPitch)
    {
        uint64_t to[16];
        uint64_t from[16];
        const TransDataTo5HDParams params{false, false, static_cast<uint8_t>(cols / 8),
                                          static_cast<uint16_t>(dstPitch), 1};
        const uint64_t srcAt = reinterpret_cast<uint64_t>(src.GetPhyAddr());
        const uint64_t dstAt = reinterpret_cast<uint64_t>(dst.GetPhyAddr());
        for (uint32_t r0 = 0; r0 < rows; r0 += 16) {
            for (uint32_t r = 0; r < 16; ++r) {
                from[r] = srcAt + (r0 + r) * srcPitch * 4;
            }
            for (uint32_t c = 0; c < 8; ++c) {
                to[2 * c] = dstAt + (c * dstPitch + r0) * 4;
                to[2 * c + 1] = dstAt + (c * dstPitch + r0 + 8) * 4;
            }
            TransDataTo5HD<uint32_t>(to, from, params);
        }
    }

    __aicore__ inline void LoadSelf(const LocalTensor<float>& pool, uint64_t group, uint16_t live)
    {
        const uint32_t rows = tiling_.unit;
        const uint32_t selfLen = tiling_.selfLen;
        const uint32_t tile = tiling_.selfTile;
        const LocalTensor<float> turned = pool;
        const LocalTensor<float> staged = pool[tile * rows];
        for (uint32_t p0 = 0; p0 < selfLen; p0 += tile) {
            const uint32_t width = selfLen - p0 < tile ? selfLen - p0 : tile;
            const uint32_t pitch = (width + 7) / 8 * 8;
            if (p0 > 0) {
                Sync<HardEvent::V_MTE2>();
            }
            DataCopyExtParams load{live, width * 4, (selfLen - width) * 4, 0, 0};
            DataCopyPad(staged, selfGm_[group * selfLen + p0], load, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
            Sync<HardEvent::MTE2_V>();
            Turn(turned, staged, rows, pitch, pitch, rows);
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams store{1, width * rows * 4, 0, 0, 0};
            DataCopyPad(slabGm_[group * selfLen + p0 * rows], turned, store);
            Sync<HardEvent::MTE3_V>();
        }
    }

    __aicore__ inline void AddSource(const LocalTensor<float>& pool, uint64_t group, uint16_t live, uint64_t slab)
    {
        const uint32_t rows = tiling_.unit;
        const uint32_t indexLen = tiling_.indexLen;
        const uint32_t tile = tiling_.sourceTile;
        const uint32_t tiles = (indexLen + tile - 1) / tile;
        const uint32_t slot = tile * rows;
        const GlobalTensor<float> slabAt = slabGm_[slab];
        const __ubuf__ uint64_t* words = reinterpret_cast<__ubuf__ uint64_t*>(indexBuf_.Get<int16_t>().GetPhyAddr());
        TPipe* pipe = GetTPipePtr();
        TEventID loaded[2];
        TEventID turnedFrom[2];
        TEventID added[2];
        for (uint32_t s = 0; s < 2; ++s) {
            loaded[s] = pipe->AllocEventID<HardEvent::MTE2_V>();
            turnedFrom[s] = pipe->AllocEventID<HardEvent::V_MTE2>();
            added[s] = pipe->AllocEventID<HardEvent::MTE3_V>();
        }
        LoadTile(pool, group, live, 0);
        SetFlag<HardEvent::MTE2_V>(loaded[0]);
        for (uint32_t t = 0; t < tiles; ++t) {
            const uint32_t s = t & 1;
            if (t + 1 < tiles) {
                if (t >= 1) {
                    WaitFlag<HardEvent::V_MTE2>(turnedFrom[1 - s]);
                }
                LoadTile(pool[(1 - s) * slot], group, live, t + 1);
                SetFlag<HardEvent::MTE2_V>(loaded[1 - s]);
            }
            WaitFlag<HardEvent::MTE2_V>(loaded[s]);
            if (t >= 2) {
                WaitFlag<HardEvent::MTE3_V>(added[s]);
            }
            uint32_t width = 0;
            const uint32_t k0 = TileSpan(t, width);
            const uint32_t pitch = (width + 7) / 8 * 8;
            const LocalTensor<float> turned = pool[(2 + s) * slot];
            Turn(turned, pool[s * slot], rows, pitch, pitch, rows);
            SetFlag<HardEvent::V_MTE2>(turnedFrom[s]);
            Sync<HardEvent::V_MTE3>();
            if (rows == 64) {
                AddColumns<6>(slabAt, turned, words, k0, width);
            } else if (rows == 32) {
                AddColumns<5>(slabAt, turned, words, k0, width);
            } else {
                AddColumns<4>(slabAt, turned, words, k0, width);
            }
            SetFlag<HardEvent::MTE3_V>(added[s]);
        }
        const uint32_t tail = tiles < 2 ? tiles : 2;
        for (uint32_t t = tiles - tail; t < tiles; ++t) {
            WaitFlag<HardEvent::MTE3_V>(added[t & 1]);
            WaitFlag<HardEvent::V_MTE2>(turnedFrom[t & 1]);
        }
        for (uint32_t s = 0; s < 2; ++s) {
            pipe->ReleaseEventID<HardEvent::MTE2_V>(loaded[s]);
            pipe->ReleaseEventID<HardEvent::V_MTE2>(turnedFrom[s]);
            pipe->ReleaseEventID<HardEvent::MTE3_V>(added[s]);
        }
    }

    template <uint32_t kShift>
    __aicore__ inline static void AddColumns(const GlobalTensor<float>& slabAt, const LocalTensor<float>& turned,
                                             const __ubuf__ uint64_t* words, uint32_t k0, uint32_t width)
    {
        const uint32_t rows = 1u << kShift;
        uint32_t k = 0;
        uint32_t at = 0;
        for (; k + 4 <= width; k += 4) {
            const uint64_t word = words[(k0 + k) >> 2];
            DataCopy(slabAt[(word & 0xFFFF) << kShift], turned[at], rows);
            DataCopy(slabAt[((word >> 16) & 0xFFFF) << kShift], turned[at + rows], rows);
            DataCopy(slabAt[((word >> 32) & 0xFFFF) << kShift], turned[at + 2 * rows], rows);
            DataCopy(slabAt[(word >> 48) << kShift], turned[at + 3 * rows], rows);
            at += 4 * rows;
        }
        for (; k < width; ++k, at += rows) {
            const uint64_t word = words[(k0 + k) >> 2];
            DataCopy(slabAt[((word >> (16 * ((k0 + k) & 3))) & 0xFFFF) << kShift], turned[at], rows);
        }
    }

    __aicore__ inline uint32_t TileSpan(uint32_t t, uint32_t& width) const
    {
        const uint32_t indexLen = tiling_.indexLen;
        const uint32_t tiles = (indexLen + tiling_.sourceTile - 1) / tiling_.sourceTile;
        const uint32_t units = (indexLen + 15) / 16;
        const uint32_t base = units / tiles;
        const uint32_t extra = units % tiles;
        const uint32_t k0 = 16 * (t * base + (t < extra ? t : extra));
        const uint32_t span = 16 * (base + (t < extra ? 1 : 0));
        width = indexLen - k0 < span ? indexLen - k0 : span;
        return k0;
    }

    __aicore__ inline void LoadTile(const LocalTensor<float>& dst, uint64_t group, uint16_t live, uint32_t t)
    {
        const uint32_t indexLen = tiling_.indexLen;
        uint32_t width = 0;
        const uint32_t k0 = TileSpan(t, width);
        DataCopyExtParams load{live, width * 4, (indexLen - width) * 4, 0, 0};
        DataCopyPad(dst, sourceGm_[group * indexLen + k0], load, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
    }

    __aicore__ inline void StoreOut(const LocalTensor<float>& pool, uint64_t group, uint16_t live, uint64_t slab)
    {
        const uint32_t rows = tiling_.unit;
        const uint32_t selfLen = tiling_.selfLen;
        const uint32_t tile = tiling_.selfTile;
        const LocalTensor<float> columns = pool;
        const LocalTensor<float> outRows = pool[tile * rows];
        for (uint32_t p0 = 0; p0 < selfLen; p0 += tile) {
            const uint32_t width = selfLen - p0 < tile ? selfLen - p0 : tile;
            const uint32_t turnRows = (width + 15) / 16 * 16;
            if (p0 > 0) {
                Sync<HardEvent::V_MTE2>();
            }
            DataCopyExtParams load{1, width * rows * 4, 0, 0, 0};
            DataCopyPad(columns, slabGm_[slab + p0 * rows], load, DataCopyPadExtParams<float>{false, 0, 0, 0.0f});
            Sync<HardEvent::MTE2_V>();
            Turn(outRows, columns, turnRows, rows, rows, turnRows);
            Sync<HardEvent::V_MTE3>();
            const uint32_t ubGap = (turnRows * 4 - (width * 4 + 31) / 32 * 32) / 32;
            DataCopyExtParams store{live, width * 4, ubGap, (selfLen - width) * 4, 0};
            DataCopyPad(outGm_[group * selfLen + p0], outRows, store);
            Sync<HardEvent::MTE3_V>();
        }
    }

    IndexAddTilingData tiling_;
    GlobalTensor<float> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<float> sourceGm_;
    GlobalTensor<float> outGm_;
    GlobalTensor<float> slabGm_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> poolBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
};

template <typename T, typename E, Kind K>
__aicore__ inline void RunVector(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out, GM_ADDR workspace,
                                 const IndexAddTilingData& tiling)
{
    if (tiling.path == 4) {
        SplitRowKernel<T> op;
        op.Init(self, index, source, out, tiling);
        op.Process();
        return;
    }
    if (tiling.path == 1) {
        RowKernel<T> op;
        op.Init(self, index, source, out, tiling);
        op.Process();
        return;
    }
    if constexpr (std::is_same<T, float>::value) {
        if (tiling.path == 10) {
            PairWaveKernel op;
            op.Init(self, index, source, out, workspace, tiling);
            op.Process();
            return;
        }
    }
    if (tiling.path == 2 || tiling.path == 8) {
        uint64_t first = 0;
        uint64_t last = 0;
        OwnedSlices(tiling, first, last);
        LayerKernel<T> op;
        op.Init(self, index, source, out, workspace, tiling);
        if constexpr (sizeof(T) == 2) {
            if (tiling.path == 8) {
                if (op.PlanSlots(first, last)) {
                    op.ProcessLanes(first, last);
                    return;
                }
                if (first < last) {
                    ScalarKernel<E, K> scalar;
                    scalar.Init(self, index, source, out, tiling);
                    scalar.Process();
                }
                return;
            }
        }
        if (op.Plan()) {
            if constexpr (sizeof(T) == 2) {
                op.Process(first, last);
            } else {
                op.Process(first, last);
            }
            return;
        }
        if (first >= last) {
            return;
        }
    }
    ScalarKernel<E, K> op;
    op.Init(self, index, source, out, tiling);
    op.Process();
}
}

extern "C" __global__ __aicore__ void index_add(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR out,
                                                GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (std::is_same<DTYPE_SELF, float>::value) {
        if (TILING_KEY_IS(1)) {
            ColumnSlabKernel op;
            op.Init(self, index, source, out, workspace, tilingData);
            op.Process();
            return;
        }
    }
    if constexpr (std::is_same<DTYPE_SELF, bfloat16_t>::value) {
        if (TILING_KEY_IS(2)) {
            Bf16LayerKernel op;
            op.Init(self, index, source, out, tilingData);
            op.ProcessHits();
            return;
        }
    }

    if (!TILING_KEY_IS(0)) {
        return;
    }
    if constexpr (std::is_same<DTYPE_SELF, int8_t>::value) {
        if (tilingData.path == 5) {
            WideInt8Kernel op;
            op.Init(self, index, source, out, tilingData);
            op.Process();
        } else if (tilingData.path == 3) {
            OrderedRowKernel<int8_t> op;
            op.Init(self, index, source, out, tilingData);
            op.Process();
        } else {
            ScalarKernel<int8_t, Kind::kInt8> op;
            op.Init(self, index, source, out, tilingData);
            op.Process();
        }
    } else if constexpr (std::is_same<DTYPE_SELF, int32_t>::value) {
        if (tilingData.path == 9) {
            AtomicRowKernel op;
            op.Init(self, index, source, out, tilingData);
            op.Process();
        } else {
            RunVector<int32_t, int32_t, Kind::kInt32>(self, index, source, out, workspace, tilingData);
        }
    } else if constexpr (std::is_same<DTYPE_SELF, float>::value) {
        RunVector<float, float, Kind::kFloat>(self, index, source, out, workspace, tilingData);
    } else if constexpr (std::is_same<DTYPE_SELF, half>::value) {
        RunVector<half, uint16_t, Kind::kFp16>(self, index, source, out, workspace, tilingData);
    } else if (tilingData.path == 7 || tilingData.path == 11) {
        Bf16LayerKernel op;
        op.Init(self, index, source, out, tilingData);
        if (tilingData.path == 11) {
            op.ProcessStacks();
        } else {
            op.Process();
        }
    } else if (tilingData.path == 3) {
        OrderedRowKernel<DTYPE_SELF> op;
        op.Init(self, index, source, out, tilingData);
        op.Process();
    } else {
        ScalarKernel<uint16_t, Kind::kBf16> op;
        op.Init(self, index, source, out, tilingData);
        op.Process();
    }
}
