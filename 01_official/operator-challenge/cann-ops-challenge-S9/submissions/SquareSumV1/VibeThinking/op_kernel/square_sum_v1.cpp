#include "kernel_operator.h"

#include <cstddef>
#include <type_traits>

namespace {
using namespace AscendC;

constexpr uint32_t kMaxGroups = 8;
constexpr uint32_t kStageElems = 16384;
constexpr uint32_t kAccElems = 1024;
constexpr uint32_t kFlatElems = 16384;
constexpr uint32_t kLanes = 64;
constexpr uint32_t kMaxRepeat = 255;
constexpr uint32_t kMaxRepStride = 255;
constexpr uint32_t kGroupRows = 8;
constexpr uint32_t kFloatLanes = 8;
constexpr uint32_t kMaxSlabRows = 32;
constexpr uint32_t kPackRows = 16;
constexpr uint32_t kMaxBlockFoldBatch = 15;
constexpr uint32_t kUbBytes = 190 * 1024;

template <typename T>
class SquareSumV1Kernel {
public:

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const SquareSumV1TilingData& tiling)
    {
        const uint32_t flags = tiling.flags;
        groupCount_ = (flags >> 16) & 0xF;
        reduceCores_ = tiling.reduceCores;
        partialPitch_ = tiling.partialPitch;
        flatBatch_ = tiling.flatBatch;
        blockFold_ = (flags & 8) != 0;
        slotFloats_ = tiling.slotFloats;
        partialBase_ = tiling.partialBase;
        tailReduced_ = (flags & 1) != 0;
        rowSplit_ = (flags & 4) != 0;
        blockElems_ = 32 / sizeof(T);
        outputCount_ = tiling.outputCount;
        reduceCount_ = tiling.reduceCount;
        runLength_ = tiling.runLength;
        runCount_ = reduceCount_ / runLength_;
        innerWidth_ = tiling.innerWidth;
        contiguousOutputs_ = tiling.contiguousOutputs;
        contiguousRows_ = tiling.contiguousRows;
        for (uint32_t i = 0; i < kMaxGroups; ++i) {
            size_[i] = tiling.groupSize[i];
            reduced_[i] = (flags >> (8 + i)) & 1;
        }
        uint64_t stride = 1;
        for (uint32_t i = groupCount_; i > 0; --i) {
            stride_[i - 1] = stride;
            stride *= size_[i - 1];
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));

        xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if (reduceCores_ > 1 || rowSplit_ || slotFloats_ > 0) {
            wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace)));
        }

        flatPacked_ = tailReduced_ && flatBatch_ > 0 && !blockFold_ && kFusedAccum && CanFoldPacked();

        narrowAccum_ = kNarrowCapable && flatPacked_ && runCount_ > 1 &&
                       (kPackRows * runLength_ * sizeof(T)) % 32 == 0 &&
                       runLength_ + kPackRows <= 128;
        scratchFloats_ = flatBatch_ > kMaxRepeat ? flatBatch_ : kMaxRepeat;
        if (!tailReduced_ && innerWidth_ > scratchFloats_) {
            scratchFloats_ = static_cast<uint32_t>(innerWidth_);
        }
        scratchFloats_ = (scratchFloats_ + 7) / 8 * 8;
        if (scratchFloats_ > kAccElems) {
            scratchFloats_ = kAccElems;
        }

        flatFloats_ = kFlatElems;
        stageElems_ = kStageElems;
        if (tailReduced_ && flatBatch_ == 0) {

            const uint32_t accBytes = (3 * scratchFloats_ + kAccElems) * sizeof(float);
            const uint32_t want = (kUbBytes - accBytes) / 8 / 16 * 16;
            if (want > stageElems_) {
                stageElems_ = want;
            }
            flatFloats_ = stageElems_ * sizeof(T) / sizeof(float);
        } else if (!tailReduced_) {

            const uint32_t width = static_cast<uint32_t>(innerWidth_);
            const uint32_t padded = kGroupRows * ((width + 7) / 8 * 8);
            flatFloats_ = kGroupRows * width > padded ? kGroupRows * width : padded;

            const uint32_t fixed =
                (flatFloats_ + 3 * scratchFloats_ + kAccElems) * sizeof(float);
            const uint32_t want = (kUbBytes - fixed) / 4 / kStageAlign * kStageAlign;
            if (want > stageElems_) {
                stageElems_ = want;
            }
        } else if (flatBatch_ > 0 && !blockFold_) {

            const uint32_t padded = flatPacked_ ? static_cast<uint32_t>(runLength_)
                                                : (static_cast<uint32_t>(runLength_) + 7) / 8 * 8;

            const uint32_t half = (flatBatch_ * padded + kStageAlign / 2 - 1) / (kStageAlign / 2) *
                                  (kStageAlign / 2);
            stageElems_ = 2 * half;
            flatFloats_ = (flatBatch_ * static_cast<uint32_t>(runLength_) + 128 + 7) / 8 * 8;
            if (narrowAccum_) {

                flatFloats_ = (flatBatch_ * static_cast<uint32_t>(runLength_) + 256 + 15) / 16 * 8;
            }
        }

        stageElems_ = stageElems_ / kStageAlign * kStageAlign;
        halfElems_ = stageElems_ / 2;

        const uint32_t stageBytes = stageElems_ * sizeof(T);
        const uint32_t fixedBytes =
            (flatFloats_ + 3 * scratchFloats_ + kAccElems) * sizeof(float) +
            2 * (kMaxBlockFoldBatch + 1) * kLanes * sizeof(float);
        doubleStage_ = blockFold_ && fixedBytes + 2 * stageBytes <= kUbBytes;
        pipe_.InitBuffer(rawBuf_, doubleStage_ ? 2 * stageBytes : stageBytes);
        if constexpr (sizeof(T) != sizeof(float)) {

            if (!blockFold_ && !flatPacked_) {
                const uint32_t valueFloats =
                    (tailReduced_ && flatBatch_ == 0) ? stageElems_ : halfElems_;
                pipe_.InitBuffer(valueBuf_, valueFloats * sizeof(float));
            }
        }
        pipe_.InitBuffer(flatBuf_, flatFloats_ * sizeof(float));
        pipe_.InitBuffer(accBuf_, scratchFloats_ * sizeof(float));
        pipe_.InitBuffer(partialBuf_, scratchFloats_ * sizeof(float));
        pipe_.InitBuffer(outBuf_, scratchFloats_ * sizeof(float));
        pipe_.InitBuffer(chunkBuf_, kAccElems * sizeof(float));

        flatGather_ = !tailReduced_ && innerWidth_ % kFloatLanes != 0 &&
                      static_cast<uint64_t>(innerWidth_) <= kAccElems;
        if (rowSplit_ || flatGather_) {

            LocalTensor<int32_t> ramp = chunkBuf_.Get<int32_t>();
            for (int32_t i = 0; i < static_cast<int32_t>(kGroupRows); ++i) {
                ramp.SetValue(i, i * static_cast<int32_t>(sizeof(float)));
            }
            uint32_t built = kGroupRows;
            while (built < kAccElems) {
                PipeBarrier<PIPE_V>();
                Adds(ramp[built], ramp, static_cast<int32_t>(built * sizeof(float)),
                     static_cast<int32_t>(built));
                built *= 2;
            }
            PipeBarrier<PIPE_V>();
        }
        if (blockFold_) {

            pipe_.InitBuffer(foldBuf_, 2 * (kMaxBlockFoldBatch + 1) * kLanes * sizeof(float));

            pipe_.InitBuffer(redBuf_, (flatFloats_ / 8 + flatFloats_ / 64 + 2 * kLanes) * sizeof(float));

            Duplicate(flatBuf_.Get<float>(), 0.0f, static_cast<int32_t>(flatFloats_));
            LocalTensor<float> mask = foldBuf_.Get<float>();
            const uint32_t spill = static_cast<uint32_t>(runLength_ % kLanes);
            Duplicate(mask, 0.0f, static_cast<int32_t>((flatBatch_ + 1) * kLanes));
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 1; r <= flatBatch_; ++r) {
                Duplicate(mask[r * kLanes], 1.0f, static_cast<int32_t>(r * spill));
            }
            PipeBarrier<PIPE_V>();
        }

        if (reduceCores_ > 1) {

            const uint64_t index = GetBlockIdx();
            const uint64_t slab = SlabRows(static_cast<uint32_t>(innerWidth_));
            if (FusedSlab(static_cast<uint32_t>(innerWidth_)) && slab * reduceCores_ < reduceCount_) {

                rowBegin_ = index * slab;
                rowEnd_ = reduceCount_;
                stepStride_ = reduceCores_ * slab;
                if (rowBegin_ > reduceCount_) {
                    rowBegin_ = reduceCount_;
                }
            } else {
                const uint64_t base = reduceCount_ / reduceCores_;
                const uint64_t extra = reduceCount_ % reduceCores_;
                rowBegin_ = index * base + (index < extra ? index : extra);
                rowEnd_ = rowBegin_ + base + (index < extra ? 1 : 0);
            }
            begin_ = 0;
            end_ = outputCount_;
            return;
        }
        if (rowSplit_) {

            gBegin_ = SpanBegin(GetBlockIdx());
            gEnd_ = SpanBegin(GetBlockIdx() + 1);
            begin_ = 0;
            end_ = outputCount_;
            return;
        }
        if ((flags & 2) != 0) {

            const uint64_t rows = outputCount_ / innerWidth_;
            const uint64_t cores = GetBlockNum();
            const uint64_t index = GetBlockIdx();
            const uint64_t base = rows / cores;
            const uint64_t extra = rows % cores;
            const uint64_t first = index * base + (index < extra ? index : extra);
            const uint64_t owned = base + (index < extra ? 1 : 0);
            begin_ = first * innerWidth_;
            end_ = begin_ + owned * innerWidth_;
            return;
        }

        const uint64_t blocks = (outputCount_ + blockElems_ - 1) / blockElems_;
        const uint64_t cores = GetBlockNum();
        const uint64_t index = GetBlockIdx();
        const uint64_t base = blocks / cores;
        const uint64_t extra = blocks % cores;
        const uint64_t first = index * base + (index < extra ? index : extra);
        const uint64_t owned = base + (index < extra ? 1 : 0);
        begin_ = first * blockElems_;
        end_ = begin_ + owned * blockElems_;
        if (end_ > outputCount_) {
            end_ = outputCount_;
        }
    }

    template <int kMode>
    __aicore__ inline void Process()
    {
        if constexpr (kMode == 0) {
            SplitColumnMode();
            return;
        }
        if (begin_ >= end_) {
            return;
        }
        if constexpr (kMode == 1) {
            FlatRowMode();
        } else if constexpr (kMode == 2) {
            RowMode();
        } else {
            ColumnMode();
        }
    }

private:
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const TEventID id = GetTPipePtr()->FetchEventID<event>();
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline uint64_t SpanBegin(uint64_t core) const
    {
        const uint64_t total = (outputCount_ / innerWidth_) * reduceCount_;
        const uint64_t cores = GetBlockNum();
        const uint64_t base = total / cores;
        const uint64_t extra = total % cores;
        return core * base + (core < extra ? core : extra);
    }

    // 分别解码输出坐标与归约坐标，支持多个不相邻归约轴。
    __aicore__ inline uint64_t OutputBase(uint64_t index) const
    {
        uint64_t offset = 0;
        for (uint32_t i = groupCount_; i > 0; --i) {
            const uint32_t g = i - 1;
            if (reduced_[g]) {
                continue;
            }
            offset += (index % size_[g]) * stride_[g];
            index /= size_[g];
        }
        return offset;
    }

    __aicore__ inline uint64_t ReduceBase(uint64_t index, bool skipTail) const
    {
        uint64_t offset = 0;
        for (uint32_t i = groupCount_; i > 0; --i) {
            const uint32_t g = i - 1;
            if (!reduced_[g] || (skipTail && g == groupCount_ - 1)) {
                continue;
            }
            offset += (index % size_[g]) * stride_[g];
            index /= size_[g];
        }
        return offset;
    }

    __aicore__ inline uint32_t PaddedPitch(uint64_t length) const
    {
        const uint32_t bytes = static_cast<uint32_t>(length) * static_cast<uint32_t>(sizeof(T));
        return ((bytes + 31) / 32 * 32) / static_cast<uint32_t>(sizeof(T));
    }

    __aicore__ inline void LoadRows(const LocalTensor<T>& dst, uint64_t offset, uint32_t rows, uint32_t length,
                                    uint64_t gapElems)
    {
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(length * sizeof(T)),
                                 static_cast<uint32_t>(gapElems * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, xGm_[offset], params, pad);
    }

    __aicore__ inline void Load(const LocalTensor<T>& dst, uint64_t offset, uint32_t count)
    {

        constexpr uint32_t lanes = 32 / sizeof(T);
        if ((offset % lanes) == 0 && (count % lanes) == 0) {
            DataCopy(dst, xGm_[offset], count);
            return;
        }
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(dst, xGm_[offset], params, pad);
    }

    __aicore__ inline void Store(uint64_t offset, const LocalTensor<T>& src, uint32_t count)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[offset], src, params);
    }

    __aicore__ inline LocalTensor<float> Value()
    {
        if constexpr (sizeof(T) == sizeof(float)) {
            return rawBuf_.Get<float>();
        } else {
            return valueBuf_.Get<float>();
        }
    }

    static constexpr uint32_t kStageAlign = 64 / sizeof(T);

    static constexpr bool kFusedAccum =
        IsSameType<T, float>::value || IsSameType<T, half>::value;

    // 平方结果累加到 float 缓冲；支持融合指令时直接完成平方与累加。
    __aicore__ inline void AccumSquares(const LocalTensor<float>& acc, const LocalTensor<T>& src, uint32_t count)
    {
        if constexpr (kFusedAccum) {
            PipeBarrier<PIPE_V>();
            MulAddDst(acc, src, src, static_cast<int32_t>(count));
        } else {
            LocalTensor<float> v = SquareSlab(src, count);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, v, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void SquareInto(const LocalTensor<float>& dst, const LocalTensor<T>& src, uint32_t count)
    {
        PipeBarrier<PIPE_V>();
        if constexpr (sizeof(T) == sizeof(float)) {
            LocalTensor<float> v = src.template ReinterpretCast<float>();
            Mul(dst, v, v, static_cast<int32_t>(count));
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Mul(dst, dst, dst, static_cast<int32_t>(count));
        }
    }

    __aicore__ inline void Square(uint32_t count)
    {
        LocalTensor<float> value = Value();
        if constexpr (sizeof(T) == sizeof(float)) {
            Mul(value, value, value, static_cast<int32_t>(count));
        } else {
            Cast(value, rawBuf_.Get<T>(), RoundMode::CAST_NONE, count);
            Mul(value, value, value, static_cast<int32_t>(count));
        }
    }

    // 归约完成后统一转换回输出类型，并在写回结束后复用缓冲。
    __aicore__ inline void StoreFloat(const LocalTensor<float>& source, uint64_t offset, uint32_t count)
    {
        LocalTensor<T> out = outBuf_.Get<float>().template ReinterpretCast<T>();
        if constexpr (sizeof(T) == sizeof(float)) {
            Sync<HardEvent::V_MTE3>();
            Store(offset, source.template ReinterpretCast<T>(), count);
        } else {
            Cast(out, source, RoundMode::CAST_RINT, count);
            Sync<HardEvent::V_MTE3>();
            Store(offset, out, count);
        }

        Sync<HardEvent::MTE3_V>();
    }

    __aicore__ inline void FoldStage(const LocalTensor<float>& value, uint64_t rows, uint32_t pitch)
    {
        uint64_t n = rows;
        while (n > 1) {
            const uint64_t half = n / 2;
            const uint64_t rest = n - half;
            PipeBarrier<PIPE_V>();
            Add(value, value, value[rest * pitch], static_cast<int32_t>(half * pitch));
            n = rest;
        }
    }

    __aicore__ inline void FoldRows(const LocalTensor<float>& sums, const LocalTensor<float>& value, uint32_t batch,
                                    uint32_t pitch)
    {
        const int32_t stride = static_cast<int32_t>(pitch * sizeof(float) / 32);

        const uint32_t head = runLength_ < kLanes ? static_cast<uint32_t>(runLength_) : kLanes;
        if (runLength_ <= kLanes) {
            WholeReduceSum(sums, value, static_cast<int32_t>(head), static_cast<int32_t>(batch), 1, 1, stride);
            return;
        }
        for (uint64_t c = kLanes; c < runLength_; c += kLanes) {
            const uint64_t rest = runLength_ - c;
            const int32_t n = static_cast<int32_t>(rest < kLanes ? rest : kLanes);
            PipeBarrier<PIPE_V>();
            Add(value, value, value[c], static_cast<uint64_t>(n), static_cast<uint8_t>(batch),
                {1, 1, 1, static_cast<uint8_t>(stride), static_cast<uint8_t>(stride),
                 static_cast<uint8_t>(stride)});
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sums, value, static_cast<int32_t>(head), static_cast<int32_t>(batch), 1, 1, stride);
    }

    __aicore__ inline void LaneRange128(uint32_t lo, uint32_t hi, uint64_t out[2]) const
    {
        out[0] = LaneRange(lo < 64 ? lo : 64, hi < 64 ? hi : 64);
        out[1] = LaneRange(lo > 64 ? lo - 64 : 0, hi > 64 ? hi - 64 : 0);
    }

    __aicore__ inline uint64_t LaneRange(uint32_t lo, uint32_t hi) const
    {
        const uint64_t high = hi >= 64 ? ~static_cast<uint64_t>(0) : ((static_cast<uint64_t>(1) << hi) - 1);
        return high & ~((static_cast<uint64_t>(1) << lo) - 1);
    }

    __aicore__ inline bool CanFoldPacked() const
    {
        return runLength_ <= kMaxRepStride && runLength_ + kGroupRows <= 128;
    }

    __aicore__ inline void FoldPacked(const LocalTensor<float>& sums, const LocalTensor<float>& carry,
                                      const LocalTensor<float>& flat, uint32_t batch)
    {
        const uint32_t run = static_cast<uint32_t>(runLength_);
        bool carried = false;
        for (uint32_t p = 0; p < kGroupRows && p < batch; ++p) {
            const uint32_t start = p * run;
            const uint32_t base = start / 8 * 8;
            const uint32_t off = start - base;
            const uint32_t span = off + run;
            const int32_t reps = static_cast<int32_t>((batch - p + kGroupRows - 1) / kGroupRows);
            uint64_t head[2] = {LaneRange(off, span < 64 ? span : 64), 0};
            PipeBarrier<PIPE_V>();
            WholeReduceSum(sums[p], flat[base], head, reps, static_cast<int32_t>(kGroupRows), 1,
                           static_cast<int32_t>(run));
            if (span > 64) {
                uint64_t tail[2] = {LaneRange(0, span - 64), 0};
                PipeBarrier<PIPE_V>();
                WholeReduceSum(carry[p], flat[base + 64], tail, reps, static_cast<int32_t>(kGroupRows), 1,
                               static_cast<int32_t>(run));
                carried = true;
            }
        }
        if (carried) {
            PipeBarrier<PIPE_V>();
            Add(sums, sums, carry, static_cast<int32_t>(batch));
        }
    }

    static constexpr bool kNarrowCapable = kFusedAccum && sizeof(T) == 2;

    __aicore__ inline void FoldPackedNarrow(const LocalTensor<T>& sums, const LocalTensor<T>& flat,
                                            uint32_t batch)
    {
        if constexpr (!kNarrowCapable) {
            return;
        } else {
        const uint32_t run = static_cast<uint32_t>(runLength_);
        const uint32_t lanes = 32 / sizeof(T);

        PipeBarrier<PIPE_V>();
        for (uint32_t p = 0; p < kPackRows && p < batch; ++p) {
            const uint32_t start = p * run;
            const uint32_t base = start / lanes * lanes;
            const uint32_t off = start - base;
            const int32_t reps = static_cast<int32_t>((batch - p + kPackRows - 1) / kPackRows);
            uint64_t mask[2];
            LaneRange128(off, off + run, mask);
            WholeReduceSum(sums[p], flat[base], mask, reps, static_cast<int32_t>(kPackRows), 1,
                           static_cast<int32_t>(run));
        }
        }
    }

    __aicore__ inline void FoldBlocks(const LocalTensor<float>& sums, const LocalTensor<float>& flat,
                                      uint32_t batch)
    {
        const uint32_t run = static_cast<uint32_t>(runLength_);
        const uint32_t span = run / kLanes;
        const uint32_t stride = span * kLanes * sizeof(float) / 32;
        LocalTensor<float> chunk = chunkBuf_.Get<float>();

        LocalTensor<float> edge = chunk;
        LocalTensor<float> env = chunk[256];
        PipeBarrier<PIPE_V>();

        LocalTensor<float> mask = foldBuf_.Get<float>();
        LocalTensor<float> strip = mask[(kMaxBlockFoldBatch + 1) * kLanes];
        Mul(strip, flat, mask, static_cast<uint64_t>(kLanes), static_cast<uint8_t>(batch + 1),
            {1, 1, 1, 8, static_cast<uint8_t>(stride), 8});
        PipeBarrier<PIPE_V>();
        WholeReduceSum(edge, strip, static_cast<int32_t>(kLanes), static_cast<int32_t>(batch + 1), 8,
                       1, 8);

        LocalTensor<float> lvl1 = redBuf_.Get<float>();
        LocalTensor<float> blk = lvl1[flatFloats_ / 8];
        const uint32_t envFloats = batch * span * kLanes;
        PipeBarrier<PIPE_V>();
        BlockReduceSum(lvl1, flat, static_cast<int32_t>(envFloats / kLanes),
                       static_cast<int32_t>(kLanes), 1, 1, 8);
        PipeBarrier<PIPE_V>();

        BlockReduceSum(blk, lvl1, static_cast<int32_t>(envFloats / 8 / kLanes),
                       static_cast<int32_t>(kLanes), 1, 1, 8);
        PipeBarrier<PIPE_V>();
        uint64_t rowMask[2] = {LaneRange(0, span), 0};
        WholeReduceSum(env, blk, rowMask, static_cast<int32_t>(batch), 8, 1,
                       static_cast<int32_t>(span * sizeof(float) / 32));
        PipeBarrier<PIPE_V>();
        Sub(env, env, edge, static_cast<int32_t>(batch * 8));
        PipeBarrier<PIPE_V>();
        Add(env, env, edge[8], static_cast<int32_t>(batch * 8));
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sums, env, 1, static_cast<int32_t>(batch), 1, 1, 1);
    }

    __aicore__ inline uint64_t ChunkRows(uint64_t output) const
    {
        uint64_t batch = flatBatch_;
        const uint64_t left = contiguousOutputs_ - (output % contiguousOutputs_);
        if (batch > left) {
            batch = left;
        }
        if (batch > end_ - output) {
            batch = end_ - output;
        }
        return batch;
    }

    __aicore__ inline void BlockFoldMode()
    {
        LocalTensor<float> flat = flatBuf_.Get<float>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<T> raw = rawBuf_.Get<T>();

        Duplicate(flat, 0.0f, static_cast<int32_t>(flatFloats_));
        TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        TEventID spent[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                             GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
        bool live[2] = {false, false};
        uint64_t output = begin_;
        uint64_t batch = output < end_ ? ChunkRows(output) : 0;
        uint32_t cur = 0;
        if (batch > 0) {
            Load(raw[0], OutputBase(output) + ReduceBase(0, true),
                 static_cast<uint32_t>(batch * runLength_));
            SetFlag<HardEvent::MTE2_V>(full[0]);
        }
        while (output < end_) {
            const uint64_t next = output + batch;
            const uint64_t nextBatch = next < end_ ? ChunkRows(next) : 0;
            const uint32_t nxt = 1 - cur;
            if (nextBatch > 0) {
                if (live[nxt]) {
                    WaitFlag<HardEvent::V_MTE2>(spent[nxt]);
                    live[nxt] = false;
                }
                Load(raw[nxt * stageElems_], OutputBase(next) + ReduceBase(0, true),
                     static_cast<uint32_t>(nextBatch * runLength_));
                SetFlag<HardEvent::MTE2_V>(full[nxt]);
            }
            WaitFlag<HardEvent::MTE2_V>(full[cur]);
            SquareInto(flat, raw[cur * stageElems_], static_cast<uint32_t>(batch * runLength_));
            SetFlag<HardEvent::V_MTE2>(spent[cur]);
            live[cur] = true;
            FoldBlocks(acc, flat, static_cast<uint32_t>(batch));
            PipeBarrier<PIPE_V>();
            StoreFloat(acc, output, static_cast<uint32_t>(batch));
            output = next;
            batch = nextBatch;
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

    __aicore__ inline uint64_t ChunkBatch(uint64_t output) const
    {
        uint64_t batch = flatBatch_;
        const uint64_t left = contiguousOutputs_ - (output % contiguousOutputs_);
        if (batch > left) {
            batch = left;
        }
        if (batch > end_ - output) {
            batch = end_ - output;
        }
        return batch;
    }

    __aicore__ inline void FlatRowMode()
    {
        if (doubleStage_) {
            BlockFoldMode();
            return;
        }
        LocalTensor<float> flat = flatBuf_.Get<float>();
        LocalTensor<T> flatNarrow = flatBuf_.Get<float>().template ReinterpretCast<T>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<T> raw = rawBuf_.Get<T>();
        const uint32_t halfElems = runCount_ > 1 ? halfElems_ : stageElems_;
        const uint64_t slot = GetBlockIdx() * slotFloats_;
        const uint32_t floatPitch = (static_cast<uint32_t>(runLength_) + 7) / 8 * 8;
        const bool packedFold = !blockFold_ && CanFoldPacked();
        const bool single = runCount_ == 1 && !packedFold;
        if (blockFold_) {

            Duplicate(flat, 0.0f, static_cast<int32_t>(flatFloats_));
        }

        TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        TEventID spent[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                             GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
        bool live[2] = {false, false};
        uint64_t output = begin_;
        uint32_t head = 0;
        uint64_t batch = ChunkBatch(output);
        uint32_t count = static_cast<uint32_t>(batch * runLength_);
        Load(raw[0], OutputBase(output) + ReduceBase(0, true), count);
        SetFlag<HardEvent::MTE2_V>(full[0]);
        while (output < end_) {
            const uint64_t myBatch = batch;
            const uint32_t myCount = count;
            const uint64_t myBase = OutputBase(output);
            const uint64_t next = output + myBatch;
            uint32_t cur = head;
            uint32_t clear = myCount;
            if (packedFold) {
                clear = myCount + 128 > flatFloats_ ? flatFloats_ : myCount + 128;
                PipeBarrier<PIPE_V>();
                Duplicate(partialBuf_.Get<float>(), 0.0f, static_cast<int32_t>(myBatch));
            }
            if (!single) {
                if constexpr (kNarrowCapable) {
                    if (narrowAccum_) {

                        uint32_t wipe = myCount + 256;
                        if (wipe > 2 * flatFloats_) {
                            wipe = 2 * flatFloats_;
                        }
                        Duplicate(flatNarrow, static_cast<T>(0), static_cast<int32_t>(wipe));
                    } else {
                        Duplicate(flat, 0.0f, static_cast<int32_t>(clear));
                    }
                } else {
                    Duplicate(flat, 0.0f, static_cast<int32_t>(clear));
                }
            }
            for (uint64_t run = 0; run < runCount_; ++run) {
                const uint32_t nxt = 1 - cur;
                uint64_t loadBase = 0;
                uint32_t loadCount = 0;
                if (run + 1 < runCount_) {
                    loadBase = myBase + ReduceBase(run + 1, true);
                    loadCount = myCount;
                } else if (next < end_) {
                    batch = ChunkBatch(next);
                    count = static_cast<uint32_t>(batch * runLength_);
                    loadBase = OutputBase(next) + ReduceBase(0, true);
                    loadCount = count;
                    head = nxt;
                }
                if (loadCount > 0) {
                    if (live[nxt]) {
                        WaitFlag<HardEvent::V_MTE2>(spent[nxt]);
                        live[nxt] = false;
                    }
                    Load(raw[nxt * halfElems], loadBase, loadCount);
                    SetFlag<HardEvent::MTE2_V>(full[nxt]);
                }
                WaitFlag<HardEvent::MTE2_V>(full[cur]);
                if (single) {
                    SquareInto(flat, raw[cur * halfElems], myCount);
                } else {
                    bool done2 = false;
                    if constexpr (kNarrowCapable) {
                        if (narrowAccum_) {
                            PipeBarrier<PIPE_V>();
                            MulAddDst(flatNarrow, raw[cur * halfElems], raw[cur * halfElems],
                                      static_cast<int32_t>(myCount));
                            done2 = true;
                        }
                    }
                    if (!done2) {
                        AccumSquares(flat, raw[cur * halfElems], myCount);
                    }
                }
                SetFlag<HardEvent::V_MTE2>(spent[cur]);
                live[cur] = true;
                cur = nxt;
            }
            if (kNarrowCapable && narrowAccum_) {
                LocalTensor<T> sums = accBuf_.Get<float>().template ReinterpretCast<T>();
                FoldPackedNarrow(sums, flatNarrow, static_cast<uint32_t>(myBatch));
                PipeBarrier<PIPE_V>();
                Sync<HardEvent::V_MTE3>();
                Store(output, sums, static_cast<uint32_t>(myBatch));
                Sync<HardEvent::MTE3_V>();
                output = next;
                continue;
            }
            if (blockFold_) {
                FoldBlocks(acc, flat, static_cast<uint32_t>(myBatch));
            } else if (packedFold) {
                FoldPacked(acc, partialBuf_.Get<float>(), flat, static_cast<uint32_t>(myBatch));
            } else {
                Sync<HardEvent::V_MTE3>();
                DataCopyExtParams store{1, static_cast<uint32_t>(myCount * sizeof(float)), 0, 0, 0};
                DataCopyPad(wsGm_[slot], flat, store);
                Sync<HardEvent::MTE3_MTE2>();
                LocalTensor<float> padded = Value();
                DataCopyExtParams reload{static_cast<uint16_t>(myBatch),
                                         static_cast<uint32_t>(runLength_ * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
                DataCopyPad(padded, wsGm_[slot], reload, pad);
                Sync<HardEvent::MTE2_V>();
                FoldRows(acc, padded, static_cast<uint32_t>(myBatch), floatPitch);
            }
            PipeBarrier<PIPE_V>();
            StoreFloat(acc, output, static_cast<uint32_t>(myBatch));
            output = next;
        }
        for (uint32_t b = 0; b < 2; ++b) {
            if (live[b]) {
                WaitFlag<HardEvent::V_MTE2>(spent[b]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(full[b]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(spent[b]);
        }
    }

    __aicore__ inline uint32_t BatchAt(uint64_t output, uint32_t pitch) const
    {
        uint64_t batch = stageElems_ / pitch;
        if (pitch * sizeof(float) / 32 > kMaxRepStride) {
            batch = 1;
        }
        if (batch > kMaxRepeat) {
            batch = kMaxRepeat;
        }
        const uint64_t left = contiguousOutputs_ - (output % contiguousOutputs_);
        if (batch > left) {
            batch = left;
        }
        if (batch > end_ - output) {
            batch = end_ - output;
        }
        if (batch == 0) {
            batch = 1;
        }
        return static_cast<uint32_t>(batch);
    }

    __aicore__ inline void RowMode()
    {
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> sums = partialBuf_.Get<float>();
        const uint32_t pitch = PaddedPitch(runLength_);
        if (runLength_ > stageElems_) {
            for (uint64_t output = begin_; output < end_; ++output) {
                LongRun(output, acc, Value(), sums);
            }
            return;
        }
        LocalTensor<T> half[2] = {rawBuf_.Get<T>(), flatBuf_.Get<float>().template ReinterpretCast<T>()};
        TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        TEventID spent[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                             GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
        bool live[2] = {false, false};
        uint64_t output = begin_;
        uint64_t run = 0;
        uint32_t batch = BatchAt(output, pitch);
        uint32_t cur = 0;
        LoadRows(half[0], OutputBase(output) + ReduceBase(0, true), batch, static_cast<uint32_t>(runLength_), 0);
        SetFlag<HardEvent::MTE2_V>(full[0]);
        Duplicate(acc, 0.0f, static_cast<int32_t>(batch));
        while (true) {
            uint64_t nextOutput = output;
            uint64_t nextRun = run + 1;
            uint32_t nextBatch = batch;
            const bool lastRun = nextRun == runCount_;
            if (lastRun) {
                nextRun = 0;
                nextOutput = output + batch;
                nextBatch = 0;
            }
            const bool more = nextOutput < end_;
            const uint32_t nxt = 1 - cur;
            if (more) {
                if (nextBatch == 0) {
                    nextBatch = BatchAt(nextOutput, pitch);
                }
                if (live[nxt]) {
                    WaitFlag<HardEvent::V_MTE2>(spent[nxt]);
                    live[nxt] = false;
                }
                LoadRows(half[nxt], OutputBase(nextOutput) + ReduceBase(nextRun, true), nextBatch,
                         static_cast<uint32_t>(runLength_), 0);
                SetFlag<HardEvent::MTE2_V>(full[nxt]);
            }
            WaitFlag<HardEvent::MTE2_V>(full[cur]);
            LocalTensor<float> v = SquareSlab(half[cur], batch * pitch);
            PipeBarrier<PIPE_V>();
            FoldRows(sums, v, batch, pitch);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, sums, static_cast<int32_t>(batch));
            SetFlag<HardEvent::V_MTE2>(spent[cur]);
            live[cur] = true;
            if (lastRun) {
                PipeBarrier<PIPE_V>();
                StoreFloat(acc, output, batch);
            }
            if (!more) {
                break;
            }
            if (lastRun) {
                Duplicate(acc, 0.0f, static_cast<int32_t>(nextBatch));
            }
            output = nextOutput;
            run = nextRun;
            batch = nextBatch;
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

    __aicore__ inline void LongRun(uint64_t output, const LocalTensor<float>& acc, const LocalTensor<float>& value,
                                   const LocalTensor<float>& sums)
    {
        Duplicate(acc, 0.0f, static_cast<int32_t>(kLanes));
        const uint64_t base = OutputBase(output);
        for (uint64_t run = 0; run < runCount_; ++run) {
            const uint64_t runBase = base + ReduceBase(run, true);
            for (uint64_t done = 0; done < runLength_; done += stageElems_) {
                const uint64_t rest = runLength_ - done;
                const uint32_t count = static_cast<uint32_t>(rest < stageElems_ ? rest : stageElems_);
                Sync<HardEvent::V_MTE2>();
                Load(rawBuf_.Get<T>(), runBase + done, count);
                Sync<HardEvent::MTE2_V>();
                Square(count);
                for (uint32_t c = 0; c < count; c += kLanes) {
                    const uint32_t rest2 = count - c;
                    Add(acc, acc, value[c], static_cast<int32_t>(rest2 < kLanes ? rest2 : kLanes));
                }
            }
        }
        WholeReduceSum(sums, acc, static_cast<int32_t>(kLanes), 1, 1, 1,
                       static_cast<int32_t>(kLanes * sizeof(float) / 32));
        PipeBarrier<PIPE_V>();
        StoreFloat(sums, output, 1);
    }

    __aicore__ inline bool FusedSlab(uint32_t w) const
    {
        return kFusedAccum && (kGroupRows * w * sizeof(T)) % 32 == 0;
    }

    __aicore__ inline uint64_t SlabRows(uint32_t w) const
    {
        uint64_t rows = halfElems_ / w;
        if (!FusedSlab(w)) {
            uint64_t p = kGroupRows;
            while (p * 2 <= rows) {
                p *= 2;
            }
            return p;
        }

        if (rows > kMaxSlabRows) {
            rows = kMaxSlabRows;
        }
        return rows;
    }

    __aicore__ inline uint64_t SlabTake(uint64_t step, uint64_t last, uint64_t stageRows, bool fused) const
    {
        uint64_t rows = last - step;
        const uint64_t reach = contiguousRows_ - (step % contiguousRows_);
        if (rows > reach) {
            rows = reach;
        }
        if (rows > stageRows) {
            rows = stageRows;
        }
        if (fused || rows < kGroupRows) {

            return rows;
        }
        uint64_t take = kGroupRows;
        while (take * 2 <= rows) {
            take *= 2;
        }
        return take;
    }

    __aicore__ inline LocalTensor<float> SquareSlab(const LocalTensor<T>& src, uint32_t count)
    {
        if constexpr (sizeof(T) == sizeof(float)) {
            LocalTensor<float> v = src.template ReinterpretCast<float>();
            Mul(v, v, v, static_cast<int32_t>(count));
            return v;
        } else {
            LocalTensor<float> v = valueBuf_.Get<float>();
            Cast(v, src, RoundMode::CAST_NONE, count);
            Mul(v, v, v, static_cast<int32_t>(count));
            return v;
        }
    }

    __aicore__ inline uint64_t AlignTake(uint64_t offset, uint32_t w, uint64_t take) const
    {
        if constexpr (sizeof(T) == 2) {
            constexpr uint64_t lanes = 32 / sizeof(T);
            if (stepStride_ == 0 && offset % lanes != 0) {
                for (uint64_t k = 1; k < lanes && k < take; ++k) {
                    if ((offset + k * w) % lanes == 0) {
                        return k < kGroupRows ? k : kGroupRows;
                    }
                }
            }
        }
        return take;
    }

    __aicore__ inline void AccumulateFlat(const LocalTensor<float>& acc, uint64_t base, uint32_t w, uint64_t first,
                                          uint64_t last)
    {
        LocalTensor<float> group = flatBuf_.Get<float>();
        LocalTensor<T> raw = rawBuf_.Get<T>();
        const uint32_t groupElems = kGroupRows * w;
        Duplicate(group, 0.0f, static_cast<int32_t>(groupElems));

        const uint32_t halfElems = runCount_ > 1 ? halfElems_ : stageElems_;

        const bool fused = FusedSlab(w);
        const uint64_t stageRows = SlabRows(w);

        TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        TEventID spent[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                             GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
        bool live[2] = {false, false};

        uint64_t step = first;
        uint32_t slot = 0;
        uint64_t take = 0;
        if (step < last) {
            take = AlignTake(base + ReduceBase(step, false), w, SlabTake(step, last, stageRows, fused));
            Load(raw[slot * halfElems], base + ReduceBase(step, false), static_cast<uint32_t>(take * w));
            SetFlag<HardEvent::MTE2_V>(full[slot]);
        }
        while (step < last) {
            const uint64_t curTake = take;
            const uint32_t cur = slot;
            const uint32_t nxt = 1 - cur;
            const uint64_t nextStep = stepStride_ != 0 ? step + stepStride_ : step + curTake;

            if (nextStep < last) {
                take = AlignTake(base + ReduceBase(nextStep, false), w, SlabTake(nextStep, last, stageRows, fused));
                if (live[nxt]) {
                    WaitFlag<HardEvent::V_MTE2>(spent[nxt]);
                    live[nxt] = false;
                }
                Load(raw[nxt * halfElems], base + ReduceBase(nextStep, false), static_cast<uint32_t>(take * w));
                SetFlag<HardEvent::MTE2_V>(full[nxt]);
            }
            WaitFlag<HardEvent::MTE2_V>(full[cur]);
            const uint32_t count = static_cast<uint32_t>(curTake * w);
            const uint32_t at = cur * halfElems;
            bool done = false;
            if constexpr (kFusedAccum) {
                if (fused) {

                    for (uint64_t r = 0; r < curTake; r += kGroupRows) {
                        uint64_t slab = curTake - r;
                        if (slab > kGroupRows) {
                            slab = kGroupRows;
                        }
                        PipeBarrier<PIPE_V>();
                        MulAddDst(group, raw[at + r * w], raw[at + r * w], static_cast<int32_t>(slab * w));
                    }
                    done = true;
                }
            }
            if (!done) {
                LocalTensor<float> v = SquareSlab(raw[at], count);
                PipeBarrier<PIPE_V>();
                uint64_t n = curTake;
                while (n > kGroupRows) {
                    const uint64_t half = n / 2;
                    const uint64_t rest = n - half;
                    Add(v, v, v[rest * w], static_cast<int32_t>(half * w));
                    PipeBarrier<PIPE_V>();
                    n = rest;
                }
                Add(group, group, v, static_cast<int32_t>(n * w));
            }
            SetFlag<HardEvent::V_MTE2>(spent[cur]);
            live[cur] = true;
            step = nextStep;
            slot = nxt;
        }
        for (uint32_t b = 0; b < 2; ++b) {
            if (live[b]) {
                WaitFlag<HardEvent::V_MTE2>(spent[b]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(full[b]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(spent[b]);
        }

        const uint32_t pitch = (w + kFloatLanes - 1) / kFloatLanes * kFloatLanes;
        if (pitch == w) {

            FoldStage(group, kGroupRows, pitch);
            PipeBarrier<PIPE_V>();
            Add(acc, acc, group, static_cast<int32_t>(w));
            return;
        }
        LocalTensor<float> value = Value();
        if (flatGather_) {

            LocalTensor<uint32_t> ramp = chunkBuf_.Get<uint32_t>();
            for (uint32_t r = 0; r < kGroupRows; ++r) {
                PipeBarrier<PIPE_V>();
                Gather(value[r * pitch], group, ramp, r * w * static_cast<uint32_t>(sizeof(float)), w);
            }
            PipeBarrier<PIPE_V>();
        } else {
            const uint64_t slotBase = GetBlockIdx() * slotFloats_;
            Sync<HardEvent::V_MTE3>();
            DataCopyExtParams store{1, static_cast<uint32_t>(groupElems * sizeof(float)), 0, 0, 0};
            DataCopyPad(wsGm_[slotBase], group, store);
            Sync<HardEvent::MTE3_MTE2>();
            DataCopyExtParams reload{static_cast<uint16_t>(kGroupRows),
                                     static_cast<uint32_t>(w * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            DataCopyPad(value, wsGm_[slotBase], reload, pad);
            Sync<HardEvent::MTE2_V>();
        }
        FoldStage(value, kGroupRows, pitch);
        PipeBarrier<PIPE_V>();
        Add(acc, acc, value, static_cast<int32_t>(w));
    }

    __aicore__ inline void SplitColumnMode()
    {
        LocalTensor<float> value = Value();
        LocalTensor<float> acc = accBuf_.Get<float>();
        const uint32_t w = static_cast<uint32_t>(innerWidth_);
        Duplicate(acc, 0.0f, static_cast<int32_t>(w));
        if constexpr (sizeof(T) == sizeof(float)) {

            if (GetBlockIdx() == 0) {
                StoreFloat(acc, 0, w);
            }
        }
        if (rowBegin_ < rowEnd_) {
            AccumulateFlat(acc, OutputBase(0), w, rowBegin_, rowEnd_);
        }

        if constexpr (sizeof(T) == sizeof(float)) {
            SyncAll<true>();
            SetAtomicAdd<float>();
            StoreFloat(acc, 0, w);
            SetAtomicNone();
            return;
        }
        const uint64_t slot = partialBase_ + GetBlockIdx() * partialPitch_;
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams out{1, static_cast<uint32_t>(w * sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm_[slot], acc, out);
        Sync<HardEvent::MTE3_S>();
        SyncAll<true>();
        if (GetBlockIdx() != 0) {
            return;
        }

        LocalTensor<float> rows = rawBuf_.Get<float>();
        const uint32_t pitch = (w + 7) / 8 * 8;
        uint32_t group = stageElems_ * sizeof(T) / sizeof(float) / pitch;
        if (group > reduceCores_) {
            group = static_cast<uint32_t>(reduceCores_);
        }
        if (group == 0) {
            group = 1;
        }
        Duplicate(acc, 0.0f, static_cast<int32_t>(w));
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        for (uint64_t first = 0; first < reduceCores_; first += group) {
            uint32_t take = static_cast<uint32_t>(reduceCores_ - first);
            if (take > group) {
                take = group;
            }

            PipeBarrier<PIPE_V>();
            Duplicate(rows, 0.0f, static_cast<int32_t>(take * pitch));
            Sync<HardEvent::V_MTE2>();
            DataCopyExtParams in{static_cast<uint16_t>(take), static_cast<uint32_t>(w * sizeof(float)),
                                 static_cast<uint32_t>((partialPitch_ - w) * sizeof(float)), 0, 0};
            DataCopyPad(rows, wsGm_[partialBase_ + first * partialPitch_], in, pad);
            Sync<HardEvent::MTE2_V>();
            uint32_t n = take;
            while (n > 1) {
                const uint32_t half = n / 2;
                const uint32_t rest = n - half;
                Add(rows, rows, rows[rest * pitch], static_cast<int32_t>(half * pitch));
                PipeBarrier<PIPE_V>();
                n = rest;
            }
            Add(acc, acc, rows, static_cast<int32_t>(w));
        }
        StoreFloat(acc, 0, w);
    }

    __aicore__ inline uint64_t SpanTake(uint64_t g, uint64_t stageRows, bool fused) const
    {
        uint64_t rows = gEnd_ - g;
        const uint64_t reach = reduceCount_ - (g % reduceCount_);
        if (rows > reach) {
            rows = reach;
        }
        if (rows > stageRows) {
            rows = stageRows;
        }
        if (fused || rows < kGroupRows) {
            return rows;
        }
        uint64_t take = kGroupRows;
        while (take * 2 <= rows) {
            take *= 2;
        }
        return take;
    }

    __aicore__ inline uint64_t SpanAddr(uint64_t g) const
    {
        return OutputBase(g / reduceCount_ * innerWidth_) + ReduceBase(g % reduceCount_, false);
    }

    __aicore__ inline void FoldSlab(const LocalTensor<float>& group, const LocalTensor<T>& src, uint64_t rows,
                                    uint32_t w, bool fused)
    {
        if constexpr (kFusedAccum) {
            if (fused) {
                for (uint64_t r = 0; r < rows; r += kGroupRows) {
                    uint64_t slab = rows - r;
                    if (slab > kGroupRows) {
                        slab = kGroupRows;
                    }
                    PipeBarrier<PIPE_V>();
                    MulAddDst(group, src[r * w], src[r * w], static_cast<int32_t>(slab * w));
                }
                return;
            }
        }
        LocalTensor<float> v = SquareSlab(src, static_cast<uint32_t>(rows * w));
        PipeBarrier<PIPE_V>();
        uint64_t n = rows;
        while (n > kGroupRows) {
            const uint64_t half = n / 2;
            const uint64_t rest = n - half;
            Add(v, v, v[rest * w], static_cast<int32_t>(half * w));
            PipeBarrier<PIPE_V>();
            n = rest;
        }
        Add(group, group, v, static_cast<int32_t>(n * w));
    }

    __aicore__ inline void CloseRow(const LocalTensor<float>& group, uint64_t row, uint32_t w)
    {
        const uint64_t rowStart = row * reduceCount_;
        const uint64_t lo = gBegin_ > rowStart ? gBegin_ - rowStart : 0;
        const uint64_t hi = gEnd_ - rowStart < reduceCount_ ? gEnd_ - rowStart : reduceCount_;
        LocalTensor<float> value = Value();
        const uint32_t pitch = (w + 7) / 8 * 8;
        LocalTensor<uint32_t> ramp = chunkBuf_.Get<uint32_t>();

        for (uint32_t r = 0; r < kGroupRows; ++r) {
            PipeBarrier<PIPE_V>();
            Gather(value[r * pitch], group, ramp, r * w * static_cast<uint32_t>(sizeof(float)), w);
        }
        PipeBarrier<PIPE_V>();
        FoldStage(value, kGroupRows, pitch);
        PipeBarrier<PIPE_V>();
        if (lo == 0 && hi == reduceCount_) {
            StoreFloat(value, row * innerWidth_, w);
            return;
        }
        const uint64_t firstRow = gBegin_ / reduceCount_;
        const uint64_t slot =
            partialBase_ + (2 * GetBlockIdx() + (row == firstRow ? 0 : 1)) * partialPitch_;
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams out{1, static_cast<uint32_t>(w * sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm_[slot], value, out);
        Sync<HardEvent::MTE3_V>();
    }

    __aicore__ inline void SplitRowMode()
    {
        const uint32_t w = static_cast<uint32_t>(innerWidth_);
        const uint64_t rc = reduceCount_;
        const uint64_t me = GetBlockIdx();
        LocalTensor<float> group = flatBuf_.Get<float>();
        LocalTensor<T> raw = rawBuf_.Get<T>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        const bool fused = FusedSlab(w);
        const uint64_t stageRows = SlabRows(w);
        if (gBegin_ < gEnd_) {
            TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                                GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
            TEventID spent[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                                 GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
            bool live[2] = {false, false};
            uint64_t g = gBegin_;
            uint32_t cur = 0;
            uint64_t take = SpanTake(g, stageRows, fused);
            Load(raw[0], SpanAddr(g), static_cast<uint32_t>(take * w));
            SetFlag<HardEvent::MTE2_V>(full[0]);
            uint64_t row = gBegin_ / rc;
            bool opened = false;
            while (g < gEnd_) {
                const uint64_t curTake = take;
                const uint32_t nxt = 1 - cur;
                const uint64_t nextG = g + curTake;
                if (nextG < gEnd_) {
                    take = SpanTake(nextG, stageRows, fused);
                    if (live[nxt]) {
                        WaitFlag<HardEvent::V_MTE2>(spent[nxt]);
                        live[nxt] = false;
                    }
                    Load(raw[nxt * halfElems_], SpanAddr(nextG), static_cast<uint32_t>(take * w));
                    SetFlag<HardEvent::MTE2_V>(full[nxt]);
                }
                if (!opened) {
                    Duplicate(group, 0.0f, static_cast<int32_t>(kGroupRows * w));
                    opened = true;
                }
                WaitFlag<HardEvent::MTE2_V>(full[cur]);
                FoldSlab(group, raw[cur * halfElems_], curTake, w, fused);
                SetFlag<HardEvent::V_MTE2>(spent[cur]);
                live[cur] = true;
                g = nextG;
                cur = nxt;
                if (g % rc == 0 || g >= gEnd_) {
                    CloseRow(group, row, w);
                    opened = false;
                    row = g / rc;
                }
            }
            for (uint32_t b = 0; b < 2; ++b) {
                if (live[b]) {
                    WaitFlag<HardEvent::V_MTE2>(spent[b]);
                }
                GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(full[b]);
                GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(spent[b]);
            }
        }
        SyncAll<true>();

        if (gEnd_ <= gBegin_ || gEnd_ % rc == 0) {
            return;
        }
        const uint64_t row = gEnd_ / rc;
        if (row * rc < gBegin_) {
            return;
        }
        const uint64_t rowEnd = (row + 1) * rc;
        const uint64_t cores = GetBlockNum();
        LocalTensor<float> part = partialBuf_.Get<float>();
        Duplicate(acc, 0.0f, static_cast<int32_t>(w));
        DataCopyExtParams in{1, static_cast<uint32_t>(w * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        for (uint64_t core = me; core < cores; ++core) {
            const uint64_t spanBegin = SpanBegin(core);
            if (spanBegin >= rowEnd) {
                break;
            }
            const uint64_t slot =
                partialBase_ + (2 * core + (spanBegin / rc == row ? 0 : 1)) * partialPitch_;
            Sync<HardEvent::V_MTE2>();
            DataCopyPad(part, wsGm_[slot], in, pad);
            Sync<HardEvent::MTE2_V>();
            Add(acc, acc, part, static_cast<int32_t>(w));
        }
        StoreFloat(acc, row * innerWidth_, w);
    }

    __aicore__ inline void ColumnMode()
    {
        LocalTensor<float> value = Value();
        LocalTensor<float> acc = accBuf_.Get<float>();
        uint64_t output = begin_;
        while (output < end_) {
            const uint64_t row = output / innerWidth_;
            const uint64_t column = output % innerWidth_;
            uint64_t width = innerWidth_ - column;
            if (width > end_ - output) {
                width = end_ - output;
            }
            if (width > scratchFloats_) {
                width = scratchFloats_;
            }
            const uint32_t w = static_cast<uint32_t>(width);
            Duplicate(acc, 0.0f, static_cast<int32_t>(w));
            const uint64_t base = OutputBase(row * innerWidth_) + column;
            if (width == innerWidth_ && contiguousRows_ > 1 && kGroupRows * w <= kStageElems &&
                kGroupRows * w <= kFlatElems) {
                AccumulateFlat(acc, base, w, 0, reduceCount_);
                StoreFloat(acc, output, w);
                output += width;
                continue;
            }

            const uint32_t pitch = PaddedPitch(width);
            const uint64_t rowsPerStage = contiguousRows_ > 1 ? (halfElems_ / pitch) : 1;
            uint64_t step = 0;
            while (step < reduceCount_) {
                uint64_t rows = 1;
                if (rowsPerStage > 1) {
                    rows = contiguousRows_ - (step % contiguousRows_);
                    if (rows > rowsPerStage) {
                        rows = rowsPerStage;
                    }
                    if (rows > reduceCount_ - step) {
                        rows = reduceCount_ - step;
                    }
                }
                const uint32_t count = static_cast<uint32_t>(rows * pitch);
                Sync<HardEvent::V_MTE2>();
                LoadRows(rawBuf_.Get<T>(), base + ReduceBase(step, false), static_cast<uint32_t>(rows), w,
                         innerWidth_ - width);
                Sync<HardEvent::MTE2_V>();
                Square(count);

                PipeBarrier<PIPE_V>();
                FoldStage(value, rows, pitch);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, value, static_cast<int32_t>(w));
                step += rows;
            }
            StoreFloat(acc, output, w);
            output += width;
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> valueBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> partialBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    TBuf<TPosition::VECCALC> chunkBuf_;
    TBuf<TPosition::VECCALC> foldBuf_;
    TBuf<TPosition::VECCALC> redBuf_;
    bool doubleStage_ = false;
    TBuf<TPosition::VECCALC> flatBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint64_t size_[kMaxGroups] = {0};
    uint64_t stride_[kMaxGroups] = {0};
    uint32_t reduced_[kMaxGroups] = {0};
    uint32_t groupCount_ = 0;
    uint32_t blockElems_ = 0;
    bool tailReduced_ = false;
    uint64_t outputCount_ = 0;
    uint64_t reduceCount_ = 0;
    uint64_t runLength_ = 0;
    uint64_t runCount_ = 0;
    uint64_t innerWidth_ = 0;
    uint64_t contiguousOutputs_ = 1;
    uint64_t contiguousRows_ = 1;
    uint64_t begin_ = 0;
    uint64_t end_ = 0;
    uint32_t scratchFloats_ = kAccElems;
    uint32_t halfElems_ = 0;
    uint64_t stepStride_ = 0;
    bool flatPacked_ = false;
    bool narrowAccum_ = false;
    bool rowSplit_ = false;
    uint64_t gBegin_ = 0;
    uint64_t gEnd_ = 0;
    uint32_t reduceCores_ = 1;
    uint32_t partialPitch_ = 0;
    uint32_t flatBatch_ = 0;
    bool blockFold_ = false;
    uint32_t stageElems_ = kStageElems;
    uint32_t flatFloats_ = kFlatElems;
    uint32_t slotFloats_ = 0;
    uint32_t partialBase_ = 0;
    bool flatGather_ = false;
    uint64_t rowBegin_ = 0;
    uint64_t rowEnd_ = 0;
    GlobalTensor<float> wsGm_;
};

#define TILING_FIELD(tiling, field) \
    (reinterpret_cast<const __gm__ uint32_t*>(tiling)[offsetof(SquareSumV1TilingData, field) / sizeof(uint32_t)])
#define TILING_SIZE(tiling, group) \
    (reinterpret_cast<const __gm__ uint32_t*>(tiling)[offsetof(SquareSumV1TilingData, groupSize) / sizeof(uint32_t) + (group)])

template <typename T>
class WaveSliceKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
    {
        slices_ = TILING_SIZE(tiling, 0);
        rows_ = TILING_SIZE(tiling, 1);
        width_ = TILING_SIZE(tiling, 2);
        band_ = TILING_FIELD(tiling, pairBand);
        slotElems_ = BlockUp(band_ * width_) + kGuard;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(inBuf_, 2 * slotElems_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        cores_ = GetBlockNum();
        core_ = GetBlockIdx();
        const uint32_t wave = cores_ * band_;
        waves_ = rows_ / wave;
        const uint32_t rest = rows_ - waves_ * wave;
        const uint32_t base = rest / cores_;
        const uint32_t extra = rest % cores_;
        tailBegin_ = waves_ * wave + core_ * base + (core_ < extra ? core_ : extra);
        tailRows_ = base + (core_ < extra ? 1 : 0);
        const uint32_t count = waves_ + (tailRows_ != 0 ? 1 : 0);
        if (count == 0) {
            return;
        }
        LocalTensor<T> in = inBuf_.Get<T>();
        TEventID full[2] = {GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>(),
                            GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>()};
        TEventID spent[2] = {GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>(),
                             GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>()};
        bool live[2] = {false, false};
        const uint32_t total = count * slices_;
        Load(in, 0, full[0], spent[0], live[0]);
        if (total > 1) {
            Load(in, 1, full[1], spent[1], live[1]);
        }
        const uint32_t outElems = BlockUp(band_);
        const uint32_t held = count < kMaxHeld ? count : kMaxHeld;
        pipe_.InitBuffer(accBuf_, (BlockUp(band_ * width_) + kGuard) * sizeof(T));
        pipe_.InitBuffer(outBuf_, held * outElems * sizeof(T));
        for (uint32_t r = 0; r < 16; ++r) {
            const uint32_t shift = r * width_ & (kLanes - 1);
            const uint32_t low = 64 - shift < width_ ? 64 - shift : width_;
            const uint32_t high = width_ - low;
            mask_[r][0] = ((1ULL << (low - 1)) * 2 - 1) << shift;
            mask_[r][1] = high == 0 ? 0ULL : (1ULL << (high - 1)) * 2 - 1;
            start_[r] = r * width_ - shift;
        }
        LocalTensor<T> acc = accBuf_.Get<T>();
        LocalTensor<T> out = outBuf_.Get<T>();
        uint32_t kept = 0;
        uint32_t first = 0;
        for (uint32_t g = 0; g < total; ++g) {
            const uint32_t s = g & 1;
            const uint32_t b = g / slices_;
            const uint32_t rows = b < waves_ ? band_ : tailRows_;
            const int32_t active = static_cast<int32_t>(BlockUp(rows * width_));
            WaitFlag<HardEvent::MTE2_V>(full[s]);
            if (g % slices_ == 0) {
                Mul(acc, in[s * slotElems_], in[s * slotElems_], active);
            } else {
                MulAddDst(acc, in[s * slotElems_], in[s * slotElems_], active);
            }
            SetFlag<HardEvent::V_MTE2>(spent[s]);
            live[s] = true;
            if (g + 2 < total) {
                Load(in, g + 2, full[s], spent[s], live[s]);
            }
            if (g % slices_ != slices_ - 1) {
                continue;
            }
            PipeBarrier<PIPE_V>();
            const uint32_t at = kept * outElems;
            const uint32_t reach = rows < 16 ? rows : 16;
            for (uint32_t r = 0; r < reach; ++r) {
                WholeReduceSum(out[at + r], acc[start_[r]], mask_[r], static_cast<int32_t>((rows - r + 15) >> 4), 16,
                               1, static_cast<int32_t>(width_));
            }
            ++kept;
            if (kept == held || b + 1 == count) {
                Flush(out, first, kept, outElems);
                first = b + 1;
                kept = 0;
            }
        }
        for (uint32_t k = 0; k < 2; ++k) {
            if (live[k]) {
                WaitFlag<HardEvent::V_MTE2>(spent[k]);
            }
            GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(full[k]);
            GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(spent[k]);
        }
    }

private:
    static constexpr uint32_t kLanes = 32 / sizeof(T);
    static constexpr uint32_t kGuard = 128;
    static constexpr uint32_t kMaxHeld = 8;

    __aicore__ inline uint32_t BlockUp(uint32_t elems) const
    {
        return (elems + kLanes - 1) & ~(kLanes - 1);
    }

    __aicore__ inline void Load(const LocalTensor<T>& in, uint32_t g, TEventID full, TEventID spent, bool& live)
    {
        const uint32_t b = g / slices_;
        const uint32_t j = g % slices_;
        const uint32_t others = slices_ - 1;
        const uint32_t slice = j == 0 ? 0 : 1 + (j - 1 + core_ + b) % others;
        const uint32_t begin = b < waves_ ? (b * cores_ + core_) * band_ : tailBegin_;
        const uint32_t rows = b < waves_ ? band_ : tailRows_;
        const uint32_t s = g & 1;
        if (live) {
            WaitFlag<HardEvent::V_MTE2>(spent);
            live = false;
        }
        const uint64_t offset = (static_cast<uint64_t>(slice) * rows_ + begin) * width_;
        DataCopyExtParams run{1, rows * width_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(in[s * slotElems_], xGm_[offset], run, pad);
        SetFlag<HardEvent::MTE2_V>(full);
    }

    __aicore__ inline void Flush(const LocalTensor<T>& out, uint32_t first, uint32_t kept, uint32_t outElems)
    {
        const TEventID id = GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(id);
        WaitFlag<HardEvent::V_MTE3>(id);
        const uint32_t whole = first + kept <= waves_ ? kept : kept - 1;
        const uint32_t bytes = band_ * static_cast<uint32_t>(sizeof(T));
        if (whole != 0) {
            DataCopyExtParams store{static_cast<uint16_t>(whole), bytes, 0, (cores_ - 1) * bytes, 0};
            DataCopyPad(yGm_[(first * cores_ + core_) * band_], out, store);
        }
        if (whole != kept) {
            DataCopyExtParams store{1, tailRows_ * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[tailBegin_], out[whole * outElems], store);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECIN> inBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint32_t slices_ = 0;
    uint32_t rows_ = 0;
    uint32_t width_ = 0;
    uint32_t band_ = 0;
    uint32_t slotElems_ = 0;
    uint32_t cores_ = 0;
    uint32_t core_ = 0;
    uint32_t waves_ = 0;
    uint32_t tailBegin_ = 0;
    uint32_t tailRows_ = 0;
    uint64_t mask_[16][2] = {{0}};
    uint32_t start_[16] = {0};
};

template <typename T>
class SpanEdgeKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
    {
        outputs_ = TILING_FIELD(tiling, outputCount);
        run_ = TILING_FIELD(tiling, runLength);
        blocks_ = (run_ - 1) / 64;
        batch_ = TILING_FIELD(tiling, spanBatch);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        const uint32_t elems = batch_ * run_;
        pipe_.InitBuffer(inQue_, 2, Up(elems, 16) * sizeof(T));
        pipe_.InitBuffer(valueBuf_, Up(elems, 8) * sizeof(float));
        pipe_.InitBuffer(levelBuf_, batch_ * blocks_ * 8 * sizeof(float));
        pipe_.InitBuffer(blockBuf_, (batch_ * blocks_ + 64) * sizeof(float));
        pipe_.InitBuffer(edgeIdxBuf_, (batch_ * 64 + 8) * sizeof(int32_t));
        pipe_.InitBuffer(edgeBuf_, batch_ * 64 * sizeof(float));
        pipe_.InitBuffer(sideBuf_, Up(6 * batch_, 8) * sizeof(float));
        pipe_.InitBuffer(outQue_, 2, Up(batch_, 16) * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        const uint32_t cores = GetBlockNum();
        const uint32_t idx = GetBlockIdx();
        const uint32_t units = (outputs_ + 15) / 16;
        const uint32_t base = units / cores;
        const uint32_t extra = units % cores;
        uint32_t begin = (idx * base + (idx < extra ? idx : extra)) * 16;
        uint32_t end = begin + (base + (idx < extra ? 1 : 0)) * 16;
        if (end > outputs_) {
            end = outputs_;
        }
        if (begin >= end) {
            return;
        }
        uint32_t rows = end - begin < batch_ ? end - begin : batch_;
        Fetch(begin, rows);
        BuildIndices();
        LocalTensor<float> value = valueBuf_.Get<float>();
        LocalTensor<float> level = levelBuf_.Get<float>();
        LocalTensor<float> block = blockBuf_.Get<float>();
        LocalTensor<uint32_t> edgeIdx = edgeIdxBuf_.Get<uint32_t>();
        LocalTensor<float> edge = edgeBuf_.Get<float>();
        LocalTensor<float> side = sideBuf_.Get<float>();
        LocalTensor<float> sums = side;
        LocalTensor<float> edgeSums = side[Up(batch_, 8)];
        LocalTensor<uint32_t> lastIdx = side[2 * Up(batch_, 8)].template ReinterpretCast<uint32_t>();
        LocalTensor<float> last = side[3 * Up(batch_, 8)];
        uint64_t inner[2] = {((1ULL << (blocks_ - 1)) * 2 - 1) & ~1ULL, 0};
        if (blocks_ == 64) {
            inner[0] = ~1ULL;
        }
        const int32_t innerStride = static_cast<int32_t>(blocks_ / 8);
        bool pending = false;
        uint32_t doneBegin = 0;
        uint32_t doneRows = 0;
        while (begin < end) {
            LocalTensor<T> in = inQue_.template DeQue<T>();
            const uint32_t next = begin + rows;
            const uint32_t nextRows = end - next < batch_ ? end - next : batch_;
            if (next < end) {
                Fetch(next, nextRows);
            }
            if (pending) {
                Flush(doneBegin, doneRows);
            }
            const int32_t count = static_cast<int32_t>(rows * run_);
            Cast(value, in, RoundMode::CAST_NONE, count);
            Mul(value, value, value, count);
            const uint32_t windows = rows * blocks_;
            for (uint32_t w = 0; w < windows; w += 248) {
                const uint32_t reps = windows - w < 248 ? windows - w : 248;
                BlockReduceSum(level[w * 8], value[w * 64], static_cast<int32_t>(reps), 64, 1, 1, 8);
            }
            BlockReduceSum(block, level, static_cast<int32_t>(windows / 8), 64, 1, 1, 8);
            WholeReduceSum(sums, block, inner, static_cast<int32_t>(rows), 1, 1, innerStride);
            Gather(edge, value, edgeIdx, 0, rows * 64);
            WholeReduceSum(edgeSums, edge, 64, static_cast<int32_t>(rows), 1, 1, 8);
            Gather(last, value, lastIdx, 0, rows);
            Add(sums, sums, edgeSums, static_cast<int32_t>(rows));
            Add(sums, sums, last, static_cast<int32_t>(rows));
            inQue_.FreeTensor(in);
            LocalTensor<T> out = outQue_.template AllocTensor<T>();
            Cast(out, sums, RoundMode::CAST_RINT, static_cast<int32_t>(Up(rows, 16)));
            outQue_.EnQue(out);
            pending = true;
            doneBegin = begin;
            doneRows = rows;
            begin = next;
            rows = nextRows;
        }
        Flush(doneBegin, doneRows);
    }

private:
    __aicore__ inline uint32_t Up(uint32_t n, uint32_t a) const
    {
        return (n + a - 1) / a * a;
    }

    __aicore__ inline void Fetch(uint32_t begin, uint32_t rows)
    {
        LocalTensor<T> in = inQue_.template AllocTensor<T>();
        const uint32_t count = rows * run_;
        const uint64_t offset = static_cast<uint64_t>(begin) * run_;
        if ((count & 15) == 0 && (offset & 15) == 0) {
            DataCopy(in, xGm_[offset], count);
        } else {
            DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(in, xGm_[offset], params, pad);
        }
        inQue_.EnQue(in);
    }

    __aicore__ inline void Flush(uint32_t begin, uint32_t rows)
    {
        LocalTensor<T> out = outQue_.template DeQue<T>();
        if (((rows | begin) & 15) == 0) {
            DataCopy(yGm_[begin], out, rows);
        } else {
            DataCopyExtParams params{1, rows * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(yGm_[begin], out, params);
        }
        outQue_.FreeTensor(out);
    }

    __aicore__ inline void BuildIndices()
    {
        LocalTensor<int32_t> edgeIdx = edgeIdxBuf_.Get<int32_t>();
        LocalTensor<int32_t> side = sideBuf_.Get<int32_t>();
        LocalTensor<int32_t> lastIdx = side[2 * Up(batch_, 8)];
        const int32_t lane = static_cast<int32_t>(sizeof(float));
        const int32_t jump = static_cast<int32_t>(64 * (blocks_ - 1)) * lane;
        ArithProgression<int32_t>(edgeIdx, 0, lane, 64);
        ArithProgression<int32_t>(lastIdx, static_cast<int32_t>(64 * blocks_) * lane,
                                  static_cast<int32_t>(run_) * lane, static_cast<int32_t>(batch_));
        PipeBarrier<PIPE_V>();
        for (uint32_t r = batch_ - 1; r > 0; --r) {
            Adds(edgeIdx[r * 64], edgeIdx, static_cast<int32_t>(64 * blocks_ * r + r) * lane, 64);
            uint64_t tail[2] = {~0ULL << (64 - r), 0};
            Adds<int32_t>(edgeIdx[r * 64], edgeIdx[r * 64], jump, tail, 1, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 2> outQue_;
    TBuf<TPosition::VECCALC> valueBuf_;
    TBuf<TPosition::VECCALC> levelBuf_;
    TBuf<TPosition::VECCALC> blockBuf_;
    TBuf<TPosition::VECCALC> edgeIdxBuf_;
    TBuf<TPosition::VECCALC> edgeBuf_;
    TBuf<TPosition::VECCALC> sideBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint32_t outputs_ = 0;
    uint32_t run_ = 0;
    uint32_t blocks_ = 0;
    uint32_t batch_ = 0;
};

template <typename T>
class ShortRunKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
    {
        outputs_ = TILING_FIELD(tiling, outputCount);
        run_ = TILING_FIELD(tiling, runLength);
        const uint32_t rows = TILING_FIELD(tiling, shortRows);
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        pipe_.InitBuffer(rawBuf_, ((rows * run_ + 15) & ~15U) * sizeof(T));
        pipe_.InitBuffer(valueBuf_, ((rows * run_ + 64 + 7) & ~7U) * sizeof(float));
        pipe_.InitBuffer(sumBuf_, ((rows + 7) & ~7U) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t units = (outputs_ + 15) / 16;
        const uint32_t cores = GetBlockNum();
        const uint32_t idx = GetBlockIdx();
        const uint32_t first = units * idx / cores * 16;
        uint32_t end = units * (idx + 1) / cores * 16;
        if (end > outputs_) {
            end = outputs_;
        }
        if (first >= end) {
            return;
        }
        const uint32_t rows = end - first;
        const uint32_t count = rows * run_;
        const uint32_t offset = first * run_;
        LocalTensor<T> raw = rawBuf_.Get<T>();
        if (((offset | count) & 15) == 0) {
            DataCopy(raw, xGm_[offset], count);
        } else {
            DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(raw, xGm_[offset], params, pad);
        }
        Sync<HardEvent::MTE2_V>();
        Mul(raw, raw, raw, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        LocalTensor<float> value = valueBuf_.Get<float>();
        LocalTensor<float> sums = sumBuf_.Get<float>();
        Cast(value, raw, RoundMode::CAST_NONE, static_cast<int32_t>(count));
        PipeBarrier<PIPE_V>();
        uint32_t period = 8;
        while (period > 1 && (period / 2 * run_) % 8 == 0) {
            period /= 2;
        }
        const int32_t stride = static_cast<int32_t>(period * run_ / 8);
        const uint64_t rowBits = (1ULL << run_) - 1;
        for (uint32_t j = 0; j < period && j < rows; ++j) {
            const uint32_t start = j * run_;
            const uint32_t shift = start & 7;
            uint64_t mask[2] = {rowBits << shift, 0};
            WholeReduceSum(sums[j], value[start - shift], mask, static_cast<int32_t>((rows - j + period - 1) / period),
                           static_cast<int32_t>(period), 1, stride);
        }
        PipeBarrier<PIPE_V>();
        Cast(raw, sums, RoundMode::CAST_RINT, static_cast<int32_t>(rows));
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams store{1, rows * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[first], raw, store);
        Sync<HardEvent::MTE3_V>();
    }

private:
    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const TEventID id = GetTPipePtr()->FetchEventID<event>();
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> rawBuf_;
    TBuf<TPosition::VECCALC> valueBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    uint32_t outputs_ = 0;
    uint32_t run_ = 0;
};

template <typename T>
class LaneCycleKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling)
    {
        rows_ = TILING_FIELD(tiling, reduceCount);
        width_ = TILING_FIELD(tiling, innerWidth);
        pitch_ = (width_ + 7) & ~7U;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        pipe_.InitBuffer(inQue_, 2, 2 * kGroup * width_ * sizeof(float));

        pipe_.InitBuffer(groupBuf_, kGroup * width_ * sizeof(float) + kBankOffset);
        pipe_.InitBuffer(foldBuf_, (kGroup / 2) * pitch_ * sizeof(float));
        pipe_.InitBuffer(rampBuf_, width_ * sizeof(uint32_t));
        pipe_.InitBuffer(outQue_, 1, pitch_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const uint32_t cores = GetBlockNum();
        const uint32_t idx = GetBlockIdx();
        const uint32_t pairRows = 2 * kGroup;
        const uint32_t pairs = (rows_ + pairRows - 1) / pairRows;
        const uint32_t rounds = pairs / cores;
        const uint32_t spill = pairs % cores;
        const uint32_t rest = rows_ - rounds * cores * pairRows;
        const uint32_t restGroups = (rest + kGroup - 1) / kGroup;

        const bool spread = spill != 0 && restGroups <= cores;
        const uint32_t mine = spread ? rounds : rounds + (idx < spill ? 1 : 0);
        const uint32_t extraBase = rounds * cores * pairRows + idx * kGroup;
        uint32_t extraRows = 0;
        if (spread && idx < restGroups) {
            extraRows = rows_ - extraBase < kGroup ? rows_ - extraBase : kGroup;
        }
        const uint32_t shift = kLaneShift % cores;
        uint32_t lane = idx;
        uint32_t pair = lane;
        if (mine > 0) {
            FetchPair(pair);
        } else if (extraRows > 0) {
            FetchRows(extraBase, extraRows);
        }
        if (idx == 0) {

            LocalTensor<float> zero = outQue_.template AllocTensor<float>();
            Duplicate(zero, 0.0f, static_cast<int32_t>(pitch_));
            outQue_.EnQue(zero);
            zero = outQue_.template DeQue<float>();
            Store(zero);
            Sync<HardEvent::MTE3_S>();
            outQue_.FreeTensor(zero);
        }
        SyncAll<true>();
        if (mine == 0 && extraRows == 0) {
            return;
        }
        LocalTensor<float> group = groupBuf_.GetWithOffset<float>(kGroup * width_, kBankOffset);
        LocalTensor<float> fold = foldBuf_.Get<float>();
        LocalTensor<int32_t> ramp = rampBuf_.Get<int32_t>();
        Duplicate(group, 0.0f, static_cast<int32_t>(kGroup * width_));
        Duplicate(fold, 0.0f, static_cast<int32_t>((kGroup / 2) * pitch_));
        ArithProgression<int32_t>(ramp, 0, static_cast<int32_t>(sizeof(float)), static_cast<int32_t>(width_));
        PipeBarrier<PIPE_V>();
        const uint32_t half = kGroup * width_;
        for (uint32_t k = 0; k < mine; ++k) {
            LocalTensor<float> in = inQue_.template DeQue<float>();
            uint32_t next = (k + 1) * cores + idx;
            if (k + 1 < rounds) {
                lane = lane + shift < cores ? lane + shift : lane + shift - cores;
                next = (k + 1) * cores + lane;
            }
            if (k + 1 < mine) {
                FetchPair(next);
            } else if (extraRows > 0) {
                FetchRows(extraBase, extraRows);
            }
            const uint32_t base = pair * pairRows;
            const uint32_t length = rows_ - base < pairRows ? rows_ - base : pairRows;
            const uint32_t first = length < kGroup ? length : kGroup;
            MulAddDst(group, in, in, static_cast<int32_t>(first * width_));
            if (length > kGroup) {
                MulAddDst(group, in[half], in[half], static_cast<int32_t>((length - kGroup) * width_));
            }
            inQue_.FreeTensor(in);
            pair = next;
        }
        if (extraRows > 0) {
            LocalTensor<float> in = inQue_.template DeQue<float>();
            MulAddDst(group, in, in, static_cast<int32_t>(extraRows * width_));
            inQue_.FreeTensor(in);
        }
        PipeBarrier<PIPE_V>();
        Add(group, group, group[(kGroup / 2) * width_], static_cast<int32_t>((kGroup / 2) * width_));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> offsets = ramp.ReinterpretCast<uint32_t>();
        for (uint32_t r = 0; r < kGroup / 2; ++r) {
            Gather(fold[r * pitch_], group, offsets, r * width_ * static_cast<uint32_t>(sizeof(float)), width_);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t n = kGroup / 2; n > 1; n /= 2) {
            Add(fold, fold, fold[(n / 2) * pitch_], static_cast<int32_t>((n / 2) * pitch_));
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicAdd<float>();
        Store(fold);
        SetAtomicNone();
    }

private:
    static constexpr uint32_t kGroup = 16;
    static constexpr uint32_t kBankOffset = 320;
    static constexpr uint32_t kLaneShift = 25;

    template <HardEvent event>
    __aicore__ inline void Sync()
    {
        const TEventID id = GetTPipePtr()->FetchEventID<event>();
        SetFlag<event>(id);
        WaitFlag<event>(id);
    }

    __aicore__ inline void FetchPair(uint32_t pair)
    {
        const uint32_t base = pair * 2 * kGroup;
        const uint32_t length = rows_ - base < 2 * kGroup ? rows_ - base : 2 * kGroup;
        FetchRows(base, length);
    }

    __aicore__ inline void FetchRows(uint32_t base, uint32_t rows)
    {
        LocalTensor<float> in = inQue_.template AllocTensor<float>();
        const uint64_t offset = static_cast<uint64_t>(base) * width_;
        const uint32_t count = rows * width_;
        if (((offset | count) & 7) == 0) {
            DataCopy(in, xGm_[offset], count);
        } else {
            DataCopyExtParams params{1, count * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            DataCopyPad(in, xGm_[offset], params, pad);
        }
        inQue_.EnQue(in);
    }

    __aicore__ inline void Store(const LocalTensor<float>& row)
    {
        DataCopyExtParams params{1, width_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(yGm_[0], row, params);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inQue_;
    TQue<TPosition::VECOUT, 1> outQue_;
    TBuf<TPosition::VECCALC> groupBuf_;
    TBuf<TPosition::VECCALC> foldBuf_;
    TBuf<TPosition::VECCALC> rampBuf_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    uint32_t rows_ = 0;
    uint32_t width_ = 0;
    uint32_t pitch_ = 0;
};
}

extern "C" __global__ __aicore__ void square_sum_v1(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    if constexpr (std::is_same<DTYPE_X, half>::value) {
        if (TILING_KEY_IS(401)) {
            WaveSliceKernel<DTYPE_X> op;
            op.Init(x, y, tiling);
            op.Process();
            return;
        }
    }
    if constexpr (sizeof(DTYPE_X) == 2) {
        if (TILING_KEY_IS(402)) {
            SpanEdgeKernel<DTYPE_X> op;
            op.Init(x, y, tiling);
            op.Process();
            return;
        }
    }
    if constexpr (std::is_same<DTYPE_X, half>::value) {
        if (TILING_KEY_IS(403)) {
            ShortRunKernel<DTYPE_X> op;
            op.Init(x, y, tiling);
            op.Process();
            return;
        }
    }
    if constexpr (std::is_same<DTYPE_X, float>::value) {
        if (TILING_KEY_IS(404)) {
            LaneCycleKernel<DTYPE_X> op;
            op.Init(x, y, tiling);
            op.Process();
            return;
        }
    }
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(405)) {
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.template Process<0>();
        return;
    }
    if (TILING_KEY_IS(406)) {
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.template Process<0>();
        return;
    }
    if (TILING_KEY_IS(407)) {
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.template Process<1>();
        return;
    }
    if (TILING_KEY_IS(408)) {
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.template Process<1>();
        return;
    }
    if (TILING_KEY_IS(409)) {
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.template Process<1>();
        return;
    }
    if (TILING_KEY_IS(410)) {
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.template Process<2>();
        return;
    }
    if (TILING_KEY_IS(411)) {
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.template Process<2>();
        return;
    }
    if (TILING_KEY_IS(412)) {
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.template Process<3>();
        return;
    }
    if (TILING_KEY_IS(413)) {
        SquareSumV1Kernel<DTYPE_X> op;
        op.Init(x, y, workspace, tilingData);
        op.template Process<3>();
        return;
    }
}
