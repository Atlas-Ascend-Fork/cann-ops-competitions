#include "index_add_tiling.h"
#include "kernel_operator.h"

#include <type_traits>

using namespace AscendC;

namespace {

constexpr int64_t kTileAlignElems = 32;
constexpr int64_t kAlignBytes = 32;
constexpr int64_t kNarrowArenaBytes = 184 * 1024;
constexpr int64_t kNarrowWorkspaceBytes = 2 * 1024 * 1024;
constexpr int64_t kNarrowChunkElems = 4096;
constexpr int64_t kL2BypassMinBytes = 128 * 1024 * 1024;
constexpr int64_t kSingleGatherMaxBytes = 255 * 256;
constexpr int64_t kMinParallelRowsPerCore = 16;

template <typename T>
__aicore__ inline int64_t AlignElems(int64_t elems)
{
    const int64_t alignElems = kAlignBytes / static_cast<int64_t>(sizeof(T));
    return ((elems + alignElems - 1) / alignElems) * alignElems;
}

__aicore__ inline int64_t AlignUp(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

__aicore__ inline void BuildByteOffsets(LocalTensor<int32_t> offsets, int32_t stride, int32_t count)
{
    ArithProgression(offsets, 0, stride, count < 8 ? count : 8);
    for (int32_t length = 8; length < count; length *= 2) {
        const int32_t extra = length < count - length ? length : count - length;
        Adds(offsets[length], offsets, length * stride, extra);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void CopyInTile(LocalTensor<T>& local, GlobalTensor<T>& global, int64_t offset,
                                  int64_t length)
{
    if (((offset * static_cast<int64_t>(sizeof(T))) % kAlignBytes) == 0 &&
        ((length * static_cast<int64_t>(sizeof(T))) % kAlignBytes) == 0) {
        DataCopy(local, global[offset], static_cast<uint32_t>(length));
        return;
    }
    DataCopyExtParams copy = {1, static_cast<uint32_t>(length * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
    DataCopyPad(local, global[offset], copy, pad);
}

template <typename T>
__aicore__ inline void CopyOutTile(GlobalTensor<T>& global, LocalTensor<T>& local, int64_t offset, int64_t length)
{
    if (((offset * static_cast<int64_t>(sizeof(T))) % kAlignBytes) == 0 &&
        ((length * static_cast<int64_t>(sizeof(T))) % kAlignBytes) == 0) {
        DataCopy(global[offset], local, static_cast<uint32_t>(length));
        return;
    }
    DataCopyExtParams copy = {1, static_cast<uint32_t>(length * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
    DataCopyPad(global[offset], local, copy);
}

template <typename T>
class KernelIndexAddAtomic {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                const IndexAddTilingData& tiling)
    {
        totalElems_ = tiling.totalElems;
        dimSize_ = tiling.dimSize;
        innerSize_ = tiling.innerSize;
        indexRows_ = tiling.indexRows;
        tileElems_ = tiling.tileElems;
        orderedTileElems_ = tiling.orderedTileElems;
        inputGm_.SetGlobalBuffer((__gm__ T*)input, totalElems_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output, totalElems_);
        sourceGm_.SetGlobalBuffer((__gm__ T*)source, indexRows_ * (totalElems_ / dimSize_));
        indexGm_.SetGlobalBuffer((__gm__ int32_t*)index, indexRows_);
        pipe_.InitBuffer(dataQueue_, 1, static_cast<uint32_t>(tileElems_ * sizeof(T)));
        pipe_.InitBuffer(indexQueue_, 1, static_cast<uint32_t>(AlignElems<int32_t>(indexRows_) * sizeof(int32_t)));
        mte2ToMte3Event_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_MTE3));
    }

    __aicore__ inline void Process(bool orderedAtomic)
    {
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        CopyInputToOutput(blockId, blockNum);
        SyncAll<true>();

        LocalTensor<int32_t> indexLocal = indexQueue_.AllocTensor<int32_t>();
        CopyInTile(indexLocal, indexGm_, 0, indexRows_);
        indexQueue_.EnQue(indexLocal);
        indexLocal = indexQueue_.DeQue<int32_t>();

        const int64_t atomicTileElems = orderedAtomic ? orderedTileElems_ : tileElems_;
        const int64_t innerTileCount = (innerSize_ + atomicTileElems - 1) / atomicTileElems;
        const int64_t outerCount = totalElems_ / (dimSize_ * innerSize_);
        const int64_t taskCount = outerCount * (orderedAtomic ? innerTileCount : indexRows_ * innerTileCount);
        SetAtomicAdd<T>();
        for (int64_t taskId = blockId; taskId < taskCount; taskId += blockNum) {
            if (orderedAtomic) {
                AtomicAddOrderedTile(taskId, innerTileCount, indexLocal);
            } else {
                AtomicAddSourceTile(taskId, innerTileCount, indexLocal);
            }
        }
        SetAtomicNone();
        indexQueue_.FreeTensor(indexLocal);
    }

private:
    __aicore__ inline void CopyInputToOutput(int64_t blockId, int64_t blockNum)
    {
        int64_t copyTileElems = AlignUp((totalElems_ + blockNum - 1) / blockNum, kTileAlignElems);
        if (copyTileElems > tileElems_) {
            copyTileElems = tileElems_;
        }
        const int64_t taskCount = (totalElems_ + copyTileElems - 1) / copyTileElems;
        for (int64_t taskId = blockId; taskId < taskCount; taskId += blockNum) {
            const int64_t offset = taskId * copyTileElems;
            const int64_t length = offset + copyTileElems <= totalElems_ ? copyTileElems : totalElems_ - offset;
            TransferTile(inputGm_, offset, outputGm_, offset, length);
        }
    }

    __aicore__ inline void AtomicAddSourceTile(int64_t taskId, int64_t innerTileCount,
                                               LocalTensor<int32_t>& indexLocal)
    {
        const int64_t innerTile = taskId % innerTileCount;
        const int64_t sourceRowTask = taskId / innerTileCount;
        const int64_t srcRow = sourceRowTask % indexRows_;
        const int64_t outer = sourceRowTask / indexRows_;
        const int64_t dimIndex = static_cast<int64_t>(indexLocal.GetValue(srcRow));
        const int64_t innerOffset = innerTile * tileElems_;
        const int64_t length = innerOffset + tileElems_ <= innerSize_ ? tileElems_ : innerSize_ - innerOffset;
        const int64_t sourceOffset = (outer * indexRows_ + srcRow) * innerSize_ + innerOffset;
        const int64_t outputOffset = (outer * dimSize_ + dimIndex) * innerSize_ + innerOffset;
        TransferTile(sourceGm_, sourceOffset, outputGm_, outputOffset, length);
    }

    __aicore__ inline void AtomicAddOrderedTile(int64_t taskId, int64_t innerTileCount,
                                                LocalTensor<int32_t>& indexLocal)
    {
        const int64_t innerTile = taskId % innerTileCount;
        const int64_t outer = taskId / innerTileCount;
        const int64_t innerOffset = innerTile * orderedTileElems_;
        const int64_t length =
            innerOffset + orderedTileElems_ <= innerSize_ ? orderedTileElems_ : innerSize_ - innerOffset;
        for (int64_t srcRow = 0; srcRow < indexRows_; ++srcRow) {
            const int64_t dimIndex = static_cast<int64_t>(indexLocal.GetValue(srcRow));
            const int64_t sourceOffset = (outer * indexRows_ + srcRow) * innerSize_ + innerOffset;
            const int64_t outputOffset = (outer * dimSize_ + dimIndex) * innerSize_ + innerOffset;
            TransferTile(sourceGm_, sourceOffset, outputGm_, outputOffset, length);
        }
    }

    __aicore__ inline void TransferTile(GlobalTensor<T>& source, int64_t sourceOffset,
                                        GlobalTensor<T>& destination, int64_t destinationOffset, int64_t length)
    {
        LocalTensor<T> dataLocal = dataQueue_.AllocTensor<T>();
        CopyInTile(dataLocal, source, sourceOffset, length);
        dataQueue_.EnQue(dataLocal);
        dataLocal = dataQueue_.DeQue<T>();
        SetFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        WaitFlag<HardEvent::MTE2_MTE3>(mte2ToMte3Event_);
        CopyOutTile(destination, dataLocal, destinationOffset, length);
        dataQueue_.FreeTensor(dataLocal);
    }

private:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQueue_;
    TQue<QuePosition::VECIN, 1> indexQueue_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<int32_t> indexGm_;
    int64_t totalElems_ = 0;
    int64_t dimSize_ = 0;
    int64_t innerSize_ = 0;
    int64_t indexRows_ = 0;
    int64_t tileElems_ = 0;
    int64_t orderedTileElems_ = 0;
    event_t mte2ToMte3Event_;
};

class KernelIndexAddSmallInt8 {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                const IndexAddTilingData& tiling)
    {
        rows_ = tiling.indexRows;
        inner_ = tiling.innerSize;
        dim_ = tiling.dimSize;
        sourceElems_ = rows_ * inner_;
        input_.SetGlobalBuffer((__gm__ int8_t*)input, tiling.totalElems);
        output_.SetGlobalBuffer((__gm__ int8_t*)output, tiling.totalElems);
        source_.SetGlobalBuffer((__gm__ int8_t*)source, sourceElems_);
        index_.SetGlobalBuffer((__gm__ int32_t*)index, rows_);
        pipe_.InitBuffer(arena_, 184 * 1024);
    }

    __aicore__ inline void Process()
    {
        const int64_t target = GetBlockIdx();
        const int64_t scan = AlignUp(rows_, 64);
        const int64_t sourceSpan = AlignUp(sourceElems_, 128);
        const int64_t offsetCount = AlignUp(sourceElems_ / 16, 8);
        const int64_t maskBytes = AlignUp(scan / 8, 32);
        LocalTensor<int32_t> index = arena_.Get<int32_t>();
        LocalTensor<int32_t> sequence = index[scan];
        LocalTensor<int32_t> selected = sequence[scan];
        LocalTensor<uint32_t> offsets = selected[scan].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> mask = offsets[offsetCount].ReinterpretCast<uint8_t>();
        LocalTensor<int8_t> raw = mask[maskBytes].ReinterpretCast<int8_t>();
        LocalTensor<half> sourceHalf = raw[sourceSpan].ReinterpretCast<half>();
        LocalTensor<half> gathered = sourceHalf[sourceSpan];
        LocalTensor<float> values = gathered[sourceSpan].ReinterpretCast<float>();
        LocalTensor<int8_t> input = values[sourceSpan].ReinterpretCast<int8_t>();
        LocalTensor<float> acc = input[inner_].ReinterpretCast<float>();
        LocalTensor<float> temp = acc[inner_];
        Duplicate(index, 0, static_cast<int32_t>(scan));
        Duplicate(offsets, static_cast<uint32_t>(0), static_cast<int32_t>(offsetCount));
        ArithProgression(sequence, 0, static_cast<int32_t>(inner_ * sizeof(half)), static_cast<int32_t>(scan));
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInTile(index, index_, 0, rows_);
        CopyInTile(raw, source_, 0, sourceElems_);
        CopyInTile(input, input_, target * inner_, inner_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(index.ReinterpretCast<float>(), index, RoundMode::CAST_NONE, static_cast<int32_t>(scan));
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, index.ReinterpretCast<float>(), static_cast<float>(target), CMPMODE::EQ,
                      static_cast<int32_t>(scan));
        PipeBarrier<PIPE_V>();
        uint64_t hits = 0;
        GatherMask(selected, sequence, mask.ReinterpretCast<uint32_t>(), true,
                   static_cast<uint32_t>(rows_), {1, 1, 8, 8}, hits);
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        if (hits > 0) {
            Cast(sourceHalf, raw, RoundMode::CAST_NONE, static_cast<int32_t>(sourceElems_));
            const int64_t blocks = inner_ / 16;
            for (int64_t hit = 0; hit < static_cast<int64_t>(hits); ++hit) {
                const uint32_t base = static_cast<uint32_t>(selected.GetValue(hit));
                for (int64_t block = 0; block < blocks; ++block) {
                    offsets.SetValue(hit * blocks + block, base + block * 32);
                }
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            PipeBarrier<PIPE_V>();
            Gatherb(gathered.ReinterpretCast<uint16_t>(), sourceHalf.ReinterpretCast<uint16_t>(), offsets,
                    static_cast<uint8_t>((hits * blocks + 7) / 8), {1, 8});
            PipeBarrier<PIPE_V>();
            Cast(values, gathered, RoundMode::CAST_NONE, static_cast<int32_t>(hits * inner_));
            PipeBarrier<PIPE_V>();
            for (int64_t active = static_cast<int64_t>(hits); active > 1; active = (active + 1) / 2) {
                const int64_t left = (active + 1) / 2;
                Add(values, values, values[left * inner_], static_cast<int32_t>((active - left) * inner_));
                PipeBarrier<PIPE_V>();
            }
        }
        Cast(gathered, input, RoundMode::CAST_NONE, static_cast<int32_t>(inner_));
        PipeBarrier<PIPE_V>();
        Cast(acc, gathered, RoundMode::CAST_NONE, static_cast<int32_t>(inner_));
        PipeBarrier<PIPE_V>();
        if (hits > 0) {
            Add(acc, acc, values, static_cast<int32_t>(inner_));
            PipeBarrier<PIPE_V>();
        }
        Adds(acc, acc, 128.0f, static_cast<int32_t>(inner_));
        PipeBarrier<PIPE_V>();
        Muls(temp, acc, 1.0f / 256.0f, static_cast<int32_t>(inner_));
        PipeBarrier<PIPE_V>();
        Cast(temp, temp, RoundMode::CAST_FLOOR, static_cast<int32_t>(inner_));
        PipeBarrier<PIPE_V>();
        Muls(temp, temp, 256.0f, static_cast<int32_t>(inner_));
        PipeBarrier<PIPE_V>();
        Sub(acc, acc, temp, static_cast<int32_t>(inner_));
        PipeBarrier<PIPE_V>();
        Adds(acc, acc, -128.0f, static_cast<int32_t>(inner_));
        PipeBarrier<PIPE_V>();
        Cast(gathered, acc, RoundMode::CAST_RINT, static_cast<int32_t>(inner_));
        PipeBarrier<PIPE_V>();
        Cast(input, gathered, RoundMode::CAST_NONE, static_cast<int32_t>(inner_));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        CopyOutTile(output_, input, target * inner_, inner_);
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<int8_t> input_, source_, output_;
    GlobalTensor<int32_t> index_;
    int64_t rows_, inner_, dim_, sourceElems_;
};

class KernelIndexAddReducedAtomic {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                const IndexAddTilingData& tiling)
    {
        total_ = tiling.totalElems;
        dim_ = tiling.dimSize;
        inner_ = tiling.innerSize;
        rows_ = tiling.indexRows;
        stride_ = AlignUp(inner_, kTileAlignElems);
        accElems_ = dim_ * stride_;
        const int64_t indexElems = AlignUp(rows_, kTileAlignElems);
        gatherElems_ = AlignUp(inner_ + 7, 64);
        const int64_t rawElems = AlignUp(4 * inner_, 64) + 64;
        batch_ = 4;
        input_.SetGlobalBuffer((__gm__ int32_t*)input, total_);
        output_.SetGlobalBuffer((__gm__ int32_t*)output, total_);
        source_.SetGlobalBuffer((__gm__ int32_t*)source, total_ / dim_ * rows_);
        source_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        index_.SetGlobalBuffer((__gm__ int32_t*)index, rows_);
        pipe_.InitBuffer(accBuf_, accElems_ * sizeof(int32_t));
        pipe_.InitBuffer(indexBuf_, indexElems * sizeof(int32_t));
        pipe_.InitBuffer(sourceQueue_, 1, rawElems * sizeof(int32_t));
        pipe_.InitBuffer(gatherBuf_, gatherElems_ * sizeof(int32_t));
        pipe_.InitBuffer(offsetBuf_, 8 * AlignUp(gatherElems_ / 8, 32));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> acc = accBuf_.Get<int32_t>();
        const int64_t block = GetBlockIdx();
        const int64_t cores = GetBlockNum();
        const int64_t copyElems = AlignUp((total_ + cores - 1) / cores, kTileAlignElems);
        const int64_t tile = copyElems < accElems_ ? copyElems : accElems_;
        for (int64_t offset = block * tile; offset < total_; offset += cores * tile) {
            const int64_t length = offset + tile <= total_ ? tile : total_ - offset;
            CopyInTile(acc, input_, offset, length);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            CopyOutTile(output_, acc, offset, length);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        SyncAll<true>();
        LocalTensor<int32_t> indices = indexBuf_.Get<int32_t>();
        CopyInTile(indices, index_, 0, rows_);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        LocalTensor<int32_t> gathered = gatherBuf_.Get<int32_t>();
        LocalTensor<uint8_t> mask = offsetBuf_.Get<uint8_t>();
        const int64_t maskBytes = AlignUp(gatherElems_ / 8, 32);
        Duplicate(mask.ReinterpretCast<uint32_t>(), UINT32_MAX,
                  static_cast<int32_t>(8 * maskBytes / 4));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t shift = 0; shift < 8; ++shift) {
            mask.ReinterpretCast<uint32_t>().SetValue(shift * maskBytes / 4, UINT32_MAX << shift);
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        const int64_t totalRows = total_ / (dim_ * inner_) * rows_;
        const int64_t end = totalRows * (block + 1) / cores;
        for (int64_t begin = totalRows * block / cores; begin < end;) {
            const int64_t task = begin / rows_;
            const int64_t outer = task;
            offset_ = 0;
            length_ = inner_;
            const int64_t limit = (task + 1) * rows_ < end ? (task + 1) * rows_ : end;
            Duplicate(acc, 0, static_cast<int32_t>(accElems_));
            const int64_t firstRows = begin + batch_ <= limit ? batch_ : limit - begin;
            Prefetch(outer * rows_ + begin % rows_, firstRows);
            for (int64_t row = begin; row < limit; row += batch_) {
                LocalTensor<int32_t> data = sourceQueue_.DeQue<int32_t>();
                const int64_t count = row + batch_ <= limit ? batch_ : limit - row;
                for (int64_t r = 0; r < count; ++r) {
                    const int64_t target = indices.GetValue(row - task * rows_ + r);
                    const int64_t rawOffset = r * inner_;
                    const int64_t shift = rawOffset % 8;
                    if (shift != 0) {
                        uint64_t selected = 0;
                        GatherMask(gathered, data[rawOffset / 8 * 8],
                                   mask[shift * maskBytes].ReinterpretCast<uint32_t>(),
                                   true, static_cast<uint32_t>(inner_ + shift), {1, 1, 8, 8}, selected);
                        PipeBarrier<PIPE_V>();
                    }
                    Add(acc[target * stride_], acc[target * stride_], shift == 0 ? data[rawOffset] : gathered,
                        static_cast<int32_t>(length_));
                    PipeBarrier<PIPE_V>();
                }
                sourceQueue_.FreeTensor(data);
                if (row + batch_ < limit) {
                    const int64_t nextRows = row + 2 * batch_ <= limit ? batch_ : limit - row - batch_;
                    Prefetch(outer * rows_ + row % rows_ + batch_, nextRows);
                }
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const uint32_t localStride = (stride_ - AlignElems<int32_t>(length_)) / 8;
            DataCopyExtParams copy = {static_cast<uint16_t>(dim_),
                static_cast<uint32_t>(length_ * sizeof(int32_t)), localStride,
                static_cast<uint32_t>((inner_ - length_) * sizeof(int32_t)), 0};
            SetAtomicAdd<int32_t>();
            DataCopyPad(output_[outer * dim_ * inner_ + offset_], acc, copy);
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            begin = limit;
        }
    }

private:
    __aicore__ inline void Prefetch(int64_t row, int64_t count)
    {
        LocalTensor<int32_t> data = sourceQueue_.AllocTensor<int32_t>();
        CopyInTile(data, source_, row * inner_, count * inner_);
        sourceQueue_.EnQue(data);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> gatherBuf_;
    TBuf<TPosition::VECCALC> offsetBuf_;
    TQue<TPosition::VECIN, 1> sourceQueue_;
    GlobalTensor<int32_t> input_, output_, source_, index_;
    int64_t total_, dim_, inner_, rows_, stride_, accElems_, batch_, gatherElems_, offset_, length_;
};

template <typename T>
class KernelIndexAddOuterOwned {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                const IndexAddTilingData& tiling)
    {
        totalElems_ = tiling.totalElems;
        dimSize_ = tiling.dimSize;
        innerSize_ = tiling.innerSize;
        indexRows_ = tiling.indexRows;
        outputElems_ = dimSize_ * innerSize_;
        sourceElems_ = indexRows_ * innerSize_;
        paddedInner_ = AlignElems<T>(innerSize_);
        paddedOutputElems_ = dimSize_ * paddedInner_;
        inputGm_.SetGlobalBuffer((__gm__ T*)input, totalElems_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output, totalElems_);
        sourceGm_.SetGlobalBuffer((__gm__ T*)source, indexRows_ * (totalElems_ / dimSize_));
        indexGm_.SetGlobalBuffer((__gm__ int32_t*)index, indexRows_);
        pipe_.InitBuffer(indexQueue_, 1, static_cast<uint32_t>(AlignElems<int32_t>(indexRows_) * sizeof(int32_t)));
        pipe_.InitBuffer(sourceBuf_, static_cast<uint32_t>(AlignElems<T>(sourceElems_) * sizeof(T)));
        pipe_.InitBuffer(roundedBuf_, static_cast<uint32_t>(paddedOutputElems_ * sizeof(T)));
        pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(paddedOutputElems_ * sizeof(float)));
        pipe_.InitBuffer(tempBuf_, static_cast<uint32_t>(paddedInner_ * sizeof(float)));
        pipe_.InitBuffer(rowBuf_, static_cast<uint32_t>(paddedInner_ * sizeof(T)));
        pipe_.InitBuffer(offsetBuf_, static_cast<uint32_t>(paddedInner_ * sizeof(uint32_t)));
        if constexpr (std::is_same_v<T, int8_t>) {
            pipe_.InitBuffer(halfBuf_, static_cast<uint32_t>(paddedInner_ * sizeof(half)));
            pipe_.InitBuffer(sourceHalfBuf_,
                             static_cast<uint32_t>(AlignElems<half>(sourceElems_) * sizeof(half)));
        }
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> indexLocal = indexQueue_.AllocTensor<int32_t>();
        CopyInTile(indexLocal, indexGm_, 0, indexRows_);
        indexQueue_.EnQue(indexLocal);
        indexLocal = indexQueue_.DeQue<int32_t>();
        LocalTensor<int32_t> signedOffsets = offsetBuf_.Get<int32_t>();
        constexpr int32_t gatherBytes = std::is_same_v<T, int8_t> ? sizeof(half) : sizeof(T);
        ArithProgression<int32_t>(signedOffsets, 0, gatherBytes,
                                  static_cast<uint32_t>(innerSize_));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> offsets = signedOffsets.ReinterpretCast<uint32_t>();
        const int64_t outerCount = totalElems_ / outputElems_;
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        for (int64_t outer = blockId; outer < outerCount; outer += blockNum) {
            ProcessOuter(outer, indexLocal, offsets);
        }
        indexQueue_.FreeTensor(indexLocal);
    }

private:
    __aicore__ inline void ProcessOuter(int64_t outer, LocalTensor<int32_t>& indexLocal,
                                        LocalTensor<uint32_t>& offsets)
    {
        LocalTensor<T> sourceLocal = sourceBuf_.Get<T>();
        LocalTensor<T> rounded = roundedBuf_.Get<T>();
        LocalTensor<T> row = rowBuf_.Get<T>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> temp = tempBuf_.Get<float>();
        CopyRowsIn(rounded, inputGm_, outer * outputElems_, dimSize_);
        CopyInTile(sourceLocal, sourceGm_, outer * sourceElems_, sourceElems_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        LocalTensor<half> sourceHalf;
        if constexpr (std::is_same_v<T, int8_t>) {
            sourceHalf = sourceHalfBuf_.Get<half>();
            Cast(sourceHalf, sourceLocal, RoundMode::CAST_NONE, static_cast<int32_t>(sourceElems_));
            PipeBarrier<PIPE_V>();
        }

        if constexpr (std::is_same_v<T, bfloat16_t>) {
            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                const int64_t offset = dimIndex * paddedInner_;
                Cast(acc[offset], rounded[offset], RoundMode::CAST_NONE, static_cast<int32_t>(innerSize_));
            }
            for (int64_t srcRow = 0; srcRow < indexRows_; ++srcRow) {
                const int64_t dstOffset = static_cast<int64_t>(indexLocal.GetValue(srcRow)) * paddedInner_;
                Gather(row, sourceLocal, offsets, static_cast<uint32_t>(srcRow * innerSize_ * sizeof(T)),
                       static_cast<uint32_t>(innerSize_));
                Cast(temp, row, RoundMode::CAST_NONE, static_cast<int32_t>(innerSize_));
                Add(acc[dstOffset], acc[dstOffset], temp, static_cast<int32_t>(innerSize_));
                Cast(rounded[dstOffset], acc[dstOffset], RoundMode::CAST_RINT, static_cast<int32_t>(innerSize_));
                Cast(acc[dstOffset], rounded[dstOffset], RoundMode::CAST_NONE, static_cast<int32_t>(innerSize_));
            }
        } else if constexpr (std::is_same_v<T, int8_t>) {
            LocalTensor<half> halfLocal = halfBuf_.Get<half>();
            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                const int64_t offset = dimIndex * paddedInner_;
                Cast(halfLocal, rounded[offset], RoundMode::CAST_NONE, static_cast<int32_t>(innerSize_));
                Cast(acc[offset], halfLocal, RoundMode::CAST_NONE, static_cast<int32_t>(innerSize_));
            }
            for (int64_t srcRow = 0; srcRow < indexRows_; ++srcRow) {
                const int64_t dstOffset = static_cast<int64_t>(indexLocal.GetValue(srcRow)) * paddedInner_;
                Gather(halfLocal, sourceHalf, offsets,
                       static_cast<uint32_t>(srcRow * innerSize_ * sizeof(half)),
                       static_cast<uint32_t>(innerSize_));
                Cast(temp, halfLocal, RoundMode::CAST_NONE, static_cast<int32_t>(innerSize_));
                Add(acc[dstOffset], acc[dstOffset], temp, static_cast<int32_t>(innerSize_));
            }
            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                WrapInt8(rounded[dimIndex * paddedInner_], acc[dimIndex * paddedInner_], temp, halfLocal);
            }
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        CopyRowsOut(outputGm_, rounded, outer * outputElems_, dimSize_);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void CopyRowsIn(LocalTensor<T> local, GlobalTensor<T>& global,
                                      int64_t globalOffset, int64_t rows)
    {
        DataCopyExtParams copy = {static_cast<uint16_t>(rows),
                                  static_cast<uint32_t>(innerSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, global[globalOffset], copy, pad);
    }

    __aicore__ inline void CopyRowsOut(GlobalTensor<T>& global, LocalTensor<T> local,
                                       int64_t globalOffset, int64_t rows)
    {
        DataCopyExtParams copy = {static_cast<uint16_t>(rows),
                                  static_cast<uint32_t>(innerSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(global[globalOffset], local, copy);
    }

    __aicore__ inline void WrapInt8(LocalTensor<int8_t> output, LocalTensor<float> acc,
                                    LocalTensor<float> temp, LocalTensor<half> halfLocal)
    {
        const int32_t count = static_cast<int32_t>(innerSize_);
        Adds(acc, acc, 128.0F, count);
        Muls(temp, acc, 1.0F / 256.0F, count);
        Cast(temp, temp, RoundMode::CAST_FLOOR, count);
        Muls(temp, temp, 256.0F, count);
        Sub(acc, acc, temp, count);
        Adds(acc, acc, -128.0F, count);
        Cast(halfLocal, acc, RoundMode::CAST_RINT, count);
        Cast(output, halfLocal, RoundMode::CAST_NONE, count);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> indexQueue_;
    TBuf<QuePosition::VECCALC> sourceBuf_;
    TBuf<QuePosition::VECCALC> roundedBuf_;
    TBuf<QuePosition::VECCALC> accBuf_;
    TBuf<QuePosition::VECCALC> tempBuf_;
    TBuf<QuePosition::VECCALC> halfBuf_;
    TBuf<QuePosition::VECCALC> sourceHalfBuf_;
    TBuf<QuePosition::VECCALC> rowBuf_;
    TBuf<QuePosition::VECCALC> offsetBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<int32_t> indexGm_;
    int64_t totalElems_ = 0;
    int64_t dimSize_ = 0;
    int64_t innerSize_ = 0;
    int64_t indexRows_ = 0;
    int64_t outputElems_ = 0;
    int64_t sourceElems_ = 0;
    int64_t paddedInner_ = 0;
    int64_t paddedOutputElems_ = 0;
};

class KernelIndexAddBf16TargetOwned {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                const IndexAddTilingData& tiling)
    {
        totalElems_ = tiling.totalElems;
        dimSize_ = tiling.dimSize;
        innerSize_ = tiling.innerSize;
        indexRows_ = tiling.indexRows;
        outputElems_ = dimSize_ * innerSize_;
        sourceElems_ = indexRows_ * innerSize_;
        outerCount_ = totalElems_ / outputElems_;
        paddedInner_ = AlignElems<bfloat16_t>(innerSize_);
        const int64_t maxTargets = (dimSize_ + GetBlockNum() - 1) / GetBlockNum();
        const int64_t indexBytes = AlignUp(indexRows_, 64) * sizeof(int32_t);
        stateTileElems_ = outerCount_ * paddedInner_;

        inputGm_.SetGlobalBuffer((__gm__ bfloat16_t*)input, totalElems_);
        outputGm_.SetGlobalBuffer((__gm__ bfloat16_t*)output, totalElems_);
        sourceGm_.SetGlobalBuffer((__gm__ bfloat16_t*)source, outerCount_ * sourceElems_);
        indexGm_.SetGlobalBuffer((__gm__ int32_t*)index, indexRows_);
        pipe_.InitBuffer(indexQueue_, 1, static_cast<uint32_t>(indexBytes));
        pipe_.InitBuffer(stateBuf_, static_cast<uint32_t>(maxTargets * stateTileElems_ * sizeof(bfloat16_t)));
        pipe_.InitBuffer(sourceBuf_, static_cast<uint32_t>(2 * stateTileElems_ * sizeof(bfloat16_t)));
        pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(stateTileElems_ * sizeof(float)));
        pipe_.InitBuffer(tempBuf_, static_cast<uint32_t>(stateTileElems_ * sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> index = indexQueue_.AllocTensor<int32_t>();
        CopyInTile(index, indexGm_, 0, indexRows_);
        indexQueue_.EnQue(index);
        index = indexQueue_.DeQue<int32_t>();

        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t dimBegin = dimSize_ * blockId / blockNum;
        const int64_t dimEnd = dimSize_ * (blockId + 1) / blockNum;
        const int64_t targetCount = dimEnd - dimBegin;
        LocalTensor<bfloat16_t> states = stateBuf_.Get<bfloat16_t>();
        for (int64_t localTarget = 0; localTarget < targetCount; ++localTarget) {
            CopyRowsIn(states[localTarget * stateTileElems_], inputGm_, dimBegin + localTarget,
                       outputElems_);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        LocalTensor<bfloat16_t> source = sourceBuf_.Get<bfloat16_t>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> temp = tempBuf_.Get<float>();
        const int32_t count = static_cast<int32_t>(stateTileElems_);
        const int64_t maskBytes = AlignUp((indexRows_ + 7) / 8, 32);
        const int32_t compareCount = static_cast<int32_t>(AlignUp(indexRows_, 64));
        int32_t targetShift = 0;
        while ((int64_t(1) << targetShift) < dimSize_) {
            ++targetShift;
        }
        const int32_t targetRadix = int32_t(1) << targetShift;
        const bool filter = compareCount <= stateTileElems_ &&
                            2 * maskBytes <= stateTileElems_ * sizeof(bfloat16_t) &&
                            indexRows_ * targetRadix <= INT32_MAX;
        int64_t hitCount = indexRows_;
        if (filter) {
            LocalTensor<uint8_t> low = source.template ReinterpretCast<uint8_t>();
            LocalTensor<uint8_t> high = low[maskBytes];
            LocalTensor<int32_t> sequence = acc.template ReinterpretCast<int32_t>();
            Cast(temp, index, RoundMode::CAST_NONE, compareCount);
            CreateVecIndex(sequence, 0, static_cast<uint32_t>(indexRows_));
            PipeBarrier<PIPE_V>();
            Muls(sequence, sequence, targetRadix, static_cast<int32_t>(indexRows_));
            CompareScalar(low, temp, static_cast<float>(dimBegin), CMPMODE::GE, compareCount);
            CompareScalar(high, temp, static_cast<float>(dimEnd), CMPMODE::LT, compareCount);
            PipeBarrier<PIPE_V>();
            Add(sequence, sequence, index, static_cast<int32_t>(indexRows_));
            And(low.template ReinterpretCast<uint16_t>(), low.template ReinterpretCast<uint16_t>(),
                high.template ReinterpretCast<uint16_t>(), static_cast<int32_t>(maskBytes / 2));
            PipeBarrier<PIPE_V>();
            uint64_t selected = 0;
            GatherMask(index, sequence, low.template ReinterpretCast<uint32_t>(), true,
                       static_cast<uint32_t>(indexRows_), {1, 1, 8, 8}, selected);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            hitCount = static_cast<int64_t>(selected);
        }
        if (filter && hitCount > 0) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyRowsIn(source, sourceGm_, index.GetValue(0) >> targetShift, sourceElems_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }
        for (int64_t hit = 0; hit < hitCount; ++hit) {
            const int32_t entry = index.GetValue(hit);
            const int64_t srcRow = filter ? entry >> targetShift : hit;
            const int32_t dst = filter ? entry & (targetRadix - 1) : entry;
            if (dst >= dimBegin && dst < dimEnd) {
                LocalTensor<bfloat16_t> state = states[(dst - dimBegin) * stateTileElems_];
                const event_t event = static_cast<event_t>(filter ? (hit & 1) : 0);
                LocalTensor<bfloat16_t> data = source[filter ? (hit & 1) * stateTileElems_ : 0];
                if (filter) {
                    WaitFlag<HardEvent::MTE2_V>(event);
                    if (hit + 1 < hitCount) {
                        const event_t nextEvent = static_cast<event_t>((hit + 1) & 1);
                        SetFlag<HardEvent::V_MTE2>(nextEvent);
                        WaitFlag<HardEvent::V_MTE2>(nextEvent);
                        CopyRowsIn(source[((hit + 1) & 1) * stateTileElems_], sourceGm_,
                                   index.GetValue(hit + 1) >> targetShift, sourceElems_);
                        SetFlag<HardEvent::MTE2_V>(nextEvent);
                    }
                } else {
                    SetFlag<HardEvent::V_MTE2>(event);
                    WaitFlag<HardEvent::V_MTE2>(event);
                    CopyRowsIn(data, sourceGm_, srcRow, sourceElems_);
                    SetFlag<HardEvent::MTE2_V>(event);
                }
                Cast(acc, state, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                if (!filter) {
                    WaitFlag<HardEvent::MTE2_V>(event);
                }
                Cast(temp, data, RoundMode::CAST_NONE, count);
                PipeBarrier<PIPE_V>();
                Add(acc, acc, temp, count);
                PipeBarrier<PIPE_V>();
                Cast(state, acc, RoundMode::CAST_RINT, count);
                PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for (int64_t localTarget = 0; localTarget < targetCount; ++localTarget) {
            CopyRowsOut(outputGm_, states[localTarget * stateTileElems_], dimBegin + localTarget);
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        indexQueue_.FreeTensor(index);
    }

private:
    __aicore__ inline void CopyRowsIn(LocalTensor<bfloat16_t> local,
                                      GlobalTensor<bfloat16_t>& global, int64_t row,
                                      int64_t outerStride)
    {
        const uint32_t rowBytes = static_cast<uint32_t>(innerSize_ * sizeof(bfloat16_t));
        const uint32_t sourceStride = static_cast<uint32_t>((outerStride - innerSize_) * sizeof(bfloat16_t));
        DataCopyExtParams copy = {static_cast<uint16_t>(outerCount_), rowBytes, sourceStride, 0, 0};
        DataCopyPadExtParams<bfloat16_t> pad = {false, 0, 0, static_cast<bfloat16_t>(0)};
        DataCopyPad(local, global[row * innerSize_], copy, pad);
    }

    __aicore__ inline void CopyRowsOut(GlobalTensor<bfloat16_t>& global,
                                       LocalTensor<bfloat16_t> local, int64_t row)
    {
        const uint32_t rowBytes = static_cast<uint32_t>(innerSize_ * sizeof(bfloat16_t));
        const uint32_t destinationStride = static_cast<uint32_t>((outputElems_ - innerSize_) * sizeof(bfloat16_t));
        DataCopyExtParams copy = {static_cast<uint16_t>(outerCount_), rowBytes, 0, destinationStride, 0};
        DataCopyPad(global[row * innerSize_], local, copy);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> indexQueue_;
    TBuf<QuePosition::VECCALC> stateBuf_;
    TBuf<QuePosition::VECCALC> sourceBuf_;
    TBuf<QuePosition::VECCALC> accBuf_;
    TBuf<QuePosition::VECCALC> tempBuf_;
    GlobalTensor<bfloat16_t> inputGm_;
    GlobalTensor<bfloat16_t> sourceGm_;
    GlobalTensor<bfloat16_t> outputGm_;
    GlobalTensor<int32_t> indexGm_;
    int64_t totalElems_ = 0;
    int64_t dimSize_ = 0;
    int64_t innerSize_ = 0;
    int64_t indexRows_ = 0;
    int64_t outputElems_ = 0;
    int64_t sourceElems_ = 0;
    int64_t outerCount_ = 0;
    int64_t paddedInner_ = 0;
    int64_t stateTileElems_ = 0;
};

template <typename T, bool UseHitList>
class KernelIndexAddOutputOwned {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output, GM_ADDR workspace,
                                const IndexAddTilingData& tiling)
    {
        totalElems_ = tiling.totalElems;
        dimSize_ = tiling.dimSize;
        innerSize_ = tiling.innerSize;
        indexRows_ = tiling.indexRows;
        tileElems_ = tiling.outputTileElems;
        inputGm_.SetGlobalBuffer((__gm__ T*)input, totalElems_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output, totalElems_);
        sourceGm_.SetGlobalBuffer((__gm__ T*)source, indexRows_ * (totalElems_ / dimSize_));
        indexGm_.SetGlobalBuffer((__gm__ int32_t*)index, indexRows_);
        if constexpr (UseHitList && std::is_same_v<T, int32_t>) {
            const int64_t outerCount = totalElems_ / (dimSize_ * innerSize_);
            const int64_t sourceBytes = indexRows_ * (totalElems_ / dimSize_) * sizeof(T);
            if (sourceBytes >= kL2BypassMinBytes) {
                sourceGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                outputGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            const int64_t indexBytes = AlignElems<int32_t>(indexRows_) * sizeof(int32_t);
            const int64_t cursorBytes = AlignUp(dimSize_ + 1, 64) * sizeof(int32_t);
            const int64_t metadataBytes = AlignElems<int32_t>(dimSize_ + 1 + indexRows_) * sizeof(int32_t);
            const int64_t rowBytes = tileElems_ * sizeof(T);
            const int64_t metadataFixedBytes = indexBytes + cursorBytes + metadataBytes;
            const int64_t availableRows = metadataFixedBytes < kNarrowArenaBytes ?
                (kNarrowArenaBytes - metadataFixedBytes) / rowBytes : 0;
            const int64_t maxBatchRows = availableRows - 4;
            if (sourceBytes >= kL2BypassMinBytes && tileElems_ >= innerSize_ && maxBatchRows >= 2) {
                int64_t candidateRows = maxBatchRows < 8 ? maxBatchRows : 8;
                int64_t candidateSlots = 1;
                if (maxBatchRows >= 6 && (innerSize_ * sizeof(T)) % kAlignBytes != 0) {
                    candidateRows = maxBatchRows / 2 < 4 ? maxBatchRows / 2 : 4;
                    candidateSlots = 2;
                }
                const int64_t taskCount =
                    outerCount * ((dimSize_ + candidateRows - 1) / candidateRows);
                if (taskCount >= GetBlockNum()) {
                    batchRows_ = candidateRows;
                    batchSlots_ = candidateSlots;
                    sourceBatchRows_ = 2;
                }
            }
        }
        if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, bfloat16_t>) {
            pipe_.InitBuffer(dataQueue_, 1, static_cast<uint32_t>(tileElems_ * sizeof(T)));
            pipe_.InitBuffer(outputQueue_, 1, static_cast<uint32_t>(tileElems_ * sizeof(T)));
        } else {
            pipe_.InitBuffer(dataQueue_, 2,
                             static_cast<uint32_t>(tileElems_ * sourceBatchRows_ * sizeof(T)));
        }
        pipe_.InitBuffer(indexQueue_, 1, static_cast<uint32_t>(AlignElems<int32_t>(indexRows_) * sizeof(int32_t)));
        if constexpr (UseHitList) {
            metadataElems_ = dimSize_ + 1 + indexRows_;
            metadataGm_.SetGlobalBuffer((__gm__ int32_t*)workspace, metadataElems_);
            countStride_ = AlignUp(dimSize_ + 1, 64);
            const int64_t countsOffset = AlignUp(metadataElems_ * static_cast<int64_t>(sizeof(int32_t)), 512);
            countsGm_.SetGlobalBuffer((__gm__ int32_t*)(workspace + countsOffset), 40 * countStride_);
            pipe_.InitBuffer(cursorBuf_, static_cast<uint32_t>(countStride_ * sizeof(int32_t)));
            pipe_.InitBuffer(metadataBuf_,
                             static_cast<uint32_t>(AlignElems<int32_t>(metadataElems_) * sizeof(int32_t)));
        }
        if constexpr (std::is_same_v<T, int8_t>) {
            pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(tileElems_ * sizeof(float)));
            pipe_.InitBuffer(tempBuf_, static_cast<uint32_t>(tileElems_ * sizeof(float)));
            pipe_.InitBuffer(halfBuf_, static_cast<uint32_t>(tileElems_ * sizeof(half)));
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            pipe_.InitBuffer(accBuf_, static_cast<uint32_t>(tileElems_ * sizeof(float)));
            pipe_.InitBuffer(tempBuf_, static_cast<uint32_t>(tileElems_ * sizeof(float)));
        } else {
            pipe_.InitBuffer(accBuf_,
                             static_cast<uint32_t>(tileElems_ * batchRows_ * batchSlots_ * sizeof(T)));
        }
        sToMte3_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        mte3ToS_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<int32_t> indexLocal = indexQueue_.AllocTensor<int32_t>();
        CopyInTile(indexLocal, indexGm_, 0, indexRows_);
        indexQueue_.EnQue(indexLocal);
        indexLocal = indexQueue_.DeQue<int32_t>();
        LocalTensor<int32_t> metadata = indexLocal;
        if constexpr (UseHitList) {
            metadata = metadataBuf_.Get<int32_t>();
            if constexpr (std::is_same_v<T, int32_t>) {
                if (batchRows_ > 1) {
                    BuildMetadataParallel(indexLocal, metadata, batchRows_);
                    ProcessWideBatches(metadata);
                    indexQueue_.FreeTensor(indexLocal);
                    return;
                }
            }
            BuildMetadata(indexLocal, metadata);
        }

        const int64_t innerTileCount = (innerSize_ + tileElems_ - 1) / tileElems_;
        const int64_t outerCount = totalElems_ / (dimSize_ * innerSize_);
        const int64_t taskCount = outerCount * dimSize_ * innerTileCount;
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        for (int64_t taskId = blockId; taskId < taskCount; taskId += blockNum) {
            ProcessTile(taskId, innerTileCount, indexLocal, metadata);
        }
        indexQueue_.FreeTensor(indexLocal);
    }

private:
    __aicore__ inline void BuildMetadata(LocalTensor<int32_t>& indexLocal, LocalTensor<int32_t>& metadata)
    {
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t rowCountMax = (indexRows_ + blockNum - 1) / blockNum;
        const int64_t scratchElems = AlignElems<int32_t>(rowCountMax) + 2 * countStride_;
        if constexpr (std::is_same_v<T, int32_t>) {
            if (blockNum > 1 && indexRows_ >= blockNum * kMinParallelRowsPerCore &&
                scratchElems <= AlignElems<int32_t>(metadataElems_)) {
                BuildMetadataParallel(indexLocal, metadata, 1);
                return;
            }
        }
        if (GetBlockIdx() == 0) {
            LocalTensor<int32_t> cursor = cursorBuf_.Get<int32_t>();
            Duplicate(cursor, 0, static_cast<int32_t>(dimSize_));
            PipeBarrier<PIPE_ALL>();
            for (int64_t srcRow = 0; srcRow < indexRows_; ++srcRow) {
                const int32_t dimIndex = indexLocal.GetValue(srcRow);
                cursor.SetValue(dimIndex, cursor.GetValue(dimIndex) + 1);
            }

            int32_t prefix = 0;
            metadata.SetValue(0, 0);
            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                prefix += cursor.GetValue(dimIndex);
                metadata.SetValue(dimIndex + 1, prefix);
            }

            Duplicate(cursor, 0, static_cast<int32_t>(dimSize_));
            PipeBarrier<PIPE_ALL>();
            for (int64_t srcRow = 0; srcRow < indexRows_; ++srcRow) {
                const int32_t dimIndex = indexLocal.GetValue(srcRow);
                const int32_t slot = metadata.GetValue(dimIndex) + cursor.GetValue(dimIndex);
                metadata.SetValue(dimSize_ + 1 + slot, static_cast<int32_t>(srcRow));
                cursor.SetValue(dimIndex, cursor.GetValue(dimIndex) + 1);
            }
            CopyOutTile(metadataGm_, metadata, 0, metadataElems_);
            PipeBarrier<PIPE_ALL>();
        }
        SyncAll<true>();
        CopyInTile(metadata, metadataGm_, 0, metadataElems_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BuildMetadataParallel(LocalTensor<int32_t>& indexLocal,
                                                 LocalTensor<int32_t>& scratch, int64_t groupRows)
    {
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t groupCount = (dimSize_ + groupRows - 1) / groupRows;
        const int64_t groupStride = AlignUp(groupCount + 1, 64);
        const int64_t rowBegin = indexRows_ * blockId / blockNum;
        const int64_t rowEnd = indexRows_ * (blockId + 1) / blockNum;
        const int64_t rowCount = rowEnd - rowBegin;
        const int64_t rowAligned = AlignElems<int32_t>((indexRows_ + blockNum - 1) / blockNum);
        LocalTensor<int32_t> slots = scratch;
        LocalTensor<int32_t> counts = slots[rowAligned];
        LocalTensor<int32_t> corePrefix = counts[groupStride];
        LocalTensor<int32_t> temp = cursorBuf_.Get<int32_t>();
        const bool packedCounts = groupRows > 1 && blockNum * groupStride <= tileElems_ * batchRows_;
        const int64_t countGmStride = packedCounts ? groupStride : countStride_;

        Duplicate(counts, 0, static_cast<int32_t>(groupStride));
        PipeBarrier<PIPE_ALL>();
        for (int64_t row = 0; row < rowCount; ++row) {
            const int32_t dst = indexLocal.GetValue(rowBegin + row);
            const int32_t group = dst / groupRows;
            const int32_t slot = counts.GetValue(group);
            slots.SetValue(row, slot);
            counts.SetValue(group, slot + 1);
        }
        PipeBarrier<PIPE_ALL>();
        CopyOutTile(countsGm_, counts, blockId * countGmStride, groupStride);
        SyncAll<true>();

        LocalTensor<int32_t> total = counts;
        LocalTensor<int32_t> countMatrix = accBuf_.Get<int32_t>();
        if (packedCounts) {
            CopyInTile(countMatrix, countsGm_, 0, blockNum * groupStride);
            PipeBarrier<PIPE_ALL>();
        }
        Duplicate(total, 0, static_cast<int32_t>(groupStride));
        PipeBarrier<PIPE_V>();
        for (int64_t core = 0; core < blockNum; ++core) {
            LocalTensor<int32_t> coreCounts = temp;
            if (packedCounts) {
                coreCounts = countMatrix[core * groupStride];
            } else {
                CopyInTile(temp, countsGm_, core * countStride_, groupStride);
                PipeBarrier<PIPE_ALL>();
            }
            if (core == blockId) {
                Adds(corePrefix, total, 0, static_cast<int32_t>(groupStride));
                PipeBarrier<PIPE_V>();
            }
            Add(total, total, coreCounts, static_cast<int32_t>(groupStride));
            PipeBarrier<PIPE_V>();
        }

        int32_t globalPrefix = 0;
        for (int64_t group = 0; group < groupCount; ++group) {
            const int32_t count = total.GetValue(group);
            total.SetValue(group, globalPrefix);
            globalPrefix += count;
        }
        total.SetValue(groupCount, globalPrefix);
        if (blockId == 0) {
            CopyOutTile(metadataGm_, total, 0, groupCount + 1);
        }
        const int64_t zeroChunk = AlignUp((indexRows_ + blockNum - 1) / blockNum, 8);
        const int64_t zeroBegin = blockId * zeroChunk;
        const int64_t zeroEnd = zeroBegin + zeroChunk < indexRows_ ? zeroBegin + zeroChunk : indexRows_;
        for (int64_t begin = zeroBegin; begin < zeroEnd; begin += countStride_) {
            const int64_t length = begin + countStride_ < zeroEnd ? countStride_ : zeroEnd - begin;
            Duplicate(temp, 0, static_cast<int32_t>(length));
            PipeBarrier<PIPE_V>();
            CopyOutTile(metadataGm_, temp, groupCount + 1 + begin, length);
        }
        SyncAll<true>();

        SetAtomicAdd<int32_t>();
        if (packedCounts) {
            Duplicate(countMatrix, 0, static_cast<int32_t>(AlignElems<int32_t>(indexRows_)));
            PipeBarrier<PIPE_ALL>();
            for (int64_t row = 0; row < rowCount; ++row) {
                const int32_t dst = indexLocal.GetValue(rowBegin + row);
                const int32_t group = dst / groupRows;
                const int64_t position = static_cast<int64_t>(total.GetValue(group)) +
                                         corePrefix.GetValue(group) + slots.GetValue(row);
                const int32_t srcRow = static_cast<int32_t>(rowBegin + row);
                const int32_t lane = dst - group * groupRows;
                countMatrix.SetValue(position, (srcRow << 4) | lane);
            }
            SetFlag<HardEvent::S_MTE3>(sToMte3_);
            WaitFlag<HardEvent::S_MTE3>(sToMte3_);
            CopyOutTile(metadataGm_, countMatrix, groupCount + 1, indexRows_);
            SetFlag<HardEvent::MTE3_S>(mte3ToS_);
            WaitFlag<HardEvent::MTE3_S>(mte3ToS_);
        } else {
            for (int64_t row = 0; row < rowCount; ++row) {
                const int32_t dst = indexLocal.GetValue(rowBegin + row);
                const int32_t group = dst / groupRows;
                const int64_t position = static_cast<int64_t>(total.GetValue(group)) +
                                         corePrefix.GetValue(group) + slots.GetValue(row);
                temp.SetValue(0, static_cast<int32_t>(rowBegin + row));
                SetFlag<HardEvent::S_MTE3>(sToMte3_);
                WaitFlag<HardEvent::S_MTE3>(sToMte3_);
                CopyOutTile(metadataGm_, temp, groupCount + 1 + position, 1);
                SetFlag<HardEvent::MTE3_S>(mte3ToS_);
                WaitFlag<HardEvent::MTE3_S>(mte3ToS_);
            }
        }
        SetAtomicNone();
        SyncAll<true>();
        CopyInTile(scratch, metadataGm_, 0, groupCount + 1 + indexRows_);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void AccumulateHitList(LocalTensor<T> acc, int64_t sourceBase,
                                             int64_t hitBegin, int64_t hitEnd, int64_t length,
                                             LocalTensor<int32_t>& metadata)
    {
        if (hitBegin >= hitEnd) {
            return;
        }
        int64_t srcRow = metadata.GetValue(dimSize_ + 1 + hitBegin);
        LocalTensor<T> dataLocal = dataQueue_.AllocTensor<T>();
        CopyInTile(dataLocal, sourceGm_, sourceBase + srcRow * innerSize_, length);
        dataQueue_.EnQue(dataLocal);
        for (int64_t hit = hitBegin; hit < hitEnd; ++hit) {
            dataLocal = dataQueue_.DeQue<T>();
            if (hit + 1 < hitEnd) {
                srcRow = metadata.GetValue(dimSize_ + 1 + hit + 1);
                LocalTensor<T> nextLocal = dataQueue_.AllocTensor<T>();
                CopyInTile(nextLocal, sourceGm_, sourceBase + srcRow * innerSize_, length);
                dataQueue_.EnQue(nextLocal);
            }
            Add(acc, acc, dataLocal, static_cast<int32_t>(length));
            dataQueue_.FreeTensor(dataLocal);
        }
    }

    __aicore__ inline void ProcessWideBatches(LocalTensor<int32_t>& metadata)
    {
        const int64_t outerCount = totalElems_ / (dimSize_ * innerSize_);
        const int64_t dimBatchCount = (dimSize_ + batchRows_ - 1) / batchRows_;
        const int64_t taskCount = outerCount * dimBatchCount;
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        int64_t batchRotation = (dimBatchCount & 1) == 0 ? 3 : 2;
        if ((dimBatchCount - batchRotation) % 5 == 0) {
            batchRotation = (dimBatchCount & 1) == 0 ? 1 : 4;
        }
        const bool balancedWaves = batchSlots_ == 2 && dimSize_ % batchRows_ == 0 &&
                                   dimBatchCount >= 3 * blockNum;
        if (balancedWaves) {
            batchRotation = 0;
        }
        LocalTensor<T> accBase = accBuf_.Get<T>();
        const int64_t accSlotElems = tileElems_ * batchRows_;
        const bool paddedRows = (innerSize_ * sizeof(T)) % kAlignBytes != 0;
        const uint32_t rowBytes = static_cast<uint32_t>(innerSize_ * sizeof(T));
        const uint32_t localRowStride = static_cast<uint32_t>(
            (tileElems_ * sizeof(T) - AlignUp(rowBytes, kAlignBytes)) / kAlignBytes);
        DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
        int64_t iteration = 0;
        int64_t lane = blockId;
        const int64_t laneRotation = batchSlots_ == 2 ?
            (balancedWaves ? blockNum - blockNum / 4 - 1 : blockNum / 2 + blockNum / 8) : 0;
        while (true) {
            const int64_t taskId = iteration * blockNum + lane;
            if (taskId >= taskCount) {
                break;
            }
            const event_t currentEvent = static_cast<event_t>(
                batchSlots_ == 2 && (iteration & 1) != 0 ? EVENT_ID1 : EVENT_ID0);
            LocalTensor<T> acc = accBase[batchSlots_ == 2 && (iteration & 1) != 0 ? accSlotElems : 0];
            const int64_t logicalBatch = taskId % dimBatchCount;
            const int64_t outer = taskId / dimBatchCount;
            const int64_t batch = (logicalBatch + outer * batchRotation) % dimBatchCount;
            const int64_t dimBegin = batch * batchRows_;
            const int64_t rows = dimBegin + batchRows_ <= dimSize_ ? batchRows_ : dimSize_ - dimBegin;
            const int64_t outputOffset = (outer * dimSize_ + dimBegin) * innerSize_;
            const int64_t sourceBase = outer * indexRows_ * innerSize_;
            if (batchSlots_ == 1 || iteration == 0) {
                if (paddedRows) {
                    DataCopyExtParams inputCopy = {
                        static_cast<uint16_t>(rows), rowBytes, 0, localRowStride, 0};
                    DataCopyPad(acc, inputGm_[outputOffset], inputCopy, pad);
                } else {
                    CopyInTile(acc, inputGm_, outputOffset, rows * innerSize_);
                }
                SetFlag<HardEvent::MTE2_V>(currentEvent);
            }
            WaitFlag<HardEvent::MTE2_V>(currentEvent);
            const int64_t hitBegin = metadata.GetValue(batch);
            const int64_t hitEnd = metadata.GetValue(batch + 1);
            if (hitBegin < hitEnd) {
                int32_t entry0 = metadata.GetValue(dimBatchCount + 1 + hitBegin);
                int32_t entry1 = hitBegin + 1 < hitEnd ?
                    metadata.GetValue(dimBatchCount + 1 + hitBegin + 1) : entry0;
                int64_t srcRow0 = entry0 >> 4;
                int64_t srcRow1 = entry1 >> 4;
                int64_t pairRows = hitBegin + 1 < hitEnd ? 2 : 1;
                LocalTensor<T> dataLocal = dataQueue_.AllocTensor<T>();
                CopyWideSourcePair(dataLocal, sourceBase, srcRow0, srcRow1, pairRows,
                                   rowBytes, localRowStride, pad);
                dataQueue_.EnQue(dataLocal);
                for (int64_t hit = hitBegin; hit < hitEnd; hit += 2) {
                    dataLocal = dataQueue_.DeQue<T>();
                    const int32_t currentEntry0 = entry0;
                    const int32_t currentEntry1 = entry1;
                    const int64_t currentRows = pairRows;
                    if (hit + 2 < hitEnd) {
                        entry0 = metadata.GetValue(dimBatchCount + 1 + hit + 2);
                        entry1 = hit + 3 < hitEnd ?
                            metadata.GetValue(dimBatchCount + 1 + hit + 3) : entry0;
                        srcRow0 = entry0 >> 4;
                        srcRow1 = entry1 >> 4;
                        pairRows = hit + 3 < hitEnd ? 2 : 1;
                        LocalTensor<T> nextLocal = dataQueue_.AllocTensor<T>();
                        CopyWideSourcePair(nextLocal, sourceBase, srcRow0, srcRow1, pairRows,
                                           rowBytes, localRowStride, pad);
                        dataQueue_.EnQue(nextLocal);
                    }
                    const int64_t accStride = paddedRows ? tileElems_ : innerSize_;
                    const int64_t accOffset = (currentEntry0 & 15) * accStride;
                    Add(acc[accOffset], acc[accOffset], dataLocal,
                        static_cast<int32_t>(innerSize_));
                    if (currentRows == 2) {
                        const int64_t nextAccOffset = (currentEntry1 & 15) * accStride;
                        Add(acc[nextAccOffset], acc[nextAccOffset], dataLocal[tileElems_],
                            static_cast<int32_t>(innerSize_));
                    }
                    dataQueue_.FreeTensor(dataLocal);
                }
            }
            SetFlag<HardEvent::V_MTE3>(currentEvent);
            WaitFlag<HardEvent::V_MTE3>(currentEvent);
            if (paddedRows) {
                DataCopyExtParams outputCopy = {
                    static_cast<uint16_t>(rows), rowBytes, localRowStride, 0, 0};
                DataCopyPad(outputGm_[outputOffset], acc, outputCopy);
            } else {
                CopyOutTile(outputGm_, acc, outputOffset, rows * innerSize_);
            }
            SetFlag<HardEvent::MTE3_MTE2>(currentEvent);
            int64_t nextLane = lane + laneRotation;
            if (nextLane >= blockNum) {
                nextLane -= blockNum;
            }
            const int64_t nextTaskId = (iteration + 1) * blockNum + nextLane;
            if (batchSlots_ == 2 && nextTaskId < taskCount) {
                const event_t nextEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID1 : EVENT_ID0);
                LocalTensor<T> nextAcc = (iteration & 1) == 0 ? accBase[accSlotElems] : accBase;
                if (iteration > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(nextEvent);
                }
                const int64_t nextLogicalBatch = nextTaskId % dimBatchCount;
                const int64_t nextOuter = nextTaskId / dimBatchCount;
                const int64_t nextBatch =
                    (nextLogicalBatch + nextOuter * batchRotation) % dimBatchCount;
                const int64_t nextDimBegin = nextBatch * batchRows_;
                const int64_t nextRows = nextDimBegin + batchRows_ <= dimSize_ ?
                    batchRows_ : dimSize_ - nextDimBegin;
                const int64_t nextOutputOffset =
                    (nextOuter * dimSize_ + nextDimBegin) * innerSize_;
                if (paddedRows) {
                    DataCopyExtParams inputCopy = {
                        static_cast<uint16_t>(nextRows), rowBytes, 0, localRowStride, 0};
                    DataCopyPad(nextAcc, inputGm_[nextOutputOffset], inputCopy, pad);
                } else {
                    CopyInTile(nextAcc, inputGm_, nextOutputOffset, nextRows * innerSize_);
                }
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            } else if (batchSlots_ == 1) {
                WaitFlag<HardEvent::MTE3_MTE2>(currentEvent);
            }
            ++iteration;
            lane = nextLane;
        }
        if (batchSlots_ == 2) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            if (iteration > 1) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            }
        }
    }

    __aicore__ inline void CopyWideSourcePair(LocalTensor<T> local, int64_t sourceBase,
                                              int64_t srcRow0, int64_t srcRow1, int64_t rows,
                                              uint32_t rowBytes, uint32_t localRowStride,
                                              DataCopyPadExtParams<T>& pad)
    {
        const uint32_t sourceStride = rows == 2 ?
            static_cast<uint32_t>((srcRow1 - srcRow0 - 1) * rowBytes) : 0;
        DataCopyExtParams copy = {
            static_cast<uint16_t>(rows), rowBytes, sourceStride, localRowStride, 0};
        DataCopyPad(local, sourceGm_[sourceBase + srcRow0 * innerSize_], copy, pad);
    }

    __aicore__ inline void ProcessTile(int64_t taskId, int64_t innerTileCount,
                                       LocalTensor<int32_t>& indexLocal, LocalTensor<int32_t>& metadata)
    {
        const int64_t innerTile = taskId % innerTileCount;
        const int64_t rowTask = taskId / innerTileCount;
        const int64_t dimIndex = rowTask % dimSize_;
        const int64_t outer = rowTask / dimSize_;
        const int64_t innerOffset = innerTile * tileElems_;
        const int64_t length = innerOffset + tileElems_ <= innerSize_ ? tileElems_ : innerSize_ - innerOffset;
        const int64_t outputOffset = (outer * dimSize_ + dimIndex) * innerSize_ + innerOffset;
        const int64_t sourceBase = outer * indexRows_ * innerSize_ + innerOffset;
        const int64_t hitBegin = UseHitList ? metadata.GetValue(dimIndex) : 0;
        const int64_t hitEnd = UseHitList ? metadata.GetValue(dimIndex + 1) : indexRows_;

        if constexpr (!std::is_same_v<T, int8_t> && !std::is_same_v<T, bfloat16_t>) {
            ProcessPlainTile(outputOffset, sourceBase, dimIndex, hitBegin, hitEnd, length, indexLocal, metadata);
            return;
        }

        LocalTensor<T> dataLocal = dataQueue_.AllocTensor<T>();
        CopyInTile(dataLocal, inputGm_, outputOffset, length);
        dataQueue_.EnQue(dataLocal);
        dataLocal = dataQueue_.DeQue<T>();
        LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();

        if constexpr (std::is_same_v<T, int8_t>) {
            ProcessInt8(dataLocal, outputLocal, sourceBase, dimIndex, hitBegin, hitEnd, length, indexLocal, metadata);
        } else if constexpr (std::is_same_v<T, bfloat16_t>) {
            ProcessBf16(dataLocal, outputLocal, sourceBase, dimIndex, hitBegin, hitEnd, length, indexLocal, metadata);
        }

        outputQueue_.EnQue(outputLocal);
        outputLocal = outputQueue_.DeQue<T>();
        CopyOutTile(outputGm_, outputLocal, outputOffset, length);
        outputQueue_.FreeTensor(outputLocal);
    }

    __aicore__ inline void ProcessPlainTile(int64_t outputOffset, int64_t sourceBase, int64_t dimIndex,
                                            int64_t hitBegin, int64_t hitEnd, int64_t length,
                                            LocalTensor<int32_t>& indexLocal, LocalTensor<int32_t>& metadata)
    {
        LocalTensor<T> acc = accBuf_.Get<T>();
        CopyInTile(acc, inputGm_, outputOffset, length);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        if constexpr (UseHitList) {
            AccumulateHitList(acc, sourceBase, hitBegin, hitEnd, length, metadata);
        } else {
            for (int64_t hit = hitBegin; hit < hitEnd; ++hit) {
                if (indexLocal.GetValue(hit) != static_cast<int32_t>(dimIndex)) {
                    continue;
                }
                LocalTensor<T> dataLocal = dataQueue_.AllocTensor<T>();
                CopyInTile(dataLocal, sourceGm_, sourceBase + hit * innerSize_, length);
                dataQueue_.EnQue(dataLocal);
                dataLocal = dataQueue_.DeQue<T>();
                Add(acc, acc, dataLocal, static_cast<int32_t>(length));
                dataQueue_.FreeTensor(dataLocal);
            }
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        CopyOutTile(outputGm_, acc, outputOffset, length);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void ProcessBf16(LocalTensor<bfloat16_t>& dataLocal,
                                       LocalTensor<bfloat16_t>& outputLocal, int64_t sourceBase,
                                       int64_t dimIndex, int64_t hitBegin, int64_t hitEnd, int64_t length,
                                       LocalTensor<int32_t>& indexLocal, LocalTensor<int32_t>& metadata)
    {
        const int32_t count = static_cast<int32_t>(length);
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> temp = tempBuf_.Get<float>();
        Cast(acc, dataLocal, RoundMode::CAST_NONE, count);
        dataQueue_.FreeTensor(dataLocal);
        for (int64_t hit = hitBegin; hit < hitEnd; ++hit) {
            if constexpr (!UseHitList) {
                if (indexLocal.GetValue(hit) != static_cast<int32_t>(dimIndex)) {
                    continue;
                }
            }
            const int64_t srcRow = UseHitList ? metadata.GetValue(dimSize_ + 1 + hit) : hit;
            dataLocal = dataQueue_.AllocTensor<bfloat16_t>();
            CopyInTile(dataLocal, sourceGm_, sourceBase + srcRow * innerSize_, length);
            dataQueue_.EnQue(dataLocal);
            dataLocal = dataQueue_.DeQue<bfloat16_t>();
            Cast(temp, dataLocal, RoundMode::CAST_NONE, count);
            Add(acc, acc, temp, count);
            Cast(dataLocal, acc, RoundMode::CAST_RINT, count);
            Cast(acc, dataLocal, RoundMode::CAST_NONE, count);
            dataQueue_.FreeTensor(dataLocal);
        }
        Cast(outputLocal, acc, RoundMode::CAST_RINT, count);
    }

    __aicore__ inline void ProcessInt8(LocalTensor<int8_t>& dataLocal, LocalTensor<int8_t>& outputLocal,
                                       int64_t sourceBase, int64_t dimIndex, int64_t hitBegin, int64_t hitEnd,
                                       int64_t length, LocalTensor<int32_t>& indexLocal,
                                       LocalTensor<int32_t>& metadata)
    {
        const int32_t count = static_cast<int32_t>(length);
        LocalTensor<half> halfLocal = halfBuf_.Get<half>();
        LocalTensor<float> acc = accBuf_.Get<float>();
        LocalTensor<float> temp = tempBuf_.Get<float>();
        Cast(halfLocal, dataLocal, RoundMode::CAST_NONE, count);
        Cast(acc, halfLocal, RoundMode::CAST_NONE, count);
        dataQueue_.FreeTensor(dataLocal);
        for (int64_t hit = hitBegin; hit < hitEnd; ++hit) {
            if constexpr (!UseHitList) {
                if (indexLocal.GetValue(hit) != static_cast<int32_t>(dimIndex)) {
                    continue;
                }
            }
            const int64_t srcRow = UseHitList ? metadata.GetValue(dimSize_ + 1 + hit) : hit;
            dataLocal = dataQueue_.AllocTensor<int8_t>();
            CopyInTile(dataLocal, sourceGm_, sourceBase + srcRow * innerSize_, length);
            dataQueue_.EnQue(dataLocal);
            dataLocal = dataQueue_.DeQue<int8_t>();
            Cast(halfLocal, dataLocal, RoundMode::CAST_NONE, count);
            Cast(temp, halfLocal, RoundMode::CAST_NONE, count);
            Add(acc, acc, temp, count);
            dataQueue_.FreeTensor(dataLocal);
        }
        Adds(acc, acc, 128.0F, count);
        Muls(temp, acc, 1.0F / 256.0F, count);
        Cast(temp, temp, RoundMode::CAST_FLOOR, count);
        Muls(temp, temp, 256.0F, count);
        Sub(acc, acc, temp, count);
        Adds(acc, acc, -128.0F, count);
        Cast(halfLocal, acc, RoundMode::CAST_RINT, count);
        Cast(outputLocal, halfLocal, RoundMode::CAST_NONE, count);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> dataQueue_;
    TQue<QuePosition::VECIN, 1> indexQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> cursorBuf_;
    TBuf<QuePosition::VECCALC> metadataBuf_;
    TBuf<QuePosition::VECCALC> accBuf_;
    TBuf<QuePosition::VECCALC> tempBuf_;
    TBuf<QuePosition::VECCALC> halfBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<int32_t> metadataGm_;
    GlobalTensor<int32_t> countsGm_;
    int64_t totalElems_ = 0;
    int64_t dimSize_ = 0;
    int64_t innerSize_ = 0;
    int64_t indexRows_ = 0;
    int64_t tileElems_ = 0;
    int64_t metadataElems_ = 0;
    int64_t countStride_ = 0;
    int64_t batchRows_ = 1;
    int64_t batchSlots_ = 1;
    int64_t sourceBatchRows_ = 1;
    event_t sToMte3_;
    event_t mte3ToS_;
};

template <typename T>
class KernelIndexAddNarrowOuter {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                GM_ADDR workspace, const IndexAddTilingData& tiling)
    {
        totalElems_ = tiling.totalElems;
        dimSize_ = tiling.dimSize;
        innerSize_ = tiling.innerSize;
        indexRows_ = tiling.indexRows;
        outputElems_ = dimSize_ * innerSize_;
        sourceElems_ = indexRows_ * innerSize_;
        paddedInner_ = AlignElems<T>(innerSize_);
        metadataElems_ = dimSize_ + 1 + indexRows_;
        inputGm_.SetGlobalBuffer((__gm__ T*)input, totalElems_);
        outputGm_.SetGlobalBuffer((__gm__ T*)output, totalElems_);
        sourceGm_.SetGlobalBuffer((__gm__ T*)source, indexRows_ * (totalElems_ / dimSize_));
        indexGm_.SetGlobalBuffer((__gm__ int32_t*)index, indexRows_);
        if constexpr (std::is_same_v<T, float>) {
            const int64_t sourceBytes =
                indexRows_ * (totalElems_ / dimSize_) * static_cast<int64_t>(sizeof(T));
            if (sourceBytes >= kL2BypassMinBytes / 2) {
                sourceGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
        }
        metadataGm_.SetGlobalBuffer((__gm__ int32_t*)workspace, metadataElems_);
        denseWorkspaceOffset_ = AlignUp(metadataElems_ * static_cast<int64_t>(sizeof(int32_t)), 512);
        denseGm_.SetGlobalBuffer((__gm__ int32_t*)(workspace + denseWorkspaceOffset_),
                                 kNarrowWorkspaceBytes / sizeof(int32_t));
        pipe_.InitBuffer(arenaBuf_, kNarrowArenaBytes);
        mte2ToS_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_S));
        sToMte2_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE2));
        vToS_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
        sToV_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
        sToMte3_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
        mte3ToS_ = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
    }

    __aicore__ inline void Process(bool columnWave = false)
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, int32_t> || std::is_same_v<T, half>) {
            ProcessSupported(columnWave);
        }
    }

private:
    __aicore__ inline void ProcessSupported(bool columnWave)
    {
        LocalTensor<T> arena = arenaBuf_.Get<T>();
        LocalTensor<int32_t> arenaI32 = arena.template ReinterpretCast<int32_t>();
        if constexpr (std::is_same_v<T, int32_t>) {
            if (innerSize_ > 8) {
                ProcessWideInt32(arenaI32);
                return;
            }
        }
        if (!columnWave && TryTargetDense(arena, arenaI32)) {
            return;
        }
        if (!columnWave && TryParallelDense(arena, arenaI32)) {
            return;
        }
        if (!columnWave && TryDirectDense(arena, arenaI32)) {
            return;
        }
        BuildMetadata(arenaI32);

        LocalTensor<int32_t> metadata = arenaI32;
        int64_t maxHits = 0;
        for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
            const int64_t hits = metadata.GetValue(dimIndex + 1) - metadata.GetValue(dimIndex);
            maxHits = hits > maxHits ? hits : maxHits;
        }
        const int64_t arenaElems = kNarrowArenaBytes / sizeof(T);
        const int64_t arenaI32Elems = kNarrowArenaBytes / sizeof(int32_t);
        if (columnWave) {
            const int64_t paddedOutput = AlignElems<int32_t>(outputElems_);
            const int64_t paddedDim = AlignElems<int32_t>(dimSize_);
            const int64_t denseElems = maxHits * paddedOutput;
            if (denseElems <= kNarrowWorkspaceBytes / static_cast<int64_t>(sizeof(int32_t))) {
                const int64_t metadataAligned = AlignElems<int32_t>(metadataElems_);
                const int64_t vectorBuildElems =
                    metadataAligned + paddedDim + paddedDim * 8 + 2 * paddedOutput;
                if (paddedDim / 8 <= UINT8_MAX && vectorBuildElems <= arenaI32Elems) {
                    BuildDenseWaveVector(metadata, arenaI32[metadataAligned], maxHits, paddedDim, paddedOutput);
                } else {
                    BuildDenseWaves(metadata, arenaI32[metadataAligned], maxHits, paddedOutput);
                }
                ProcessTiledDense(arena, maxHits, paddedOutput);
            } else {
                ProcessOrderedRows(arena, metadata);
            }
            return;
        }
        const int64_t paddedOutput = AlignElems<T>(outputElems_);
        const int64_t denseElems = maxHits * paddedOutput;
        const int64_t cacheProcessElems = AlignElems<T>(sourceElems_) + kTileAlignElems +
                                          AlignElems<T>(outputElems_) + paddedOutput;
        const int64_t streamProcessElems = AlignElems<T>(sourceElems_) + kTileAlignElems +
                                           AlignElems<T>(outputElems_) + paddedOutput +
                                           2 * paddedOutput * sizeof(int32_t) / sizeof(T);
        if (denseElems <= kNarrowWorkspaceBytes / static_cast<int64_t>(sizeof(int32_t)) &&
            streamProcessElems <= arenaElems) {
            const int64_t metadataAligned = AlignElems<int32_t>(metadataElems_);
            const int64_t paddedDim = AlignElems<int32_t>(dimSize_);
            const int64_t denseRows = maxHits * paddedDim;
            const int64_t vectorBuildElems = metadataAligned + denseRows + paddedDim * 8 + 2 * paddedOutput;
            if (paddedDim / 8 <= UINT8_MAX && vectorBuildElems <= arenaI32Elems) {
                BuildDenseVector(metadata, arenaI32[metadataAligned], maxHits, paddedDim, paddedOutput);
            } else if (metadataAligned + denseElems <= arenaI32Elems) {
                BuildDense(metadata, arenaI32[metadataAligned], denseElems, paddedOutput);
            } else {
                BuildDenseWaves(metadata, arenaI32[metadataAligned], maxHits, paddedOutput);
            }
            const int64_t denseDataElems = denseElems * sizeof(int32_t) / sizeof(T);
            ProcessDense(arena, maxHits, paddedOutput, denseDataElems + cacheProcessElems <= arenaElems);
            return;
        }

        if constexpr (sizeof(T) != sizeof(int32_t)) {
            ProcessOrderedRows(arena, metadata);
            return;
        }

        const int64_t sparseProcessElems = AlignElems<int32_t>(metadataElems_) + AlignElems<T>(sourceElems_) +
                                           dimSize_ * paddedInner_ + 2 * kNarrowChunkElems +
                                           AlignElems<T>(innerSize_ * paddedInner_ + paddedInner_);
        if (sparseProcessElems > arenaElems) {
            ProcessOrderedRows(arena, metadata);
            return;
        }

        const int64_t metadataAligned = AlignElems<int32_t>(metadataElems_);
        const int64_t sourceAligned = AlignElems<T>(sourceElems_);
        LocalTensor<T> source = arena[metadataAligned];
        LocalTensor<T> output = source[sourceAligned];
        LocalTensor<int32_t> offsets = output[dimSize_ * paddedInner_].template ReinterpretCast<int32_t>();
        LocalTensor<T> values = offsets[kNarrowChunkElems].template ReinterpretCast<T>();
        LocalTensor<T> sums = values[kNarrowChunkElems];
        LocalTensor<int32_t> sumOffsets = sums[innerSize_ * paddedInner_].template ReinterpretCast<int32_t>();
        ArithProgression(sumOffsets, 0, static_cast<int32_t>(paddedInner_ * sizeof(T)),
                         static_cast<int32_t>(innerSize_));
        PipeBarrier<PIPE_V>();

        const int64_t outerCount = totalElems_ / outputElems_;
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        for (int64_t outer = blockId; outer < outerCount; outer += blockNum) {
            CopyInTile(source, sourceGm_, outer * sourceElems_, sourceElems_);
            CopyRowsIn(output, inputGm_, outer * outputElems_);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                const int64_t hitBegin = metadata.GetValue(dimIndex);
                const int64_t hitEnd = metadata.GetValue(dimIndex + 1);
                if (hitBegin == hitEnd) {
                    continue;
                }
                ReduceRow(source, output[dimIndex * paddedInner_], metadata, offsets,
                          values, sums, sumOffsets, hitBegin, hitEnd);
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyRowsOut(outputGm_, output, outer * outputElems_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void ProcessWideInt32(LocalTensor<int32_t>& arena)
    {
        const int64_t arenaElems = kNarrowArenaBytes / static_cast<int64_t>(sizeof(int32_t));
        const int64_t indexAligned = AlignElems<int32_t>(indexRows_);
        const int64_t availableRows = (arenaElems - indexAligned) / paddedInner_;
        const int64_t chunkRows = dimSize_ <= availableRows - kTileAlignElems
                                      ? dimSize_
                                      : availableRows - kTileAlignElems;
        const int64_t tileRows = availableRows - chunkRows;
        LocalTensor<int32_t> index = arena;
        LocalTensor<int32_t> output = index[indexAligned];
        LocalTensor<int32_t> source = output[chunkRows * paddedInner_];
        CopyInTile(index, indexGm_, 0, indexRows_);
        SetFlag<HardEvent::MTE2_S>(mte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(mte2ToS_);

        const int64_t outerCount = totalElems_ / outputElems_;
        const int64_t chunkCount = (dimSize_ + chunkRows - 1) / chunkRows;
        const int64_t taskCount = outerCount * chunkCount;
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        for (int64_t task = blockId; task < taskCount; task += blockNum) {
            const int64_t outer = task / chunkCount;
            const int64_t dimBegin = (task % chunkCount) * chunkRows;
            const int64_t dimEnd = dimBegin + chunkRows <= dimSize_ ? dimBegin + chunkRows : dimSize_;
            const int64_t outputRows = dimEnd - dimBegin;
            DataCopyExtParams outputCopy = {static_cast<uint16_t>(outputRows),
                                            static_cast<uint32_t>(innerSize_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> outputPad = {false, 0, 0, 0};
            const int64_t outputOffset = (outer * dimSize_ + dimBegin) * innerSize_;
            DataCopyPad(output, inputGm_[outputOffset], outputCopy, outputPad);
            for (int64_t tileStart = 0; tileStart < indexRows_; tileStart += tileRows) {
                const int64_t rows = tileStart + tileRows <= indexRows_ ? tileRows : indexRows_ - tileStart;
                DataCopyExtParams copy = {static_cast<uint16_t>(rows),
                                          static_cast<uint32_t>(innerSize_ * sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> pad = {false, 0, 0, 0};
                DataCopyPad(source, sourceGm_[outer * sourceElems_ + tileStart * innerSize_], copy, pad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if (chunkCount == 1) {
                    for (int64_t row = 0; row < rows; ++row) {
                        const int64_t localOffset =
                            static_cast<int64_t>(index.GetValue(tileStart + row)) * paddedInner_;
                        Add(output[localOffset], output[localOffset], source[row * paddedInner_],
                            static_cast<int32_t>(innerSize_));
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    for (int64_t row = 0; row < rows; ++row) {
                        const int64_t dst = static_cast<int64_t>(index.GetValue(tileStart + row));
                        if (dst < dimBegin || dst >= dimEnd) {
                            continue;
                        }
                        const int64_t localOffset = (dst - dimBegin) * paddedInner_;
                        Add(output[localOffset], output[localOffset], source[row * paddedInner_],
                            static_cast<int32_t>(innerSize_));
                        PipeBarrier<PIPE_V>();
                    }
                }
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(outputGm_[outputOffset], output, outputCopy);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline bool TryTargetDense(LocalTensor<T>& arena, LocalTensor<int32_t>& integer)
    {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, half>) {
            return false;
        }
        const int64_t cores = GetBlockNum(), block = GetBlockIdx();
        const int64_t scan = AlignUp(indexRows_, 64);
        const int64_t width = AlignUp((dimSize_ + cores - 1) / cores, 8);
        const int64_t countStride = AlignUp(width, 32);
        const int64_t maskBytes = AlignUp(scan / 8, 32);
        const int64_t arenaI32Elems = kNarrowArenaBytes / sizeof(int32_t);
        const int64_t paddedOutput = AlignElems<T>(outputElems_);
        const int64_t denseBase = cores * 8;
        const int64_t capacity = AlignUp(2 * ((indexRows_ + dimSize_ - 1) / dimSize_) + 4, 8);
        if (innerSize_ != 1 || cores < 8 || indexRows_ < 512 || dimSize_ > 2040 ||
            4 * scan + 2 * maskBytes / 4 + countStride > arenaI32Elems ||
            capacity * width > scan ||
            denseBase + capacity * paddedOutput > kNarrowWorkspaceBytes / sizeof(int32_t)) {
            return false;
        }
        const int64_t first = block * width;
        const int64_t last = first + width < dimSize_ ? first + width : dimSize_;
        LocalTensor<int32_t> index = integer;
        LocalTensor<int32_t> codes = index[scan];
        LocalTensor<int32_t> selected = codes[scan];
        LocalTensor<int32_t> selectedSources = selected[scan];
        LocalTensor<uint8_t> low = selectedSources[scan].template ReinterpretCast<uint8_t>();
        LocalTensor<uint8_t> high = low[maskBytes];
        LocalTensor<int32_t> counts = high[maskBytes].template ReinterpretCast<int32_t>();
        Duplicate(index, 0, static_cast<int32_t>(scan));
        Duplicate(counts, 0, static_cast<int32_t>(countStride));
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyInTile(index, indexGm_, 0, indexRows_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(codes.template ReinterpretCast<float>(), index, RoundMode::CAST_NONE,
             static_cast<int32_t>(scan));
        PipeBarrier<PIPE_V>();
        CompareScalar(low, codes.template ReinterpretCast<float>(), static_cast<float>(first),
                      CMPMODE::GE, static_cast<int32_t>(scan));
        CompareScalar(high, codes.template ReinterpretCast<float>(), static_cast<float>(last),
                      CMPMODE::LT, static_cast<int32_t>(scan));
        PipeBarrier<PIPE_V>();
        And(low.template ReinterpretCast<uint16_t>(), low.template ReinterpretCast<uint16_t>(),
            high.template ReinterpretCast<uint16_t>(), static_cast<int32_t>(maskBytes / 2));
        PipeBarrier<PIPE_V>();
        uint64_t selectedCount = 0;
        GatherMask(selected, index, low.template ReinterpretCast<uint32_t>(), true,
                   static_cast<uint32_t>(indexRows_), {1, 1, 8, 8}, selectedCount);
        BuildByteOffsets(codes, static_cast<int32_t>(sizeof(T)), static_cast<int32_t>(scan));
        PipeBarrier<PIPE_V>();
        GatherMask(selectedSources, codes, low.template ReinterpretCast<uint32_t>(), true,
                   static_cast<uint32_t>(indexRows_), {1, 1, 8, 8}, selectedCount);
        PipeBarrier<PIPE_V>();
        Duplicate(index, static_cast<int32_t>(AlignElems<T>(sourceElems_) * sizeof(T)),
                  static_cast<int32_t>(scan));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        int32_t localMax = 0;
        for (int64_t row = 0; row < static_cast<int64_t>(selectedCount); ++row) {
            const int64_t target = selected.GetValue(row) - first;
            const int32_t slot = counts.GetValue(target);
            counts.SetValue(target, slot + 1);
            const int64_t position = static_cast<int64_t>(slot) * width + target;
            if (position < scan) {
                index.SetValue(position, selectedSources.GetValue(row));
            }
            localMax = slot + 1 > localMax ? slot + 1 : localMax;
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        if constexpr (std::is_same_v<T, half>) {
            Duplicate(codes.template ReinterpretCast<float>(), static_cast<float>(localMax), 8);
        } else {
            Duplicate(codes, localMax, 8);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        CopyOutTile(denseGm_, codes, block * 8, 8);
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
        if (first < paddedOutput) {
            const int64_t columns = first + width <= paddedOutput ? width : paddedOutput - first;
            DataCopyExtParams copy = {static_cast<uint16_t>(capacity),
                static_cast<uint32_t>(columns * sizeof(int32_t)),
                static_cast<uint32_t>((width - columns) / 8),
                static_cast<uint32_t>((paddedOutput - columns) * sizeof(int32_t)), 0};
            DataCopyPad(denseGm_[denseBase + first], index, copy);
        }
        SyncAll<true>();
        CopyInTile(codes, denseGm_, 0, cores * 8);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        if constexpr (std::is_same_v<T, half>) {
            ReduceMax(index.template ReinterpretCast<float>(), codes.template ReinterpretCast<float>(),
                      selected.template ReinterpretCast<float>(), static_cast<int32_t>(cores * 8), false);
        } else {
            for (int64_t active = cores; active > 1; active = (active + 1) / 2) {
                const int64_t left = (active + 1) / 2;
                Max(codes, codes, codes[left * 8], static_cast<int32_t>((active - left) * 8));
                PipeBarrier<PIPE_V>();
            }
        }
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        const int64_t maxHits = std::is_same_v<T, half>
            ? static_cast<int64_t>(index.template ReinterpretCast<float>().GetValue(0)) : codes.GetValue(0);
        const int64_t denseElems = maxHits * paddedOutput;
        if (maxHits > capacity) {
            return false;
        }
        const int64_t denseDataElems = denseElems * sizeof(int32_t) / sizeof(T);
        const int64_t processElems = AlignElems<T>(sourceElems_) + kTileAlignElems +
                                     AlignElems<T>(outputElems_) + paddedOutput;
        ProcessDense(arena, maxHits, paddedOutput,
                     denseDataElems + processElems <= kNarrowArenaBytes / sizeof(T), denseBase);
        return true;
    }

    __aicore__ inline bool TryParallelDense(LocalTensor<T>& arena, LocalTensor<int32_t>& arenaI32)
    {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, half> && !std::is_same_v<T, int32_t>) {
            return false;
        }
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t outerCount = totalElems_ / outputElems_;
        if (innerSize_ > 8 || blockNum <= 1 ||
            indexRows_ < blockNum * kMinParallelRowsPerCore || dimSize_ > UINT8_MAX * 8) {
            return false;
        }

        const int64_t rowBegin = indexRows_ * blockId / blockNum;
        const int64_t rowEnd = indexRows_ * (blockId + 1) / blockNum;
        const int64_t rowCount = rowEnd - rowBegin;
        const int64_t rowAligned = AlignElems<int32_t>(rowCount);
        const int64_t countStride = AlignUp(dimSize_, 64);
        LocalTensor<int32_t> index = arenaI32;
        LocalTensor<int32_t> slots = index[rowAligned];
        LocalTensor<int32_t> counts = slots[rowAligned];
        LocalTensor<int32_t> prefix = counts[countStride];
        LocalTensor<int32_t> total = prefix[countStride];
        LocalTensor<int32_t> temp = total[countStride];

        CopyInTile(index, indexGm_, rowBegin, rowCount);
        Duplicate(counts, 0, static_cast<int32_t>(countStride));
        SetFlag<HardEvent::MTE2_S>(mte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(mte2ToS_);
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        for (int64_t row = 0; row < rowCount; ++row) {
            const int32_t dst = index.GetValue(row);
            const int32_t slot = counts.GetValue(dst);
            slots.SetValue(row, slot);
            counts.SetValue(dst, slot + 1);
        }
        SetFlag<HardEvent::S_MTE3>(sToMte3_);
        WaitFlag<HardEvent::S_MTE3>(sToMte3_);
        CopyOutTile(denseGm_, counts, blockId * countStride, countStride);
        SyncAll<true>();

        const int64_t matrixMergeElems =
            2 * rowAligned + (blockNum + 2) * countStride;
        if ((std::is_same_v<T, float> || std::is_same_v<T, half>) &&
            matrixMergeElems <= kNarrowArenaBytes / static_cast<int64_t>(sizeof(int32_t))) {
            Duplicate(prefix, 0, static_cast<int32_t>(countStride));
            PipeBarrier<PIPE_V>();
            CopyInTile(total, denseGm_, 0, blockNum * countStride);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (int64_t core = 0; core < blockId; ++core) {
                Add(prefix, prefix, total[core * countStride], static_cast<int32_t>(countStride));
                PipeBarrier<PIPE_V>();
            }
            int64_t activeCores = blockNum;
            while (activeCores > 1) {
                const int64_t leftCores = (activeCores + 1) / 2;
                const int64_t pairedCores = activeCores - leftCores;
                Add(total, total, total[leftCores * countStride],
                    static_cast<int32_t>(pairedCores * countStride));
                PipeBarrier<PIPE_V>();
                activeCores = leftCores;
            }
        } else {
            Duplicate(total, 0, static_cast<int32_t>(countStride));
            PipeBarrier<PIPE_V>();
            for (int64_t core = 0; core < blockNum; ++core) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                CopyInTile(temp, denseGm_, core * countStride, countStride);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                if (core == blockId) {
                    Adds(prefix, total, 0, static_cast<int32_t>(countStride));
                    PipeBarrier<PIPE_V>();
                }
                Add(total, total, temp, static_cast<int32_t>(countStride));
                PipeBarrier<PIPE_V>();
            }
        }

        int32_t activeBlocks = static_cast<int32_t>(countStride / 8);
        while (activeBlocks > 1) {
            const int32_t leftBlocks = (activeBlocks + 1) / 2;
            Max(total, total, total[leftBlocks * 8], (activeBlocks - leftBlocks) * 8);
            PipeBarrier<PIPE_V>();
            activeBlocks = leftBlocks;
        }
        for (int32_t width = 4; width > 0; width /= 2) {
            ArithProgression(counts, width * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(sizeof(int32_t)), width);
            PipeBarrier<PIPE_V>();
            Gather(temp, total, counts.ReinterpretCast<uint32_t>(), 0, width);
            PipeBarrier<PIPE_V>();
            Max(total, total, temp, width);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        const int32_t maxHits = total.GetValue(0);

        const int64_t paddedOutput = AlignElems<T>(outputElems_);
        const int64_t scratchElems = blockNum * countStride + indexRows_;
        const int64_t denseBase = AlignElems<int32_t>(scratchElems);
        const int64_t denseElems = static_cast<int64_t>(maxHits) * paddedOutput;
        const int64_t workspaceElems = kNarrowWorkspaceBytes / static_cast<int64_t>(sizeof(int32_t));
        if (denseElems <= 0 || denseBase + denseElems > workspaceElems) {
            return false;
        }
        const int32_t sentinel = static_cast<int32_t>(AlignElems<T>(sourceElems_) * sizeof(T));
        const int64_t partialBase = denseBase + denseElems;
        const int64_t arenaI32Elems = kNarrowArenaBytes / sizeof(int32_t);
        const int64_t mergeChunk = AlignUp((denseElems + blockNum - 1) / blockNum, 8);
        if (innerSize_ == 1 && partialBase + blockNum * denseElems <= workspaceElems &&
            2 * rowAligned + 2 * countStride + denseElems <= arenaI32Elems &&
            blockNum * mergeChunk <= arenaI32Elems) {
            LocalTensor<int32_t> localDense = total;
            Duplicate(localDense, sentinel, static_cast<int32_t>(denseElems));
            SetFlag<HardEvent::V_S>(vToS_);
            WaitFlag<HardEvent::V_S>(vToS_);
            for (int64_t row = 0; row < rowCount; ++row) {
                const int32_t target = index.GetValue(row);
                const int64_t wave = static_cast<int64_t>(prefix.GetValue(target)) + slots.GetValue(row);
                localDense.SetValue(wave * paddedOutput + target,
                                   static_cast<int32_t>((rowBegin + row) * sizeof(T)));
            }
            SetFlag<HardEvent::S_MTE3>(sToMte3_);
            WaitFlag<HardEvent::S_MTE3>(sToMte3_);
            CopyOutTile(denseGm_, localDense, partialBase + blockId * denseElems, denseElems);
            SyncAll<true>();

            const int64_t begin = blockId * mergeChunk;
            const int64_t length = begin + mergeChunk <= denseElems ? mergeChunk : denseElems - begin;
            if (length > 0) {
                LocalTensor<int32_t> matrix = arenaI32;
                DataCopyExtParams copy = {static_cast<uint16_t>(blockNum),
                    static_cast<uint32_t>(length * sizeof(int32_t)),
                    static_cast<uint32_t>((denseElems - length) * sizeof(int32_t)), 0, 0};
                DataCopyPadExtParams<int32_t> pad = {false, 0, 0, 0};
                DataCopyPad(matrix, denseGm_[partialBase + begin], copy, pad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                for (int64_t active = blockNum; active > 1; active = (active + 1) / 2) {
                    const int64_t left = (active + 1) / 2;
                    Min(matrix, matrix, matrix[left * length],
                        static_cast<int32_t>((active - left) * length));
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyOutTile(denseGm_, matrix, denseBase + begin, length);
            }
            SyncAll<true>();
            const int64_t denseDataElems = denseElems * sizeof(int32_t) / sizeof(T);
            const int64_t processElems = AlignElems<T>(sourceElems_) + kTileAlignElems +
                                         AlignElems<T>(outputElems_) + paddedOutput;
            ProcessDense(arena, maxHits, paddedOutput,
                         denseDataElems + processElems <= kNarrowArenaBytes / sizeof(T), denseBase);
            return true;
        }
        const int64_t initChunk = AlignUp((denseElems + blockNum - 1) / blockNum, 8);
        const int64_t initBegin = blockId * initChunk;
        const int64_t initEnd = initBegin + initChunk < denseElems ? initBegin + initChunk : denseElems;
        for (int64_t begin = initBegin; begin < initEnd; begin += countStride) {
            const int64_t length = begin + countStride < initEnd ? countStride : initEnd - begin;
            Duplicate(total, sentinel, static_cast<int32_t>(length));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyOutTile(denseGm_, total, denseBase + begin, length);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        SyncAll<true>();

        if (innerSize_ == 1) {
            for (int64_t row = 0; row < rowCount; ++row) {
                const int32_t dst = index.GetValue(row);
                const int64_t wave = static_cast<int64_t>(prefix.GetValue(dst)) + slots.GetValue(row);
                counts.SetValue(0, static_cast<int32_t>((rowBegin + row) * sizeof(T)));
                SetFlag<HardEvent::S_MTE3>(sToMte3_);
                WaitFlag<HardEvent::S_MTE3>(sToMte3_);
                CopyOutTile(denseGm_, counts, denseBase + wave * paddedOutput + dst, 1);
                SetFlag<HardEvent::MTE3_S>(mte3ToS_);
                WaitFlag<HardEvent::MTE3_S>(mte3ToS_);
            }
        } else {
            for (int64_t row = 0; row < rowCount; ++row) {
                const int32_t dst = index.GetValue(row);
                const int64_t wave = static_cast<int64_t>(prefix.GetValue(dst)) + slots.GetValue(row);
                for (int64_t column = 0; column < innerSize_; ++column) {
                    counts.SetValue(column,
                                    static_cast<int32_t>(((rowBegin + row) * innerSize_ + column) * sizeof(T)));
                }
                SetFlag<HardEvent::S_MTE3>(sToMte3_);
                WaitFlag<HardEvent::S_MTE3>(sToMte3_);
                CopyOutTile(denseGm_, counts, denseBase + wave * paddedOutput + dst * innerSize_, innerSize_);
                SetFlag<HardEvent::MTE3_S>(mte3ToS_);
                WaitFlag<HardEvent::MTE3_S>(mte3ToS_);
            }
        }
        SyncAll<true>();

        const int64_t arenaElems = kNarrowArenaBytes / sizeof(T);
        const int64_t denseDataElems = denseElems * sizeof(int32_t) / sizeof(T);
        const int64_t cacheProcessElems = AlignElems<T>(sourceElems_) + kTileAlignElems +
                                          AlignElems<T>(outputElems_) + paddedOutput;
        ProcessDense(arena, maxHits, paddedOutput, denseDataElems + cacheProcessElems <= arenaElems, denseBase);
        return true;
    }

    __aicore__ inline bool TryDirectDense(LocalTensor<T>& arena, LocalTensor<int32_t>& arenaI32)
    {
        const int64_t arenaElems = kNarrowArenaBytes / sizeof(T);
        const int64_t paddedDim = AlignElems<int32_t>(dimSize_);
        const int64_t paddedOutput = AlignElems<T>(outputElems_);
        const int64_t cacheProcessElems = AlignElems<T>(sourceElems_) + kTileAlignElems +
                                          AlignElems<T>(outputElems_) + paddedOutput;
        const int64_t streamProcessElems = AlignElems<T>(sourceElems_) + kTileAlignElems +
                                           AlignElems<T>(outputElems_) + paddedOutput +
                                           2 * paddedOutput * sizeof(int32_t) / sizeof(T);
        if (paddedDim / 8 > UINT8_MAX || streamProcessElems > arenaElems) {
            return false;
        }

        const int64_t indexAligned = AlignElems<int32_t>(indexRows_);
        const int64_t cursorAligned = AlignElems<int32_t>(dimSize_);
        const int64_t fixedElems = indexAligned + cursorAligned + paddedDim * 8 + 2 * paddedOutput;
        const int64_t arenaI32Elems = kNarrowArenaBytes / sizeof(int32_t);
        const int64_t localCapacity = (arenaI32Elems - fixedElems) / paddedDim;
        const int64_t workspaceCapacity =
            (kNarrowWorkspaceBytes / static_cast<int64_t>(sizeof(int32_t))) / paddedOutput;
        const int64_t rowCapacity = localCapacity < workspaceCapacity ? localCapacity : workspaceCapacity;
        if (rowCapacity <= 0) {
            return false;
        }

        LocalTensor<int32_t> index = arenaI32;
        LocalTensor<int32_t> cursor = index[indexAligned];
        LocalTensor<int32_t> rowTable = cursor[cursorAligned];
        if (GetBlockIdx() == 0) {
            CopyInTile(index, indexGm_, 0, indexRows_);
            Duplicate(cursor, 0, static_cast<int32_t>(dimSize_));
            Duplicate(rowTable, static_cast<int32_t>(indexRows_), static_cast<int32_t>(rowCapacity * paddedDim));
            SetFlag<HardEvent::MTE2_S>(mte2ToS_);
            WaitFlag<HardEvent::MTE2_S>(mte2ToS_);
            SetFlag<HardEvent::V_S>(vToS_);
            WaitFlag<HardEvent::V_S>(vToS_);

            int32_t maxHits = 0;
            int32_t overflow = 0;
            for (int64_t row = 0; row < indexRows_; ++row) {
                const int32_t dst = index.GetValue(row);
                const int32_t slot = cursor.GetValue(dst);
                cursor.SetValue(dst, slot + 1);
                maxHits = slot + 1 > maxHits ? slot + 1 : maxHits;
                if (slot < rowCapacity) {
                    rowTable.SetValue(static_cast<int64_t>(slot) * paddedDim + dst, static_cast<int32_t>(row));
                } else {
                    overflow = 1;
                }
            }
            index.SetValue(0, maxHits);
            index.SetValue(1, overflow);
            SetFlag<HardEvent::S_MTE3>(sToMte3_);
            WaitFlag<HardEvent::S_MTE3>(sToMte3_);
            CopyOutTile(metadataGm_, index, 0, 8);
        }
        SyncAll<true>();
        CopyInTile(arenaI32, metadataGm_, 0, 8);
        SetFlag<HardEvent::MTE2_S>(mte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(mte2ToS_);
        const int64_t maxHits = arenaI32.GetValue(0);
        if (arenaI32.GetValue(1) != 0) {
            return false;
        }

        LocalTensor<int32_t> expanded = rowTable[rowCapacity * paddedDim];
        LocalTensor<int32_t> waveOffsets = expanded[paddedDim * 8];
        LocalTensor<int32_t> compactOffsets = waveOffsets[paddedOutput];
        WriteDenseElements(rowTable, expanded, waveOffsets, compactOffsets,
                           maxHits, paddedDim, paddedOutput);
        const int64_t denseElems = maxHits * paddedOutput;
        const int64_t denseDataElems = denseElems * sizeof(int32_t) / sizeof(T);
        ProcessDense(arena, maxHits, paddedOutput, denseDataElems + cacheProcessElems <= arenaElems);
        return true;
    }

    __aicore__ inline void ProcessOrderedRows(LocalTensor<T>& arena, LocalTensor<int32_t>& metadata)
    {
        const int64_t metadataAligned =
            AlignElems<int32_t>(metadataElems_) * static_cast<int64_t>(sizeof(int32_t)) / sizeof(T);
        LocalTensor<T> output = arena[metadataAligned];
        LocalTensor<T> sourceRow = output[dimSize_ * paddedInner_];
        const int64_t outerCount = totalElems_ / outputElems_;
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        for (int64_t outer = blockId; outer < outerCount; outer += blockNum) {
            CopyRowsIn(output, inputGm_, outer * outputElems_);
            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                const int64_t begin = metadata.GetValue(dimIndex);
                const int64_t end = metadata.GetValue(dimIndex + 1);
                for (int64_t hit = begin; hit < end; ++hit) {
                    const int64_t row = metadata.GetValue(dimSize_ + 1 + hit);
                    CopyInTile(sourceRow, sourceGm_, outer * sourceElems_ + row * innerSize_, innerSize_);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Add(output[dimIndex * paddedInner_], output[dimIndex * paddedInner_], sourceRow,
                        static_cast<int32_t>(innerSize_));
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyRowsOut(outputGm_, output, outer * outputElems_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void BuildMetadata(LocalTensor<int32_t>& arena)
    {
        if (GetBlockIdx() == 0) {
            const int64_t indexAligned = AlignElems<int32_t>(indexRows_);
            const int64_t cursorAligned = AlignElems<int32_t>(dimSize_);
            LocalTensor<int32_t> index = arena;
            LocalTensor<int32_t> cursor = index[indexAligned];
            LocalTensor<int32_t> metadata = cursor[cursorAligned];
            CopyInTile(index, indexGm_, 0, indexRows_);
            SetFlag<HardEvent::MTE2_S>(mte2ToS_);
            WaitFlag<HardEvent::MTE2_S>(mte2ToS_);

            Duplicate(cursor, 0, static_cast<int32_t>(dimSize_));
            SetFlag<HardEvent::V_S>(vToS_);
            WaitFlag<HardEvent::V_S>(vToS_);
            for (int64_t row = 0; row < indexRows_; ++row) {
                const int32_t dst = index.GetValue(row);
                cursor.SetValue(dst, cursor.GetValue(dst) + 1);
            }
            int32_t prefix = 0;
            metadata.SetValue(0, 0);
            for (int64_t dst = 0; dst < dimSize_; ++dst) {
                prefix += cursor.GetValue(dst);
                metadata.SetValue(dst + 1, prefix);
            }

            Duplicate(cursor, 0, static_cast<int32_t>(dimSize_));
            SetFlag<HardEvent::V_S>(vToS_);
            WaitFlag<HardEvent::V_S>(vToS_);
            for (int64_t row = 0; row < indexRows_; ++row) {
                const int32_t dst = index.GetValue(row);
                const int32_t slot = metadata.GetValue(dst) + cursor.GetValue(dst);
                metadata.SetValue(dimSize_ + 1 + slot, static_cast<int32_t>(row));
                cursor.SetValue(dst, cursor.GetValue(dst) + 1);
            }
            SetFlag<HardEvent::S_MTE3>(sToMte3_);
            WaitFlag<HardEvent::S_MTE3>(sToMte3_);
            CopyOutTile(metadataGm_, metadata, 0, metadataElems_);
        }
        SyncAll<true>();
        CopyInTile(arena, metadataGm_, 0, metadataElems_);
        SetFlag<HardEvent::MTE2_S>(mte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(mte2ToS_);
    }

    __aicore__ inline void BuildDense(LocalTensor<int32_t>& metadata, LocalTensor<int32_t> dense,
                                      int64_t denseElems, int64_t paddedOutput)
    {
        if (GetBlockIdx() == 0) {
            const int32_t sentinel = static_cast<int32_t>(AlignElems<T>(sourceElems_) * sizeof(T));
            Duplicate(dense, sentinel, static_cast<int32_t>(denseElems));
            SetFlag<HardEvent::V_S>(vToS_);
            WaitFlag<HardEvent::V_S>(vToS_);
            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                const int64_t begin = metadata.GetValue(dimIndex);
                const int64_t end = metadata.GetValue(dimIndex + 1);
                for (int64_t hit = begin; hit < end; ++hit) {
                    const int64_t row = metadata.GetValue(dimSize_ + 1 + hit);
                    for (int64_t column = 0; column < innerSize_; ++column) {
                        dense.SetValue((hit - begin) * paddedOutput + dimIndex * innerSize_ + column,
                                       static_cast<int32_t>((row * innerSize_ + column) * sizeof(T)));
                    }
                }
            }
            SetFlag<HardEvent::S_MTE3>(sToMte3_);
            WaitFlag<HardEvent::S_MTE3>(sToMte3_);
            CopyOutTile(denseGm_, dense, 0, denseElems);
        }
        SyncAll<true>();
    }

    __aicore__ inline void BuildDenseWaves(LocalTensor<int32_t>& metadata, LocalTensor<int32_t> dense,
                                           int64_t maxHits, int64_t paddedOutput)
    {
        if (GetBlockIdx() == 0) {
            const int32_t sentinel = static_cast<int32_t>(AlignElems<T>(sourceElems_) * sizeof(T));
            for (int64_t wave = 0; wave < maxHits; ++wave) {
                Duplicate(dense, sentinel, static_cast<int32_t>(paddedOutput));
                SetFlag<HardEvent::V_S>(vToS_);
                WaitFlag<HardEvent::V_S>(vToS_);
                for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                    const int64_t hit = metadata.GetValue(dimIndex) + wave;
                    if (hit >= metadata.GetValue(dimIndex + 1)) {
                        continue;
                    }
                    const int64_t row = metadata.GetValue(dimSize_ + 1 + hit);
                    for (int64_t column = 0; column < innerSize_; ++column) {
                        dense.SetValue(dimIndex * innerSize_ + column,
                                       static_cast<int32_t>((row * innerSize_ + column) * sizeof(T)));
                    }
                }
                SetFlag<HardEvent::S_MTE3>(sToMte3_);
                WaitFlag<HardEvent::S_MTE3>(sToMte3_);
                CopyOutTile(denseGm_, dense, wave * paddedOutput, paddedOutput);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        SyncAll<true>();
    }

    __aicore__ inline void BuildDenseWaveVector(LocalTensor<int32_t>& metadata, LocalTensor<int32_t> rowIds,
                                                int64_t maxHits, int64_t paddedDim, int64_t paddedOutput)
    {
        LocalTensor<int32_t> expanded = rowIds[paddedDim];
        LocalTensor<int32_t> waveOffsets = expanded[paddedDim * 8];
        LocalTensor<int32_t> compactOffsets = waveOffsets[paddedOutput];
        if (GetBlockIdx() == 0) {
            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                for (int64_t column = 0; column < innerSize_; ++column) {
                    compactOffsets.SetValue(dimIndex * innerSize_ + column,
                                            static_cast<int32_t>((dimIndex * 8 + column) * sizeof(int32_t)));
                }
            }
            const uint8_t repeats = static_cast<uint8_t>(paddedDim / 8);
            const BrcbRepeatParams brcbParams(1, 8);
            const UnaryRepeatParams addsParams(1, 1, 8, 8);
            for (int64_t wave = 0; wave < maxHits; ++wave) {
                Duplicate(rowIds, static_cast<int32_t>(indexRows_), static_cast<int32_t>(paddedDim));
                SetFlag<HardEvent::V_S>(vToS_);
                WaitFlag<HardEvent::V_S>(vToS_);
                for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                    const int64_t hit = metadata.GetValue(dimIndex) + wave;
                    if (hit < metadata.GetValue(dimIndex + 1)) {
                        rowIds.SetValue(dimIndex, metadata.GetValue(dimSize_ + 1 + hit));
                    }
                }
                SetFlag<HardEvent::S_V>(sToV_);
                WaitFlag<HardEvent::S_V>(sToV_);
                Brcb(expanded, rowIds, repeats, brcbParams);
                PipeBarrier<PIPE_V>();
                Muls(expanded, expanded, static_cast<int32_t>(innerSize_ * sizeof(T)),
                     static_cast<int32_t>(paddedDim * 8));
                PipeBarrier<PIPE_V>();
                for (int64_t column = 1; column < innerSize_; ++column) {
                    uint64_t mask[2] = {0x0101010101010101ULL << column, 0};
                    Adds(expanded, expanded, static_cast<int32_t>(column * sizeof(T)), mask, repeats, addsParams);
                    PipeBarrier<PIPE_V>();
                }
                Gather(waveOffsets, expanded, compactOffsets.template ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(outputElems_));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyOutTile(denseGm_, waveOffsets, wave * paddedOutput, paddedOutput);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        SyncAll<true>();
    }

    __aicore__ inline void BuildDenseVector(LocalTensor<int32_t>& metadata, LocalTensor<int32_t> rowTable,
                                            int64_t maxHits, int64_t paddedDim, int64_t paddedOutput)
    {
        const int64_t denseRows = maxHits * paddedDim;
        LocalTensor<int32_t> expanded = rowTable[denseRows];
        LocalTensor<int32_t> waveOffsets = expanded[paddedDim * 8];
        LocalTensor<int32_t> compactOffsets = waveOffsets[paddedOutput];
        if (GetBlockIdx() == 0) {
            Duplicate(rowTable, static_cast<int32_t>(indexRows_), static_cast<int32_t>(denseRows));
            SetFlag<HardEvent::V_S>(vToS_);
            WaitFlag<HardEvent::V_S>(vToS_);
            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                const int64_t begin = metadata.GetValue(dimIndex);
                const int64_t end = metadata.GetValue(dimIndex + 1);
                for (int64_t hit = begin; hit < end; ++hit) {
                    rowTable.SetValue((hit - begin) * paddedDim + dimIndex,
                                      metadata.GetValue(dimSize_ + 1 + hit));
                }
            }
        }
        WriteDenseElements(rowTable, expanded, waveOffsets, compactOffsets,
                           maxHits, paddedDim, paddedOutput);
    }

    __aicore__ inline void WriteDenseElements(LocalTensor<int32_t>& rowTable, LocalTensor<int32_t>& expanded,
                                              LocalTensor<int32_t>& waveOffsets,
                                              LocalTensor<int32_t>& compactOffsets,
                                              int64_t maxHits, int64_t paddedDim, int64_t paddedOutput)
    {
        if (GetBlockIdx() == 0) {
            for (int64_t dimIndex = 0; dimIndex < dimSize_; ++dimIndex) {
                for (int64_t column = 0; column < innerSize_; ++column) {
                    compactOffsets.SetValue(dimIndex * innerSize_ + column,
                                            static_cast<int32_t>((dimIndex * 8 + column) * sizeof(int32_t)));
                }
            }
            SetFlag<HardEvent::S_V>(sToV_);
            WaitFlag<HardEvent::S_V>(sToV_);

            const uint8_t repeats = static_cast<uint8_t>(paddedDim / 8);
            const BrcbRepeatParams brcbParams(1, 8);
            const UnaryRepeatParams addsParams(1, 1, 8, 8);
            for (int64_t wave = 0; wave < maxHits; ++wave) {
                Brcb(expanded, rowTable[wave * paddedDim], repeats, brcbParams);
                PipeBarrier<PIPE_V>();
                Muls(expanded, expanded, static_cast<int32_t>(innerSize_ * sizeof(T)),
                     static_cast<int32_t>(paddedDim * 8));
                PipeBarrier<PIPE_V>();
                for (int64_t column = 1; column < innerSize_; ++column) {
                    uint64_t mask[2] = {0x0101010101010101ULL << column, 0};
                    Adds(expanded, expanded, static_cast<int32_t>(column * sizeof(T)), mask, repeats, addsParams);
                    PipeBarrier<PIPE_V>();
                }
                Gather(waveOffsets, expanded, compactOffsets.template ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(outputElems_));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                CopyOutTile(denseGm_, waveOffsets, wave * paddedOutput, paddedOutput);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
        SyncAll<true>();
    }

    __aicore__ inline void ProcessTiledDense(LocalTensor<T>& arena, int64_t maxHits, int64_t paddedOutput)
    {
        constexpr int64_t padElems = kAlignBytes / sizeof(T);
        const int64_t arenaElems = kNarrowArenaBytes / sizeof(T);
        const int64_t maxTileRows = (arenaElems - 3 * paddedOutput - 2 * padElems) / innerSize_;
        const int64_t tileRows = (maxTileRows / kTileAlignElems) * kTileAlignElems;
        const int64_t sourceSpan = AlignElems<T>(tileRows * innerSize_ + 2 * padElems);
        LocalTensor<T> source = arena;
        LocalTensor<T> sourceData = source[padElems];
        LocalTensor<T> output = source[sourceSpan];
        LocalTensor<int32_t> offsets = output[paddedOutput].template ReinterpretCast<int32_t>();
        LocalTensor<T> values = offsets[paddedOutput].template ReinterpretCast<T>();
        Duplicate(source, static_cast<T>(0), padElems);
        PipeBarrier<PIPE_V>();

        const int64_t outerCount = totalElems_ / outputElems_;
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        for (int64_t outer = blockId; outer < outerCount; outer += blockNum) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyInTile(output, inputGm_, outer * outputElems_, outputElems_);
            for (int64_t tileStart = 0; tileStart < indexRows_; tileStart += tileRows) {
                const int64_t tileLen = tileStart + tileRows <= indexRows_ ? tileRows : indexRows_ - tileStart;
                const int64_t tileElems = tileLen * innerSize_;
                const int64_t sentinel = padElems + AlignElems<T>(tileElems);
                const int64_t tailBlock = padElems + (tileElems / padElems) * padElems;
                Duplicate(source[tailBlock], static_cast<T>(0),
                          static_cast<int32_t>(sentinel - tailBlock + padElems));
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                DataCopyExtParams sourceCopy = {
                    1, static_cast<uint32_t>(tileElems * static_cast<int64_t>(sizeof(T))), 0, 0, 0};
                DataCopyPadExtParams<T> sourcePad = {
                    true, 0, static_cast<uint8_t>(AlignElems<T>(tileElems) - tileElems), static_cast<T>(0)};
                DataCopyPad(sourceData, sourceGm_[outer * sourceElems_ + tileStart * innerSize_],
                            sourceCopy, sourcePad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

                const int32_t shift = static_cast<int32_t>(
                    (padElems - tileStart * innerSize_) * static_cast<int64_t>(sizeof(T)));
                const int32_t upper = static_cast<int32_t>(sentinel * static_cast<int64_t>(sizeof(T)));
                for (int64_t wave = 0; wave < maxHits; ++wave) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    CopyInTile(offsets, denseGm_, wave * paddedOutput, paddedOutput);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Adds(offsets, offsets, shift, static_cast<int32_t>(paddedOutput));
                    PipeBarrier<PIPE_V>();
                    Maxs(offsets, offsets, static_cast<int32_t>(0), static_cast<int32_t>(paddedOutput));
                    PipeBarrier<PIPE_V>();
                    Mins(offsets, offsets, upper, static_cast<int32_t>(paddedOutput));
                    PipeBarrier<PIPE_V>();
                    Gather(values, source, offsets.template ReinterpretCast<uint32_t>(), 0,
                           static_cast<uint32_t>(outputElems_));
                    PipeBarrier<PIPE_V>();
                    Add(output, output, values, static_cast<int32_t>(outputElems_));
                    PipeBarrier<PIPE_V>();
                }
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyOutTile(outputGm_, output, outer * outputElems_, outputElems_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void ProcessDense(LocalTensor<T>& arena, int64_t maxHits, int64_t paddedOutput,
                                        bool cacheDense, int64_t denseBase = 0)
    {
        const int64_t denseElems = maxHits * paddedOutput;
        const int64_t denseDataElems = denseElems * sizeof(int32_t) / sizeof(T);
        const int64_t sourceAligned = AlignElems<T>(sourceElems_);
        const int64_t outputAligned = AlignElems<T>(outputElems_);
        if (innerSize_ == 1 && cacheDense) {
            const bool preferCrossOuter = sourceAligned <= 4 * outputAligned;
            if (preferCrossOuter) {
                if (ProcessDenseInner1SparseTail<3>(arena, maxHits, paddedOutput, denseBase,
                                                    sourceAligned, outputAligned)) {
                    return;
                }
                if (ProcessDenseInner1SparseTail<2>(arena, maxHits, paddedOutput, denseBase,
                                                    sourceAligned, outputAligned)) {
                    return;
                }
            }
            if constexpr (std::is_same_v<T, float>) {
                const int64_t outerCount = totalElems_ / outputElems_;
                const int64_t rowsPerCore = outerCount / GetBlockNum();
                const int64_t gatherBytes =
                    maxHits * 2 * paddedOutput * static_cast<int64_t>(sizeof(T));
                if (preferCrossOuter && gatherBytes <= kSingleGatherMaxBytes && rowsPerCore >= 16 &&
                    ProcessDenseInner1Batched<2>(arena, maxHits, paddedOutput, denseBase,
                                                 denseElems, sourceAligned, outputAligned)) {
                    return;
                }
            }
            if (ProcessCompactTail(arena, maxHits, paddedOutput, denseBase, sourceAligned, outputAligned)) {
                return;
            }
            if (ProcessDenseInner1Tree(arena, maxHits, paddedOutput, denseBase,
                                       sourceAligned, outputAligned)) {
                return;
            }
            if constexpr (std::is_same_v<T, half>) {
                if (ProcessDenseHalfBlocks(arena, maxHits, paddedOutput, denseBase,
                                           sourceAligned, outputAligned)) {
                    return;
                }
                if (ProcessDenseInner1Batched<8>(arena, maxHits, paddedOutput, denseBase,
                                                 denseElems, sourceAligned, outputAligned)) {
                    return;
                }
                if (ProcessDenseInner1OrderedTail<5>(arena, maxHits, paddedOutput, denseBase,
                                                     sourceAligned, outputAligned)) {
                    return;
                }
                if (ProcessDenseInner1Batched<5>(arena, maxHits, paddedOutput, denseBase,
                                                 denseElems, sourceAligned, outputAligned)) {
                    return;
                }
            }
            if (ProcessDenseInner1Batched<4>(arena, maxHits, paddedOutput, denseBase,
                                             denseElems, sourceAligned, outputAligned)) {
                return;
            }
            if (ProcessDenseInner1Batched<3>(arena, maxHits, paddedOutput, denseBase,
                                             denseElems, sourceAligned, outputAligned)) {
                return;
            }
            if (ProcessDenseInner1Batched<2>(arena, maxHits, paddedOutput, denseBase,
                                             denseElems, sourceAligned, outputAligned)) {
                return;
            }
        }
        const int64_t sourceSpan = sourceAligned + kTileAlignElems;
        const int64_t arenaElems = kNarrowArenaBytes / sizeof(T);
        if (cacheDense && denseDataElems + 2 * sourceSpan + 2 * outputAligned + outputElems_ <= arenaElems) {
            ProcessDenseDoubleBuffered(arena, maxHits, paddedOutput, denseBase,
                                       denseElems, denseDataElems, sourceAligned, outputAligned);
            return;
        }
        LocalTensor<int32_t> dense = arena.template ReinterpretCast<int32_t>();
        LocalTensor<T> source = arena[cacheDense ? denseDataElems : 0];
        LocalTensor<T> output = source[sourceAligned + kTileAlignElems];
        LocalTensor<int32_t> offsets0 = output[outputAligned].template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> offsets1 = offsets0[paddedOutput];
        LocalTensor<T> values = cacheDense ? output[outputAligned] :
                                             offsets1[paddedOutput].template ReinterpretCast<T>();
        if (cacheDense) {
            CopyInTile(dense, denseGm_, denseBase, denseElems);
        }
        Duplicate(source[sourceAligned], static_cast<T>(0), kTileAlignElems);
        PipeBarrier<PIPE_V>();

        const int64_t outerCount = totalElems_ / outputElems_;
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        for (int64_t outer = blockId; outer < outerCount; outer += blockNum) {
            CopyInTile(source, sourceGm_, outer * sourceElems_, sourceElems_);
            CopyInTile(output, inputGm_, outer * outputElems_, outputElems_);
            if (!cacheDense) {
                CopyInTile(offsets0, denseGm_, denseBase, outputElems_);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            if (cacheDense) {
                for (int64_t wave = 0; wave < maxHits; ++wave) {
                    Gather(values, source, dense[wave * paddedOutput].template ReinterpretCast<uint32_t>(), 0,
                           static_cast<uint32_t>(outputElems_));
                    PipeBarrier<PIPE_V>();
                    Add(output, output, values, static_cast<int32_t>(outputElems_));
                    PipeBarrier<PIPE_V>();
                }
            } else {
                for (int64_t wave = 0; wave < maxHits; ++wave) {
                    const event_t currentEvent = static_cast<event_t>((wave & 1) == 0 ? EVENT_ID0 : EVENT_ID1);
                    LocalTensor<int32_t> waveOffsets = (wave & 1) == 0 ? offsets0 : offsets1;
                    if (wave > 0) {
                        WaitFlag<HardEvent::MTE2_V>(currentEvent);
                    }
                    if (wave + 1 < maxHits) {
                        const event_t nextEvent = static_cast<event_t>((wave & 1) == 0 ? EVENT_ID1 : EVENT_ID0);
                        LocalTensor<int32_t> nextOffsets = (wave & 1) == 0 ? offsets1 : offsets0;
                        if (wave > 0) {
                            WaitFlag<HardEvent::V_MTE2>(nextEvent);
                        }
                        CopyInTile(nextOffsets, denseGm_, denseBase + (wave + 1) * paddedOutput, outputElems_);
                        SetFlag<HardEvent::MTE2_V>(nextEvent);
                    }
                    Gather(values, source, waveOffsets.template ReinterpretCast<uint32_t>(), 0,
                           static_cast<uint32_t>(outputElems_));
                    PipeBarrier<PIPE_V>();
                    Add(output, output, values, static_cast<int32_t>(outputElems_));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(currentEvent);
                }
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                if (maxHits > 1) {
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                }
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            CopyOutTile(outputGm_, output, outer * outputElems_, outputElems_);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    template <int64_t BatchRows>
    __aicore__ inline bool ProcessDenseInner1SparseTail(LocalTensor<T>& arena, int64_t maxHits,
                                                        int64_t paddedOutput, int64_t denseBase,
                                                        int64_t sourceAligned, int64_t outputAligned)
    {
        if constexpr (!std::is_same_v<T, float>) {
            return false;
        }
        constexpr int64_t minDenseWaves = 5;
        constexpr int64_t maxDenseWaves = 12;
        const int64_t outerCount = totalElems_ / outputElems_;
        if (outerCount / GetBlockNum() < BatchRows * 8 ||
            maxHits <= minDenseWaves || maxHits > 64) {
            return false;
        }

        const int64_t sourceSpan = sourceAligned + kTileAlignElems;
        const int64_t arenaElems = kNarrowArenaBytes / static_cast<int64_t>(sizeof(T));
        const int64_t bufferedSourceElems = 2 * BatchRows * sourceSpan;
        if (2 * bufferedSourceElems > arenaElems) {
            return false;
        }
        const int64_t prefixLimit = maxHits - 1 < maxDenseWaves ? maxHits - 1 : maxDenseWaves;
        const int64_t scanSpan = AlignUp(paddedOutput, 64);
        const int64_t detectMaskI32 = AlignUp(scanSpan / 8, kAlignBytes) / sizeof(int32_t);
        if (3 * scanSpan + detectMaskI32 > kNarrowArenaBytes / static_cast<int64_t>(sizeof(int32_t))) {
            return false;
        }
        LocalTensor<int32_t> scan = arena.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> dimSequence = scan[scanSpan];
        LocalTensor<int32_t> targetOffsets = dimSequence[scanSpan];
        LocalTensor<uint8_t> activeMask = targetOffsets[scanSpan].template ReinterpretCast<uint8_t>();

        CreateVecIndex(dimSequence, static_cast<int32_t>(0), static_cast<uint32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Muls(dimSequence, dimSequence, static_cast<int32_t>(sizeof(T)), static_cast<int32_t>(scanSpan));
        Duplicate(scan, static_cast<int32_t>(sourceAligned * sizeof(T)), static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

        const int32_t sentinel = static_cast<int32_t>(sourceAligned * sizeof(T));
        int64_t denseWaves = 0;
        int64_t sparseTargets = 0;
        for (int64_t candidate = minDenseWaves; candidate <= prefixLimit; ++candidate) {
            CopyInTile(scan, denseGm_, denseBase + candidate * paddedOutput, paddedOutput);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            CompareScalar(activeMask, scan, sentinel, CMPMODE::EQ, static_cast<int32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Not(activeMask.template ReinterpretCast<uint16_t>(), activeMask.template ReinterpretCast<uint16_t>(),
                static_cast<int32_t>(scanSpan / 16));
            PipeBarrier<PIPE_V>();
            uint64_t gathered = 0;
            GatherMaskParams params = {1, 1, 8, 8};
            GatherMask(targetOffsets, dimSequence, activeMask.template ReinterpretCast<uint32_t>(),
                       true, static_cast<uint32_t>(scanSpan), params, gathered);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_S>(vToS_);
            WaitFlag<HardEvent::V_S>(vToS_);
            const int64_t activeTargets = static_cast<int64_t>(gathered);
            const int64_t candidateStride = AlignElems<int32_t>(activeTargets);
            const int64_t candidateSparseElems =
                (maxHits - candidate) * BatchRows * candidateStride;
            const int64_t candidateExpandedElems = candidate * BatchRows * paddedOutput;
            const int64_t candidateMetadataElems =
                candidateExpandedElems + BatchRows * paddedOutput +
                2 * AlignElems<int32_t>(candidateSparseElems) +
                BatchRows * paddedOutput;
            const int64_t candidateRequiredElems = candidateMetadataElems + 2 * BatchRows * sourceSpan +
                2 * BatchRows * outputAligned;
            if (activeTargets > 0 &&
                candidateRequiredElems <= kNarrowArenaBytes / static_cast<int64_t>(sizeof(T)) &&
                candidateSparseElems * static_cast<int64_t>(sizeof(T)) <= kSingleGatherMaxBytes) {
                denseWaves = candidate;
                sparseTargets = activeTargets;
                break;
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }
        if (denseWaves == 0) {
            return false;
        }
        if (maxHits <= 2 * minDenseWaves && denseWaves == minDenseWaves &&
            sparseTargets * 4 >= dimSize_) {
            return false;
        }
        const int64_t overflowWaves = maxHits - denseWaves;
        const int64_t expandedElems = denseWaves * BatchRows * paddedOutput;
        LocalTensor<int32_t> expanded = arena.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> dimOffsets = expanded[expandedElems];
        LocalTensor<int32_t> waveOffsets = dimOffsets[paddedOutput];

        const int64_t sparseStride = AlignElems<int32_t>(sparseTargets);
        const int64_t sparseWaveElems = BatchRows * sparseStride;
        const int64_t sparseDataElems = overflowWaves * sparseWaveElems;
        const int64_t sparseDataAligned = AlignElems<int32_t>(sparseDataElems);
        const int64_t sparseSumElems = BatchRows * sparseStride;
        LocalTensor<int32_t> sparseOffsets = dimOffsets[BatchRows * paddedOutput];
        LocalTensor<T> sparseWork = sparseOffsets[sparseDataAligned].template ReinterpretCast<T>();
        LocalTensor<int32_t> sparseSumOffsets =
            sparseWork[sparseDataAligned].template ReinterpretCast<int32_t>();
        LocalTensor<T> source0 = sparseSumOffsets[BatchRows * paddedOutput].template ReinterpretCast<T>();
        LocalTensor<T> source1 = source0[BatchRows * sourceSpan];
        LocalTensor<T> output0 = source1[BatchRows * sourceSpan];
        LocalTensor<T> output1 = output0[BatchRows * outputAligned];
        LocalTensor<T> values = dimOffsets.template ReinterpretCast<T>();

        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        Adds(dimOffsets, targetOffsets, 0, static_cast<int32_t>(sparseTargets));
        Adds(waveOffsets, scan, 0, static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        for (int64_t wave = 0; wave < denseWaves; ++wave) {
            LocalTensor<int32_t> denseRow = expanded[wave * BatchRows * paddedOutput];
            CopyInTile(denseRow, denseGm_, denseBase + wave * paddedOutput, paddedOutput);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (int64_t wave = 0; wave < denseWaves; ++wave) {
            for (int64_t row = 1; row < BatchRows; ++row) {
                Adds(expanded[(wave * BatchRows + row) * paddedOutput],
                     expanded[wave * BatchRows * paddedOutput],
                     static_cast<int32_t>(row * sourceSpan * sizeof(T)), static_cast<int32_t>(paddedOutput));
                PipeBarrier<PIPE_V>();
            }
        }

        Duplicate(sparseOffsets, sentinel, static_cast<int32_t>(sparseDataElems));
        PipeBarrier<PIPE_V>();
        for (int64_t wave = 0; wave < overflowWaves; ++wave) {
            if (wave > 0) {
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                CopyInTile(waveOffsets, denseGm_,
                           denseBase + (denseWaves + wave) * paddedOutput, paddedOutput);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            }
            LocalTensor<int32_t> rowOffsets = sparseOffsets[wave * sparseWaveElems];
            Gather(rowOffsets, waveOffsets, dimOffsets.template ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(sparseTargets));
            PipeBarrier<PIPE_V>();
            for (int64_t row = 1; row < BatchRows; ++row) {
                Adds(rowOffsets[row * sparseStride], rowOffsets,
                     static_cast<int32_t>(row * sourceSpan * sizeof(T)),
                     static_cast<int32_t>(sparseTargets));
                PipeBarrier<PIPE_V>();
            }
        }

        const int32_t sparseSentinel = static_cast<int32_t>(sparseSumElems * sizeof(T));
        Duplicate(sparseSumOffsets, sparseSentinel, static_cast<int32_t>(paddedOutput));
        Duplicate(source0, static_cast<T>(0), static_cast<int32_t>(BatchRows * sourceSpan));
        Duplicate(source1, static_cast<T>(0), static_cast<int32_t>(BatchRows * sourceSpan));
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        int64_t target = 0;
        for (; target + 3 < sparseTargets; target += 4) {
            const int64_t dim0 = dimOffsets.GetValue(target) / sizeof(T);
            const int64_t dim1 = dimOffsets.GetValue(target + 1) / sizeof(T);
            const int64_t dim2 = dimOffsets.GetValue(target + 2) / sizeof(T);
            const int64_t dim3 = dimOffsets.GetValue(target + 3) / sizeof(T);
            sparseSumOffsets.SetValue(dim0, static_cast<int32_t>(target * sizeof(T)));
            sparseSumOffsets.SetValue(dim1, static_cast<int32_t>((target + 1) * sizeof(T)));
            sparseSumOffsets.SetValue(dim2, static_cast<int32_t>((target + 2) * sizeof(T)));
            sparseSumOffsets.SetValue(dim3, static_cast<int32_t>((target + 3) * sizeof(T)));
        }
        for (; target < sparseTargets; ++target) {
            const int64_t dim = dimOffsets.GetValue(target) / sizeof(T);
            sparseSumOffsets.SetValue(dim, static_cast<int32_t>(target * sizeof(T)));
        }
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        for (int64_t row = 1; row < BatchRows; ++row) {
            Adds(sparseSumOffsets[row * paddedOutput], sparseSumOffsets,
                 static_cast<int32_t>(row * sparseStride * sizeof(T)), static_cast<int32_t>(paddedOutput));
            PipeBarrier<PIPE_V>();
            Mins(sparseSumOffsets[row * paddedOutput], sparseSumOffsets[row * paddedOutput], sparseSentinel,
                 static_cast<int32_t>(paddedOutput));
            PipeBarrier<PIPE_V>();
        }

        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t outerBegin = outerCount * blockId / blockNum;
        const int64_t outerEnd = outerCount * (blockId + 1) / blockNum;
        if (outerBegin >= outerEnd) {
            return true;
        }
        const int64_t firstRows = outerBegin + BatchRows <= outerEnd ? BatchRows : outerEnd - outerBegin;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyDenseBatchIn(source0, output0, sourceSpan, outputAligned, outerBegin, firstRows);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        int64_t iteration = 0;
        for (int64_t outer = outerBegin; outer < outerEnd; outer += BatchRows, ++iteration) {
            const event_t currentEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID0 : EVENT_ID1);
            LocalTensor<T> source = (iteration & 1) == 0 ? source0 : source1;
            LocalTensor<T> output = (iteration & 1) == 0 ? output0 : output1;
            const int64_t rows = outer + BatchRows <= outerEnd ? BatchRows : outerEnd - outer;
            WaitFlag<HardEvent::MTE2_V>(currentEvent);

            const int64_t nextOuter = outer + BatchRows;
            if (nextOuter < outerEnd) {
                const event_t nextEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID1 : EVENT_ID0);
                LocalTensor<T> nextSource = (iteration & 1) == 0 ? source1 : source0;
                LocalTensor<T> nextOutput = (iteration & 1) == 0 ? output1 : output0;
                if (iteration > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(nextEvent);
                }
                const int64_t nextRows = nextOuter + BatchRows <= outerEnd ? BatchRows : outerEnd - nextOuter;
                CopyDenseBatchIn(nextSource, nextOutput, sourceSpan, outputAligned, nextOuter, nextRows);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }

            for (int64_t wave = 0; wave < denseWaves; ++wave) {
                Gather(values, source,
                       expanded[wave * BatchRows * paddedOutput].template ReinterpretCast<uint32_t>(),
                       0, static_cast<uint32_t>(rows * paddedOutput));
                PipeBarrier<PIPE_V>();
                Add(output, output, values, static_cast<int32_t>(rows * paddedOutput));
                PipeBarrier<PIPE_V>();
            }
            if (rows == BatchRows) {
                Gather(sparseWork, source, sparseOffsets.template ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(sparseDataElems));
                PipeBarrier<PIPE_V>();
                int64_t activeWaves = overflowWaves;
                while (activeWaves > 1) {
                    const int64_t leftWaves = (activeWaves + 1) / 2;
                    const int64_t pairedWaves = activeWaves - leftWaves;
                    Add(sparseWork, sparseWork, sparseWork[leftWaves * sparseWaveElems],
                        static_cast<int32_t>(pairedWaves * sparseWaveElems));
                    PipeBarrier<PIPE_V>();
                    activeWaves = leftWaves;
                }
            } else {
                const int64_t tempOffset = rows * sparseStride;
                for (int64_t row = 0; row < rows; ++row) {
                    LocalTensor<T> rowSum = sparseWork[row * sparseStride];
                    Gather(rowSum, source,
                           sparseOffsets[row * sparseStride].template ReinterpretCast<uint32_t>(), 0,
                           static_cast<uint32_t>(sparseTargets));
                    PipeBarrier<PIPE_V>();
                    for (int64_t wave = 1; wave < overflowWaves; ++wave) {
                        Gather(sparseWork[tempOffset], source,
                               sparseOffsets[wave * sparseWaveElems + row * sparseStride]
                                   .template ReinterpretCast<uint32_t>(),
                               0, static_cast<uint32_t>(sparseTargets));
                        PipeBarrier<PIPE_V>();
                        Add(rowSum, rowSum, sparseWork[tempOffset], static_cast<int32_t>(sparseTargets));
                        PipeBarrier<PIPE_V>();
                    }
                }
            }
            Duplicate(sparseWork[sparseSumElems], static_cast<T>(0), kAlignBytes / sizeof(T));
            PipeBarrier<PIPE_V>();
            Gather(values, sparseWork, sparseSumOffsets.template ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(rows * paddedOutput));
            PipeBarrier<PIPE_V>();
            Add(output, output, values, static_cast<int32_t>(rows * paddedOutput));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(currentEvent);
            WaitFlag<HardEvent::V_MTE3>(currentEvent);
            CopyDenseBatchOut(output, outputAligned, outer, rows);
            SetFlag<HardEvent::MTE3_MTE2>(currentEvent);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        if (iteration > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        return true;
    }

    template <int64_t BatchRows>
    __aicore__ inline bool ProcessDenseInner1OrderedTail(LocalTensor<T>& arena, int64_t maxHits,
                                                         int64_t paddedOutput, int64_t denseBase,
                                                         int64_t sourceAligned, int64_t outputAligned)
    {
        if constexpr (!std::is_same_v<T, half> || BatchRows != 5) {
            return false;
        }
        constexpr int64_t denseWaves = 5;
        const int64_t overflowWaves = maxHits - denseWaves;
        const int64_t outerCount = totalElems_ / outputElems_;
        if (innerSize_ != 1 || outerCount < GetBlockNum() || overflowWaves < 4 || maxHits > 64) {
            return false;
        }

        const int64_t scanSpan = AlignUp(paddedOutput, 64);
        const int64_t arenaI32Elems = kNarrowArenaBytes / sizeof(int32_t);
        LocalTensor<int32_t> arenaI32 = arena.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> scan = arenaI32;
        LocalTensor<int32_t> dimSequence = scan[scanSpan];
        LocalTensor<int32_t> targetIndices = dimSequence[scanSpan];
        LocalTensor<uint8_t> activeMask = targetIndices[scanSpan].template ReinterpretCast<uint8_t>();
        const int64_t detectMaskI32 = AlignUp(scanSpan / 8, kAlignBytes) / sizeof(int32_t);

        CopyInTile(scan, denseGm_, denseBase + denseWaves * paddedOutput, paddedOutput);
        CreateVecIndex(dimSequence, static_cast<int32_t>(0), static_cast<uint32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        const int32_t sentinel = static_cast<int32_t>(sourceAligned * sizeof(T));
        CompareScalar(activeMask, scan, sentinel, CMPMODE::EQ, static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Not(activeMask.template ReinterpretCast<uint16_t>(), activeMask.template ReinterpretCast<uint16_t>(),
            static_cast<int32_t>(scanSpan / 16));
        PipeBarrier<PIPE_V>();
        uint64_t gathered = 0;
        GatherMaskParams gatherMaskParams = {1, 1, 8, 8};
        GatherMask(targetIndices, dimSequence, activeMask.template ReinterpretCast<uint32_t>(),
                   true, static_cast<uint32_t>(scanSpan), gatherMaskParams, gathered);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        const int64_t sparseTargets = static_cast<int64_t>(gathered);
        if (sparseTargets <= 0 || sparseTargets * 8 > paddedOutput * 3) {
            return false;
        }

        const int64_t sparseStride = AlignElems<int32_t>(sparseTargets);
        const int64_t expandedElems = denseWaves * BatchRows * paddedOutput;
        const int64_t targetOffsetElems = BatchRows * sparseStride;
        const int64_t sparseOffsetElems = overflowWaves * targetOffsetElems;
        const int64_t mapElems = BatchRows * paddedOutput;
        const int64_t fixedI32 = 3 * scanSpan + detectMaskI32;
        const int64_t scheduleI32 = fixedI32 + expandedElems + targetOffsetElems +
                                    sparseOffsetElems + mapElems;
        const int64_t sourceSpan = sourceAligned + kTileAlignElems;
        const int64_t maskRowBytes = AlignUp(paddedOutput / 8, kAlignBytes);
        const int64_t maskBytes = BatchRows * maskRowBytes;
        const int64_t maskElems = maskBytes / sizeof(T);
        constexpr int64_t sparseSentinelBytes = 4096;
        constexpr int64_t sparseSentinelElems = sparseSentinelBytes / sizeof(T);
        constexpr int64_t sparseValueElems = sparseSentinelElems + kTileAlignElems;
        const int64_t requiredElems = scheduleI32 * sizeof(int32_t) / sizeof(T) +
            2 * BatchRows * sourceSpan + 2 * BatchRows * outputAligned +
            2 * sparseValueElems + mapElems + maskElems;
        if (scheduleI32 > arenaI32Elems || requiredElems > kNarrowArenaBytes / sizeof(T)) {
            return false;
        }

        LocalTensor<int32_t> expanded = arenaI32[fixedI32];
        LocalTensor<int32_t> targetOffsets = expanded[expandedElems];
        LocalTensor<int32_t> sparseOffsets = targetOffsets[targetOffsetElems];
        LocalTensor<int32_t> outputMap = sparseOffsets[sparseOffsetElems];
        LocalTensor<T> source0 = arena[scheduleI32 * sizeof(int32_t) / sizeof(T)];
        LocalTensor<T> source1 = source0[BatchRows * sourceSpan];
        LocalTensor<T> output0 = source1[BatchRows * sourceSpan];
        LocalTensor<T> output1 = output0[BatchRows * outputAligned];
        LocalTensor<T> sparseState = output1[BatchRows * outputAligned];
        LocalTensor<T> sparseValues = sparseState[sparseValueElems];
        LocalTensor<T> replacement = sparseValues[sparseValueElems];
        LocalTensor<uint8_t> outputMask = replacement[mapElems].template ReinterpretCast<uint8_t>();

        for (int64_t wave = 0; wave < denseWaves; ++wave) {
            LocalTensor<int32_t> expandedWave = expanded[wave * BatchRows * paddedOutput];
            CopyInTile(expandedWave, denseGm_, denseBase + wave * paddedOutput, paddedOutput);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        for (int64_t wave = 0; wave < denseWaves; ++wave) {
            for (int64_t row = 1; row < BatchRows; ++row) {
                Adds(expanded[(wave * BatchRows + row) * paddedOutput],
                     expanded[wave * BatchRows * paddedOutput],
                     static_cast<int32_t>(row * sourceSpan * sizeof(T)),
                     static_cast<int32_t>(paddedOutput));
                PipeBarrier<PIPE_V>();
            }
        }

        Duplicate(targetOffsets, 0, static_cast<int32_t>(targetOffsetElems));
        Muls(targetOffsets, targetIndices, static_cast<int32_t>(sizeof(T)),
             static_cast<int32_t>(sparseTargets));
        PipeBarrier<PIPE_V>();
        for (int64_t row = 1; row < BatchRows; ++row) {
            Adds(targetOffsets[row * sparseStride], targetOffsets,
                 static_cast<int32_t>(row * outputAligned * sizeof(T)),
                 static_cast<int32_t>(sparseTargets));
            PipeBarrier<PIPE_V>();
        }

        Muls(dimSequence, targetIndices, static_cast<int32_t>(sizeof(int32_t)),
             static_cast<int32_t>(sparseTargets));
        Duplicate(sparseOffsets, sentinel, static_cast<int32_t>(sparseOffsetElems));
        PipeBarrier<PIPE_V>();
        Duplicate(outputMap, static_cast<int32_t>(0), static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Muls(outputMap, scan, static_cast<int32_t>(-1), static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        Adds(outputMap, outputMap, sentinel, static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        Mins(outputMap, outputMap, static_cast<int32_t>(1), static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        for (int64_t wave = 0; wave < overflowWaves; ++wave) {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            CopyInTile(scan, denseGm_, denseBase + (denseWaves + wave) * paddedOutput,
                       paddedOutput);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            LocalTensor<int32_t> waveOffsets = sparseOffsets[wave * targetOffsetElems];
            Gather(waveOffsets, scan, dimSequence.template ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(sparseTargets));
            PipeBarrier<PIPE_V>();
            for (int64_t row = 1; row < BatchRows; ++row) {
                Adds(waveOffsets[row * sparseStride], waveOffsets,
                     static_cast<int32_t>(row * sourceSpan * sizeof(T)),
                     static_cast<int32_t>(sparseTargets));
                PipeBarrier<PIPE_V>();
            }
        }

        LocalTensor<int32_t> prefix = scan;
        LocalTensor<int32_t> shifted = dimSequence;
        LocalTensor<int32_t> shiftOffsets = targetIndices;
        Adds(prefix, outputMap, static_cast<int32_t>(0), static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        for (int64_t shift = 1; shift < scanSpan; shift <<= 1) {
            CreateVecIndex(shifted, static_cast<int32_t>(0), static_cast<uint32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Adds(shiftOffsets, shifted, static_cast<int32_t>(-shift),
                 static_cast<int32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Maxs(shiftOffsets, shiftOffsets, static_cast<int32_t>(0),
                 static_cast<int32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Muls(shiftOffsets, shiftOffsets, static_cast<int32_t>(sizeof(int32_t)),
                 static_cast<int32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Gather(shifted, prefix, shiftOffsets.template ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            CreateVecIndex(outputMap, static_cast<int32_t>(0), static_cast<uint32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Adds(outputMap, outputMap, static_cast<int32_t>(1 - shift),
                 static_cast<int32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Maxs(outputMap, outputMap, static_cast<int32_t>(0), static_cast<int32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Mins(outputMap, outputMap, static_cast<int32_t>(1), static_cast<int32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Mul(shifted, shifted, outputMap, static_cast<int32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
            Add(prefix, prefix, shifted, static_cast<int32_t>(scanSpan));
            PipeBarrier<PIPE_V>();
        }
        CreateVecIndex(shifted, static_cast<int32_t>(0), static_cast<uint32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Adds(shiftOffsets, shifted, static_cast<int32_t>(-1), static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Maxs(shiftOffsets, shiftOffsets, static_cast<int32_t>(0), static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Muls(shiftOffsets, shiftOffsets, static_cast<int32_t>(sizeof(int32_t)),
             static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Gather(shifted, prefix, shiftOffsets.template ReinterpretCast<uint32_t>(), 0,
               static_cast<uint32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        CreateVecIndex(outputMap, static_cast<int32_t>(0), static_cast<uint32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Mins(outputMap, outputMap, static_cast<int32_t>(1), static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Mul(shifted, shifted, outputMap, static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        Sub(shifted, prefix, shifted, static_cast<int32_t>(scanSpan));
        PipeBarrier<PIPE_V>();
        for (int64_t row = 0; row < BatchRows; ++row) {
            CompareScalar(outputMask[row * maskRowBytes], shifted, static_cast<int32_t>(0),
                          CMPMODE::GT, static_cast<int32_t>(paddedOutput));
            PipeBarrier<PIPE_V>();
            LocalTensor<uint16_t> rowMask =
                outputMask[row * maskRowBytes].template ReinterpretCast<uint16_t>();
            Not(rowMask, rowMask, static_cast<int32_t>(maskRowBytes / sizeof(uint16_t)));
            PipeBarrier<PIPE_V>();
        }
        constexpr int32_t mapSentinel = sparseSentinelBytes;
        Adds(outputMap, prefix, static_cast<int32_t>(-1), static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        Muls(outputMap, outputMap, static_cast<int32_t>(sizeof(T)),
             static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        Mul(outputMap, outputMap, shifted, static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        Muls(shiftOffsets, shifted, static_cast<int32_t>(-mapSentinel),
             static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        Adds(shiftOffsets, shiftOffsets, mapSentinel, static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        Add(outputMap, outputMap, shiftOffsets, static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        for (int64_t row = 1; row < BatchRows; ++row) {
            Adds(outputMap[row * paddedOutput], outputMap,
                 static_cast<int32_t>(row * sparseStride * sizeof(T)),
                 static_cast<int32_t>(paddedOutput));
            PipeBarrier<PIPE_V>();
            Mins(outputMap[row * paddedOutput], outputMap[row * paddedOutput], mapSentinel,
                 static_cast<int32_t>(paddedOutput));
            PipeBarrier<PIPE_V>();
        }
        Duplicate(source0, static_cast<T>(0), BatchRows * sourceSpan);
        Duplicate(source1, static_cast<T>(0), BatchRows * sourceSpan);
        PipeBarrier<PIPE_V>();

        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t outerBegin = outerCount * blockId / blockNum;
        const int64_t outerEnd = outerCount * (blockId + 1) / blockNum;
        if (outerBegin >= outerEnd) {
            return true;
        }
        const int64_t firstRows = outerBegin + BatchRows <= outerEnd ? BatchRows : outerEnd - outerBegin;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyDenseBatchIn(source0, output0, sourceSpan, outputAligned, outerBegin, firstRows);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        int64_t iteration = 0;
        for (int64_t outer = outerBegin; outer < outerEnd; outer += BatchRows, ++iteration) {
            const event_t currentEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID0 : EVENT_ID1);
            LocalTensor<T> source = (iteration & 1) == 0 ? source0 : source1;
            LocalTensor<T> output = (iteration & 1) == 0 ? output0 : output1;
            const int64_t rows = outer + BatchRows <= outerEnd ? BatchRows : outerEnd - outer;
            WaitFlag<HardEvent::MTE2_V>(currentEvent);

            const int64_t nextOuter = outer + BatchRows;
            if (nextOuter < outerEnd) {
                const event_t nextEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID1 : EVENT_ID0);
                LocalTensor<T> nextSource = (iteration & 1) == 0 ? source1 : source0;
                LocalTensor<T> nextOutput = (iteration & 1) == 0 ? output1 : output0;
                if (iteration > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(nextEvent);
                }
                const int64_t nextRows = nextOuter + BatchRows <= outerEnd ? BatchRows : outerEnd - nextOuter;
                CopyDenseBatchIn(nextSource, nextOutput, sourceSpan, outputAligned, nextOuter, nextRows);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }

            for (int64_t wave = 0; wave < denseWaves; ++wave) {
                Gather(replacement, source,
                       expanded[wave * BatchRows * paddedOutput].template ReinterpretCast<uint32_t>(),
                       0, static_cast<uint32_t>(rows * paddedOutput));
                PipeBarrier<PIPE_V>();
                Add(output, output, replacement, static_cast<int32_t>(rows * paddedOutput));
                PipeBarrier<PIPE_V>();
            }
            Gather(sparseState, output, targetOffsets.template ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(rows * sparseStride));
            PipeBarrier<PIPE_V>();
            for (int64_t wave = 0; wave < overflowWaves; ++wave) {
                Gather(sparseValues, source,
                       sparseOffsets[wave * targetOffsetElems].template ReinterpretCast<uint32_t>(),
                       0, static_cast<uint32_t>(rows * sparseStride));
                PipeBarrier<PIPE_V>();
                Add(sparseState, sparseState, sparseValues,
                    static_cast<int32_t>(rows * sparseStride));
                PipeBarrier<PIPE_V>();
            }
            Duplicate(sparseState[sparseSentinelElems], static_cast<T>(0), kTileAlignElems);
            PipeBarrier<PIPE_V>();
            Gather(replacement, sparseState, outputMap.template ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(rows * paddedOutput));
            PipeBarrier<PIPE_V>();
            for (int64_t row = 0; row < rows; ++row) {
                Select(output[row * paddedOutput], outputMask[row * maskRowBytes],
                       replacement[row * paddedOutput], output[row * paddedOutput],
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<int32_t>(paddedOutput));
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(currentEvent);
            WaitFlag<HardEvent::V_MTE3>(currentEvent);
            CopyDenseBatchOut(output, outputAligned, outer, rows);
            SetFlag<HardEvent::MTE3_MTE2>(currentEvent);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        if (iteration > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        return true;
    }

    __aicore__ inline void TransposeHalf16(LocalTensor<T> destination, LocalTensor<T> source,
                                           int64_t columns, bool reverse)
    {
        const uint64_t srcBase = reinterpret_cast<uint64_t>(source.GetPhyAddr());
        const uint64_t dstBase = reinterpret_cast<uint64_t>(destination.GetPhyAddr());
        uint64_t src[16], dst[16];
        for (int64_t row = 0; row < 16; ++row) {
            src[row] = srcBase + row * (reverse ? 16 : columns) * sizeof(T);
            dst[row] = dstBase + row * (reverse ? columns : 16) * sizeof(T);
        }
        const uint8_t repeats = static_cast<uint8_t>(columns / 16);
        TransDataTo5HD<uint16_t>(dst, src, {false, false, repeats,
            static_cast<uint16_t>(repeats == 1 ? 0 : (reverse ? 1 : 16)),
            static_cast<uint16_t>(repeats == 1 ? 0 : (reverse ? 16 : 1))});
    }

    __aicore__ inline bool ProcessDenseHalfBlocks(LocalTensor<T>& arena, int64_t maxHits,
                                                  int64_t paddedOutput, int64_t denseBase,
                                                  int64_t sourceAligned, int64_t outputAligned)
    {
        constexpr int64_t batch = 16;
        const int64_t outerCount = totalElems_ / outputElems_;
        const int64_t denseElems = maxHits * paddedOutput;
        const int64_t denseDataElems = denseElems * sizeof(int32_t) / sizeof(T);
        const int64_t transposedElems = batch * (sourceAligned + 16);
        const int64_t waveElems = batch * paddedOutput;
        const int64_t fixedElems = denseDataElems + batch * sourceAligned +
                                   3 * batch * outputAligned + transposedElems;
        const int64_t arenaElems = kNarrowArenaBytes / sizeof(T);
        const int64_t padding = outputAligned - outputElems_;
        const int64_t maskBytes = padding ? AlignUp(batch * outputAligned / 8, 32) : 0;
        const int64_t waveCapacity = (arenaElems - fixedElems - maskBytes / sizeof(T)) / waveElems;
        const int64_t repeatCapacity = 255 * 8 / paddedOutput;
        if (outerCount / GetBlockNum() < batch || sourceAligned > 255 * 16 ||
            outputAligned > 255 * 16 || fixedElems >= arenaElems ||
            waveCapacity < 1 || repeatCapacity < 1) {
            return false;
        }
        const int64_t gatherWaves = waveCapacity < repeatCapacity ? waveCapacity : repeatCapacity;
        sourceGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        LocalTensor<int32_t> dense = arena.template ReinterpretCast<int32_t>();
        LocalTensor<T> source0 = arena[denseDataElems];
        LocalTensor<T> output0 = source0[batch * sourceAligned];
        LocalTensor<T> output1 = output0[batch * outputAligned];
        LocalTensor<T> transposed = output1[batch * outputAligned];
        LocalTensor<T> state = transposed[transposedElems];
        LocalTensor<T> values = state[batch * outputAligned];
        LocalTensor<uint32_t> outputMask = arena[arenaElems - maskBytes / sizeof(T)].template ReinterpretCast<uint32_t>();
        CopyInTile(dense, denseGm_, denseBase, denseElems);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Muls(dense, dense, static_cast<int32_t>(batch), static_cast<int32_t>(denseElems));
        Duplicate(transposed[batch * sourceAligned], static_cast<T>(0), batch * 16);
        if (padding) {
            Duplicate(outputMask, UINT32_MAX, static_cast<int32_t>(maskBytes / 4));
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            for (int64_t row = 0; row < batch; ++row) {
                const int64_t last = (row + 1) * outputAligned - 1;
                const uint32_t clear = ((uint32_t(1) << padding) - 1) << (last % 32 + 1 - padding);
                outputMask.SetValue(last / 32, outputMask.GetValue(last / 32) & ~clear);
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        const int64_t tasks = (outerCount + batch - 1) / batch;
        const int64_t begin = batch * (tasks * GetBlockIdx() / GetBlockNum());
        const int64_t tileEnd = batch * (tasks * (GetBlockIdx() + 1) / GetBlockNum());
        const int64_t end = tileEnd < outerCount ? tileEnd : outerCount;
        const int64_t firstRows = begin + batch <= end ? batch : end - begin;
        CopyDenseBatchIn(source0, output0, sourceAligned, outputAligned, begin, firstRows);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        int64_t iteration = 0;
        for (int64_t outer = begin; outer < end; outer += batch, ++iteration) {
            const event_t event = static_cast<event_t>(iteration & 1);
            LocalTensor<T> output = (iteration & 1) ? output1 : output0;
            const int64_t rows = outer + batch <= end ? batch : end - outer;
            WaitFlag<HardEvent::MTE2_V>(event);
            TransposeHalf16(transposed, source0, sourceAligned, false);
            TransposeHalf16(state, output, outputAligned, false);
            PipeBarrier<PIPE_V>();
            if (outer + batch < end) {
                const event_t nextEvent = static_cast<event_t>((iteration + 1) & 1);
                if (iteration > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(nextEvent);
                }
                const int64_t nextRows = outer + 2 * batch <= end ? batch : end - outer - batch;
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                CopyDenseBatchIn(source0, (iteration & 1) ? output0 : output1,
                                 sourceAligned, outputAligned, outer + batch, nextRows);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }
            if (padding && iteration > 0) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
            }
            for (int64_t base = 0; base < maxHits; base += gatherWaves) {
                const int64_t waves = base + gatherWaves <= maxHits ? gatherWaves : maxHits - base;
                Gatherb(values.template ReinterpretCast<uint16_t>(),
                        transposed.template ReinterpretCast<uint16_t>(),
                        dense[base * paddedOutput].template ReinterpretCast<uint32_t>(),
                        static_cast<uint8_t>(waves * paddedOutput / 8), {1, 8});
                PipeBarrier<PIPE_V>();
                for (int64_t wave = 0; wave < waves; ++wave) {
                    Add(state, state, values[wave * waveElems], static_cast<int32_t>(waveElems));
                    PipeBarrier<PIPE_V>();
                }
            }
            TransposeHalf16(output, state, outputAligned, true);
            if (padding) {
                PipeBarrier<PIPE_V>();
                uint64_t selected = 0;
                GatherMask(values, output, outputMask.template ReinterpretCast<uint16_t>(), true,
                           static_cast<uint32_t>(rows * outputAligned), {1, 1, 8, 8}, selected);
            }
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            if (padding) {
                CopyOutTile(outputGm_, values, outer * outputElems_, rows * outputElems_);
                SetFlag<HardEvent::MTE3_V>(EVENT_ID2);
            } else {
                CopyDenseBatchOut(output, outputAligned, outer, rows);
            }
            SetFlag<HardEvent::MTE3_MTE2>(event);
        }
        if (padding) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        if (iteration > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        return true;
    }

    __aicore__ inline bool ProcessCompactTail(LocalTensor<T>& arena, int64_t maxHits,
                                               int64_t paddedOutput, int64_t denseBase,
                                               int64_t sourceAligned, int64_t outputAligned)
    {
        if constexpr (!std::is_same_v<T, float>) {
            return false;
        }
        constexpr int64_t prefixWaves = 10;
        const int64_t outerCount = totalElems_ / outputElems_;
        if (maxHits <= prefixWaves || maxHits > 32 || outerCount / GetBlockNum() < 16) {
            return false;
        }
        const int64_t scan = AlignUp(paddedOutput, 64);
        const int64_t denseElems = maxHits * paddedOutput;
        const int64_t prefixElems = prefixWaves * paddedOutput;
        const int64_t maxTail = (maxHits - prefixWaves) * paddedOutput;
        const int64_t maskBytes = AlignUp(scan / 8, 32);
        const int64_t arenaElems = kNarrowArenaBytes / sizeof(T);
        if (denseElems + 2 * maxTail + 3 * scan + maskBytes / 4 > arenaElems ||
            prefixElems * sizeof(T) > kSingleGatherMaxBytes) {
            return false;
        }
        LocalTensor<int32_t> dense = arena.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> compact = dense[denseElems];
        LocalTensor<int32_t> gatherPlan = compact[maxTail];
        LocalTensor<int32_t> targets = gatherPlan[maxTail];
        LocalTensor<int32_t> sequence = targets[scan];
        LocalTensor<int32_t> inverse = sequence[scan];
        LocalTensor<uint8_t> mask = inverse[scan].template ReinterpretCast<uint8_t>();
        CopyInTile(dense, denseGm_, denseBase, denseElems);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Duplicate(sequence, static_cast<int32_t>(sourceAligned * sizeof(T)), static_cast<int32_t>(scan));
        PipeBarrier<PIPE_V>();
        Adds(sequence, dense[prefixElems], 0, static_cast<int32_t>(paddedOutput));
        PipeBarrier<PIPE_V>();
        CompareScalar(mask, sequence, static_cast<int32_t>(sourceAligned * sizeof(T)),
                      CMPMODE::EQ, static_cast<int32_t>(scan));
        PipeBarrier<PIPE_V>();
        Not(mask.template ReinterpretCast<uint16_t>(), mask.template ReinterpretCast<uint16_t>(),
            static_cast<int32_t>(maskBytes / 2));
        BuildByteOffsets(sequence, static_cast<int32_t>(sizeof(T)), static_cast<int32_t>(scan));
        Duplicate(targets, 0, static_cast<int32_t>(scan));
        PipeBarrier<PIPE_V>();
        uint64_t selected = 0;
        GatherMask(targets, sequence, mask.template ReinterpretCast<uint32_t>(), true,
                   static_cast<uint32_t>(paddedOutput), {1, 1, 8, 8}, selected);
        PipeBarrier<PIPE_V>();
        const int64_t stride = AlignUp(static_cast<int64_t>(selected), 8);
        const int64_t tailElems = (maxHits - prefixWaves) * stride;
        const int64_t sourceSpan = sourceAligned + 32;
        const int64_t gatherElems = AlignUp(prefixElems + tailElems, 64);
        const int64_t metadata = gatherElems + scan;
        const int64_t required = metadata + 2 * sourceSpan + 2 * outputAligned + gatherElems + 32;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if (stride == 0 || stride * 2 >= paddedOutput || required > arenaElems ||
            gatherElems * sizeof(T) > kSingleGatherMaxBytes) {
            return false;
        }
        for (int64_t wave = 0; wave < maxHits - prefixWaves; ++wave) {
            Adds(gatherPlan[wave * stride], targets,
                 static_cast<int32_t>((prefixWaves + wave) * paddedOutput * sizeof(T)),
                 static_cast<int32_t>(stride));
        }
        PipeBarrier<PIPE_V>();
        Gather(compact, dense, gatherPlan.template ReinterpretCast<uint32_t>(), 0,
               static_cast<uint32_t>(tailElems));
        Duplicate(inverse, static_cast<int32_t>(tailElems * sizeof(T)), static_cast<int32_t>(scan));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int64_t target = 0; target < static_cast<int64_t>(selected); ++target) {
            inverse.SetValue(targets.GetValue(target) / sizeof(T), static_cast<int32_t>(target * sizeof(T)));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Adds(dense[prefixElems], compact, 0, static_cast<int32_t>(tailElems));
        if (gatherElems > prefixElems + tailElems) {
            Duplicate(dense[prefixElems + tailElems], static_cast<int32_t>(sourceAligned * sizeof(T)),
                      static_cast<int32_t>(gatherElems - prefixElems - tailElems));
        }
        Adds(dense[gatherElems], inverse, 0, static_cast<int32_t>(scan));
        PipeBarrier<PIPE_V>();
        LocalTensor<T> source0 = arena[metadata];
        LocalTensor<T> source1 = source0[sourceSpan];
        LocalTensor<T> output0 = source1[sourceSpan];
        LocalTensor<T> output1 = output0[outputAligned];
        LocalTensor<T> prefix = output1[outputAligned];
        LocalTensor<T> tail = prefix[prefixElems];
        Duplicate(source0[sourceAligned], static_cast<T>(0), 32);
        Duplicate(source1[sourceAligned], static_cast<T>(0), 32);
        Duplicate(tail[tailElems], static_cast<T>(0), 32);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        sourceGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        const int64_t block = GetBlockIdx(), cores = GetBlockNum();
        CopyInTile(source0, sourceGm_, block * sourceElems_, sourceElems_);
        CopyInTile(output0, inputGm_, block * outputElems_, outputElems_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        int64_t iteration = 0;
        for (int64_t outer = block; outer < outerCount; outer += cores, ++iteration) {
            const event_t event = static_cast<event_t>(iteration & 1);
            LocalTensor<T> source = (iteration & 1) ? source1 : source0;
            LocalTensor<T> output = (iteration & 1) ? output1 : output0;
            WaitFlag<HardEvent::MTE2_V>(event);
            if (outer + cores < outerCount) {
                const event_t next = static_cast<event_t>((iteration + 1) & 1);
                if (iteration > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(next);
                }
                LocalTensor<T> nextSource = (iteration & 1) ? source0 : source1;
                LocalTensor<T> nextOutput = (iteration & 1) ? output0 : output1;
                CopyInTile(nextSource, sourceGm_, (outer + cores) * sourceElems_, sourceElems_);
                CopyInTile(nextOutput, inputGm_, (outer + cores) * outputElems_, outputElems_);
                SetFlag<HardEvent::MTE2_V>(next);
            }
            Gather(prefix, source, dense.template ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(gatherElems));
            PipeBarrier<PIPE_V>();
            for (int64_t waves = prefixWaves; waves > 1; waves = (waves + 1) / 2) {
                const int64_t left = (waves + 1) / 2;
                Add(prefix, prefix, prefix[left * paddedOutput],
                    static_cast<int32_t>((waves - left) * paddedOutput));
                PipeBarrier<PIPE_V>();
            }
            Add(output, output, prefix, static_cast<int32_t>(outputElems_));
            for (int64_t waves = maxHits - prefixWaves; waves > 1; waves = (waves + 1) / 2) {
                const int64_t left = (waves + 1) / 2;
                Add(tail, tail, tail[left * stride], static_cast<int32_t>((waves - left) * stride));
                PipeBarrier<PIPE_V>();
            }
            Gather(prefix, tail, dense[gatherElems].template ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(scan));
            PipeBarrier<PIPE_V>();
            Add(output, output, prefix, static_cast<int32_t>(outputElems_));
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            CopyOutTile(outputGm_, output, outer * outputElems_, outputElems_);
            SetFlag<HardEvent::MTE3_MTE2>(event);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        if (iteration > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        return true;
    }

    __aicore__ inline bool ProcessDenseInner1Tree(LocalTensor<T>& arena, int64_t maxHits,
                                                  int64_t paddedOutput, int64_t denseBase,
                                                  int64_t sourceAligned, int64_t outputAligned)
    {
        if constexpr (!std::is_same_v<T, float>) {
            return false;
        }
        const int64_t outerCount = totalElems_ / outputElems_;
        if (outerCount / GetBlockNum() < 16) {
            return false;
        }
        const int64_t sourceSpan = sourceAligned + kTileAlignElems;
        const int64_t valueStride = outputAligned;
        const int64_t treeDenseElems = maxHits * valueStride;
        const int64_t treeDenseDataElems = treeDenseElems * sizeof(int32_t) / sizeof(T);
        const int64_t maxGatherWaves = (kSingleGatherMaxBytes / sizeof(T)) / valueStride;
        if (maxGatherWaves <= 0) {
            return false;
        }
        const int64_t gatherWaves = maxHits < maxGatherWaves ? maxHits : maxGatherWaves;
        const int64_t gatherValueElems = gatherWaves * valueStride;
        const int64_t arenaElems = kNarrowArenaBytes / sizeof(T);
        if (treeDenseDataElems + 2 * sourceSpan + 2 * outputAligned + gatherValueElems > arenaElems) {
            return false;
        }
        if (sourceElems_ * outerCount * sizeof(T) >= kL2BypassMinBytes / 8) {
            sourceGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        LocalTensor<int32_t> dense = arena.template ReinterpretCast<int32_t>();
        LocalTensor<T> source0 = arena[treeDenseDataElems];
        LocalTensor<T> source1 = source0[sourceSpan];
        LocalTensor<T> output0 = source1[sourceSpan];
        LocalTensor<T> output1 = output0[outputAligned];
        LocalTensor<T> values = output1[outputAligned];
        Duplicate(source0[sourceAligned], static_cast<T>(0), kTileAlignElems);
        Duplicate(source1[sourceAligned], static_cast<T>(0), kTileAlignElems);
        if (valueStride != paddedOutput) {
            Duplicate(dense, static_cast<int32_t>(sourceAligned * sizeof(T)),
                      static_cast<int32_t>(treeDenseElems));
        }
        PipeBarrier<PIPE_V>();
        if (valueStride == paddedOutput) {
            CopyInTile(dense, denseGm_, denseBase, treeDenseElems);
        } else {
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            for (int64_t wave = 0; wave < maxHits; ++wave) {
                LocalTensor<int32_t> denseRow = dense[wave * valueStride];
                CopyInTile(denseRow, denseGm_, denseBase + wave * paddedOutput, paddedOutput);
            }
        }

        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        if (blockId >= outerCount) {
            return true;
        }
        CopyInTile(source0, sourceGm_, blockId * sourceElems_, sourceElems_);
        CopyInTile(output0, inputGm_, blockId * outputElems_, outputElems_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        int64_t iteration = 0;
        for (int64_t outer = blockId; outer < outerCount; outer += blockNum, ++iteration) {
            const event_t currentEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID0 : EVENT_ID1);
            LocalTensor<T> source = (iteration & 1) == 0 ? source0 : source1;
            LocalTensor<T> output = (iteration & 1) == 0 ? output0 : output1;
            WaitFlag<HardEvent::MTE2_V>(currentEvent);

            const int64_t nextOuter = outer + blockNum;
            if (nextOuter < outerCount) {
                const event_t nextEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID1 : EVENT_ID0);
                LocalTensor<T> nextSource = (iteration & 1) == 0 ? source1 : source0;
                LocalTensor<T> nextOutput = (iteration & 1) == 0 ? output1 : output0;
                if (iteration > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(nextEvent);
                }
                CopyInTile(nextSource, sourceGm_, nextOuter * sourceElems_, sourceElems_);
                CopyInTile(nextOutput, inputGm_, nextOuter * outputElems_, outputElems_);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }

            for (int64_t waveBase = 0; waveBase < maxHits; waveBase += gatherWaves) {
                const int64_t chunkWaves = waveBase + gatherWaves <= maxHits ? gatherWaves : maxHits - waveBase;
                Gather(values, source, dense[waveBase * valueStride].template ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(chunkWaves * valueStride));
                PipeBarrier<PIPE_V>();
                int64_t activeWaves = chunkWaves;
                while (activeWaves > 1) {
                    const int64_t leftWaves = (activeWaves + 1) / 2;
                    const int64_t pairedWaves = activeWaves - leftWaves;
                    Add(values, values, values[leftWaves * valueStride],
                        static_cast<int32_t>(pairedWaves * valueStride));
                    PipeBarrier<PIPE_V>();
                    activeWaves = leftWaves;
                }
                Add(output, output, values, static_cast<int32_t>(outputElems_));
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(currentEvent);
            WaitFlag<HardEvent::V_MTE3>(currentEvent);
            CopyOutTile(outputGm_, output, outer * outputElems_, outputElems_);
            SetFlag<HardEvent::MTE3_MTE2>(currentEvent);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        if (iteration > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        return true;
    }

    template <int64_t BatchRows>
    __aicore__ inline bool ProcessDenseInner1Batched(LocalTensor<T>& arena, int64_t maxHits,
                                                     int64_t paddedOutput, int64_t denseBase,
                                                     int64_t denseElems, int64_t sourceAligned,
                                                     int64_t outputAligned)
    {
        const int64_t sourceSpan = sourceAligned + kTileAlignElems;
        const int64_t expandedElems = BatchRows * denseElems;
        int64_t scheduleElems = denseElems + expandedElems;
        int64_t scheduleDataElems = scheduleElems * sizeof(int32_t) / sizeof(T);
        const int64_t arenaElems = kNarrowArenaBytes / sizeof(T);
        int64_t requiredElems = scheduleDataElems + BatchRows *
            (2 * sourceSpan + 2 * outputAligned + paddedOutput);
        if (sourceSpan - sourceElems_ > UINT8_MAX) {
            return false;
        }
        const int64_t outerCount = totalElems_ / outputElems_;
        const int64_t rowsPerCore = outerCount / GetBlockNum();
        const int64_t compactGatherBytes =
            maxHits * BatchRows * paddedOutput * static_cast<int64_t>(sizeof(T));
        const bool preferCompactFp32 = BatchRows == 2 && std::is_same_v<T, float> &&
                                       compactGatherBytes <= kSingleGatherMaxBytes && rowsPerCore >= 16;
        const bool preferCompactFp16 = BatchRows == 5 && std::is_same_v<T, half> &&
                                       rowsPerCore >= 1;
        const bool preferCompact = preferCompactFp32 || preferCompactFp16;
        bool compactSchedule = false;
        if (preferCompact || requiredElems > arenaElems) {
            if constexpr (!((BatchRows == 2 && std::is_same_v<T, float>) ||
                            (BatchRows == 5 && std::is_same_v<T, half>))) {
                return false;
            }
            scheduleElems = expandedElems;
            scheduleDataElems = scheduleElems * sizeof(int32_t) / sizeof(T);
            requiredElems = scheduleDataElems + BatchRows *
                (2 * sourceSpan + 2 * outputAligned + paddedOutput);
            if (requiredElems > arenaElems) {
                return false;
            }
            compactSchedule = true;
        }
        LocalTensor<int32_t> dense = arena.template ReinterpretCast<int32_t>();
        LocalTensor<int32_t> expanded = compactSchedule ? dense : dense[denseElems];
        LocalTensor<T> source0 = arena[scheduleDataElems];
        LocalTensor<T> source1 = source0[BatchRows * sourceSpan];
        LocalTensor<T> output0 = source1[BatchRows * sourceSpan];
        LocalTensor<T> output1 = output0[BatchRows * outputAligned];
        LocalTensor<T> values = output1[BatchRows * outputAligned];
        const int64_t valueCapacity = arenaElems - scheduleDataElems -
            2 * BatchRows * sourceSpan - 2 * BatchRows * outputAligned;
        const int64_t treeWaveElems = BatchRows * paddedOutput;
        const int64_t treeGatherCapacity = valueCapacity / treeWaveElems;
        const int64_t treeGatherLimit =
            (kSingleGatherMaxBytes / static_cast<int64_t>(sizeof(T))) / treeWaveElems;
        const int64_t treeGatherWaves = treeGatherCapacity < treeGatherLimit ?
            treeGatherCapacity : treeGatherLimit;

        if (compactSchedule) {
            for (int64_t wave = 0; wave < maxHits; ++wave) {
                LocalTensor<int32_t> row = expanded[wave * BatchRows * paddedOutput];
                CopyInTile(row, denseGm_, denseBase + wave * paddedOutput, paddedOutput);
            }
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (int64_t wave = 0; wave < maxHits; ++wave) {
                for (int64_t row = 1; row < BatchRows; ++row) {
                    Adds(expanded[(wave * BatchRows + row) * paddedOutput],
                         expanded[wave * BatchRows * paddedOutput],
                         static_cast<int32_t>(row * sourceSpan * sizeof(T)),
                         static_cast<int32_t>(paddedOutput));
                    PipeBarrier<PIPE_V>();
                }
            }
        } else {
            CopyInTile(dense, denseGm_, denseBase, denseElems);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            for (int64_t wave = 0; wave < maxHits; ++wave) {
                for (int64_t row = 0; row < BatchRows; ++row) {
                    Adds(expanded[(wave * BatchRows + row) * paddedOutput], dense[wave * paddedOutput],
                         static_cast<int32_t>(row * sourceSpan * sizeof(T)),
                         static_cast<int32_t>(paddedOutput));
                    PipeBarrier<PIPE_V>();
                }
            }
        }
        Duplicate(source0, static_cast<T>(0), static_cast<int32_t>(BatchRows * sourceSpan));
        Duplicate(source1, static_cast<T>(0), static_cast<int32_t>(BatchRows * sourceSpan));
        PipeBarrier<PIPE_V>();

        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t outerBegin = outerCount * blockId / blockNum;
        const int64_t outerEnd = outerCount * (blockId + 1) / blockNum;
        if (outerBegin >= outerEnd) {
            return true;
        }
        const int64_t firstRows = outerBegin + BatchRows <= outerEnd ? BatchRows : outerEnd - outerBegin;
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        CopyDenseBatchIn(source0, output0, sourceSpan, outputAligned, outerBegin, firstRows);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        int64_t iteration = 0;
        for (int64_t outer = outerBegin; outer < outerEnd; outer += BatchRows, ++iteration) {
            const event_t currentEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID0 : EVENT_ID1);
            LocalTensor<T> source = (iteration & 1) == 0 ? source0 : source1;
            LocalTensor<T> output = (iteration & 1) == 0 ? output0 : output1;
            const int64_t rows = outer + BatchRows <= outerEnd ? BatchRows : outerEnd - outer;
            WaitFlag<HardEvent::MTE2_V>(currentEvent);

            const int64_t nextOuter = outer + BatchRows;
            if (nextOuter < outerEnd) {
                const event_t nextEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID1 : EVENT_ID0);
                LocalTensor<T> nextSource = (iteration & 1) == 0 ? source1 : source0;
                LocalTensor<T> nextOutput = (iteration & 1) == 0 ? output1 : output0;
                if (iteration > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(nextEvent);
                }
                const int64_t nextRows = nextOuter + BatchRows <= outerEnd ? BatchRows : outerEnd - nextOuter;
                CopyDenseBatchIn(nextSource, nextOutput, sourceSpan, outputAligned, nextOuter, nextRows);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }

            if (preferCompactFp32 && rows == BatchRows && treeGatherWaves > 1) {
                for (int64_t waveBase = 0; waveBase < maxHits; waveBase += treeGatherWaves) {
                    const int64_t chunkWaves = waveBase + treeGatherWaves <= maxHits ?
                        treeGatherWaves : maxHits - waveBase;
                    Gather(values, source,
                           expanded[waveBase * treeWaveElems].template ReinterpretCast<uint32_t>(),
                           0, static_cast<uint32_t>(chunkWaves * treeWaveElems));
                    PipeBarrier<PIPE_V>();
                    int64_t activeWaves = chunkWaves;
                    while (activeWaves > 1) {
                        const int64_t leftWaves = (activeWaves + 1) / 2;
                        const int64_t pairedWaves = activeWaves - leftWaves;
                        Add(values, values, values[leftWaves * treeWaveElems],
                            static_cast<int32_t>(pairedWaves * treeWaveElems));
                        PipeBarrier<PIPE_V>();
                        activeWaves = leftWaves;
                    }
                    Add(output, output, values, static_cast<int32_t>(treeWaveElems));
                    PipeBarrier<PIPE_V>();
                }
            } else {
                for (int64_t wave = 0; wave < maxHits; ++wave) {
                    Gather(values, source,
                           expanded[wave * BatchRows * paddedOutput].template ReinterpretCast<uint32_t>(),
                           0, static_cast<uint32_t>(rows * paddedOutput));
                    PipeBarrier<PIPE_V>();
                    Add(output, output, values, static_cast<int32_t>(rows * paddedOutput));
                    PipeBarrier<PIPE_V>();
                }
            }
            SetFlag<HardEvent::V_MTE3>(currentEvent);
            WaitFlag<HardEvent::V_MTE3>(currentEvent);
            CopyDenseBatchOut(output, outputAligned, outer, rows);
            SetFlag<HardEvent::MTE3_MTE2>(currentEvent);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        if (iteration > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
        return true;
    }

    __aicore__ inline void CopyDenseBatchIn(LocalTensor<T> source, LocalTensor<T> output,
                                            int64_t sourceSpan, int64_t outputAligned,
                                            int64_t outer, int64_t rows)
    {
        const uint32_t sourceBytes = static_cast<uint32_t>(sourceElems_ * static_cast<int64_t>(sizeof(T)));
        const uint32_t sourceStride = static_cast<uint32_t>(
            (sourceSpan * static_cast<int64_t>(sizeof(T)) - AlignUp(sourceBytes, kAlignBytes)) / kAlignBytes);
        DataCopyExtParams sourceCopy = {static_cast<uint16_t>(rows), sourceBytes, 0, sourceStride, 0};
        DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
        if (sourceSpan == sourceElems_) {
            CopyInTile(source, sourceGm_, outer * sourceElems_, rows * sourceElems_);
        } else {
            DataCopyPad(source, sourceGm_[outer * sourceElems_], sourceCopy, pad);
        }

        const uint32_t outputBytes = static_cast<uint32_t>(outputElems_ * static_cast<int64_t>(sizeof(T)));
        const uint32_t outputStride = static_cast<uint32_t>(
            (outputAligned * static_cast<int64_t>(sizeof(T)) - AlignUp(outputBytes, kAlignBytes)) / kAlignBytes);
        DataCopyExtParams outputCopy = {static_cast<uint16_t>(rows), outputBytes, 0, outputStride, 0};
        DataCopyPad(output, inputGm_[outer * outputElems_], outputCopy, pad);
    }

    __aicore__ inline void CopyDenseBatchOut(LocalTensor<T> output, int64_t outputAligned,
                                             int64_t outer, int64_t rows)
    {
        const uint32_t outputBytes = static_cast<uint32_t>(outputElems_ * static_cast<int64_t>(sizeof(T)));
        const uint32_t sourceStride = static_cast<uint32_t>(
            (outputAligned * static_cast<int64_t>(sizeof(T)) - AlignUp(outputBytes, kAlignBytes)) / kAlignBytes);
        DataCopyExtParams copy = {static_cast<uint16_t>(rows), outputBytes, sourceStride, 0, 0};
        DataCopyPad(outputGm_[outer * outputElems_], output, copy);
    }

    __aicore__ inline void ProcessDenseDoubleBuffered(LocalTensor<T>& arena, int64_t maxHits,
                                                      int64_t paddedOutput, int64_t denseBase,
                                                      int64_t denseElems, int64_t denseDataElems,
                                                      int64_t sourceAligned, int64_t outputAligned)
    {
        const int64_t sourceSpan = sourceAligned + kTileAlignElems;
        LocalTensor<int32_t> dense = arena.template ReinterpretCast<int32_t>();
        LocalTensor<T> source0 = arena[denseDataElems];
        LocalTensor<T> source1 = source0[sourceSpan];
        LocalTensor<T> output0 = source1[sourceSpan];
        LocalTensor<T> output1 = output0[outputAligned];
        LocalTensor<T> values = output1[outputAligned];
        Duplicate(source0[sourceAligned], static_cast<T>(0), kTileAlignElems);
        Duplicate(source1[sourceAligned], static_cast<T>(0), kTileAlignElems);
        PipeBarrier<PIPE_V>();

        const int64_t outerCount = totalElems_ / outputElems_;
        const int64_t blockId = static_cast<int64_t>(GetBlockIdx());
        const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
        const int64_t firstOuter = blockId;
        if (firstOuter >= outerCount) {
            return;
        }
        const int64_t denseWaves = maxHits - 12;
        int64_t sparseStart = 0;
        int64_t sparseLength = 0;
        bool sparseOverflow = false;
        bool denseLoaded = false;
        if constexpr (std::is_same_v<T, half>) {
            if (outerCount / GetBlockNum() >= 4 && maxHits > 16 && maxHits <= 64) {
                CopyInTile(dense, denseGm_, denseBase, denseElems);
                SetFlag<HardEvent::MTE2_S>(mte2ToS_);
                WaitFlag<HardEvent::MTE2_S>(mte2ToS_);
                denseLoaded = true;
                const int32_t sentinel = static_cast<int32_t>(sourceAligned * sizeof(T));
                const int64_t scanBlocks = (outputElems_ + 7) / 8;
                const int32_t scanRepeats = static_cast<int32_t>((scanBlocks + 7) / 8);
                LocalTensor<float> blockMins = output0.template ReinterpretCast<float>();
                BlockReduceMin(blockMins,
                               dense[denseWaves * paddedOutput].template ReinterpretCast<float>(),
                               scanRepeats, 64, 1, 1, 8);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_S>(vToS_);
                WaitFlag<HardEvent::V_S>(vToS_);
                LocalTensor<int32_t> blockMinBits = blockMins.template ReinterpretCast<int32_t>();
                int64_t previousBlock = -2;
                bool oneRun = true;
                for (int64_t block = 0; block < scanBlocks; ++block) {
                    if (blockMinBits.GetValue(block) >= sentinel) {
                        continue;
                    }
                    if (sparseLength == 0) {
                        sparseStart = block * 8;
                    } else if (block != previousBlock + 1) {
                        oneRun = false;
                        break;
                    }
                    sparseLength += block * 8 + 8 <= outputElems_ ? 8 : outputElems_ - block * 8;
                    previousBlock = block;
                }
                sparseOverflow = oneRun && sparseLength > 0 && sparseLength * 4 <= outputElems_;
            }
        }
        if (!denseLoaded) {
            CopyInTile(dense, denseGm_, denseBase, denseElems);
        }
        CopyInTile(source0, sourceGm_, firstOuter * sourceElems_, sourceElems_);
        CopyInTile(output0, inputGm_, firstOuter * outputElems_, outputElems_);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

        int64_t iteration = 0;
        for (int64_t outer = firstOuter; outer < outerCount; outer += blockNum, ++iteration) {
            const event_t currentEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID0 : EVENT_ID1);
            LocalTensor<T> source = (iteration & 1) == 0 ? source0 : source1;
            LocalTensor<T> output = (iteration & 1) == 0 ? output0 : output1;
            WaitFlag<HardEvent::MTE2_V>(currentEvent);

            const int64_t nextOuter = outer + blockNum;
            if (nextOuter < outerCount) {
                const event_t nextEvent = static_cast<event_t>((iteration & 1) == 0 ? EVENT_ID1 : EVENT_ID0);
                LocalTensor<T> nextSource = (iteration & 1) == 0 ? source1 : source0;
                LocalTensor<T> nextOutput = (iteration & 1) == 0 ? output1 : output0;
                if (iteration > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(nextEvent);
                }
                CopyInTile(nextSource, sourceGm_, nextOuter * sourceElems_, sourceElems_);
                CopyInTile(nextOutput, inputGm_, nextOuter * outputElems_, outputElems_);
                SetFlag<HardEvent::MTE2_V>(nextEvent);
            }

            const int64_t fullWaves = sparseOverflow ? denseWaves : maxHits;
            for (int64_t wave = 0; wave < fullWaves; ++wave) {
                Gather(values, source, dense[wave * paddedOutput].template ReinterpretCast<uint32_t>(), 0,
                       static_cast<uint32_t>(outputElems_));
                PipeBarrier<PIPE_V>();
                Add(output, output, values, static_cast<int32_t>(outputElems_));
                PipeBarrier<PIPE_V>();
            }
            if (sparseOverflow) {
                for (int64_t wave = denseWaves; wave < maxHits; ++wave) {
                    Gather(values, source,
                           dense[wave * paddedOutput + sparseStart].template ReinterpretCast<uint32_t>(),
                           0, static_cast<uint32_t>(sparseLength));
                    PipeBarrier<PIPE_V>();
                    Add(output[sparseStart], output[sparseStart], values, static_cast<int32_t>(sparseLength));
                    PipeBarrier<PIPE_V>();
                }
            }
            SetFlag<HardEvent::V_MTE3>(currentEvent);
            WaitFlag<HardEvent::V_MTE3>(currentEvent);
            CopyOutTile(outputGm_, output, outer * outputElems_, outputElems_);
            SetFlag<HardEvent::MTE3_MTE2>(currentEvent);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        if (iteration > 1) {
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
        }
    }

    __aicore__ inline void ReduceRow(LocalTensor<T>& source, LocalTensor<T> output,
                                     LocalTensor<int32_t>& metadata, LocalTensor<int32_t>& offsets,
                                     LocalTensor<T>& values, LocalTensor<T>& sums,
                                     LocalTensor<int32_t>& sumOffsets, int64_t hitBegin, int64_t hitEnd)
    {
        if constexpr (std::is_same_v<T, float>) {
            if (hitEnd - hitBegin > kNarrowChunkElems) {
                for (int64_t hit = hitBegin; hit < hitEnd; ++hit) {
                    const int32_t row = metadata.GetValue(dimSize_ + 1 + hit);
                    ArithProgression(offsets, row * static_cast<int32_t>(innerSize_ * sizeof(T)),
                                     static_cast<int32_t>(sizeof(T)), static_cast<int32_t>(innerSize_));
                    PipeBarrier<PIPE_V>();
                    Gather(values, source, offsets.template ReinterpretCast<uint32_t>(), 0,
                           static_cast<uint32_t>(innerSize_));
                    PipeBarrier<PIPE_V>();
                    Add(output, output, values, static_cast<int32_t>(innerSize_));
                    PipeBarrier<PIPE_V>();
                }
                return;
            }
        }
        Duplicate(sums, static_cast<T>(0), static_cast<int32_t>(innerSize_ * paddedInner_));
        for (int64_t column = 0; column < innerSize_; ++column) {
            for (int64_t hit = hitBegin; hit < hitEnd; hit += kNarrowChunkElems) {
                const int32_t count = static_cast<int32_t>(
                    hit + kNarrowChunkElems <= hitEnd ? kNarrowChunkElems : hitEnd - hit);
                const int32_t paddedCount = static_cast<int32_t>(AlignElems<T>(count));
                Duplicate(values, static_cast<T>(0), paddedCount);
                ArithProgression(offsets, static_cast<int32_t>((dimSize_ + 1 + hit) * sizeof(int32_t)),
                                 static_cast<int32_t>(sizeof(int32_t)), count);
                PipeBarrier<PIPE_V>();
                Gather(values.template ReinterpretCast<int32_t>(), metadata,
                       offsets.template ReinterpretCast<uint32_t>(), 0, count);
                PipeBarrier<PIPE_V>();
                Muls(offsets, values.template ReinterpretCast<int32_t>(),
                     static_cast<int32_t>(innerSize_ * sizeof(T)), count);
                PipeBarrier<PIPE_V>();
                Gather(values, source, offsets.template ReinterpretCast<uint32_t>(),
                       static_cast<uint32_t>(column * sizeof(T)), count);
                TreeReduce(values, offsets.template ReinterpretCast<T>(), sumOffsets, paddedCount);
                Add(sums[column * paddedInner_], sums[column * paddedInner_], values, 1);
                PipeBarrier<PIPE_V>();
            }
        }
        ArithProgression(sumOffsets, 0, static_cast<int32_t>(paddedInner_ * sizeof(T)),
                         static_cast<int32_t>(innerSize_));
        PipeBarrier<PIPE_V>();
        Gather(values, sums, sumOffsets.template ReinterpretCast<uint32_t>(), 0,
               static_cast<uint32_t>(innerSize_));
        PipeBarrier<PIPE_V>();
        Add(output, output, values, static_cast<int32_t>(innerSize_));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TreeReduce(LocalTensor<T>& values, LocalTensor<T> temp,
                                      LocalTensor<int32_t>& laneOffsets, int32_t paddedCount)
    {
        constexpr int32_t blockElems = kAlignBytes / sizeof(T);
        int32_t activeBlocks = paddedCount / blockElems;
        while (activeBlocks > 1) {
            const int32_t leftBlocks = (activeBlocks + 1) / 2;
            Add(values, values, values[leftBlocks * blockElems],
                (activeBlocks - leftBlocks) * blockElems);
            PipeBarrier<PIPE_V>();
            activeBlocks = leftBlocks;
        }
        for (int32_t width = blockElems / 2; width > 0; width /= 2) {
            ArithProgression(laneOffsets, width * static_cast<int32_t>(sizeof(T)),
                             static_cast<int32_t>(sizeof(T)), width);
            PipeBarrier<PIPE_V>();
            Gather(temp, values, laneOffsets.template ReinterpretCast<uint32_t>(), 0, width);
            PipeBarrier<PIPE_V>();
            Add(values, values, temp, width);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyRowsIn(LocalTensor<T> local, GlobalTensor<T>& global, int64_t offset)
    {
        DataCopyExtParams copy = {static_cast<uint16_t>(dimSize_),
                                  static_cast<uint32_t>(innerSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad = {false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, global[offset], copy, pad);
    }

    __aicore__ inline void CopyRowsOut(GlobalTensor<T>& global, LocalTensor<T> local, int64_t offset)
    {
        DataCopyExtParams copy = {static_cast<uint16_t>(dimSize_),
                                  static_cast<uint32_t>(innerSize_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(global[offset], local, copy);
    }

private:
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> arenaBuf_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> sourceGm_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<int32_t> metadataGm_;
    GlobalTensor<int32_t> denseGm_;
    int64_t totalElems_ = 0;
    int64_t dimSize_ = 0;
    int64_t innerSize_ = 0;
    int64_t indexRows_ = 0;
    int64_t outputElems_ = 0;
    int64_t sourceElems_ = 0;
    int64_t paddedInner_ = 0;
    int64_t metadataElems_ = 0;
    int64_t denseWorkspaceOffset_ = 0;
    event_t mte2ToS_;
    event_t sToMte2_;
    event_t vToS_;
    event_t sToV_;
    event_t sToMte3_;
    event_t mte3ToS_;
};

}

extern "C" __global__ __aicore__ void index_add(
    GM_ADDR input, GM_ADDR index, GM_ADDR source, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    static_cast<void>(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE(1, KERNEL_TYPE_MIX_AIV_1_0);
        KernelIndexAddAtomic<DTYPE_INPUT> op;
        op.Init(input, index, source, output, tilingData);
        op.Process(false);
    } else if (TILING_KEY_IS(2)) {
        KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIV_1_0);
        KernelIndexAddAtomic<DTYPE_INPUT> op;
        op.Init(input, index, source, output, tilingData);
        op.Process(true);
    } else if (TILING_KEY_IS(3)) {
        KernelIndexAddOutputOwned<DTYPE_INPUT, false> op;
        op.Init(input, index, source, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelIndexAddOuterOwned<DTYPE_INPUT> op;
        op.Init(input, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KERNEL_TASK_TYPE(5, KERNEL_TYPE_MIX_AIV_1_0);
        KernelIndexAddOutputOwned<DTYPE_INPUT, true> op;
        op.Init(input, index, source, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(6)) {
        KERNEL_TASK_TYPE(6, KERNEL_TYPE_MIX_AIV_1_0);
        KernelIndexAddNarrowOuter<DTYPE_INPUT> op;
        op.Init(input, index, source, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(7)) {
        KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIV_1_0);
        KernelIndexAddNarrowOuter<DTYPE_INPUT> op;
        op.Init(input, index, source, output, workspace, tilingData);
        op.Process(true);
    } else if (TILING_KEY_IS(8)) {
        KernelIndexAddBf16TargetOwned op;
        op.Init(input, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10)) {
        KernelIndexAddSmallInt8 op;
        op.Init(input, index, source, output, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(9)) {
        KERNEL_TASK_TYPE(9, KERNEL_TYPE_MIX_AIV_1_0);
        KernelIndexAddReducedAtomic op;
        op.Init(input, index, source, output, tilingData);
        op.Process();
    }
}
