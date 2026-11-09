/**
 * SquareSumV1 设备侧实现。
 * 对输入执行平方与求和归约；按照分块信息组织输入流水线、局部累加和输出存储。
 */

#include "kernel_operator.h"
#include "lib/reduce/sum.h"
#include "../op_host/square_sum_v1_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kOutTile = 512U;
constexpr uint32_t kDeepInBytes = 86016U;
constexpr uint32_t kPartialTile = 256U;
constexpr uint32_t kSegTile = 512U;

template <typename A, typename B>
struct SameType {
    static constexpr bool value = false;
};
template <typename A>
struct SameType<A, A> {
    static constexpr bool value = true;
};

template <typename T, uint32_t kPath4Mode = 0U>
// 根据最内层维度是否参与归约组织平方求和；分段归约使用工作区合并部分结果。
class KernelSquareSumV1 {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                const SquareSumV1TilingData& tiling)
    {
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if (tiling.redSegCount > 1 || tiling.keepRedSeg > 1 ||
            tiling.slabScratch != 0U) {

            GM_ADDR user = GetUserWorkspace(workspace);
            const uint32_t blocks = static_cast<uint32_t>(GetBlockNum());
            syncGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(user), blocks * 8);
            ws_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(user + blocks * 32));
        }
        for (uint32_t i = 0; i < S9_SQ_MAX_GROUP; ++i) {
            groupLen_[i] = tiling.groupLen[i];
            groupStride_[i] = tiling.groupStride[i];
            groupIsReduce_[i] = tiling.groupIsReduce[i];
        }
        groupCount_ = tiling.groupCount;
        innerLen_ = tiling.innerLen;
        innerIsReduce_ = tiling.innerIsReduce;
        innerChunk_ = tiling.innerChunk;
        innerChunkCount_ = tiling.innerChunkCount;
        prefixRedCount_ = tiling.prefixRedCount;
        outCount_ = tiling.outCount;
        taskCount_ = tiling.taskCount;
        redSegCount_ = tiling.redSegCount;
        redSegLen_ = tiling.redSegLen;
        keepRedSeg_ = tiling.keepRedSeg;
        keepRedSegLen_ = tiling.keepRedSegLen;
        slabScratch_ = tiling.slabScratch;
        tile_ = tiling.tileElements;

        outSlot_ = 0U;
        outPend_[0] = 0U;
        outPend_[1] = 0U;

        const uint32_t inSlots =
            (kPath4Mode == 0U && sizeof(T) == 2U && tile_ <= 8192U) ? 4U : 2U;

        const uint32_t inBytes =
            inSlots == 4U ? kDeepInBytes : inSlots * tile_ * sizeof(T);
        pipe_.InitBuffer(inBuf_, inBytes < 32U ? 32U : inBytes);
        if constexpr (kPath4Mode == 3U || kPath4Mode == 5U) {
            const uint32_t accElements = innerLen_ > kOutTile ?
                static_cast<uint32_t>(innerLen_) : kOutTile;
            pipe_.InitBuffer(sqBuf_, tile_ * sizeof(float));
            pipe_.InitBuffer(accBuf_, accElements * sizeof(float));
            pipe_.InitBuffer(workBuf_, 32U);
        } else {
            pipe_.InitBuffer(sqBuf_, tile_ * sizeof(float));
            pipe_.InitBuffer(accBuf_, tile_ * sizeof(float));
            pipe_.InitBuffer(
                workBuf_, kPath4Mode == 2U ? 32U : tile_ * sizeof(float));
        }
        pipe_.InitBuffer(
            redBuf_, (kPath4Mode == 2U || kPath4Mode == 3U ||
                      kPath4Mode == 5U) ?
            32U : 32U * sizeof(float));
        pipe_.InitBuffer(outF32Buf_, kOutTile * sizeof(float));

        pipe_.InitBuffer(outTBuf_,
                         SameType<T, float>::value ? (kOutTile * sizeof(T))
                                                   : (2U * kOutTile * sizeof(T)));
        pipe_.InitBuffer(
            partialBuf_, (kPath4Mode == 2U || kPath4Mode == 3U ||
                          kPath4Mode == 5U) ?
            32U : kPartialTile * sizeof(float));
        pipe_.InitBuffer(
            segBuf_, (kPath4Mode == 2U || kPath4Mode == 3U ||
                      kPath4Mode == 5U) ?
            32U : kSegTile * sizeof(float));
        if (tiling.redSegCount > 1 || tiling.keepRedSeg > 1) {
            pipe_.InitBuffer(syncUbBuf_, 64 * 32);
            InitGlobalMemory(syncGm_, static_cast<uint32_t>(GetBlockNum()) * 8, 0);
        }
    }

    __aicore__ inline void Process()
    {
        ProcessImpl();
        DrainStore();
    }

    __aicore__ inline void ProcessImpl()
    {
        if constexpr (kPath4Mode == 4U || kPath4Mode == 5U) {
            ProcessKeepInnerCyclic();
            return;
        } else if constexpr (kPath4Mode == 3U) {
            ProcessKeepInnerContiguousCompact();
            return;
        } else if constexpr (kPath4Mode != 0U) {
            ProcessReduceInnerPrefixCyclic();
            return;
        }
        if (redSegCount_ > 1) {

            ReduceMultiCore();
            return;
        }
        if (keepRedSeg_ > 1) {
            KeepInnerReduceMultiCore();
            return;
        }
        if (taskCount_ == 0) {
            return;
        }
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t share = taskCount_ / blocks;
        const uint64_t extra = taskCount_ % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1 : 0);
        if (begin >= end) {
            return;
        }
        if (innerIsReduce_ != 0) {
            ProcessReduceInner(begin, end);
        } else {
            ProcessKeepInner(begin, end);
        }
    }

    __aicore__ inline void ReduceMultiCore()
    {
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t share = taskCount_ / blocks;
        const uint64_t extra = taskCount_ % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1 : 0);

        LocalTensor<float> acc = accBuf_.template Get<float>();
        LocalTensor<float> work = workBuf_.template Get<float>();
        LocalTensor<float> red = redBuf_.template Get<float>();
        LocalTensor<float> partial = partialBuf_.template Get<float>();

        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t out = task / redSegCount_;
            const uint64_t seg = task % redSegCount_;
            const uint64_t segStart = seg * redSegLen_;
            const uint64_t segEnd =
                (segStart + redSegLen_ < innerLen_) ? (segStart + redSegLen_) : innerLen_;
            const uint64_t keepOffset = KeepOffset(out);
            Duplicate(acc, 0.0f, static_cast<int32_t>(tile_));
            PipeBarrier<PIPE_V>();
            for (uint64_t r = 0; r < prefixRedCount_; ++r) {
                const uint64_t runOffset = keepOffset + ReduceOffset(r);
                for (uint64_t pos = segStart; pos < segEnd; pos += tile_) {
                    const uint64_t left = segEnd - pos;
                    const uint32_t count = static_cast<uint32_t>(left < tile_ ? left : tile_);
                    LoadTile(runOffset + pos, count);
                    LocalTensor<float> squared = SquareTile(count);
                    Add(acc, acc, squared, static_cast<int32_t>(count));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
            }
            ReduceSum<float>(red, acc, work, static_cast<int32_t>(tile_));

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const DataCopyExtParams store(1, sizeof(float), 0, 0, 0);
            DataCopyPad(ws_[task * 8], red, store);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        (void)partial;

        LocalTensor<int32_t> ubSync = syncUbBuf_.template Get<int32_t>();
        SyncAll(syncGm_, ubSync, static_cast<int32_t>(GetBlockNum()));

        LocalTensor<float> seg = segBuf_.template Get<float>();
        LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
        const DataCopyPadExtParams<float> pad;
        for (uint64_t out = block; out < outCount_; out += blocks) {

            float total = 0.0f;
            for (uint64_t i = 0; i < redSegCount_; ++i) {
                const DataCopyExtParams load(1, sizeof(float), 0, 0, 0);
                DataCopyPad(seg, ws_[(out * redSegCount_ + i) * 8], load, pad);
                SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
                total += seg.GetValue(0);
            }
            outF32.SetValue(0, total);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            PipeBarrier<PIPE_ALL>();
            StoreOutput(out, 1);
        }
    }

private:

    __aicore__ inline uint64_t KeepOffset(uint64_t index) const
    {
        uint64_t offset = 0;
        for (int32_t i = static_cast<int32_t>(groupCount_) - 1; i >= 0; --i) {
            if (groupIsReduce_[i] != 0) {
                continue;
            }
            const uint64_t len = groupLen_[i];
            offset += (index % len) * groupStride_[i];
            index /= len;
        }
        return offset;
    }

    __aicore__ inline uint64_t ReduceOffset(uint64_t index) const
    {
        uint64_t offset = 0;
        for (int32_t i = static_cast<int32_t>(groupCount_) - 1; i >= 0; --i) {
            if (groupIsReduce_[i] == 0) {
                continue;
            }
            const uint64_t len = groupLen_[i];
            offset += (index % len) * groupStride_[i];
            index /= len;
        }
        return offset;
    }

    __aicore__ inline void LoadTile(uint64_t offset, uint32_t count)
    {
        LocalTensor<T> local = inBuf_.template Get<T>();
        const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
        const DataCopyPadExtParams<T> pad;
        DataCopyPad(local, x_[offset], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline LocalTensor<float> SquareTile(uint32_t count)
    {
        if constexpr (SameType<T, float>::value) {
            LocalTensor<float> src = inBuf_.template Get<float>();
            Mul(src, src, src, static_cast<int32_t>(count));
            return src;
        } else {
            LocalTensor<T> src = inBuf_.template Get<T>();
            LocalTensor<float> dst = sqBuf_.template Get<float>();
            Cast(dst, src, RoundMode::CAST_NONE, static_cast<int32_t>(count));
            Mul(dst, dst, dst, static_cast<int32_t>(count));
            return dst;
        }
    }

    __aicore__ inline void StoreOutput(uint64_t outIndex, uint32_t count)
    {
        LocalTensor<float> src = outF32Buf_.template Get<float>();
        if constexpr (SameType<T, float>::value) {

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
            DataCopyPad(y_[outIndex], src, copy);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        } else {

            const uint32_t slot = outSlot_;
            if (outPend_[slot] != 0U) {
                if (slot == 0U) { WaitFlag<HardEvent::MTE3_V>(EVENT_ID4); }
                else { WaitFlag<HardEvent::MTE3_V>(EVENT_ID5); }
                outPend_[slot] = 0U;
            }
            LocalTensor<T> dst =
                outTBuf_.template Get<T>()[slot * kOutTile];
            if constexpr (SameType<T, half>::value) {
                Cast(dst, src, RoundMode::CAST_NONE, static_cast<int32_t>(count));
            } else {
                Cast(dst, src, RoundMode::CAST_RINT, static_cast<int32_t>(count));
            }
            if (slot == 0U) {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
            } else {
                SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
            }
            const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
            DataCopyPad(y_[outIndex], dst, copy);
            if (slot == 0U) { SetFlag<HardEvent::MTE3_V>(EVENT_ID4); }
            else { SetFlag<HardEvent::MTE3_V>(EVENT_ID5); }
            outPend_[slot] = 1U;
            outSlot_ = slot ^ 1U;
        }
    }

    __aicore__ inline void DrainStore()
    {
        if constexpr (!SameType<T, float>::value) {
            if (outPend_[0] != 0U) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID4); outPend_[0] = 0U;
            }
            if (outPend_[1] != 0U) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID5); outPend_[1] = 0U;
            }
        }
    }

    __aicore__ inline void KeepInnerReduceMultiCore()
    {
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t keptTasks = taskCount_ / keepRedSeg_;
        const uint64_t share = taskCount_ / blocks;
        const uint64_t extra = taskCount_ % blocks;
        const uint64_t begin = block * share + (block < extra ? block : extra);
        const uint64_t end = begin + share + (block < extra ? 1 : 0);

        LocalTensor<float> acc = accBuf_.template Get<float>();
        LocalTensor<float> sq = sqBuf_.template Get<float>();
        LocalTensor<T> inbuf = inBuf_.template Get<T>();
        const DataCopyPadExtParams<T> pd;

        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t keptIdx = task / keepRedSeg_;
            const uint64_t redSegIdx = task % keepRedSeg_;
            const uint64_t keepIndex = keptIdx / innerChunkCount_;
            const uint64_t chunkIndex = keptIdx % innerChunkCount_;
            const uint64_t innerOffset = chunkIndex * innerChunk_;
            const uint32_t count = static_cast<uint32_t>(
                (innerLen_ - innerOffset) < innerChunk_ ? (innerLen_ - innerOffset) : innerChunk_);
            const uint64_t keepOffset = KeepOffset(keepIndex);
            const uint64_t rStart = redSegIdx * keepRedSegLen_;
            const uint64_t rEnd =
                (rStart + keepRedSegLen_ < prefixRedCount_) ? (rStart + keepRedSegLen_) : prefixRedCount_;

            Duplicate(acc, 0.0f, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            const int32_t evk[2] = {EVENT_ID0, EVENT_ID1};
            const DataCopyExtParams cp(1, count * sizeof(T), 0, 0, 0);

            uint32_t nReduceGroups = 0;
            uint64_t reduceStride = 0;
            for (uint32_t gi = 0; gi < groupCount_; ++gi) {
                if (groupIsReduce_[gi] != 0) {
                    ++nReduceGroups;
                    reduceStride = groupStride_[gi];
                }
            }
            const uint32_t rowBytes = count * static_cast<uint32_t>(sizeof(T));
            const uint32_t rowPitch =
                (((rowBytes + 31U) / 32U) * 32U) / static_cast<uint32_t>(sizeof(T));
            const uint32_t rowsPerTile = (rowPitch > 0) ? (tile_ / rowPitch) : 0U;
            const bool contiguousBatch =
                (nReduceGroups == 1) && (reduceStride == innerLen_) &&
                (count == innerLen_) && (innerOffset == 0) && (rowsPerTile >= 2) &&
                (rowPitch <= tile_);

            uint32_t rowGcd = rowBytes;
            uint32_t align = 32U;
            while (align != 0U) {
                const uint32_t next = rowGcd % align;
                rowGcd = align;
                align = next;
            }
            const uint32_t alignRows = 32U / rowGcd;
            const uint32_t superWidth = alignRows * count;
            const bool packedUnaligned =
                contiguousBatch && (rowPitch != count) && (superWidth <= tile_) &&
                ((rEnd - rStart) >= 2U * alignRows);

            if (packedUnaligned && rEnd > rStart) {

                Duplicate(acc, 0.0f, static_cast<int32_t>(superWidth));
                PipeBarrier<PIPE_V>();
                const uint64_t segmentRows = rEnd - rStart;
                const uint64_t fullRows = (segmentRows / alignRows) * alignRows;
                const uint32_t groupsPerTile = tile_ / superWidth;

                const uint32_t slots = static_cast<uint32_t>(
                    (2U * tile_) / superWidth > 4U ? 4U : (2U * tile_) / superWidth);
                const uint32_t slotN = slots >= 2U ? slots : 1U;
                const uint64_t rowsPerBatch = static_cast<uint64_t>(alignRows) *
                    (slotN >= 2U ? 1U : groupsPerTile);
                const uint64_t batches = (rowsPerBatch == 0U) ? 0U :
                    (fullRows + rowsPerBatch - 1U) / rowsPerBatch;
                const int32_t evp[4] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3};
                const uint32_t slotElems = (slotN >= 2U) ? superWidth : tile_;

                for (uint64_t pre = 0U; pre + 1U < slotN && pre < batches; ++pre) {
                    const uint64_t prb = rStart + pre * rowsPerBatch;
                    const uint64_t gl = (rStart + fullRows - prb) / alignRows;
                    const uint32_t g = static_cast<uint32_t>(
                        gl < (rowsPerBatch / alignRows) ? gl : (rowsPerBatch / alignRows));
                    const DataCopyExtParams cpp(
                        1, g * superWidth * sizeof(T), 0, 0, 0);
                    DataCopyPad(inbuf[pre * slotElems],
                                x_[keepOffset + ReduceOffset(prb) + innerOffset],
                                cpp, pd);
                    SetFlag<HardEvent::MTE2_V>(evp[pre]);
                }
                for (uint64_t batch = 0U; batch < batches; ++batch) {
                    const uint32_t cur = static_cast<uint32_t>(batch % slotN);
                    const uint64_t rb = rStart + batch * rowsPerBatch;
                    const uint64_t groupsLeft = (rStart + fullRows - rb) / alignRows;
                    const uint64_t gCap = rowsPerBatch / alignRows;
                    const uint32_t groups = static_cast<uint32_t>(
                        groupsLeft < gCap ? groupsLeft : gCap);
                    const uint32_t total = groups * superWidth;
                    const uint64_t issue = batch + slotN - 1U;
                    if (issue < batches) {
                        const uint32_t nxt = static_cast<uint32_t>(issue % slotN);
                        if (issue >= slotN) {
                            WaitFlag<HardEvent::V_MTE2>(evp[nxt]);
                        }
                        const uint64_t nrb = rStart + issue * rowsPerBatch;
                        const uint64_t nGroupsLeft = (rStart + fullRows - nrb) / alignRows;
                        const uint32_t nGroups = static_cast<uint32_t>(
                            nGroupsLeft < gCap ? nGroupsLeft : gCap);
                        const DataCopyExtParams ncpb(
                            1, nGroups * superWidth * sizeof(T), 0, 0, 0);
                        DataCopyPad(inbuf[nxt * slotElems],
                                    x_[keepOffset + ReduceOffset(nrb) + innerOffset],
                                    ncpb, pd);
                        SetFlag<HardEvent::MTE2_V>(evp[nxt]);
                    }
                    WaitFlag<HardEvent::MTE2_V>(evp[cur]);
                    LocalTensor<float> squared;
                    if constexpr (SameType<T, float>::value) {
                        LocalTensor<float> sf =
                            inBuf_.template Get<float>()[cur * slotElems];
                        Mul(sf, sf, sf, static_cast<int32_t>(total));
                        squared = sf;
                    } else {
                        Cast(sq, inbuf[cur * slotElems], RoundMode::CAST_NONE,
                             static_cast<int32_t>(total));
                        Mul(sq, sq, sq, static_cast<int32_t>(total));
                        squared = sq;
                    }
                    uint32_t w = groups;
                    while (w > 1U) {
                        const uint32_t h = w >> 1U;
                        Add(squared, squared, squared[h * superWidth],
                            static_cast<int32_t>(h * superWidth));
                        PipeBarrier<PIPE_V>();
                        if ((w & 1U) != 0U) {
                            Add(squared, squared, squared[(w - 1U) * superWidth],
                                static_cast<int32_t>(superWidth));
                            PipeBarrier<PIPE_V>();
                        }
                        w = h;
                    }
                    Add(acc, acc, squared, static_cast<int32_t>(superWidth));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(evp[cur]);
                }

                for (uint64_t d = 0U; d < slotN && d < batches; ++d) {
                    const uint32_t sl = static_cast<uint32_t>((batches - 1U - d) % slotN);
                    WaitFlag<HardEvent::V_MTE2>(evp[sl]);
                }
                uint64_t rb = rStart + fullRows;

                while (rb < rEnd) {
                    DataCopyPad(inbuf, x_[keepOffset + ReduceOffset(rb) + innerOffset], cp, pd);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    LocalTensor<float> squared;
                    if constexpr (SameType<T, float>::value) {
                        LocalTensor<float> sf = inBuf_.template Get<float>();
                        Mul(sf, sf, sf, static_cast<int32_t>(count));
                        squared = sf;
                    } else {
                        Cast(sq, inbuf, RoundMode::CAST_NONE, static_cast<int32_t>(count));
                        Mul(sq, sq, sq, static_cast<int32_t>(count));
                        squared = sq;
                    }
                    Add(acc, acc, squared, static_cast<int32_t>(count));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    ++rb;
                }

                const uint64_t partialFloats = keepRedSeg_ * outCount_;
                const uint64_t scratchBase = ((partialFloats + 7U) & ~7UL) + block * tile_;
                const DataCopyExtParams scratchStore(1, superWidth * sizeof(float), 0, 0, 0);
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                DataCopyPad(ws_[scratchBase], acc, scratchStore);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                const DataCopyExtParams scratchLoad(
                    static_cast<uint16_t>(alignRows), count * sizeof(float), 0, 0, 0);
                const DataCopyPadExtParams<float> scratchPad;
                DataCopyPad(sq, ws_[scratchBase], scratchLoad, scratchPad);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                const uint32_t scratchPitch = ((count * sizeof(float) + 31U) / 32U) * 8U;
                uint32_t w = alignRows;
                while (w > 1U) {
                    const uint32_t h = w >> 1U;
                    Add(sq, sq, sq[h * scratchPitch], static_cast<int32_t>(h * scratchPitch));
                    PipeBarrier<PIPE_V>();
                    w = h;
                }
                Adds(acc, sq, 0.0f, static_cast<int32_t>(count));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            } else if (contiguousBatch && rEnd > rStart) {

                const uint32_t B = rowsPerTile;
                const uint64_t nb = (rEnd - rStart + B - 1) / B;
                const bool rowAligned = (rowPitch == count);
                const int32_t batchEv[2] = {EVENT_ID2, EVENT_ID3};
                const uint32_t rows0 = static_cast<uint32_t>(
                    (rEnd - rStart) < B ? (rEnd - rStart) : B);
                const uint32_t total0 = rows0 * rowPitch;
                const DataCopyExtParams cp0(
                    rowAligned ? 1U : rows0,
                    rowAligned ? total0 * sizeof(T) : count * sizeof(T),
                    0, 0, 0);
                DataCopyPad(
                    inbuf, x_[keepOffset + ReduceOffset(rStart) + innerOffset], cp0, pd);
                SetFlag<HardEvent::MTE2_V>(batchEv[0]);
                for (uint64_t b = 0; b < nb; ++b) {
                    const uint32_t cur = static_cast<uint32_t>(b & 1U);
                    const uint64_t rb = rStart + b * B;
                    const uint32_t rows = static_cast<uint32_t>((rEnd - rb) < B ? (rEnd - rb) : B);
                    const uint32_t total = rows * rowPitch;
                    if (b + 1U < nb) {
                        const uint32_t nextSlot = static_cast<uint32_t>((b + 1U) & 1U);
                        const uint64_t nextRb = rb + B;
                        const uint32_t nextRows = static_cast<uint32_t>(
                            (rEnd - nextRb) < B ? (rEnd - nextRb) : B);
                        const uint32_t nextTotal = nextRows * rowPitch;
                        if (b + 1U >= 2U) {
                            WaitFlag<HardEvent::V_MTE2>(batchEv[nextSlot]);
                        }
                        const DataCopyExtParams nextCp(
                            rowAligned ? 1U : nextRows,
                            rowAligned ? nextTotal * sizeof(T) : count * sizeof(T),
                            0, 0, 0);
                        DataCopyPad(
                            inbuf[nextSlot * tile_],
                            x_[keepOffset + ReduceOffset(nextRb) + innerOffset],
                            nextCp, pd);
                        SetFlag<HardEvent::MTE2_V>(batchEv[nextSlot]);
                    }
                    WaitFlag<HardEvent::MTE2_V>(batchEv[cur]);
                    LocalTensor<float> squared;
                    if constexpr (SameType<T, float>::value) {
                        LocalTensor<float> sf = inBuf_.template Get<float>();
                        LocalTensor<float> current = sf[cur * tile_];
                        Mul(current, current, current, static_cast<int32_t>(total));
                        squared = current;
                    } else {
                        LocalTensor<T> src = inbuf[cur * tile_];
                        Cast(sq, src, RoundMode::CAST_NONE, static_cast<int32_t>(total));
                        Mul(sq, sq, sq, static_cast<int32_t>(total));
                        squared = sq;
                    }

                    {
                        uint32_t w = rows;
                        while (w > 1U) {
                            const uint32_t h = w >> 1U;
                            Add(squared, squared, squared[h * rowPitch],
                                static_cast<int32_t>(h * rowPitch));
                            PipeBarrier<PIPE_V>();
                            if ((w & 1U) != 0U) {
                                Add(squared, squared, squared[(w - 1U) * rowPitch],
                                    static_cast<int32_t>(rowPitch));
                                PipeBarrier<PIPE_V>();
                            }
                            w = h;
                        }
                        Add(acc, acc, squared, static_cast<int32_t>(count));
                    }
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(batchEv[cur]);
                }
                WaitFlag<HardEvent::V_MTE2>(batchEv[(nb - 1U) & 1U]);
                if (nb >= 2U) {
                    WaitFlag<HardEvent::V_MTE2>(batchEv[(nb - 2U) & 1U]);
                }
            } else if (rEnd > rStart) {
                DataCopyPad(inbuf, x_[keepOffset + ReduceOffset(rStart) + innerOffset], cp, pd);
                SetFlag<HardEvent::MTE2_V>(evk[0]);
                uint64_t idx = 0;
                for (uint64_t r = rStart; r < rEnd; ++r, ++idx) {
                    const uint32_t cur = static_cast<uint32_t>(idx & 1U);
                    if (r + 1 < rEnd) {
                        const uint32_t ns = static_cast<uint32_t>((idx + 1) & 1U);
                        if (idx + 1 >= 2) {
                            WaitFlag<HardEvent::V_MTE2>(evk[ns]);
                        }
                        DataCopyPad(inbuf[ns * tile_], x_[keepOffset + ReduceOffset(r + 1) + innerOffset], cp, pd);
                        SetFlag<HardEvent::MTE2_V>(evk[ns]);
                    }
                    WaitFlag<HardEvent::MTE2_V>(evk[cur]);
                    LocalTensor<float> squared;
                    if constexpr (SameType<T, float>::value) {
                        LocalTensor<float> sf = inBuf_.template Get<float>();
                        LocalTensor<float> ss = sf[cur * tile_];
                        Mul(ss, ss, ss, static_cast<int32_t>(count));
                        squared = ss;
                    } else {
                        LocalTensor<T> src = inbuf[cur * tile_];
                        Cast(sq, src, RoundMode::CAST_NONE, static_cast<int32_t>(count));
                        Mul(sq, sq, sq, static_cast<int32_t>(count));
                        squared = sq;
                    }
                    Add(acc, acc, squared, static_cast<int32_t>(count));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(evk[cur]);
                }
                const uint64_t doneTiles = rEnd - rStart;
                WaitFlag<HardEvent::V_MTE2>(evk[(doneTiles - 1) & 1U]);
                if (doneTiles >= 2) {
                    WaitFlag<HardEvent::V_MTE2>(evk[(doneTiles - 2) & 1U]);
                }
            }

            const uint64_t keptGlobal = keepIndex * innerLen_ + innerOffset;
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const DataCopyExtParams st(1, count * sizeof(float), 0, 0, 0);
            DataCopyPad(ws_[redSegIdx * outCount_ + keptGlobal], acc, st);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
        (void)keptTasks;

        LocalTensor<int32_t> ubSync = syncUbBuf_.template Get<int32_t>();
        SyncAll(syncGm_, ubSync, static_cast<int32_t>(blocks));

        const uint64_t share2 = outCount_ / blocks;
        const uint64_t extra2 = outCount_ % blocks;
        const uint64_t b2 = block * share2 + (block < extra2 ? block : extra2);
        const uint64_t e2 = b2 + share2 + (block < extra2 ? 1 : 0);
        LocalTensor<float> acc2 = accBuf_.template Get<float>();
        LocalTensor<float> segv = sqBuf_.template Get<float>();
        LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
        const DataCopyPadExtParams<float> pad2;
        for (uint64_t pos = b2; pos < e2; pos += kOutTile) {
            const uint32_t cnt = static_cast<uint32_t>((e2 - pos) < kOutTile ? (e2 - pos) : kOutTile);
            Duplicate(acc2, 0.0f, static_cast<int32_t>(cnt));
            PipeBarrier<PIPE_V>();
            for (uint64_t seg = 0; seg < keepRedSeg_; ++seg) {
                const DataCopyExtParams ld(1, cnt * sizeof(float), 0, 0, 0);
                DataCopyPad(segv, ws_[seg * outCount_ + pos], ld, pad2);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Add(acc2, acc2, segv, static_cast<int32_t>(cnt));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            Adds(outF32, acc2, 0.0f, static_cast<int32_t>(cnt));
            PipeBarrier<PIPE_V>();
            StoreOutput(pos, cnt);
        }
    }

    __aicore__ inline uint32_t RowPitchT() const
    {
        const uint32_t bytes = static_cast<uint32_t>(innerLen_) * static_cast<uint32_t>(sizeof(T));
        return (((bytes + 31U) / 32U) * 32U) / static_cast<uint32_t>(sizeof(T));
    }

    __aicore__ inline uint32_t FoldRowsForSum(
        const LocalTensor<float> &buf, uint32_t rows, uint32_t P, uint32_t n) const
    {
        constexpr uint32_t kLanes = 64U;
        constexpr uint32_t kMaxRep = 255U;
        if (n <= kLanes || rows == 0U || (P & 7U) != 0U || (P >> 3U) > kMaxRep) {
            return n;
        }
        const uint8_t pitchBlk = static_cast<uint8_t>(P >> 3U);
        BinaryRepeatParams rp;
        rp.dstBlkStride = 1U;
        rp.src0BlkStride = 1U;
        rp.src1BlkStride = 1U;
        rp.dstRepStride = pitchBlk;
        rp.src0RepStride = pitchBlk;
        rp.src1RepStride = pitchBlk;
        SetMaskNorm();
        uint32_t cur = n;
        while (cur > kLanes) {

            uint32_t half = (cur + 1U) >> 1U;
            half = ((half + kLanes - 1U) / kLanes) * kLanes;
            const uint32_t tail = cur - half;
            for (uint32_t off = 0U; off < tail; off += kLanes) {
                const uint32_t cnt = (tail - off) < kLanes ? (tail - off) : kLanes;
                for (uint32_t r0 = 0U; r0 < rows; r0 += kMaxRep) {
                    const uint32_t rep =
                        (rows - r0) < kMaxRep ? (rows - r0) : kMaxRep;
                    const uint64_t rowBase = static_cast<uint64_t>(r0) * P;
                    Add(buf[rowBase + off], buf[rowBase + off],
                        buf[rowBase + half + off],
                        static_cast<uint64_t>(cnt),
                        static_cast<uint8_t>(rep), rp);
                }
            }
            PipeBarrier<PIPE_V>();
            cur = half;
        }
        return kLanes;
    }

    __aicore__ inline void FastRowSum(
        const LocalTensor<float> &dst, const LocalTensor<float> &src,
        const LocalTensor<uint8_t> &tmp, uint32_t rows, uint32_t P, uint32_t n)
    {
        constexpr uint32_t kLanes = 64U;

        constexpr uint32_t kMaxRep = 248U;
        if (rows > 0U && n > 0U && n <= kLanes && (P & 7U) == 0U &&
            (P >> 3U) <= 255U) {
            SetMaskNorm();
            for (uint32_t r0 = 0U; r0 < rows; r0 += kMaxRep) {
                const uint32_t rep = (rows - r0) < kMaxRep ? (rows - r0) : kMaxRep;
                WholeReduceSum<float>(
                    dst[r0], src[static_cast<uint64_t>(r0) * P],
                    static_cast<int32_t>(n), static_cast<int32_t>(rep),
                    1, 1, static_cast<int32_t>(P >> 3U));
            }
            ResetMask();
            return;
        }
        SumParams sp;
        sp.outter = rows;
        sp.inner = P;
        sp.n = n;
        Sum<float>(dst, src, tmp, sp);
        SetMaskNorm();
        ResetMask();
    }

    __aicore__ inline bool ReduceInnerFastEligible() const
    {
        if (prefixRedCount_ != 1) {
            return false;
        }
        if (innerLen_ == 0) {
            return false;
        }

        if (RowPitchT() > tile_) {
            return false;
        }
        return true;
    }

    __aicore__ inline bool ReduceInnerPrefixBatchEligible() const
    {
        if (groupCount_ < 2U) {
            return false;
        }
        const uint32_t innerPrefix = groupCount_ - 1U;
        return prefixRedCount_ > 1 &&
            groupIsReduce_[innerPrefix] == 0U &&
            groupStride_[innerPrefix] == innerLen_ && RowPitchT() <= tile_;
    }

    __aicore__ inline void ProcessReduceInnerPrefixCompact(
        uint64_t begin, uint64_t end)
    {
        const uint32_t W = static_cast<uint32_t>(innerLen_);
        const uint32_t P = RowPitchT();
        uint32_t B = static_cast<uint32_t>(tile_ / P);
        if (B > kOutTile) {
            B = kOutTile;
        }
        if (B < 1U) {
            B = 1U;
        }
        LocalTensor<T> inputStorage = inBuf_.template Get<T>();
        LocalTensor<float> accPacked = sqBuf_.template Get<float>();
        LocalTensor<float> padded = accBuf_.template Get<float>();
        LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
        LocalTensor<int32_t> signedOffsets = workBuf_.template Get<int32_t>();
        LocalTensor<uint32_t> offsets =
            signedOffsets.template ReinterpretCast<uint32_t>();

        LocalTensor<uint8_t> tmp = sqBuf_.template Get<uint8_t>();
        const DataCopyPadExtParams<T> pad;
        const int32_t loadEvent[2] = {EVENT_ID2, EVENT_ID3};
        const uint64_t keptRun = groupLen_[groupCount_ - 1U];
        cachedOffsetRows_ = 0U;
        uint32_t reduceGroupCount = 0U;
        uint64_t singleReduceStride = 0U;
        for (uint32_t group = 0U; group < groupCount_; ++group) {
            if (groupIsReduce_[group] != 0U) {
                ++reduceGroupCount;
                singleReduceStride = groupStride_[group];
            }
        }
        const bool linearReduce = reduceGroupCount == 1U;

        uint64_t base = begin;
        while (base < end) {
            const uint64_t runLeft = keptRun - (base % keptRun);
            uint64_t batchRows = (end - base) < B ? (end - base) : B;
            if (batchRows > runLeft) {
                batchRows = runLeft;
            }
            const uint32_t rows = static_cast<uint32_t>(batchRows);
            const uint32_t compactTotal = rows * W;
            const uint32_t paddedTotal = rows * P;
            const uint32_t zeroIndex = compactTotal;

            if (rows != cachedOffsetRows_) {
                for (uint32_t lane = 0U; lane < P; ++lane) {
                    signedOffsets.SetValue(
                        lane,
                        static_cast<int32_t>(
                            (lane < W ? lane : zeroIndex) * sizeof(float)));
                }
                SetFlag<HardEvent::S_V>(EVENT_ID0);
                WaitFlag<HardEvent::S_V>(EVENT_ID0);
                for (uint32_t row = 1U; row < rows; ++row) {
                    Adds(
                        signedOffsets[row * P], signedOffsets,
                        static_cast<int32_t>(row * W * sizeof(float)),
                        static_cast<int32_t>(W));
                    for (uint32_t lane = W; lane < P; ++lane) {
                        signedOffsets.SetValue(
                            row * P + lane,
                            static_cast<int32_t>(zeroIndex * sizeof(float)));
                    }
                }
                PipeBarrier<PIPE_V>();
                cachedOffsetRows_ = rows;
            }

            Duplicate(accPacked, 0.0f, static_cast<int32_t>(paddedTotal));
            PipeBarrier<PIPE_V>();

            const uint64_t keepOffset = KeepOffset(base);
            const DataCopyExtParams copy(
                1, compactTotal * sizeof(T), 0, 0, 0);

            const uint32_t slotStride = ((compactTotal + 15U) / 16U) * 16U;
            const uint32_t inCap =
                (kPath4Mode == 0U && sizeof(T) == 2U && tile_ <= 8192U)
                ? (kDeepInBytes / static_cast<uint32_t>(sizeof(T)))
                : (2U * tile_);
            uint32_t kDeepR = slotStride == 0U ? 1U : inCap / slotStride;
            if (kDeepR > 8U) { kDeepR = 8U; }
            if (kDeepR < 1U) { kDeepR = 1U; }
            const int32_t evR[8] = {EVENT_ID0, EVENT_ID1, EVENT_ID2, EVENT_ID3,
                                    EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7};
            for (uint64_t p = 0U; p + 1U < kDeepR && p < prefixRedCount_; ++p) {
                const uint64_t off = linearReduce ? p * singleReduceStride
                                                  : ReduceOffset(p);
                DataCopyPad(inputStorage[(p % kDeepR) * slotStride],
                            x_[keepOffset + off], copy, pad);
                SetFlag<HardEvent::MTE2_V>(evR[p % kDeepR]);
            }
            for (uint64_t r = 0U; r < prefixRedCount_; ++r) {
                const uint32_t cur = static_cast<uint32_t>(r % kDeepR);
                const uint64_t issue = r + kDeepR - 1U;
                if (issue < prefixRedCount_) {
                    const uint32_t ns = static_cast<uint32_t>(issue % kDeepR);
                    if (issue >= kDeepR) {
                        WaitFlag<HardEvent::V_MTE2>(evR[ns]);
                    }
                    const uint64_t nextReduceOffset = linearReduce ?
                        issue * singleReduceStride : ReduceOffset(issue);
                    DataCopyPad(
                        inputStorage[ns * slotStride],
                        x_[keepOffset + nextReduceOffset], copy, pad);
                    SetFlag<HardEvent::MTE2_V>(evR[ns]);
                }
                WaitFlag<HardEvent::MTE2_V>(evR[cur]);
                LocalTensor<T> in = inputStorage[cur * slotStride];
                if constexpr (SameType<T, float>::value ||
                              SameType<T, half>::value) {

                    MulAddDst<float, T>(
                        accPacked, in, in, static_cast<int32_t>(compactTotal));
                    PipeBarrier<PIPE_V>();
                } else {

                    Cast(padded, in, RoundMode::CAST_NONE,
                         static_cast<int32_t>(compactTotal));
                    PipeBarrier<PIPE_V>();
                    MulAddDst<float, float>(
                        accPacked, padded, padded,
                        static_cast<int32_t>(compactTotal));
                    PipeBarrier<PIPE_V>();
                }
                SetFlag<HardEvent::V_MTE2>(evR[cur]);
            }
            {
                const uint64_t it = prefixRedCount_;
                const uint64_t dr = it < kDeepR ? it : kDeepR;
                for (uint64_t d = 0U; d < dr; ++d) {
                    WaitFlag<HardEvent::V_MTE2>(
                        evR[static_cast<uint32_t>((it - 1U - d) % kDeepR)]);
                }
            }

            Gather(padded, accPacked, offsets, 0U, paddedTotal);
            PipeBarrier<PIPE_V>();
            FastRowSum(outF32, padded, tmp, rows, P,
                       FoldRowsForSum(padded, rows, P, W));
            PipeBarrier<PIPE_V>();
            StoreOutput(base, rows);
            base += rows;
        }
    }

    __aicore__ inline void ProcessReduceInnerPrefixBatched(uint64_t begin, uint64_t end)
    {
        const uint32_t W = static_cast<uint32_t>(innerLen_);
        const uint32_t P = RowPitchT();

        if (P != W && W * sizeof(T) <= 512U && end - begin >= 8U) {
            ProcessReduceInnerPrefixCompact(begin, end);
            return;
        }
        uint32_t B = static_cast<uint32_t>(tile_ / P);
        if (B > kOutTile) {
            B = kOutTile;
        }
        LocalTensor<T> inbuf = inBuf_.template Get<T>();
        LocalTensor<float> sq = sqBuf_.template Get<float>();
        LocalTensor<float> acc = accBuf_.template Get<float>();
        LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
        LocalTensor<uint8_t> tmp = workBuf_.template Get<uint8_t>();
        const DataCopyPadExtParams<T> pd;
        const int32_t loadEvent[2] = {EVENT_ID2, EVENT_ID3};

        const uint64_t keptRun = groupLen_[groupCount_ - 1U];
        uint32_t reduceGroupCount = 0U;
        uint64_t singleReduceStride = 0U;
        for (uint32_t group = 0; group < groupCount_; ++group) {
            if (groupIsReduce_[group] != 0U) {
                ++reduceGroupCount;
                singleReduceStride = groupStride_[group];
            }
        }
        const bool linearReduce = reduceGroupCount == 1U;
        uint64_t base = begin;
        while (base < end) {
            const uint64_t runLeft = keptRun - (base % keptRun);
            uint64_t batchRows = (end - base) < B ? (end - base) : B;
            if (batchRows > runLeft) {
                batchRows = runLeft;
            }
            const uint32_t rows = static_cast<uint32_t>(batchRows);
            const uint32_t total = rows * P;
            Duplicate(acc, 0.0f, static_cast<int32_t>(total));
            PipeBarrier<PIPE_V>();
            const uint64_t keepOffset = KeepOffset(base);
            const DataCopyExtParams firstCopy(
                static_cast<uint16_t>(P == W ? 1U : rows),
                (P == W ? total : W) * sizeof(T), 0, 0, 0);
            DataCopyPad(inbuf, x_[keepOffset], firstCopy, pd);
            SetFlag<HardEvent::MTE2_V>(loadEvent[0]);
            for (uint64_t r = 0; r < prefixRedCount_; ++r) {
                const uint32_t current = static_cast<uint32_t>(r & 1U);
                if (r + 1U < prefixRedCount_) {
                    const uint32_t next = current ^ 1U;
                    if (r + 1U >= 2U) {
                        WaitFlag<HardEvent::V_MTE2>(loadEvent[next]);
                    }
                    const uint64_t nextReduceOffset = linearReduce ?
                        (r + 1U) * singleReduceStride : ReduceOffset(r + 1U);
                    const uint64_t nextOffset = keepOffset + nextReduceOffset;
                    const DataCopyExtParams nextCopy(
                        static_cast<uint16_t>(P == W ? 1U : rows),
                        (P == W ? total : W) * sizeof(T), 0, 0, 0);
                    DataCopyPad(inbuf[next * tile_], x_[nextOffset], nextCopy, pd);
                    SetFlag<HardEvent::MTE2_V>(loadEvent[next]);
                }
                WaitFlag<HardEvent::MTE2_V>(loadEvent[current]);
                LocalTensor<float> squared;
                if constexpr (SameType<T, float>::value) {
                    LocalTensor<float> sf = inBuf_.template Get<float>();
                    LocalTensor<float> currentInput = sf[current * tile_];
                    Mul(currentInput, currentInput, currentInput,
                        static_cast<int32_t>(total));
                    squared = currentInput;
                } else {
                    Cast(sq, inbuf[current * tile_], RoundMode::CAST_NONE,
                        static_cast<int32_t>(total));
                    Mul(sq, sq, sq, static_cast<int32_t>(total));
                    squared = sq;
                }
                Add(acc, acc, squared, static_cast<int32_t>(total));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(loadEvent[current]);
            }
            WaitFlag<HardEvent::V_MTE2>(
                loadEvent[(prefixRedCount_ - 1U) & 1U]);
            if (prefixRedCount_ >= 2U) {
                WaitFlag<HardEvent::V_MTE2>(
                    loadEvent[(prefixRedCount_ - 2U) & 1U]);
            }
            FastRowSum(outF32, acc, tmp, rows, P,
                       FoldRowsForSum(acc, rows, P, W));
            PipeBarrier<PIPE_V>();
            StoreOutput(base, rows);
            base += rows;
        }
    }

    __aicore__ inline void ProcessReduceInnerPrefixCyclic()
    {
        if constexpr (kPath4Mode != 0U) {
            const uint32_t W = static_cast<uint32_t>(innerLen_);
            const uint32_t P = kPath4Mode == 2U ? W : RowPitchT();
            uint32_t B = static_cast<uint32_t>(tile_ / P);
            if (B > kOutTile) {
                B = kOutTile;
            }
            if constexpr (kPath4Mode == 2U) {
                const uint32_t alignRows = 16U / GcdBlock(W);
                B = (B / alignRows) * alignRows;
            }
            const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
            const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
            const uint64_t keptRun = groupLen_[groupCount_ - 1U];
            const uint64_t groupsPerRun = (keptRun + B - 1U) / B;
            const uint64_t outerRuns = outCount_ / keptRun;
            const uint64_t totalGroups = outerRuns * groupsPerRun;
            LocalTensor<T> inbuf = inBuf_.template Get<T>();
            LocalTensor<float> sq = sqBuf_.template Get<float>();
            LocalTensor<float> acc = accBuf_.template Get<float>();
            LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
            LocalTensor<uint8_t> tmp = workBuf_.template Get<uint8_t>();
            const DataCopyPadExtParams<T> pad;
            const int32_t loadEvent[2] = {EVENT_ID2, EVENT_ID3};
            uint32_t reduceGroupCount = 0U;
            uint64_t singleReduceStride = 0U;
            for (uint32_t group = 0U; group < groupCount_; ++group) {
                if (groupIsReduce_[group] != 0U) {
                    ++reduceGroupCount;
                    singleReduceStride = groupStride_[group];
                }
            }
            const bool linearReduce = reduceGroupCount == 1U;

            for (uint64_t groupTask = block; groupTask < totalGroups;
                 groupTask += blocks) {
                const uint64_t outer = groupTask / groupsPerRun;
                const uint64_t group = groupTask - outer * groupsPerRun;
                const uint64_t base = outer * keptRun + group * B;
                const uint32_t rows = static_cast<uint32_t>(
                    (keptRun - group * B) < B ? (keptRun - group * B) : B);
                const uint32_t total = rows * P;
                Duplicate(acc, 0.0f, static_cast<int32_t>(total));
                PipeBarrier<PIPE_V>();
                const uint64_t keepOffset = KeepOffset(base);
                const DataCopyExtParams copy(
                    1U, total * sizeof(T), 0U, 0U, 0U);
                DataCopyPad(inbuf, x_[keepOffset], copy, pad);
                SetFlag<HardEvent::MTE2_V>(loadEvent[0]);
                for (uint64_t r = 0U; r < prefixRedCount_; ++r) {
                    const uint32_t current = static_cast<uint32_t>(r & 1U);
                    if (r + 1U < prefixRedCount_) {
                        const uint32_t next = current ^ 1U;
                        if (r + 1U >= 2U) {
                            WaitFlag<HardEvent::V_MTE2>(loadEvent[next]);
                        }
                        const uint64_t nextReduceOffset = linearReduce ?
                            (r + 1U) * singleReduceStride :
                            ReduceOffset(r + 1U);
                        DataCopyPad(
                            inbuf[next * tile_],
                            x_[keepOffset + nextReduceOffset], copy, pad);
                        SetFlag<HardEvent::MTE2_V>(loadEvent[next]);
                    }
                    WaitFlag<HardEvent::MTE2_V>(loadEvent[current]);
                    LocalTensor<float> squared;
                    if constexpr (SameType<T, float>::value) {
                        LocalTensor<float> input =
                            inBuf_.template Get<float>()[current * tile_];
                        Mul(input, input, input, static_cast<int32_t>(total));
                        squared = input;
                    } else {
                        Cast(
                            sq, inbuf[current * tile_], RoundMode::CAST_NONE,
                            static_cast<int32_t>(total));
                        Mul(sq, sq, sq, static_cast<int32_t>(total));
                        squared = sq;
                    }
                    Add(acc, acc, squared, static_cast<int32_t>(total));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(loadEvent[current]);
                }
                WaitFlag<HardEvent::V_MTE2>(
                    loadEvent[(prefixRedCount_ - 1U) & 1U]);
                WaitFlag<HardEvent::V_MTE2>(
                    loadEvent[(prefixRedCount_ - 2U) & 1U]);
                FastRowSum(outF32, acc, tmp, rows, P,
                           FoldRowsForSum(acc, rows, P, W));
                PipeBarrier<PIPE_V>();
                if constexpr (kPath4Mode == 2U) {
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
                }
                StoreOutput(base, rows);
            }
        }
    }

    __aicore__ inline void ProcessKeepInnerContiguousCompact()
    {
        if constexpr (kPath4Mode == 3U) {
            const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
            const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
            const uint64_t share = taskCount_ / blocks;
            const uint64_t extra = taskCount_ % blocks;
            const uint64_t begin = block * share +
                (block < extra ? block : extra);
            const uint64_t end = begin + share +
                (block < extra ? 1U : 0U);
            const uint32_t W = static_cast<uint32_t>(innerLen_);
            const uint32_t alignRows = 16U / GcdBlock(W);
            uint32_t B = static_cast<uint32_t>(tile_ / W);
            B = (B / alignRows) * alignRows;
            LocalTensor<T> inbuf = inBuf_.template Get<T>();
            LocalTensor<float> sq = sqBuf_.template Get<float>();
            LocalTensor<float> acc = accBuf_.template Get<float>();
            LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
            const DataCopyPadExtParams<T> pad;
            const int32_t loadEvent[2] = {EVENT_ID2, EVENT_ID3};

            for (uint64_t task = begin; task < end; ++task) {
                const uint64_t keepOffset = KeepOffset(task);
                Duplicate(acc, 0.0f, static_cast<int32_t>(W));
                PipeBarrier<PIPE_V>();
                const uint32_t rows0 = static_cast<uint32_t>(
                    prefixRedCount_ < B ? prefixRedCount_ : B);
                const DataCopyExtParams firstCopy(
                    1U, rows0 * W * sizeof(T), 0U, 0U, 0U);
                DataCopyPad(inbuf, x_[keepOffset], firstCopy, pad);
                SetFlag<HardEvent::MTE2_V>(loadEvent[0]);
                const uint64_t batches =
                    (prefixRedCount_ + B - 1U) / B;
                for (uint64_t batch = 0U; batch < batches; ++batch) {
                    const uint32_t current =
                        static_cast<uint32_t>(batch & 1U);
                    const uint64_t rb = batch * B;
                    const uint32_t rows = static_cast<uint32_t>(
                        (prefixRedCount_ - rb) < B ?
                        (prefixRedCount_ - rb) : B);
                    if (batch + 1U < batches) {
                        const uint32_t next = current ^ 1U;
                        if (batch + 1U >= 2U) {
                            WaitFlag<HardEvent::V_MTE2>(loadEvent[next]);
                        }
                        const uint64_t nextRb = rb + B;
                        const uint32_t nextRows = static_cast<uint32_t>(
                            (prefixRedCount_ - nextRb) < B ?
                            (prefixRedCount_ - nextRb) : B);
                        const DataCopyExtParams nextCopy(
                            1U, nextRows * W * sizeof(T), 0U, 0U, 0U);
                        DataCopyPad(
                            inbuf[next * tile_],
                            x_[keepOffset + ReduceOffset(nextRb)],
                            nextCopy, pad);
                        SetFlag<HardEvent::MTE2_V>(loadEvent[next]);
                    }
                    WaitFlag<HardEvent::MTE2_V>(loadEvent[current]);
                    const uint32_t total = rows * W;
                    Cast(
                        sq, inbuf[current * tile_], RoundMode::CAST_NONE,
                        static_cast<int32_t>(total));
                    PipeBarrier<PIPE_V>();
                    Mul(sq, sq, sq, static_cast<int32_t>(total));
                    PipeBarrier<PIPE_V>();
                    uint32_t lanes = rows;
                    while (lanes > 1U) {
                        const uint32_t half = lanes >> 1U;
                        Add(sq, sq, sq[half * W],
                            static_cast<int32_t>(half * W));
                        PipeBarrier<PIPE_V>();
                        if ((lanes & 1U) != 0U) {
                            Add(sq, sq, sq[(lanes - 1U) * W],
                                static_cast<int32_t>(W));
                            PipeBarrier<PIPE_V>();
                        }
                        lanes = half;
                    }
                    Add(acc, acc, sq, static_cast<int32_t>(W));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(loadEvent[current]);
                }
                WaitFlag<HardEvent::V_MTE2>(
                    loadEvent[(batches - 1U) & 1U]);
                if (batches >= 2U) {
                    WaitFlag<HardEvent::V_MTE2>(
                        loadEvent[(batches - 2U) & 1U]);
                }
                for (uint32_t done = 0U; done < W; done += kOutTile) {
                    const uint32_t part = (W - done) < kOutTile ?
                        (W - done) : kOutTile;
                    Adds(outF32, acc[done], 0.0f,
                        static_cast<int32_t>(part));
                    PipeBarrier<PIPE_V>();
                    StoreOutput(task * innerLen_ + done, part);
                }
            }
        }
    }

    __aicore__ inline void ProcessKeepInnerCyclic()
    {
        if constexpr ((kPath4Mode == 4U && SameType<T, float>::value) ||
                      (kPath4Mode == 5U && !SameType<T, float>::value)) {
            const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
            const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
            const uint32_t W = static_cast<uint32_t>(innerLen_);
            uint32_t B = static_cast<uint32_t>(tile_ / W);
            if constexpr (kPath4Mode == 5U) {
                const uint32_t alignRows = 16U / GcdBlock(W);
                B = (B / alignRows) * alignRows;
            }
            const uint64_t groups = (prefixRedCount_ + B - 1U) / B;
            LocalTensor<T> inbuf = inBuf_.template Get<T>();
            LocalTensor<float> acc = accBuf_.template Get<float>();
            LocalTensor<float> sq = sqBuf_.template Get<float>();
            LocalTensor<float> segv = sqBuf_.template Get<float>();
            LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
            const DataCopyPadExtParams<T> pad;
            const DataCopyPadExtParams<float> floatPad;
            const int32_t loadEvent[2] = {EVENT_ID2, EVENT_ID3};
            Duplicate(acc, 0.0f, static_cast<int32_t>(W));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(loadEvent[0]);
            SetFlag<HardEvent::V_MTE2>(loadEvent[1]);

            uint64_t group = block;
            uint64_t ordinal = 0U;
            if (group < groups) {
                const uint64_t rb = group * B;
                const uint32_t rows = static_cast<uint32_t>(
                    (prefixRedCount_ - rb) < B ?
                    (prefixRedCount_ - rb) : B);
                WaitFlag<HardEvent::V_MTE2>(loadEvent[0]);
                const DataCopyExtParams firstCopy(
                    1U, rows * W * sizeof(T), 0U, 0U, 0U);
                DataCopyPad(inbuf, x_[rb * W], firstCopy, pad);
                SetFlag<HardEvent::MTE2_V>(loadEvent[0]);
            }
            while (group < groups) {
                const uint32_t current =
                    static_cast<uint32_t>(ordinal & 1U);
                const uint64_t rb = group * B;
                const uint32_t rows = static_cast<uint32_t>(
                    (prefixRedCount_ - rb) < B ?
                    (prefixRedCount_ - rb) : B);
                const uint64_t nextGroup = group + blocks;
                if (nextGroup < groups) {
                    const uint32_t next = current ^ 1U;
                    const uint64_t nextRb = nextGroup * B;
                    const uint32_t nextRows = static_cast<uint32_t>(
                        (prefixRedCount_ - nextRb) < B ?
                        (prefixRedCount_ - nextRb) : B);
                    WaitFlag<HardEvent::V_MTE2>(loadEvent[next]);
                    const DataCopyExtParams nextCopy(
                        1U, nextRows * W * sizeof(T), 0U, 0U, 0U);
                    DataCopyPad(
                        inbuf[next * tile_], x_[nextRb * W], nextCopy, pad);
                    SetFlag<HardEvent::MTE2_V>(loadEvent[next]);
                }
                WaitFlag<HardEvent::MTE2_V>(loadEvent[current]);
                const uint32_t total = rows * W;
                LocalTensor<float> squared;
                if constexpr (SameType<T, float>::value) {
                    squared = inBuf_.template Get<float>()[current * tile_];
                } else {
                    Cast(
                        sq, inbuf[current * tile_], RoundMode::CAST_NONE,
                        static_cast<int32_t>(total));
                    PipeBarrier<PIPE_V>();
                    squared = sq;
                }
                Mul(squared, squared, squared, static_cast<int32_t>(total));
                PipeBarrier<PIPE_V>();
                uint32_t lanes = rows;
                while (lanes > 1U) {
                    const uint32_t half = lanes >> 1U;
                    Add(squared, squared, squared[half * W],
                        static_cast<int32_t>(half * W));
                    PipeBarrier<PIPE_V>();
                    if ((lanes & 1U) != 0U) {
                        Add(squared, squared, squared[(lanes - 1U) * W],
                            static_cast<int32_t>(W));
                        PipeBarrier<PIPE_V>();
                    }
                    lanes = half;
                }
                Add(acc, acc, squared, static_cast<int32_t>(W));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(loadEvent[current]);
                group = nextGroup;
                ++ordinal;
            }
            WaitFlag<HardEvent::V_MTE2>(loadEvent[0]);
            WaitFlag<HardEvent::V_MTE2>(loadEvent[1]);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const DataCopyExtParams partialStore(
                1U, W * sizeof(float), 0U, 0U, 0U);
            DataCopyPad(ws_[block * outCount_], acc, partialStore);
            SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
            SyncAll();

            const uint64_t share = outCount_ / blocks;
            const uint64_t extra = outCount_ % blocks;
            const uint64_t begin = block * share +
                (block < extra ? block : extra);
            const uint64_t end = begin + share +
                (block < extra ? 1U : 0U);
            for (uint64_t pos = begin; pos < end; pos += kOutTile) {
                const uint32_t count = static_cast<uint32_t>(
                    (end - pos) < kOutTile ? (end - pos) : kOutTile);
                Duplicate(acc, 0.0f, static_cast<int32_t>(count));
                PipeBarrier<PIPE_V>();
                for (uint64_t core = 0U; core < blocks; ++core) {
                    const DataCopyExtParams load(
                        1U, count * sizeof(float), 0U, 0U, 0U);
                    DataCopyPad(
                        segv, ws_[core * outCount_ + pos], load, floatPad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    Add(acc, acc, segv, static_cast<int32_t>(count));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
                Adds(outF32, acc, 0.0f, static_cast<int32_t>(count));
                PipeBarrier<PIPE_V>();
                StoreOutput(pos, count);
            }
        }
    }

    __aicore__ inline void ProcessReduceInnerBatched(uint64_t begin, uint64_t end)
    {
        const uint32_t W = static_cast<uint32_t>(innerLen_);
        const uint32_t P = RowPitchT();
        uint32_t B = static_cast<uint32_t>(tile_ / P);
        if (B > kOutTile) {
            B = kOutTile;
        }
        if (B < 1U) {
            B = 1U;
        }
        LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
        LocalTensor<uint8_t> tmp = workBuf_.template Get<uint8_t>();
        LocalTensor<T> inbuf = inBuf_.template Get<T>();
        const DataCopyPadExtParams<T> pd;

        const int32_t evk[2] = {EVENT_ID2, EVENT_ID3};
        if (begin >= end) {
            return;
        }

        {
            const uint32_t rows0 = static_cast<uint32_t>((end - begin) < B ? (end - begin) : B);

            const DataCopyExtParams cp0(static_cast<uint16_t>(rows0), W * sizeof(T), 0, 0, 0);
            DataCopyPad(inbuf, x_[KeepOffset(begin)], cp0, pd);
            SetFlag<HardEvent::MTE2_V>(evk[0]);
        }
        uint64_t out = begin;
        uint32_t idx = 0;
        while (out < end) {
            const uint32_t cur = static_cast<uint32_t>(idx & 1U);
            const uint32_t rows = static_cast<uint32_t>((end - out) < B ? (end - out) : B);
            const uint32_t total = rows * P;
            const uint64_t next = out + rows;
            if (next < end) {
                const uint32_t ns = static_cast<uint32_t>((idx + 1U) & 1U);
                const uint32_t rowsN = static_cast<uint32_t>((end - next) < B ? (end - next) : B);
                if (idx + 1U >= 2U) {
                    WaitFlag<HardEvent::V_MTE2>(evk[ns]);
                }
                const DataCopyExtParams cpn(static_cast<uint16_t>(rowsN), W * sizeof(T), 0, 0, 0);
                DataCopyPad(inbuf[ns * tile_], x_[KeepOffset(next)], cpn, pd);
                SetFlag<HardEvent::MTE2_V>(evk[ns]);
            }
            WaitFlag<HardEvent::MTE2_V>(evk[cur]);
            LocalTensor<float> sq;
            if constexpr (SameType<T, float>::value) {
                LocalTensor<float> sf = inBuf_.template Get<float>();
                sq = sf[cur * tile_];
                Mul(sq, sq, sq, static_cast<int32_t>(total));
            } else {
                sq = sqBuf_.template Get<float>();
                Cast(sq, inbuf[cur * tile_], RoundMode::CAST_NONE, static_cast<int32_t>(total));
                Mul(sq, sq, sq, static_cast<int32_t>(total));
            }
            PipeBarrier<PIPE_V>();
            FastRowSum(outF32, sq, tmp, rows, P,
                       FoldRowsForSum(sq, rows, P, W));
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(evk[cur]);
            StoreOutput(out, rows);
            out = next;
            ++idx;
        }
        WaitFlag<HardEvent::V_MTE2>(evk[(idx - 1U) & 1U]);
        if (idx >= 2U) {
            WaitFlag<HardEvent::V_MTE2>(evk[(idx - 2U) & 1U]);
        }
    }
    __aicore__ inline void ProcessReduceInner(uint64_t begin, uint64_t end)
    {
        if (ReduceInnerFastEligible()) {
            ProcessReduceInnerBatched(begin, end);
            return;
        }
        if (ReduceInnerPrefixBatchEligible()) {
            ProcessReduceInnerPrefixBatched(begin, end);
            return;
        }

        LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
        LocalTensor<float> acc = accBuf_.template Get<float>();
        LocalTensor<float> work = workBuf_.template Get<float>();
        LocalTensor<float> red = redBuf_.template Get<float>();
        uint64_t base = begin;
        while (base < end) {
            const uint64_t remaining = end - base;
            const uint32_t batch = static_cast<uint32_t>(remaining < kOutTile ? remaining : kOutTile);
            for (uint32_t j = 0; j < batch; ++j) {
                const uint64_t keepOffset = KeepOffset(base + j);
                Duplicate(acc, 0.0f, static_cast<int32_t>(tile_));
                PipeBarrier<PIPE_V>();
                for (uint64_t r = 0; r < prefixRedCount_; ++r) {
                    const uint64_t runOffset = keepOffset + ReduceOffset(r);
                    for (uint64_t done = 0; done < innerLen_; done += tile_) {
                        const uint64_t left = innerLen_ - done;
                        const uint32_t count =
                            static_cast<uint32_t>(left < tile_ ? left : tile_);
                        LoadTile(runOffset + done, count);
                        LocalTensor<float> squared = SquareTile(count);

                        Add(acc, acc, squared, static_cast<int32_t>(count));
                        PipeBarrier<PIPE_V>();
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    }
                }
                ReduceSum<float>(red, acc, work, static_cast<int32_t>(tile_));
                SetFlag<HardEvent::V_S>(EVENT_ID0);
                WaitFlag<HardEvent::V_S>(EVENT_ID0);
                outF32.SetValue(j, red.GetValue(0));
            }
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            PipeBarrier<PIPE_ALL>();
            StoreOutput(base, batch);
            base += batch;
        }
    }

    __aicore__ inline uint32_t GcdBlock(uint32_t value) const
    {
        uint32_t divisor = 32U;
        while (divisor != 0U) {
            const uint32_t next = value % divisor;
            value = divisor;
            divisor = next;
        }
        return value;
    }

    __aicore__ inline bool KeepInnerSuperBatchEligible() const
    {

        if (slabScratch_ != 1U || groupCount_ < 3U ||
            innerChunkCount_ != 1U || innerLen_ == 0U || innerLen_ > 64U) {
            return false;
        }
        const uint32_t innerPrefix = groupCount_ - 1U;
        const uint32_t keptPrefix = groupCount_ - 2U;
        if (groupIsReduce_[innerPrefix] == 0U ||
            groupIsReduce_[keptPrefix] != 0U ||
            groupStride_[innerPrefix] != innerLen_) {
            return false;
        }
        const uint32_t W = static_cast<uint32_t>(innerLen_);
        const uint32_t reduceRows = static_cast<uint32_t>(groupLen_[innerPrefix]);
        const uint32_t alignRows = 32U / GcdBlock(W * sizeof(T));
        const uint32_t superWidth = alignRows * W;
        const uint32_t floatPitch = ((W * sizeof(float) + 31U) / 32U) * 8U;
        const uint32_t inputPitch =
            ((W * sizeof(T) + 31U) / 32U) * (32U / sizeof(T));
        return reduceRows >= 64U && reduceRows / alignRows >= 2U &&
            superWidth <= tile_ && alignRows * floatPitch <= tile_ &&
            inputPitch <= tile_ &&
            groupStride_[keptPrefix] >= superWidth &&
            (groupStride_[keptPrefix] - W) <=
                (0xffffffffULL / static_cast<uint32_t>(sizeof(T)));
    }

    __aicore__ inline void ProcessKeepInnerSuperBatch(uint64_t begin, uint64_t end)
    {
        const uint32_t innerPrefix = groupCount_ - 1U;
        const uint32_t keptPrefix = groupCount_ - 2U;
        const uint32_t W = static_cast<uint32_t>(innerLen_);
        const uint32_t reduceRows = static_cast<uint32_t>(groupLen_[innerPrefix]);
        const uint32_t alignRows = 32U / GcdBlock(W * sizeof(T));
        const uint32_t fullRows = (reduceRows / alignRows) * alignRows;
        const uint32_t tailRows = reduceRows - fullRows;
        const uint32_t reduceGroups = fullRows / alignRows;
        const uint32_t superWidth = alignRows * W;
        const uint32_t floatPitch = ((W * sizeof(float) + 31U) / 32U) * 8U;
        const uint32_t batchCap = tile_ / superWidth;
        const uint64_t keptRun = groupLen_[keptPrefix];
        const uint64_t keepStride = groupStride_[keptPrefix];
        const uint64_t outerRedCount = prefixRedCount_ / reduceRows;
        const uint32_t superGap =
            static_cast<uint32_t>((keepStride - superWidth) * sizeof(T));
        const uint64_t scratchBase = static_cast<uint64_t>(GetBlockIdx()) * tile_;
        LocalTensor<T> inbuf = inBuf_.template Get<T>();
        LocalTensor<float> sq = sqBuf_.template Get<float>();
        LocalTensor<float> acc = accBuf_.template Get<float>();
        LocalTensor<float> work = workBuf_.template Get<float>();
        LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
        const DataCopyPadExtParams<T> inputPad;
        const DataCopyPadExtParams<float> floatPad;

        uint64_t base = begin;
        while (base < end) {
            const uint64_t runLeft = keptRun - (base % keptRun);
            uint64_t batchRows = (end - base) < batchCap ? (end - base) : batchCap;
            if (batchRows > runLeft) {
                batchRows = runLeft;
            }
            const uint32_t rows = static_cast<uint32_t>(batchRows);
            const uint32_t accTotal = rows * superWidth;
            Duplicate(acc, 0.0f, static_cast<int32_t>(accTotal));
            PipeBarrier<PIPE_V>();
            for (uint64_t outerRed = 0; outerRed < outerRedCount; ++outerRed) {
                const uint64_t outerBase = outerRed * reduceRows;
                for (uint32_t group = 0; group < reduceGroups; ++group) {
                    const uint64_t reduceIndex = outerBase + group * alignRows;
                    const uint64_t inputOffset = KeepOffset(base) + ReduceOffset(reduceIndex);
                    const DataCopyExtParams cp(
                        static_cast<uint16_t>(rows), superWidth * sizeof(T),
                        superGap, 0, 0);
                    DataCopyPad(inbuf, x_[inputOffset], cp, inputPad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    LocalTensor<float> squared;
                    if constexpr (SameType<T, float>::value) {
                        LocalTensor<float> sf = inBuf_.template Get<float>();
                        Mul(sf, sf, sf, static_cast<int32_t>(accTotal));
                        squared = sf;
                    } else {
                        Cast(sq, inbuf, RoundMode::CAST_NONE,
                            static_cast<int32_t>(accTotal));
                        Mul(sq, sq, sq, static_cast<int32_t>(accTotal));
                        squared = sq;
                    }
                    Add(acc, acc, squared, static_cast<int32_t>(accTotal));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }

                if (tailRows != 0U) {
                    const uint32_t tailWidth = tailRows * W;
                    const uint32_t tailPitch =
                        ((tailWidth * sizeof(T) + 31U) / 32U) *
                        (32U / sizeof(T));
                    const uint64_t reduceIndex = outerBase + fullRows;
                    const uint64_t inputOffset =
                        KeepOffset(base) + ReduceOffset(reduceIndex);
                    const uint32_t packedTailGap = static_cast<uint32_t>(
                        (keepStride - tailWidth) * sizeof(T));
                    const DataCopyExtParams cp(
                        static_cast<uint16_t>(rows), tailWidth * sizeof(T),
                        packedTailGap, 0, 0);
                    DataCopyPad(inbuf, x_[inputOffset], cp, inputPad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    LocalTensor<float> squared;
                    if constexpr (SameType<T, float>::value) {
                        LocalTensor<float> sf = inBuf_.template Get<float>();
                        Mul(sf, sf, sf, static_cast<int32_t>(rows * tailPitch));
                        squared = sf;
                    } else {
                        Cast(sq, inbuf, RoundMode::CAST_NONE,
                            static_cast<int32_t>(rows * tailPitch));
                        Mul(sq, sq, sq, static_cast<int32_t>(rows * tailPitch));
                        squared = sq;
                    }
                    const BinaryRepeatParams addPackedTail{
                        1, 1, 1, static_cast<uint8_t>(superWidth / 8U),
                        static_cast<uint8_t>(superWidth / 8U),
                        static_cast<uint8_t>(tailPitch / 8U)};
                    for (uint32_t done = 0U; done < tailWidth; done += 64U) {
                        const uint32_t part =
                            (tailWidth - done) < 64U ? (tailWidth - done) : 64U;
                        Add(
                            acc[done], acc[done], squared[done],
                            static_cast<uint64_t>(part),
                            static_cast<uint8_t>(rows), addPackedTail);
                    }
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
            }

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            const DataCopyExtParams scratchStore(
                1, accTotal * sizeof(float), 0, 0, 0);
            DataCopyPad(ws_[scratchBase], acc, scratchStore);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            const DataCopyExtParams scratchLoad(
                static_cast<uint16_t>(rows * alignRows),
                W * sizeof(float), 0, 0, 0);
            DataCopyPad(work, ws_[scratchBase], scratchLoad, floatPad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

            const uint32_t rowSpan = alignRows * floatPitch;
            for (uint32_t row = 0; row < rows; ++row) {
                LocalTensor<float> reduced = work[row * rowSpan];
                uint32_t lanes = alignRows;
                while (lanes > 1U) {
                    const uint32_t halfLanes = lanes >> 1U;
                    Add(reduced, reduced, reduced[halfLanes * floatPitch],
                        static_cast<int32_t>(halfLanes * floatPitch));
                    PipeBarrier<PIPE_V>();
                    lanes = halfLanes;
                }
                for (uint32_t done = 0; done < W; done += kOutTile) {
                    const uint32_t part =
                        (W - done) < kOutTile ? (W - done) : kOutTile;
                    Adds(outF32, reduced[done], 0.0f, static_cast<int32_t>(part));
                    PipeBarrier<PIPE_V>();
                    StoreOutput((base + row) * innerLen_ + done, part);
                }
            }
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            base += rows;
        }
    }

    __aicore__ inline bool KeepInnerPrefixBatchEligible() const
    {
        if (groupCount_ < 3U || innerChunkCount_ != 1U || innerLen_ == 0U) {
            return false;
        }
        const uint32_t innerPrefix = groupCount_ - 1U;
        const uint32_t keptPrefix = groupCount_ - 2U;
        if (groupIsReduce_[innerPrefix] == 0U ||
            groupIsReduce_[keptPrefix] != 0U) {
            return false;
        }
        const uint32_t bytes = static_cast<uint32_t>(innerLen_) * sizeof(T);
        const uint32_t pitch = ((bytes + 31U) / 32U) * (32U / sizeof(T));
        return pitch <= tile_ &&
            groupStride_[keptPrefix] >= innerLen_ &&
            (groupStride_[keptPrefix] - innerLen_) <=
                (0xffffffffULL / static_cast<uint32_t>(sizeof(T)));
    }

    __aicore__ inline void ProcessKeepInnerPrefixBatched(uint64_t begin, uint64_t end)
    {
        const uint32_t keptPrefix = groupCount_ - 2U;
        const uint32_t W = static_cast<uint32_t>(innerLen_);
        const uint32_t rowBytes = W * static_cast<uint32_t>(sizeof(T));
        const uint32_t P = ((rowBytes + 31U) / 32U) * (32U / sizeof(T));
        const uint32_t B = tile_ / P;
        const uint64_t keptRun = groupLen_[keptPrefix];
        const uint64_t keepStride = groupStride_[keptPrefix];
        const uint32_t srcGap =
            static_cast<uint32_t>((keepStride - innerLen_) * sizeof(T));
        LocalTensor<T> inbuf = inBuf_.template Get<T>();
        LocalTensor<float> sq = sqBuf_.template Get<float>();
        LocalTensor<float> acc = accBuf_.template Get<float>();
        LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
        const DataCopyPadExtParams<T> pd;

        uint64_t base = begin;
        while (base < end) {
            const uint64_t runLeft = keptRun - (base % keptRun);
            uint64_t batchRows = (end - base) < B ? (end - base) : B;
            if (batchRows > runLeft) {
                batchRows = runLeft;
            }
            const uint32_t rows = static_cast<uint32_t>(batchRows);
            const uint32_t total = rows * P;
            Duplicate(acc, 0.0f, static_cast<int32_t>(total));
            PipeBarrier<PIPE_V>();
            for (uint64_t r = 0; r < prefixRedCount_; ++r) {
                const uint64_t offset = KeepOffset(base) + ReduceOffset(r);
                const DataCopyExtParams cp(
                    static_cast<uint16_t>(rows), W * sizeof(T), srcGap, 0, 0);
                DataCopyPad(inbuf, x_[offset], cp, pd);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                LocalTensor<float> squared;
                if constexpr (SameType<T, float>::value) {
                    LocalTensor<float> sf = inBuf_.template Get<float>();
                    Mul(sf, sf, sf, static_cast<int32_t>(total));
                    squared = sf;
                } else {
                    Cast(sq, inbuf, RoundMode::CAST_NONE, static_cast<int32_t>(total));
                    Mul(sq, sq, sq, static_cast<int32_t>(total));
                    squared = sq;
                }
                Add(acc, acc, squared, static_cast<int32_t>(total));
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
            }
            if (P == W) {
                for (uint32_t done = 0; done < total; done += kOutTile) {
                    const uint32_t part =
                        (total - done) < kOutTile ? (total - done) : kOutTile;
                    Adds(outF32, acc[done], 0.0f, static_cast<int32_t>(part));
                    PipeBarrier<PIPE_V>();
                    StoreOutput(base * innerLen_ + done, part);
                }
            } else {
                for (uint32_t row = 0; row < rows; ++row) {
                    for (uint32_t done = 0; done < W; done += kOutTile) {
                        const uint32_t part =
                            (W - done) < kOutTile ? (W - done) : kOutTile;
                        Adds(outF32, acc[row * P + done], 0.0f,
                            static_cast<int32_t>(part));
                        PipeBarrier<PIPE_V>();
                        StoreOutput((base + row) * innerLen_ + done, part);
                    }
                }
            }
            base += rows;
        }
    }

    __aicore__ inline void ProcessKeepInner(uint64_t begin, uint64_t end)
    {
        if (KeepInnerSuperBatchEligible()) {
            ProcessKeepInnerSuperBatch(begin, end);
            return;
        }
        if (KeepInnerPrefixBatchEligible()) {
            ProcessKeepInnerPrefixBatched(begin, end);
            return;
        }
        LocalTensor<float> acc = accBuf_.template Get<float>();
        LocalTensor<float> outF32 = outF32Buf_.template Get<float>();
        uint32_t nReduceGroups = 0;
        uint64_t reduceStride = 0;
        for (uint32_t gi = 0; gi < groupCount_; ++gi) {
            if (groupIsReduce_[gi] != 0U) {
                ++nReduceGroups;
                reduceStride = groupStride_[gi];
            }
        }
        const uint32_t fullCount = static_cast<uint32_t>(innerLen_);
        const uint32_t rowBytes = fullCount * static_cast<uint32_t>(sizeof(T));
        const uint32_t rowPitch =
            (((rowBytes + 31U) / 32U) * 32U) / static_cast<uint32_t>(sizeof(T));
        const uint32_t rowsPerTile = rowPitch > 0U ? tile_ / rowPitch : 0U;
        const bool contiguousBatch =
            nReduceGroups == 1U && reduceStride == innerLen_ &&
            innerChunkCount_ == 1U && rowsPerTile >= 2U && rowPitch <= tile_;
        for (uint64_t task = begin; task < end; ++task) {
            const uint64_t keepIndex = task / innerChunkCount_;
            const uint64_t chunkIndex = task % innerChunkCount_;
            const uint64_t innerOffset = chunkIndex * innerChunk_;
            const uint64_t left = innerLen_ - innerOffset;
            const uint32_t count = static_cast<uint32_t>(left < innerChunk_ ? left : innerChunk_);
            const uint64_t keepOffset = KeepOffset(keepIndex);
            Duplicate(acc, 0.0f, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            if (contiguousBatch) {
                LocalTensor<T> inbuf = inBuf_.template Get<T>();
                LocalTensor<float> sq = sqBuf_.template Get<float>();
                const DataCopyPadExtParams<T> pd;
                const bool rowAligned = rowPitch == count;

                uint32_t rowGcd = count * static_cast<uint32_t>(sizeof(T));
                uint32_t alignStep = 32U;
                while (alignStep != 0U) {
                    const uint32_t nx = rowGcd % alignStep;
                    rowGcd = alignStep;
                    alignStep = nx;
                }
                const uint32_t alignRows = rowGcd == 0U ? 1U : (32U / rowGcd);
                const uint32_t superWidth = alignRows * count;
                const uint32_t scratchPitch =
                    ((count * sizeof(float) + 31U) / 32U) * 8U;
                const bool packedRows =
                    slabScratch_ != 0U && !rowAligned && alignRows >= 2U &&
                    superWidth <= tile_ && alignRows * scratchPitch <= tile_ &&
                    prefixRedCount_ >= 2U * static_cast<uint64_t>(alignRows);
                if (packedRows) {

                    const uint32_t accSpanAll =
                        (tile_ / superWidth) * superWidth;
                    Duplicate(acc, 0.0f, static_cast<int32_t>(accSpanAll));
                    PipeBarrier<PIPE_V>();
                    const uint64_t fullRows =
                        (prefixRedCount_ / alignRows) * alignRows;
                    const uint32_t groupsPerTile = tile_ / superWidth;

                    const uint64_t rowsPerBatch =
                        static_cast<uint64_t>(groupsPerTile) * alignRows;
                    const uint64_t nBatch = rowsPerBatch == 0U ? 0U :
                        (fullRows + rowsPerBatch - 1U) / rowsPerBatch;
                    const int32_t evk2[2] = {EVENT_ID0, EVENT_ID1};
                    if (nBatch > 0U) {
                        const uint64_t g0 = (fullRows / alignRows) < groupsPerTile ?
                            (fullRows / alignRows) : groupsPerTile;
                        const DataCopyExtParams cpb0(
                            1, static_cast<uint32_t>(g0) * superWidth * sizeof(T),
                            0, 0, 0);
                        DataCopyPad(inbuf,
                                    x_[keepOffset + ReduceOffset(0) + innerOffset],
                                    cpb0, pd);
                        SetFlag<HardEvent::MTE2_V>(evk2[0]);
                    }
                    uint64_t rb = 0;
                    for (uint64_t bi = 0U; bi < nBatch; ++bi) {
                        const uint32_t cur = static_cast<uint32_t>(bi & 1U);
                        rb = bi * rowsPerBatch;
                        const uint64_t groupsLeft = (fullRows - rb) / alignRows;
                        const uint32_t groups = static_cast<uint32_t>(
                            groupsLeft < groupsPerTile ? groupsLeft : groupsPerTile);
                        const uint32_t total = groups * superWidth;
                        if (bi + 1U < nBatch) {
                            const uint32_t nxt = cur ^ 1U;
                            if (bi + 1U >= 2U) {
                                WaitFlag<HardEvent::V_MTE2>(evk2[nxt]);
                            }
                            const uint64_t nrb = rb + rowsPerBatch;
                            const uint64_t ngl = (fullRows - nrb) / alignRows;
                            const uint32_t ng = static_cast<uint32_t>(
                                ngl < groupsPerTile ? ngl : groupsPerTile);
                            const DataCopyExtParams ncpb(
                                1, ng * superWidth * sizeof(T), 0, 0, 0);
                            DataCopyPad(
                                inbuf[nxt * tile_],
                                x_[keepOffset + ReduceOffset(nrb) + innerOffset],
                                ncpb, pd);
                            SetFlag<HardEvent::MTE2_V>(evk2[nxt]);
                        }
                        WaitFlag<HardEvent::MTE2_V>(evk2[cur]);
                        if constexpr (SameType<T, float>::value) {
                            LocalTensor<float> sf =
                                inBuf_.template Get<float>()[cur * tile_];
                            MulAddDst<float, float>(
                                acc, sf, sf, static_cast<int32_t>(total));
                        } else if constexpr (SameType<T, half>::value) {
                            MulAddDst<float, T>(
                                acc, inbuf[cur * tile_], inbuf[cur * tile_],
                                static_cast<int32_t>(total));
                        } else {
                            Cast(sq, inbuf[cur * tile_], RoundMode::CAST_NONE,
                                 static_cast<int32_t>(total));
                            PipeBarrier<PIPE_V>();
                            MulAddDst<float, float>(
                                acc, sq, sq, static_cast<int32_t>(total));
                        }
                        PipeBarrier<PIPE_V>();
                        SetFlag<HardEvent::V_MTE2>(evk2[cur]);
                    }
                    {
                        uint32_t wAll = tile_ / superWidth;
                        while (wAll > 1U) {
                            const uint32_t h = wAll >> 1U;
                            Add(acc, acc, acc[h * superWidth],
                                static_cast<int32_t>(h * superWidth));
                            PipeBarrier<PIPE_V>();
                            if ((wAll & 1U) != 0U) {
                                Add(acc, acc, acc[(wAll - 1U) * superWidth],
                                    static_cast<int32_t>(superWidth));
                                PipeBarrier<PIPE_V>();
                            }
                            wAll = h;
                        }
                    }
                    if (nBatch > 0U) {
                        WaitFlag<HardEvent::V_MTE2>(evk2[(nBatch - 1U) & 1U]);
                        if (nBatch >= 2U) {
                            WaitFlag<HardEvent::V_MTE2>(evk2[(nBatch - 2U) & 1U]);
                        }
                    }
                    rb = fullRows;

                    const DataCopyExtParams cpt(1, count * sizeof(T), 0, 0, 0);
                    while (rb < prefixRedCount_) {
                        DataCopyPad(
                            inbuf, x_[keepOffset + ReduceOffset(rb) + innerOffset],
                            cpt, pd);
                        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                        LocalTensor<float> squared;
                        if constexpr (SameType<T, float>::value) {
                            LocalTensor<float> sf = inBuf_.template Get<float>();
                            Mul(sf, sf, sf, static_cast<int32_t>(count));
                            squared = sf;
                        } else {
                            Cast(sq, inbuf, RoundMode::CAST_NONE,
                                 static_cast<int32_t>(count));
                            Mul(sq, sq, sq, static_cast<int32_t>(count));
                            squared = sq;
                        }
                        Add(acc, acc, squared, static_cast<int32_t>(count));
                        PipeBarrier<PIPE_V>();
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                        ++rb;
                    }
                    const uint64_t scratchBase =
                        static_cast<uint64_t>(GetBlockIdx()) * tile_;
                    const DataCopyExtParams scratchStore(
                        1, superWidth * sizeof(float), 0, 0, 0);
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                    DataCopyPad(ws_[scratchBase], acc, scratchStore);
                    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    const DataCopyExtParams scratchLoad(
                        static_cast<uint16_t>(alignRows), count * sizeof(float),
                        0, 0, 0);
                    const DataCopyPadExtParams<float> scratchPad;
                    DataCopyPad(sq, ws_[scratchBase], scratchLoad, scratchPad);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    uint32_t w3 = alignRows;
                    while (w3 > 1U) {
                        const uint32_t h = w3 >> 1U;
                        Add(sq, sq, sq[h * scratchPitch],
                            static_cast<int32_t>(h * scratchPitch));
                        PipeBarrier<PIPE_V>();
                        if ((w3 & 1U) != 0U) {
                            Add(sq, sq, sq[(w3 - 1U) * scratchPitch],
                                static_cast<int32_t>(scratchPitch));
                            PipeBarrier<PIPE_V>();
                        }
                        w3 = h;
                    }
                    Adds(acc, sq, 0.0f, static_cast<int32_t>(count));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                } else {
                const uint32_t B = rowsPerTile;
                for (uint64_t rb = 0; rb < prefixRedCount_; rb += B) {
                    const uint32_t rows = static_cast<uint32_t>(
                        (prefixRedCount_ - rb) < B ? (prefixRedCount_ - rb) : B);
                    const uint32_t total = rows * rowPitch;
                    const DataCopyExtParams cp(
                        rowAligned ? 1U : rows,
                        rowAligned ? total * sizeof(T) : count * sizeof(T),
                        0, 0, 0);
                    DataCopyPad(
                        inbuf, x_[keepOffset + ReduceOffset(rb) + innerOffset], cp, pd);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                    LocalTensor<float> squared;
                    if constexpr (SameType<T, float>::value) {
                        LocalTensor<float> sf = inBuf_.template Get<float>();
                        Mul(sf, sf, sf, static_cast<int32_t>(total));
                        squared = sf;
                    } else {
                        Cast(sq, inbuf, RoundMode::CAST_NONE, static_cast<int32_t>(total));
                        Mul(sq, sq, sq, static_cast<int32_t>(total));
                        squared = sq;
                    }
                    uint32_t w = rows;
                    while (w > 1U) {
                        const uint32_t h = w >> 1U;
                        Add(squared, squared, squared[h * rowPitch],
                            static_cast<int32_t>(h * rowPitch));
                        PipeBarrier<PIPE_V>();
                        if ((w & 1U) != 0U) {
                            Add(squared, squared, squared[(w - 1U) * rowPitch],
                                static_cast<int32_t>(rowPitch));
                            PipeBarrier<PIPE_V>();
                        }
                        w = h;
                    }
                    Add(acc, acc, squared, static_cast<int32_t>(count));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
                }
                }
            } else {

            {
                LocalTensor<T> inbuf = inBuf_.template Get<T>();
                LocalTensor<float> sq = sqBuf_.template Get<float>();
                const int32_t evk[2] = {EVENT_ID0, EVENT_ID1};
                const DataCopyPadExtParams<T> pd;
                const DataCopyExtParams cp(1, count * sizeof(T), 0, 0, 0);
                DataCopyPad(inbuf, x_[keepOffset + ReduceOffset(0) + innerOffset], cp, pd);
                SetFlag<HardEvent::MTE2_V>(evk[0]);
                for (uint64_t r = 0; r < prefixRedCount_; ++r) {
                    const uint32_t cur = static_cast<uint32_t>(r & 1U);
                    if (r + 1 < prefixRedCount_) {
                        const uint32_t ns = static_cast<uint32_t>((r + 1) & 1U);
                        if (r + 1 >= 2) {
                            WaitFlag<HardEvent::V_MTE2>(evk[ns]);
                        }
                        DataCopyPad(inbuf[ns * tile_], x_[keepOffset + ReduceOffset(r + 1) + innerOffset], cp, pd);
                        SetFlag<HardEvent::MTE2_V>(evk[ns]);
                    }
                    WaitFlag<HardEvent::MTE2_V>(evk[cur]);
                    LocalTensor<float> squared;
                    if constexpr (SameType<T, float>::value) {
                        LocalTensor<float> sf = inBuf_.template Get<float>();
                        LocalTensor<float> ss = sf[cur * tile_];
                        Mul(ss, ss, ss, static_cast<int32_t>(count));
                        squared = ss;
                    } else {
                        LocalTensor<T> src = inbuf[cur * tile_];
                        Cast(sq, src, RoundMode::CAST_NONE, static_cast<int32_t>(count));
                        Mul(sq, sq, sq, static_cast<int32_t>(count));
                        squared = sq;
                    }
                    Add(acc, acc, squared, static_cast<int32_t>(count));
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE2>(evk[cur]);
                }
                WaitFlag<HardEvent::V_MTE2>(evk[(prefixRedCount_ - 1) & 1U]);
                if (prefixRedCount_ >= 2) {
                    WaitFlag<HardEvent::V_MTE2>(evk[(prefixRedCount_ - 2) & 1U]);
                }
            }
            }

            if (count <= kOutTile) {
                Adds(outF32, acc, 0.0f, static_cast<int32_t>(count));
                PipeBarrier<PIPE_V>();
                StoreOutput(keepIndex * innerLen_ + innerOffset, count);
            } else {
                for (uint32_t done = 0; done < count; done += kOutTile) {
                    const uint32_t left2 = count - done;
                    const uint32_t part = left2 < kOutTile ? left2 : kOutTile;
                    Adds(outF32, acc[done], 0.0f, static_cast<int32_t>(part));
                    PipeBarrier<PIPE_V>();
                    StoreOutput(keepIndex * innerLen_ + innerOffset + done, part);
                }
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> inBuf_;
    TBuf<TPosition::VECCALC> sqBuf_;
    TBuf<TPosition::VECCALC> accBuf_;
    TBuf<TPosition::VECCALC> workBuf_;
    TBuf<TPosition::VECCALC> redBuf_;
    uint32_t outSlot_ = 0U;
    uint32_t outPend_[2] = {0U, 0U};
    TBuf<TPosition::VECCALC> outF32Buf_;
    TBuf<TPosition::VECCALC> outTBuf_;
    TBuf<TPosition::VECCALC> partialBuf_;
    TBuf<TPosition::VECCALC> segBuf_;
    TBuf<TPosition::VECCALC> syncUbBuf_;
    GlobalTensor<T> x_;
    GlobalTensor<T> y_;
    GlobalTensor<float> ws_;
    GlobalTensor<int32_t> syncGm_;
    uint64_t groupLen_[S9_SQ_MAX_GROUP];
    uint64_t groupStride_[S9_SQ_MAX_GROUP];
    uint64_t groupIsReduce_[S9_SQ_MAX_GROUP];
    uint64_t innerLen_;
    uint64_t innerChunk_;
    uint64_t innerChunkCount_;
    uint64_t prefixRedCount_;
    uint64_t outCount_;
    uint64_t taskCount_;
    uint64_t redSegCount_;
    uint64_t redSegLen_;
    uint64_t keepRedSeg_;
    uint64_t keepRedSegLen_;
    uint32_t slabScratch_;
    uint32_t groupCount_;
    uint32_t innerIsReduce_;
    uint32_t tile_;
    uint32_t cachedOffsetRows_;
};
}

extern "C" __global__ __aicore__ void square_sum_v1(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SquareSumV1TilingData);
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(1)) {
        KernelSquareSumV1<DTYPE_INPUT, 1U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelSquareSumV1<DTYPE_INPUT, 2U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3)) {
        KernelSquareSumV1<DTYPE_INPUT, 3U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        KernelSquareSumV1<DTYPE_INPUT, 4U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(5)) {
        KernelSquareSumV1<DTYPE_INPUT, 5U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        KernelSquareSumV1<DTYPE_INPUT, 0U> op;
        op.Init(input, output, workspace, tilingData);
        op.Process();
    }
}
