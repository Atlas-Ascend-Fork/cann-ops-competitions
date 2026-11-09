#include "concat_tiling.h"
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

namespace {

constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kAlignBytes = 32;
constexpr uint64_t kMaxUint32 = 0xFFFFFFFFULL;
constexpr int64_t kMaxInputs = 256;
constexpr int64_t kPackedWideChunkElems = 256;
constexpr int64_t kMaxBalancedGroupInputs = 24;
constexpr int64_t kMaxDirectInputs = 8;
constexpr int64_t kScheduleRows = 0;
constexpr int64_t kScheduleInputRows = 1;
constexpr int64_t kScheduleFlatContiguous = 2;
constexpr int64_t kScheduleInputFullOuter = 3;
constexpr int64_t kScheduleInputMajorBalanced = 15;

template <typename T>
__aicore__ inline uint32_t AlignElems(uint32_t elemCount) {
    const uint32_t alignElems = kAlignBytes / sizeof(T);
    return ((elemCount + alignElems - 1) / alignElems) * alignElems;
}

template <typename T, bool kInputMajorBalanced = false, bool kInputReadBypass = false>
class ConcatCopyKernel {
  public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
        inputCount_ = tiling.input_count;
        outerCount_ = tiling.outer_count;
        innerSpan_ = tiling.inner_span;
        concatDim_ = tiling.concat_dim;
        outputElems_ = tiling.output_elems;
        workItems_ = tiling.work_items;
        workItemsPerCore_ = tiling.work_items_per_core;
        rowsPerTask_ = tiling.rows_per_task;
        tileElems_ = tiling.tile_elems;
        scheduleMode_ = tiling.schedule_mode;
        flatInputIndex_ = tiling.flat_input_index;

        outputGm_.SetGlobalBuffer((__gm__ T *)output, static_cast<uint64_t>(tiling.output_elems));
        inputDesc_.Init((__gm__ void *)inputs);

        for (int64_t idx = 0; idx < inputCount_ && idx < kMaxInputs; ++idx) {
            splits_[idx] = tiling.splits[idx];
        }

        int64_t maxSplit = 1;
        for (int64_t idx = 0; idx < inputCount_ && idx < kMaxInputs; ++idx) {
            if (splits_[idx] > maxSplit) {
                maxSplit = splits_[idx];
            }
        }
        uint32_t bufferElems = 1;
        if constexpr (kInputMajorBalanced) {
            bufferElems = AlignElems<T>(static_cast<uint32_t>(tileElems_));
        } else if (scheduleMode_ == kScheduleFlatContiguous) {
            bufferElems = AlignElems<T>(static_cast<uint32_t>(rowsPerTask_));
        } else {
            int64_t elemsPerRowCopy = maxSplit * innerSpan_;
            if (tileElems_ > 0 && tileElems_ < elemsPerRowCopy) {
                elemsPerRowCopy = tileElems_;
            }
            bufferElems = AlignElems<T>(static_cast<uint32_t>(elemsPerRowCopy)) *
                          static_cast<uint32_t>(rowsPerTask_);
        }
        if (bufferElems == 0) {
            bufferElems = 1;
        }
        pipe_.InitBuffer(copyQueue_, kBufferNum, bufferElems * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (inputCount_ <= 0 || outerCount_ <= 0 || innerSpan_ <= 0 || concatDim_ <= 0 || workItems_ <= 0) {
            return;
        }
        if constexpr (kInputMajorBalanced) {
            ProcessInputMajorBalanced();
            FlushPending();
            return;
        }
        const int64_t coreIdx = GetBlockIdx();
        const int64_t blockCount = GetBlockNum();
        const int64_t smallTaskCount = workItems_ / blockCount;
        const int64_t largeCoreCount = workItems_ - smallTaskCount * blockCount;
        const int64_t firstTask = coreIdx * smallTaskCount + (coreIdx < largeCoreCount ? coreIdx : largeCoreCount);
        const int64_t lastTask = firstTask + smallTaskCount + (coreIdx < largeCoreCount ? 1 : 0);
        if (scheduleMode_ == kScheduleFlatContiguous) {
            ProcessFlat(firstTask, lastTask);
            return;
        }
        if (scheduleMode_ == kScheduleInputRows) {
            ProcessInputRows(firstTask, lastTask);
            return;
        }
        if (scheduleMode_ == kScheduleInputFullOuter) {
            ProcessInputFullOuter(firstTask, lastTask);
            return;
        }
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t outerBegin = task * rowsPerTask_;
            if (outerBegin >= outerCount_) {
                return;
            }
            int64_t rows = outerCount_ - outerBegin;
            if (rows > rowsPerTask_) {
                rows = rowsPerTask_;
            }
            CopyRows(outerBegin, rows);
        }
    }

  private:
    __aicore__ inline void FlushPending() {
        if (!hasPending_) {
            return;
        }
        LocalTensor<T> local = copyQueue_.template DeQue<T>();
        const DataCopyExtParams params = {pendingRows_, pendingRowBytes_, 0, pendingStride_, 0};
        DataCopyPad(outputGm_[pendingOffset_], local, params);
        copyQueue_.FreeTensor(local);
        hasPending_ = false;
    }

    __aicore__ inline void QueueOutput(int64_t offset, uint16_t rows, uint32_t bytes, uint32_t stride) {
        FlushPending();
        pendingOffset_ = offset;
        pendingRows_ = rows;
        pendingRowBytes_ = bytes;
        pendingStride_ = stride;
        hasPending_ = true;
    }

    __aicore__ inline void ProcessInputMajorBalanced() {
        const int64_t coreIdx = GetBlockIdx();
        const int64_t blockCount = GetBlockNum();
        const int64_t alignBytes = outputElems_ * sizeof(T) >= 2 * 1024 * 1024 &&
                                           (outerCount_ == 1 || (innerSpan_ * sizeof(T)) % 512 == 0) ? 512 : kAlignBytes;
        const int64_t alignElems = alignBytes / sizeof(T);
        const int64_t alignedUnits = outputElems_ / alignElems;
        const int64_t firstElem = alignedUnits * coreIdx / blockCount * alignElems;
        const int64_t lastElem = coreIdx + 1 == blockCount
                                     ? outputElems_
                                     : alignedUnits * (coreIdx + 1) / blockCount * alignElems;
        if (lastElem <= firstElem) {
            return;
        }

        int64_t inputBase = 0;
        int64_t concatOffset = 0;
        for (int64_t inputIdx = 0; inputIdx < inputCount_ && inputIdx < kMaxInputs; ++inputIdx) {
            const int64_t split = splits_[inputIdx];
            const int64_t inputElems = outerCount_ * split * innerSpan_;
            const int64_t inputEnd = inputBase + inputElems;
            if (split > 0 && firstElem < inputEnd && lastElem > inputBase) {
                const int64_t localBegin = firstElem > inputBase ? firstElem - inputBase : 0;
                const int64_t localEnd = lastElem < inputEnd ? lastElem - inputBase : inputElems;
                __gm__ T *inputPtr = inputDesc_.GetDataPtr<T>(static_cast<uint32_t>(inputIdx));
                if (inputPtr != nullptr && localEnd > localBegin) {
                    GlobalTensor<T> inputGm;
                    inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(inputElems));
                    if constexpr (kInputReadBypass) {
                        inputGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
                    } else if (outputElems_ * sizeof(T) >= 2 * 1024 * 1024) {
                        inputGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
                    }
                    MoveInputMajorRange(inputGm, localBegin, localEnd - localBegin, split, concatOffset);
                }
            }
            inputBase = inputEnd;
            concatOffset += split;
            if (inputBase >= lastElem) {
                return;
            }
        }
    }

    __aicore__ inline void MoveInputMajorRange(GlobalTensor<T> &inputGm, int64_t localOffset,
                                               int64_t elemCount, int64_t split, int64_t concatOffset) {
        const int64_t rowElems = split * innerSpan_;
        if ((innerSpan_ * sizeof(T)) % kAlignBytes == 0) {
            MoveLinearInputRange(inputGm, localOffset, elemCount, split, concatOffset);
            return;
        }
        while (elemCount > 0) {
            const int64_t outerIdx = localOffset / rowElems;
            const int64_t rowOffset = localOffset - outerIdx * rowElems;
            const int64_t paddedRowElems = (rowElems + kAlignBytes / sizeof(T) - 1) /
                                          (kAlignBytes / sizeof(T)) * (kAlignBytes / sizeof(T));
            if (rowOffset == 0 && elemCount >= rowElems && tileElems_ >= paddedRowElems) {
                int64_t rows = elemCount / rowElems;
                const int64_t ubRows = tileElems_ / paddedRowElems;
                rows = rows < ubRows ? rows : ubRows;
                rows = rows < 65535 ? rows : 65535;
                MoveRowChunk(inputGm, outerIdx, rows, split, concatOffset, 0, rowElems);
                localOffset += rows * rowElems;
                elemCount -= rows * rowElems;
                continue;
            }
            int64_t rowCount = rowElems - rowOffset;
            if (rowCount > elemCount) {
                rowCount = elemCount;
            }
            for (int64_t chunkOffset = 0; chunkOffset < rowCount; chunkOffset += tileElems_) {
                int64_t chunkElems = rowCount - chunkOffset;
                if (chunkElems > tileElems_) {
                    chunkElems = tileElems_;
                }
                MoveRowChunkSingle(inputGm, outerIdx, split, concatOffset,
                                   rowOffset + chunkOffset, chunkElems);
            }
            localOffset += rowCount;
            elemCount -= rowCount;
        }
    }

    __aicore__ inline void MoveLinearInputRange(GlobalTensor<T> &inputGm, int64_t localOffset,
                                                int64_t elemCount, int64_t split, int64_t concatOffset) {
        const int64_t rowElems = split * innerSpan_;
        const int64_t outputRowElems = concatDim_ * innerSpan_;
        int64_t count = elemCount < tileElems_ ? elemCount : tileElems_;
        LocalTensor<T> first = copyQueue_.template AllocTensor<T>();
        DataCopy(first, inputGm[localOffset], static_cast<uint32_t>(count));
        copyQueue_.EnQue(first);
        while (elemCount > 0) {
            LocalTensor<T> local = copyQueue_.template DeQue<T>();
            const int64_t remaining = elemCount - count;
            const int64_t nextCount = remaining < tileElems_ ? remaining : tileElems_;
            if (nextCount > 0) {
                LocalTensor<T> next = copyQueue_.template AllocTensor<T>();
                DataCopy(next, inputGm[localOffset + count], static_cast<uint32_t>(nextCount));
                copyQueue_.EnQue(next);
            }
            int64_t row = localOffset / rowElems;
            const int64_t rowOffset = localOffset - row * rowElems;
            int64_t used = 0;
            if (rowOffset != 0) {
                const int64_t head = count < rowElems - rowOffset ? count : rowElems - rowOffset;
                DataCopy(outputGm_[row * outputRowElems + concatOffset * innerSpan_ + rowOffset],
                         local, static_cast<uint32_t>(head));
                used += head;
                ++row;
            }
            const int64_t fullRows = (count - used) / rowElems;
            const uint64_t strideBytes = static_cast<uint64_t>(outputRowElems - rowElems) * sizeof(T);
            if (fullRows > 0) {
                if (strideBytes <= kMaxUint32) {
                    const DataCopyExtParams params = {
                        static_cast<uint16_t>(fullRows), static_cast<uint32_t>(rowElems * sizeof(T)),
                        0, static_cast<uint32_t>(strideBytes), 0};
                    DataCopyPad(outputGm_[row * outputRowElems + concatOffset * innerSpan_], local[used], params);
                } else {
                    for (int64_t r = 0; r < fullRows; ++r) {
                        DataCopy(outputGm_[(row + r) * outputRowElems + concatOffset * innerSpan_],
                                 local[used + r * rowElems], static_cast<uint32_t>(rowElems));
                    }
                }
                used += fullRows * rowElems;
                row += fullRows;
            }
            if (used < count) {
                DataCopy(outputGm_[row * outputRowElems + concatOffset * innerSpan_],
                         local[used], static_cast<uint32_t>(count - used));
            }
            copyQueue_.FreeTensor(local);
            localOffset += count;
            elemCount -= count;
            count = nextCount;
        }
    }

    __aicore__ inline void ProcessFlat(int64_t firstTask, int64_t lastTask) {
        __gm__ T *inputPtr = inputDesc_.GetDataPtr<T>(static_cast<uint32_t>(flatInputIndex_));
        if (inputPtr == nullptr) {
            return;
        }
        GlobalTensor<T> inputGm;
        inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outputElems_));
        if constexpr (kInputReadBypass) {
            inputGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }

        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t elemOffset = task * rowsPerTask_;
            if (elemOffset >= outputElems_) {
                return;
            }
            int64_t elemCount = outputElems_ - elemOffset;
            if (elemCount > rowsPerTask_) {
                elemCount = rowsPerTask_;
            }
            MoveFlat(inputGm, elemOffset, elemCount);
        }
    }

    __aicore__ inline void ProcessInputRows(int64_t firstTask, int64_t lastTask) {
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t inputIdx = task / outerCount_;
            const int64_t outerIdx = task - inputIdx * outerCount_;
            if (inputIdx >= inputCount_ || outerIdx >= outerCount_) {
                return;
            }
            const int64_t split = splits_[inputIdx];
            if (split <= 0) {
                continue;
            }
            __gm__ T *inputPtr = inputDesc_.GetDataPtr<T>(static_cast<uint32_t>(inputIdx));
            if (inputPtr == nullptr) {
                continue;
            }
            GlobalTensor<T> inputGm;
            inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outerCount_ * split * innerSpan_));
            if constexpr (kInputReadBypass) {
                inputGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            }
            MoveRows(inputGm, outerIdx, 1, split, PrefixSplit(inputIdx));
        }
    }

    __aicore__ inline int64_t PrefixSplit(int64_t inputIdx) const {
        int64_t offset = 0;
        for (int64_t idx = 0; idx < inputIdx && idx < kMaxInputs; ++idx) {
            offset += splits_[idx];
        }
        return offset;
    }

    __aicore__ inline void ProcessInputFullOuter(int64_t firstTask, int64_t lastTask) {
        for (int64_t inputIdx = firstTask; inputIdx < lastTask; ++inputIdx) {
            if (inputIdx >= inputCount_) {
                return;
            }
            const int64_t split = splits_[inputIdx];
            if (split <= 0) {
                continue;
            }
            __gm__ T *inputPtr = inputDesc_.GetDataPtr<T>(static_cast<uint32_t>(inputIdx));
            if (inputPtr == nullptr) {
                continue;
            }
            GlobalTensor<T> inputGm;
            inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outerCount_ * split * innerSpan_));
            if constexpr (kInputReadBypass) {
                inputGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            }
            MoveRows(inputGm, 0, outerCount_, split, PrefixSplit(inputIdx));
        }
    }


    __aicore__ inline void CopyRows(int64_t outerBegin, int64_t rowCount) {
        int64_t concatOffset = 0;
        for (int64_t inputIdx = 0; inputIdx < inputCount_ && inputIdx < kMaxInputs; ++inputIdx) {
            const int64_t split = splits_[inputIdx];
            if (split <= 0) {
                continue;
            }
            __gm__ T *inputPtr = inputDesc_.GetDataPtr<T>(static_cast<uint32_t>(inputIdx));
            if (inputPtr == nullptr) {
                concatOffset += split;
                continue;
            }
            GlobalTensor<T> inputGm;
            inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outerCount_ * split * innerSpan_));
            if constexpr (kInputReadBypass) {
                inputGm.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            }
            MoveRows(inputGm, outerBegin, rowCount, split, concatOffset);
            concatOffset += split;
        }
    }

    __aicore__ inline void MoveRows(GlobalTensor<T> &inputGm, int64_t outerBegin, int64_t rowCount, int64_t split,
                                    int64_t concatOffset) {
        const int64_t rowElems = split * innerSpan_;
        if (rowElems <= 0) {
            return;
        }
        int64_t chunkElems = tileElems_;
        if (chunkElems <= 0 || chunkElems > rowElems) {
            chunkElems = rowElems;
        }
        for (int64_t chunkOffset = 0; chunkOffset < rowElems; chunkOffset += chunkElems) {
            int64_t currentElems = rowElems - chunkOffset;
            if (currentElems > chunkElems) {
                currentElems = chunkElems;
            }
            MoveRowChunk(inputGm, outerBegin, rowCount, split, concatOffset, chunkOffset, currentElems);
        }
    }

    __aicore__ inline void MoveRowChunk(GlobalTensor<T> &inputGm, int64_t outerBegin, int64_t rowCount, int64_t split,
                                        int64_t concatOffset, int64_t chunkOffset, int64_t chunkElems) {
        const int64_t inputRowElems = split * innerSpan_;
        const int64_t outputRowElems = concatDim_ * innerSpan_;
        const uint32_t rowBytes = static_cast<uint32_t>(chunkElems * sizeof(T));
        const uint64_t inputStrideBytes = static_cast<uint64_t>(inputRowElems - chunkElems) * sizeof(T);
        const uint64_t outputStrideBytes = static_cast<uint64_t>(outputRowElems - chunkElems) * sizeof(T);
        if (inputStrideBytes > kMaxUint32 || outputStrideBytes > kMaxUint32) {
            for (int64_t row = 0; row < rowCount; ++row) {
                MoveRowChunkSingle(inputGm, outerBegin + row, split, concatOffset, chunkOffset, chunkElems);
            }
            return;
        }
        const uint16_t blockCount = static_cast<uint16_t>(rowCount);
        LocalTensor<T> local = copyQueue_.template AllocTensor<T>();
        DataCopyExtParams inputParams = {blockCount, rowBytes, static_cast<uint32_t>(inputStrideBytes), 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, inputGm[outerBegin * inputRowElems + chunkOffset], inputParams, padParams);
        copyQueue_.EnQue(local);
        const int64_t outputOffset = (outerBegin * concatDim_ + concatOffset) * innerSpan_ + chunkOffset;
        if constexpr (kInputMajorBalanced) {
            QueueOutput(outputOffset, blockCount, rowBytes, static_cast<uint32_t>(outputStrideBytes));
            return;
        }
        local = copyQueue_.template DeQue<T>();
        DataCopyExtParams outputParams = {blockCount, rowBytes, 0, static_cast<uint32_t>(outputStrideBytes), 0};
        DataCopyPad(outputGm_[outputOffset], local, outputParams);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void MoveRowChunkSingle(GlobalTensor<T> &inputGm, int64_t outerIdx, int64_t split,
                                             int64_t concatOffset, int64_t chunkOffset, int64_t chunkElems) {
        const int64_t inputRowElems = split * innerSpan_;
        const uint32_t bytes = static_cast<uint32_t>(chunkElems * sizeof(T));
        LocalTensor<T> local = copyQueue_.template AllocTensor<T>();
        DataCopyExtParams params = {1, bytes, 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, inputGm[outerIdx * inputRowElems + chunkOffset], params, padParams);
        copyQueue_.EnQue(local);
        const int64_t outputOffset = (outerIdx * concatDim_ + concatOffset) * innerSpan_ + chunkOffset;
        if constexpr (kInputMajorBalanced) {
            QueueOutput(outputOffset, 1, bytes, 0);
            return;
        }
        local = copyQueue_.template DeQue<T>();
        DataCopyPad(outputGm_[outputOffset], local, params);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void MoveFlat(GlobalTensor<T> &inputGm, int64_t elemOffset, int64_t elemCount) {
        const uint32_t count = static_cast<uint32_t>(elemCount);
        LocalTensor<T> local = copyQueue_.template AllocTensor<T>();
        if ((count * sizeof(T)) % kAlignBytes == 0) {
            DataCopy(local, inputGm[elemOffset], count);
            copyQueue_.EnQue(local);
            local = copyQueue_.template DeQue<T>();
            DataCopy(outputGm_[elemOffset], local, count);
        } else {
            const uint32_t bytes = count * sizeof(T);
            DataCopyExtParams params = {1, bytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(local, inputGm[elemOffset], params, padParams);
            copyQueue_.EnQue(local);
            local = copyQueue_.template DeQue<T>();
            DataCopyPad(outputGm_[elemOffset], local, params);
        }
        copyQueue_.FreeTensor(local);
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kBufferNum> copyQueue_;
    ListTensorDesc inputDesc_;
    GlobalTensor<T> outputGm_;
    int64_t inputCount_ = 0;
    int64_t outerCount_ = 0;
    int64_t innerSpan_ = 0;
    int64_t concatDim_ = 0;
    int64_t outputElems_ = 0;
    int64_t workItems_ = 0;
    int64_t workItemsPerCore_ = 0;
    int64_t rowsPerTask_ = 1;
    int64_t tileElems_ = 1;
    int64_t scheduleMode_ = kScheduleRows;
    int64_t flatInputIndex_ = 0;
    bool hasPending_ = false;
    int64_t pendingOffset_ = 0;
    uint16_t pendingRows_ = 0;
    uint32_t pendingRowBytes_ = 0;
    uint32_t pendingStride_ = 0;
    uint16_t splits_[kMaxInputs] = {};
};

template <typename T, bool kStriped = false>
class ConcatSmallInputDirectRowsKernel {
  public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
        inputCount_ = tiling.input_count;
        outerCount_ = tiling.outer_count;
        innerSpan_ = tiling.inner_span;
        concatDim_ = tiling.concat_dim;
        rowsPerTile_ = tiling.rows_per_task;
        outputGm_.SetGlobalBuffer((__gm__ T *)output, static_cast<uint64_t>(tiling.output_elems));

        inputDesc_.Init((__gm__ void *)inputs);
        int64_t concatOffset = 0;
        uint32_t queueRowElems[2] = {};
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            const int64_t split = tiling.splits[inputIdx];
            if (split <= 0) {
                continue;
            }
            const int64_t activeIdx = activeInputCount_++;
            splits_[activeIdx] = static_cast<uint16_t>(split);
            outputOffsets_[activeIdx] = concatOffset;
            __gm__ T *inputPtr = inputDesc_.GetDataPtr<T>(static_cast<uint32_t>(inputIdx));
            inputGm_[activeIdx].SetGlobalBuffer(
                inputPtr, static_cast<uint64_t>(outerCount_ * split * innerSpan_));
            inputGm_[activeIdx].template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
            const int64_t queueIdx = activeIdx & 1;
            const uint32_t rowElems = static_cast<uint32_t>(split * innerSpan_);
            queueRowElems[queueIdx] = queueRowElems[queueIdx] > rowElems ? queueRowElems[queueIdx] : rowElems;
            concatOffset += split;
        }

        const uint32_t firstBufferElems = AlignElems<T>(queueRowElems[0]) * static_cast<uint32_t>(rowsPerTile_);
        const uint32_t secondBufferElems = AlignElems<T>(queueRowElems[1]) * static_cast<uint32_t>(rowsPerTile_);
        pipe_.InitBuffer(firstQueue_, 1, firstBufferElems * sizeof(T));
        pipe_.InitBuffer(secondQueue_, 1, secondBufferElems * sizeof(T));
    }

    __aicore__ inline void Process() {
        const int64_t coreIdx = GetBlockIdx();
        const int64_t blockCount = GetBlockNum();
        const int64_t smallRows = outerCount_ / blockCount;
        const int64_t largeCoreCount = outerCount_ - smallRows * blockCount;
        const int64_t coreRowBegin = kStriped ? coreIdx : coreIdx * smallRows + (coreIdx < largeCoreCount ? coreIdx : largeCoreCount);
        const int64_t coreRows = smallRows + (coreIdx < largeCoreCount ? 1 : 0);

        for (int64_t localRow = 0; localRow < coreRows; localRow += rowsPerTile_) {
            const int64_t rows = coreRows - localRow > rowsPerTile_ ? rowsPerTile_ : coreRows - localRow;
            const int64_t rowBegin = coreRowBegin + localRow * (kStriped ? 40 : 1);
            ProcessActiveInput<0>(rowBegin, rows);
            const int64_t firstPending = (activeInputCount_ & 1) != 0 ? activeInputCount_ - 1
                                                                      : activeInputCount_ - 2;
            const int64_t secondPending = (activeInputCount_ & 1) != 0 ? activeInputCount_ - 2
                                                                       : activeInputCount_ - 1;
            StoreSegment<0>(rowBegin, rows, firstPending);
            StoreSegment<1>(rowBegin, rows, secondPending);
        }
    }

  private:
    template <int kActiveInput>
    __aicore__ inline void ProcessActiveInput(int64_t rowBegin, int64_t rowCount) {
        if constexpr (kActiveInput < kMaxDirectInputs) {
            if (kActiveInput >= activeInputCount_) {
                return;
            }
            if constexpr (kActiveInput >= 2) {
                StoreSegment<kActiveInput & 1>(rowBegin, rowCount, kActiveInput - 2);
            }
            LoadSegment<kActiveInput & 1>(rowBegin, rowCount, kActiveInput);
            ProcessActiveInput<kActiveInput + 1>(rowBegin, rowCount);
        }
    }

    template <int kQueue>
    __aicore__ inline void LoadSegment(int64_t rowBegin, int64_t rowCount, int64_t inputIdx) {
        const int64_t split = splits_[inputIdx];
        const int64_t rowElems = split * innerSpan_;
        const uint32_t rowBytes = static_cast<uint32_t>(rowElems * sizeof(T));
        if constexpr (kQueue == 0) {
            LocalTensor<T> local = firstQueue_.AllocTensor<T>();
            const DataCopyExtParams inputParams = {static_cast<uint16_t>(rowCount), rowBytes, kStriped ? 39U * rowBytes : 0U, 0, 0};
            const DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(local, inputGm_[inputIdx][rowBegin * rowElems], inputParams, padParams);
            firstQueue_.EnQue(local);
            return;
        }
        LocalTensor<T> local = secondQueue_.AllocTensor<T>();
        DataCopyExtParams inputParams = {static_cast<uint16_t>(rowCount), rowBytes, kStriped ? 39U * rowBytes : 0U, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(local, inputGm_[inputIdx][rowBegin * rowElems], inputParams, padParams);
        secondQueue_.EnQue(local);
    }

    template <int kQueue>
    __aicore__ inline void StoreSegment(int64_t rowBegin, int64_t rowCount, int64_t inputIdx) {
        const int64_t split = splits_[inputIdx];
        const uint32_t rowBytes = static_cast<uint32_t>(split * innerSpan_ * sizeof(T));
        const uint32_t outputStrideBytes =
            static_cast<uint32_t>(((kStriped ? 40 : 1) * concatDim_ - split) * innerSpan_ * sizeof(T));
        const DataCopyExtParams outputParams = {
            static_cast<uint16_t>(rowCount), rowBytes, 0, outputStrideBytes, 0};
        const int64_t outputOffset = (rowBegin * concatDim_ + outputOffsets_[inputIdx]) * innerSpan_;
        if constexpr (kQueue == 0) {
            LocalTensor<T> local = firstQueue_.DeQue<T>();
            DataCopyPad(outputGm_[outputOffset], local, outputParams);
            firstQueue_.FreeTensor(local);
            return;
        }
        LocalTensor<T> local = secondQueue_.DeQue<T>();
        DataCopyPad(outputGm_[outputOffset], local, outputParams);
        secondQueue_.FreeTensor(local);
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> firstQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> secondQueue_;
    ListTensorDesc inputDesc_;
    GlobalTensor<T> outputGm_;
    GlobalTensor<T> inputGm_[kMaxDirectInputs];
    uint16_t splits_[kMaxDirectInputs] = {};
    int64_t outputOffsets_[kMaxDirectInputs] = {};
    int64_t inputCount_ = 0;
    int64_t activeInputCount_ = 0;
    int64_t outerCount_ = 0;
    int64_t innerSpan_ = 0;
    int64_t concatDim_ = 0;
    int64_t rowsPerTile_ = 1;
};

template <typename T>
class ConcatInputSegmentKernel {
  public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
        inputCount_ = tiling.input_count;
        outerCount_ = tiling.outer_count;
        innerSpan_ = tiling.inner_span;
        concatDim_ = tiling.concat_dim;
        workItems_ = tiling.work_items;
        rowsPerTask_ = tiling.rows_per_task;

        outputGm_.SetGlobalBuffer((__gm__ T *)output, static_cast<uint64_t>(tiling.output_elems));
        inputDesc_.Init((__gm__ void *)inputs);
        for (int64_t idx = 0; idx < inputCount_ && idx < kMaxInputs; ++idx) {
            splits_[idx] = tiling.splits[idx];
        }

        uint32_t bufferElems = AlignElems<T>(static_cast<uint32_t>(rowsPerTask_));
        if (bufferElems == 0) {
            bufferElems = 1;
        }
        pipe_.InitBuffer(copyQueue_, kBufferNum, bufferElems * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (inputCount_ <= 0 || outerCount_ <= 0 || innerSpan_ <= 0 || concatDim_ <= 0 || workItems_ <= 0 ||
            rowsPerTask_ <= 0) {
            return;
        }

        const int64_t tasksPerOuter = InputSegmentTasksPerOuter();
        if (tasksPerOuter <= 0) {
            return;
        }

        const int64_t coreIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        const int64_t firstTask = workItems_ * coreIdx / blockNum;
        const int64_t lastTask = workItems_ * (coreIdx + 1) / blockNum;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t outerIdx = task / tasksPerOuter;
            if (outerIdx >= outerCount_) {
                return;
            }

            int64_t inputIdx = 0;
            int64_t prefix = 0;
            int64_t segmentOffset = 0;
            int64_t segmentElems = 0;
            const int64_t taskInOuter = task - outerIdx * tasksPerOuter;
            if (!LocateInputSegmentTask(taskInOuter, inputIdx, prefix, segmentOffset, segmentElems)) {
                continue;
            }

            __gm__ T *inputPtr = inputDesc_.GetDataPtr<T>(static_cast<uint32_t>(inputIdx));
            if (inputPtr == nullptr) {
                continue;
            }
            const int64_t split = splits_[inputIdx];
            GlobalTensor<T> inputGm;
            inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outerCount_ * split * innerSpan_));
            const int64_t inputOffset = outerIdx * split * innerSpan_ + segmentOffset;
            const int64_t outputOffset = (outerIdx * concatDim_ + prefix) * innerSpan_ + segmentOffset;
            MoveSegment(inputGm, inputOffset, outputOffset, segmentElems);
        }
    }

  private:
    __aicore__ inline int64_t InputSegmentTasksPerOuter() const {
        int64_t tasks = 0;
        for (int64_t idx = 0; idx < inputCount_ && idx < kMaxInputs; ++idx) {
            const int64_t split = splits_[idx];
            if (split <= 0) {
                continue;
            }
            const int64_t elems = split * innerSpan_;
            tasks += (elems + rowsPerTask_ - 1) / rowsPerTask_;
        }
        return tasks;
    }

    __aicore__ inline bool LocateInputSegmentTask(int64_t taskInOuter, int64_t &inputIdx, int64_t &prefix,
                                                  int64_t &segmentOffset, int64_t &segmentElems) const {
        int64_t currentTask = taskInOuter;
        int64_t currentPrefix = 0;
        for (int64_t idx = 0; idx < inputCount_ && idx < kMaxInputs; ++idx) {
            const int64_t split = splits_[idx];
            if (split <= 0) {
                continue;
            }
            const int64_t inputElemsPerOuter = split * innerSpan_;
            const int64_t tasksForInput = (inputElemsPerOuter + rowsPerTask_ - 1) / rowsPerTask_;
            if (currentTask < tasksForInput) {
                inputIdx = idx;
                prefix = currentPrefix;
                segmentOffset = currentTask * rowsPerTask_;
                segmentElems = inputElemsPerOuter - segmentOffset;
                if (segmentElems > rowsPerTask_) {
                    segmentElems = rowsPerTask_;
                }
                return segmentElems > 0;
            }
            currentTask -= tasksForInput;
            currentPrefix += split;
        }
        return false;
    }

    __aicore__ inline void MoveSegment(GlobalTensor<T> &inputGm, int64_t inputOffset, int64_t outputOffset,
                                       int64_t elemCount) {
        const uint32_t count = static_cast<uint32_t>(elemCount);
        LocalTensor<T> local = copyQueue_.template AllocTensor<T>();
        if ((count * sizeof(T)) % kAlignBytes == 0) {
            DataCopy(local, inputGm[inputOffset], count);
            copyQueue_.EnQue(local);
            local = copyQueue_.template DeQue<T>();
            DataCopy(outputGm_[outputOffset], local, count);
        } else {
            const uint32_t bytes = count * sizeof(T);
            DataCopyExtParams params = {1, bytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(local, inputGm[inputOffset], params, padParams);
            copyQueue_.EnQue(local);
            local = copyQueue_.template DeQue<T>();
            DataCopyPad(outputGm_[outputOffset], local, params);
        }
        copyQueue_.FreeTensor(local);
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kBufferNum> copyQueue_;
    ListTensorDesc inputDesc_;
    GlobalTensor<T> outputGm_;
    int64_t inputCount_ = 0;
    int64_t outerCount_ = 0;
    int64_t innerSpan_ = 0;
    int64_t concatDim_ = 0;
    int64_t workItems_ = 0;
    int64_t rowsPerTask_ = 1;
    uint16_t splits_[kMaxInputs] = {};
};

class ConcatFp16UnitTransposeKernel {
  public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
        inputCount_ = tiling.input_count;
        outerCount_ = tiling.outer_count;
        rowsPerTask_ = tiling.rows_per_task;
        coreBoundaryRows_ = tiling.tile_elems;
        inputDesc_.Init((__gm__ void *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ half *)output, static_cast<uint64_t>(tiling.output_elems));

        const uint32_t paddedInputs = static_cast<uint32_t>((inputCount_ + 15) & ~15LL);
        const uint32_t storageTileElems = static_cast<uint32_t>(rowsPerTask_) * paddedInputs;
        pipe_.InitBuffer(loadQueue_, 1, storageTileElems * sizeof(half));
        pipe_.InitBuffer(storeQueue_, 1, storageTileElems * sizeof(half));
    }

    __aicore__ inline void Process() {
        if (inputCount_ <= 0 || outerCount_ <= 0 || rowsPerTask_ <= 0 || coreBoundaryRows_ <= 0) {
            return;
        }

        const int64_t coreIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        const int64_t boundaryCount = (outerCount_ + coreBoundaryRows_ - 1) / coreBoundaryRows_;
        const int64_t boundariesPerCore = boundaryCount / blockNum;
        const int64_t extraCores = boundaryCount - boundariesPerCore * blockNum;
        const int64_t firstBoundary = coreIdx * boundariesPerCore + (coreIdx < extraCores ? coreIdx : extraCores);
        const int64_t lastBoundary = firstBoundary + boundariesPerCore + (coreIdx < extraCores ? 1 : 0);
        const int64_t coreBegin = firstBoundary * coreBoundaryRows_;
        int64_t coreEnd = lastBoundary * coreBoundaryRows_;
        if (coreEnd > outerCount_) {
            coreEnd = outerCount_;
        }

        const int64_t rowCount = coreEnd - coreBegin;
        const int64_t tileCount = (rowCount + rowsPerTask_ - 1) / rowsPerTask_;
        for (int64_t tile = 0; tile < tileCount; ++tile) {
            const int64_t outerBegin = coreBegin + tile * rowsPerTask_;
            int64_t rows = coreEnd - outerBegin;
            if (rows > rowsPerTask_) {
                rows = rowsPerTask_;
            }
            ProcessRows(outerBegin, rows);
        }
    }

  private:
    __aicore__ inline void Transpose(LocalTensor<half> &outputLocal, LocalTensor<half> &inputLocal) {
        constexpr int64_t kTransposeWidth = 16;
        TransDataTo5HDParams params;
        params.dstHighHalf = false;
        params.srcHighHalf = false;
        params.repeatTimes = static_cast<uint8_t>((rowsPerTask_ + kTransposeWidth - 1) / kTransposeWidth);
        const int64_t paddedInputs = (inputCount_ + kTransposeWidth - 1) & ~(kTransposeWidth - 1);
        params.dstRepStride = static_cast<uint16_t>(paddedInputs);
        params.srcRepStride = 1;

        const uint64_t inputBase = reinterpret_cast<uint64_t>(inputLocal.GetPhyAddr());
        const uint64_t outputBase = reinterpret_cast<uint64_t>(outputLocal.GetPhyAddr());
        for (int64_t inputBaseIdx = 0; inputBaseIdx < paddedInputs; inputBaseIdx += kTransposeWidth) {
            uint64_t dstList[kTransposeWidth];
            uint64_t srcList[kTransposeWidth];
            for (int64_t lane = 0; lane < kTransposeWidth; ++lane) {
                srcList[lane] = inputBase +
                                static_cast<uint64_t>((inputBaseIdx + lane) * rowsPerTask_ * sizeof(half));
                dstList[lane] = outputBase +
                                static_cast<uint64_t>((lane * paddedInputs + inputBaseIdx) * sizeof(half));
            }
            TransDataTo5HD<uint16_t>(dstList, srcList, params);
        }
    }

    __aicore__ inline void ProcessRows(int64_t outerBegin, int64_t rowCount) {
        LocalTensor<half> inputLocal = loadQueue_.AllocTensor<half>();
        const uint32_t rowElems = static_cast<uint32_t>(rowCount);
        const uint32_t rowBytes = rowElems * sizeof(half);
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            __gm__ half *inputPtr = inputDesc_.GetDataPtr<half>(static_cast<uint32_t>(inputIdx));
            if (inputPtr == nullptr) {
                continue;
            }
            GlobalTensor<half> inputGm;
            inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outerCount_));
            LocalTensor<half> inputSlot = inputLocal[inputIdx * rowsPerTask_];
            if ((rowBytes % kAlignBytes) == 0) {
                DataCopy(inputSlot, inputGm[outerBegin], rowElems);
            } else {
                DataCopyExtParams params = {1, rowBytes, 0, 0, 0};
                DataCopyPadExtParams<half> padParams{false, 0, 0, static_cast<half>(0)};
                DataCopyPad(inputSlot, inputGm[outerBegin], params, padParams);
            }
        }
        loadQueue_.EnQue(inputLocal);
        inputLocal = loadQueue_.DeQue<half>();

        LocalTensor<half> outputLocal = storeQueue_.AllocTensor<half>();
        Transpose(outputLocal, inputLocal);
        storeQueue_.EnQue(outputLocal);
        outputLocal = storeQueue_.DeQue<half>();

        const uint32_t outputElems = static_cast<uint32_t>(rowCount * inputCount_);
        const int64_t paddedInputs = (inputCount_ + 15) & ~15LL;
        if (paddedInputs == inputCount_) {
            DataCopy(outputGm_[outerBegin * inputCount_], outputLocal, outputElems);
        } else {
            const uint32_t actualRowBytes = static_cast<uint32_t>(inputCount_ * sizeof(half));
            const uint32_t paddedRowBytes = static_cast<uint32_t>(paddedInputs * sizeof(half));
            const uint32_t alignedActualRowBytes =
                (actualRowBytes + kAlignBytes - 1U) / kAlignBytes * kAlignBytes;
            const DataCopyExtParams params = {
                static_cast<uint16_t>(rowCount), actualRowBytes,
                (paddedRowBytes - alignedActualRowBytes) / kAlignBytes, 0, 0};
            DataCopyPad(outputGm_[outerBegin * inputCount_], outputLocal, params);
        }
        storeQueue_.FreeTensor(outputLocal);
        loadQueue_.FreeTensor(inputLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> loadQueue_;
    TQue<QuePosition::VECOUT, 1> storeQueue_;
    ListTensorDesc inputDesc_;
    GlobalTensor<half> outputGm_;
    int64_t inputCount_ = 0;
    int64_t outerCount_ = 0;
    int64_t rowsPerTask_ = 1;
    int64_t coreBoundaryRows_ = 1;
};

template <typename T>
class ConcatPackedNarrowKernel {
  public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
        inputCount_ = tiling.input_count;
        outerCount_ = tiling.outer_count;
        workItems_ = tiling.work_items;
        rowsPerTask_ = tiling.rows_per_task;
        inputDesc_.Init((__gm__ void *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ T *)output, static_cast<uint64_t>(tiling.output_elems));

        const uint32_t tileElems = static_cast<uint32_t>(rowsPerTask_ * inputCount_);
        pipe_.InitBuffer(loadQueue_, 1, tileElems * sizeof(T));
        pipe_.InitBuffer(storeQueue_, 1, tileElems * sizeof(T));
        pipe_.InitBuffer(gatherOffsetsBuf_, tileElems * sizeof(uint32_t));
    }

    __aicore__ inline void Process() {
        if (inputCount_ <= 0 || outerCount_ <= 0 || workItems_ <= 0 || rowsPerTask_ <= 0) {
            return;
        }
        BuildGatherOffsets();

        const int64_t coreIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        const int64_t tasksPerCore = workItems_ / blockNum;
        const int64_t extraCores = workItems_ - tasksPerCore * blockNum;
        const int64_t firstTask = coreIdx * tasksPerCore + (coreIdx < extraCores ? coreIdx : extraCores);
        const int64_t lastTask = firstTask + tasksPerCore + (coreIdx < extraCores ? 1 : 0);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t outerBegin = task * rowsPerTask_;
            int64_t rows = outerCount_ - outerBegin;
            if (rows > rowsPerTask_) {
                rows = rowsPerTask_;
            }
            ProcessRows(outerBegin, rows);
        }
    }

  private:
    __aicore__ inline void BuildGatherOffsets() {
        LocalTensor<int32_t> offsets = gatherOffsetsBuf_.Get<int32_t>();
        const int32_t rowStrideBytes = static_cast<int32_t>(rowsPerTask_ * sizeof(T));
        ArithProgression<int32_t>(offsets, 0, rowStrideBytes, static_cast<uint32_t>(inputCount_));
        for (int64_t row = 1; row < rowsPerTask_; ++row) {
            Adds(offsets[row * inputCount_], offsets, static_cast<int32_t>(row * sizeof(T)),
                 static_cast<uint32_t>(inputCount_));
        }
    }

    __aicore__ inline void ProcessRows(int64_t outerBegin, int64_t rowCount) {
        LocalTensor<T> inputLocal = loadQueue_.AllocTensor<T>();
        const uint32_t rowElems = static_cast<uint32_t>(rowCount);
        const uint32_t rowBytes = rowElems * sizeof(T);
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            __gm__ T *inputPtr = inputDesc_.GetDataPtr<T>(static_cast<uint32_t>(inputIdx));
            if (inputPtr == nullptr) {
                continue;
            }
            GlobalTensor<T> inputGm;
            inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outerCount_));
            LocalTensor<T> inputSlot = inputLocal[inputIdx * rowsPerTask_];
            if ((rowBytes % kAlignBytes) == 0) {
                DataCopy(inputSlot, inputGm[outerBegin], rowElems);
            } else {
                DataCopyExtParams params = {1, rowBytes, 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(inputSlot, inputGm[outerBegin], params, padParams);
            }
        }
        loadQueue_.EnQue(inputLocal);
        inputLocal = loadQueue_.DeQue<T>();

        const uint32_t outputElems = static_cast<uint32_t>(rowCount * inputCount_);
        LocalTensor<T> outputLocal = storeQueue_.AllocTensor<T>();
        LocalTensor<uint32_t> offsets = gatherOffsetsBuf_.Get<uint32_t>();
        Gather(outputLocal, inputLocal, offsets, 0, outputElems);
        storeQueue_.EnQue(outputLocal);
        outputLocal = storeQueue_.DeQue<T>();
        if ((outputElems * sizeof(T)) % kAlignBytes == 0) {
            DataCopy(outputGm_[outerBegin * inputCount_], outputLocal, outputElems);
        } else {
            DataCopyExtParams params = {1, static_cast<uint32_t>(outputElems * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[outerBegin * inputCount_], outputLocal, params);
        }
        storeQueue_.FreeTensor(outputLocal);
        loadQueue_.FreeTensor(inputLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> loadQueue_;
    TQue<QuePosition::VECOUT, 1> storeQueue_;
    TBuf<QuePosition::VECCALC> gatherOffsetsBuf_;
    ListTensorDesc inputDesc_;
    GlobalTensor<T> outputGm_;
    int64_t inputCount_ = 0;
    int64_t outerCount_ = 0;
    int64_t workItems_ = 0;
    int64_t rowsPerTask_ = 1;
};

template <typename T, bool kPadded = false, bool kCompactIndices = false, bool kVectorInit = false>
class ConcatPackedRaggedNarrowKernel {
  public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
        inputCount_ = tiling.input_count;
        outerCount_ = tiling.outer_count;
        concatDim_ = tiling.concat_dim;
        workItems_ = tiling.work_items;
        rowsPerTask_ = tiling.rows_per_task;
        inputDesc_.Init((__gm__ void *)inputs);
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            splits_[inputIdx] = tiling.splits[inputIdx];
        }
        outputGm_.SetGlobalBuffer((__gm__ T *)output, static_cast<uint64_t>(tiling.output_elems));

        storageDim_ = kPadded ? AlignElems<T>(static_cast<uint32_t>(concatDim_)) : concatDim_;
        const uint32_t loadElems = static_cast<uint32_t>(rowsPerTask_ * concatDim_);
        const uint32_t outputElems = static_cast<uint32_t>(rowsPerTask_ * storageDim_);
        pipe_.InitBuffer(loadQueue_, 1, loadElems * sizeof(T));
        pipe_.InitBuffer(storeQueue_, 1, outputElems * sizeof(T));
        pipe_.InitBuffer(gatherOffsetsBuf_, outputElems * sizeof(uint32_t));
        pipe_.InitBuffer(gatherStridesBuf_,
                         AlignElems<int32_t>(static_cast<uint32_t>(storageDim_)) * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        if (inputCount_ <= 0 || outerCount_ <= 0 || workItems_ <= 0 || rowsPerTask_ <= 0) {
            return;
        }
        if constexpr (!kCompactIndices) BuildGatherOffsets();

        const int64_t coreIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        const int64_t tasksPerCore = workItems_ / blockNum;
        const int64_t extraCores = workItems_ - tasksPerCore * blockNum;
        const int64_t firstTask = coreIdx * tasksPerCore + (coreIdx < extraCores ? coreIdx : extraCores);
        const int64_t lastTask = firstTask + tasksPerCore + (coreIdx < extraCores ? 1 : 0);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t outerBegin = task * rowsPerTask_;
            int64_t rows = outerCount_ - outerBegin;
            if (rows > rowsPerTask_) {
                rows = rowsPerTask_;
            }
            ProcessRows(outerBegin, rows);
        }
    }

  private:
    __aicore__ inline bool IsUniformSplit(int64_t &split) const {
        split = splits_[0];
        if (split <= 0 || split * inputCount_ != concatDim_) {
            return false;
        }
        for (int64_t inputIdx = 1; inputIdx < inputCount_; ++inputIdx) {
            if (splits_[inputIdx] != split) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void BuildUniformGatherOffsets(LocalTensor<int32_t> offsets, int64_t split) {
        const int32_t elemBytes = static_cast<int32_t>(sizeof(T));
        ArithProgression<int32_t>(offsets, 0, elemBytes, static_cast<uint32_t>(split));
        for (int64_t inputIdx = 1; inputIdx < inputCount_; ++inputIdx) {
            Adds(offsets[inputIdx * split], offsets,
                 static_cast<int32_t>(inputIdx * split * rowsPerTask_ * elemBytes),
                 static_cast<uint32_t>(split));
        }
        const int32_t rowStrideBytes = static_cast<int32_t>(split * elemBytes);
        for (int64_t row = 1; row < rowsPerTask_; ++row) {
            Adds(offsets[row * concatDim_], offsets, static_cast<int32_t>(row) * rowStrideBytes,
                 static_cast<uint32_t>(concatDim_));
        }
    }

    __aicore__ inline void BuildGatherOffsets() {
        LocalTensor<int32_t> offsets = gatherOffsetsBuf_.Get<int32_t>();
        int64_t uniformSplit = 0;
        if (!kCompactIndices && !kPadded && IsUniformSplit(uniformSplit)) {
            BuildUniformGatherOffsets(offsets, uniformSplit);
            return;
        }
        LocalTensor<int32_t> strides = gatherStridesBuf_.Get<int32_t>();
        if constexpr (kVectorInit) {
            LocalTensor<int32_t> ramp = offsets[storageDim_];
            ArithProgression<int32_t>(ramp, 0, sizeof(T), static_cast<uint32_t>(storageDim_));
            if constexpr (kPadded) {
                Duplicate(offsets, 0, static_cast<uint32_t>(storageDim_));
                Duplicate(strides, 0, static_cast<uint32_t>(storageDim_));
            }
            int64_t prefix = concatDim_;
            for (int64_t i = inputCount_ - 1; i >= 0; --i) {
                const int64_t split = splits_[i];
                prefix -= split;
                if (split <= 0) continue;
                const int64_t head = prefix & 7;
                const int64_t begin = prefix - head;
                const uint32_t count = split + head;
                Adds(offsets[begin], ramp,
                    static_cast<int32_t>((prefix * rowsPerTask_ - head) * sizeof(T)), count);
                Duplicate(strides[begin], static_cast<int32_t>(split * sizeof(T)), count);
            }
        } else {
        int64_t concatOffset = 0;
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            const int64_t split = splits_[inputIdx];
            const int64_t stageBase = concatOffset * rowsPerTask_;
            if constexpr (kCompactIndices) {
                int64_t local = 0;
                for (; local < split && ((concatOffset + local) & 7) != 0; ++local) {
                    offsets.SetValue(concatOffset + local,
                                     static_cast<int32_t>((stageBase + local) * sizeof(T)));
                    strides.SetValue(concatOffset + local, static_cast<int32_t>(split * sizeof(T)));
                }
                if (local < split) {
                    ArithProgression<int32_t>(offsets[concatOffset + local],
                        static_cast<int32_t>((stageBase + local) * sizeof(T)),
                        static_cast<int32_t>(sizeof(T)), static_cast<uint32_t>(split - local));
                    Duplicate(strides[concatOffset + local], static_cast<int32_t>(split * sizeof(T)),
                              static_cast<uint32_t>(split - local));
                }
            } else {
            for (int64_t local = 0; local < split; ++local) {
                offsets.SetValue(concatOffset + local,
                                 static_cast<int32_t>((stageBase + local) * sizeof(T)));
                strides.SetValue(concatOffset + local, static_cast<int32_t>(split * sizeof(T)));
            }
            }
            concatOffset += split;
        }
        for (int64_t local = concatDim_; local < storageDim_; ++local) {
            offsets.SetValue(local, 0);
            strides.SetValue(local, 0);
        }
        }
        for (int64_t row = 1; row < rowsPerTask_; ++row) {
            if constexpr (kCompactIndices) {
                Add(offsets[row * storageDim_], offsets[(row - 1) * storageDim_], strides,
                    static_cast<uint32_t>(storageDim_));
            } else {
            Muls(offsets[row * storageDim_], strides, static_cast<int32_t>(row),
                 static_cast<uint32_t>(storageDim_));
            Add(offsets[row * storageDim_], offsets[row * storageDim_], offsets,
                static_cast<uint32_t>(storageDim_));
            }
        }
    }

    __aicore__ inline void ProcessRows(int64_t outerBegin, int64_t rowCount) {
        LocalTensor<T> inputLocal = loadQueue_.AllocTensor<T>();
        int64_t concatOffset = 0;
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            const int64_t split = splits_[inputIdx];
            if (split <= 0) {
                continue;
            }
            __gm__ T *inputPtr = inputDesc_.GetDataPtr<T>(static_cast<uint32_t>(inputIdx));
            if (inputPtr == nullptr) {
                concatOffset += split;
                continue;
            }
            GlobalTensor<T> inputGm;
            inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outerCount_ * split));
            const uint32_t inputElems = static_cast<uint32_t>(rowCount * split);
            const uint32_t inputBytes = inputElems * sizeof(T);
            LocalTensor<T> inputSlot = inputLocal[concatOffset * rowsPerTask_];
            if ((inputBytes % kAlignBytes) == 0) {
                DataCopy(inputSlot, inputGm[outerBegin * split], inputElems);
            } else {
                DataCopyExtParams params = {1, inputBytes, 0, 0, 0};
                DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(inputSlot, inputGm[outerBegin * split], params, padParams);
            }
            concatOffset += split;
        }
        loadQueue_.EnQue(inputLocal);
        if constexpr (kCompactIndices) {
            if (!offsetsReady_) {
                BuildGatherOffsets();
                offsetsReady_ = true;
            }
        }
        inputLocal = loadQueue_.DeQue<T>();

        const uint32_t outputElems = static_cast<uint32_t>(rowCount * storageDim_);
        LocalTensor<T> outputLocal = storeQueue_.AllocTensor<T>();
        LocalTensor<uint32_t> offsets = gatherOffsetsBuf_.Get<uint32_t>();
        Gather(outputLocal, inputLocal, offsets, 0, outputElems);
        storeQueue_.EnQue(outputLocal);
        outputLocal = storeQueue_.DeQue<T>();
        if (kPadded) {
            const DataCopyExtParams params = {static_cast<uint16_t>(rowCount),
                                              static_cast<uint32_t>(concatDim_ * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[outerBegin * concatDim_], outputLocal, params);
        } else if ((outputElems * sizeof(T)) % kAlignBytes == 0) {
            DataCopy(outputGm_[outerBegin * concatDim_], outputLocal, outputElems);
        } else {
            DataCopyExtParams params = {1, static_cast<uint32_t>(outputElems * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[outerBegin * concatDim_], outputLocal, params);
        }
        storeQueue_.FreeTensor(outputLocal);
        loadQueue_.FreeTensor(inputLocal);
    }

    bool offsetsReady_ = false;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> loadQueue_;
    TQue<QuePosition::VECOUT, 1> storeQueue_;
    TBuf<QuePosition::VECCALC> gatherOffsetsBuf_;
    TBuf<QuePosition::VECCALC> gatherStridesBuf_;
    ListTensorDesc inputDesc_;
    GlobalTensor<T> outputGm_;
    int64_t inputCount_ = 0;
    int64_t outerCount_ = 0;
    int64_t concatDim_ = 0;
    int64_t storageDim_ = 0;
    int64_t workItems_ = 0;
    int64_t rowsPerTask_ = 1;
    uint16_t splits_[kMaxInputs];
};

class ConcatPackedWideFp32Kernel {
  public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
        inputCount_ = tiling.input_count;
        outerCount_ = tiling.outer_count;
        concatDim_ = tiling.concat_dim;
        workItems_ = tiling.work_items;
        rowsPerTask_ = tiling.rows_per_task;
        rowTasks_ = (outerCount_ + rowsPerTask_ - 1) / rowsPerTask_;
        inputDesc_.Init((__gm__ void *)inputs);
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            splits_[inputIdx] = tiling.splits[inputIdx];
            inputPtrs_[inputIdx] = inputDesc_.GetDataPtr<float>(static_cast<uint32_t>(inputIdx));
        }
        BuildChunks();
        outputGm_.SetGlobalBuffer((__gm__ float *)output, static_cast<uint64_t>(tiling.output_elems));

        const uint32_t loadElems = static_cast<uint32_t>(rowsPerTask_ * maxChunkWidth_);
        const uint32_t storageWidth = AlignElems<float>(static_cast<uint32_t>(maxChunkWidth_));
        const uint32_t outputElems = static_cast<uint32_t>(rowsPerTask_) * storageWidth;
        pipe_.InitBuffer(loadQueue_, 1, loadElems * sizeof(float));
        pipe_.InitBuffer(storeQueue_, 1, outputElems * sizeof(float));
        pipe_.InitBuffer(gatherOffsetsBuf_, outputElems * sizeof(uint32_t));
        pipe_.InitBuffer(gatherStridesBuf_, storageWidth * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        if (chunkCount_ <= 0 || rowTasks_ <= 0 || workItems_ <= 0) {
            return;
        }
        const int64_t coreIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        const int64_t firstTask = workItems_ * coreIdx / blockNum;
        const int64_t lastTask = workItems_ * (coreIdx + 1) / blockNum;
        int64_t preparedChunk = -1;
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t chunkIdx = task / rowTasks_;
            const int64_t rowTask = task - chunkIdx * rowTasks_;
            if (chunkIdx != preparedChunk) {
                BuildGatherOffsets(chunkIdx);
                preparedChunk = chunkIdx;
            }
            const int64_t outerBegin = rowTask * rowsPerTask_;
            int64_t rows = outerCount_ - outerBegin;
            if (rows > rowsPerTask_) {
                rows = rowsPerTask_;
            }
            ProcessRows(chunkIdx, outerBegin, rows);
        }
    }

  private:
    __aicore__ inline void RecordChunk(int64_t firstInput, int64_t lastInput, int64_t width,
                                       int64_t globalOffset) {
        chunkFirst_[chunkCount_] = static_cast<int32_t>(firstInput);
        chunkLast_[chunkCount_] = static_cast<int32_t>(lastInput);
        chunkWidth_[chunkCount_] = static_cast<int32_t>(width);
        chunkOffset_[chunkCount_] = static_cast<int32_t>(globalOffset);
        if (width > maxChunkWidth_) {
            maxChunkWidth_ = width;
        }
        ++chunkCount_;
    }

    __aicore__ inline void BuildChunks() {
        int64_t firstInput = 0;
        int64_t width = 0;
        int64_t globalOffset = 0;
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            const int64_t split = splits_[inputIdx];
            if (split <= 0) {
                continue;
            }
            if (width > 0 && width + split > kPackedWideChunkElems) {
                RecordChunk(firstInput, inputIdx, width, globalOffset);
                globalOffset += width;
                firstInput = inputIdx;
                width = 0;
            }
            width += split;
            if (width == kPackedWideChunkElems) {
                RecordChunk(firstInput, inputIdx + 1, width, globalOffset);
                globalOffset += width;
                firstInput = inputIdx + 1;
                width = 0;
            }
        }
        if (width > 0) {
            RecordChunk(firstInput, inputCount_, width, globalOffset);
        }
    }

    __aicore__ inline void BuildGatherOffsets(int64_t chunkIdx) {
        LocalTensor<int32_t> offsets = gatherOffsetsBuf_.Get<int32_t>();
        LocalTensor<int32_t> strides = gatherStridesBuf_.Get<int32_t>();
        const int64_t width = chunkWidth_[chunkIdx];
        const int64_t storageWidth = AlignElems<float>(static_cast<uint32_t>(width));
        int64_t localOffset = 0;
        for (int64_t inputIdx = chunkFirst_[chunkIdx]; inputIdx < chunkLast_[chunkIdx]; ++inputIdx) {
            const int64_t split = splits_[inputIdx];
            if (split <= 0) {
                continue;
            }
            const int64_t stageBase = localOffset * rowsPerTask_;
            for (int64_t local = 0; local < split; ++local) {
                offsets.SetValue(localOffset + local,
                                 static_cast<int32_t>((stageBase + local) * sizeof(float)));
                strides.SetValue(localOffset + local, static_cast<int32_t>(split * sizeof(float)));
            }
            localOffset += split;
        }
        for (int64_t local = width; local < storageWidth; ++local) {
            offsets.SetValue(local, 0);
            strides.SetValue(local, 0);
        }
        for (int64_t row = 1; row < rowsPerTask_; ++row) {
            Muls(offsets[row * storageWidth], strides, static_cast<int32_t>(row),
                 static_cast<uint32_t>(storageWidth));
            Add(offsets[row * storageWidth], offsets[row * storageWidth], offsets,
                static_cast<uint32_t>(storageWidth));
        }
    }

    __aicore__ inline void ProcessRows(int64_t chunkIdx, int64_t outerBegin, int64_t rowCount) {
        LocalTensor<float> inputLocal = loadQueue_.AllocTensor<float>();
        int64_t localOffset = 0;
        for (int64_t inputIdx = chunkFirst_[chunkIdx]; inputIdx < chunkLast_[chunkIdx]; ++inputIdx) {
            const int64_t split = splits_[inputIdx];
            if (split <= 0) {
                continue;
            }
            __gm__ float *inputPtr = inputPtrs_[inputIdx];
            if (inputPtr == nullptr) {
                localOffset += split;
                continue;
            }
            GlobalTensor<float> inputGm;
            inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outerCount_ * split));
            const uint32_t inputElems = static_cast<uint32_t>(rowCount * split);
            LocalTensor<float> inputSlot = inputLocal[localOffset * rowsPerTask_];
            if ((inputElems * sizeof(float)) % kAlignBytes == 0) {
                DataCopy(inputSlot, inputGm[outerBegin * split], inputElems);
            } else {
                const DataCopyExtParams params = {
                    1, static_cast<uint32_t>(inputElems * sizeof(float)), 0, 0, 0};
                const DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
                DataCopyPad(inputSlot, inputGm[outerBegin * split], params, padParams);
            }
            localOffset += split;
        }
        loadQueue_.EnQue(inputLocal);
        inputLocal = loadQueue_.DeQue<float>();

        const int64_t width = chunkWidth_[chunkIdx];
        const int64_t storageWidth = AlignElems<float>(static_cast<uint32_t>(width));
        LocalTensor<float> outputLocal = storeQueue_.AllocTensor<float>();
        LocalTensor<uint32_t> offsets = gatherOffsetsBuf_.Get<uint32_t>();
        Gather(outputLocal, inputLocal, offsets, 0, static_cast<uint32_t>(rowCount * storageWidth));
        storeQueue_.EnQue(outputLocal);
        outputLocal = storeQueue_.DeQue<float>();
        const DataCopyExtParams params = {static_cast<uint16_t>(rowCount),
                                          static_cast<uint32_t>(width * sizeof(float)), 0,
                                          static_cast<uint32_t>((concatDim_ - width) * sizeof(float)), 0};
        const int64_t outputOffset = outerBegin * concatDim_ + chunkOffset_[chunkIdx];
        DataCopyPad(outputGm_[outputOffset], outputLocal, params);
        storeQueue_.FreeTensor(outputLocal);
        loadQueue_.FreeTensor(inputLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> loadQueue_;
    TQue<QuePosition::VECOUT, 1> storeQueue_;
    TBuf<QuePosition::VECCALC> gatherOffsetsBuf_;
    TBuf<QuePosition::VECCALC> gatherStridesBuf_;
    ListTensorDesc inputDesc_;
    GlobalTensor<float> outputGm_;
    __gm__ float *inputPtrs_[kMaxInputs];
    int64_t inputCount_ = 0;
    int64_t outerCount_ = 0;
    int64_t concatDim_ = 0;
    int64_t workItems_ = 0;
    int64_t rowsPerTask_ = 1;
    int64_t rowTasks_ = 0;
    int64_t chunkCount_ = 0;
    int64_t maxChunkWidth_ = 1;
    uint16_t splits_[kMaxInputs] = {};
    int32_t chunkFirst_[kMaxInputs] = {};
    int32_t chunkLast_[kMaxInputs] = {};
    int32_t chunkWidth_[kMaxInputs] = {};
    int32_t chunkOffset_[kMaxInputs] = {};
};

template <bool kWide = false, bool kVectorInit = false>
class ConcatBalancedGroupKernel {
  public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
        outerCount_ = tiling.outer_count;
        concatDim_ = tiling.concat_dim;
        rowsPerTask_ = tiling.rows_per_task;
        storeRows_ = kWide ? 2 : (rowsPerTask_ >= 16 ? 4 : rowsPerTask_);
        const int64_t coreIdx = GetBlockIdx();
        const int64_t kRowParts = GetBlockNum() / tiling.group_count;
        const int64_t groupIdx = coreIdx / kRowParts;
        valid_ = groupIdx < tiling.group_count;
        if (!valid_) {
            return;
        }
        firstInput_ = tiling.group_input_bounds[groupIdx];
        groupInputCount_ = tiling.group_input_bounds[groupIdx + 1] - firstInput_;
        outputOffset_ = tiling.group_output_offsets[groupIdx];
        groupWidth_ = tiling.group_output_offsets[groupIdx + 1] - outputOffset_;
        const int64_t rowPart = coreIdx % kRowParts;
        const int64_t taskCount = (outerCount_ + rowsPerTask_ - 1) / rowsPerTask_;
        const int64_t tasksPerPart = taskCount / kRowParts;
        const int64_t extraParts = taskCount - tasksPerPart * kRowParts;
        const int64_t firstTask = rowPart * tasksPerPart + (rowPart < extraParts ? rowPart : extraParts);
        const int64_t partTasks = tasksPerPart + (rowPart < extraParts ? 1 : 0);
        outerBegin_ = firstTask * rowsPerTask_;
        outerEnd_ = (firstTask + partTasks) * rowsPerTask_;
        if (outerEnd_ > outerCount_) {
            outerEnd_ = outerCount_;
        }

        inputDesc_.Init((__gm__ void *)inputs);
        uint32_t stageOffset = 0;
        int64_t localOffset = 0;
        for (int64_t localInput = 0; localInput < groupInputCount_; ++localInput) {
            const int64_t inputIdx = firstInput_ + localInput;
            splits_[localInput] = tiling.splits[inputIdx];
            localOffsets_[localInput] = static_cast<uint16_t>(localOffset);
            stageOffset = AlignElems<float>(stageOffset);
            stageOffsets_[localInput] = static_cast<uint16_t>(stageOffset);
            stageOffset += static_cast<uint32_t>(rowsPerTask_ * splits_[localInput]);
            localOffset += splits_[localInput];
            __gm__ float *inputPtr = inputDesc_.GetDataPtr<float>(static_cast<uint32_t>(inputIdx));
            inputGm_[localInput].SetGlobalBuffer(
                inputPtr, static_cast<uint64_t>(outerCount_ * splits_[localInput]));
            inputGm_[localInput].template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        outputGm_.SetGlobalBuffer((__gm__ float *)output, static_cast<uint64_t>(tiling.output_elems));

        const uint32_t storageWidth = AlignElems<float>(static_cast<uint32_t>(groupWidth_));
        const uint32_t inputBufferElems = AlignElems<float>(stageOffset);
        const uint32_t expandedOffsetBytes =
            static_cast<uint32_t>(3 * storeRows_ * storageWidth * sizeof(uint32_t));
        expandedStrides_ = (kWide ? 1 : 2) * inputBufferElems * sizeof(float) +
                               2 * storeRows_ * storageWidth * sizeof(float) + expandedOffsetBytes <=
                           192 * 1024 - 3 * 1024;
        pipe_.InitBuffer(loadQueue_, kWide ? 1 : 2, inputBufferElems * sizeof(float));
        pipe_.InitBuffer(storeQueue_, 2,
                         static_cast<uint32_t>(storeRows_ * storageWidth * sizeof(float)));
        pipe_.InitBuffer(gatherOffsetsBuf_,
                         static_cast<uint32_t>((2 * storeRows_ + (expandedStrides_ ? storeRows_ : 1)) *
                                               storageWidth * sizeof(uint32_t)));
    }

    __aicore__ inline void Process() {
        if (!valid_) {
            return;
        }
        if (outerBegin_ >= outerEnd_) {
            return;
        }
        int64_t currentBegin = outerBegin_;
        int64_t currentRows = CappedRows(currentBegin);
        PrefetchRows(currentBegin, currentRows);
        BuildGatherOffsets();
        while (currentBegin < outerEnd_) {
            LocalTensor<float> inputLocal = loadQueue_.DeQue<float>();
            const int64_t nextBegin = currentBegin + currentRows;
            const int64_t nextRows = CappedRows(nextBegin);
            if (!kWide && nextRows > 0) {
                PrefetchRows(nextBegin, nextRows);
            }
            GatherAndStore(inputLocal, currentBegin, currentRows);
            loadQueue_.FreeTensor(inputLocal);
            if constexpr (kWide) {
                if (nextRows > 0) PrefetchRows(nextBegin, nextRows);
            }
            currentBegin = nextBegin;
            currentRows = nextRows;
        }
    }

  private:
    __aicore__ inline void BuildGatherOffsets() {
        LocalTensor<int32_t> offsets = gatherOffsetsBuf_.Get<int32_t>();
        const int64_t storageWidth = AlignElems<float>(static_cast<uint32_t>(groupWidth_));
        LocalTensor<int32_t> strides = offsets[2 * storeRows_ * storageWidth];
        if constexpr (kVectorInit) {
            LocalTensor<int32_t> ramp = offsets[storeRows_ * storageWidth];
            const uint32_t rampCount = storageWidth < 264 ? storageWidth : 264;
            ArithProgression<int32_t>(ramp, 0, 4, rampCount);
            Duplicate(offsets, 0, static_cast<uint32_t>(storageWidth));
            Duplicate(strides, 0, static_cast<uint32_t>(storageWidth));
            for (int64_t i = groupInputCount_ - 1; i >= 0; --i) {
                const int64_t split = splits_[i];
                if (split <= 0) continue;
                const int64_t head = localOffsets_[i] & 7;
                const int64_t begin = localOffsets_[i] - head;
                const uint32_t count = split + head;
                Adds(offsets[begin], ramp,
                    static_cast<int32_t>((stageOffsets_[i] - head) * sizeof(float)), count);
                Duplicate(strides[begin], static_cast<int32_t>(split * sizeof(float)), count);
            }
        } else {
        for (int64_t localInput = 0; localInput < groupInputCount_; ++localInput) {
            const int64_t split = splits_[localInput];
            if (split <= 0) {
                continue;
            }
            const int64_t localOffset = localOffsets_[localInput];
            const int64_t stageBase = stageOffsets_[localInput];
            const int64_t segmentEnd = localOffset + split;
            int64_t vectorBegin = localOffset;
            while (vectorBegin < segmentEnd && (vectorBegin & 7) != 0) {
                const int64_t local = vectorBegin - localOffset;
                offsets.SetValue(vectorBegin, static_cast<int32_t>((stageBase + local) * sizeof(float)));
                strides.SetValue(vectorBegin, static_cast<int32_t>(split * sizeof(float)));
                ++vectorBegin;
            }
            if (vectorBegin < segmentEnd) {
                const int64_t local = vectorBegin - localOffset;
                const uint32_t vectorCount = static_cast<uint32_t>(segmentEnd - vectorBegin);
                ArithProgression<int32_t>(offsets[vectorBegin],
                                          static_cast<int32_t>((stageBase + local) * sizeof(float)),
                                          static_cast<int32_t>(sizeof(float)), vectorCount);
                Duplicate(strides[vectorBegin], static_cast<int32_t>(split * sizeof(float)), vectorCount);
            }
        }
        int64_t paddingBegin = groupWidth_;
        while (paddingBegin < storageWidth && (paddingBegin & 7) != 0) {
            offsets.SetValue(paddingBegin, 0);
            strides.SetValue(paddingBegin, 0);
            ++paddingBegin;
        }
        if (paddingBegin < storageWidth) {
            const uint32_t paddingCount = static_cast<uint32_t>(storageWidth - paddingBegin);
            Duplicate(offsets[paddingBegin], 0, paddingCount);
            Duplicate(strides[paddingBegin], 0, paddingCount);
        }
        }
        for (int64_t row = 1; row < storeRows_; ++row) {
            Add(offsets[row * storageWidth], offsets[(row - 1) * storageWidth], strides,
                static_cast<uint32_t>(storageWidth));
        }
        if (rowsPerTask_ > storeRows_) {
            Add(strides, strides, strides, static_cast<uint32_t>(storageWidth));
            if constexpr (!kWide) Add(strides, strides, strides, static_cast<uint32_t>(storageWidth));
        }
        if (expandedStrides_) {
            for (int64_t row = 1; row < storeRows_; ++row) {
                DataCopy(strides[row * storageWidth], strides, static_cast<uint32_t>(storageWidth));
            }
        }
    }

    __aicore__ inline int64_t CappedRows(int64_t outerBegin) const {
        if (outerBegin >= outerEnd_) {
            return 0;
        }
        const int64_t remain = outerEnd_ - outerBegin;
        return remain < rowsPerTask_ ? remain : rowsPerTask_;
    }

    __aicore__ inline void PrefetchRows(int64_t outerBegin, int64_t rowCount) {
        LocalTensor<float> inputLocal = loadQueue_.AllocTensor<float>();
        LoadInputs(inputLocal, outerBegin, rowCount);
        loadQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void LoadInputs(LocalTensor<float> inputLocal, int64_t outerBegin, int64_t rowCount) {
        for (int64_t localInput = 0; localInput < groupInputCount_; ++localInput) {
            const int64_t split = splits_[localInput];
            if (split > 0) {
                LocalTensor<float> inputSlot = inputLocal[stageOffsets_[localInput]];
                const uint32_t inputElems = static_cast<uint32_t>(rowCount * split);
                if ((inputElems * sizeof(float)) % kAlignBytes == 0) {
                    DataCopy(inputSlot, inputGm_[localInput][outerBegin * split], inputElems);
                } else {
                    const DataCopyExtParams params = {
                        1, static_cast<uint32_t>(inputElems * sizeof(float)), 0, 0, 0};
                    const DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
                    DataCopyPad(inputSlot, inputGm_[localInput][outerBegin * split], params, padParams);
                }
            }
        }
    }

    __aicore__ inline void GatherAndStore(LocalTensor<float> inputLocal, int64_t outerBegin,
                                          int64_t rowCount) {
        const int64_t storageWidth = AlignElems<float>(static_cast<uint32_t>(groupWidth_));
        LocalTensor<int32_t> base = gatherOffsetsBuf_.Get<int32_t>();
        LocalTensor<int32_t> shifted = base[storeRows_ * storageWidth];
        LocalTensor<int32_t> strides = base[2 * storeRows_ * storageWidth];
        for (int64_t firstRow = 0; firstRow < rowCount; firstRow += storeRows_) {
            LocalTensor<int32_t> offsets = base;
            if (firstRow > 0) {
                LocalTensor<int32_t> previous = firstRow == storeRows_ ? base : shifted;
                if (expandedStrides_) {
                    Add(shifted, previous, strides, static_cast<uint32_t>(storeRows_ * storageWidth));
                } else {
                    for (int64_t row = 0; row < storeRows_; ++row) {
                        Add(shifted[row * storageWidth], previous[row * storageWidth], strides,
                            static_cast<uint32_t>(storageWidth));
                    }
                }
                offsets = shifted;
            }
            const int64_t rows = rowCount - firstRow < storeRows_ ? rowCount - firstRow : storeRows_;
            LocalTensor<float> outputLocal = storeQueue_.AllocTensor<float>();
            Gather(outputLocal, inputLocal, offsets.ReinterpretCast<uint32_t>(), 0,
                   static_cast<uint32_t>(rows * storageWidth));
            storeQueue_.EnQue(outputLocal);
            outputLocal = storeQueue_.DeQue<float>();
            const DataCopyExtParams params = {
                static_cast<uint16_t>(rows), static_cast<uint32_t>(groupWidth_ * sizeof(float)), 0,
                static_cast<uint32_t>((concatDim_ - groupWidth_) * sizeof(float)), 0};
            const int64_t outputBase = (outerBegin + firstRow) * concatDim_ + outputOffset_;
            DataCopyPad(outputGm_[outputBase], outputLocal, params);
            storeQueue_.FreeTensor(outputLocal);
        }
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> loadQueue_;
    TQue<QuePosition::VECOUT, 1> storeQueue_;
    TBuf<QuePosition::VECCALC> gatherOffsetsBuf_;
    ListTensorDesc inputDesc_;
    GlobalTensor<float> outputGm_;
    GlobalTensor<float> inputGm_[kWide ? 32 : kMaxBalancedGroupInputs];
    uint16_t splits_[kWide ? 32 : kMaxBalancedGroupInputs] = {};
    uint16_t localOffsets_[kWide ? 32 : kMaxBalancedGroupInputs] = {};
    uint16_t stageOffsets_[kWide ? 32 : kMaxBalancedGroupInputs] = {};
    bool valid_ = false;
    bool expandedStrides_ = false;
    int64_t outerCount_ = 0;
    int64_t concatDim_ = 0;
    int64_t rowsPerTask_ = 1;
    int64_t firstInput_ = 0;
    int64_t storeRows_ = 1;
    int64_t groupInputCount_ = 0;
    int64_t outputOffset_ = 0;
    int64_t groupWidth_ = 0;
    int64_t outerBegin_ = 0;
    int64_t outerEnd_ = 0;
};

template <bool kRagged, bool kPadded = false>
class ConcatPackedInt8Kernel {
  public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
        inputCount_ = tiling.input_count;
        outerCount_ = tiling.outer_count;
        concatDim_ = tiling.concat_dim;
        workItems_ = tiling.work_items;
        rowsPerTask_ = tiling.rows_per_task;
        inputDesc_.Init((__gm__ void *)inputs);
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            splits_[inputIdx] = tiling.splits[inputIdx];
        }
        outputGm_.SetGlobalBuffer((__gm__ int8_t *)output, static_cast<uint64_t>(tiling.output_elems));

        storageDim_ = kPadded ? AlignElems<int8_t>(static_cast<uint32_t>(concatDim_)) : concatDim_;
        loadTileElems_ = static_cast<uint32_t>(rowsPerTask_ * concatDim_);
        outputTileElems_ = static_cast<uint32_t>(rowsPerTask_ * storageDim_);
        const uint32_t inputBufferElems = loadTileElems_ > outputTileElems_ ? loadTileElems_ : outputTileElems_;
        pipe_.InitBuffer(inputBuf_, inputBufferElems * sizeof(int8_t));
        pipe_.InitBuffer(halfInputBuf_, loadTileElems_ * sizeof(half));
        pipe_.InitBuffer(halfOutputBuf_, outputTileElems_ * sizeof(half));
        pipe_.InitBuffer(gatherOffsetsBuf_, outputTileElems_ * sizeof(uint32_t));
        pipe_.InitBuffer(gatherStridesBuf_,
                         AlignElems<int32_t>(static_cast<uint32_t>(storageDim_)) * sizeof(int32_t));
    }

    __aicore__ inline void Process() {
        if (inputCount_ <= 0 || outerCount_ <= 0 || concatDim_ <= 0 || workItems_ <= 0 || rowsPerTask_ <= 0) {
            return;
        }
        BuildGatherOffsets();

        const int64_t coreIdx = GetBlockIdx();
        const int64_t blockNum = GetBlockNum();
        const int64_t tasksPerCore = workItems_ / blockNum;
        const int64_t extraCores = workItems_ - tasksPerCore * blockNum;
        const int64_t firstTask = coreIdx * tasksPerCore + (coreIdx < extraCores ? coreIdx : extraCores);
        const int64_t lastTask = firstTask + tasksPerCore + (coreIdx < extraCores ? 1 : 0);
        for (int64_t task = firstTask; task < lastTask; ++task) {
            const int64_t outerBegin = task * rowsPerTask_;
            int64_t rows = outerCount_ - outerBegin;
            if (rows > rowsPerTask_) {
                rows = rowsPerTask_;
            }
            ProcessRows(outerBegin, rows);
        }
    }

  private:
    __aicore__ inline bool IsUniformSplit(int64_t &split) const {
        split = splits_[0];
        if (!kRagged || split <= 0 || split * inputCount_ != concatDim_) {
            return false;
        }
        for (int64_t inputIdx = 1; inputIdx < inputCount_; ++inputIdx) {
            if (splits_[inputIdx] != split) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void BuildUniformGatherOffsets(LocalTensor<int32_t> offsets, int64_t split) {
        const int32_t elemBytes = static_cast<int32_t>(sizeof(half));
        ArithProgression<int32_t>(offsets, 0, elemBytes, static_cast<uint32_t>(split));
        for (int64_t inputIdx = 1; inputIdx < inputCount_; ++inputIdx) {
            Adds(offsets[inputIdx * split], offsets,
                 static_cast<int32_t>(inputIdx * split * rowsPerTask_ * elemBytes),
                 static_cast<uint32_t>(split));
        }
        const int32_t rowStrideBytes = static_cast<int32_t>(split * elemBytes);
        for (int64_t row = 1; row < rowsPerTask_; ++row) {
            Adds(offsets[row * concatDim_], offsets, static_cast<int32_t>(row) * rowStrideBytes,
                 static_cast<uint32_t>(concatDim_));
        }
    }

    __aicore__ inline void BuildGatherOffsets() {
        LocalTensor<int32_t> offsets = gatherOffsetsBuf_.Get<int32_t>();
        int64_t uniformSplit = 0;
        if (!kPadded && IsUniformSplit(uniformSplit)) {
            BuildUniformGatherOffsets(offsets, uniformSplit);
            return;
        }
        LocalTensor<int32_t> strides = gatherStridesBuf_.Get<int32_t>();
        if (kRagged) {
            int64_t concatOffset = 0;
            for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
                const int64_t split = splits_[inputIdx];
                const int64_t stageBase = concatOffset * rowsPerTask_;
                for (int64_t local = 0; local < split; ++local) {
                    offsets.SetValue(concatOffset + local,
                                     static_cast<int32_t>((stageBase + local) * sizeof(half)));
                    strides.SetValue(concatOffset + local, static_cast<int32_t>(split * sizeof(half)));
                }
                concatOffset += split;
            }
        } else {
            const int32_t rowStrideBytes = static_cast<int32_t>(rowsPerTask_ * sizeof(half));
            ArithProgression<int32_t>(offsets, 0, rowStrideBytes, static_cast<uint32_t>(concatDim_));
        }
        for (int64_t local = concatDim_; local < storageDim_; ++local) {
            offsets.SetValue(local, 0);
            strides.SetValue(local, 0);
        }
        for (int64_t row = 1; row < rowsPerTask_; ++row) {
            if (kRagged) {
                Muls(offsets[row * storageDim_], strides, static_cast<int32_t>(row),
                     static_cast<uint32_t>(storageDim_));
                Add(offsets[row * storageDim_], offsets[row * storageDim_], offsets,
                    static_cast<uint32_t>(storageDim_));
            } else {
                Adds(offsets[row * storageDim_], offsets, static_cast<int32_t>(row * sizeof(half)),
                     static_cast<uint32_t>(storageDim_));
            }
        }
    }

    __aicore__ inline void ProcessRows(int64_t outerBegin, int64_t rowCount) {
        LocalTensor<int8_t> inputLocal = inputBuf_.Get<int8_t>();
        int64_t concatOffset = 0;
        for (int64_t inputIdx = 0; inputIdx < inputCount_; ++inputIdx) {
            const int64_t split = kRagged ? splits_[inputIdx] : 1;
            if (split <= 0) {
                continue;
            }
            __gm__ int8_t *inputPtr = inputDesc_.GetDataPtr<int8_t>(static_cast<uint32_t>(inputIdx));
            if (inputPtr == nullptr) {
                concatOffset += split;
                continue;
            }
            GlobalTensor<int8_t> inputGm;
            inputGm.SetGlobalBuffer(inputPtr, static_cast<uint64_t>(outerCount_ * split));
            const uint32_t inputElems = static_cast<uint32_t>(rowCount * split);
            LocalTensor<int8_t> inputSlot = inputLocal[concatOffset * rowsPerTask_];
            if ((inputElems % kAlignBytes) == 0) {
                DataCopy(inputSlot, inputGm[outerBegin * split], inputElems);
            } else {
                DataCopyExtParams params = {1, inputElems, 0, 0, 0};
                DataCopyPadExtParams<int8_t> padParams{false, 0, 0, static_cast<int8_t>(0)};
                DataCopyPad(inputSlot, inputGm[outerBegin * split], params, padParams);
            }
            concatOffset += split;
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        LocalTensor<half> halfInput = halfInputBuf_.Get<half>();
        LocalTensor<half> halfOutput = halfOutputBuf_.Get<half>();
        Cast(halfInput, inputLocal, RoundMode::CAST_NONE, loadTileElems_);
        const uint32_t outputElems = static_cast<uint32_t>(rowCount * storageDim_);
        LocalTensor<uint32_t> offsets = gatherOffsetsBuf_.Get<uint32_t>();
        Gather(halfOutput, halfInput, offsets, 0, outputElems);
        Cast<int8_t>(inputLocal, halfOutput, RoundMode::CAST_RINT, outputElems);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        if (kPadded) {
            const DataCopyExtParams params = {static_cast<uint16_t>(rowCount),
                                              static_cast<uint32_t>(concatDim_), 0, 0, 0};
            DataCopyPad(outputGm_[outerBegin * concatDim_], inputLocal, params);
        } else if ((outputElems % kAlignBytes) == 0) {
            DataCopy(outputGm_[outerBegin * concatDim_], inputLocal, outputElems);
        } else {
            DataCopyExtParams params = {1, outputElems, 0, 0, 0};
            DataCopyPad(outputGm_[outerBegin * concatDim_], inputLocal, params);
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    TPipe pipe_;
    TBuf<QuePosition::VECCALC> inputBuf_;
    TBuf<QuePosition::VECCALC> halfInputBuf_;
    TBuf<QuePosition::VECCALC> halfOutputBuf_;
    TBuf<QuePosition::VECCALC> gatherOffsetsBuf_;
    TBuf<QuePosition::VECCALC> gatherStridesBuf_;
    ListTensorDesc inputDesc_;
    GlobalTensor<int8_t> outputGm_;
    uint32_t loadTileElems_ = 0;
    uint32_t outputTileElems_ = 0;
    int64_t inputCount_ = 0;
    int64_t outerCount_ = 0;
    int64_t concatDim_ = 0;
    int64_t storageDim_ = 0;
    int64_t workItems_ = 0;
    int64_t rowsPerTask_ = 1;
    uint16_t splits_[kMaxInputs] = {};
};

template <typename T, bool kInputMajorBalanced = false, bool kInputReadBypass = false>
__aicore__ inline void RunConcat(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
    ConcatCopyKernel<T, kInputMajorBalanced, kInputReadBypass> kernel;
    kernel.Init(inputs, output, tiling);
    kernel.Process();
}

__aicore__ inline void RunConcatFp16UnitTranspose(GM_ADDR inputs, GM_ADDR output,
                                                  const ConcatTilingData &tiling) {
    ConcatFp16UnitTransposeKernel kernel;
    kernel.Init(inputs, output, tiling);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunConcatInputSegments(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
    ConcatInputSegmentKernel<T> kernel;
    kernel.Init(inputs, output, tiling);
    kernel.Process();
}

template <typename T>
__aicore__ inline void RunConcatPackedNarrow(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
    ConcatPackedNarrowKernel<T> kernel;
    kernel.Init(inputs, output, tiling);
    kernel.Process();
}

template <typename T, bool kPadded = false>
__aicore__ inline void RunConcatPackedRaggedNarrow(GM_ADDR inputs, GM_ADDR output,
                                                   const ConcatTilingData &tiling) {
    ConcatPackedRaggedNarrowKernel<T, kPadded> kernel;
    kernel.Init(inputs, output, tiling);
    kernel.Process();
}

template <bool kRagged, bool kPadded = false>
__aicore__ inline void RunConcatPackedInt8(GM_ADDR inputs, GM_ADDR output, const ConcatTilingData &tiling) {
    ConcatPackedInt8Kernel<kRagged, kPadded> kernel;
    kernel.Init(inputs, output, tiling);
    kernel.Process();
}

}

extern "C" __global__ __aicore__ void concat(GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    (void)workspace;
    ConcatTilingData tilingData;
    const __gm__ ConcatTilingData *src = (const __gm__ ConcatTilingData *)tiling;
    tilingData.input_count = src->input_count;
    tilingData.output_elems = src->output_elems;
    tilingData.outer_count = src->outer_count;
    tilingData.inner_span = src->inner_span;
    tilingData.concat_dim = src->concat_dim;
    tilingData.element_bytes = src->element_bytes;
    tilingData.work_items = src->work_items;
    tilingData.work_items_per_core = src->work_items_per_core;
    tilingData.rows_per_task = src->rows_per_task;
    tilingData.tile_elems = src->tile_elems;
    tilingData.schedule_mode = src->schedule_mode;
    tilingData.flat_input_index = src->flat_input_index;
    tilingData.group_count = src->group_count;
    int64_t firstInput = 0;
    int64_t lastInput = tilingData.input_count;
    if (TILING_KEY_IS(72) || TILING_KEY_IS(51)) {
        const int64_t groups = tilingData.group_count;
        for (int64_t i = 0; i <= groups; ++i) {
            tilingData.group_input_bounds[i] = src->group_input_bounds[i];
            tilingData.group_output_offsets[i] = src->group_output_offsets[i];
        }
        const int64_t group = GetBlockIdx() / (GetBlockNum() / groups);
        firstInput = tilingData.group_input_bounds[group];
        lastInput = tilingData.group_input_bounds[group + 1];
    }
    for (int64_t i = firstInput; i < lastInput && i < kMaxInputs; ++i) {
        tilingData.splits[i] = src->splits[i];
    }

    if (TILING_KEY_IS(74)) {
        ConcatSmallInputDirectRowsKernel<DTYPE_INPUTS, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(72)) {
        ConcatBalancedGroupKernel<true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(70)) {
        ConcatPackedRaggedNarrowKernel<half, false, true, true> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(10771)) {
        RunConcatFp16UnitTranspose(inputs, output, tilingData);
    } else if (TILING_KEY_IS(0)) {
        RunConcat<half>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(1)) {
        RunConcat<float>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(2)) {
        RunConcat<int32_t>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(3)) {
        RunConcat<int8_t>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(10)) {
        RunConcatInputSegments<half>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(11)) {
        RunConcatInputSegments<float>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(12)) {
        RunConcatInputSegments<int32_t>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(13)) {
        RunConcatInputSegments<int8_t>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(20)) {
        RunConcatPackedNarrow<half>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(21)) {
        RunConcatPackedNarrow<float>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(22)) {
        RunConcatPackedNarrow<int32_t>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(23)) {
        RunConcatPackedInt8<false>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(30)) {
        RunConcatPackedRaggedNarrow<half>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(31)) {
        RunConcatPackedRaggedNarrow<float>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(32)) {
        RunConcatPackedRaggedNarrow<int32_t>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(33)) {
        RunConcatPackedInt8<true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(40)) {
        RunConcatPackedRaggedNarrow<half, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(41)) {
        RunConcatPackedRaggedNarrow<float, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(42)) {
        RunConcatPackedRaggedNarrow<int32_t, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(43)) {
        RunConcatPackedInt8<true, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(50)) {
        ConcatPackedWideFp32Kernel kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(51)) {
        ConcatBalancedGroupKernel<> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(52)) {
        ConcatSmallInputDirectRowsKernel<DTYPE_INPUTS> kernel;
        kernel.Init(inputs, output, tilingData);
        kernel.Process();
    } else if (TILING_KEY_IS(54)) {
        RunConcat<half, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(55)) {
        RunConcat<float, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(56)) {
        RunConcat<int32_t, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(57)) {
        RunConcat<int8_t, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(60)) {
        RunConcat<half, false, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(61)) {
        RunConcat<float, false, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(62)) {
        RunConcat<int32_t, false, true>(inputs, output, tilingData);
    } else if (TILING_KEY_IS(63)) {
        RunConcat<int8_t, false, true>(inputs, output, tilingData);
    }
}
