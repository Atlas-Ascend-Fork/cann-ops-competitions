/*!
 * \file inplace_update.h
 * \brief Kernel implementation for InplaceUpdate.
 */

#ifndef INPLACEUPDATE_H
#define INPLACEUPDATE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "inplace_update_tiling_data.h"
#include "inplace_update_tiling_key.h"

namespace NsInplaceUpdate {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t MAX_DATA_UB_BYTES = 144 * 1024;
constexpr int64_t MAX_INDEX_UB_COUNT = 1024;
constexpr int64_t SMALL_INDEX_SKIP_LIMIT = 8;
constexpr int64_t LARGE_ROW_SKIP_BYTES = 8 * 1024;
constexpr int64_t PINGPONG_COPY_BYTES_MIN = 64 * 1024;
constexpr int64_t PINGPONG_COPY_BYTES_LIMIT = 2 * 1024 * 1024;
constexpr int64_t MAX_INT32_ROW = 2147483647LL;
constexpr int64_t SMALL_DIRECT_DATA_UB_BYTES = 256;
constexpr uint32_t TINY_STATIC_BYTES = 128;
constexpr uint32_t PACKED_STATIC_BYTES = 512;
constexpr int32_t MAX_LARGE_UPDATE_BATCH = 32;

__aicore__ inline int64_t MinInt64(int64_t lhs, int64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline int64_t MaxInt64(int64_t lhs, int64_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

class InplaceUpdate {
public:
    __aicore__ inline InplaceUpdate(){};

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR i,
        GM_ADDR v,
        GM_ADDR y,
        const InplaceUpdateTilingData* tilingData,
        bool scalarSingleIndex = false);
    __aicore__ inline void InitSmall(GM_ADDR x, GM_ADDR i, GM_ADDR v, GM_ADDR y, const InplaceUpdateTilingData* tilingData);
    __aicore__ inline void InitLargeIndex(GM_ADDR x, GM_ADDR i, GM_ADDR v, GM_ADDR y, const InplaceUpdateTilingData* tilingData);
    __aicore__ inline void InitSmallStatic(GM_ADDR x, GM_ADDR i, GM_ADDR v, GM_ADDR y, const InplaceUpdateTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessWithRowFilter();
    __aicore__ inline void ProcessTinySingleCore();
    __aicore__ inline void ProcessSmallDirect();
    __aicore__ inline void ProcessSmallStatic();
    template <bool ROW_ALIGNED>
    __aicore__ inline void ProcessLargeIndexRowAligned();
    __aicore__ inline void ProcessLargeIndexRowAlignedStatic2048();
    __aicore__ inline void ProcessWideRowBalanced();
    __aicore__ inline void ProcessWideRowBalancedIndex4();
    __aicore__ inline void ProcessSingleIndexScalar();

private:
    template <bool FROM_V, bool GUARANTEED_ALIGNED = false>
    __aicore__ inline void CopyBytes(int64_t srcOffset, int64_t dstOffset, int64_t byteCount);
    template <bool FROM_V, bool GUARANTEED_STREAM = false>
    __aicore__ inline void CopyWideBytesStream(int64_t srcOffset, int64_t dstOffset, int64_t byteCount);
    template <bool GUARANTEED_STREAM>
    __aicore__ inline void ProcessWideRowBalancedImpl();
    template <bool GUARANTEED_STREAM, int32_t INDEX_COUNT>
    __aicore__ inline void ProcessWidePrepared(
        int64_t ownedRowStart, int64_t ownedRowEnd);
    __aicore__ inline void FinalizeWideBytesStream();
    template <bool FROM_V>
    __aicore__ inline void CopyTinyBytes(int64_t srcOffset, int64_t dstOffset, int64_t byteCount);
    template <bool FROM_V>
    __aicore__ inline void CopyTinyStaticBytes(int64_t srcOffset, int64_t dstOffset, int64_t byteCount);
    __aicore__ inline void ApplySmallDirectUpdate(int64_t index, int64_t innerBytes);
    __aicore__ inline void ApplySmallStaticUpdate(int64_t index, int64_t innerBytes);
    __aicore__ inline bool RowIsUpdated(int64_t row, LocalTensor<int32_t> indexLocal, int64_t currentIndexNum);
    __aicore__ inline bool HasLaterUpdateForRow(
        int64_t row,
        LocalTensor<int32_t> indexLocal,
        int64_t startIdx,
        int64_t currentIndexNum);
    __aicore__ inline bool EnableSmallIndexFormulaSkip();
    __aicore__ inline bool EnablePreparedSmallIndex();
    __aicore__ inline bool EnablePreparedInputSkip();
    __aicore__ inline bool CopyOwnedInputSkipSmallUpdates();
    __aicore__ inline void CopyOwnedInputSkipPrepared(LocalTensor<int32_t> indexLocal, int64_t currentIndexNum);
    __aicore__ inline void CopyOwnedInput();
    __aicore__ inline void ProcessPreparedSmallIndex();
    __aicore__ inline void ProcessPreparedSingleIndexValue(int64_t rawIndex);
    template <bool ENABLE_ROW_FILTER>
    __aicore__ inline void ApplyUpdatesImpl();
    __aicore__ inline void ApplyUpdatesPrepared(LocalTensor<int32_t> indexLocal, int64_t currentIndexNum);
    __aicore__ inline void ApplyUpdates();
    __aicore__ inline void ApplyUpdatesWithRowFilter();
    __aicore__ inline void ApplyUpdatesTinySingleCore();
    __aicore__ inline void ApplyLargeIndexRowAlignedOne(
        LocalTensor<int32_t> indexLocal,
        int64_t globalIdx,
        int64_t localIdx,
        int64_t ownedRowStart,
        int64_t ownedRowEnd);
    __aicore__ inline void ApplyLargeIndexRowAlignedOneInt32(
        LocalTensor<int32_t> indexLocal,
        int64_t globalIdx,
        int64_t localIdx,
        int32_t ownedRowStart,
        int32_t ownedRowEnd);
    __aicore__ inline void ApplyLargeIndexRowAlignedRawInt32(
        int32_t targetRow,
        int64_t globalIdx,
        int32_t ownedRowStart,
        int32_t ownedRowEnd);
    template <bool ROW_ALIGNED>
    __aicore__ inline void FlushLargeIndexOutputBatch(
        const uint16_t* targetOffsets,
        int32_t batchCount,
        int64_t ownedRowStart,
        int64_t slotBytes,
        int64_t rowBytes,
        bool waitForReuse,
        int32_t slotIdx,
        int64_t batchBytes);
    __aicore__ inline void FlushStatic2048ContiguousInputBatch(
        const uint16_t* targetOffsets,
        const uint16_t* sourceOffsets,
        int32_t batchCount,
        int64_t indexBase,
        int64_t ownedRowStart,
        bool waitForReuse,
        int32_t slotIdx,
        int64_t batchBytes);

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> dataBuffer;
    TBuf<TPosition::VECCALC> indexBuffer;

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockFactor_ = 1;
    int64_t ubFactor_ = 8;
    int64_t rowNum_ = 0;
    int64_t innerNum_ = 0;
    int64_t indexNum_ = 0;
    int64_t indexUbCount_ = 8;
    int64_t typeSize_ = 4;
    int64_t dataUbBytes_ = BLOCK_BYTES;
    int64_t wideStreamTileCount_ = 0;
    int32_t wideStreamBufferIdx_ = 0;
};

__aicore__ inline void InplaceUpdate::Init(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData,
    bool scalarSingleIndex)
{
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    totalNum_ = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor > 0 ? tilingData->blockFactor : totalNum_;
    ubFactor_ = tilingData->ubFactor > 0 ? tilingData->ubFactor : 8;
    rowNum_ = tilingData->rowNum;
    innerNum_ = tilingData->innerNum;
    indexNum_ = tilingData->indexNum;
    indexUbCount_ = tilingData->indexUbCount > 0 ? tilingData->indexUbCount : 8;
    typeSize_ = tilingData->typeSize > 0 ? tilingData->typeSize : 4;

    if (rowNum_ <= 0 && totalNum_ > 0) {
        rowNum_ = 1;
    }
    if (innerNum_ <= 0 && totalNum_ > 0) {
        innerNum_ = totalNum_;
    }
    if (indexUbCount_ > MAX_INDEX_UB_COUNT) {
        indexUbCount_ = MAX_INDEX_UB_COUNT;
    }

    dataUbBytes_ = ubFactor_ * typeSize_;
    if (dataUbBytes_ > MAX_DATA_UB_BYTES) {
        dataUbBytes_ = MAX_DATA_UB_BYTES;
        ubFactor_ = dataUbBytes_ / typeSize_;
    }
    if (dataUbBytes_ < BLOCK_BYTES) {
        dataUbBytes_ = BLOCK_BYTES;
    }
    int64_t indexBytes = indexUbCount_ * static_cast<int64_t>(sizeof(int32_t));
    if (indexBytes < BLOCK_BYTES) {
        indexBytes = BLOCK_BYTES;
    }

    pipe.InitBuffer(dataBuffer, dataUbBytes_);
    if (!scalarSingleIndex || indexNum_ != 1) {
        pipe.InitBuffer(indexBuffer, indexBytes);
    }
}

__aicore__ inline void InplaceUpdate::InitSmall(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    totalNum_ = tilingData->totalNum;
    blockFactor_ = tilingData->blockFactor > 0 ? tilingData->blockFactor : totalNum_;
    ubFactor_ = tilingData->ubFactor > 0 ? tilingData->ubFactor : 8;
    rowNum_ = tilingData->rowNum;
    innerNum_ = tilingData->innerNum;
    indexNum_ = tilingData->indexNum;
    typeSize_ = tilingData->typeSize > 0 ? tilingData->typeSize : 4;

    if (rowNum_ <= 0 && totalNum_ > 0) {
        rowNum_ = 1;
    }
    if (innerNum_ <= 0 && totalNum_ > 0) {
        innerNum_ = totalNum_;
    }

    dataUbBytes_ = ubFactor_ * typeSize_;
    if (dataUbBytes_ > SMALL_DIRECT_DATA_UB_BYTES) {
        dataUbBytes_ = SMALL_DIRECT_DATA_UB_BYTES;
    }
    if (dataUbBytes_ < BLOCK_BYTES) {
        dataUbBytes_ = BLOCK_BYTES;
    }

    pipe.InitBuffer(dataBuffer, dataUbBytes_);
}

__aicore__ inline void InplaceUpdate::InitLargeIndex(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    Init(x, i, v, y, tilingData);
}

__aicore__ inline void InplaceUpdate::InitSmallStatic(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    totalNum_ = tilingData->totalNum;
    rowNum_ = tilingData->rowNum;
    innerNum_ = tilingData->innerNum;
    indexNum_ = tilingData->indexNum;
    typeSize_ = tilingData->typeSize > 0 ? tilingData->typeSize : 4;
    if (rowNum_ <= 0 && totalNum_ > 0) {
        rowNum_ = 1;
    }
    if (innerNum_ <= 0 && totalNum_ > 0) {
        innerNum_ = totalNum_;
    }
}

template <bool FROM_V, bool GUARANTEED_ALIGNED>
__aicore__ inline void InplaceUpdate::CopyBytes(int64_t srcOffset, int64_t dstOffset, int64_t byteCount)
{
    if (byteCount <= 0) {
        return;
    }
    const bool fullyAligned = GUARANTEED_ALIGNED ||
        ((srcOffset | dstOffset | byteCount) & (BLOCK_BYTES - 1)) == 0;
    const int64_t pingPongBytes = ((dataUbBytes_ / BUFFER_NUM) / BLOCK_BYTES) * BLOCK_BYTES;
    const bool usePingPong = pingPongBytes >= BLOCK_BYTES && byteCount >= PINGPONG_COPY_BYTES_MIN &&
        byteCount <= PINGPONG_COPY_BYTES_LIMIT;
    if (!usePingPong) {
        int64_t done = 0;
        while (done < byteCount) {
            const int64_t currentBytes = MinInt64(byteCount - done, dataUbBytes_);
            const bool alignedCopy = fullyAligned || (((srcOffset + done) % BLOCK_BYTES == 0) &&
                ((dstOffset + done) % BLOCK_BYTES == 0) && (currentBytes % BLOCK_BYTES == 0));
            LocalTensor<uint8_t> local = dataBuffer.Get<uint8_t>();
            if constexpr (GUARANTEED_ALIGNED) {
                if constexpr (FROM_V) {
                    DataCopy(local, inputGMV[srcOffset + done], static_cast<uint32_t>(currentBytes));
                } else {
                    DataCopy(local, inputGMX[srcOffset + done], static_cast<uint32_t>(currentBytes));
                }
            } else if (alignedCopy) {
                if constexpr (FROM_V) {
                    DataCopy(local, inputGMV[srcOffset + done], static_cast<uint32_t>(currentBytes));
                } else {
                    DataCopy(local, inputGMX[srcOffset + done], static_cast<uint32_t>(currentBytes));
                }
            } else {
                DataCopyExtParams copyInParams{1, static_cast<uint32_t>(currentBytes), 0, 0, 0};
                DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
                if constexpr (FROM_V) {
                    DataCopyPad(local, inputGMV[srcOffset + done], copyInParams, padParams);
                } else {
                    DataCopyPad(local, inputGMX[srcOffset + done], copyInParams, padParams);
                }
            }
            SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
            WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
            if constexpr (GUARANTEED_ALIGNED) {
                DataCopy(outputGMY[dstOffset + done], local, static_cast<uint32_t>(currentBytes));
            } else if (alignedCopy) {
                DataCopy(outputGMY[dstOffset + done], local, static_cast<uint32_t>(currentBytes));
            } else {
                DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(currentBytes), 0, 0, 0};
                DataCopyPad(outputGMY[dstOffset + done], local, copyOutParams);
            }
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
            done += currentBytes;
        }
        return;
    }

    int64_t done = 0;
    bool pendingOut0 = false;
    bool pendingOut1 = false;
    int32_t bufferIdx = 0;
    LocalTensor<uint8_t> baseLocal = dataBuffer.Get<uint8_t>();
    while (done < byteCount) {
        const int64_t currentBytes = MinInt64(byteCount - done, pingPongBytes);
        const bool alignedCopy = fullyAligned || (((srcOffset + done) % BLOCK_BYTES == 0) &&
            ((dstOffset + done) % BLOCK_BYTES == 0) && (currentBytes % BLOCK_BYTES == 0));
        if (bufferIdx == 0 && pendingOut0) {
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
            pendingOut0 = false;
        }
        if (bufferIdx == 1 && pendingOut1) {
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(1));
            pendingOut1 = false;
        }
        LocalTensor<uint8_t> local = baseLocal[bufferIdx * pingPongBytes];
        if constexpr (GUARANTEED_ALIGNED) {
            if constexpr (FROM_V) {
                DataCopy(local, inputGMV[srcOffset + done], static_cast<uint32_t>(currentBytes));
            } else {
                DataCopy(local, inputGMX[srcOffset + done], static_cast<uint32_t>(currentBytes));
            }
        } else if (alignedCopy) {
            if constexpr (FROM_V) {
                DataCopy(local, inputGMV[srcOffset + done], static_cast<uint32_t>(currentBytes));
            } else {
                DataCopy(local, inputGMX[srcOffset + done], static_cast<uint32_t>(currentBytes));
            }
        } else {
            DataCopyExtParams copyInParams{1, static_cast<uint32_t>(currentBytes), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
            if constexpr (FROM_V) {
                DataCopyPad(local, inputGMV[srcOffset + done], copyInParams, padParams);
            } else {
                DataCopyPad(local, inputGMX[srcOffset + done], copyInParams, padParams);
            }
        }
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(bufferIdx));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(bufferIdx));
        if constexpr (GUARANTEED_ALIGNED) {
            DataCopy(outputGMY[dstOffset + done], local, static_cast<uint32_t>(currentBytes));
        } else if (alignedCopy) {
            DataCopy(outputGMY[dstOffset + done], local, static_cast<uint32_t>(currentBytes));
        } else {
            DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(currentBytes), 0, 0, 0};
            DataCopyPad(outputGMY[dstOffset + done], local, copyOutParams);
        }
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(bufferIdx));
        if (bufferIdx == 0) {
            pendingOut0 = true;
        } else {
            pendingOut1 = true;
        }
        done += currentBytes;
        bufferIdx = 1 - bufferIdx;
    }
    if (pendingOut0) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
    }
    if (pendingOut1) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(1));
    }
}

__aicore__ inline void InplaceUpdate::FinalizeWideBytesStream()
{
    if (wideStreamTileCount_ > 0) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
    }
    if (wideStreamTileCount_ > 1) {
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(1));
    }
    wideStreamTileCount_ = 0;
    wideStreamBufferIdx_ = 0;
}

template <bool FROM_V, bool GUARANTEED_STREAM>
__aicore__ inline void InplaceUpdate::CopyWideBytesStream(
    int64_t srcOffset, int64_t dstOffset, int64_t byteCount)
{
    if (byteCount <= 0) {
        return;
    }
    const int64_t pingPongBytes = ((dataUbBytes_ / BUFFER_NUM) / BLOCK_BYTES) * BLOCK_BYTES;
    if constexpr (!GUARANTEED_STREAM) {
        const bool canKeepStream = pingPongBytes >= BLOCK_BYTES && byteCount >= PINGPONG_COPY_BYTES_MIN &&
            byteCount <= PINGPONG_COPY_BYTES_LIMIT &&
            ((srcOffset | dstOffset | byteCount) & (BLOCK_BYTES - 1)) == 0;
        if (!canKeepStream) {
            FinalizeWideBytesStream();
            CopyBytes<FROM_V>(srcOffset, dstOffset, byteCount);
            return;
        }
    }

    LocalTensor<uint8_t> baseLocal = dataBuffer.Get<uint8_t>();
    int64_t done = 0;
    while (done < byteCount) {
        const int64_t currentBytes = MinInt64(byteCount - done, pingPongBytes);
        const TEventID eventId = static_cast<TEventID>(wideStreamBufferIdx_);
        if (wideStreamTileCount_ >= BUFFER_NUM) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventId);
        }

        LocalTensor<uint8_t> local = baseLocal[wideStreamBufferIdx_ * pingPongBytes];
        if constexpr (FROM_V) {
            DataCopy(local, inputGMV[srcOffset + done], static_cast<uint32_t>(currentBytes));
        } else {
            DataCopy(local, inputGMX[srcOffset + done], static_cast<uint32_t>(currentBytes));
        }
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        DataCopy(outputGMY[dstOffset + done], local, static_cast<uint32_t>(currentBytes));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        ++wideStreamTileCount_;
        done += currentBytes;
        wideStreamBufferIdx_ = 1 - wideStreamBufferIdx_;
    }
}

template <bool FROM_V>
__aicore__ inline void InplaceUpdate::CopyTinyBytes(int64_t srcOffset, int64_t dstOffset, int64_t byteCount)
{
    if (byteCount <= 0) {
        return;
    }
    LocalTensor<uint8_t> local = dataBuffer.Get<uint8_t>();
    const bool alignedCopy =
        (srcOffset % BLOCK_BYTES == 0) && (dstOffset % BLOCK_BYTES == 0) && (byteCount % BLOCK_BYTES == 0);
    if (alignedCopy) {
        if constexpr (FROM_V) {
            DataCopy(local, inputGMV[srcOffset], static_cast<uint32_t>(byteCount));
        } else {
            DataCopy(local, inputGMX[srcOffset], static_cast<uint32_t>(byteCount));
        }
    } else {
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(byteCount), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        if constexpr (FROM_V) {
            DataCopyPad(local, inputGMV[srcOffset], copyInParams, padParams);
        } else {
            DataCopyPad(local, inputGMX[srcOffset], copyInParams, padParams);
        }
    }
    SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
    if (alignedCopy) {
        DataCopy(outputGMY[dstOffset], local, static_cast<uint32_t>(byteCount));
    } else {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(byteCount), 0, 0, 0};
        DataCopyPad(outputGMY[dstOffset], local, copyOutParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
}

template <bool FROM_V>
__aicore__ inline void InplaceUpdate::CopyTinyStaticBytes(int64_t srcOffset, int64_t dstOffset, int64_t byteCount)
{
    if (byteCount <= 0 || byteCount > static_cast<int64_t>(TINY_STATIC_BYTES)) {
        return;
    }
    LocalTensor<uint8_t> local(TPosition::VECCALC, 0, TINY_STATIC_BYTES);
    const bool alignedCopy =
        (srcOffset % BLOCK_BYTES == 0) && (dstOffset % BLOCK_BYTES == 0) && (byteCount % BLOCK_BYTES == 0);
    if (alignedCopy) {
        if constexpr (FROM_V) {
            DataCopy(local, inputGMV[srcOffset], static_cast<uint32_t>(byteCount));
        } else {
            DataCopy(local, inputGMX[srcOffset], static_cast<uint32_t>(byteCount));
        }
    } else {
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(byteCount), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        if constexpr (FROM_V) {
            DataCopyPad(local, inputGMV[srcOffset], copyInParams, padParams);
        } else {
            DataCopyPad(local, inputGMX[srcOffset], copyInParams, padParams);
        }
    }
    SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
    if (alignedCopy) {
        DataCopy(outputGMY[dstOffset], local, static_cast<uint32_t>(byteCount));
    } else {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(byteCount), 0, 0, 0};
        DataCopyPad(outputGMY[dstOffset], local, copyOutParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
}

__aicore__ inline void InplaceUpdate::ApplySmallDirectUpdate(int64_t index, int64_t innerBytes)
{
    int64_t targetRow = static_cast<int64_t>(inputGMI.GetValue(index));
    if (targetRow < 0) {
        targetRow += rowNum_;
    }
    if (targetRow < 0 || targetRow >= rowNum_) {
        return;
    }
    const int64_t srcBase = index * innerBytes;
    const int64_t dstBase = targetRow * innerBytes;
    CopyTinyBytes<true>(srcBase, dstBase, innerBytes);
}

__aicore__ inline void InplaceUpdate::ApplySmallStaticUpdate(int64_t index, int64_t innerBytes)
{
    int64_t targetRow = static_cast<int64_t>(inputGMI.GetValue(index));
    if (targetRow < 0) {
        targetRow += rowNum_;
    }
    if (targetRow < 0 || targetRow >= rowNum_) {
        return;
    }
    const int64_t srcBase = index * innerBytes;
    const int64_t dstBase = targetRow * innerBytes;
    CopyTinyStaticBytes<true>(srcBase, dstBase, innerBytes);
}

__aicore__ inline bool InplaceUpdate::RowIsUpdated(
    int64_t row,
    LocalTensor<int32_t> indexLocal,
    int64_t currentIndexNum)
{
    for (int64_t localIdx = 0; localIdx < currentIndexNum; ++localIdx) {
        int64_t targetRow = static_cast<int64_t>(indexLocal.GetValue(localIdx));
        if (targetRow < 0) {
            targetRow += rowNum_;
        }
        if (targetRow == row) {
            return true;
        }
    }
    return false;
}

__aicore__ inline bool InplaceUpdate::HasLaterUpdateForRow(
    int64_t row,
    LocalTensor<int32_t> indexLocal,
    int64_t startIdx,
    int64_t currentIndexNum)
{
    for (int64_t localIdx = startIdx; localIdx < currentIndexNum; ++localIdx) {
        int64_t targetRow = static_cast<int64_t>(indexLocal.GetValue(localIdx));
        if (targetRow < 0) {
            targetRow += rowNum_;
        }
        if (targetRow == row) {
            return true;
        }
    }
    return false;
}

__aicore__ inline bool InplaceUpdate::EnableSmallIndexFormulaSkip()
{
    if (indexNum_ <= 0 || indexNum_ > SMALL_INDEX_SKIP_LIMIT || indexNum_ > indexUbCount_ || rowNum_ <= 0 ||
        innerNum_ <= 0 || typeSize_ <= 0) {
        return false;
    }
    return rowNum_ == 1 || innerNum_ * typeSize_ >= LARGE_ROW_SKIP_BYTES;
}

__aicore__ inline bool InplaceUpdate::EnablePreparedSmallIndex()
{
    return indexNum_ > 0 && indexNum_ <= SMALL_INDEX_SKIP_LIMIT && indexNum_ <= indexUbCount_ && rowNum_ > 0 &&
        innerNum_ > 0 && typeSize_ > 0;
}

__aicore__ inline bool InplaceUpdate::EnablePreparedInputSkip()
{
    return EnablePreparedSmallIndex() && (rowNum_ == 1 || innerNum_ * typeSize_ >= LARGE_ROW_SKIP_BYTES);
}

__aicore__ inline bool InplaceUpdate::CopyOwnedInputSkipSmallUpdates()
{
    if (!EnableSmallIndexFormulaSkip()) {
        return false;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t ownedStart = blockIdx * blockFactor_;
    if (ownedStart >= totalNum_) {
        return true;
    }
    const int64_t ownedEnd = MinInt64(totalNum_, ownedStart + blockFactor_);

    LocalTensor<int32_t> indexLocal = indexBuffer.Get<int32_t>();
    DataCopyExtParams indexCopyParams{
        1,
        static_cast<uint32_t>(indexNum_ * static_cast<int64_t>(sizeof(int32_t))),
        0,
        0,
        0};
    DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
    DataCopyPad(indexLocal, inputGMI[0], indexCopyParams, indexPadParams);
    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

    int64_t cursor = ownedStart;
    int64_t copyStart = ownedStart;
    while (cursor < ownedEnd) {
        const int64_t row = cursor / innerNum_;
        const int64_t rowEnd = MinInt64(ownedEnd, (row + 1) * innerNum_);
        if (RowIsUpdated(row, indexLocal, indexNum_)) {
            if (copyStart < cursor) {
                CopyBytes<false>(copyStart * typeSize_, copyStart * typeSize_, (cursor - copyStart) * typeSize_);
            }
            copyStart = rowEnd;
        }
        cursor = rowEnd;
    }
    if (copyStart < ownedEnd) {
        CopyBytes<false>(copyStart * typeSize_, copyStart * typeSize_, (ownedEnd - copyStart) * typeSize_);
    }
    return true;
}

__aicore__ inline void InplaceUpdate::CopyOwnedInputSkipPrepared(
    LocalTensor<int32_t> indexLocal,
    int64_t currentIndexNum)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t ownedStart = blockIdx * blockFactor_;
    if (ownedStart >= totalNum_) {
        return;
    }
    const int64_t ownedEnd = MinInt64(totalNum_, ownedStart + blockFactor_);

    int64_t cursor = ownedStart;
    int64_t copyStart = ownedStart;
    while (cursor < ownedEnd) {
        const int64_t row = cursor / innerNum_;
        const int64_t rowEnd = MinInt64(ownedEnd, (row + 1) * innerNum_);
        if (RowIsUpdated(row, indexLocal, currentIndexNum)) {
            if (copyStart < cursor) {
                CopyBytes<false>(copyStart * typeSize_, copyStart * typeSize_, (cursor - copyStart) * typeSize_);
            }
            copyStart = rowEnd;
        }
        cursor = rowEnd;
    }
    if (copyStart < ownedEnd) {
        CopyBytes<false>(copyStart * typeSize_, copyStart * typeSize_, (ownedEnd - copyStart) * typeSize_);
    }
}

__aicore__ inline void InplaceUpdate::CopyOwnedInput()
{
    if (totalNum_ <= 0 || blockFactor_ <= 0 || typeSize_ <= 0) {
        return;
    }

    if (CopyOwnedInputSkipSmallUpdates()) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t start = blockIdx * blockFactor_;
    if (start >= totalNum_) {
        return;
    }
    const int64_t currentNum = MinInt64(blockFactor_, totalNum_ - start);
    CopyBytes<false>(start * typeSize_, start * typeSize_, currentNum * typeSize_);
}

__aicore__ inline void InplaceUpdate::ApplyUpdatesPrepared(
    LocalTensor<int32_t> indexLocal,
    int64_t currentIndexNum)
{
    if (totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 || currentIndexNum <= 0 || blockFactor_ <= 0 ||
        typeSize_ <= 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t ownedStart = blockIdx * blockFactor_;
    if (ownedStart >= totalNum_) {
        return;
    }
    const int64_t ownedEnd = MinInt64(totalNum_, ownedStart + blockFactor_);

    for (int64_t localIdx = 0; localIdx < currentIndexNum; ++localIdx) {
        int64_t targetRow = static_cast<int64_t>(indexLocal.GetValue(localIdx));
        if (targetRow < 0) {
            targetRow += rowNum_;
        }
        if (targetRow < 0 || targetRow >= rowNum_) {
            continue;
        }
        if (rowNum_ == 1 && HasLaterUpdateForRow(targetRow, indexLocal, localIdx + 1, currentIndexNum)) {
            continue;
        }

        const int64_t rowStart = targetRow * innerNum_;
        const int64_t rowEnd = rowStart + innerNum_;
        const int64_t overlapStart = MaxInt64(ownedStart, rowStart);
        const int64_t overlapEnd = MinInt64(ownedEnd, rowEnd);
        if (overlapStart >= overlapEnd) {
            continue;
        }

        const int64_t srcElem = localIdx * innerNum_ + (overlapStart - rowStart);
        const int64_t srcBytes = srcElem * typeSize_;
        const int64_t dstBytes = overlapStart * typeSize_;
        const int64_t copyBytes = (overlapEnd - overlapStart) * typeSize_;
        CopyBytes<true>(srcBytes, dstBytes, copyBytes);
    }
}

__aicore__ inline void InplaceUpdate::ProcessPreparedSmallIndex()
{
    if (blockFactor_ <= 0 || static_cast<int64_t>(GetBlockIdx()) * blockFactor_ >= totalNum_) {
        return;
    }

    if (indexNum_ == 1) {
        ProcessPreparedSingleIndexValue(static_cast<int64_t>(inputGMI.GetValue(0)));
        return;
    }

    LocalTensor<int32_t> indexLocal = indexBuffer.Get<int32_t>();
    DataCopyExtParams indexCopyParams{
        1,
        static_cast<uint32_t>(indexNum_ * static_cast<int64_t>(sizeof(int32_t))),
        0,
        0,
        0};
    DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
    DataCopyPad(indexLocal, inputGMI[0], indexCopyParams, indexPadParams);
    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

    if (EnablePreparedInputSkip()) {
        CopyOwnedInputSkipPrepared(indexLocal, indexNum_);
    } else {
        CopyOwnedInput();
    }
    ApplyUpdatesPrepared(indexLocal, indexNum_);
}

__aicore__ inline void InplaceUpdate::ProcessPreparedSingleIndexValue(int64_t rawIndex)
{
    if (totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 || blockFactor_ <= 0 || typeSize_ <= 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t ownedStart = blockIdx * blockFactor_;
    if (ownedStart >= totalNum_) {
        return;
    }
    const int64_t ownedEnd = MinInt64(totalNum_, ownedStart + blockFactor_);

    int64_t targetRow = rawIndex;
    if (targetRow < 0) {
        targetRow += rowNum_;
    }
    if (targetRow < 0 || targetRow >= rowNum_) {
        CopyBytes<false>(ownedStart * typeSize_, ownedStart * typeSize_, (ownedEnd - ownedStart) * typeSize_);
        return;
    }

    const int64_t rowStart = targetRow * innerNum_;
    const int64_t rowEnd = rowStart + innerNum_;
    const int64_t overlapStart = MaxInt64(ownedStart, rowStart);
    const int64_t overlapEnd = MinInt64(ownedEnd, rowEnd);

    if (!EnablePreparedInputSkip() || overlapStart >= overlapEnd) {
        CopyBytes<false>(ownedStart * typeSize_, ownedStart * typeSize_, (ownedEnd - ownedStart) * typeSize_);
    } else {
        if (ownedStart < overlapStart) {
            CopyBytes<false>(
                ownedStart * typeSize_,
                ownedStart * typeSize_,
                (overlapStart - ownedStart) * typeSize_);
        }
        if (overlapEnd < ownedEnd) {
            CopyBytes<false>(
                overlapEnd * typeSize_,
                overlapEnd * typeSize_,
                (ownedEnd - overlapEnd) * typeSize_);
        }
    }

    if (overlapStart >= overlapEnd) {
        return;
    }
    const int64_t srcElem = overlapStart - rowStart;
    const int64_t srcBytes = srcElem * typeSize_;
    const int64_t dstBytes = overlapStart * typeSize_;
    const int64_t copyBytes = (overlapEnd - overlapStart) * typeSize_;
    CopyBytes<true>(srcBytes, dstBytes, copyBytes);
}

__aicore__ inline void InplaceUpdate::ProcessSingleIndexScalar()
{
    if (totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 || blockFactor_ <= 0 || typeSize_ <= 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t ownedStart = blockIdx * blockFactor_;
    if (ownedStart >= totalNum_) {
        return;
    }
    const int64_t ownedEnd = MinInt64(totalNum_, ownedStart + blockFactor_);
    const int64_t ownedByteStart = ownedStart * typeSize_;
    const int64_t ownedBytes = (ownedEnd - ownedStart) * typeSize_;
    if (((ownedByteStart | ownedBytes) & (BLOCK_BYTES - 1)) == 0) {
        CopyBytes<false, true>(ownedByteStart, ownedByteStart, ownedBytes);
    } else {
        CopyBytes<false>(ownedByteStart, ownedByteStart, ownedBytes);
    }

    int64_t targetRow = static_cast<int64_t>(inputGMI.GetValue(0));
    if (targetRow < 0) {
        targetRow += rowNum_;
    }
    if (targetRow < 0 || targetRow >= rowNum_) {
        return;
    }

    const int64_t rowStart = targetRow * innerNum_;
    const int64_t overlapStart = MaxInt64(ownedStart, rowStart);
    const int64_t overlapEnd = MinInt64(ownedEnd, rowStart + innerNum_);
    if (overlapStart >= overlapEnd) {
        return;
    }
    const int64_t srcElem = overlapStart - rowStart;
    const int64_t updateSrcBytes = srcElem * typeSize_;
    const int64_t updateDstBytes = overlapStart * typeSize_;
    const int64_t updateBytes = (overlapEnd - overlapStart) * typeSize_;
    if (((updateSrcBytes | updateDstBytes | updateBytes) & (BLOCK_BYTES - 1)) == 0) {
        CopyBytes<true, true>(updateSrcBytes, updateDstBytes, updateBytes);
    } else {
        CopyBytes<true>(updateSrcBytes, updateDstBytes, updateBytes);
    }
}

template <bool ENABLE_ROW_FILTER>
__aicore__ inline void InplaceUpdate::ApplyUpdatesImpl()
{
    if (totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 || indexNum_ <= 0 || blockFactor_ <= 0 || typeSize_ <= 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t ownedStart = blockIdx * blockFactor_;
    if (ownedStart >= totalNum_) {
        return;
    }
    const int64_t ownedEnd = MinInt64(totalNum_, ownedStart + blockFactor_);
    int64_t ownedRowStart = 0;
    int64_t ownedRowEnd = 0;
    if constexpr (ENABLE_ROW_FILTER) {
        ownedRowStart = ownedStart / innerNum_;
        ownedRowEnd = (ownedEnd + innerNum_ - 1) / innerNum_;
    }

    for (int64_t indexBase = 0; indexBase < indexNum_; indexBase += indexUbCount_) {
        const int64_t currentIndexNum = MinInt64(indexUbCount_, indexNum_ - indexBase);
        LocalTensor<int32_t> indexLocal = indexBuffer.Get<int32_t>();
        DataCopyExtParams indexCopyParams{
            1,
            static_cast<uint32_t>(currentIndexNum * static_cast<int64_t>(sizeof(int32_t))),
            0,
            0,
            0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, inputGMI[indexBase], indexCopyParams, indexPadParams);
        SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

        for (int64_t localIdx = 0; localIdx < currentIndexNum; ++localIdx) {
            int64_t targetRow = static_cast<int64_t>(indexLocal.GetValue(localIdx));
            if (targetRow < 0) {
                targetRow += rowNum_;
            }
            if (targetRow < 0 || targetRow >= rowNum_) {
                continue;
            }
            if constexpr (ENABLE_ROW_FILTER) {
                if (targetRow < ownedRowStart || targetRow >= ownedRowEnd) {
                    continue;
                }
            }
            if (rowNum_ == 1 && indexNum_ <= SMALL_INDEX_SKIP_LIMIT && indexNum_ <= indexUbCount_ &&
                HasLaterUpdateForRow(targetRow, indexLocal, localIdx + 1, currentIndexNum)) {
                continue;
            }

            const int64_t rowStart = targetRow * innerNum_;
            const int64_t rowEnd = rowStart + innerNum_;
            const int64_t overlapStart = MaxInt64(ownedStart, rowStart);
            const int64_t overlapEnd = MinInt64(ownedEnd, rowEnd);
            if (overlapStart >= overlapEnd) {
                continue;
            }

            const int64_t srcElem =
                (indexBase + localIdx) * innerNum_ + (overlapStart - rowStart);
            CopyBytes<true>(
                srcElem * typeSize_,
                overlapStart * typeSize_,
                (overlapEnd - overlapStart) * typeSize_);
        }
    }
}

__aicore__ inline void InplaceUpdate::ApplyUpdates()
{
    ApplyUpdatesImpl<false>();
}

__aicore__ inline void InplaceUpdate::ApplyUpdatesWithRowFilter()
{
    ApplyUpdatesImpl<true>();
}

__aicore__ inline void InplaceUpdate::ApplyUpdatesTinySingleCore()
{
    if (totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 || indexNum_ <= 0 || typeSize_ <= 0) {
        return;
    }

    for (int64_t indexBase = 0; indexBase < indexNum_; indexBase += indexUbCount_) {
        const int64_t currentIndexNum = MinInt64(indexUbCount_, indexNum_ - indexBase);
        LocalTensor<int32_t> indexLocal = indexBuffer.Get<int32_t>();
        DataCopyExtParams indexCopyParams{
            1,
            static_cast<uint32_t>(currentIndexNum * static_cast<int64_t>(sizeof(int32_t))),
            0,
            0,
            0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, inputGMI[indexBase], indexCopyParams, indexPadParams);
        SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

        for (int64_t localIdx = 0; localIdx < currentIndexNum; ++localIdx) {
            int64_t targetRow = static_cast<int64_t>(indexLocal.GetValue(localIdx));
            if (targetRow < 0) {
                targetRow += rowNum_;
            }
            if (targetRow < 0 || targetRow >= rowNum_) {
                continue;
            }
            if (rowNum_ == 1 && indexNum_ <= SMALL_INDEX_SKIP_LIMIT && indexNum_ <= indexUbCount_ &&
                HasLaterUpdateForRow(targetRow, indexLocal, localIdx + 1, currentIndexNum)) {
                continue;
            }

            const int64_t srcElem = (indexBase + localIdx) * innerNum_;
            const int64_t dstElem = targetRow * innerNum_;
            CopyBytes<true>(srcElem * typeSize_, dstElem * typeSize_, innerNum_ * typeSize_);
        }
    }
}

__aicore__ inline void InplaceUpdate::Process()
{
    if (EnablePreparedSmallIndex()) {
        ProcessPreparedSmallIndex();
        return;
    }
    CopyOwnedInput();
    ApplyUpdates();
}

__aicore__ inline void InplaceUpdate::ProcessWithRowFilter()
{
    CopyOwnedInput();
    ApplyUpdatesWithRowFilter();
}

__aicore__ inline void InplaceUpdate::ProcessTinySingleCore()
{
    if (static_cast<int64_t>(GetBlockIdx()) != 0 || totalNum_ <= 0 || typeSize_ <= 0) {
        return;
    }
    CopyBytes<false>(0, 0, totalNum_ * typeSize_);
    ApplyUpdatesTinySingleCore();
}

__aicore__ inline void InplaceUpdate::ProcessSmallDirect()
{
    if (static_cast<int64_t>(GetBlockIdx()) != 0 || totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 ||
        typeSize_ <= 0) {
        return;
    }

    const int64_t totalBytes = totalNum_ * typeSize_;
    const int64_t innerBytes = innerNum_ * typeSize_;
    CopyTinyBytes<false>(0, 0, totalBytes);
    if (indexNum_ == 1) {
        ApplySmallDirectUpdate(0, innerBytes);
        return;
    }
    if (indexNum_ == 2) {
        ApplySmallDirectUpdate(0, innerBytes);
        ApplySmallDirectUpdate(1, innerBytes);
        return;
    }
    if (indexNum_ == 3) {
        ApplySmallDirectUpdate(0, innerBytes);
        ApplySmallDirectUpdate(1, innerBytes);
        ApplySmallDirectUpdate(2, innerBytes);
        return;
    }
    for (int64_t index = 0; index < indexNum_; ++index) {
        ApplySmallDirectUpdate(index, innerBytes);
    }
}

__aicore__ inline void InplaceUpdate::ProcessSmallStatic()
{
    if (static_cast<int64_t>(GetBlockIdx()) != 0 || totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 ||
        typeSize_ <= 0) {
        return;
    }

    const int64_t totalBytes = totalNum_ * typeSize_;
    const int64_t innerBytes = innerNum_ * typeSize_;
    CopyTinyStaticBytes<false>(0, 0, totalBytes);
    if (indexNum_ == 1) {
        ApplySmallStaticUpdate(0, innerBytes);
        return;
    }
    if (indexNum_ == 2) {
        ApplySmallStaticUpdate(0, innerBytes);
        ApplySmallStaticUpdate(1, innerBytes);
        return;
    }
    if (indexNum_ == 3) {
        ApplySmallStaticUpdate(0, innerBytes);
        ApplySmallStaticUpdate(1, innerBytes);
        ApplySmallStaticUpdate(2, innerBytes);
        return;
    }
    for (int64_t index = 0; index < indexNum_; ++index) {
        ApplySmallStaticUpdate(index, innerBytes);
    }
}

template <bool ROW_ALIGNED>
__aicore__ inline void InplaceUpdate::ProcessLargeIndexRowAligned()
{
    if (totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 || indexNum_ <= 0 || blockFactor_ <= 0 ||
        typeSize_ <= 0 || blockFactor_ % innerNum_ != 0) {
        ProcessWithRowFilter();
        return;
    }

    const int64_t rowsPerCore = blockFactor_ / innerNum_;
    if (rowsPerCore <= 0) {
        ProcessWithRowFilter();
        return;
    }
    if (rowsPerCore > 65535) {
        ProcessWithRowFilter();
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t ownedRowStart = blockIdx * rowsPerCore;
    if (ownedRowStart >= rowNum_) {
        return;
    }
    const int64_t ownedRowEnd = MinInt64(rowNum_, ownedRowStart + rowsPerCore);
    const int64_t ownedStart = ownedRowStart * innerNum_;
    const int64_t ownedNum = (ownedRowEnd - ownedRowStart) * innerNum_;

    const bool useInt32RowFastPath = rowNum_ <= MAX_INT32_ROW && ownedRowEnd <= MAX_INT32_ROW;
    const int32_t ownedRowStart32 = static_cast<int32_t>(ownedRowStart);
    const int32_t ownedRowEnd32 = static_cast<int32_t>(ownedRowEnd);

    const int64_t firstIndexNum = MinInt64(indexUbCount_, indexNum_);
    LocalTensor<int32_t> prefetchedIndexLocal = indexBuffer.Get<int32_t>();
    DataCopyExtParams firstIndexCopyParams{
        1,
        static_cast<uint32_t>(firstIndexNum * static_cast<int64_t>(sizeof(int32_t))),
        0,
        0,
        0};
    DataCopyPadExtParams<int32_t> firstIndexPadParams{false, 0, 0, 0};
    DataCopyPad(prefetchedIndexLocal, inputGMI[0], firstIndexCopyParams, firstIndexPadParams);
    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

    if constexpr (ROW_ALIGNED) {
        CopyBytes<false, true>(ownedStart * typeSize_, ownedStart * typeSize_, ownedNum * typeSize_);
    } else {
        CopyBytes<false>(ownedStart * typeSize_, ownedStart * typeSize_, ownedNum * typeSize_);
    }

    for (int64_t indexBase = 0; indexBase < indexNum_; indexBase += indexUbCount_) {
        const int64_t currentIndexNum = MinInt64(indexUbCount_, indexNum_ - indexBase);
        LocalTensor<int32_t> indexLocal = indexBuffer.Get<int32_t>();
        if (indexBase == 0) {
            WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
        } else {
            DataCopyExtParams indexCopyParams{
                1,
                static_cast<uint32_t>(currentIndexNum * static_cast<int64_t>(sizeof(int32_t))),
                0,
                0,
                0};
            DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
            DataCopyPad(indexLocal, inputGMI[indexBase], indexCopyParams, indexPadParams);
            SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
            WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
        }

        if (useInt32RowFastPath) {
            const int64_t rowBytes = innerNum_ * typeSize_;
            const int64_t slotBytes = ROW_ALIGNED ? rowBytes :
                ((rowBytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
            const int32_t batchCapacity = static_cast<int32_t>(
                MinInt64(MAX_LARGE_UPDATE_BATCH, dataUbBytes_ / slotBytes));
            uint16_t targetOffsets[MAX_LARGE_UPDATE_BATCH];
            LocalTensor<uint8_t> updateLocal = dataBuffer.Get<uint8_t>();
            DataCopyExtParams updateCopyInParams{1, static_cast<uint32_t>(rowBytes), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> updatePadParams{false, 0, 0, 0};
            __ubuf__ int32_t* indexAddress = reinterpret_cast<__ubuf__ int32_t*>(indexLocal.GetPhyAddr());
            int32_t batchCount = 0;
            int32_t slotIdx = 0;
            const int64_t batchBytes = static_cast<int64_t>(batchCapacity) * slotBytes;
            bool slotHasPendingMte3[2] = {false, false};
            int64_t localIdx = 0;
            for (; localIdx + 7 < currentIndexNum; localIdx += 8) {
                int32_t rows[8];
#pragma unroll
                for (int32_t lane = 0; lane < 8; ++lane) {
                    rows[lane] = indexAddress[localIdx + lane];
                }
#pragma unroll
                for (int32_t lane = 0; lane < 8; ++lane) {
                    int32_t targetRow = rows[lane];
                    targetRow += (targetRow >> 31) & static_cast<int32_t>(rowNum_);
                    const uint32_t ownedOffset =
                        static_cast<uint32_t>(targetRow) - static_cast<uint32_t>(ownedRowStart32);
                    if (ownedOffset < static_cast<uint32_t>(ownedRowEnd32 - ownedRowStart32)) {
                        if (batchCount == batchCapacity) {
                            FlushLargeIndexOutputBatch<ROW_ALIGNED>(
                                targetOffsets, batchCount, ownedRowStart, slotBytes, rowBytes,
                                false, slotIdx, batchBytes);
                            slotHasPendingMte3[slotIdx] = true;
                            slotIdx = 1 - slotIdx;
                            if (slotHasPendingMte3[slotIdx]) {
                                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slotIdx));
                                slotHasPendingMte3[slotIdx] = false;
                            }
                            batchCount = 0;
                        }
                        if constexpr (ROW_ALIGNED) {
                            DataCopy(updateLocal[static_cast<int64_t>(slotIdx) * batchBytes +
                                    static_cast<int64_t>(batchCount) * slotBytes],
                                inputGMV[(indexBase + localIdx + lane) * rowBytes], static_cast<uint32_t>(rowBytes));
                        } else {
                            DataCopyPad(updateLocal[static_cast<int64_t>(slotIdx) * batchBytes +
                                    static_cast<int64_t>(batchCount) * slotBytes],
                                inputGMV[(indexBase + localIdx + lane) * rowBytes],
                                updateCopyInParams, updatePadParams);
                        }
                        targetOffsets[batchCount] = static_cast<uint16_t>(ownedOffset);
                        ++batchCount;
                    }
                }
            }
            for (; localIdx < currentIndexNum; ++localIdx) {
                int32_t targetRow = indexAddress[localIdx];
                targetRow += (targetRow >> 31) & static_cast<int32_t>(rowNum_);
                const uint32_t ownedOffset =
                    static_cast<uint32_t>(targetRow) - static_cast<uint32_t>(ownedRowStart32);
                if (ownedOffset < static_cast<uint32_t>(ownedRowEnd32 - ownedRowStart32)) {
                    if (batchCount == batchCapacity) {
                        FlushLargeIndexOutputBatch<ROW_ALIGNED>(
                            targetOffsets, batchCount, ownedRowStart, slotBytes, rowBytes,
                            false, slotIdx, batchBytes);
                        slotHasPendingMte3[slotIdx] = true;
                        slotIdx = 1 - slotIdx;
                        if (slotHasPendingMte3[slotIdx]) {
                            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slotIdx));
                            slotHasPendingMte3[slotIdx] = false;
                        }
                        batchCount = 0;
                    }
                    if constexpr (ROW_ALIGNED) {
                        DataCopy(updateLocal[static_cast<int64_t>(slotIdx) * batchBytes +
                                static_cast<int64_t>(batchCount) * slotBytes],
                            inputGMV[(indexBase + localIdx) * rowBytes], static_cast<uint32_t>(rowBytes));
                    } else {
                        DataCopyPad(updateLocal[static_cast<int64_t>(slotIdx) * batchBytes +
                                static_cast<int64_t>(batchCount) * slotBytes],
                            inputGMV[(indexBase + localIdx) * rowBytes], updateCopyInParams, updatePadParams);
                    }
                    targetOffsets[batchCount] = static_cast<uint16_t>(ownedOffset);
                    ++batchCount;
                }
            }
            if (batchCount > 0) {
                FlushLargeIndexOutputBatch<ROW_ALIGNED>(
                    targetOffsets, batchCount, ownedRowStart, slotBytes, rowBytes,
                    true, slotIdx, batchBytes);
            } else if (slotHasPendingMte3[0] || slotHasPendingMte3[1]) {
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(1));
            }
        } else {
            int64_t localIdx = 0;
            for (; localIdx + 3 < currentIndexNum; localIdx += 4) {
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx, localIdx, ownedRowStart, ownedRowEnd);
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx + 1, localIdx + 1, ownedRowStart, ownedRowEnd);
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx + 2, localIdx + 2, ownedRowStart, ownedRowEnd);
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx + 3, localIdx + 3, ownedRowStart, ownedRowEnd);
            }
            for (; localIdx < currentIndexNum; ++localIdx) {
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx, localIdx, ownedRowStart, ownedRowEnd);
            }
        }
    }
    PipeBarrier<PIPE_MTE3>();
}

__aicore__ inline void InplaceUpdate::ProcessLargeIndexRowAlignedStatic2048()
{
    constexpr int64_t ROW_BYTES = 2048;
    if (totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 || indexNum_ <= 0 || blockFactor_ <= 0 ||
        typeSize_ <= 0 || blockFactor_ % innerNum_ != 0 || innerNum_ * typeSize_ != ROW_BYTES ||
        dataUbBytes_ < ROW_BYTES) {
        return;
    }

    const int64_t rowsPerCore = blockFactor_ / innerNum_;
    if (rowsPerCore <= 0 || rowsPerCore > 65535) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t ownedRowStart = blockIdx * rowsPerCore;
    if (ownedRowStart >= rowNum_) {
        return;
    }
    const int64_t ownedRowEnd = MinInt64(rowNum_, ownedRowStart + rowsPerCore);

    const bool useInt32RowFastPath = rowNum_ <= MAX_INT32_ROW && ownedRowEnd <= MAX_INT32_ROW;
    const int32_t ownedRowStart32 = static_cast<int32_t>(ownedRowStart);
    const int32_t ownedRowEnd32 = static_cast<int32_t>(ownedRowEnd);

    const int64_t firstIndexNum = MinInt64(indexUbCount_, indexNum_);
    LocalTensor<int32_t> prefetchedIndexLocal = indexBuffer.Get<int32_t>();
    DataCopyExtParams firstIndexCopyParams{
        1,
        static_cast<uint32_t>(firstIndexNum * static_cast<int64_t>(sizeof(int32_t))),
        0,
        0,
        0};
    DataCopyPadExtParams<int32_t> firstIndexPadParams{false, 0, 0, 0};
    DataCopyPad(prefetchedIndexLocal, inputGMI[0], firstIndexCopyParams, firstIndexPadParams);
    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

    CopyBytes<false, true>(
        ownedRowStart * ROW_BYTES, ownedRowStart * ROW_BYTES, (ownedRowEnd - ownedRowStart) * ROW_BYTES);

    for (int64_t indexBase = 0; indexBase < indexNum_; indexBase += indexUbCount_) {
        const int64_t currentIndexNum = MinInt64(indexUbCount_, indexNum_ - indexBase);
        LocalTensor<int32_t> indexLocal = indexBuffer.Get<int32_t>();
        if (indexBase == 0) {
            WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
        } else {
            DataCopyExtParams indexCopyParams{
                1,
                static_cast<uint32_t>(currentIndexNum * static_cast<int64_t>(sizeof(int32_t))),
                0,
                0,
                0};
            DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
            DataCopyPad(indexLocal, inputGMI[indexBase], indexCopyParams, indexPadParams);
            SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
            WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
        }

        if (useInt32RowFastPath) {
            constexpr int64_t slotBytes = ROW_BYTES;
            const int32_t batchCapacity = static_cast<int32_t>(
                MinInt64(MAX_LARGE_UPDATE_BATCH, dataUbBytes_ / slotBytes));
            uint16_t targetOffsets[MAX_LARGE_UPDATE_BATCH];
            uint16_t sourceOffsets[MAX_LARGE_UPDATE_BATCH];
            int32_t batchCount = 0;
            int32_t slotIdx = 0;
            const int64_t batchBytes = static_cast<int64_t>(batchCapacity) * slotBytes;
            bool slotHasPendingMte3[2] = {false, false};
            __ubuf__ int32_t* indexAddress = reinterpret_cast<__ubuf__ int32_t*>(indexLocal.GetPhyAddr());
            int64_t localIdx = 0;
            for (; localIdx + 7 < currentIndexNum; localIdx += 8) {
                int32_t rows[8];
#pragma unroll
                for (int32_t lane = 0; lane < 8; ++lane) {
                    rows[lane] = indexAddress[localIdx + lane];
                }
#pragma unroll
                for (int32_t lane = 0; lane < 8; ++lane) {
                    int32_t targetRow = rows[lane];
                    targetRow += (targetRow >> 31) & static_cast<int32_t>(rowNum_);
                    const uint32_t ownedOffset =
                        static_cast<uint32_t>(targetRow) - static_cast<uint32_t>(ownedRowStart32);
                    if (ownedOffset < static_cast<uint32_t>(ownedRowEnd32 - ownedRowStart32)) {
                        if (batchCount == batchCapacity) {
                            FlushStatic2048ContiguousInputBatch(
                                targetOffsets, sourceOffsets, batchCount, indexBase,
                                ownedRowStart, false, slotIdx, batchBytes);
                            slotHasPendingMte3[slotIdx] = true;
                            slotIdx = 1 - slotIdx;
                            if (slotHasPendingMte3[slotIdx]) {
                                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slotIdx));
                                slotHasPendingMte3[slotIdx] = false;
                            }
                            batchCount = 0;
                        }
                        targetOffsets[batchCount] = static_cast<uint16_t>(ownedOffset);
                        sourceOffsets[batchCount] = static_cast<uint16_t>(localIdx + lane);
                        ++batchCount;
                    }
                }
            }
            for (; localIdx < currentIndexNum; ++localIdx) {
                int32_t targetRow = indexAddress[localIdx];
                targetRow += (targetRow >> 31) & static_cast<int32_t>(rowNum_);
                const uint32_t ownedOffset =
                    static_cast<uint32_t>(targetRow) - static_cast<uint32_t>(ownedRowStart32);
                if (ownedOffset < static_cast<uint32_t>(ownedRowEnd32 - ownedRowStart32)) {
                    if (batchCount == batchCapacity) {
                        FlushStatic2048ContiguousInputBatch(
                            targetOffsets, sourceOffsets, batchCount, indexBase,
                            ownedRowStart, false, slotIdx, batchBytes);
                        slotHasPendingMte3[slotIdx] = true;
                        slotIdx = 1 - slotIdx;
                        if (slotHasPendingMte3[slotIdx]) {
                            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slotIdx));
                            slotHasPendingMte3[slotIdx] = false;
                        }
                        batchCount = 0;
                    }
                    targetOffsets[batchCount] = static_cast<uint16_t>(ownedOffset);
                    sourceOffsets[batchCount] = static_cast<uint16_t>(localIdx);
                    ++batchCount;
                }
            }
            if (batchCount > 0) {
                FlushStatic2048ContiguousInputBatch(
                    targetOffsets, sourceOffsets, batchCount, indexBase,
                    ownedRowStart, true, slotIdx, batchBytes);
            } else if (slotHasPendingMte3[0] || slotHasPendingMte3[1]) {
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(1));
            }
        } else {
            int64_t localIdx = 0;
            for (; localIdx + 3 < currentIndexNum; localIdx += 4) {
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx, localIdx, ownedRowStart, ownedRowEnd);
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx + 1, localIdx + 1, ownedRowStart, ownedRowEnd);
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx + 2, localIdx + 2, ownedRowStart, ownedRowEnd);
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx + 3, localIdx + 3, ownedRowStart, ownedRowEnd);
            }
            for (; localIdx < currentIndexNum; ++localIdx) {
                ApplyLargeIndexRowAlignedOne(
                    indexLocal, indexBase + localIdx, localIdx, ownedRowStart, ownedRowEnd);
            }
        }
    }
    PipeBarrier<PIPE_MTE3>();
}

__aicore__ inline void InplaceUpdate::ApplyLargeIndexRowAlignedOne(
    LocalTensor<int32_t> indexLocal,
    int64_t globalIdx,
    int64_t localIdx,
    int64_t ownedRowStart,
    int64_t ownedRowEnd)
{
    int64_t targetRow = static_cast<int64_t>(indexLocal.GetValue(localIdx));
    if (targetRow < 0) {
        targetRow += rowNum_;
    }
    if (targetRow < ownedRowStart || targetRow >= ownedRowEnd || targetRow < 0 || targetRow >= rowNum_) {
        return;
    }

    const int64_t srcElem = globalIdx * innerNum_;
    const int64_t dstElem = targetRow * innerNum_;
    const int64_t srcBytes = srcElem * typeSize_;
    const int64_t dstBytes = dstElem * typeSize_;
    const int64_t copyBytes = innerNum_ * typeSize_;
    CopyBytes<true>(srcBytes, dstBytes, copyBytes);
}

__aicore__ inline void InplaceUpdate::ApplyLargeIndexRowAlignedOneInt32(
    LocalTensor<int32_t> indexLocal,
    int64_t globalIdx,
    int64_t localIdx,
    int32_t ownedRowStart,
    int32_t ownedRowEnd)
{
    ApplyLargeIndexRowAlignedRawInt32(
        indexLocal.GetValue(localIdx), globalIdx, ownedRowStart, ownedRowEnd);
}

__aicore__ inline void InplaceUpdate::ApplyLargeIndexRowAlignedRawInt32(
    int32_t targetRow,
    int64_t globalIdx,
    int32_t ownedRowStart,
    int32_t ownedRowEnd)
{
    targetRow += (targetRow >> 31) & static_cast<int32_t>(rowNum_);
    const uint32_t ownedOffset =
        static_cast<uint32_t>(targetRow) - static_cast<uint32_t>(ownedRowStart);
    if (ownedOffset >= static_cast<uint32_t>(ownedRowEnd - ownedRowStart)) {
        return;
    }

    const int64_t srcElem = globalIdx * innerNum_;
    const int64_t dstElem = static_cast<int64_t>(targetRow) * innerNum_;
    const int64_t srcBytes = srcElem * typeSize_;
    const int64_t dstBytes = dstElem * typeSize_;
    const int64_t copyBytes = innerNum_ * typeSize_;
    CopyBytes<true>(srcBytes, dstBytes, copyBytes);
}

template <bool ROW_ALIGNED>
__aicore__ inline void InplaceUpdate::FlushLargeIndexOutputBatch(
    const uint16_t* targetOffsets,
    int32_t batchCount,
    int64_t ownedRowStart,
    int64_t slotBytes,
    int64_t rowBytes,
    bool waitForReuse,
    int32_t slotIdx,
    int64_t batchBytes)
{
    LocalTensor<uint8_t> updateLocal = dataBuffer.Get<uint8_t>();
    const int64_t slotBase = static_cast<int64_t>(slotIdx) * batchBytes;
    SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(slotIdx));
    WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(slotIdx));

    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(rowBytes), 0, 0, 0};
    for (int32_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
        if constexpr (ROW_ALIGNED) {
            DataCopy(outputGMY[(ownedRowStart + static_cast<int64_t>(targetOffsets[batchIdx])) * rowBytes],
                updateLocal[slotBase + static_cast<int64_t>(batchIdx) * slotBytes],
                static_cast<uint32_t>(rowBytes));
        } else {
            DataCopyPad(outputGMY[(ownedRowStart + static_cast<int64_t>(targetOffsets[batchIdx])) * rowBytes],
                updateLocal[slotBase + static_cast<int64_t>(batchIdx) * slotBytes], copyOutParams);
        }
    }
    if (waitForReuse) {
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slotIdx));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slotIdx));
    }
}

__aicore__ inline void InplaceUpdate::FlushStatic2048ContiguousInputBatch(
    const uint16_t* targetOffsets,
    const uint16_t* sourceOffsets,
    int32_t batchCount,
    int64_t indexBase,
    int64_t ownedRowStart,
    bool waitForReuse,
    int32_t slotIdx,
    int64_t batchBytes)
{
    constexpr int64_t ROW_BYTES = 2048;
    LocalTensor<uint8_t> updateLocal = dataBuffer.Get<uint8_t>();
    const int64_t slotBase = static_cast<int64_t>(slotIdx) * batchBytes;
    int32_t runStart = 0;
    while (runStart < batchCount) {
        int32_t runEnd = runStart + 1;
        while (runEnd < batchCount &&
            static_cast<uint16_t>(sourceOffsets[runEnd - 1] + 1) == sourceOffsets[runEnd]) {
            ++runEnd;
        }
        const int64_t runBytes = static_cast<int64_t>(runEnd - runStart) * ROW_BYTES;
        DataCopy(updateLocal[slotBase + static_cast<int64_t>(runStart) * ROW_BYTES],
            inputGMV[(indexBase + static_cast<int64_t>(sourceOffsets[runStart])) * ROW_BYTES],
            static_cast<uint32_t>(runBytes));
        runStart = runEnd;
    }
    SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(slotIdx));
    WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(slotIdx));
    for (int32_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
        DataCopy(outputGMY[(ownedRowStart + static_cast<int64_t>(targetOffsets[batchIdx])) * ROW_BYTES],
            updateLocal[slotBase + static_cast<int64_t>(batchIdx) * ROW_BYTES],
            static_cast<uint32_t>(ROW_BYTES));
    }
    if (waitForReuse) {
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slotIdx));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(slotIdx));
    }
}

__aicore__ inline void InplaceUpdate::ProcessWideRowBalanced()
{
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    const int64_t rowBytes = innerNum_ * typeSize_;
    const int64_t maxOwnedRows = blockNum > 0 ? (rowNum_ + blockNum - 1) / blockNum : rowNum_;
    const bool guaranteedStream = blockNum > 0 && rowBytes >= PINGPONG_COPY_BYTES_MIN &&
        rowBytes <= PINGPONG_COPY_BYTES_LIMIT && maxOwnedRows * rowBytes <= PINGPONG_COPY_BYTES_LIMIT &&
        (rowBytes & (BLOCK_BYTES - 1)) == 0;
    if (guaranteedStream) {
        ProcessWideRowBalancedImpl<true>();
    } else {
        ProcessWideRowBalancedImpl<false>();
    }
}

__aicore__ inline void InplaceUpdate::ProcessWideRowBalancedIndex4()
{
    if (totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 || typeSize_ <= 0 || indexNum_ != 4) {
        ProcessWideRowBalanced();
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockNum <= 0 || blockIdx >= blockNum) {
        return;
    }
    const int64_t baseRows = rowNum_ / blockNum;
    const int64_t extraRows = rowNum_ - baseRows * blockNum;
    const int64_t ownedRowStart = blockIdx < extraRows ?
        blockIdx * (baseRows + 1) :
        extraRows * (baseRows + 1) + (blockIdx - extraRows) * baseRows;
    const int64_t ownedRows = baseRows + (blockIdx < extraRows ? 1 : 0);
    if (ownedRows <= 0 || ownedRowStart >= rowNum_) {
        return;
    }
    const int64_t ownedRowEnd = MinInt64(rowNum_, ownedRowStart + ownedRows);
    ProcessWidePrepared<true, 4>(ownedRowStart, ownedRowEnd);
}

template <bool GUARANTEED_STREAM>
__aicore__ inline void InplaceUpdate::ProcessWideRowBalancedImpl()
{
    if (totalNum_ <= 0 || rowNum_ <= 0 || innerNum_ <= 0 || typeSize_ <= 0) {
        return;
    }

    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t blockNum = static_cast<int64_t>(GetBlockNum());
    if (blockNum <= 0 || blockIdx >= blockNum) {
        return;
    }
    const int64_t baseRows = rowNum_ / blockNum;
    const int64_t extraRows = rowNum_ - baseRows * blockNum;
    const int64_t ownedRowStart = blockIdx < extraRows ?
        blockIdx * (baseRows + 1) :
        extraRows * (baseRows + 1) + (blockIdx - extraRows) * baseRows;
    const int64_t ownedRows = baseRows + (blockIdx < extraRows ? 1 : 0);
    if (ownedRows <= 0 || ownedRowStart >= rowNum_) {
        return;
    }
    const int64_t ownedRowEnd = MinInt64(rowNum_, ownedRowStart + ownedRows);

    if (EnablePreparedSmallIndex()) {
        if (indexNum_ == 1) {
            ProcessWidePrepared<GUARANTEED_STREAM, 1>(ownedRowStart, ownedRowEnd);
        } else if (indexNum_ == 2) {
            ProcessWidePrepared<GUARANTEED_STREAM, 2>(ownedRowStart, ownedRowEnd);
        } else if (indexNum_ == 3) {
            ProcessWidePrepared<GUARANTEED_STREAM, 3>(ownedRowStart, ownedRowEnd);
        } else if (indexNum_ == 4) {
            ProcessWidePrepared<GUARANTEED_STREAM, 4>(ownedRowStart, ownedRowEnd);
        } else {
            ProcessWidePrepared<GUARANTEED_STREAM, 0>(ownedRowStart, ownedRowEnd);
        }
        return;
    }

    const int64_t ownedStart = ownedRowStart * innerNum_;
    const int64_t ownedNum = (ownedRowEnd - ownedRowStart) * innerNum_;
    CopyBytes<false>(ownedStart * typeSize_, ownedStart * typeSize_, ownedNum * typeSize_);

    for (int64_t indexBase = 0; indexBase < indexNum_; indexBase += indexUbCount_) {
        const int64_t currentIndexNum = MinInt64(indexUbCount_, indexNum_ - indexBase);
        LocalTensor<int32_t> indexLocal = indexBuffer.Get<int32_t>();
        DataCopyExtParams indexCopyParams{
            1,
            static_cast<uint32_t>(currentIndexNum * static_cast<int64_t>(sizeof(int32_t))),
            0,
            0,
            0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, inputGMI[indexBase], indexCopyParams, indexPadParams);
        SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

        for (int64_t localIdx = 0; localIdx < currentIndexNum; ++localIdx) {
            int64_t targetRow = static_cast<int64_t>(indexLocal.GetValue(localIdx));
            if (targetRow < 0) {
                targetRow += rowNum_;
            }
            if (targetRow < ownedRowStart || targetRow >= ownedRowEnd || targetRow < 0 || targetRow >= rowNum_) {
                continue;
            }

            const int64_t srcElem = (indexBase + localIdx) * innerNum_;
            const int64_t dstElem = targetRow * innerNum_;
            CopyBytes<true>(srcElem * typeSize_, dstElem * typeSize_, innerNum_ * typeSize_);
        }
    }
}

template <bool GUARANTEED_STREAM, int32_t INDEX_COUNT>
__aicore__ inline void InplaceUpdate::ProcessWidePrepared(
    int64_t ownedRowStart, int64_t ownedRowEnd)
{
    const int32_t indexCount = INDEX_COUNT > 0 ? INDEX_COUNT : static_cast<int32_t>(indexNum_);
    LocalTensor<int32_t> indexLocal = indexBuffer.Get<int32_t>();
    DataCopyExtParams indexCopyParams{
        1,
        static_cast<uint32_t>(indexCount * static_cast<int32_t>(sizeof(int32_t))),
        0,
        0,
        0};
    DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
    DataCopyPad(indexLocal, inputGMI[0], indexCopyParams, indexPadParams);
    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    __ubuf__ int32_t* indexAddress = reinterpret_cast<__ubuf__ int32_t*>(indexLocal.GetPhyAddr());
    int64_t updateRows[SMALL_INDEX_SKIP_LIMIT] = {};
#pragma unroll
    for (int32_t localIdx = 0; localIdx < indexCount; ++localIdx) {
        int64_t targetRow = static_cast<int64_t>(indexAddress[localIdx]);
        if (targetRow < 0) {
            targetRow += rowNum_;
        }
        updateRows[localIdx] = targetRow;
    }

    int64_t copyStartRow = ownedRowStart;
    for (int64_t rowBase = ownedRowStart; rowBase < ownedRowEnd; rowBase += 8) {
        const int64_t rowBaseEnd = MinInt64(ownedRowEnd, rowBase + 8);
        bool updated[8] = {false, false, false, false, false, false, false, false};
#pragma unroll
        for (int32_t localIdx = 0; localIdx < indexCount; ++localIdx) {
            const int64_t updateRow = updateRows[localIdx];
            if (updateRow >= rowBase && updateRow < rowBaseEnd) {
                updated[static_cast<int32_t>(updateRow - rowBase)] = true;
            }
        }
#pragma unroll
        for (int32_t k = 0; k < 8; ++k) {
            const int64_t row = rowBase + k;
            if (row >= rowBaseEnd) {
                break;
            }
            if (updated[k]) {
                if (copyStartRow < row) {
                    const int64_t copyStart = copyStartRow * innerNum_;
                    const int64_t copyNum = (row - copyStartRow) * innerNum_;
                    CopyWideBytesStream<false, GUARANTEED_STREAM>(
                        copyStart * typeSize_, copyStart * typeSize_, copyNum * typeSize_);
                }
                copyStartRow = row + 1;
            }
        }
    }
    if (copyStartRow < ownedRowEnd) {
        const int64_t copyStart = copyStartRow * innerNum_;
        const int64_t copyNum = (ownedRowEnd - copyStartRow) * innerNum_;
        CopyWideBytesStream<false, GUARANTEED_STREAM>(
            copyStart * typeSize_, copyStart * typeSize_, copyNum * typeSize_);
    }

#pragma unroll
    for (int32_t localIdx = 0; localIdx < indexCount; ++localIdx) {
        const int64_t targetRow = updateRows[localIdx];
        if (targetRow < ownedRowStart || targetRow >= ownedRowEnd || targetRow < 0 || targetRow >= rowNum_) {
            continue;
        }
        bool hasLaterUpdate = false;
#pragma unroll
        for (int32_t laterIdx = localIdx + 1; laterIdx < indexCount; ++laterIdx) {
            if (updateRows[laterIdx] == targetRow) {
                hasLaterUpdate = true;
                break;
            }
        }
        if (!hasLaterUpdate) {
            const int64_t srcElem = static_cast<int64_t>(localIdx) * innerNum_;
            const int64_t dstElem = targetRow * innerNum_;
            CopyWideBytesStream<true, GUARANTEED_STREAM>(
                srcElem * typeSize_, dstElem * typeSize_, innerNum_ * typeSize_);
        }
    }
    PipeBarrier<PIPE_MTE3>();
    FinalizeWideBytesStream();
}

__aicore__ inline int32_t NormalizeTinyTargetRow(int32_t rawIndex, int32_t rowNum)
{
    int32_t targetRow = rawIndex;
    if (targetRow < 0) {
        targetRow += rowNum;
    }
    if (static_cast<uint32_t>(targetRow) >= static_cast<uint32_t>(rowNum)) {
        return -1;
    }
    return targetRow;
}

template <bool FROM_V>
__aicore__ inline void CopyTinyStaticFastBytes(
    GlobalTensor<uint8_t>& inputGMX,
    GlobalTensor<uint8_t>& inputGMV,
    GlobalTensor<uint8_t>& outputGMY,
    int64_t srcOffset,
    int64_t dstOffset,
    int64_t byteCount)
{
    if (byteCount <= 0 || byteCount > static_cast<int64_t>(TINY_STATIC_BYTES)) {
        return;
    }
    LocalTensor<uint8_t> local(TPosition::VECCALC, 0, TINY_STATIC_BYTES);
    const bool alignedCopy =
        (srcOffset % BLOCK_BYTES == 0) && (dstOffset % BLOCK_BYTES == 0) && (byteCount % BLOCK_BYTES == 0);
    if (alignedCopy) {
        if constexpr (FROM_V) {
            DataCopy(local, inputGMV[srcOffset], static_cast<uint32_t>(byteCount));
        } else {
            DataCopy(local, inputGMX[srcOffset], static_cast<uint32_t>(byteCount));
        }
    } else {
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(byteCount), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        if constexpr (FROM_V) {
            DataCopyPad(local, inputGMV[srcOffset], copyInParams, padParams);
        } else {
            DataCopyPad(local, inputGMX[srcOffset], copyInParams, padParams);
        }
    }
    SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
    if (alignedCopy) {
        DataCopy(outputGMY[dstOffset], local, static_cast<uint32_t>(byteCount));
    } else {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(byteCount), 0, 0, 0};
        DataCopyPad(outputGMY[dstOffset], local, copyOutParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
}

template <bool FROM_V>
__aicore__ inline void CopyPackedSmallStaticBytes(
    GlobalTensor<uint8_t>& inputGMX,
    GlobalTensor<uint8_t>& inputGMV,
    GlobalTensor<uint8_t>& outputGMY,
    int64_t srcOffset,
    int64_t dstOffset,
    int64_t byteCount)
{
    if (byteCount <= 0 || byteCount > static_cast<int64_t>(PACKED_STATIC_BYTES)) {
        return;
    }
    LocalTensor<uint8_t> local(TPosition::VECCALC, 0, PACKED_STATIC_BYTES);
    const bool alignedCopy =
        (srcOffset % BLOCK_BYTES == 0) && (dstOffset % BLOCK_BYTES == 0) && (byteCount % BLOCK_BYTES == 0);
    if (alignedCopy) {
        if constexpr (FROM_V) {
            DataCopy(local, inputGMV[srcOffset], static_cast<uint32_t>(byteCount));
        } else {
            DataCopy(local, inputGMX[srcOffset], static_cast<uint32_t>(byteCount));
        }
    } else {
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(byteCount), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        if constexpr (FROM_V) {
            DataCopyPad(local, inputGMV[srcOffset], copyInParams, padParams);
        } else {
            DataCopyPad(local, inputGMX[srcOffset], copyInParams, padParams);
        }
    }
    SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
    if (alignedCopy) {
        DataCopy(outputGMY[dstOffset], local, static_cast<uint32_t>(byteCount));
    } else {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(byteCount), 0, 0, 0};
        DataCopyPad(outputGMY[dstOffset], local, copyOutParams);
    }
    SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
}

__aicore__ inline void ApplyPackedSmallStaticUpdate(
    GlobalTensor<uint8_t>& inputGMX,
    GlobalTensor<uint8_t>& inputGMV,
    GlobalTensor<uint8_t>& outputGMY,
    LocalTensor<int32_t> indexLocal,
    int64_t rowNum,
    int64_t innerBytes,
    int64_t index)
{
    const int64_t targetRow = NormalizeTinyTargetRow(static_cast<int64_t>(indexLocal.GetValue(index)), rowNum);
    if (targetRow < 0) {
        return;
    }
    CopyPackedSmallStaticBytes<true>(
        inputGMX, inputGMV, outputGMY, index * innerBytes, targetRow * innerBytes, innerBytes);
}

__aicore__ inline bool TryProcessPackedSmallStaticFastPath(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    if (tilingData == nullptr) {
        return false;
    }

    int64_t totalNum = tilingData->totalNum;
    int64_t rowNum = tilingData->rowNum;
    int64_t innerNum = tilingData->innerNum;
    const int64_t indexNum = tilingData->indexNum;
    const int64_t typeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 4;
    if (rowNum <= 0 && totalNum > 0) {
        rowNum = 1;
    }
    if (innerNum <= 0 && totalNum > 0) {
        innerNum = totalNum;
    }
    if (totalNum <= 0 || rowNum <= 1 || innerNum <= 0 || typeSize <= 0 || indexNum <= 1 ||
        indexNum > SMALL_INDEX_SKIP_LIMIT) {
        return false;
    }

    const int64_t totalBytes = totalNum * typeSize;
    const int64_t innerBytes = innerNum * typeSize;
    if (totalBytes <= static_cast<int64_t>(TINY_STATIC_BYTES) ||
        totalBytes > static_cast<int64_t>(PACKED_STATIC_BYTES) ||
        innerBytes > static_cast<int64_t>(TINY_STATIC_BYTES)) {
        return false;
    }
    if (static_cast<int64_t>(GetBlockIdx()) != 0) {
        return true;
    }

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    LocalTensor<int32_t> indexLocal(TPosition::VECCALC, PACKED_STATIC_BYTES, 32);
    DataCopyExtParams indexCopyParams{
        1,
        static_cast<uint32_t>(indexNum * static_cast<int64_t>(sizeof(int32_t))),
        0,
        0,
        0};
    DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
    DataCopyPad(indexLocal, inputGMI[0], indexCopyParams, indexPadParams);
    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

    CopyPackedSmallStaticBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);
    WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    if (innerBytes % BLOCK_BYTES != 0) {
        for (int64_t index = 0; index < indexNum; ++index) {
            ApplyPackedSmallStaticUpdate(inputGMX, inputGMV, outputGMY, indexLocal, rowNum, innerBytes, index);
        }
        return true;
    }
    if (indexNum == 2 || indexNum == 3) {
        int32_t targetRows[3] = {-1, -1, -1};
        for (int32_t index = 0; index < static_cast<int32_t>(indexNum); ++index) {
            targetRows[index] = NormalizeTinyTargetRow(
                indexLocal.GetValue(index), static_cast<int32_t>(rowNum));
        }
        LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, 0, PACKED_STATIC_BYTES);
        const int64_t updateSlotBytes = ((innerBytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
        DataCopyExtParams updateCopyInParams{
            static_cast<uint16_t>(indexNum), static_cast<uint32_t>(innerBytes), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> updatePadParams{
            false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
        DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));

        DataCopyExtParams updateCopyOutParams{1, static_cast<uint32_t>(innerBytes), 0, 0, 0};
        bool hasOutput = false;
        for (int32_t index = 0; index < static_cast<int32_t>(indexNum); ++index) {
            if (targetRows[index] >= 0) {
                DataCopyPad(outputGMY[static_cast<int64_t>(targetRows[index]) * innerBytes],
                    updateLocal[static_cast<int64_t>(index) * updateSlotBytes], updateCopyOutParams);
                hasOutput = true;
            }
        }
        if (hasOutput) {
            SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
        }
        return true;
    }
    for (int64_t index = 0; index < indexNum; ++index) {
        ApplyPackedSmallStaticUpdate(inputGMX, inputGMV, outputGMY, indexLocal, rowNum, innerBytes, index);
    }
    return true;
}

__aicore__ inline void ApplyTinyStaticFastUpdate(
    GlobalTensor<uint8_t>& inputGMX,
    GlobalTensor<uint8_t>& inputGMV,
    GlobalTensor<uint8_t>& outputGMY,
    LocalTensor<int32_t> indexLocal,
    int64_t rowNum,
    int64_t innerBytes,
    int64_t index)
{
    const int64_t targetRow = NormalizeTinyTargetRow(static_cast<int64_t>(indexLocal.GetValue(index)), rowNum);
    if (targetRow < 0) {
        return;
    }
    CopyTinyStaticFastBytes<true>(
        inputGMX, inputGMV, outputGMY, index * innerBytes, targetRow * innerBytes, innerBytes);
}

__aicore__ inline void ApplyTinyStaticFastUpdateFromGM(
    GlobalTensor<int32_t>& inputGMI,
    GlobalTensor<uint8_t>& inputGMX,
    GlobalTensor<uint8_t>& inputGMV,
    GlobalTensor<uint8_t>& outputGMY,
    int64_t rowNum,
    int64_t innerBytes,
    int64_t index)
{
    const int64_t targetRow = NormalizeTinyTargetRow(static_cast<int64_t>(inputGMI.GetValue(index)), rowNum);
    if (targetRow < 0) {
        return;
    }
    CopyTinyStaticFastBytes<true>(
        inputGMX, inputGMV, outputGMY, index * innerBytes, targetRow * innerBytes, innerBytes);
}

__aicore__ inline void ApplyTinyIndex1UbComposeRow(
    __ubuf__ uint32_t* xAddress,
    __ubuf__ uint32_t* updateAddress,
    int32_t targetRow,
    int64_t rowWords)
{
    if (targetRow < 0) {
        return;
    }
    const int64_t dstBase = static_cast<int64_t>(targetRow) * rowWords;
    xAddress[dstBase] = updateAddress[0];
    if (rowWords > 1) {
        xAddress[dstBase + 1] = updateAddress[1];
    }
    if (rowWords > 2) {
        xAddress[dstBase + 2] = updateAddress[2];
    }
    if (rowWords > 3) {
        xAddress[dstBase + 3] = updateAddress[3];
    }
}

__aicore__ inline void ProcessTinyStaticIndex1UbComposeFastPath(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    if (tilingData == nullptr || static_cast<int64_t>(GetBlockIdx()) != 0) {
        return;
    }

    const int64_t totalNum = tilingData->totalNum;
    const int64_t rowNum = tilingData->rowNum;
    const int64_t innerNum = tilingData->innerNum;
    const int64_t typeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 4;
    const int64_t totalBytes = totalNum * typeSize;
    const int64_t innerBytes = innerNum * typeSize;
    const int64_t rowWords = innerBytes / static_cast<int64_t>(sizeof(uint32_t));
    if (tilingData->indexNum != 1 || rowNum <= 1 || totalBytes <= 0 ||
        totalBytes > static_cast<int64_t>(TINY_STATIC_BYTES) || typeSize != 4 ||
        innerBytes < 4 || innerBytes > 16 || (innerBytes & 3) != 0) {
        return;
    }

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    LocalTensor<uint8_t> xLocal(TPosition::VECCALC, 0, TINY_STATIC_BYTES);
    LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, TINY_STATIC_BYTES, TINY_STATIC_BYTES);
    DataCopyExtParams xCopyInParams{1, static_cast<uint32_t>(totalBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[0], xCopyInParams, xPadParams);
    DataCopyExtParams updateCopyInParams{1, static_cast<uint32_t>(innerBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> updatePadParams{false, 0, 0, 0};
    DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);

    const int32_t targetRow = NormalizeTinyTargetRow(inputGMI.GetValue(0), static_cast<int32_t>(rowNum));
    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    __ubuf__ uint32_t* xAddress = reinterpret_cast<__ubuf__ uint32_t*>(xLocal.GetPhyAddr());
    __ubuf__ uint32_t* updateAddress = reinterpret_cast<__ubuf__ uint32_t*>(updateLocal.GetPhyAddr());
    ApplyTinyIndex1UbComposeRow(xAddress, updateAddress, targetRow, rowWords);

    SetFlag<HardEvent::S_MTE3>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::S_MTE3>(static_cast<TEventID>(0));
    if ((totalBytes & (BLOCK_BYTES - 1)) == 0) {
        DataCopy(outputGMY[0], xLocal, static_cast<uint32_t>(totalBytes));
    } else {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(totalBytes), 0, 0, 0};
        DataCopyPad(outputGMY[0], xLocal, copyOutParams);
    }
}

__aicore__ inline void ProcessTinyStaticIndex1UbCompose8BFastPath(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    if (tilingData == nullptr || static_cast<int64_t>(GetBlockIdx()) != 0) {
        return;
    }

    const int64_t totalNum = tilingData->totalNum;
    const int64_t rowNum = tilingData->rowNum;
    const int64_t innerNum = tilingData->innerNum;
    const int64_t typeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 4;
    const int64_t totalBytes = totalNum * typeSize;
    if (tilingData->indexNum != 1 || rowNum <= 1 || totalBytes <= 0 ||
        totalBytes > static_cast<int64_t>(TINY_STATIC_BYTES) || typeSize != 4 || innerNum != 2) {
        return;
    }

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    LocalTensor<uint8_t> xLocal(TPosition::VECCALC, 0, TINY_STATIC_BYTES);
    LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, TINY_STATIC_BYTES, TINY_STATIC_BYTES);
    DataCopyExtParams xCopyInParams{1, static_cast<uint32_t>(totalBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[0], xCopyInParams, xPadParams);
    DataCopyExtParams updateCopyInParams{1, 8, 0, 0, 0};
    DataCopyPadExtParams<uint8_t> updatePadParams{false, 0, 0, 0};
    DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);

    const int32_t targetRow = NormalizeTinyTargetRow(inputGMI.GetValue(0), static_cast<int32_t>(rowNum));
    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    if (targetRow >= 0) {
        __ubuf__ uint32_t* xAddress = reinterpret_cast<__ubuf__ uint32_t*>(xLocal.GetPhyAddr());
        __ubuf__ uint32_t* updateAddress = reinterpret_cast<__ubuf__ uint32_t*>(updateLocal.GetPhyAddr());
        const int64_t dstBase = static_cast<int64_t>(targetRow) * 2;
        xAddress[dstBase] = updateAddress[0];
        xAddress[dstBase + 1] = updateAddress[1];
    }

    SetFlag<HardEvent::S_MTE3>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::S_MTE3>(static_cast<TEventID>(0));
    if ((totalBytes & (BLOCK_BYTES - 1)) == 0) {
        DataCopy(outputGMY[0], xLocal, static_cast<uint32_t>(totalBytes));
    } else {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(totalBytes), 0, 0, 0};
        DataCopyPad(outputGMY[0], xLocal, copyOutParams);
    }
}

__aicore__ inline void ProcessSmallStaticFastPath(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    if (tilingData == nullptr) {
        return;
    }
    const int64_t dispatchTypeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 4;
    const int64_t dispatchTotalBytes = tilingData->totalNum * dispatchTypeSize;
    if (tilingData->rowNum > 1 && dispatchTotalBytes > static_cast<int64_t>(TINY_STATIC_BYTES) &&
        dispatchTotalBytes <= static_cast<int64_t>(PACKED_STATIC_BYTES) &&
        TryProcessPackedSmallStaticFastPath(x, i, v, y, tilingData)) {
        return;
    }
    if (static_cast<int64_t>(GetBlockIdx()) != 0) {
        return;
    }

    int64_t totalNum = tilingData->totalNum;
    int64_t rowNum = tilingData->rowNum;
    int64_t innerNum = tilingData->innerNum;
    const int64_t indexNum = tilingData->indexNum;
    const int64_t typeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 4;
    if (rowNum <= 0 && totalNum > 0) {
        rowNum = 1;
    }
    if (innerNum <= 0 && totalNum > 0) {
        innerNum = totalNum;
    }
    if (totalNum <= 0 || rowNum <= 0 || innerNum <= 0 || typeSize <= 0) {
        return;
    }

    const int64_t totalBytes = totalNum * typeSize;
    const int64_t innerBytes = innerNum * typeSize;
    if (rowNum == 1 && totalBytes > static_cast<int64_t>(TINY_STATIC_BYTES) &&
        totalBytes <= static_cast<int64_t>(PACKED_STATIC_BYTES) && indexNum > 0 &&
        indexNum <= SMALL_INDEX_SKIP_LIMIT) {
        GlobalTensor<uint8_t> inputGMX;
        GlobalTensor<int32_t> inputGMI;
        GlobalTensor<uint8_t> inputGMV;
        GlobalTensor<uint8_t> outputGMY;
        inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
        inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
        inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
        outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

        int64_t lastValidIndex = -1;
        for (int64_t index = 0; index < indexNum; ++index) {
            if (NormalizeTinyTargetRow(inputGMI.GetValue(index), 1) == 0) {
                lastValidIndex = index;
            }
        }
        if (lastValidIndex >= 0) {
            CopyPackedSmallStaticBytes<true>(
                inputGMX, inputGMV, outputGMY, lastValidIndex * innerBytes, 0, totalBytes);
        } else {
            CopyPackedSmallStaticBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);
        }
        return;
    }
    if (totalBytes > static_cast<int64_t>(TINY_STATIC_BYTES) ||
        innerBytes > static_cast<int64_t>(TINY_STATIC_BYTES)) {
        InplaceUpdate op;
        op.InitSmallStatic(x, i, v, y, tilingData);
        op.ProcessSmallStatic();
        return;
    }
    if (indexNum > SMALL_INDEX_SKIP_LIMIT) {
        InplaceUpdate op;
        op.InitSmallStatic(x, i, v, y, tilingData);
        op.ProcessSmallStatic();
        return;
    }

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    if (indexNum == 1) {
        if (rowNum == 1 && NormalizeTinyTargetRow(static_cast<int64_t>(inputGMI.GetValue(0)), rowNum) == 0) {
            CopyTinyStaticFastBytes<true>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);
            return;
        }
        CopyTinyStaticFastBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);
        ApplyTinyStaticFastUpdateFromGM(inputGMI, inputGMX, inputGMV, outputGMY, rowNum, innerBytes, 0);
        return;
    }

    if (indexNum == 2 || indexNum == 3) {
        if (rowNum == 1) {
            int64_t lastValidIndex = -1;
            for (int64_t index = 0; index < indexNum; ++index) {
                if (NormalizeTinyTargetRow(static_cast<int64_t>(inputGMI.GetValue(index)), rowNum) == 0) {
                    lastValidIndex = index;
                }
            }
            if (lastValidIndex >= 0) {
                CopyTinyStaticFastBytes<true>(
                    inputGMX, inputGMV, outputGMY, lastValidIndex * innerBytes, 0, totalBytes);
                return;
            }
        }

        if (indexNum == 2 && typeSize <= 2) {
            const int32_t targetRow0 = NormalizeTinyTargetRow(inputGMI.GetValue(0), static_cast<int32_t>(rowNum));
            const int32_t targetRow1 = NormalizeTinyTargetRow(inputGMI.GetValue(1), static_cast<int32_t>(rowNum));
            LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, TINY_STATIC_BYTES, TINY_STATIC_BYTES);
            const int64_t updateSlotBytes = ((innerBytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
            DataCopyExtParams updateCopyInParams{2, static_cast<uint32_t>(innerBytes), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> updatePadParams{
                false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
            DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);
            SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

            CopyTinyStaticFastBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);
            WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));

            DataCopyExtParams updateCopyOutParams{1, static_cast<uint32_t>(innerBytes), 0, 0, 0};
            bool hasOutput = false;
            if (targetRow0 >= 0) {
                DataCopyPad(outputGMY[static_cast<int64_t>(targetRow0) * innerBytes],
                    updateLocal[0], updateCopyOutParams);
                hasOutput = true;
            }
            if (targetRow1 >= 0) {
                DataCopyPad(outputGMY[static_cast<int64_t>(targetRow1) * innerBytes],
                    updateLocal[updateSlotBytes], updateCopyOutParams);
                hasOutput = true;
            }
            if (hasOutput) {
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
            }
            return;
        }

        CopyTinyStaticFastBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);
        if (indexNum == 2) {
            const int32_t targetRow0 = NormalizeTinyTargetRow(inputGMI.GetValue(0), static_cast<int32_t>(rowNum));
            const int32_t targetRow1 = NormalizeTinyTargetRow(inputGMI.GetValue(1), static_cast<int32_t>(rowNum));
            LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, 0, TINY_STATIC_BYTES);
            const int64_t updateSlotBytes = ((innerBytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
            DataCopyExtParams updateCopyInParams{2, static_cast<uint32_t>(innerBytes), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> updatePadParams{
                false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
            DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);
            SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
            WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));

            DataCopyExtParams updateCopyOutParams{1, static_cast<uint32_t>(innerBytes), 0, 0, 0};
            bool hasOutput = false;
            if (targetRow0 >= 0) {
                DataCopyPad(outputGMY[static_cast<int64_t>(targetRow0) * innerBytes],
                    updateLocal[0], updateCopyOutParams);
                hasOutput = true;
            }
            if (targetRow1 >= 0) {
                DataCopyPad(outputGMY[static_cast<int64_t>(targetRow1) * innerBytes],
                    updateLocal[updateSlotBytes], updateCopyOutParams);
                hasOutput = true;
            }
            if (hasOutput) {
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
            }
            return;
        }
        if (indexNum == 3) {
            const int32_t targetRow0 = NormalizeTinyTargetRow(inputGMI.GetValue(0), static_cast<int32_t>(rowNum));
            const int32_t targetRow1 = NormalizeTinyTargetRow(inputGMI.GetValue(1), static_cast<int32_t>(rowNum));
            const int32_t targetRow2 = NormalizeTinyTargetRow(inputGMI.GetValue(2), static_cast<int32_t>(rowNum));
            LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, 0, TINY_STATIC_BYTES);
            const int64_t updateSlotBytes = ((innerBytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
            DataCopyExtParams updateCopyInParams{3, static_cast<uint32_t>(innerBytes), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> updatePadParams{
                false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
            DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);
            SetFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));
            WaitFlag<HardEvent::MTE2_MTE3>(static_cast<TEventID>(0));

            DataCopyExtParams updateCopyOutParams{1, static_cast<uint32_t>(innerBytes), 0, 0, 0};
            bool hasOutput = false;
            if (targetRow0 >= 0 && targetRow0 != targetRow1 && targetRow0 != targetRow2) {
                DataCopyPad(outputGMY[static_cast<int64_t>(targetRow0) * innerBytes],
                    updateLocal[0], updateCopyOutParams);
                hasOutput = true;
            }
            if (targetRow1 >= 0 && targetRow1 != targetRow2) {
                DataCopyPad(outputGMY[static_cast<int64_t>(targetRow1) * innerBytes],
                    updateLocal[updateSlotBytes], updateCopyOutParams);
                hasOutput = true;
            }
            if (targetRow2 >= 0) {
                DataCopyPad(outputGMY[static_cast<int64_t>(targetRow2) * innerBytes],
                    updateLocal[2 * updateSlotBytes], updateCopyOutParams);
                hasOutput = true;
            }
            if (hasOutput) {
                SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
            }
            return;
        }
        ApplyTinyStaticFastUpdateFromGM(inputGMI, inputGMX, inputGMV, outputGMY, rowNum, innerBytes, 0);
        ApplyTinyStaticFastUpdateFromGM(inputGMI, inputGMX, inputGMV, outputGMY, rowNum, innerBytes, 1);
        if (indexNum == 3) {
            ApplyTinyStaticFastUpdateFromGM(inputGMI, inputGMX, inputGMV, outputGMY, rowNum, innerBytes, 2);
        }
        return;
    }

    LocalTensor<int32_t> indexLocal(TPosition::VECCALC, TINY_STATIC_BYTES, 32);
    const bool useLocalIndex = indexNum > 1;
    if (useLocalIndex) {
        DataCopyExtParams indexCopyParams{
            1,
            static_cast<uint32_t>(indexNum * static_cast<int64_t>(sizeof(int32_t))),
            0,
            0,
            0};
        DataCopyPadExtParams<int32_t> indexPadParams{false, 0, 0, 0};
        DataCopyPad(indexLocal, inputGMI[0], indexCopyParams, indexPadParams);
        SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    }

    if (rowNum == 1 && indexNum > 0) {
        int64_t lastValidIndex = -1;
        if (useLocalIndex) {
            for (int64_t index = 0; index < indexNum; ++index) {
                if (NormalizeTinyTargetRow(static_cast<int64_t>(indexLocal.GetValue(index)), rowNum) == 0) {
                    lastValidIndex = index;
                }
            }
        } else {
            if (NormalizeTinyTargetRow(static_cast<int64_t>(inputGMI.GetValue(0)), rowNum) == 0) {
                lastValidIndex = 0;
            }
        }
        if (lastValidIndex >= 0) {
            CopyTinyStaticFastBytes<true>(
                inputGMX, inputGMV, outputGMY, lastValidIndex * innerBytes, 0, totalBytes);
            return;
        }
    }

    CopyTinyStaticFastBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);
    if (indexNum == 2) {
        ApplyTinyStaticFastUpdate(inputGMX, inputGMV, outputGMY, indexLocal, rowNum, innerBytes, 0);
        ApplyTinyStaticFastUpdate(inputGMX, inputGMV, outputGMY, indexLocal, rowNum, innerBytes, 1);
        return;
    }
    if (indexNum == 3) {
        ApplyTinyStaticFastUpdate(inputGMX, inputGMV, outputGMY, indexLocal, rowNum, innerBytes, 0);
        ApplyTinyStaticFastUpdate(inputGMX, inputGMV, outputGMY, indexLocal, rowNum, innerBytes, 1);
        ApplyTinyStaticFastUpdate(inputGMX, inputGMV, outputGMY, indexLocal, rowNum, innerBytes, 2);
        return;
    }
    for (int64_t index = 0; index < indexNum; ++index) {
        ApplyTinyStaticFastUpdate(inputGMX, inputGMV, outputGMY, indexLocal, rowNum, innerBytes, index);
    }
}

__aicore__ inline void ProcessTinyStaticIndex2NarrowFastPath(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    if (tilingData == nullptr || static_cast<int64_t>(GetBlockIdx()) != 0) {
        return;
    }

    const int64_t totalNum = tilingData->totalNum;
    const int64_t rowNum = tilingData->rowNum;
    const int64_t innerNum = tilingData->innerNum;
    const int64_t typeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 2;
    const int64_t totalBytes = totalNum * typeSize;
    const int64_t innerBytes = innerNum * typeSize;
    if (totalBytes <= 0 || totalBytes > static_cast<int64_t>(TINY_STATIC_BYTES) || rowNum <= 0 ||
        innerBytes <= 0 || innerBytes > static_cast<int64_t>(TINY_STATIC_BYTES) || typeSize > 4) {
        return;
    }

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, TINY_STATIC_BYTES, TINY_STATIC_BYTES);
    const int64_t updateSlotBytes = ((innerBytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
    DataCopyExtParams updateCopyInParams{2, static_cast<uint32_t>(innerBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> updatePadParams{
        false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
    DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);
    const int32_t rowNum32 = static_cast<int32_t>(rowNum);
    const int32_t targetRow0 = NormalizeTinyTargetRow(inputGMI.GetValue(0), rowNum32);
    const int32_t targetRow1 = NormalizeTinyTargetRow(inputGMI.GetValue(1), rowNum32);
    CopyTinyStaticFastBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);

    DataCopyExtParams updateCopyOutParams{1, static_cast<uint32_t>(innerBytes), 0, 0, 0};
    bool hasOutput = false;
    if (targetRow0 >= 0 && targetRow0 != targetRow1) {
        DataCopyPad(outputGMY[static_cast<int64_t>(targetRow0) * innerBytes], updateLocal[0], updateCopyOutParams);
        hasOutput = true;
    }
    if (targetRow1 >= 0) {
        DataCopyPad(outputGMY[static_cast<int64_t>(targetRow1) * innerBytes],
            updateLocal[updateSlotBytes], updateCopyOutParams);
        hasOutput = true;
    }
    if (hasOutput) {
        PipeBarrier<PIPE_MTE3>();
    }
}

__aicore__ inline void ProcessPackedStaticIndex3FastPath(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    if (tilingData == nullptr || static_cast<int64_t>(GetBlockIdx()) != 0) {
        return;
    }
    const int64_t totalNum = tilingData->totalNum;
    const int64_t rowNum = tilingData->rowNum;
    const int64_t innerNum = tilingData->innerNum;
    const int64_t typeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 4;
    const int64_t totalBytes = totalNum * typeSize;
    const int64_t innerBytes = innerNum * typeSize;
    if (totalBytes <= static_cast<int64_t>(TINY_STATIC_BYTES) ||
        totalBytes > static_cast<int64_t>(PACKED_STATIC_BYTES) || rowNum <= 1 || innerBytes <= 0 ||
        innerBytes > static_cast<int64_t>(TINY_STATIC_BYTES)) {
        return;
    }

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    const int32_t rowNum32 = static_cast<int32_t>(rowNum);
    const int32_t targetRow0 = NormalizeTinyTargetRow(inputGMI.GetValue(0), rowNum32);
    const int32_t targetRow1 = NormalizeTinyTargetRow(inputGMI.GetValue(1), rowNum32);
    const int32_t targetRow2 = NormalizeTinyTargetRow(inputGMI.GetValue(2), rowNum32);

    if (innerBytes % BLOCK_BYTES != 0) {
        const int64_t updateSlotBytes = ((innerBytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
        LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, PACKED_STATIC_BYTES, 3 * TINY_STATIC_BYTES);
        DataCopyExtParams updateCopyInParams{3, static_cast<uint32_t>(innerBytes), 0, 0, 0};
        DataCopyPadExtParams<uint8_t> updatePadParams{
            false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
        DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);
        CopyPackedSmallStaticBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);

        const int32_t targetRows[3] = {targetRow0, targetRow1, targetRow2};
        DataCopyExtParams updateCopyOutParams{1, static_cast<uint32_t>(innerBytes), 0, 0, 0};
        for (int32_t index = 0; index < 3; ++index) {
            const int32_t targetRow = targetRows[index];
            if (targetRow < 0 || (index < 2 && targetRows[index + 1] == targetRow) ||
                (index == 0 && targetRows[2] == targetRow)) {
                continue;
            }
            DataCopyPad(outputGMY[static_cast<int64_t>(targetRow) * innerBytes],
                updateLocal[static_cast<int64_t>(index) * updateSlotBytes], updateCopyOutParams);
            PipeBarrier<PIPE_MTE3>();
        }
        return;
    }

    const int64_t updateSlotBytes = ((innerBytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
    LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, PACKED_STATIC_BYTES, 3 * TINY_STATIC_BYTES);
    DataCopyExtParams updateCopyInParams{3, static_cast<uint32_t>(innerBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> updatePadParams{
        false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
    DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);
    CopyPackedSmallStaticBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);

    DataCopyExtParams updateCopyOutParams{1, static_cast<uint32_t>(innerBytes), 0, 0, 0};
    bool hasOutput = false;
    if (targetRow0 >= 0 && targetRow0 != targetRow1 && targetRow0 != targetRow2) {
        DataCopyPad(outputGMY[static_cast<int64_t>(targetRow0) * innerBytes], updateLocal[0], updateCopyOutParams);
        hasOutput = true;
    }
    if (targetRow1 >= 0 && targetRow1 != targetRow2) {
        DataCopyPad(outputGMY[static_cast<int64_t>(targetRow1) * innerBytes],
            updateLocal[updateSlotBytes], updateCopyOutParams);
        hasOutput = true;
    }
    if (targetRow2 >= 0) {
        DataCopyPad(outputGMY[static_cast<int64_t>(targetRow2) * innerBytes],
            updateLocal[2 * updateSlotBytes], updateCopyOutParams);
        hasOutput = true;
    }
    if (hasOutput) {
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
    }
}

template <bool PREFETCH_UPDATE>
__aicore__ inline void ProcessTinyStaticIndex3NarrowFastPath(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    if (tilingData == nullptr || static_cast<int64_t>(GetBlockIdx()) != 0) {
        return;
    }
    const int64_t totalNum = tilingData->totalNum;
    const int64_t rowNum = tilingData->rowNum;
    const int64_t innerNum = tilingData->innerNum;
    const int64_t typeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 4;
    const int64_t totalBytes = totalNum * typeSize;
    const int64_t innerBytes = innerNum * typeSize;
    if (totalBytes <= 0 || totalBytes > static_cast<int64_t>(TINY_STATIC_BYTES) || rowNum <= 1 ||
        innerBytes <= 0 || innerBytes > static_cast<int64_t>(TINY_STATIC_BYTES) || tilingData->indexNum != 3) {
        return;
    }

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    const int64_t updateSlotBytes = ((innerBytes + BLOCK_BYTES - 1) / BLOCK_BYTES) * BLOCK_BYTES;
    LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, TINY_STATIC_BYTES, 3 * TINY_STATIC_BYTES);
    DataCopyExtParams copyInParams{3, static_cast<uint32_t>(innerBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> padParams{
        false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
    if constexpr (PREFETCH_UPDATE) {
        DataCopyPad(updateLocal, inputGMV[0], copyInParams, padParams);
    }
    const int32_t rowNum32 = static_cast<int32_t>(rowNum);
    const int32_t targetRow0 = NormalizeTinyTargetRow(inputGMI.GetValue(0), rowNum32);
    const int32_t targetRow1 = NormalizeTinyTargetRow(inputGMI.GetValue(1), rowNum32);
    const int32_t targetRow2 = NormalizeTinyTargetRow(inputGMI.GetValue(2), rowNum32);
    if constexpr (!PREFETCH_UPDATE) {
        DataCopyPad(updateLocal, inputGMV[0], copyInParams, padParams);
    }
    CopyTinyStaticFastBytes<false>(inputGMX, inputGMV, outputGMY, 0, 0, totalBytes);

    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(innerBytes), 0, 0, 0};
    bool hasOutput = false;
    if (targetRow0 >= 0 && targetRow0 != targetRow1 && targetRow0 != targetRow2) {
        DataCopyPad(outputGMY[static_cast<int64_t>(targetRow0) * innerBytes], updateLocal[0], copyOutParams);
        hasOutput = true;
    }
    if (targetRow1 >= 0 && targetRow1 != targetRow2) {
        DataCopyPad(outputGMY[static_cast<int64_t>(targetRow1) * innerBytes],
            updateLocal[updateSlotBytes], copyOutParams);
        hasOutput = true;
    }
    if (targetRow2 >= 0) {
        DataCopyPad(outputGMY[static_cast<int64_t>(targetRow2) * innerBytes],
            updateLocal[2 * updateSlotBytes], copyOutParams);
        hasOutput = true;
    }
    if (hasOutput) {
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
        WaitFlag<HardEvent::MTE3_MTE2>(static_cast<TEventID>(0));
    }
}

template <int32_t INNER_NUM>
__aicore__ inline void ApplyTinyIndex3UbComposeRow(
    __ubuf__ uint32_t* xAddress,
    __ubuf__ uint32_t* updateAddress,
    int32_t targetRow,
    int32_t updateIndex,
    int64_t updateSlotWords)
{
    static_assert(INNER_NUM == 3 || INNER_NUM == 4, "tiny index3 UB-compose supports width 3 or 4");
    if (targetRow < 0) {
        return;
    }
    const int64_t dstBase = static_cast<int64_t>(targetRow) * INNER_NUM;
    const int64_t srcBase = static_cast<int64_t>(updateIndex) * updateSlotWords;
    xAddress[dstBase] = updateAddress[srcBase];
    xAddress[dstBase + 1] = updateAddress[srcBase + 1];
    xAddress[dstBase + 2] = updateAddress[srcBase + 2];
    if constexpr (INNER_NUM == 4) {
        xAddress[dstBase + 3] = updateAddress[srcBase + 3];
    }
}

template <int32_t INNER_NUM>
__aicore__ inline void ProcessTinyStaticIndex3UbComposeFastPath(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    if (tilingData == nullptr || static_cast<int64_t>(GetBlockIdx()) != 0) {
        return;
    }
    const int64_t totalNum = tilingData->totalNum;
    const int64_t rowNum = tilingData->rowNum;
    const int64_t innerNum = tilingData->innerNum;
    const int64_t typeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 4;
    const int64_t totalBytes = totalNum * typeSize;
    constexpr int64_t innerBytes = INNER_NUM * static_cast<int64_t>(sizeof(uint32_t));
    if (totalBytes <= 0 || totalBytes > static_cast<int64_t>(TINY_STATIC_BYTES) || rowNum <= 1 ||
        innerNum != INNER_NUM || typeSize != static_cast<int64_t>(sizeof(uint32_t)) ||
        tilingData->indexNum != 3) {
        return;
    }

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    LocalTensor<uint8_t> xLocal(TPosition::VECCALC, 0, TINY_STATIC_BYTES);
    constexpr int64_t updateSlotBytes = BLOCK_BYTES;
    LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, TINY_STATIC_BYTES, 3 * TINY_STATIC_BYTES);

    DataCopyExtParams xCopyInParams{1, static_cast<uint32_t>(totalBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[0], xCopyInParams, xPadParams);

    DataCopyExtParams updateCopyInParams{3, static_cast<uint32_t>(innerBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> updatePadParams{
        false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
    DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);

    const int32_t rowNum32 = static_cast<int32_t>(rowNum);
    const int32_t targetRow0 = NormalizeTinyTargetRow(inputGMI.GetValue(0), rowNum32);
    const int32_t targetRow1 = NormalizeTinyTargetRow(inputGMI.GetValue(1), rowNum32);
    const int32_t targetRow2 = NormalizeTinyTargetRow(inputGMI.GetValue(2), rowNum32);

    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    __ubuf__ uint32_t* xAddress = reinterpret_cast<__ubuf__ uint32_t*>(xLocal.GetPhyAddr());
    __ubuf__ uint32_t* updateAddress = reinterpret_cast<__ubuf__ uint32_t*>(updateLocal.GetPhyAddr());
    constexpr int64_t updateSlotWords = updateSlotBytes / static_cast<int64_t>(sizeof(uint32_t));
    ApplyTinyIndex3UbComposeRow<INNER_NUM>(xAddress, updateAddress, targetRow0, 0, updateSlotWords);
    ApplyTinyIndex3UbComposeRow<INNER_NUM>(xAddress, updateAddress, targetRow1, 1, updateSlotWords);
    ApplyTinyIndex3UbComposeRow<INNER_NUM>(xAddress, updateAddress, targetRow2, 2, updateSlotWords);

    SetFlag<HardEvent::S_MTE3>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::S_MTE3>(static_cast<TEventID>(0));
    if ((totalBytes & (BLOCK_BYTES - 1)) == 0) {
        DataCopy(outputGMY[0], xLocal, static_cast<uint32_t>(totalBytes));
    } else {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(totalBytes), 0, 0, 0};
        DataCopyPad(outputGMY[0], xLocal, copyOutParams);
    }
}

template <int32_t INNER_BYTES>
__aicore__ inline void ApplyTinyIndex2UbComposeRow(
    __ubuf__ uint32_t* xAddress,
    __ubuf__ uint32_t* updateAddress,
    int32_t targetRow,
    int32_t updateIndex)
{
    static_assert(INNER_BYTES == 16 || INNER_BYTES == 24,
        "tiny index2 UB-compose supports 16B or 24B rows");
    if (targetRow < 0) {
        return;
    }
    constexpr int64_t rowWords = INNER_BYTES / static_cast<int64_t>(sizeof(uint32_t));
    constexpr int64_t updateSlotWords = BLOCK_BYTES / static_cast<int64_t>(sizeof(uint32_t));
    const int64_t dstBase = static_cast<int64_t>(targetRow) * rowWords;
    const int64_t srcBase = static_cast<int64_t>(updateIndex) * updateSlotWords;
    xAddress[dstBase] = updateAddress[srcBase];
    xAddress[dstBase + 1] = updateAddress[srcBase + 1];
    xAddress[dstBase + 2] = updateAddress[srcBase + 2];
    xAddress[dstBase + 3] = updateAddress[srcBase + 3];
    if constexpr (INNER_BYTES == 24) {
        xAddress[dstBase + 4] = updateAddress[srcBase + 4];
        xAddress[dstBase + 5] = updateAddress[srcBase + 5];
    }
}

template <int32_t INNER_BYTES>
__aicore__ inline void ProcessTinyStaticIndex2UbComposeFastPath(
    GM_ADDR x,
    GM_ADDR i,
    GM_ADDR v,
    GM_ADDR y,
    const InplaceUpdateTilingData* tilingData)
{
    if (tilingData == nullptr || static_cast<int64_t>(GetBlockIdx()) != 0) {
        return;
    }
    const int64_t totalNum = tilingData->totalNum;
    const int64_t rowNum = tilingData->rowNum;
    const int64_t innerNum = tilingData->innerNum;
    const int64_t typeSize = tilingData->typeSize > 0 ? tilingData->typeSize : 2;
    const int64_t totalBytes = totalNum * typeSize;
    constexpr int64_t innerBytes = INNER_BYTES;
    if (totalBytes <= 0 || totalBytes > static_cast<int64_t>(TINY_STATIC_BYTES) || rowNum <= 1 ||
        typeSize <= 0 || typeSize > 4 || innerNum * typeSize != innerBytes || tilingData->indexNum != 2) {
        return;
    }

    GlobalTensor<uint8_t> inputGMX;
    GlobalTensor<int32_t> inputGMI;
    GlobalTensor<uint8_t> inputGMV;
    GlobalTensor<uint8_t> outputGMY;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(x));
    inputGMI.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(i));
    inputGMV.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(v));
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));

    LocalTensor<uint8_t> xLocal(TPosition::VECCALC, 0, TINY_STATIC_BYTES);
    constexpr int64_t updateSlotBytes = BLOCK_BYTES;
    LocalTensor<uint8_t> updateLocal(TPosition::VECCALC, TINY_STATIC_BYTES, 2 * TINY_STATIC_BYTES);

    DataCopyExtParams xCopyInParams{1, static_cast<uint32_t>(totalBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocal, inputGMX[0], xCopyInParams, xPadParams);

    DataCopyExtParams updateCopyInParams{2, static_cast<uint32_t>(innerBytes), 0, 0, 0};
    DataCopyPadExtParams<uint8_t> updatePadParams{
        false, 0, static_cast<uint8_t>(updateSlotBytes - innerBytes), 0};
    DataCopyPad(updateLocal, inputGMV[0], updateCopyInParams, updatePadParams);

    const int32_t rowNum32 = static_cast<int32_t>(rowNum);
    const int32_t targetRow0 = NormalizeTinyTargetRow(inputGMI.GetValue(0), rowNum32);
    const int32_t targetRow1 = NormalizeTinyTargetRow(inputGMI.GetValue(1), rowNum32);

    SetFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::MTE2_S>(static_cast<TEventID>(0));
    __ubuf__ uint32_t* xAddress = reinterpret_cast<__ubuf__ uint32_t*>(xLocal.GetPhyAddr());
    __ubuf__ uint32_t* updateAddress = reinterpret_cast<__ubuf__ uint32_t*>(updateLocal.GetPhyAddr());
    ApplyTinyIndex2UbComposeRow<INNER_BYTES>(xAddress, updateAddress, targetRow0, 0);
    ApplyTinyIndex2UbComposeRow<INNER_BYTES>(xAddress, updateAddress, targetRow1, 1);

    SetFlag<HardEvent::S_MTE3>(static_cast<TEventID>(0));
    WaitFlag<HardEvent::S_MTE3>(static_cast<TEventID>(0));
    if ((totalBytes & (BLOCK_BYTES - 1)) == 0) {
        DataCopy(outputGMY[0], xLocal, static_cast<uint32_t>(totalBytes));
    } else {
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(totalBytes), 0, 0, 0};
        DataCopyPad(outputGMY[0], xLocal, copyOutParams);
    }
}

} // namespace NsInplaceUpdate
#endif // INPLACEUPDATE_H
