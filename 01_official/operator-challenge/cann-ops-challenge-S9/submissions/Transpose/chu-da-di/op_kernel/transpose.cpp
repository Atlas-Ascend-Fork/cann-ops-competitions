#include "kernel_operator.h"
#include "../op_host/transpose_tiling.h"
#include "transpose_macro.h"
#include "transpose_suffix_dma.h"
#include "transpose_suffix_block.h"
#include "transpose_l2_staged.h"
#include "transpose_direct_mask.h"
#include "transpose_direct_mask_unroll.h"
#include "transpose_owner64.h"
#include "transpose_nocount.h"
#include "transpose_column_owner.h"
#include "transpose_row_aligned.h"
#include "transpose_int8_pairs.h"

namespace s9transpose {
constexpr uint32_t kBufferNum = 1;
constexpr uint32_t kSwapBufferNum = 2;
constexpr uint32_t kCopyBufferNum = 2;
constexpr uint64_t kBlockBytes = 32;
constexpr uint64_t kMicro = 16;
constexpr uint64_t kGroupedRowTiles = 8;
constexpr uint64_t kGroupedRowMinimumTiles = 64;

template <typename T, bool kStaged = false>
class KernelTranspose {
public:
    __aicore__ inline void InitStages()
    {
        // Two slots per queue, at most 128 rows x eight 32-byte columns.
        // Queue storage is reused only after the previous stage's SyncAll.
        pipe_.InitBuffer(inQueue_, 2, 32768);
        pipe_.InitBuffer(outQueue_, 2, 32768);
    }
    __aicore__ inline void ProcessBatchedMatrices(GM_ADDR mapAddress,
        uint32_t batchMatrices, uint32_t mapCount,
        const AscendC::LocalTensor<uint32_t>& indices)
    {
        const uint64_t matrix = data_.matrixH * data_.matrixW;
        const uint64_t tasks = (data_.outer + batchMatrices - 1) / batchMatrices;
        if (AscendC::GetBlockIdx() >= tasks) { return; }
        AscendC::GlobalTensor<uint32_t> map;
        map.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(mapAddress));
        AscendC::DataCopyExtParams mapCopy{1, mapCount * 4, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> mapPad{false, 0, 0, 0};
        AscendC::DataCopyPad(indices, map, mapCopy, mapPad);
        BatchSync<AscendC::HardEvent::MTE2_V>();
        for (uint64_t task = AscendC::GetBlockIdx(); task < tasks;
             task += AscendC::GetBlockNum()) {
            const uint64_t firstMatrix = task * batchMatrices;
            const uint64_t matrices = data_.outer - firstMatrix < batchMatrices
                ? data_.outer - firstMatrix : batchMatrices;
            const uint32_t count = static_cast<uint32_t>(matrices * matrix);
            const uint64_t begin = firstMatrix * matrix;
            auto raw = inQueue_.AllocTensor<T>();
            if constexpr (sizeof(T) == 1) {
                const uint32_t padded = (count + 127) / 128 * 128;
                auto source = inQueue_.AllocTensor<half>();
                auto result = outQueue_.AllocTensor<half>();
                auto bytes = outQueue_.AllocTensor<T>();
                AscendC::Duplicate(raw.template ReinterpretCast<uint16_t>(),
                    static_cast<uint16_t>(0), padded / 2);
                BatchSync<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopyExtParams copy{1, count, 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                AscendC::DataCopyPad(raw, xGm_[begin], copy, pad);
                BatchSync<AscendC::HardEvent::MTE2_V>();
                AscendC::Cast(source, raw.template ReinterpretCast<int8_t>(),
                    AscendC::RoundMode::CAST_NONE, padded);
                AscendC::Duplicate(result, static_cast<half>(0), padded);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t offset = 0; offset < count; offset += 2048) {
                    const uint32_t once = count - offset < 2048 ? count - offset : 2048;
                    AscendC::Gather(result[offset], source, indices[offset], 0, once);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(bytes.template ReinterpretCast<int8_t>(), result,
                    AscendC::RoundMode::CAST_NONE, padded);
                BatchSync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_[begin], bytes, copy);
                BatchSync<AscendC::HardEvent::MTE3_V>();
                inQueue_.FreeTensor(raw); inQueue_.FreeTensor(source);
                outQueue_.FreeTensor(result); outQueue_.FreeTensor(bytes);
            } else {
                auto result = outQueue_.AllocTensor<T>();
                BatchSync<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopyExtParams copy{1,
                    static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                AscendC::DataCopyPad(raw, xGm_[begin], copy, pad);
                BatchSync<AscendC::HardEvent::MTE2_V>();
                for (uint32_t offset = 0; offset < count; offset += 2048) {
                    const uint32_t once = count - offset < 2048 ? count - offset : 2048;
                    if constexpr (sizeof(T) == 4) {
                        AscendC::Gather(result.template ReinterpretCast<float>()[offset],
                            raw.template ReinterpretCast<float>(), indices[offset], 0, once);
                    } else {
                        AscendC::Gather(result.template ReinterpretCast<half>()[offset],
                            raw.template ReinterpretCast<half>(), indices[offset], 0, once);
                    }
                }
                BatchSync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_[begin], result, copy);
                BatchSync<AscendC::HardEvent::MTE3_V>();
                inQueue_.FreeTensor(raw); outQueue_.FreeTensor(result);
            }
        }
    }

    __aicore__ inline void ProcessWideMatrices(GM_ADDR mapAddress,
        uint32_t tileCols, uint32_t mapCount,
        const AscendC::LocalTensor<uint32_t>& indices)
    {
        const uint64_t matrix = data_.matrixH * data_.matrixW;
        const uint64_t columns = data_.outer * data_.matrixW;
        const uint64_t tasks = (columns + tileCols - 1) / tileCols;
        if (AscendC::GetBlockIdx() >= tasks) { return; }
        AscendC::GlobalTensor<uint32_t> map;
        map.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(mapAddress));
        AscendC::DataCopyExtParams mapCopy{1, mapCount * 4, 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> mapPad{false, 0, 0, 0};
        AscendC::DataCopyPad(indices, map, mapCopy, mapPad);
        BatchSync<AscendC::HardEvent::MTE2_V>();
        for (uint64_t task = AscendC::GetBlockIdx(); task < tasks;
             task += AscendC::GetBlockNum()) {
            uint64_t globalColumn = task * tileCols;
            const uint64_t taskEnd = columns - globalColumn < tileCols
                ? columns : globalColumn + tileCols;
            while (globalColumn < taskEnd) {
            const uint64_t outer = globalColumn / data_.matrixW;
            const uint64_t column = globalColumn % data_.matrixW;
            const uint32_t validCols = static_cast<uint32_t>(
                data_.matrixW - column < taskEnd - globalColumn
                    ? data_.matrixW - column : taskEnd - globalColumn);
            const uint32_t count = static_cast<uint32_t>(validCols * data_.matrixH);
            const uint64_t begin = globalColumn * data_.matrixH;
            const uint64_t inputBegin = outer * matrix + column;
            const uint32_t alignment = 32 / sizeof(T);
            AscendC::DataCopyExtParams inputCopy{static_cast<uint16_t>(data_.matrixH),
                static_cast<uint32_t>(validCols * sizeof(T)),
                static_cast<uint32_t>((data_.matrixW - validCols) * sizeof(T)),
                static_cast<uint32_t>((tileCols - (validCols + alignment - 1) / alignment * alignment) / alignment), 0};
            auto raw = inQueue_.AllocTensor<T>();
            if constexpr (sizeof(T) == 1) {
                const uint32_t padded = (count + 127) / 128 * 128;
                const uint32_t sourcePadded = (mapCount + 127) / 128 * 128;
                auto source = inQueue_.AllocTensor<half>();
                auto result = outQueue_.AllocTensor<half>();
                auto bytes = outQueue_.AllocTensor<T>();
                AscendC::Duplicate(raw.template ReinterpretCast<uint16_t>(),
                    static_cast<uint16_t>(0), sourcePadded / 2);
                BatchSync<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopyExtParams copy{1, count, 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                AscendC::DataCopyPad(raw, xGm_[inputBegin], inputCopy, pad);
                BatchSync<AscendC::HardEvent::MTE2_V>();
                AscendC::Cast(source, raw.template ReinterpretCast<int8_t>(),
                    AscendC::RoundMode::CAST_NONE, sourcePadded);
                AscendC::Duplicate(result, static_cast<half>(0), padded);
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t offset = 0; offset < count; offset += 2048) {
                    const uint32_t once = count - offset < 2048 ? count - offset : 2048;
                    AscendC::Gather(result[offset], source, indices[offset], 0, once);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(bytes.template ReinterpretCast<int8_t>(), result,
                    AscendC::RoundMode::CAST_NONE, padded);
                BatchSync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_[begin], bytes, copy);
                BatchSync<AscendC::HardEvent::MTE3_V>();
                inQueue_.FreeTensor(raw); inQueue_.FreeTensor(source);
                outQueue_.FreeTensor(result); outQueue_.FreeTensor(bytes);
            } else {
                auto result = outQueue_.AllocTensor<T>();
                BatchSync<AscendC::HardEvent::V_MTE2>();
                AscendC::DataCopyExtParams copy{1,
                    static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                AscendC::DataCopyPad(raw, xGm_[inputBegin], inputCopy, pad);
                BatchSync<AscendC::HardEvent::MTE2_V>();
                for (uint32_t offset = 0; offset < count; offset += 2048) {
                    const uint32_t once = count - offset < 2048 ? count - offset : 2048;
                    if constexpr (sizeof(T) == 4) {
                        AscendC::Gather(result.template ReinterpretCast<float>()[offset],
                            raw.template ReinterpretCast<float>(), indices[offset], 0, once);
                    } else {
                        AscendC::Gather(result.template ReinterpretCast<half>()[offset],
                            raw.template ReinterpretCast<half>(), indices[offset], 0, once);
                    }
                }
                BatchSync<AscendC::HardEvent::V_MTE3>();
                AscendC::DataCopyPad(yGm_[begin], result, copy);
                BatchSync<AscendC::HardEvent::MTE3_V>();
                inQueue_.FreeTensor(raw); outQueue_.FreeTensor(result);
            }
            globalColumn += validCols;
            }
        }
    }

    __aicore__ inline void ConfigureStage(GM_ADDR x, GM_ADDR y,
        const TransposeTilingData& data)
    {
        data_ = data;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.totalElements);
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeTilingData& tiling)
    {
        data_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.totalElements);
        const uint64_t bufferElements = data_.tileElements < kMicro * kMicro ? kMicro * kMicro : data_.tileElements;
        const Mode mode = static_cast<Mode>(data_.mode);
        if (mode == Mode::kIdentity || mode == Mode::kContiguousSuffix) {
            // Pure-movement modes need no VEC instruction between MTE2 and
            // MTE3.  A bound two-slot queue carries the hardware dependency
            // while one slot is filled and the preceding slot is drained.
            pipe_.InitBuffer(copyQueue_, kCopyBufferNum, bufferElements * sizeof(T));
        } else if (mode == Mode::kSwapLastTwo && data_.fastSwap16 != 0) {
            // A core can own several swap tasks.  Two physical slots are
            // required so the next MTE2/TransData sequence cannot recycle UB
            // still referenced by the previous task's MTE3.  Size for the
            // two-row-tile whole-width path, which is the largest swap slab.
            const bool groupedRows = UseGroupedRows();
            const uint64_t bufferedRowTiles = groupedRows ? kGroupedRowTiles : 2;
            const uint32_t swapBufferNum = groupedRows ? kBufferNum : kSwapBufferNum;
            const uint64_t swapBufferElements = bufferedRowTiles * kMicro *
                data_.tileBatchCols * data_.microCols;
            pipe_.InitBuffer(
                inQueue_, swapBufferNum, swapBufferElements * sizeof(T));
            pipe_.InitBuffer(
                outQueue_, swapBufferNum, swapBufferElements * sizeof(T));
        } else if (mode == Mode::kPermutedMatrix) {
            const uint64_t reverseWidth = data_.tileBatchCols * data_.microCols;
            const uint64_t reverseTileElements = kMicro * (sizeof(T) == 1
                ? (reverseWidth + 31) / 32 * 32 : reverseWidth);
            pipe_.InitBuffer(
                inQueue_, kBufferNum, reverseTileElements * sizeof(T));
            pipe_.InitBuffer(
                outQueue_, kBufferNum, reverseTileElements * sizeof(T) * (sizeof(T) == 1 ? 2 : 1));
            if constexpr (sizeof(T) == 1) {
                pipe_.InitBuffer(matrixInputHalf_, reverseTileElements * sizeof(half));
                pipe_.InitBuffer(matrixOutputHalf_, 2 * reverseTileElements * sizeof(half));
            }
        } else {
            pipe_.InitBuffer(inQueue_, kBufferNum, bufferElements * sizeof(T));
            pipe_.InitBuffer(outQueue_, kBufferNum, bufferElements * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        const Mode mode = static_cast<Mode>(data_.mode);
        if (mode == Mode::kIdentity) {
            ProcessIdentity();
        } else if (mode == Mode::kContiguousSuffix) {
            ProcessSuffix();
        } else if (mode == Mode::kSwapLastTwo && data_.fastSwap16 != 0) {
            if constexpr (sizeof(T) == 1) {
                if (UseInt8RowPairParallel()) {
                    ProcessSwapInt8RowPairs();
                } else {
                    ProcessSwapInt8Blocks();
                }
            } else if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
                if (UseGroupedRows()) {
                    ProcessSwapGroupedRows();
                } else if (data_.columnBatches == 1 && data_.matrixH % kMicro == 0) {
                    ProcessSwapWholeWidthRows();
                } else {
                    ProcessSwapBlocks();
                }
            } else {
                ProcessGeneric();
            }
        } else if (mode == Mode::kPermutedMatrix) {
            if constexpr (sizeof(T) == 1) { ProcessPermutedMatrixInt8(); }
            else { ProcessPermutedMatrix(); }
        } else {
            ProcessGeneric();
        }
    }

private:
    template <AscendC::HardEvent event> __aicore__ inline void BatchSync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        AscendC::SetFlag<event>(id); AscendC::WaitFlag<event>(id);
    }
    __aicore__ inline bool UseGroupedRows() const
    {
        if constexpr (sizeof(T) == 1) {
            return false;
        }
        const uint64_t groupedTasks = data_.outer *
            ((data_.tileRows + kGroupedRowTiles - 1) / kGroupedRowTiles) *
            data_.columnBatches;
        const bool stagedBatch = kStaged && data_.rowOwnershipQuantum == 2 &&
            data_.tileRows > 1 && groupedTasks >= 2 * AscendC::GetBlockNum();
        return stagedBatch || (data_.tileRows >= kGroupedRowMinimumTiles &&
            data_.columnBatches > 1);
    }

    __aicore__ inline bool UseInt8RowPairParallel() const
    {
        if constexpr (sizeof(T) != 1) {
            return false;
        }
        // The host enlarges blockDim beyond the column-owned task count only
        // when full 32-row pairs are needed to fill otherwise-idle AIVs.
        return data_.outer * data_.columnBatches < AscendC::GetBlockNum();
    }

    __aicore__ inline void Partition(uint64_t total, uint64_t& start, uint64_t& count) const
    {
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        if constexpr (kStaged) {
        if (data_.rowOwnershipQuantum == 3 && data_.tileRows == 1) {
            // Very short output rows can share a 32-byte GM block. Assign
            // whole aligned batches of matrices to one AIV, including all
            // of their column tiles, so core boundaries stay disjoint.
            uint64_t a = data_.matrixH * data_.matrixW * sizeof(T), b = 32;
            while (b != 0) { const uint64_t r = a % b; a = b; b = r; }
            const uint64_t quantum = 32 / a;
            const uint64_t groups = (data_.outer + quantum - 1) / quantum;
            const uint64_t each = groups / blocks, extra = groups % blocks;
            const uint64_t first = block * each + (block < extra ? block : extra);
            const uint64_t owned = each + (block < extra ? 1 : 0);
            const uint64_t outerStart = first * quantum;
            const uint64_t outerEnd = (first + owned) * quantum < data_.outer
                ? (first + owned) * quantum : data_.outer;
            start = outerStart * data_.columnBatches;
            count = outerStart < outerEnd ? (outerEnd - outerStart) * data_.columnBatches : 0;
            return;
        }
        }
        const uint64_t small = total / blocks;
        const uint64_t remainder = total % blocks;
        count = small + (block < remainder ? 1 : 0);
        start = block * small + (block < remainder ? block : remainder);
    }

    __aicore__ inline void PartitionAlignedElements(
        uint64_t total, uint64_t& start, uint64_t& count) const
    {
        const uint64_t blockElements = kBlockBytes / sizeof(T);
        const uint64_t units = (total + blockElements - 1) / blockElements;
        uint64_t unitStart = 0;
        uint64_t unitCount = 0;
        Partition(units, unitStart, unitCount);
        start = unitStart * blockElements;
        const uint64_t end = (unitStart + unitCount) * blockElements < total
                                 ? (unitStart + unitCount) * blockElements
                                 : total;
        count = start < total ? end - start : 0;
    }

    __aicore__ inline void WaitVectorBeforeMte2Reuse() const
    {
        const event_t event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
    }

    __aicore__ inline void WaitMte3BeforeVectorReuse() const
    {
        const event_t event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
    }

    __aicore__ inline uint64_t InputOffset(uint64_t outputLinear) const
    {
        uint64_t remaining = outputLinear;
        uint64_t offset = 0;
        for (int32_t outputAxis = static_cast<int32_t>(data_.rank) - 1; outputAxis >= 0; --outputAxis) {
            const uint64_t extent = data_.outputShape[outputAxis];
            const uint64_t coordinate = remaining % extent;
            remaining /= extent;
            offset += coordinate * data_.inputStride[data_.perm[outputAxis]];
        }
        return offset;
    }

    __aicore__ inline uint64_t AlignElements(uint64_t elements) const
    {
        const uint64_t blockElements = kBlockBytes / sizeof(T);
        return (elements + blockElements - 1) / blockElements * blockElements;
    }

    __aicore__ inline void CopySegment(uint64_t inputOffset, uint64_t outputOffset, uint64_t elements)
    {
        bool hasPending = false;
        uint64_t pendingOutputOffset = 0;
        uint32_t pendingElements = 0;
        uint64_t copied = 0;
        while (copied < elements) {
            const uint64_t count = elements - copied < data_.tileElements ? elements - copied : data_.tileElements;
            AscendC::LocalTensor<T> local = copyQueue_.AllocTensor<T>();
            AscendC::DataCopyExtParams inputParams{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            AscendC::DataCopyPad(local, xGm_[inputOffset + copied], inputParams, padParams);
            copyQueue_.EnQue(local);
            if (hasPending) {
                local = copyQueue_.DeQue<T>();
                AscendC::DataCopyExtParams outputParams{
                    1, static_cast<uint32_t>(pendingElements * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(yGm_[pendingOutputOffset], local, outputParams);
                copyQueue_.FreeTensor(local);
            }
            pendingOutputOffset = outputOffset + copied;
            pendingElements = static_cast<uint32_t>(count);
            hasPending = true;
            copied += count;
        }
        if (hasPending) {
            AscendC::LocalTensor<T> local = copyQueue_.DeQue<T>();
            AscendC::DataCopyExtParams outputParams{
                1, static_cast<uint32_t>(pendingElements * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[pendingOutputOffset], local, outputParams);
            copyQueue_.FreeTensor(local);
        }
    }

    __aicore__ inline void ProcessIdentity()
    {
        uint64_t start = 0;
        uint64_t count = 0;
        Partition(data_.totalElements, start, count);
        if (count != 0) {
            CopySegment(start, start, count);
        }
    }

    __aicore__ inline void ProcessSuffix()
    {
        uint64_t blockStart = 0;
        uint64_t blockCount = 0;
        Partition(data_.outputBlocks, blockStart, blockCount);
        for (uint64_t block = blockStart; block < blockStart + blockCount; ++block) {
            const uint64_t outputOffset = block * data_.innerElements;
            CopySegment(InputOffset(outputOffset), outputOffset, data_.innerElements);
        }
    }

    __aicore__ inline void ProcessSwapBlocks()
    {
        // Batch adjacent 16-row x one-datablock column tiles.  The datablock is
        // 16 FP16 or 8 FP32/INT32 values.  One 2-D MTE2 transfer gathers the
        // slab, TransData consumes all tiles as repeats, and one 2-D MTE3
        // transfer scatters the packed transpose.  The destination address
        // list follows the A2 32-bit interleave documented by the stock CANN
        // transpose implementation; for 16-bit values groups==1 and it
        // reduces to the ordinary 16x16 layout.
        const uint64_t tasksPerMatrix = data_.tileRows * data_.columnBatches;
        const uint64_t totalTasks = data_.outer * tasksPerMatrix;
        uint64_t taskStart = 0;
        uint64_t taskCount = 0;
        Partition(totalTasks, taskStart, taskCount);
        for (uint64_t task = taskStart; task < taskStart + taskCount; ++task) {
            const uint64_t outerIndex = task / tasksPerMatrix;
            const uint64_t matrixTask = task % tasksPerMatrix;
            const uint64_t rowTile = matrixTask / data_.columnBatches;
            const uint64_t columnBatch = matrixTask % data_.columnBatches;
            const uint64_t firstColumnTile = columnBatch * data_.tileBatchCols;
            const uint64_t remainingTiles = data_.tileCols - firstColumnTile;
            const uint64_t currentTiles = remainingTiles < data_.tileBatchCols
                                              ? remainingTiles
                                              : data_.tileBatchCols;
            const uint64_t batchWidth = currentTiles * data_.microCols;
            const uint64_t row = rowTile * kMicro;
            const uint64_t column = firstColumnTile * data_.microCols;
            const uint64_t validRows = data_.matrixH - row < kMicro
                                           ? data_.matrixH - row
                                           : kMicro;
            const uint64_t validWidth = data_.matrixW - column < batchWidth
                                            ? data_.matrixW - column
                                            : batchWidth;
            const uint64_t inputOffset = outerIndex * data_.matrixH * data_.matrixW + row * data_.matrixW + column;
            const uint64_t outputOffset = outerIndex * data_.matrixH * data_.matrixW + column * data_.matrixH + row;

            AscendC::LocalTensor<T> input = inQueue_.AllocTensor<T>();
            AscendC::DataCopyExtParams inputParams{static_cast<uint16_t>(validRows),
                static_cast<uint32_t>(validWidth * sizeof(T)),
                static_cast<uint32_t>((data_.matrixW - validWidth) * sizeof(T)), 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{
                validWidth != batchWidth, 0,
                static_cast<uint8_t>(batchWidth - validWidth), static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm_[inputOffset], inputParams, padParams);
            inQueue_.EnQue(input);

            input = inQueue_.DeQue<T>();
            AscendC::LocalTensor<T> output = outQueue_.AllocTensor<T>();
            uint64_t sourceList[kMicro];
            uint64_t destinationList[kMicro];
            for (uint64_t i = 0; i < kMicro; ++i) {
                sourceList[i] = reinterpret_cast<uint64_t>(input[i * batchWidth].GetPhyAddr());
            }
            const uint64_t groups = kMicro / data_.microCols;
            for (uint64_t columnInBlock = 0; columnInBlock < data_.microCols; ++columnInBlock) {
                for (uint64_t group = 0; group < groups; ++group) {
                    const uint64_t listIndex = columnInBlock * groups + group;
                    const uint64_t outputElement = columnInBlock * kMicro + group * data_.microCols;
                    destinationList[listIndex] =
                        reinterpret_cast<uint64_t>(output[outputElement].GetPhyAddr());
                }
            }
            const uint16_t destinationRepeatStride = currentTiles == 1 ? 0 : kMicro;
            const uint16_t sourceRepeatStride = currentTiles == 1 ? 0 : 1;
            AscendC::TransDataTo5HDParams transParams{false, false, static_cast<uint8_t>(currentTiles),
                destinationRepeatStride, sourceRepeatStride};
            AscendC::TransDataTo5HD<T>(destinationList, sourceList, transParams);
            // The next task reuses both single-slot queues.  Complete the
            // vector read/write before either UB tensor can be recycled.
            AscendC::PipeBarrier<PIPE_V>();
            outQueue_.EnQue(output);
            WaitVectorBeforeMte2Reuse();
            inQueue_.FreeTensor(input);

            output = outQueue_.DeQue<T>();
            const uint64_t outputBlockBytes = validRows * sizeof(T);
            const uint64_t consumedLocalBytes =
                (outputBlockBytes + kBlockBytes - 1) / kBlockBytes * kBlockBytes;
            const uint32_t sourceStrideBlocks = static_cast<uint32_t>(
                (kMicro * sizeof(T) - consumedLocalBytes) / kBlockBytes);
            AscendC::DataCopyExtParams outputParams{static_cast<uint16_t>(validWidth),
                static_cast<uint32_t>(outputBlockBytes), sourceStrideBlocks,
                static_cast<uint32_t>((data_.matrixH - validRows) * sizeof(T)), 0};
            AscendC::DataCopyPad(yGm_[outputOffset], output, outputParams);
            WaitMte3BeforeVectorReuse();
            outQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline void ProcessSwapGroupedRows()
    {
        // The ordinary path emits one strided MTE3 block per output column
        // for only sixteen input rows.  On large matrices that means thousands
        // of 32/64-byte stores.  Group eight adjacent row tiles so each store
        // carries 128 FP16 or 256 FP32/INT32 bytes while preserving the same
        // independent output-column ownership.
        const uint64_t rowGroupsPerMatrix =
            (data_.tileRows + kGroupedRowTiles - 1) / kGroupedRowTiles;
        const uint64_t tasksPerMatrix = rowGroupsPerMatrix * data_.columnBatches;
        const uint64_t totalTasks = data_.outer * tasksPerMatrix;
        uint64_t taskStart = 0;
        uint64_t taskCount = 0;
        Partition(totalTasks, taskStart, taskCount);
        for (uint64_t task = taskStart; task < taskStart + taskCount; ++task) {
            const uint64_t outerIndex = task / tasksPerMatrix;
            const uint64_t matrixTask = task % tasksPerMatrix;
            const uint64_t rowGroup = matrixTask / data_.columnBatches;
            const uint64_t columnBatch = matrixTask % data_.columnBatches;

            const uint64_t firstRowTile = rowGroup * kGroupedRowTiles;
            const uint64_t remainingRowTiles = data_.tileRows - firstRowTile;
            const uint64_t currentRowTiles = remainingRowTiles < kGroupedRowTiles
                ? remainingRowTiles : kGroupedRowTiles;
            const uint64_t paddedRowSpan = currentRowTiles * kMicro;
            const uint64_t row = firstRowTile * kMicro;
            const uint64_t validRows = data_.matrixH - row < paddedRowSpan
                ? data_.matrixH - row : paddedRowSpan;

            const uint64_t firstColumnTile = columnBatch * data_.tileBatchCols;
            const uint64_t remainingColumnTiles = data_.tileCols - firstColumnTile;
            const uint64_t currentColumnTiles =
                remainingColumnTiles < data_.tileBatchCols
                    ? remainingColumnTiles : data_.tileBatchCols;
            const uint64_t batchWidth = currentColumnTiles * data_.microCols;
            const uint64_t column = firstColumnTile * data_.microCols;
            const uint64_t validWidth = data_.matrixW - column < batchWidth
                ? data_.matrixW - column : batchWidth;
            const uint64_t matrixBase =
                outerIndex * data_.matrixH * data_.matrixW;

            AscendC::LocalTensor<T> input = inQueue_.AllocTensor<T>();
            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(validRows),
                static_cast<uint32_t>(validWidth * sizeof(T)),
                static_cast<uint32_t>((data_.matrixW - validWidth) * sizeof(T)),
                0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{
                validWidth != batchWidth, 0,
                static_cast<uint8_t>(batchWidth - validWidth), static_cast<T>(0)};
            AscendC::DataCopyPad(
                input, xGm_[matrixBase + row * data_.matrixW + column],
                inputParams, padParams);
            inQueue_.EnQue(input);

            input = inQueue_.DeQue<T>();
            AscendC::LocalTensor<T> output = outQueue_.AllocTensor<T>();
            uint64_t sourceList[kMicro];
            uint64_t destinationList[kMicro];
            const uint64_t groups = kMicro / data_.microCols;
            for (uint64_t localRowTile = 0;
                 localRowTile < currentRowTiles; ++localRowTile) {
                for (uint64_t i = 0; i < kMicro; ++i) {
                    sourceList[i] = reinterpret_cast<uint64_t>(
                        input[(localRowTile * kMicro + i) * batchWidth].GetPhyAddr());
                }
                for (uint64_t columnInBlock = 0;
                     columnInBlock < data_.microCols; ++columnInBlock) {
                    for (uint64_t group = 0; group < groups; ++group) {
                        const uint64_t listIndex = columnInBlock * groups + group;
                        const uint64_t outputElement =
                            columnInBlock * paddedRowSpan +
                            localRowTile * kMicro + group * data_.microCols;
                        destinationList[listIndex] = reinterpret_cast<uint64_t>(
                            output[outputElement].GetPhyAddr());
                    }
                }
                const uint16_t destinationRepeatStride =
                    currentColumnTiles == 1
                        ? 0 : static_cast<uint16_t>(paddedRowSpan);
                const uint16_t sourceRepeatStride =
                    currentColumnTiles == 1 ? 0 : 1;
                AscendC::TransDataTo5HDParams transParams{
                    false, false, static_cast<uint8_t>(currentColumnTiles),
                    destinationRepeatStride, sourceRepeatStride};
                AscendC::TransDataTo5HD<T>(
                    destinationList, sourceList, transParams);
                AscendC::PipeBarrier<PIPE_V>();
            }
            outQueue_.EnQue(output);
            WaitVectorBeforeMte2Reuse();
            inQueue_.FreeTensor(input);

            output = outQueue_.DeQue<T>();
            const uint64_t outputBlockBytes = validRows * sizeof(T);
            const uint64_t consumedLocalBytes =
                (outputBlockBytes + kBlockBytes - 1) / kBlockBytes * kBlockBytes;
            const uint32_t sourceStrideBlocks = static_cast<uint32_t>(
                (paddedRowSpan * sizeof(T) - consumedLocalBytes) / kBlockBytes);
            AscendC::DataCopyExtParams outputParams{
                static_cast<uint16_t>(validWidth),
                static_cast<uint32_t>(outputBlockBytes), sourceStrideBlocks,
                static_cast<uint32_t>((data_.matrixH - validRows) * sizeof(T)), 0};
            AscendC::DataCopyPad(
                yGm_[matrixBase + column * data_.matrixH + row],
                output, outputParams);
            WaitMte3BeforeVectorReuse();
            outQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline void ProcessSwapInt8Blocks()
    {
        // Column ownership is the established race-free route when the
        // column grid already fills the device.  Each task retains every
        // 16-row segment of its output rows, so partial 16-byte stores never
        // overlap another AIV.
        const uint64_t tasksPerMatrix = data_.columnBatches;
        const uint64_t totalTasks = data_.outer * tasksPerMatrix;
        uint64_t taskStart = 0;
        uint64_t taskCount = 0;
        Partition(totalTasks, taskStart, taskCount);
        for (uint64_t task = taskStart; task < taskStart + taskCount; ++task) {
            const uint64_t outerIndex = task / tasksPerMatrix;
            const uint64_t columnBatch = task % tasksPerMatrix;
            const uint64_t firstColumnTile = columnBatch * data_.tileBatchCols;
            const uint64_t remainingTiles = data_.tileCols - firstColumnTile;
            const uint64_t currentTiles = remainingTiles < data_.tileBatchCols
                                              ? remainingTiles
                                              : data_.tileBatchCols;
            const uint64_t batchWidth = currentTiles * data_.microCols;
            const uint64_t column = firstColumnTile * data_.microCols;
            const uint64_t validWidth = data_.matrixW - column < batchWidth
                                            ? data_.matrixW - column
                                            : batchWidth;
            const uint64_t matrixBase = outerIndex * data_.matrixH * data_.matrixW;
            for (uint64_t rowTile = 0; rowTile < data_.tileRows; ++rowTile) {
                const uint64_t row = rowTile * kMicro;
                const uint64_t validRows = data_.matrixH - row < kMicro
                                               ? data_.matrixH - row
                                               : kMicro;
                const uint64_t inputOffset =
                    matrixBase + row * data_.matrixW + column;
                const uint64_t outputOffset =
                    matrixBase + column * data_.matrixH + row;

                AscendC::LocalTensor<T> input = inQueue_.AllocTensor<T>();
                AscendC::DataCopyExtParams inputParams{
                    static_cast<uint16_t>(validRows),
                    static_cast<uint32_t>(validWidth),
                    static_cast<uint32_t>(data_.matrixW - validWidth), 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams{
                    validWidth != batchWidth, 0,
                    static_cast<uint8_t>(batchWidth - validWidth), static_cast<T>(0)};
                AscendC::DataCopyPad(input, xGm_[inputOffset], inputParams, padParams);
                inQueue_.EnQue(input);

                input = inQueue_.DeQue<T>();
                AscendC::LocalTensor<T> output = outQueue_.AllocTensor<T>();
                uint64_t sourceList[kMicro];
                uint64_t destinationList[kMicro];
                for (uint64_t i = 0; i < kMicro; ++i) {
                    sourceList[i] = reinterpret_cast<uint64_t>(
                        input[i * batchWidth].GetPhyAddr());
                    destinationList[i] = reinterpret_cast<uint64_t>(
                        output[i * kBlockBytes].GetPhyAddr());
                }
                const uint16_t destinationRepeatStride = currentTiles == 1 ? 0 : 32;
                const uint16_t sourceRepeatStride = currentTiles == 1 ? 0 : 1;
                AscendC::TransDataTo5HDParams lowParams{
                    false, false, static_cast<uint8_t>(currentTiles),
                    destinationRepeatStride, sourceRepeatStride};
                AscendC::TransDataTo5HD<T>(destinationList, sourceList, lowParams);
                AscendC::PipeBarrier<PIPE_V>();

                for (uint64_t i = 0; i < kMicro; ++i) {
                    destinationList[i] = reinterpret_cast<uint64_t>(
                        output[(kMicro + i) * kBlockBytes].GetPhyAddr());
                }
                AscendC::TransDataTo5HDParams highParams{
                    false, true, static_cast<uint8_t>(currentTiles),
                    destinationRepeatStride, sourceRepeatStride};
                AscendC::TransDataTo5HD<T>(destinationList, sourceList, highParams);
                AscendC::PipeBarrier<PIPE_V>();
                outQueue_.EnQue(output);
                inQueue_.FreeTensor(input);

                output = outQueue_.DeQue<T>();
                AscendC::DataCopyExtParams outputParams{
                    static_cast<uint16_t>(validWidth),
                    static_cast<uint32_t>(validRows), 0,
                    static_cast<uint32_t>(data_.matrixH - validRows), 0};
                AscendC::DataCopyPad(yGm_[outputOffset], output, outputParams);
                AscendC::PipeBarrier<PIPE_MTE3>();
                outQueue_.FreeTensor(output);
            }
        }
    }

    __aicore__ inline void ProcessSwapInt8RowPairs()
    {
        // TransDataTo5HD consumes 16 source datablocks.  For INT8, each
        // datablock holds 32 columns and one instruction emits sixteen
        // 16-byte half-blocks.  Keep every output row padded to one 32-byte
        // UB block, run the low and high source halves separately, then let
        // one strided MTE3 copy compact the 32x16 result into GM.  The rule is
        // purely dtype/datablock based and is valid for every Ascend 910B
        // family revision supported by the generic CANN target.
        // A pair owns a complete aligned 32-byte segment of every affected
        // output row.  Row-pair tasks are therefore disjoint and can safely
        // fill all AIVs when the older column-owned grid underfills them.
        const uint64_t rowPairs = (data_.tileRows + 1) / 2;
        const uint64_t tasksPerMatrix = rowPairs * data_.columnBatches;
        const uint64_t totalTasks = data_.outer * tasksPerMatrix;
        uint64_t taskStart = 0;
        uint64_t taskCount = 0;
        Partition(totalTasks, taskStart, taskCount);
        for (uint64_t task = taskStart; task < taskStart + taskCount; ++task) {
            const uint64_t outerIndex = task / tasksPerMatrix;
            const uint64_t matrixTask = task % tasksPerMatrix;
            const uint64_t rowPair = matrixTask / data_.columnBatches;
            const uint64_t columnBatch = matrixTask % data_.columnBatches;
            const uint64_t firstColumnTile = columnBatch * data_.tileBatchCols;
            const uint64_t remainingTiles = data_.tileCols - firstColumnTile;
            const uint64_t currentTiles = remainingTiles < data_.tileBatchCols
                                              ? remainingTiles
                                              : data_.tileBatchCols;
            const uint64_t batchWidth = currentTiles * data_.microCols;
            const uint64_t column = firstColumnTile * data_.microCols;
            const uint64_t validWidth = data_.matrixW - column < batchWidth
                                            ? data_.matrixW - column
                                            : batchWidth;
            const uint64_t matrixBase = outerIndex * data_.matrixH * data_.matrixW;
            // Pair adjacent 16-row tiles.  INT8 TransData can write either
            // half of each 32-byte destination datablock: the first row tile
            // fills the low half and the second fills the high half.  The
            // paired result is therefore one naturally aligned 32-byte span
            // per output row, so MTE2 and MTE3 each issue one command for 32
            // rows instead of two commands carrying 16-byte partial blocks.
            // A final odd/tail tile keeps the same layout and copies only its
            // valid low-half bytes.
            {
                const uint64_t rowTile = rowPair * 2;
                const uint64_t row = rowTile * kMicro;
                const uint64_t remainingRows = data_.matrixH - row;
                const uint64_t validRows = remainingRows < 2 * kMicro
                                               ? remainingRows
                                               : 2 * kMicro;
                const uint64_t inputOffset =
                    matrixBase + row * data_.matrixW + column;
                const uint64_t outputOffset =
                    matrixBase + column * data_.matrixH + row;

                AscendC::LocalTensor<T> input = inQueue_.AllocTensor<T>();
                AscendC::DataCopyExtParams inputParams{
                    static_cast<uint16_t>(validRows),
                    static_cast<uint32_t>(validWidth),
                    static_cast<uint32_t>(data_.matrixW - validWidth), 0, 0};
                AscendC::DataCopyPadExtParams<T> padParams{
                    validWidth != batchWidth, 0,
                    static_cast<uint8_t>(batchWidth - validWidth), static_cast<T>(0)};
                AscendC::DataCopyPad(input, xGm_[inputOffset], inputParams, padParams);
                inQueue_.EnQue(input);

                input = inQueue_.DeQue<T>();
                AscendC::LocalTensor<T> output = outQueue_.AllocTensor<T>();
                uint64_t sourceList[kMicro];
                uint64_t destinationList[kMicro];
                const uint16_t destinationRepeatStride = currentTiles == 1 ? 0 : 32;
                const uint16_t sourceRepeatStride = currentTiles == 1 ? 0 : 1;
                const uint64_t rowHalves = validRows > kMicro ? 2 : 1;
                for (uint64_t rowHalf = 0; rowHalf < rowHalves; ++rowHalf) {
                    for (uint64_t i = 0; i < kMicro; ++i) {
                        sourceList[i] = reinterpret_cast<uint64_t>(
                            input[(rowHalf * kMicro + i) * batchWidth].GetPhyAddr());
                        destinationList[i] = reinterpret_cast<uint64_t>(
                            output[i * kBlockBytes].GetPhyAddr());
                    }
                    AscendC::TransDataTo5HDParams lowParams{
                        rowHalf != 0, false, static_cast<uint8_t>(currentTiles),
                        destinationRepeatStride, sourceRepeatStride};
                    AscendC::TransDataTo5HD<T>(destinationList, sourceList, lowParams);
                    AscendC::PipeBarrier<PIPE_V>();

                    for (uint64_t i = 0; i < kMicro; ++i) {
                        destinationList[i] = reinterpret_cast<uint64_t>(
                            output[(kMicro + i) * kBlockBytes].GetPhyAddr());
                    }
                    AscendC::TransDataTo5HDParams highParams{
                        rowHalf != 0, true, static_cast<uint8_t>(currentTiles),
                        destinationRepeatStride, sourceRepeatStride};
                    AscendC::TransDataTo5HD<T>(destinationList, sourceList, highParams);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                outQueue_.EnQue(output);
                inQueue_.FreeTensor(input);

                output = outQueue_.DeQue<T>();
                AscendC::DataCopyExtParams outputParams{
                    static_cast<uint16_t>(validWidth),
                    static_cast<uint32_t>(validRows), 0,
                    static_cast<uint32_t>(data_.matrixH - validRows), 0};
                AscendC::DataCopyPad(yGm_[outputOffset], output, outputParams);
                // The next row tile reuses this single UB slot.  Waiting for
                // MTE3 here prevents a late partial-row store from reading
                // bytes already overwritten by the following TransData.
                AscendC::PipeBarrier<PIPE_MTE3>();
                outQueue_.FreeTensor(output);
            }
        }
    }

    __aicore__ inline void ProcessSwapWholeWidthRows()
    {
        // For a whole-width batch, adjacent 16-row tiles have contiguous
        // input rectangles and contiguous output column segments.  Grouping
        // them changes N independent MTE2 + N independent MTE3 commands into
        // one MTE2 + one MTE3 while retaining N TransData operations.  The
        // transformation is purely a change of UB layout:
        //   input  [row_tile, row, column]
        //   output [column, row_tile, row]
        // so every output element keeps the exact same source coordinate.
        const uint64_t rowTilesPerMatrix = data_.tileRows;
        constexpr uint64_t kTargetGroupTiles = 2;
        const uint64_t groupsPerMatrix =
            (rowTilesPerMatrix + kTargetGroupTiles - 1) / kTargetGroupTiles;
        const uint64_t totalGroups = data_.outer * groupsPerMatrix;
        uint64_t groupTaskStart = 0;
        uint64_t groupTaskCount = 0;
        Partition(totalGroups, groupTaskStart, groupTaskCount);

        const uint64_t paddedWidth = data_.tileCols * data_.microCols;
        const uint64_t elementsPerRowTile = kMicro * paddedWidth;
        const uint64_t maxGroupTiles = data_.tileElements / elementsPerRowTile;
        if (maxGroupTiles < kTargetGroupTiles) {
            ProcessSwapBlocks();
            return;
        }

        const uint64_t groupTaskEnd = groupTaskStart + groupTaskCount;
        for (uint64_t groupTask = groupTaskStart;
             groupTask < groupTaskEnd; ++groupTask) {
            const uint64_t outerIndex = groupTask / groupsPerMatrix;
            const uint64_t groupInMatrix = groupTask % groupsPerMatrix;
            const uint64_t firstRowTile = groupInMatrix * kTargetGroupTiles;
            const uint64_t remainingTiles = rowTilesPerMatrix - firstRowTile;
            const uint64_t groupTiles = remainingTiles < kTargetGroupTiles
                ? remainingTiles : kTargetGroupTiles;
            const uint64_t row = firstRowTile * kMicro;
            const uint64_t rowSpan = groupTiles * kMicro;
            const uint64_t matrixBase = outerIndex * data_.matrixH * data_.matrixW;

            AscendC::LocalTensor<T> input = inQueue_.AllocTensor<T>();
            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(rowSpan),
                static_cast<uint32_t>(data_.matrixW * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T> padParams{
                paddedWidth != data_.matrixW, 0,
                static_cast<uint8_t>(paddedWidth - data_.matrixW), static_cast<T>(0)};
            AscendC::DataCopyPad(
                input, xGm_[matrixBase + row * data_.matrixW], inputParams, padParams);
            inQueue_.EnQue(input);

            input = inQueue_.DeQue<T>();
            AscendC::LocalTensor<T> output = outQueue_.AllocTensor<T>();
            uint64_t sourceList[kMicro];
            uint64_t destinationList[kMicro];
            const uint64_t groups = kMicro / data_.microCols;
            for (uint64_t rowGroup = 0; rowGroup < groupTiles; ++rowGroup) {
                for (uint64_t i = 0; i < kMicro; ++i) {
                    sourceList[i] = reinterpret_cast<uint64_t>(
                        input[(rowGroup * kMicro + i) * paddedWidth].GetPhyAddr());
                }
                for (uint64_t columnInBlock = 0;
                     columnInBlock < data_.microCols; ++columnInBlock) {
                    for (uint64_t group = 0; group < groups; ++group) {
                        const uint64_t listIndex = columnInBlock * groups + group;
                        const uint64_t outputElement = columnInBlock * rowSpan +
                            rowGroup * kMicro + group * data_.microCols;
                        destinationList[listIndex] = reinterpret_cast<uint64_t>(
                            output[outputElement].GetPhyAddr());
                    }
                }
                const uint16_t destinationRepeatStride =
                    data_.tileCols == 1 ? 0 : static_cast<uint16_t>(rowSpan);
                const uint16_t sourceRepeatStride = data_.tileCols == 1 ? 0 : 1;
                AscendC::TransDataTo5HDParams transParams{
                    false, false, static_cast<uint8_t>(data_.tileCols),
                    destinationRepeatStride, sourceRepeatStride};
                AscendC::TransDataTo5HD<T>(destinationList, sourceList, transParams);
                AscendC::PipeBarrier<PIPE_V>();
            }
            outQueue_.EnQue(output);
            WaitVectorBeforeMte2Reuse();
            inQueue_.FreeTensor(input);

            output = outQueue_.DeQue<T>();
            AscendC::DataCopyExtParams outputParams{
                static_cast<uint16_t>(data_.matrixW),
                static_cast<uint32_t>(rowSpan * sizeof(T)), 0,
                static_cast<uint32_t>((data_.matrixH - rowSpan) * sizeof(T)), 0};
            AscendC::DataCopyPad(yGm_[matrixBase + row], output, outputParams);
            WaitMte3BeforeVectorReuse();
            outQueue_.FreeTensor(output);
        }
    }

    __aicore__ inline void ProcessPermutedMatrixInt8()
    {
        const uint32_t rowAxis = data_.perm[data_.rank - 1];
        const uint64_t rowStride = data_.inputStride[rowAxis];
        uint64_t columnStride = 1;
        for (int32_t axis = data_.rank - 1; axis >= 0; --axis) {
            if (data_.perm[axis] == static_cast<int32_t>(data_.rank - 1)) { break; }
            columnStride *= data_.outputShape[axis];
        }
        // All sixteen-byte pieces of a GM block belong to one AIV.
        const uint64_t rowQuantum = data_.rowOwnershipQuantum;
        const uint64_t tasks = (data_.outer / rowQuantum) * data_.columnBatches;
        uint64_t start = 0, count = 0;
        Partition(tasks, start, count);
        for (uint64_t task = start; task < start + count; ++task) {
            const uint64_t firstOuter = (task / data_.columnBatches) * rowQuantum;
            const uint64_t batch = task % data_.columnBatches;
            const uint64_t firstTile = batch * data_.tileBatchCols;
            const uint64_t tiles = data_.tileCols - firstTile < data_.tileBatchCols
                ? data_.tileCols - firstTile : data_.tileBatchCols;
            const uint64_t width = tiles * 16;
            const uint64_t column = firstTile * 16;
            const uint64_t validWidth = data_.matrixW - column < width ? data_.matrixW - column : width;
            for (uint64_t o = firstOuter; o < firstOuter + rowQuantum; ++o) {
                uint64_t outer = o, inputBase = 0, outputBase = 0, outputStride = 1;
                for (int32_t axis = data_.rank - 1; axis >= 0; --axis) {
                    const uint32_t inputAxis = data_.perm[axis];
                    if (inputAxis != rowAxis && inputAxis != data_.rank - 1) {
                        const uint64_t coordinate = outer % data_.outputShape[axis];
                        outer /= data_.outputShape[axis];
                        inputBase += coordinate * data_.inputStride[inputAxis];
                        outputBase += coordinate * outputStride;
                    }
                    outputStride *= data_.outputShape[axis];
                }
                for (uint64_t row = 0; row < data_.matrixH; row += 16) {
                    const uint64_t validRows = data_.matrixH - row < 16 ? data_.matrixH - row : 16;
                    AscendC::LocalTensor<T> raw = inQueue_.AllocTensor<T>();
                    // Raw INT8 rows need 32-byte alignment even though a
                    // sixteen-column TransData micro-tile is only 16 bytes.
                    const uint64_t rawWidth = (width + 31) / 32 * 32;
                    AscendC::DataCopyExtParams inputParams{static_cast<uint16_t>(validRows), static_cast<uint32_t>(validWidth),
                        static_cast<uint32_t>(rowStride - validWidth), 0, 0};
                    AscendC::DataCopyPadExtParams<T> pad{true,0,static_cast<uint8_t>(rawWidth-validWidth),0};
                    AscendC::DataCopyPad(raw, xGm_[inputBase + row * rowStride + column], inputParams, pad);
                    inQueue_.EnQue(raw); raw = inQueue_.DeQue<T>();
                    AscendC::LocalTensor<half> input = matrixInputHalf_.Get<half>();
                    AscendC::LocalTensor<half> output = matrixOutputHalf_.Get<half>();
                    AscendC::Cast(input, raw, AscendC::RoundMode::CAST_NONE, 16 * rawWidth);
                    AscendC::Duplicate(output, static_cast<half>(0), width * 32);
                    AscendC::PipeBarrier<PIPE_V>();
                    uint64_t sources[16], destinations[16];
                    for (uint32_t i = 0; i < 16; ++i) {
                        sources[i] = reinterpret_cast<uint64_t>(input[i * rawWidth].GetPhyAddr());
                        destinations[i] = reinterpret_cast<uint64_t>(output[i * 32].GetPhyAddr());
                    }
                    AscendC::TransDataTo5HDParams params{false,false,static_cast<uint8_t>(tiles),
                        static_cast<uint16_t>(tiles == 1 ? 0 : 32), static_cast<uint16_t>(tiles == 1 ? 0 : 1)};
                    AscendC::TransDataTo5HD<half>(destinations, sources, params);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::LocalTensor<T> packed = outQueue_.AllocTensor<T>();
                    AscendC::Cast(packed, output, AscendC::RoundMode::CAST_NONE, width * 32);
                    AscendC::PipeBarrier<PIPE_V>();
                    outQueue_.EnQue(packed);
                    WaitVectorBeforeMte2Reuse(); inQueue_.FreeTensor(raw);
                    packed = outQueue_.DeQue<T>();
                    AscendC::DataCopyExtParams outputParams{static_cast<uint16_t>(validWidth),static_cast<uint32_t>(validRows),0,
                        static_cast<uint32_t>(columnStride - validRows),0};
                    AscendC::DataCopyPad(yGm_[outputBase + column * columnStride + row], packed, outputParams);
                    WaitMte3BeforeVectorReuse(); outQueue_.FreeTensor(packed);
                }
            }
        }
    }

    __aicore__ inline void ProcessPermutedMatrix()
    {
        const uint32_t rowAxis = data_.perm[data_.rank - 1];
        const uint64_t rowStride = data_.inputStride[rowAxis];
        uint64_t columnStride = 1;
        for (int32_t outAxis = data_.rank - 1; outAxis >= 0; --outAxis) {
            if (data_.perm[outAxis] == static_cast<int32_t>(data_.rank - 1)) { break; }
            columnStride *= data_.outputShape[outAxis];
        }
        const uint64_t tasksPerMatrix = data_.tileRows * data_.columnBatches;
        const uint64_t quantum = data_.rowOwnershipQuantum;
        const uint64_t totalTasks = quantum == 1 ? data_.outer * tasksPerMatrix
            : (data_.outer / quantum) * data_.columnBatches;
        uint64_t taskStart = 0;
        uint64_t taskCount = 0;
        Partition(totalTasks, taskStart, taskCount);

        for (uint64_t task = taskStart; task < taskStart + taskCount; ++task) {
            const uint64_t ownedTiles = quantum == 1 ? 1 : quantum * data_.tileRows;
            for (uint64_t owned = 0; owned < ownedTiles; ++owned) {
            uint64_t outerIndex = quantum == 1 ? task / tasksPerMatrix
                : (task / data_.columnBatches) * quantum + owned / data_.tileRows;
            const uint64_t withinMatrix = quantum == 1 ? task % tasksPerMatrix
                : (owned % data_.tileRows) * data_.columnBatches + task % data_.columnBatches;
            const uint64_t row = (withinMatrix / data_.columnBatches) * kMicro;
            const uint64_t validRows = data_.matrixH - row < kMicro
                ? data_.matrixH - row : kMicro;
            const uint64_t columnBatch = withinMatrix % data_.columnBatches;
            uint64_t inputBase = 0, outputBase = 0, outputStride = 1;
            for (int32_t outAxis = data_.rank - 1; outAxis >= 0; --outAxis) {
                const uint32_t inputAxis = data_.perm[outAxis];
                if (inputAxis != rowAxis && inputAxis != data_.rank - 1) {
                    const uint64_t coordinate = outerIndex % data_.outputShape[outAxis];
                    outerIndex /= data_.outputShape[outAxis];
                    inputBase += coordinate * data_.inputStride[inputAxis];
                    outputBase += coordinate * outputStride;
                }
                outputStride *= data_.outputShape[outAxis];
            }
            const uint64_t firstColumnTile =
                columnBatch * data_.tileBatchCols;
            const uint64_t remainingTiles =
                data_.tileCols - firstColumnTile;
            const uint64_t currentTiles =
                remainingTiles < data_.tileBatchCols
                    ? remainingTiles : data_.tileBatchCols;
            const uint64_t batchWidth = currentTiles * data_.microCols;
            const uint64_t column = firstColumnTile * data_.microCols;
            const uint64_t validWidth = data_.matrixW - column < batchWidth
                ? data_.matrixW - column : batchWidth;

            const uint64_t inputOffset = inputBase + row * rowStride + column;
            AscendC::LocalTensor<T> input = inQueue_.AllocTensor<T>();
            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(validRows),
                static_cast<uint32_t>(validWidth * sizeof(T)),
                static_cast<uint32_t>((rowStride - validWidth) * sizeof(T)),
                0, 0};
            AscendC::DataCopyPadExtParams<T> inputPad{
                validWidth != batchWidth, 0,
                static_cast<uint8_t>(batchWidth - validWidth),
                static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm_[inputOffset], inputParams, inputPad);
            inQueue_.EnQue(input);

            input = inQueue_.DeQue<T>();
            AscendC::LocalTensor<T> output = outQueue_.AllocTensor<T>();
            uint64_t sourceList[kMicro];
            uint64_t destinationList[kMicro];
            for (uint64_t i = 0; i < kMicro; ++i) {
                sourceList[i] = reinterpret_cast<uint64_t>(
                    input[i * batchWidth].GetPhyAddr());
            }
            const uint64_t groups = kMicro / data_.microCols;
            for (uint64_t columnInBlock = 0;
                 columnInBlock < data_.microCols; ++columnInBlock) {
                for (uint64_t group = 0; group < groups; ++group) {
                    const uint64_t listIndex =
                        columnInBlock * groups + group;
                    const uint64_t outputElement =
                        columnInBlock * kMicro + group * data_.microCols;
                    destinationList[listIndex] = reinterpret_cast<uint64_t>(
                        output[outputElement].GetPhyAddr());
                }
            }
            const uint16_t destinationRepeatStride = currentTiles == 1
                ? 0 : static_cast<uint16_t>(kMicro);
            const uint16_t sourceRepeatStride = currentTiles == 1 ? 0 : 1;
            AscendC::TransDataTo5HDParams transParams{
                false, false, static_cast<uint8_t>(currentTiles),
                destinationRepeatStride, sourceRepeatStride};
            AscendC::TransDataTo5HD<T>(
                destinationList, sourceList, transParams);
            AscendC::PipeBarrier<PIPE_V>();
            outQueue_.EnQue(output);
            WaitVectorBeforeMte2Reuse();
            inQueue_.FreeTensor(input);

            output = outQueue_.DeQue<T>();
            const uint64_t outputOffset =
                outputBase + column * columnStride + row;
            AscendC::DataCopyExtParams outputParams{
                static_cast<uint16_t>(validWidth),
                static_cast<uint32_t>(validRows * sizeof(T)),
                static_cast<uint32_t>((kMicro - validRows) * sizeof(T) / kBlockBytes),
                static_cast<uint32_t>((columnStride - validRows) * sizeof(T)),
                0};
            AscendC::DataCopyPad(yGm_[outputOffset], output, outputParams);
            WaitMte3BeforeVectorReuse();
            outQueue_.FreeTensor(output);
            }
        }
    }

    __aicore__ inline void InitInputCursor(
        uint64_t outputLinear, uint64_t (&coordinates)[kMaxRank],
        uint64_t& inputOffset) const
    {
        uint64_t remaining = outputLinear;
        inputOffset = 0;
        for (int32_t outputAxis = static_cast<int32_t>(data_.rank) - 1;
             outputAxis >= 0; --outputAxis) {
            const uint64_t extent = data_.outputShape[outputAxis];
            const uint64_t coordinate = remaining % extent;
            remaining /= extent;
            coordinates[outputAxis] = coordinate;
            inputOffset +=
                coordinate * data_.inputStride[data_.perm[outputAxis]];
        }
    }

    __aicore__ inline void AdvanceInputCursor(
        uint64_t (&coordinates)[kMaxRank], uint64_t& inputOffset) const
    {
        // Output-owned elements are visited in row-major order.  Updating a
        // mixed-radix coordinate cursor replaces rank divisions/modulos for
        // every element with one add and a usually single-iteration carry.
        for (int32_t outputAxis = static_cast<int32_t>(data_.rank) - 1;
             outputAxis >= 0; --outputAxis) {
            const uint64_t extent = data_.outputShape[outputAxis];
            const uint64_t inputStride =
                data_.inputStride[data_.perm[outputAxis]];
            ++coordinates[outputAxis];
            inputOffset += inputStride;
            if (coordinates[outputAxis] < extent) {
                return;
            }
            coordinates[outputAxis] = 0;
            inputOffset -= extent * inputStride;
        }
    }

    __aicore__ inline void ProcessGeneric()
    {
        uint64_t start = 0;
        uint64_t count = 0;
        // Align ownership boundaries to one GM datablock.  The scalar gather
        // is staged in UB and MTE3 writes each owned interval, so no two AIVs
        // can perform partial updates to the same 32-byte output block.
        PartitionAlignedElements(data_.totalElements, start, count);
        if (count == 0) {
            return;
        }
        uint64_t coordinates[kMaxRank] = {};
        uint64_t inputOffset = 0;
        InitInputCursor(start, coordinates, inputOffset);
        // GlobalTensor::SetValue writes through each core's DCache.  Besides
        // needing an explicit DCCI, adjacent cores can then dirty the same
        // 64-byte cache line.  Stage each output-owned interval in UB and use
        // MTE3 so non-aligned tails remain visible and race-free.
        for (uint64_t begin = start; begin < start + count; begin += data_.tileElements) {
            const uint64_t once = start + count - begin < data_.tileElements
                                      ? start + count - begin
                                      : data_.tileElements;
            AscendC::LocalTensor<T> outputLocal = outQueue_.AllocTensor<T>();
            for (uint64_t local = 0; local < once; ++local) {
                outputLocal.SetValue(local, xGm_.GetValue(inputOffset));
                AdvanceInputCursor(coordinates, inputOffset);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            outQueue_.EnQue(outputLocal);
            outputLocal = outQueue_.DeQue<T>();
            AscendC::DataCopyExtParams outputParams{
                1, static_cast<uint32_t>(once * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[begin], outputLocal, outputParams);
            WaitMte3BeforeVectorReuse();
            outQueue_.FreeTensor(outputLocal);
        }
    }

private:
    TransposeTilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT,
                      kCopyBufferNum> copyQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixInputHalf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> matrixOutputHalf_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
};

template <typename T>
class KernelTransposeGenericDma {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeTilingData& tiling)
    {
        data_ = tiling;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), data_.totalElements);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), data_.totalElements);
        pipe_.InitBuffer(inQueue_, kBufferNum, kGenericRows * kBlockBytes);
        pipe_.InitBuffer(outQueue_, kBufferNum, kGenericRows * sizeof(T));
        pipe_.InitBuffer(genericIndex_, kGenericRows * sizeof(uint32_t));
        if constexpr (sizeof(T) == 1) {
            pipe_.InitBuffer(genericInputHalf_, kGenericRows * kBlockBytes * sizeof(half));
            pipe_.InitBuffer(genericOutputHalf_, kGenericRows * sizeof(half));
        }
    }
    __aicore__ inline void Process() { ProcessGenericStridedDma(); }
private:
    static constexpr uint32_t kGenericRows = 256;
    __aicore__ inline void Partition(uint64_t total, uint64_t& start, uint64_t& count) const
    {
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t small = total / blocks;
        const uint64_t remainder = total % blocks;
        count = small + (block < remainder ? 1 : 0);
        start = block * small + (block < remainder ? block : remainder);
    }
    __aicore__ inline void PartitionAlignedElements(
        uint64_t total, uint64_t& start, uint64_t& count) const
    {
        const uint64_t blockElements = kBlockBytes / sizeof(T);
        const uint64_t units = (total + blockElements - 1) / blockElements;
        uint64_t unitStart = 0;
        uint64_t unitCount = 0;
        Partition(units, unitStart, unitCount);
        start = unitStart * blockElements;
        const uint64_t end = (unitStart + unitCount) * blockElements < total
                                 ? (unitStart + unitCount) * blockElements
                                 : total;
        count = start < total ? end - start : 0;
    }
    __aicore__ inline uint64_t InputOffset(uint64_t outputLinear) const
    {
        uint64_t remaining = outputLinear;
        uint64_t offset = 0;
        for (int32_t outputAxis = static_cast<int32_t>(data_.rank) - 1; outputAxis >= 0; --outputAxis) {
            const uint64_t extent = data_.outputShape[outputAxis];
            const uint64_t coordinate = remaining % extent;
            remaining /= extent;
            offset += coordinate * data_.inputStride[data_.perm[outputAxis]];
        }
        return offset;
    }
    __aicore__ inline void WaitVectorBeforeMte2Reuse() const
    {
        const event_t event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(event);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(event);
    }
    __aicore__ inline void WaitMte3BeforeVectorReuse() const
    {
        const event_t event = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(event);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(event);
    }
    __aicore__ inline void ProcessGenericStridedDma()
    {
        uint64_t start = 0, count = 0;
        PartitionAlignedElements(data_.totalElements, start, count);
        if (count == 0) { return; }
        const uint64_t width = data_.outputShape[data_.rank - 1];
        const uint64_t stride = data_.inputStride[data_.perm[data_.rank - 1]];
        AscendC::LocalTensor<uint32_t> indices = genericIndex_.Get<uint32_t>();
        for (uint32_t i = 0; i < kGenericRows; ++i) {
            indices.SetValue(i, i * kBlockBytes * (sizeof(T) == 1 ? sizeof(half) : 1));
        }
        const event_t indicesReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
        AscendC::SetFlag<AscendC::HardEvent::S_V>(indicesReady);
        AscendC::WaitFlag<AscendC::HardEvent::S_V>(indicesReady);
        const uint64_t end = start + count;
        // Decode once per AIV, then advance the mixed-radix cursor. This
        // removes rank many dynamic 64-bit divisions from every DMA batch.
        uint64_t coordinates[kMaxRank] = {};
        uint64_t remaining = start, inputOffset = 0;
        for (int32_t axis = static_cast<int32_t>(data_.rank) - 1; axis >= 0; --axis) {
            coordinates[axis] = remaining % data_.outputShape[axis];
            remaining /= data_.outputShape[axis];
            inputOffset += coordinates[axis] * data_.inputStride[data_.perm[axis]];
        }
        uint64_t rowPosition = coordinates[data_.rank - 1];
        for (uint64_t begin = start; begin < end;) {
            uint64_t rows = width - rowPosition;
            if (rows > kGenericRows) { rows = kGenericRows; }
            if (rows > end - begin) { rows = end - begin; }
            AscendC::LocalTensor<T> input = inQueue_.AllocTensor<T>();
            if constexpr (sizeof(T) == 1) {
                // The INT8 Cast consumes 128 raw lanes. Clear only the
                // padded cast span, then let MTE2 fill its valid row slots.
                const uint32_t rawCount = static_cast<uint32_t>((rows * kBlockBytes + 127) / 128 * 128);
                auto words = input.template ReinterpretCast<uint16_t>();
                AscendC::Duplicate(words, static_cast<uint16_t>(0), rawCount / 2);
                WaitVectorBeforeMte2Reuse();
            }
            AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(sizeof(T)),
                static_cast<uint32_t>((stride - 1) * sizeof(T)), 0, 0};
            AscendC::DataCopyPadExtParams<T> pad{
                true, 0, static_cast<uint8_t>(kBlockBytes / sizeof(T) - 1), static_cast<T>(0)};
            AscendC::DataCopyPad(input, xGm_[inputOffset], inputParams, pad);
            inQueue_.EnQue(input);
            input = inQueue_.DeQue<T>();
            AscendC::LocalTensor<T> output = outQueue_.AllocTensor<T>();
            if constexpr (sizeof(T) == 1) {
                const uint32_t rawCount = static_cast<uint32_t>((rows * kBlockBytes + 127) / 128 * 128);
                const uint32_t outCount = static_cast<uint32_t>((rows + 127) / 128 * 128);
                auto wideInput = genericInputHalf_.Get<half>();
                auto wideOutput = genericOutputHalf_.Get<half>();
                AscendC::Cast(wideInput, input, AscendC::RoundMode::CAST_NONE, rawCount);
                AscendC::Duplicate(wideOutput, static_cast<half>(0), outCount);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Gather(wideOutput, wideInput, indices, 0, static_cast<uint32_t>(rows));
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(output, wideOutput, AscendC::RoundMode::CAST_NONE, outCount);
            } else if constexpr (sizeof(T) == 2) {
                auto sourceBits = input.template ReinterpretCast<uint16_t>();
                auto outputBits = output.template ReinterpretCast<uint16_t>();
                AscendC::Gather(outputBits, sourceBits, indices, 0, static_cast<uint32_t>(rows));
            } else {
                auto sourceBits = input.template ReinterpretCast<uint32_t>();
                auto outputBits = output.template ReinterpretCast<uint32_t>();
                AscendC::Gather(outputBits, sourceBits, indices, 0, static_cast<uint32_t>(rows));
            }
            AscendC::PipeBarrier<PIPE_V>();
            outQueue_.EnQue(output);
            // FreeTensor retains the queue slot until its vector use ends.
            inQueue_.FreeTensor(input);
            output = outQueue_.DeQue<T>();
            AscendC::DataCopyExtParams outputParams{
                1, static_cast<uint32_t>(rows * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(yGm_[begin], output, outputParams);
            // VECOUT queue reuse already waits for this slot's MTE3 completion.
            outQueue_.FreeTensor(output);
            begin += rows;
            rowPosition += rows;
            inputOffset += rows * stride;
            if (rowPosition == width) {
                rowPosition = 0;
                inputOffset -= width * stride;
                for (int32_t axis = static_cast<int32_t>(data_.rank) - 2; axis >= 0; --axis) {
                    const uint64_t axisStride = data_.inputStride[data_.perm[axis]];
                    inputOffset += axisStride;
                    if (++coordinates[axis] < data_.outputShape[axis]) break;
                    coordinates[axis] = 0;
                    inputOffset -= data_.outputShape[axis] * axisStride;
                }
            }
        }
    }
    TransposeTilingData data_{};
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, kBufferNum> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, kBufferNum> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> genericIndex_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> genericInputHalf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> genericOutputHalf_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
};
}  // namespace s9transpose

extern "C" __global__ __aicore__ void transpose(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(2, KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(s9transpose::TransposeTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2", s9transpose::TransposeStagesGatherTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 3", s9transpose::TransposeMacroTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 5", s9transpose::TransposeColumnOwnerTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 6", s9transpose::TransposeMacroTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 7", s9transpose::TransposeSuffixBlockTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 11", s9transpose::TransposeL2StagedTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 12", s9transpose::TransposeDirectMaskTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 16", s9transpose::TransposeRowAlignedTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 17", s9transpose::TransposeInt8PairsTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 18", s9transpose::TransposeDirectMaskTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 19", s9transpose::TransposeDirectMaskTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 28", s9transpose::TransposeRowAlignedTilingData);
    if(TILING_KEY_IS(28)){
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeColumnOwnerTilingData, owned, tiling);
        if constexpr(sizeof(DTYPE_X)==4){
            if((reinterpret_cast<uint64_t>(x)&31U)==0){s9transpose::KernelTransposeRowAligned<DTYPE_X,1> op;op.Init(x,y,owned,tiling+sizeof(owned));op.Process();}
            else{s9transpose::KernelTransposeColumnOwner<DTYPE_X> op;op.Init(x,y,owned);op.Process();}
        }return;
    }
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 23", s9transpose::TransposeDirectMaskTilingData);
    if (TILING_KEY_IS(18) || TILING_KEY_IS(19) || TILING_KEY_IS(23)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeDirectMaskHeader, packed, tiling);
        if constexpr(sizeof(DTYPE_X)==4){
            if((reinterpret_cast<uint64_t>(x)&31U)!=0){s9transpose::KernelTranspose<DTYPE_X> op;op.Init(x,y,packed.base);op.Process();}
            else if(TILING_KEY_IS(23)){s9transpose::KernelTransposeOwner64<DTYPE_X,true,12> op;op.Init(x,y,packed,tiling+sizeof(packed));op.Process();}
            else if(TILING_KEY_IS(19)){s9transpose::KernelTransposeDirectMaskUnroll<DTYPE_X,true> op;op.Init(x,y,packed,tiling+sizeof(packed));op.Process();}
            else {s9transpose::KernelTransposeDirectMask<DTYPE_X,true> op;op.Init(x,y,packed,tiling+sizeof(packed));op.Process();}
        }
        return;
    }
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 30", s9transpose::TransposeDirectMaskTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 31", s9transpose::TransposeDirectMaskTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 32", s9transpose::TransposeDirectMaskTilingData);
    if(TILING_KEY_IS(30)||TILING_KEY_IS(31)||TILING_KEY_IS(32)){
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeDirectMaskHeader, packed, tiling);
        if constexpr(sizeof(DTYPE_X)==4){
            if((reinterpret_cast<uint64_t>(x)&31U)!=0){s9transpose::KernelTranspose<DTYPE_X> op;op.Init(x,y,packed.base);op.Process();}
            else if(TILING_KEY_IS(30)){s9transpose::KernelTransposeNoCount<DTYPE_X,true,12,1> op;op.Init(x,y,packed,tiling+sizeof(packed));op.Process();}
            else if(TILING_KEY_IS(31)){s9transpose::KernelTransposeNoCount<DTYPE_X,true,12,3> op;op.Init(x,y,packed,tiling+sizeof(packed));op.Process();}
            else if(TILING_KEY_IS(32)){s9transpose::KernelTransposeNoCount<DTYPE_X,true,12,2> op;op.Init(x,y,packed,tiling+sizeof(packed));op.Process();}
        }return;
    }
    if (TILING_KEY_IS(17)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeInt8PairsTilingData, paired, tiling);
        s9transpose::KernelTransposeInt8Pairs op;op.Init(x,y,paired);op.Process();return;
    }
    GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeTilingData, tilingData, tiling);
    if (TILING_KEY_IS(16)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeColumnOwnerTilingData, owned, tiling);
        if constexpr(sizeof(DTYPE_X)==4){
            if((reinterpret_cast<uint64_t>(x)&31U)==0){s9transpose::KernelTransposeRowAligned<DTYPE_X> op;op.Init(x,y,owned,tiling+sizeof(s9transpose::TransposeColumnOwnerTilingData));op.Process();}
            else{s9transpose::KernelTransposeColumnOwner<DTYPE_X> op;op.Init(x,y,owned);op.Process();}
        }else{s9transpose::KernelTranspose<DTYPE_X> op;op.Init(x,y,tilingData);op.Process();}
    } else if (TILING_KEY_IS(12)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeDirectMaskHeader, packed, tiling);
        if ((reinterpret_cast<uint64_t>(x)&31U) != 0) {s9transpose::KernelTranspose<DTYPE_X> op;op.Init(x,y,tilingData);op.Process();}
        else {s9transpose::KernelTransposeDirectMask<DTYPE_X> op;op.Init(x,y,packed,tiling+sizeof(s9transpose::TransposeDirectMaskHeader));op.Process();}
    } else if (TILING_KEY_IS(11)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeL2StagedHeader, packed, tiling);
        if ((reinterpret_cast<uint64_t>(x)&31U) != 0) {s9transpose::KernelTranspose<DTYPE_X> op;op.Init(x,y,tilingData);op.Process();}
        else {AscendC::SetSysWorkspace(workspace);s9transpose::KernelTransposeL2Staged<DTYPE_X> op;op.Init(x,y,AscendC::GetUserWorkspace(workspace),packed,tiling+sizeof(s9transpose::TransposeL2StagedHeader));op.Process();}
    } else if (TILING_KEY_IS(7)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeSuffixBlockHeader, packed, tiling);
        s9transpose::KernelTransposeSuffixBlock<DTYPE_X> op;
        op.Init(x, y, packed, tiling + sizeof(s9transpose::TransposeSuffixBlockHeader)); op.Process();
    } else if (TILING_KEY_IS(6)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeMacroTilingData, macro, tiling);
        s9transpose::KernelTransposeMacro<DTYPE_X, 1> op;
        op.Init(x, y, macro); op.Process();
    } else if (TILING_KEY_IS(5)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeColumnOwnerTilingData, owned, tiling);
        s9transpose::KernelTransposeColumnOwner<DTYPE_X> op;
        op.Init(x, y, owned); op.Process();
    } else if (TILING_KEY_IS(4)) {
        s9transpose::KernelTransposeSuffixDma<DTYPE_X> op;
        op.Init(x, y, tilingData); op.Process();
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeMacroTilingData, macro, tiling);
        s9transpose::KernelTransposeMacro<DTYPE_X> op;
        op.Init(x, y, macro); op.Process();
    } else if (TILING_KEY_IS(1)) {
        s9transpose::KernelTransposeGenericDma<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(0)) {
        s9transpose::KernelTranspose<DTYPE_X> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(s9transpose::TransposeStagesTilingData, staged, tiling);
        AscendC::SetSysWorkspace(workspace);
        GM_ADDR temp = AscendC::GetUserWorkspace(workspace);
        GM_ADDR source = x;
        s9transpose::KernelTranspose<DTYPE_X, true> op;
        op.InitStages();
        AscendC::TBuf<AscendC::TPosition::VECCALC> batchIndices;
        GetTPipePtr()->InitBuffer(batchIndices, 8192 * sizeof(uint32_t));
        for (uint32_t i = 0; i < staged.stageCount; ++i) {
            const auto& specification = staged.stages[i];
            s9transpose::TransposeTilingData step{};
            step.mode = static_cast<uint32_t>(s9transpose::Mode::kSwapLastTwo);
            step.dtypeBytes = sizeof(DTYPE_X);
            step.fastSwap16 = 1;
            step.rowOwnershipQuantum = 2; // stage-local row grouping when the task grid remains full
            step.totalElements = staged.base.totalElements;
            step.outer = specification.outer;
            step.matrixH = specification.matrixH;
            step.matrixW = specification.matrixW;
            step.microCols = 32 / sizeof(DTYPE_X);
            step.tileRows = (step.matrixH + 15) / 16;
            // A column tile spans microCols=32/sizeof(T) output rows, so its
            // complete byte range is aligned even when one row is short.
            // Only boundaries between unaligned matrices need whole-matrix ownership.
            if (step.tileRows == 1 && step.matrixH * sizeof(DTYPE_X) < 32 &&
                step.outer > 1 && (step.matrixH * step.matrixW * sizeof(DTYPE_X)) % 32 != 0) {
                step.rowOwnershipQuantum = 3;
            }
            step.tileCols = (step.matrixW + step.microCols - 1) / step.microCols;
            step.tileBatchCols = specification.tileBatchCols;
            step.columnBatches = (step.tileCols + step.tileBatchCols - 1) / step.tileBatchCols;
            step.tileElements = 32768 / sizeof(DTYPE_X);
            GM_ADDR destination = (staged.stageCount - i) % 2 != 0 ? y : temp;
            op.ConfigureStage(source, destination, step);
            if (staged.reserved == i + 1) {
                op.ProcessBatchedMatrices(tiling + sizeof(s9transpose::TransposeStagesTilingData),
                    staged.gatherBatchMatrices, staged.gatherCount, batchIndices.Get<uint32_t>());
            } else if (staged.wideStage != 0 &&
                step.matrixH == staged.stages[staged.wideStage - 1].matrixH &&
                step.matrixH * step.matrixW > 2048) {
                op.ProcessWideMatrices(tiling + sizeof(s9transpose::TransposeStagesTilingData) + 8192 * sizeof(uint32_t),
                    staged.wideTileCols, staged.wideGatherCount, batchIndices.Get<uint32_t>());
            } else {
                op.Process();
            }
            AscendC::SyncAll<true>();
            source = destination;
        }
    }
}
