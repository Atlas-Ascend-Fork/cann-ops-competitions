/**
 * Concat 设备侧实现。
 * 按主机侧分块信息执行连续拷贝、行分块或 Gather 拼接；缓冲区复用遵循流水线事件依赖。
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../op_host/concat_tiling.h"

using namespace AscendC;
using PackedLongKey4Tiling = typename ConcatPackedKey4<DTYPE_INPUTS>::Type;

namespace {
template <typename T> struct ConcatUsePhasedCompact { static constexpr bool value = false; };
template <> struct ConcatUsePhasedCompact<int32_t> { static constexpr bool value = true; };
constexpr uint32_t kTileBytes = 64U * 1024U;
constexpr uint32_t kBufferNum = 2U;
constexpr uint32_t kPackTileBytes = 64U * 1024U;

template <typename T>
// 按连续片段或二维矩形搬运输入；队列负责相邻分块的流水线依赖。
class KernelConcat {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
                                const ConcatTilingData& tiling)
    {
        list_.Init(reinterpret_cast<__gm__ void*>(inputs));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        outer_ = tiling.outer;
        inner_ = tiling.inner;
        outputAxis_ = tiling.outputAxis;
        taskCount_ = tiling.taskCount;
        inputCount_ = tiling.inputCount;
        axis_ = tiling.axis;
        mode_ = tiling.mode;
        rowGroups_ = tiling.rowGroups;
        inputGroups_ = tiling.inputGroups;
        tileElements_ = kTileBytes / sizeof(T);
        pipe_.InitBuffer(copyQueue_, kBufferNum, kTileBytes);
    }

    __aicore__ inline void Process()
    {
        if (mode_ == 1) {
            ProcessColumnSplit();
        } else if (mode_ == 2) {
            ProcessLinearTasks();
        } else {
            ProcessGrid();
        }
    }

private:
    __aicore__ inline void ProcessPackedHalf(
        const ConcatManyTilingData& tiling)
    {
        if constexpr (sizeof(T) == 2) {
            const uint32_t block = GetBlockIdx();
            const uint64_t outputRowElements = outputAxis_ * inner_;
            const uint32_t rowGroup = block / inputGroups_;
            const uint64_t processRowStart = outer_ * rowGroup / rowGroups_;
            const uint64_t processRowEnd =
                outer_ * (rowGroup + 1U) / rowGroups_;
            const uint64_t processRows = processRowEnd - processRowStart;
            TensorDesc<T> desc;
            uint64_t shape[S9_CONCAT_MAX_RANK];
            desc.SetShapeAddr(shape);

            const uint16_t scheduleBegin = tiling.groupOffsets[block];
            const uint16_t scheduleEnd = tiling.groupOffsets[block + 1];
            for (uint16_t schedule = scheduleBegin;
                 schedule < scheduleEnd; ++schedule) {
                const uint16_t group = tiling.inputOrder[schedule];
                const uint16_t inputBegin = tiling.packGroupOffsets[group];
                const uint16_t inputEnd = tiling.packGroupOffsets[group + 1];
                const uint64_t groupPrefix = tiling.prefixes[inputBegin];

                for (int32_t inputIndex = static_cast<int32_t>(inputEnd) - 1;
                     inputIndex >= static_cast<int32_t>(inputBegin); --inputIndex) {
                    list_.GetDesc(desc, static_cast<uint32_t>(inputIndex));
                    const uint32_t segment = static_cast<uint32_t>(
                        desc.GetShape(axis_) * inner_);
                    if (segment == 0) {
                        continue;
                    }
                    input_.SetGlobalBuffer(desc.GetDataPtr());
                    const uint64_t prefix =
                        tiling.prefixes[static_cast<uint32_t>(inputIndex)];
                    if (prefix - groupPrefix < 15U) {
                        CopyRectangle(processRowStart * segment,
                                      processRowStart * outputRowElements + prefix,
                                      processRows, segment, segment,
                                      outputRowElements);
                        continue;
                    }

                    uint32_t divisor = segment;
                    uint32_t other = 16U;
                    while (other != 0U) {
                        const uint32_t remainder = divisor % other;
                        divisor = other;
                        other = remainder;
                    }
                    const uint32_t alignmentCycle = 16U / divisor;
                    uint32_t rowsPerBatch = kTileBytes /
                        (segment * static_cast<uint32_t>(sizeof(T)));
                    if (rowsPerBatch > 4095U) {
                        rowsPerBatch = 4095U;
                    }

                    for (uint64_t row = 0; row < processRows;
                         row += rowsPerBatch) {
                        const uint32_t batchRows = static_cast<uint32_t>(
                            (processRows - row) < rowsPerBatch ?
                                (processRows - row) : rowsPerBatch);
                        LocalTensor<T> compact = copyQueue_.AllocTensor<T>();
                        const DataCopyExtParams inputCopy(
                            1, batchRows * segment * sizeof(T), 0, 0, 0);
                        const DataCopyPadExtParams<T> noPad(
                            false, 0, 0, static_cast<T>(0));
                        DataCopyPad(compact,
                                    input_[(processRowStart + row) * segment],
                                    inputCopy, noPad);
                        copyQueue_.EnQue(compact);
                        compact = copyQueue_.DeQue<T>();

                        for (uint32_t rowClass = 0;
                             rowClass < alignmentCycle && rowClass < batchRows;
                             ++rowClass) {
                            const uint32_t compactOffset = rowClass * segment;
                            const uint32_t leading = compactOffset & 15U;
                            const uint32_t sourceOffset = compactOffset - leading;
                            const uint32_t blockBytes =
                                (leading + segment) * sizeof(T);
                            const uint32_t alignedBlockBytes =
                                (blockBytes + 31U) & ~31U;
                            const uint32_t sourceStepBytes =
                                alignmentCycle * segment * sizeof(T);
                            const uint32_t blockCount =
                                (batchRows - rowClass + alignmentCycle - 1U) /
                                    alignmentCycle;
                            const uint32_t sourceStrideBlocks =
                                (sourceStepBytes - alignedBlockBytes) / 32U;
                            const uint32_t outputStrideBytes = static_cast<uint32_t>(
                                alignmentCycle * outputRowElements * sizeof(T) -
                                blockBytes);
                            const uint64_t outputOffset =
                                (processRowStart + row + rowClass) *
                                    outputRowElements +
                                prefix - leading;
                            const DataCopyExtParams outputCopy(
                                blockCount, blockBytes, sourceStrideBlocks,
                                outputStrideBytes, 0);
                            DataCopyPad(output_[outputOffset],
                                        compact[sourceOffset], outputCopy);
                        }
                        copyQueue_.FreeTensor(compact);
                    }

                    if (alignmentCycle > 1U) {
                        PipeBarrier<PIPE_MTE3>();
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyIn(uint64_t inputOffset, uint32_t count)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        DataCopyPad(local, input_[inputOffset], copy, pad);
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset, uint32_t count)
    {
        LocalTensor<T> local = copyQueue_.DeQue<T>();
        const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
        DataCopyPad(output_[outputOffset], local, copy);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void CopyRectangleIn(uint64_t inputOffset,
                                            uint32_t rows,
                                            uint32_t columnBytes,
                                            uint32_t inputStrideBytes)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        const DataCopyExtParams copy(
            rows, columnBytes, inputStrideBytes, 0, 0);
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        DataCopyPad(local, input_[inputOffset], copy, pad);
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyRectangleOut(uint64_t outputOffset,
                                             uint32_t rows,
                                             uint32_t columnBytes,
                                             uint32_t outputStrideBytes)
    {
        LocalTensor<T> local = copyQueue_.DeQue<T>();
        const DataCopyExtParams copy(
            rows, columnBytes, 0, outputStrideBytes, 0);
        DataCopyPad(output_[outputOffset], local, copy);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void CopyRange(uint64_t inputOffset,
                                     uint64_t outputOffset,
                                     uint64_t count)
    {
        if (count == 0) {
            return;
        }
        uint64_t done = 0;
        uint32_t previousCount = static_cast<uint32_t>(
            count < tileElements_ ? count : tileElements_);
        CopyIn(inputOffset, previousCount);
        done += previousCount;
        uint64_t previousOutputOffset = outputOffset;
        while (done < count) {
            const uint32_t currentCount = static_cast<uint32_t>(
                (count - done) < tileElements_ ? (count - done) : tileElements_);
            CopyIn(inputOffset + done, currentCount);
            CopyOut(previousOutputOffset, previousCount);
            previousOutputOffset = outputOffset + done;
            previousCount = currentCount;
            done += currentCount;
        }
        CopyOut(previousOutputOffset, previousCount);
    }

    __aicore__ inline void CopyRectangle(uint64_t inputOffset,
                                         uint64_t outputOffset,
                                         uint64_t rows,
                                         uint64_t columns,
                                         uint64_t inputRowElements,
                                         uint64_t outputRowElements)
    {

        if (columns == 0 || rows == 0) {
            return;
        }
        if (columns > tileElements_ ||
            (inputRowElements - columns) * sizeof(T) > 0xffffffffULL ||
            (outputRowElements - columns) * sizeof(T) > 0xffffffffULL) {
            for (uint64_t row = 0; row < rows; ++row) {
                CopyRange(inputOffset + row * inputRowElements,
                          outputOffset + row * outputRowElements,
                          columns);
            }
            return;
        }

        const uint64_t columnBytes = columns * sizeof(T);
        const uint64_t alignedRowBytes = (columnBytes + 31ULL) & ~31ULL;
        uint64_t rowsPerBatch = kTileBytes / alignedRowBytes;

        if (rowsPerBatch > 4095ULL) {
            rowsPerBatch = 4095ULL;
        }
        const uint32_t columnBytes32 = static_cast<uint32_t>(columnBytes);
        const uint32_t inputStrideBytes = static_cast<uint32_t>(
            (inputRowElements - columns) * sizeof(T));
        const uint32_t outputStrideBytes = static_cast<uint32_t>(
            (outputRowElements - columns) * sizeof(T));
        uint64_t row = 0;
        uint64_t previousRow = 0;
        uint32_t previousRows = static_cast<uint32_t>(
            rows < rowsPerBatch ? rows : rowsPerBatch);
        CopyRectangleIn(inputOffset, previousRows, columnBytes32,
                        inputStrideBytes);
        row += previousRows;
        while (row < rows) {
            const uint32_t currentRows = static_cast<uint32_t>(
                (rows - row) < rowsPerBatch ? (rows - row) : rowsPerBatch);
            CopyRectangleIn(inputOffset + row * inputRowElements,
                            currentRows, columnBytes32, inputStrideBytes);
            CopyRectangleOut(outputOffset + previousRow * outputRowElements,
                             previousRows, columnBytes32, outputStrideBytes);
            previousRow = row;
            previousRows = currentRows;
            row += currentRows;
        }
        CopyRectangleOut(outputOffset + previousRow * outputRowElements,
                         previousRows, columnBytes32, outputStrideBytes);
    }

    __aicore__ inline void ProcessGrid()
    {
        const uint32_t block = GetBlockIdx();
        const uint32_t rowGroup = block / inputGroups_;
        const uint32_t inputGroup = block - rowGroup * inputGroups_;
        const uint64_t rowStart = outer_ * rowGroup / rowGroups_;
        const uint64_t rowEnd = outer_ * (rowGroup + 1) / rowGroups_;
        const uint64_t outputRowElements = outputAxis_ * inner_;
        constexpr uint64_t minimumElements = 16ULL * 1024ULL * 1024ULL / sizeof(T);
        const bool bypassInput = sizeof(T) == 4U && inputCount_ >= 2U && inputCount_ <= 8U &&
            outputRowElements != 0U && outer_ >= minimumElements / outputRowElements +
                static_cast<uint64_t>(minimumElements % outputRowElements != 0U);
        TensorDesc<T> desc;
        uint64_t shape[S9_CONCAT_MAX_RANK];
        desc.SetShapeAddr(shape);
        uint64_t axisPrefix = 0;
        for (uint32_t inputIndex = 0; inputIndex < inputCount_; ++inputIndex) {
            list_.GetDesc(desc, inputIndex);
            const uint64_t segment = desc.GetShape(axis_) * inner_;
            if (segment != 0 && inputIndex % inputGroups_ == inputGroup) {
                input_.SetGlobalBuffer(desc.GetDataPtr());
                if (bypassInput) { input_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE); }
                CopyRectangle(rowStart * segment,
                              rowStart * outputRowElements + axisPrefix,
                              rowEnd - rowStart, segment, segment,
                              outputRowElements);
            }
            axisPrefix += segment;
        }
    }

    __aicore__ inline void ProcessColumnSplit()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        const uint64_t alignElements = 32U / sizeof(T);
        const uint64_t outputRowElements = outputAxis_ * inner_;
        TensorDesc<T> desc;
        uint64_t shape[S9_CONCAT_MAX_RANK];
        desc.SetShapeAddr(shape);
        uint64_t axisPrefix = 0;
        for (uint32_t inputIndex = 0; inputIndex < inputCount_; ++inputIndex) {
            list_.GetDesc(desc, inputIndex);
            const uint64_t segment = desc.GetShape(axis_) * inner_;
            if (segment == 0) {
                continue;
            }
            input_.SetGlobalBuffer(desc.GetDataPtr());
            for (uint64_t row = 0; row < outer_; ++row) {
                const uint64_t inputBase = row * segment;
                const uint64_t outputBase =
                    row * outputRowElements + axisPrefix;
                uint64_t leading =
                    (alignElements - outputBase % alignElements) % alignElements;
                if (leading > segment) {
                    leading = segment;
                }
                const uint64_t alignedElements =
                    ((segment - leading) / alignElements) * alignElements;
                const uint64_t alignedBlocks = alignedElements / alignElements;
                const uint64_t firstBlock = alignedBlocks * block / blocks;
                const uint64_t lastBlock = alignedBlocks * (block + 1) / blocks;
                const uint64_t middleStart = leading + firstBlock * alignElements;
                const uint64_t middleCount = (lastBlock - firstBlock) * alignElements;
                if (middleCount != 0) {
                    CopyRange(inputBase + middleStart,
                              outputBase + middleStart, middleCount);
                }
                if (block == 0) {
                    CopyRange(inputBase, outputBase, leading);
                    const uint64_t tailStart = leading + alignedElements;
                    CopyRange(inputBase + tailStart, outputBase + tailStart,
                              segment - tailStart);
                }
            }
            axisPrefix += segment;
        }
    }

    __aicore__ inline void ProcessLinearTasks()
    {
        const uint64_t block = static_cast<uint64_t>(GetBlockIdx());
        const uint64_t blocks = static_cast<uint64_t>(GetBlockNum());
        for (uint64_t task = block; task < taskCount_; task += blocks) {
            const uint32_t inputIndex = static_cast<uint32_t>(task % inputCount_);
            const uint64_t outerIndex = task / inputCount_;
            TensorDesc<T> desc;
            uint64_t shape[S9_CONCAT_MAX_RANK];
            desc.SetShapeAddr(shape);
            uint64_t axisPrefix = 0;
            for (uint32_t i = 0; i <= inputIndex; ++i) {
                list_.GetDesc(desc, i);
                if (i < inputIndex) {
                    axisPrefix += desc.GetShape(axis_);
                }
            }
            const uint64_t segment = desc.GetShape(axis_) * inner_;
            if (segment == 0) {
                continue;
            }
            input_.SetGlobalBuffer(desc.GetDataPtr());
            const uint64_t inputOffset = outerIndex * segment;
            const uint64_t outputOffset =
                outerIndex * outputAxis_ * inner_ + axisPrefix * inner_;
            CopyRange(inputOffset, outputOffset, segment);
        }
    }

public:
    __aicore__ inline void ProcessFlatIntervals(const ConcatFlatTilingData& tiling)
    {
        const uint64_t count = outputAxis_ * inner_;
        constexpr uint64_t align = 32U / sizeof(T);
        const uint64_t blocks = count / align + static_cast<uint64_t>(count % align != 0U);
        const uint64_t core = GetBlockIdx(), cores = GetBlockNum();
        const uint64_t firstBlock = blocks / cores * core + blocks % cores * core / cores;
        const uint64_t lastBlock = blocks / cores * (core + 1U) + blocks % cores * (core + 1U) / cores;
        uint64_t cursor = firstBlock * align < count ? firstBlock * align : count;
        const uint64_t end = lastBlock * align < count ? lastBlock * align : count;
        if (cursor == end) { return; }
        uint32_t low = 0, high = inputCount_;
        while (low < high) {
            const uint32_t middle = low + (high - low) / 2U;
            if (tiling.prefixes[middle + 1U] <= cursor) { low = middle + 1U; }
            else { high = middle; }
        }
        for (uint32_t i = low; i < inputCount_ && cursor < end; ++i) {
            const uint64_t stop = tiling.prefixes[i + 1U] < end ? tiling.prefixes[i + 1U] : end;
            if (stop <= cursor) { continue; }
            input_.SetGlobalBuffer(list_.GetDataPtr<T>(i));
            CopyRange(cursor - tiling.prefixes[i], cursor, stop - cursor);
            cursor = stop;
        }
    }

    __aicore__ inline void ProcessWeightedRows(const ConcatRowsTilingData& tiling)
    {
        const auto& owner = tiling.owners[GetBlockIdx()];
        if (owner.rows == 0U || owner.segment == 0U) { return; }
        input_.SetGlobalBuffer(list_.GetDataPtr<T>(owner.input));
        const uint64_t rowElements = outputAxis_ * inner_;
        const uint64_t minimumElements =
            2ULL * kTileBytes * GetBlockNum() / sizeof(T);

        if constexpr (sizeof(T) == 4U) {
            if (rowElements != 0U && outer_ >= minimumElements / rowElements +
                    static_cast<uint64_t>(minimumElements % rowElements != 0U)) {
                input_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            }
        }
        CopyRectangle(owner.rowStart * owner.segment,
                      owner.rowStart * outputAxis_ * inner_ + owner.prefix,
                      owner.rows, owner.segment, owner.segment, outputAxis_ * inner_);
    }

    __aicore__ inline void ProcessManyGrid(const ConcatManyTilingData& tiling)
    {
        if (mode_ == 3) {
            ProcessPackedHalf(tiling);
            return;
        }
        const uint32_t block = GetBlockIdx();
        const uint32_t rowGroup = block / inputGroups_;
        const uint32_t inputGroup = block - rowGroup * inputGroups_;
        const uint64_t rowStart = outer_ * rowGroup / rowGroups_;
        const uint64_t rowEnd = outer_ * (rowGroup + 1) / rowGroups_;
        const uint64_t outputRowElements = outputAxis_ * inner_;
        const uint16_t scheduleBegin = tiling.groupOffsets[inputGroup];
        const uint16_t scheduleEnd = tiling.groupOffsets[inputGroup + 1];
        for (uint16_t scheduleIndex = scheduleBegin;
             scheduleIndex < scheduleEnd; ++scheduleIndex) {
            const uint32_t inputIndex = tiling.inputOrder[scheduleIndex];
            TensorDesc<T> desc;
            uint64_t shape[S9_CONCAT_MAX_RANK];
            desc.SetShapeAddr(shape);
            list_.GetDesc(desc, inputIndex);
            const uint64_t segment = desc.GetShape(axis_) * inner_;
            if (segment != 0) {
                input_.SetGlobalBuffer(desc.GetDataPtr());
                CopyRectangle(rowStart * segment,
                              rowStart * outputRowElements + tiling.prefixes[inputIndex],
                              rowEnd - rowStart, segment, segment,
                              outputRowElements);
            }
        }
    }

    __aicore__ inline void ProcessBalancedLinear(const ConcatLinearTilingData& tiling)
    {
        const uint32_t block = GetBlockIdx();
        const uint64_t outputRowElements = outputAxis_ * inner_;
        const uint16_t scheduleBegin = tiling.groupOffsets[block];
        const uint16_t scheduleEnd = tiling.groupOffsets[block + 1];
        for (uint16_t schedule = scheduleBegin; schedule < scheduleEnd; ++schedule) {
            const uint32_t task = tiling.linearTaskOrder[schedule];
            const uint32_t inputIndex = task % inputCount_;
            const uint64_t outerIndex = task / inputCount_;
            const uint64_t prefix = tiling.prefixes[inputIndex];
            const uint64_t end = inputIndex + 1 < inputCount_ ?
                tiling.prefixes[inputIndex + 1] : outputRowElements;
            const uint64_t segment = end - prefix;
            TensorDesc<T> desc;
            uint64_t shape[S9_CONCAT_MAX_RANK];
            desc.SetShapeAddr(shape);
            list_.GetDesc(desc, inputIndex);
            input_.SetGlobalBuffer(desc.GetDataPtr());
            CopyRange(outerIndex * segment,
                      outerIndex * outputRowElements + prefix, segment);
        }
    }

private:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kBufferNum> copyQueue_;
    TQue<QuePosition::VECIN, 1> packInQueue_;
    TBuf<QuePosition::VECCALC> packMidBuffer_;
    TQue<QuePosition::VECOUT, 1> packOutQueue_;
    ListTensorDesc list_;
    GlobalTensor<T> input_;
    GlobalTensor<T> output_;
    uint64_t outer_;
    uint64_t inner_;
    uint64_t outputAxis_;
    uint64_t taskCount_;
    uint32_t inputCount_;
    uint32_t axis_;
    uint32_t mode_;
    uint32_t rowGroups_;
    uint32_t inputGroups_;
    uint32_t tileElements_;
};

template <typename Bits, uint32_t MicroRows = S9_CONCAT_STREAM_MICRO_ROWS>
// 在输入宏块内复用索引，通过 Gather 组织窄片段拼接。
class KernelConcatStreaming {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling)
    {
        list_.Init(reinterpret_cast<__gm__ void*>(inputs));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ Bits*>(output));
        auto* data = reinterpret_cast<__gm__ ConcatStreamingTilingData*>(tiling);
        const uint32_t block = GetBlockIdx();
        group_ = block % data->groupCount;
        const uint32_t rowGroup = block / data->groupCount;
        rowStart_ = data->outer * rowGroup / data->rowGroups;
        processRows_ = data->outer * (rowGroup + 1U) / data->rowGroups - rowStart_;
        rowBytes_ = data->rowBytes;
        groupBytes_ = data->groupBytes[group_];
        groupPitch_ = data->groupPitch[group_];
        macroRows_ = data->macroRows[group_];
        const uint32_t begin = data->groupBegin[group_];
        const bool alignedOwnership = data->bufferPolicy == 3U || data->bufferPolicy == 5U;
        serpentine_ = data->bufferPolicy == 5U;
        const uint32_t end = alignedOwnership ? data->inputCount : data->groupBegin[group_ + 1U];
        outputPrefix_ = alignedOwnership ? data->microInputStride[group_] : data->prefixes[begin];
        const uint32_t outputEnd = outputPrefix_ + groupBytes_;

        const bool wider = data->bufferPolicy != 0U;
        const bool aspect = data->bufferPolicy == 8U;
        constexpr bool shorter = MicroRows == S9_CONCAT_STREAM_SHORT_MICRO_ROWS;
        const uint32_t inputBytes = aspect ? data->microInputStride[group_] :
                                    shorter ? S9_CONCAT_STREAM_SHORT_INPUT_BYTES :
                                    wider ? S9_CONCAT_STREAM_WIDE_INPUT_BYTES :
                                           S9_CONCAT_STREAM_INPUT_BYTES;
        const uint32_t outputBytes = aspect ? kMicroRows * groupPitch_ :
                                     shorter ? S9_CONCAT_STREAM_SHORT_OUTPUT_BYTES :
                                     wider ? S9_CONCAT_STREAM_WIDE_OUTPUT_BYTES :
                                            S9_CONCAT_STREAM_OUTPUT_BYTES;
        const uint32_t offsetBytes = aspect ?
            kMicroRows * groupPitch_ / sizeof(Bits) * sizeof(int32_t) :
                                     shorter ? S9_CONCAT_STREAM_SHORT_OFFSET_BYTES :
                                     wider ? S9_CONCAT_STREAM_WIDE_OFFSET_BYTES :
                                            S9_CONCAT_STREAM_OFFSET_BYTES;
        const uint32_t maxColumns = aspect ? groupPitch_ / sizeof(Bits) :
                                    (wider ? S9_CONCAT_STREAM_WIDE_ROW_BYTES :
                                            S9_CONCAT_STREAM_MAX_ROW_BYTES) / 2U;
        pipe_.InitBuffer(inputQueue_, 2, inputBytes + 32U);
        pipe_.InitBuffer(outputQueue_, 2, outputBytes + 32U);
        pipe_.InitBuffer(offsetBuffer_, offsetBytes + 32U);
        pipe_.InitBuffer(scratchBuffer_, (2U * maxColumns + 8U) * sizeof(int32_t));
        offsets_ = offsetBuffer_.Get<int32_t>()[8];
        baseRow_ = scratchBuffer_.Get<int32_t>();
        strideRow_ = scratchBuffer_.Get<int32_t>()[maxColumns + 8U];

        const uint32_t columns = groupPitch_ / sizeof(Bits);
        CreateVecIndex(baseRow_, static_cast<int32_t>(0), columns);
        PipeBarrier<PIPE_V>();
        Muls(baseRow_, baseRow_, static_cast<int32_t>(sizeof(Bits)), columns);
        Duplicate(strideRow_, static_cast<int32_t>(0), columns);
        PipeBarrier<PIPE_V>();
        cachedInputs_ = 0;
        uint32_t sourceBase = 0;
        uint32_t localPrefix = 0;
        for (uint32_t index = begin; index < end; ++index) {
            const uint32_t prefix = alignedOwnership ? data->prefixes[index] : 0U;
            if (alignedOwnership && prefix >= outputEnd) {
                break;
            }
            const uint32_t segment = data->segments[index];
            if (segment == 0U) {
                continue;
            }
            const uint32_t skip = alignedOwnership && outputPrefix_ > prefix ?
                                  outputPrefix_ - prefix : 0U;
            const uint32_t kept = alignedOwnership && prefix + segment > outputEnd ?
                                  outputEnd - prefix - skip : segment - skip;
            pointers_[cachedInputs_] = list_.GetDataPtr<Bits>(index);
            segments_[cachedInputs_] = segment;
            sourceBases_[cachedInputs_] = sourceBase;
            ++cachedInputs_;
            UpdateIndexRange(localPrefix / sizeof(Bits), kept / sizeof(Bits),
                             static_cast<int32_t>(sourceBase + skip - localPrefix),
                             static_cast<int32_t>(segment), false);

            sourceBase += Align32(macroRows_ * segment);
            localPrefix += kept;
        }

        UpdateIndexRange(groupBytes_ / sizeof(Bits),
                         (groupPitch_ - groupBytes_) / sizeof(Bits), 0, 0, true);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < kMicroRows; ++row) {
            LocalTensor<int32_t> rowOffsets = offsets_[row * columns];
            Muls(rowOffsets, strideRow_, static_cast<int32_t>(row), columns);
            PipeBarrier<PIPE_V>();
            Add(rowOffsets, rowOffsets, baseRow_, columns);
        }
        PipeBarrier<PIPE_V>();

        Muls(strideRow_, strideRow_, static_cast<int32_t>(kMicroRows), columns);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        if (processRows_ == 0U || cachedInputs_ == 0U) {
            return;
        }
        if constexpr (MicroRows == S9_CONCAT_STREAM_MICRO_ROWS) {
            if (serpentine_) {
                ProcessSerpentine();
                return;
            }
        }
        const uint64_t macroCount = (processRows_ + macroRows_ - 1U) / macroRows_;

        uint64_t ordinal = group_ % macroCount;
        uint32_t currentRows = RowsAt(ordinal);
        LocalTensor<Bits> first = inputQueue_.AllocTensor<Bits>();
        LoadMacro(first, ordinal * macroRows_, currentRows);
        inputQueue_.EnQue(first);
        uint64_t previousOutputRow = 0;
        uint32_t previousOutputRows = 0;

        for (uint64_t step = 0; step < macroCount; ++step) {
            const bool hasNext = step + 1U < macroCount;
            const uint64_t nextOrdinal = ordinal + 1U == macroCount ? 0U : ordinal + 1U;
            uint32_t nextRows = 0;
            if (hasNext) {
                nextRows = RowsAt(nextOrdinal);
                LocalTensor<Bits> next = inputQueue_.AllocTensor<Bits>();
                LoadMacro(next, nextOrdinal * macroRows_, nextRows);
                inputQueue_.EnQue(next);
            }
            LocalTensor<Bits> compact = inputQueue_.DeQue<Bits>();
            const uint32_t micros = (currentRows + kMicroRows - 1U) / kMicroRows;
            for (uint32_t micro = 0; micro < micros; ++micro) {

                if (previousOutputRows != 0U) {
                    StorePrevious(previousOutputRow, previousOutputRows);
                }
                const uint32_t remaining = currentRows - micro * kMicroRows;
                const uint32_t rows = remaining < kMicroRows ? remaining : kMicroRows;
                LocalTensor<Bits> assembled = outputQueue_.AllocTensor<Bits>();
                Gather(assembled, compact, offsets_.ReinterpretCast<uint32_t>(),
                       0U, rows * groupPitch_ / sizeof(Bits));
                outputQueue_.EnQue(assembled);
                previousOutputRow = ordinal * macroRows_ + micro * kMicroRows;
                previousOutputRows = rows;
                if (micro + 1U < micros) {

                    ShiftOffsets<false>(strideRow_);
                }
            }
            if (hasNext && micros > 1U) {
                Muls(baseRow_, strideRow_, static_cast<int32_t>(micros - 1U),
                     groupPitch_ / sizeof(Bits));
                PipeBarrier<PIPE_V>();
                ShiftOffsets<true>(baseRow_);
            }

            inputQueue_.FreeTensor(compact);
            ordinal = nextOrdinal;
            currentRows = nextRows;
        }
        if (previousOutputRows != 0U) {
            StorePrevious(previousOutputRow, previousOutputRows);
        }
    }

private:
    static constexpr uint32_t kMicroRows = MicroRows;

    __aicore__ inline void ProcessSerpentine()
    {
        const uint64_t macroCount = (processRows_ + macroRows_ - 1U) / macroRows_;
        uint64_t ordinal = group_ % macroCount;
        uint32_t currentRows = RowsAt(ordinal);
        LocalTensor<Bits> first = inputQueue_.AllocTensor<Bits>();
        LoadMacro(first, ordinal * macroRows_, currentRows);
        inputQueue_.EnQue(first);
        bool upper = false;
        uint64_t previousOutputRow = 0U;
        uint32_t previousOutputRows = 0U;

        for (uint64_t step = 0U; step < macroCount; ++step) {
            const bool hasNext = step + 1U < macroCount;
            const uint64_t nextOrdinal = ordinal + 1U == macroCount ? 0U : ordinal + 1U;
            uint32_t nextRows = 0U;
            if (hasNext) {
                nextRows = RowsAt(nextOrdinal);
                LocalTensor<Bits> next = inputQueue_.AllocTensor<Bits>();
                LoadMacro(next, nextOrdinal * macroRows_, nextRows);
                inputQueue_.EnQue(next);
            }
            LocalTensor<Bits> compact = inputQueue_.DeQue<Bits>();
            const uint32_t micros = currentRows > kMicroRows ? 2U : 1U;
            if (micros == 1U && upper) {

                ShiftOffsets<true>(strideRow_);
                upper = false;
            }
            for (uint32_t visit = 0U; visit < micros; ++visit) {
                if (previousOutputRows != 0U) {
                    StorePrevious(previousOutputRow, previousOutputRows);
                }
                const uint32_t micro = upper ? 1U : 0U;
                const uint32_t remaining = currentRows - micro * kMicroRows;
                const uint32_t rows = remaining < kMicroRows ? remaining : kMicroRows;
                LocalTensor<Bits> assembled = outputQueue_.AllocTensor<Bits>();
                Gather(assembled, compact, offsets_.ReinterpretCast<uint32_t>(),
                       0U, rows * groupPitch_ / sizeof(Bits));
                outputQueue_.EnQue(assembled);
                previousOutputRow = ordinal * macroRows_ + micro * kMicroRows;
                previousOutputRows = rows;
                if (visit + 1U < micros) {
                    if (upper) {
                        ShiftOffsets<true>(strideRow_);
                    } else {
                        ShiftOffsets<false>(strideRow_);
                    }
                    upper = !upper;
                }
            }

            inputQueue_.FreeTensor(compact);
            ordinal = nextOrdinal;
            currentRows = nextRows;
        }
        if (previousOutputRows != 0U) {
            StorePrevious(previousOutputRow, previousOutputRows);
        }
    }

    __aicore__ inline uint32_t Align32(uint32_t bytes)
    {
        return (bytes + 31U) & ~31U;
    }

    __aicore__ inline uint32_t RowsAt(uint64_t ordinal)
    {
        const uint64_t remaining = processRows_ - ordinal * macroRows_;
        return static_cast<uint32_t>(remaining < macroRows_ ? remaining : macroRows_);
    }

    __aicore__ inline void UpdateIndexRange(uint32_t first, uint32_t count,
                                           int32_t correction, int32_t stride,
                                           bool padding)
    {

        while (count != 0U) {
            const uint32_t aligned = first & ~7U;
            const uint32_t head = first - aligned;
            const uint32_t available = 64U - head;
            const uint32_t span = count < available ? count : available;
            const uint64_t bits = span == 64U ? ~0ULL : ((1ULL << span) - 1ULL) << head;
            uint64_t mask[2] = {bits, 0ULL};
            if (padding) {
                Duplicate(baseRow_[aligned], static_cast<int32_t>(0), mask, 1, 1, 8);
            } else {
                Adds(baseRow_[aligned], baseRow_[aligned], correction, mask, 1,
                     UnaryRepeatParams{1, 1, 8, 8});
            }
            Duplicate(strideRow_[aligned], stride, mask, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            first += span;
            count -= span;
        }
    }

    __aicore__ inline void LoadMacro(LocalTensor<Bits> destination,
                                     uint64_t localRow, uint32_t rows)
    {
        const DataCopyPadExtParams<Bits> noPad(false, 0, 0, static_cast<Bits>(0));
        for (uint32_t input = 0; input < cachedInputs_; ++input) {
            const uint32_t segment = segments_[input];
            input_.SetGlobalBuffer(pointers_[input]);
            const uint64_t sourceOffset = (rowStart_ + localRow) * segment / sizeof(Bits);
            const DataCopyExtParams copy(1, rows * segment, 0, 0, 0);
            DataCopyPad(destination[sourceBases_[input] / sizeof(Bits)],
                        input_[sourceOffset], copy, noPad);
        }
    }

    template <bool subtract>
    __aicore__ inline void ShiftOffsets(LocalTensor<int32_t> delta)
    {
        const uint32_t columns = groupPitch_ / sizeof(Bits);
        const uint8_t rowStride = static_cast<uint8_t>(columns / 8U);
        const BinaryRepeatParams repeat{1, 1, 1, rowStride, rowStride, 0};
        PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < columns; column += 64U) {
            const uint64_t lanes = columns - column < 64U ? columns - column : 64U;
            if constexpr (subtract) {
                Sub(offsets_[column], offsets_[column], delta[column], lanes,
                    static_cast<uint8_t>(kMicroRows), repeat);
            } else {
                Add(offsets_[column], offsets_[column], delta[column], lanes,
                    static_cast<uint8_t>(kMicroRows), repeat);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StorePrevious(uint64_t localRow, uint32_t rows)
    {
        LocalTensor<Bits> assembled = outputQueue_.DeQue<Bits>();
        const uint64_t offset = ((rowStart_ + localRow) * rowBytes_ + outputPrefix_) / sizeof(Bits);
        const DataCopyExtParams copy(rows, groupBytes_, 0, rowBytes_ - groupBytes_, 0);
        DataCopyPad(output_[offset], assembled, copy);
        outputQueue_.FreeTensor(assembled);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    TBuf<QuePosition::VECCALC> offsetBuffer_;
    TBuf<QuePosition::VECCALC> scratchBuffer_;
    LocalTensor<int32_t> offsets_;
    LocalTensor<int32_t> baseRow_;
    LocalTensor<int32_t> strideRow_;
    ListTensorDesc list_;
    GlobalTensor<Bits> input_;
    GlobalTensor<Bits> output_;
    __gm__ Bits* pointers_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint32_t segments_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint32_t sourceBases_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint64_t rowStart_;
    uint64_t processRows_;
    uint32_t group_;
    uint32_t rowBytes_;
    uint32_t groupBytes_;
    uint32_t groupPitch_;
    uint32_t macroRows_;
    uint32_t outputPrefix_;
    uint32_t cachedInputs_;
    bool serpentine_;
};

template <typename Bits>
class KernelConcatWideStatic {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR y, GM_ADDR tiling)
    {
        constexpr uint32_t microRows = 4U;
        list_.Init(reinterpret_cast<__gm__ void*>(inputs));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ Bits*>(y));
        auto* p = reinterpret_cast<__gm__ ConcatStreamingTilingData*>(tiling);
        group_ = GetBlockIdx() % p->groupCount;
        const uint32_t rowGroup = GetBlockIdx() / p->groupCount;
        rowStart_ = p->outer * rowGroup / p->rowGroups;
        uint64_t rowEnd = p->outer * (rowGroup + 1U) / p->rowGroups;
        if constexpr (sizeof(Bits) == 4U) {

            constexpr uint64_t rowQuantum = 32U / sizeof(Bits);
            const uint64_t units = p->outer / rowQuantum + (p->outer % rowQuantum != 0U);
            rowStart_ = (units * rowGroup / p->rowGroups) * rowQuantum;
            rowEnd = (units * (rowGroup + 1U) / p->rowGroups) * rowQuantum;
            if (rowStart_ > p->outer) { rowStart_ = p->outer; }
            if (rowEnd > p->outer) { rowEnd = p->outer; }
        }
        rows_ = rowEnd - rowStart_;
        rowBytes_ = p->rowBytes;
        bytes_ = p->groupBytes[group_];
        pitch_ = p->groupPitch[group_];
        columns_ = pitch_ / sizeof(Bits);
        macroRows_ = p->macroRows[group_];
        const uint32_t begin = p->groupBegin[group_];
        const uint32_t end = p->groupBegin[group_ + 1U];
        prefix_ = p->prefixes[begin];
        pipe_.InitBuffer(inputQueue_, 2, p->microInputStride[group_] + 32U);
        pipe_.InitBuffer(outputQueue_, 2, microRows * pitch_ + 32U);
        pipe_.InitBuffer(indexBuffer_, macroRows_ * columns_ * 4U + 32U);
        pipe_.InitBuffer(scratchBuffer_, (2U * columns_ + 8U) * 4U);
        offsets_ = indexBuffer_.Get<int32_t>()[8];
        base_ = scratchBuffer_.Get<int32_t>();
        stride_ = scratchBuffer_.Get<int32_t>()[columns_ + 8U];
        CreateVecIndex(base_, static_cast<int32_t>(0), columns_);
        PipeBarrier<PIPE_V>();
        Muls(base_, base_, static_cast<int32_t>(sizeof(Bits)), columns_);
        Duplicate(stride_, static_cast<int32_t>(0), columns_);
        PipeBarrier<PIPE_V>();
        count_ = 0U;
        uint32_t sourceBase = 0U, localPrefix = 0U;
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t segment = p->segments[i];
            if (segment == 0U) { continue; }
            pointers_[count_] = list_.GetDataPtr<Bits>(i);
            segments_[count_] = segment;
            bases_[count_++] = sourceBase;
            FillIndexRange(localPrefix / sizeof(Bits), segment / sizeof(Bits),
                static_cast<int32_t>(sourceBase) - static_cast<int32_t>(localPrefix),
                static_cast<int32_t>(segment), false);
            sourceBase += Align32(macroRows_ * segment);
            localPrefix += segment;
        }
        FillIndexRange(bytes_ / sizeof(Bits), (pitch_ - bytes_) / sizeof(Bits), 0, 0, true);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < macroRows_; ++row) {
            LocalTensor<int32_t> rowOffsets = offsets_[row * columns_];
            Muls(rowOffsets, stride_, static_cast<int32_t>(row), columns_);
            PipeBarrier<PIPE_V>();
            Add(rowOffsets, rowOffsets, base_, columns_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t microRows = 4U;
        if (rows_ == 0U || count_ == 0U) { return; }
        const uint64_t macros = (rows_ + macroRows_ - 1U) / macroRows_;
        uint64_t ordinal = group_ % macros;
        uint32_t currentRows = RowsAt(ordinal);
        LocalTensor<Bits> first = inputQueue_.AllocTensor<Bits>();
        Load(first, ordinal * macroRows_, currentRows);
        inputQueue_.EnQue(first);
        uint64_t previousRow = 0U;
        uint32_t previousRows = 0U;
        for (uint64_t step = 0U; step < macros; ++step) {
            const bool hasNext = step + 1U < macros;
            const uint64_t nextOrdinal = ordinal + 1U == macros ? 0U : ordinal + 1U;
            const uint32_t nextRows = hasNext ? RowsAt(nextOrdinal) : 0U;
            if (hasNext) {
                LocalTensor<Bits> next = inputQueue_.AllocTensor<Bits>();
                Load(next, nextOrdinal * macroRows_, nextRows);
                inputQueue_.EnQue(next);
            }
            LocalTensor<Bits> input = inputQueue_.DeQue<Bits>();
            for (uint32_t row = 0U; row < currentRows; row += microRows) {
                if (previousRows != 0U) { Store(previousRow, previousRows); }
                const uint32_t remaining = currentRows - row;
                const uint32_t count = remaining < microRows ? remaining : microRows;
                LocalTensor<Bits> output = outputQueue_.AllocTensor<Bits>();
                LocalTensor<uint32_t> indices = offsets_[row * columns_].ReinterpretCast<uint32_t>();
                Gather(output, input, indices, 0U, count * columns_);
                outputQueue_.EnQue(output);
                previousRow = ordinal * macroRows_ + row;
                previousRows = count;
            }
            inputQueue_.FreeTensor(input);
            ordinal = nextOrdinal;
            currentRows = nextRows;
        }
        if (previousRows != 0U) { Store(previousRow, previousRows); }
    }

private:
    __aicore__ inline uint32_t Align32(uint32_t bytes)
    {
        return (bytes + 31U) & ~31U;
    }

    __aicore__ inline uint32_t RowsAt(uint64_t ordinal)
    {
        const uint64_t remaining = rows_ - ordinal * macroRows_;
        return static_cast<uint32_t>(remaining < macroRows_ ? remaining : macroRows_);
    }

    __aicore__ inline void FillIndexRange(uint32_t first, uint32_t count,
                                         int32_t correction, int32_t stride, bool padding)
    {
        while (count != 0U) {
            const uint32_t aligned = first & ~7U;
            const uint32_t head = first - aligned;
            const uint32_t span = count < 64U - head ? count : 64U - head;
            const uint64_t bits = span == 64U ? ~0ULL : ((1ULL << span) - 1ULL) << head;
            uint64_t mask[2] = {bits, 0ULL};
            if (padding) {
                Duplicate(base_[aligned], static_cast<int32_t>(0), mask, 1, 1, 8);
            } else {
                Adds(base_[aligned], base_[aligned], correction, mask, 1,
                     UnaryRepeatParams{1, 1, 8, 8});
            }
            Duplicate(stride_[aligned], stride, mask, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            first += span;
            count -= span;
        }
    }

    __aicore__ inline void Load(LocalTensor<Bits> local, uint64_t row, uint32_t rows)
    {
        const DataCopyPadExtParams<Bits> noPad(false, 0, 0, static_cast<Bits>(0));
        for (uint32_t i = 0; i < count_; ++i) {
            input_.SetGlobalBuffer(pointers_[i]);
            input_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            const DataCopyExtParams copy(1, rows * segments_[i], 0, 0, 0);
            DataCopyPad(local[bases_[i] / sizeof(Bits)],
                        input_[(rowStart_ + row) * segments_[i] / sizeof(Bits)], copy, noPad);
        }
    }

    __aicore__ inline void Store(uint64_t row, uint32_t rows)
    {
        LocalTensor<Bits> local = outputQueue_.DeQue<Bits>();

        const uint32_t sourceStride = (pitch_ - Align32(bytes_)) / 32U;
        const DataCopyExtParams copy(rows, bytes_, sourceStride, rowBytes_ - bytes_, 0);
        DataCopyPad(output_[((rowStart_ + row) * rowBytes_ + prefix_) / sizeof(Bits)],
                    local, copy);
        outputQueue_.FreeTensor(local);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    TBuf<QuePosition::VECCALC> indexBuffer_, scratchBuffer_;
    LocalTensor<int32_t> offsets_, base_, stride_;
    ListTensorDesc list_;
    GlobalTensor<Bits> input_, output_;
    __gm__ Bits* pointers_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint32_t segments_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint32_t bases_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint64_t rows_, rowStart_;
    uint32_t group_, rowBytes_, bytes_, pitch_, columns_, macroRows_, prefix_, count_;
};

template <typename Bits>
class KernelConcatSmallResident {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR y, GM_ADDR tiling)
    {
        list_.Init(reinterpret_cast<__gm__ void*>(inputs));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ Bits*>(y));
        auto* p = reinterpret_cast<__gm__ ConcatStreamingTilingData*>(tiling);
        group_ = GetBlockIdx() % p->groupCount;
        const uint32_t rowGroup = GetBlockIdx() / p->groupCount;
        rowStart_ = p->outer * rowGroup / p->rowGroups;
        uint64_t rowEnd = p->outer * (rowGroup + 1U) / p->rowGroups;
        if constexpr (sizeof(Bits) == 2U) {

            constexpr uint64_t rowQuantum = 32U / sizeof(Bits);
            const uint64_t units = p->outer / rowQuantum + (p->outer % rowQuantum != 0U);
            rowStart_ = (units * rowGroup / p->rowGroups) * rowQuantum;
            rowEnd = (units * (rowGroup + 1U) / p->rowGroups) * rowQuantum;
            if (rowStart_ > p->outer) { rowStart_ = p->outer; }
            if (rowEnd > p->outer) { rowEnd = p->outer; }
        }
        rows_ = rowEnd - rowStart_;
        rowBytes_ = p->rowBytes;
        bytes_ = p->groupBytes[group_];
        pitch_ = p->groupPitch[group_];
        columns_ = pitch_ / sizeof(Bits);
        macroRows_ = p->macroRows[group_];
        const uint32_t begin = p->groupBegin[group_];
        const uint32_t end = p->groupBegin[group_ + 1U];
        prefix_ = p->prefixes[begin];
        pipe_.InitBuffer(inputQueue_, 1, p->microInputStride[group_] + 32U);
        pipe_.InitBuffer(outputQueue_, 1, macroRows_ * pitch_ + 32U);
        pipe_.InitBuffer(indexBuffer_, macroRows_ * columns_ * 4U + 32U);
        pipe_.InitBuffer(scratchBuffer_, (2U * columns_ + 8U) * 4U);
        offsets_ = indexBuffer_.Get<int32_t>()[8];
        base_ = scratchBuffer_.Get<int32_t>();
        stride_ = scratchBuffer_.Get<int32_t>()[columns_ + 8U];
        CreateVecIndex(base_, static_cast<int32_t>(0), columns_);
        PipeBarrier<PIPE_V>();
        Muls(base_, base_, static_cast<int32_t>(sizeof(Bits)), columns_);
        Duplicate(stride_, static_cast<int32_t>(0), columns_);
        PipeBarrier<PIPE_V>();
        count_ = 0U;
        uint32_t sourceBase = 0U, localPrefix = 0U;
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t segment = p->segments[i];
            if (segment == 0U) { continue; }
            pointers_[count_] = list_.GetDataPtr<Bits>(i);
            segments_[count_] = segment;
            bases_[count_++] = sourceBase;
            FillIndexRange(localPrefix / sizeof(Bits), segment / sizeof(Bits),
                static_cast<int32_t>(sourceBase) - static_cast<int32_t>(localPrefix),
                static_cast<int32_t>(segment), false);
            sourceBase += Align32(macroRows_ * segment);
            localPrefix += segment;
        }
        FillIndexRange(bytes_ / sizeof(Bits), (pitch_ - bytes_) / sizeof(Bits), 0, 0, true);
        PipeBarrier<PIPE_V>();

        LocalTensor<Bits> initial = inputQueue_.AllocTensor<Bits>();
        Load(initial, 0U, static_cast<uint32_t>(rows_));
        inputQueue_.EnQue(initial);
        for (uint32_t row = 0; row < macroRows_; ++row) {
            LocalTensor<int32_t> rowOffsets = offsets_[row * columns_];
            Muls(rowOffsets, stride_, static_cast<int32_t>(row), columns_);
            PipeBarrier<PIPE_V>();
            Add(rowOffsets, rowOffsets, base_, columns_);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        if (rows_ == 0U || count_ == 0U) { return; }
        LocalTensor<Bits> input = inputQueue_.DeQue<Bits>();
        LocalTensor<Bits> output = outputQueue_.AllocTensor<Bits>();
        Gather(output, input, offsets_.ReinterpretCast<uint32_t>(), 0U,
               static_cast<uint32_t>(rows_) * columns_);
        outputQueue_.EnQue(output);
        inputQueue_.FreeTensor(input);
        Store(0U, static_cast<uint32_t>(rows_));
    }

private:
    __aicore__ inline uint32_t Align32(uint32_t bytes)
    {
        return (bytes + 31U) & ~31U;
    }

    __aicore__ inline uint32_t RowsAt(uint64_t ordinal)
    {
        const uint64_t remaining = rows_ - ordinal * macroRows_;
        return static_cast<uint32_t>(remaining < macroRows_ ? remaining : macroRows_);
    }

    __aicore__ inline void FillIndexRange(uint32_t first, uint32_t count,
                                         int32_t correction, int32_t stride, bool padding)
    {
        while (count != 0U) {
            const uint32_t aligned = first & ~7U;
            const uint32_t head = first - aligned;
            const uint32_t span = count < 64U - head ? count : 64U - head;
            const uint64_t bits = span == 64U ? ~0ULL : ((1ULL << span) - 1ULL) << head;
            uint64_t mask[2] = {bits, 0ULL};
            if (padding) {
                Duplicate(base_[aligned], static_cast<int32_t>(0), mask, 1, 1, 8);
            } else {
                Adds(base_[aligned], base_[aligned], correction, mask, 1,
                     UnaryRepeatParams{1, 1, 8, 8});
            }
            Duplicate(stride_[aligned], stride, mask, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            first += span;
            count -= span;
        }
    }

    __aicore__ inline void Load(LocalTensor<Bits> local, uint64_t row, uint32_t rows)
    {
        const DataCopyPadExtParams<Bits> noPad(false, 0, 0, static_cast<Bits>(0));
        for (uint32_t i = 0; i < count_; ++i) {
            input_.SetGlobalBuffer(pointers_[i]);
            input_.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
            const DataCopyExtParams copy(1, rows * segments_[i], 0, 0, 0);
            DataCopyPad(local[bases_[i] / sizeof(Bits)],
                        input_[(rowStart_ + row) * segments_[i] / sizeof(Bits)], copy, noPad);
        }
    }

    __aicore__ inline void Store(uint64_t row, uint32_t rows)
    {
        LocalTensor<Bits> local = outputQueue_.DeQue<Bits>();

        const uint32_t sourceStride = (pitch_ - Align32(bytes_)) / 32U;
        const DataCopyExtParams copy(1U, rows * bytes_, 0U, 0U, 0U);
        DataCopyPad(output_[((rowStart_ + row) * rowBytes_ + prefix_) / sizeof(Bits)],
                    local, copy);
        outputQueue_.FreeTensor(local);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<QuePosition::VECCALC> indexBuffer_, scratchBuffer_;
    LocalTensor<int32_t> offsets_, base_, stride_;
    ListTensorDesc list_;
    GlobalTensor<Bits> input_, output_;
    __gm__ Bits* pointers_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint32_t segments_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint32_t bases_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint64_t rows_, rowStart_;
    uint32_t group_, rowBytes_, bytes_, pitch_, columns_, macroRows_, prefix_, count_;
};
template <typename T>
// 由输出区间的唯一所有者执行拷贝，避免核间重复写入。
class KernelConcatDirectOwner {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling)
    {
        list_.Init(reinterpret_cast<__gm__ void*>(inputs));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        auto* plan = reinterpret_cast<__gm__ ConcatRowsTilingData*>(tiling);
        auto* owner = &plan->owners[GetBlockIdx()];
        rowStart_ = owner->rowStart;
        rows_ = owner->rows;
        segment_ = owner->segment;
        prefix_ = owner->prefix;
        inputIndex_ = owner->input;
        rowElements_ = plan->base.outputAxis * plan->base.inner;
        const uint64_t threshold = 2ULL * kTileBytes * GetBlockNum() / sizeof(T);
        bypass_ = sizeof(T) == 4U && rowElements_ != 0U &&
            plan->base.outer >= threshold / rowElements_ +
                static_cast<uint64_t>(threshold % rowElements_ != 0U);
        tileElements_ = kTileBytes / sizeof(T);
        pipe_.InitBuffer(copyQueue_, kBufferNum, kTileBytes);
    }
    __aicore__ inline void Process()
    {
        if (rows_ == 0U || segment_ == 0U) { return; }
        input_.SetGlobalBuffer(list_.GetDataPtr<T>(inputIndex_));
        if (bypass_) { input_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE); }
        CopyRectangle(rowStart_ * segment_, rowStart_ * rowElements_ + prefix_,
                      rows_, segment_, segment_, rowElements_);
    }
private:
    __aicore__ inline void CopyIn(uint64_t inputOffset, uint32_t count)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        DataCopyPad(local, input_[inputOffset], copy, pad);
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset, uint32_t count)
    {
        LocalTensor<T> local = copyQueue_.DeQue<T>();
        const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
        DataCopyPad(output_[outputOffset], local, copy);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void CopyRectangleIn(uint64_t inputOffset,
                                            uint32_t rows,
                                            uint32_t columnBytes,
                                            uint32_t inputStrideBytes)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        const DataCopyExtParams copy(
            rows, columnBytes, inputStrideBytes, 0, 0);
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        DataCopyPad(local, input_[inputOffset], copy, pad);
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyRectangleOut(uint64_t outputOffset,
                                             uint32_t rows,
                                             uint32_t columnBytes,
                                             uint32_t outputStrideBytes)
    {
        LocalTensor<T> local = copyQueue_.DeQue<T>();
        const DataCopyExtParams copy(
            rows, columnBytes, 0, outputStrideBytes, 0);
        DataCopyPad(output_[outputOffset], local, copy);
        copyQueue_.FreeTensor(local);
    }

    __aicore__ inline void CopyRange(uint64_t inputOffset,
                                     uint64_t outputOffset,
                                     uint64_t count)
    {
        if (count == 0) {
            return;
        }
        uint64_t done = 0;
        uint32_t previousCount = static_cast<uint32_t>(
            count < tileElements_ ? count : tileElements_);
        CopyIn(inputOffset, previousCount);
        done += previousCount;
        uint64_t previousOutputOffset = outputOffset;
        while (done < count) {
            const uint32_t currentCount = static_cast<uint32_t>(
                (count - done) < tileElements_ ? (count - done) : tileElements_);
            CopyIn(inputOffset + done, currentCount);
            CopyOut(previousOutputOffset, previousCount);
            previousOutputOffset = outputOffset + done;
            previousCount = currentCount;
            done += currentCount;
        }
        CopyOut(previousOutputOffset, previousCount);
    }

    __aicore__ inline void CopyRectangle(uint64_t inputOffset,
                                         uint64_t outputOffset,
                                         uint64_t rows,
                                         uint64_t columns,
                                         uint64_t inputRowElements,
                                         uint64_t outputRowElements)
    {

        if (columns == 0 || rows == 0) {
            return;
        }
        if (columns > tileElements_ ||
            (inputRowElements - columns) * sizeof(T) > 0xffffffffULL ||
            (outputRowElements - columns) * sizeof(T) > 0xffffffffULL) {
            for (uint64_t row = 0; row < rows; ++row) {
                CopyRange(inputOffset + row * inputRowElements,
                          outputOffset + row * outputRowElements,
                          columns);
            }
            return;
        }

        const uint64_t columnBytes = columns * sizeof(T);
        const uint64_t alignedRowBytes = (columnBytes + 31ULL) & ~31ULL;
        uint64_t rowsPerBatch = kTileBytes / alignedRowBytes;

        if (rowsPerBatch > 4095ULL) {
            rowsPerBatch = 4095ULL;
        }
        const uint32_t columnBytes32 = static_cast<uint32_t>(columnBytes);
        const uint32_t inputStrideBytes = static_cast<uint32_t>(
            (inputRowElements - columns) * sizeof(T));
        const uint32_t outputStrideBytes = static_cast<uint32_t>(
            (outputRowElements - columns) * sizeof(T));
        uint64_t row = 0;
        uint64_t previousRow = 0;
        uint32_t previousRows = static_cast<uint32_t>(
            rows < rowsPerBatch ? rows : rowsPerBatch);
        CopyRectangleIn(inputOffset, previousRows, columnBytes32,
                        inputStrideBytes);
        row += previousRows;
        while (row < rows) {
            const uint32_t currentRows = static_cast<uint32_t>(
                (rows - row) < rowsPerBatch ? (rows - row) : rowsPerBatch);
            CopyRectangleIn(inputOffset + row * inputRowElements,
                            currentRows, columnBytes32, inputStrideBytes);
            CopyRectangleOut(outputOffset + previousRow * outputRowElements,
                             previousRows, columnBytes32, outputStrideBytes);
            previousRow = row;
            previousRows = currentRows;
            row += currentRows;
        }
        CopyRectangleOut(outputOffset + previousRow * outputRowElements,
                         previousRows, columnBytes32, outputStrideBytes);
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, kBufferNum> copyQueue_;
    ListTensorDesc list_;
    GlobalTensor<T> input_, output_;
    uint64_t rowStart_, rows_, segment_, prefix_, rowElements_;
    uint32_t inputIndex_, tileElements_;
    bool bypass_;
};

template <typename T>
class KernelConcatDirectFlat {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling)
    {
        list_.Init(reinterpret_cast<__gm__ void*>(inputs));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output));
        plan_ = reinterpret_cast<__gm__ ConcatFlatTilingData*>(tiling);
        count_ = plan_->base.outputAxis * plan_->base.inner;
        inputCount_ = plan_->base.inputCount;
        tileElements_ = kTileBytes / sizeof(T);
        pipe_.InitBuffer(copyQueue_, kBufferNum, kTileBytes);
    }
    __aicore__ inline void Process()
    {
        const uint64_t count = count_;
        constexpr uint64_t align = 32U / sizeof(T);
        const uint64_t blocks = count / align + static_cast<uint64_t>(count % align != 0U);
        const uint64_t core = GetBlockIdx(), cores = GetBlockNum();
        const uint64_t firstBlock = blocks / cores * core + blocks % cores * core / cores;
        const uint64_t lastBlock = blocks / cores * (core + 1U) + blocks % cores * (core + 1U) / cores;
        uint64_t cursor = firstBlock * align < count ? firstBlock * align : count;
        const uint64_t end = lastBlock * align < count ? lastBlock * align : count;
        if (cursor == end) { return; }
        uint32_t low = 0, high = inputCount_;
        while (low < high) {
            const uint32_t middle = low + (high - low) / 2U;
            if (plan_->prefixes[middle + 1U] <= cursor) { low = middle + 1U; }
            else { high = middle; }
        }
        for (uint32_t i = low; i < inputCount_ && cursor < end; ++i) {
            const uint64_t stop = plan_->prefixes[i + 1U] < end ? plan_->prefixes[i + 1U] : end;
            if (stop <= cursor) { continue; }
            input_.SetGlobalBuffer(list_.GetDataPtr<T>(i));
            CopyRange(cursor - plan_->prefixes[i], cursor, stop - cursor);
            cursor = stop;
        }
    }
private:
    __aicore__ inline void CopyIn(uint64_t inputOffset, uint32_t count)
    {
        LocalTensor<T> local = copyQueue_.AllocTensor<T>();
        const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
        const DataCopyPadExtParams<T> pad(false, 0, 0, static_cast<T>(0));
        DataCopyPad(local, input_[inputOffset], copy, pad);
        copyQueue_.EnQue(local);
    }
    __aicore__ inline void CopyOut(uint64_t outputOffset, uint32_t count)
    {
        LocalTensor<T> local = copyQueue_.DeQue<T>();
        const DataCopyExtParams copy(1, count * sizeof(T), 0, 0, 0);
        DataCopyPad(output_[outputOffset], local, copy);
        copyQueue_.FreeTensor(local);
    }
    __aicore__ inline void CopyRange(uint64_t inputOffset,
                                     uint64_t outputOffset,
                                     uint64_t count)
    {
        if (count == 0) {
            return;
        }
        uint64_t done = 0;
        uint32_t previousCount = static_cast<uint32_t>(
            count < tileElements_ ? count : tileElements_);
        CopyIn(inputOffset, previousCount);
        done += previousCount;
        uint64_t previousOutputOffset = outputOffset;
        while (done < count) {
            const uint32_t currentCount = static_cast<uint32_t>(
                (count - done) < tileElements_ ? (count - done) : tileElements_);
            CopyIn(inputOffset + done, currentCount);
            CopyOut(previousOutputOffset, previousCount);
            previousOutputOffset = outputOffset + done;
            previousCount = currentCount;
            done += currentCount;
        }
        CopyOut(previousOutputOffset, previousCount);
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> copyQueue_;
    ListTensorDesc list_;
    GlobalTensor<T> input_, output_;
    __gm__ ConcatFlatTilingData* plan_;
    uint64_t count_;
    uint32_t inputCount_, tileElements_;
};


// 输入与紧凑输出复用物理存储，阶段间必须完成相应读写同步。
class KernelConcatInplaceCompact {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling)
    {
        auto* p = reinterpret_cast<__gm__ ConcatRowCompactTilingData*>(tiling);
        list_.Init(reinterpret_cast<__gm__ void*>(inputs));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output));
        start_ = p->base.outer * GetBlockIdx() / GetBlockNum();
        rows_ = p->base.outer * (GetBlockIdx() + 1U) / GetBlockNum() - start_;
        inputPitch_ = p->inputPitch; rowBytes_ = p->outputRowBytes; batchRows_ = p->batchRows;
        const uint64_t threshold = 2ULL * kTileBytes * GetBlockNum();
        bypass_ = p->base.outer >= threshold / rowBytes_ + static_cast<uint64_t>(threshold % rowBytes_ != 0U);
        count_ = 0U;
        for (uint32_t i = 0U; i < p->base.inputCount; ++i) {
            if (p->segments[i] == 0U) { continue; }
            pointers_[count_] = list_.GetDataPtr<uint32_t>(i);
            segments_[count_] = p->segments[i]; offsets_[count_++] = p->offsets[i];
        }
        for (uint32_t s = 0U; s < 2U; ++s) {
            pipe_.InitBuffer(slots_[s], batchRows_ * inputPitch_ + 32U);
            loaded_[s] = pipe_.AllocEventID<HardEvent::MTE2_V>();
            stored_[s] = pipe_.AllocEventID<HardEvent::MTE3_MTE2>();
            live_[s] = false;
        }
        ready_ = pipe_.AllocEventID<HardEvent::V_MTE3>();
        pipe_.InitBuffer(maskBuffer_, Align32(p->maskBytes) + 32U);
        mask_ = maskBuffer_.Get<uint8_t>();
        GlobalTensor<uint8_t> metadata; metadata.SetGlobalBuffer(p->mask);
        DataCopyPad(mask_, metadata, DataCopyExtParams(1, p->maskBytes, 0, 0, 0),
                    DataCopyPadExtParams<uint8_t>(false, 0, 0, 0U));
        PipeBarrier<PIPE_ALL>();
    }
    __aicore__ inline void Process()
    {
        uint32_t current = static_cast<uint32_t>(rows_ < batchRows_ ? rows_ : batchRows_);
        uint32_t slot = 0U;
        if (current != 0U) { Load(slot, 0U, current); }
        for (uint64_t row = 0U; row < rows_;) {
            const uint64_t nextRow = row + current;
            const uint32_t next = static_cast<uint32_t>(rows_ - nextRow < batchRows_ ? rows_ - nextRow : batchRows_);
            const uint32_t other = slot ^ 1U;
            if (next != 0U) {
                if (live_[other]) { WaitFlag<HardEvent::MTE3_MTE2>(stored_[other]); live_[other] = false; }
                Load(other, nextRow, next);
            }
            WaitFlag<HardEvent::MTE2_V>(loaded_[slot]);
            LocalTensor<uint32_t> data = slots_[slot].Get<uint32_t>();
            uint64_t retained = 0U;
            GatherMask(data, data, mask_.ReinterpretCast<uint32_t>(), true,
                       current * inputPitch_ / 4U, GatherMaskParams(1, 1, 8, 0), retained);
            SetFlag<HardEvent::V_MTE3>(ready_); WaitFlag<HardEvent::V_MTE3>(ready_);
            DataCopyPad(output_[(start_ + row) * rowBytes_ / 4U], data,
                        DataCopyExtParams(1, current * rowBytes_, 0, 0, 0));
            SetFlag<HardEvent::MTE3_MTE2>(stored_[slot]); live_[slot] = true;
            row = nextRow; current = next; slot = other;
        }
        for (uint32_t s = 0U; s < 2U; ++s) {
            if (live_[s]) { WaitFlag<HardEvent::MTE3_MTE2>(stored_[s]); }
            pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(stored_[s]);
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(loaded_[s]);
        }
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(ready_);
    }

    __aicore__ inline void ProcessPhased()
    {
        if (!bypass_ || rows_ <= 2U * batchRows_) { Process(); return; }
        uint64_t row = 0U;
        while (row < rows_) {
            uint64_t starts[2];
            uint32_t counts[2];
            uint32_t held = 0U;
            for (; held < 2U && row < rows_; ++held) {
                starts[held] = row;
                counts[held] = static_cast<uint32_t>(rows_ - row < batchRows_ ? rows_ - row : batchRows_);
                Load(held, row, counts[held]);
                row += counts[held];
            }

            for (uint32_t slot = 0U; slot < held; ++slot) {
                WaitFlag<HardEvent::MTE2_V>(loaded_[slot]);
            }
            for (uint32_t slot = 0U; slot < held; ++slot) {
                LocalTensor<uint32_t> data = slots_[slot].Get<uint32_t>();
                uint64_t retained = 0U;
                GatherMask(data, data, mask_.ReinterpretCast<uint32_t>(), true,
                           counts[slot] * inputPitch_ / 4U, GatherMaskParams(1, 1, 8, 0), retained);
                SetFlag<HardEvent::V_MTE3>(ready_); WaitFlag<HardEvent::V_MTE3>(ready_);
                DataCopyPad(output_[(start_ + starts[slot]) * rowBytes_ / 4U], data,
                            DataCopyExtParams(1, counts[slot] * rowBytes_, 0, 0, 0));
                SetFlag<HardEvent::MTE3_MTE2>(stored_[slot]);
            }
            for (uint32_t slot = 0U; slot < held; ++slot) {
                WaitFlag<HardEvent::MTE3_MTE2>(stored_[slot]);
            }
        }
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(stored_[slot]);
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(loaded_[slot]);
        }
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(ready_);
    }
private:
    __aicore__ inline uint32_t Align32(uint32_t x) { return (x + 31U) & ~31U; }
    __aicore__ inline void Load(uint32_t slot, uint64_t row, uint32_t rows)
    {
        LocalTensor<uint32_t> input = slots_[slot].Get<uint32_t>();
        for (uint32_t i = 0U; i < count_; ++i) {
            input_.SetGlobalBuffer(pointers_[i]);
            if (bypass_) { input_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE); }
            const uint32_t stride = (inputPitch_ - Align32(segments_[i])) / 32U;
            DataCopyPad(input[offsets_[i] / 4U], input_[(start_ + row) * segments_[i] / 4U],
                        DataCopyExtParams(rows, segments_[i], 0, stride, 0),
                        DataCopyPadExtParams<uint32_t>(false, 0, 0, 0U));
        }
        SetFlag<HardEvent::MTE2_V>(loaded_[slot]);
    }
    TPipe pipe_;
    TBuf<QuePosition::VECCALC> slots_[2], maskBuffer_;
    TEventID loaded_[2], stored_[2], ready_;
    bool live_[2];
    LocalTensor<uint8_t> mask_;
    ListTensorDesc list_;
    GlobalTensor<uint32_t> input_, output_;
    __gm__ uint32_t* pointers_[8];
    uint32_t segments_[8], offsets_[8];
    uint64_t start_, rows_;
    uint32_t inputPitch_, rowBytes_, batchRows_, count_;
    bool bypass_;
};

class KernelConcatPackedBlocks {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR y, GM_ADDR tiling)
    {
        auto* packed = reinterpret_cast<__gm__ ConcatPackedBlockTilingData*>(tiling);
        auto* p = &packed->base;
        group_ = GetBlockIdx() % p->groupCount;
        const uint32_t owner = GetBlockIdx() / p->groupCount;
        const uint64_t units = p->outer / 8U + (p->outer % 8U != 0U);
        rowStart_ = (units * owner / p->rowGroups) * 8U;
        uint64_t rowEnd = (units * (owner + 1U) / p->rowGroups) * 8U;
        if (rowStart_ > p->outer) { rowStart_ = p->outer; }
        if (rowEnd > p->outer) { rowEnd = p->outer; }
        rows_ = rowEnd - rowStart_;
        rowBytes_ = p->rowBytes;
        bytes_ = p->groupBytes[group_];
        pitch_ = p->groupPitch[group_];
        macro_ = p->macroRows[group_];
        blocks_ = packed->rowBlocks[group_];
        const uint32_t begin = p->groupBegin[group_], end = p->groupBegin[group_ + 1U];
        prefix_ = p->prefixes[begin];
        list_.Init(reinterpret_cast<__gm__ void*>(inputs));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y));
        pipe_.InitBuffer(inputQueue_, 2, p->microInputStride[group_] + 32U);
        pipe_.InitBuffer(outputQueue_, 2, 4U * pitch_ + 32U);
        pipe_.InitBuffer(indexBuffer_, macro_ * blocks_ * 4U + 32U);
        pipe_.InitBuffer(maskBuffer_, 8U * blocks_ + 32U);
        pipe_.InitBuffer(expandedBuffer_, 4U * blocks_ * 32U + 32U);
        indices_ = indexBuffer_.Get<uint32_t>();
        masks_ = maskBuffer_.Get<uint8_t>();
        expanded_ = expandedBuffer_.Get<uint32_t>();
        count_ = 0U;
        uint32_t sourceBase = 0U;
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t width = p->segments[i];
            if (width == 0U) { continue; }
            pointers_[count_] = list_.GetDataPtr<uint32_t>(i);
            widths_[count_] = width;
            bases_[count_++] = sourceBase;
            sourceBase += Align32(macro_ * width);
        }
        const uint32_t indexCount = macro_ * blocks_;
        __gm__ uint8_t* data = reinterpret_cast<__gm__ uint8_t*>(packed->payload) + packed->payloadOffset[group_];
        GlobalTensor<int16_t> packedIndices;
        packedIndices.SetGlobalBuffer(reinterpret_cast<__gm__ int16_t*>(data));
        DataCopyPad(indices_.ReinterpretCast<int16_t>(), packedIndices,
                    DataCopyExtParams(1, indexCount * 2U, 0, 0, 0),
                    DataCopyPadExtParams<int16_t>(false, 0, 0, 0));
        GlobalTensor<uint8_t> packedMasks;
        packedMasks.SetGlobalBuffer(data + indexCount * 2U);
        DataCopyPad(masks_, packedMasks, DataCopyExtParams(1, 8U * blocks_, 0, 0, 0),
                    DataCopyPadExtParams<uint8_t>(false, 0, 0, 0));
        PipeBarrier<PIPE_ALL>();

        auto scratch = expanded_.ReinterpretCast<float>();
        auto compact = indices_.ReinterpretCast<int16_t>();
        auto decoded = indices_.ReinterpretCast<int32_t>();
        for (uint32_t endIndex = (indexCount + 127U) / 128U * 128U;
             endIndex != 0U; endIndex -= 128U) {
            const uint32_t firstIndex = endIndex - 128U;
            const uint32_t remaining = indexCount - firstIndex;
            const uint32_t count = remaining < 128U ? remaining : 128U;
            Cast(scratch, compact[firstIndex], RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Muls(scratch, scratch, 32.0f, count);
            PipeBarrier<PIPE_V>();
            Cast(decoded[firstIndex], scratch, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void Process()
    {
        if (rows_ == 0U || count_ == 0U) { return; }
        const uint64_t macros = (rows_ + macro_ - 1U) / macro_;
        uint64_t ordinal = group_ % macros;
        uint32_t currentRows = RowsAt(ordinal);
        auto first = inputQueue_.AllocTensor<uint32_t>();
        Load(first, ordinal * macro_, currentRows);
        inputQueue_.EnQue(first);
        uint64_t previousRow = 0U;
        uint32_t previousRows = 0U;
        for (uint64_t step = 0U; step < macros; ++step) {
            const bool hasNext = step + 1U < macros;
            const uint64_t nextOrdinal = ordinal + 1U == macros ? 0U : ordinal + 1U;
            const uint32_t nextRows = hasNext ? RowsAt(nextOrdinal) : 0U;
            if (hasNext) {
                auto next = inputQueue_.AllocTensor<uint32_t>();
                Load(next, nextOrdinal * macro_, nextRows);
                inputQueue_.EnQue(next);
            }
            auto input = inputQueue_.DeQue<uint32_t>();
            for (uint32_t row = 0U; row < currentRows; row += 4U) {
                if (previousRows != 0U) { Store(previousRow, previousRows); }
                const uint32_t count = currentRows - row < 4U ? currentRows - row : 4U;
                auto output = outputQueue_.AllocTensor<uint32_t>();
                Gatherb(expanded_, input, indices_[row * blocks_],
                        static_cast<uint8_t>(count * blocks_ / 8U), GatherRepeatParams(1, 8));
                PipeBarrier<PIPE_V>();
                uint64_t retained = 0U;
                GatherMask(output, expanded_, masks_[(row % 8U) * blocks_].ReinterpretCast<uint32_t>(),
                           true, count * blocks_ * 8U, GatherMaskParams(1, 1, 8, 0), retained);
                PipeBarrier<PIPE_V>();
                outputQueue_.EnQue(output);
                previousRow = ordinal * macro_ + row;
                previousRows = count;
            }
            inputQueue_.FreeTensor(input);
            ordinal = nextOrdinal;
            currentRows = nextRows;
        }
        if (previousRows != 0U) { Store(previousRow, previousRows); }
    }

private:
    __aicore__ inline uint32_t Align32(uint32_t x) { return (x + 31U) & ~31U; }
    __aicore__ inline uint32_t RowsAt(uint64_t ordinal) {
        const uint64_t remaining = rows_ - ordinal * macro_;
        return static_cast<uint32_t>(remaining < macro_ ? remaining : macro_);
    }
    __aicore__ inline void Load(LocalTensor<uint32_t> local, uint64_t row, uint32_t rows) {
        const DataCopyPadExtParams<uint32_t> noPad(false, 0, 0, 0U);
        for (uint32_t i = 0U; i < count_; ++i) {
            input_.SetGlobalBuffer(pointers_[i]);
            input_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            DataCopyPad(local[bases_[i] / 4U], input_[(rowStart_ + row) * widths_[i] / 4U],
                        DataCopyExtParams(1, rows * widths_[i], 0, 0, 0), noPad);
        }
    }
    __aicore__ inline void Store(uint64_t row, uint32_t rows) {
        auto output = outputQueue_.DeQue<uint32_t>();
        DataCopyPad(output_[((rowStart_ + row) * rowBytes_ + prefix_) / 4U], output,
                    DataCopyExtParams(rows, bytes_, (pitch_ - Align32(bytes_)) / 32U, rowBytes_ - bytes_, 0));
        outputQueue_.FreeTensor(output);
    }
    TPipe pipe_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    TBuf<QuePosition::VECCALC> indexBuffer_, maskBuffer_, expandedBuffer_;
    LocalTensor<uint32_t> indices_, expanded_;
    LocalTensor<uint8_t> masks_;
    ListTensorDesc list_;
    GlobalTensor<uint32_t> input_, output_;
    __gm__ uint32_t* pointers_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint32_t widths_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS], bases_[S9_CONCAT_STREAM_MAX_GROUP_INPUTS];
    uint64_t rows_, rowStart_;
    uint32_t group_, rowBytes_, bytes_, pitch_, macro_, prefix_, count_, blocks_;
};

}

extern "C" __global__ __aicore__ void concat(
    GM_ADDR inputs, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ConcatTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1", ConcatTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 2", ConcatManyTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 3", ConcatLinearTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 4", PackedLongKey4Tiling);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 15", ConcatRowsTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 24", ConcatRowCompactTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 30", ConcatFlatTilingData);
    if (TILING_KEY_IS(30)) {
        KernelConcatDirectFlat<DTYPE_INPUTS> op;
        op.Init(inputs, y, tiling); op.Process(); return;
    }

    if (TILING_KEY_IS(24)) {
        if constexpr (sizeof(DTYPE_INPUTS) == 4U) {
            KernelConcatInplaceCompact op;
            op.Init(inputs, y, tiling);
            if constexpr (ConcatUsePhasedCompact<DTYPE_INPUTS>::value) {
                op.ProcessPhased();
            } else {
                op.Process();
            }
        }
        return;
    }
    if (TILING_KEY_IS(1)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, tilingData, tiling);
        KernelConcat<DTYPE_INPUTS> op;
        op.Init(inputs, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatManyTilingData, tilingData, tiling);
        KernelConcat<DTYPE_INPUTS> op;
        op.Init(inputs, y, tilingData.base);
        op.ProcessManyGrid(tilingData);
    } else if (TILING_KEY_IS(3)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatLinearTilingData, tilingData, tiling);
        KernelConcat<DTYPE_INPUTS> op;
        op.Init(inputs, y, tilingData.base);
        op.ProcessBalancedLinear(tilingData);
    } else if (TILING_KEY_IS(15)) {
        KernelConcatDirectOwner<DTYPE_INPUTS> op;
        op.Init(inputs, y, tiling);
        op.Process();
    } else if (TILING_KEY_IS(4)) {
        const uint32_t policy =
            reinterpret_cast<__gm__ ConcatStreamingTilingData*>(tiling)->bufferPolicy;
        if constexpr (ConcatPackedKey4<DTYPE_INPUTS>::enabled) {
            if (policy == 14U) {
                KernelConcatPackedBlocks op;
                op.Init(inputs, y, tiling);
                op.Process();
                return;
            }
        }
        if constexpr (sizeof(DTYPE_INPUTS) == 2U) {
            if (policy == 18U) {
                KernelConcatSmallResident<uint16_t> op;
                op.Init(inputs, y, tiling);
                op.Process();
            } else if (policy == 9U || policy == 12U) {
                KernelConcatWideStatic<uint16_t> op;
                op.Init(inputs, y, tiling);
                op.Process();
            } else if (policy == 2U || policy == 3U || policy == 8U) {
                KernelConcatStreaming<uint16_t, S9_CONCAT_STREAM_SHORT_MICRO_ROWS> op;
                op.Init(inputs, y, tiling);
                op.Process();
            } else {
                KernelConcatStreaming<uint16_t> op;
                op.Init(inputs, y, tiling);
                op.Process();
            }
        } else if constexpr (sizeof(DTYPE_INPUTS) == 4U) {
            if (policy == 9U || policy == 12U) {
                KernelConcatWideStatic<uint32_t> op;
                op.Init(inputs, y, tiling);
                op.Process();
            } else if (policy == 2U || policy == 3U || policy == 8U) {
                KernelConcatStreaming<uint32_t, S9_CONCAT_STREAM_SHORT_MICRO_ROWS> op;
                op.Init(inputs, y, tiling);
                op.Process();
            } else {
                KernelConcatStreaming<uint32_t> op;
                op.Init(inputs, y, tiling);
                op.Process();
            }
        }
    }
}
