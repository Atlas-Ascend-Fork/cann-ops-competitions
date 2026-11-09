#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kMaxInputs = 256U;
constexpr uint32_t kRows = 128U;
constexpr uint32_t kColumns = 256U;
constexpr uint32_t kRowsPerCore = 16U;
constexpr uint32_t kOutputElements = kRows * kColumns;
constexpr uint32_t kStagingElements = 5376U;
constexpr uint32_t kGenericTileBytes = 32U * 1024U;
// A row-batched path keeps one complete source fragment per logical output
// row in UB.  64 KiB is the largest conservative allocation that still leaves
// room for the runtime bookkeeping on 910B.
constexpr uint32_t kRowBatchBytes = 96U * 1024U;
// Keep two row batches resident so MTE2 can fetch the next batch while MTE3
// drains the current one.  The previous depth-one queue serialized every
// input and output DMA pair and left one of the two engines idle.
constexpr uint32_t kRowBatchQueueDepth = 2U;

struct ConcatKernelTilingData {
    uint64_t totalElements;
    uint64_t inner;
    uint64_t outputAxis;
    uint32_t inputCount;
    uint32_t dtypeBytes;
    uint16_t inputAxis[kMaxInputs];
    uint32_t inputRowStrideBytes[kMaxInputs];
};

static_assert(sizeof(ConcatKernelTilingData) == 1568U,
              "Concat host/kernel tiling ABI must match");

template <typename T>
__aicore__ inline __gm__ T *GetDynamicInputAddress(
    GM_ADDR inputs,
    uint32_t index)
{
    __gm__ uint64_t *descriptor =
        reinterpret_cast<__gm__ uint64_t *>(inputs);
    const uint64_t pointerOffset = *descriptor;
    __gm__ uint64_t *pointers = descriptor + (pointerOffset >> 3U);
    return reinterpret_cast<__gm__ T *>(*(pointers + index));
}

template <typename T>
__aicore__ inline void CopyGenericSpan(
    GM_ADDR inputs,
    GlobalTensor<T> &outputGm,
    LocalTensor<T> &staging,
    uint32_t inputIndex,
    uint64_t sourceOffset,
    uint64_t destinationOffset,
    uint64_t elementCount)
{
    constexpr uint32_t kTileElements = kGenericTileBytes / sizeof(T);
    constexpr uint32_t kElementsPerBlock = 32U / sizeof(T);
    GlobalTensor<T> inputGm;
    inputGm.SetGlobalBuffer(GetDynamicInputAddress<T>(inputs, inputIndex));

    while (elementCount != 0U) {
        const uint32_t copyElements = static_cast<uint32_t>(
            elementCount > kTileElements ? kTileElements : elementCount);
        const uint32_t alignedElements =
            (copyElements + kElementsPerBlock - 1U) /
            kElementsPerBlock * kElementsPerBlock;
        const DataCopyExtParams copyParams{
            1U,
            static_cast<uint32_t>(copyElements * sizeof(T)),
            0U,
            0U,
            0U};
        const DataCopyPadExtParams<T> padParams{
            true,
            0U,
            static_cast<uint8_t>(alignedElements - copyElements),
            static_cast<T>(0)};
        DataCopyPad(staging, inputGm[sourceOffset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(outputGm[destinationOffset], staging, copyParams);
        PipeBarrier<PIPE_ALL>();

        sourceOffset += copyElements;
        destinationOffset += copyElements;
        elementCount -= copyElements;
    }
}

template <typename T>
__aicore__ inline void ConcatGenericTyped(
    GM_ADDR inputs,
    GM_ADDR output,
    __gm__ const ConcatKernelTilingData *data)
{
    if (data->totalElements == 0U || data->inner == 0U ||
        data->outputAxis == 0U) {
        return;
    }

    constexpr uint64_t kElementsPerBlock = 32U / sizeof(T);
    const uint64_t transferBlocks =
        (data->totalElements + kElementsPerBlock - 1U) /
        kElementsPerBlock;
    const uint64_t blocksPerCore =
        (transferBlocks + GetBlockNum() - 1U) / GetBlockNum();
    const uint64_t firstElement =
        static_cast<uint64_t>(GetBlockIdx()) * blocksPerCore *
        kElementsPerBlock;
    uint64_t endElement =
        (static_cast<uint64_t>(GetBlockIdx()) + 1U) * blocksPerCore *
        kElementsPerBlock;
    if (endElement > data->totalElements) {
        endElement = data->totalElements;
    }
    if (firstElement >= endElement) {
        return;
    }

    GlobalTensor<T> outputGm;
    outputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ T *>(output), data->totalElements);
    TPipe pipe;
    TBuf<TPosition::VECCALC> stagingBuffer;
    pipe.InitBuffer(stagingBuffer, kGenericTileBytes);
    LocalTensor<T> staging = stagingBuffer.Get<T>();

    const uint64_t outputRun = data->outputAxis * data->inner;
    uint64_t outputIndex = firstElement;
    while (outputIndex < endElement) {
        const uint64_t outerCoordinate = outputIndex / outputRun;
        const uint64_t withinOutputRun = outputIndex % outputRun;
        uint64_t inputRunBegin = 0U;
        bool copied = false;
        for (uint32_t inputIndex = 0U;
             inputIndex < data->inputCount;
             ++inputIndex) {
            const uint64_t inputRunElements =
                static_cast<uint64_t>(data->inputAxis[inputIndex]) *
                data->inner;
            const uint64_t inputRunEnd =
                inputRunBegin + inputRunElements;
            if (withinOutputRun < inputRunEnd) {
                const uint64_t sourceOffset =
                    outerCoordinate * inputRunElements +
                    (withinOutputRun - inputRunBegin);
                uint64_t copyElements = inputRunEnd - withinOutputRun;
                if (copyElements > endElement - outputIndex) {
                    copyElements = endElement - outputIndex;
                }
                CopyGenericSpan(
                    inputs,
                    outputGm,
                    staging,
                    inputIndex,
                    sourceOffset,
                    outputIndex,
                    copyElements);
                outputIndex += copyElements;
                copied = true;
                break;
            }
            inputRunBegin = inputRunEnd;
        }
        if (!copied) {
            return;
        }
    }
}

__aicore__ inline void ConcatGeneric(
    GM_ADDR inputs,
    GM_ADDR output,
    __gm__ const ConcatKernelTilingData *data)
{
    if (data->totalElements == 0U) {
        return;
    }
    if (data->dtypeBytes == 1U) {
        ConcatGenericTyped<int8_t>(inputs, output, data);
    } else if (data->dtypeBytes == 2U) {
        ConcatGenericTyped<uint16_t>(inputs, output, data);
    } else {
        ConcatGenericTyped<uint32_t>(inputs, output, data);
    }
}

// The row-batched kernels use an int8 view so that one implementation serves
// all supported element types.  DataCopyPad is needed for arbitrary byte
// lengths, but it carries extra descriptor/padding work.  Once the fragment,
// source pitch, and destination pitch are all 32-byte aligned, the regular
// DataCopy instruction describes the same 2-D transfer more cheaply.
__aicore__ inline void CopyAlignedOrPaddedInput(
    LocalTensor<int8_t> destination,
    GlobalTensor<int8_t> source,
    uint64_t sourceOffset,
    uint16_t rows,
    uint32_t bytes,
    uint64_t sourceGapBytes,
    bool aligned)
{
    if (aligned) {
        const DataCopyParams params{
            rows,
            static_cast<uint16_t>(bytes / 32U),
            static_cast<uint16_t>(sourceGapBytes / 32U),
            0U};
        DataCopy(destination, source[sourceOffset], params);
        return;
    }
    const DataCopyExtParams params{
        rows,
        bytes,
        static_cast<uint32_t>(sourceGapBytes),
        0U,
        0U};
    const DataCopyPadExtParams<int8_t> padParams{
        false, 0U, 0U, static_cast<int8_t>(0)};
    DataCopyPad(destination, source[sourceOffset], params, padParams);
}

__aicore__ inline void CopyAlignedOrPaddedOutput(
    GlobalTensor<int8_t> destination,
    LocalTensor<int8_t> source,
    uint64_t destinationOffset,
    uint16_t rows,
    uint32_t bytes,
    uint64_t destinationGapBytes,
    bool aligned)
{
    if (aligned) {
        const DataCopyParams params{
            rows,
            static_cast<uint16_t>(bytes / 32U),
            0U,
            static_cast<uint16_t>(destinationGapBytes / 32U)};
        DataCopy(destination[destinationOffset], source, params);
        return;
    }
    const DataCopyExtParams params{
        rows,
        bytes,
        0U,
        static_cast<uint32_t>(destinationGapBytes),
        0U};
    DataCopyPad(destination[destinationOffset], source, params);
}

__aicore__ inline void CopyAlignedOrPaddedLinearInput(
    LocalTensor<int8_t> destination,
    GlobalTensor<int8_t> source,
    uint64_t sourceOffset,
    uint32_t bytes,
    bool aligned)
{
    if (aligned) {
        const DataCopyParams params{
            1U, static_cast<uint16_t>(bytes / 32U), 0U, 0U};
        DataCopy(destination, source[sourceOffset], params);
        return;
    }
    const DataCopyExtParams params{1U, bytes, 0U, 0U, 0U};
    const DataCopyPadExtParams<int8_t> padParams{
        false, 0U, 0U, static_cast<int8_t>(0)};
    DataCopyPad(destination, source[sourceOffset], params, padParams);
}

__aicore__ inline void CopyAlignedOrPaddedLinearOutput(
    GlobalTensor<int8_t> destination,
    LocalTensor<int8_t> source,
    uint64_t destinationOffset,
    uint32_t bytes,
    bool aligned)
{
    if (aligned) {
        const DataCopyParams params{
            1U, static_cast<uint16_t>(bytes / 32U), 0U, 0U};
        DataCopy(destination[destinationOffset], source, params);
        return;
    }
    const DataCopyExtParams params{1U, bytes, 0U, 0U, 0U};
    DataCopyPad(destination[destinationOffset], source, params);
}

// A tensor list produced by split() often consists of adjacent views into one
// dense base tensor.  When every view has the full output row pitch and the
// first-row addresses form one contiguous span, the whole concat is already
// laid out in output order in GM.  Detect that case on device so that the
// row-batch path can issue one 2-D stream instead of restarting DMA for every
// narrow input fragment.  The predicate is conservative: unrelated tensors
// (or views with padding between rows) continue through the normal path.
__aicore__ inline bool InputRunsFormContiguousRows(
    GM_ADDR inputs,
    __gm__ const ConcatKernelTilingData *data,
    uint64_t outputRunBytes,
    uint32_t *firstNonzeroIndex)
{
    // A one-input concat is already handled efficiently by the regular
    // row-batch stream.  The contiguous-span variant adds an extra view
    // discovery/2-D setup cost for that case and was measurably worse on the
    // large single-input benchmark.  Reserve this shortcut for true
    // multi-input split views, where it can remove many fragment launches.
    if (data->inputCount < 2U) {
        return false;
    }
    uint64_t previousEnd = 0U;
    bool haveInput = false;
    uint32_t firstIndex = 0U;
    for (uint32_t inputIndex = 0U;
         inputIndex < data->inputCount;
         ++inputIndex) {
        const uint64_t inputRunBytes = static_cast<uint64_t>(
            data->inputAxis[inputIndex]) * data->inner * data->dtypeBytes;
        if (inputRunBytes == 0U) {
            continue;
        }
        const uint64_t sourceRowStrideBytes =
            data->inputRowStrideBytes[inputIndex] != 0U
                ? static_cast<uint64_t>(data->inputRowStrideBytes[inputIndex])
                : inputRunBytes;
        if (sourceRowStrideBytes != outputRunBytes) {
            return false;
        }
        const uint64_t address = reinterpret_cast<uint64_t>(
            GetDynamicInputAddress<int8_t>(inputs, inputIndex));
        if (haveInput && address != previousEnd) {
            return false;
        }
        if (!haveInput) {
            firstIndex = inputIndex;
        }
        previousEnd = address + inputRunBytes;
        haveInput = true;
    }
    if (haveInput && firstNonzeroIndex != nullptr) {
        *firstNonzeroIndex = firstIndex;
    }
    return haveInput;
}

// Concatenate along an arbitrary axis by treating each outer coordinate as a
// logical row.  Unlike ConcatGeneric, all rows in a batch are described by a
// single strided DMA, so tiny fragments do not pay one pair of barriers per
// row.  The source and destination segments are disjoint for every input.
__aicore__ inline void ConcatRowBatch(
    GM_ADDR inputs,
    GM_ADDR output,
    __gm__ const ConcatKernelTilingData *data)
{
    if (data->totalElements == 0U || data->inner == 0U ||
        data->outputAxis == 0U || data->dtypeBytes == 0U) {
        return;
    }

    const uint64_t outputRunElements = data->outputAxis * data->inner;
    const uint64_t outer = data->totalElements / outputRunElements;
    const uint64_t coreCount = static_cast<uint64_t>(GetBlockNum());
    const uint64_t coreIndex = static_cast<uint64_t>(GetBlockIdx());
    const uint64_t rowsPerCore = outer / coreCount;
    const uint64_t extraRows = outer % coreCount;
    const uint64_t firstRow =
        coreIndex * rowsPerCore +
        (coreIndex < extraRows ? coreIndex : extraRows);
    const uint64_t endRow =
        firstRow + rowsPerCore + (coreIndex < extraRows ? 1U : 0U);
    if (firstRow >= endRow) {
        return;
    }

    const uint64_t outputRunBytes =
        outputRunElements * static_cast<uint64_t>(data->dtypeBytes);
    GlobalTensor<int8_t> outputGm;
    outputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t *>(output),
        data->totalElements * static_cast<uint64_t>(data->dtypeBytes));

    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT,
             kRowBatchQueueDepth> copyQueue;
    pipe.InitBuffer(copyQueue, kRowBatchQueueDepth, kRowBatchBytes);

    // split() along any axis of a dense tensor preserves the full output-row
    // pitch.  If its input views are adjacent, copy all rows as one stream;
    // this removes the per-input DMA/queue overhead which otherwise dominates
    // large hidden cases made from many narrow split views.
    uint32_t contiguousFirstInput = 0U;
    if (InputRunsFormContiguousRows(
            inputs, data, outputRunBytes, &contiguousFirstInput)) {
        GlobalTensor<int8_t> inputGm;
        inputGm.SetGlobalBuffer(
            GetDynamicInputAddress<int8_t>(inputs, contiguousFirstInput));
        uint64_t columnBytes = 0U;
        while (columnBytes < outputRunBytes) {
            const uint32_t copyColumns = static_cast<uint32_t>(
                outputRunBytes - columnBytes > kRowBatchBytes
                    ? kRowBatchBytes
                    : outputRunBytes - columnBytes);
            const uint32_t alignedColumns =
                (copyColumns + 31U) / 32U * 32U;
            uint32_t maxRows = kRowBatchBytes / alignedColumns;
            if (maxRows == 0U) {
                maxRows = 1U;
            }
            if (maxRows > 4095U) {
                maxRows = 4095U;
            }
            const bool alignedTransfer =
                (copyColumns % 32U) == 0U &&
                (outputRunBytes % 32U) == 0U &&
                (columnBytes % 32U) == 0U &&
                (outputRunBytes - copyColumns) <= 65535U * 32U;
            uint64_t row = firstRow;
            if (row < endRow) {
                uint32_t copyRows = static_cast<uint32_t>(
                    endRow - row > maxRows ? maxRows : endRow - row);
                uint64_t batchRow = row;
                LocalTensor<int8_t> first =
                    copyQueue.AllocTensor<int8_t>();
                CopyAlignedOrPaddedInput(
                    first,
                    inputGm,
                    batchRow * outputRunBytes + columnBytes,
                    static_cast<uint16_t>(copyRows),
                    copyColumns,
                    outputRunBytes - copyColumns,
                    alignedTransfer);
                copyQueue.EnQue(first);
                row += copyRows;
                while (true) {
                    LocalTensor<int8_t> local =
                        copyQueue.DeQue<int8_t>();
                    const bool hasNext = row < endRow;
                    uint64_t nextRow = row;
                    uint32_t nextRows = 0U;
                    if (hasNext) {
                        nextRows = static_cast<uint32_t>(
                            endRow - row > maxRows ? maxRows : endRow - row);
                        LocalTensor<int8_t> next =
                            copyQueue.AllocTensor<int8_t>();
                        CopyAlignedOrPaddedInput(
                            next,
                            inputGm,
                            nextRow * outputRunBytes + columnBytes,
                            static_cast<uint16_t>(nextRows),
                            copyColumns,
                            outputRunBytes - copyColumns,
                            alignedTransfer);
                        copyQueue.EnQue(next);
                        row += nextRows;
                    }
                    CopyAlignedOrPaddedOutput(
                        outputGm,
                        local,
                        batchRow * outputRunBytes + columnBytes,
                        static_cast<uint16_t>(copyRows),
                        copyColumns,
                        outputRunBytes - copyColumns,
                        alignedTransfer);
                    copyQueue.FreeTensor(local);
                    if (!hasNext) {
                        break;
                    }
                    batchRow = nextRow;
                    copyRows = nextRows;
                }
            }
            columnBytes += copyColumns;
        }
        return;
    }

    uint64_t destinationColumnBytes = 0U;
    for (uint32_t inputIndex = 0U;
         inputIndex < data->inputCount;
         ++inputIndex) {
        const uint64_t inputRunElements =
            static_cast<uint64_t>(data->inputAxis[inputIndex]) * data->inner;
        const uint64_t inputRunBytes =
            inputRunElements * static_cast<uint64_t>(data->dtypeBytes);
        if (inputRunBytes == 0U) {
            continue;
        }
        const uint64_t sourceRowStrideBytes =
            data->inputRowStrideBytes[inputIndex] != 0U
                ? static_cast<uint64_t>(data->inputRowStrideBytes[inputIndex])
                : inputRunBytes;
        if (sourceRowStrideBytes < inputRunBytes) {
            return;
        }

        GlobalTensor<int8_t> inputGm;
        inputGm.SetGlobalBuffer(
            GetDynamicInputAddress<int8_t>(inputs, inputIndex));
        uint64_t columnBytes = 0U;
        while (columnBytes < inputRunBytes) {
            const uint32_t copyColumns = static_cast<uint32_t>(
                inputRunBytes - columnBytes > kRowBatchBytes
                    ? kRowBatchBytes
                    : inputRunBytes - columnBytes);
            const uint32_t alignedColumns =
                (copyColumns + 31U) / 32U * 32U;
            uint32_t maxRows = kRowBatchBytes / alignedColumns;
            if (maxRows == 0U) {
                maxRows = 1U;
            }
            if (maxRows > 4095U) {
                maxRows = 4095U;
            }

            const bool alignedTransfer =
                (copyColumns % 32U) == 0U &&
                (inputRunBytes % 32U) == 0U &&
                (outputRunBytes % 32U) == 0U &&
                (columnBytes % 32U) == 0U &&
                ((destinationColumnBytes + columnBytes) % 32U) == 0U &&
                // DataCopyParams expresses the inter-row gap in 32-byte
                // units.  A merely bounded but non-32B source pitch would
                // be truncated by the division in CopyAlignedOrPaddedInput.
                (sourceRowStrideBytes % 32U) == 0U &&
                (sourceRowStrideBytes - copyColumns) <= 65535U * 32U &&
                (outputRunBytes - copyColumns) <= 65535U * 32U;

            uint64_t row = firstRow;
            if (row < endRow) {
                uint32_t copyRows = static_cast<uint32_t>(
                    endRow - row > maxRows ? maxRows : endRow - row);
                uint64_t batchRow = row;
                LocalTensor<int8_t> first =
                    copyQueue.AllocTensor<int8_t>();
                CopyAlignedOrPaddedInput(
                    first,
                    inputGm,
                    batchRow * sourceRowStrideBytes + columnBytes,
                    static_cast<uint16_t>(copyRows),
                    copyColumns,
                    sourceRowStrideBytes - copyColumns,
                    alignedTransfer);
                copyQueue.EnQue(first);
                row += copyRows;

                while (true) {
                    LocalTensor<int8_t> local =
                        copyQueue.DeQue<int8_t>();
                    const bool hasNext = row < endRow;
                    uint64_t nextRow = row;
                    uint32_t nextRows = 0U;
                    if (hasNext) {
                        nextRows = static_cast<uint32_t>(
                            endRow - row > maxRows ? maxRows : endRow - row);
                        LocalTensor<int8_t> next =
                            copyQueue.AllocTensor<int8_t>();
                        CopyAlignedOrPaddedInput(
                            next,
                            inputGm,
                            nextRow * sourceRowStrideBytes + columnBytes,
                            static_cast<uint16_t>(nextRows),
                            copyColumns,
                            sourceRowStrideBytes - copyColumns,
                            alignedTransfer);
                        copyQueue.EnQue(next);
                        row += nextRows;
                    }

                    CopyAlignedOrPaddedOutput(
                        outputGm,
                        local,
                        batchRow * outputRunBytes +
                            destinationColumnBytes + columnBytes,
                        static_cast<uint16_t>(copyRows),
                        copyColumns,
                        outputRunBytes - copyColumns,
                        alignedTransfer);
                    copyQueue.FreeTensor(local);
                    if (!hasNext) {
                        break;
                    }
                    batchRow = nextRow;
                    copyRows = nextRows;
                }
            }
            columnBytes += copyColumns;
        }
        destinationColumnBytes += inputRunBytes;
    }
}

// For a long list of narrow inputs, assigning rows to every core makes each
// core walk the full input list.  Assigning disjoint input ranges instead lets
// the cores copy different source tensors concurrently while preserving exact
// output ownership.  This is especially effective for the many-input cases
// where each fragment is only one or a few elements wide.
__aicore__ inline void ConcatInputParallel(
    GM_ADDR inputs,
    GM_ADDR output,
    __gm__ const ConcatKernelTilingData *data)
{
    if (data->totalElements == 0U || data->inner == 0U ||
        data->outputAxis == 0U || data->dtypeBytes == 0U) {
        return;
    }
    const uint64_t outputRunElements = data->outputAxis * data->inner;
    const uint64_t outer = data->totalElements / outputRunElements;
    const uint64_t coreCount = static_cast<uint64_t>(GetBlockNum());
    const uint64_t coreIndex = static_cast<uint64_t>(GetBlockIdx());
    // Do not assign an input owner before checking the contiguous-row mode.
    // An owner can legitimately have no whole input when there are fewer
    // fragments than blocks (or one fragment is much wider than the target).
    // Such a block can still own output rows in the contiguous mode.
    uint32_t firstInput = 0U;
    uint32_t lastInput = 0U;

    const uint64_t outputRunBytes =
        outputRunElements * static_cast<uint64_t>(data->dtypeBytes);
    const uint64_t rowsPerCore = outer / coreCount;
    const uint64_t extraRows = outer % coreCount;
    const uint64_t firstRow =
        coreIndex * rowsPerCore +
        (coreIndex < extraRows ? coreIndex : extraRows);
    const uint64_t endRow =
        firstRow + rowsPerCore + (coreIndex < extraRows ? 1U : 0U);
    if (firstRow >= endRow) {
        return;
    }
    GlobalTensor<int8_t> outputGm;
    outputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ int8_t *>(output),
        data->totalElements * static_cast<uint64_t>(data->dtypeBytes));
    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT,
             kRowBatchQueueDepth> copyQueue;
    pipe.InitBuffer(copyQueue, kRowBatchQueueDepth, kRowBatchBytes);

    // For adjacent split views the input-parallel key should use the same
    // whole-row stream as the row-parallel key; otherwise each core would own
    // only a subset of input fragments and lose the contiguous-span win.
    uint32_t contiguousFirstInput = 0U;
    if (InputRunsFormContiguousRows(
            inputs, data, outputRunBytes, &contiguousFirstInput)) {
        GlobalTensor<int8_t> inputGm;
        inputGm.SetGlobalBuffer(
            GetDynamicInputAddress<int8_t>(inputs, contiguousFirstInput));
        uint64_t columnBytes = 0U;
        while (columnBytes < outputRunBytes) {
            const uint32_t copyColumns = static_cast<uint32_t>(
                outputRunBytes - columnBytes > kRowBatchBytes
                    ? kRowBatchBytes
                    : outputRunBytes - columnBytes);
            const uint32_t alignedColumns =
                (copyColumns + 31U) / 32U * 32U;
            uint32_t maxRows = kRowBatchBytes / alignedColumns;
            if (maxRows == 0U) {
                maxRows = 1U;
            }
            if (maxRows > 4095U) {
                maxRows = 4095U;
            }
            const bool alignedTransfer =
                (copyColumns % 32U) == 0U &&
                (outputRunBytes % 32U) == 0U &&
                (columnBytes % 32U) == 0U &&
                (outputRunBytes - copyColumns) <= 65535U * 32U;
            uint64_t row = firstRow;
            if (row < endRow) {
                uint32_t copyRows = static_cast<uint32_t>(
                    endRow - row > maxRows ? maxRows : endRow - row);
                uint64_t batchRow = row;
                LocalTensor<int8_t> first =
                    copyQueue.AllocTensor<int8_t>();
                CopyAlignedOrPaddedInput(
                    first,
                    inputGm,
                    batchRow * outputRunBytes + columnBytes,
                    static_cast<uint16_t>(copyRows),
                    copyColumns,
                    outputRunBytes - copyColumns,
                    alignedTransfer);
                copyQueue.EnQue(first);
                row += copyRows;
                while (true) {
                    LocalTensor<int8_t> local =
                        copyQueue.DeQue<int8_t>();
                    const bool hasNext = row < endRow;
                    uint64_t nextRow = row;
                    uint32_t nextRows = 0U;
                    if (hasNext) {
                        nextRows = static_cast<uint32_t>(
                            endRow - row > maxRows ? maxRows : endRow - row);
                        LocalTensor<int8_t> next =
                            copyQueue.AllocTensor<int8_t>();
                        CopyAlignedOrPaddedInput(
                            next,
                            inputGm,
                            nextRow * outputRunBytes + columnBytes,
                            static_cast<uint16_t>(nextRows),
                            copyColumns,
                            outputRunBytes - copyColumns,
                            alignedTransfer);
                        copyQueue.EnQue(next);
                        row += nextRows;
                    }
                    CopyAlignedOrPaddedOutput(
                        outputGm,
                        local,
                        batchRow * outputRunBytes + columnBytes,
                        static_cast<uint16_t>(copyRows),
                        copyColumns,
                        outputRunBytes - copyColumns,
                        alignedTransfer);
                    copyQueue.FreeTensor(local);
                    if (!hasNext) {
                        break;
                    }
                    batchRow = nextRow;
                    copyRows = nextRows;
                }
            }
            columnBytes += copyColumns;
        }
        return;
    }

    // The non-contiguous mode owns complete input fragments.  It is safe for
    // a block to have no fragment here because every output byte belongs to
    // exactly one owner selected below.  This calculation deliberately comes
    // after the contiguous-mode return above.
    const uint64_t totalBytes = outputRunBytes;
    const uint64_t targetBytes =
        (totalBytes + coreCount - 1U) / coreCount;
    uint64_t prefixBytes = 0U;
    for (uint64_t owner = 0U; owner < coreCount; ++owner) {
        firstInput = lastInput;
        const uint64_t ownerLimit =
            owner + 1U == coreCount
                ? totalBytes
                : (owner + 1U) * targetBytes;
        while (lastInput < data->inputCount && prefixBytes < ownerLimit) {
            prefixBytes += static_cast<uint64_t>(
                data->inputAxis[lastInput]) * data->inner * data->dtypeBytes;
            ++lastInput;
        }
        if (owner == coreIndex) {
            break;
        }
    }
    if (firstInput >= lastInput) {
        return;
    }

    uint64_t destinationColumnBytes = 0U;
    for (uint32_t i = 0U; i < firstInput; ++i) {
        destinationColumnBytes += static_cast<uint64_t>(data->inputAxis[i]) *
                                  data->inner * data->dtypeBytes;
    }
    for (uint32_t inputIndex = firstInput;
         inputIndex < lastInput;
         ++inputIndex) {
        const uint64_t inputRunBytes = static_cast<uint64_t>(
            data->inputAxis[inputIndex]) * data->inner * data->dtypeBytes;
        if (inputRunBytes == 0U) {
            destinationColumnBytes += inputRunBytes;
            continue;
        }
        const uint64_t sourceRowStrideBytes =
            data->inputRowStrideBytes[inputIndex] != 0U
                ? static_cast<uint64_t>(data->inputRowStrideBytes[inputIndex])
                : inputRunBytes;
        if (sourceRowStrideBytes < inputRunBytes) {
            return;
        }
        GlobalTensor<int8_t> inputGm;
        inputGm.SetGlobalBuffer(
            GetDynamicInputAddress<int8_t>(inputs, inputIndex));
        uint64_t columnBytes = 0U;
        while (columnBytes < inputRunBytes) {
            const uint32_t copyColumns = static_cast<uint32_t>(
                inputRunBytes - columnBytes > kRowBatchBytes
                    ? kRowBatchBytes
                    : inputRunBytes - columnBytes);
            const uint32_t alignedColumns =
                (copyColumns + 31U) / 32U * 32U;
            uint32_t maxRows = kRowBatchBytes / alignedColumns;
            if (maxRows == 0U) {
                maxRows = 1U;
            }
            if (maxRows > 4095U) {
                maxRows = 4095U;
            }
            const bool alignedTransfer =
                (copyColumns % 32U) == 0U &&
                (inputRunBytes % 32U) == 0U &&
                (outputRunBytes % 32U) == 0U &&
                (columnBytes % 32U) == 0U &&
                ((destinationColumnBytes + columnBytes) % 32U) == 0U &&
                // The source gap is encoded in 32-byte units by DataCopy.
                // Keep an unaligned physical row pitch on the exact-byte
                // DataCopyPad path instead of truncating it below.
                (sourceRowStrideBytes % 32U) == 0U &&
                (sourceRowStrideBytes - copyColumns) <= 65535U * 32U &&
                (outputRunBytes - copyColumns) <= 65535U * 32U;
            uint64_t row = 0U;
            if (row < outer) {
                uint32_t copyRows = static_cast<uint32_t>(
                    outer - row > maxRows ? maxRows : outer - row);
                uint64_t batchRow = row;
                LocalTensor<int8_t> first =
                    copyQueue.AllocTensor<int8_t>();
                CopyAlignedOrPaddedInput(
                    first,
                    inputGm,
                    batchRow * sourceRowStrideBytes + columnBytes,
                    static_cast<uint16_t>(copyRows),
                    copyColumns,
                    sourceRowStrideBytes - copyColumns,
                    alignedTransfer);
                copyQueue.EnQue(first);
                row += copyRows;

                while (true) {
                    LocalTensor<int8_t> local =
                        copyQueue.DeQue<int8_t>();
                    const bool hasNext = row < outer;
                    uint64_t nextRow = row;
                    uint32_t nextRows = 0U;
                    if (hasNext) {
                        nextRows = static_cast<uint32_t>(
                            outer - row > maxRows ? maxRows : outer - row);
                        LocalTensor<int8_t> next =
                            copyQueue.AllocTensor<int8_t>();
                        CopyAlignedOrPaddedInput(
                            next,
                            inputGm,
                            nextRow * sourceRowStrideBytes + columnBytes,
                            static_cast<uint16_t>(nextRows),
                            copyColumns,
                            sourceRowStrideBytes - copyColumns,
                            alignedTransfer);
                        copyQueue.EnQue(next);
                        row += nextRows;
                    }

                    CopyAlignedOrPaddedOutput(
                        outputGm,
                        local,
                        batchRow * outputRunBytes +
                            destinationColumnBytes + columnBytes,
                        static_cast<uint16_t>(copyRows),
                        copyColumns,
                        outputRunBytes - copyColumns,
                        alignedTransfer);
                    copyQueue.FreeTensor(local);
                    if (!hasNext) {
                        break;
                    }
                    batchRow = nextRow;
                    copyRows = nextRows;
                }
            }
            columnBytes += copyColumns;
        }
        destinationColumnBytes += inputRunBytes;
    }
}

// When the concatenation axis is the leading axis (outer == 1), the logical
// output is simply the input tensors laid out back-to-back.  A row-batch
// transfer cannot expose parallelism in this case because there is only one
// row, while the standard Concat kernel may still split the operation into a
// large number of small tasks.  Partition the flat byte range across AIVs and
// stream each core's range through a double-buffered UB queue.  The range is
// allowed to cross input boundaries; each intersection is copied as a normal
// contiguous span, so no element-wise address arithmetic is needed on the
// device.
__aicore__ inline void ConcatLinear(
    GM_ADDR inputs,
    GM_ADDR output,
    __gm__ const ConcatKernelTilingData *data)
{
    if (data->totalElements == 0U || data->inner == 0U ||
        data->outputAxis == 0U || data->dtypeBytes == 0U) {
        return;
    }

    const uint64_t totalBytes =
        data->totalElements * static_cast<uint64_t>(data->dtypeBytes);
    const uint64_t coreCount = static_cast<uint64_t>(GetBlockNum());
    const uint64_t coreIndex = static_cast<uint64_t>(GetBlockIdx());
    // Partition whole 32-byte blocks.  This keeps every core's destination
    // range aligned, allowing the regular DataCopy instruction for the large
    // contiguous portions while the final short tail still uses DataCopyPad.
    const uint64_t totalBlocks = (totalBytes + 31U) / 32U;
    const uint64_t blocksPerCore =
        (totalBlocks + coreCount - 1U) / coreCount;
    const uint64_t firstByte = coreIndex * blocksPerCore * 32U;
    uint64_t endByte = (coreIndex + 1U) * blocksPerCore * 32U;
    if (endByte > totalBytes) {
        endByte = totalBytes;
    }
    if (firstByte >= endByte) {
        return;
    }

    GlobalTensor<int8_t> outputGm;
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(output),
                             totalBytes);

    TPipe pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT,
             kRowBatchQueueDepth> copyQueue;
    pipe.InitBuffer(copyQueue, kRowBatchQueueDepth, kRowBatchBytes);

    uint64_t inputBegin = 0U;
    for (uint32_t inputIndex = 0U;
         inputIndex < data->inputCount && inputBegin < endByte;
         ++inputIndex) {
        const uint64_t inputBytes = static_cast<uint64_t>(
            data->inputAxis[inputIndex]) * data->inner * data->dtypeBytes;
        const uint64_t inputEnd = inputBegin + inputBytes;
        const uint64_t segmentBegin =
            firstByte > inputBegin ? firstByte : inputBegin;
        const uint64_t segmentEnd = endByte < inputEnd ? endByte : inputEnd;
        if (segmentBegin < segmentEnd) {
            GlobalTensor<int8_t> inputGm;
            inputGm.SetGlobalBuffer(
                GetDynamicInputAddress<int8_t>(inputs, inputIndex));
            uint64_t segmentOffset = segmentBegin - inputBegin;
            uint64_t remaining = segmentEnd - segmentBegin;

            // Keep one transfer in flight while the next one is issued.  Use
            // plain DataCopy whenever both GM addresses and the transfer
            // length are aligned; otherwise retain the exact-length pad form.
            LocalTensor<int8_t> first = copyQueue.AllocTensor<int8_t>();
            uint32_t currentBytes = static_cast<uint32_t>(
                remaining > kRowBatchBytes ? kRowBatchBytes : remaining);
            bool currentInputAligned =
                (segmentOffset % 32U) == 0U &&
                (currentBytes % 32U) == 0U;
            CopyAlignedOrPaddedLinearInput(
                first, inputGm, segmentOffset, currentBytes,
                currentInputAligned);
            copyQueue.EnQue(first);
            segmentOffset += currentBytes;
            remaining -= currentBytes;
            uint64_t currentDestination = segmentBegin;
            bool currentOutputAligned =
                (currentDestination % 32U) == 0U &&
                (currentBytes % 32U) == 0U;

            while (true) {
                LocalTensor<int8_t> local = copyQueue.DeQue<int8_t>();
                const bool hasNext = remaining != 0U;
                uint32_t nextBytes = 0U;
                bool nextInputAligned = false;
                bool nextOutputAligned = false;
                if (hasNext) {
                    LocalTensor<int8_t> next =
                        copyQueue.AllocTensor<int8_t>();
                    nextBytes = static_cast<uint32_t>(
                        remaining > kRowBatchBytes ? kRowBatchBytes : remaining);
                    nextInputAligned =
                        (segmentOffset % 32U) == 0U &&
                        (nextBytes % 32U) == 0U;
                    CopyAlignedOrPaddedLinearInput(
                        next, inputGm, segmentOffset, nextBytes,
                        nextInputAligned);
                    copyQueue.EnQue(next);
                    segmentOffset += nextBytes;
                    remaining -= nextBytes;
                    nextOutputAligned =
                        ((currentDestination + currentBytes) % 32U) == 0U &&
                        (nextBytes % 32U) == 0U;
                }

                CopyAlignedOrPaddedLinearOutput(
                    outputGm, local, currentDestination, currentBytes,
                    currentOutputAligned && currentInputAligned);
                copyQueue.FreeTensor(local);
                if (!hasNext) {
                    break;
                }
                currentDestination += currentBytes;
                currentBytes = nextBytes;
                currentInputAligned = nextInputAligned;
                currentOutputAligned = nextOutputAligned;
            }
        }
        inputBegin = inputEnd;
    }
}

template <uint32_t Width, uint32_t LocalOffset>
__aicore__ inline void LoadPart(
    LocalTensor<half>& destination,
    GlobalTensor<half>& source,
    uint32_t sourceRow)
{
    constexpr uint32_t kAlignedWidth = (Width + 15U) / 16U * 16U;
    const DataCopyExtParams params{
        static_cast<uint16_t>(kRowsPerCore),
        static_cast<uint32_t>(Width * sizeof(half)),
        0,
        0,
        0};
    const DataCopyPadExtParams<half> pad{
        true,
        0,
        static_cast<uint8_t>(kAlignedWidth - Width),
        static_cast<half>(0.0f)};
    DataCopyPad(destination[LocalOffset], source[sourceRow * Width], params, pad);
}

template <uint32_t Width, uint32_t LocalOffset, uint32_t OutputColumn>
__aicore__ inline void StorePart(
    GlobalTensor<half>& destination,
    LocalTensor<half>& source,
    uint32_t destinationRow)
{
    const DataCopyExtParams params{
        static_cast<uint16_t>(kRowsPerCore),
        static_cast<uint32_t>(Width * sizeof(half)),
        0,
        static_cast<uint32_t>((kColumns - Width) * sizeof(half)),
        0};
    DataCopyPad(
        destination[destinationRow * kColumns + OutputColumn],
        source[LocalOffset],
        params);
}

__aicore__ inline void ConcatFast(GM_ADDR inputs, GM_ADDR output)
{
    GlobalTensor<half> input0Gm;
    GlobalTensor<half> input1Gm;
    GlobalTensor<half> input2Gm;
    GlobalTensor<half> input3Gm;
    GlobalTensor<half> input4Gm;
    GlobalTensor<half> input5Gm;
    GlobalTensor<half> input6Gm;
    GlobalTensor<half> input7Gm;
    GlobalTensor<half> input8Gm;
    GlobalTensor<half> outputGm;
    input0Gm.SetGlobalBuffer(
        GetDynamicInputAddress<half>(inputs, 0U), kRows * 27U);
    input1Gm.SetGlobalBuffer(
        GetDynamicInputAddress<half>(inputs, 1U), kRows * 40U);
    input2Gm.SetGlobalBuffer(
        GetDynamicInputAddress<half>(inputs, 2U), kRows * 63U);
    input3Gm.SetGlobalBuffer(
        GetDynamicInputAddress<half>(inputs, 3U), kRows * 24U);
    input4Gm.SetGlobalBuffer(
        GetDynamicInputAddress<half>(inputs, 4U), kRows * 50U);
    input5Gm.SetGlobalBuffer(
        GetDynamicInputAddress<half>(inputs, 5U), kRows * 26U);
    input6Gm.SetGlobalBuffer(
        GetDynamicInputAddress<half>(inputs, 6U), kRows * 19U);
    input7Gm.SetGlobalBuffer(
        GetDynamicInputAddress<half>(inputs, 7U), kRows * 2U);
    input8Gm.SetGlobalBuffer(
        GetDynamicInputAddress<half>(inputs, 8U), kRows * 5U);
    outputGm.SetGlobalBuffer(
        reinterpret_cast<__gm__ half *>(output), kOutputElements);

    const uint32_t rowOffset = GetBlockIdx() * kRowsPerCore;

    TPipe pipe;
    TBuf<TPosition::VECCALC> stagingBuf;
    pipe.InitBuffer(stagingBuf, kStagingElements * sizeof(half));
    LocalTensor<half> staging = stagingBuf.Get<half>();

    LoadPart<27U, 0U>(staging, input0Gm, rowOffset);
    LoadPart<40U, 512U>(staging, input1Gm, rowOffset);
    LoadPart<63U, 1280U>(staging, input2Gm, rowOffset);
    LoadPart<24U, 2304U>(staging, input3Gm, rowOffset);
    LoadPart<50U, 2816U>(staging, input4Gm, rowOffset);
    LoadPart<26U, 3840U>(staging, input5Gm, rowOffset);
    LoadPart<19U, 4352U>(staging, input6Gm, rowOffset);
    LoadPart<2U, 4864U>(staging, input7Gm, rowOffset);
    LoadPart<5U, 5120U>(staging, input8Gm, rowOffset);
    PipeBarrier<PIPE_ALL>();

    StorePart<27U, 0U, 0U>(outputGm, staging, rowOffset);
    StorePart<40U, 512U, 27U>(outputGm, staging, rowOffset);
    StorePart<63U, 1280U, 67U>(outputGm, staging, rowOffset);
    StorePart<24U, 2304U, 130U>(outputGm, staging, rowOffset);
    StorePart<50U, 2816U, 154U>(outputGm, staging, rowOffset);
    StorePart<26U, 3840U, 204U>(outputGm, staging, rowOffset);
    StorePart<19U, 4352U, 230U>(outputGm, staging, rowOffset);
    StorePart<2U, 4864U, 249U>(outputGm, staging, rowOffset);
    StorePart<5U, 5120U, 251U>(outputGm, staging, rowOffset);
}
}  // namespace

extern "C" __global__ __aicore__ void concat(
    GM_ADDR inputs,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(1)) {
        ConcatFast(inputs, output);
        return;
    }
    if (TILING_KEY_IS(0)) {
        __gm__ const auto *data = reinterpret_cast<__gm__
            const ConcatKernelTilingData *>(tiling);
        ConcatGeneric(inputs, output, data);
        return;
    }
    if (TILING_KEY_IS(2)) {
        __gm__ const auto *data = reinterpret_cast<__gm__
            const ConcatKernelTilingData *>(tiling);
        ConcatRowBatch(inputs, output, data);
        return;
    }
    if (TILING_KEY_IS(3)) {
        __gm__ const auto *data = reinterpret_cast<__gm__
            const ConcatKernelTilingData *>(tiling);
        ConcatInputParallel(inputs, output, data);
        return;
    }
    if (TILING_KEY_IS(4)) {
        __gm__ const auto *data = reinterpret_cast<__gm__
            const ConcatKernelTilingData *>(tiling);
        ConcatLinear(inputs, output, data);
    }
}
