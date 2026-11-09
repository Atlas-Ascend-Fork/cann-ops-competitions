#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "adv_api/pad/pad.h"
#include "../op_host/concat_tiling.h"

using namespace AscendC;

// A single contiguous burst can amortize a streaming read. Short padded-row
// bursts retain L2 reuse of neighboring cache lines across the AIVs.
template <typename T>
__aicore__ inline void ConfigureConcatStreamRead(GlobalTensor<T>& input,
    uint64_t totalBytes, uint64_t burstBytes) {
    if (totalBytes >= 4ULL * 1024 * 1024 && burstBytes >= 4096 &&
        (burstBytes % 32 == 0 || burstBytes >= 8192)) {
        input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }
}


namespace {
constexpr uint64_t kDmaBlockBytes = 32;

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t AlignUp32(uint64_t value)
{
    return (value + kDmaBlockBytes - 1) / kDmaBlockBytes * kDmaBlockBytes;
}

__aicore__ inline uint64_t GcdU64(uint64_t lhs, uint64_t rhs)
{
    while (rhs != 0) {
        const uint64_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

template <typename TilingDataT, uint32_t kMode, bool kUseQueue>
class KernelConcat {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), tiling_.totalBytes);
        if constexpr (kUseQueue) {
            pipe_.InitBuffer(moveQueue_, 2, tiling_.tileBytes);
        } else {
            pipe_.InitBuffer(buffer_, tiling_.tileBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (kMode == 0) {
            return;
        } else if constexpr (kMode == 2) {
            ProcessFlat();
        } else if constexpr (kMode == 3) {
            if (tiling_.elementBytes == 1) {
                ProcessGatherInputs<uint8_t>();
            } else if (tiling_.elementBytes == 2) {
                ProcessGatherInputs<uint16_t>();
            } else {
                ProcessGatherInputs<uint32_t>();
            }
        } else if constexpr (kMode == 5) {
            if (tiling_.elementBytes == 1) {
                ProcessUniformGatherInputs<uint8_t>();
            } else if (tiling_.elementBytes == 2) {
                ProcessUniformGatherInputs<uint16_t>();
            } else {
                ProcessUniformGatherInputs<uint32_t>();
            }
        } else if constexpr (kMode == 4) {
            ProcessDirectRowTask();
        } else {
            ProcessRows();
        }
    }

private:
    __aicore__ inline uint64_t InputRowBytes(uint32_t input)
    {
        return tiling_.inputRowBytes[input];
    }

    __aicore__ inline void CopyContiguous(
        const GlobalTensor<uint8_t>& source, uint64_t sourceOffset, uint64_t destinationOffset, uint64_t bytes)
    {
        if constexpr (kUseQueue) {
            while (bytes != 0) {
                const uint64_t once = MinU64(bytes, tiling_.tileBytes);
                LocalTensor<uint8_t> local = moveQueue_.AllocTensor<uint8_t>();
                DataCopyExtParams params{1, static_cast<uint32_t>(once), 0, 0, 0};
                DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                DataCopyPad(local, source[sourceOffset], params, pad);
                moveQueue_.EnQue(local);
                local = moveQueue_.DeQue<uint8_t>();
                DataCopyPad(output_[destinationOffset], local, params);
                moveQueue_.FreeTensor(local);
                sourceOffset += once;
                destinationOffset += once;
                bytes -= once;
            }
        } else {
            LocalTensor<uint8_t> local = buffer_.Get<uint8_t>();
            while (bytes != 0) {
                const uint64_t once = MinU64(bytes, tiling_.tileBytes);
                DataCopyExtParams params{1, static_cast<uint32_t>(once), 0, 0, 0};
                DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                DataCopyPad(local, source[sourceOffset], params, pad);
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(output_[destinationOffset], local, params);
                PipeBarrier<PIPE_ALL>();
                sourceOffset += once;
                destinationOffset += once;
                bytes -= once;
            }
        }
    }

    __aicore__ inline void ProcessFlat()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t begin = core * tiling_.flatChunkBytes;
        if (begin >= tiling_.totalBytes) {
            return;
        }
        const uint64_t end = MinU64(begin + tiling_.flatChunkBytes, tiling_.totalBytes);
        uint64_t streamCursor = 0;
        for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
            const uint64_t inputBytes = tiling_.outer * InputRowBytes(input);
            const uint64_t left = begin > streamCursor ? begin : streamCursor;
            const uint64_t inputEnd = streamCursor + inputBytes;
            const uint64_t right = end < inputEnd ? end : inputEnd;
            if (left < right) {
                GlobalTensor<uint8_t> source;
                source.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(input), inputBytes);
            ConfigureConcatStreamRead(source, tiling_.totalBytes, inputBytes);
                CopyContiguous(source, left - streamCursor, left, right - left);
            }
            streamCursor = inputEnd;
        }
    }

    template <typename T>
    __aicore__ inline void ProcessGatherInputs()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        const uint64_t elementBytes = sizeof(T);
        const uint64_t bytesPerCore = AlignUp32(
            (tiling_.totalBytes + blockNum - 1) / blockNum);
        const uint64_t beginBytes = core * bytesPerCore;
        if (beginBytes >= tiling_.totalBytes) {
            return;
        }
        const uint64_t endBytes = MinU64(beginBytes + bytesPerCore, tiling_.totalBytes);
        const uint64_t beginElements = beginBytes / elementBytes;
        const uint64_t endElements = endBytes / elementBytes;
        LocalTensor<T> localValues = buffer_.Get<T>();
        LocalTensor<uint8_t> localBytes = buffer_.Get<uint8_t>();

        uint64_t streamCursorElements = 0;
        for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
            if (streamCursorElements >= endElements) {
                break;
            }
            const uint64_t inputElements = InputRowBytes(input) / elementBytes;
            const uint64_t inputEndElements = streamCursorElements + inputElements;
            const uint64_t left = beginElements > streamCursorElements ?
                                      beginElements : streamCursorElements;
            const uint64_t right = endElements < inputEndElements ?
                                       endElements : inputEndElements;
            if (left < right) {
                GlobalTensor<T> source;
                source.SetGlobalBuffer(inputs_.GetDataPtr<T>(input), inputElements);
                for (uint64_t element = left; element < right; ++element) {
                    localValues.SetValue(
                        element - beginElements, source.GetValue(element - streamCursorElements));
                }
            }
            streamCursorElements = inputEndElements;
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(endBytes - beginBytes), 0, 0, 0};
        DataCopyPad(output_[beginBytes], localBytes, outputParams);
        PipeBarrier<PIPE_ALL>();
    }

    template <typename T>
    __aicore__ inline void ProcessUniformGatherInputs()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        const uint64_t elementBytes = sizeof(T);
        const uint64_t bytesPerCore = AlignUp32(
            (tiling_.totalBytes + blockNum - 1) / blockNum);
        const uint64_t beginBytes = core * bytesPerCore;
        if (beginBytes >= tiling_.totalBytes) {
            return;
        }
        const uint64_t endBytes = MinU64(beginBytes + bytesPerCore, tiling_.totalBytes);
        const uint64_t beginElements = beginBytes / elementBytes;
        const uint64_t endElements = endBytes / elementBytes;
        const uint64_t inputElements = InputRowBytes(0) / elementBytes;
        LocalTensor<T> localValues = buffer_.Get<T>();
        LocalTensor<uint8_t> localBytes = buffer_.Get<uint8_t>();

        uint32_t input = static_cast<uint32_t>(beginElements / inputElements);
        uint64_t streamCursorElements = static_cast<uint64_t>(input) * inputElements;
        while (input < tiling_.inputCount && streamCursorElements < endElements) {
            const uint64_t inputEndElements = streamCursorElements + inputElements;
            const uint64_t left = beginElements > streamCursorElements ?
                                      beginElements : streamCursorElements;
            const uint64_t right = endElements < inputEndElements ?
                                       endElements : inputEndElements;
            GlobalTensor<T> source;
            source.SetGlobalBuffer(inputs_.GetDataPtr<T>(input), inputElements);
            for (uint64_t element = left; element < right; ++element) {
                localValues.SetValue(
                    element - beginElements, source.GetValue(element - streamCursorElements));
            }
            ++input;
            streamCursorElements = inputEndElements;
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(endBytes - beginBytes), 0, 0, 0};
        DataCopyPad(output_[beginBytes], localBytes, outputParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessRowInterval(
        uint32_t input, uint64_t rowBegin, uint64_t rowEnd, uint64_t outputPrefixBytes)
    {
        const uint64_t segmentBytes = InputRowBytes(input);
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(input), tiling_.outer * segmentBytes);
            ConfigureConcatStreamRead(source, tiling_.totalBytes, segmentBytes);
        const uint64_t localRowBytes = AlignUp32(segmentBytes);
        const bool canUse2d = localRowBytes <= tiling_.tileBytes &&
                              segmentBytes <= 2097151 &&
                              tiling_.outputRowBytes - segmentBytes <= static_cast<uint64_t>(UINT32_MAX);
        if (canUse2d) {
            const uint64_t maxRows = MinU64(4095, tiling_.tileBytes / localRowBytes);
            if constexpr (kUseQueue) {
                for (uint64_t row = rowBegin; row < rowEnd; row += maxRows) {
                    const uint64_t rows = MinU64(maxRows, rowEnd - row);
                    LocalTensor<uint8_t> queuedLocal = moveQueue_.AllocTensor<uint8_t>();
                    DataCopyExtParams inputParams{
                        static_cast<uint16_t>(rows), static_cast<uint32_t>(segmentBytes), 0, 0, 0};
                    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                    DataCopyPad(queuedLocal, source[row * segmentBytes], inputParams, pad);
                    moveQueue_.EnQue(queuedLocal);
                    queuedLocal = moveQueue_.DeQue<uint8_t>();
                    DataCopyExtParams outputParams{
                        static_cast<uint16_t>(rows), static_cast<uint32_t>(segmentBytes),
                        0, static_cast<uint32_t>(tiling_.outputRowBytes - segmentBytes), 0};
                    const uint64_t outputOffset = row * tiling_.outputRowBytes + outputPrefixBytes;
                    DataCopyPad(output_[outputOffset], queuedLocal, outputParams);
                    moveQueue_.FreeTensor(queuedLocal);
                }
            } else {
                LocalTensor<uint8_t> local = buffer_.Get<uint8_t>();
                for (uint64_t row = rowBegin; row < rowEnd; row += maxRows) {
                    const uint64_t rows = MinU64(maxRows, rowEnd - row);
                    DataCopyExtParams inputParams{
                        static_cast<uint16_t>(rows), static_cast<uint32_t>(segmentBytes), 0, 0, 0};
                    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                    DataCopyPad(local, source[row * segmentBytes], inputParams, pad);
                    PipeBarrier<PIPE_ALL>();
                    DataCopyExtParams outputParams{
                        static_cast<uint16_t>(rows), static_cast<uint32_t>(segmentBytes),
                        0, static_cast<uint32_t>(tiling_.outputRowBytes - segmentBytes), 0};
                    const uint64_t outputOffset = row * tiling_.outputRowBytes + outputPrefixBytes;
                    DataCopyPad(output_[outputOffset], local, outputParams);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        } else {
            for (uint64_t row = rowBegin; row < rowEnd; ++row) {
                const uint64_t outputOffset = row * tiling_.outputRowBytes + outputPrefixBytes;
                CopyContiguous(source, row * segmentBytes, outputOffset, segmentBytes);
            }
        }
    }

    __aicore__ inline void ProcessDirectRowTask()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t groups = (tiling_.outer + tiling_.blockRows - 1) / tiling_.blockRows;
        const uint32_t input = static_cast<uint32_t>(core / groups);
        const uint64_t group = core - static_cast<uint64_t>(input) * groups;
        if (input >= tiling_.inputCount) {
            return;
        }
        const uint64_t rowBegin = group * tiling_.blockRows;
        const uint64_t rowEnd = MinU64(rowBegin + tiling_.blockRows, tiling_.outer);
        uint64_t outputPrefixBytes = 0;
        for (uint32_t previous = 0; previous < input; ++previous) {
            outputPrefixBytes += InputRowBytes(previous);
        }
        ProcessRowInterval(input, rowBegin, rowEnd, outputPrefixBytes);
    }

    __aicore__ inline void ProcessRows()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        uint64_t outputPrefixBytes = 0;
        uint64_t taskOrdinal = 0;
        for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
            const uint64_t segmentBytes = InputRowBytes(input);
            if (segmentBytes == 0) {
                continue;
            }
            for (uint64_t rowBegin = 0; rowBegin < tiling_.outer; rowBegin += tiling_.blockRows) {
                const uint64_t rowEnd = MinU64(rowBegin + tiling_.blockRows, tiling_.outer);
                const bool ownsTask = taskOrdinal % blockNum == core;
                ++taskOrdinal;
                if (ownsTask) {
                    ProcessRowInterval(input, rowBegin, rowEnd, outputPrefixBytes);
                }
            }
            outputPrefixBytes += segmentBytes;
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> buffer_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> moveQueue_;
    ListTensorDesc inputs_;
    GlobalTensor<uint8_t> output_;
    TilingDataT tiling_{};
};

// Large row grids are bandwidth-bound but a cyclic task map makes the widest
// input determine the whole launch.  The host solves the small scheduling
// problem once and serializes a compact per-core task list into tiling data;
// every AIV therefore jumps directly to its balanced work with no global task
// scan and with far fewer 2-D DMA descriptors than a very fine row grid.
template <typename TilingDataT, bool kPrefixOnly = false>
class KernelConcatScheduledRows {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), tiling_.totalBytes);
        pipe_.InitBuffer(moveQueue_, 2, tiling_.tileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint16_t begin = tiling_.coreTaskOffsets[core];
        const uint16_t end = tiling_.coreTaskOffsets[core + 1];
        const uint64_t groups = (tiling_.outer + tiling_.blockRows - 1) / tiling_.blockRows;
        for (uint16_t slot = begin; slot < end; ++slot) {
            const uint16_t ordinal = tiling_.taskOrdinals[slot];
            const uint32_t input = static_cast<uint32_t>(ordinal / groups);
            const uint64_t group = ordinal - static_cast<uint64_t>(input) * groups;
            ProcessTask(input, group);
        }
    }

private:
    __aicore__ inline uint64_t SegmentBytes(uint32_t input)
    {
        if constexpr (kPrefixOnly) {
            const uint64_t begin = tiling_.inputPrefixBytes[input];
            const uint64_t end = input + 1 < tiling_.inputCount ?
                tiling_.inputPrefixBytes[input + 1] : tiling_.outputRowBytes;
            return end - begin;
        } else {
            return tiling_.inputRowBytes[input];
        }
    }

    __aicore__ inline void ProcessTask(uint32_t input, uint64_t group)
    {
        const uint64_t segmentBytes = SegmentBytes(input);
        const uint64_t outputPrefixBytes = tiling_.inputPrefixBytes[input];
        const uint64_t rowBegin = group * tiling_.blockRows;
        const uint64_t rowEnd = MinU64(rowBegin + tiling_.blockRows, tiling_.outer);

        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(input), tiling_.outer * segmentBytes);
            ConfigureConcatStreamRead(source, tiling_.totalBytes, segmentBytes);
        const uint64_t localRowBytes = AlignUp32(segmentBytes);
        const bool canUse2d = localRowBytes <= tiling_.tileBytes &&
                              segmentBytes <= 2097151 &&
                              tiling_.outputRowBytes - segmentBytes <= static_cast<uint64_t>(UINT32_MAX);
        if (canUse2d) {
            CopyRows2d(source, rowBegin, rowEnd, outputPrefixBytes, segmentBytes, localRowBytes);
            return;
        }
        for (uint64_t row = rowBegin; row < rowEnd; ++row) {
            const uint64_t outputOffset = row * tiling_.outputRowBytes + outputPrefixBytes;
            CopyContiguous(source, row * segmentBytes, outputOffset, segmentBytes);
        }
    }

    __aicore__ inline void CopyRows2d(
        const GlobalTensor<uint8_t>& source,
        uint64_t rowBegin,
        uint64_t rowEnd,
        uint64_t outputPrefixBytes,
        uint64_t segmentBytes,
        uint64_t localRowBytes)
    {
        const uint64_t maxRows = MinU64(4095, tiling_.tileBytes / localRowBytes);
        bool hasPending = false;
        uint64_t pendingOutputOffset = 0;
        uint32_t pendingRows = 0;
        for (uint64_t row = rowBegin; row < rowEnd; row += maxRows) {
            const uint64_t rows = MinU64(maxRows, rowEnd - row);
            DataCopyExtParams inputParams{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(segmentBytes), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            const uint64_t outputOffset = row * tiling_.outputRowBytes + outputPrefixBytes;
            LocalTensor<uint8_t> local = moveQueue_.AllocTensor<uint8_t>();
            DataCopyPad(local, source[row * segmentBytes], inputParams, pad);
            moveQueue_.EnQue(local);
            if (hasPending) {
                local = moveQueue_.DeQue<uint8_t>();
                DataCopyExtParams pendingParams{
                    static_cast<uint16_t>(pendingRows),
                    static_cast<uint32_t>(segmentBytes), 0,
                    static_cast<uint32_t>(tiling_.outputRowBytes - segmentBytes), 0};
                DataCopyPad(output_[pendingOutputOffset], local, pendingParams);
                moveQueue_.FreeTensor(local);
            }
            pendingOutputOffset = outputOffset;
            pendingRows = static_cast<uint32_t>(rows);
            hasPending = true;
        }
        if (hasPending) {
            LocalTensor<uint8_t> local = moveQueue_.DeQue<uint8_t>();
            DataCopyExtParams pendingParams{
                static_cast<uint16_t>(pendingRows), static_cast<uint32_t>(segmentBytes), 0,
                static_cast<uint32_t>(tiling_.outputRowBytes - segmentBytes), 0};
            DataCopyPad(output_[pendingOutputOffset], local, pendingParams);
            moveQueue_.FreeTensor(local);
        }
    }

    __aicore__ inline void CopyContiguous(
        const GlobalTensor<uint8_t>& source,
        uint64_t sourceOffset,
        uint64_t destinationOffset,
        uint64_t bytes)
    {
        bool hasPending = false;
        uint64_t pendingDestinationOffset = 0;
        uint32_t pendingBytes = 0;
        while (bytes != 0) {
            const uint64_t once = MinU64(bytes, tiling_.tileBytes);
            DataCopyExtParams params{1, static_cast<uint32_t>(once), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            LocalTensor<uint8_t> local = moveQueue_.AllocTensor<uint8_t>();
            DataCopyPad(local, source[sourceOffset], params, pad);
            moveQueue_.EnQue(local);
            if (hasPending) {
                local = moveQueue_.DeQue<uint8_t>();
                DataCopyExtParams pendingParams{1, pendingBytes, 0, 0, 0};
                DataCopyPad(output_[pendingDestinationOffset], local, pendingParams);
                moveQueue_.FreeTensor(local);
            }
            pendingDestinationOffset = destinationOffset;
            pendingBytes = static_cast<uint32_t>(once);
            hasPending = true;
            sourceOffset += once;
            destinationOffset += once;
            bytes -= once;
        }
        if (hasPending) {
            LocalTensor<uint8_t> local = moveQueue_.DeQue<uint8_t>();
            DataCopyExtParams pendingParams{1, pendingBytes, 0, 0, 0};
            DataCopyPad(output_[pendingDestinationOffset], local, pendingParams);
            moveQueue_.FreeTensor(local);
        }
    }

private:
    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> moveQueue_;
    ListTensorDesc inputs_;
    GlobalTensor<uint8_t> output_;
    TilingDataT tiling_{};
};

// Width-skewed TensorLists need a different row-group count per input.  The
// host stores only the <=256 useful tasks and balances them with LPT; each AIV
// therefore executes its direct list instead of scanning the Cartesian
// product of every input and the widest input's group count.
template <typename TilingDataT>
class KernelConcatAdaptiveRows {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), tiling_.totalBytes);
        pipe_.InitBuffer(moveQueue_, 2, tiling_.tileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint16_t begin = tiling_.coreTaskOffsets[core];
        const uint16_t end = tiling_.coreTaskOffsets[core + 1];
        for (uint16_t slot = begin; slot < end; ++slot) {
            ProcessTask(tiling_.taskInputs[slot], tiling_.taskGroups[slot]);
        }
    }

private:
    __aicore__ inline uint64_t SegmentBytes(uint32_t input)
    {
        const uint64_t begin = tiling_.inputPrefixBytes[input];
        const uint64_t end = input + 1 < tiling_.inputCount ?
            tiling_.inputPrefixBytes[input + 1] : tiling_.outputRowBytes;
        return end - begin;
    }

    __aicore__ inline void ProcessTask(uint32_t input, uint32_t group)
    {
        const uint64_t groups = tiling_.inputGroupCounts[input];
        const uint64_t blockRows = (tiling_.outer + groups - 1) / groups;
        const uint64_t rowBegin = static_cast<uint64_t>(group) * blockRows;
        const uint64_t rowEnd = MinU64(rowBegin + blockRows, tiling_.outer);
        const uint64_t segmentBytes = SegmentBytes(input);
        const uint64_t outputPrefixBytes = tiling_.inputPrefixBytes[input];

        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(input), tiling_.outer * segmentBytes);
            ConfigureConcatStreamRead(source, tiling_.totalBytes, segmentBytes);
        const uint64_t localRowBytes = AlignUp32(segmentBytes);
        const bool canUse2d = localRowBytes <= tiling_.tileBytes &&
                              segmentBytes <= 2097151 &&
                              tiling_.outputRowBytes - segmentBytes <=
                                  static_cast<uint64_t>(UINT32_MAX);
        if (canUse2d) {
            CopyRows2d(
                source, rowBegin, rowEnd, outputPrefixBytes, segmentBytes, localRowBytes);
            return;
        }
        for (uint64_t row = rowBegin; row < rowEnd; ++row) {
            CopyContiguous(
                source, row * segmentBytes,
                row * tiling_.outputRowBytes + outputPrefixBytes, segmentBytes);
        }
    }

    __aicore__ inline void CopyRows2d(
        const GlobalTensor<uint8_t>& source,
        uint64_t rowBegin,
        uint64_t rowEnd,
        uint64_t outputPrefixBytes,
        uint64_t segmentBytes,
        uint64_t localRowBytes)
    {
        const uint64_t maxRows = MinU64(4095, tiling_.tileBytes / localRowBytes);
        bool hasPending = false;
        uint64_t pendingOutputOffset = 0;
        uint32_t pendingRows = 0;
        for (uint64_t row = rowBegin; row < rowEnd; row += maxRows) {
            const uint64_t rows = MinU64(maxRows, rowEnd - row);
            DataCopyExtParams inputParams{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(segmentBytes), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            const uint64_t outputOffset =
                row * tiling_.outputRowBytes + outputPrefixBytes;
            LocalTensor<uint8_t> local = moveQueue_.AllocTensor<uint8_t>();
            DataCopyPad(local, source[row * segmentBytes], inputParams, pad);
            moveQueue_.EnQue(local);
            if (hasPending) {
                local = moveQueue_.DeQue<uint8_t>();
                DataCopyExtParams pendingParams{
                    static_cast<uint16_t>(pendingRows),
                    static_cast<uint32_t>(segmentBytes), 0,
                    static_cast<uint32_t>(tiling_.outputRowBytes - segmentBytes), 0};
                DataCopyPad(output_[pendingOutputOffset], local, pendingParams);
                moveQueue_.FreeTensor(local);
            }
            pendingOutputOffset = outputOffset;
            pendingRows = static_cast<uint32_t>(rows);
            hasPending = true;
        }
        if (hasPending) {
            LocalTensor<uint8_t> local = moveQueue_.DeQue<uint8_t>();
            DataCopyExtParams pendingParams{
                static_cast<uint16_t>(pendingRows),
                static_cast<uint32_t>(segmentBytes), 0,
                static_cast<uint32_t>(tiling_.outputRowBytes - segmentBytes), 0};
            DataCopyPad(output_[pendingOutputOffset], local, pendingParams);
            moveQueue_.FreeTensor(local);
        }
    }

    __aicore__ inline void CopyContiguous(
        const GlobalTensor<uint8_t>& source,
        uint64_t sourceOffset,
        uint64_t destinationOffset,
        uint64_t bytes)
    {
        bool hasPending = false;
        uint64_t pendingDestinationOffset = 0;
        uint32_t pendingBytes = 0;
        while (bytes != 0) {
            const uint64_t once = MinU64(bytes, tiling_.tileBytes);
            DataCopyExtParams params{1, static_cast<uint32_t>(once), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            LocalTensor<uint8_t> local = moveQueue_.AllocTensor<uint8_t>();
            DataCopyPad(local, source[sourceOffset], params, pad);
            moveQueue_.EnQue(local);
            if (hasPending) {
                local = moveQueue_.DeQue<uint8_t>();
                DataCopyExtParams pendingParams{1, pendingBytes, 0, 0, 0};
                DataCopyPad(output_[pendingDestinationOffset], local, pendingParams);
                moveQueue_.FreeTensor(local);
            }
            pendingDestinationOffset = destinationOffset;
            pendingBytes = static_cast<uint32_t>(once);
            hasPending = true;
            sourceOffset += once;
            destinationOffset += once;
            bytes -= once;
        }
        if (hasPending) {
            LocalTensor<uint8_t> local = moveQueue_.DeQue<uint8_t>();
            DataCopyExtParams pendingParams{1, pendingBytes, 0, 0, 0};
            DataCopyPad(output_[pendingDestinationOffset], local, pendingParams);
            moveQueue_.FreeTensor(local);
        }
    }

private:
    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> moveQueue_;
    ListTensorDesc inputs_;
    GlobalTensor<uint8_t> output_;
    TilingDataT tiling_{};
};

template <typename TilingDataT>
class KernelConcatDirectRowTasks {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), tiling_.totalBytes);
        pipe_.InitBuffer(moveQueue_, 2, tiling_.tileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        const uint64_t groups = (tiling_.outer + tiling_.blockRows - 1) / tiling_.blockRows;
        uint64_t outputPrefixBytes = 0;
        for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
            const uint64_t segmentBytes = tiling_.inputRowBytes[input];
            const uint64_t inputPhase = (static_cast<uint64_t>(input) * groups) % blockNum;
            const uint64_t firstGroup = (core + blockNum - inputPhase) % blockNum;
            if (firstGroup < groups && segmentBytes != 0) {
                ProcessInput(input, firstGroup, groups, blockNum, outputPrefixBytes, segmentBytes);
            }
            outputPrefixBytes += segmentBytes;
        }
    }

private:
    __aicore__ inline void ProcessInput(
        uint32_t input,
        uint64_t firstGroup,
        uint64_t groups,
        uint64_t groupStride,
        uint64_t outputPrefixBytes,
        uint64_t segmentBytes)
    {
        GlobalTensor<uint8_t> source;
        source.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(input), tiling_.outer * segmentBytes);
            ConfigureConcatStreamRead(source, tiling_.totalBytes, segmentBytes);
        const uint64_t localRowBytes = AlignUp32(segmentBytes);
        const bool canUse2d = localRowBytes <= tiling_.tileBytes && segmentBytes <= 2097151 &&
                              tiling_.outputRowBytes - segmentBytes <= static_cast<uint64_t>(UINT32_MAX);
        for (uint64_t group = firstGroup; group < groups; group += groupStride) {
            const uint64_t rowBegin = group * tiling_.blockRows;
            const uint64_t rowEnd = MinU64(rowBegin + tiling_.blockRows, tiling_.outer);
            if (canUse2d) {
                CopyRows2d(source, rowBegin, rowEnd, outputPrefixBytes, segmentBytes, localRowBytes);
                continue;
            }
            for (uint64_t row = rowBegin; row < rowEnd; ++row) {
                CopyContiguous(source, row * segmentBytes,
                               row * tiling_.outputRowBytes + outputPrefixBytes, segmentBytes);
            }
        }
    }

    __aicore__ inline void CopyRows2d(
        const GlobalTensor<uint8_t>& source,
        uint64_t rowBegin,
        uint64_t rowEnd,
        uint64_t outputPrefixBytes,
        uint64_t segmentBytes,
        uint64_t localRowBytes)
    {
        const uint64_t maxRows = MinU64(4095, tiling_.tileBytes / localRowBytes);
        for (uint64_t row = rowBegin; row < rowEnd; row += maxRows) {
            const uint64_t rows = MinU64(maxRows, rowEnd - row);
            DataCopyExtParams inputParams{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(segmentBytes), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyExtParams outputParams{
                static_cast<uint16_t>(rows), static_cast<uint32_t>(segmentBytes), 0,
                static_cast<uint32_t>(tiling_.outputRowBytes - segmentBytes), 0};
            const uint64_t outputOffset = row * tiling_.outputRowBytes + outputPrefixBytes;
            LocalTensor<uint8_t> local = moveQueue_.AllocTensor<uint8_t>();
            DataCopyPad(local, source[row * segmentBytes], inputParams, pad);
            moveQueue_.EnQue(local);
            local = moveQueue_.DeQue<uint8_t>();
            DataCopyPad(output_[outputOffset], local, outputParams);
            moveQueue_.FreeTensor(local);
        }
    }

    __aicore__ inline void CopyContiguous(
        const GlobalTensor<uint8_t>& source,
        uint64_t sourceOffset,
        uint64_t destinationOffset,
        uint64_t bytes)
    {
        while (bytes != 0) {
            const uint64_t once = MinU64(bytes, tiling_.tileBytes);
            DataCopyExtParams params{1, static_cast<uint32_t>(once), 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            LocalTensor<uint8_t> local = moveQueue_.AllocTensor<uint8_t>();
            DataCopyPad(local, source[sourceOffset], params, pad);
            moveQueue_.EnQue(local);
            local = moveQueue_.DeQue<uint8_t>();
            DataCopyPad(output_[destinationOffset], local, params);
            moveQueue_.FreeTensor(local);
            sourceOffset += once;
            destinationOffset += once;
            bytes -= once;
        }
    }

private:
    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> moveQueue_;
    ListTensorDesc inputs_;
    GlobalTensor<uint8_t> output_;
    TilingDataT tiling_{};
};

// CANN's host PadTilingFunc cannot execute on the AIV.  This is the CANN 8.5
// tiling arithmetic for Pad's 2-D, right-padding-only case, kept next to the
// only kernel that consumes it.  Every operand is at least one 32-byte block
// wide when this path is selected.
template <typename ElementT>
__aicore__ inline void BuildRowPadTiling(
    const ShapeInfo& srcShape, uint32_t stackBufferSize, PadTiling& tiling)
{
    constexpr uint32_t typeSize = sizeof(ElementT);
    const uint32_t srcHeight = srcShape.shape[0];
    const uint32_t srcWidth = srcShape.shape[1];
    const uint32_t srcOriWidth = srcShape.originalShape[1];
    tiling.srcHeight = srcHeight;
    tiling.srcWidth = srcWidth;
    tiling.srcOriWidth = srcOriWidth;

    const uint32_t elementsPerBlock = ONE_BLK_SIZE / typeSize;
    const uint32_t widthWithoutLastBlock = srcWidth - elementsPerBlock;
    tiling.widthWithoutLastBlock = widthWithoutLastBlock;
    tiling.blocksPerRow = srcWidth * typeSize / ONE_BLK_SIZE;

    const uint32_t heightTiling = MAX_REPEAT_TIMES;
    const uint32_t heightFractal = srcHeight / heightTiling;
    const uint32_t heightFractalTail = srcHeight % heightTiling;
    tiling.heightTiling = heightTiling;
    tiling.heightFractal = heightFractal;
    tiling.heightFractalTail = heightFractalTail;
    tiling.mainLoopOffset = heightTiling * srcWidth;
    tiling.tailBlockOffset =
        heightFractal * heightTiling * srcWidth + widthWithoutLastBlock;

    constexpr uint32_t baseBlockLen = 16 * ONE_BLK_SIZE;
    constexpr uint32_t baseBlockSize = baseBlockLen / typeSize;
    uint32_t tmpBuffer1BlockNum =
        stackBufferSize / typeSize / baseBlockSize / 2;
    tmpBuffer1BlockNum = tmpBuffer1BlockNum == 0 ? 1 : tmpBuffer1BlockNum;
    tiling.tmpBuffer1BlockNum = tmpBuffer1BlockNum;
    tiling.tmpBuffer1RowNum = 16 * tmpBuffer1BlockNum;
    tiling.tmpBuffer2Offset = tmpBuffer1BlockNum * baseBlockSize;

    const uint32_t widthTiling = 16 * tmpBuffer1BlockNum;
    const uint32_t widthFractal = srcWidth / widthTiling;
    const uint32_t widthFractalTail = srcWidth % widthTiling;
    const uint32_t widthFractalTailAligned =
        ((widthFractalTail - 1) / elementsPerBlock + 1) * elementsPerBlock;
    tiling.widthTiling = widthTiling;
    tiling.widthFractal = widthFractal;
    tiling.widthFractalTail = widthFractalTail;
    tiling.widthFractalTailAlingned = widthFractalTailAligned;

    const uint32_t brcbTiling = 16 * tmpBuffer1BlockNum;
    const uint32_t brcbFractal = srcHeight * srcWidth / brcbTiling;
    const uint32_t brcbFractalTail = srcHeight * srcWidth % brcbTiling;
    tiling.brcbTiling = brcbTiling;
    tiling.brcbFractal = brcbFractal;
    tiling.brcbFractalTail = brcbFractalTail;

    constexpr uint32_t maxRepeatTimes = 254;
    tiling.maxRepeatTimes = maxRepeatTimes;
    tiling.brcbTilingRepeatTimes = brcbTiling / 8 / maxRepeatTimes;
    tiling.brcbTilingRepeatTimesTail = brcbTiling / 8 % maxRepeatTimes;
    tiling.brcbFractalTailRepeatTimes =
        brcbFractalTail / 8 / maxRepeatTimes;
    tiling.brcbFractalTailRepeatTimesTail =
        brcbFractalTail / 8 % maxRepeatTimes;
}

// Large row grids are MTE2-bound when DataCopyPad has to materialize
// one aligned UB row for every unaligned source row.  Load each input shard as
// one contiguous stream, then use the dedicated Pad vector primitive to
// rebuild all aligned UB rows in one operation.  FP32/INT32 share uint32_t as
// a bit-preserving transport type; even-width INT8 rows use uint16_t pairs.
// The outgoing 2-D DMA is unchanged, so destination ranges remain disjoint.
template <typename TilingDataT, typename ElementT>
class KernelConcatContiguousRowPad {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementT*>(y), tiling_.totalBytes / sizeof(ElementT));
        pipe_.InitBuffer(inputQueue_, 1, tiling_.tileBytes);
        pipe_.InitBuffer(outputQueue_, 1, tiling_.tileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        const uint64_t baseRows = tiling_.outer / blockNum;
        const uint64_t extraRows = tiling_.outer % blockNum;
        const uint64_t ownedBegin =
            core * baseRows + MinU64(core, extraRows);
        const uint64_t ownedEnd =
            ownedBegin + baseRows + static_cast<uint64_t>(core < extraRows);
        for (uint64_t rowBegin = ownedBegin; rowBegin < ownedEnd;
             rowBegin += tiling_.blockRows) {
            const uint64_t rowEnd = MinU64(rowBegin + tiling_.blockRows, ownedEnd);
            const uint32_t rows = static_cast<uint32_t>(rowEnd - rowBegin);
            uint64_t outputPrefixElements = 0;
            for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
                const uint32_t segmentBytes = tiling_.inputRowBytes[input];
                const uint32_t segmentElements = segmentBytes / sizeof(ElementT);
                if (segmentElements != 0) {
                    CopyIn(input, rowBegin, rows, segmentBytes, segmentElements);
                    if (segmentBytes % kDmaBlockBytes == 0) {
                        CopyAlignedOut(
                            rowBegin, rows, outputPrefixElements, segmentBytes);
                    } else {
                        PadRows(rows, segmentBytes, segmentElements);
                        CopyPaddedOut(
                            rowBegin, rows, outputPrefixElements, segmentBytes);
                    }
                }
                outputPrefixElements += segmentElements;
            }
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t input,
        uint64_t rowBegin,
        uint32_t rows,
        uint32_t segmentBytes,
        uint32_t segmentElements)
    {
        GlobalTensor<ElementT> source;
        source.SetGlobalBuffer(
            inputs_.GetDataPtr<ElementT>(input), tiling_.outer * segmentElements);
        LocalTensor<ElementT> local = inputQueue_.AllocTensor<ElementT>();
        DataCopyExtParams params{
            1, static_cast<uint32_t>(static_cast<uint64_t>(rows) * segmentBytes), 0, 0, 0};
        DataCopyPadExtParams<ElementT> pad{false, 0, 0, 0};
        DataCopyPad(local, source[rowBegin * segmentElements], params, pad);
        inputQueue_.EnQue(local);
    }

    __aicore__ inline void PadRows(
        uint32_t rows, uint32_t segmentBytes, uint32_t segmentElements)
    {
        LocalTensor<ElementT> inputLocal = inputQueue_.DeQue<ElementT>();
        LocalTensor<ElementT> outputLocal = outputQueue_.AllocTensor<ElementT>();
        const uint32_t alignedElements =
            static_cast<uint32_t>(AlignUp32(segmentBytes) / sizeof(ElementT));

        uint32_t shape[] = {rows, segmentElements};
        uint32_t originalShape[] = {rows, segmentElements};
        ShapeInfo shapeInfo(2, shape, 2, originalShape, DataFormat::ND);
        inputLocal.SetShapeInfo(shapeInfo);

        LocalTensor<uint8_t> stackBuffer;
        PopStackBuffer<uint8_t, TPosition::LCM>(stackBuffer);
        PadTiling padTiling;
        BuildRowPadTiling<ElementT>(shapeInfo, stackBuffer.GetSize(), padTiling);
        PadParams padParams{
            0, static_cast<uint16_t>(alignedElements - segmentElements), 0};
        Pad(outputLocal, inputLocal, padParams, padTiling);

        inputQueue_.FreeTensor(inputLocal);
        outputQueue_.EnQue(outputLocal);
    }

    __aicore__ inline void CopyPaddedOut(
        uint64_t rowBegin,
        uint32_t rows,
        uint64_t outputPrefixElements,
        uint32_t segmentBytes)
    {
        LocalTensor<ElementT> local = outputQueue_.DeQue<ElementT>();
        DataCopyExtParams params{
            static_cast<uint16_t>(rows), segmentBytes, 0,
            static_cast<uint32_t>(tiling_.outputRowBytes - segmentBytes), 0};
        const uint64_t outputOffset =
            rowBegin * (tiling_.outputRowBytes / sizeof(ElementT)) + outputPrefixElements;
        DataCopyPad(output_[outputOffset], local, params);
        outputQueue_.FreeTensor(local);
    }

    __aicore__ inline void CopyAlignedOut(
        uint64_t rowBegin,
        uint32_t rows,
        uint64_t outputPrefixElements,
        uint32_t segmentBytes)
    {
        LocalTensor<ElementT> local = inputQueue_.DeQue<ElementT>();
        DataCopyExtParams params{
            static_cast<uint16_t>(rows), segmentBytes, 0,
            static_cast<uint32_t>(tiling_.outputRowBytes - segmentBytes), 0};
        const uint64_t outputOffset =
            rowBegin * (tiling_.outputRowBytes / sizeof(ElementT)) + outputPrefixElements;
        DataCopyPad(output_[outputOffset], local, params);
        inputQueue_.FreeTensor(local);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inputQueue_;
    TQue<TPosition::VECOUT, 1> outputQueue_;
    ListTensorDesc inputs_;
    GlobalTensor<ElementT> output_;
    TilingDataT tiling_{};
};

// All input segments are 32-byte aligned.  Stage a complete output-row slab
// directly into its final UB layout, then write the slab contiguously.  MTE2
// destination stride is expressed in 32-byte blocks; MTE3 stride remains
// byte-based, matching the CANN 8.5 all-align Concat implementation.
template <typename TilingDataT>
class KernelConcatAlignedRowPack {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), tiling_.totalBytes);
        pipe_.InitBuffer(packQueue_, 2, tiling_.tileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t blockNum = GetBlockNum();
        const uint64_t baseRows = tiling_.outer / blockNum;
        const uint64_t extraRows = tiling_.outer % blockNum;
        const uint64_t ownedBegin = core * baseRows + MinU64(core, extraRows);
        const uint64_t ownedEnd =
            ownedBegin + baseRows + static_cast<uint64_t>(core < extraRows);
        // Keep one packed slab in flight while the next slab is being staged.
        // The queue is VECIN/VECOUT bound with depth two, so EnQue of the
        // current slab permits MTE2 to fill the other slot while CopyOut drains
        // the previous slot on MTE3.  The former implementation packed and
        // drained each slab back-to-back, which serialized the two engines and
        // made the mode-23 path pay the full MTE2+MTE3 sum on large rows.
        bool hasPending = false;
        uint64_t pendingRowBegin = 0;
        uint32_t pendingRows = 0;
        for (uint64_t rowBegin = ownedBegin; rowBegin < ownedEnd;
             rowBegin += tiling_.blockRows) {
            const uint32_t rows = static_cast<uint32_t>(
                MinU64(rowBegin + tiling_.blockRows, ownedEnd) - rowBegin);
            PackRows(rowBegin, rows);
            if (hasPending) {
                CopyOut(pendingRowBegin, pendingRows);
            }
            pendingRowBegin = rowBegin;
            pendingRows = rows;
            hasPending = true;
        }
        if (hasPending) {
            CopyOut(pendingRowBegin, pendingRows);
        }
    }

private:
    __aicore__ inline void PackRows(uint64_t rowBegin, uint32_t rows)
    {
        LocalTensor<uint8_t> local = packQueue_.AllocTensor<uint8_t>();
        uint32_t outputPrefixBytes = 0;
        for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
            const uint32_t segmentBytes = tiling_.inputRowBytes[input];
            if (segmentBytes != 0) {
                GlobalTensor<uint8_t> source;
                source.SetGlobalBuffer(
                    inputs_.GetDataPtr<uint8_t>(input), tiling_.outer * segmentBytes);
            ConfigureConcatStreamRead(source, tiling_.totalBytes, segmentBytes);
                DataCopyExtParams params{
                    static_cast<uint16_t>(rows), segmentBytes, 0,
                    static_cast<uint32_t>(
                        (tiling_.outputRowBytes - segmentBytes) / kDmaBlockBytes),
                    0};
                DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                DataCopyPad(
                    local[outputPrefixBytes], source[rowBegin * segmentBytes], params, pad);
            }
            outputPrefixBytes += segmentBytes;
        }
        packQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOut(uint64_t rowBegin, uint32_t rows)
    {
        LocalTensor<uint8_t> local = packQueue_.DeQue<uint8_t>();
        DataCopyExtParams params{
            1, static_cast<uint32_t>(rows * tiling_.outputRowBytes), 0, 0, 0};
        DataCopyPad(
            output_[rowBegin * tiling_.outputRowBytes], local, params);
        packQueue_.FreeTensor(local);
    }

private:
    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> packQueue_;
    ListTensorDesc inputs_;
    GlobalTensor<uint8_t> output_;
    TilingDataT tiling_{};
};

// A row-wise Concat with uniform 1--16 byte input shards is not a bandwidth
// problem: direct 2-D MTE3 emits one tiny strided block per output row and can
// be two orders of magnitude slower than the byte volume suggests.  This
// kernel instead loads one contiguous row interval from every input, gathers
// the input-major UB slabs into row-major order, and writes one contiguous GM
// interval.  FP16 and 32-bit payloads use bit-preserving transport.  INT8 is
// widened exactly to FP16 because A2 vgather has 16/32-bit lanes, then cast
// back before the single output DMA.
template <typename TilingDataT, typename ElementT, typename GatherT, bool kConvertInt8>
class KernelConcatUniformNarrowRows {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementT*>(y), tiling_.totalBytes / sizeof(ElementT));
        pipe_.InitBuffer(sourceBuffer_, kMaxSourceElements * sizeof(GatherT));
        pipe_.InitBuffer(outputBuffer_, kMaxGatherElements * sizeof(GatherT));
        pipe_.InitBuffer(indexBuffer_, kMaxGatherElements * sizeof(uint32_t));
        pipe_.InitBuffer(rawSourceBuffer_, kMaxRawSourceBytes);
        pipe_.InitBuffer(rawOutputBuffer_, kMaxGatherElements);
    }

    __aicore__ inline void Process()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t rowBegin = core * tiling_.blockRows;
        if (rowBegin >= tiling_.outer) {
            return;
        }
        const uint64_t rowEnd = MinU64(rowBegin + tiling_.blockRows, tiling_.outer);
        const uint32_t inputElements =
            static_cast<uint32_t>(tiling_.inputRowBytes[0] / sizeof(ElementT));
        const uint32_t outputElements =
            static_cast<uint32_t>(tiling_.outputRowBytes / sizeof(ElementT));
        if (inputElements == 0 || outputElements == 0) {
            return;
        }
        const uint32_t rowQuantum = static_cast<uint32_t>(
            kDmaBlockBytes / GcdU64(tiling_.outputRowBytes, kDmaBlockBytes));
        uint32_t tileRows = kMaxGatherElements / outputElements;
        tileRows = tileRows / rowQuantum * rowQuantum;
        if (tileRows == 0) {
            // A whole alignment quantum can exceed the Gather capacity for
            // odd, wide output rows. Only core boundaries require alignment;
            // internal tiles are serialized by the MTE3 completion event.
            tileRows = kMaxGatherElements / outputElements;
        }
        while (tileRows > 1 && !TileFits(tileRows, inputElements, outputElements)) {
            tileRows -= tileRows > rowQuantum ? rowQuantum : 1;
        }

        for (uint64_t tileBegin = rowBegin; tileBegin < rowEnd; tileBegin += tileRows) {
            const uint32_t rows = static_cast<uint32_t>(
                MinU64(tileRows, rowEnd - tileBegin));
            ProcessTile(tileBegin, rows, inputElements, outputElements);
        }
    }

private:
    __aicore__ inline uint32_t AlignElements(uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline uint32_t SourceStrideElements(uint32_t rows, uint32_t inputElements)
    {
        const uint32_t actual = rows * inputElements;
        if constexpr (kConvertInt8) {
            return AlignElements(actual, kCastAlignmentElements);
        }
        const uint32_t blockElements = kDmaBlockBytes / sizeof(ElementT);
        return AlignElements(actual, blockElements);
    }

    __aicore__ inline bool TileFits(
        uint32_t rows, uint32_t inputElements, uint32_t outputElements)
    {
        const uint32_t outputCount = rows * outputElements;
        const uint32_t gatherCount = kConvertInt8 ?
            AlignElements(outputCount, kCastAlignmentElements) : outputCount;
        const uint32_t sourceStride = SourceStrideElements(rows, inputElements);
        return rows != 0 && gatherCount <= kMaxGatherElements &&
               sourceStride * tiling_.inputCount <= kMaxSourceElements &&
               (!kConvertInt8 || sourceStride * tiling_.inputCount <= kMaxRawSourceBytes);
    }

    __aicore__ inline void BuildIndices(
        uint32_t rows,
        uint32_t inputElements,
        uint32_t outputElements,
        uint32_t sourceStride,
        uint32_t gatherCount)
    {
        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        const uint32_t outputCount = rows * outputElements;
        // Build the row-major mapping with additions only. Division and
        // modulo per output lane are unnecessary for this regular tensor.
        uint32_t element = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
                uint32_t sourceElement = input * sourceStride + row * inputElements;
                for (uint32_t column = 0; column < inputElements; ++column) {
                    offsets.SetValue(element++, sourceElement++ * sizeof(GatherT));
                }
            }
        }
        for (; element < gatherCount; ++element) { offsets.SetValue(element, 0); }
        // SetValue runs on the scalar pipeline, so establish the dependency
        // explicitly before the vector pipeline consumes the offset table.
        const event_t scalarToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
    }

    __aicore__ inline void LoadSources(
        uint64_t rowBegin,
        uint32_t rows,
        uint32_t inputElements,
        uint32_t sourceStride)
    {
        const uint32_t actualElements = rows * inputElements;
        LocalTensor<GatherT> gatheredSource = sourceBuffer_.Get<GatherT>();
        if constexpr (kConvertInt8) {
            LocalTensor<int8_t> raw = rawSourceBuffer_.Get<int8_t>();
            const uint32_t dmaAlignedElements =
                AlignElements(actualElements, kDmaBlockBytes / sizeof(int8_t));
            for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
                GlobalTensor<int8_t> source;
                source.SetGlobalBuffer(
                    inputs_.GetDataPtr<int8_t>(input), tiling_.outer * inputElements);
                DataCopyExtParams params{
                    1, static_cast<uint32_t>(actualElements * sizeof(int8_t)), 0, 0, 0};
                // A2 DataCopyPad can append at most one 32-byte DMA block.
                // Pad only to that boundary here; requesting the complete
                // 256-element Cast alignment silently corrupts layouts whose
                // tile length is not already Cast-aligned (for example five
                // one-byte shards per row).
                DataCopyPadExtParams<int8_t> pad{
                    true, 0,
                    static_cast<uint8_t>(dmaAlignedElements - actualElements),
                    static_cast<int8_t>(0)};
                DataCopyPad(
                    raw[input * sourceStride], source[rowBegin * inputElements], params, pad);
            }
            const event_t inputToVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(inputToVector);
            WaitFlag<HardEvent::MTE2_V>(inputToVector);
            if (dmaAlignedElements < sourceStride) {
                // Duplicate<int8_t> is unavailable on A2.  Both offsets are
                // even because they are DMA/Cast aligned, so clear the rest
                // through a bit-preserving uint16 view before the wide Cast.
                LocalTensor<uint16_t> raw16 = raw.template ReinterpretCast<uint16_t>();
                const uint32_t tail16 = (sourceStride - dmaAlignedElements) / sizeof(uint16_t);
                for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
                    const uint32_t begin16 =
                        (input * sourceStride + dmaAlignedElements) / sizeof(uint16_t);
                    Duplicate(raw16[begin16], static_cast<uint16_t>(0), tail16);
                }
                PipeBarrier<PIPE_V>();
            }
            for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
                Cast(gatheredSource[input * sourceStride], raw[input * sourceStride],
                     RoundMode::CAST_NONE, sourceStride);
            }
            PipeBarrier<PIPE_V>();
        } else {
            const uint32_t blockElements = kDmaBlockBytes / sizeof(ElementT);
            for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
                GlobalTensor<ElementT> source;
                source.SetGlobalBuffer(
                    inputs_.GetDataPtr<ElementT>(input), tiling_.outer * inputElements);
                DataCopyExtParams params{
                    1, static_cast<uint32_t>(actualElements * sizeof(ElementT)), 0, 0, 0};
                DataCopyPadExtParams<ElementT> pad{
                    true, 0,
                    static_cast<uint8_t>(sourceStride - actualElements),
                    static_cast<ElementT>(0)};
                DataCopyPad(
                    gatheredSource[input * sourceStride],
                    source[rowBegin * inputElements], params, pad);
            }
            (void)blockElements;
            const event_t inputToVector = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(inputToVector);
            WaitFlag<HardEvent::MTE2_V>(inputToVector);
        }
    }

    __aicore__ inline void ProcessTile(
        uint64_t rowBegin,
        uint32_t rows,
        uint32_t inputElements,
        uint32_t outputElements)
    {
        // Every tile reuses the same four TBuf regions.  PipeBarrier only
        // orders instructions within a pipeline; it does not protect these
        // regions from the previous tile's asynchronous consumers.  Drain
        // vector reads before scalar/MTE2 overwrite the index/source buffers.
        const event_t vectorToScalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);
        const event_t vectorToInput = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vectorToInput);
        WaitFlag<HardEvent::V_MTE2>(vectorToInput);

        const uint32_t outputCount = rows * outputElements;
        const uint32_t gatherCount = kConvertInt8 ?
            AlignElements(outputCount, kCastAlignmentElements) : outputCount;
        const uint32_t sourceStride = SourceStrideElements(rows, inputElements);
        // Every full tile has the same UB layout. Retain its mapping until
        // the final short tile changes the row count.
        if (cachedIndexRows_ != rows) {
            BuildIndices(rows, inputElements, outputElements, sourceStride, gatherCount);
            cachedIndexRows_ = rows;
        }
        LoadSources(rowBegin, rows, inputElements, sourceStride);

        LocalTensor<GatherT> gatheredSource = sourceBuffer_.Get<GatherT>();
        LocalTensor<GatherT> gatheredOutput = outputBuffer_.Get<GatherT>();
        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        Gather(gatheredOutput, gatheredSource, offsets, 0, gatherCount);
        PipeBarrier<PIPE_V>();

        const uint64_t outputOffset = rowBegin * outputElements;
        DataCopyExtParams params{
            1, static_cast<uint32_t>(outputCount * sizeof(ElementT)), 0, 0, 0};
        if constexpr (kConvertInt8) {
            LocalTensor<int8_t> rawOutput = rawOutputBuffer_.Get<int8_t>();
            Cast(rawOutput, gatheredOutput, RoundMode::CAST_NONE, gatherCount);
            PipeBarrier<PIPE_V>();
            const event_t vectorToOutput = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vectorToOutput);
            WaitFlag<HardEvent::V_MTE3>(vectorToOutput);
            DataCopyPad(output_[outputOffset], rawOutput, params);
        } else {
            LocalTensor<ElementT> output =
                gatheredOutput.template ReinterpretCast<ElementT>();
            const event_t vectorToOutput = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vectorToOutput);
            WaitFlag<HardEvent::V_MTE3>(vectorToOutput);
            DataCopyPad(output_[outputOffset], output, params);
        }
        // Do not let the next tile's vector stage overwrite outputBuffer_ or
        // rawOutputBuffer_ while MTE3 still consumes it.
        const event_t outputToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(outputToVector);
        WaitFlag<HardEvent::MTE3_V>(outputToVector);
    }

private:
    // The Compare/Gather APIs are issued over at most 2,048 lanes.  Correct
    // inter-pipeline events above make this full tile deterministic; reducing
    // it to 768 was useful only as the hazard-localization probe.
    uint32_t cachedIndexRows_ = UINT32_MAX;
    static constexpr uint32_t kMaxGatherElements = 2048;
    static constexpr uint32_t kMaxSourceElements = 4096;
    static constexpr uint32_t kMaxRawSourceBytes = 4096;
    // INT8 <-> FP16 Cast consumes 128 INT8 lanes per 256-byte vector repeat.
    // Padding to exactly that quantum avoids reading an uninitialized tail.
    static constexpr uint32_t kCastAlignmentElements = 128;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> sourceBuffer_;
    TBuf<TPosition::VECCALC> outputBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    TBuf<TPosition::VECCALC> rawSourceBuffer_;
    TBuf<TPosition::VECCALC> rawOutputBuffer_;
    ListTensorDesc inputs_;
    GlobalTensor<ElementT> output_;
    TilingDataT tiling_{};
};

// Narrow shards of different widths share the same input-major to row-major
// mapping; prefix sums replace the equal-width assumption.
template <typename TilingDataT, typename ElementT, typename GatherT, bool kConvertInt8>
class KernelConcatRaggedNarrowRows {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(
            reinterpret_cast<__gm__ ElementT*>(y), tiling_.totalBytes / sizeof(ElementT));
        pipe_.InitBuffer(sourceBuffer_, kMaxSourceElements * sizeof(GatherT));
        pipe_.InitBuffer(outputBuffer_, kMaxGatherElements * sizeof(GatherT));
        pipe_.InitBuffer(indexBuffer_, kMaxGatherElements * sizeof(uint32_t));
        pipe_.InitBuffer(rawSourceBuffer_, kMaxRawSourceBytes);
        pipe_.InitBuffer(rawOutputBuffer_, kMaxGatherElements);
    }

    __aicore__ inline void Process()
    {
        const uint64_t core = GetBlockIdx();
        const uint64_t rowBegin = core * tiling_.blockRows;
        if (rowBegin >= tiling_.outer) {
            return;
        }
        const uint64_t rowEnd = MinU64(rowBegin + tiling_.blockRows, tiling_.outer);
        const uint32_t inputElements =
            static_cast<uint32_t>(tiling_.inputRowBytes[0] / sizeof(ElementT));
        const uint32_t outputElements =
            static_cast<uint32_t>(tiling_.outputRowBytes / sizeof(ElementT));
        if (inputElements == 0 || outputElements == 0) {
            return;
        }
        // Core ownership is 32-byte aligned. Internal row tiles may end
        // inside a block because this core drains every MTE3 before reuse.
        uint32_t tileRows = kMaxGatherElements / outputElements;
        while (tileRows > 1 && !TileFits(tileRows, inputElements, outputElements)) {
            --tileRows;
        }

        for (uint64_t tileBegin = rowBegin; tileBegin < rowEnd; tileBegin += tileRows) {
            const uint32_t rows = static_cast<uint32_t>(
                MinU64(tileRows, rowEnd - tileBegin));
            ProcessTile(tileBegin, rows, inputElements, outputElements);
        }
    }

private:
    __aicore__ inline uint32_t AlignElements(uint32_t value, uint32_t alignment)
    {
        return (value + alignment - 1) / alignment * alignment;
    }

    __aicore__ inline uint32_t SourceStrideElements(uint32_t rows, uint32_t inputElements)
    {
        const uint32_t actual = rows * inputElements;
        if constexpr (kConvertInt8) {
            return AlignElements(actual, kCastAlignmentElements);
        }
        const uint32_t blockElements = kDmaBlockBytes / sizeof(ElementT);
        return AlignElements(actual, blockElements);
    }

    __aicore__ inline bool TileFits(
        uint32_t rows, uint32_t inputElements, uint32_t outputElements)
    {
        (void)inputElements;
        const uint32_t outputCount = rows * outputElements;
        const uint32_t gatherCount = kConvertInt8 ?
            AlignElements(outputCount, kCastAlignmentElements) : outputCount;
        uint32_t sourceElements = 0;
        for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
            const uint32_t width = tiling_.inputRowBytes[input] / sizeof(ElementT);
            sourceElements += SourceStrideElements(rows, width);
        }
        return rows != 0 && gatherCount <= kMaxGatherElements &&
               sourceElements <= kMaxSourceElements &&
               (!kConvertInt8 || sourceElements <= kMaxRawSourceBytes);
    }

    __aicore__ inline void BuildIndices(
        uint32_t rows, uint32_t inputElements, uint32_t outputElements,
        uint32_t sourceStride, uint32_t gatherCount)
    {
        (void)inputElements; (void)outputElements; (void)sourceStride;
        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        uint32_t element = 0;
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t sourceBase = 0;
            for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
                const uint32_t width = tiling_.inputRowBytes[input] / sizeof(ElementT);
                uint32_t sourceElement = sourceBase + row * width;
                for (uint32_t column = 0; column < width; ++column) {
                    offsets.SetValue(element++, sourceElement++ * sizeof(GatherT));
                }
                sourceBase += SourceStrideElements(rows, width);
            }
        }
        for (; element < gatherCount; ++element) { offsets.SetValue(element, 0); }
        const event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(ready); WaitFlag<HardEvent::S_V>(ready);
    }

    __aicore__ inline void LoadSources(
        uint64_t rowBegin, uint32_t rows, uint32_t inputElements, uint32_t sourceStride)
    {
        (void)inputElements; (void)sourceStride;
        LocalTensor<GatherT> gatheredSource = sourceBuffer_.Get<GatherT>();
        uint32_t totalSourceElements = 0;
        for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
            totalSourceElements += SourceStrideElements(
                rows, tiling_.inputRowBytes[input] / sizeof(ElementT));
        }
        if constexpr (kConvertInt8) {
            auto raw = rawSourceBuffer_.Get<int8_t>();
            auto raw16 = raw.template ReinterpretCast<uint16_t>();
            Duplicate(raw16, static_cast<uint16_t>(0), totalSourceElements / 2);
            const event_t cleared = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(cleared); WaitFlag<HardEvent::V_MTE2>(cleared);
            uint32_t sourceBase = 0;
            for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
                const uint32_t width = tiling_.inputRowBytes[input];
                const uint32_t actual = rows * width;
                const uint32_t dmaAligned = AlignElements(actual, kDmaBlockBytes);
                GlobalTensor<int8_t> source;
                source.SetGlobalBuffer(inputs_.GetDataPtr<int8_t>(input), tiling_.outer * width);
                DataCopyExtParams params{1, actual, 0, 0, 0};
                DataCopyPadExtParams<int8_t> pad{true, 0,
                    static_cast<uint8_t>(dmaAligned - actual), static_cast<int8_t>(0)};
                DataCopyPad(raw[sourceBase], source[rowBegin * width], params, pad);
                sourceBase += SourceStrideElements(rows, width);
            }
            const event_t loaded = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(loaded); WaitFlag<HardEvent::MTE2_V>(loaded);
            Cast(gatheredSource, raw, RoundMode::CAST_NONE, totalSourceElements);
            PipeBarrier<PIPE_V>();
        } else {
            uint32_t sourceBase = 0;
            for (uint32_t input = 0; input < tiling_.inputCount; ++input) {
                const uint32_t width = tiling_.inputRowBytes[input] / sizeof(ElementT);
                const uint32_t actual = rows * width;
                const uint32_t aligned = SourceStrideElements(rows, width);
                GlobalTensor<ElementT> source;
                source.SetGlobalBuffer(inputs_.GetDataPtr<ElementT>(input), tiling_.outer * width);
                DataCopyExtParams params{1, static_cast<uint32_t>(actual * sizeof(ElementT)), 0, 0, 0};
                DataCopyPadExtParams<ElementT> pad{true, 0,
                    static_cast<uint8_t>(aligned - actual), static_cast<ElementT>(0)};
                DataCopyPad(gatheredSource[sourceBase], source[rowBegin * width], params, pad);
                sourceBase += aligned;
            }
            const event_t loaded = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(loaded); WaitFlag<HardEvent::MTE2_V>(loaded);
        }
    }

    __aicore__ inline void ProcessTile(
        uint64_t rowBegin,
        uint32_t rows,
        uint32_t inputElements,
        uint32_t outputElements)
    {
        // Every tile reuses the same four TBuf regions.  PipeBarrier only
        // orders instructions within a pipeline; it does not protect these
        // regions from the previous tile's asynchronous consumers.  Drain
        // vector reads before scalar/MTE2 overwrite the index/source buffers.
        const event_t vectorToScalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(vectorToScalar);
        WaitFlag<HardEvent::V_S>(vectorToScalar);
        const event_t vectorToInput = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(vectorToInput);
        WaitFlag<HardEvent::V_MTE2>(vectorToInput);

        const uint32_t outputCount = rows * outputElements;
        const uint32_t gatherCount = kConvertInt8 ?
            AlignElements(outputCount, kCastAlignmentElements) : outputCount;
        const uint32_t sourceStride = SourceStrideElements(rows, inputElements);
        // Every full tile has the same UB layout. Retain its mapping until
        // the final short tile changes the row count.
        if (cachedIndexRows_ != rows) {
            BuildIndices(rows, inputElements, outputElements, sourceStride, gatherCount);
            cachedIndexRows_ = rows;
        }
        LoadSources(rowBegin, rows, inputElements, sourceStride);

        LocalTensor<GatherT> gatheredSource = sourceBuffer_.Get<GatherT>();
        LocalTensor<GatherT> gatheredOutput = outputBuffer_.Get<GatherT>();
        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        Gather(gatheredOutput, gatheredSource, offsets, 0, gatherCount);
        PipeBarrier<PIPE_V>();

        const uint64_t outputOffset = rowBegin * outputElements;
        DataCopyExtParams params{
            1, static_cast<uint32_t>(outputCount * sizeof(ElementT)), 0, 0, 0};
        if constexpr (kConvertInt8) {
            LocalTensor<int8_t> rawOutput = rawOutputBuffer_.Get<int8_t>();
            Cast(rawOutput, gatheredOutput, RoundMode::CAST_NONE, gatherCount);
            PipeBarrier<PIPE_V>();
            const event_t vectorToOutput = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vectorToOutput);
            WaitFlag<HardEvent::V_MTE3>(vectorToOutput);
            DataCopyPad(output_[outputOffset], rawOutput, params);
        } else {
            LocalTensor<ElementT> output =
                gatheredOutput.template ReinterpretCast<ElementT>();
            const event_t vectorToOutput = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(vectorToOutput);
            WaitFlag<HardEvent::V_MTE3>(vectorToOutput);
            DataCopyPad(output_[outputOffset], output, params);
        }
        // Do not let the next tile's vector stage overwrite outputBuffer_ or
        // rawOutputBuffer_ while MTE3 still consumes it.
        const event_t outputToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(outputToVector);
        WaitFlag<HardEvent::MTE3_V>(outputToVector);
    }

private:
    // The Compare/Gather APIs are issued over at most 2,048 lanes.  Correct
    // inter-pipeline events above make this full tile deterministic; reducing
    // it to 768 was useful only as the hazard-localization probe.
    uint32_t cachedIndexRows_ = UINT32_MAX;
    static constexpr uint32_t kMaxGatherElements = 2048;
    static constexpr uint32_t kMaxSourceElements = 4096;
    static constexpr uint32_t kMaxRawSourceBytes = 4096;
    // INT8 <-> FP16 Cast consumes 128 INT8 lanes per 256-byte vector repeat.
    // Padding to exactly that quantum avoids reading an uninitialized tail.
    static constexpr uint32_t kCastAlignmentElements = 128;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> sourceBuffer_;
    TBuf<TPosition::VECCALC> outputBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    TBuf<TPosition::VECCALC> rawSourceBuffer_;
    TBuf<TPosition::VECCALC> rawOutputBuffer_;
    ListTensorDesc inputs_;
    GlobalTensor<ElementT> output_;
    TilingDataT tiling_{};
};


template <typename TilingDataT>
class KernelConcatDirectFlat {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), tiling_.totalBytes);
        pipe_.InitBuffer(buffer_, tiling_.tileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        LocalTensor<uint8_t> local = buffer_.Get<uint8_t>();
        for (uint32_t input = core; input < tiling_.inputCount; input += blockNum) {
            const uint64_t outputBegin = tiling_.inputRowBytes[input];
            const uint64_t outputEnd = input + 1 < tiling_.inputCount ?
                                           tiling_.inputRowBytes[input + 1] : tiling_.totalBytes;
            uint64_t sourceOffset = 0;
            uint64_t outputOffset = outputBegin;
            uint64_t bytes = outputEnd - outputBegin;
            GlobalTensor<uint8_t> source;
            source.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(input), bytes);
            ConfigureConcatStreamRead(source, tiling_.totalBytes, bytes);
            while (bytes != 0) {
                const uint64_t once = MinU64(bytes, tiling_.tileBytes);
                DataCopyExtParams params{1, static_cast<uint32_t>(once), 0, 0, 0};
                DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                DataCopyPad(local, source[sourceOffset], params, pad);
                PipeBarrier<PIPE_ALL>();
                DataCopyPad(output_[outputOffset], local, params);
                PipeBarrier<PIPE_ALL>();
                sourceOffset += once;
                outputOffset += once;
                bytes -= once;
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> buffer_;
    ListTensorDesc inputs_;
    GlobalTensor<uint8_t> output_;
    TilingDataT tiling_{};
};

template <typename TilingDataT>
class KernelConcatDirectFlatQueue {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y), tiling_.totalBytes);
        pipe_.InitBuffer(moveQueue_, 2, tiling_.tileBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        bool hasPending = false;
        uint64_t pendingOutputOffset = 0;
        uint32_t pendingBytes = 0;
        for (uint32_t input = core; input < tiling_.inputCount; input += blockNum) {
            const uint64_t outputBegin = tiling_.inputRowBytes[input];
            const uint64_t outputEnd = input + 1 < tiling_.inputCount ?
                                           tiling_.inputRowBytes[input + 1] : tiling_.totalBytes;
            uint64_t sourceOffset = 0;
            uint64_t outputOffset = outputBegin;
            uint64_t bytes = outputEnd - outputBegin;
            GlobalTensor<uint8_t> source;
            source.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(input), bytes);
            ConfigureConcatStreamRead(source, tiling_.totalBytes, bytes);
            while (bytes != 0) {
                const uint64_t once = MinU64(bytes, tiling_.tileBytes);
                DataCopyExtParams params{1, static_cast<uint32_t>(once), 0, 0, 0};
                DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                LocalTensor<uint8_t> local = moveQueue_.AllocTensor<uint8_t>();
                DataCopyPad(local, source[sourceOffset], params, pad);
                moveQueue_.EnQue(local);
                if (hasPending) {
                    local = moveQueue_.DeQue<uint8_t>();
                    DataCopyExtParams pendingParams{1, pendingBytes, 0, 0, 0};
                    DataCopyPad(output_[pendingOutputOffset], local, pendingParams);
                    moveQueue_.FreeTensor(local);
                }
                pendingOutputOffset = outputOffset;
                pendingBytes = static_cast<uint32_t>(once);
                hasPending = true;
                sourceOffset += once;
                outputOffset += once;
                bytes -= once;
            }
        }
        if (hasPending) {
            LocalTensor<uint8_t> local = moveQueue_.DeQue<uint8_t>();
            DataCopyExtParams pendingParams{1, pendingBytes, 0, 0, 0};
            DataCopyPad(output_[pendingOutputOffset], local, pendingParams);
            moveQueue_.FreeTensor(local);
        }
    }

private:
    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> moveQueue_;
    ListTensorDesc inputs_;
    GlobalTensor<uint8_t> output_;
    TilingDataT tiling_{};
};

template <typename TilingDataT>
class KernelConcatCacheLines {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        outputBase_ = reinterpret_cast<__gm__ uint64_t*>(y);
    }

    __aicore__ inline void Process()
    {
        constexpr uint64_t wordsPerCacheLine = 8;
        const uint32_t core = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        for (uint32_t input = core; input < tiling_.inputCount; input += blockNum) {
            const uint64_t outputBegin = tiling_.inputRowBytes[input];
            const uint64_t outputEnd = input + 1 < tiling_.inputCount ?
                                           tiling_.inputRowBytes[input + 1] : tiling_.totalBytes;
            const uint64_t wordCount = (outputEnd - outputBegin) / sizeof(uint64_t);
            GlobalTensor<uint64_t> source;
            GlobalTensor<uint64_t> destination;
            source.SetGlobalBuffer(inputs_.GetDataPtr<uint64_t>(input), wordCount);
            destination.SetGlobalBuffer(outputBase_ + outputBegin / sizeof(uint64_t), wordCount);
            for (uint64_t line = 0; line < wordCount; line += wordsPerCacheLine) {
                DataCacheCleanAndInvalid<
                    uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(source[line]);
                for (uint64_t word = 0; word < wordsPerCacheLine; ++word) {
                    destination.SetValue(line + word, source.GetValue(line + word));
                }
                DataCacheCleanAndInvalid<
                    uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(destination[line]);
            }
        }
    }

private:
    ListTensorDesc inputs_;
    __gm__ uint64_t* outputBase_ = nullptr;
    TilingDataT tiling_{};
};

// Flat FP16 TensorLists containing one value per input are dominated by
// pointer chasing in the scalar gather path.  One AIV owns exactly one
// 32-byte output block.  It DMA-stages the 16 independent values into aligned
// UB slots, then one vector gather compacts those slots before the final DMA.
// The transformation is exact: dst[j] = src_j[0], with no arithmetic on data.
template <typename TilingDataT>
class KernelConcatUnitFp16Gather {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(y), tiling_.inputCount);
        pipe_.InitBuffer(sourceBuffer_, kInputsPerCore * kSlotBytes);
        pipe_.InitBuffer(outputBuffer_, kDmaBlockBytes);
        pipe_.InitBuffer(indexBuffer_, kIndexBytes);
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t kSlotElements = kSlotBytes / sizeof(uint16_t);
        const uint32_t begin = GetBlockIdx() * kInputsPerCore;
        if (begin >= tiling_.inputCount) {
            return;
        }
        const uint32_t count = static_cast<uint32_t>(MinU64(
            kInputsPerCore, static_cast<uint64_t>(tiling_.inputCount - begin)));
        LocalTensor<uint16_t> sourceLocal = sourceBuffer_.Get<uint16_t>();
        LocalTensor<uint16_t> outputLocal = outputBuffer_.Get<uint16_t>();
        DataCopyExtParams inputParams{1, sizeof(uint16_t), 0, 0, 0};
        DataCopyPadExtParams<uint16_t> pad{false, 0, 0, 0};
        bool arithmeticPointers = count > 1;
        uint64_t firstAddress = reinterpret_cast<uint64_t>(
            inputs_.GetDataPtr<uint16_t>(begin));
        uint64_t sourceStrideBytes = 0;
        if (arithmeticPointers) {
            const uint64_t secondAddress = reinterpret_cast<uint64_t>(
                inputs_.GetDataPtr<uint16_t>(begin + 1));
            sourceStrideBytes = secondAddress - firstAddress;
            arithmeticPointers = sourceStrideBytes >= sizeof(uint16_t) &&
                                 sourceStrideBytes <= UINT32_MAX;
            for (uint32_t input = 2; input < count && arithmeticPointers; ++input) {
                const uint64_t address = reinterpret_cast<uint64_t>(
                    inputs_.GetDataPtr<uint16_t>(begin + input));
                arithmeticPointers = address == firstAddress +
                                      static_cast<uint64_t>(input) * sourceStrideBytes;
            }
        }
        if (arithmeticPointers) {
            GlobalTensor<uint16_t> source;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(firstAddress), count);
            // A single 2-D DMA gathers all regularly spaced TensorList
            // elements into the same padded UB slots used by the fallback.
            inputParams = DataCopyExtParams{
                static_cast<uint16_t>(count), sizeof(uint16_t),
                static_cast<uint32_t>(sourceStrideBytes - sizeof(uint16_t)), 0, 0};
            DataCopyPad(sourceLocal, source, inputParams, pad);
        } else {
            for (uint32_t input = 0; input < count; ++input) {
                GlobalTensor<uint16_t> source;
                source.SetGlobalBuffer(inputs_.GetDataPtr<uint16_t>(begin + input), 1);
                inputParams = DataCopyExtParams{1, sizeof(uint16_t), 0, 0, 0};
                DataCopyPad(sourceLocal[input * kSlotElements], source, inputParams, pad);
                if ((input & 3U) == 3U) {
                    PipeBarrier<PIPE_MTE2>();
                }
            }
        }
        PipeBarrier<PIPE_MTE2>();

        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        // CANN 8.5's Arange implementation has a tail-specific event path
        // which is not reliable for the short (1..16 element) gather used
        // here on the reference 910B-family device.  Materialize the 32-byte
        // slot offsets explicitly;
        // this is a handful of scalar stores, but keeps the result bit-exact.
        for (uint32_t index = 0; index < count; ++index) {
            offsets.SetValue(index, index * kSlotBytes);
        }
        PipeBarrier<PIPE_ALL>();
        Gather(outputLocal, sourceLocal, offsets, 0, count);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(count * sizeof(uint16_t)), 0, 0, 0};
        DataCopyPad(output_[begin], outputLocal, outputParams);
        PipeBarrier<PIPE_MTE3>();
    }

private:
    static constexpr uint32_t kInputsPerCore = 16;
    static constexpr uint32_t kSlotBytes = 32;
    static constexpr uint32_t kIndexBytes = 256;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> sourceBuffer_;
    TBuf<TPosition::VECCALC> outputBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    ListTensorDesc inputs_;
    GlobalTensor<uint16_t> output_;
    TilingDataT tiling_{};
};

// The same regular-allocation transformation for two FP16 values per input.
// Keeping this as a separate key avoids adding a runtime width branch to the
// already hot one-value kernel.  Each input occupies one 32-byte UB slot;
// one 2-D MTE2 command stages up to 16 regularly spaced tensors and vgather
// compacts their 32 values before one MTE3 command.
template <typename TilingDataT, uint32_t kValuesPerInput>
class KernelConcatNarrowFp16Gather {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingDataT& data)
    {
        tiling_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(y),
                                tiling_.inputCount * kValuesPerInput);
        pipe_.InitBuffer(sourceBuffer_, kInputsPerCore * kSlotBytes);
        pipe_.InitBuffer(outputBuffer_, kOutputBytes);
        pipe_.InitBuffer(indexBuffer_, kIndexBytes);
    }

    __aicore__ inline void Process()
    {
        const uint32_t begin = GetBlockIdx() * kInputsPerCore;
        if (begin >= tiling_.inputCount) {
            return;
        }
        const uint32_t count = static_cast<uint32_t>(MinU64(
            kInputsPerCore, static_cast<uint64_t>(tiling_.inputCount - begin)));
        LocalTensor<uint8_t> sourceBytes = sourceBuffer_.Get<uint8_t>();
        LocalTensor<uint16_t> sourceLocal = sourceBytes.template ReinterpretCast<uint16_t>();
        LocalTensor<uint16_t> outputLocal = outputBuffer_.Get<uint16_t>();
        DataCopyExtParams inputParams{1, kInputBytes, 0, 0, 0};
        // For GM->UB 2-D copies on 910B, a short block is physically
        // transferred in a 32-byte slot.  Explicitly request right padding
        // instead of relying on the undocumented implicit stride behavior;
        // otherwise block lengths 8/16 bytes can be truncated to 4 bytes.
        DataCopyPadExtParams<uint8_t> pad{
            true, 0, static_cast<uint8_t>(kSlotBytes - kInputBytes), 0};

        bool arithmeticPointers = count > 1;
        const uint64_t firstAddress = reinterpret_cast<uint64_t>(
            inputs_.GetDataPtr<uint16_t>(begin));
        uint64_t sourceStrideBytes = 0;
        if (arithmeticPointers) {
            const uint64_t secondAddress = reinterpret_cast<uint64_t>(
                inputs_.GetDataPtr<uint16_t>(begin + 1));
            sourceStrideBytes = secondAddress - firstAddress;
            arithmeticPointers = sourceStrideBytes >= kInputBytes &&
                                  sourceStrideBytes <= UINT32_MAX;
            for (uint32_t input = 2; input < count && arithmeticPointers; ++input) {
                const uint64_t address = reinterpret_cast<uint64_t>(
                    inputs_.GetDataPtr<uint16_t>(begin + input));
                arithmeticPointers = address == firstAddress +
                                      static_cast<uint64_t>(input) * sourceStrideBytes;
            }
        }
        if (arithmeticPointers) {
            GlobalTensor<uint8_t> source;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(firstAddress),
                                   count * kInputBytes);
            inputParams = DataCopyExtParams{
                static_cast<uint16_t>(count), kInputBytes,
                static_cast<uint32_t>(sourceStrideBytes - kInputBytes), 0, 0};
            DataCopyPad(sourceBytes, source, inputParams, pad);
        } else {
            for (uint32_t input = 0; input < count; ++input) {
                GlobalTensor<uint8_t> source;
                source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(
                                           inputs_.GetDataPtr<uint16_t>(begin + input)),
                                       kInputBytes);
                inputParams = DataCopyExtParams{1, kInputBytes, 0, 0, 0};
                DataCopyPad(sourceBytes[input * kSlotBytes], source, inputParams, pad);
                if ((input & 3U) == 3U) {
                    PipeBarrier<PIPE_MTE2>();
                }
            }
        }
        PipeBarrier<PIPE_MTE2>();

        LocalTensor<uint32_t> offsets = indexBuffer_.Get<uint32_t>();
        for (uint32_t input = 0; input < count; ++input) {
            const uint32_t slotBase = input * kSlotBytes;
            for (uint32_t value = 0; value < kValuesPerInput; ++value) {
                offsets.SetValue(input * kValuesPerInput + value,
                                 slotBase + value * sizeof(uint16_t));
            }
        }
        PipeBarrier<PIPE_ALL>();
        Gather(outputLocal, sourceLocal, offsets, 0, count * kValuesPerInput);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(count * kInputBytes), 0, 0, 0};
        DataCopyPad(output_[begin * kValuesPerInput], outputLocal, outputParams);
        PipeBarrier<PIPE_MTE3>();
    }

private:
    static constexpr uint32_t kInputsPerCore = 16;
    static constexpr uint32_t kInputBytes = kValuesPerInput * sizeof(uint16_t);
    static constexpr uint32_t kSlotBytes = 32;
    static constexpr uint32_t kSlotElements = kSlotBytes / sizeof(uint16_t);
    static constexpr uint32_t kOutputBytes = kInputsPerCore * kInputBytes;
    static constexpr uint32_t kIndexBytes = kInputsPerCore * kValuesPerInput * sizeof(uint32_t);
    TPipe pipe_;
    TBuf<TPosition::VECCALC> sourceBuffer_;
    TBuf<TPosition::VECCALC> outputBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    ListTensorDesc inputs_;
    GlobalTensor<uint16_t> output_;
    TilingDataT tiling_{};
};

} // namespace


template <typename ElementT, typename GatherT, bool kConvert>
class KernelConcatGatherRows {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
        const ConcatGatherRowsHeader& data) {
        data_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ ElementT*>(y));
        for (uint32_t i = 0; i < data_.base.inputCount; ++i)
            bases_[i] = inputs_.GetDataPtr<ElementT>(i);
        pipe_.InitBuffer(source_, kSourceElements * sizeof(GatherT));
        pipe_.InitBuffer(outputBuffer_, kSlabElements * sizeof(GatherT));
        pipe_.InitBuffer(indices_, 4096 * sizeof(uint32_t));
        pipe_.InitBuffer(steps_, 4096 * sizeof(uint32_t));
        pipe_.InitBuffer(working_, 4096 * sizeof(uint32_t));
        if constexpr (kConvert) {
            pipe_.InitBuffer(raw_, kSourceElements);
            pipe_.InitBuffer(rawOutput_, kSlabElements);
        }
        GlobalTensor<uint32_t> map;
        map.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(tiling + sizeof(ConcatGatherRowsHeader)));
        DataCopyExtParams copy{1, data_.gatherCount * 4, 0, 0, 0};
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        DataCopyPad(indices_.Get<uint32_t>(), map, copy, pad);
        DataCopyPad(steps_.Get<uint32_t>(), map[4096], copy, pad);
        Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void Process() {
        const auto& d = data_.base;
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * d.blockRows;
        const uint64_t end = MinU64(begin + d.blockRows, d.outer);
        const uint32_t outputWidth = d.outputRowBytes / sizeof(ElementT);
        for (uint64_t row = begin; row < end; row += data_.tileRows) {
            const uint32_t rows = static_cast<uint32_t>(MinU64(data_.tileRows, end - row));
            Sync<HardEvent::V_MTE2>();
            auto values = source_.Get<GatherT>();
            if constexpr (kConvert) {
                Duplicate(raw_.Get<uint16_t>(), static_cast<uint16_t>(0), data_.sourceElements / 2);
                Sync<HardEvent::V_MTE2>();
            }
            for (uint32_t i = 0; i < d.inputCount; ++i) {
                const uint32_t width = d.inputRowBytes[i] / sizeof(ElementT);
                GlobalTensor<ElementT> input; input.SetGlobalBuffer(bases_[i]);
            ConfigureConcatStreamRead(input, data_.base.totalBytes, static_cast<uint64_t>(rows) * width * sizeof(ElementT));
                DataCopyExtParams copy{1, static_cast<uint32_t>(rows * width * sizeof(ElementT)), 0, 0, 0};
                DataCopyPadExtParams<ElementT> pad{false, 0, 0, 0};
                if constexpr (kConvert)
                    DataCopyPad(raw_.Get<ElementT>()[data_.inputOffsets[i]], input[row * width], copy, pad);
                else
                    DataCopyPad(values.template ReinterpretCast<ElementT>()[data_.inputOffsets[i]], input[row * width], copy, pad);
            }
            Sync<HardEvent::MTE2_V>();
            if constexpr (kConvert) {
                Cast(values, raw_.Get<int8_t>(), RoundMode::CAST_NONE, data_.sourceElements);
                PipeBarrier<PIPE_V>();
            }
            auto result = outputBuffer_.Get<GatherT>();
            auto base = indices_.Get<uint32_t>();
            auto step = steps_.Get<int32_t>();
            auto working = working_.Get<int32_t>();
            const uint32_t groups = (rows + data_.groupRows - 1) / data_.groupRows;
            for (uint32_t group = 0; group < groups; ++group) {
                LocalTensor<uint32_t> index = base;
                if (group != 0) {
                    auto previous = group == 1 ? base.template ReinterpretCast<int32_t>() : working;
                    Add(working, previous, step, data_.gatherCount);
                    PipeBarrier<PIPE_V>();
                    index = working.template ReinterpretCast<uint32_t>();
                }
                const uint32_t validRows = static_cast<uint32_t>(MinU64(data_.groupRows, rows - group * data_.groupRows));
                const uint32_t count = validRows * outputWidth;
                const uint32_t gatherCount = kConvert ? (count + 127) / 128 * 128 : count;
                for (uint32_t offset = 0; offset < gatherCount; offset += 2048) {
                    const uint32_t once = gatherCount - offset < 2048 ? gatherCount - offset : 2048;
                    Gather(result[group * data_.gatherCount + offset], values, index[offset], 0, once);
                }
                PipeBarrier<PIPE_V>();
            }
            if constexpr (kConvert) {
                auto rawOut = rawOutput_.Get<int8_t>();
                Cast(rawOut, result, RoundMode::CAST_NONE, groups * data_.gatherCount);
                Sync<HardEvent::V_MTE3>();
                Store(row, rows, outputWidth, rawOut);
            } else {
                Sync<HardEvent::V_MTE3>();
                Store(row, rows, outputWidth, result.template ReinterpretCast<ElementT>());
            }
            Sync<HardEvent::MTE3_V>();
        }
    }
private:
    __aicore__ inline void Store(uint64_t row, uint32_t rows, uint32_t width,
        const LocalTensor<ElementT>& values) {
        const uint32_t fullGroups = rows / data_.groupRows;
        const uint32_t fullBytes = data_.groupRows * width * sizeof(ElementT);
        const uint32_t consumed = static_cast<uint32_t>(AlignUp32(fullBytes));
        if (fullGroups != 0) {
            DataCopyExtParams store{static_cast<uint16_t>(fullGroups), fullBytes,
                static_cast<uint32_t>((data_.gatherCount * sizeof(ElementT) - consumed) / 32), 0, 0};
            DataCopyPad(output_[row * width], values, store);
        }
        const uint32_t tailRows = rows - fullGroups * data_.groupRows;
        if (tailRows != 0) {
            DataCopyExtParams tail{1, static_cast<uint32_t>(tailRows * width * sizeof(ElementT)), 0, 0, 0};
            DataCopyPad(output_[(row + fullGroups * data_.groupRows) * width], values[fullGroups * data_.gatherCount], tail);
        }
    }
    template <HardEvent event> __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
        SetFlag<event>(id); WaitFlag<event>(id);
    }
    static constexpr uint32_t kSourceElements = sizeof(ElementT) == 4 ? 16384 : 20480;
    static constexpr uint32_t kSlabElements = sizeof(ElementT) == 4 ? 12288 : 16384;
    ConcatGatherRowsHeader data_{};
    TPipe pipe_;
    TBuf<TPosition::VECCALC> source_, outputBuffer_, indices_, steps_, working_, raw_, rawOutput_;
    ListTensorDesc inputs_;
    __gm__ ElementT* bases_[S9_CONCAT_SMALL_INPUTS];
    GlobalTensor<ElementT> output_;
};


// Full-slab static map eliminates the per-group index Add / step table.
// Queue order: prefetch next input -> gather current -> store current.
// TQue carries MTE2->V, V->MTE2, V->MTE3, MTE3->V dependencies.
template <typename LaneT, typename GatherT, bool kConvert>
class KernelConcatPipeline {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
        const ConcatPipelineHeader& data) {
        d_ = data;
        ListTensorDesc inputs(reinterpret_cast<__gm__ void*>(x));
        auto* packed = reinterpret_cast<__gm__ ConcatPipelineTilingData*>(tiling);
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            widths_[i] = packed->inputRowBytes[i] / sizeof(LaneT);
            offsets_[i] = packed->inputOffsets[i];
            bases_[i] = inputs.GetDataPtr<LaneT>(packed->originalInputs[i]);
        }
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ LaneT*>(y));
        pipe_.InitBuffer(inputQueue_, 2, d_.sourceElements * sizeof(LaneT));
        pipe_.InitBuffer(outputQueue_, 2, d_.mapCount * sizeof(LaneT));
        pipe_.InitBuffer(map_, d_.mapCount * sizeof(uint32_t));
        if constexpr (kConvert) {
            pipe_.InitBuffer(converted_, d_.sourceElements * sizeof(GatherT));
            pipe_.InitBuffer(gathered_, d_.mapCount * sizeof(GatherT));
        }
        GlobalTensor<uint32_t> indices;
        indices.SetGlobalBuffer(packed->indices);
        DataCopyExtParams copy{1, d_.mapCount * 4, 0, 0, 0};
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        DataCopyPad(map_.Get<uint32_t>(), indices, copy, pad);
        const event_t ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready); WaitFlag<HardEvent::MTE2_V>(ready);
    }
    __aicore__ inline void Process() {
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * d_.blockRows;
        const uint64_t end = MinU64(begin + d_.blockRows, d_.outer);
        if (begin >= end) { return; }
        CopyIn(begin, static_cast<uint32_t>(MinU64(d_.tileRows, end - begin)));
        for (uint64_t row = begin; row < end; row += d_.tileRows) {
            const uint32_t rows = static_cast<uint32_t>(MinU64(d_.tileRows, end - row));
            const uint64_t next = row + d_.tileRows;
            if (next < end) CopyIn(next, static_cast<uint32_t>(MinU64(d_.tileRows, end - next)));
            Compute(rows);
            CopyOut(row, rows);
        }
    }
private:
    __aicore__ inline void CopyIn(uint64_t row, uint32_t rows) {
        auto local = inputQueue_.AllocTensor<LaneT>();
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            const uint32_t width = widths_[i];
            GlobalTensor<LaneT> input; input.SetGlobalBuffer(bases_[i]);
            ConfigureConcatStreamRead(input, d_.outer * d_.outputRowBytes, static_cast<uint64_t>(rows) * width * sizeof(LaneT));
            DataCopyExtParams copy{1, static_cast<uint32_t>(rows * width * sizeof(LaneT)), 0, 0, 0};
            DataCopyPadExtParams<LaneT> pad{false, 0, 0, 0};
            DataCopyPad(local[offsets_[i]], input[row * width], copy, pad);
        }
        inputQueue_.EnQue(local);
    }
    __aicore__ inline void Compute(uint32_t rows) {
        auto input = inputQueue_.DeQue<LaneT>();
        auto output = outputQueue_.AllocTensor<LaneT>();
        auto values = input.template ReinterpretCast<GatherT>();
        auto result = output.template ReinterpretCast<GatherT>();
        const uint32_t count = rows * (d_.outputRowBytes / sizeof(LaneT));
        uint32_t gatherCount = count;
        if constexpr (kConvert) {
            values = converted_.Get<GatherT>();
            result = gathered_.Get<GatherT>();
            Cast(values, input, RoundMode::CAST_NONE, d_.sourceElements);
            PipeBarrier<PIPE_V>();
            gatherCount = (count + 127) / 128 * 128;
        }
        auto indices = map_.Get<uint32_t>();
        for (uint32_t offset = 0; offset < gatherCount; offset += 2048) {
            const uint32_t once = static_cast<uint32_t>(MinU64(2048, gatherCount - offset));
            Gather(result[offset], values, indices[offset], 0, once);
        }
        if constexpr (kConvert) {
            PipeBarrier<PIPE_V>();
            Cast(output, result, RoundMode::CAST_NONE, gatherCount);
        }
        outputQueue_.EnQue(output);
        inputQueue_.FreeTensor(input);
    }
    __aicore__ inline void CopyOut(uint64_t row, uint32_t rows) {
        auto local = outputQueue_.DeQue<LaneT>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(rows * d_.outputRowBytes), 0, 0, 0};
        DataCopyPad(output_[row * (d_.outputRowBytes / sizeof(LaneT))], local, copy);
        if ((d_.tileRows * d_.outputRowBytes) % 32 != 0) {
            // Adjacent partial 32B writes belong to this same core. Finish
            // the preceding MTE3 transaction before another can touch it.
            PipeBarrier<PIPE_MTE3>();
        }
        outputQueue_.FreeTensor(local);
    }
    static constexpr uint32_t kCapacity = sizeof(LaneT) == 4 ? 9216 : 15360;
    static constexpr uint32_t kSource = sizeof(LaneT) == 4 ? 9344 : 15616;
    ConcatPipelineHeader d_{};
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inputQueue_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> map_, converted_, gathered_;
    __gm__ LaneT* bases_[S9_CONCAT_MAX_INPUTS];
    uint32_t widths_[S9_CONCAT_MAX_INPUTS];
    uint32_t offsets_[S9_CONCAT_MAX_INPUTS];
    GlobalTensor<LaneT> output_;
};



// AIVs own full row intervals aligned to the 32-byte ownership quantum.
// Within one AIV, MTE3 serialization handles neighboring partial stripes.
template <typename LaneT>
class KernelConcatStripe {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
        const ConcatStripeHeader& data) {
        d_ = data;
        table_ = reinterpret_cast<__gm__ ConcatStripeTilingData*>(tiling);
        ListTensorDesc inputs(reinterpret_cast<__gm__ void*>(x));
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            widths_[i] = table_->inputRowBytes[i] / sizeof(LaneT);
            offsets_[i] = table_->inputOffsets[i];
            bases_[i] = inputs.GetDataPtr<LaneT>(table_->originalInputs[i]);
        }
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ LaneT*>(y));
        if ((d_.cachePolicy & 4U) != 0) output_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(inputQueue_, d_.inputBuffers, d_.inputBytes);
        pipe_.InitBuffer(outputQueue_, 2, d_.outputBytes);
        pipe_.InitBuffer(map_, AlignUp32((d_.mapCount + d_.mapCount / d_.tileRows) * 4));
        pipe_.InitBuffer(compact_, ((d_.mapCount / d_.tileRows + 15) / 16 * 16) * 4);
    }
    __aicore__ inline void Process() {
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * d_.blockRows;
        const uint64_t end = MinU64(begin + d_.blockRows, d_.outer);
        if (begin >= end) return;
        for (uint32_t gi = 0; gi < d_.groupCount; ++gi) {
            g_.prefixBytes = table_->groups[gi].prefixBytes;
            g_.rowBytes = table_->groups[gi].rowBytes;
            g_.paddedRowBytes = table_->groups[gi].paddedRowBytes;
            g_.inputBegin = table_->groups[gi].inputBegin;
            g_.inputCount = table_->groups[gi].inputCount;
            g_.mapBegin = table_->groups[gi].mapBegin;
            g_.mapCount = table_->groups[gi].mapCount;
            const uint32_t firstRows = static_cast<uint32_t>(MinU64(d_.tileRows, end - begin));
            const bool preloaded = ((d_.flags & 2U) != 0) && ((sizeof(LaneT) == 4 && d_.mapFormat == 1) || d_.inputBuffers > 1);
            PrepareMap(begin, firstRows, preloaded);
            if (!preloaded) CopyIn(begin, firstRows);
            for (uint64_t row = begin; row < end; row += d_.tileRows) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(d_.tileRows, end - row));
                const uint64_t next = row + d_.tileRows;
                if (d_.inputBuffers > 1 && next < end) CopyIn(next, static_cast<uint32_t>(MinU64(d_.tileRows, end - next)));
                auto input = inputQueue_.DeQue<LaneT>();
                auto output = outputQueue_.AllocTensor<LaneT>();
                Gather(output, input, map_.Get<uint32_t>(), 0,
                    rows * (g_.paddedRowBytes / sizeof(LaneT)));
                outputQueue_.EnQue(output);
                inputQueue_.FreeTensor(input);
                if (d_.inputBuffers == 1 && next < end) CopyIn(next, static_cast<uint32_t>(MinU64(d_.tileRows, end - next)));
                output = outputQueue_.DeQue<LaneT>();
                if ((d_.flags & 1U) == 0 && gi != 0 && row == begin) { PipeBarrier<PIPE_MTE3>(); }
                DataCopyExtParams write{static_cast<uint16_t>(rows), g_.rowBytes, 0,
                    static_cast<uint32_t>(d_.outputRowBytes - g_.rowBytes), 0};
                DataCopyPad(output_[(row * d_.outputRowBytes + g_.prefixBytes) / sizeof(LaneT)], output, write);
                // Same-group row strips are separated by >=32 bytes (host checked).
                // Queue events protect buffer reuse; only a group transition can
                // share a partial GM block, and PipeBarrier<PIPE_ALL> below orders it.
                outputQueue_.FreeTensor(output);
            }
            if ((d_.flags & 1U) != 0) { PipeBarrier<PIPE_ALL>(); } else { PipeBarrier<PIPE_V>(); }
        }
        PipeBarrier<PIPE_ALL>();
    }
private:
    __aicore__ inline void PrepareMap(uint64_t firstRow, uint32_t firstRows, bool preload) {
        if (sizeof(LaneT) == 4 && d_.mapFormat == 1) {
            const uint32_t rowLanes = g_.paddedRowBytes / sizeof(LaneT);
            const uint32_t stride = (rowLanes + 15) / 16 * 16;
            GlobalTensor<int32_t> src;
            src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table_->indices + g_.mapBegin));
            auto packed = compact_.Get<int32_t>();
            DataCopyExtParams read{1, stride * 4, 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(packed, src, read, pad);
            const event_t ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(ready); WaitFlag<HardEvent::MTE2_V>(ready);
        if (preload) CopyIn(firstRow, firstRows);
            auto map = map_.Get<int32_t>();
            ShiftRight(map[d_.mapCount], packed, static_cast<int32_t>(16), rowLanes);
            PipeBarrier<PIPE_V>();
            ShiftLeft(map, map[d_.mapCount], static_cast<int32_t>(16), rowLanes);
            PipeBarrier<PIPE_V>();
            Sub(map, packed, map, rowLanes);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 1; r < d_.tileRows; ++r) {
                Add(map[r * rowLanes], map[(r - 1) * rowLanes], map[d_.mapCount], rowLanes);
                PipeBarrier<PIPE_V>();
            }
            // No VECIN scratch is borrowed: the next MTE2 input can overlap map work.
            return;
        }

        const uint32_t rowLanes = g_.paddedRowBytes / sizeof(LaneT);
        // Each 16-bit Cast source must independently start on a 32B block.
        const uint32_t compactStride = (rowLanes + 15) / 16 * 16;
        GlobalTensor<int16_t> src; src.SetGlobalBuffer(table_->indices + g_.mapBegin);
        auto compressed = compact_.Get<int16_t>();
        DataCopyExtParams copy{1, compactStride * 4, 0, 0, 0};
        DataCopyPadExtParams<int16_t> pad{false, 0, 0, 0};
        DataCopyPad(compressed, src, copy, pad);
        const event_t ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready); WaitFlag<HardEvent::MTE2_V>(ready);
        if (preload) CopyIn(firstRow, firstRows);
        auto scratch = inputQueue_.AllocTensor<float>();
        auto map = map_.Get<int32_t>();
        Cast(scratch, compressed, RoundMode::CAST_NONE, rowLanes);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, (sizeof(LaneT) == 4 ? 4.0f : 2.0f), rowLanes);
        PipeBarrier<PIPE_V>();
        Cast(map, scratch, RoundMode::CAST_RINT, rowLanes);
        PipeBarrier<PIPE_V>();
        Cast(scratch, compressed[compactStride], RoundMode::CAST_NONE, rowLanes);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, (sizeof(LaneT) == 4 ? 4.0f : 2.0f), rowLanes);
        PipeBarrier<PIPE_V>();
        Cast(map[d_.mapCount], scratch, RoundMode::CAST_RINT, rowLanes);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 1; r < d_.tileRows; ++r) {
            Add(map[r * rowLanes], map[(r - 1) * rowLanes], map[d_.mapCount], rowLanes);
            PipeBarrier<PIPE_V>();
        }
        const event_t consumed = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(consumed); WaitFlag<HardEvent::V_MTE2>(consumed);
        inputQueue_.FreeTensor(scratch);
    }
    __aicore__ inline void CopyIn(uint64_t row, uint32_t rows) {
        auto local = inputQueue_.AllocTensor<LaneT>();
        for (uint32_t i = g_.inputBegin; i < g_.inputBegin + g_.inputCount; ++i) {
            GlobalTensor<LaneT> input; input.SetGlobalBuffer(bases_[i]);
            const uint32_t bytes = rows * widths_[i] * sizeof(LaneT);
            if ((d_.cachePolicy & 1U) != 0) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            else if ((d_.cachePolicy & 2U) == 0) ConfigureConcatStreamRead(input, d_.outer * d_.outputRowBytes, bytes);
            DataCopyExtParams read{1, bytes, 0, 0, 0};
            DataCopyPadExtParams<LaneT> pad{false, 0, 0, 0};
            DataCopyPad(local[offsets_[i]], input[row * widths_[i]], read, pad);
        }
        inputQueue_.EnQue(local);
    }
    ConcatStripeHeader d_{};
    ConcatStripeGroup g_{};
    __gm__ ConcatStripeTilingData* table_;
    __gm__ LaneT* bases_[256];
    uint32_t widths_[256], offsets_[256];
    GlobalTensor<LaneT> output_;
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inputQueue_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> map_, compact_;
};

template <typename LaneT>
class KernelConcatStripeMetadata {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
        const ConcatStripeHeader& data) {
        d_ = data;
        table_ = reinterpret_cast<__gm__ ConcatStripeTilingData*>(tiling);
        inputs_.Init(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ LaneT*>(y));
        if ((d_.cachePolicy & 4U) != 0) output_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(inputQueue_, d_.inputBuffers, d_.inputBytes);
        pipe_.InitBuffer(outputQueue_, 2, d_.outputBytes);
        pipe_.InitBuffer(map_, AlignUp32((d_.mapCount + d_.mapCount / d_.tileRows) * 4));
        pipe_.InitBuffer(compact_, ((d_.mapCount / d_.tileRows + 15) / 16 * 16) * 4);
    }
    __aicore__ inline void Process() {
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * d_.blockRows;
        const uint64_t end = MinU64(begin + d_.blockRows, d_.outer);
        if (begin >= end) return;
        for (uint32_t gi = 0; gi < d_.groupCount; ++gi) {
            g_.prefixBytes = table_->groups[gi].prefixBytes;
            g_.rowBytes = table_->groups[gi].rowBytes;
            g_.paddedRowBytes = table_->groups[gi].paddedRowBytes;
            g_.inputBegin = table_->groups[gi].inputBegin;
            g_.inputCount = table_->groups[gi].inputCount;
            g_.mapBegin = table_->groups[gi].mapBegin;
            g_.mapCount = table_->groups[gi].mapCount;
            const uint32_t firstRows = static_cast<uint32_t>(MinU64(d_.tileRows, end - begin));
            const bool preloaded = ((d_.flags & 2U) != 0) && ((sizeof(LaneT) == 4 && d_.mapFormat == 1) || d_.inputBuffers > 1);
            PrepareMap(begin, firstRows, preloaded);
            if (!preloaded) CopyIn(begin, firstRows);
            for (uint64_t row = begin; row < end; row += d_.tileRows) {
                const uint32_t rows = static_cast<uint32_t>(MinU64(d_.tileRows, end - row));
                const uint64_t next = row + d_.tileRows;
                if (d_.inputBuffers > 1 && next < end) CopyIn(next, static_cast<uint32_t>(MinU64(d_.tileRows, end - next)));
                auto input = inputQueue_.DeQue<LaneT>();
                auto output = outputQueue_.AllocTensor<LaneT>();
                Gather(output, input, map_.Get<uint32_t>(), 0,
                    rows * (g_.paddedRowBytes / sizeof(LaneT)));
                outputQueue_.EnQue(output);
                inputQueue_.FreeTensor(input);
                if (d_.inputBuffers == 1 && next < end) CopyIn(next, static_cast<uint32_t>(MinU64(d_.tileRows, end - next)));
                output = outputQueue_.DeQue<LaneT>();
                if ((d_.flags & 1U) == 0 && gi != 0 && row == begin) { PipeBarrier<PIPE_MTE3>(); }
                DataCopyExtParams write{static_cast<uint16_t>(rows), g_.rowBytes, 0,
                    static_cast<uint32_t>(d_.outputRowBytes - g_.rowBytes), 0};
                DataCopyPad(output_[(row * d_.outputRowBytes + g_.prefixBytes) / sizeof(LaneT)], output, write);
                // Same-group row strips are separated by >=32 bytes (host checked).
                // Queue events protect buffer reuse; only a group transition can
                // share a partial GM block, and PipeBarrier<PIPE_ALL> below orders it.
                outputQueue_.FreeTensor(output);
            }
            if ((d_.flags & 1U) != 0) { PipeBarrier<PIPE_ALL>(); } else { PipeBarrier<PIPE_V>(); }
        }
        PipeBarrier<PIPE_ALL>();
    }
private:
    __aicore__ inline void PrepareMap(uint64_t firstRow, uint32_t firstRows, bool preload) {
        if (sizeof(LaneT) == 4 && d_.mapFormat == 1) {
            const uint32_t rowLanes = g_.paddedRowBytes / sizeof(LaneT);
            const uint32_t stride = (rowLanes + 15) / 16 * 16;
            GlobalTensor<int32_t> src;
            src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table_->indices + g_.mapBegin));
            auto packed = compact_.Get<int32_t>();
            DataCopyExtParams read{1, stride * 4, 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
            DataCopyPad(packed, src, read, pad);
            const event_t ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(ready); WaitFlag<HardEvent::MTE2_V>(ready);
        if (preload) CopyIn(firstRow, firstRows);
            auto map = map_.Get<int32_t>();
            ShiftRight(map[d_.mapCount], packed, static_cast<int32_t>(16), rowLanes);
            PipeBarrier<PIPE_V>();
            ShiftLeft(map, map[d_.mapCount], static_cast<int32_t>(16), rowLanes);
            PipeBarrier<PIPE_V>();
            Sub(map, packed, map, rowLanes);
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 1; r < d_.tileRows; ++r) {
                Add(map[r * rowLanes], map[(r - 1) * rowLanes], map[d_.mapCount], rowLanes);
                PipeBarrier<PIPE_V>();
            }
            // No VECIN scratch is borrowed: the next MTE2 input can overlap map work.
            return;
        }

        const uint32_t rowLanes = g_.paddedRowBytes / sizeof(LaneT);
        // Each 16-bit Cast source must independently start on a 32B block.
        const uint32_t compactStride = (rowLanes + 15) / 16 * 16;
        GlobalTensor<int16_t> src; src.SetGlobalBuffer(table_->indices + g_.mapBegin);
        auto compressed = compact_.Get<int16_t>();
        DataCopyExtParams copy{1, compactStride * 4, 0, 0, 0};
        DataCopyPadExtParams<int16_t> pad{false, 0, 0, 0};
        DataCopyPad(compressed, src, copy, pad);
        const event_t ready = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready); WaitFlag<HardEvent::MTE2_V>(ready);
        if (preload) CopyIn(firstRow, firstRows);
        auto scratch = inputQueue_.AllocTensor<float>();
        auto map = map_.Get<int32_t>();
        Cast(scratch, compressed, RoundMode::CAST_NONE, rowLanes);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, (sizeof(LaneT) == 4 ? 4.0f : 2.0f), rowLanes);
        PipeBarrier<PIPE_V>();
        Cast(map, scratch, RoundMode::CAST_RINT, rowLanes);
        PipeBarrier<PIPE_V>();
        Cast(scratch, compressed[compactStride], RoundMode::CAST_NONE, rowLanes);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, (sizeof(LaneT) == 4 ? 4.0f : 2.0f), rowLanes);
        PipeBarrier<PIPE_V>();
        Cast(map[d_.mapCount], scratch, RoundMode::CAST_RINT, rowLanes);
        PipeBarrier<PIPE_V>();
        for (uint32_t r = 1; r < d_.tileRows; ++r) {
            Add(map[r * rowLanes], map[(r - 1) * rowLanes], map[d_.mapCount], rowLanes);
            PipeBarrier<PIPE_V>();
        }
        const event_t consumed = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(consumed); WaitFlag<HardEvent::V_MTE2>(consumed);
        inputQueue_.FreeTensor(scratch);
    }
    __aicore__ inline void CopyIn(uint64_t row, uint32_t rows) {
        auto local = inputQueue_.AllocTensor<LaneT>();
        for (uint32_t i = g_.inputBegin; i < g_.inputBegin + g_.inputCount; ++i) {
            const uint32_t width = table_->inputRowBytes[i] / sizeof(LaneT);
            const uint32_t offset = table_->inputOffsets[i];
            GlobalTensor<LaneT> input; input.SetGlobalBuffer(inputs_.GetDataPtr<LaneT>(table_->originalInputs[i]));
            const uint32_t bytes = rows * width * sizeof(LaneT);
            if ((d_.cachePolicy & 1U) != 0) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            else if ((d_.cachePolicy & 2U) == 0) ConfigureConcatStreamRead(input, d_.outer * d_.outputRowBytes, bytes);
            DataCopyExtParams read{1, bytes, 0, 0, 0};
            DataCopyPadExtParams<LaneT> pad{false, 0, 0, 0};
            DataCopyPad(local[offset], input[row * width], read, pad);
        }
        inputQueue_.EnQue(local);
    }
    ConcatStripeHeader d_{};
    ConcatStripeGroup g_{};
    __gm__ ConcatStripeTilingData* table_;
    ListTensorDesc inputs_;
    GlobalTensor<LaneT> output_;
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inputQueue_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> map_, compact_;
};


class KernelConcatSubmitFirst {
    using LaneT = uint32_t;
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, const ConcatStripeHeader& data) {
        d_ = data; inputRows_ = d_.flags >> 8;
        table_ = reinterpret_cast<__gm__ ConcatStripeTilingData*>(tiling);
        inputs_.Init(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ LaneT*>(y));
        if(d_.cachePolicy&4U) output_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(inputQueue_,d_.inputBuffers,d_.inputBytes);
        pipe_.InitBuffer(outputQueue_,2,d_.outputBytes);
        pipe_.InitBuffer(map_,AlignUp32((d_.mapCount+2*d_.mapCount/d_.tileRows)*4));
        pipe_.InitBuffer(compact_,AlignUp32((d_.mapCount/d_.tileRows+15)/16*16*4));
    }
    __aicore__ inline void Process() {
        const uint64_t begin=static_cast<uint64_t>(GetBlockIdx())*d_.blockRows;
        const uint64_t end=MinU64(begin+d_.blockRows,d_.outer); if(begin>=end)return;
        for(uint32_t gi=0;gi<d_.groupCount;++gi) {
            g_.prefixBytes=table_->groups[gi].prefixBytes; g_.rowBytes=table_->groups[gi].rowBytes;
            g_.paddedRowBytes=table_->groups[gi].paddedRowBytes;g_.inputBegin=table_->groups[gi].inputBegin;
            g_.inputCount=table_->groups[gi].inputCount;g_.mapBegin=table_->groups[gi].mapBegin;
            g_.mapCount=table_->groups[gi].mapCount;
            PrepareMap(begin,static_cast<uint32_t>(MinU64(inputRows_,end-begin)));
            for(uint64_t row=begin;row<end;row+=inputRows_) {
                const uint32_t rows=static_cast<uint32_t>(MinU64(inputRows_,end-row));
                const uint64_t next=row+inputRows_;
                if(d_.inputBuffers>1 && next<end)CopyIn(next,static_cast<uint32_t>(MinU64(inputRows_,end-next)));
                auto input=inputQueue_.DeQue<LaneT>();
                for(uint32_t sub=0;sub<rows;sub+=d_.tileRows) {
                    const uint32_t count=static_cast<uint32_t>(MinU64(d_.tileRows,rows-sub));
                    auto output=outputQueue_.AllocTensor<LaneT>();
                    Gather(output,input,map_.Get<uint32_t>(),0,count*(g_.paddedRowBytes/4));
                    outputQueue_.EnQue(output);
                    if((d_.flags&2U)==0 && sub+d_.tileRows<rows)AdvanceMap(false);
                    output=outputQueue_.DeQue<LaneT>();
                    if(gi!=0 && row==begin && sub==0)PipeBarrier<PIPE_MTE3>();
                    DataCopyExtParams write{static_cast<uint16_t>(count),g_.rowBytes,0,static_cast<uint32_t>(d_.outputRowBytes-g_.rowBytes),0};
                    DataCopyPad(output_[((row+sub)*d_.outputRowBytes+g_.prefixBytes)/4],output,write);
                    outputQueue_.FreeTensor(output);
                    if((d_.flags&2U)!=0 && sub+d_.tileRows<rows)AdvanceMap(false);
                }
                inputQueue_.FreeTensor(input);
                if(next<end) {
                    if(d_.inputBuffers==1)CopyIn(next,static_cast<uint32_t>(MinU64(inputRows_,end-next)));
                    AdvanceMap(true);
                }
            }
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
    }
private:
    __aicore__ inline void PrepareMap(uint64_t row,uint32_t rows) {
        const uint32_t lanes=g_.paddedRowBytes/4, stride=(lanes+15)/16*16;
        GlobalTensor<int32_t> src;src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table_->indices+g_.mapBegin));
        auto packed=compact_.Get<int32_t>();DataCopyExtParams read{1,stride*4,0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
        DataCopyPad(packed,src,read,pad);
        const event_t ready=static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready);WaitFlag<HardEvent::MTE2_V>(ready);
        CopyIn(row,rows);
        auto map=map_.Get<int32_t>();auto step=map[d_.mapCount];auto reset=map[d_.mapCount+lanes];
        ShiftRight(step,packed,static_cast<int32_t>(16),lanes);PipeBarrier<PIPE_V>();
        ShiftLeft(map,step,static_cast<int32_t>(16),lanes);PipeBarrier<PIPE_V>();
        Sub(map,packed,map,lanes);PipeBarrier<PIPE_V>();
        ShiftLeft(map,map,static_cast<int32_t>(2),lanes);
        ShiftLeft(step,step,static_cast<int32_t>(2),lanes);PipeBarrier<PIPE_V>();
        for(uint32_t r=1;r<d_.tileRows;++r){Add(map[r*lanes],map[(r-1)*lanes],step,lanes);PipeBarrier<PIPE_V>();}
        uint32_t inputShift=0,outputShift=0;while((1U<<inputShift)<inputRows_)++inputShift;while((1U<<outputShift)<d_.tileRows)++outputShift;
        ShiftLeft(reset,step,static_cast<int32_t>(inputShift),lanes);PipeBarrier<PIPE_V>();
        ShiftLeft(step,step,static_cast<int32_t>(outputShift),lanes);PipeBarrier<PIPE_V>();
        Sub(reset,reset,step,lanes);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void AdvanceMap(bool reset) {
        const uint32_t lanes=g_.paddedRowBytes/4;auto map=map_.Get<int32_t>();PipeBarrier<PIPE_V>();
        for(uint32_t r=0;r<d_.tileRows;++r){
            if(reset)Sub(map[r*lanes],map[r*lanes],map[d_.mapCount+lanes],lanes);
            else Add(map[r*lanes],map[r*lanes],map[d_.mapCount],lanes);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CopyIn(uint64_t row,uint32_t rows) {
        auto local=inputQueue_.AllocTensor<LaneT>();
        for(uint32_t i=g_.inputBegin;i<g_.inputBegin+g_.inputCount;++i){
            const uint32_t width=table_->inputRowBytes[i]/4;
            const uint32_t offset=table_->inputOffsets[i];
            GlobalTensor<LaneT> input;input.SetGlobalBuffer(inputs_.GetDataPtr<LaneT>(table_->originalInputs[i]));uint32_t bytes=rows*width*4;
            if(d_.cachePolicy&1U)input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            else if((d_.cachePolicy&2U)==0)ConfigureConcatStreamRead(input,d_.outer*d_.outputRowBytes,bytes);
            DataCopyExtParams read{1,bytes,0,0,0};DataCopyPadExtParams<LaneT> pad{false,0,0,0};
            DataCopyPad(local[offset],input[row*width],read,pad);
        }
        inputQueue_.EnQue(local);
    }
    ConcatStripeHeader d_{};ConcatStripeGroup g_{};uint32_t inputRows_;
    __gm__ ConcatStripeTilingData* table_;ListTensorDesc inputs_;
    GlobalTensor<LaneT> output_;TPipe pipe_;TQue<TPosition::VECIN,2> inputQueue_;TQue<TPosition::VECOUT,2> outputQueue_;TBuf<TPosition::VECCALC> map_,compact_;
};


class KernelConcatStripeBatch {
    using LaneT = uint32_t;
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, const ConcatStripeHeader& data) {
        d_ = data; inputRows_ = d_.flags >> 8;
        table_ = reinterpret_cast<__gm__ ConcatStripeTilingData*>(tiling);
        inputs_.Init(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ LaneT*>(y));
        if(d_.cachePolicy&4U) output_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(inputQueue_,d_.inputBuffers,d_.inputBytes);
        pipe_.InitBuffer(outputQueue_,2,d_.outputBytes);
        pipe_.InitBuffer(map_,AlignUp32((d_.mapCount+2*d_.mapCount/d_.tileRows)*4));
        pipe_.InitBuffer(compact_,AlignUp32((d_.mapCount/d_.tileRows+15)/16*16*4));
    }
    __aicore__ inline void Process() {
        const uint64_t begin=static_cast<uint64_t>(GetBlockIdx())*d_.blockRows;
        const uint64_t end=MinU64(begin+d_.blockRows,d_.outer); if(begin>=end)return;
        for(uint32_t gi=0;gi<d_.groupCount;++gi) {
            g_.prefixBytes=table_->groups[gi].prefixBytes; g_.rowBytes=table_->groups[gi].rowBytes;
            g_.paddedRowBytes=table_->groups[gi].paddedRowBytes;g_.inputBegin=table_->groups[gi].inputBegin;
            g_.inputCount=table_->groups[gi].inputCount;g_.mapBegin=table_->groups[gi].mapBegin;
            g_.mapCount=table_->groups[gi].mapCount;
            PrepareMap(begin,static_cast<uint32_t>(MinU64(inputRows_,end-begin)));
            for(uint64_t row=begin;row<end;row+=inputRows_) {
                const uint32_t rows=static_cast<uint32_t>(MinU64(inputRows_,end-row));
                const uint64_t next=row+inputRows_;
                if(d_.inputBuffers>1 && next<end)CopyIn(next,static_cast<uint32_t>(MinU64(inputRows_,end-next)));
                auto input=inputQueue_.DeQue<LaneT>();
                for(uint32_t sub=0;sub<rows;sub+=d_.tileRows) {
                    const uint32_t count=static_cast<uint32_t>(MinU64(d_.tileRows,rows-sub));
                    auto output=outputQueue_.AllocTensor<LaneT>();
                    Gather(output,input,map_.Get<uint32_t>(),0,count*(g_.paddedRowBytes/4));
                    outputQueue_.EnQue(output);
                    if(sub+d_.tileRows<rows)AdvanceMap(false);
                    output=outputQueue_.DeQue<LaneT>();
                    if(gi!=0 && row==begin && sub==0)PipeBarrier<PIPE_MTE3>();
                    DataCopyExtParams write{static_cast<uint16_t>(count),g_.rowBytes,0,static_cast<uint32_t>(d_.outputRowBytes-g_.rowBytes),0};
                    DataCopyPad(output_[((row+sub)*d_.outputRowBytes+g_.prefixBytes)/4],output,write);
                    outputQueue_.FreeTensor(output);
                }
                inputQueue_.FreeTensor(input);
                if(next<end) {
                    if(d_.inputBuffers==1)CopyIn(next,static_cast<uint32_t>(MinU64(inputRows_,end-next)));
                    AdvanceMap(true);
                }
            }
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
    }
private:
    __aicore__ inline void PrepareMap(uint64_t row,uint32_t rows) {
        const uint32_t lanes=g_.paddedRowBytes/4, stride=(lanes+15)/16*16;
        GlobalTensor<int32_t> src;src.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table_->indices+g_.mapBegin));
        auto packed=compact_.Get<int32_t>();DataCopyExtParams read{1,stride*4,0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
        DataCopyPad(packed,src,read,pad);
        const event_t ready=static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ready);WaitFlag<HardEvent::MTE2_V>(ready);
        CopyIn(row,rows);
        auto map=map_.Get<int32_t>();auto step=map[d_.mapCount];auto reset=map[d_.mapCount+lanes];
        ShiftRight(step,packed,static_cast<int32_t>(16),lanes);PipeBarrier<PIPE_V>();
        ShiftLeft(map,step,static_cast<int32_t>(16),lanes);PipeBarrier<PIPE_V>();
        Sub(map,packed,map,lanes);PipeBarrier<PIPE_V>();
        ShiftLeft(map,map,static_cast<int32_t>(2),lanes);
        ShiftLeft(step,step,static_cast<int32_t>(2),lanes);PipeBarrier<PIPE_V>();
        for(uint32_t r=1;r<d_.tileRows;++r){Add(map[r*lanes],map[(r-1)*lanes],step,lanes);PipeBarrier<PIPE_V>();}
        uint32_t inputShift=0,outputShift=0;while((1U<<inputShift)<inputRows_)++inputShift;while((1U<<outputShift)<d_.tileRows)++outputShift;
        ShiftLeft(reset,step,static_cast<int32_t>(inputShift),lanes);PipeBarrier<PIPE_V>();
        ShiftLeft(step,step,static_cast<int32_t>(outputShift),lanes);PipeBarrier<PIPE_V>();
        Sub(reset,reset,step,lanes);PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void AdvanceMap(bool reset) {
        const uint32_t lanes=g_.paddedRowBytes/4;auto map=map_.Get<int32_t>();PipeBarrier<PIPE_V>();
        for(uint32_t r=0;r<d_.tileRows;++r){
            if(reset)Sub(map[r*lanes],map[r*lanes],map[d_.mapCount+lanes],lanes);
            else Add(map[r*lanes],map[r*lanes],map[d_.mapCount],lanes);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CopyIn(uint64_t row,uint32_t rows) {
        auto local=inputQueue_.AllocTensor<LaneT>();
        for(uint32_t i=g_.inputBegin;i<g_.inputBegin+g_.inputCount;++i){
            const uint32_t width=table_->inputRowBytes[i]/4;
            const uint32_t offset=table_->inputOffsets[i];
            GlobalTensor<LaneT> input;input.SetGlobalBuffer(inputs_.GetDataPtr<LaneT>(table_->originalInputs[i]));uint32_t bytes=rows*width*4;
            if(d_.cachePolicy&1U)input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            else if((d_.cachePolicy&2U)==0)ConfigureConcatStreamRead(input,d_.outer*d_.outputRowBytes,bytes);
            DataCopyExtParams read{1,bytes,0,0,0};DataCopyPadExtParams<LaneT> pad{false,0,0,0};
            DataCopyPad(local[offset],input[row*width],read,pad);
        }
        inputQueue_.EnQue(local);
    }
    ConcatStripeHeader d_{};ConcatStripeGroup g_{};uint32_t inputRows_;
    __gm__ ConcatStripeTilingData* table_;ListTensorDesc inputs_;
    GlobalTensor<LaneT> output_;TPipe pipe_;TQue<TPosition::VECIN,2> inputQueue_;TQue<TPosition::VECOUT,2> outputQueue_;TBuf<TPosition::VECCALC> map_,compact_;
};


class KernelConcatCompactDirect {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
        const ConcatCompactDirectHeader& data) {
        d_ = data;
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        table_ = reinterpret_cast<__gm__ ConcatCompactDirectTilingData*>(tiling);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        pipe_.InitBuffer(queue_, 2, d_.tileBytes);
    }
    __aicore__ inline void Process() {
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * d_.blockRows;
        const uint64_t end = MinU64(begin + d_.blockRows, d_.outer);
        if (begin >= end) { return; }
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            const uint32_t width = table_->inputRowBytes[i];
            const uint32_t prefix = table_->prefixBytes[i];
            GlobalTensor<uint8_t> input;
            input.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(table_->originalInputs[i]));
            ConfigureConcatStreamRead(input, d_.outer * d_.outputRowBytes, width);
            if (d_.inputCount == 1) {
                CopyLinear(input, begin * width, begin * d_.outputRowBytes, (end - begin) * width);
                continue;
            }
            const uint64_t padded = AlignUp32(width);
            if (padded > d_.tileBytes) {
                for (uint64_t row = begin; row < end; ++row) {
                    CopyLinear(input, row * width, row * d_.outputRowBytes + prefix, width);
                }
                continue;
            }
            const uint64_t maxRows = MinU64(4095, d_.tileBytes / padded);
            // The 2-D descriptor reads contiguous GM rows; use its complete span.
            ConfigureConcatStreamRead(input, d_.outer * d_.outputRowBytes, MinU64(maxRows, end - begin) * width);
            if (d_.outputRowBytes - width >= 32) {
                bool pending = false; uint64_t pendingRow = 0; uint16_t pendingRows = 0;
                for (uint64_t row = begin; row < end; row += maxRows) {
                    const uint16_t rows = static_cast<uint16_t>(MinU64(maxRows, end - row));
                    auto local = queue_.AllocTensor<uint8_t>();
                    DataCopyExtParams read{rows, width, 0, 0, 0};
                    DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                    DataCopyPad(local, input[row * width], read, pad); queue_.EnQue(local);
                    if (pending) {
                        local = queue_.DeQue<uint8_t>();
                        DataCopyExtParams write{pendingRows, width, 0, static_cast<uint32_t>(d_.outputRowBytes - width), 0};
                        DataCopyPad(output_[pendingRow * d_.outputRowBytes + prefix], local, write);
                        queue_.FreeTensor(local);
                    }
                    pending = true; pendingRow = row; pendingRows = rows;
                }
                if (pending) {
                    auto local = queue_.DeQue<uint8_t>();
                    DataCopyExtParams write{pendingRows, width, 0, static_cast<uint32_t>(d_.outputRowBytes - width), 0};
                    DataCopyPad(output_[pendingRow * d_.outputRowBytes + prefix], local, write);
                    queue_.FreeTensor(local);
                }
                // Different input segments may share a partial destination block.
                PipeBarrier<PIPE_MTE3>();
                continue;
            }
            for (uint64_t row = begin; row < end; row += maxRows) {
                const uint16_t rows = static_cast<uint16_t>(MinU64(maxRows, end - row));
                auto local = queue_.AllocTensor<uint8_t>();
                DataCopyExtParams read{rows, width, 0, 0, 0};
                DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                DataCopyPad(local, input[row * width], read, pad);
                queue_.EnQue(local);
                local = queue_.DeQue<uint8_t>();
                DataCopyExtParams write{rows, width, 0, static_cast<uint32_t>(d_.outputRowBytes - width), 0};
                DataCopyPad(output_[row * d_.outputRowBytes + prefix], local, write);
                if (((width | prefix | d_.outputRowBytes) & 31) != 0) { PipeBarrier<PIPE_MTE3>(); }
                queue_.FreeTensor(local);
            }
        }
    }
private:
    __aicore__ inline void CopyLinear(const GlobalTensor<uint8_t>& input,
        uint64_t source, uint64_t destination, uint64_t bytes) {
        while (bytes != 0) {
            const uint32_t once = static_cast<uint32_t>(MinU64(bytes, d_.tileBytes));
            auto local = queue_.AllocTensor<uint8_t>();
            DataCopyExtParams copy{1, once, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyPad(local, input[source], copy, pad);
            queue_.EnQue(local);
            local = queue_.DeQue<uint8_t>();
            DataCopyPad(output_[destination], local, copy);
            if (((destination | once) & 31) != 0) { PipeBarrier<PIPE_MTE3>(); }
            queue_.FreeTensor(local);
            source += once; destination += once; bytes -= once;
        }
    }
    ConcatCompactDirectHeader d_{};
    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 2> queue_;
    ListTensorDesc inputs_;
    __gm__ ConcatCompactDirectTilingData* table_;
    GlobalTensor<uint8_t> output_;
};

template <typename LaneT, typename GatherT, typename MaskT, bool kConvert>
class KernelConcatMaskedRows {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
        const ConcatMaskedRowsHeader& data) {
        d_ = data;
        ListTensorDesc inputs(reinterpret_cast<__gm__ void*>(x));
        auto* p = reinterpret_cast<__gm__ ConcatMaskedRowsTilingData*>(tiling);
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            widths_[i] = p->inputRowBytes[i];
            offsets_[i] = p->paddedOffsets[i] / sizeof(LaneT);
            bases_[i] = inputs.GetDataPtr<LaneT>(p->originalInputs[i]);
        }
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ LaneT*>(y));
        pipe_.InitBuffer(inputQueue_, 2, d_.inputBytes);
        pipe_.InitBuffer(outputQueue_, 2, d_.outputBytes);
        pipe_.InitBuffer(pattern_, d_.maskBytes);
        if constexpr (kConvert) {
            pipe_.InitBuffer(converted_, 2 * d_.inputBytes);
            pipe_.InitBuffer(gathered_, 2 * d_.outputBytes);
        }
        GlobalTensor<MaskT> pattern;
        pattern.SetGlobalBuffer(reinterpret_cast<__gm__ MaskT*>(p->maskBits));
        DataCopyExtParams copy{1, d_.maskBytes, 0, 0, 0};
        DataCopyPadExtParams<MaskT> pad{false, 0, 0, 0};
        DataCopyPad(pattern_.Get<MaskT>(), pattern, copy, pad);
        const event_t ev = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(ev); WaitFlag<HardEvent::MTE2_V>(ev);
    }
    __aicore__ inline void Process() {
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * d_.blockRows;
        const uint64_t end = MinU64(begin + d_.blockRows, d_.outer);
        if (begin >= end) { return; }
        CopyIn(begin, static_cast<uint32_t>(MinU64(d_.tileRows, end - begin)));
        for (uint64_t row = begin; row < end; row += d_.tileRows) {
            const uint32_t rows = static_cast<uint32_t>(MinU64(d_.tileRows, end - row));
            const uint64_t next = row + d_.tileRows;
            if (next < end) { CopyIn(next, static_cast<uint32_t>(MinU64(d_.tileRows, end - next))); }
            Compute(rows);
            CopyOut(row, rows);
        }
    }
private:
    __aicore__ inline void CopyIn(uint64_t row, uint32_t rows) {
        auto local = inputQueue_.AllocTensor<LaneT>();
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            const uint32_t width = widths_[i];
            GlobalTensor<LaneT> input; input.SetGlobalBuffer(bases_[i]);
            ConfigureConcatStreamRead(input, d_.outer * d_.outputRowBytes, width);
            DataCopyExtParams copy{static_cast<uint16_t>(rows), width, 0,
                static_cast<uint32_t>((d_.paddedRowBytes - AlignUp32(width)) / 32), 0};
            DataCopyPadExtParams<LaneT> pad{false, 0, 0, 0};
            DataCopyPad(local[offsets_[i]], input[row * (width / sizeof(LaneT))], copy, pad);
        }
        inputQueue_.EnQue(local);
    }
    __aicore__ inline void Compute(uint32_t rows) {
        auto input = inputQueue_.DeQue<LaneT>();
        auto output = outputQueue_.AllocTensor<LaneT>();
        auto values = input.template ReinterpretCast<GatherT>();
        auto result = output.template ReinterpretCast<GatherT>();
        if constexpr (kConvert) {
            values = converted_.Get<GatherT>();
            result = gathered_.Get<GatherT>();
            Cast(values, input, RoundMode::CAST_NONE, d_.inputBytes);
            PipeBarrier<PIPE_V>();
        }
        auto mask = pattern_.Get<MaskT>();
        uint64_t kept = 0;
        GatherMask(result, values, mask, true, d_.paddedRowBytes / sizeof(LaneT),
            {1, static_cast<uint16_t>(rows),
             static_cast<uint16_t>(d_.paddedRowBytes * sizeof(GatherT) / sizeof(LaneT) / 32), 0}, kept);
        if constexpr (kConvert) {
            PipeBarrier<PIPE_V>();
            const uint32_t outCount = static_cast<uint32_t>(
                (rows * d_.outputRowBytes + 127) / 128 * 128);
            Cast(output, result, RoundMode::CAST_NONE, outCount);
        }
        outputQueue_.EnQue(output);
        inputQueue_.FreeTensor(input);
    }
    __aicore__ inline void CopyOut(uint64_t row, uint32_t rows) {
        auto local = outputQueue_.DeQue<LaneT>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(rows * d_.outputRowBytes), 0, 0, 0};
        DataCopyPad(output_[row * (d_.outputRowBytes / sizeof(LaneT))], local, copy);
        if ((d_.tileRows * d_.outputRowBytes) % 32 != 0) { PipeBarrier<PIPE_MTE3>(); }
        outputQueue_.FreeTensor(local);
    }
    ConcatMaskedRowsHeader d_{};
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inputQueue_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> pattern_, converted_, gathered_;
    __gm__ LaneT* bases_[S9_CONCAT_MAX_INPUTS];
    uint32_t widths_[S9_CONCAT_MAX_INPUTS], offsets_[S9_CONCAT_MAX_INPUTS];
    GlobalTensor<LaneT> output_;
};

class KernelConcatAlignedSlab {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
        const ConcatAlignedSlabHeader& data) {
        d_ = data;
        auto* p = reinterpret_cast<__gm__ ConcatAlignedSlabTilingData*>(tiling);
        ListTensorDesc inputs(reinterpret_cast<__gm__ void*>(x));
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            widths_[i] = p->inputRowBytes[i];
            prefixes_[i] = p->outputPrefixes[i];
            offsets_[i] = p->inputOffsets[i];
            bases_[i] = inputs.GetDataPtr<uint8_t>(p->originalInputs[i]);
        }
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        pipe_.InitBuffer(inputQueue_, 2, d_.tileBytes);
        pipe_.InitBuffer(outputQueue_, 2, d_.tileBytes);
    }
    __aicore__ inline void Process() {
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * d_.blockRows;
        const uint64_t end = MinU64(begin + d_.blockRows, d_.outer);
        if (begin >= end) { return; }
        CopyIn(begin, static_cast<uint32_t>(MinU64(d_.tileRows, end - begin)));
        for (uint64_t row = begin; row < end; row += d_.tileRows) {
            const uint32_t rows = static_cast<uint32_t>(MinU64(d_.tileRows, end - row));
            const uint64_t next = row + d_.tileRows;
            if (next < end) { CopyIn(next, static_cast<uint32_t>(MinU64(d_.tileRows, end - next))); }
            Compute(rows);
            CopyOut(row, rows);
        }
    }
private:
    __aicore__ inline void CopyIn(uint64_t row, uint32_t rows) {
        auto input = inputQueue_.AllocTensor<uint8_t>();
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            GlobalTensor<uint8_t> source; source.SetGlobalBuffer(bases_[i]);
            const uint32_t bytes = rows * widths_[i];
            ConfigureConcatStreamRead(source, d_.outer * d_.outputRowBytes, bytes);
            DataCopyExtParams copy{1, bytes, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyPad(input[offsets_[i]], source[row * widths_[i]], copy, pad);
        }
        inputQueue_.EnQue(input);
    }
    __aicore__ inline void Compute(uint32_t rows) {
        auto input = inputQueue_.DeQue<uint8_t>();
        auto output = outputQueue_.AllocTensor<uint8_t>();
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            DataCopyParams copy{static_cast<uint16_t>(rows),
                static_cast<uint16_t>(widths_[i] / 32), 0,
                static_cast<uint16_t>((d_.outputRowBytes - widths_[i]) / 32)};
            DataCopy(output[prefixes_[i]], input[offsets_[i]], copy);
        }
        outputQueue_.EnQue(output);
        inputQueue_.FreeTensor(input);
    }
    __aicore__ inline void CopyOut(uint64_t row, uint32_t rows) {
        auto output = outputQueue_.DeQue<uint8_t>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(rows * d_.outputRowBytes), 0, 0, 0};
        DataCopyPad(output_[row * d_.outputRowBytes], output, copy);
        outputQueue_.FreeTensor(output);
    }
    ConcatAlignedSlabHeader d_{};
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inputQueue_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
    __gm__ uint8_t* bases_[32];
    uint32_t widths_[32], prefixes_[32], offsets_[32];
    GlobalTensor<uint8_t> output_;
};

class KernelConcatOutputChunks {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
        const ConcatOutputChunksHeader& data) {
        d_ = data;
        auto* p = reinterpret_cast<__gm__ ConcatOutputChunksTilingData*>(tiling);
        ListTensorDesc inputs(reinterpret_cast<__gm__ void*>(x));
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            widths_[i] = p->inputRowBytes[i]; prefixes_[i] = p->prefixes[i];
            bases_[i] = inputs.GetDataPtr<uint8_t>(p->originalInputs[i]);
        }
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        pipe_.InitBuffer(queue_, 2, d_.tileBytes);
    }
    __aicore__ inline void Process() {
        uint64_t pos = static_cast<uint64_t>(GetBlockIdx()) * d_.chunkBytes;
        const uint64_t end = MinU64(pos + d_.chunkBytes, d_.totalBytes);
        if (pos >= end) { return; }
        uint64_t row = pos / d_.outputRowBytes;
        const uint32_t col = static_cast<uint32_t>(pos % d_.outputRowBytes);
        uint32_t lo = 0, hi = d_.inputCount;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) / 2;
            if (prefixes_[mid] <= col) { lo = mid; } else { hi = mid; }
        }
        uint32_t input = lo;
        uint32_t inRow = col - prefixes_[input];
        while (pos < end) {
            const uint64_t amount = MinU64(widths_[input] - inRow, end - pos);
            Move(input, row * widths_[input] + inRow, pos, amount);
            pos += amount;
            inRow += static_cast<uint32_t>(amount);
            if (inRow == widths_[input]) {
                inRow = 0; ++input;
                if (input == d_.inputCount) { input = 0; ++row; }
            }
        }
    }
private:
    __aicore__ inline void Move(uint32_t input, uint64_t src, uint64_t dst, uint64_t bytes) {
        while (bytes != 0) {
            const uint32_t n = static_cast<uint32_t>(MinU64(bytes, d_.tileBytes));
            GlobalTensor<uint8_t> source; source.SetGlobalBuffer(bases_[input]);
            ConfigureConcatStreamRead(source, d_.totalBytes, n);
            auto local = queue_.AllocTensor<uint8_t>();
            DataCopyExtParams copy{1, n, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyPad(local, source[src], copy, pad);
            queue_.EnQue(local); local = queue_.DeQue<uint8_t>();
            DataCopyPad(output_[dst], local, copy);
            // Consecutive fragments inside one core can share a partial
            // 32-byte block; finish that store before the next fragment.
            if (((dst | n) & 31) != 0) { PipeBarrier<PIPE_MTE3>(); }
            queue_.FreeTensor(local);
            src += n; dst += n; bytes -= n;
        }
    }
    ConcatOutputChunksHeader d_{};
    TPipe pipe_;
    TQueBind<TPosition::VECIN,TPosition::VECOUT,2> queue_;
    __gm__ uint8_t* bases_[32];
    uint32_t widths_[32], prefixes_[32];
    GlobalTensor<uint8_t> output_;
};


class KernelConcatBytePair {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling,
        const ConcatBytePairHeader& data) {
        d_ = data;
        table_ = reinterpret_cast<__gm__ ConcatBytePairTilingData*>(tiling);
        ListTensorDesc inputs(reinterpret_cast<__gm__ void*>(x));
        uint64_t prefix = 0;
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            widths_[i] = table_->inputs[i].width;
            offsets0_[i] = table_->inputs[i].offset0;
            offsets1_[i] = table_->inputs[i].offset1;
            shiftedOnly_[i] = prefix & 1;
            dual_[i] = (widths_[i] ^ d_.outputRowBytes) & 1;
            prefix += widths_[i];
            bases_[i] = inputs.GetDataPtr<uint8_t>(table_->inputs[i].original);
        }
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        pipe_.InitBuffer(inputQueue_, 2, d_.sourceBytes);
        pipe_.InitBuffer(outputQueue_, 2, d_.outputBytes);
        pipe_.InitBuffer(map_, d_.mapCount * 4);
        if (d_.patchCount != 0) {
            pipe_.InitBuffer(lowMap_, d_.patchCount * 64);
            pipe_.InitBuffer(highMap_, d_.patchCount * 64);
            pipe_.InitBuffer(low_, d_.patchCount * 32);
            pipe_.InitBuffer(high_, d_.patchCount * 32);
            GlobalTensor<uint32_t> lm, hm;
            lm.SetGlobalBuffer(table_->lowIndices); hm.SetGlobalBuffer(table_->highIndices);
            DataCopyExtParams cp{1, static_cast<uint32_t>(d_.patchCount * 64), 0, 0, 0};
            DataCopyPadExtParams<uint32_t> pp{false, 0, 0, 0};
            DataCopyPad(lowMap_.Get<uint32_t>(), lm, cp, pp);
            DataCopyPad(highMap_.Get<uint32_t>(), hm, cp, pp);
        }
        GlobalTensor<uint32_t> indices; indices.SetGlobalBuffer(table_->indices);
        DataCopyExtParams copy{1, static_cast<uint32_t>(d_.mapCount * 4), 0, 0, 0};
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        DataCopyPad(map_.Get<uint32_t>(), indices, copy, pad);
        Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline void Process() {
        const uint64_t begin = GetBlockIdx() * d_.blockRows;
        const uint64_t end = MinU64(begin + d_.blockRows, d_.outer);
        if (begin >= end) { return; }
        CopyIn(begin, MinU64(d_.tileRows, end - begin));
        for (uint64_t row = begin; row < end; row += d_.tileRows) {
            const uint64_t rows = MinU64(d_.tileRows, end - row);
            if (row + d_.tileRows < end) { CopyIn(row + d_.tileRows, MinU64(d_.tileRows, end - row - d_.tileRows)); }
            Compute(rows);
            CopyOut(row, rows);
        }
    }
private:
    __aicore__ inline void CopyIn(uint64_t row, uint64_t rows) {
        auto local = inputQueue_.AllocTensor<uint8_t>();
        for (uint32_t i = 0; i < d_.inputCount; ++i) {
            GlobalTensor<uint8_t> source; source.SetGlobalBuffer(bases_[i]);
            // The second phase immediately reuses the first phase's cache lines.
            const uint32_t bytes = static_cast<uint32_t>(rows * widths_[i]);
            DataCopyExtParams copy{1, bytes, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> normal{false, 0, 0, 0};
            if (dual_[i] || !shiftedOnly_[i]) { DataCopyPad(local[offsets0_[i]], source[row * widths_[i]], copy, normal); }
            DataCopyPadExtParams<uint8_t> shifted{true, 1, static_cast<uint8_t>((32 - ((bytes + 1) % 32)) % 32), 0};
            if (dual_[i] || shiftedOnly_[i]) { DataCopyPad(local[offsets1_[i]], source[row * widths_[i]], copy, shifted); }
        }
        inputQueue_.EnQue(local);
    }
    __aicore__ inline void Compute(uint64_t rows) {
        auto input = inputQueue_.DeQue<uint8_t>();
        auto output = outputQueue_.AllocTensor<uint8_t>();
        auto source = input.ReinterpretCast<uint16_t>();
        auto result = output.ReinterpretCast<uint16_t>();
        const uint32_t count = static_cast<uint32_t>((rows * d_.outputRowBytes + 1) / 2);
        Gather(result, source, map_.Get<uint32_t>(), 0, count);
        if (d_.patchCount != 0) {
            const uint32_t n = static_cast<uint32_t>(d_.patchCount * 16);
            auto low = low_.Get<uint16_t>(), high = high_.Get<uint16_t>();
            Gather(low, source, lowMap_.Get<uint32_t>(), 0, n);
            Gather(high, source, highMap_.Get<uint32_t>(), 0, n);
            PipeBarrier<PIPE_V>();
            ShiftLeft(low, low, static_cast<uint16_t>(8), n);
            ShiftRight(high, high, static_cast<uint16_t>(8), n);
            PipeBarrier<PIPE_V>();
            ShiftRight(low, low, static_cast<uint16_t>(8), n);
            ShiftLeft(high, high, static_cast<uint16_t>(8), n);
            PipeBarrier<PIPE_V>();
            Or(low, low, high, n);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < d_.patchCount; ++i) {
                const uint64_t word = table_->patchBlocks[i] * 16;
                if (word >= count) { break; }
                DataCopy(result[word], low[i * 16], 16);
            }
        }
        outputQueue_.EnQue(output);
        inputQueue_.FreeTensor(input);
    }
    __aicore__ inline void CopyOut(uint64_t row, uint64_t rows) {
        auto output = outputQueue_.DeQue<uint8_t>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(rows * d_.outputRowBytes), 0, 0, 0};
        DataCopyPad(output_[row * d_.outputRowBytes], output, copy);
        if ((d_.tileRows * d_.outputRowBytes) % 32 != 0) { PipeBarrier<PIPE_MTE3>(); }
        outputQueue_.FreeTensor(output);
    }
    template <HardEvent E> __aicore__ inline void Sync() {
        const event_t id = static_cast<event_t>(pipe_.FetchEventID(E));
        SetFlag<E>(id); WaitFlag<E>(id);
    }
    ConcatBytePairHeader d_{};
    __gm__ ConcatBytePairTilingData* table_;
    TPipe pipe_;
    TQue<TPosition::VECIN, 2> inputQueue_;
    TQue<TPosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> map_, lowMap_, highMap_, low_, high_;
    __gm__ uint8_t* bases_[32];
    uint64_t widths_[32], offsets0_[32], offsets1_[32], dual_[32], shiftedOnly_[32];
    GlobalTensor<uint8_t> output_;
};



class KernelConcatBoundaryBody {
public:
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,GM_ADDR tiling,const ConcatBoundaryHeader& h){
  h_=h;t_=reinterpret_cast<__gm__ ConcatBoundaryTilingData*>(tiling);inputs_.Init(reinterpret_cast<__gm__ void*>(x));out_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));pipe_.InitBuffer(q_,2,h_.tileBytes);pending_=false;
 }
 __aicore__ inline void Process(){
  const uint32_t c=GetBlockIdx();
  for(uint32_t j=t_->coreOffsets[c];j<t_->coreOffsets[c+1];++j){
   const uint32_t width=t_->tasks[j].width,stride=t_->tasks[j].stride,original=t_->tasks[j].input;uint32_t remaining=t_->tasks[j].rows;
   uint64_t src=t_->tasks[j].src,dst=t_->tasks[j].dst;
   GlobalTensor<uint8_t> input;input.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(original));
   const uint32_t maxRows=static_cast<uint32_t>(MinU64(4095,h_.tileBytes/width));ConfigureConcatStreamRead(input,h_.totalBytes,static_cast<uint64_t>(width)*MinU64(maxRows,remaining));
   while(remaining){const uint32_t rows=static_cast<uint32_t>(MinU64(maxRows,remaining));auto local=q_.AllocTensor<uint8_t>();
    DataCopyExtParams read{static_cast<uint16_t>(rows),width,stride-width,0,0};DataCopyPadExtParams<uint8_t> pad{false,0,0,0};DataCopyPad(local,input[src],read,pad);q_.EnQue(local);
    Flush();pending_=true;pendingDst_=dst;pendingRows_=rows;pendingWidth_=width;pendingStride_=static_cast<uint32_t>(h_.quantum*h_.rowBytes-width);
    src+=static_cast<uint64_t>(rows)*stride;dst+=static_cast<uint64_t>(rows)*h_.quantum*h_.rowBytes;remaining-=rows;
   }
  }Flush();
  if(c==0&&h_.tailBytes){
   const uint32_t last=h_.inputCount-1,width=t_->inputs[last].width,original=t_->inputs[last].original;GlobalTensor<uint8_t> input;input.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(original));
   auto local=q_.AllocTensor<uint8_t>();DataCopyExtParams copy{1,h_.tailBytes,0,0,0};DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
   DataCopyPad(local,input[h_.outer*width-h_.tailBytes],copy,pad);q_.EnQue(local);local=q_.DeQue<uint8_t>();DataCopyPad(out_[h_.totalBytes-h_.tailBytes],local,copy);q_.FreeTensor(local);
  }PipeBarrier<PIPE_ALL>();
 }
private:
 __aicore__ inline void Flush(){if(!pending_)return;auto local=q_.DeQue<uint8_t>();DataCopyExtParams write{static_cast<uint16_t>(pendingRows_),pendingWidth_,0,pendingStride_,0};if(pendingRows_==1 || pendingStride_/32<=65535){DataCopyParams fast{static_cast<uint16_t>(pendingRows_),static_cast<uint16_t>(pendingWidth_/32),0,static_cast<uint16_t>(pendingRows_==1?0:pendingStride_/32)};DataCopy(out_[pendingDst_],local,fast);}else DataCopyPad(out_[pendingDst_],local,write);q_.FreeTensor(local);pending_=false;}
 ConcatBoundaryHeader h_{};__gm__ ConcatBoundaryTilingData* t_;ListTensorDesc inputs_;GlobalTensor<uint8_t> out_;TPipe pipe_;TQueBind<TPosition::VECIN,TPosition::VECOUT,2> q_;
 bool pending_;uint64_t pendingDst_;uint32_t pendingRows_,pendingWidth_,pendingStride_;
};

class KernelConcatBoundaryPackedEdges {
public:
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,GM_ADDR tiling,const ConcatBoundaryHeader& h){
  h_=h;t_=reinterpret_cast<__gm__ ConcatBoundaryTilingData*>(tiling);inputs_.Init(reinterpret_cast<__gm__ void*>(x));out_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(y));
  const uint32_t e=GetBlockIdx()-h_.bodyCores;first_=t_->edges[e].firstRow;rows_=t_->edges[e].rows;begin_=t_->edges[e].begin;count_=t_->edges[e].count;prev_=t_->edges[e].needsPrev;mask_=t_->edges[e].slotMask;mapBegin_=t_->edges[e].mapBegin;mapCount_=t_->edges[e].mapCount;shift_=t_->edges[e].shift;
  pipe_.InitBuffer(input_,(2*h_.inputCount+1)*h_.edgeSlotBytes);pipe_.InitBuffer(result_,h_.edgeOutputBytes);pipe_.InitBuffer(low_,h_.edgeOutputBytes);pipe_.InitBuffer(high_,h_.edgeOutputBytes);pipe_.InitBuffer(extra_,h_.edgeOutputBytes);
  ready_=static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());written_=static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
 }
 __aicore__ inline void Process(){
  if(count_){ReadInputs();GlobalTensor<uint32_t> lo,hi;lo.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(t_->lowMaps)+mapBegin_);hi.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(t_->highMaps)+mapBegin_);
   DataCopyExtParams copy{1,mapCount_*4,0,0,0};DataCopyPadExtParams<uint32_t> pad{false,0,0,0};DataCopyPad(low_.Get<uint32_t>(),lo,copy,pad);if(shift_)DataCopyPad(high_.Get<uint32_t>(),hi,copy,pad);
   SetFlag<HardEvent::MTE2_V>(ready_);WaitFlag<HardEvent::MTE2_V>(ready_);
   auto result=result_.Get<uint32_t>();Gather(result,input_.Get<uint32_t>(),low_.Get<uint32_t>(),0,mapCount_);
   if(shift_){auto high=extra_.Get<uint32_t>();Gather(high,input_.Get<uint32_t>(),high_.Get<uint32_t>(),0,mapCount_);PipeBarrier<PIPE_V>();ShiftRight(result,result,shift_,mapCount_);ShiftLeft(high,high,32-shift_,mapCount_);PipeBarrier<PIPE_V>();Or(result.ReinterpretCast<uint16_t>(),result.ReinterpretCast<uint16_t>(),high.ReinterpretCast<uint16_t>(),mapCount_*2);}
   PipeBarrier<PIPE_V>();SetFlag<HardEvent::V_MTE3>(written_);WaitFlag<HardEvent::V_MTE3>(written_);
   for(uint32_t b=0;b<count_;++b){const int32_t relative=t_->records[begin_+b].dstRelative;const uint64_t dst=static_cast<uint64_t>(static_cast<int64_t>(first_*h_.rowBytes)+relative);auto local=result[b*rows_*8];
    if(rows_==1){DataCopyParams fast{1,1,0,0};DataCopy(out_[dst/4],local,fast);}else{DataCopyExtParams write{static_cast<uint16_t>(rows_),32,0,static_cast<uint32_t>(h_.quantum*h_.rowBytes-32),0};DataCopyPad(out_[dst/4],local,write);}
   }
  }PipeBarrier<PIPE_ALL>();pipe_.ReleaseEventID<HardEvent::MTE2_V>(ready_);pipe_.ReleaseEventID<HardEvent::V_MTE3>(written_);
 }
private:
 __aicore__ inline void ReadInputs(){
  auto local=input_.Get<uint8_t>();DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
  for(uint32_t i=0;i<h_.inputCount;++i){const bool head=((mask_>>(2*i))&1)!=0,tail=((mask_>>(2*i+1))&1)!=0;if(!head&&!tail)continue;uint32_t w=t_->inputs[i].width;GlobalTensor<uint8_t> input;input.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(t_->inputs[i].original));DataCopyExtParams read{static_cast<uint16_t>(rows_),32,h_.quantum*w-32,0,0};
   if(head)DataCopyPad(local[2*i*h_.edgeSlotBytes],input[static_cast<uint64_t>(first_)*w],read,pad);if(tail)DataCopyPad(local[(2*i+1)*h_.edgeSlotBytes],input[(static_cast<uint64_t>(first_)+1)*w-32],read,pad);
  }
  if(prev_){uint32_t i=h_.inputCount-1,w=t_->inputs[i].width;GlobalTensor<uint8_t> input;input.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(t_->inputs[i].original));DataCopyExtParams read{static_cast<uint16_t>(rows_),32,h_.quantum*w-32,0,0};DataCopyPad(local[2*h_.inputCount*h_.edgeSlotBytes],input[static_cast<uint64_t>(first_)*w-32],read,pad);}
 }
 ConcatBoundaryHeader h_{};__gm__ ConcatBoundaryTilingData* t_;ListTensorDesc inputs_;GlobalTensor<uint32_t> out_;TPipe pipe_;
 TBuf<TPosition::VECCALC> input_,result_,low_,high_,extra_;event_t ready_,written_;uint32_t first_,rows_,begin_,count_,prev_,mapBegin_,mapCount_,shift_;uint64_t mask_;
};

class KernelConcatLinearTasks {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,GM_ADDR tiling,const ConcatDecodedTaskHeader& d){
        d_=d;table_=reinterpret_cast<__gm__ ConcatLinearTaskTilingData*>(tiling);
        inputs_=ListTensorDesc(reinterpret_cast<__gm__ void*>(x));output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));pipe_.InitBuffer(queue_,2,d_.tileBytes);
    }
    __aicore__ inline void Process(){
        const uint32_t core=GetBlockIdx(),begin=table_->coreOffsets[core],end=table_->coreOffsets[core+1];
        bool pending=false,written=false;uint32_t pendingBytes=0;uint64_t pendingDst=0;
        for(uint32_t slot=begin;slot<end;++slot){
            const uint32_t bytes=table_->tasks[slot].width,input=table_->tasks[slot].input;
            const uint64_t src=table_->tasks[slot].sourceOffset,dst=table_->tasks[slot].outputOffset;
            GlobalTensor<uint8_t> source;source.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(input));ConfigureConcatStreamRead(source,d_.totalBytes,bytes);
            auto local=queue_.AllocTensor<uint8_t>();DataCopyExtParams read{1,bytes,0,0,0};DataCopyPadExtParams<uint8_t> pad{false,0,0,0};DataCopyPad(local,source[src],read,pad);queue_.EnQue(local);
            if(pending){local=queue_.DeQue<uint8_t>();if(written)PipeBarrier<PIPE_MTE3>();DataCopyExtParams write{1,pendingBytes,0,0,0};DataCopyPad(output_[pendingDst],local,write);queue_.FreeTensor(local);written=true;}
            pending=true;pendingBytes=bytes;pendingDst=dst;
        }
        if(pending){auto local=queue_.DeQue<uint8_t>();if(written)PipeBarrier<PIPE_MTE3>();DataCopyExtParams write{1,pendingBytes,0,0,0};DataCopyPad(output_[pendingDst],local,write);queue_.FreeTensor(local);}
    }
private:
    ConcatDecodedTaskHeader d_{};__gm__ ConcatLinearTaskTilingData* table_;ListTensorDesc inputs_;GlobalTensor<uint8_t> output_;TPipe pipe_;TQueBind<TPosition::VECIN,TPosition::VECOUT,2> queue_;
};


class KernelConcatDecodedTasks {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tiling, const ConcatDecodedTaskHeader& d) {
        d_ = d; table_ = reinterpret_cast<__gm__ ConcatDecodedTaskTilingData*>(tiling);
        inputs_ = ListTensorDesc(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        pipe_.InitBuffer(queue_, 2, d_.tileBytes);
    }
    __aicore__ inline void Process() {
        const uint32_t core = GetBlockIdx();
        const uint32_t begin = table_->coreOffsets[core], end = table_->coreOffsets[core + 1];
        for (uint32_t slot = begin; slot < end; ++slot) Run(slot);
    }
private:
    __aicore__ inline void Run(uint32_t slot) {
        const uint32_t input = table_->tasks[slot].input;
        const uint32_t width = table_->tasks[slot].width;
        const uint32_t maxRows = table_->tasks[slot].rowsPerTile;
        uint32_t remaining = table_->tasks[slot].rows;
        uint64_t srcOffset = table_->tasks[slot].sourceOffset, dstOffset = table_->tasks[slot].outputOffset;
        GlobalTensor<uint8_t> source; source.SetGlobalBuffer(inputs_.GetDataPtr<uint8_t>(input));
        ConfigureConcatStreamRead(source, d_.totalBytes, width);
        bool pending = false; uint16_t pendingRows = 0; uint64_t pendingDst = 0;
        while (remaining != 0) {
            const uint16_t rows = static_cast<uint16_t>(MinU64(remaining, maxRows));
            auto local = queue_.AllocTensor<uint8_t>();
            DataCopyExtParams read{rows, width, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            DataCopyPad(local, source[srcOffset], read, pad); queue_.EnQue(local);
            if (pending) {
                local = queue_.DeQue<uint8_t>();
                DataCopyExtParams write{pendingRows, width, 0, static_cast<uint32_t>(d_.outputRowBytes - width), 0};
                DataCopyPad(output_[pendingDst], local, write); queue_.FreeTensor(local);
            }
            pending = true; pendingRows = rows; pendingDst = dstOffset;
            remaining -= rows; srcOffset += static_cast<uint64_t>(rows) * width; dstOffset += static_cast<uint64_t>(rows) * d_.outputRowBytes;
        }
        if (pending) {
            auto local = queue_.DeQue<uint8_t>();
            DataCopyExtParams write{pendingRows, width, 0, static_cast<uint32_t>(d_.outputRowBytes - width), 0};
            DataCopyPad(output_[pendingDst], local, write); queue_.FreeTensor(local);
        }
    }
    ConcatDecodedTaskHeader d_{}; __gm__ ConcatDecodedTaskTilingData* table_;
    ListTensorDesc inputs_; GlobalTensor<uint8_t> output_; TPipe pipe_;
    TQueBind<TPosition::VECIN,TPosition::VECOUT,2> queue_;
};

class KernelConcatSmallPredecoded {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,GM_ADDR tiling,const ConcatStripeHeader& d){
        d_=d;table_=reinterpret_cast<__gm__ ConcatStripeTilingData*>(tiling);inputs_.Init(reinterpret_cast<__gm__ void*>(x));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(y));pipe_.InitBuffer(input_,d_.inputBytes);
        pipe_.InitBuffer(outputBuffer_,d_.outputBytes);pipe_.InitBuffer(map_,AlignUp32(d_.mapCount*4));
    }
    __aicore__ inline void Process(){
        const uint64_t row=static_cast<uint64_t>(GetBlockIdx())*d_.blockRows;if(row>=d_.outer)return;
        const uint32_t rows=static_cast<uint32_t>(MinU64(d_.blockRows,d_.outer-row));
        GlobalTensor<uint32_t> map;map.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(table_->indices));
        DataCopyExtParams mp{1,d_.mapCount*4,0,0,0};DataCopyPadExtParams<uint32_t> np{false,0,0,0};DataCopyPad(map_.Get<uint32_t>(),map,mp,np);
        auto local=input_.Get<uint16_t>();
        if(d_.flags&1U){
            CopySmallKnown<0,27,0>(local,row,rows);
            CopySmallKnown<1,40,432>(local,row,rows);
            CopySmallKnown<2,63,1072>(local,row,rows);
            CopySmallKnown<3,24,2080>(local,row,rows);
            CopySmallKnown<4,50,2464>(local,row,rows);
            CopySmallKnown<5,26,3264>(local,row,rows);
            CopySmallKnown<6,19,3680>(local,row,rows);
            CopySmallKnown<7,2,3984>(local,row,rows);
            CopySmallKnown<8,5,4016>(local,row,rows);
        }else for(uint32_t i=0;i<d_.inputCount;++i){
            const uint32_t width=table_->inputRowBytes[i];GlobalTensor<uint16_t> source;source.SetGlobalBuffer(inputs_.GetDataPtr<uint16_t>(table_->originalInputs[i]));
            DataCopyExtParams read{1,rows*width,0,0,0};DataCopyPadExtParams<uint16_t> pad{false,0,0,0};
            DataCopyPad(local[table_->inputOffsets[i]],source[row*width/2],read,pad);
        }
        const event_t inputReady=static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));SetFlag<HardEvent::MTE2_V>(inputReady);WaitFlag<HardEvent::MTE2_V>(inputReady);
        auto result=outputBuffer_.Get<uint16_t>();Gather(result,local,map_.Get<uint32_t>(),0,rows*d_.outputRowBytes/2);
        const event_t outputReady=static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_MTE3));SetFlag<HardEvent::V_MTE3>(outputReady);WaitFlag<HardEvent::V_MTE3>(outputReady);
        DataCopyExtParams write{1,static_cast<uint32_t>(rows*d_.outputRowBytes),0,0,0};DataCopyPad(output_[row*d_.outputRowBytes/2],result,write);
    }
private:
    template<uint32_t Index,uint32_t Width,uint32_t Offset>
    __aicore__ inline void CopySmallKnown(const LocalTensor<uint16_t>& local,uint64_t row,uint32_t rows){
        GlobalTensor<uint16_t> source;source.SetGlobalBuffer(inputs_.GetDataPtr<uint16_t>(Index));
        DataCopyExtParams read{1,rows*Width*2,0,0,0};DataCopyPadExtParams<uint16_t> pad{false,0,0,0};
        AscendC::DataCopyPadGm2UBImpl(reinterpret_cast<__ubuf__ uint16_t*>(local.GetPhyAddr())+Offset,const_cast<__gm__ uint16_t*>(source.GetPhyAddr())+row*Width,read,pad);
    }
    ConcatStripeHeader d_{};__gm__ ConcatStripeTilingData* table_;ListTensorDesc inputs_;GlobalTensor<uint16_t> output_;
    TPipe pipe_;TBuf<TPosition::VECCALC> input_,outputBuffer_,map_;
};

extern "C" __global__ __aicore__ void concat(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(ConcatTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1600", ConcatStripeTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1371", ConcatStripeTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1361", ConcatBytePairTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1311", ConcatOutputChunksTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1309", ConcatAlignedSlabTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1305", ConcatMaskedRowsTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1303", ConcatCompactDirectTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1301", ConcatPipelineTilingData);
    REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1300", ConcatGatherRowsTilingData);
    // Use numeric literals: the CANN 8.5 precompiler enumerates kernel entries
    // before ordinary C++ constant folding.  Every branch is one data-flow,
    // one buffer policy, and one tiling ABI; unrelated code cannot pollute the
    // hot kernel's instruction/register footprint.
    if (TILING_KEY_IS(100)) {
        return;
    }
    if (TILING_KEY_IS(1100)) {
        return;
    }
    if (TILING_KEY_IS(102)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcat<ConcatTilingData, 2, false> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(103)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcat<ConcatTilingData, 2, true> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(104)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcat<ConcatTilingData, 3, false> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(105)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcat<ConcatTilingData, 5, false> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(106)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcat<ConcatTilingData, 1, false> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(107)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcat<ConcatTilingData, 1, true> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(108)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcat<ConcatTilingData, 4, false> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(109)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcat<ConcatTilingData, 4, true> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(110)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatDirectFlat<ConcatTilingData> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(126)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatDirectFlatQueue<ConcatTilingData> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(127)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        if (fullData.elementBytes == 1) {
            KernelConcatUniformNarrowRows<ConcatTilingData, int8_t, half, true> kernel;
            kernel.Init(x, y, fullData);
            kernel.Process();
        } else if (fullData.elementBytes == 2) {
            KernelConcatUniformNarrowRows<ConcatTilingData, half, half, false> kernel;
            kernel.Init(x, y, fullData);
            kernel.Process();
        } else {
            KernelConcatUniformNarrowRows<ConcatTilingData, uint32_t, uint32_t, false> kernel;
            kernel.Init(x, y, fullData);
            kernel.Process();
        }
        return;
    }
    if (TILING_KEY_IS(111)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatCacheLines<ConcatTilingData> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(112)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatUnitFp16Gather<ConcatTilingData> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(114)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatNarrowFp16Gather<ConcatTilingData, 2> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(115)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatNarrowFp16Gather<ConcatTilingData, 4> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(116)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatNarrowFp16Gather<ConcatTilingData, 8> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(117)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatNarrowFp16Gather<ConcatTilingData, 16> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(118)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatDirectRowTasks<ConcatTilingData> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(119)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatBalancedRowTilingData, balancedData, tiling);
        KernelConcatScheduledRows<ConcatBalancedRowTilingData> kernel;
        kernel.Init(x, y, balancedData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(120)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatContiguousRowPad<ConcatTilingData, half> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(121)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatInputMajorRowTilingData, inputMajorData, tiling);
        KernelConcatScheduledRows<ConcatInputMajorRowTilingData, true> kernel;
        kernel.Init(x, y, inputMajorData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(122)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatContiguousRowPad<ConcatTilingData, uint32_t> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(123)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatContiguousRowPad<ConcatTilingData, uint16_t> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(124)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatTilingData, fullData, tiling);
        KernelConcatAlignedRowPack<ConcatTilingData> kernel;
        kernel.Init(x, y, fullData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(125)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatAdaptiveRowTilingData, adaptiveData, tiling);
        KernelConcatAdaptiveRows<ConcatAdaptiveRowTilingData> kernel;
        kernel.Init(x, y, adaptiveData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1102)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcat<ConcatSmallTilingData, 2, false> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1103)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcat<ConcatSmallTilingData, 2, true> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1104)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcat<ConcatSmallTilingData, 3, false> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1105)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcat<ConcatSmallTilingData, 5, false> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1106)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcat<ConcatSmallTilingData, 1, false> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1107)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcat<ConcatSmallTilingData, 1, true> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1108)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcat<ConcatSmallTilingData, 4, false> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1109)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcat<ConcatSmallTilingData, 4, true> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1110)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatDirectFlat<ConcatSmallTilingData> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1126)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatDirectFlatQueue<ConcatSmallTilingData> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1128)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        if (smallData.elementBytes == 1) {
            KernelConcatRaggedNarrowRows<ConcatSmallTilingData, int8_t, half, true> kernel;
            kernel.Init(x, y, smallData); kernel.Process();
        } else if (smallData.elementBytes == 2) {
            KernelConcatRaggedNarrowRows<ConcatSmallTilingData, half, half, false> kernel;
            kernel.Init(x, y, smallData); kernel.Process();
        } else {
            KernelConcatRaggedNarrowRows<ConcatSmallTilingData, uint32_t, uint32_t, false> kernel;
            kernel.Init(x, y, smallData); kernel.Process();
        }
        return;
    }
    if (TILING_KEY_IS(1127)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        if (smallData.elementBytes == 1) {
            KernelConcatUniformNarrowRows<ConcatSmallTilingData, int8_t, half, true> kernel;
            kernel.Init(x, y, smallData);
            kernel.Process();
        } else if (smallData.elementBytes == 2) {
            KernelConcatUniformNarrowRows<ConcatSmallTilingData, half, half, false> kernel;
            kernel.Init(x, y, smallData);
            kernel.Process();
        } else {
            KernelConcatUniformNarrowRows<ConcatSmallTilingData, uint32_t, uint32_t, false> kernel;
            kernel.Init(x, y, smallData);
            kernel.Process();
        }
        return;
    }
    if (TILING_KEY_IS(1111)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatCacheLines<ConcatSmallTilingData> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1112)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatUnitFp16Gather<ConcatSmallTilingData> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1114)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatNarrowFp16Gather<ConcatSmallTilingData, 2> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1115)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatNarrowFp16Gather<ConcatSmallTilingData, 4> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1116)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatNarrowFp16Gather<ConcatSmallTilingData, 8> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1117)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatNarrowFp16Gather<ConcatSmallTilingData, 16> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1118)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatDirectRowTasks<ConcatSmallTilingData> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1120)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatContiguousRowPad<ConcatSmallTilingData, half> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1122)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatContiguousRowPad<ConcatSmallTilingData, uint32_t> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1123)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatContiguousRowPad<ConcatSmallTilingData, uint16_t> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1124)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatSmallTilingData, smallData, tiling);
        KernelConcatAlignedRowPack<ConcatSmallTilingData> kernel;
        kernel.Init(x, y, smallData);
        kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1381)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatDecodedTaskHeader, tasks, tiling);
        KernelConcatDecodedTasks kernel; kernel.Init(x, y, tiling, tasks); kernel.Process(); return;
    }
    if (TILING_KEY_IS(1361)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatBytePairHeader, pairs, tiling);
        KernelConcatBytePair kernel; kernel.Init(x, y, tiling, pairs); kernel.Process(); return;
    }
    if (TILING_KEY_IS(1311)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatOutputChunksHeader, data, tiling);
        KernelConcatOutputChunks op;
        op.Init(x, y, tiling, data); op.Process();
        return;
    }
    if (TILING_KEY_IS(1309)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatAlignedSlabHeader, data, tiling);
        KernelConcatAlignedSlab op;
        op.Init(x, y, tiling, data); op.Process();
        return;
    }
    if (TILING_KEY_IS(1305)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatMaskedRowsHeader, data, tiling);
        if (data.laneBytes == 4) {
            KernelConcatMaskedRows<uint32_t, uint32_t, uint32_t, false> op;
            op.Init(x, y, tiling, data); op.Process();
        } else if (data.laneBytes == 2) {
            KernelConcatMaskedRows<uint16_t, uint16_t, uint16_t, false> op;
            op.Init(x, y, tiling, data); op.Process();
        } else {
            KernelConcatMaskedRows<int8_t, half, uint16_t, true> op;
            op.Init(x, y, tiling, data); op.Process();
        }
        return;
    }
    if (TILING_KEY_IS(1303)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatCompactDirectHeader, direct, tiling);
        KernelConcatCompactDirect kernel;
        kernel.Init(x, y, tiling, direct); kernel.Process();
        return;
    }
    if (TILING_KEY_IS(1414)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatStripeHeader, stripe, tiling);
        KernelConcatStripeMetadata<uint32_t> op; op.Init(x, y, tiling, stripe); op.Process(); return;
    }
    if (TILING_KEY_IS(1929)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatBoundaryHeader, data, tiling);
        if(GetBlockIdx()<data.bodyCores){KernelConcatBoundaryBody op;op.Init(x,y,tiling,data);op.Process();}
        else{KernelConcatBoundaryPackedEdges op;op.Init(x,y,tiling,data);op.Process();}return;
    }
    if (TILING_KEY_IS(1411)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatDecodedTaskHeader, linear, tiling);
        KernelConcatLinearTasks op;op.Init(x,y,tiling,linear);op.Process();return;
    }
    if (TILING_KEY_IS(1600)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatStripeHeader, stripe, tiling);
        KernelConcatSmallPredecoded op;op.Init(x,y,tiling,stripe);op.Process();return;
    }
    if(TILING_KEY_IS(1945)){GET_TILING_DATA_WITH_STRUCT(ConcatStripeHeader,stripe,tiling);KernelConcatSubmitFirst op;op.Init(x,y,tiling,stripe);op.Process();return;}
    if (TILING_KEY_IS(1403)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatStripeHeader, stripe, tiling);
        KernelConcatStripeBatch op;op.Init(x,y,tiling,stripe);op.Process();return;
    }
    if (TILING_KEY_IS(1371)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatStripeHeader, stripe, tiling);
        if (stripe.laneBytes == 4) {
            KernelConcatStripe<uint32_t> op; op.Init(x, y, tiling, stripe); op.Process();
        } else {
            KernelConcatStripe<uint16_t> op; op.Init(x, y, tiling, stripe); op.Process();
        }
        return;
    }
    if (TILING_KEY_IS(1301)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatPipelineHeader, pipeline, tiling);
        if (pipeline.laneBytes == 4) {
            KernelConcatPipeline<uint32_t, uint32_t, false> kernel;
            kernel.Init(x, y, tiling, pipeline); kernel.Process();
        } else if (pipeline.laneBytes == 2) {
            KernelConcatPipeline<uint16_t, uint16_t, false> kernel;
            kernel.Init(x, y, tiling, pipeline); kernel.Process();
        } else {
            KernelConcatPipeline<int8_t, half, true> kernel;
            kernel.Init(x, y, tiling, pipeline); kernel.Process();
        }
        return;
    }
    if (TILING_KEY_IS(1300)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatGatherRowsHeader, data, tiling);
        if (data.base.elementBytes == 1) {
            KernelConcatGatherRows<int8_t, half, true> op;
            op.Init(x, y, tiling, data); op.Process();
        } else if (data.base.elementBytes == 2) {
            KernelConcatGatherRows<uint16_t, uint16_t, false> op;
            op.Init(x, y, tiling, data); op.Process();
        } else {
            KernelConcatGatherRows<uint32_t, uint32_t, false> op;
            op.Init(x, y, tiling, data); op.Process();
        }
        return;
    }
}
