#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t SINGLE_BUFFER_NUM = 1;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t MAX_RANK = 8;
constexpr uint32_t MATRIX_TILE = 16;
constexpr uint32_t MATRIX_TILE_ELEMENTS = MATRIX_TILE * MATRIX_TILE;
constexpr uint32_t MATRIX_TILE_BYTES = MATRIX_TILE_ELEMENTS * sizeof(half);
constexpr uint64_t MAX_UINT32_VALUE = 0xFFFFFFFFULL;

}

#include "transpose_tiling.h"

class KernelTransposeBlockSwapFp16 {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint64_t a,
        uint64_t b, uint64_t c, uint32_t d, uint64_t tasks, TPipe *pipe)
    {
        inputGm_.SetGlobalBuffer((__gm__ half *)input);
        outputGm_.SetGlobalBuffer((__gm__ half *)output);
        a_ = a; b_ = b; c_ = c; d_ = d; tasks_ = tasks;
        columnTasks_ = (c_ + TILE_C - 1) / TILE_C;
        groupElements_ = static_cast<uint32_t>(a_ * d_);
        inputPitch_ = (TILE_C * d_ + 15U) / 16U * 16U;
        outputPitch_ = (groupElements_ + 15U) / 16U * 16U;
        const uint32_t inputBytes = static_cast<uint32_t>(a_) * inputPitch_ * sizeof(half);
        const uint32_t outputBytes = outputPitch_ * 4U * sizeof(half);
        const uint32_t offsetBytes = outputPitch_ * 4U * sizeof(uint32_t);
        pipe->InitBuffer(inputQueue_, 2, inputBytes);
        pipe->InitBuffer(outputQueue_, 2, outputBytes);
        pipe->InitBuffer(offsets_, offsetBytes);
        const uint32_t count = outputPitch_ * 4U;
        auto scratch = inputQueue_.AllocTensor<float>();
        auto rowStore = inputQueue_.AllocTensor<int32_t>();
        auto index = offsets_.Get<float>();
        auto temp = scratch;
        auto row = rowStore;
        CreateVecIndex(index, 0.0f, count);
        PipeBarrier<PIPE_V>();
        Muls(temp, index, 1.00000012f / static_cast<float>(static_cast<int32_t>(outputPitch_)), count);
        PipeBarrier<PIPE_V>();
        Cast(row, temp, RoundMode::CAST_FLOOR, count);
        PipeBarrier<PIPE_V>();
        Cast(temp, row, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(temp, temp, -static_cast<float>(static_cast<int32_t>(outputPitch_)), count);
        PipeBarrier<PIPE_V>();
        Add(row.ReinterpretCast<float>(), temp, index, count);
        PipeBarrier<PIPE_V>();
        Mins(row.ReinterpretCast<float>(), row.ReinterpretCast<float>(),
             static_cast<float>(static_cast<int32_t>(groupElements_ - 1)), count);
        PipeBarrier<PIPE_V>();
        Muls(temp, row.ReinterpretCast<float>(), 1.00000012f / static_cast<float>(static_cast<int32_t>(d_)), count);
        PipeBarrier<PIPE_V>();
        Cast(offsets_.Get<int32_t>(), temp, RoundMode::CAST_FLOOR, count);
        PipeBarrier<PIPE_V>();
        Cast(index, offsets_.Get<int32_t>(), RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(index, index, static_cast<float>(static_cast<int32_t>(inputPitch_ - d_)), count);
        PipeBarrier<PIPE_V>();
        Add(index, index, row.ReinterpretCast<float>(), count);
        PipeBarrier<PIPE_V>();
        CreateVecIndex(temp, 0.0f, count);
        PipeBarrier<PIPE_V>();
        Muls(temp, temp, 1.00000012f / static_cast<float>(static_cast<int32_t>(outputPitch_)), count);
        PipeBarrier<PIPE_V>();
        Cast(row, temp, RoundMode::CAST_FLOOR, count);
        PipeBarrier<PIPE_V>();
        Cast(temp, row, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Muls(temp, temp, static_cast<float>(static_cast<int32_t>(d_)), count);
        PipeBarrier<PIPE_V>();
        Add(index, index, temp, count);
        PipeBarrier<PIPE_V>();
        Muls(index, index, 2.0f, count);
        PipeBarrier<PIPE_V>();
        Cast(offsets_.Get<int32_t>(), index, RoundMode::CAST_FLOOR, count);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        inputQueue_.FreeTensor(scratch);
        inputQueue_.FreeTensor(rowStore);
    }

    __aicore__ inline void Process()
    {
        const uint64_t block = GetBlockIdx();
        const uint64_t cores = GetBlockNum();
        if (block >= tasks_) return;
        CopyIn(block);
        const auto offsets = offsets_.Get<uint32_t>();
        for (uint64_t task = block; task < tasks_; task += cores) {
            if (task + cores < tasks_) CopyIn(task + cores);
            auto input = inputQueue_.DeQue<half>();
            const uint64_t batch = task / columnTasks_;
            const uint64_t firstC = task % columnTasks_ * TILE_C;
            const uint32_t validC = static_cast<uint32_t>(
                c_ - firstC < TILE_C ? c_ - firstC : TILE_C);
            for (uint32_t column = 0; column < validC; column += 4) {
                const uint32_t take = validC - column < 4 ? validC - column : 4;
                auto output = outputQueue_.AllocTensor<half>();
                Gather(output, input, offsets, column * d_ * sizeof(half), take * outputPitch_);
                outputQueue_.EnQue(output);
                output = outputQueue_.DeQue<half>();
                const uint64_t destination = ((firstC + column) * b_ + batch) * groupElements_;
                DataCopyPad(outputGm_[destination], output,
                    DataCopyExtParams{static_cast<uint16_t>(take),
                        static_cast<uint32_t>(groupElements_ * sizeof(half)), 0,
                        static_cast<uint32_t>((b_ - 1) * groupElements_ * sizeof(half)), 0});
                outputQueue_.FreeTensor(output);
            }
            inputQueue_.FreeTensor(input);
        }
    }

private:
    static constexpr uint32_t TILE_C = 24;
    __aicore__ inline void CopyIn(uint64_t task)
    {
        const uint64_t batch = task / columnTasks_;
        const uint64_t firstC = task % columnTasks_ * TILE_C;
        const uint32_t validC = static_cast<uint32_t>(
            c_ - firstC < TILE_C ? c_ - firstC : TILE_C);
        const uint32_t valid = validC * d_;
        const uint32_t aligned = (valid + 15U) / 16U * 16U;
        auto input = inputQueue_.AllocTensor<half>();
        DataCopyPad(input, inputGm_[(batch * c_ + firstC) * d_],
            DataCopyExtParams{static_cast<uint16_t>(a_),
                static_cast<uint32_t>(valid * sizeof(half)),
                static_cast<uint32_t>((b_ * c_ * d_ - valid) * sizeof(half)),
                (inputPitch_ - aligned) / 16U, 0},
            DataCopyPadExtParams<half>{true, 0, static_cast<uint8_t>(aligned - valid), 0});
        inputQueue_.EnQue(input);
    }

    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> offsets_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    uint64_t a_ = 0, b_ = 0, c_ = 0, tasks_ = 0, columnTasks_ = 0;
    uint32_t d_ = 0, groupElements_ = 0, inputPitch_ = 0, outputPitch_ = 0;
};

class KernelTransposeSlab2dByte {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint64_t rows,
        uint32_t columns, uint32_t paddedColumns, uint64_t tasks,
        const AscendC::tiling::ConfusionTransposeTiling &tiling, TPipe *pipe)
    {
        inputGm_.SetGlobalBuffer((__gm__ int8_t *)input);
        outputGm_.SetGlobalBuffer((__gm__ int8_t *)output);
        rows_ = rows;
        columns_ = columns;
        paddedColumns_ = paddedColumns;
        tasks_ = tasks;
        tiling_ = tiling;
        const uint32_t elements = ROWS * paddedColumns_;
        pipe->InitBuffer(inputQueue_, 1, elements);
        pipe->InitBuffer(outputQueue_, 1, elements);
        pipe->InitBuffer(halfInput_, elements * sizeof(half));
        pipe->InitBuffer(halfOutput_, elements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        const uint32_t elements = ROWS * paddedColumns_;
        auto halfInput = halfInput_.Get<half>();
        auto halfOutput = halfOutput_.Get<half>();
        for (uint64_t task = GetBlockIdx(); task < tasks_; task += GetBlockNum()) {
            const uint64_t firstRow = task * ROWS;
            const uint32_t rows = static_cast<uint32_t>(
                rows_ - firstRow < ROWS ? rows_ - firstRow : ROWS);
            auto input = inputQueue_.AllocTensor<int8_t>();
            DataCopyPad(input, inputGm_[firstRow * columns_],
                DataCopyExtParams{static_cast<uint16_t>(rows), columns_, 0, 0, 0},
                DataCopyPadExtParams<int8_t>{true, 0,
                    static_cast<uint8_t>(paddedColumns_ - columns_), 0});
            if (rows < ROWS) {
                Duplicate(input.ReinterpretCast<uint16_t>()[rows * paddedColumns_ / 2],
                    static_cast<uint16_t>(0), (ROWS - rows) * paddedColumns_ / 2);
            }
            inputQueue_.EnQue(input);
            input = inputQueue_.DeQue<int8_t>();
            PipeBarrier<PIPE_V>();
            Cast(halfInput, input, RoundMode::CAST_NONE, elements);
            PipeBarrier<PIPE_V>();
            Transpose(halfOutput, halfInput, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling_);
            PipeBarrier<PIPE_V>();
            auto output = outputQueue_.AllocTensor<int8_t>();
            Cast(output, halfOutput, RoundMode::CAST_NONE, elements);
            inputQueue_.FreeTensor(input);
            outputQueue_.EnQue(output);
            output = outputQueue_.DeQue<int8_t>();
            // A transposed byte row occupies exactly one 32-byte block.
            DataCopyPad(outputGm_[firstRow], output,
                DataCopyExtParams{static_cast<uint16_t>(columns_), rows, 0,
                    static_cast<uint32_t>(rows_ - rows), 0});
            outputQueue_.FreeTensor(output);
        }
    }

private:
    static constexpr uint32_t ROWS = 32;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TBuf<TPosition::VECCALC> halfInput_;
    TBuf<TPosition::VECCALC> halfOutput_;
    GlobalTensor<int8_t> inputGm_;
    GlobalTensor<int8_t> outputGm_;
    AscendC::tiling::ConfusionTransposeTiling tiling_;
    uint64_t rows_ = 0;
    uint64_t tasks_ = 0;
    uint32_t columns_ = 0;
    uint32_t paddedColumns_ = 0;
};

// Aligned FP16 matrices: transpose 16-row slabs in UB and scatter columns.
class KernelTransposeSlab2dFp16 {
public:
    __aicore__ inline KernelTransposeSlab2dFp16() = default;

    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output, uint64_t inputRows,
        uint64_t inputColumns, uint64_t taskCount,
        uint32_t rowsPerTask,
        const AscendC::tiling::ConfusionTransposeTiling &transposeTiling,
        TPipe *pipe)
    {
        inputGm_.SetGlobalBuffer((__gm__ half *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ half *)output);
        inputRows_ = inputRows;
        inputColumns_ = inputColumns;
        taskCount_ = taskCount;
        rowsPerTask_ = rowsPerTask;
        slabElements_ = static_cast<uint32_t>(rowsPerTask_ * inputColumns_);
        outputRowGapBytes_ = static_cast<uint32_t>(
            (inputRows_ - rowsPerTask_) * sizeof(half));
        transposeTiling_ = transposeTiling;
        pipe_ = pipe;
        const uint32_t slabBytes = slabElements_ * sizeof(half);
        pipe_->InitBuffer(inputQueue_, SINGLE_BUFFER_NUM, slabBytes);
        pipe_->InitBuffer(outputQueue_, SINGLE_BUFFER_NUM, slabBytes);
    }

    __aicore__ inline void Process()
    {
        if (inputRows_ == 0 || inputColumns_ == 0 || taskCount_ == 0 ||
            rowsPerTask_ == 0 || slabElements_ == 0) {
            return;
        }
        const uint64_t blockIndex = GetBlockIdx();
        const uint64_t blockCount = GetBlockNum();
        for (uint64_t task = blockIndex; task < taskCount_; task += blockCount) {
            const uint64_t rowStart = task * rowsPerTask_;
            CopyIn(rowStart * inputColumns_);
            Compute();
            CopyOut(rowStart);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t inputOffset)
    {
        LocalTensor<half> inputLocal = inputQueue_.AllocTensor<half>();
        DataCopy(inputLocal, inputGm_[inputOffset], slabElements_);
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<half> inputLocal = inputQueue_.DeQue<half>();
        LocalTensor<half> outputLocal = outputQueue_.AllocTensor<half>();
        Transpose(
            outputLocal, inputLocal, TransposeType::TRANSPOSE_ND2ND_ONLY,
            transposeTiling_);
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset)
    {
        LocalTensor<half> outputLocal = outputQueue_.DeQue<half>();
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(inputColumns_),
            static_cast<uint32_t>(rowsPerTask_ * sizeof(half)),
            0, outputRowGapBytes_, 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    TPipe *pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    AscendC::tiling::ConfusionTransposeTiling transposeTiling_;
    uint64_t inputRows_ = 0;
    uint64_t inputColumns_ = 0;
    uint64_t taskCount_ = 0;
    uint32_t rowsPerTask_ = 0;
    uint32_t slabElements_ = 0;
    uint32_t outputRowGapBytes_ = 0;
};

class KernelTransposeSlab2dWord32 {
public:
    __aicore__ inline KernelTransposeSlab2dWord32() = default;

    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output, uint64_t inputRows,
        uint64_t inputColumns, uint64_t taskCount,
        uint32_t rowsPerTask, uint32_t columnsPerTask,
        const AscendC::tiling::ConfusionTransposeTiling &transposeTiling,
        TPipe *pipe)
    {
        inputGm_.SetGlobalBuffer((__gm__ uint32_t *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ uint32_t *)output);
        inputRows_ = inputRows;
        inputColumns_ = inputColumns;
        taskCount_ = taskCount;
        rowsPerTask_ = rowsPerTask;
        columnsPerTask_ = columnsPerTask;
        columnTaskCount_ = inputColumns_ / columnsPerTask_;
        slabElements_ = rowsPerTask_ * columnsPerTask_;
        inputRowGapBytes_ = static_cast<uint32_t>(
            (inputColumns_ - columnsPerTask_) * sizeof(uint32_t));
        outputRowGapBytes_ = static_cast<uint32_t>(
            (inputRows_ - rowsPerTask_) * sizeof(uint32_t));
        transposeTiling_ = transposeTiling;
        pipe_ = pipe;
        const uint32_t slabBytes = slabElements_ * sizeof(uint32_t);
        pipe_->InitBuffer(inputQueue_, SINGLE_BUFFER_NUM, slabBytes);
        pipe_->InitBuffer(outputQueue_, SINGLE_BUFFER_NUM, slabBytes);
    }

    __aicore__ inline void Process()
    {
        if (inputRows_ == 0 || inputColumns_ == 0 || taskCount_ == 0 ||
            rowsPerTask_ == 0 || columnsPerTask_ == 0 ||
            columnTaskCount_ == 0 || slabElements_ == 0) {
            return;
        }
        const uint64_t blockIndex = GetBlockIdx();
        const uint64_t blockCount = GetBlockNum();
        for (uint64_t task = blockIndex; task < taskCount_; task += blockCount) {
            const uint64_t rowTask = task / columnTaskCount_;
            const uint64_t columnTask = task % columnTaskCount_;
            const uint64_t rowStart = rowTask * rowsPerTask_;
            const uint64_t columnStart = columnTask * columnsPerTask_;
            const uint64_t inputOffset =
                rowStart * inputColumns_ + columnStart;
            const uint64_t outputOffset =
                columnStart * inputRows_ + rowStart;
            CopyIn(inputOffset);
            Compute();
            CopyOut(outputOffset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t inputOffset)
    {
        LocalTensor<uint32_t> inputLocal =
            inputQueue_.AllocTensor<uint32_t>();
        if (columnsPerTask_ == inputColumns_) {
            DataCopy(inputLocal, inputGm_[inputOffset], slabElements_);
        } else {
            const DataCopyExtParams copyParams{
                static_cast<uint16_t>(rowsPerTask_),
                static_cast<uint32_t>(
                    columnsPerTask_ * sizeof(uint32_t)),
                inputRowGapBytes_, 0, 0};
            const DataCopyPadExtParams<uint32_t> padParams{
                false, 0, 0, 0};
            DataCopyPad(
                inputLocal, inputGm_[inputOffset], copyParams, padParams);
        }
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<uint32_t> inputLocal = inputQueue_.DeQue<uint32_t>();
        LocalTensor<uint32_t> outputLocal =
            outputQueue_.AllocTensor<uint32_t>();
        Transpose(
            outputLocal, inputLocal, TransposeType::TRANSPOSE_ND2ND_ONLY,
            transposeTiling_);
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset)
    {
        LocalTensor<uint32_t> outputLocal = outputQueue_.DeQue<uint32_t>();
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(columnsPerTask_),
            static_cast<uint32_t>(rowsPerTask_ * sizeof(uint32_t)),
            0, outputRowGapBytes_, 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    TPipe *pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    GlobalTensor<uint32_t> inputGm_;
    GlobalTensor<uint32_t> outputGm_;
    AscendC::tiling::ConfusionTransposeTiling transposeTiling_;
    uint64_t inputRows_ = 0;
    uint64_t inputColumns_ = 0;
    uint64_t taskCount_ = 0;
    uint64_t columnTaskCount_ = 0;
    uint32_t rowsPerTask_ = 0;
    uint32_t columnsPerTask_ = 0;
    uint32_t slabElements_ = 0;
    uint32_t inputRowGapBytes_ = 0;
    uint32_t outputRowGapBytes_ = 0;
};

// Stream full input rows, then use 16-row/8-column TransDataTo5HD groups.
// Two output buffers alternate writes; only valid tail rows and columns are stored.
class KernelTransposeFullRowVnchwWord32 {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
        uint64_t inputRows, uint64_t inputColumns, uint64_t taskCount,
        uint32_t prefixRank, const uint64_t *dims, const uint64_t *strides)
    {
        inputGm_.SetGlobalBuffer((__gm__ uint32_t *)inputs);
        inputGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        outputGm_.SetGlobalBuffer((__gm__ uint32_t *)output);
        rows_ = inputRows; columns_ = inputColumns; pitch_ = (columns_ + 7) / 8 * 8; taskCount_ = taskCount;
        rowTiles_ = (rows_ + ROWS - 1) / ROWS;
        prefixRank_ = prefixRank; dims_ = dims; strides_ = strides;
        pipe_.InitBuffer(inputQueue_, 1, ROWS * pitch_ * sizeof(uint32_t));
        pipe_.InitBuffer(outputQueue_, 2, ROWS * COLS * sizeof(uint32_t));

    }

    __aicore__ inline void Process()
    {
        for (uint64_t task = GetBlockIdx(); task < taskCount_; task += GetBlockNum()) {
            const uint64_t batch = task / rowTiles_;
            const uint64_t rowStart = task % rowTiles_ * ROWS;
            const uint32_t validRows = static_cast<uint32_t>(
                rows_ - rowStart < ROWS ? rows_ - rowStart : ROWS);
            auto raw = inputQueue_.AllocTensor<uint32_t>();
            if (validRows < ROWS) {
                Duplicate(raw, static_cast<uint32_t>(0), ROWS * pitch_);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
            }
            DataCopyPad(raw, inputGm_[InputBatchOffset(batch) + rowStart * columns_],
                DataCopyExtParams{static_cast<uint16_t>(validRows), static_cast<uint32_t>(columns_ * 4), 0, 0, 0},
                DataCopyPadExtParams<uint32_t>{pitch_ != columns_, 0,
                    static_cast<uint8_t>(pitch_ - columns_), 0});
            inputQueue_.EnQue(raw);
            raw = inputQueue_.DeQue<uint32_t>();
            for (uint32_t column = 0; column < columns_; column += COLS) {
                const uint32_t take = columns_ - column < COLS ? columns_ - column : COLS;
                auto result = outputQueue_.AllocTensor<uint32_t>();
                TransDataTo5HDParams params;
                params.repeatTimes = (take + 7) / 8;
                params.dstRepStride = params.repeatTimes > 1 ? ROWS : 0;
                params.srcRepStride = params.repeatTimes > 1 ? 1 : 0;
                for (uint32_t row = 0; row < ROWS; row += 16) {
                    uint64_t dstList[16], srcList[16];
                    for (uint32_t lane = 0; lane < 16; lane += 2) {
                        dstList[lane] = reinterpret_cast<uint64_t>(result[row + ROWS * (lane / 2)].GetPhyAddr());
                        dstList[lane + 1] = reinterpret_cast<uint64_t>(result[row + ROWS * (lane / 2) + 8].GetPhyAddr());
                    }
                    for (uint32_t lane = 0; lane < 16; ++lane) {
                        srcList[lane] = reinterpret_cast<uint64_t>(raw[(row + lane) * pitch_ + column].GetPhyAddr());
                    }
                    TransDataTo5HD<uint32_t>(dstList, srcList, params);
                }
                outputQueue_.EnQue(result);
                result = outputQueue_.DeQue<uint32_t>();
                const uint64_t outputOffset = batch * rows_ * columns_ +
                    static_cast<uint64_t>(column) * rows_ + rowStart;
                DataCopyPad(outputGm_[outputOffset], result,
                    DataCopyExtParams{static_cast<uint16_t>(take), validRows * 4,
                        (ROWS - (validRows + 7) / 8 * 8) / 8,
                        static_cast<uint32_t>((rows_ - validRows) * 4), 0});
                outputQueue_.FreeTensor(result);
            }
            inputQueue_.FreeTensor(raw);
        }
    }
private:
    __aicore__ inline uint64_t InputBatchOffset(uint64_t batch) const
    {
        if (prefixRank_ == 0) return batch * rows_ * columns_;
        uint64_t offset = 0;
        for (uint32_t reverse = prefixRank_; reverse > 0; --reverse) {
            const uint32_t axis = reverse - 1;
            offset += (batch % dims_[axis]) * strides_[axis];
            batch /= dims_[axis];
        }
        return offset;
    }
    static constexpr uint32_t ROWS = 48;
    static constexpr uint32_t COLS = 240;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    GlobalTensor<uint32_t> inputGm_, outputGm_;
    uint64_t pitch_ = 0;
    uint64_t rows_ = 0, columns_ = 0, taskCount_ = 0, rowTiles_ = 0;
    uint32_t prefixRank_ = 0;
    const uint64_t *dims_ = nullptr, *strides_ = nullptr;
};


class KernelTransposePackedGatherWord32 {
public:
    __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output,
        uint64_t inputRows, uint64_t inputColumns, uint64_t taskCount,
        uint32_t prefixRank, const uint64_t *dims, const uint64_t *strides)
    {
        inputGm_.SetGlobalBuffer((__gm__ uint32_t *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ uint32_t *)output);
        rows_ = inputRows; columns_ = inputColumns; taskCount_ = taskCount;
        rowTiles_ = (rows_ + ROWS - 1) / ROWS;
        prefixRank_ = prefixRank; dims_ = dims; strides_ = strides;
        pipe_.InitBuffer(inputQueue_, 1, ROWS * columns_ * sizeof(uint32_t));
        pipe_.InitBuffer(outputQueue_, 2, ROWS * COLS * sizeof(uint32_t));
        pipe_.InitBuffer(offsetBuffer_, ROWS * COLS * sizeof(uint32_t));
        auto scratch = inputQueue_.AllocTensor<float>();
        auto offsetsFloat = offsetBuffer_.Get<float>();
        auto offsets = offsetBuffer_.Get<int32_t>();
        CreateVecIndex(scratch, 0.0f, ROWS * COLS);
        PipeBarrier<PIPE_V>();
        Muls(offsetsFloat, scratch, 1.0f / ROWS, ROWS * COLS);
        PipeBarrier<PIPE_V>();
        Cast(offsets, offsetsFloat, RoundMode::CAST_FLOOR, ROWS * COLS);
        PipeBarrier<PIPE_V>();
        Cast(offsetsFloat, offsets, RoundMode::CAST_NONE, ROWS * COLS);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, static_cast<float>(static_cast<int32_t>(columns_)), ROWS * COLS);
        Muls(offsetsFloat, offsetsFloat,
             1.0f - static_cast<float>(static_cast<int32_t>(ROWS * columns_)), ROWS * COLS);
        PipeBarrier<PIPE_V>();
        Add(offsetsFloat, offsetsFloat, scratch, ROWS * COLS);
        PipeBarrier<PIPE_V>();
        Muls(offsetsFloat, offsetsFloat, 4.0f, ROWS * COLS);
        PipeBarrier<PIPE_V>();
        Cast(offsets, offsetsFloat, RoundMode::CAST_FLOOR, ROWS * COLS);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        inputQueue_.FreeTensor(scratch);
    }

    __aicore__ inline void Process()
    {
        for (uint64_t task = GetBlockIdx(); task < taskCount_; task += GetBlockNum()) {
            const uint64_t batch = task / rowTiles_;
            const uint64_t rowStart = task % rowTiles_ * ROWS;
            const uint32_t validRows = static_cast<uint32_t>(
                rows_ - rowStart < ROWS ? rows_ - rowStart : ROWS);
            auto raw = inputQueue_.AllocTensor<uint32_t>();
            if (validRows < ROWS) {
                Duplicate(raw, static_cast<uint32_t>(0), ROWS * columns_);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
            }
            const uint32_t count = validRows * columns_;
            DataCopyPad(raw, inputGm_[InputBatchOffset(batch) + rowStart * columns_],
                DataCopyExtParams{1, count * 4, 0, 0, 0},
                DataCopyPadExtParams<uint32_t>{true, 0,
                    static_cast<uint8_t>((8 - count % 8) % 8), 0});
            inputQueue_.EnQue(raw);
            raw = inputQueue_.DeQue<uint32_t>();
            for (uint32_t column = 0; column < columns_; column += COLS) {
                const uint32_t take = columns_ - column < COLS ? columns_ - column : COLS;
                auto result = outputQueue_.AllocTensor<uint32_t>();
                Gather(result, raw, offsetBuffer_.Get<uint32_t>(), column * 4, take * ROWS);
                outputQueue_.EnQue(result);
                result = outputQueue_.DeQue<uint32_t>();
                const uint64_t outputOffset = batch * rows_ * columns_ +
                    static_cast<uint64_t>(column) * rows_ + rowStart;
                DataCopyPad(outputGm_[outputOffset], result,
                    DataCopyExtParams{static_cast<uint16_t>(take), validRows * 4,
                        (ROWS - (validRows + 7) / 8 * 8) / 8,
                        static_cast<uint32_t>((rows_ - validRows) * 4), 0});
                outputQueue_.FreeTensor(result);
            }
            inputQueue_.FreeTensor(raw);
        }
    }
private:
    __aicore__ inline uint64_t InputBatchOffset(uint64_t batch) const
    {
        if (prefixRank_ == 0) return batch * rows_ * columns_;
        uint64_t offset = 0;
        for (uint32_t reverse = prefixRank_; reverse > 0; --reverse) {
            const uint32_t axis = reverse - 1;
            offset += (batch % dims_[axis]) * strides_[axis];
            batch /= dims_[axis];
        }
        return offset;
    }
    static constexpr uint32_t ROWS = 64;
    static constexpr uint32_t COLS = 64;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
    GlobalTensor<uint32_t> inputGm_, outputGm_;
    uint64_t rows_ = 0, columns_ = 0, taskCount_ = 0, rowTiles_ = 0;
    uint32_t prefixRank_ = 0;
    const uint64_t *dims_ = nullptr, *strides_ = nullptr;
};


// Batched matrices: use UB-bounded slabs and preserve original batch strides.
class KernelTransposeBatchedSlabWord32 {
public:
    __aicore__ inline KernelTransposeBatchedSlabWord32() = default;

    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output, uint64_t batchCount,
        uint64_t inputRows, uint64_t inputColumns, uint64_t taskCount,
        uint32_t rowsPerTask, uint32_t columnsPerTask,
        uint32_t matrixPrefixRank, const uint64_t *outputDims,
        const uint64_t *sourceStrides,
        const AscendC::tiling::ConfusionTransposeTiling &transposeTiling,
        const AscendC::tiling::ConfusionTransposeTiling &tailTransposeTiling,
        TPipe *pipe)
    {
        inputGm_.SetGlobalBuffer((__gm__ uint32_t *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ uint32_t *)output);
        batchCount_ = batchCount;
        inputRows_ = inputRows;
        inputColumns_ = inputColumns;
        taskCount_ = taskCount;
        rowsPerTask_ = rowsPerTask;
        columnsPerTask_ = columnsPerTask;
        matrixPrefixRank_ = matrixPrefixRank;
        outputDims_ = outputDims;
        sourceStrides_ = sourceStrides;
        matrixElements_ = inputRows_ * inputColumns_;
        rowTaskCount_ =
            (inputRows_ + rowsPerTask_ - 1) / rowsPerTask_;
        fullColumnTaskCount_ = inputColumns_ / columnsPerTask_;
        const uint64_t remainingColumns =
            inputColumns_ % columnsPerTask_;
        paddedColumnTaskCount_ =
            remainingColumns != 0 &&
                    columnsPerTask_ - remainingColumns <
                        BLOCK_BYTES / sizeof(uint32_t)
                ? 1
                : 0;
        tailColumnTaskCount_ = paddedColumnTaskCount_ == 0
            ? (remainingColumns + MATRIX_TILE - 1) / MATRIX_TILE
            : 0;
        columnTaskCount_ =
            fullColumnTaskCount_ + paddedColumnTaskCount_ +
            tailColumnTaskCount_;
        transposeTiling_ = transposeTiling;
        tailTransposeTiling_ = tailTransposeTiling;
        pipe_ = pipe;
        const uint32_t slabBytes =
            rowsPerTask_ * columnsPerTask_ * sizeof(uint32_t);
        pipe_->InitBuffer(inputQueue_, SINGLE_BUFFER_NUM, slabBytes);
        const uint32_t outputBufferCount =
            inputColumns_ <= 32 ? BUFFER_NUM : SINGLE_BUFFER_NUM;
        pipe_->InitBuffer(outputQueue_, outputBufferCount, slabBytes);
    }

    __aicore__ inline void Process()
    {
        if (batchCount_ == 0 || inputRows_ == 0 || inputColumns_ == 0 ||
            taskCount_ == 0 || rowsPerTask_ == 0 ||
            columnsPerTask_ == 0 || rowTaskCount_ == 0 ||
            columnTaskCount_ == 0) {
            return;
        }
        const uint64_t tasksPerBatch =
            rowTaskCount_ * columnTaskCount_;
        const uint64_t blockIndex = GetBlockIdx();
        const uint64_t blockCount = GetBlockNum();
        for (uint64_t task = blockIndex; task < taskCount_; task += blockCount) {
            const uint64_t batch = task / tasksPerBatch;
            const uint64_t batchTask = task % tasksPerBatch;
            const uint64_t rowTask = batchTask / columnTaskCount_;
            const uint64_t columnTask = batchTask % columnTaskCount_;
            const bool mainColumnTask =
                columnTask < fullColumnTaskCount_ + paddedColumnTaskCount_;
            const uint64_t rowStart = rowTask * rowsPerTask_;
            const uint64_t columnStart = mainColumnTask
                ? columnTask * columnsPerTask_
                : fullColumnTaskCount_ * columnsPerTask_ +
                    (columnTask - fullColumnTaskCount_) * MATRIX_TILE;
            const uint32_t validRows = static_cast<uint32_t>(
                inputRows_ - rowStart < rowsPerTask_
                    ? inputRows_ - rowStart : rowsPerTask_);
            const uint32_t taskColumns =
                mainColumnTask ? columnsPerTask_ : MATRIX_TILE;
            const uint32_t validColumns = static_cast<uint32_t>(
                inputColumns_ - columnStart < taskColumns
                    ? inputColumns_ - columnStart : taskColumns);
            const uint64_t inputOffset = InputBatchOffset(batch) +
                rowStart * inputColumns_ + columnStart;
            const uint64_t outputOffset = batch * matrixElements_ +
                columnStart * inputRows_ + rowStart;
            CopyIn(inputOffset, validRows, validColumns, taskColumns);
            Compute(mainColumnTask);
            CopyOut(outputOffset, validRows, validColumns);
        }
    }

private:
    __aicore__ inline uint64_t InputBatchOffset(uint64_t batch) const
    {
        if (matrixPrefixRank_ == 0) {
            return batch * matrixElements_;
        }
        uint64_t remaining = batch;
        uint64_t offset = 0;
        for (uint32_t reverse = matrixPrefixRank_; reverse > 0; --reverse) {
            const uint32_t axis = reverse - 1;
            const uint64_t coordinate = remaining % outputDims_[axis];
            remaining /= outputDims_[axis];
            offset += coordinate * sourceStrides_[axis];
        }
        return offset;
    }

    __aicore__ inline void CopyIn(
        uint64_t inputOffset, uint32_t validRows,
        uint32_t validColumns, uint32_t taskColumns)
    {
        LocalTensor<uint32_t> inputLocal =
            inputQueue_.AllocTensor<uint32_t>();
        const uint32_t alignedColumns = (validColumns + 7) / 8 * 8;
        if (alignedColumns < taskColumns) {
            Duplicate<uint32_t>(inputLocal, 0, rowsPerTask_ * taskColumns);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        }
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(validRows),
            static_cast<uint32_t>(validColumns * sizeof(uint32_t)),
            static_cast<uint32_t>(
                (inputColumns_ - validColumns) * sizeof(uint32_t)),
            (taskColumns - alignedColumns) / 8, 0};
        const DataCopyPadExtParams<uint32_t> padParams{
            true, 0,
            static_cast<uint8_t>(alignedColumns - validColumns), 0};
        DataCopyPad(
            inputLocal, inputGm_[inputOffset], copyParams, padParams);
        if (validRows < rowsPerTask_ && alignedColumns == taskColumns) {
            Duplicate<uint32_t>(
                inputLocal[validRows * taskColumns], 0,
                (rowsPerTask_ - validRows) * taskColumns);
        }
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(bool fullColumnTask)
    {
        LocalTensor<uint32_t> inputLocal =
            inputQueue_.DeQue<uint32_t>();
        LocalTensor<uint32_t> outputLocal =
            outputQueue_.AllocTensor<uint32_t>();
        Transpose(
            outputLocal, inputLocal, TransposeType::TRANSPOSE_ND2ND_ONLY,
            fullColumnTask ? transposeTiling_ : tailTransposeTiling_);
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(
        uint64_t outputOffset, uint32_t validRows,
        uint32_t validColumns)
    {
        LocalTensor<uint32_t> outputLocal =
            outputQueue_.DeQue<uint32_t>();
        const uint32_t rowBytes = rowsPerTask_ * sizeof(uint32_t);
        const uint32_t validRowBytes = validRows * sizeof(uint32_t);
        const uint32_t alignedValidRowBytes =
            (validRowBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(validColumns), validRowBytes,
            static_cast<uint32_t>(
                (rowBytes - alignedValidRowBytes) / BLOCK_BYTES),
            static_cast<uint32_t>(
                (inputRows_ - validRows) * sizeof(uint32_t)),
            0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    TPipe *pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    GlobalTensor<uint32_t> inputGm_;
    GlobalTensor<uint32_t> outputGm_;
    AscendC::tiling::ConfusionTransposeTiling transposeTiling_;
    AscendC::tiling::ConfusionTransposeTiling tailTransposeTiling_;
    uint64_t batchCount_ = 0;
    uint64_t inputRows_ = 0;
    uint64_t inputColumns_ = 0;
    uint64_t matrixElements_ = 0;
    uint64_t taskCount_ = 0;
    uint64_t rowTaskCount_ = 0;
    uint64_t fullColumnTaskCount_ = 0;
    uint64_t paddedColumnTaskCount_ = 0;
    uint64_t tailColumnTaskCount_ = 0;
    uint64_t columnTaskCount_ = 0;
    uint32_t rowsPerTask_ = 0;
    uint32_t columnsPerTask_ = 0;
    uint32_t matrixPrefixRank_ = 0;
    const uint64_t *outputDims_ = nullptr;
    const uint64_t *sourceStrides_ = nullptr;
};

class KernelTransposeStridedTileWord32 {
public:
    __aicore__ inline KernelTransposeStridedTileWord32() = default;

    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output, uint64_t taskCount,
        uint64_t matrixRows, uint64_t matrixColumns,
        uint32_t paddedRows, uint32_t paddedColumns,
        uint32_t rank, uint32_t matrixRowAxis, uint32_t matrixColumnAxis,
        uint32_t firstColumnAxis,
        uint64_t outputMatrixRowStride, const uint64_t *outputDims,
        const uint64_t *sourceStrides,
        const AscendC::tiling::ConfusionTransposeTiling &transposeTiling,
        TPipe *pipe)
    {
        inputGm_.SetGlobalBuffer((__gm__ uint32_t *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ uint32_t *)output);
        taskCount_ = taskCount;
        matrixRows_ = static_cast<uint32_t>(matrixRows);
        matrixColumns_ = static_cast<uint32_t>(matrixColumns);
        paddedRows_ = paddedRows;
        paddedColumns_ = paddedColumns;
        rank_ = rank;
        matrixRowAxis_ = matrixRowAxis;
        matrixColumnAxis_ = matrixColumnAxis;
        firstColumnAxis_ = firstColumnAxis;
        outputMatrixRowStride_ = outputMatrixRowStride;
        transposeTiling_ = transposeTiling;
        for (uint32_t axis = 0; axis < MAX_RANK; ++axis) {
            outputDims_[axis] = outputDims[axis];
            sourceStrides_[axis] = sourceStrides[axis];
        }
        pipe_ = pipe;
        const uint32_t tileBytes =
            paddedRows_ * paddedColumns_ * sizeof(uint32_t);
        pipe_->InitBuffer(inputQueue_, paddedColumns > 32 ? 2 : 1, tileBytes);
        pipe_->InitBuffer(outputQueue_, paddedColumns > 32 ? 2 : 1, tileBytes);
    }

    __aicore__ inline void Process()
    {
        if (taskCount_ == 0 || matrixRows_ == 0 || matrixColumns_ == 0 ||
            paddedRows_ == 0 || paddedColumns_ == 0) {
            return;
        }
        const uint64_t blockIndex = GetBlockIdx();
        const uint64_t blockCount = GetBlockNum();
        uint64_t inputOffsets[2], outputOffsets[2];
        uint32_t validColumns[2];
        if (blockIndex >= taskCount_) return;
        PrepareTask(blockIndex, inputOffsets[0], outputOffsets[0], validColumns[0]);
        CopyIn(inputOffsets[0], validColumns[0]);
        uint32_t slot = 0;
        for (uint64_t task = blockIndex; task < taskCount_; task += blockCount) {
            if (paddedColumns_ > 32 && task + blockCount < taskCount_) {
                PrepareTask(task + blockCount, inputOffsets[1 - slot], outputOffsets[1 - slot], validColumns[1 - slot]);
                CopyIn(inputOffsets[1 - slot], validColumns[1 - slot]);
            }
            Compute();
            CopyOut(outputOffsets[slot], validColumns[slot]);
            slot = 1 - slot;
            if (paddedColumns_ == 32 && task + blockCount < taskCount_) {
                PrepareTask(task + blockCount, inputOffsets[slot], outputOffsets[slot], validColumns[slot]);
                CopyIn(inputOffsets[slot], validColumns[slot]);
            }
        }
    }

private:
    __aicore__ inline void PrepareTask(uint64_t task, uint64_t &inputOffset,
                                       uint64_t &outputOffset, uint32_t &columns) const
    {
        const uint64_t columnTiles = (matrixColumns_ + paddedColumns_ - 1) / paddedColumns_;
        const uint64_t columnStart = task % columnTiles * paddedColumns_;
        DecodeBatchOffsets(task / columnTiles, inputOffset, outputOffset);
        inputOffset += columnStart;
        outputOffset += columnStart * outputMatrixRowStride_;
        columns = static_cast<uint32_t>(matrixColumns_ - columnStart < paddedColumns_ ?
            matrixColumns_ - columnStart : paddedColumns_);
    }

    __aicore__ inline void DecodeBatchOffsets(
        uint64_t batch, uint64_t &inputOffset, uint64_t &outputOffset) const
    {
        uint64_t remaining = batch;
        uint64_t outputStride = 1;
        inputOffset = 0;
        outputOffset = 0;
        for (uint32_t reverse = rank_; reverse > 0; --reverse) {
            const uint32_t axis = reverse - 1;
            if (axis == matrixRowAxis_ || (axis >= firstColumnAxis_ && axis <= matrixColumnAxis_)) {
                outputStride *= outputDims_[axis];
                continue;
            }
            const uint64_t coordinate = remaining % outputDims_[axis];
            remaining /= outputDims_[axis];
            inputOffset += coordinate * sourceStrides_[axis];
            outputOffset += coordinate * outputStride;
            outputStride *= outputDims_[axis];
        }
    }

    __aicore__ inline void CopyIn(uint64_t inputOffset, uint32_t columns)
    {
        LocalTensor<uint32_t> inputLocal =
            inputQueue_.AllocTensor<uint32_t>();
        const uint32_t inputGapBytes = static_cast<uint32_t>(
            (sourceStrides_[matrixRowAxis_] - columns) *
            sizeof(uint32_t));
        const uint32_t alignedColumns = (columns + 7U) / 8U * 8U;
        if (columns < paddedColumns_) {
            Duplicate(inputLocal, static_cast<uint32_t>(0), paddedRows_ * paddedColumns_);
            SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
        }
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(matrixRows_), columns * 4,
            inputGapBytes, (paddedColumns_ - alignedColumns) / 8, 0};
        const DataCopyPadExtParams<uint32_t> padParams{
            true, 0,
            static_cast<uint8_t>(alignedColumns - columns), 0};
        DataCopyPad(inputLocal, inputGm_[inputOffset], copyParams, padParams);
        if (matrixRows_ < paddedRows_) {
            Duplicate<uint32_t>(
                inputLocal[matrixRows_ * paddedColumns_], 0,
                (paddedRows_ - matrixRows_) * paddedColumns_);
        }
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<uint32_t> inputLocal =
            inputQueue_.DeQue<uint32_t>();
        LocalTensor<uint32_t> outputLocal =
            outputQueue_.AllocTensor<uint32_t>();
        Transpose(
            outputLocal, inputLocal, TransposeType::TRANSPOSE_ND2ND_ONLY,
            transposeTiling_);
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset, uint32_t columns)
    {
        LocalTensor<uint32_t> outputLocal =
            outputQueue_.DeQue<uint32_t>();
        const uint32_t validRowBytes = matrixRows_ * sizeof(uint32_t);
        const uint32_t paddedRowBytes = paddedRows_ * sizeof(uint32_t);
        const uint32_t alignedValidRowBytes =
            (validRowBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
        const uint32_t outputGapBytes = static_cast<uint32_t>(
            (outputMatrixRowStride_ - matrixRows_) * sizeof(uint32_t));
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(columns), validRowBytes,
            (paddedRowBytes - alignedValidRowBytes) / BLOCK_BYTES,
            outputGapBytes, 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    TPipe *pipe_ = nullptr;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TQue<QuePosition::VECOUT, 2> outputQueue_;
    GlobalTensor<uint32_t> inputGm_;
    GlobalTensor<uint32_t> outputGm_;
    AscendC::tiling::ConfusionTransposeTiling transposeTiling_;
    uint64_t outputDims_[MAX_RANK] = {0};
    uint64_t sourceStrides_[MAX_RANK] = {0};
    uint64_t taskCount_ = 0;
    uint64_t outputMatrixRowStride_ = 0;
    uint32_t matrixRows_ = 0;
    uint32_t matrixColumns_ = 0;
    uint32_t paddedRows_ = 0;
    uint32_t paddedColumns_ = 0;
    uint32_t rank_ = 0;
    uint32_t matrixRowAxis_ = 0;
    uint32_t matrixColumnAxis_ = 0;
    uint32_t firstColumnAxis_ = 0;
};

class KernelTransposeTiled2dFp16 {
public:
    __aicore__ inline KernelTransposeTiled2dFp16() = default;

    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output, uint64_t inputRows,
        uint64_t inputColumns, uint64_t taskCount)
    {
        inputGm_.SetGlobalBuffer((__gm__ half *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ half *)output);
        inputRows_ = inputRows;
        inputColumns_ = inputColumns;
        taskCount_ = taskCount;
        tileColumns_ = inputColumns_ / MATRIX_TILE;
        inputRowGapBytes_ = static_cast<uint32_t>(
            (inputColumns_ - MATRIX_TILE) * sizeof(half));
        outputRowGapBytes_ = static_cast<uint32_t>(
            (inputRows_ - MATRIX_TILE) * sizeof(half));
        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, MATRIX_TILE_BYTES);
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, MATRIX_TILE_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (inputRows_ == 0 || inputColumns_ == 0 || tileColumns_ == 0 ||
            taskCount_ == 0) {
            return;
        }
        const uint64_t blockIndex = GetBlockIdx();
        const uint64_t blockCount = GetBlockNum();
        for (uint64_t task = blockIndex; task < taskCount_; task += blockCount) {
            const uint64_t tileRow = task / tileColumns_;
            const uint64_t tileColumn = task % tileColumns_;
            const uint64_t inputOffset =
                tileRow * MATRIX_TILE * inputColumns_ +
                tileColumn * MATRIX_TILE;
            const uint64_t outputOffset =
                tileColumn * MATRIX_TILE * inputRows_ +
                tileRow * MATRIX_TILE;
            CopyIn(inputOffset);
            Compute();
            CopyOut(outputOffset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t inputOffset)
    {
        LocalTensor<half> inputLocal = inputQueue_.AllocTensor<half>();
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(MATRIX_TILE), BLOCK_BYTES,
            inputRowGapBytes_, 0, 0};
        const DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGm_[inputOffset], copyParams, padParams);
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<half> inputLocal = inputQueue_.DeQue<half>();
        LocalTensor<half> outputLocal = outputQueue_.AllocTensor<half>();
        Transpose<half>(outputLocal, inputLocal);
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset)
    {
        LocalTensor<half> outputLocal = outputQueue_.DeQue<half>();
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(MATRIX_TILE), BLOCK_BYTES, 0,
            outputRowGapBytes_, 0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    uint64_t inputRows_ = 0;
    uint64_t inputColumns_ = 0;
    uint64_t tileColumns_ = 0;
    uint64_t taskCount_ = 0;
    uint32_t inputRowGapBytes_ = 0;
    uint32_t outputRowGapBytes_ = 0;
};

class KernelTransposeTiledMoveLastFp16 {
public:
    __aicore__ inline KernelTransposeTiledMoveLastFp16() = default;

    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output, uint64_t inputRows,
        uint64_t inputColumns, uint64_t taskCount)
    {
        inputGm_.SetGlobalBuffer((__gm__ half *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ half *)output);
        inputRows_ = inputRows;
        inputColumns_ = inputColumns;
        taskCount_ = taskCount;
        tileColumns_ =
            (inputColumns_ + MATRIX_TILE - 1) / MATRIX_TILE;
        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, MATRIX_TILE_BYTES);
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, MATRIX_TILE_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (inputRows_ == 0 || inputColumns_ == 0 || tileColumns_ == 0 ||
            taskCount_ == 0) {
            return;
        }
        const uint64_t blockIndex = GetBlockIdx();
        const uint64_t blockCount = GetBlockNum();
        for (uint64_t task = blockIndex; task < taskCount_; task += blockCount) {
            const uint64_t tileRow = task / tileColumns_;
            const uint64_t tileColumn = task % tileColumns_;
            const uint64_t rowStart = tileRow * MATRIX_TILE;
            const uint64_t columnStart = tileColumn * MATRIX_TILE;
            const uint32_t validRows = static_cast<uint32_t>(
                inputRows_ - rowStart < MATRIX_TILE
                    ? inputRows_ - rowStart
                    : MATRIX_TILE);
            const uint32_t validColumns = static_cast<uint32_t>(
                inputColumns_ - columnStart < MATRIX_TILE
                    ? inputColumns_ - columnStart
                    : MATRIX_TILE);
            const uint64_t inputOffset =
                rowStart * inputColumns_ + columnStart;
            const uint64_t outputOffset =
                columnStart * inputRows_ + rowStart;
            CopyIn(inputOffset, validRows, validColumns);
            Compute(validRows);
            CopyOut(outputOffset, validRows, validColumns);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t inputOffset,
                                  uint32_t validRows,
                                  uint32_t validColumns)
    {
        LocalTensor<half> inputLocal = inputQueue_.AllocTensor<half>();
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(validRows),
            static_cast<uint32_t>(validColumns * sizeof(half)),
            static_cast<uint32_t>(
                (inputColumns_ - validColumns) * sizeof(half)),
            0, 0};
        const DataCopyPadExtParams<half> padParams{
            true, 0,
            static_cast<uint8_t>(MATRIX_TILE - validColumns),
            static_cast<half>(0)};
        DataCopyPad(inputLocal, inputGm_[inputOffset], copyParams, padParams);
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t validRows)
    {
        LocalTensor<half> inputLocal = inputQueue_.DeQue<half>();
        if (validRows < MATRIX_TILE) {
            Duplicate<half>(
                inputLocal[validRows * MATRIX_TILE], static_cast<half>(0),
                (MATRIX_TILE - validRows) * MATRIX_TILE);
        }
        LocalTensor<half> outputLocal = outputQueue_.AllocTensor<half>();
        Transpose<half>(outputLocal, inputLocal);
        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint64_t outputOffset,
                                   uint32_t validRows,
                                   uint32_t validColumns)
    {
        LocalTensor<half> outputLocal = outputQueue_.DeQue<half>();
        const DataCopyExtParams copyParams{
            static_cast<uint16_t>(validColumns),
            static_cast<uint32_t>(validRows * sizeof(half)), 0,
            static_cast<uint32_t>(
                (inputRows_ - validRows) * sizeof(half)),
            0};
        DataCopyPad(outputGm_[outputOffset], outputLocal, copyParams);
        outputQueue_.FreeTensor(outputLocal);
    }

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    GlobalTensor<half> inputGm_;
    GlobalTensor<half> outputGm_;
    uint64_t inputRows_ = 0;
    uint64_t inputColumns_ = 0;
    uint64_t tileColumns_ = 0;
    uint64_t taskCount_ = 0;
};

// Fold contiguous output axes into runs to replace element gathers with DMA.
class KernelTransposeContiguousRun {
public:
    __aicore__ inline KernelTransposeContiguousRun() = default;

    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output, uint64_t totalElements,
        uint64_t runElements, uint64_t runCount, uint64_t chunksPerUnit,
        uint64_t taskCount, uint32_t prefixRank, uint32_t tileElements,
        uint32_t elementSize, uint32_t rank, uint32_t runAxis,
        uint64_t outputRunStride, uint64_t matrixRows,
        const uint64_t *outputDims,
        const uint64_t *sourceStrides)
    {
        inputGm_.SetGlobalBuffer((__gm__ uint8_t *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ uint8_t *)output);
        totalElements_ = totalElements;
        runElements_ = runElements;
        runCount_ = runCount;
        chunksPerUnit_ = chunksPerUnit;
        taskCount_ = taskCount;
        prefixRank_ = prefixRank;
        tileElements_ = tileElements;
        elementSize_ = elementSize;
        rank_ = rank;
        runAxis_ = runAxis;
        outputRunStride_ = outputRunStride == 0 ? 1 : outputRunStride;
        matrixRows_ = matrixRows == 0 ? 1 : matrixRows;
        for (uint32_t axis = 0; axis < MAX_RANK; ++axis) {
            outputDims_[axis] = outputDims[axis];
            sourceStrides_[axis] = sourceStrides[axis];
        }
        uint32_t bufferBytes = tileElements_ * elementSize_;
        if (matrixRows_ > 1 && outputRunStride_ > 1) {
            const uint32_t rowBytes = static_cast<uint32_t>(
                runElements_ * elementSize_);
            const uint32_t paddedRow = (rowBytes + BLOCK_BYTES - 1) /
                BLOCK_BYTES * BLOCK_BYTES;
            const uint32_t matrixBytes = static_cast<uint32_t>(matrixRows_) * paddedRow;
            if (matrixBytes > bufferBytes) {
                bufferBytes = matrixBytes;
            }
        }
        pipe_.InitBuffer(copyQueue_, BUFFER_NUM, bufferBytes);
    }

    __aicore__ inline void Process()
    {
        if (totalElements_ == 0 || runElements_ == 0 || runCount_ == 0 ||
            chunksPerUnit_ == 0 || taskCount_ == 0) {
            return;
        }
        const uint64_t blockIndex = GetBlockIdx();
        const uint64_t blockCount = GetBlockNum();
        for (uint64_t task = blockIndex; task < taskCount_; task += blockCount) {
            if (matrixRows_ > 1 && outputRunStride_ > 1) {
                const uint64_t runIndex = task * matrixRows_;
                const uint64_t inputBaseElements = DecodeInputBase(runIndex);
                const uint64_t outputBaseElements = DecodeOutputBase(runIndex);
                CopyMatrix(inputBaseElements * elementSize_,
                           outputBaseElements * elementSize_);
                continue;
            }
            const uint64_t runIndex = task / chunksPerUnit_;
            const uint64_t chunkIndex = task % chunksPerUnit_;
            const uint64_t chunkElementOffset = chunkIndex * tileElements_;
            const uint64_t remaining = runElements_ - chunkElementOffset;
            const uint32_t count = static_cast<uint32_t>(
                remaining < tileElements_ ? remaining : tileElements_);
            const uint64_t inputBaseElements = DecodeInputBase(runIndex);
            const uint64_t outputBaseElements = DecodeOutputBase(runIndex);
            const uint64_t inputOffsetBytes =
                (inputBaseElements + chunkElementOffset) * elementSize_;
            const uint64_t outputOffsetBytes =
                (outputBaseElements + chunkElementOffset * outputRunStride_) *
                elementSize_;
            CopyTile(inputOffsetBytes, outputOffsetBytes, count);
        }
    }

private:
    __aicore__ inline bool IsRunAxis(uint32_t axis) const
    {
        if (outputRunStride_ > 1) {
            return axis == runAxis_;
        }
        return axis >= prefixRank_;
    }

    __aicore__ inline uint64_t DecodeInputBase(uint64_t runIndex) const
    {
        uint64_t remaining = runIndex;
        uint64_t inputBase = 0;
        const uint32_t limit = outputRunStride_ > 1 ? rank_ : prefixRank_;
        for (uint32_t reverse = limit; reverse > 0; --reverse) {
            const uint32_t axis = reverse - 1;
            if (IsRunAxis(axis)) {
                continue;
            }
            const uint64_t coordinate = remaining % outputDims_[axis];
            remaining /= outputDims_[axis];
            inputBase += coordinate * sourceStrides_[axis];
        }
        return inputBase;
    }

    __aicore__ inline uint64_t DecodeOutputBase(uint64_t runIndex) const
    {
        if (outputRunStride_ <= 1) {
            return runIndex * runElements_;
        }
        uint64_t remaining = runIndex;
        uint64_t outputBase = 0;
        uint64_t outputStride = 1;
        for (uint32_t reverse = rank_; reverse > 0; --reverse) {
            const uint32_t axis = reverse - 1;
            if (axis == runAxis_) {
                outputStride *= outputDims_[axis];
                continue;
            }
            const uint64_t coordinate = remaining % outputDims_[axis];
            remaining /= outputDims_[axis];
            outputBase += coordinate * outputStride;
            outputStride *= outputDims_[axis];
        }
        return outputBase;
    }

    __aicore__ inline void CopyMatrix(uint64_t inputOffset, uint64_t outputOffset)
    {
        LocalTensor<uint8_t> copyLocal = copyQueue_.AllocTensor<uint8_t>();
        const uint32_t rowBytes = static_cast<uint32_t>(runElements_ * elementSize_);
        const uint32_t paddedRow = (rowBytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
        const uint32_t ubGap = paddedRow - rowBytes;
        const DataCopyExtParams inputParams{
            static_cast<uint16_t>(matrixRows_), rowBytes, 0, ubGap, 0};
        const DataCopyPadExtParams<uint8_t> padParams{
            ubGap != 0, 0, static_cast<uint8_t>(ubGap), 0};
        DataCopyPad(copyLocal, inputGm_[inputOffset], inputParams, padParams);
        copyQueue_.EnQue(copyLocal);
        copyLocal = copyQueue_.DeQue<uint8_t>();
        const uint64_t outputRowElements = runElements_ * outputRunStride_;
        const uint32_t dstGap = static_cast<uint32_t>(
            (outputRowElements - runElements_) * elementSize_);
        const DataCopyExtParams outputParams{
            static_cast<uint16_t>(matrixRows_), rowBytes, ubGap, dstGap, 0};
        DataCopyPad(outputGm_[outputOffset], copyLocal, outputParams);
        copyQueue_.FreeTensor(copyLocal);
    }

    __aicore__ inline void CopyTile(uint64_t inputOffset, uint64_t outputOffset,
                                    uint32_t count)
    {
        LocalTensor<uint8_t> copyLocal = copyQueue_.AllocTensor<uint8_t>();
        if (outputRunStride_ <= 1) {
            const uint32_t length = count * elementSize_;
            const DataCopyExtParams copyParams{1, length, 0, 0, 0};
            const DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
            DataCopyPad(copyLocal, inputGm_[inputOffset], copyParams, padParams);
            copyQueue_.EnQue(copyLocal);
            copyLocal = copyQueue_.DeQue<uint8_t>();
            DataCopyPad(outputGm_[outputOffset], copyLocal, copyParams);
        } else {
            const DataCopyExtParams inputParams{
                static_cast<uint16_t>(count), elementSize_, 0, 0, 0};
            const DataCopyPadExtParams<uint8_t> padParams{
                true, 0, static_cast<uint8_t>(BLOCK_BYTES - elementSize_), 0};
            DataCopyPad(copyLocal, inputGm_[inputOffset], inputParams, padParams);
            copyQueue_.EnQue(copyLocal);
            copyLocal = copyQueue_.DeQue<uint8_t>();
            const uint32_t dstGap = static_cast<uint32_t>(
                (outputRunStride_ - 1) * elementSize_);
            const DataCopyExtParams outputParams{
                static_cast<uint16_t>(count), elementSize_, 0, dstGap, 0};
            DataCopyPad(outputGm_[outputOffset], copyLocal, outputParams);
        }
        copyQueue_.FreeTensor(copyLocal);
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue_;
    GlobalTensor<uint8_t> inputGm_;
    GlobalTensor<uint8_t> outputGm_;
    uint64_t outputDims_[MAX_RANK] = {0};
    uint64_t sourceStrides_[MAX_RANK] = {0};
    uint64_t totalElements_ = 0;
    uint64_t runElements_ = 0;
    uint64_t runCount_ = 0;
    uint64_t chunksPerUnit_ = 0;
    uint64_t taskCount_ = 0;
    uint32_t prefixRank_ = 0;
    uint32_t tileElements_ = 0;
    uint32_t elementSize_ = 0;
    uint32_t rank_ = 0;
    uint32_t runAxis_ = 0;
    uint64_t outputRunStride_ = 1;
    uint64_t matrixRows_ = 1;
};

// General permutations: gather short runs into aligned UB slots, then pack output.
// DataCopyPad handles partial blocks without reading or writing beyond tensor bounds.
class KernelTransposeStridedGather {
public:
    __aicore__ inline KernelTransposeStridedGather() = default;

    __aicore__ inline void Init(
        GM_ADDR inputs, GM_ADDR output, uint64_t runElements,
        uint64_t groupCount, uint64_t blocksPerGroup,
        uint64_t chunksPerUnit, uint64_t taskCount,
        uint64_t sourceStepBytes, uint32_t prefixRank,
        uint32_t gatherBlocks, uint32_t elementSize,
        const uint64_t *outputDims, const uint64_t *sourceStrides,
        bool packedByteMode, bool packedWordMode)
    {
        inputGm_.SetGlobalBuffer((__gm__ uint8_t *)inputs);
        outputGm_.SetGlobalBuffer((__gm__ uint8_t *)output);
        runElements_ = runElements;
        groupCount_ = groupCount;
        blocksPerGroup_ = blocksPerGroup;
        chunksPerUnit_ = chunksPerUnit;
        taskCount_ = taskCount;
        sourceStepBytes_ = sourceStepBytes;
        prefixRank_ = prefixRank;
        gatherAxis_ = prefixRank_ - 1;
        gatherBlocks_ = gatherBlocks;
        elementSize_ = elementSize;
        runBytes_ = static_cast<uint32_t>(runElements_ * elementSize_);
        localBlockBytes_ =
            (runBytes_ + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
        packedByteMode_ = packedByteMode;
        packedWordMode_ = packedWordMode;
        for (uint32_t axis = 0; axis < MAX_RANK; ++axis) {
            outputDims_[axis] = outputDims[axis];
            sourceStrides_[axis] = sourceStrides[axis];
        }
        if (packedByteMode_ || packedWordMode_) {
            pipe_.InitBuffer(
                packedInputBuffer_, gatherBlocks_ * BLOCK_BYTES);
            pipe_.InitBuffer(
                packedOutputBuffer_, gatherBlocks_ * runBytes_);
            pipe_.InitBuffer(
                indexBuffer_, gatherBlocks_ * sizeof(int32_t));
            pipe_.InitBuffer(
                offsetBuffer_, gatherBlocks_ * sizeof(uint32_t));
            if (packedByteMode_) {
                pipe_.InitBuffer(
                    gatheredBuffer_, gatherBlocks_ * sizeof(int16_t));
                pipe_.InitBuffer(
                    halfBuffer_, gatherBlocks_ * sizeof(half));
            }
            eventIdVToMte2_ = pipe_.FetchEventID(HardEvent::V_MTE2);
            eventIdMte2ToV_ = pipe_.FetchEventID(HardEvent::MTE2_V);
            eventIdVToMte3_ = pipe_.FetchEventID(HardEvent::V_MTE3);
            eventIdMte3ToV_ = pipe_.FetchEventID(HardEvent::MTE3_V);

            LocalTensor<int32_t> indices = indexBuffer_.Get<int32_t>();
            LocalTensor<int32_t> offsets = offsetBuffer_.Get<int32_t>();
            CreateVecIndex(indices, static_cast<int32_t>(0), gatherBlocks_);
            Muls(offsets, indices, static_cast<int32_t>(BLOCK_BYTES),
                 gatherBlocks_);
            PipeBarrier<PIPE_V>();
        } else {
            pipe_.InitBuffer(
                copyQueue_, BUFFER_NUM, gatherBlocks_ * localBlockBytes_);
        }
    }

    __aicore__ inline void Process()
    {
        if (runElements_ == 0 || groupCount_ == 0 || blocksPerGroup_ == 0 ||
            chunksPerUnit_ == 0 || taskCount_ == 0 || runBytes_ == 0) {
            return;
        }
        const uint64_t blockIndex = GetBlockIdx();
        const uint64_t blockCount = GetBlockNum();
        for (uint64_t task = blockIndex; task < taskCount_; task += blockCount) {
            const uint64_t groupIndex = task / chunksPerUnit_;
            const uint64_t chunkIndex = task % chunksPerUnit_;
            const uint64_t blockStart = chunkIndex * gatherBlocks_;
            const uint64_t remaining = blocksPerGroup_ - blockStart;
            const uint32_t blocks = static_cast<uint32_t>(
                remaining < gatherBlocks_ ? remaining : gatherBlocks_);
            const uint64_t inputBaseElements = DecodeGroupInputBase(groupIndex);
            const uint64_t inputOffsetBytes = inputBaseElements * elementSize_ +
                blockStart * sourceStepBytes_;
            const uint64_t outputOffsetBytes =
                (groupIndex * blocksPerGroup_ + blockStart) * runBytes_;
            GatherBlocks(inputOffsetBytes, outputOffsetBytes, blocks);
        }
    }

    __aicore__ inline void ProcessPackedWordFlatGather()
    {
        if (runElements_ == 0 || groupCount_ == 0 || blocksPerGroup_ == 0 ||
            taskCount_ == 0 || runBytes_ == 0) {
            return;
        }
        const uint64_t groupsPerTask = gatherBlocks_ / blocksPerGroup_;
        const uint32_t groupBlocks =
            static_cast<uint32_t>(blocksPerGroup_);
        const uint64_t blockIndex = GetBlockIdx();
        const uint64_t blockCount = GetBlockNum();
        for (uint64_t task = blockIndex; task < taskCount_; task += blockCount) {
            const uint64_t groupStart = task * groupsPerTask;
            const uint64_t remainingGroups = groupCount_ - groupStart;
            const uint32_t groups = static_cast<uint32_t>(
                remainingGroups < groupsPerTask
                    ? remainingGroups
                    : groupsPerTask);
            const uint32_t blocks = groups * groupBlocks;
            const uint64_t outputOffsetBytes =
                groupStart * blocksPerGroup_ * runBytes_;
            GatherPackedWordGroups(
                groupStart, groups, outputOffsetBytes, blocks);
        }
    }

private:
    __aicore__ inline uint64_t DecodeGroupInputBase(uint64_t groupIndex) const
    {
        uint64_t remaining = groupIndex;
        uint64_t inputBase = 0;
        for (uint32_t reverse = gatherAxis_; reverse > 0; --reverse) {
            const uint32_t axis = reverse - 1;
            const uint64_t coordinate = remaining % outputDims_[axis];
            remaining /= outputDims_[axis];
            inputBase += coordinate * sourceStrides_[axis];
        }
        return inputBase;
    }

    __aicore__ inline void GatherBlocks(uint64_t inputOffset,
                                        uint64_t outputOffset,
                                        uint32_t blocks)
    {
        if (packedByteMode_) {
            GatherPackedByteBlocks(inputOffset, outputOffset, blocks);
            return;
        }
        if (packedWordMode_) {
            GatherPackedWordBlocks(inputOffset, outputOffset, blocks);
            return;
        }

        LocalTensor<uint8_t> copyLocal = copyQueue_.AllocTensor<uint8_t>();
        const DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        const uint64_t sourceGap = sourceStepBytes_ - runBytes_;
        if (sourceGap <= MAX_UINT32_VALUE) {
            const DataCopyExtParams inputParams{
                static_cast<uint16_t>(blocks), runBytes_,
                static_cast<uint32_t>(sourceGap), 0, 0};
            DataCopyPad(copyLocal, inputGm_[inputOffset], inputParams, padParams);
        } else {
            const DataCopyExtParams inputParams{1, runBytes_, 0, 0, 0};
            for (uint32_t block = 0; block < blocks; ++block) {
                DataCopyPad(copyLocal[block * localBlockBytes_],
                            inputGm_[inputOffset + block * sourceStepBytes_],
                            inputParams, padParams);
            }
        }
        copyQueue_.EnQue(copyLocal);
        copyLocal = copyQueue_.DeQue<uint8_t>();
        if (runBytes_ == 1 && blocks > 1) {
            for (uint32_t block = 1; block < blocks; ++block) {
                copyLocal.SetValue(
                    block, copyLocal.GetValue(block * localBlockBytes_));
            }
            const DataCopyExtParams outputParams{1, blocks, 0, 0, 0};
            DataCopyPad(outputGm_[outputOffset], copyLocal, outputParams);
        } else {
            const DataCopyExtParams outputParams{
                static_cast<uint16_t>(blocks), runBytes_, 0, 0, 0};
            DataCopyPad(outputGm_[outputOffset], copyLocal, outputParams);
        }
        copyQueue_.FreeTensor(copyLocal);
    }

    __aicore__ inline void GatherPackedByteBlocks(uint64_t inputOffset,
                                                  uint64_t outputOffset,
                                                  uint32_t blocks)
    {
        LocalTensor<uint8_t> staged = packedInputBuffer_.Get<uint8_t>();
        const DataCopyPadExtParams<uint8_t> padParams{
            true, 0, static_cast<uint8_t>(BLOCK_BYTES - 1), 0};
        const uint64_t sourceGap = sourceStepBytes_ - 1;

        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        if (sourceGap <= MAX_UINT32_VALUE) {
            const DataCopyExtParams inputParams{
                static_cast<uint16_t>(blocks), 1,
                static_cast<uint32_t>(sourceGap), 0, 0};
            DataCopyPad(staged, inputGm_[inputOffset], inputParams, padParams);
        } else {
            const DataCopyExtParams inputParams{1, 1, 0, 0, 0};
            for (uint32_t block = 0; block < blocks; ++block) {
                DataCopyPad(staged[block * BLOCK_BYTES],
                            inputGm_[inputOffset + block * sourceStepBytes_],
                            inputParams, padParams);
            }
        }
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);

        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<uint32_t>();
        LocalTensor<int16_t> gathered = gatheredBuffer_.Get<int16_t>();
        LocalTensor<half> halfLocal = halfBuffer_.Get<half>();
        LocalTensor<uint8_t> packed = packedOutputBuffer_.Get<uint8_t>();
        Gather(gathered, staged.ReinterpretCast<int16_t>(), offsets, 0, blocks);
        PipeBarrier<PIPE_V>();
        Cast(halfLocal, gathered, RoundMode::CAST_NONE, blocks);
        PipeBarrier<PIPE_V>();
        Cast(packed, halfLocal, RoundMode::CAST_NONE, blocks);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        const DataCopyExtParams outputParams{1, blocks, 0, 0, 0};
        DataCopyPad(outputGm_[outputOffset], packed, outputParams);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
    }

    __aicore__ inline void GatherPackedWordGroups(
        uint64_t groupStart, uint32_t groups, uint64_t outputOffset,
        uint32_t blocks)
    {
        LocalTensor<uint8_t> staged = packedInputBuffer_.Get<uint8_t>();
        const DataCopyPadExtParams<uint8_t> padParams{
            true, 0, static_cast<uint8_t>(BLOCK_BYTES - runBytes_), 0};
        const uint64_t sourceGap = sourceStepBytes_ - runBytes_;
        const uint32_t groupBlocks =
            static_cast<uint32_t>(blocksPerGroup_);
        const DataCopyExtParams inputParams{
            static_cast<uint16_t>(groupBlocks), runBytes_,
            static_cast<uint32_t>(sourceGap), 0, 0};

        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        for (uint32_t localGroup = 0; localGroup < groups; ++localGroup) {
            const uint64_t groupIndex = groupStart + localGroup;
            const uint64_t inputOffsetBytes =
                DecodeGroupInputBase(groupIndex) * elementSize_;
            const uint32_t stagedBlockStart = localGroup * groupBlocks;
            DataCopyPad(staged[stagedBlockStart * BLOCK_BYTES],
                        inputGm_[inputOffsetBytes], inputParams, padParams);
        }
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);

        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<uint32_t>();
        LocalTensor<uint8_t> packed = packedOutputBuffer_.Get<uint8_t>();
        if (runBytes_ == sizeof(uint16_t)) {
            Gather(packed.ReinterpretCast<uint16_t>(),
                   staged.ReinterpretCast<uint16_t>(), offsets, 0, blocks);
        } else {
            Gather(packed.ReinterpretCast<uint32_t>(),
                   staged.ReinterpretCast<uint32_t>(), offsets, 0, blocks);
        }
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        const DataCopyExtParams outputParams{
            1, static_cast<uint32_t>(blocks * runBytes_), 0, 0, 0};
        DataCopyPad(outputGm_[outputOffset], packed, outputParams);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
    }

    __aicore__ inline void GatherPackedWordBlocks(uint64_t inputOffset,
                                                  uint64_t outputOffset,
                                                  uint32_t blocks)
    {
        LocalTensor<uint8_t> staged = packedInputBuffer_.Get<uint8_t>();
        const DataCopyPadExtParams<uint8_t> padParams{
            true, 0, static_cast<uint8_t>(BLOCK_BYTES - runBytes_), 0};
        const uint64_t sourceGap = sourceStepBytes_ - runBytes_;

        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2_);
        if (sourceGap <= MAX_UINT32_VALUE) {
            const DataCopyExtParams inputParams{
                static_cast<uint16_t>(blocks), runBytes_,
                static_cast<uint32_t>(sourceGap), 0, 0};
            DataCopyPad(staged, inputGm_[inputOffset], inputParams, padParams);
        } else {
            const DataCopyExtParams inputParams{1, runBytes_, 0, 0, 0};
            for (uint32_t block = 0; block < blocks; ++block) {
                DataCopyPad(staged[block * BLOCK_BYTES],
                            inputGm_[inputOffset + block * sourceStepBytes_],
                            inputParams, padParams);
            }
        }
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV_);

        LocalTensor<uint32_t> offsets = offsetBuffer_.Get<uint32_t>();
        LocalTensor<uint8_t> packed = packedOutputBuffer_.Get<uint8_t>();
        if (runBytes_ == sizeof(uint16_t)) {
            Gather(packed.ReinterpretCast<uint16_t>(),
                   staged.ReinterpretCast<uint16_t>(), offsets, 0, blocks);
        } else {
            Gather(packed.ReinterpretCast<uint32_t>(),
                   staged.ReinterpretCast<uint32_t>(), offsets, 0, blocks);
        }
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3_);
        const DataCopyExtParams outputParams{
            1, blocks * runBytes_, 0, 0, 0};
        DataCopyPad(outputGm_[outputOffset], packed, outputParams);
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV_);
    }

    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> copyQueue_;
    TBuf<TPosition::VECCALC> packedInputBuffer_;
    TBuf<TPosition::VECCALC> packedOutputBuffer_;
    TBuf<TPosition::VECCALC> indexBuffer_;
    TBuf<TPosition::VECCALC> offsetBuffer_;
    TBuf<TPosition::VECCALC> gatheredBuffer_;
    TBuf<TPosition::VECCALC> halfBuffer_;
    TEventID eventIdVToMte2_;
    TEventID eventIdMte2ToV_;
    TEventID eventIdVToMte3_;
    TEventID eventIdMte3ToV_;
    GlobalTensor<uint8_t> inputGm_;
    GlobalTensor<uint8_t> outputGm_;
    uint64_t outputDims_[MAX_RANK] = {0};
    uint64_t sourceStrides_[MAX_RANK] = {0};
    uint64_t runElements_ = 0;
    uint64_t groupCount_ = 0;
    uint64_t blocksPerGroup_ = 0;
    uint64_t chunksPerUnit_ = 0;
    uint64_t taskCount_ = 0;
    uint64_t sourceStepBytes_ = 0;
    uint32_t prefixRank_ = 0;
    uint32_t gatherAxis_ = 0;
    uint32_t gatherBlocks_ = 0;
    uint32_t elementSize_ = 0;
    uint32_t runBytes_ = 0;
    uint32_t localBlockBytes_ = 0;
    bool packedByteMode_ = false;
    bool packedWordMode_ = false;
};

extern "C" __global__ __aicore__ void transpose(
    GM_ADDR inputs, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;
    REGISTER_TILING_DEFAULT(TransposeKernelTilingData);
    REGISTER_TILING_FOR_TILINGKEY(
        "(TILING_KEY_VAR == 100) || (TILING_KEY_VAR == 200) || (TILING_KEY_VAR == 220) || (TILING_KEY_VAR == 225) || (TILING_KEY_VAR == 230) || (TILING_KEY_VAR == 240) || (TILING_KEY_VAR == 250) || (TILING_KEY_VAR == 260) || (TILING_KEY_VAR == 900) || (TILING_KEY_VAR == 910) || (TILING_KEY_VAR == 920) || (TILING_KEY_VAR == 921) || (TILING_KEY_VAR == 930)",
        TransposeKernelTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        TransposeKernelTilingData, tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(930)) {
        TPipe pipe;
        KernelTransposeBlockSwapFp16 op;
        op.Init(inputs, output, tilingData.matrixRows, tilingData.batchCount,
                tilingData.matrixColumns, static_cast<uint32_t>(tilingData.runElements),
                tilingData.taskCount, &pipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(225)) {
        TPipe pipe;
        KernelTransposeSlab2dByte op;
        op.Init(inputs, output, tilingData.matrixRows,
                static_cast<uint32_t>(tilingData.matrixColumns),
                tilingData.matrixColumnsPerTask, tilingData.taskCount,
                tilingData.transposeTiling, &pipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(220)) {
        TPipe pipe;
        KernelTransposeSlab2dFp16 op;
        op.Init(inputs, output, tilingData.outputDims[1],
                tilingData.outputDims[0], tilingData.taskCount,
                tilingData.matrixRowsPerTask,
                tilingData.transposeTiling, &pipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(230)) {
        TPipe pipe;
        KernelTransposeSlab2dWord32 op;
        op.Init(inputs, output, tilingData.outputDims[1],
                tilingData.outputDims[0], tilingData.taskCount,
                tilingData.matrixRowsPerTask,
                tilingData.matrixColumnsPerTask,
                tilingData.transposeTiling, &pipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(256)) {
        KernelTransposeFullRowVnchwWord32 op;
        op.Init(inputs, output, tilingData.matrixRows, tilingData.matrixColumns,
                tilingData.taskCount, tilingData.matrixPrefixRank,
                tilingData.outputDims, tilingData.sourceStrides);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(251)) {
        KernelTransposePackedGatherWord32 op;
        op.Init(inputs, output, tilingData.matrixRows, tilingData.matrixColumns,
                tilingData.taskCount, tilingData.matrixPrefixRank,
                tilingData.outputDims, tilingData.sourceStrides);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(250)) {
        TPipe pipe;
        KernelTransposeBatchedSlabWord32 op;
        op.Init(inputs, output, tilingData.batchCount,
                tilingData.matrixRows, tilingData.matrixColumns,
                tilingData.taskCount, tilingData.matrixRowsPerTask,
                tilingData.matrixColumnsPerTask,
                tilingData.matrixPrefixRank, tilingData.outputDims,
                tilingData.sourceStrides,
                tilingData.transposeTiling,
                tilingData.tailTransposeTiling, &pipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(260)) {
        TPipe pipe;
        KernelTransposeStridedTileWord32 op;
        op.Init(inputs, output, tilingData.taskCount,
                tilingData.matrixRows, tilingData.matrixColumns,
                tilingData.matrixRowsPerTask,
                tilingData.matrixColumnsPerTask, tilingData.rank,
                tilingData.matrixRowAxis, tilingData.matrixColumnAxis,
                static_cast<uint32_t>(tilingData.matrixPrefixRank),
                tilingData.outputMatrixRowStride, tilingData.outputDims,
                tilingData.sourceStrides, tilingData.transposeTiling, &pipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(200)) {
        KernelTransposeTiled2dFp16 op;
        op.Init(inputs, output, tilingData.outputDims[1],
                tilingData.outputDims[0], tilingData.taskCount);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(240)) {
        KernelTransposeTiledMoveLastFp16 op;
        op.Init(inputs, output,
                tilingData.totalElements / tilingData.outputDims[0],
                tilingData.outputDims[0], tilingData.taskCount);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(100)) {
        KernelTransposeContiguousRun op;
        op.Init(inputs, output, tilingData.totalElements,
                tilingData.runElements, tilingData.runCount,
                tilingData.chunksPerUnit, tilingData.taskCount,
                tilingData.prefixRank, tilingData.tileElements,
                tilingData.elementSize, tilingData.rank, tilingData.runAxis,
                tilingData.outputRunStride, tilingData.matrixRows,
                tilingData.outputDims,
                tilingData.sourceStrides);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(900)) {
        KernelTransposeStridedGather op;
        op.Init(inputs, output, tilingData.runElements,
                tilingData.groupCount, tilingData.blocksPerGroup,
                tilingData.chunksPerUnit, tilingData.taskCount,
                tilingData.sourceStepBytes, tilingData.prefixRank,
                tilingData.gatherBlocks, tilingData.elementSize,
                tilingData.outputDims, tilingData.sourceStrides, false, false);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(910)) {
        KernelTransposeStridedGather op;
        op.Init(inputs, output, tilingData.runElements,
                tilingData.groupCount, tilingData.blocksPerGroup,
                tilingData.chunksPerUnit, tilingData.taskCount,
                tilingData.sourceStepBytes, tilingData.prefixRank,
                tilingData.gatherBlocks, tilingData.elementSize,
                tilingData.outputDims, tilingData.sourceStrides, true, false);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(921)) {
        KernelTransposeStridedGather op;
        op.Init(inputs, output, tilingData.runElements,
                tilingData.groupCount, tilingData.blocksPerGroup,
                tilingData.chunksPerUnit, tilingData.taskCount,
                tilingData.sourceStepBytes, tilingData.prefixRank,
                tilingData.gatherBlocks, tilingData.elementSize,
                tilingData.outputDims, tilingData.sourceStrides, false, true);
        op.ProcessPackedWordFlatGather();
        return;
    }

    if (TILING_KEY_IS(920)) {
        KernelTransposeStridedGather op;
        op.Init(inputs, output, tilingData.runElements,
                tilingData.groupCount, tilingData.blocksPerGroup,
                tilingData.chunksPerUnit, tilingData.taskCount,
                tilingData.sourceStepBytes, tilingData.prefixRank,
                tilingData.gatherBlocks, tilingData.elementSize,
                tilingData.outputDims, tilingData.sourceStrides, false, true);
        op.Process();
    }
}
