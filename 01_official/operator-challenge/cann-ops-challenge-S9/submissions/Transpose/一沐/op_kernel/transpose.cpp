#include "transpose_tiling.h"

#include "kernel_operator.h"
#include "transpose_pipeline.h"
#include "transpose_copy.h"

namespace {
constexpr uint32_t kContiguousBufferBytes = 32 * 1024;
constexpr uint32_t kStridedElements = 2048;
constexpr uint32_t kScratchBytes = kStridedElements * 32;
// Three 64 KB regions fill the 192 KB UB. Int8 intermediates occupy the
// upper 32 KB of input/output; at most 16384 values are gathered per tile.
constexpr uint32_t kCompactBytes = 64 * 1024;
constexpr uint32_t kIndexBytes = 16384 * sizeof(uint32_t);
}

template <typename T>
class KernelTranspose {
public:
    __aicore__ inline KernelTranspose() = default;

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR tiling)
    {
        tiling_ = reinterpret_cast<__gm__ const CompetitionTransposeTilingData *>(tiling);
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(input),
                              tiling_->totalElements);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output),
                               tiling_->totalElements);
        pipe_.InitBuffer(inputBuffer_, kScratchBytes);
        pipe_.InitBuffer(outputBuffer_, kCompactBytes);
        pipe_.InitBuffer(indexBuffer_, kIndexBytes);
    }

    __aicore__ inline void Process()
    {
        if (tiling_->strategy == 0) {
            ProcessContiguous();
        } else if (tiling_->strategy == 1) {
            ProcessLastTwoSwap();
        } else if (tiling_->strategy == 5) {
            ProcessBlockCopies();
        } else if (tiling_->strategy == 4) {
            ProcessTiles();
        } else if (tiling_->strategy == 2) {
            ProcessStridedRows();
        } else {
            ProcessScalar();
        }
    }

private:
    __aicore__ inline void ProcessBlockCopies()
    {
        const uint32_t rank = tiling_->rank, fast = tiling_->inputDims[0];
        const uint64_t group = tiling_->contiguousElements;
        const uint64_t cols = tiling_->outputDims[fast];
        const uint64_t stride = tiling_->inputDims[1];
        const uint32_t side = tiling_->tileSide;
        const uint64_t chunks = (cols + side - 1) / side;
        auto local = inputBuffer_.Get<T>();
        const AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        for (uint64_t task = AscendC::GetBlockIdx(); task < tiling_->taskCount;
             task += AscendC::GetBlockNum()) {
            uint64_t batch = task / chunks;
            const uint64_t c = task % chunks * side;
            uint64_t ib = 0, ob = 0, os = group;
            for (uint32_t rev = 1; rev < rank; ++rev) {
                const uint32_t a = rank - 1 - rev;
                const uint64_t dim = tiling_->outputDims[a];
                if (a != fast) {
                    const uint64_t x = batch % dim;
                    batch /= dim;
                    ib += x * tiling_->inputStrides[a];
                    ob += x * os;
                }
                os *= dim;
            }
            const uint16_t n = Min(side, cols - c);
            const AscendC::DataCopyExtParams ip{n, static_cast<uint32_t>(group * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(local, input_[ib + c * group], ip, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            const AscendC::DataCopyExtParams op{n, static_cast<uint32_t>(group * sizeof(T)),
                0, static_cast<uint32_t>((stride - group) * sizeof(T)), 0};
            AscendC::DataCopyPad(output_[ob + c * stride], local, op);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    // Tile the input-fast and output-fast axes, with every other axis as a batch.
    // This works for arbitrary permutations, not only swaps of the last two axes.
    __aicore__ inline void ProcessTiles()
    {
        const uint32_t side = tiling_->tileSide;
        const uint32_t width = tiling_->inputDims[4] ? tiling_->inputDims[4] : side;
        const uint32_t group = tiling_->inputDims[2];
        const uint32_t rowAxis = tiling_->inputDims[3];
        const uint32_t pitch = ((width * group * sizeof(T) + 31) / 32) * (32 / sizeof(T));
        const uint32_t outPitch = ((side * group * sizeof(T) + 31) / 32) * (32 / sizeof(T));
        const uint32_t count = width * outPitch;
        const uint32_t inputCount = side * pitch;
        constexpr uint32_t bytes = sizeof(T) == 1 ? 2 : sizeof(T);
        const bool hardware = sizeof(T) <= 2 && group == 1;
        auto indices = indexBuffer_.Get<uint32_t>();
        if (!hardware) {
        for (uint32_t r = 0; r < outPitch; ++r)
            indices.SetValue(r, r < side * group ? (r / group * pitch + r % group) * bytes : 0);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (uint32_t c = 1; c < width; ++c) {
            AscendC::Adds(indices[c * outPitch].template ReinterpretCast<int32_t>(),
                indices.template ReinterpretCast<int32_t>(), static_cast<int32_t>(c * group * bytes), outPitch);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
        }
        const uint32_t rank = tiling_->rank;
        const uint32_t fast = tiling_->inputDims[0];
        const uint64_t rows = tiling_->outputDims[rowAxis];
        const uint64_t cols = tiling_->outputDims[fast];
        const uint64_t inStride = tiling_->inputStrides[rowAxis];
        const uint64_t outStride = tiling_->inputDims[1];
        const uint64_t rt = (rows + side - 1) / side;
        const uint64_t ct = (cols + width - 1) / width;
        auto src = inputBuffer_.Get<T>();
        auto dst = outputBuffer_.Get<T>();
        uint64_t srcAddresses[4][16], dstAddresses[4][16];
        if (hardware) {
            auto hsrc = inputBuffer_.Get<uint16_t>();
            auto hdst = outputBuffer_.Get<uint16_t>();
            if constexpr (sizeof(T) == 1) { hsrc = hsrc[16384]; hdst = hdst[16384]; }
            for (uint32_t rb = 0; rb < 4; ++rb) {
                for (uint32_t a = 0; a < 16; ++a) {
                    srcAddresses[rb][a] = reinterpret_cast<uint64_t>(hsrc[(rb * 16 + a) * 64].GetPhyAddr());
                    dstAddresses[rb][a] = reinterpret_cast<uint64_t>(hdst[a * 64 + rb * 16].GetPhyAddr());
                }
            }
        }
        const AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        for (uint64_t task = AscendC::GetBlockIdx(); task < tiling_->taskCount;
             task += AscendC::GetBlockNum()) {
            uint64_t batch = task / (rt * ct);
            const uint64_t tile = task % (rt * ct);
            const uint64_t r = tile / ct * side;
            const uint64_t c = tile % ct * width;
            uint64_t ib = 0, ob = 0, os = 1;
            for (uint32_t rev = 0; rev < rank; ++rev) {
                const uint32_t a = rank - 1 - rev;
                const uint64_t dim = tiling_->outputDims[a];
                if (a != fast && a != rowAxis && a <= rowAxis) {
                    const uint64_t x = batch % dim;
                    batch /= dim;
                    ib += x * tiling_->inputStrides[a];
                    ob += x * os;
                }
                os *= dim;
            }
            const uint32_t nr = Min(side, rows - r), nc = Min(width, cols - c);
            const uint32_t inPadBlocks = (pitch * sizeof(T) - ((nc * group * sizeof(T) + 31) / 32) * 32) / 32;
            const AscendC::DataCopyExtParams ip{static_cast<uint16_t>(nr),
                static_cast<uint32_t>(nc * group * sizeof(T)), static_cast<uint32_t>((inStride - nc * group) * sizeof(T)), inPadBlocks, 0};
            AscendC::DataCopyPad(src, input_[ib + r * inStride + c * group], ip, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            if constexpr (sizeof(T) == 1) {
                auto halfIn = inputBuffer_.Get<half>()[16384];
                auto halfOut = outputBuffer_.Get<half>()[16384];
                AscendC::Cast(halfIn, src, AscendC::RoundMode::CAST_NONE, inputCount);
                AscendC::PipeBarrier<PIPE_V>();
                if (hardware) {
                    const AscendC::TransDataTo5HDParams tp{false, false, 4, 64, 1};
                    for (uint32_t rb = 0; rb < 4; ++rb)
                        AscendC::TransDataTo5HD<uint16_t>(dstAddresses[rb], srcAddresses[rb], tp);
                } else AscendC::Gather(halfOut, halfIn, indices, 0, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(dst, halfOut, AscendC::RoundMode::CAST_NONE, count);
            } else if constexpr (sizeof(T) == 2) {
                if (hardware) {
                    const AscendC::TransDataTo5HDParams tp{false, false, 4, 64, 1};
                    for (uint32_t rb = 0; rb < 4; ++rb)
                        AscendC::TransDataTo5HD<uint16_t>(dstAddresses[rb], srcAddresses[rb], tp);
                } else AscendC::Gather(dst.template ReinterpretCast<uint16_t>(),
                    src.template ReinterpretCast<uint16_t>(), indices, 0, count);
            } else {
                AscendC::Gather(dst.template ReinterpretCast<uint32_t>(),
                    src.template ReinterpretCast<uint32_t>(), indices, 0, count);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            const uint32_t outPadBlocks = (outPitch * sizeof(T) - ((nr * group * sizeof(T) + 31) / 32) * 32) / 32;
            const AscendC::DataCopyExtParams op{static_cast<uint16_t>(nc),
                static_cast<uint32_t>(nr * group * sizeof(T)), outPadBlocks, static_cast<uint32_t>((outStride - nr * group) * sizeof(T)), 0};
            AscendC::DataCopyPad(output_[ob + c * outStride + r * group], dst, op);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline uint64_t InputOffset(uint64_t outputIndex) const
    {
        uint64_t remaining = outputIndex;
        uint64_t inputIndex = 0;
        for (uint32_t reverse = 0; reverse < tiling_->rank; ++reverse) {
            const uint32_t axis = tiling_->rank - 1 - reverse;
            const uint64_t extent = tiling_->outputDims[axis];
            const uint64_t coordinate = remaining % extent;
            remaining /= extent;
            inputIndex += coordinate * tiling_->inputStrides[axis];
        }
        return inputIndex;
    }

    __aicore__ inline void CopyExact(uint64_t inputOffset,
                                     uint64_t outputOffset,
                                     uint32_t elements)
    {
        AscendC::LocalTensor<T> local = inputBuffer_.Get<T>();
        const AscendC::DataCopyExtParams params{
            1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
        const AscendC::DataCopyPadExtParams<T> pad{
            false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(local, input_[inputOffset], params, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyPad(output_[outputOffset], local, params);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ProcessContiguous()
    {
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t rowElements = tiling_->contiguousElements;
        const uint64_t chunks = tiling_->chunksPerRow;
        constexpr uint32_t maxElements = kContiguousBufferBytes / sizeof(T);
        for (uint64_t task = block; task < tiling_->taskCount; task += blocks) {
            const uint64_t row = task / chunks;
            const uint64_t chunk = task - row * chunks;
            const uint64_t first = chunk * maxElements;
            const uint32_t count = static_cast<uint32_t>(
                Min(maxElements, rowElements - first));
            const uint64_t outputOffset = row * rowElements + first;
            const uint64_t inputOffset = InputOffset(row * rowElements) + first;
            CopyExact(inputOffset, outputOffset, count);
        }
    }

    __aicore__ inline void PrepareTransposeIndices(uint32_t side)
    {
        AscendC::LocalTensor<uint32_t> indices = indexBuffer_.Get<uint32_t>();
        for (uint32_t outputRow = 0; outputRow < side; ++outputRow) {
            for (uint32_t outputColumn = 0; outputColumn < side; ++outputColumn) {
                indices.SetValue(outputRow * side + outputColumn,
                    (outputColumn * side + outputRow) * sizeof(T));
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void TransposeLocal(AscendC::LocalTensor<T> outputLocal,
                                          AscendC::LocalTensor<T> inputLocal,
                                          uint32_t side)
    {
        if constexpr (sizeof(T) == 2) {
            AscendC::Transpose(outputLocal.template ReinterpretCast<uint16_t>(),
                               inputLocal.template ReinterpretCast<uint16_t>());
        } else if constexpr (sizeof(T) == 4) {
            AscendC::LocalTensor<uint32_t> indices = indexBuffer_.Get<uint32_t>();
            AscendC::Gather(outputLocal.template ReinterpretCast<uint32_t>(),
                            inputLocal.template ReinterpretCast<uint32_t>(),
                            indices, 0, side * side);
        } else {
            for (uint32_t outputRow = 0; outputRow < side; ++outputRow) {
                for (uint32_t outputColumn = 0; outputColumn < side; ++outputColumn) {
                    outputLocal.SetValue(outputRow * side + outputColumn,
                        inputLocal.GetValue(outputColumn * side + outputRow));
                }
            }
        }
    }

    __aicore__ inline void ProcessLastTwoSwap()
    {
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint32_t rank = tiling_->rank;
        const uint32_t side = tiling_->tileSide;
        const uint64_t rows = tiling_->inputDims[rank - 2];
        const uint64_t columns = tiling_->inputDims[rank - 1];
        const uint64_t rowTiles = (rows + side - 1) / side;
        const uint64_t columnTiles = (columns + side - 1) / side;
        const uint64_t tilesPerMatrix = rowTiles * columnTiles;
        if constexpr (sizeof(T) == 4) {
            PrepareTransposeIndices(side);
        }
        AscendC::LocalTensor<T> inputLocal = inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> outputLocal = outputBuffer_.Get<T>();
        const AscendC::DataCopyPadExtParams<T> pad{
            false, 0, 0, static_cast<T>(0)};

        for (uint64_t task = block; task < tiling_->taskCount; task += blocks) {
            const uint64_t matrix = task / tilesPerMatrix;
            const uint64_t tile = task - matrix * tilesPerMatrix;
            const uint64_t rowTile = tile / columnTiles;
            const uint64_t columnTile = tile - rowTile * columnTiles;
            const uint64_t rowStart = rowTile * side;
            const uint64_t columnStart = columnTile * side;
            const uint32_t tileRows = static_cast<uint32_t>(Min(side, rows - rowStart));
            const uint32_t tileColumns = static_cast<uint32_t>(Min(side, columns - columnStart));
            const uint64_t matrixInput = matrix * rows * columns;
            const uint64_t matrixOutput = matrixInput;
            const uint64_t inputOffset = matrixInput + rowStart * columns + columnStart;
            const AscendC::DataCopyExtParams inputParams{
                static_cast<uint16_t>(tileRows),
                static_cast<uint32_t>(tileColumns * sizeof(T)),
                static_cast<uint32_t>((columns - tileColumns) * sizeof(T)),
                0, 0};
            AscendC::DataCopyPad(inputLocal, input_[inputOffset], inputParams, pad);
            AscendC::PipeBarrier<PIPE_ALL>();
            TransposeLocal(outputLocal, inputLocal, side);
            AscendC::PipeBarrier<PIPE_ALL>();

            const uint64_t outputOffset = matrixOutput + columnStart * rows + rowStart;
            const AscendC::DataCopyExtParams outputParams{
                static_cast<uint16_t>(tileColumns),
                static_cast<uint32_t>(tileRows * sizeof(T)),
                0,
                static_cast<uint32_t>((rows - tileRows) * sizeof(T)),
                0};
            AscendC::DataCopyPad(output_[outputOffset], outputLocal, outputParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void PrepareStridedIndices()
    {
        if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
            AscendC::LocalTensor<uint32_t> indices = indexBuffer_.Get<uint32_t>();
            for (uint32_t index = 0; index < kStridedElements; ++index) {
                indices.SetValue(index, index * 32);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CompactStrided(AscendC::LocalTensor<T> outputLocal,
                                          AscendC::LocalTensor<T> scratch,
                                          uint32_t count)
    {
        if constexpr (sizeof(T) == 2) {
            AscendC::Gather(outputLocal.template ReinterpretCast<uint16_t>(),
                            scratch.template ReinterpretCast<uint16_t>(),
                            indexBuffer_.Get<uint32_t>(), 0, count);
        } else if constexpr (sizeof(T) == 4) {
            AscendC::Gather(outputLocal.template ReinterpretCast<uint32_t>(),
                            scratch.template ReinterpretCast<uint32_t>(),
                            indexBuffer_.Get<uint32_t>(), 0, count);
        } else {
            for (uint32_t index = 0; index < count; ++index) {
                outputLocal.SetValue(index, scratch.GetValue(index * 32));
            }
        }
    }

    __aicore__ inline void ProcessStridedRows()
    {
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t blocks = AscendC::GetBlockNum();
        const uint64_t rowElements = tiling_->outputDims[tiling_->rank - 1];
        const uint64_t step = tiling_->inputStrides[tiling_->rank - 1];
        PrepareStridedIndices();
        AscendC::LocalTensor<T> scratch = inputBuffer_.Get<T>();
        AscendC::LocalTensor<T> compact = outputBuffer_.Get<T>();
        const AscendC::DataCopyPadExtParams<T> pad{
            false, 0, 0, static_cast<T>(0)};
        for (uint64_t row = block; row < tiling_->taskCount; row += blocks) {
            const uint64_t outputBase = row * rowElements;
            const uint64_t inputBase = InputOffset(outputBase);
            for (uint64_t first = 0; first < rowElements; first += kStridedElements) {
                const uint32_t count = static_cast<uint32_t>(
                    Min(kStridedElements, rowElements - first));
                const AscendC::DataCopyExtParams inputParams{
                    static_cast<uint16_t>(count), static_cast<uint32_t>(sizeof(T)),
                    static_cast<uint32_t>((step - 1) * sizeof(T)), 0, 0};
                AscendC::DataCopyPad(scratch, input_[inputBase + first * step],
                                     inputParams, pad);
                AscendC::PipeBarrier<PIPE_ALL>();
                CompactStrided(compact, scratch, count);
                AscendC::PipeBarrier<PIPE_ALL>();
                const AscendC::DataCopyExtParams outputParams{
                    1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
                AscendC::DataCopyPad(output_[outputBase + first], compact,
                                     outputParams);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessScalar()
    {
        const uint64_t block = AscendC::GetBlockIdx();
        const uint64_t blocks = AscendC::GetBlockNum();
        AscendC::LocalTensor<T> local = outputBuffer_.Get<T>();
        for (uint64_t task = block; task < tiling_->taskCount; task += blocks) {
            const uint64_t outputStart = task * kStridedElements;
            const uint32_t count = static_cast<uint32_t>(Min(
                kStridedElements, tiling_->totalElements - outputStart));
            for (uint32_t index = 0; index < count; ++index) {
                const uint64_t outputIndex = outputStart + index;
                local.SetValue(index,
                               input_.GetValue(InputOffset(outputIndex)));
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            const AscendC::DataCopyExtParams outputParams{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(output_[outputStart], local, outputParams);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuffer_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> indexBuffer_;
    AscendC::GlobalTensor<T> input_;
    AscendC::GlobalTensor<T> output_;
    __gm__ const CompetitionTransposeTilingData *tiling_ = nullptr;
};

extern "C" __global__ __aicore__ void transpose(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto *data = reinterpret_cast<__gm__ const CompetitionTransposeTilingData *>(tiling);
    if (data->strategy == 8) {
        KernelTransposeCopy<DTYPE_INPUT> copy;
        copy.Init(input, output, data);
        copy.Process();
        return;
    }
    if (data->strategy == 6 || data->strategy == 7) {
        if constexpr (sizeof(DTYPE_INPUT) == 1) {
            if (data->strategy == 6 && (data->inputDims[5] & 4)) {
                if (data->inputDims[6]) {
                    KernelTransposePipeline<DTYPE_INPUT, true, true> native;
                    native.Init(input, output, data);
                    native.Process();
                } else {
                    KernelTransposePipeline<DTYPE_INPUT, false, true> native;
                    native.Init(input, output, data);
                    native.Process();
                }
                return;
            }
        }
        if (data->inputDims[6]) {
            KernelTransposePipeline<DTYPE_INPUT, true> merged;
            merged.Init(input, output, data);
            merged.Process();
            return;
        }
        KernelTransposePipeline<DTYPE_INPUT> fast;
        fast.Init(input, output, data);
        fast.Process();
        return;
    }
    KernelTranspose<DTYPE_INPUT> kernel;
    kernel.Init(input, output, tiling);
    kernel.Process();
}
