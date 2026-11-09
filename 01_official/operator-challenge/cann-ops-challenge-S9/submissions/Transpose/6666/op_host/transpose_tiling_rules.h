#ifndef TRANSPOSE_TILING_RULES_H
#define TRANSPOSE_TILING_RULES_H

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {
constexpr uint32_t TRANSPOSE_MAX_RANK = 8;
constexpr uint32_t TRANSPOSE_TILE_BYTES = 32 * 1024;
constexpr uint32_t TRANSPOSE_BLOCK_BYTES = 32;
constexpr uint32_t TRANSPOSE_MAX_COPY_BLOCKS = 4095;
constexpr uint32_t TRANSPOSE_MATRIX_TILE = 16;
constexpr uint32_t TRANSPOSE_MAX_TRANSPOSE_REPEATS = 255;
constexpr uint32_t TRANSPOSE_WORD32_COLUMNS_PER_REPEAT =
    TRANSPOSE_BLOCK_BYTES / sizeof(uint32_t);

constexpr uint64_t TRANSPOSE_KEY_CONTIGUOUS_RUN = 100;
constexpr uint64_t TRANSPOSE_KEY_TILED_2D_FP16 = 200;
constexpr uint64_t TRANSPOSE_KEY_SLAB_2D_FP16 = 220;
constexpr uint64_t TRANSPOSE_KEY_SLAB_2D_WORD32 = 230;
constexpr uint64_t TRANSPOSE_KEY_SLAB_2D_BYTE = 225;
constexpr uint64_t TRANSPOSE_KEY_TILED_MOVE_LAST_FP16 = 240;
constexpr uint64_t TRANSPOSE_KEY_BATCHED_SLAB_WORD32 = 250;
constexpr uint64_t TRANSPOSE_KEY_PACKED_GATHER_WORD32 = 251;
constexpr uint64_t TRANSPOSE_KEY_FULL_ROW_WORD32 = 256;
constexpr uint64_t TRANSPOSE_KEY_STRIDED_TILE_WORD32 = 260;
constexpr uint64_t TRANSPOSE_KEY_STRIDED_GATHER = 900;
constexpr uint64_t TRANSPOSE_KEY_PACKED_BYTE_GATHER = 910;
constexpr uint64_t TRANSPOSE_KEY_PACKED_WORD_GATHER = 920;
constexpr uint64_t TRANSPOSE_KEY_PACKED_WORD_FLAT_GATHER = 921;
constexpr uint64_t TRANSPOSE_KEY_BLOCK_SWAP_FP16 = 930;

enum class TransposeStrategy : uint8_t {
    CONTIGUOUS_RUN_COPY,
    TILED_2D_FP16,
    SLAB_2D_FP16,
    SLAB_2D_WORD32,
    SLAB_2D_BYTE,
    TILED_MOVE_LAST_FP16,
    BATCHED_SLAB_WORD32,
    PACKED_GATHER_WORD32,
    FULL_ROW_WORD32,
    STRIDED_TILE_WORD32,
    STRIDED_GATHER,
    BLOCK_SWAP_FP16,
};

struct TransposePlan {
    TransposeStrategy strategy = TransposeStrategy::STRIDED_GATHER;
    uint64_t tilingKey = TRANSPOSE_KEY_STRIDED_GATHER;
    uint64_t runElements = 1;
    uint64_t runCount = 1;
    uint64_t groupCount = 1;
    uint64_t blocksPerGroup = 1;
    uint64_t chunksPerUnit = 1;
    uint64_t taskCount = 1;
    uint64_t sourceStepBytes = 0;
    uint64_t batchCount = 0;
    uint64_t matrixRows = 0;
    uint64_t matrixColumns = 0;
    uint32_t matrixPrefixRank = 0;
    uint32_t matrixRowAxis = 0;
    uint32_t matrixColumnAxis = 0;
    uint64_t outputMatrixRowStride = 0;
    uint32_t prefixRank = 0;
    uint32_t tileElements = 1;
    uint32_t gatherBlocks = 1;
    uint32_t matrixRowsPerTask = 0;
    uint32_t matrixColumnsPerTask = 0;
    uint32_t blockDim = 1;
    uint32_t runAxis = 0;
    uint64_t outputRunStride = 1;
};

inline uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0 ? 1 : 0);
}

inline uint32_t SelectBatchedWord32SlabColumns(
    uint64_t columns, uint64_t ubSize, uint32_t rowsPerTask,
    uint32_t outputBufferCount)
{
    const uint64_t bufferBytesPerColumn =
        (1ULL + outputBufferCount) * sizeof(uint32_t);
    constexpr uint64_t maxColumnsByTranspose =
        TRANSPOSE_MAX_TRANSPOSE_REPEATS *
        TRANSPOSE_WORD32_COLUMNS_PER_REPEAT;
    if (columns < TRANSPOSE_MATRIX_TILE || ubSize <= 1) {
        return 0;
    }
    uint64_t maxColumns =
        (ubSize - 1) / (bufferBytesPerColumn * rowsPerTask);
    maxColumns = std::min<uint64_t>(
        maxColumns, TRANSPOSE_MAX_COPY_BLOCKS);
    maxColumns = std::min(maxColumns, maxColumnsByTranspose);
    maxColumns -= maxColumns % TRANSPOSE_MATRIX_TILE;
    const uint64_t alignedColumns =
        CeilDiv(columns, TRANSPOSE_MATRIX_TILE) *
        TRANSPOSE_MATRIX_TILE;
    if (alignedColumns <= maxColumns) {
        return static_cast<uint32_t>(alignedColumns);
    }
    return static_cast<uint32_t>(maxColumns);
}

inline uint32_t SelectWord32SlabColumns(uint64_t columns, uint64_t ubSize)
{
    constexpr uint64_t bufferBytesPerColumn =
        2ULL * TRANSPOSE_MATRIX_TILE * sizeof(uint32_t);
    constexpr uint64_t maxColumnsByTranspose =
        TRANSPOSE_MAX_TRANSPOSE_REPEATS *
        TRANSPOSE_WORD32_COLUMNS_PER_REPEAT;
    if (columns < TRANSPOSE_MATRIX_TILE || ubSize <= 1) {
        return 0;
    }
    uint64_t maxColumns = (ubSize - 1) / bufferBytesPerColumn;
    maxColumns = std::min(maxColumns, columns);
    maxColumns = std::min<uint64_t>(
        maxColumns, TRANSPOSE_MAX_COPY_BLOCKS);
    maxColumns = std::min(maxColumns, maxColumnsByTranspose);
    maxColumns -= maxColumns % TRANSPOSE_MATRIX_TILE;
    for (uint64_t candidate = maxColumns;
         candidate >= TRANSPOSE_MATRIX_TILE;
         candidate -= TRANSPOSE_MATRIX_TILE) {
        if (columns % candidate == 0) {
            return static_cast<uint32_t>(candidate);
        }
    }
    return 0;
}

inline TransposePlan SelectTransposePlan(
    const uint64_t *outputDims, const uint64_t *sourceStrides,
    uint32_t rank, uint64_t totalElements, uint32_t elementSize,
    uint32_t availableAivCores, uint64_t ubSize)
{
    TransposePlan plan;
    const uint32_t coreCount = std::max(1U, availableAivCores);
    plan.tileElements = std::max(1U, TRANSPOSE_TILE_BYTES / elementSize);
    plan.gatherBlocks = std::min(
        TRANSPOSE_MAX_COPY_BLOCKS,
        TRANSPOSE_TILE_BYTES / TRANSPOSE_BLOCK_BYTES);

    uint64_t expectedStride = 1;
    uint64_t runElements = 1;
    uint32_t prefixRank = rank;
    for (uint32_t reverse = rank; reverse > 0; --reverse) {
        const uint32_t axis = reverse - 1;
        if (outputDims[axis] == 1) {
            prefixRank = axis;
            continue;
        }
        if (sourceStrides[axis] != expectedStride) {
            break;
        }
        runElements *= outputDims[axis];
        expectedStride *= outputDims[axis];
        prefixRank = axis;
    }

    plan.runElements = runElements;
    plan.runCount = totalElements / runElements;
    plan.prefixRank = prefixRank;
    plan.runAxis = prefixRank;
    plan.outputRunStride = 1;
    uint64_t interiorRunElements = 0;
    uint32_t interiorRunAxis = 0;
    uint64_t interiorOutputStride = 1;
    if (runElements == 1) {
        uint32_t bestAxis = rank;
        uint64_t bestDim = 1;
        for (uint32_t axis = 0; axis < rank; ++axis) {
            if (sourceStrides[axis] == 1 && outputDims[axis] > bestDim) {
                bestAxis = axis;
                bestDim = outputDims[axis];
            }
        }
        const uint64_t bestBytes = bestDim * elementSize;
        if (bestAxis > 0 && bestAxis + 1 < rank &&
            bestBytes >= TRANSPOSE_BLOCK_BYTES) {
            uint64_t outputStride = 1;
            for (uint32_t axis = bestAxis + 1; axis < rank; ++axis) {
                outputStride *= outputDims[axis];
            }
            interiorRunElements = bestDim;
            interiorRunAxis = bestAxis;
            interiorOutputStride = outputStride;
        }
    }
    const uint64_t runBytes = runElements * elementSize;
    const bool fullTensorRun = runElements == totalElements;
    const bool stridedRunCanGather =
        !fullTensorRun && prefixRank > 0 &&
        runBytes <= TRANSPOSE_TILE_BYTES &&
        sourceStrides[prefixRank - 1] >= runElements;
    const uint64_t maxCopyStride = std::numeric_limits<uint32_t>::max();
    if (elementSize == 1 && rank >= 2 && sourceStrides[0] == 1 &&
        outputDims[0] >= 32 && outputDims[0] <= TRANSPOSE_MAX_COPY_BLOCKS) {
        uint64_t expected = outputDims[0];
        bool foldedMatrix = true;
        for (uint32_t reverse = rank; reverse > 1; --reverse) {
            const uint32_t axis = reverse - 1;
            if (outputDims[axis] != 1 && sourceStrides[axis] != expected) {
                foldedMatrix = false;
                break;
            }
            expected *= outputDims[axis];
        }
        const uint64_t rows = totalElements / outputDims[0];
        const uint64_t columns = CeilDiv(outputDims[0], 32) * 32;
        constexpr uint32_t rowsPerTask = 32;
        // One byte input/output and two half intermediates.
        const uint64_t bufferBytes = 6ULL * rowsPerTask * columns;
        if (foldedMatrix && expected == totalElements && rows >= rowsPerTask &&
            rows <= maxCopyStride && bufferBytes < ubSize) {
            plan.strategy = TransposeStrategy::SLAB_2D_BYTE;
            plan.tilingKey = TRANSPOSE_KEY_SLAB_2D_BYTE;
            plan.matrixRows = rows;
            plan.matrixColumns = outputDims[0];
            plan.matrixRowsPerTask = rowsPerTask;
            plan.matrixColumnsPerTask = static_cast<uint32_t>(columns);
            plan.taskCount = CeilDiv(rows, rowsPerTask);
            plan.blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreCount, plan.taskCount));
            return plan;
        }
    }
    if (elementSize == 2 && rank == 4 && sourceStrides[3] == 1 &&
        sourceStrides[0] == outputDims[3] &&
        sourceStrides[1] == outputDims[0] * outputDims[3] &&
        sourceStrides[2] == outputDims[1] * outputDims[0] * outputDims[3] &&
        outputDims[2] <= TRANSPOSE_MAX_COPY_BLOCKS && outputDims[2] > 1 &&
        outputDims[0] >= 16 && sourceStrides[2] <= maxCopyStride / 2) {
        const uint64_t group = outputDims[2] * outputDims[3];
        const uint64_t outputBytes = CeilDiv(group * 2, 32) * 32 * 4;
        const uint64_t offsetBytes = outputBytes * 2;
        constexpr uint32_t tileC = 24;
        const uint64_t inputPitch = CeilDiv(tileC * outputDims[3], 16) * 16;
        const uint64_t inputBytes = outputDims[2] * inputPitch * 2;
        const uint64_t bufferBytes = 2 * inputBytes + 2 * outputBytes + offsetBytes;
        if (bufferBytes < ubSize && group <= TRANSPOSE_TILE_BYTES / 2) {
            plan.strategy = TransposeStrategy::BLOCK_SWAP_FP16;
            plan.tilingKey = TRANSPOSE_KEY_BLOCK_SWAP_FP16;
            plan.matrixRows = outputDims[2];
            plan.matrixColumns = outputDims[0];
            plan.batchCount = outputDims[1];
            plan.matrixColumnsPerTask = tileC;
            plan.taskCount = outputDims[1] * CeilDiv(outputDims[0], tileC);
            plan.blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreCount, plan.taskCount));
            return plan;
        }
    }
    const bool physical2dTranspose =
        rank == 2 &&
        outputDims[0] >= TRANSPOSE_MATRIX_TILE &&
        outputDims[1] >= TRANSPOSE_MATRIX_TILE &&
        outputDims[0] % TRANSPOSE_MATRIX_TILE == 0 &&
        outputDims[1] % TRANSPOSE_MATRIX_TILE == 0 &&
        sourceStrides[0] == 1 && sourceStrides[1] == outputDims[0] &&
        (outputDims[0] - TRANSPOSE_MATRIX_TILE) * elementSize <=
            maxCopyStride &&
        (outputDims[1] - TRANSPOSE_MATRIX_TILE) * elementSize <=
            maxCopyStride;
    const bool tiled2dFp16 =
        physical2dTranspose && elementSize == sizeof(uint16_t);
    const uint64_t slabElements =
        TRANSPOSE_MATRIX_TILE * outputDims[0];
    const uint64_t slabBufferBytesPerElement = 2ULL * elementSize;
    const bool slab2dFp16FitsUb =
        tiled2dFp16 && outputDims[0] > TRANSPOSE_MATRIX_TILE &&
        outputDims[0] <= TRANSPOSE_MAX_COPY_BLOCKS && ubSize > 0 &&
        slabElements <= (ubSize - 1) / slabBufferBytesPerElement;
    const uint32_t word32SlabColumns =
        physical2dTranspose && elementSize == sizeof(uint32_t)
            ? SelectWord32SlabColumns(outputDims[0], ubSize)
            : 0;

    uint64_t stridedTileWord32BatchCount = 0;
    uint64_t stridedTileWord32Rows = 0;
    uint64_t stridedTileWord32Columns = 0;
    uint64_t stridedTileWord32OutputRowStride = 0;
    uint32_t stridedTileWord32RowAxis = 0;
    uint32_t stridedTileWord32ColumnAxis = 0;
    uint32_t stridedTileWord32FirstColumnAxis = 0;
    uint32_t stridedTileWord32TileColumns = 32;
    constexpr uint32_t stridedTileWord32PaddedSize = 32;
    constexpr uint64_t stridedTileWord32BufferBytes =
        2ULL * stridedTileWord32PaddedSize * stridedTileWord32PaddedSize *
        sizeof(uint32_t);
    if (rank >= 3 && elementSize == sizeof(uint32_t) && runElements == 1 &&
        ubSize > stridedTileWord32BufferBytes) {
        const uint32_t rowAxis = rank - 1;
        const uint64_t rows = outputDims[rowAxis];
        uint32_t columnAxis = rank;
        uint64_t columns = 0;
        if (rows >= TRANSPOSE_MATRIX_TILE &&
            rows <= stridedTileWord32PaddedSize) {
            for (uint32_t axis = 0; axis < rowAxis; ++axis) {
                if (sourceStrides[axis] == 1 &&
                    outputDims[axis] >= TRANSPOSE_MATRIX_TILE &&
                    outputDims[axis] <= stridedTileWord32PaddedSize &&
                    outputDims[axis] > columns) {
                    columnAxis = axis;
                    columns = outputDims[axis];
                }
            }
        }
        if (columnAxis < rank && sourceStrides[rowAxis] >= columns) {
            uint64_t outputRowStride = 1;
            for (uint32_t axis = columnAxis + 1; axis < rank; ++axis) {
                outputRowStride *= outputDims[axis];
            }
            uint32_t firstColumnAxis = columnAxis;
            if (ubSize >= 4ULL * 32 * 256 * sizeof(uint32_t)) {
                while (firstColumnAxis > 0 && firstColumnAxis - 1 != rowAxis &&
                       sourceStrides[firstColumnAxis - 1] == columns) {
                    columns *= outputDims[firstColumnAxis - 1];
                    --firstColumnAxis;
                }
            }
            const uint64_t matrixElements = rows * columns;
            const uint64_t inputGapBytes =
                (sourceStrides[rowAxis] - columns) * elementSize;
            const uint64_t outputGapBytes =
                (outputRowStride - rows) * elementSize;
            if (outputRowStride >= rows && matrixElements != 0 &&
                totalElements % matrixElements == 0 &&
                inputGapBytes <= maxCopyStride &&
                outputGapBytes <= maxCopyStride) {
                stridedTileWord32BatchCount = totalElements / matrixElements;
                stridedTileWord32Rows = rows;
                stridedTileWord32Columns = columns;
                stridedTileWord32OutputRowStride = outputRowStride;
                stridedTileWord32RowAxis = rowAxis;
                stridedTileWord32ColumnAxis = columnAxis;
                stridedTileWord32FirstColumnAxis = firstColumnAxis;
                // Four 32-row buffers share UB; keep the column tile aligned
                // for the 16-column vector transpose blocks.
                const uint32_t maxTileColumns = static_cast<uint32_t>(
                    ubSize / (4ULL * 32 * sizeof(uint32_t)) / 16 * 16);
                stridedTileWord32TileColumns = columns > 32 ? maxTileColumns : 32;
            }
        }
    }

    uint64_t batchedWord32Count = 0;
    uint64_t batchedWord32Rows = 0;
    uint64_t batchedWord32Columns = 0;
    uint32_t batchedWord32SlabColumns = 0;
    uint32_t batchedWord32RowsPerTask = 0;
    uint32_t batchedWord32PrefixRank = 0;
    if (rank >= 2 && elementSize == sizeof(uint32_t)) {
        for (uint32_t movedAxis = 0; movedAxis + 1 < rank; ++movedAxis) {
            if (sourceStrides[movedAxis] != 1 ||
                outputDims[movedAxis] < TRANSPOSE_MATRIX_TILE) {
                continue;
            }
            const uint64_t columns = outputDims[movedAxis];
            uint64_t expected = columns;
            uint64_t rows = 1;
            bool compatible = true;
            for (uint32_t reverse = rank; reverse > movedAxis + 1; --reverse) {
                const uint32_t axis = reverse - 1;
                if (outputDims[axis] != 1 && sourceStrides[axis] != expected) {
                    compatible = false;
                    break;
                }
                if (expected > totalElements / outputDims[axis]) {
                    compatible = false;
                    break;
                }
                expected *= outputDims[axis];
                rows *= outputDims[axis];
            }
            if (!compatible || rows < TRANSPOSE_MATRIX_TILE) {
                continue;
            }
            const uint64_t matrixElements = rows * columns;
            if (expected != matrixElements || matrixElements == 0 ||
                totalElements % matrixElements != 0) {
                continue;
            }
            const uint64_t batches = totalElements / matrixElements;
            if (batches < 2) {
                continue;
            }
            uint64_t prefixExpected = matrixElements;
            bool contiguousPrefix = true;
            for (uint32_t reverse = movedAxis; reverse > 0; --reverse) {
                const uint32_t axis = reverse - 1;
                if (outputDims[axis] != 1 &&
                    sourceStrides[axis] != prefixExpected) {
                    contiguousPrefix = false;
                    break;
                }
                prefixExpected *= outputDims[axis];
            }
            const bool narrowMatrix = columns <= 32;
            const uint32_t rowsPerTask = narrowMatrix ? 96 : 128;
            const uint32_t outputBufferCount = narrowMatrix ? 2 : 1;
            const uint32_t slabColumns = SelectBatchedWord32SlabColumns(
                columns, ubSize, rowsPerTask, outputBufferCount);
            if (slabColumns == 0) {
                continue;
            }
            batchedWord32Count = batches;
            batchedWord32Rows = rows;
            batchedWord32Columns = columns;
            batchedWord32SlabColumns = slabColumns;
            batchedWord32RowsPerTask = rowsPerTask;
            batchedWord32PrefixRank = contiguousPrefix ? 0 : movedAxis;
            break;
        }
    }

    uint64_t moveLastRows = 0;
    bool tiledMoveLastFp16 = false;
    if (rank >= 2 && elementSize == sizeof(uint16_t) &&
        outputDims[0] >= TRANSPOSE_MATRIX_TILE &&
        sourceStrides[0] == 1 &&
        totalElements % outputDims[0] == 0) {
        const uint64_t columns = outputDims[0];
        uint64_t flattenedElements = columns;
        bool physicallyContiguous = true;
        for (uint32_t reverse = rank; reverse > 1; --reverse) {
            const uint32_t axis = reverse - 1;
            if (outputDims[axis] == 1) {
                continue;
            }
            if (sourceStrides[axis] != flattenedElements ||
                flattenedElements > totalElements / outputDims[axis]) {
                physicallyContiguous = false;
                break;
            }
            flattenedElements *= outputDims[axis];
        }
        moveLastRows = totalElements / columns;
        const uint64_t maxStride = std::numeric_limits<uint32_t>::max();
        const uint64_t lastColumnStart =
            ((columns - 1) / TRANSPOSE_MATRIX_TILE) *
            TRANSPOSE_MATRIX_TILE;
        const uint64_t lastRowStart =
            ((moveLastRows - 1) / TRANSPOSE_MATRIX_TILE) *
            TRANSPOSE_MATRIX_TILE;
        tiledMoveLastFp16 =
            physicallyContiguous && flattenedElements == totalElements &&
            moveLastRows >= TRANSPOSE_MATRIX_TILE &&
            lastColumnStart * elementSize <= maxStride &&
            lastRowStart * elementSize <= maxStride;
    }

    if (slab2dFp16FitsUb) {
        plan.strategy = TransposeStrategy::SLAB_2D_FP16;
        plan.tilingKey = TRANSPOSE_KEY_SLAB_2D_FP16;
        plan.matrixRowsPerTask = TRANSPOSE_MATRIX_TILE;
        plan.taskCount = outputDims[1] / plan.matrixRowsPerTask;
    } else if (tiled2dFp16) {
        plan.strategy = TransposeStrategy::TILED_2D_FP16;
        plan.tilingKey = TRANSPOSE_KEY_TILED_2D_FP16;
        plan.taskCount =
            (outputDims[0] / TRANSPOSE_MATRIX_TILE) *
            (outputDims[1] / TRANSPOSE_MATRIX_TILE);
    } else if (word32SlabColumns != 0) {
        plan.strategy = TransposeStrategy::SLAB_2D_WORD32;
        plan.tilingKey = TRANSPOSE_KEY_SLAB_2D_WORD32;
        plan.matrixRowsPerTask = TRANSPOSE_MATRIX_TILE;
        plan.matrixColumnsPerTask = word32SlabColumns;
        plan.taskCount =
            (outputDims[1] / plan.matrixRowsPerTask) *
            (outputDims[0] / plan.matrixColumnsPerTask);
    } else if (batchedWord32SlabColumns != 0 && batchedWord32Columns > 64 &&
               batchedWord32Rows >= 48 &&
               48ULL * CeilDiv(batchedWord32Columns, 8) * 8 * sizeof(uint32_t) +
                   2ULL * 48 * 240 * sizeof(uint32_t) <= ubSize) {
        plan.strategy = TransposeStrategy::FULL_ROW_WORD32;
        plan.tilingKey = TRANSPOSE_KEY_FULL_ROW_WORD32;
        plan.batchCount = batchedWord32Count;
        plan.matrixRows = batchedWord32Rows;
        plan.matrixColumns = batchedWord32Columns;
        plan.matrixPrefixRank = batchedWord32PrefixRank;
        plan.matrixRowsPerTask = 48;
        plan.matrixColumnsPerTask = 240;
        plan.taskCount = batchedWord32Count * CeilDiv(batchedWord32Rows, 48);
    } else if (batchedWord32SlabColumns != 0 && batchedWord32Columns > 64 &&
               batchedWord32Rows >= 64 &&
               64ULL * batchedWord32Columns * sizeof(uint32_t) +
                   3ULL * 64 * 64 * sizeof(uint32_t) <= ubSize) {
        plan.strategy = TransposeStrategy::PACKED_GATHER_WORD32;
        plan.tilingKey = TRANSPOSE_KEY_PACKED_GATHER_WORD32;
        plan.batchCount = batchedWord32Count;
        plan.matrixRows = batchedWord32Rows;
        plan.matrixColumns = batchedWord32Columns;
        plan.matrixPrefixRank = batchedWord32PrefixRank;
        plan.matrixRowsPerTask = 64;
        plan.matrixColumnsPerTask = 64;
        plan.taskCount = batchedWord32Count * CeilDiv(batchedWord32Rows, 64);
    } else if (batchedWord32SlabColumns != 0) {
        plan.strategy = TransposeStrategy::BATCHED_SLAB_WORD32;
        plan.tilingKey = TRANSPOSE_KEY_BATCHED_SLAB_WORD32;
        plan.batchCount = batchedWord32Count;
        plan.matrixRows = batchedWord32Rows;
        plan.matrixColumns = batchedWord32Columns;
        plan.matrixPrefixRank = batchedWord32PrefixRank;
        plan.matrixRowsPerTask = batchedWord32RowsPerTask;
        plan.matrixColumnsPerTask = batchedWord32SlabColumns;
        const uint64_t fullColumnTasks =
            batchedWord32Columns / batchedWord32SlabColumns;
        const uint64_t remainingColumns =
            batchedWord32Columns % batchedWord32SlabColumns;
        const uint64_t paddedColumnTasks =
            remainingColumns != 0 &&
                    batchedWord32SlabColumns - remainingColumns <=
                        TRANSPOSE_BLOCK_BYTES / sizeof(uint32_t) - 1
                ? 1
                : 0;
        const uint64_t tailColumnTasks = paddedColumnTasks == 0
            ? CeilDiv(remainingColumns, TRANSPOSE_MATRIX_TILE)
            : 0;
        plan.taskCount = batchedWord32Count *
            CeilDiv(batchedWord32Rows, plan.matrixRowsPerTask) *
            (fullColumnTasks + paddedColumnTasks + tailColumnTasks);
    } else if (stridedTileWord32BatchCount != 0) {
        plan.strategy = TransposeStrategy::STRIDED_TILE_WORD32;
        plan.tilingKey = TRANSPOSE_KEY_STRIDED_TILE_WORD32;
        plan.batchCount = stridedTileWord32BatchCount;
        plan.matrixRows = stridedTileWord32Rows;
        plan.matrixColumns = stridedTileWord32Columns;
        plan.matrixRowsPerTask = stridedTileWord32PaddedSize;
        plan.matrixColumnsPerTask = stridedTileWord32TileColumns;
        plan.matrixPrefixRank = stridedTileWord32FirstColumnAxis;
        plan.matrixRowAxis = stridedTileWord32RowAxis;
        plan.matrixColumnAxis = stridedTileWord32ColumnAxis;
        plan.outputMatrixRowStride = stridedTileWord32OutputRowStride;
        plan.taskCount = stridedTileWord32BatchCount * CeilDiv(
            stridedTileWord32Columns, stridedTileWord32TileColumns);
    } else if (tiledMoveLastFp16) {
        plan.strategy = TransposeStrategy::TILED_MOVE_LAST_FP16;
        plan.tilingKey = TRANSPOSE_KEY_TILED_MOVE_LAST_FP16;
        const uint64_t tileRows =
            CeilDiv(moveLastRows, TRANSPOSE_MATRIX_TILE);
        const uint64_t tileColumns =
            CeilDiv(outputDims[0], TRANSPOSE_MATRIX_TILE);
        plan.taskCount = tileRows * tileColumns;
    } else if (false && interiorRunElements != 0) {
        plan.strategy = TransposeStrategy::CONTIGUOUS_RUN_COPY;
        plan.tilingKey = TRANSPOSE_KEY_CONTIGUOUS_RUN;
        plan.runElements = interiorRunElements;
        plan.runCount = totalElements / interiorRunElements;
        plan.prefixRank = rank;
        plan.runAxis = interiorRunAxis;
        plan.outputRunStride = interiorOutputStride;
        plan.chunksPerUnit = 1;
        if (interiorRunAxis > 0 &&
            sourceStrides[interiorRunAxis - 1] == interiorRunElements &&
            outputDims[interiorRunAxis - 1] > 1 &&
            outputDims[interiorRunAxis - 1] <= TRANSPOSE_MAX_COPY_BLOCKS) {
            plan.matrixRows = outputDims[interiorRunAxis - 1];
            plan.taskCount = plan.runCount / plan.matrixRows;
        } else {
            plan.chunksPerUnit = CeilDiv(interiorRunElements, plan.tileElements);
            plan.taskCount = plan.runCount * plan.chunksPerUnit;
        }
    } else if (fullTensorRun ||
               (runBytes >= TRANSPOSE_BLOCK_BYTES &&
                (!stridedRunCanGather ||
                 plan.runCount < static_cast<uint64_t>(coreCount) * 2))) {
        plan.strategy = TransposeStrategy::CONTIGUOUS_RUN_COPY;
        plan.tilingKey = TRANSPOSE_KEY_CONTIGUOUS_RUN;
        plan.chunksPerUnit = CeilDiv(runElements, plan.tileElements);
        plan.taskCount = plan.runCount * plan.chunksPerUnit;
    } else {
        const uint32_t gatherAxis = prefixRank - 1;
        const uint64_t sourceStepElements = sourceStrides[gatherAxis];
        const uint64_t localBlockBytes =
            CeilDiv(runBytes, TRANSPOSE_BLOCK_BYTES) *
            TRANSPOSE_BLOCK_BYTES;
        plan.gatherBlocks = static_cast<uint32_t>(std::min<uint64_t>(
            TRANSPOSE_MAX_COPY_BLOCKS,
            std::max<uint64_t>(
                1, TRANSPOSE_TILE_BYTES / localBlockBytes)));
        uint32_t groupPrefixRank = prefixRank;
        uint64_t blocksPerGroup = outputDims[gatherAxis];
        const bool packedByteRun = runBytes == 1;
        const bool packedWordRun =
            runBytes == sizeof(uint16_t) || runBytes == sizeof(uint32_t);
        if (packedByteRun || packedWordRun ||
            runBytes >= TRANSPOSE_BLOCK_BYTES) {
            while (groupPrefixRank > 1) {
                const uint32_t axis = groupPrefixRank - 2;
                if (outputDims[axis] == 0 ||
                    blocksPerGroup >
                        std::numeric_limits<uint64_t>::max() /
                            outputDims[axis]) {
                    break;
                }
                if (outputDims[axis] != 1) {
                    if (sourceStepElements == 0 ||
                        blocksPerGroup >
                            std::numeric_limits<uint64_t>::max() /
                                sourceStepElements ||
                        sourceStrides[axis] !=
                            blocksPerGroup * sourceStepElements) {
                        break;
                    }
                }
                blocksPerGroup *= outputDims[axis];
                groupPrefixRank = axis + 1;
            }
        }
        plan.strategy = TransposeStrategy::STRIDED_GATHER;
        plan.prefixRank = groupPrefixRank;
        plan.blocksPerGroup = blocksPerGroup;
        plan.groupCount = totalElements / runElements / plan.blocksPerGroup;
        plan.sourceStepBytes = sourceStepElements * elementSize;
        plan.chunksPerUnit = CeilDiv(plan.blocksPerGroup, plan.gatherBlocks);
        const bool wholeGroupWindow =
            plan.blocksPerGroup < plan.gatherBlocks &&
            plan.gatherBlocks % plan.blocksPerGroup == 0;
        const uint64_t groupsPerFlatTask = wholeGroupWindow
            ? plan.gatherBlocks / plan.blocksPerGroup
            : 0;
        const uint64_t flatTaskCount = groupsPerFlatTask != 0
            ? CeilDiv(plan.groupCount, groupsPerFlatTask)
            : 0;
        const bool sourceGapFitsCopyStride =
            plan.sourceStepBytes >= runBytes &&
            plan.sourceStepBytes - runBytes <= maxCopyStride;
        const bool packedWordFlatGather =
            packedWordRun &&
            plan.blocksPerGroup == outputDims[gatherAxis] &&
            groupsPerFlatTask >= 2 && groupsPerFlatTask <= 16 &&
            flatTaskCount >= coreCount && sourceGapFitsCopyStride;
        if (packedByteRun) {
            plan.tilingKey = TRANSPOSE_KEY_PACKED_BYTE_GATHER;
        } else if (packedWordFlatGather) {
            plan.tilingKey = TRANSPOSE_KEY_PACKED_WORD_FLAT_GATHER;
        } else if (packedWordRun) {
            plan.tilingKey = TRANSPOSE_KEY_PACKED_WORD_GATHER;
        } else {
            plan.tilingKey = TRANSPOSE_KEY_STRIDED_GATHER;
        }
        plan.taskCount = packedWordFlatGather
            ? flatTaskCount
            : plan.groupCount * plan.chunksPerUnit;
    }
    plan.blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(coreCount, std::max<uint64_t>(1, plan.taskCount)));
    return plan;
}
}  // namespace optiling

#endif  // TRANSPOSE_TILING_RULES_H
