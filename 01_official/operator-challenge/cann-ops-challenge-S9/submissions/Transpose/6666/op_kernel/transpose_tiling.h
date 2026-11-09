#ifndef TRANSPOSE_KERNEL_TILING_H
#define TRANSPOSE_KERNEL_TILING_H

#include "kernel_operator.h"

// Device layout mirrors op_host/transpose_tiling.h, including 8-byte alignment.
#pragma pack(push, 8)
struct TransposeKernelTilingData {
    uint64_t totalElements;
    uint64_t runElements;
    uint64_t runCount;
    uint64_t groupCount;
    uint64_t blocksPerGroup;
    uint64_t chunksPerUnit;
    uint64_t taskCount;
    uint64_t sourceStepBytes;
    uint64_t batchCount;
    uint64_t matrixRows;
    uint64_t matrixColumns;
    uint64_t matrixPrefixRank;
    uint64_t outputMatrixRowStride;
    uint32_t rank;
    uint32_t prefixRank;
    uint32_t tileElements;
    uint32_t gatherBlocks;
    uint32_t elementSize;
    uint32_t matrixRowsPerTask;
    uint32_t matrixColumnsPerTask;
    uint32_t matrixRowAxis;
    uint32_t matrixColumnAxis;
    uint32_t runAxis;
    uint64_t outputRunStride;
    uint64_t outputDims[8];
    uint64_t sourceStrides[8];
    AscendC::tiling::ConfusionTransposeTiling transposeTiling;
    AscendC::tiling::ConfusionTransposeTiling tailTransposeTiling;
};
#pragma pack(pop)

#endif  // TRANSPOSE_KERNEL_TILING_H
