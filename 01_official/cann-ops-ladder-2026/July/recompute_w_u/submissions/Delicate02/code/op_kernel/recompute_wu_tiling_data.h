/*!
 * \file recompute_wu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RECOMPUTEWU_TILING_DATA_H_
#define _RECOMPUTEWU_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

struct alignas(8) RecomputeWuTilingData {
    uint32_t batchSize = 0;
    uint32_t totalTokens = 0;
    uint32_t headNum = 0;
    uint32_t groupHeadNum = 0;
    uint32_t keyDim = 0;
    uint32_t valueDim = 0;
    uint32_t chunkSize = 0;
    uint32_t numChunks = 0;
    uint32_t taskCount = 0;
    uint32_t blockDim = 0;
    uint32_t workspacePerCore = 0;
    uint32_t headsPerTask = 1;
    uint32_t headBatchCount = 1;
    uint32_t chunksPerTask = 1;
    uint32_t chunkGroupCount = 0;
    TCubeTiling cubeTiling;
};
#endif
