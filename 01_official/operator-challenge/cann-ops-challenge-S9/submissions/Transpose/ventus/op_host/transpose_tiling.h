/**
 * Transpose 主机侧与设备侧共享的分块数据结构。
 * 描述规范化维度、输入输出步长、连续片段和二维 tile 的调度参数。
 */

#ifndef S9_TRANSPOSE_TILING_H
#define S9_TRANSPOSE_TILING_H

#include <cstdint>

constexpr uint32_t S9_TRANSPOSE_MAX_RANK = 8;
constexpr uint32_t S9_TRANSPOSE_MAX_GROUP = 8;

constexpr uint32_t S9_TRANSPOSE_REGIME_RUN = 0;
constexpr uint32_t S9_TRANSPOSE_REGIME_TILE = 1;

struct TransposeTilingData {
    uint64_t outerLen[S9_TRANSPOSE_MAX_GROUP];
    uint64_t outerInStride[S9_TRANSPOSE_MAX_GROUP];
    uint64_t outerOutStride[S9_TRANSPOSE_MAX_GROUP];
    uint64_t outerTotal;
    uint64_t taskCount;

    uint64_t runLen;
    uint64_t runChunk;
    uint64_t runChunkCount;
    uint64_t midLen;
    uint64_t midInStride;
    uint64_t midBatch;
    uint64_t midChunkCount;

    uint64_t lenA;
    uint64_t lenB;
    uint64_t tileA;
    uint64_t tileB;
    uint64_t tileCountA;
    uint64_t tileCountB;
    uint64_t inStrideB;
    uint64_t outStrideA;

    uint32_t outerCount;
    uint32_t regime;
    uint32_t blockDim;

    uint32_t wholeTile;

    uint32_t storeGroup;

    uint32_t aFastShift;

    uint32_t bypassOutput;

    uint32_t fullRowStore;
};

#endif
