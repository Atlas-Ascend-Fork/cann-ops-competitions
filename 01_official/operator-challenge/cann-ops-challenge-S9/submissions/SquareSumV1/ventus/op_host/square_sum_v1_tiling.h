/**
 * SquareSumV1 主机侧与设备侧共享的分块数据结构。
 * 维度按相邻归约或保留分组规范化，字段记录各组长度、步长及计算路径参数。
 */

#ifndef S9_SQUARE_SUM_V1_TILING_H
#define S9_SQUARE_SUM_V1_TILING_H

#include <cstdint>

constexpr uint32_t S9_SQ_MAX_RANK = 8;
constexpr uint32_t S9_SQ_MAX_GROUP = 8;

struct SquareSumV1TilingData {
    uint64_t groupLen[S9_SQ_MAX_GROUP];
    uint64_t groupStride[S9_SQ_MAX_GROUP];
    uint64_t groupIsReduce[S9_SQ_MAX_GROUP];
    uint64_t innerLen;
    uint64_t outCount;
    uint64_t prefixOutCount;
    uint64_t prefixRedCount;
    uint64_t innerChunk;
    uint64_t innerChunkCount;
    uint64_t taskCount;

    uint64_t redSegCount;
    uint64_t redSegLen;

    uint64_t keepRedSeg;
    uint64_t keepRedSegLen;
    uint32_t groupCount;
    uint32_t innerIsReduce;
    uint32_t blockDim;
    uint32_t tileElements;
    uint32_t slabScratch;
};

#endif
