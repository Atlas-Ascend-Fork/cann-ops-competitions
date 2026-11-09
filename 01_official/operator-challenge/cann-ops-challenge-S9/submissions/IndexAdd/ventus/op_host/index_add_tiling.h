/**
 * IndexAdd 主机侧与设备侧共享的分块数据结构。
 * 输入和输出按 outer、axis、inner 展开；字段描述任务划分、索引分桶与工作区布局。
 */

#ifndef S9_INDEX_ADD_TILING_H
#define S9_INDEX_ADD_TILING_H

#include <cstdint>

constexpr uint32_t S9_INDEX_ADD_MAX_RANK = 8;

struct IndexAddTilingData {
    uint64_t outer;
    uint64_t axisLen;
    uint64_t inner;
    uint64_t indexCount;
    uint64_t innerChunk;
    uint64_t innerChunkCount;
    uint64_t axisChunk;
    uint64_t axisChunkCount;
    uint64_t taskCount;
    uint32_t blockDim;
    uint32_t useBucket;

    uint64_t numSlices;
    uint64_t sliceRows;
    uint64_t sliceShift;
    uint64_t bucketCap;
    uint64_t bucketCores;
    uint32_t useResident;
    uint32_t singleSubblock;
};

#endif
