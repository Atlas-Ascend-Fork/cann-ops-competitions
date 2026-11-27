/*!
 * \file group_norm_tiling_data.h
 * \brief Runtime tiling data for GroupNorm.
 */

#ifndef GROUP_NORM_TILING_DATA_H
#define GROUP_NORM_TILING_DATA_H

#include <cstdint>

struct GroupNormTilingData {
    int64_t outputFactor = 8192;   // elements handled by one output tile
    int64_t blockFactor = 1;      // groups handled by a regular core
    int64_t ubFactor = 8192;      // elements handled by one statistics tile
    int64_t totalGroups = 0;      // N * numGroups
    int64_t lastCoreGroups = 0;
    int64_t numGroups = 1;
    int64_t channels = 0;
    int64_t channelsPerGroup = 0;
    int64_t spatial = 0;
    int64_t groupElements = 0;
    int64_t channelsAlign = 0;
    float epsilon = 1.0e-5f;
    int32_t statisticsPrefetchMode = 1;  // 1=off, 2=FP32 depth2, 3=2-byte depth1
    int32_t outputSplitWorkers = 1;  // identical moments, disjoint output channel ranges
};

#endif // GROUP_NORM_TILING_DATA_H
