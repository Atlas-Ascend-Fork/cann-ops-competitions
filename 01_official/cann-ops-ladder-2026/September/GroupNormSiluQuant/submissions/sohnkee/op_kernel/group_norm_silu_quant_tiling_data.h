/*!
 * \file group_norm_silu_quant_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _GROUPNORMSILUQUANT_TILING_DATA_H_
#define _GROUPNORMSILUQUANT_TILING_DATA_H_

struct GroupNormSiluQuantTilingData {
    int64_t batch = 0;
    int64_t channels = 0;
    int64_t height = 0;
    int64_t width = 0;
    int64_t numGroups = 0;
    int64_t channelsPerGroup = 0;
    int64_t spatial = 0;
    int64_t groupElements = 0;
    int64_t totalGroups = 0;
    int64_t tileElements = 0;
    int64_t workerNum = 0;
    int64_t quantScaleCount = 0;
    int64_t parameterBytes = 0;
    int64_t exactResidentCapacity = 0;
    float epsilon = 1.0e-5f;
    int32_t activateSilu = 1;
};
#endif
