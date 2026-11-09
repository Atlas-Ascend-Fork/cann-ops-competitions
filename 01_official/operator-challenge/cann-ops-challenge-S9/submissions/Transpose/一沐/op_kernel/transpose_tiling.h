#ifndef COMPETITION_TRANSPOSE_TILING_H
#define COMPETITION_TRANSPOSE_TILING_H

#include <cstdint>
#include "register/tilingdata_base.h"

constexpr uint32_t TRANSPOSE_MAX_DIMS = 16;

struct CompetitionTransposeTilingData {
    uint64_t totalElements;
    uint64_t contiguousElements;
    uint64_t taskCount;
    uint64_t chunksPerRow;
    uint64_t inputDims[TRANSPOSE_MAX_DIMS];
    uint64_t outputDims[TRANSPOSE_MAX_DIMS];
    uint64_t inputStrides[TRANSPOSE_MAX_DIMS];
    uint32_t rank;
    uint32_t strategy;
    uint32_t elementBytes;
    uint32_t tileSide;
};

namespace optiling {
BEGIN_TILING_DATA_DEF(TransposeTilingData)
TILING_DATA_FIELD_DEF_ARR(uint8_t, sizeof(CompetitionTransposeTilingData), rawData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)
} // namespace optiling

#endif // COMPETITION_TRANSPOSE_TILING_H
