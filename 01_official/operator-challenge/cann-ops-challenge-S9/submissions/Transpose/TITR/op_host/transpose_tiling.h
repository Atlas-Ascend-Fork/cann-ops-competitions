#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(TransposeTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, total);
    TILING_DATA_FIELD_DEF(uint64_t, inner);
    TILING_DATA_FIELD_DEF(uint64_t, innerStride);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 5, stride);
    TILING_DATA_FIELD_DEF(uint32_t, rank);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, shape);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, perm);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, outShape);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 16, p);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)
}
