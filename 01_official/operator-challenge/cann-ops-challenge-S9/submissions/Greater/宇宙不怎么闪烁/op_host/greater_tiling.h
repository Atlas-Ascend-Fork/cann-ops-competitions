#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GreaterTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, total_size);
    TILING_DATA_FIELD_DEF(uint32_t, block_elems);
    TILING_DATA_FIELD_DEF(uint32_t, total_blocks);
    TILING_DATA_FIELD_DEF(uint32_t, tile_elems);
    TILING_DATA_FIELD_DEF(uint32_t, elem_bytes);
    TILING_DATA_FIELD_DEF(uint32_t, same_shape);
    TILING_DATA_FIELD_DEF(uint32_t, self_dense);
    TILING_DATA_FIELD_DEF(uint32_t, other_dense);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, out_dims);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, self_dims);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, other_dims);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, out_strides);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, self_strides);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, other_strides);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Greater, GreaterTilingData)
}  // namespace optiling
