#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SquareSumV1TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, mode);
    TILING_DATA_FIELD_DEF(uint32_t, total_size);
    TILING_DATA_FIELD_DEF(uint32_t, input_size);
    TILING_DATA_FIELD_DEF(uint32_t, output_size);
    TILING_DATA_FIELD_DEF(uint32_t, block_elems);
    TILING_DATA_FIELD_DEF(uint32_t, total_blocks);
    TILING_DATA_FIELD_DEF(uint32_t, tile_elems);
    TILING_DATA_FIELD_DEF(uint32_t, acc_tile_elems);
    TILING_DATA_FIELD_DEF(uint32_t, outer_size);
    TILING_DATA_FIELD_DEF(uint32_t, reduce_size);
    TILING_DATA_FIELD_DEF(uint32_t, inner_size);
    TILING_DATA_FIELD_DEF(uint32_t, task_num);
    TILING_DATA_FIELD_DEF(uint32_t, bypass_l2);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, in_dims);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, out_dims);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, in_strides);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, out_strides);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 5, reduce_mask);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumV1, SquareSumV1TilingData)
}  // namespace optiling
