
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SquareSumV1TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, axis_count);
  TILING_DATA_FIELD_DEF(uint32_t, axis_mask);
  TILING_DATA_FIELD_DEF(uint32_t, total_size);
  TILING_DATA_FIELD_DEF(uint32_t, output_size);
  TILING_DATA_FIELD_DEF(uint32_t, outer_size);
  TILING_DATA_FIELD_DEF(uint32_t, reduce_size);
  TILING_DATA_FIELD_DEF(uint32_t, inner_size);
  TILING_DATA_FIELD_DEF(uint32_t, tile_elems);
  TILING_DATA_FIELD_DEF(uint32_t, row_group);
  TILING_DATA_FIELD_DEF(uint32_t, workspace_offset);
  TILING_DATA_FIELD_DEF(uint32_t, partial_count);
  TILING_DATA_FIELD_DEF(uint32_t, partial_stride);
  TILING_DATA_FIELD_DEF(uint32_t, dtype_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, reduce_axis0);
  TILING_DATA_FIELD_DEF(uint32_t, reduce_axis1);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, dims);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, strides);
  TILING_DATA_FIELD_DEF(uint32_t, separated_fp16_layout_mode);
  TILING_DATA_FIELD_DEF(uint32_t, separated_fp16_layout_rows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumV1, SquareSumV1TilingData)
}
