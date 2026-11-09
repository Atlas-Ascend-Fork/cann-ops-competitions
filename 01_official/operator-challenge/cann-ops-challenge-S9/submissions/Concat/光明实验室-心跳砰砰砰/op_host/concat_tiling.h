#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ConcatTilingData)
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 256, splits);
  TILING_DATA_FIELD_DEF_ARR(uint16_t, 21, group_input_bounds);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 21, group_output_offsets);
  TILING_DATA_FIELD_DEF(int64_t, input_count);
  TILING_DATA_FIELD_DEF(int64_t, output_elems);
  TILING_DATA_FIELD_DEF(int64_t, outer_count);
  TILING_DATA_FIELD_DEF(int64_t, inner_span);
  TILING_DATA_FIELD_DEF(int64_t, concat_dim);
  TILING_DATA_FIELD_DEF(int64_t, element_bytes);
  TILING_DATA_FIELD_DEF(int64_t, work_items);
  TILING_DATA_FIELD_DEF(int64_t, work_items_per_core);
  TILING_DATA_FIELD_DEF(int64_t, rows_per_task);
  TILING_DATA_FIELD_DEF(int64_t, tile_elems);
  TILING_DATA_FIELD_DEF(int64_t, schedule_mode);
  TILING_DATA_FIELD_DEF(int64_t, flat_input_index);
  TILING_DATA_FIELD_DEF(int64_t, group_count);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)

}

#endif
