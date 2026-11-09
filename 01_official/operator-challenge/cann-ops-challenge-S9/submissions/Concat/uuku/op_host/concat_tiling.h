#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConcatTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, input_count);
    TILING_DATA_FIELD_DEF(uint32_t, outer_size);
    TILING_DATA_FIELD_DEF(uint32_t, inner_size);
    TILING_DATA_FIELD_DEF(uint32_t, output_axis_size);
    TILING_DATA_FIELD_DEF(uint32_t, total_elems);
    TILING_DATA_FIELD_DEF(uint32_t, block_elems);
    TILING_DATA_FIELD_DEF(uint32_t, total_blocks);
    TILING_DATA_FIELD_DEF(uint32_t, tile_elems);
    TILING_DATA_FIELD_DEF(uint32_t, elem_bytes);
    TILING_DATA_FIELD_DEF(uint32_t, ub_bytes);
    TILING_DATA_FIELD_DEF(uint32_t, aligned_group_count);
    TILING_DATA_FIELD_DEF(uint32_t, phase_boundary_count);
    TILING_DATA_FIELD_DEF(uint32_t, phase_unit_count);
    TILING_DATA_FIELD_DEF(uint32_t, phase_owner_word_count);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 1792, axis_lens);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 1792, axis_offsets);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, 1792, schedule_order);
    // High words are consumed only when the wide-geometry flag is set.
    // Existing scalar and array offsets remain unchanged.
    TILING_DATA_FIELD_DEF(uint32_t, outer_size_hi);
    TILING_DATA_FIELD_DEF(uint32_t, inner_size_hi);
    TILING_DATA_FIELD_DEF(uint32_t, total_elems_hi);
    // Only generic dispatch reads this host-computed DMA/single-input contract.
    TILING_DATA_FIELD_DEF(uint32_t, linear_plan);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}  // namespace optiling
