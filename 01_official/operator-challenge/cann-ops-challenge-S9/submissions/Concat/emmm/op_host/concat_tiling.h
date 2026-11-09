#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t CONCAT_MAX_INPUTS = 256;
constexpr uint32_t CONCAT_LAUNCH_PADDING_WORDS = 91;
constexpr uint32_t CONCAT_PACKED_WIDTH_WORDS = CONCAT_MAX_INPUTS / 4;
constexpr uint32_t CONCAT_BALANCED_SCHEDULE_WORDS = 10;
constexpr uint32_t CONCAT_DYNAMIC_SLAB_HEADER_WORD =
    CONCAT_PACKED_WIDTH_WORDS + CONCAT_BALANCED_SCHEDULE_WORDS;
constexpr uint32_t CONCAT_DYNAMIC_SLAB_BOUNDARY_WORDS = 16;
static_assert(CONCAT_DYNAMIC_SLAB_HEADER_WORD + 1U +
    CONCAT_DYNAMIC_SLAB_BOUNDARY_WORDS <= CONCAT_LAUNCH_PADDING_WORDS);

BEGIN_TILING_DATA_DEF(ConcatTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileBytes);
  TILING_DATA_FIELD_DEF(uint64_t, totalBytes);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowBytes);
  TILING_DATA_FIELD_DEF(uint64_t, blockFactorBytes);
  TILING_DATA_FIELD_DEF(uint64_t, outerRows);
  TILING_DATA_FIELD_DEF(uint64_t, blockFactorRows);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerBatch);
  TILING_DATA_FIELD_DEF(uint32_t, elementBytes);
  TILING_DATA_FIELD_DEF(uint32_t, concatDim);
  TILING_DATA_FIELD_DEF(uint64_t, innerElements);
  TILING_DATA_FIELD_DEF(uint32_t, uniformInputRowBytes);
  // Preserve the 448-byte launch footprint measured fastest on Ascend 910B.
  TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_LAUNCH_PADDING_WORDS, launchPadding);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}  // namespace optiling

#endif  // CONCAT_TILING_H
