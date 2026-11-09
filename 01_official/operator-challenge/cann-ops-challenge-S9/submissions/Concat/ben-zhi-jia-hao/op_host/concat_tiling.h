#ifndef CONCAT_BASELINE_TILING_H
#define CONCAT_BASELINE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConcatTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputCount);
  TILING_DATA_FIELD_DEF(uint32_t, axis);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, reserved);
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, innerBytes);
  TILING_DATA_FIELD_DEF(uint64_t, rowBytes);
  TILING_DATA_FIELD_DEF(uint64_t, totalBytes);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 256, inputRowBytes);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 256, inputOffsets);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 256, columnFirst);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8192, panelPlan);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}
#endif
