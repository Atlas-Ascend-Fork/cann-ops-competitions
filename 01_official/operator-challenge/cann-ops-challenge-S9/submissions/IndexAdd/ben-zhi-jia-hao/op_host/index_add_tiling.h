#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, outer);
  TILING_DATA_FIELD_DEF(uint64_t, axisSize);
  TILING_DATA_FIELD_DEF(uint64_t, inner);
  TILING_DATA_FIELD_DEF(uint64_t, indexCount);
  TILING_DATA_FIELD_DEF(uint64_t, total);
  TILING_DATA_FIELD_DEF(uint32_t, listBytes);
  TILING_DATA_FIELD_DEF(uint32_t, dtype);
  TILING_DATA_FIELD_DEF(uint32_t, parallelPlan);
  TILING_DATA_FIELD_DEF(uint32_t, bf16Batch);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}
