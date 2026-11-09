#ifndef S9_TRANSPOSE_TILING_H
#define S9_TRANSPOSE_TILING_H
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(TransposeTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, total);
  TILING_DATA_FIELD_DEF(uint64_t, taskCount);
  TILING_DATA_FIELD_DEF(uint64_t, planeRows);
  TILING_DATA_FIELD_DEF(uint64_t, planeCols);
  TILING_DATA_FIELD_DEF(uint64_t, inputRowStride);
  TILING_DATA_FIELD_DEF(uint64_t, outputRowStride);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, mode);
  TILING_DATA_FIELD_DEF(uint32_t, tileRows);
  TILING_DATA_FIELD_DEF(uint32_t, tileCols);
  TILING_DATA_FIELD_DEF(uint32_t, planeGroup);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 64, dims);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 64, inputStrides);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, 64, outputStrides);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)
}
#endif
