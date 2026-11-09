#ifndef GREATER_TILING_H
#define GREATER_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GreaterTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, lowSize);
  TILING_DATA_FIELD_DEF(uint32_t, middleSize);
  TILING_DATA_FIELD_DEF(uint32_t, highSize);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, broadcastIsX1);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, x1Dims);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, x2Dims);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, outDims);
  TILING_DATA_FIELD_DEF_ARR(float, 8, inverseStrides);
  TILING_DATA_FIELD_DEF_ARR(float, 8, inversePeriods);
  TILING_DATA_FIELD_DEF_ARR(int64_t, 3, broadcastOffsets);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Greater, GreaterTilingData)

}

#endif
