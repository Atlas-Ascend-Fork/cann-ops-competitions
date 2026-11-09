#ifndef SQUARE_SUM_V1_TILING_H
#define SQUARE_SUM_V1_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SquareSumV1TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, outerLength);
  TILING_DATA_FIELD_DEF(uint32_t, reduceLength);
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreItems);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreCount);
  TILING_DATA_FIELD_DEF(uint32_t, alignedReduceLength);
  TILING_DATA_FIELD_DEF(uint32_t, inputSize);
  TILING_DATA_FIELD_DEF(uint32_t, outputSize);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, axisMask);
  TILING_DATA_FIELD_DEF(uint32_t, genericTileCols);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, dims);
  TILING_DATA_FIELD_DEF(uint64_t, wideInputSize);
  TILING_DATA_FIELD_DEF(uint64_t, wideOutputSize);
  TILING_DATA_FIELD_DEF(uint64_t, wideReduceLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumV1, SquareSumV1TilingData)

}

#endif
