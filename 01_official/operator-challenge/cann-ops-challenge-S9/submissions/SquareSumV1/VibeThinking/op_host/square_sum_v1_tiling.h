#ifndef SQUARE_SUM_V1_TILING_H
#define SQUARE_SUM_V1_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

enum class TilingRoute : uint32_t {
    kWaveSlice = 401,
    kSpanEdge = 402,
    kShortRun = 403,
    kLaneCycle = 404,
    kSplitAligned = 405,
    kSplitPadded = 406,
    kBlockFold = 407,
    kFlatPacked = 408,
    kFlatStaged = 409,
    kRowSingle = 410,
    kRowMany = 411,
    kColumnOwned = 412,
    kColumnStrided = 413,
};

BEGIN_TILING_DATA_DEF(SquareSumV1TilingData)

  TILING_DATA_FIELD_DEF(uint32_t, flags);
  TILING_DATA_FIELD_DEF(uint32_t, reduceCores);
  TILING_DATA_FIELD_DEF(uint32_t, partialPitch);
  TILING_DATA_FIELD_DEF(uint32_t, flatBatch);
  TILING_DATA_FIELD_DEF(uint32_t, slotFloats);
  TILING_DATA_FIELD_DEF(uint32_t, partialBase);
  TILING_DATA_FIELD_DEF(uint32_t, pairBand);
  TILING_DATA_FIELD_DEF(uint32_t, spanBatch);
  TILING_DATA_FIELD_DEF(uint32_t, shortRows);
  TILING_DATA_FIELD_DEF(uint32_t, outputCount);
  TILING_DATA_FIELD_DEF(uint32_t, reduceCount);
  TILING_DATA_FIELD_DEF(uint32_t, runLength);
  TILING_DATA_FIELD_DEF(uint32_t, innerWidth);
  TILING_DATA_FIELD_DEF(uint32_t, contiguousOutputs);
  TILING_DATA_FIELD_DEF(uint32_t, contiguousRows);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, groupSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumV1, SquareSumV1TilingData)
}

#endif
