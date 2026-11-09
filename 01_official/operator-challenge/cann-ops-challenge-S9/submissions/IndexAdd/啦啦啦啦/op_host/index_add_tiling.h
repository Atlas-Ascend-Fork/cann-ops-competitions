
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, outerSize);
  TILING_DATA_FIELD_DEF(uint64_t, innerSize);
  TILING_DATA_FIELD_DEF(uint32_t, axisSize);
  TILING_DATA_FIELD_DEF(uint32_t, indexSize);
  TILING_DATA_FIELD_DEF(uint32_t, tilesPerOuter);
  TILING_DATA_FIELD_DEF(uint32_t, tileElements);
  TILING_DATA_FIELD_DEF(uint32_t, sourceChunk);
  TILING_DATA_FIELD_DEF(uint32_t, outerBatch);
  TILING_DATA_FIELD_DEF(uint32_t, batchAxisChunk);
  TILING_DATA_FIELD_DEF(uint32_t, sourceTranspose);
  TILING_DATA_FIELD_DEF(uint32_t, sourceBufferCount);
  TILING_DATA_FIELD_DEF(uint32_t, atomicAdd);
  TILING_DATA_FIELD_DEF(uint32_t, typeBytes);
  TILING_DATA_FIELD_DEF(uint32_t, promoteBfloat16);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}
