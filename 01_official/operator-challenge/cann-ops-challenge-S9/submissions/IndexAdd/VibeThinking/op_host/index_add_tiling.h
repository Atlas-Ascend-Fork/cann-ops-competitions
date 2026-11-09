#ifndef INDEX_ADD_TILING_H
#define INDEX_ADD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, outer);
  TILING_DATA_FIELD_DEF(uint32_t, selfLen);
  TILING_DATA_FIELD_DEF(uint32_t, inner);
  TILING_DATA_FIELD_DEF(uint32_t, indexLen);
  TILING_DATA_FIELD_DEF(uint32_t, unit);
  TILING_DATA_FIELD_DEF(uint32_t, path);
  TILING_DATA_FIELD_DEF(uint32_t, slabRows);
  TILING_DATA_FIELD_DEF(uint32_t, selfTile);
  TILING_DATA_FIELD_DEF(uint32_t, sourceTile);
  TILING_DATA_FIELD_DEF(uint32_t, l2Low);
  TILING_DATA_FIELD_DEF(uint32_t, l2High);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}

#endif
