
#ifndef INDEX_ADD_TILING_H
#define INDEX_ADD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalSize);
  TILING_DATA_FIELD_DEF(uint32_t, blockSize);
  TILING_DATA_FIELD_DEF(uint32_t, outerSize);
  TILING_DATA_FIELD_DEF(uint32_t, dimSize);
  TILING_DATA_FIELD_DEF(uint32_t, indexSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}

#endif
