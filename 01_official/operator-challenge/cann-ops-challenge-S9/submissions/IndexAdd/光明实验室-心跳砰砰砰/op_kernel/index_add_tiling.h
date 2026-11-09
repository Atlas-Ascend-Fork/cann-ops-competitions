#ifndef HARNESS_INDEX_ADD_TILING_H
#define HARNESS_INDEX_ADD_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  TILING_DATA_FIELD_DEF(int64_t, totalElems);
  TILING_DATA_FIELD_DEF(int64_t, dimSize);
  TILING_DATA_FIELD_DEF(int64_t, innerSize);
  TILING_DATA_FIELD_DEF(int64_t, indexRows);
  TILING_DATA_FIELD_DEF(int64_t, tileElems);
  TILING_DATA_FIELD_DEF(int64_t, orderedTileElems);
  TILING_DATA_FIELD_DEF(int64_t, outputTileElems);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)

}

#endif
