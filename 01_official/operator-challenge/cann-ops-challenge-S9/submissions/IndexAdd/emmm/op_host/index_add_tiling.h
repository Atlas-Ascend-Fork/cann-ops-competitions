
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalLength);
  TILING_DATA_FIELD_DEF(uint64_t, innerLength);
  TILING_DATA_FIELD_DEF(uint32_t, axisLength);
  TILING_DATA_FIELD_DEF(uint32_t, indexLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}
