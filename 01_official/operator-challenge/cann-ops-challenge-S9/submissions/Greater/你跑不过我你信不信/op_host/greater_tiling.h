#ifndef GREATER_TILING_H
#define GREATER_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, perCore);
  TILING_DATA_FIELD_DEF(uint32_t, tileLen);
  TILING_DATA_FIELD_DEF(uint32_t, inner);
  TILING_DATA_FIELD_DEF(uint32_t, outer);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, innerTile);
  TILING_DATA_FIELD_DEF(uint32_t, bcastFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Greater, TilingData)
}  // namespace optiling
#endif  // GREATER_TILING_H
