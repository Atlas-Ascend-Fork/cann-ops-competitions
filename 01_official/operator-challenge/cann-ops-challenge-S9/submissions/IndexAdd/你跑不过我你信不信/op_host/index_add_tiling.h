#ifndef INDEX_ADD_TILING_H
#define INDEX_ADD_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexAddTilingData)
  // Challenge shapes fit in uint32.  Keep only values that cannot be derived
  // cheaply in the kernel from the shape, dtype and launch block count.
  TILING_DATA_FIELD_DEF(uint32_t, outer);      // prod(dims before dim)
  TILING_DATA_FIELD_DEF(uint32_t, dimLen);     // dims[dim]
  TILING_DATA_FIELD_DEF(uint32_t, inner);      // prod(dims after dim)
  TILING_DATA_FIELD_DEF(uint32_t, m);          // index length (source dim-axis)
  TILING_DATA_FIELD_DEF(uint32_t, accInnerTile);    // fp16/bf16 inner chunk (fp32 acc)
  // key 7 (vectorized transpose, inner==1 outer>1): B outer-rows -> lanes.
  TILING_DATA_FIELD_DEF(uint32_t, k7B);             // lane block (mult of 8)
  TILING_DATA_FIELD_DEF(uint32_t, k7Ntile);         // N-tile for input/output transpose (mult 8)
  TILING_DATA_FIELD_DEF(uint32_t, k7Mtile);         // M-tile for source transpose (mult 8)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddTilingData)
}  // namespace optiling
#endif  // INDEX_ADD_TILING_H
