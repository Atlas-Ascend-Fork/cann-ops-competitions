#ifndef GREATER_TILING_H
#define GREATER_TILING_H
#include "greater_tiling_fields.h"
#include "register/tilingdata_base.h"

namespace optiling {
constexpr uint32_t GT_MAX_DIM = GT_MAX_DIM_VALUE;

// Keep field order, types and array extents aligned with GT_TILING_FIELDS.
// Expand the vendor field macros on separate lines so the registered layout
// matches the host Plan and kernel readers.
BEGIN_TILING_DATA_DEF(GreaterTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, perCore);    // output elements handled by one core
    TILING_DATA_FIELD_DEF(uint64_t, totalNum);   // total output elements
    TILING_DATA_FIELD_DEF(uint64_t, innerLen);   // extent of the innermost collapsed dim
    TILING_DATA_FIELD_DEF(uint64_t, rowNum);     // totalNum / innerLen
    TILING_DATA_FIELD_DEF(uint32_t, ubSize);     // usable UB bytes, queried from the platform
    TILING_DATA_FIELD_DEF(uint32_t, usedCores);
    TILING_DATA_FIELD_DEF(uint32_t, dimNum);     // number of collapsed dims (D)
    TILING_DATA_FIELD_DEF(uint32_t, innerMode);  // 0 = both dense, 1 = x1 scalar per row, 2 = x2 scalar per row
    TILING_DATA_FIELD_DEF(uint32_t, cacheMode);  // bit0: x1 row block invariant, bit1: x2 row block invariant
    TILING_DATA_FIELD_DEF(uint32_t, pad0);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, GT_MAX_DIM, shape);          // collapsed output shape
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 2 * GT_MAX_DIM, strides);    // [2*i] = x1, [2*i+1] = x2
    TILING_DATA_FIELD_DEF(uint64_t, tilingKey);  // same value passed to SetTilingKey
    TILING_DATA_FIELD_DEF(uint64_t, diagCode);   // diagnostic delay probe; op_host/greater_diag.h
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Greater, GreaterTilingData)
}  // namespace optiling
#endif  // GREATER_TILING_H
