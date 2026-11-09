#ifndef S9_CONCAT_TILING_H_
#define S9_CONCAT_TILING_H_

#include "register/tilingdata_base.h"

#include <cstdint>

// Nnopbase accepts at most 256 tensors in a dynamic input.  The spreadsheet
// bounds every input extent by 10000, so uint16_t stores every legal extent
// while covering that complete runtime-supported tensor-list range.
constexpr uint32_t CONCAT_MAX_INPUTS = 256U;

namespace optiling {

BEGIN_TILING_DATA_DEF(ConcatTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, totalElements);
    TILING_DATA_FIELD_DEF(uint64_t, inner);
    TILING_DATA_FIELD_DEF(uint64_t, outputAxis);
    TILING_DATA_FIELD_DEF(uint32_t, inputCount);
    TILING_DATA_FIELD_DEF(uint32_t, dtypeBytes);
    TILING_DATA_FIELD_DEF_ARR(uint16_t, CONCAT_MAX_INPUTS, inputAxis);
    TILING_DATA_FIELD_DEF_ARR(uint32_t, CONCAT_MAX_INPUTS, inputRowStrideBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)

}  // namespace optiling

#endif  // S9_CONCAT_TILING_H_
