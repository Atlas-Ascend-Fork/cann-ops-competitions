#pragma once
#include "register/tilingdata_base.h"
namespace optiling {
BEGIN_TILING_DATA_DEF(ConcatTilingData)
TILING_DATA_FIELD_DEF_ARR(uint64_t, 4096, words);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Concat, ConcatTilingData)
}
