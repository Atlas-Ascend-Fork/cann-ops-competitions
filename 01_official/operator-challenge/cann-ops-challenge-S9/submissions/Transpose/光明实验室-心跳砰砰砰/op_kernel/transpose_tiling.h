#ifndef TRANSPOSE_TILING_H
#define TRANSPOSE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(TransposeTilingData)
  TILING_DATA_FIELD_DEF(int64_t, totalLength);
  TILING_DATA_FIELD_DEF(int64_t, dim0);
  TILING_DATA_FIELD_DEF(int64_t, dim1);
  TILING_DATA_FIELD_DEF(int64_t, rank);
  TILING_DATA_FIELD_DEF(int64_t, blockRows);
  TILING_DATA_FIELD_DEF(int64_t, smallCoreTasks);
  TILING_DATA_FIELD_DEF(int64_t, bigCoreCount);
  TILING_DATA_FIELD_DEF(int64_t, schedule);
  TILING_DATA_FIELD_DEF(int64_t, in0);
  TILING_DATA_FIELD_DEF(int64_t, in1);
  TILING_DATA_FIELD_DEF(int64_t, in2);
  TILING_DATA_FIELD_DEF(int64_t, in3);
  TILING_DATA_FIELD_DEF(int64_t, in4);
  TILING_DATA_FIELD_DEF(int64_t, in5);
  TILING_DATA_FIELD_DEF(int64_t, in6);
  TILING_DATA_FIELD_DEF(int64_t, in7);
  TILING_DATA_FIELD_DEF(int64_t, perm0);
  TILING_DATA_FIELD_DEF(int64_t, perm1);
  TILING_DATA_FIELD_DEF(int64_t, perm2);
  TILING_DATA_FIELD_DEF(int64_t, perm3);
  TILING_DATA_FIELD_DEF(int64_t, perm4);
  TILING_DATA_FIELD_DEF(int64_t, perm5);
  TILING_DATA_FIELD_DEF(int64_t, perm6);
  TILING_DATA_FIELD_DEF(int64_t, perm7);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)

}

#endif
