#ifndef SQUARE_SUM_V1_TILING_H
#define SQUARE_SUM_V1_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, mode);        // 0=ROW 1=GENERAL 2=RPAR 3=COLUMN 4=MODE4
  TILING_DATA_FIELD_DEF(uint32_t, tileElems);
  TILING_DATA_FIELD_DEF(uint32_t, outer);       // L
  TILING_DATA_FIELD_DEF(uint32_t, reduceLen);   // n (ROW) / m (COL)
  TILING_DATA_FIELD_DEF(uint32_t, inner);       // 1 (ROW) / n (COL)
  // ROW
  TILING_DATA_FIELD_DEF(uint32_t, outerPerCore);
  // COLUMN
  TILING_DATA_FIELD_DEF(uint32_t, colCores);
  TILING_DATA_FIELD_DEF(uint32_t, mPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, innerP);
  // RPAR
  TILING_DATA_FIELD_DEF(uint32_t, O);
  TILING_DATA_FIELD_DEF(uint32_t, Oslot);
  TILING_DATA_FIELD_DEF(uint32_t, rCores);
  TILING_DATA_FIELD_DEF(uint32_t, rPerCore);
  // GENERAL
  TILING_DATA_FIELD_DEF(uint32_t, unitReduced);
  TILING_DATA_FIELD_DEF(uint32_t, unit);
  TILING_DATA_FIELD_DEF(uint32_t, klProd);
  TILING_DATA_FIELD_DEF(uint32_t, rlProd);
  TILING_DATA_FIELD_DEF(uint32_t, koPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, nKL);
  TILING_DATA_FIELD_DEF(uint32_t, nRL);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, klDim);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, klStr);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, rlDim);
  TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, rlStr);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumV1, TilingData)
}  // namespace optiling
#endif  // SQUARE_SUM_V1_TILING_H
