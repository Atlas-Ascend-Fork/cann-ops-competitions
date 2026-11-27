/*!
 * \file quant_batch_mat_mul_v3_tiling_data.h
 * \brief Tiling data for QuantBatchMatMulV3.
 */

#ifndef _QUANTBATCHMATMULV3_TILING_DATA_H_
#define _QUANTBATCHMATMULV3_TILING_DATA_H_

#include "register/tilingdata_base.h"
#if __has_include("tiling/tiling_api.h")
#include "tiling/tiling_api.h"
#else
#include "tiling_api.h"
#endif

namespace optiling {
BEGIN_TILING_DATA_DEF(QuantBatchMatMulV3TilingData)
    TILING_DATA_FIELD_DEF(int64_t, totalNum);
    TILING_DATA_FIELD_DEF(int64_t, blockFactor);
    TILING_DATA_FIELD_DEF(int64_t, ubFactor);
    TILING_DATA_FIELD_DEF(int64_t, m);
    TILING_DATA_FIELD_DEF(int64_t, n);
    TILING_DATA_FIELD_DEF(int64_t, k);
    TILING_DATA_FIELD_DEF(int64_t, x1Dim0);
    TILING_DATA_FIELD_DEF(int64_t, x1Dim1);
    TILING_DATA_FIELD_DEF(int64_t, x2Dim0);
    TILING_DATA_FIELD_DEF(int64_t, x2Dim1);
    TILING_DATA_FIELD_DEF(int64_t, scaleLen);
    TILING_DATA_FIELD_DEF(int64_t, transposeX1);
    TILING_DATA_FIELD_DEF(int64_t, transposeX2);
    TILING_DATA_FIELD_DEF(int64_t, nTileNum);
    TILING_DATA_FIELD_DEF(int64_t, colTile);
    TILING_DATA_FIELD_DEF(uint64_t, workspacePerCore);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, matmulTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(QuantBatchMatMulV3, QuantBatchMatMulV3TilingData)
} // namespace optiling

#endif
