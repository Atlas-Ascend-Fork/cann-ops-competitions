#ifndef TRANSPOSE_TILING_H
#define TRANSPOSE_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "transpose_tiling_rules.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TransposeTilingData)
  TILING_DATA_FIELD_DEF(uint64_t, totalElements);
  TILING_DATA_FIELD_DEF(uint64_t, runElements);
  TILING_DATA_FIELD_DEF(uint64_t, runCount);
  TILING_DATA_FIELD_DEF(uint64_t, groupCount);
  TILING_DATA_FIELD_DEF(uint64_t, blocksPerGroup);
  TILING_DATA_FIELD_DEF(uint64_t, chunksPerUnit);
  TILING_DATA_FIELD_DEF(uint64_t, taskCount);
  TILING_DATA_FIELD_DEF(uint64_t, sourceStepBytes);
  TILING_DATA_FIELD_DEF(uint64_t, batchCount);
  TILING_DATA_FIELD_DEF(uint64_t, matrixRows);
  TILING_DATA_FIELD_DEF(uint64_t, matrixColumns);
  TILING_DATA_FIELD_DEF(uint64_t, matrixPrefixRank);
  TILING_DATA_FIELD_DEF(uint64_t, outputMatrixRowStride);
  TILING_DATA_FIELD_DEF(uint32_t, rank);
  TILING_DATA_FIELD_DEF(uint32_t, prefixRank);
  TILING_DATA_FIELD_DEF(uint32_t, tileElements);
  TILING_DATA_FIELD_DEF(uint32_t, gatherBlocks);
  TILING_DATA_FIELD_DEF(uint32_t, elementSize);
  TILING_DATA_FIELD_DEF(uint32_t, matrixRowsPerTask);
  TILING_DATA_FIELD_DEF(uint32_t, matrixColumnsPerTask);
  TILING_DATA_FIELD_DEF(uint32_t, matrixRowAxis);
  TILING_DATA_FIELD_DEF(uint32_t, matrixColumnAxis);
  TILING_DATA_FIELD_DEF(uint32_t, runAxis);
  TILING_DATA_FIELD_DEF(uint64_t, outputRunStride);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, TRANSPOSE_MAX_RANK, outputDims);
  TILING_DATA_FIELD_DEF_ARR(uint64_t, TRANSPOSE_MAX_RANK, sourceStrides);
  TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, transposeTiling);
  TILING_DATA_FIELD_DEF_STRUCT(ConfusionTransposeTiling, tailTransposeTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)
}  // namespace optiling

#endif  // TRANSPOSE_TILING_H
