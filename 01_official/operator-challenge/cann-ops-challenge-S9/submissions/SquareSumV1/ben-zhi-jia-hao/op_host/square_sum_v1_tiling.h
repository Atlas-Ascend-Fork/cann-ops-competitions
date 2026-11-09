#ifndef S9_SQUARE_SUM_V1_TILING_H
#define S9_SQUARE_SUM_V1_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SquareSumV1TilingData)
// --- canonical dimension description ---
TILING_DATA_FIELD_DEF(uint32_t, innerIsReduce);  // 1: innermost contiguous run is reduced
TILING_DATA_FIELD_DEF(uint32_t, nOuterK);        // kept groups outside the innermost run
TILING_DATA_FIELD_DEF(uint32_t, nOuterR);        // reduced groups outside the innermost run
TILING_DATA_FIELD_DEF(uint32_t, usedCores);
TILING_DATA_FIELD_DEF(uint32_t, redChunks);   // reduction split factor (>1 => 2 stage)
TILING_DATA_FIELD_DEF(uint32_t, splitRuns);   // 1: redChunks splits runs, 0: splits innerLen
TILING_DATA_FIELD_DEF(uint32_t, rowBatch);    // rows fused into one DMA on the row routes
TILING_DATA_FIELD_DEF(uint32_t, rowPitch);    // UB element pitch per row (ROWS_SHORT)
TILING_DATA_FIELD_DEF(uint32_t, accLen);      // fp32 accumulator length (elements)
TILING_DATA_FIELD_DEF(uint32_t, ubChunk);     // max input elements per UB tile
TILING_DATA_FIELD_DEF(uint32_t, workLen);     // ReduceSum shared tmp length (floats)
TILING_DATA_FIELD_DEF(uint32_t, outStage);    // output staging capacity (elements)
TILING_DATA_FIELD_DEF(uint32_t, runGroup);    // >0: contiguous run-span route, runs folded per accumulator
TILING_DATA_FIELD_DEF(uint32_t, runLanes);    // runGroup * innerLen (accumulator width of that route)
TILING_DATA_FIELD_DEF(uint32_t, runFold);     // partials left to the scalar residual fold (1 = all vector)
TILING_DATA_FIELD_DEF(uint32_t, tmpLen);      // tmpBuf floats (row route needs far fewer than accLen)
TILING_DATA_FIELD_DEF(uint64_t, innerLen);    // innermost contiguous run length
TILING_DATA_FIELD_DEF(uint64_t, colPiece);    // output columns per independent unit
TILING_DATA_FIELD_DEF(uint64_t, colChunks);   // number of column pieces per tile
TILING_DATA_FIELD_DEF(uint64_t, outTiles);    // product of outer kept group sizes
TILING_DATA_FIELD_DEF(uint64_t, redPerTile);  // product of outer reduced group sizes (runs)
TILING_DATA_FIELD_DEF(uint64_t, totalOut);    // output element count
TILING_DATA_FIELD_DEF(uint64_t, indepUnits);  // independent (workspace free) work units
TILING_DATA_FIELD_DEF(uint64_t, outAligned);  // workspace row pitch for partial results
TILING_DATA_FIELD_DEF_ARR(uint64_t, 4, kSize);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 4, kStride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 4, rSize);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 4, rStride);
// Packed header for KEY_ROWS_HDR / KEY_ROWS_BARE. Append fields rather than
// inserting them so existing offsets remain stable. These routes read oneHdr
// as their only tiling word; PackOneHdr documents its bit layout and bounds.
// Other routes leave it zero.
TILING_DATA_FIELD_DEF(uint64_t, oneHdr);
// Trailing diagnostic iteration count; appending preserves earlier offsets.
// Zero disables the diagnostic delay. See diag_probe.h for the encoding.
TILING_DATA_FIELD_DEF(uint64_t, diagSpin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareSumV1, SquareSumV1TilingData)
}  // namespace optiling

#endif  // S9_SQUARE_SUM_V1_TILING_H
