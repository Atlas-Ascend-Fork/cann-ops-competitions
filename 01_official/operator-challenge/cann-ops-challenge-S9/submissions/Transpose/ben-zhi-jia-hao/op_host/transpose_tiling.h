#ifndef ASCEND_S9_TRANSPOSE_TILING_H
#define ASCEND_S9_TRANSPOSE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// Field order is shared with the kernel tiling layout. Keep declarations
// and the readers of individual fields synchronized.
BEGIN_TILING_DATA_DEF(TransposeTilingData)
    // --- CACHE LINE 0 (bytes 0..63) -------------------------------------
    // Field order affects how many GM cache lines a route reads. Keep each
    // RouteNeeds field set within as few lines as practical when extending the layout.
    TILING_DATA_FIELD_DEF(uint64_t, numel);          // 0x00  all
    TILING_DATA_FIELD_DEF(uint64_t, rowCount);       // 0x08  all
    TILING_DATA_FIELD_DEF(uint64_t, rowLen);         // 0x10  all
    TILING_DATA_FIELD_DEF(uint64_t, srcBlkPitch);    // 0x18  kPitch (keys 2,3)
    TILING_DATA_FIELD_DEF(uint32_t, outerRank);      // 0x20  all
    TILING_DATA_FIELD_DEF(uint32_t, usedCores);      // 0x24  all
    TILING_DATA_FIELD_DEF(uint32_t, blkPerRow);      // 0x28  kRows  (key 2)
    TILING_DATA_FIELD_DEF(uint32_t, blkLenElems);    // 0x2c  kRows  (key 2)
    TILING_DATA_FIELD_DEF(uint32_t, rowsPerBatch);   // 0x30  kStage (keys 2,4)
    TILING_DATA_FIELD_DEF(uint32_t, chunkBlks);      // 0x34  kStage (keys 2,4)
    TILING_DATA_FIELD_DEF(uint32_t, slotBytes);      // 0x38  kStage (keys 2,4)
    TILING_DATA_FIELD_DEF(uint32_t, tilePrefetch);   // 0x3c  kTile  (key 3)

    // --- CACHE LINE 1 (bytes 64..127) -----------------------------------
    TILING_DATA_FIELD_DEF(uint64_t, tileK);          // 0x40  kTile
    TILING_DATA_FIELD_DEF(uint64_t, dstRowStride);   // 0x48  kTile
    TILING_DATA_FIELD_DEF(uint32_t, tileMT);         // 0x50  kTile
    TILING_DATA_FIELD_DEF(uint32_t, tileKT);         // 0x54  kTile
    TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);      // 0x58  (kernel uses sizeof(T))
    TILING_DATA_FIELD_DEF(uint32_t, foldLen);        // 0x5c  kFold  (key 4)
    TILING_DATA_FIELD_DEF(uint64_t, chunkElems);     // 0x60  kBulk  (key 1)
    TILING_DATA_FIELD_DEF(uint64_t, foldExt);        // 0x68  kFold
    TILING_DATA_FIELD_DEF(uint64_t, foldSrcStride);  // 0x70  kFold
    TILING_DATA_FIELD_DEF(uint64_t, foldDstRowStride); // 0x78 kFold

    // --- CACHE LINE 2+ : only read when outerRank > 0 --------------------
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, outerShape);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, outerInStride);
    TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, outerOutStride);
    // The diagnostic delay word is trailing and outside transpose_fields.h.
    // Appending it preserves the offsets of shared planner fields. Only block 0
    // reads it, behind !ASCENDC_CPU_DEBUG; zero disables the delay.
    // See transpose_diag.h for the encoding.
    TILING_DATA_FIELD_DEF(uint64_t, diagSpin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)

}  // namespace optiling

#endif  // ASCEND_S9_TRANSPOSE_TILING_H
