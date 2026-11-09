// Shared list of plan fields serialized into kernel tiling data.
// Every plan-to-tiling copy expands these lists so declarations and assignments
// stay synchronized. Fields with different host and kernel names remain explicit.
// This header intentionally has no include dependencies.

#ifndef ASCEND_S9_TRANSPOSE_FIELDS_H
#define ASCEND_S9_TRANSPOSE_FIELDS_H

// X(type, name)
#define S9_TRANSPOSE_TILING_SCALARS(X) \
    X(uint64_t, numel)                 \
    X(uint64_t, rowCount)              \
    X(uint64_t, rowLen)                \
    X(uint64_t, srcBlkPitch)           \
    X(uint32_t, outerRank)             \
    X(uint32_t, blkPerRow)             \
    X(uint32_t, blkLenElems)           \
    X(uint32_t, rowsPerBatch)          \
    X(uint32_t, chunkBlks)             \
    X(uint32_t, slotBytes)             \
    X(uint32_t, tilePrefetch)          \
    X(uint64_t, tileK)                 \
    X(uint64_t, dstRowStride)          \
    X(uint32_t, tileMT)                \
    X(uint32_t, tileKT)                \
    X(uint32_t, dtypeSize)             \
    X(uint32_t, foldLen)               \
    X(uint64_t, chunkElems)            \
    X(uint64_t, foldExt)               \
    X(uint64_t, foldSrcStride)         \
    X(uint64_t, foldDstRowStride)

// X(elementType, name); every array is 8 wide (kPlanMaxOdo).
#define S9_TRANSPOSE_TILING_ARRAYS(X) \
    X(uint64_t, outerShape)           \
    X(uint64_t, outerInStride)        \
    X(uint64_t, outerOutStride)

#endif  // ASCEND_S9_TRANSPOSE_FIELDS_H
