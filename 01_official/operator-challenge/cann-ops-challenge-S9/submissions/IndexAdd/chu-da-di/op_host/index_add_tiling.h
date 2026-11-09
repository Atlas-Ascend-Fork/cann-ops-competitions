#ifndef CONTEST_INDEX_ADD_TILING_H
#define CONTEST_INDEX_ADD_TILING_H

#include <cstdint>

// Keep this structure plain and shared by Host and Kernel.  CANN 8.5's
// REGISTER_TILING_DEFAULT/GET_TILING_DATA pair copies it byte-for-byte.
struct IndexAddTilingData {
    uint64_t outer;
    uint64_t destinationAxis;
    uint64_t indexCount;
    uint64_t inner;
    uint64_t selfElements;
    uint64_t sourceElements;
    uint64_t tileInner;
    uint64_t tileInnerAligned;
    uint64_t innerTileCount;
    uint64_t taskCount;
    uint64_t tasksPerCore;
    uint32_t extraCoreTasks;
    uint32_t usedCoreNum;
    uint32_t mode;
    uint32_t dtypeBytes;
    uint32_t accumulatorRows;
    uint32_t indexBufferBytes;
    uint32_t indexVectorElements;
    uint32_t matchMaskBufferBytes;
    uint32_t rawBufferBytes;
    uint32_t castBufferBytes;
    uint32_t accumulatorBufferBytes;
    uint32_t sourceCacheBufferBytes;
    uint32_t rowPresenceBufferBytes;
    uint32_t inverseWaveCapacity;
    uint32_t inverseAlignedDestinations;
    uint32_t inverseSourceSentinel;
    uint32_t inverseConvertBufferBytes;
    uint32_t crossOuterRows;
    uint32_t crossLanes;
    uint32_t crossSourceTile;
    uint32_t crossActiveCapacity;
    uint32_t inverseBatchRows;
    uint32_t inverseSourceRowElements;
    uint32_t inverseWideRowElements;
    uint32_t csrHeadBytes;
    uint32_t csrPrefixBytes;
    uint32_t csrBatchSize;
    uint32_t csrSourceSlotBytes;
    uint32_t plan[48];
    uint64_t sourceL2BypassOffset;
};

// TILING_KEY_IS is parsed by the Ascend C precompiler before the C++
// compiler runs.  CANN 8.5 accepts numeric macros/literals there, but does
// not resolve scoped/unscoped enum identifiers (the generated binary then
// lacks the requested entry and fails at runtime with
// BinaryGetFunctionByEntry).  Keep these keys as preprocessor constants and
// retain the uint32_t fields in the plain tiling structure above.
#define INDEX_ADD_EMPTY 0
#define INDEX_ADD_SLAB_OWNER 1
#define INDEX_ADD_ROW_OWNER 2
#define INDEX_ADD_DESTINATION_SLAB_OWNER 3
#define INDEX_ADD_SPARSE_ROW_OWNER 4
#define INDEX_ADD_CONTIGUOUS_AXIS_OWNER 5
#define INDEX_ADD_ADAPTIVE_AXIS_OWNER 6
#define INDEX_ADD_LINEAR_AXIS_OWNER 7
#define INDEX_ADD_ORDERED_CSR 8
#define INDEX_ADD_FILTERED_OWNER 9
#define INDEX_ADD_TILED_AXIS 10
#define INDEX_ADD_COOPERATIVE_AXIS 13
#define INDEX_ADD_ATOMIC_SOURCE 14



#define INDEX_ADD_LINKED_AXIS 15

#define INDEX_ADD_CACHED_INT8_ROWS 16


#define INDEX_ADD_FLAT_WAVES 19



#define INDEX_ADD_RANKED_WAVES 25

#define INDEX_ADD_BLOCK_WAVES 26

#define INDEX_ADD_RANKED_PIPE 27


#define INDEX_ADD_TRANSPOSE_WAVES 30

#define INDEX_ADD_ORDERED_WAVES 34
#define INDEX_ADD_RESIDENT_BF16 35
#define INDEX_ADD_STREAMING_INT32 39

#endif  // CONTEST_INDEX_ADD_TILING_H
