#ifndef S9_CONCAT_TILING_H
#define S9_CONCAT_TILING_H

#include <cstdint>

// ACLNN on CANN 8.5 accepts at most 256 tensors in a dynamic input list.
constexpr uint32_t S9_CONCAT_MAX_INPUTS = 256;

// The kernel is split by data-flow and movement policy.  The 100-series keys
// carry the full 256-entry table; the 1100-series keys carry a compact 16-entry
// uint16 table.  Common TensorLists therefore do not make every AIV read a
// ~1.1 KiB tiling payload before doing a few microseconds of useful work.
constexpr uint64_t S9_CONCAT_TILING_KEY_EMPTY = 100;
constexpr uint64_t S9_CONCAT_TILING_KEY_FLAT_BUFFER = 102;
constexpr uint64_t S9_CONCAT_TILING_KEY_FLAT_QUEUE = 103;
constexpr uint64_t S9_CONCAT_TILING_KEY_GATHER = 104;
constexpr uint64_t S9_CONCAT_TILING_KEY_UNIFORM_GATHER = 105;
constexpr uint64_t S9_CONCAT_TILING_KEY_ROWS_BUFFER = 106;
constexpr uint64_t S9_CONCAT_TILING_KEY_ROWS_QUEUE = 107;
constexpr uint64_t S9_CONCAT_TILING_KEY_DIRECT_ROW_BUFFER = 108;
constexpr uint64_t S9_CONCAT_TILING_KEY_DIRECT_ROW_QUEUE = 109;
constexpr uint64_t S9_CONCAT_TILING_KEY_DIRECT_FLAT = 110;
constexpr uint64_t S9_CONCAT_TILING_KEY_CACHE_LINES = 111;
constexpr uint64_t S9_CONCAT_TILING_KEY_UNIT_FP16_GATHER = 112;
constexpr uint64_t S9_CONCAT_TILING_KEY_PAIR_FP16_GATHER = 114;
constexpr uint64_t S9_CONCAT_TILING_KEY_QUAD_FP16_GATHER = 115;
constexpr uint64_t S9_CONCAT_TILING_KEY_OCT_FP16_GATHER = 116;
constexpr uint64_t S9_CONCAT_TILING_KEY_HEX_FP16_GATHER = 117;
constexpr uint64_t S9_CONCAT_TILING_KEY_DIRECT_ROW_TASKS_QUEUE = 118;
// Large row grids use a compact host-computed task schedule.  The schedule is
// byte-balanced across AIVs, avoiding the long tail created by cyclic
// (input,row-group) ownership when input widths are highly skewed.
constexpr uint64_t S9_CONCAT_TILING_KEY_BALANCED_ROWS_QUEUE = 119;
// FP16 large-row path: one contiguous GM->UB transfer per input shard,
// followed by the vector Pad primitive that restores the 32-byte row layout
// required by the strided UB->GM transfer.  This removes the dominant padded
// MTE2 walk without issuing one vector Gather instruction for every row.
constexpr uint64_t S9_CONCAT_TILING_KEY_CONTIGUOUS_ROW_PAD = 120;
// Medium/large row grids with many inputs use a host-built per-AIV task list.
// This removes the O(AIVs * inputs) task/prefix scan from the generic kernel;
// the byte ordinal encodes (input,row-group), bounded to 256 exact tasks.
constexpr uint64_t S9_CONCAT_TILING_KEY_INPUT_MAJOR_ROWS = 121;
// The same Pad data flow is valid for FP32 and INT32 when the payload is
// treated as uint32_t.  Keep a distinct key so the hot kernel is compiled at
// the physical 32-bit vector width without a runtime dtype branch.
constexpr uint64_t S9_CONCAT_TILING_KEY_CONTIGUOUS_ROW_PAD_32 = 122;
// Even-width INT8 rows can be transported as uint16_t pairs without changing
// a bit.  Odd widths never select this key because a pair would cross a row.
constexpr uint64_t S9_CONCAT_TILING_KEY_CONTIGUOUS_ROW_PAD_INT8_PAIR = 123;
// For <=32 fully aligned inputs, stage a complete output-row slab in UB and
// issue one contiguous MTE3 transfer.  This mirrors CANN's all-align Concat
// data flow and removes one output DMA descriptor per input without changing
// the MTE2 source partition.
constexpr uint64_t S9_CONCAT_TILING_KEY_ALIGNED_ROW_PACK = 124;
// Up to 64 width-skewed inputs use independent row-group counts.  Wide inputs
// are split near the measured 32-KiB target while narrow inputs remain whole;
// a compact host LPT schedule avoids the generic all-AIV ownership scan.
constexpr uint64_t S9_CONCAT_TILING_KEY_ADAPTIVE_ROWS = 125;
// Balanced FP16 flat inputs larger than one UB tile can overlap the next
// MTE2 load with the preceding MTE3 store through a two-slot bound queue.
constexpr uint64_t S9_CONCAT_TILING_KEY_DIRECT_FLAT_QUEUE = 126;
// Uniform, very narrow row shards are descriptor-bound when each input is
// copied straight to its strided GM destination.  Stage a row tile from all
// inputs in UB, gather it into final row-major order, and emit one contiguous
// output DMA.  This is especially important for last-axis 1--16 byte shards.
constexpr uint64_t S9_CONCAT_TILING_KEY_UNIFORM_NARROW_ROWS = 127;
// Prefix-aware Gather also covers unequal shards and up to 16 inputs.
constexpr uint64_t S9_CONCAT_TILING_KEY_RAGGED_NARROW_ROWS = 128;

constexpr uint64_t S9_CONCAT_TILING_KEY_SMALL_OFFSET = 1000;
constexpr uint32_t S9_CONCAT_SMALL_INPUTS = 16;
// Keep this dedicated payload below 1 KiB.  Larger TensorLists retain the
// generic 256-entry ABI and its measured 32-KiB row grid.
constexpr uint32_t S9_CONCAT_BALANCED_MAX_INPUTS = 32;
constexpr uint32_t S9_CONCAT_BALANCED_MAX_CORES = 48;
constexpr uint32_t S9_CONCAT_BALANCED_MAX_GROUPS = 8;
constexpr uint32_t S9_CONCAT_BALANCED_MAX_TASKS =
    S9_CONCAT_BALANCED_MAX_INPUTS * S9_CONCAT_BALANCED_MAX_GROUPS;
constexpr uint32_t S9_CONCAT_ADAPTIVE_MAX_INPUTS = 64;
constexpr uint32_t S9_CONCAT_ADAPTIVE_MAX_TASKS = 256;
constexpr uint64_t S9_CONCAT_DOUBLE_BUFFER_THRESHOLD_BYTES = 32 * 1024;

#define S9_CONCAT_COMMON_TILING_FIELDS \
    uint64_t outer;                     \
    uint64_t inner;                     \
    uint64_t outputRowBytes;            \
    uint64_t totalBytes;                \
    uint64_t blockRows;                 \
    uint64_t flatChunkBytes;            \
    uint64_t tileBytes;                 \
    uint32_t inputCount;                \
    uint32_t mode;                      \
    uint32_t dim;                       \
    uint32_t elementBytes;

// POD on purpose: CANN's tiling macros consume these definitions in both host
// g++ and the Ascend C compiler.  Keep the common prefix byte-identical.
struct ConcatTilingData {
    S9_CONCAT_COMMON_TILING_FIELDS
    uint32_t inputRowBytes[S9_CONCAT_MAX_INPUTS];
};

struct ConcatSmallTilingData {
    S9_CONCAT_COMMON_TILING_FIELDS
    uint16_t inputRowBytes[S9_CONCAT_SMALL_INPUTS];
};

struct ConcatBalancedRowTilingData {
    S9_CONCAT_COMMON_TILING_FIELDS
    uint16_t inputRowBytes[S9_CONCAT_BALANCED_MAX_INPUTS];
    uint16_t inputPrefixBytes[S9_CONCAT_BALANCED_MAX_INPUTS];
    uint16_t coreTaskOffsets[S9_CONCAT_BALANCED_MAX_CORES + 1];
    uint16_t taskOrdinals[S9_CONCAT_BALANCED_MAX_TASKS];
};

struct ConcatInputMajorRowTilingData {
    S9_CONCAT_COMMON_TILING_FIELDS
    uint16_t inputPrefixBytes[S9_CONCAT_MAX_INPUTS];
    uint16_t coreTaskOffsets[S9_CONCAT_BALANCED_MAX_CORES + 1];
    uint8_t taskOrdinals[S9_CONCAT_MAX_INPUTS];
};

struct ConcatAdaptiveRowTilingData {
    S9_CONCAT_COMMON_TILING_FIELDS
    uint32_t inputPrefixBytes[S9_CONCAT_ADAPTIVE_MAX_INPUTS];
    uint16_t coreTaskOffsets[S9_CONCAT_BALANCED_MAX_CORES + 1];
    uint8_t taskInputs[S9_CONCAT_ADAPTIVE_MAX_TASKS];
    uint8_t taskGroups[S9_CONCAT_ADAPTIVE_MAX_TASKS];
    uint16_t inputGroupCounts[S9_CONCAT_ADAPTIVE_MAX_INPUTS];
};

#undef S9_CONCAT_COMMON_TILING_FIELDS

static_assert(sizeof(ConcatTilingData) <= 2048, "Concat full tiling data exceeds the CANN 8.5 limit");
static_assert(sizeof(ConcatSmallTilingData) <= 128, "Concat small tiling data must stay within four GM blocks");
static_assert(sizeof(ConcatBalancedRowTilingData) <= 2048,
              "Concat balanced-row tiling data exceeds the CANN 8.5 limit");
static_assert(sizeof(ConcatInputMajorRowTilingData) <= 2048,
              "Concat input-major tiling data exceeds the CANN 8.5 limit");
static_assert(sizeof(ConcatInputMajorRowTilingData) <= sizeof(ConcatTilingData),
              "Concat input-major payload must fit the registered default tiling buffer");
static_assert(sizeof(ConcatAdaptiveRowTilingData) <= sizeof(ConcatTilingData),
              "Concat adaptive-row payload must fit the registered default tiling buffer");

struct ConcatGatherRowsHeader {
    ConcatSmallTilingData base;
    uint32_t tileRows;
    uint32_t gatherCount;
    uint32_t sourceElements;
    uint32_t groupRows;
    uint32_t groupsPerSlab;
    uint32_t inputOffsets[S9_CONCAT_SMALL_INPUTS];
};
struct ConcatGatherRowsTilingData {
    ConcatGatherRowsHeader header;
    uint32_t indices[4096];
    uint32_t steps[4096];
};
// Only the fixed 48-byte header is decoded by GET_TILING_DATA. Per-input
// metadata is read once for each active input, not for all 256 possible slots.
struct ConcatPipelineHeader {
    uint64_t outer;
    uint64_t blockRows;
    uint64_t outputRowBytes;
    uint32_t inputCount;
    uint32_t tileRows;
    uint32_t mapCount;
    uint32_t sourceElements;
    uint32_t laneBytes;
};
struct ConcatPipelineTilingData {
    ConcatPipelineHeader header;
    uint32_t inputRowBytes[S9_CONCAT_MAX_INPUTS];
    uint32_t inputOffsets[S9_CONCAT_MAX_INPUTS];
    uint16_t originalInputs[S9_CONCAT_MAX_INPUTS];
    uint32_t indices[15360];
};
static_assert(sizeof(ConcatPipelineTilingData) < 65536, "pipeline tiling must fit 64 KiB");

struct ConcatCompactDirectHeader {
    uint64_t outer;
    uint64_t blockRows;
    uint64_t outputRowBytes;
    uint32_t inputCount;
    uint32_t tileBytes;
};
struct ConcatCompactDirectTilingData {
    ConcatCompactDirectHeader header;
    uint32_t inputRowBytes[S9_CONCAT_MAX_INPUTS];
    uint32_t prefixBytes[S9_CONCAT_MAX_INPUTS];
    uint16_t originalInputs[S9_CONCAT_MAX_INPUTS];
};

// Pack each input at a 32-byte boundary inside every row, then remove only
// padding lanes with a one-bit pattern. The same row mask is reused by all
// repeat iterations; its size is independent of the tile row count.
struct ConcatMaskedRowsHeader {
    uint64_t outer;
    uint64_t blockRows;
    uint64_t outputRowBytes;
    uint32_t paddedRowBytes;
    uint32_t tileRows;
    uint32_t inputCount;
    uint32_t laneBytes;
    uint32_t inputBytes;
    uint32_t outputBytes;
    uint32_t maskBytes;
};
struct ConcatMaskedRowsTilingData {
    ConcatMaskedRowsHeader header;
    uint32_t inputRowBytes[S9_CONCAT_MAX_INPUTS];
    uint32_t paddedOffsets[S9_CONCAT_MAX_INPUTS];
    uint16_t originalInputs[S9_CONCAT_MAX_INPUTS];
    uint32_t maskBits[2048];
};
static_assert(sizeof(ConcatMaskedRowsTilingData) < 65536, "masked rows tiling must fit 64 KiB");

// A block-aligned tile is a permutation of 32-byte blocks. Stage each input
// contiguously and transpose the row/input layout entirely inside UB.
struct ConcatAlignedSlabHeader {
    uint64_t outer;
    uint64_t blockRows;
    uint64_t outputRowBytes;
    uint32_t inputCount;
    uint32_t tileRows;
    uint32_t tileBytes;
};
struct ConcatAlignedSlabTilingData {
    ConcatAlignedSlabHeader header;
    uint32_t inputRowBytes[32];
    uint32_t outputPrefixes[32];
    uint32_t inputOffsets[32];
    uint16_t originalInputs[32];
};
static_assert(sizeof(ConcatAlignedSlabTilingData) <= 512, "aligned slab metadata must stay compact");

struct ConcatOutputChunksHeader {
    uint64_t totalBytes;
    uint64_t outputRowBytes;
    uint64_t chunkBytes;
    uint32_t tileBytes;
    uint32_t inputCount;
};
struct ConcatOutputChunksTilingData {
    ConcatOutputChunksHeader header;
    uint32_t inputRowBytes[32];
    uint32_t prefixes[32];
    uint16_t originalInputs[32];
};
static_assert(sizeof(ConcatOutputChunksTilingData) <= 512, "output chunk metadata must stay compact");

struct ConcatBytePairHeader {
    uint64_t outer, blockRows, outputRowBytes, tileRows;
    uint64_t sourceBytes, outputBytes, mapCount, inputCount, patchCount;
};
struct ConcatBytePairInput { uint64_t width, offset0, offset1, original; };
struct ConcatBytePairTilingData {
    ConcatBytePairHeader header;
    ConcatBytePairInput inputs[32];
    uint64_t patchBlocks[64];
    uint32_t lowIndices[1024], highIndices[1024];
    uint32_t indices[12288];
};
static_assert(sizeof(ConcatBytePairTilingData) < 65536, "byte pair tiling fits 64 KiB");


// Whole-input column stripes retain complete row ownership per AIV.
// Affine base/step maps keep host payload independent of row batch depth.
struct ConcatStripeHeader {
    uint64_t outer, blockRows, outputRowBytes;
    uint32_t inputCount, groupCount, tileRows, laneBytes;
    uint32_t inputBytes, outputBytes, mapCount, indexCount;
    uint32_t cachePolicy, mapFormat, inputBuffers, flags;
};
struct ConcatStripeGroup {
    uint32_t prefixBytes, rowBytes, paddedRowBytes;
    uint32_t inputBegin, inputCount, mapBegin, mapCount;
};
struct ConcatStripeTilingData {
    ConcatStripeHeader header;
    ConcatStripeGroup groups[64];
    uint32_t inputRowBytes[256], inputOffsets[256];
    uint16_t originalInputs[256];
    int16_t indices[30000];
};
static_assert(sizeof(ConcatStripeTilingData) < 65536, "stripe map must fit 64 KiB");


// Per-core DMA jobs contain all row arithmetic needed by the device.
struct ConcatDecodedTaskHeader {
    uint64_t totalBytes, outputRowBytes;
    uint32_t tileBytes, inputCount, taskCount, blockCount;
};
struct ConcatDecodedTask {
    uint64_t sourceOffset, outputOffset;
    uint32_t input, rows, rowsPerTile, width;
};
struct ConcatDecodedTaskTilingData {
    ConcatDecodedTaskHeader header;
    uint16_t coreOffsets[S9_CONCAT_BALANCED_MAX_CORES + 1];
    ConcatDecodedTask tasks[S9_CONCAT_ADAPTIVE_MAX_TASKS];
};
static_assert(sizeof(ConcatDecodedTaskTilingData) < 65536, "decoded task tiling fits 64 KiB");



struct ConcatLinearTaskTilingData {
    ConcatDecodedTaskHeader header;
    uint16_t coreOffsets[S9_CONCAT_BALANCED_MAX_CORES + 1];
    ConcatDecodedTask tasks[1024];
};
static_assert(sizeof(ConcatLinearTaskTilingData)<65536,"linear task table fits tiling storage");


struct ConcatBoundaryHeader {
 uint64_t totalBytes,rowBytes,outer;
 uint32_t quantum,inputCount,bodyCores,edgeCores,tileBytes,edgeSlotBytes,bodyTasks,boundaries,tailBytes,edgeOutputBytes;
};
struct ConcatBoundaryInput {uint32_t width,original;};
struct ConcatBoundaryBody {uint64_t src,dst;uint32_t input,width,rows,stride;};
struct ConcatBoundaryEdge {uint32_t firstRow,rows,begin,count,needsPrev,mapBegin,mapCount,shift;uint64_t slotMask;};
struct ConcatBoundaryRecord {int32_t dstRelative;uint8_t left,right,startWord,shift;};
struct ConcatBoundaryTilingData {
 ConcatBoundaryHeader header;ConcatBoundaryInput inputs[32];uint16_t coreOffsets[65];
 ConcatBoundaryBody tasks[512];ConcatBoundaryEdge edges[16];ConcatBoundaryRecord records[256];uint32_t lowMaps[2048],highMaps[2048];
};
static_assert(sizeof(ConcatBoundaryHeader)==64 && sizeof(ConcatBoundaryBody)==32 && sizeof(ConcatBoundaryEdge)==40 && sizeof(ConcatBoundaryRecord)==8,"boundary ABI");
static_assert(sizeof(ConcatBoundaryTilingData)==35912,"boundary plan ABI and capacity");

#endif
