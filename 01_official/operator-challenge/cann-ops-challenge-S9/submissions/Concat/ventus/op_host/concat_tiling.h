/**
 * Concat 主机侧与设备侧共享的分块数据结构及容量常量。
 * 结构布局与设备侧解码配套使用，字段顺序和对齐不可单独调整。
 */

#ifndef S9_CONCAT_TILING_H
#define S9_CONCAT_TILING_H

#include <cstdint>
#include <cstddef>

constexpr uint32_t S9_CONCAT_MAX_RANK = 8;
constexpr uint32_t S9_CONCAT_MAX_INPUTS = 256;
constexpr uint32_t S9_CONCAT_PREFIX_THRESHOLD = 32;
constexpr uint32_t S9_CONCAT_MAX_INPUT_GROUPS = 64;
constexpr uint32_t S9_CONCAT_MAX_LINEAR_TASKS = 512;
constexpr uint32_t S9_CONCAT_MAX_PACK_GROUPS = 256;

struct ConcatTilingData {
    uint64_t outer;
    uint64_t inner;
    uint64_t outputAxis;
    uint64_t taskCount;
    uint64_t maxSegment;
    uint32_t inputCount;
    uint32_t blockDim;
    uint32_t axis;
    uint32_t rank;
    uint32_t mode;
    uint32_t rowGroups;
    uint32_t inputGroups;
    uint32_t reserved;
};

struct ConcatManyTilingData {
    ConcatTilingData base;
    uint64_t prefixes[S9_CONCAT_MAX_INPUTS];
    uint16_t groupOffsets[S9_CONCAT_MAX_INPUT_GROUPS + 1];
    uint16_t inputOrder[S9_CONCAT_MAX_INPUTS];
    uint16_t packGroupCount;
    uint16_t packGroupOffsets[S9_CONCAT_MAX_PACK_GROUPS + 1];
    uint16_t packGroupRealElements[S9_CONCAT_MAX_PACK_GROUPS];
};

struct ConcatLinearTilingData {
    ConcatTilingData base;
    uint64_t prefixes[S9_CONCAT_MAX_INPUTS];
    uint16_t groupOffsets[S9_CONCAT_MAX_INPUT_GROUPS + 1];
    uint16_t linearTaskOrder[S9_CONCAT_MAX_LINEAR_TASKS];
};

constexpr uint32_t S9_CONCAT_STREAM_MICRO_ROWS = 8;
constexpr uint32_t S9_CONCAT_STREAM_INPUT_BYTES = 64U * 1024U;
constexpr uint32_t S9_CONCAT_STREAM_OUTPUT_BYTES = 12U * 1024U;
constexpr uint32_t S9_CONCAT_STREAM_OFFSET_BYTES = 24U * 1024U;
constexpr uint32_t S9_CONCAT_STREAM_MAX_ROW_BYTES =
    S9_CONCAT_STREAM_OUTPUT_BYTES / S9_CONCAT_STREAM_MICRO_ROWS;
constexpr uint32_t S9_CONCAT_STREAM_MAX_GROUP_INPUTS = 64;

constexpr uint32_t S9_CONCAT_STREAM_WIDE_OUTPUT_BYTES = 18U * 1024U;
constexpr uint32_t S9_CONCAT_STREAM_WIDE_OFFSET_BYTES =
    2U * S9_CONCAT_STREAM_WIDE_OUTPUT_BYTES;
constexpr uint32_t S9_CONCAT_STREAM_WIDE_ROW_BYTES =
    S9_CONCAT_STREAM_WIDE_OUTPUT_BYTES / S9_CONCAT_STREAM_MICRO_ROWS;
constexpr uint32_t S9_CONCAT_STREAM_WIDE_SCRATCH_BYTES =
    (S9_CONCAT_STREAM_WIDE_ROW_BYTES + 8U) * sizeof(int32_t);
constexpr uint32_t S9_CONCAT_STREAM_WIDE_INPUT_BYTES =
    (186560U - 2U * (S9_CONCAT_STREAM_WIDE_OUTPUT_BYTES + 32U) -
     (S9_CONCAT_STREAM_WIDE_OFFSET_BYTES + 32U) -
     S9_CONCAT_STREAM_WIDE_SCRATCH_BYTES) / 2U - 32U;
static_assert(S9_CONCAT_STREAM_WIDE_INPUT_BYTES == 51712U,
              "Recompute the complete UB budget when changing geometry");

constexpr uint32_t S9_CONCAT_STREAM_SHORT_MICRO_ROWS = 4U;
constexpr uint32_t S9_CONCAT_STREAM_SHORT_OUTPUT_BYTES =
    S9_CONCAT_STREAM_WIDE_ROW_BYTES * S9_CONCAT_STREAM_SHORT_MICRO_ROWS;
constexpr uint32_t S9_CONCAT_STREAM_SHORT_OFFSET_BYTES =
    2U * S9_CONCAT_STREAM_SHORT_OUTPUT_BYTES;
constexpr uint32_t S9_CONCAT_STREAM_SHORT_INPUT_BYTES =
    (186560U - 2U * (S9_CONCAT_STREAM_SHORT_OUTPUT_BYTES + 32U) -
     (S9_CONCAT_STREAM_SHORT_OFFSET_BYTES + 32U) -
     S9_CONCAT_STREAM_WIDE_SCRATCH_BYTES) / 2U - 32U;
static_assert(S9_CONCAT_STREAM_SHORT_INPUT_BYTES == 70144U,
              "Four-row resource transfer must retain the complete UB budget");

constexpr uint32_t S9_CONCAT_SERP_MACRO_ROWS = 2U * S9_CONCAT_STREAM_MICRO_ROWS;

struct ConcatStreamingTilingData {
    uint64_t outer;
    uint32_t rowBytes;
    uint32_t inputCount;
    uint32_t elementBytes;
    uint32_t groupCount;
    uint32_t rowGroups;
    uint32_t blockDim;
    uint32_t segments[S9_CONCAT_MAX_INPUTS];
    uint32_t prefixes[S9_CONCAT_MAX_INPUTS];
    uint16_t groupBegin[S9_CONCAT_MAX_INPUT_GROUPS + 1];
    uint32_t groupBytes[S9_CONCAT_MAX_INPUT_GROUPS];
    uint32_t groupPitch[S9_CONCAT_MAX_INPUT_GROUPS];

    uint32_t microInputStride[S9_CONCAT_MAX_INPUT_GROUPS];
    uint32_t macroRows[S9_CONCAT_MAX_INPUT_GROUPS];

    uint32_t bufferPolicy;
};
static_assert(sizeof(ConcatStreamingTilingData) == 3240U,
              "Streaming metadata size must not grow for this policy change");

struct ConcatPackedBlockTilingData {
    ConcatStreamingTilingData base;
    uint32_t rowBlocks[S9_CONCAT_MAX_INPUT_GROUPS];
    uint32_t payloadOffset[S9_CONCAT_MAX_INPUT_GROUPS];
    uint8_t alignment[24];
    uint8_t payload[94520];
};
static_assert(offsetof(ConcatPackedBlockTilingData, payload) == 3776U);
static_assert(sizeof(ConcatPackedBlockTilingData) == 98296U);
template <typename T> struct ConcatPackedKey4 {
    using Type = ConcatStreamingTilingData;
    static constexpr bool enabled = false;
};
template <> struct ConcatPackedKey4<float> {
    using Type = ConcatPackedBlockTilingData;
    static constexpr bool enabled = true;
};

struct ConcatRowOwner {
    uint64_t rowStart;
    uint64_t rows;
    uint64_t segment;
    uint64_t prefix;
    uint32_t input;
    uint32_t reserved;
};
struct ConcatRowsTilingData {
    ConcatTilingData base;
    ConcatRowOwner owners[S9_CONCAT_MAX_INPUT_GROUPS];
};
static_assert(sizeof(ConcatRowsTilingData) == 2632U, "Bounded row-owner ABI");

struct ConcatFlatTilingData {
    ConcatTilingData base;
    uint64_t prefixes[S9_CONCAT_MAX_INPUTS + 1U];
};
static_assert(sizeof(ConcatFlatTilingData) == 2128U, "Bounded flat-interval ABI");

struct ConcatRowCompactTilingData {
    ConcatTilingData base;
    uint32_t segments[8];
    uint32_t offsets[8];
    uint32_t inputPitch;
    uint32_t outputRowBytes;
    uint32_t batchRows;
    uint32_t maskBytes;
    uint8_t mask[2048];
};
static_assert(sizeof(ConcatRowCompactTilingData) <= sizeof(ConcatManyTilingData),
              "Row compaction must not enlarge the existing argument ceiling");

#endif
