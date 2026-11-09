#include "concat_tiling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint64_t UB_RESERVED_BYTES = 8U * 1024U;
constexpr uint64_t MIN_TILE_BYTES = 32U;
constexpr uint64_t MAX_TILE_BYTES = 64U * 1024U - MIN_TILE_BYTES;
constexpr uint64_t FOUR_BUFFER_MAX_TILE_BYTES = 32U * 1024U;
constexpr uint64_t FIVE_BUFFER_MAX_TILE_BYTES = 24U * 1024U;
constexpr uint64_t SMALL_OUTPUT_BYTES = 32U * 1024U;
constexpr uint64_t BYTES_PER_CORE = 256U * 1024U;
constexpr uint32_t ROW_PARALLEL_CORES = 8U;
constexpr uint64_t DATA_COPY_MAX_ROWS = 255U;
constexpr uint64_t DATA_COPY_MAX_BLOCKS = 4095U;
constexpr uint64_t DATA_COPY_MAX_FIELD = 0xFFFFFFFFULL;
constexpr uint64_t ALIGNED_ASSEMBLY_MIN_ROWS = 2048U;
constexpr uint64_t ALIGNED_ASSEMBLY_MAX_INPUT_ROW_BYTES = 64U;
constexpr uint32_t ALIGNED_ASSEMBLY_SMALL_CORES = 24U;
constexpr uint32_t ALIGNED_ASSEMBLY_MID_CORES = 32U;
constexpr uint32_t ALIGNED_ASSEMBLY_SMALL_MAX_BATCHES = 2U;
constexpr uint64_t ALIGNED_ASSEMBLY_MID_MAX_BYTES_PER_CORE = 816U * 1024U;
constexpr uint32_t SEGMENTED_NARROW_MIN_ROWS = 4096U;
constexpr uint32_t SEGMENTED_NARROW_MAX_BATCH_ROWS = 240U;
constexpr uint32_t SEGMENTED_NARROW_OUTPUT_BYTES = 512U;
constexpr uint32_t SEGMENTED_NARROW16_MAX_BATCH_ROWS = 288U;
constexpr uint32_t SEGMENTED_B32_MAX_BATCH_ROWS = 280U;
constexpr uint32_t SEGMENTED_ALIGNED_MAX_BATCH_ROWS = 376U;
constexpr uint32_t SEGMENTED_ALIGNED128_MAX_BATCH_ROWS = 240U;
constexpr uint32_t SEGMENTED_ALIGNED128_TRIPLE_MAX_BATCH_ROWS = 208U;
constexpr uint32_t SEGMENTED_ALIGNED32_OUTPUT_BYTES = 512U;
constexpr uint32_t SEGMENTED_ALIGNED_POINTER_BYTES =
    optiling::CONCAT_MAX_INPUTS * sizeof(uint64_t);
constexpr uint64_t COMPACT_GATHER_MIN_ROW_BYTES = 48U;
constexpr uint64_t COMPACT_GATHER_ROW_BYTE_LIMIT = 128U;
constexpr uint64_t COMPACT_GATHER_MAX_STORAGE_ROW_BYTES = 124U;
constexpr uint64_t COMPACT_GATHER_OFFSET_BYTES = 1024U;
constexpr uint32_t COMPACT_GATHER_MIN_INPUTS = 16U;
constexpr uint32_t COMPACT_GATHER_MIN_BATCH_ROWS = 32U;
constexpr uint32_t DYNAMIC_SLAB_MIN_INPUTS = 64U;
constexpr uint32_t DYNAMIC_SLAB_MIN_BATCH_ROWS = 32U;
constexpr uint32_t DYNAMIC_SLAB_MIN_GROUP_BYTES = 1536U;
constexpr uint32_t DYNAMIC_SLAB_MAX_GROUP_BYTES = 1920U;
constexpr uint32_t DYNAMIC_SLAB_MIN_GROUPS = 20U;
constexpr uint32_t DYNAMIC_SLAB_MAX_GROUPS = 32U;
constexpr uint32_t DYNAMIC_SLAB_TARGET_GROUPS = 20U;
constexpr uint32_t DYNAMIC_SLAB_TARGET_BYTES = 1760U;
constexpr uint32_t DYNAMIC_SLAB_METADATA_VERSION = 1U;
constexpr uint32_t DYNAMIC_SLAB_COMPACT_MIN_INPUTS = 20U;
constexpr uint32_t DYNAMIC_SLAB_COMPACT_MIN_ROW_BYTES = 1536U;
constexpr uint64_t DYNAMIC_SLAB_COMPACT_OFFSET_BYTES = 2048U;
constexpr uint64_t DYNAMIC_SLAB_WIDE_OFFSET_BYTES = 4096U;
constexpr uint64_t DYNAMIC_SLAB_GATHER_BASE_BYTES = 256U;
constexpr uint64_t DYNAMIC_SLAB_SCRATCH_OFFSET32_BYTES =
    32U * 64U * sizeof(uint32_t);
constexpr uint32_t DYNAMIC_SLAB_SCRATCH_OFFSET32_MIN_SAVINGS = 64U;
constexpr uint64_t DYNAMIC_SLAB_DUAL_WIDE_OFFSET_BYTES =
    2U * DYNAMIC_SLAB_WIDE_OFFSET_BYTES;
constexpr uint64_t DYNAMIC_DOUBLE_SLAB_BOUNDARY_OFFSET_BYTES = 2048U;
constexpr uint64_t DYNAMIC_DOUBLE_SLAB_BOUNDARY_BASE_BYTES = 256U;
constexpr uint64_t DYNAMIC_DOUBLE_SLAB_BOUNDARY_SCRATCH_ROW_BYTES = 224U;
constexpr uint64_t DYNAMIC_DOUBLE_SLAB_BOUNDARY_MAX_SLAB_STRIDE = 1568U;
constexpr uint64_t DYNAMIC_SLAB_COMPACT_GATHER_ROW_BYTE_LIMIT = 224U;
constexpr uint64_t DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES = 224U;
constexpr uint32_t DYNAMIC_SLAB_SEGMENTED_MIN_BATCH_ROWS = 48U;
constexpr uint32_t DYNAMIC_SLAB_SEGMENTED_MIN_INPUTS = 20U;
constexpr uint32_t DYNAMIC_SLAB_SEGMENTED_MIN_ROW_BYTES = 8192U;
constexpr uint32_t DYNAMIC_SLAB_ADAPTIVE_MIN_AUX_LOADS = 16U;
constexpr uint32_t DYNAMIC_SLAB_ADAPTIVE_METADATA_VERSION = 1U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_METADATA_VERSION = 7U;
// Use all forty AIVs as twenty symmetric two-shard group owners.  A smaller
// owner count keeps the MTE2/Vector steady-state dependency covered while
// retaining the balanced group-owner schedule for descriptor-heavy inputs.
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_GROUPS = 20U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_MIN_ROWS = 512U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_MIN_NONZERO = 3U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS = 16U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_ARENA_PAIR_WORD =
    optiling::CONCAT_DYNAMIC_SLAB_HEADER_WORD + 1U +
    (DYNAMIC_SLAB_GROUP_OWNER_GROUPS + 1U) / 2U;
constexpr uint8_t DYNAMIC_SLAB_GROUP_OWNER_NO_ARENA_PAIR = 0xffU;
// A valid pair always has first < second, so zero is reserved for an arena
// request that keeps every Gather table distinct.  This lets the Host select
// one dependency-heavy group for software pipelining even when it has no
// duplicate compact-Gather geometry.
constexpr uint8_t DYNAMIC_SLAB_GROUP_OWNER_ARENA_WITHOUT_PAIR = 0x00U;
// A tight arena is admitted only when the ordinary 8-KiB-reserve budget
// misses the bounded batch-count proof and an independently encoded budget
// satisfies it.  The marker has low nibble >= high nibble, so it cannot
// collide with a valid compact-Gather pair (whose first ordinal is smaller).
constexpr uint8_t DYNAMIC_SLAB_GROUP_OWNER_TIGHT_ARENA = 0xefU;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_TIGHT_BUDGET_WORD =
    optiling::CONCAT_LAUNCH_PADDING_WORDS - 1U;
// TPipe owns every byte charged below.  Keep a 2.5-KiB physical guard while
// reclaiming only the otherwise conservative portion of the global 8-KiB
// reserve for a capacity-bound persistent arena.
constexpr uint64_t DYNAMIC_SLAB_GROUP_OWNER_TIGHT_RESERVED_BYTES =
    5U * 512U;
// After a pipelined owner leaves the completion path, the next-highest
// dependency chain becomes exposed.  Permit two runtime-ranked unpaired
// groups to use the same arena implementation; duplicate-pair arenas remain
// independent and do not consume this budget.
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_MAX_UNPAIRED_ARENAS = 2U;
static_assert(DYNAMIC_SLAB_GROUP_OWNER_ARENA_PAIR_WORD +
    (DYNAMIC_SLAB_GROUP_OWNER_GROUPS + 3U) / 4U <=
    optiling::CONCAT_LAUNCH_PADDING_WORDS);
static_assert(DYNAMIC_SLAB_GROUP_OWNER_ARENA_PAIR_WORD +
    (DYNAMIC_SLAB_GROUP_OWNER_GROUPS + 3U) / 4U <=
    DYNAMIC_SLAB_GROUP_OWNER_TIGHT_BUDGET_WORD);
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_MAX_BATCH_ROWS = 255U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_MAX_MTE2_PERCENT = 105U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_MAX_MTE3_PERCENT = 112U;
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_MIN_OFFSET_AMORTIZATION = 4U;
// Group ownership amortizes one compact load and its Gather table over all
// rows of a group shard.  Unlike the row-owned keys, even a one-element FP32
// row is therefore cheaper to fetch contiguously than as a long-stride 2-D
// DataCopyPad.  Keep this threshold private to key 92 so the verified key-79
// fallback retains its established request geometry.
constexpr uint32_t DYNAMIC_SLAB_GROUP_OWNER_COMPACT_MIN_ROW_BYTES = 4U;
// MemoryDetail-derived balancing weights.  They are an empirical ordering
// proxy only: the Host never interprets them as cycles or bandwidth.  The
// fixed descriptor/Gather/batch terms prevent a byte-balanced partition from
// leaving one descriptor-dense output group on the completion path.
constexpr uint64_t DYNAMIC_SLAB_GROUP_OWNER_MTE2_WEIGHT = 5400U;
constexpr uint64_t DYNAMIC_SLAB_GROUP_OWNER_GATHER_WEIGHT = 1160U;
constexpr uint64_t DYNAMIC_SLAB_GROUP_OWNER_BATCH_WEIGHT = 65500U;
constexpr uint64_t DYNAMIC_SLAB_GROUP_OWNER_SEMANTIC_KIB_WEIGHT = 50U;
constexpr uint64_t DYNAMIC_SLAB_GROUP_OWNER_PADDED_ROW_WEIGHT = 80U;
// Each persistent Gather table is initialized once per AIV before the row
// batches start, so it participates in the minimax objective instead of only
// serving as a final tie-break.  Keep its weight below the equivalent cost of
// a descriptor-heavy batch: paired MemoryDetail partitions show that about
// 38 MTE2 descriptors match one persistent-table setup, while a larger weight
// merely moves the completion path from table setup to MTE2 issue service.
constexpr uint64_t DYNAMIC_SLAB_GROUP_OWNER_PERSISTENT_TABLE_WEIGHT =
    180000U;
constexpr uint64_t DYNAMIC_SLAB_GROUP_OWNER_MIN_FIXED_BYTES =
    DYNAMIC_SLAB_WIDE_OFFSET_BYTES + DYNAMIC_SLAB_GATHER_BASE_BYTES;
constexpr uint32_t INPUT_ALIGNED_OWNERSHIP_MAX_INPUTS = 8U;
constexpr uint32_t INPUT_ALIGNED_OWNERSHIP_MIN_NONZERO_INPUTS = 4U;
constexpr uint64_t INPUT_ALIGNED_OWNERSHIP_MIN_ROW_BYTES = 1024U;
constexpr uint64_t INPUT_ALIGNED_OWNERSHIP_MIN_TOTAL_BYTES =
    8U * 1024U * 1024U;
constexpr uint32_t INPUT_ALIGNED_OWNERSHIP_MAX_GROWTH_PERCENT = 125U;
constexpr uint32_t INCREMENTAL_FLAT_MIN_INPUTS = 8U;
constexpr uint32_t INCREMENTAL_FLAT_MAX_INPUTS = 32U;
constexpr uint64_t INCREMENTAL_FLAT_MIN_TOTAL_BYTES = 4U * 1024U * 1024U;
constexpr uint32_t INCREMENTAL_FLAT_VERSION_WORD =
    INCREMENTAL_FLAT_MAX_INPUTS;
constexpr uint32_t INCREMENTAL_FLAT_METADATA_VERSION = 1U;
static_assert(INCREMENTAL_FLAT_VERSION_WORD <
    optiling::CONCAT_LAUNCH_PADDING_WORDS);
constexpr uint32_t DEDICATED_DIRECT_MAX_INPUTS = 8U;
constexpr uint32_t DEDICATED_DIRECT_MAX_SLOTS = 5U;
constexpr uint32_t DEDICATED_DIRECT_MIN_NONZERO_INPUTS = 4U;
constexpr uint64_t DEDICATED_DIRECT_MIN_TOTAL_BYTES =
    8U * 1024U * 1024U;
// Dedicated direct owns only its per-input slabs (no TPipe scratch or
// metadata buffers).  It can safely spend the otherwise conservative global
// reserve and admit a larger row batch when the runtime UB proof leaves a
// four-KiB guard for the kernel prologue/epilogue.
constexpr uint64_t DEDICATED_DIRECT_UB_RESERVED_BYTES = 4U * 1024U;
constexpr uint32_t DEDICATED_DIRECT_SLOT_PITCH_WORD =
    DEDICATED_DIRECT_MAX_INPUTS;
constexpr uint32_t DEDICATED_DIRECT_VERSION_WORD =
    DEDICATED_DIRECT_SLOT_PITCH_WORD + DEDICATED_DIRECT_MAX_SLOTS;
constexpr uint32_t DEDICATED_DIRECT_METADATA_VERSION = 1U;
static_assert(DEDICATED_DIRECT_VERSION_WORD <
    optiling::CONCAT_LAUNCH_PADDING_WORDS);
uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return alignment == 0 ? value : (value + alignment - 1) / alignment * alignment;
}

bool CapIncrementalFlatLaunch(uint64_t totalBytes, uint32_t tileBytes,
    uint32_t availableCores, uint32_t& usedCores, uint64_t& blockFactorBytes)
{
    if (tileBytes == 0U || tileBytes % MIN_TILE_BYTES != 0U ||
        usedCores == 0U || availableCores == 0U) {
        return false;
    }
    const uint32_t capped = static_cast<uint32_t>(std::max<uint64_t>(1U,
        static_cast<uint64_t>(availableCores) * 4U / 5U));
    if (capped >= usedCores) {
        return false;
    }
    const uint64_t rawFactor = totalBytes / capped +
        static_cast<uint64_t>(totalBytes % capped != 0U);
    if (rawFactor > std::numeric_limits<uint64_t>::max() - (MIN_TILE_BYTES - 1U)) {
        return false;
    }
    const uint64_t factor = AlignUp(rawFactor, MIN_TILE_BYTES);
    if (factor > std::numeric_limits<uint64_t>::max() / capped) {
        return false;
    }
    const uint64_t finalStart = factor * (capped - 1U);
    if (finalStart >= totalBytes ||
        totalBytes - finalStart < 2U * static_cast<uint64_t>(tileBytes)) {
        return false;
    }
    usedCores = capped;
    blockFactorBytes = factor;
    return true;
}

// A separate wide-row dataflow: two flat input arenas and one complete
// output row. The index table covers every column in bounded wide copies.
constexpr uint32_t WIDE_ROW_MAX_INPUTS = 8U;
constexpr uint32_t WIDE_ROW_ARENA_WORD = WIDE_ROW_MAX_INPUTS;
constexpr uint32_t WIDE_ROW_VERSION_WORD = WIDE_ROW_ARENA_WORD + 1U;
constexpr uint32_t WIDE_ROW_METADATA_VERSION = 1U;
constexpr uint32_t WIDE_ROW_BAND_WORD = WIDE_ROW_VERSION_WORD + 1U;
constexpr uint32_t WIDE_ROW_BAND_VERSION = 2U;
constexpr uint32_t WIDE_ROW_INDEX_BYTES = 16U * 1024U;
static_assert(WIDE_ROW_BAND_WORD < optiling::CONCAT_LAUNCH_PADDING_WORDS);
struct WideRowPlan {
    uint32_t rows = 0U;
    uint32_t arenaBytes = 0U;
    uint32_t bandBytes = 0U;
};

WideRowPlan BuildWideRowPlan(const uint32_t* widths, uint32_t inputNum,
                            uint64_t blockRows, uint64_t ubSize)
{
    WideRowPlan plan;
    if (inputNum == 0U || inputNum > WIDE_ROW_MAX_INPUTS || blockRows < 2U) {
        return plan;
    }
    uint64_t rowBytes = 0U;
    uint32_t active = 0U;
    for (uint32_t i = 0U; i < inputNum; ++i) {
        if (widths[i] == 0U) {
            continue;
        }
        if (widths[i] % sizeof(uint32_t) != 0U || widths[i] < 1024U) {
            return plan;
        }
        rowBytes += widths[i];
        ++active;
    }
    if (active < 2U || rowBytes > DATA_COPY_MAX_FIELD) {
        return plan;
    }
    const uint64_t fixed = AlignUp(rowBytes, MIN_TILE_BYTES) +
        WIDE_ROW_INDEX_BYTES + UB_RESERVED_BYTES;
    if (ubSize <= fixed) {
        return plan;
    }
    for (uint32_t rows = 2U;
         rows <= std::min<uint64_t>(blockRows, DATA_COPY_MAX_ROWS); ++rows) {
        uint64_t arena = 0U;
        for (uint32_t i = 0U; i < inputNum; ++i) {
            arena += AlignUp(static_cast<uint64_t>(widths[i]) * rows,
                             MIN_TILE_BYTES);
        }
        if (2U * arena > ubSize - fixed) {
            break;
        }
        plan.rows = rows;
        plan.arenaBytes = static_cast<uint32_t>(arena);
    }
    // Do not trade input prefetch depth or admission for output overlap.
    // Keep the parent plan when the extra alignment block cannot fit.
    const uint64_t band = AlignUp((rowBytes + 1U) / 2U, MIN_TILE_BYTES);
    if (plan.rows != 0U && band < rowBytes &&
        2U * static_cast<uint64_t>(plan.arenaBytes) + 2U * band +
        WIDE_ROW_INDEX_BYTES + UB_RESERVED_BYTES <= ubSize) {
        plan.bandBytes = static_cast<uint32_t>(band);
    }
    return plan;
}

uint64_t GreatestCommonDivisor(uint64_t lhs, uint64_t rhs)
{
    while (rhs != 0U) {
        const uint64_t remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    return lhs;
}

uint64_t LeastCommonMultiple(uint64_t lhs, uint64_t rhs)
{
    return lhs / GreatestCommonDivisor(lhs, rhs) * rhs;
}

uint32_t DataTypeSize(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4U;
        case ge::DT_FLOAT16:
            return 2U;
        case ge::DT_INT8:
            return 1U;
        default:
            return 0U;
    }
}

bool IsDynamicSlabSegmentedGatherRow(uint64_t rowBytes)
{
    return rowBytes >= DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES &&
        rowBytes % MIN_TILE_BYTES != 0U;
}

struct DynamicSlabGroupOwnerLayout {
    uint16_t begin = 0U;
    uint16_t end = 0U;
    uint32_t rowBytes = 0U;
    uint32_t slabStride = 0U;
    std::array<uint32_t, 4> capacities = {};
    uint32_t rowsPerBatch = 0U;
    uint32_t nonzeroInputs = 0U;
    uint32_t gatherInputs = 0U;
    uint32_t fixedBytes = 0U;
};

struct DynamicSlabGroupOwnerMetrics {
    uint64_t maximumProxy = 0U;
    uint64_t mte2Descriptors = 0U;
    uint64_t mte3Descriptors = 0U;
    uint64_t gatherIssues = 0U;
    uint64_t dynamicOffsetTables = 0U;
    uint64_t persistentOffsetTables = 0U;
    bool valid = false;
};

struct DynamicSlabGroupOwnerPlan {
    std::array<uint16_t, DYNAMIC_SLAB_GROUP_OWNER_GROUPS> groupEnds = {};
    std::array<uint8_t, DYNAMIC_SLAB_GROUP_OWNER_GROUPS> arenaPairs = {};
    std::array<uint8_t, 40> schedule = {};
    uint32_t groupCount = 0U;
    uint32_t scheduleCount = 0U;
    uint32_t availableUbBytes = 0U;
    uint32_t tightArenaAvailableUbBytes = 0U;
    uint64_t mte2Descriptors = 0U;
    uint64_t mte3Descriptors = 0U;
    uint64_t gatherIssues = 0U;
    uint64_t dynamicOffsetTables = 0U;
    uint64_t persistentOffsetTables = 0U;
};

bool IsGroupOwnerGatherRow(uint32_t rowBytes, uint32_t localPrefix);
uint32_t GroupOwnerGatherSegments(uint32_t rowBytes, uint32_t localPrefix);

uint8_t FindGroupOwnerArenaPair(
    const uint32_t* inputRowBytes,
    const DynamicSlabGroupOwnerLayout& layout)
{
    if (inputRowBytes == nullptr || layout.begin >= layout.end) {
        return DYNAMIC_SLAB_GROUP_OWNER_NO_ARENA_PAIR;
    }
    uint32_t bestReferences = 1U;
    uint32_t bestFirst = 0U;
    uint32_t bestSecond = 0U;
    uint32_t localPrefix = 0U;
    uint32_t gatherOrdinal = 0U;
    for (uint32_t input = layout.begin; input < layout.end; ++input) {
        const uint32_t rowBytes = inputRowBytes[input];
        if (rowBytes == 0U) {
            continue;
        }
        const bool gather = IsGroupOwnerGatherRow(rowBytes, localPrefix);
        const bool compact = gather &&
            rowBytes < DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES;
        if (compact) {
            uint32_t references = 0U;
            uint32_t first = 0U;
            uint32_t second = 0U;
            uint32_t candidatePrefix = 0U;
            uint32_t candidateOrdinal = 0U;
            for (uint32_t candidate = layout.begin;
                 candidate < layout.end; ++candidate) {
                const uint32_t candidateBytes = inputRowBytes[candidate];
                if (candidateBytes == 0U) {
                    continue;
                }
                const bool candidateGather = IsGroupOwnerGatherRow(
                    candidateBytes, candidatePrefix);
                if (candidateGather &&
                    candidateBytes <
                        DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES &&
                    candidateBytes == rowBytes &&
                    candidatePrefix % MIN_TILE_BYTES ==
                        localPrefix % MIN_TILE_BYTES) {
                    if (references == 0U) {
                        first = candidateOrdinal;
                    } else if (references == 1U) {
                        second = candidateOrdinal;
                    }
                    ++references;
                }
                candidateOrdinal += static_cast<uint32_t>(candidateGather);
                candidatePrefix += candidateBytes;
            }
            if (references > bestReferences && first < second &&
                second < DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS) {
                bestReferences = references;
                bestFirst = first;
                bestSecond = second;
            }
        }
        gatherOrdinal += static_cast<uint32_t>(gather);
        localPrefix += rowBytes;
    }
    if (bestReferences < 2U || bestFirst >= bestSecond || bestSecond >= 16U) {
        return DYNAMIC_SLAB_GROUP_OWNER_NO_ARENA_PAIR;
    }
    return static_cast<uint8_t>(bestFirst | (bestSecond << 4U));
}

uint32_t GroupOwnerArenaSourceStride(uint32_t rowBytes,
                                     uint32_t localPrefix)
{
    if (IsGroupOwnerGatherRow(rowBytes, localPrefix)) {
        return rowBytes;
    }
    return static_cast<uint32_t>(AlignUp(
        localPrefix % MIN_TILE_BYTES + rowBytes, MIN_TILE_BYTES));
}

uint64_t GroupOwnerArenaBytes(const uint32_t* inputRowBytes,
                              const DynamicSlabGroupOwnerLayout& layout,
                              uint32_t rows)
{
    if (inputRowBytes == nullptr || rows == 0U) {
        return 0U;
    }
    uint64_t total = 0U;
    uint32_t localPrefix = 0U;
    for (uint32_t input = layout.begin; input < layout.end; ++input) {
        const uint32_t rowBytes = inputRowBytes[input];
        if (rowBytes == 0U) {
            continue;
        }
        const uint32_t stride =
            GroupOwnerArenaSourceStride(rowBytes, localPrefix);
        uint64_t slice = static_cast<uint64_t>(rows) * stride;
        if (IsGroupOwnerGatherRow(rowBytes, localPrefix)) {
            slice = AlignUp(slice, MIN_TILE_BYTES);
        }
        if (slice > std::numeric_limits<uint64_t>::max() - total) {
            return 0U;
        }
        total += slice;
        localPrefix += rowBytes;
    }
    return total;
}

uint32_t FindGroupOwnerArenaRows(
    const uint32_t* inputRowBytes,
    const DynamicSlabGroupOwnerLayout& layout,
    uint32_t availableUbBytes)
{
    const uint32_t tableCount = std::max(1U, layout.gatherInputs);
    const uint64_t fixedBytes = static_cast<uint64_t>(tableCount) *
            DYNAMIC_SLAB_WIDE_OFFSET_BYTES +
        DYNAMIC_SLAB_GATHER_BASE_BYTES;
    for (uint32_t rows = layout.rowsPerBatch;
         rows >= DYNAMIC_SLAB_MIN_BATCH_ROWS; --rows) {
        const uint64_t arenaBytes =
            GroupOwnerArenaBytes(inputRowBytes, layout, rows);
        const uint64_t total = fixedBytes + arenaBytes +
            static_cast<uint64_t>(rows) * layout.slabStride;
        if (arenaBytes != 0U && total <= availableUbBytes) {
            return rows;
        }
    }
    return 0U;
}

uint32_t GroupOwnerArenaDependencyPressure(
    const uint32_t* inputRowBytes,
    const DynamicSlabGroupOwnerLayout& layout)
{
    uint32_t pressure = 0U;
    uint32_t localPrefix = 0U;
    for (uint32_t input = layout.begin; input < layout.end; ++input) {
        const uint32_t rowBytes = inputRowBytes[input];
        if (rowBytes == 0U) {
            continue;
        }
        pressure += GroupOwnerGatherSegments(rowBytes, localPrefix);
        localPrefix += rowBytes;
    }
    return pressure;
}

bool HasSingleInteriorDirectGroupOwnerRow(
    const uint32_t* inputRowBytes,
    const DynamicSlabGroupOwnerLayout& layout)
{
    uint32_t directRows = 0U;
    bool gatherBeforeDirect = false;
    bool gatherAfterDirect = false;
    bool seenGather = false;
    bool seenDirect = false;
    uint32_t localPrefix = 0U;
    for (uint32_t input = layout.begin; input < layout.end; ++input) {
        const uint32_t rowBytes = inputRowBytes[input];
        if (rowBytes == 0U) {
            continue;
        }
        const bool gather = IsGroupOwnerGatherRow(rowBytes, localPrefix);
        if (gather) {
            gatherAfterDirect |= seenDirect;
            seenGather = true;
        } else {
            ++directRows;
            gatherBeforeDirect |= seenGather;
            seenDirect = true;
        }
        localPrefix += rowBytes;
    }
    return directRows == 1U && gatherBeforeDirect && gatherAfterDirect;
}

uint32_t GroupOwnerScratchRequirement(uint32_t rowBytes,
                                      uint32_t localPrefix)
{
    // Keep planning capacities conservative for block-sized rows.  Device
    // may Gather a misaligned block row from this larger slot, but never
    // relies on the saved bytes when proving the Host schedule.
    if (rowBytes >= DYNAMIC_SLAB_GROUP_OWNER_COMPACT_MIN_ROW_BYTES &&
        rowBytes % MIN_TILE_BYTES != 0U) {
        return static_cast<uint32_t>(AlignUp(rowBytes, MIN_TILE_BYTES));
    }
    return static_cast<uint32_t>(AlignUp(
        localPrefix % MIN_TILE_BYTES + rowBytes, MIN_TILE_BYTES));
}

bool IsGroupOwnerGatherRow(uint32_t rowBytes, uint32_t localPrefix)
{
    return rowBytes >= DYNAMIC_SLAB_GROUP_OWNER_COMPACT_MIN_ROW_BYTES &&
        (rowBytes % MIN_TILE_BYTES != 0U ||
         localPrefix % MIN_TILE_BYTES != 0U);
}

bool BuildDynamicSlabGroupOwnerLayout(
    const uint32_t* inputRowBytes, uint32_t inputNum,
    uint32_t begin, uint32_t end, uint32_t availableUbBytes,
    DynamicSlabGroupOwnerLayout& layout)
{
    if (inputRowBytes == nullptr || begin >= end || end > inputNum ||
        end - begin > DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS) {
        return false;
    }
    uint64_t groupBytes = 0U;
    uint32_t localPrefix = 0U;
    std::array<uint32_t, 4> capacities = {};
    uint32_t nonzero = 0U;
    uint32_t gatherInputs = 0U;
    for (uint32_t input = begin; input < end; ++input) {
        const uint64_t rowBytes64 = inputRowBytes[input];
        if (rowBytes64 > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        groupBytes += rowBytes64;
        if (rowBytes64 == 0U) {
            continue;
        }
        const uint32_t rowBytes = static_cast<uint32_t>(rowBytes64);
        const uint32_t requirement =
            GroupOwnerScratchRequirement(rowBytes, localPrefix);
        gatherInputs += static_cast<uint32_t>(
            IsGroupOwnerGatherRow(rowBytes, localPrefix));
        uint32_t insert = 4U;
        for (uint32_t rank = 0U; rank < 4U; ++rank) {
            if (requirement > capacities[rank]) {
                insert = rank;
                break;
            }
        }
        if (insert < 4U) {
            for (uint32_t rank = 3U; rank > insert; --rank) {
                capacities[rank] = capacities[rank - 1U];
            }
            capacities[insert] = requirement;
        }
        localPrefix += rowBytes;
        ++nonzero;
    }
    if (nonzero < DYNAMIC_SLAB_GROUP_OWNER_MIN_NONZERO ||
        groupBytes == 0U || groupBytes > 0xffU * MIN_TILE_BYTES ||
        groupBytes != localPrefix) {
        return false;
    }
    for (uint32_t rank = 0U; rank < 4U; ++rank) {
        if (capacities[rank] == 0U) {
            capacities[rank] = MIN_TILE_BYTES;
        }
        if (capacities[rank] / MIN_TILE_BYTES > 0xffU) {
            return false;
        }
    }
    const uint32_t slabStride = static_cast<uint32_t>(
        AlignUp(groupBytes, MIN_TILE_BYTES));
    uint64_t perRowBytes = slabStride;
    for (const uint32_t capacity : capacities) {
        perRowBytes += capacity;
    }
    const uint32_t offsetTableCount = std::max(1U, gatherInputs);
    const uint64_t fixedBytes64 =
        static_cast<uint64_t>(offsetTableCount) *
            DYNAMIC_SLAB_WIDE_OFFSET_BYTES +
        DYNAMIC_SLAB_GATHER_BASE_BYTES;
    if (fixedBytes64 > std::numeric_limits<uint32_t>::max() ||
        perRowBytes == 0U || availableUbBytes < fixedBytes64 ||
        availableUbBytes - fixedBytes64 < perRowBytes) {
        return false;
    }
    const uint32_t rowsPerBatch = static_cast<uint32_t>(
        std::min<uint64_t>(DYNAMIC_SLAB_GROUP_OWNER_MAX_BATCH_ROWS,
            (availableUbBytes - fixedBytes64) / perRowBytes));
    if (rowsPerBatch < DYNAMIC_SLAB_MIN_BATCH_ROWS) {
        return false;
    }
    layout.begin = static_cast<uint16_t>(begin);
    layout.end = static_cast<uint16_t>(end);
    layout.rowBytes = static_cast<uint32_t>(groupBytes);
    layout.slabStride = slabStride;
    layout.capacities = capacities;
    layout.rowsPerBatch = rowsPerBatch;
    layout.nonzeroInputs = nonzero;
    layout.gatherInputs = gatherInputs;
    layout.fixedBytes = static_cast<uint32_t>(fixedBytes64);
    return true;
}

uint32_t GroupOwnerGatherSegments(uint32_t rowBytes, uint32_t localPrefix)
{
    if (!IsGroupOwnerGatherRow(rowBytes, localPrefix)) {
        return 0U;
    }
    if (rowBytes < DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES) {
        return 1U;
    }
    const uint32_t leftBytes = localPrefix % MIN_TILE_BYTES;
    uint32_t segmentSpan = 192U;
    const uint32_t widenedSpan = leftBytes == 0U ? 256U : 224U;
    const uint32_t baselineSegments =
        (rowBytes + segmentSpan - 1U) / segmentSpan;
    const uint32_t widenedSegments =
        (rowBytes + widenedSpan - 1U) / widenedSpan;
    if (widenedSegments <= baselineSegments) {
        segmentSpan = widenedSpan;
    }
    return (rowBytes + segmentSpan - 1U) / segmentSpan;
}

bool BuildGroupOwnerInputSlots(
    const uint32_t* inputRowBytes,
    const DynamicSlabGroupOwnerLayout& layout,
    std::array<uint16_t, DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS>& widths,
    std::array<uint16_t, DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS>& prefixes,
    std::array<uint8_t, DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS>& slots,
    uint32_t& count)
{
    uint32_t freeMask = 0xfU;
    std::array<uint8_t, 4> queue = {};
    uint32_t queueHead = 0U;
    uint32_t queueCount = 0U;
    uint32_t localPrefix = 0U;
    count = 0U;
    for (uint32_t input = layout.begin; input < layout.end; ++input) {
        const uint32_t rowBytes = static_cast<uint32_t>(inputRowBytes[input]);
        if (rowBytes == 0U) {
            continue;
        }
        const uint32_t requirement =
            GroupOwnerScratchRequirement(rowBytes, localPrefix);
        uint32_t selected = 4U;
        while (selected == 4U) {
            uint32_t selectedCapacity = std::numeric_limits<uint32_t>::max();
            for (uint32_t slot = 0U; slot < 4U; ++slot) {
                const uint32_t capacity = layout.capacities[slot];
                if ((freeMask & (1U << slot)) != 0U &&
                    capacity >= requirement &&
                    capacity < selectedCapacity) {
                    selected = slot;
                    selectedCapacity = capacity;
                }
            }
            if (selected != 4U) {
                break;
            }
            if (queueCount == 0U) {
                return false;
            }
            const uint32_t released = queue[queueHead];
            queueHead = (queueHead + 1U) & 3U;
            --queueCount;
            freeMask |= 1U << released;
        }
        if (count >= DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS ||
            rowBytes > 0xffffU || localPrefix > 0xffffU) {
            return false;
        }
        widths[count] = static_cast<uint16_t>(rowBytes);
        prefixes[count] = static_cast<uint16_t>(localPrefix);
        slots[count] = static_cast<uint8_t>(selected);
        ++count;
        freeMask &= ~(1U << selected);
        queue[(queueHead + queueCount) & 3U] =
            static_cast<uint8_t>(selected);
        ++queueCount;
        localPrefix += rowBytes;
    }
    return count == layout.nonzeroInputs && localPrefix == layout.rowBytes;
}

DynamicSlabGroupOwnerMetrics MeasureGroupOwnerShard(
    const uint32_t* inputRowBytes,
    const DynamicSlabGroupOwnerLayout& layout, uint64_t rows)
{
    DynamicSlabGroupOwnerMetrics result;
    if (rows == 0U || layout.rowsPerBatch == 0U) {
        return result;
    }
    std::array<uint16_t, DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS> widths = {};
    std::array<uint16_t, DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS> prefixes = {};
    std::array<uint8_t, DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS> slots = {};
    uint32_t inputCount = 0U;
    if (!BuildGroupOwnerInputSlots(
            inputRowBytes, layout, widths, prefixes, slots, inputCount)) {
        return result;
    }
    const uint64_t fullBatches = rows / layout.rowsPerBatch;
    const uint32_t tailRows = static_cast<uint32_t>(
        rows % layout.rowsPerBatch);
    const uint64_t batchCount = fullBatches + (tailRows != 0U);
    result.mte2Descriptors = batchCount * inputCount;
    result.mte3Descriptors = batchCount;
    result.dynamicOffsetTables = batchCount * layout.gatherInputs;
    result.persistentOffsetTables = layout.gatherInputs;
    auto gatherForBatch = [&](uint32_t batchRows) {
        uint64_t issues = 0U;
        for (uint32_t input = 0U; input < inputCount; ++input) {
            const uint32_t rowBytes = widths[input];
            const uint32_t segments =
                GroupOwnerGatherSegments(rowBytes, prefixes[input]);
            if (segments == 0U) {
                continue;
            }
            const uint32_t capacity = layout.capacities[slots[input]];
            const uint64_t tableOffset = AlignUp(
                static_cast<uint64_t>(batchRows) * rowBytes,
                MIN_TILE_BYTES);
            const bool scratchOffset32 =
                rowBytes >= DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES &&
                tableOffset + DYNAMIC_SLAB_SCRATCH_OFFSET32_BYTES <=
                    static_cast<uint64_t>(batchRows) * capacity;
            const uint32_t repeatRows = scratchOffset32 ? 32U : 16U;
            issues += static_cast<uint64_t>(segments) *
                ((batchRows + repeatRows - 1U) / repeatRows);
        }
        return issues;
    };
    result.gatherIssues = fullBatches * gatherForBatch(layout.rowsPerBatch);
    if (tailRows != 0U) {
        result.gatherIssues += gatherForBatch(tailRows);
    }
    if (rows > std::numeric_limits<uint64_t>::max() / layout.rowBytes) {
        return DynamicSlabGroupOwnerMetrics{};
    }
    const uint64_t semanticBytes = rows * layout.rowBytes;
    const uint64_t semanticKiB = semanticBytes / 1024U +
        static_cast<uint64_t>(semanticBytes % 1024U != 0U);
    uint32_t paddedInputs = 0U;
    for (uint32_t input = 0U; input < inputCount; ++input) {
        const uint32_t rowBytes = widths[input];
        const uint32_t prefix = prefixes[input];
        const bool gather =
            rowBytes >= DYNAMIC_SLAB_GROUP_OWNER_COMPACT_MIN_ROW_BYTES &&
            (rowBytes % MIN_TILE_BYTES != 0U ||
             prefix % MIN_TILE_BYTES != 0U);
        const bool aligned = prefix % MIN_TILE_BYTES == 0U &&
            rowBytes % MIN_TILE_BYTES == 0U;
        paddedInputs += static_cast<uint32_t>(!gather && !aligned);
    }
    if (paddedInputs != 0U &&
        rows > std::numeric_limits<uint64_t>::max() / paddedInputs) {
        return DynamicSlabGroupOwnerMetrics{};
    }
    const uint64_t paddedRows = rows * paddedInputs;
    uint64_t proxy = 0U;
    const auto addWeighted = [&proxy](uint64_t count, uint64_t weight) {
        if (count != 0U &&
            count > (std::numeric_limits<uint64_t>::max() - proxy) / weight) {
            return false;
        }
        proxy += count * weight;
        return true;
    };
    if (!addWeighted(result.mte2Descriptors,
            DYNAMIC_SLAB_GROUP_OWNER_MTE2_WEIGHT) ||
        !addWeighted(result.gatherIssues,
            DYNAMIC_SLAB_GROUP_OWNER_GATHER_WEIGHT) ||
        !addWeighted(result.mte3Descriptors,
            DYNAMIC_SLAB_GROUP_OWNER_BATCH_WEIGHT) ||
        !addWeighted(layout.gatherInputs,
            DYNAMIC_SLAB_GROUP_OWNER_PERSISTENT_TABLE_WEIGHT) ||
        !addWeighted(semanticKiB,
            DYNAMIC_SLAB_GROUP_OWNER_SEMANTIC_KIB_WEIGHT) ||
        !addWeighted(paddedRows,
            DYNAMIC_SLAB_GROUP_OWNER_PADDED_ROW_WEIGHT)) {
        return DynamicSlabGroupOwnerMetrics{};
    }
    result.maximumProxy = proxy;
    result.valid = true;
    return result;
}

DynamicSlabGroupOwnerMetrics MeasureGroupOwnerGroup(
    const uint32_t* inputRowBytes,
    const DynamicSlabGroupOwnerLayout& layout,
    uint64_t outerRows, uint32_t shards)
{
    DynamicSlabGroupOwnerMetrics result;
    if (shards == 0U || shards > 2U) {
        return result;
    }
    const uint64_t baseRows = outerRows / shards;
    const uint32_t extraRows = static_cast<uint32_t>(outerRows % shards);
    result.valid = true;
    for (uint32_t shard = 0U; shard < shards; ++shard) {
        const DynamicSlabGroupOwnerMetrics unit = MeasureGroupOwnerShard(
            inputRowBytes, layout, baseRows + (shard < extraRows));
        if (!unit.valid) {
            result.valid = false;
            return result;
        }
        result.maximumProxy = std::max(
            result.maximumProxy, unit.maximumProxy);
        result.mte2Descriptors += unit.mte2Descriptors;
        result.mte3Descriptors += unit.mte3Descriptors;
        result.gatherIssues += unit.gatherIssues;
        result.dynamicOffsetTables += unit.dynamicOffsetTables;
        result.persistentOffsetTables += unit.persistentOffsetTables;
    }
    return result;
}

bool BetterGroupOwnerMetrics(const DynamicSlabGroupOwnerMetrics& lhs,
                             const DynamicSlabGroupOwnerMetrics& rhs)
{
    if (!lhs.valid) {
        return false;
    }
    if (!rhs.valid) {
        return true;
    }
    if (lhs.maximumProxy != rhs.maximumProxy) {
        return lhs.maximumProxy < rhs.maximumProxy;
    }
    if (lhs.mte2Descriptors != rhs.mte2Descriptors) {
        return lhs.mte2Descriptors < rhs.mte2Descriptors;
    }
    if (lhs.mte3Descriptors != rhs.mte3Descriptors) {
        return lhs.mte3Descriptors < rhs.mte3Descriptors;
    }
    if (lhs.gatherIssues != rhs.gatherIssues) {
        return lhs.gatherIssues < rhs.gatherIssues;
    }
    return lhs.persistentOffsetTables < rhs.persistentOffsetTables;
}

bool BuildDynamicSlabGroupOwnerPlan(
    const uint32_t* inputRowBytes, uint32_t inputNum, uint64_t outputRowBytes,
    uint64_t outerRows, uint32_t usedCores, uint64_t ubSize,
    uint32_t baselineGroupCount, DynamicSlabGroupOwnerPlan& plan)
{
    constexpr uint32_t GROUPS = DYNAMIC_SLAB_GROUP_OWNER_GROUPS;
    plan.arenaPairs.fill(DYNAMIC_SLAB_GROUP_OWNER_NO_ARENA_PAIR);
    if (inputRowBytes == nullptr || inputNum < GROUPS ||
        inputNum > optiling::CONCAT_MAX_INPUTS || usedCores < GROUPS ||
        usedCores > plan.schedule.size() || outerRows <
            DYNAMIC_SLAB_GROUP_OWNER_MIN_ROWS || baselineGroupCount == 0U ||
        ubSize <=
            UB_RESERVED_BYTES + DYNAMIC_SLAB_GROUP_OWNER_MIN_FIXED_BYTES) {
        return false;
    }
    const uint64_t available64 =
        (ubSize - UB_RESERVED_BYTES) /
        MIN_TILE_BYTES * MIN_TILE_BYTES;
    if (available64 == 0U ||
        available64 / MIN_TILE_BYTES > 0xffffU) {
        return false;
    }
    const uint32_t availableUbBytes = static_cast<uint32_t>(available64);
    std::vector<uint32_t> suffixNonzero(inputNum + 1U, 0U);
    for (uint32_t input = inputNum; input != 0U; --input) {
        suffixNonzero[input - 1U] = suffixNonzero[input] +
            static_cast<uint32_t>(inputRowBytes[input - 1U] != 0U);
    }
    if (suffixNonzero[0] <
        GROUPS * DYNAMIC_SLAB_GROUP_OWNER_MIN_NONZERO) {
        return false;
    }
    if (outputRowBytes == 0U) {
        return false;
    }
    struct JointState {
        DynamicSlabGroupOwnerMetrics metrics;
        uint16_t previousInput = 0xffffU;
        uint8_t previousCores = 0xffU;
        uint8_t shards = 0U;
    };
    const uint32_t columns = inputNum + 1U;
    const uint32_t coreStates = usedCores + 1U;
    const auto state = [columns, coreStates](
            uint32_t groups, uint32_t input, uint32_t cores) {
        return (static_cast<size_t>(groups) * columns + input) *
            coreStates + cores;
    };
    std::vector<JointState> states(
        static_cast<size_t>(GROUPS + 1U) * columns * coreStates);
    states[state(0U, 0U, 0U)].metrics.valid = true;

    // Jointly choose each contiguous input group and whether it owns one or
    // two row shards.  A prefix with a worse maximum service and no better
    // aggregate tie-break can never become the optimal completion path after
    // adding the same non-negative suffix, so one state per
    // (group,input,core) is sufficient.
    for (uint32_t groups = 0U; groups < GROUPS; ++groups) {
        const uint32_t remainingGroups = GROUPS - groups - 1U;
        for (uint32_t begin = 0U; begin < inputNum; ++begin) {
            const uint32_t maximumEnd = std::min<uint32_t>(
                inputNum,
                begin + DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS);
            for (uint32_t end = begin + 1U; end <= maximumEnd; ++end) {
                if (suffixNonzero[end] < remainingGroups *
                        DYNAMIC_SLAB_GROUP_OWNER_MIN_NONZERO) {
                    break;
                }
                if (inputNum - end > remainingGroups *
                        DYNAMIC_SLAB_GROUP_OWNER_MAX_INPUTS) {
                    continue;
                }
                DynamicSlabGroupOwnerLayout layout;
                if (!BuildDynamicSlabGroupOwnerLayout(
                        inputRowBytes, inputNum, begin, end,
                        availableUbBytes, layout)) {
                    continue;
                }
                std::array<DynamicSlabGroupOwnerMetrics, 2> service = {
                    MeasureGroupOwnerGroup(
                        inputRowBytes, layout, outerRows, 1U),
                    MeasureGroupOwnerGroup(
                        inputRowBytes, layout, outerRows, 2U),
                };
                for (uint32_t previousCores = groups;
                     previousCores <= std::min<uint32_t>(
                         usedCores, 2U * groups); ++previousCores) {
                    const JointState& parent =
                        states[state(groups, begin, previousCores)];
                    if (!parent.metrics.valid) {
                        continue;
                    }
                    for (uint32_t shards = 1U; shards <= 2U; ++shards) {
                        const uint32_t nextCores = previousCores + shards;
                        if (nextCores > usedCores ||
                            !service[shards - 1U].valid) {
                            continue;
                        }
                        const uint32_t groupsAfter = groups + 1U;
                        const uint32_t remainingAfter = GROUPS - groupsAfter;
                        if (nextCores + remainingAfter > usedCores ||
                            nextCores + 2U * remainingAfter < usedCores) {
                            continue;
                        }
                        JointState candidate = parent;
                        candidate.metrics.maximumProxy = std::max(
                            candidate.metrics.maximumProxy,
                            service[shards - 1U].maximumProxy);
                        candidate.metrics.mte2Descriptors +=
                            service[shards - 1U].mte2Descriptors;
                        candidate.metrics.mte3Descriptors +=
                            service[shards - 1U].mte3Descriptors;
                        candidate.metrics.gatherIssues +=
                            service[shards - 1U].gatherIssues;
                        candidate.metrics.dynamicOffsetTables +=
                            service[shards - 1U].dynamicOffsetTables;
                        candidate.metrics.persistentOffsetTables +=
                            service[shards - 1U].persistentOffsetTables;
                        candidate.metrics.valid = true;
                        candidate.previousInput =
                            static_cast<uint16_t>(begin);
                        candidate.previousCores =
                            static_cast<uint8_t>(previousCores);
                        candidate.shards = static_cast<uint8_t>(shards);
                        JointState& child = states[state(
                            groupsAfter, end, nextCores)];
                        if (BetterGroupOwnerMetrics(
                                candidate.metrics, child.metrics)) {
                            child = candidate;
                        }
                    }
                }
            }
        }
    }
    const JointState& best = states[state(GROUPS, inputNum, usedCores)];
    if (!best.metrics.valid) {
        return false;
    }

    std::array<DynamicSlabGroupOwnerLayout, GROUPS> layouts = {};
    uint32_t splitMask = 0U;
    uint32_t end = inputNum;
    uint32_t cores = usedCores;
    for (uint32_t remaining = GROUPS; remaining != 0U; --remaining) {
        const JointState& selected = states[state(remaining, end, cores)];
        const uint16_t begin = selected.previousInput;
        if (!selected.metrics.valid || begin == 0xffffU || begin >= end ||
            selected.previousCores >= cores ||
            selected.shards == 0U || selected.shards > 2U ||
            !BuildDynamicSlabGroupOwnerLayout(
                inputRowBytes, inputNum, begin, end,
                availableUbBytes, layouts[remaining - 1U])) {
            return false;
        }
        plan.groupEnds[remaining - 1U] = static_cast<uint16_t>(end);
        if (selected.shards == 2U) {
            splitMask |= 1U << (remaining - 1U);
        }
        end = begin;
        cores = selected.previousCores;
    }
    if (end != 0U || cores != 0U) {
        return false;
    }
    for (uint32_t group = 0U; group < GROUPS; ++group) {
        plan.arenaPairs[group] = FindGroupOwnerArenaPair(
            inputRowBytes, layouts[group]);
    }
    // Duplicate-pair arenas remove their original completion paths.  Rank a
    // small number of additional groups by Gather dependency pressure and
    // modeled service, re-running the selection after each winner so the
    // same group cannot consume both slots.  Exact UB and batch-count proofs
    // remain per group; the fixed budget bounds the added MTE3 descriptors.
    for (uint32_t arena = 0U;
         arena < DYNAMIC_SLAB_GROUP_OWNER_MAX_UNPAIRED_ARENAS; ++arena) {
        uint32_t selectedArenaGroup = GROUPS;
        uint32_t selectedArenaPressure = 0U;
        uint64_t selectedArenaMixedBytes = 0U;
        uint64_t selectedArenaProxy = 0U;
        for (uint32_t group = 0U; group < GROUPS; ++group) {
            if (plan.arenaPairs[group] !=
                DYNAMIC_SLAB_GROUP_OWNER_NO_ARENA_PAIR) {
                continue;
            }
            const DynamicSlabGroupOwnerLayout& layout = layouts[group];
            const uint32_t arenaRows = FindGroupOwnerArenaRows(
                inputRowBytes, layout, availableUbBytes);
            if (arenaRows < DYNAMIC_SLAB_MIN_BATCH_ROWS) {
                continue;
            }
            const uint32_t shards = 1U + ((splitMask >> group) & 1U);
            const uint64_t maxShardRows =
                (outerRows + shards - 1U) / shards;
            const uint64_t defaultBatches =
                (maxShardRows + layout.rowsPerBatch - 1U) /
                    layout.rowsPerBatch;
            const uint64_t arenaBatches =
                (maxShardRows + arenaRows - 1U) / arenaRows;
            if (arenaBatches > defaultBatches + 1U) {
                continue;
            }
            const uint32_t pressure = GroupOwnerArenaDependencyPressure(
                inputRowBytes, layout);
            const DynamicSlabGroupOwnerMetrics service =
                MeasureGroupOwnerGroup(
                    inputRowBytes, layout, outerRows, shards);
            if (!service.valid || pressure == 0U) {
                continue;
            }
            // The first arena removes the densest Gather dependency chain.
            // Once it is gone, a group containing one direct aligned DMA
            // strictly between Gather rows can become the completion path:
            // the path switch serializes scratch reuse even when its raw
            // descriptor count is smaller.  Rank that mixed topology by its
            // semantic footprint for the second arena.  All inputs, prefixes,
            // groups and sizes are derived from the runtime layout.
            const uint64_t mixedBytes = arena != 0U &&
                    HasSingleInteriorDirectGroupOwnerRow(
                        inputRowBytes, layout)
                ? static_cast<uint64_t>(layout.rowBytes) * maxShardRows
                : 0U;
            if (mixedBytes > selectedArenaMixedBytes ||
                (mixedBytes == selectedArenaMixedBytes &&
                 (pressure > selectedArenaPressure ||
                  (pressure == selectedArenaPressure &&
                   service.maximumProxy > selectedArenaProxy)))) {
                selectedArenaGroup = group;
                selectedArenaPressure = pressure;
                selectedArenaMixedBytes = mixedBytes;
                selectedArenaProxy = service.maximumProxy;
            }
        }
        if (selectedArenaGroup == GROUPS) {
            break;
        }
        plan.arenaPairs[selectedArenaGroup] =
            DYNAMIC_SLAB_GROUP_OWNER_ARENA_WITHOUT_PAIR;
    }

    // The completion group can miss the ordinary persistent-arena admission
    // by only a few KiB: the smaller arena row count then adds two batches,
    // while reclaiming part of the conservative UB reserve adds only one.
    // Search exactly that capacity-bound class after the ordinary arenas are
    // fixed.  This does not globally enlarge row batches or perturb the DP;
    // only the selected group's arena allocation sees the separately encoded
    // budget.  Rank by runtime semantic traffic first: after the ordinary
    // arenas have removed dependency-heavy paths, the largest capacity-bound
    // owner is the memory-service tail that the old scratch proxy misses.
    const uint64_t tightAvailable64 =
        ubSize > DYNAMIC_SLAB_GROUP_OWNER_TIGHT_RESERVED_BYTES ?
        (ubSize - DYNAMIC_SLAB_GROUP_OWNER_TIGHT_RESERVED_BYTES) /
            MIN_TILE_BYTES * MIN_TILE_BYTES : 0U;
    uint32_t selectedTightGroup = GROUPS;
    uint64_t selectedTightSemanticBytes = 0U;
    uint64_t selectedTightProxy = 0U;
    uint32_t selectedTightPressure = 0U;
    if (tightAvailable64 > availableUbBytes &&
        tightAvailable64 / MIN_TILE_BYTES <= 0xffffU) {
        const uint32_t tightAvailable =
            static_cast<uint32_t>(tightAvailable64);
        for (uint32_t group = 0U; group < GROUPS; ++group) {
            if (plan.arenaPairs[group] !=
                DYNAMIC_SLAB_GROUP_OWNER_NO_ARENA_PAIR) {
                continue;
            }
            const DynamicSlabGroupOwnerLayout& layout = layouts[group];
            const uint32_t ordinaryRows = FindGroupOwnerArenaRows(
                inputRowBytes, layout, availableUbBytes);
            const uint32_t tightRows = FindGroupOwnerArenaRows(
                inputRowBytes, layout, tightAvailable);
            if (tightRows <= ordinaryRows ||
                tightRows < DYNAMIC_SLAB_MIN_BATCH_ROWS) {
                continue;
            }
            const uint32_t shards = 1U + ((splitMask >> group) & 1U);
            const uint64_t maxShardRows =
                (outerRows + shards - 1U) / shards;
            const uint64_t defaultBatches =
                (maxShardRows + layout.rowsPerBatch - 1U) /
                    layout.rowsPerBatch;
            const uint64_t ordinaryBatches = ordinaryRows == 0U ?
                std::numeric_limits<uint64_t>::max() :
                (maxShardRows + ordinaryRows - 1U) / ordinaryRows;
            const uint64_t tightBatches =
                (maxShardRows + tightRows - 1U) / tightRows;
            if (ordinaryBatches <= defaultBatches + 1U ||
                tightBatches > defaultBatches + 1U) {
                continue;
            }
            const uint32_t pressure = GroupOwnerArenaDependencyPressure(
                inputRowBytes, layout);
            const DynamicSlabGroupOwnerMetrics service =
                MeasureGroupOwnerGroup(
                    inputRowBytes, layout, outerRows, shards);
            if (!service.valid || pressure == 0U) {
                continue;
            }
            const uint64_t semanticBytes = layout.rowBytes != 0U &&
                    maxShardRows >
                        std::numeric_limits<uint64_t>::max() /
                            layout.rowBytes ?
                std::numeric_limits<uint64_t>::max() :
                static_cast<uint64_t>(layout.rowBytes) * maxShardRows;
            if (semanticBytes > selectedTightSemanticBytes ||
                (semanticBytes == selectedTightSemanticBytes &&
                 (service.maximumProxy > selectedTightProxy ||
                  (service.maximumProxy == selectedTightProxy &&
                   pressure > selectedTightPressure)))) {
                selectedTightGroup = group;
                selectedTightSemanticBytes = semanticBytes;
                selectedTightProxy = service.maximumProxy;
                selectedTightPressure = pressure;
            }
        }
        if (selectedTightGroup != GROUPS) {
            plan.arenaPairs[selectedTightGroup] =
                DYNAMIC_SLAB_GROUP_OWNER_TIGHT_ARENA;
            plan.tightArenaAvailableUbBytes = tightAvailable;
        }

        // An arena already admitted by the ordinary dependency-pressure
        // selector can also sit just below a batch boundary.  Promoting that
        // existing arena is fundamentally different from admitting another
        // arena: it preserves the selected execution path and can only
        // reduce its MTE2/MTE3/Vector batch count.  Since each AIV owns a
        // private UB, this promotion can share the encoded tight budget with
        // the independently selected capacity-bound arena above.  Restrict
        // the rule to an unpaired ordinary arena so no compact-table sharing
        // semantics are changed, and promote at most one runtime-ranked
        // owner.
        uint32_t selectedTightUpgradeGroup = GROUPS;
        uint64_t selectedTightUpgradeSemanticBytes = 0U;
        uint64_t selectedTightUpgradeProxy = 0U;
        uint32_t selectedTightUpgradePressure = 0U;
        for (uint32_t group = 0U; group < GROUPS; ++group) {
            if (plan.arenaPairs[group] !=
                DYNAMIC_SLAB_GROUP_OWNER_ARENA_WITHOUT_PAIR) {
                continue;
            }
            const DynamicSlabGroupOwnerLayout& layout = layouts[group];
            const uint32_t ordinaryRows = FindGroupOwnerArenaRows(
                inputRowBytes, layout, availableUbBytes);
            const uint32_t tightRows = FindGroupOwnerArenaRows(
                inputRowBytes, layout, tightAvailable);
            if (ordinaryRows < DYNAMIC_SLAB_MIN_BATCH_ROWS ||
                tightRows <= ordinaryRows) {
                continue;
            }
            const uint32_t shards = 1U + ((splitMask >> group) & 1U);
            const uint64_t maxShardRows =
                (outerRows + shards - 1U) / shards;
            const uint64_t ordinaryBatches =
                (maxShardRows + ordinaryRows - 1U) / ordinaryRows;
            const uint64_t tightBatches =
                (maxShardRows + tightRows - 1U) / tightRows;
            if (tightBatches >= ordinaryBatches) {
                continue;
            }
            const uint32_t pressure = GroupOwnerArenaDependencyPressure(
                inputRowBytes, layout);
            const DynamicSlabGroupOwnerMetrics service =
                MeasureGroupOwnerGroup(
                    inputRowBytes, layout, outerRows, shards);
            if (!service.valid || pressure == 0U) {
                continue;
            }
            const uint64_t semanticBytes = layout.rowBytes != 0U &&
                    maxShardRows >
                        std::numeric_limits<uint64_t>::max() /
                            layout.rowBytes ?
                std::numeric_limits<uint64_t>::max() :
                static_cast<uint64_t>(layout.rowBytes) * maxShardRows;
            if (semanticBytes > selectedTightUpgradeSemanticBytes ||
                (semanticBytes == selectedTightUpgradeSemanticBytes &&
                 (service.maximumProxy > selectedTightUpgradeProxy ||
                  (service.maximumProxy == selectedTightUpgradeProxy &&
                   pressure > selectedTightUpgradePressure)))) {
                selectedTightUpgradeGroup = group;
                selectedTightUpgradeSemanticBytes = semanticBytes;
                selectedTightUpgradeProxy = service.maximumProxy;
                selectedTightUpgradePressure = pressure;
            }
        }
        if (selectedTightUpgradeGroup != GROUPS) {
            plan.arenaPairs[selectedTightUpgradeGroup] =
                DYNAMIC_SLAB_GROUP_OWNER_TIGHT_ARENA;
            plan.tightArenaAvailableUbBytes = tightAvailable;
        }
    }
    const uint64_t baselineMte2 =
        static_cast<uint64_t>(suffixNonzero[0]) * usedCores;
    const uint64_t baselineMte3 =
        static_cast<uint64_t>(baselineGroupCount) * usedCores;
    if (best.metrics.mte2Descriptors * 100U >
            baselineMte2 * DYNAMIC_SLAB_GROUP_OWNER_MAX_MTE2_PERCENT ||
        best.metrics.mte3Descriptors * 100U >
            baselineMte3 * DYNAMIC_SLAB_GROUP_OWNER_MAX_MTE3_PERCENT ||
        best.metrics.persistentOffsetTables == 0U ||
        best.metrics.dynamicOffsetTables <
            best.metrics.persistentOffsetTables *
                DYNAMIC_SLAB_GROUP_OWNER_MIN_OFFSET_AMORTIZATION) {
        return false;
    }

    uint32_t block = 0U;
    for (uint32_t group = 0U; group < GROUPS; ++group) {
        const uint32_t shards = 1U + ((splitMask >> group) & 1U);
        for (uint32_t shard = 0U; shard < shards; ++shard) {
            if (block >= usedCores) {
                return false;
            }
            plan.schedule[block++] = static_cast<uint8_t>(
                group | (shard << 5U) | ((shards - 1U) << 6U) | 0x80U);
        }
    }
    if (block != usedCores) {
        return false;
    }
    plan.groupCount = GROUPS;
    plan.scheduleCount = block;
    plan.availableUbBytes = availableUbBytes;
    plan.mte2Descriptors = best.metrics.mte2Descriptors;
    plan.mte3Descriptors = best.metrics.mte3Descriptors;
    plan.gatherIssues = best.metrics.gatherIssues;
    plan.dynamicOffsetTables = best.metrics.dynamicOffsetTables;
    plan.persistentOffsetTables = best.metrics.persistentOffsetTables;
    return true;
}

const gert::Shape& LogicalShape(const gert::StorageShape* storageShape, bool isView)
{
    return isView ? storageShape->GetShape() : storageShape->GetStorageShape();
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr || context->GetComputeNodeInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto instanceInfo = context->GetComputeNodeInfo()->GetInputInstanceInfo(0);
    if (instanceInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t inputNum = static_cast<uint32_t>(instanceInfo->GetInstanceNum());
    if (inputNum == 0U || inputNum > CONCAT_MAX_INPUTS) {
        return ge::GRAPH_FAILED;
    }

    const auto* firstStorageShape = context->GetDynamicInputShape(0, 0);
    const auto* firstDesc = context->GetDynamicInputDesc(0, 0);
    const auto* attrs = context->GetAttrs();
    if (firstStorageShape == nullptr || firstDesc == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* dimPtr = attrs->GetAttrPointer<int64_t>(0);
    if (dimPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const bool firstIsView = context->DynamicInputIsView(0, 0);
    const gert::Shape& firstShape = LogicalShape(firstStorageShape, firstIsView);
    const int64_t rank = static_cast<int64_t>(firstShape.GetDimNum());
    if (rank <= 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t dim = *dimPtr;
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dataType = firstDesc->GetDataType();
    const uint32_t typeSize = DataTypeSize(dataType);
    if (typeSize == 0U) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outer = 1U;
    uint64_t inner = 1U;
    for (int64_t i = 0; i < dim; ++i) {
        if (firstShape.GetDim(i) < 0) {
            return ge::GRAPH_FAILED;
        }
        outer *= static_cast<uint64_t>(firstShape.GetDim(i));
    }
    for (int64_t i = dim + 1; i < rank; ++i) {
        if (firstShape.GetDim(i) < 0) {
            return ge::GRAPH_FAILED;
        }
        inner *= static_cast<uint64_t>(firstShape.GetDim(i));
    }

    uint32_t inputRowBytes[CONCAT_MAX_INPUTS] = {};
    uint64_t outputRowBytes = 0U;
    uint64_t maxAlignedInputRowBytes = 0U;
    uint64_t maxPaddedInputRowBytes = 0U;
    uint64_t maxInputRowBytes = 0U;
    uint32_t uniformInputRowBytes = 0U;
    bool allInputRowsSameWidth = true;
    bool supportsBatchedRows = true;
    bool allInputRowsAligned = true;
    bool allInputRowsContiguous = true;
    bool supportsNarrowB8 = typeSize == sizeof(uint8_t) && outer >= 64U &&
        inputNum >= 32U && inputNum % 32U == 0U;
    bool supportsNarrowB16 = outer >= 32U &&
        inputNum >= 16U && inputNum % 16U == 0U;
    bool supportsNarrowB32 = outer >= 8U && outer % 8U == 0U &&
        inputNum >= 16U && inputNum % 16U == 0U;
    bool supportsNarrowChunks = outer >= 32U && inputNum >= 32U;
    uint32_t narrowChunkBytes = 0U;
    bool supportsPackedRowWidths = true;
    uint32_t compactGatherInputCount = 0U;
    uint32_t compactGatherRowBytes = 0U;
    uint32_t dynamicSlabCompactGatherInputCount = 0U;
    uint32_t dynamicSlabCompactGatherRowBytes = 0U;
    uint32_t dynamicSlabSegmentedGatherInputCount = 0U;
    uint32_t dynamicSlabSegmentedGatherRowBytes = 0U;

    for (uint32_t inputIdx = 0; inputIdx < inputNum; ++inputIdx) {
        const auto* storageShape = context->GetDynamicInputShape(0, inputIdx);
        const auto* inputDesc = context->GetDynamicInputDesc(0, inputIdx);
        if (storageShape == nullptr || inputDesc == nullptr || inputDesc->GetDataType() != dataType) {
            return ge::GRAPH_FAILED;
        }
        const bool isView = context->DynamicInputIsView(0, inputIdx);
        const gert::Shape& shape = LogicalShape(storageShape, isView);
        if (static_cast<int64_t>(shape.GetDimNum()) != rank || shape.GetDim(dim) < 0) {
            return ge::GRAPH_FAILED;
        }
        for (int64_t axis = 0; axis < rank; ++axis) {
            if (axis != dim && shape.GetDim(axis) != firstShape.GetDim(axis)) {
                return ge::GRAPH_FAILED;
            }
        }

        const uint64_t rowBytes = static_cast<uint64_t>(shape.GetDim(dim)) * inner * typeSize;
        if (rowBytes > DATA_COPY_MAX_FIELD) {
            return ge::GRAPH_FAILED;
        }
        outputRowBytes += rowBytes;
        if (rowBytes >= COMPACT_GATHER_MIN_ROW_BYTES &&
            rowBytes % MIN_TILE_BYTES != 0U) {
            if (rowBytes < COMPACT_GATHER_ROW_BYTE_LIMIT) {
                ++compactGatherInputCount;
                compactGatherRowBytes += static_cast<uint32_t>(rowBytes);
            }
            if (rowBytes < DYNAMIC_SLAB_COMPACT_GATHER_ROW_BYTE_LIMIT) {
                ++dynamicSlabCompactGatherInputCount;
                dynamicSlabCompactGatherRowBytes +=
                    static_cast<uint32_t>(rowBytes);
            }
        }
        if (IsDynamicSlabSegmentedGatherRow(rowBytes)) {
            ++dynamicSlabSegmentedGatherInputCount;
            dynamicSlabSegmentedGatherRowBytes +=
                static_cast<uint32_t>(rowBytes);
        }
        supportsPackedRowWidths = supportsPackedRowWidths &&
            rowBytes / typeSize <= 0xffU;
        if (inputIdx == 0U) {
            uniformInputRowBytes = static_cast<uint32_t>(rowBytes);
        } else if (rowBytes != uniformInputRowBytes) {
            allInputRowsSameWidth = false;
        }
        allInputRowsAligned = allInputRowsAligned && rowBytes % MIN_TILE_BYTES == 0U;
        maxAlignedInputRowBytes = std::max(maxAlignedInputRowBytes, AlignUp(rowBytes, MIN_TILE_BYTES));
        maxPaddedInputRowBytes = std::max(maxPaddedInputRowBytes,
            AlignUp(rowBytes + MIN_TILE_BYTES - 1U, MIN_TILE_BYTES));
        maxInputRowBytes = std::max(maxInputRowBytes, rowBytes);

        uint64_t outerStrideBytes = rowBytes;
        const gert::Stride* stride = context->GetDynamicInputStride(0, inputIdx);
        if (isView && stride != nullptr && stride->GetDimNum() > 0U) {
            if (stride->GetDimNum() != static_cast<size_t>(rank) || stride->GetStride(rank - 1) != 1) {
                return ge::GRAPH_FAILED;
            }
            for (int64_t axis = rank - 2; axis >= 0; --axis) {
                if (axis == dim - 1) {
                    continue;
                }
                const int64_t expected = stride->GetStride(axis + 1) * shape.GetDim(axis + 1);
                if (stride->GetStride(axis) != expected) {
                    return ge::GRAPH_FAILED;
                }
            }
            outerStrideBytes = dim == 0 ? rowBytes :
                static_cast<uint64_t>(stride->GetStride(dim - 1)) * typeSize;
        }
        if (outerStrideBytes > DATA_COPY_MAX_FIELD || outerStrideBytes < rowBytes) {
            return ge::GRAPH_FAILED;
        }
        allInputRowsContiguous = allInputRowsContiguous && outerStrideBytes == rowBytes;
        if (rowBytes != sizeof(uint32_t) || outerStrideBytes != rowBytes) {
            supportsNarrowB32 = false;
        }
        if (rowBytes != sizeof(uint16_t) || outerStrideBytes != rowBytes) {
            supportsNarrowB16 = false;
        }
        if (rowBytes != 8U && rowBytes != 16U) {
            supportsNarrowChunks = false;
        } else if (narrowChunkBytes == 0U) {
            narrowChunkBytes = static_cast<uint32_t>(rowBytes);
        } else if (rowBytes != narrowChunkBytes) {
            supportsNarrowChunks = false;
        }
        if (outerStrideBytes != rowBytes) {
            supportsNarrowChunks = false;
        }
        if (rowBytes != sizeof(uint8_t) || outerStrideBytes != rowBytes) {
            supportsNarrowB8 = false;
        }
        inputRowBytes[inputIdx] = static_cast<uint32_t>(rowBytes);
        if (outerStrideBytes - rowBytes > DATA_COPY_MAX_FIELD) {
            supportsBatchedRows = false;
        }
    }

    const uint64_t totalBytes = outer * outputRowBytes;
    uint64_t ubSize = 0U;
    int64_t coreNum = 0;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    coreNum = platform.GetCoreNumAiv();
    if (ubSize <= UB_RESERVED_BYTES + 2U * MIN_TILE_BYTES || coreNum <= 0) {
        return ge::GRAPH_FAILED;
    }

    const bool supportsInputParallel = allInputRowsAligned && allInputRowsContiguous &&
        inputNum >= 64U && outer >= 512U && maxAlignedInputRowBytes != 0U &&
        maxAlignedInputRowBytes <= MAX_TILE_BYTES;
    const bool supportsAlignedAssembly = supportsInputParallel &&
        outer >= ALIGNED_ASSEMBLY_MIN_ROWS &&
        maxAlignedInputRowBytes <= ALIGNED_ASSEMBLY_MAX_INPUT_ROW_BYTES &&
        outputRowBytes != 0U && outputRowBytes <= ubSize - UB_RESERVED_BYTES;
    const bool supportsSegmentedAligned32 = supportsInputParallel &&
        allInputRowsSameWidth && outer >= SEGMENTED_NARROW_MIN_ROWS &&
        outputRowBytes % SEGMENTED_ALIGNED32_OUTPUT_BYTES == 0U &&
        (uniformInputRowBytes == 32U || uniformInputRowBytes == 64U ||
         uniformInputRowBytes == 128U) &&
        static_cast<uint64_t>(inputNum) * uniformInputRowBytes ==
            outputRowBytes;

    const bool useThreeBufferDirect =
        outputRowBytes % MIN_TILE_BYTES != 0U || !allInputRowsAligned;
    const bool useTwoBufferDirect = allInputRowsAligned;
    const bool supportsDirectRowDma = useThreeBufferDirect || useTwoBufferDirect;
    const uint64_t directBufferCount = useThreeBufferDirect ? 3U : 2U;
    uint64_t tileBytes = supportsDirectRowDma ?
        std::min(MAX_TILE_BYTES, (ubSize - UB_RESERVED_BYTES) / directBufferCount) :
        std::min(MAX_TILE_BYTES, (ubSize - UB_RESERVED_BYTES) / 2U);
    tileBytes = std::max(MIN_TILE_BYTES, tileBytes / MIN_TILE_BYTES * MIN_TILE_BYTES);
    const uint64_t narrowTileBytes = std::max(MIN_TILE_BYTES,
        (ubSize - UB_RESERVED_BYTES) / 2U / MIN_TILE_BYTES * MIN_TILE_BYTES);
    const bool supportsBatchedNarrowChunks = supportsNarrowChunks &&
        narrowChunkBytes == 8U && inputNum == CONCAT_MAX_INPUTS;
    const bool supportsSegmentedNarrow16 = supportsNarrowChunks &&
        narrowChunkBytes == 16U && inputNum == CONCAT_MAX_INPUTS &&
        outer >= SEGMENTED_NARROW_MIN_ROWS && outer % 16U == 0U;
    // Only the maximum-input 8-byte case repays the larger scratch footprint.
    // Smaller input counts retain the original key-20 schedule and row tile.
    const uint64_t narrowChunkScratchBytesPerRow = supportsBatchedNarrowChunks ? 768U :
        (narrowChunkBytes == 8U ? 160U : 96U);
    const uint64_t narrowChunkRowsByUb = narrowChunkBytes == 0U ? 0U :
        (ubSize - UB_RESERVED_BYTES) /
        (outputRowBytes + narrowChunkScratchBytesPerRow);
    if (supportsNarrowChunks && narrowChunkBytes != 0U &&
        (inputNum % (MIN_TILE_BYTES / narrowChunkBytes) != 0U ||
         (narrowChunkRowsByUb < 48U && !supportsSegmentedNarrow16))) {
        supportsNarrowChunks = false;
    }
    if (supportsNarrowB16 || supportsNarrowB32 || supportsNarrowB8 || supportsNarrowChunks) {
        tileBytes = narrowTileBytes;
    }
    if (outputRowBytes > DATA_COPY_MAX_FIELD) {
        supportsBatchedRows = false;
    }
    for (uint32_t inputIdx = 0; inputIdx < inputNum; ++inputIdx) {
        if (outputRowBytes < inputRowBytes[inputIdx] ||
            outputRowBytes - inputRowBytes[inputIdx] > DATA_COPY_MAX_FIELD) {
            supportsBatchedRows = false;
        }
    }
    supportsBatchedRows = supportsBatchedRows && maxAlignedInputRowBytes != 0U &&
        maxAlignedInputRowBytes <= tileBytes;
    uint32_t usedCores = 1U;
    if (totalBytes >= SMALL_OUTPUT_BYTES) {
        const uint64_t coresByBytes = (totalBytes + BYTES_PER_CORE - 1U) / BYTES_PER_CORE;
        const uint64_t coresByRows = std::min<uint64_t>(outer, ROW_PARALLEL_CORES);
        usedCores = static_cast<uint32_t>(std::min<uint64_t>(coreNum, std::max(coresByBytes, coresByRows)));
    }
    usedCores = std::max(usedCores, 1U);
    if (supportsInputParallel) {
        usedCores = static_cast<uint32_t>(std::min<uint64_t>(coreNum, inputNum));
        if (supportsAlignedAssembly) {
            const uint64_t rowsByRawUb =
                (ubSize - UB_RESERVED_BYTES) / outputRowBytes;
            if (usedCores >= ALIGNED_ASSEMBLY_SMALL_CORES) {
                const uint64_t rowsPerSmallCore =
                    (outer + ALIGNED_ASSEMBLY_SMALL_CORES - 1U) /
                    ALIGNED_ASSEMBLY_SMALL_CORES;
                const uint64_t smallBatches =
                    (rowsPerSmallCore + rowsByRawUb - 1U) / rowsByRawUb;
                if (smallBatches <= ALIGNED_ASSEMBLY_SMALL_MAX_BATCHES) {
                    usedCores = ALIGNED_ASSEMBLY_SMALL_CORES;
                }
            }
            if (usedCores > ALIGNED_ASSEMBLY_MID_CORES) {
                const uint64_t rowsPerMidCore =
                    (outer + ALIGNED_ASSEMBLY_MID_CORES - 1U) /
                    ALIGNED_ASSEMBLY_MID_CORES;
                if (rowsPerMidCore * outputRowBytes <=
                    ALIGNED_ASSEMBLY_MID_MAX_BYTES_PER_CORE) {
                    usedCores = ALIGNED_ASSEMBLY_MID_CORES;
                }
            }
        }
    }

    const bool useBalancedRowGroups = useThreeBufferDirect &&
        inputNum >= 64U && totalBytes >= BYTES_PER_CORE &&
        outputRowBytes % MIN_TILE_BYTES != 0U;
    bool useBalancedRows = false;
    if (useBalancedRowGroups && supportsPackedRowWidths && outer >= usedCores) {
        const uint64_t alignedRowGroup = MIN_TILE_BYTES /
            GreatestCommonDivisor(outputRowBytes, MIN_TILE_BYTES);
        const uint64_t rowGroups = outer / alignedRowGroup;
        const uint64_t groupsPerBlock = rowGroups / usedCores;
        const uint64_t extraGroups = rowGroups % usedCores;
        const uint64_t groupedMaxRows = std::max(
            (groupsPerBlock + static_cast<uint64_t>(extraGroups != 0U)) *
                alignedRowGroup,
            groupsPerBlock * alignedRowGroup + outer % alignedRowGroup);
        const uint64_t balancedMaxRows = (outer + usedCores - 1U) / usedCores;
        useBalancedRows = groupedMaxRows >= balancedMaxRows + 4U &&
            groupedMaxRows * 100U >= balancedMaxRows * 106U;
    }
    uint64_t blockFactorRows = 0U;
    uint32_t rowsPerBatch = 0U;
    uint32_t scratchGenerations = 2U;
    if (supportsBatchedRows && outer != 0U) {
        if (supportsNarrowB8) {
            const uint64_t maxNarrowCores = std::min<uint64_t>(coreNum,
                outer / 64U);
            usedCores = static_cast<uint32_t>(std::min<uint64_t>(
                std::max<uint64_t>(usedCores, 16U), maxNarrowCores));
        } else if (supportsNarrowB16 || supportsNarrowChunks) {
            const uint64_t maxNarrowCores = std::min<uint64_t>(coreNum,
                outer / 32U);
            usedCores = static_cast<uint32_t>(std::min<uint64_t>(
                std::max<uint64_t>(usedCores, 16U), maxNarrowCores));
        }
        blockFactorRows = (outer + usedCores - 1U) / usedCores;
        if (supportsNarrowB16 || supportsNarrowChunks) {
            blockFactorRows = AlignUp(blockFactorRows, 16U);
            usedCores = static_cast<uint32_t>(std::min<uint64_t>(usedCores,
                (outer + blockFactorRows - 1U) / blockFactorRows));
            if (blockFactorRows < 32U) {
                supportsNarrowB16 = false;
            }
        } else if (supportsNarrowB32) {
            blockFactorRows = AlignUp(blockFactorRows, 8U);
            usedCores = static_cast<uint32_t>(std::min<uint64_t>(usedCores,
                (outer + blockFactorRows - 1U) / blockFactorRows));
            // scatter_vnchwconv_b32 needs at least two 8-row repeats on C220.
            // Keep tiny per-core ranges on the generic DMA path.
            if (blockFactorRows < 16U) {
                supportsNarrowB32 = false;
            }
        } else if (supportsNarrowB8) {
            blockFactorRows = AlignUp(blockFactorRows, 64U);
            usedCores = static_cast<uint32_t>(std::min<uint64_t>(usedCores,
                (outer + blockFactorRows - 1U) / blockFactorRows));
        } else if (outputRowBytes % MIN_TILE_BYTES != 0U) {
            const uint64_t alignedRowGroup = MIN_TILE_BYTES /
                GreatestCommonDivisor(outputRowBytes, MIN_TILE_BYTES);
            if (!useBalancedRows) {
                blockFactorRows = AlignUp(blockFactorRows, alignedRowGroup);
            }
            if (!useBalancedRowGroups) {
                usedCores = static_cast<uint32_t>(std::min<uint64_t>(usedCores,
                    (outer + blockFactorRows - 1U) / blockFactorRows));
            }
        }
        // For a few wide inputs, align every block's first row to the common
        // 32-byte source-address period.  This preserves synchronized
        // same-input streaming across AIVs while avoiding a systematic
        // unaligned start on alternating blocks.
        if (allInputRowsContiguous && useThreeBufferDirect &&
            typeSize == sizeof(uint32_t) &&
            inputNum <= INPUT_ALIGNED_OWNERSHIP_MAX_INPUTS &&
            maxInputRowBytes >= INPUT_ALIGNED_OWNERSHIP_MIN_ROW_BYTES &&
            totalBytes >= INPUT_ALIGNED_OWNERSHIP_MIN_TOTAL_BYTES) {
            uint32_t nonzeroInputs = 0U;
            uint64_t alignedRowGroup = 1U;
            for (uint32_t input = 0U; input < inputNum; ++input) {
                if (inputRowBytes[input] == 0U) {
                    continue;
                }
                ++nonzeroInputs;
                const uint64_t inputAlignmentRows = MIN_TILE_BYTES /
                    GreatestCommonDivisor(inputRowBytes[input], MIN_TILE_BYTES);
                alignedRowGroup = LeastCommonMultiple(
                    alignedRowGroup, inputAlignmentRows);
            }
            const uint64_t alignedBlockFactorRows =
                AlignUp(blockFactorRows, alignedRowGroup);
            if (nonzeroInputs >= INPUT_ALIGNED_OWNERSHIP_MIN_NONZERO_INPUTS &&
                alignedRowGroup > 1U &&
                alignedBlockFactorRows > blockFactorRows &&
                alignedBlockFactorRows * 100U <= blockFactorRows *
                    INPUT_ALIGNED_OWNERSHIP_MAX_GROWTH_PERCENT) {
                blockFactorRows = alignedBlockFactorRows;
                usedCores = static_cast<uint32_t>(std::min<uint64_t>(
                    usedCores,
                    (outer + blockFactorRows - 1U) / blockFactorRows));
            }
        }
        const bool useDirectRowDma = !supportsNarrowB16 && !supportsNarrowB32 &&
            !supportsNarrowB8 && !supportsNarrowChunks &&
            supportsDirectRowDma;
        const uint64_t outputStorageBytes = outputRowBytes;
        const uint64_t rowsByOutput = useDirectRowDma ? blockFactorRows :
            (outputStorageBytes == 0U ? 0U : tileBytes / outputStorageBytes);
        const uint64_t rowsByInput = (supportsNarrowB16 || supportsNarrowB32 ||
            supportsNarrowB8 || supportsNarrowChunks ||
            (typeSize == 1U && !useDirectRowDma)) ? blockFactorRows :
            (maxPaddedInputRowBytes == 0U ? 0U : tileBytes / maxPaddedInputRowBytes);
        const uint64_t rowsByUb = std::min(rowsByOutput, rowsByInput);
        rowsPerBatch = static_cast<uint32_t>(std::min<uint64_t>(
            std::min<uint64_t>(blockFactorRows, rowsByUb), DATA_COPY_MAX_ROWS));
        if (supportsNarrowB16 || supportsNarrowChunks) {
            rowsPerBatch = rowsPerBatch / 16U * 16U;
        } else if (supportsNarrowB32) {
            rowsPerBatch = rowsPerBatch / 8U * 8U;
        } else if (supportsNarrowB8) {
            rowsPerBatch = rowsPerBatch / 32U * 32U;
            if (rowsPerBatch != 0U) {
                const uint32_t remainingRows = static_cast<uint32_t>(
                    blockFactorRows % rowsPerBatch);
                if (remainingRows != 0U && remainingRows < 64U && rowsPerBatch >= 96U) {
                    rowsPerBatch -= 32U;
                }
            }
        }
        if (rowsPerBatch == 0U) {
            blockFactorRows = 0U;
            supportsNarrowB32 = false;
            supportsNarrowB16 = false;
            supportsNarrowChunks = false;
            supportsNarrowB8 = false;
        } else {
            const uint64_t batchedOutputBytes =
                static_cast<uint64_t>(rowsPerBatch) * outputRowBytes;
            const uint64_t batchedInputBytes =
                static_cast<uint64_t>(rowsPerBatch) * maxPaddedInputRowBytes;
            tileBytes = AlignUp(useDirectRowDma ? batchedInputBytes :
                std::max(batchedOutputBytes, batchedInputBytes), MIN_TILE_BYTES);
        }
    }
    if (supportsNarrowChunks) {
        rowsPerBatch = static_cast<uint32_t>(std::min<uint64_t>(
            std::min<uint64_t>(blockFactorRows, narrowChunkRowsByUb),
            DATA_COPY_MAX_ROWS));
        rowsPerBatch = rowsPerBatch / 16U * 16U;
        const uint32_t remainingRows = static_cast<uint32_t>(
            blockFactorRows % rowsPerBatch);
        if (remainingRows != 0U && remainingRows < 32U && rowsPerBatch >= 48U) {
            rowsPerBatch -= 16U;
        }
        tileBytes = AlignUp(static_cast<uint64_t>(rowsPerBatch) * outputRowBytes,
                            MIN_TILE_BYTES);
    }
    const bool supportsSegmentedNarrowChunks = supportsBatchedNarrowChunks &&
        supportsNarrowChunks && outer >= SEGMENTED_NARROW_MIN_ROWS &&
        outer % 16U == 0U;
    if (supportsSegmentedNarrowChunks) {
        // Keep one 64-input (512-byte) output segment in UB and cover many more
        // rows per input DMA.  The kernel reserves another 192 bytes per row
        // for two input generations plus a shared transpose stage.
        rowsPerBatch = static_cast<uint32_t>(std::min<uint64_t>(
            blockFactorRows, SEGMENTED_NARROW_MAX_BATCH_ROWS));
        rowsPerBatch = rowsPerBatch / 16U * 16U;
        tileBytes = AlignUp(static_cast<uint64_t>(rowsPerBatch) *
            SEGMENTED_NARROW_OUTPUT_BYTES, MIN_TILE_BYTES);
    }
    if (supportsSegmentedNarrow16) {
        // Keep one 32-input (512-byte) output segment in UB.  Two 16-byte
        // input generations plus the shared transpose stage consume another
        // 128 bytes per row, allowing long input DMA bursts despite the
        // 4-KiB complete output row.
        rowsPerBatch = static_cast<uint32_t>(std::min<uint64_t>(
            blockFactorRows, SEGMENTED_NARROW16_MAX_BATCH_ROWS));
        rowsPerBatch = rowsPerBatch / 16U * 16U;
        tileBytes = AlignUp(static_cast<uint64_t>(rowsPerBatch) *
            SEGMENTED_NARROW_OUTPUT_BYTES, MIN_TILE_BYTES);
    }
    const bool supportsSegmentedNarrowB32 = supportsNarrowB32 &&
        inputNum == CONCAT_MAX_INPUTS && outer >= SEGMENTED_NARROW_MIN_ROWS;
    if (supportsSegmentedNarrowB32) {
        // One 128-input output segment plus two 16-input staging generations
        // consumes 640 UB bytes per row.  A 280-row batch stays within the
        // compiler-visible UB while reducing descriptor setup further.
        rowsPerBatch = static_cast<uint32_t>(std::min<uint64_t>(
            blockFactorRows, SEGMENTED_B32_MAX_BATCH_ROWS));
        rowsPerBatch = rowsPerBatch / 8U * 8U;
        tileBytes = AlignUp(static_cast<uint64_t>(rowsPerBatch) *
            SEGMENTED_NARROW_OUTPUT_BYTES, MIN_TILE_BYTES);
    }
    if (supportsAlignedAssembly) {
        const uint64_t rowsByRawUb = (ubSize - UB_RESERVED_BYTES) / outputRowBytes;
        const uint64_t batchCount =
            (blockFactorRows + rowsByRawUb - 1U) / rowsByRawUb;
        rowsPerBatch = static_cast<uint32_t>(
            (blockFactorRows + batchCount - 1U) / batchCount);
        tileBytes = AlignUp(static_cast<uint64_t>(rowsPerBatch) * outputRowBytes,
                            MIN_TILE_BYTES);
    } else if (supportsInputParallel) {
        rowsPerBatch = static_cast<uint32_t>(std::min<uint64_t>(
            DATA_COPY_MAX_BLOCKS, MAX_TILE_BYTES / maxAlignedInputRowBytes));
        tileBytes = AlignUp(static_cast<uint64_t>(rowsPerBatch) *
            maxAlignedInputRowBytes, MIN_TILE_BYTES);
    }
    if (supportsSegmentedAligned32) {
        // The 128-byte rows benefit from pipelined contiguous loads and UB
        // scatter.  Narrower aligned rows retain the lower-overhead strided
        // MTE2 assembly path.
        const uint32_t maxBatchRows = uniformInputRowBytes == 128U ?
            SEGMENTED_ALIGNED128_MAX_BATCH_ROWS :
            SEGMENTED_ALIGNED_MAX_BATCH_ROWS;
        if (uniformInputRowBytes == 64U) {
            const uint64_t batchCount =
                (blockFactorRows + maxBatchRows - 1U) / maxBatchRows;
            rowsPerBatch = static_cast<uint32_t>(
                (blockFactorRows + batchCount - 1U) / batchCount);
        } else {
            rowsPerBatch = static_cast<uint32_t>(std::min<uint64_t>(
                blockFactorRows, maxBatchRows));
        }
        const uint64_t segmentBufferBytes =
            static_cast<uint64_t>(rowsPerBatch) *
            SEGMENTED_ALIGNED32_OUTPUT_BYTES;
        const uint64_t scratchGenerationBytes =
            static_cast<uint64_t>(rowsPerBatch) * uniformInputRowBytes;
        // A third input generation shortens the exposed MTE2 tail for short
        // batches.  At longer batches the larger UB footprint increases the
        // completion-controlling MTE2/MTE3 spans, so retain two generations.
        if (uniformInputRowBytes == 128U &&
            rowsPerBatch <= SEGMENTED_ALIGNED128_TRIPLE_MAX_BATCH_ROWS &&
            segmentBufferBytes + 3U * scratchGenerationBytes +
                SEGMENTED_ALIGNED_POINTER_BYTES <=
                ubSize - UB_RESERVED_BYTES) {
            scratchGenerations = 3U;
        }
        const bool supportsDoubleBufferedStridedSegments =
            uniformInputRowBytes != 128U &&
            2U * segmentBufferBytes + SEGMENTED_ALIGNED_POINTER_BYTES <=
                ubSize - UB_RESERVED_BYTES;
        tileBytes = AlignUp(segmentBufferBytes *
            (supportsDoubleBufferedStridedSegments ? 2U : 1U),
            MIN_TILE_BYTES);
    }

    // The generic persistent direct path rotates every input over three
    // equal slots, so every slot must be sized for the widest input.  When
    // there are only four or five nonzero inputs, keep one slot per input.
    // This delays all reuse until the following row batch and packs each
    // slot at its own width, allowing a larger batch without a boundary
    // drain.  Admission is based only on the runtime widths and UB proof.
    uint32_t dedicatedDirectSlotPitch[DEDICATED_DIRECT_MAX_SLOTS] = {};
    uint32_t dedicatedDirectNonzeroInputs = 0U;
    uint64_t dedicatedDirectPitch = 0U;
    for (uint32_t input = 0U; input < inputNum; ++input) {
        const uint32_t rowBytes = inputRowBytes[input];
        if (rowBytes == 0U) {
            continue;
        }
        if (dedicatedDirectNonzeroInputs < DEDICATED_DIRECT_MAX_SLOTS) {
            const uint32_t paddedRowBytes = static_cast<uint32_t>(AlignUp(
                static_cast<uint64_t>(rowBytes) + MIN_TILE_BYTES - 1U,
                MIN_TILE_BYTES));
            dedicatedDirectSlotPitch[dedicatedDirectNonzeroInputs] =
                paddedRowBytes;
            dedicatedDirectPitch += paddedRowBytes;
        }
        ++dedicatedDirectNonzeroInputs;
    }
    const uint32_t genericDirectRowsPerBatch = rowsPerBatch;
    const uint64_t dedicatedDirectRowsByUb = dedicatedDirectPitch == 0U ?
        0U : (ubSize - DEDICATED_DIRECT_UB_RESERVED_BYTES) /
            dedicatedDirectPitch;
    const uint32_t dedicatedDirectRowsPerBatch = static_cast<uint32_t>(
        std::min<uint64_t>(
            std::min<uint64_t>(blockFactorRows, dedicatedDirectRowsByUb),
            DATA_COPY_MAX_ROWS));
    const uint64_t genericDirectBatchCount = genericDirectRowsPerBatch == 0U ?
        0U : (blockFactorRows + genericDirectRowsPerBatch - 1U) /
            genericDirectRowsPerBatch;
    const uint64_t dedicatedDirectBatchCount =
        dedicatedDirectRowsPerBatch == 0U ? 0U :
        (blockFactorRows + dedicatedDirectRowsPerBatch - 1U) /
            dedicatedDirectRowsPerBatch;
    const bool supportsDedicatedDirectRows =
        allInputRowsContiguous && useThreeBufferDirect &&
        inputNum <= DEDICATED_DIRECT_MAX_INPUTS &&
        dedicatedDirectNonzeroInputs >=
            DEDICATED_DIRECT_MIN_NONZERO_INPUTS &&
        dedicatedDirectNonzeroInputs <= DEDICATED_DIRECT_MAX_SLOTS &&
        totalBytes >= DEDICATED_DIRECT_MIN_TOTAL_BYTES &&
        blockFactorRows != 0U && genericDirectRowsPerBatch != 0U &&
        genericDirectBatchCount >= 3U &&
        dedicatedDirectRowsPerBatch > genericDirectRowsPerBatch &&
        dedicatedDirectBatchCount < genericDirectBatchCount &&
            static_cast<uint64_t>(dedicatedDirectRowsPerBatch) *
            dedicatedDirectPitch <= ubSize - DEDICATED_DIRECT_UB_RESERVED_BYTES;
    if (supportsDedicatedDirectRows) {
        rowsPerBatch = dedicatedDirectRowsPerBatch;
    }
    const WideRowPlan wideRowPlan = BuildWideRowPlan(
        inputRowBytes, inputNum, blockFactorRows, ubSize);
    const bool supportsWideRowAssembly = supportsDedicatedDirectRows &&
        typeSize == sizeof(uint32_t) && wideRowPlan.rows >= 2U;
    if (supportsWideRowAssembly) {
        rowsPerBatch = wideRowPlan.rows;
    }

    uint64_t blockFactorBytes = totalBytes == 0U ? 0U :
        (totalBytes + usedCores - 1U) / usedCores;
    blockFactorBytes = AlignUp(blockFactorBytes, MIN_TILE_BYTES);
    if (blockFactorBytes != 0U) {
        usedCores = static_cast<uint32_t>(std::min<uint64_t>(usedCores,
            (totalBytes + blockFactorBytes - 1U) / blockFactorBytes));
    }
    const bool supportsPipelinedByteRange = rowsPerBatch == 0U &&
        blockFactorRows == 0U && blockFactorBytes >= 2U * tileBytes &&
        2U * tileBytes <= ubSize - UB_RESERVED_BYTES;
    const bool supportsIncrementalFlatRange = supportsPipelinedByteRange &&
        allInputRowsContiguous && inputNum >= INCREMENTAL_FLAT_MIN_INPUTS &&
        inputNum <= INCREMENTAL_FLAT_MAX_INPUTS &&
        totalBytes >= INCREMENTAL_FLAT_MIN_TOTAL_BYTES;
    if (supportsIncrementalFlatRange && outer > 1U &&
        static_cast<uint64_t>(coreNum) <= std::numeric_limits<uint32_t>::max()) {
        CapIncrementalFlatLaunch(totalBytes, static_cast<uint32_t>(tileBytes),
            static_cast<uint32_t>(coreNum), usedCores, blockFactorBytes);
    }

    ConcatTilingData tiling;
    tiling.set_inputNum(inputNum);
    tiling.set_tileBytes(static_cast<uint32_t>(tileBytes));
    tiling.set_totalBytes(totalBytes);
    tiling.set_outputRowBytes(outputRowBytes);
    tiling.set_blockFactorBytes(blockFactorBytes);
    tiling.set_outerRows(outer);
    tiling.set_blockFactorRows(blockFactorRows);
    tiling.set_rowsPerBatch(rowsPerBatch);
    tiling.set_elementBytes(typeSize);
    tiling.set_concatDim(static_cast<uint32_t>(dim));
    tiling.set_innerElements(inner);
    tiling.set_uniformInputRowBytes(
        allInputRowsSameWidth ? uniformInputRowBytes : 0U);
    uint32_t launchPadding[CONCAT_LAUNCH_PADDING_WORDS] = {};
    if (supportsPackedRowWidths) {
        for (uint32_t inputIdx = 0U; inputIdx < inputNum; ++inputIdx) {
            const uint32_t rowElements = inputRowBytes[inputIdx] / typeSize;
            launchPadding[inputIdx / 4U] |=
                (rowElements & 0xffU) << ((inputIdx % 4U) * 8U);
        }
    }
    if (supportsIncrementalFlatRange) {
        for (uint32_t input = 0U; input < inputNum; ++input) {
            launchPadding[input] = inputRowBytes[input];
        }
        launchPadding[INCREMENTAL_FLAT_VERSION_WORD] =
            INCREMENTAL_FLAT_METADATA_VERSION;
    }
    if (supportsDedicatedDirectRows) {
        for (uint32_t input = 0U; input < inputNum; ++input) {
            launchPadding[input] = inputRowBytes[input];
        }
        for (uint32_t slot = 0U;
             slot < DEDICATED_DIRECT_MAX_SLOTS; ++slot) {
            launchPadding[DEDICATED_DIRECT_SLOT_PITCH_WORD + slot] =
                dedicatedDirectSlotPitch[slot];
        }
        launchPadding[DEDICATED_DIRECT_VERSION_WORD] =
            DEDICATED_DIRECT_METADATA_VERSION;
    }
    bool supportsAlignmentBalancedRows = false;
    if (useBalancedRows && allInputRowsContiguous && usedCores <= 40U) {
        constexpr int32_t INVALID_SCORE = -1;
        int32_t scheduleScores[41][41];
        uint64_t scheduleMasks[41][41] = {};
        for (uint32_t block = 0U; block <= 40U; ++block) {
            for (uint32_t extra = 0U; extra <= 40U; ++extra) {
                scheduleScores[block][extra] = INVALID_SCORE;
            }
        }
        const uint32_t extraRows = static_cast<uint32_t>(outer % usedCores);
        const uint64_t rowsPerBlock = outer / usedCores;
        auto alignmentScore = [&](uint64_t startRow) {
            uint32_t score = 0U;
            uint64_t outputPrefix = 0U;
            const uint64_t outputStart =
                (startRow % MIN_TILE_BYTES) * (outputRowBytes % MIN_TILE_BYTES);
            for (uint32_t inputIdx = 0U; inputIdx < inputNum; ++inputIdx) {
                const uint64_t rowBytes = inputRowBytes[inputIdx];
                if (rowBytes != 0U) {
                    score += static_cast<uint32_t>(
                        (startRow % MIN_TILE_BYTES) *
                            (rowBytes % MIN_TILE_BYTES) % MIN_TILE_BYTES == 0U);
                    score += static_cast<uint32_t>(
                        (outputStart + outputPrefix) % MIN_TILE_BYTES == 0U);
                }
                outputPrefix += rowBytes;
            }
            return score;
        };
        scheduleScores[0][0] = static_cast<int32_t>(alignmentScore(0U));
        for (uint32_t block = 0U; block < usedCores; ++block) {
            for (uint32_t usedExtra = 0U; usedExtra <= extraRows; ++usedExtra) {
                if (scheduleScores[block][usedExtra] == INVALID_SCORE) {
                    continue;
                }
                for (uint32_t longBlock = 0U; longBlock <= 1U; ++longBlock) {
                    const uint32_t nextExtra = usedExtra + longBlock;
                    const uint32_t remainingBlocks = usedCores - block - 1U;
                    if (nextExtra > extraRows ||
                        extraRows - nextExtra > remainingBlocks) {
                        continue;
                    }
                    int32_t nextScore = scheduleScores[block][usedExtra];
                    if (block + 1U < usedCores) {
                        const uint64_t nextStart =
                            (block + 1U) * rowsPerBlock + nextExtra;
                        nextScore += static_cast<int32_t>(alignmentScore(nextStart));
                    }
                    if (nextScore > scheduleScores[block + 1U][nextExtra]) {
                        scheduleScores[block + 1U][nextExtra] = nextScore;
                        scheduleMasks[block + 1U][nextExtra] =
                            scheduleMasks[block][usedExtra] |
                            (static_cast<uint64_t>(longBlock) << block);
                    }
                }
            }
        }
        uint64_t currentMask = extraRows == 0U ? 0U :
            ((1ULL << extraRows) - 1U);
        uint32_t currentScore = 0U;
        uint32_t currentPrefixExtra = 0U;
        for (uint32_t block = 0U; block < usedCores; ++block) {
            currentScore += alignmentScore(
                block * rowsPerBlock + currentPrefixExtra);
            currentPrefixExtra += static_cast<uint32_t>(block < extraRows);
        }
        const int32_t bestScore = scheduleScores[usedCores][extraRows];
        const uint64_t bestMask = scheduleMasks[usedCores][extraRows];
        supportsAlignmentBalancedRows = bestScore >= 0 &&
            static_cast<uint64_t>(bestScore) * 100U >=
                static_cast<uint64_t>(currentScore) * 103U &&
            bestMask != currentMask;
        if (supportsAlignmentBalancedRows) {
            uint32_t prefixExtra = 0U;
            for (uint32_t block = 0U; block < usedCores; ++block) {
                const uint32_t longBlock = static_cast<uint32_t>(
                    (bestMask >> block) & 1U);
                const uint32_t scheduleByte = prefixExtra | (longBlock << 6U);
                launchPadding[CONCAT_PACKED_WIDTH_WORDS + block / 4U] |=
                    scheduleByte << ((block % 4U) * 8U);
                prefixExtra += longBlock;
            }
        }
    }
    bool supportsDynamicSlabRows = false;
    bool supportsDynamicSlabCompactRows = false;
    bool supportsDynamicSlabSegmentedRows = false;
    bool supportsDynamicSlabSegmentedTripleRows = false;
    bool supportsDynamicSlabSegmentedTripleWideRows = false;
    bool supportsDynamicSlabSegmentedTripleScalarFreeRows = false;
    bool supportsDynamicSlabWideAdaptiveScalarFreeRows = false;
    bool supportsDynamicSlabScratchOffset32Rows = false;
    bool supportsDynamicSlabSegmentedAdaptiveRows = false;
    bool supportsDynamicSlabWideAdaptiveRows = false;
    bool supportsDynamicSlabDualOffsetRows = false;
    bool supportsDynamicSlabVectorOffsetRows = false;
    bool supportsDynamicSlabDirectPointerRows = false;
    bool supportsDynamicSlabPrecomputedOffsetRows = false;
    bool supportsDynamicSlabGroupOwnerRows = false;
    bool supportsDynamicDoubleSlabBoundaryRows = false;
    bool prefersDynamicSlabTripleRows = false;
    uint32_t adaptiveScratchCapacities[4] = {};
    uint32_t adaptiveLayoutWord = 0U;
    uint32_t wideAdaptiveScratchCapacities[4] = {};
    uint32_t wideAdaptiveLayoutWord = 0U;
    uint32_t dualOffsetScratchCapacities[4] = {};
    uint32_t dualOffsetLayoutWord = 0U;
    DynamicSlabGroupOwnerPlan groupOwnerPlan;
    if (supportsAlignmentBalancedRows && supportsPackedRowWidths &&
        allInputRowsContiguous &&
        (dataType == ge::DT_FLOAT || dataType == ge::DT_INT32) &&
        inputNum >= DYNAMIC_SLAB_MIN_INPUTS &&
        rowsPerBatch >= DYNAMIC_SLAB_MIN_BATCH_ROWS) {
        const uint32_t columns = inputNum + 1U;
        const uint64_t infinity = std::numeric_limits<uint64_t>::max() / 8U;
        std::vector<uint64_t> costs(
            static_cast<size_t>(DYNAMIC_SLAB_MAX_GROUPS + 1U) * columns,
            infinity);
        std::vector<uint16_t> predecessors(costs.size(), 0xffffU);
        const auto state = [columns](uint32_t groups, uint32_t input) {
            return static_cast<size_t>(groups) * columns + input;
        };
        costs[state(0U, 0U)] = 0U;
        for (uint32_t groups = 0U; groups < DYNAMIC_SLAB_MAX_GROUPS;
             ++groups) {
            for (uint32_t begin = 0U; begin < inputNum; ++begin) {
                const uint64_t parent = costs[state(groups, begin)];
                if (parent == infinity) {
                    continue;
                }
                uint32_t groupBytes = 0U;
                for (uint32_t end = begin; end < inputNum; ++end) {
                    groupBytes += inputRowBytes[end];
                    if (groupBytes > DYNAMIC_SLAB_MAX_GROUP_BYTES) {
                        break;
                    }
                    if (groupBytes < DYNAMIC_SLAB_MIN_GROUP_BYTES) {
                        continue;
                    }
                    const int64_t delta = static_cast<int64_t>(groupBytes) -
                        static_cast<int64_t>(DYNAMIC_SLAB_TARGET_BYTES);
                    const uint64_t segmentCost =
                        static_cast<uint64_t>(delta * delta);
                    const size_t child = state(groups + 1U, end + 1U);
                    if (parent + segmentCost < costs[child]) {
                        costs[child] = parent + segmentCost;
                        predecessors[child] = static_cast<uint16_t>(begin);
                    }
                }
            }
        }
        uint64_t bestScore = infinity;
        uint32_t bestGroupCount = 0U;
        for (uint32_t groups = DYNAMIC_SLAB_MIN_GROUPS;
             groups <= DYNAMIC_SLAB_MAX_GROUPS; ++groups) {
            const uint64_t path = costs[state(groups, inputNum)];
            if (path == infinity) {
                continue;
            }
            const int64_t countDelta = static_cast<int64_t>(groups) -
                static_cast<int64_t>(DYNAMIC_SLAB_TARGET_GROUPS);
            const uint64_t score = path +
                static_cast<uint64_t>(countDelta * countDelta) * 100000000U;
            if (score < bestScore) {
                bestScore = score;
                bestGroupCount = groups;
            }
        }
        uint16_t groupEnds[DYNAMIC_SLAB_MAX_GROUPS] = {};
        uint32_t maxGroupBytes = 0U;
        uint32_t end = inputNum;
        bool validPath = bestGroupCount != 0U;
        for (uint32_t remaining = bestGroupCount;
             validPath && remaining != 0U; --remaining) {
            const uint16_t begin = predecessors[state(remaining, end)];
            if (begin == 0xffffU || begin >= end) {
                validPath = false;
                break;
            }
            groupEnds[remaining - 1U] = static_cast<uint16_t>(end);
            uint32_t groupBytes = 0U;
            for (uint32_t input = begin; input < end; ++input) {
                groupBytes += inputRowBytes[input];
            }
            maxGroupBytes = std::max(maxGroupBytes, groupBytes);
            end = begin;
        }
        validPath = validPath && end == 0U;
        uint64_t slabStride = AlignUp(maxGroupBytes, MIN_TILE_BYTES);
        const uint64_t scratchStride = maxPaddedInputRowBytes;
        const uint64_t pointerBytes = AlignUp(
            static_cast<uint64_t>(inputNum) * sizeof(uint64_t),
            MIN_TILE_BYTES);
        const uint64_t dynamicSlabBytes =
            static_cast<uint64_t>(rowsPerBatch) * slabStride +
            2U * static_cast<uint64_t>(rowsPerBatch) * scratchStride +
            pointerBytes;
        supportsDynamicSlabRows = validPath && maxGroupBytes != 0U &&
            slabStride / MIN_TILE_BYTES <= 0xffU &&
            scratchStride / MIN_TILE_BYTES <= 0xffU &&
            dynamicSlabBytes <= ubSize - UB_RESERVED_BYTES;
        // The eight-record Gather table costs 2 KiB and is regenerated for
        // each compact input. Admit it only when enough narrow, unaligned row
        // traffic can amortize that fixed UB and Vector setup; otherwise keep
        // the identical hard-slab plan on key 44.
        supportsDynamicSlabCompactRows = supportsDynamicSlabRows &&
            dataType == ge::DT_FLOAT &&
            dynamicSlabCompactGatherInputCount >=
                DYNAMIC_SLAB_COMPACT_MIN_INPUTS &&
            dynamicSlabCompactGatherRowBytes >=
                DYNAMIC_SLAB_COMPACT_MIN_ROW_BYTES &&
            dynamicSlabBytes + DYNAMIC_SLAB_COMPACT_OFFSET_BYTES <=
                ubSize - UB_RESERVED_BYTES;
        // Wide unaligned FP32 rows use 192-byte block-aligned Gather
        // segments. The same 2 KiB offset table is generated once per input
        // and reused by every segment, so this route does not add UB beyond
        // the compact-Gather allocation. Keep it on an independent key so a
        // failed segmented density gate falls back to the unchanged key 46
        // or key 44 implementation.
        supportsDynamicSlabSegmentedRows =
            supportsDynamicSlabCompactRows &&
            rowsPerBatch >= DYNAMIC_SLAB_SEGMENTED_MIN_BATCH_ROWS &&
            dynamicSlabSegmentedGatherInputCount >=
                DYNAMIC_SLAB_SEGMENTED_MIN_INPUTS &&
            dynamicSlabSegmentedGatherRowBytes >=
                DYNAMIC_SLAB_SEGMENTED_MIN_ROW_BYTES &&
            dynamicSlabBytes + DYNAMIC_SLAB_COMPACT_OFFSET_BYTES <=
                ubSize - UB_RESERVED_BYTES;
        // A third scratch generation keeps two future MTE2 transfers queued
        // behind the current Vector pack. Account for its complete UB layout
        // exactly and keep the established two-slot key 47 as the fallback
        // for larger row batches.
        const uint64_t dynamicSlabTripleBytes =
            static_cast<uint64_t>(rowsPerBatch) * slabStride +
            3U * static_cast<uint64_t>(rowsPerBatch) * scratchStride +
            pointerBytes + DYNAMIC_SLAB_COMPACT_OFFSET_BYTES;
        supportsDynamicSlabSegmentedTripleRows =
            supportsDynamicSlabSegmentedRows &&
            dynamicSlabTripleBytes <= ubSize - UB_RESERVED_BYTES;
        const uint64_t dynamicSlabTripleWideRowsBytes =
            static_cast<uint64_t>(rowsPerBatch) * slabStride +
            3U * static_cast<uint64_t>(rowsPerBatch) * scratchStride +
            pointerBytes + DYNAMIC_SLAB_WIDE_OFFSET_BYTES;
        supportsDynamicSlabSegmentedTripleWideRows =
            supportsDynamicSlabSegmentedTripleRows &&
            dynamicSlabTripleWideRowsBytes <= ubSize - UB_RESERVED_BYTES;
        // Key 68 removes the pointer cache and replaces per-input Scalar
        // offset seeds with one 256-byte read-only Vector base. Account for
        // that exact layout independently even though every current key-67
        // shape saves UB relative to its aligned pointer table.
        const uint64_t dynamicSlabTripleScalarFreeRowsBytes =
            static_cast<uint64_t>(rowsPerBatch) * slabStride +
            3U * static_cast<uint64_t>(rowsPerBatch) * scratchStride +
            DYNAMIC_SLAB_WIDE_OFFSET_BYTES +
            DYNAMIC_SLAB_GATHER_BASE_BYTES;
        supportsDynamicSlabSegmentedTripleScalarFreeRows =
            supportsDynamicSlabSegmentedTripleWideRows &&
            dynamicSlabTripleScalarFreeRowsBytes <=
                ubSize - UB_RESERVED_BYTES;
        prefersDynamicSlabTripleRows =
            supportsDynamicSlabSegmentedTripleRows;

        // Key 49 keeps the established row ownership and output groups.  It
        // sizes the three primary scratch generations by the largest first,
        // second and third per-group requirements, then spends the remaining
        // aligned UB on a small opportunistic fourth generation.  A FIFO
        // scheduler uses that generation only when the next input fits, so
        // the three-generation route remains available for every group.
        const uint32_t adaptiveLayoutIndex =
            CONCAT_DYNAMIC_SLAB_HEADER_WORD + 1U +
            (bestGroupCount + 1U) / 2U;
        if (supportsDynamicSlabSegmentedTripleRows &&
            adaptiveLayoutIndex < CONCAT_LAUNCH_PADDING_WORDS) {
            uint64_t rankCapacities[3] = {};
            uint32_t begin = 0U;
            bool validAdaptiveGroups = true;
            for (uint32_t group = 0U; group < bestGroupCount; ++group) {
                const uint32_t groupEnd = groupEnds[group];
                std::vector<uint64_t> requirements;
                uint64_t localPrefix = 0U;
                for (uint32_t input = begin; input < groupEnd; ++input) {
                    const uint64_t rowBytes = inputRowBytes[input];
                    if (rowBytes == 0U) {
                        continue;
                    }
                    const bool gatherRow =
                        rowBytes >= COMPACT_GATHER_MIN_ROW_BYTES &&
                        rowBytes % MIN_TILE_BYTES != 0U;
                    const uint64_t requirement = gatherRow ?
                        AlignUp(rowBytes, MIN_TILE_BYTES) :
                        AlignUp(localPrefix % MIN_TILE_BYTES + rowBytes,
                                MIN_TILE_BYTES);
                    requirements.push_back(requirement);
                    localPrefix += rowBytes;
                }
                if (requirements.empty()) {
                    validAdaptiveGroups = false;
                    break;
                }
                std::sort(requirements.begin(), requirements.end(),
                    [](uint64_t lhs, uint64_t rhs) { return lhs > rhs; });
                for (uint32_t rank = 0U;
                     rank < 3U && rank < requirements.size(); ++rank) {
                    rankCapacities[rank] = std::max(
                        rankCapacities[rank], requirements[rank]);
                }
                begin = groupEnd;
            }
            const uint64_t fixedBytes =
                pointerBytes + DYNAMIC_SLAB_COMPACT_OFFSET_BYTES;
            const uint64_t usableBytes = ubSize - UB_RESERVED_BYTES;
            const uint64_t perRowBudget = rowsPerBatch == 0U ||
                usableBytes <= fixedBytes ? 0U :
                (usableBytes - fixedBytes) / rowsPerBatch /
                    MIN_TILE_BYTES * MIN_TILE_BYTES;
            const uint64_t primaryBytes = rankCapacities[0] +
                rankCapacities[1] + rankCapacities[2];
            const uint64_t auxiliaryCapacity =
                perRowBudget > slabStride + primaryBytes ?
                perRowBudget - slabStride - primaryBytes : 0U;
            validAdaptiveGroups = validAdaptiveGroups &&
                rankCapacities[0] != 0U &&
                rankCapacities[1] != 0U &&
                rankCapacities[2] != 0U &&
                auxiliaryCapacity >= MIN_TILE_BYTES &&
                auxiliaryCapacity <= rankCapacities[2] &&
                auxiliaryCapacity / MIN_TILE_BYTES <= 0xffU;

            uint32_t auxiliaryLoads = 0U;
            begin = 0U;
            const uint64_t capacities[4] = {
                rankCapacities[0], rankCapacities[1],
                rankCapacities[2], auxiliaryCapacity};
            for (uint32_t group = 0U;
                 validAdaptiveGroups && group < bestGroupCount; ++group) {
                const uint32_t groupEnd = groupEnds[group];
                std::vector<uint64_t> requirements;
                uint64_t localPrefix = 0U;
                for (uint32_t input = begin; input < groupEnd; ++input) {
                    const uint64_t rowBytes = inputRowBytes[input];
                    if (rowBytes == 0U) {
                        continue;
                    }
                    const bool gatherRow =
                        rowBytes >= COMPACT_GATHER_MIN_ROW_BYTES &&
                        rowBytes % MIN_TILE_BYTES != 0U;
                    requirements.push_back(gatherRow ?
                        AlignUp(rowBytes, MIN_TILE_BYTES) :
                        AlignUp(localPrefix % MIN_TILE_BYTES + rowBytes,
                                MIN_TILE_BYTES));
                    localPrefix += rowBytes;
                }
                uint32_t freeMask = 0xfU;
                uint32_t queue[4] = {};
                uint32_t queueHead = 0U;
                uint32_t queueCount = 0U;
                uint32_t scan = 0U;
                while (scan < requirements.size() || queueCount != 0U) {
                    while (scan < requirements.size() && queueCount < 4U) {
                        uint32_t selected = 4U;
                        uint64_t selectedCapacity =
                            std::numeric_limits<uint64_t>::max();
                        for (uint32_t slot = 0U; slot < 4U; ++slot) {
                            if ((freeMask & (1U << slot)) != 0U &&
                                capacities[slot] >= requirements[scan] &&
                                capacities[slot] < selectedCapacity) {
                                selected = slot;
                                selectedCapacity = capacities[slot];
                            }
                        }
                        if (selected == 4U) {
                            break;
                        }
                        freeMask &= ~(1U << selected);
                        queue[(queueHead + queueCount) & 3U] = selected;
                        ++queueCount;
                        ++scan;
                        auxiliaryLoads += static_cast<uint32_t>(selected == 3U);
                    }
                    if (queueCount == 0U) {
                        validAdaptiveGroups = false;
                        break;
                    }
                    const uint32_t released = queue[queueHead];
                    queueHead = (queueHead + 1U) & 3U;
                    --queueCount;
                    freeMask |= 1U << released;
                }
                begin = groupEnd;
            }
            supportsDynamicSlabSegmentedAdaptiveRows =
                validAdaptiveGroups &&
                auxiliaryLoads >= DYNAMIC_SLAB_ADAPTIVE_MIN_AUX_LOADS;
            if (supportsDynamicSlabSegmentedAdaptiveRows) {
                for (uint32_t slot = 0U; slot < 4U; ++slot) {
                    adaptiveScratchCapacities[slot] =
                        static_cast<uint32_t>(capacities[slot]);
                }
                adaptiveLayoutWord =
                    (adaptiveScratchCapacities[1] / MIN_TILE_BYTES) |
                    ((adaptiveScratchCapacities[2] / MIN_TILE_BYTES) << 8U) |
                    ((adaptiveScratchCapacities[3] / MIN_TILE_BYTES) << 16U) |
                    (DYNAMIC_SLAB_ADAPTIVE_METADATA_VERSION << 24U);
            }
        }

        // Key 56 uses sixteen Gather records. The additional 2 KiB offset
        // table changes the feasible four-generation partitions, so search
        // its exact UB budget independently instead of weakening key 49.
        // The Device consumes a FIFO head whenever the next requirement does
        // not fit; mirror that policy exactly and rank every MTE2 admission by
        // its outstanding queue depth.
        if (supportsDynamicSlabSegmentedRows &&
            adaptiveLayoutIndex < CONCAT_LAUNCH_PADDING_WORDS) {
            std::vector<std::vector<uint32_t>> groupRequirements;
            groupRequirements.reserve(bestGroupCount);
            uint32_t maxRequirementBlocks = 0U;
            uint32_t begin = 0U;
            bool validWideGroups = true;
            for (uint32_t group = 0U; group < bestGroupCount; ++group) {
                const uint32_t groupEnd = groupEnds[group];
                std::vector<uint32_t> requirements;
                uint64_t localPrefix = 0U;
                for (uint32_t input = begin; input < groupEnd; ++input) {
                    const uint64_t rowBytes = inputRowBytes[input];
                    if (rowBytes == 0U) {
                        continue;
                    }
                    const bool gatherRow =
                        rowBytes >= COMPACT_GATHER_MIN_ROW_BYTES &&
                        rowBytes % MIN_TILE_BYTES != 0U;
                    const uint64_t requirement = gatherRow ?
                        AlignUp(rowBytes, MIN_TILE_BYTES) :
                        AlignUp(localPrefix % MIN_TILE_BYTES + rowBytes,
                                MIN_TILE_BYTES);
                    const uint32_t blocks = static_cast<uint32_t>(
                        requirement / MIN_TILE_BYTES);
                    requirements.push_back(blocks);
                    maxRequirementBlocks = std::max(
                        maxRequirementBlocks, blocks);
                    localPrefix += rowBytes;
                }
                if (requirements.empty()) {
                    validWideGroups = false;
                    break;
                }
                groupRequirements.push_back(std::move(requirements));
                begin = groupEnd;
            }

            // Search one- and two-table layouts with the same FIFO model.
            // The dual-table candidate spends another 4 KiB on independent
            // Scalar/Vector lifetimes, so its scratch capacities must be
            // proven separately rather than borrowing key 56's plan.
            const auto searchWideLayout = [&groupRequirements,
                                           maxRequirementBlocks,
                                           validWideGroups, ubSize,
                                           rowsPerBatch, slabStride,
                                           pointerBytes](
                uint64_t offsetBytes, uint32_t outputCapacities[4],
                uint32_t& outputLayoutWord) {
                const uint64_t usableBytes = ubSize - UB_RESERVED_BYTES;
                const uint64_t fixedBytes = pointerBytes + offsetBytes;
                const uint64_t perRowBudget = rowsPerBatch == 0U ||
                    usableBytes <= fixedBytes ? 0U :
                    (usableBytes - fixedBytes) / rowsPerBatch /
                        MIN_TILE_BYTES * MIN_TILE_BYTES;
                const uint32_t scratchBudgetBlocks =
                    perRowBudget > slabStride ? static_cast<uint32_t>(
                        (perRowBudget - slabStride) / MIN_TILE_BYTES) : 0U;
                const bool validBudget = validWideGroups &&
                    maxRequirementBlocks != 0U &&
                    maxRequirementBlocks <= 0xffU &&
                    scratchBudgetBlocks >= maxRequirementBlocks + 3U;

                bool foundLayout = false;
                uint32_t bestQueueScore = 0U;
                uint32_t bestDepthFourLoads = 0U;
                uint32_t bestSmallestSlotLoads = 0U;
                uint32_t bestCapacityBlocks[4] = {};
                for (uint32_t second = maxRequirementBlocks;
                     validBudget && second != 0U; --second) {
                    for (uint32_t third = second; third != 0U; --third) {
                        for (uint32_t fourth = third; fourth != 0U; --fourth) {
                            const uint32_t capacities[4] = {
                                maxRequirementBlocks, second, third, fourth};
                            if (capacities[0] + capacities[1] +
                                    capacities[2] + capacities[3] >
                                scratchBudgetBlocks) {
                                continue;
                            }

                            bool validCandidate = true;
                            uint32_t queueScore = 0U;
                            uint32_t depthFourLoads = 0U;
                            uint32_t smallestSlotLoads = 0U;
                            for (const auto& requirements : groupRequirements) {
                                uint32_t freeMask = 0xfU;
                                uint32_t queue[4] = {};
                                uint32_t queueHead = 0U;
                                uint32_t queueCount = 0U;
                                uint32_t scan = 0U;
                                while (scan < requirements.size() ||
                                       queueCount != 0U) {
                                    while (scan < requirements.size() &&
                                           queueCount < 4U) {
                                        uint32_t selected = 4U;
                                        uint32_t selectedCapacity =
                                            std::numeric_limits<uint32_t>::max();
                                        for (uint32_t slot = 0U; slot < 4U;
                                             ++slot) {
                                            if ((freeMask & (1U << slot)) != 0U &&
                                                capacities[slot] >=
                                                    requirements[scan] &&
                                                capacities[slot] <
                                                    selectedCapacity) {
                                                selected = slot;
                                                selectedCapacity =
                                                    capacities[slot];
                                            }
                                        }
                                        if (selected == 4U) {
                                            break;
                                        }
                                        freeMask &= ~(1U << selected);
                                        queue[(queueHead + queueCount) & 3U] =
                                            selected;
                                        ++queueCount;
                                        queueScore += queueCount;
                                        depthFourLoads += static_cast<uint32_t>(
                                            queueCount == 4U);
                                        smallestSlotLoads +=
                                            static_cast<uint32_t>(
                                                selected == 3U);
                                        ++scan;
                                    }
                                    if (queueCount == 0U) {
                                        validCandidate = false;
                                        break;
                                    }
                                    const uint32_t released = queue[queueHead];
                                    queueHead = (queueHead + 1U) & 3U;
                                    --queueCount;
                                    freeMask |= 1U << released;
                                }
                                if (!validCandidate) {
                                    break;
                                }
                            }

                            const bool better = validCandidate &&
                                (!foundLayout ||
                                 queueScore > bestQueueScore ||
                                 (queueScore == bestQueueScore &&
                                  depthFourLoads > bestDepthFourLoads) ||
                                 (queueScore == bestQueueScore &&
                                  depthFourLoads == bestDepthFourLoads &&
                                  smallestSlotLoads >
                                      bestSmallestSlotLoads));
                            if (better) {
                                foundLayout = true;
                                bestQueueScore = queueScore;
                                bestDepthFourLoads = depthFourLoads;
                                bestSmallestSlotLoads = smallestSlotLoads;
                                for (uint32_t slot = 0U; slot < 4U; ++slot) {
                                    bestCapacityBlocks[slot] = capacities[slot];
                                }
                            }
                        }
                    }
                }

                bool supported = validBudget && foundLayout &&
                    bestSmallestSlotLoads >=
                        DYNAMIC_SLAB_ADAPTIVE_MIN_AUX_LOADS;
                uint64_t requiredBytes = fixedBytes +
                    static_cast<uint64_t>(rowsPerBatch) * slabStride;
                if (supported) {
                    for (uint32_t slot = 0U; slot < 4U; ++slot) {
                        outputCapacities[slot] =
                            bestCapacityBlocks[slot] * MIN_TILE_BYTES;
                        requiredBytes += static_cast<uint64_t>(rowsPerBatch) *
                            outputCapacities[slot];
                    }
                    supported = requiredBytes <= usableBytes;
                }
                if (supported) {
                    outputLayoutWord =
                        (outputCapacities[1] / MIN_TILE_BYTES) |
                        ((outputCapacities[2] / MIN_TILE_BYTES) << 8U) |
                        ((outputCapacities[3] / MIN_TILE_BYTES) << 16U) |
                        (DYNAMIC_SLAB_ADAPTIVE_METADATA_VERSION << 24U);
                }
                return supported;
            };

            supportsDynamicSlabWideAdaptiveRows = searchWideLayout(
                DYNAMIC_SLAB_WIDE_OFFSET_BYTES,
                wideAdaptiveScratchCapacities, wideAdaptiveLayoutWord);
            supportsDynamicSlabDualOffsetRows = searchWideLayout(
                DYNAMIC_SLAB_DUAL_WIDE_OFFSET_BYTES,
                dualOffsetScratchCapacities, dualOffsetLayoutWord);
            // Key 58 keeps key 56's one-table UB plan and synthesizes every
            // offset on PIPE_V. It avoids key 57's extra 4 KiB table, while
            // key 59 below separately targets Scalar UB pointer aliasing.
            supportsDynamicSlabVectorOffsetRows =
                supportsDynamicSlabWideAdaptiveRows;
            // Key 59 keeps key 58's Vector-only offset table, but avoids the
            // Scalar UB reads of cached input pointers that CANN 8.5 protects
            // with a whole-Vector drain on every input.
            supportsDynamicSlabDirectPointerRows =
                supportsDynamicSlabVectorOffsetRows;
            // Key 60 uses 256 bytes of the pointer-cache space that key 59
            // releases to keep one read-only [0, 4, ..., 252] offset record.
            // The Host plan already reserves the larger pointer table, so
            // scratch capacities and the complete UB bound remain unchanged.
            supportsDynamicSlabPrecomputedOffsetRows =
                supportsDynamicSlabDirectPointerRows;
            // Key 69 combines key 68's Scalar-UB-free metadata and widened
            // Gather segments with key 60's four-slot adaptive FIFO. Reuse
            // the proven capacity tuple, then charge the actual pointer-free
            // layout instead of the conservative key-60 pointer reservation.
            const uint64_t dynamicSlabWideAdaptiveScalarFreeBytes =
                static_cast<uint64_t>(rowsPerBatch) *
                    (slabStride + wideAdaptiveScratchCapacities[0] +
                     wideAdaptiveScratchCapacities[1] +
                     wideAdaptiveScratchCapacities[2] +
                     wideAdaptiveScratchCapacities[3]) +
                DYNAMIC_SLAB_WIDE_OFFSET_BYTES +
                DYNAMIC_SLAB_GATHER_BASE_BYTES;
            supportsDynamicSlabWideAdaptiveScalarFreeRows =
                supportsDynamicSlabSegmentedTripleScalarFreeRows &&
                supportsDynamicSlabWideAdaptiveRows &&
                dynamicSlabWideAdaptiveScalarFreeBytes <=
                    ubSize - UB_RESERVED_BYTES;
            // Key 79 keeps key 69's complete UB allocation and places a
            // temporary 32-record Gather table in the unused tail of the
            // scratch generation that already owns the compact input.  This
            // removes repeat-16 Gather issues without reserving another
            // global table.  Mirror the exact four-slot FIFO and admit the
            // route only for a single row batch with a substantial predicted
            // issue reduction at the smallest block row count.
            if (supportsDynamicSlabWideAdaptiveScalarFreeRows &&
                rowsPerBatch == blockFactorRows && usedCores != 0U) {
                const uint32_t probeRows = static_cast<uint32_t>(
                    outer / static_cast<uint64_t>(usedCores));
                uint32_t gatherSavings = 0U;
                uint32_t groupStart = 0U;
                bool validScratchOffsetPlan = probeRows > 32U;
                for (uint32_t group = 0U;
                     validScratchOffsetPlan && group < bestGroupCount;
                     ++group) {
                    const uint32_t groupEnd = groupEnds[group];
                    uint32_t freeMask = 0xfU;
                    uint32_t queue[4] = {};
                    uint32_t queueHead = 0U;
                    uint32_t queueCount = 0U;
                    uint32_t scan = groupStart;
                    uint64_t localPrefix = 0U;
                    while (scan < groupEnd || queueCount != 0U) {
                        while (scan < groupEnd && queueCount < 4U) {
                            const uint64_t rowBytes = inputRowBytes[scan];
                            if (rowBytes == 0U) {
                                ++scan;
                                continue;
                            }
                            const bool gatherRow =
                                rowBytes >= COMPACT_GATHER_MIN_ROW_BYTES &&
                                rowBytes % MIN_TILE_BYTES != 0U;
                            const uint64_t requirement = gatherRow ?
                                AlignUp(rowBytes, MIN_TILE_BYTES) :
                                AlignUp(localPrefix % MIN_TILE_BYTES +
                                        rowBytes, MIN_TILE_BYTES);
                            uint32_t selected = 4U;
                            uint32_t selectedCapacity =
                                std::numeric_limits<uint32_t>::max();
                            for (uint32_t slot = 0U; slot < 4U; ++slot) {
                                const uint32_t capacity =
                                    wideAdaptiveScratchCapacities[slot];
                                if ((freeMask & (1U << slot)) != 0U &&
                                    capacity >= requirement &&
                                    capacity < selectedCapacity) {
                                    selected = slot;
                                    selectedCapacity = capacity;
                                }
                            }
                            if (selected == 4U) {
                                break;
                            }
                            if (gatherRow &&
                                rowBytes >=
                                    DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES) {
                                const uint64_t tableOffset = AlignUp(
                                    static_cast<uint64_t>(probeRows) *
                                        rowBytes,
                                    MIN_TILE_BYTES);
                                const uint64_t scratchBytes =
                                    static_cast<uint64_t>(probeRows) *
                                        selectedCapacity;
                                if (tableOffset +
                                        DYNAMIC_SLAB_SCRATCH_OFFSET32_BYTES <=
                                    scratchBytes) {
                                    uint64_t segmentSpan = rowBytes;
                                    if (rowBytes >=
                                        DYNAMIC_SLAB_SEGMENTED_GATHER_MIN_ROW_BYTES) {
                                        segmentSpan = 192U;
                                        const uint64_t leftBytes =
                                            localPrefix % MIN_TILE_BYTES;
                                        const uint64_t widenedSpan =
                                            leftBytes == 0U ? 256U : 224U;
                                        const uint64_t baselineSegments =
                                            (rowBytes + segmentSpan - 1U) /
                                                segmentSpan;
                                        const uint64_t widenedSegments =
                                            (rowBytes + widenedSpan - 1U) /
                                                widenedSpan;
                                        if (widenedSegments <= baselineSegments) {
                                            segmentSpan = widenedSpan;
                                        }
                                    }
                                    const uint32_t segments =
                                        static_cast<uint32_t>(
                                            (rowBytes + segmentSpan - 1U) /
                                            segmentSpan);
                                    const uint32_t repeat16 =
                                        (probeRows + 15U) / 16U;
                                    const uint32_t repeat32 =
                                        (probeRows + 31U) / 32U;
                                    gatherSavings +=
                                        segments * (repeat16 - repeat32);
                                }
                            }
                            freeMask &= ~(1U << selected);
                            queue[(queueHead + queueCount) & 3U] = selected;
                            ++queueCount;
                            ++scan;
                            localPrefix += rowBytes;
                        }
                        if (queueCount == 0U) {
                            validScratchOffsetPlan = false;
                            break;
                        }
                        const uint32_t released = queue[queueHead];
                        queueHead = (queueHead + 1U) & 3U;
                        --queueCount;
                        freeMask |= 1U << released;
                    }
                    groupStart = groupEnd;
                }
                supportsDynamicSlabScratchOffset32Rows =
                    validScratchOffsetPlan &&
                    gatherSavings >=
                        DYNAMIC_SLAB_SCRATCH_OFFSET32_MIN_SAVINGS;
            }
            // Key 92 changes ownership rather than tuning key 79's row-local
            // FIFO.  Thirty-two output groups own all rows, while the eight
            // heaviest groups receive a second row shard on a 40-AIV launch.
            // Each block derives a group-specific slab and four-slot layout
            // from packed widths, so the existing 448-byte tiling footprint
            // still contains the complete schedule and fallback metadata.
            if (supportsDynamicSlabScratchOffset32Rows) {
                supportsDynamicSlabGroupOwnerRows =
                    BuildDynamicSlabGroupOwnerPlan(
                        inputRowBytes, inputNum, outputRowBytes, outer,
                        usedCores, ubSize, bestGroupCount, groupOwnerPlan);
            }
            // Key 48 is the verified performance baseline whenever its
            // complete three-scratch UB proof succeeds. Later adaptive and
            // boundary experiments remain available only for shapes where
            // that route cannot fit. In particular, do not let key 64 replace
            // key 48's established group boundaries before the final choice.
            if (!prefersDynamicSlabTripleRows) {
                uint16_t boundaryGroupEnds[DYNAMIC_SLAB_MAX_GROUPS] = {};
                uint32_t boundaryGroupCount = 0U;
                uint64_t boundaryGroupBytes = 0U;
                uint64_t boundaryMaxGroupBytes = 0U;
                bool validBoundaryGroups = true;
                for (uint32_t input = 0U; input < inputNum; ++input) {
                    const uint64_t rowBytes = inputRowBytes[input];
                    if (boundaryGroupBytes != 0U &&
                        boundaryGroupBytes + rowBytes >
                            DYNAMIC_DOUBLE_SLAB_BOUNDARY_MAX_SLAB_STRIDE) {
                        if (boundaryGroupCount >= DYNAMIC_SLAB_MAX_GROUPS) {
                            validBoundaryGroups = false;
                            break;
                        }
                        boundaryGroupEnds[boundaryGroupCount++] =
                            static_cast<uint16_t>(input);
                        boundaryMaxGroupBytes = std::max(
                            boundaryMaxGroupBytes, boundaryGroupBytes);
                        boundaryGroupBytes = 0U;
                    }
                    boundaryGroupBytes += rowBytes;
                }
                if (validBoundaryGroups && boundaryGroupBytes != 0U) {
                    if (boundaryGroupCount >= DYNAMIC_SLAB_MAX_GROUPS) {
                        validBoundaryGroups = false;
                    } else {
                        boundaryGroupEnds[boundaryGroupCount++] =
                            static_cast<uint16_t>(inputNum);
                        boundaryMaxGroupBytes = std::max(
                            boundaryMaxGroupBytes, boundaryGroupBytes);
                    }
                }
                const uint64_t boundarySlabStride = AlignUp(
                    boundaryMaxGroupBytes, MIN_TILE_BYTES);
                const uint64_t doubleSlabBoundaryBytes =
                    2U * static_cast<uint64_t>(rowsPerBatch) *
                        boundarySlabStride +
                    2U * static_cast<uint64_t>(rowsPerBatch) *
                        DYNAMIC_DOUBLE_SLAB_BOUNDARY_SCRATCH_ROW_BYTES +
                    DYNAMIC_DOUBLE_SLAB_BOUNDARY_OFFSET_BYTES +
                    DYNAMIC_DOUBLE_SLAB_BOUNDARY_BASE_BYTES;
                supportsDynamicDoubleSlabBoundaryRows =
                    supportsDynamicSlabPrecomputedOffsetRows &&
                    validBoundaryGroups &&
                    boundaryGroupCount >= DYNAMIC_SLAB_MIN_GROUPS &&
                    boundaryGroupCount < DYNAMIC_SLAB_MAX_GROUPS &&
                    boundarySlabStride != 0U &&
                    boundarySlabStride <=
                        DYNAMIC_DOUBLE_SLAB_BOUNDARY_MAX_SLAB_STRIDE &&
                    doubleSlabBoundaryBytes <= ubSize - UB_RESERVED_BYTES;
                if (supportsDynamicDoubleSlabBoundaryRows) {
                    bestGroupCount = boundaryGroupCount;
                    slabStride = boundarySlabStride;
                    for (uint32_t group = 0U; group < bestGroupCount; ++group) {
                        groupEnds[group] = boundaryGroupEnds[group];
                    }
                }
            }
        }
        if (supportsDynamicSlabRows) {
            const uint64_t encodedScratchStride =
                supportsDynamicSlabWideAdaptiveScalarFreeRows ?
                wideAdaptiveScratchCapacities[0] :
                (prefersDynamicSlabTripleRows ? scratchStride :
                (supportsDynamicDoubleSlabBoundaryRows ?
                DYNAMIC_DOUBLE_SLAB_BOUNDARY_SCRATCH_ROW_BYTES :
                (supportsDynamicSlabPrecomputedOffsetRows ?
                wideAdaptiveScratchCapacities[0] :
                (supportsDynamicSlabDirectPointerRows ?
                wideAdaptiveScratchCapacities[0] :
                (supportsDynamicSlabVectorOffsetRows ?
                wideAdaptiveScratchCapacities[0] :
                (supportsDynamicSlabDualOffsetRows ?
                dualOffsetScratchCapacities[0] :
                (supportsDynamicSlabWideAdaptiveRows ?
                wideAdaptiveScratchCapacities[0] :
                (supportsDynamicSlabSegmentedAdaptiveRows ?
                    adaptiveScratchCapacities[0] : scratchStride))))))));
            launchPadding[CONCAT_DYNAMIC_SLAB_HEADER_WORD] =
                (bestGroupCount & 0xffU) |
                (static_cast<uint32_t>(slabStride / MIN_TILE_BYTES) << 8U) |
                (static_cast<uint32_t>(
                    encodedScratchStride / MIN_TILE_BYTES) << 16U) |
                (DYNAMIC_SLAB_METADATA_VERSION << 24U);
            for (uint32_t group = 0U; group < bestGroupCount; ++group) {
                launchPadding[CONCAT_DYNAMIC_SLAB_HEADER_WORD + 1U +
                    group / 2U] |= static_cast<uint32_t>(groupEnds[group]) <<
                    ((group & 1U) * 16U);
            }
            if (supportsDynamicSlabWideAdaptiveScalarFreeRows) {
                launchPadding[adaptiveLayoutIndex] =
                    wideAdaptiveLayoutWord;
            } else if (prefersDynamicSlabTripleRows) {
                // Key 48 uses the uniform scratch stride encoded above.
            } else if (supportsDynamicDoubleSlabBoundaryRows) {
                // Key 64 has a fixed two-scratch layout and no adaptive word.
            } else if (supportsDynamicSlabPrecomputedOffsetRows) {
                launchPadding[adaptiveLayoutIndex] =
                    wideAdaptiveLayoutWord;
            } else if (supportsDynamicSlabDirectPointerRows) {
                launchPadding[adaptiveLayoutIndex] =
                    wideAdaptiveLayoutWord;
            } else if (supportsDynamicSlabVectorOffsetRows) {
                launchPadding[adaptiveLayoutIndex] =
                    wideAdaptiveLayoutWord;
            } else if (supportsDynamicSlabDualOffsetRows) {
                launchPadding[adaptiveLayoutIndex] =
                    dualOffsetLayoutWord;
            } else if (supportsDynamicSlabWideAdaptiveRows) {
                launchPadding[adaptiveLayoutIndex] =
                    wideAdaptiveLayoutWord;
            } else if (supportsDynamicSlabSegmentedAdaptiveRows) {
                launchPadding[adaptiveLayoutIndex] = adaptiveLayoutWord;
            }
        }
        if (supportsDynamicSlabGroupOwnerRows) {
            std::fill(
                launchPadding + CONCAT_PACKED_WIDTH_WORDS,
                launchPadding + CONCAT_LAUNCH_PADDING_WORDS, 0U);
            launchPadding[CONCAT_DYNAMIC_SLAB_HEADER_WORD] =
                (groupOwnerPlan.groupCount & 0xffU) |
                ((groupOwnerPlan.availableUbBytes / MIN_TILE_BYTES) << 8U) |
                (DYNAMIC_SLAB_GROUP_OWNER_METADATA_VERSION << 24U);
            for (uint32_t group = 0U;
                 group < groupOwnerPlan.groupCount; ++group) {
                launchPadding[CONCAT_DYNAMIC_SLAB_HEADER_WORD + 1U +
                    group / 2U] |=
                    static_cast<uint32_t>(groupOwnerPlan.groupEnds[group]) <<
                    ((group & 1U) * 16U);
                launchPadding[DYNAMIC_SLAB_GROUP_OWNER_ARENA_PAIR_WORD +
                    group / 4U] |=
                    static_cast<uint32_t>(
                        groupOwnerPlan.arenaPairs[group]) <<
                    ((group & 3U) * 8U);
            }
            for (uint32_t block = 0U;
                 block < groupOwnerPlan.scheduleCount; ++block) {
                launchPadding[CONCAT_PACKED_WIDTH_WORDS + block / 4U] |=
                    static_cast<uint32_t>(groupOwnerPlan.schedule[block]) <<
                    ((block & 3U) * 8U);
            }
            launchPadding[DYNAMIC_SLAB_GROUP_OWNER_TIGHT_BUDGET_WORD] =
                groupOwnerPlan.tightArenaAvailableUbBytes / MIN_TILE_BYTES;
        }
    }
    const uint64_t compactPointerBytes = AlignUp(
        static_cast<uint64_t>(inputNum) * sizeof(uint64_t), MIN_TILE_BYTES);
    const uint64_t compactInputBytes = AlignUp(
        static_cast<uint64_t>(rowsPerBatch) *
            COMPACT_GATHER_MAX_STORAGE_ROW_BYTES,
        MIN_TILE_BYTES);
    const bool supportsCompactGatherRows =
        supportsAlignmentBalancedRows && supportsPackedRowWidths &&
        allInputRowsContiguous &&
        (dataType == ge::DT_FLOAT || dataType == ge::DT_INT32) &&
        rowsPerBatch >= COMPACT_GATHER_MIN_BATCH_ROWS &&
        compactGatherInputCount >= COMPACT_GATHER_MIN_INPUTS &&
        3U * tileBytes + compactPointerBytes + compactInputBytes +
            COMPACT_GATHER_OFFSET_BYTES <= ubSize - UB_RESERVED_BYTES;
    if (supportsWideRowAssembly) {
        for (uint32_t input = 0U; input < inputNum; ++input) {
            launchPadding[input] = inputRowBytes[input];
        }
        launchPadding[WIDE_ROW_ARENA_WORD] = wideRowPlan.arenaBytes;
        launchPadding[WIDE_ROW_VERSION_WORD] = wideRowPlan.bandBytes != 0U ?
            WIDE_ROW_BAND_VERSION : WIDE_ROW_METADATA_VERSION;
        launchPadding[WIDE_ROW_BAND_WORD] = wideRowPlan.bandBytes;
    }
    tiling.set_launchPadding(launchPadding);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetBlockDim(usedCores);
    const bool supportsCachedNarrowPointers = inputNum == CONCAT_MAX_INPUTS &&
        outer >= 4096U && 2U * tileBytes +
        CONCAT_MAX_INPUTS * sizeof(uint64_t) <= ubSize;
    uint64_t tilingKey = 0U;
    if (supportsSegmentedAligned32 &&
        rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = scratchGenerations == 3U ? 31U : 30U;
    } else if (supportsSegmentedNarrow16 &&
        rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = 29U;
    } else if (supportsSegmentedNarrowChunks &&
        rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = 24U;
    } else if (supportsSegmentedNarrowB32 &&
        rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = 28U;
    } else if (supportsBatchedNarrowChunks && supportsNarrowChunks &&
        rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = 23U;
    } else if (supportsNarrowChunks && rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = 20U;
    } else if (supportsNarrowB16 && rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = supportsCachedNarrowPointers ? 26U : 19U;
    } else if (supportsNarrowB32 && rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = 16U;
    } else if (supportsNarrowB8 && rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = supportsCachedNarrowPointers ? 25U : 17U;
    } else if (supportsAlignedAssembly && rowsPerBatch != 0U && blockFactorRows != 0U) {
        tilingKey = allInputRowsSameWidth ? 22U : 21U;
    } else if (supportsInputParallel && rowsPerBatch != 0U) {
        tilingKey = 18U;
    } else if (supportsPipelinedByteRange) {
        tilingKey = supportsIncrementalFlatRange ? 78U : 33U;
    } else if (useBalancedRowGroups && rowsPerBatch != 0U &&
        blockFactorRows != 0U) {
        const uint64_t pointerTableBytes = AlignUp(
            static_cast<uint64_t>(inputNum) * sizeof(uint64_t), MIN_TILE_BYTES);
        if (supportsDynamicSlabRows) {
            tilingKey = supportsDynamicSlabGroupOwnerRows ? 92U :
                (supportsDynamicSlabScratchOffset32Rows ? 79U :
                (supportsDynamicSlabWideAdaptiveScalarFreeRows ? 69U :
                (supportsDynamicSlabSegmentedTripleScalarFreeRows ? 68U :
                (supportsDynamicSlabSegmentedTripleWideRows ? 67U :
                (supportsDynamicSlabSegmentedTripleRows ? 48U :
                (supportsDynamicDoubleSlabBoundaryRows ? 64U :
                (supportsDynamicSlabPrecomputedOffsetRows ? 60U :
                (supportsDynamicSlabDirectPointerRows ? 59U :
                (supportsDynamicSlabVectorOffsetRows ? 58U :
                (supportsDynamicSlabDualOffsetRows ? 57U :
                (supportsDynamicSlabWideAdaptiveRows ? 56U :
                (supportsDynamicSlabSegmentedAdaptiveRows ? 49U :
                (supportsDynamicSlabSegmentedRows ? 47U :
                (supportsDynamicSlabCompactRows ? 46U : 44U))))))))))))));
        } else if (3U * tileBytes + pointerTableBytes <=
                   ubSize - UB_RESERVED_BYTES) {
            tilingKey = supportsCompactGatherRows ? 43U :
                (supportsAlignmentBalancedRows ? 39U :
                (useBalancedRows ? 35U :
                (supportsPackedRowWidths ? 37U : 34U)));
        } else {
            tilingKey = 32U;
        }
    } else if (supportsDedicatedDirectRows) {
        tilingKey = supportsWideRowAssembly ?
            (wideRowPlan.bandBytes != 0U ? 130U : 128U) : 91U;
    } else if (rowsPerBatch != 0U && blockFactorRows != 0U &&
        useThreeBufferDirect &&
        3U * tileBytes <= ubSize - UB_RESERVED_BYTES) {
        if (rowsPerBatch == blockFactorRows &&
            tileBytes <= FIVE_BUFFER_MAX_TILE_BYTES &&
            5U * tileBytes <= ubSize - UB_RESERVED_BYTES) {
            tilingKey = 5U;
            if (outer == static_cast<uint64_t>(usedCores) * blockFactorRows) {
                tilingKey = 15U;
            }
        } else if (rowsPerBatch == blockFactorRows &&
                   tileBytes <= FOUR_BUFFER_MAX_TILE_BYTES &&
                   4U * tileBytes <= ubSize - UB_RESERVED_BYTES) {
            tilingKey = 4U;
        } else {
            const uint64_t directBatchCount =
                (blockFactorRows + rowsPerBatch - 1U) / rowsPerBatch;
            tilingKey = inputNum <= 8U && directBatchCount >= 3U ? 40U : 3U;
        }
    } else if (rowsPerBatch != 0U && blockFactorRows != 0U && supportsDirectRowDma &&
        outputRowBytes % MIN_TILE_BYTES == 0U) {
        tilingKey = tileBytes <= FIVE_BUFFER_MAX_TILE_BYTES ? 5U :
            (tileBytes <= FOUR_BUFFER_MAX_TILE_BYTES ? 4U : 2U);
        if (tilingKey == 5U && rowsPerBatch == blockFactorRows &&
            outer == static_cast<uint64_t>(usedCores) * blockFactorRows) {
            tilingKey = 15U;
        }
    }
    context->SetTilingKey(tilingKey);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes != nullptr) {
        workspaceSizes[0] = 0U;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr || context->GetComputeNodeInfo() == nullptr) {
        return GRAPH_FAILED;
    }
    auto instanceInfo = context->GetComputeNodeInfo()->GetInputInstanceInfo(0);
    const auto* attrs = context->GetAttrs();
    auto* outputShape = context->GetOutputShape(0);
    if (instanceInfo == nullptr || attrs == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    const int64_t inputNum = static_cast<int64_t>(instanceInfo->GetInstanceNum());
    const gert::Shape* firstShape = context->GetDynamicInputShape(0, 0);
    const int64_t* dimPtr = attrs->GetAttrPointer<int64_t>(0);
    if (inputNum <= 0 || firstShape == nullptr || dimPtr == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *firstShape;
    const int64_t rank = static_cast<int64_t>(outputShape->GetDimNum());
    int64_t dim = *dimPtr;
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        return GRAPH_FAILED;
    }
    int64_t concatDim = 0;
    for (int64_t inputIdx = 0; inputIdx < inputNum; ++inputIdx) {
        const gert::Shape* shape = context->GetDynamicInputShape(0, inputIdx);
        if (shape == nullptr || static_cast<int64_t>(shape->GetDimNum()) != rank) {
            return GRAPH_FAILED;
        }
        concatDim += shape->GetDim(dim);
    }
    outputShape->SetDim(dim, concatDim);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetDynamicInputDataType(0, 0));
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Concat : public OpDef {
public:
    explicit Concat(const char* name) : OpDef(name)
    {
        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .IgnoreContiguous();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}  // namespace ops
