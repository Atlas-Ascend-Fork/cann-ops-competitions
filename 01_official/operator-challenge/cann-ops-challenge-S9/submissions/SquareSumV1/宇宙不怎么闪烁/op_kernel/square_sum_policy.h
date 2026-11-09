#pragma once
#include <cstdint>

// Shared by host tiling and device dispatch; kept under op_kernel so zip_op
// includes this policy with both source packages. No shape-specific cases.
#ifndef SQUARE_SUM_POLICY_INLINE
#define SQUARE_SUM_POLICY_INLINE inline
#endif
namespace square_sum_policy {
constexpr uint32_t kUbBytes = 192U * 1024U;
constexpr uint32_t kBalancedUbBytes = 190U * 1024U;
constexpr uint32_t kVectorElems = 64U;
constexpr uint32_t kMaxRepeat = 255U;
// Fixed wide BF16 stripes; each input slot has one unread 32B bank offset.
constexpr uint32_t kFusedSuffixCols = 1024U;
constexpr uint32_t kFusedSuffixInputPadBytes = 32U;

SQUARE_SUM_POLICY_INLINE bool UseStreamingInput(uint64_t inputBytes, uint64_t l2Bytes)
{
    // Unknown capacity keeps the default cache policy. Only input larger than
    // the platform-reported L2 bypasses allocation; official flushing is unchanged.
    return l2Bytes != 0ULL && inputBytes > l2Bytes;
}

SQUARE_SUM_POLICY_INLINE uint32_t Align(uint32_t value, uint32_t unit)
{
    return ((value + unit - 1U) / unit) * unit;
}

SQUARE_SUM_POLICY_INLINE bool ContiguousInner(const uint32_t* dims, const uint32_t* mask)
{
    uint32_t first = 5U;
    uint32_t last = 0U;
    for (uint32_t i = 0U; i < 5U; ++i) {
        if (dims[i] == 0U) return false;
        if (mask[i] != 0U) {
            if (first == 5U) first = i;
            last = i;
        }
    }
    if (first == 5U || last == 4U) return false;
    for (uint32_t i = first; i <= last; ++i) {
        if (mask[i] == 0U && dims[i] != 1U) return false;
    }
    return true;
}

SQUARE_SUM_POLICY_INLINE uint32_t BalancedRows(uint32_t innerCols)
{
    // Full trailing rows fit the existing 256-column inner tile cap. Two BF16
    // inputs + one FP32 square tile + BF16 output + FP32 accumulator.
    if (innerCols == 0U || innerCols > 256U) return 0U;
    const uint32_t pitch = Align(innerCols, 16U);
    const uint32_t rows = (kBalancedUbBytes - 6U * pitch) / (8U * pitch);
    return rows < kMaxRepeat ? rows : kMaxRepeat;
}

SQUARE_SUM_POLICY_INLINE uint32_t AtomicMainCols(uint32_t innerCols)
{
    return (innerCols / kVectorElems) * kVectorElems;
}

SQUARE_SUM_POLICY_INLINE uint32_t AtomicRows(uint32_t innerCols)
{
    // Main repeat stride is uint8_t in units of 32B. The unaligned tail uses
    // its own padded slot; zero tail allocates/copies no second queue.
    const uint32_t mainCols = AtomicMainCols(innerCols);
    if (mainCols == 0U || mainCols / 8U > kMaxRepeat) return 0U;
    const uint32_t pitch = Align(innerCols, 8U);
    const uint32_t rows = (kUbBytes - 4U * pitch) / (8U * pitch);
    return rows < kMaxRepeat ? rows : kMaxRepeat;
}

// Fixed FP32 tight-DMA tiles; existing atomic eligibility/core count unchanged.
// Every allocation, including the final output, is counted before narrowing.
constexpr uint32_t kTightAtomicBatchRows = 40U;
constexpr uint32_t kTightAtomicGatherCols = 64U;

SQUARE_SUM_POLICY_INLINE uint32_t TightAtomicGroupRows(uint32_t innerCols)
{
    if (innerCols == 0U) return 0U;
    uint32_t divisor = 8U;
    while (innerCols % divisor != 0U) divisor /= 2U;
    return 8U / divisor;
}

SQUARE_SUM_POLICY_INLINE uint64_t TightAtomicBytes(uint32_t innerCols)
{
    const uint64_t cols = innerCols;
    const uint64_t groups = TightAtomicGroupRows(innerCols);
    const uint64_t pitch = ((cols + 7ULL) / 8ULL) * 8ULL;
    // Two tight input slots, independent acc including a 32B zero sentinel,
    // [G,64] Gather scratch + uint32 byte indices, and one padded output row.
    return 2ULL * kTightAtomicBatchRows * cols * 4ULL +
           (groups * cols + 8ULL) * 4ULL +
           2ULL * groups * kTightAtomicGatherCols * 4ULL + pitch * 4ULL;
}

SQUARE_SUM_POLICY_INLINE bool TightAtomicFits(uint32_t innerCols)
{
    // This full 64-bit bound also bounds all DMA/count arguments and signed
    // index-generation byte offsets below 192KiB, before their uint32 casts.
    return innerCols != 0U && TightAtomicBytes(innerCols) <= kUbBytes;
}

// BF16 contiguous-inner uses one frozen 16-row-multiple batch, not a sweep.
constexpr uint32_t kTightBf16BatchRows = 80U;

SQUARE_SUM_POLICY_INLINE uint32_t TightBf16SliceRows(uint32_t innerCols,
                                                    uint32_t outputCount)
{
    // Called within the existing balanced/tight gates. Zero keeps full-row DMA.
    if (outputCount == 0U || innerCols == 0U || innerCols > 256U ||
        outputCount > innerCols) return 0U;
    const uint32_t pitch = Align(outputCount, 16U);
    if (2U * pitch > innerCols) return 0U;
    const uint32_t rows = kTightBf16BatchRows * innerCols / pitch;
    return rows < kMaxRepeat ? rows : kMaxRepeat;
}

SQUARE_SUM_POLICY_INLINE uint64_t TightBf16Bytes(uint32_t innerCols)
{
    const uint64_t cols = innerCols;
    const uint64_t groups = TightAtomicGroupRows(innerCols);
    const uint64_t pitch = ((cols + 15ULL) / 16ULL) * 16ULL;
    // B is divisible by both 16 and G, so this capacity covers max(DMA pad,
    // grouped tail pad), even when a short tail has N < Align(v,16).
    const uint64_t batchElems = ((kTightBf16BatchRows * cols + 15ULL) / 16ULL) * 16ULL;
    // Double BF16 input, FP32 calc, independent acc + sentinel, FP32 Gather,
    // uint32 byte indices, and ONE BF16 VECOUT slot (MTE3-protected reuse).
    return 2ULL * batchElems * 2ULL + batchElems * 4ULL +
           (groups * cols + 8ULL) * 4ULL + 2ULL * groups * pitch * 4ULL + pitch * 2ULL;
}

SQUARE_SUM_POLICY_INLINE bool TightBf16Fits(uint32_t innerCols)
{
    return innerCols != 0U && TightBf16Bytes(innerCols) <= kUbBytes;
}

// Workspace inner retains the tight full-row layout, but its one VECOUT slot
// holds FP32 partials (and later BF16 output), rather than only BF16 output.
SQUARE_SUM_POLICY_INLINE uint64_t WorkspaceBf16Bytes(uint32_t innerCols)
{
    const uint64_t pitch = ((static_cast<uint64_t>(innerCols) + 15ULL) / 16ULL) * 16ULL;
    return TightBf16Bytes(innerCols) + 2ULL * pitch;
}

// CalcRange gives each core at most ceil(O*R/C) consecutive input rows.
// A range can straddle one extra outer boundary; cap by the total outer count.
// Nonzero R/C and the full 64-bit size gate are enforced before mode 7 dispatch.
SQUARE_SUM_POLICY_INLINE uint64_t WorkspaceBf16Slots(uint32_t outerSize,
                                                     uint32_t reduceSize,
                                                     uint32_t coreCount)
{
    const uint64_t rows = static_cast<uint64_t>(outerSize) * reduceSize;
    const uint64_t maxRows = rows / coreCount + (rows % coreCount != 0ULL ? 1ULL : 0ULL);
    const uint64_t slots = maxRows / reduceSize +
        (maxRows % reduceSize != 0ULL ? 1ULL : 0ULL) + 1ULL;
    return slots < outerSize ? slots : outerSize;
}

constexpr uint32_t kDirectPlaneMaxRows = 512U;
constexpr uint32_t kDirectPlaneReserveBytes = 512U;
constexpr uint32_t kDirectPlaneDenseChunkElems = kMaxRepeat * kVectorElems;

SQUARE_SUM_POLICY_INLINE uint64_t DirectPlaneBytes(uint32_t rows, uint32_t sourceCols)
{
    const uint64_t count = rows;
    const uint64_t dense = count * static_cast<uint64_t>(sourceCols);
    // Two FP16 inputs, dense FP32 accumulator + 64-float window guard,
    // three aligned FP32 row partials, and one FP16 output slot.
    return 4ULL * ((dense + 15ULL) / 16ULL * 16ULL) +
           4ULL * ((dense + 7ULL) / 8ULL * 8ULL + 64ULL) +
           12ULL * ((count + 7ULL) / 8ULL * 8ULL) +
           2ULL * ((count + 15ULL) / 16ULL * 16ULL);
}

SQUARE_SUM_POLICY_INLINE uint32_t DirectPlaneRows(uint32_t sourceCols)
{
    if (sourceCols <= kVectorElems || sourceCols > 2U * kVectorElems) return 0U;
    for (uint32_t rows = kDirectPlaneMaxRows; rows != 0U; rows -= 16U) {
        if (DirectPlaneBytes(rows, sourceCols) + kDirectPlaneReserveBytes <= kUbBytes) {
            return rows;
        }
    }
    return 0U;
}

SQUARE_SUM_POLICY_INLINE uint32_t DirectPlaneGroupRows(uint32_t sourceCols)
{
    uint32_t divisor = 8U;
    while (sourceCols % divisor != 0U) divisor /= 2U;
    return 8U / divisor;
}

// Intersection of a 64-lane aligned window with the packed row [m,m+S).
SQUARE_SUM_POLICY_INLINE uint64_t DirectPlaneWindowMask(uint32_t sourceCols,
                                                       uint32_t misalign,
                                                       uint32_t window)
{
    const uint32_t base = window * kVectorElems;
    const uint32_t end = misalign + sourceCols;
    if (end <= base || misalign >= base + kVectorElems) return 0ULL;
    const uint32_t lo = misalign > base ? misalign - base : 0U;
    const uint32_t hi = end < base + kVectorElems ? end - base : kVectorElems;
    const uint64_t upper = hi == 64U ? ~0ULL : (1ULL << hi) - 1ULL;
    return upper & (~0ULL << lo);
}

// BF16 full-row input with a fixed FP32 microbatch of complete rows.
constexpr uint32_t kFullRowSuffixMicroRows = 16U;
constexpr uint32_t kFullRowSuffixMaxRows = 32U;
constexpr uint32_t kFullRowSuffixMinRows = 16U;
constexpr uint32_t kFullRowSuffixReserveBytes = 512U;

SQUARE_SUM_POLICY_INLINE uint64_t FullRowSuffixPitch(uint32_t reduceSize)
{
    return ((static_cast<uint64_t>(reduceSize) + 15ULL) / 16ULL) * 16ULL;
}

SQUARE_SUM_POLICY_INLINE uint64_t FullRowSuffixBytes(uint32_t reduceSize, uint32_t rows)
{
    const uint64_t pitch = FullRowSuffixPitch(reduceSize);
    const uint64_t outputPitch = ((static_cast<uint64_t>(rows) + 15ULL) / 16ULL) * 16ULL;
    // Two BF16 inputs, one FP32 microbatch, padded FP32 sums and two BF16 outputs.
    return 4ULL * pitch * (static_cast<uint64_t>(rows) + kFullRowSuffixMicroRows) +
           8ULL * outputPitch;
}

SQUARE_SUM_POLICY_INLINE uint32_t FullRowSuffixRows(uint32_t reduceSize)
{
    const uint64_t pitch = FullRowSuffixPitch(reduceSize);
    if (reduceSize <= kVectorElems || pitch / 8ULL > kMaxRepeat) return 0U;
    for (uint32_t rows = kFullRowSuffixMaxRows; rows >= kFullRowSuffixMinRows; --rows) {
        if (FullRowSuffixBytes(reduceSize, rows) + kFullRowSuffixReserveBytes <= kUbBytes) {
            return rows;
        }
    }
    return 0U;
}

// Linear BF16 input, dense FP32 square, then block-sum alignment in UB.
SQUARE_SUM_POLICY_INLINE uint32_t PackedSuffixBlockPitch(uint32_t reduceSize)
{
    return Align((reduceSize + 7U) / 8U, 8U);
}

SQUARE_SUM_POLICY_INLINE uint64_t PackedSuffixBytes(uint32_t reduceSize, uint32_t rows)
{
    const uint64_t elems = static_cast<uint64_t>(rows) * reduceSize;
    const uint64_t input = ((elems + 15ULL) / 16ULL) * 16ULL;
    const uint64_t dense = ((elems + 63ULL) / 64ULL) * 64ULL;
    const uint64_t pitch = PackedSuffixBlockPitch(reduceSize);
    const uint64_t output = ((static_cast<uint64_t>(rows) + 15ULL) / 16ULL) * 16ULL;
    // Double BF16 input, FP32 square, block sums/sentinel, gather/index,
    // two FP32 sum vectors and double BF16 output.
    return 4ULL * input + 4ULL * (dense + 64ULL) + 4ULL * (dense / 8ULL + 8ULL) +
           8ULL * rows * (pitch + 16ULL) + 256ULL + 12ULL * output;
}

SQUARE_SUM_POLICY_INLINE uint32_t PackedSuffixRows(uint32_t reduceSize)
{
    if (reduceSize % 16U == 0U || FullRowSuffixRows(reduceSize) == 0U) return 0U;
    // Whole batches retain the 32B GM alignment of the output owner (16 BF16
    // rows). Only the true final batch needs exact-byte padding.
    for (uint32_t rows = kFullRowSuffixMaxRows; rows >= kFullRowSuffixMinRows; rows -= 16U) {
        if (PackedSuffixBytes(reduceSize, rows) + kFullRowSuffixReserveBytes <= kUbBytes) {
            return rows;
        }
    }
    return 0U;
}

SQUARE_SUM_POLICY_INLINE uint32_t FusedSuffixRows()
{
    uint32_t rows = kMaxRepeat;
    while (8U * rows * kFusedSuffixCols + 2U * kFusedSuffixInputPadBytes +
           12U * Align(rows, 16U) > kUbBytes) --rows;
    return rows;
}
}  // namespace square_sum_policy
