#ifndef GREATER_CLOCK_PROBE_H
#define GREATER_CLOCK_PROBE_H

// Host-side diagnostic encoding of dtype, broadcast and element-count metadata.
#ifndef GT_CLOCK_LOCAL_OVERRIDE
#define GT_CLOCK_LOCAL_OVERRIDE 0
#endif
#ifndef GT_CLOCK_FIELD
#define GT_CLOCK_FIELD 0
#endif
#ifndef GT_CLOCK_SHIFT
#define GT_CLOCK_SHIFT 0
#endif
#ifndef GT_CLOCK_AXIS
#define GT_CLOCK_AXIS 0
#endif
#ifndef GT_CLOCK_WIDTH
#define GT_CLOCK_WIDTH 9
#endif
#if GT_CLOCK_LOCAL_OVERRIDE
#include <cstdlib>
#endif

namespace greater_clock {
static_assert(GT_CLOCK_SHIFT < 64 && GT_CLOCK_AXIS < greater_core::GT_MAX_DIM &&
              GT_CLOCK_WIDTH > 0 && GT_CLOCK_WIDTH <= 13,
              "invalid clock field slice");
inline uint32_t FloorLog2(uint64_t n)
{
    uint32_t e = 0;
    while (n > 1) { n >>= 1; ++e; }
    return e;
}

inline uint64_t Code(const greater_core::Plan &p, uint32_t family, uint32_t dtypeIdx)
{
#if GT_CLOCK_LOCAL_OVERRIDE
    // Optional environment override enabled by GT_CLOCK_LOCAL_OVERRIDE.
    if (const char *s = std::getenv("GT_CLOCK_CODE")) {
        char *end = nullptr;
        const auto code = std::strtoull(s, &end, 10);
        if (end != s && *end == '\0' && code <= 639) return code;
    }
#endif
#if GT_CLOCK_FIELD == 12 || GT_CLOCK_FIELD == 13
    if (family != greater_core::FAM_BCAST) {
        return p.tilingKey * 64 + p.usedCores + (GT_CLOCK_FIELD == 13 ? 1 : 0);
    }
    if (p.dimNum < 2 || p.rowNum == 0) return 8191;
    const uint64_t rows = p.rowNum;
    const uint64_t repeat = p.shape[p.dimNum-2];
    uint64_t rank = 0;
    // Index of the repeat extent in the sorted positive divisors of rows.
    // Integer-only and O(sqrt(rows)); the pair (rows, rank) is lossless.
    for (uint64_t d = 1; d <= rows/d; ++d) {
        if (rows % d != 0) continue;
        if (d < repeat) ++rank;
        const uint64_t q = rows/d;
        if (q != d && q < repeat) ++rank;
    }
    if (rank >= 256) return 8191;
    const uint64_t word = (rows << 8) | rank;
#if GT_CLOCK_FIELD == 12
    return word & 8191ULL;
#else
    return 1 + ((word >> 13) & 8191ULL);
#endif
#elif GT_CLOCK_FIELD == 11
    uint64_t value = p.totalNum;
    uint32_t shift = 0;
    if (family == greater_core::FAM_BCAST) {
        value = p.innerLen;
        const uint32_t e = FloorLog2(value);
        shift = e >= 13 ? e - 13 : 0;
    }
    // Encode bounded metadata fields in the diagnostic frame.
    return 1 + ((value >> shift) & 8191ULL);
#elif GT_CLOCK_FIELD == 10
    if (family != greater_core::FAM_BCAST) {
        const uint32_t e = FloorLog2(p.totalNum);
        return (p.totalNum >> (e > 12 ? e - 12 : 0)) & 8191ULL;
    }
    // Ternary axis states: 0 both dense, 1 x broadcasts, 2 y broadcasts.
    // The rank prefix makes all ranks 1..4 disjoint. 8191 is an explicit
    // diagnostic overflow symbol, not a different tiling route.
    if (p.dimNum == 0 || p.dimNum > 4 || p.totalNum == 0) return 8191;
    uint64_t power = 1, pattern = 0;
    for (uint32_t i = 0; i < p.dimNum; ++i) {
        const uint32_t axis = p.strides[2*i] == 0 ? 1 : (p.strides[2*i+1] == 0 ? 2 : 0);
        pattern += axis * power;
        power *= 3;
    }
    return 64 * ((power - 3) / 2 + pattern) + FloorLog2(p.innerLen);
#elif GT_CLOCK_FIELD != 0
    // Use bounded nine-bit slices for the diagnostic frame.
    uint64_t value = 0;
    switch (GT_CLOCK_FIELD) {
        case 1: value = p.totalNum; break;
        case 2: value = p.innerLen; break;
        case 3: value = p.rowNum; break;
        case 4: value = p.dimNum * 3ULL + p.innerMode; break;
        case 5: value = p.tilingKey; break;
        case 6: value = p.shape[GT_CLOCK_AXIS]; break;
        case 7: value = p.strides[2 * GT_CLOCK_AXIS]; break;
        case 8: value = p.strides[2 * GT_CLOCK_AXIS + 1]; break;
        case 9: value = p.usedCores; break;
    }
    return (value >> GT_CLOCK_SHIFT) & ((1ULL << GT_CLOCK_WIDTH) - 1);
#else
    return 128ULL * dtypeIdx + (family == greater_core::FAM_BCAST ? 64ULL : 0ULL)
        + FloorLog2(p.totalNum);
#endif
}
}
#endif
