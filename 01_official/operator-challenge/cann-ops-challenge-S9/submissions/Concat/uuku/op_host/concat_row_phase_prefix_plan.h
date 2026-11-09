#pragma once
#include <algorithm>
#include <cstdint>

namespace concat_row_phase_prefix {
struct Plan {
    uint32_t slotWords = 0U, indexWords = 0U, bytes = 0U;
};

inline bool Build(uint32_t elemBytes, uint32_t count, uint32_t outer,
                  uint32_t inner, uint32_t axis, uint32_t cores,
                  const uint32_t* widths, Plan& result)
{
    if (elemBytes != 4U || count < 2U || count > 16U || cores != 40U ||
        outer < 64U || inner <= 1U || axis == 0U || axis > UINT32_MAX / inner)
        return false;
    const uint32_t rowWidth = axis * inner;
    // The longest phase pitch is eight rows; GM stride fields are byte counts.
    if (rowWidth > UINT32_MAX / 32U || outer > UINT32_MAX / rowWidth ||
        (static_cast<uint64_t>(outer) + cores - 1U) / cores > 4095U)
        return false;
    uint32_t minWidth = UINT32_MAX, nonzero = 0U, unaligned = 0U;
    uint64_t sum = 0U;
    for (uint32_t i = 0U; i < count; ++i) {
        sum += widths[i];
        const uint64_t width = static_cast<uint64_t>(widths[i]) * inner;
        if (width == 0U) continue;
        if (width < 512U || width > 8192U) return false;
        ++nonzero;
        unaligned += width % 8U != 0U;
        minWidth = std::min(minWidth, static_cast<uint32_t>(width));
    }
    if (sum != axis || nonzero < 2U || unaligned == 0U) return false;

    const uint64_t total = static_cast<uint64_t>(outer) * rowWidth;
    const uint64_t blocks = (total + 7U) / 8U;
    uint32_t maxRows = 0U;
    for (uint32_t core = 0U; core < cores; ++core) {
        const uint64_t lo = std::min(total, blocks * core / cores * 8U);
        const uint64_t hi = std::min(total, blocks * (core + 1U) / cores * 8U);
        if (lo < hi)
            maxRows = std::max(maxRows, static_cast<uint32_t>((hi - 1U) / rowWidth - lo / rowWidth + 1U));
    }
    Plan p;
    // One 176 KiB raw slot with a 256 B rounded-access guard. Larger slabs
    // batch more alignment phases, at the cost of cross-slab MTE2 prefetch.
    p.slotWords = 176U * 1024U / 4U - 64U;
    maxRows = std::min(maxRows, p.slotWords / minWidth);
    if (maxRows == 0U || maxRows > 255U) return false;
    p.indexWords = (maxRows * 8U + 63U) / 64U * 64U + 64U;
    const uint32_t metaReserve = 4U * ((count + 7U) / 8U * 8U) * 4U;
    // Metadata is currently inactive for count <= 16; still reserve it here.
    p.bytes = (p.slotWords + 64U) * 4U + 2U * p.indexWords * 4U +
              64U * 4U + metaReserve + 8192U;
    if (p.bytes > 192U * 1024U) return false;
    result = p;
    return true;
}
} // namespace concat_row_phase_prefix
