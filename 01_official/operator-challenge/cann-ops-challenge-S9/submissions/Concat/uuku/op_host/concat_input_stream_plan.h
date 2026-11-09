#pragma once
#include "concat_slice_pack_plan.h"

namespace concat_input_stream {
constexpr uint32_t kMode = 3U;
constexpr uint32_t kBand = 1328U; // A wider single output for shorter row batches.
constexpr uint32_t kMaxBatchRows = 26U;
constexpr uint32_t kBaselineBand = 488U; // Preserve v194's admitted shape class.
constexpr uint32_t kGuardWords = 72U; // 8-word prefix plus 64-word suffix.
struct Plan {
    uint32_t source = 0U, output = 0U, indices = 0U;
    uint32_t maxRows = 0U, nonzero = 0U, bytes = 0U;
    uint32_t inputs[256]{};
};
inline void TryGroupAdjacent(const uint32_t* widths, Plan& p) {
    Plan grouped = p;
    uint32_t position = 0U, source = 0U, mixed = 0U;
    for (uint32_t n = 0U; n < p.nonzero;) {
        const uint32_t limit = std::min(128U - position % 8U,
                                        kBand - position % kBand);
        uint32_t members = 1U, width = widths[p.inputs[n]];
        while (members < 4U && n + members < p.nonzero &&
               width + widths[p.inputs[n + members]] <= limit) {
            width += widths[p.inputs[n + members++]];
        }
        // Every head carries a count; continuation words retain the input index.
        // Input indices are <=255, so the upper 16 bits are available in mode3.
        grouped.inputs[n] |= members << 16U;
        uint32_t words = kGuardWords;
        for (uint32_t m = 0U; m < members; ++m)
            words += concat_block_pack::Align8(p.maxRows * widths[p.inputs[n + m]]);
        source = std::max(source, words);
        mixed += members > 1U ? 1U : 0U;
        position += width;
        n += members;
    }
    // The two piecewise column vectors are private UB scratch. Recompute the
    // actual padded source capacity; narrow-input shapes may require much more
    // than the 512B scratch increment and must retain the original route.
    const uint64_t bytes = static_cast<uint64_t>(p.bytes) +
        8ULL * (source - p.source) + 512U;
    if (!mixed || bytes > 192U * 1024U) return;
    grouped.source = source;
    grouped.bytes = static_cast<uint32_t>(bytes);
    p = grouped;
}
inline void EncodeInputWidths(const uint32_t* widths, Plan& p) {
    // Every admitted mode3 word carries index[7:0] and width[15:8].
    // Heads also carry total width[31:24]. Preserve the original members
    // byte[23:16]: zero on ungrouped plans means no mixed-index scratch.
    // Bounds are guaranteed by Build/TryGroupAdjacent, not by a test shape.
    for (uint32_t n = 0U; n < p.nonzero;) {
        const uint32_t tag = (p.inputs[n] >> 16U) & 0xffU;
        const uint32_t members = tag ? tag : 1U;
        uint32_t groupWidth = 0U;
        for (uint32_t m = 0U; m < members; ++m) {
            const uint32_t input = p.inputs[n + m] & 0xffU;
            p.inputs[n + m] = input | (widths[input] << 8U);
            groupWidth += widths[input];
        }
        p.inputs[n] |= (tag << 16U) | (groupWidth << 24U);
        n += members;
    }
}
inline bool Build(uint32_t elemBytes, uint32_t count, uint32_t outer,
                  uint32_t inner, uint32_t axis, uint32_t cores,
                  const uint32_t* widths, Plan& result) {
    if (elemBytes != 4U || inner != 1U || count < 65U || count > 256U ||
        cores != 40U || outer < 8U * cores || axis < 512U ||
        axis > UINT32_MAX / 4U || static_cast<uint64_t>(outer) * axis > UINT32_MAX)
        return false;
    Plan p;
    uint32_t maxWidth = 0U;
    uint64_t sum = 0U;
    for (uint32_t i = 0; i < count; ++i) {
        if (widths[i] > 128U) return false;
        sum += widths[i];
        maxWidth = std::max(maxWidth, widths[i]);
        if (widths[i]) p.inputs[p.nonzero++] = i;
    }
    if (sum != axis || p.nonzero < 2U) return false;
    const uint32_t total = outer * axis;
    for (uint32_t c = 0; c < cores; ++c) {
        const uint32_t lo = concat_slice_pack::Boundary(total, c, cores);
        const uint32_t hi = concat_slice_pack::Boundary(total, c + 1U, cores);
        if (lo == hi) return false;
        p.maxRows = std::max(p.maxRows, (hi - 1U) / axis - lo / axis + 1U);
    }
    if (p.maxRows == 0U || p.maxRows > 255U) return false;
    const uint32_t metaWords = 4U * concat_block_pack::Align8(count) + count + 8U;
    const uint32_t fullSource = concat_block_pack::Align8(p.maxRows * maxWidth) + kGuardWords;
    const uint64_t baselineBytes = 8ULL * (fullSource + p.maxRows * 64U) +
        4ULL * (p.maxRows * kBaselineBand + 64U) + 64U * 4U +
        kBaselineBand * 4U + 4ULL * concat_block_pack::Align8(metaWords) + 8192U;
    if (baselineBytes > 192U * 1024U) return false;
    p.maxRows = std::min(p.maxRows, kMaxBatchRows);
    // The selected row capacity is a UB tiling parameter, not a shape match.
    for (; p.maxRows != 0U; --p.maxRows) {
        p.source = concat_block_pack::Align8(p.maxRows * maxWidth) + kGuardWords;
        p.output = p.maxRows * kBand + 64U;
        // Edge compaction borrows indices and can be wider than rows*64.
        // Its final index repeat also stays within this own allocation.
        p.indices = std::max(p.maxRows * 64U, (kBand + 63U) / 64U * 64U);
        const uint64_t bytes = 8ULL * (p.source + p.indices) + 4ULL * p.output +
            64U * 4U + (kBand + 64U) * 4U +
            4ULL * concat_block_pack::Align8(metaWords) + 8192U;
        if (bytes > 192U * 1024U) continue;
        p.bytes = static_cast<uint32_t>(bytes);
        TryGroupAdjacent(widths, p);
        EncodeInputWidths(widths, p);
        result = p;
        return true;
    }
    return false;
}
inline void Serialize(const Plan& p, uint32_t* schedule) {
    schedule[0] = p.source; schedule[1] = p.source;
    schedule[2] = p.output; schedule[3] = p.maxRows; // Row capacity, not an output allocation.
    schedule[4] = p.indices; schedule[5] = kBand;
    schedule[6] = p.nonzero; schedule[7] = kMode;
    for (uint32_t i = 0; i < p.nonzero; ++i) schedule[8U + i] = p.inputs[i];
}
} // namespace concat_input_stream
