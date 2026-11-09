#pragma once
#include <algorithm>
#include <cstdint>

// Shared with the standalone CPU verifier. No captured shapes or values.
namespace concat_block_pack {
constexpr uint32_t kGroupWidth = 192;
constexpr uint32_t kMicroRows = 8;
constexpr uint32_t kFlag = 0x10000000U;
struct Plan {
    uint32_t sourceElems = 0;
    uint32_t outputElems = 0;
    uint32_t expandedElems = 0;
    uint32_t groups = 0;
    uint32_t maxRows = 0;
    uint32_t bytes = 0;
};
inline uint32_t Align8(uint32_t n) { return (n + 7U) / 8U * 8U; }
inline uint32_t RowBegin(uint32_t outer, uint32_t core, uint32_t cores) {
    return core == cores ? outer : static_cast<uint64_t>(outer) * core / cores / 8U * 8U;
}
inline bool Build(uint32_t elemBytes, uint32_t count, uint32_t outer,
                  uint32_t inner, uint32_t axis, uint32_t cores,
                  const uint32_t* widths, Plan& result, bool elementGather = false) {
    if (elemBytes != 4 || inner != 1 || count < 65 || count > 256 ||
        outer < 64 || axis < 512 || axis > UINT32_MAX / 4U || cores == 0 || cores > 40)
        return false;
    Plan p;
    for (uint32_t c = 0; c < cores; ++c)
        p.maxRows = std::max(p.maxRows, RowBegin(outer, c + 1, cores) - RowBegin(outer, c, cores));
    if (p.maxRows == 0 || p.maxRows > 255) return false;
    uint32_t nonzero = 0;
    uint64_t sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (widths[i] > 128) return false;
        nonzero += widths[i] != 0;
        sum += widths[i];
    }
    if (sum != axis || nonzero < 2) return false;
    uint32_t cursor = 0;
    while (cursor < count) {
        uint32_t width = 0, slab = 8, blocks = 0;
        while (cursor < count) {
            const uint32_t w = widths[cursor];
            if (width + w > kGroupWidth) break;
            slab += Align8(p.maxRows * w);
            for (uint32_t r = 0; r < kMicroRows; ++r)
                blocks += ((r + 1) * w + 7) / 8 - r * w / 8;
            width += w;
            ++cursor;
        }
        if (width == 0) continue;
        const uint32_t pitch = Align8(width);
        if (pitch != width) blocks += kMicroRows;
        ++p.groups;
        p.sourceElems = std::max(p.sourceElems, slab);
        p.outputElems = std::max(p.outputElems, p.maxRows * pitch);
        p.expandedElems = std::max(p.expandedElems,
            elementGather ? 4U * pitch : Align8(blocks) * 8);
    }
    if (p.expandedElems / 64 > 255) return false;
    // Double source/output slots; expanded data + index/delta/mask; meta cache.
    const uint64_t bytes = 8ULL * (p.sourceElems + p.outputElems) +
        (elementGather ? 12ULL * p.expandedElems :
            4ULL * p.expandedElems + 12ULL * (p.expandedElems / 8)) +
        16ULL * Align8(count) + 8192;
    if (bytes > 192 * 1024) return false;
    p.bytes = static_cast<uint32_t>(bytes);
    result = p;
    return true;
}
} // namespace concat_block_pack
