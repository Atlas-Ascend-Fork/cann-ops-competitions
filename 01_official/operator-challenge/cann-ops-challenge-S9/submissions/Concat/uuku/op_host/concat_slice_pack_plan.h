#pragma once
#include "concat_alternating_pack_plan.h"

namespace concat_slice_pack {
constexpr uint32_t kFlag = concat_alternating_pack::kFlag;
constexpr uint32_t kHeader = concat_alternating_pack::kHeader;
using Plan = concat_alternating_pack::Plan;
inline uint32_t Boundary(uint32_t total, uint32_t core, uint32_t cores) {
    const uint64_t blocks = (static_cast<uint64_t>(total) + 7U) / 8U;
    return static_cast<uint32_t>(std::min(static_cast<uint64_t>(total), blocks * core / cores * 8U));
}
inline bool Build(uint32_t elemBytes, uint32_t count, uint32_t outer,
                  uint32_t inner, uint32_t axis, uint32_t cores,
                  const uint32_t* widths, Plan& result) {
    concat_block_pack::Plan baseline;
    if (!concat_block_pack::Build(elemBytes, count, outer, inner, axis, cores,
                                   widths, baseline, true)) return false;
    if (cores != 40U || outer < 8U * cores || static_cast<uint64_t>(outer) * axis > UINT32_MAX)
        return false;
    const uint32_t total = outer * axis;
    uint32_t maxRows = 0U;
    for (uint32_t c = 0U; c < cores; ++c) {
        const uint32_t lo = Boundary(total, c, cores), hi = Boundary(total, c + 1U, cores);
        if (lo == hi) return false;
        maxRows = std::max(maxRows, static_cast<uint32_t>((static_cast<uint64_t>(hi) + axis - 1U) / axis) - lo / axis);
    }
    if (!maxRows || maxRows > 255U) return false;
    return concat_alternating_pack::BuildGroups(
        count, widths, maxRows, 160U, 240U, 13U, result);
}
inline void Serialize(const Plan& p, uint32_t* schedule) {
    concat_alternating_pack::Serialize(p, schedule, 1U);
}
} // namespace concat_slice_pack
