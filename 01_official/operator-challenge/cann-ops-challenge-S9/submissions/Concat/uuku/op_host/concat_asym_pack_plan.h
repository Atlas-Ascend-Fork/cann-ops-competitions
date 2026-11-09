#pragma once
#include "concat_alternating_pack_plan.h"

namespace concat_asym_pack {
constexpr uint32_t kFlag = concat_alternating_pack::kFlag;
constexpr uint32_t kHeader = concat_alternating_pack::kHeader;
using Plan = concat_alternating_pack::Plan;
inline bool Build(uint32_t elemBytes, uint32_t count, uint32_t outer,
                  uint32_t inner, uint32_t axis, uint32_t cores,
                  const uint32_t* widths, Plan& result) {
    concat_block_pack::Plan baseline;
    if (!concat_block_pack::Build(elemBytes, count, outer, inner, axis, cores,
                                   widths, baseline, true)) return false;
    return concat_alternating_pack::BuildGroups(
        count, widths, baseline.maxRows, 160U, 224U, 12U, result);
}
inline void Serialize(const Plan& p, uint32_t* schedule) {
    concat_alternating_pack::Serialize(p, schedule, 0U);
}
} // namespace concat_asym_pack
