// Shared alternating-bank planner. source/output/expanded hold B32 word counts;
// bytes is a byte budget. ends[] stores exclusive original-input indices.
// The DP minimizes group count for each compact-input position and next bank.
#pragma once
#include "concat_block_pack_plan.h"

namespace concat_alternating_pack {
constexpr uint32_t kFlag = 0x04000000U;
constexpr uint32_t kHeader = 8U;
struct Plan {
    uint32_t source[2]{}, output[2]{};
    uint32_t expanded = 0, groups = 0, firstSlot = 0, maxRows = 0, bytes = 0;
    uint32_t ends[256]{};
};
// Shared host-only DP; callers retain geometry, capacity and UB policies.
inline bool BuildGroups(uint32_t count, const uint32_t* widths, uint32_t maxRows,
                        uint32_t firstCapacity, uint32_t secondCapacity,
                        uint32_t expandedBytes, Plan& result)
{
    // Compact empty inputs while retaining indices for the serialized boundaries.
    uint32_t compact[256]{}, original[256]{}, n = 0;
    for (uint32_t i = 0; i < count; ++i)
        if (widths[i]) {
            compact[n] = widths[i];
            original[n++] = i;
        }

    // cost[i][slot]: minimum remaining groups starting at input i in this bank.
    // next[i][slot]: exclusive compact-input endpoint of the selected first group.
    const uint32_t capacity[2] = {firstCapacity, secondCapacity};
    constexpr uint32_t inf = 100000U;
    uint32_t cost[257][2]{}, next[256][2]{};
    for (uint32_t i = n; i-- > 0;)
        for (uint32_t slot = 0; slot < 2; ++slot) {
            cost[i][slot] = inf;
            uint32_t width = 0;
            for (uint32_t j = i; j < n; ++j) {
                width += compact[j];
                if (width > capacity[slot])
                    break;
                if (1U + cost[j + 1U][1U - slot] < cost[i][slot]) {
                    cost[i][slot] = 1U + cost[j + 1U][1U - slot];
                    next[i][slot] = j + 1U;
                }
            }
        }

    // Replay the chosen alternating groups and record each bank's peak capacity.
    Plan p;
    p.maxRows = maxRows;
    p.firstSlot = cost[0][1] < cost[0][0] ? 1U : 0U;
    if (cost[0][p.firstSlot] >= inf)
        return false;
    uint32_t slot = p.firstSlot;
    for (uint32_t i = 0; i < n; slot = 1U - slot) {
        const uint32_t end = next[i][slot];
        uint32_t width = 0, source = 8;
        for (uint32_t j = i; j < end; ++j) {
            width += compact[j];
            source += concat_block_pack::Align8(p.maxRows * compact[j]);
        }
        const uint32_t pitch = concat_block_pack::Align8(width);
        p.source[slot] = std::max(p.source[slot], source);
        p.output[slot] = std::max(p.output[slot], p.maxRows * pitch);
        p.expanded = std::max(p.expanded, 4U * pitch);
        p.ends[p.groups++] = end == n ? count : original[end];
        i = end;
    }
    // Convert word capacities to the unchanged byte budget only at this boundary.
    const uint32_t metaWords = 4U * concat_block_pack::Align8(count) + count + kHeader;
    const uint64_t bytes = 4ULL * (p.source[0] + p.source[1] + p.output[0] + p.output[1]) +
        static_cast<uint64_t>(expandedBytes) * p.expanded + 4ULL * concat_block_pack::Align8(metaWords) + 8192U;
    if (!p.source[0] || !p.source[1] || bytes > 192U * 1024U)
        return false;
    p.bytes = static_cast<uint32_t>(bytes);
    result = p;
    return true;
}
inline void Serialize(const Plan& p, uint32_t* schedule, uint32_t mode)
{
    // Header words 0..7 must retain the kernel's existing route-5 ABI order.
    schedule[0] = p.source[0];
    schedule[1] = p.source[1];
    schedule[2] = p.output[0];
    schedule[3] = p.output[1];
    schedule[4] = p.expanded;
    schedule[5] = p.firstSlot;
    schedule[6] = p.groups;
    schedule[7] = mode;
    for (uint32_t g = 0; g < p.groups; ++g)
        schedule[kHeader + g] = p.ends[g];
}
} // namespace concat_alternating_pack
