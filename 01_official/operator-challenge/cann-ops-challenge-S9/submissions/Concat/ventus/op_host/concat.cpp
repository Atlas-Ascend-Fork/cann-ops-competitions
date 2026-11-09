/**
 * Concat 算子注册、形状推导与主机侧分块规划。
 * 根据拼接轴、输入片段长度和 UB 容量选择数据搬运路径并划分输出所有权。
 */

#include "concat_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
uint64_t RowsCeilDiv(uint64_t n, uint64_t d)
{
    return n / d + static_cast<uint64_t>(n % d != 0U);
}

// 为每个输入分配独立的行区间，减少设备侧重复扫描输入描述信息。
bool BuildDescriptorFreeGrid(const ConcatTilingData& base, const uint64_t* segments,
                             uint32_t elementBytes, ConcatRowsTilingData& output)
{
    if (base.mode != 0U || base.outer == 0U || base.inputCount < 2U ||
        base.inputCount >= S9_CONCAT_PREFIX_THRESHOLD ||
        base.inputGroups != base.inputCount || base.rowGroups == 0U ||
        base.blockDim > S9_CONCAT_MAX_INPUT_GROUPS ||
        base.blockDim != static_cast<uint64_t>(base.rowGroups) * base.inputCount ||
        base.inner == 0U || base.outputAxis == 0U || elementBytes == 0U ||
        base.outputAxis > UINT64_MAX / base.inner / elementBytes / base.outer) {
        return false;
    }
    const uint64_t bytes = base.outputAxis * base.inner * elementBytes * base.outer;
    const bool gridBypass = elementBytes == 4U && base.inputCount <= 8U && bytes >= 16ULL * 1024ULL * 1024ULL;
    const bool ownerBypass = elementBytes == 4U && bytes >= 2ULL * 65536ULL * base.blockDim;
    if (gridBypass != ownerBypass) { return false; }
    output = {};
    output.base = base;
    uint64_t prefix = 0U;
    for (uint32_t i = 0U; i < base.inputCount; ++i) {
        for (uint32_t rowGroup = 0U; rowGroup < base.rowGroups; ++rowGroup) {
            const uint64_t start = base.outer / base.rowGroups * rowGroup +
                base.outer % base.rowGroups * rowGroup / base.rowGroups;
            const uint64_t end = base.outer / base.rowGroups * (rowGroup + 1U) +
                base.outer % base.rowGroups * (rowGroup + 1U) / base.rowGroups;
            output.owners[rowGroup * base.inputCount + i] = {start, end - start, segments[i], prefix, i, 0U};
        }
        prefix += segments[i];
    }
    return true;
}

bool BuildWeightedRows(const ConcatTilingData& base, const uint64_t* segments,
                       uint32_t elementBytes, ConcatRowsTilingData& output)
{
    const uint32_t cores = base.blockDim;
    if (base.mode != 0U || base.rowGroups <= 1U || base.outer == 0U ||
        cores == 0U || cores > S9_CONCAT_MAX_INPUT_GROUPS || base.inputCount > cores ||
        base.inner == 0U || base.outputAxis > UINT64_MAX / base.inner / elementBytes / base.outer) {
        return false;
    }
    uint64_t bytes[S9_CONCAT_MAX_INPUT_GROUPS] = {};
    uint32_t counts[S9_CONCAT_MAX_INPUT_GROUPS] = {};
    uint32_t nonzero = 0;
    uint64_t largest = 0;
    for (uint32_t i = 0; i < base.inputCount; ++i) {
        if (segments[i] > UINT64_MAX / elementBytes) { return false; }
        bytes[i] = segments[i] * elementBytes;
        largest = std::max(largest, bytes[i]);
        nonzero += static_cast<uint32_t>(bytes[i] != 0U);
    }
    if (nonzero == 0U || nonzero > cores || base.outer > UINT64_MAX / largest) {
        return false;
    }
    const uint32_t used = base.outer <= cores / nonzero ?
        static_cast<uint32_t>(base.outer * nonzero) : cores;
    uint64_t low = largest, high = largest * base.outer;
    while (low < high) {
        const uint64_t middle = low + (high - low) / 2U;
        uint64_t need = 0;
        for (uint32_t i = 0; i < base.inputCount && need <= used; ++i) {
            if (bytes[i] != 0U) { need += RowsCeilDiv(base.outer, middle / bytes[i]); }
        }
        if (need <= used) { high = middle; } else { low = middle + 1U; }
    }
    uint32_t allocated = 0;
    for (uint32_t i = 0; i < base.inputCount; ++i) {
        if (bytes[i] != 0U) {
            counts[i] = static_cast<uint32_t>(RowsCeilDiv(base.outer, low / bytes[i]));
            allocated += counts[i];
        }
    }
    while (allocated < used) {
        uint32_t pick = base.inputCount;
        uint64_t load = 0;
        for (uint32_t i = 0; i < base.inputCount; ++i) {
            if (counts[i] != 0U && counts[i] < base.outer) {
                const uint64_t cost = RowsCeilDiv(base.outer, counts[i]) * bytes[i];
                if (pick == base.inputCount || cost > load) { pick = i; load = cost; }
            }
        }
        if (pick == base.inputCount) { return false; }
        ++counts[pick];
        ++allocated;
    }
    uint64_t oldMax = 0, newMax = 0;
    for (uint32_t g = 0; g < base.inputGroups; ++g) {
        uint64_t groupBytes = 0;
        for (uint32_t i = g; i < base.inputCount; i += base.inputGroups) { groupBytes += bytes[i]; }
        oldMax = std::max(oldMax, groupBytes * RowsCeilDiv(base.outer, base.rowGroups));
    }
    for (uint32_t i = 0; i < base.inputCount; ++i) {
        if (counts[i] != 0U) { newMax = std::max(newMax, bytes[i] * RowsCeilDiv(base.outer, counts[i])); }
    }
    if (newMax >= oldMax) { return false; }

    constexpr uint64_t tileBytes = 64ULL * 1024ULL;
    const uint64_t tileElements = tileBytes / elementBytes;
    const uint64_t outputRowElements = base.outputAxis * base.inner;
    auto copyCalls = [&](uint64_t rows, uint64_t elements,
                         uint64_t byteCount) -> uint64_t {
        if (rows == 0U || elements == 0U) { return 0U; }
        if (elements > tileElements ||
            (outputRowElements - elements) * elementBytes > UINT32_MAX) {
            return rows * RowsCeilDiv(elements, tileElements);
        }
        const uint64_t aligned = (byteCount + 31ULL) & ~31ULL;
        const uint64_t rowsPerCall = std::min<uint64_t>(4095ULL, tileBytes / aligned);
        return RowsCeilDiv(rows, rowsPerCall);
    };
    uint64_t oldCalls = 0U, newCalls = 0U;
    for (uint32_t i = 0; i < base.inputCount; ++i) {
        if (bytes[i] == 0U) { continue; }
        for (uint32_t g = 0; g < base.rowGroups; ++g) {
            const uint64_t start = base.outer / base.rowGroups * g +
                base.outer % base.rowGroups * g / base.rowGroups;
            const uint64_t end = base.outer / base.rowGroups * (g + 1U) +
                base.outer % base.rowGroups * (g + 1U) / base.rowGroups;
            oldCalls += copyCalls(end - start, segments[i], bytes[i]);
        }
        for (uint32_t owner = 0; owner < counts[i]; ++owner) {
            const uint64_t start = base.outer / counts[i] * owner +
                base.outer % counts[i] * owner / counts[i];
            const uint64_t end = base.outer / counts[i] * (owner + 1U) +
                base.outer % counts[i] * (owner + 1U) / counts[i];
            newCalls += copyCalls(end - start, segments[i], bytes[i]);
        }
    }
    if (newCalls > oldCalls) { return false; }
    output = {};
    output.base = base;
    output.base.blockDim = used;
    uint32_t owner = 0;
    uint64_t prefix = 0;
    for (uint32_t i = 0; i < base.inputCount; ++i) {
        for (uint32_t j = 0; j < counts[i]; ++j) {

            const uint64_t start = base.outer / counts[i] * j + base.outer % counts[i] * j / counts[i];
            const uint64_t end = base.outer / counts[i] * (j + 1U) + base.outer % counts[i] * (j + 1U) / counts[i];
            output.owners[owner++] = {start, end - start, segments[i], prefix, i, 0U};
        }
        prefix += segments[i];
    }
    return owner == used;
}

bool BuildBatchAlignedOwners(const ConcatTilingData& base, const uint64_t* segments,
                              uint32_t elementBytes, ConcatRowsTilingData& output)
{
    constexpr uint64_t tileBytes = 64ULL * 1024ULL;
    const uint32_t cores = base.blockDim;
    if (elementBytes != 4U || base.mode != 0U || base.rowGroups <= 1U ||
        base.outer < base.rowGroups || cores == 0U || cores > S9_CONCAT_MAX_INPUT_GROUPS ||
        base.inputCount == 0U || base.inputCount > cores || base.inner == 0U ||
        base.outputAxis == 0U || base.outputAxis > UINT64_MAX / base.inner / elementBytes / base.outer) {
        return false;
    }
    const uint64_t rowBytes = base.outputAxis * base.inner * elementBytes;
    const uint64_t totalBytes = rowBytes * base.outer;
    uint64_t bytes[S9_CONCAT_MAX_INPUT_GROUPS] = {};
    uint64_t batchRows[S9_CONCAT_MAX_INPUT_GROUPS] = {};
    uint64_t tasks[S9_CONCAT_MAX_INPUT_GROUPS] = {};
    uint64_t weights[S9_CONCAT_MAX_INPUT_GROUPS] = {};
    uint32_t counts[S9_CONCAT_MAX_INPUT_GROUPS] = {};
    uint32_t nonzero = 0U, used = 0U;
    uint64_t low = 0U, high = 0U;
    for (uint32_t i = 0U; i < base.inputCount; ++i) {
        if (segments[i] == 0U) { continue; }
        if (segments[i] > tileBytes / elementBytes) { return false; }
        bytes[i] = segments[i] * elementBytes;
        if (bytes[i] > rowBytes || rowBytes - bytes[i] > UINT32_MAX) { return false; }
        batchRows[i] = std::min<uint64_t>(4095ULL, tileBytes / ((bytes[i] + 31ULL) & ~31ULL));
        tasks[i] = RowsCeilDiv(base.outer, batchRows[i]);
        weights[i] = batchRows[i] * bytes[i];
        if (tasks[i] > UINT64_MAX / weights[i]) { return false; }
        low = std::max(low, weights[i]);
        high = std::max(high, tasks[i] * weights[i]);
        used += static_cast<uint32_t>(std::min<uint64_t>(tasks[i], cores - used));
        ++nonzero;
    }
    if (nonzero == 0U || used < nonzero) { return false; }

    const bool gridBypass = base.inputCount >= 2U && base.inputCount <= 8U &&
        totalBytes >= 16ULL * 1024ULL * 1024ULL;
    const bool weightedBypass = totalBytes >= 2ULL * tileBytes * used;
    if (gridBypass != weightedBypass) { return false; }
    while (low < high) {
        const uint64_t middle = low + (high - low) / 2U;
        uint64_t need = 0U;
        for (uint32_t i = 0U; i < base.inputCount && need <= used; ++i) {
            if (tasks[i] != 0U) { need += RowsCeilDiv(tasks[i], middle / weights[i]); }
        }
        if (need <= used) { high = middle; } else { low = middle + 1U; }
    }
    uint32_t allocated = 0U;
    for (uint32_t i = 0U; i < base.inputCount; ++i) {
        if (tasks[i] == 0U) { continue; }
        counts[i] = static_cast<uint32_t>(RowsCeilDiv(tasks[i], low / weights[i]));
        allocated += counts[i];
    }
    while (allocated < used) {
        uint32_t pick = base.inputCount;
        uint64_t largest = 0U;
        for (uint32_t i = 0U; i < base.inputCount; ++i) {
            if (counts[i] == 0U || counts[i] >= tasks[i]) { continue; }
            const uint64_t load = RowsCeilDiv(tasks[i], counts[i]) * weights[i];
            if (pick == base.inputCount || load > largest) { pick = i; largest = load; }
        }
        if (pick == base.inputCount) { return false; }
        ++counts[pick];
        ++allocated;
    }
    uint64_t oldMax = 0U, newMax = 0U, oldCalls = 0U, newCalls = 0U;
    for (uint32_t g = 0U; g < base.inputGroups; ++g) {
        uint64_t groupBytes = 0U;
        for (uint32_t i = g; i < base.inputCount; i += base.inputGroups) { groupBytes += bytes[i]; }
        oldMax = std::max(oldMax, groupBytes * RowsCeilDiv(base.outer, base.rowGroups));
    }
    output = {};
    output.base = base;
    output.base.blockDim = used;
    uint32_t owner = 0U;
    uint64_t prefix = 0U;
    for (uint32_t i = 0U; i < base.inputCount; ++i) {
        if (tasks[i] != 0U) {
            for (uint32_t g = 0U; g < base.rowGroups; ++g) {
                const uint64_t start = base.outer / base.rowGroups * g + base.outer % base.rowGroups * g / base.rowGroups;
                const uint64_t end = base.outer / base.rowGroups * (g + 1U) + base.outer % base.rowGroups * (g + 1U) / base.rowGroups;
                oldCalls += RowsCeilDiv(end - start, batchRows[i]);
            }
            for (uint32_t j = 0U; j < counts[i]; ++j) {
                const uint64_t first = tasks[i] / counts[i] * j + tasks[i] % counts[i] * j / counts[i];
                const uint64_t last = tasks[i] / counts[i] * (j + 1U) + tasks[i] % counts[i] * (j + 1U) / counts[i];
                const uint64_t start = first * batchRows[i];
                const uint64_t end = last == tasks[i] ? base.outer : last * batchRows[i];
                if (start >= end || owner >= used) { return false; }
                output.owners[owner++] = {start, end - start, segments[i], prefix, i, 0U};
                newMax = std::max(newMax, (end - start) * bytes[i]);
                newCalls += RowsCeilDiv(end - start, batchRows[i]);
            }
        }
        prefix += segments[i];
    }
    return owner == used && newMax < oldMax && newCalls <= oldCalls;
}

uint32_t GetAivCoreCount(const gert::TilingContext* context)
{
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    return std::max<uint32_t>(1U, platform.GetCoreNumAiv());
}

int64_t NormalizeAxis(int64_t axis, size_t rank)
{
    if (axis < 0) {
        axis += static_cast<int64_t>(rank);
    }
    return axis;
}

uint32_t StreamAlign32(uint32_t bytes)
{
    return (bytes + 31U) & ~31U;
}

uint32_t StreamRequiredGroups(const uint32_t* segments, uint32_t count,
                              uint32_t byteLimit)
{
    uint32_t groups = 0;
    uint32_t bytes = 0;
    uint32_t inputs = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t segment = segments[i];
        if (segment == 0) {
            continue;
        }
        if (segment > byteLimit) {
            return S9_CONCAT_MAX_INPUTS + 1U;
        }
        if (inputs != 0U && (bytes + segment > byteLimit ||
            inputs == S9_CONCAT_STREAM_MAX_GROUP_INPUTS)) {
            ++groups;
            bytes = 0;
            inputs = 0;
        }
        bytes += segment;
        ++inputs;
    }
    return groups + static_cast<uint32_t>(inputs != 0U);
}

bool BuildStreamingPlan(uint64_t outer, const uint64_t* elementSegments,
                        uint32_t count, uint32_t elementBytes,
                        uint32_t cores, ConcatStreamingTilingData& output,
                        bool wider = false)
{
    const uint32_t rowLimit = wider ? S9_CONCAT_STREAM_WIDE_ROW_BYTES :
                                     S9_CONCAT_STREAM_MAX_ROW_BYTES;
    const uint32_t inputCapacity = wider ? S9_CONCAT_STREAM_WIDE_INPUT_BYTES :
                                         S9_CONCAT_STREAM_INPUT_BYTES;
    const uint32_t outputCapacity = wider ? S9_CONCAT_STREAM_WIDE_OUTPUT_BYTES :
                                          S9_CONCAT_STREAM_OUTPUT_BYTES;

    if ((elementBytes != 2U && elementBytes != 4U) ||
        count < S9_CONCAT_PREFIX_THRESHOLD || count > S9_CONCAT_MAX_INPUTS ||
        cores == 0U || cores > S9_CONCAT_MAX_INPUT_GROUPS || outer == 0U) {
        return false;
    }
    output = {};
    output.bufferPolicy = static_cast<uint32_t>(wider);
    output.outer = outer;
    output.inputCount = count;
    output.elementBytes = elementBytes;
    uint32_t nonzero = 0;
    uint32_t largest = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (elementSegments[i] > 512U / elementBytes) {
            return false;
        }
        const uint32_t bytes = static_cast<uint32_t>(elementSegments[i]) * elementBytes;
        output.prefixes[i] = output.rowBytes;
        output.segments[i] = bytes;
        output.rowBytes += bytes;
        nonzero += static_cast<uint32_t>(bytes != 0U);
        largest = std::max(largest, bytes);
    }
    if (nonzero == 0U || output.rowBytes < 512U ||
        outer > std::numeric_limits<uint64_t>::max() / output.rowBytes ||
        outer * output.rowBytes < static_cast<uint64_t>(cores) * 65536ULL) {
        return false;
    }
    const uint32_t minimumGroups = StreamRequiredGroups(
        output.segments, count, rowLimit);
    uint32_t groups = 0;
    for (uint32_t candidate = minimumGroups; candidate <= cores && candidate <= nonzero;
         ++candidate) {
        if (cores % candidate == 0U) {
            groups = candidate;
            break;
        }
    }
    if (groups == 0U || output.rowBytes / groups < 512U) {
        return false;
    }

    uint32_t low = std::max(largest, (output.rowBytes + groups - 1U) / groups);
    uint32_t high = rowLimit;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (StreamRequiredGroups(output.segments, count, middle) <= groups) {
            high = middle;
        } else {
            low = middle + 1U;
        }
    }
    output.groupCount = groups;
    output.rowGroups = cores / groups;
    output.blockDim = cores;
    output.groupBegin[groups] = static_cast<uint16_t>(count);
    uint32_t cursor = count;
    uint32_t remainingNonzero = nonzero;
    for (uint32_t reverse = groups; reverse != 0U; --reverse) {
        const uint32_t group = reverse - 1U;
        uint32_t bytes = 0;
        uint32_t inputs = 0;
        uint32_t stride = 0;
        while (cursor != 0U) {
            const uint32_t segment = output.segments[cursor - 1U];
            if (segment == 0U) {
                --cursor;
                continue;
            }
            if (bytes + segment > low || inputs == S9_CONCAT_STREAM_MAX_GROUP_INPUTS ||
                remainingNonzero == group) {
                break;
            }
            bytes += segment;
            stride += StreamAlign32(S9_CONCAT_STREAM_MICRO_ROWS * segment);
            ++inputs;
            --remainingNonzero;
            --cursor;
        }
        if (inputs == 0U) {
            return false;
        }
        output.groupBegin[group] = static_cast<uint16_t>(cursor);
        output.groupBytes[group] = bytes;
        output.groupPitch[group] = StreamAlign32(bytes);
        output.microInputStride[group] = stride;
        output.macroRows[group] = S9_CONCAT_STREAM_MICRO_ROWS *
            (inputCapacity / stride);

        if (output.macroRows[group] == 0U ||
            outer / output.rowGroups < 2ULL * output.macroRows[group] ||
            S9_CONCAT_STREAM_MICRO_ROWS * output.groupPitch[group] >
                outputCapacity) {
            return false;
        }
    }
    return cursor == 0U && remainingNonzero == 0U;
}
bool BuildShortMicroPlan(const ConcatStreamingTilingData& original,
                         ConcatStreamingTilingData& output)
{
    if (original.bufferPolicy != 1U) {
        return false;
    }
    output = original;
    uint64_t oldCalls = 0;
    uint64_t newCalls = 0;
    const uint64_t minimumRows = original.outer / original.rowGroups;
    for (uint32_t group = 0; group < original.groupCount; ++group) {
        uint32_t stride = 0;
        uint32_t inputs = 0;
        for (uint32_t input = original.groupBegin[group];
             input < original.groupBegin[group + 1U]; ++input) {
            const uint32_t bytes = original.segments[input];
            if (bytes != 0U) {
                stride += StreamAlign32(S9_CONCAT_STREAM_SHORT_MICRO_ROWS * bytes);
                ++inputs;
            }
        }
        if (stride == 0U) {
            return false;
        }
        const uint32_t macro = S9_CONCAT_STREAM_SHORT_MICRO_ROWS *
            (S9_CONCAT_STREAM_SHORT_INPUT_BYTES / stride);
        if (macro == 0U || macro < original.macroRows[group] ||
            minimumRows < 2ULL * macro) {
            return false;
        }
        output.microInputStride[group] = stride;
        output.macroRows[group] = macro;
        for (uint32_t rowGroup = 0; rowGroup < original.rowGroups; ++rowGroup) {
            const uint64_t rows = original.outer * (rowGroup + 1U) / original.rowGroups -
                                  original.outer * rowGroup / original.rowGroups;
            oldCalls += (rows + original.macroRows[group] - 1U) /
                            original.macroRows[group] * inputs +
                        (rows + S9_CONCAT_STREAM_MICRO_ROWS - 1U) /
                            S9_CONCAT_STREAM_MICRO_ROWS;
            newCalls += (rows + macro - 1U) / macro * inputs +
                        (rows + S9_CONCAT_STREAM_SHORT_MICRO_ROWS - 1U) /
                            S9_CONCAT_STREAM_SHORT_MICRO_ROWS;
        }
    }
    if (newCalls >= oldCalls) {
        return false;
    }
    output.bufferPolicy = 2U;
    return true;
}
bool BuildAlignedOwnershipPlan(const ConcatStreamingTilingData& original,
                               ConcatStreamingTilingData& output)
{
    if (original.bufferPolicy != 2U || original.rowBytes % 32U != 0U) {
        return false;
    }
    uint32_t boundaries[S9_CONCAT_MAX_INPUT_GROUPS + 1] = {};
    bool changed = false;
    for (uint32_t group = 0; group < original.groupCount; ++group) {
        const uint32_t prefix = original.prefixes[original.groupBegin[group]];
        boundaries[group] = (prefix + 16U) & ~31U;
        changed = changed || boundaries[group] != prefix;
    }
    boundaries[original.groupCount] = original.rowBytes;
    if (!changed) {
        return false;
    }
    output = original;
    for (uint32_t group = 0; group < original.groupCount; ++group) {
        const uint32_t beginByte = boundaries[group];
        const uint32_t endByte = boundaries[group + 1U];
        if (endByte <= beginByte || endByte - beginByte > S9_CONCAT_STREAM_WIDE_ROW_BYTES) {
            return false;
        }
        uint32_t stride = 0;
        uint32_t inputs = 0;
        uint32_t first = original.inputCount;
        for (uint32_t input = 0; input < original.inputCount; ++input) {
            const uint32_t segment = original.segments[input];
            const uint32_t prefix = original.prefixes[input];
            if (segment != 0U && prefix < endByte && prefix + segment > beginByte) {
                if (inputs == 0U) {
                    first = input;
                }
                ++inputs;

                stride += StreamAlign32(S9_CONCAT_STREAM_SHORT_MICRO_ROWS * segment);
            }
        }
        if (inputs == 0U || inputs > S9_CONCAT_STREAM_MAX_GROUP_INPUTS) {
            return false;
        }
        const uint32_t macro = S9_CONCAT_STREAM_SHORT_MICRO_ROWS *
            (S9_CONCAT_STREAM_SHORT_INPUT_BYTES / stride);
        if (macro == 0U || original.outer / original.rowGroups < 2ULL * macro) {
            return false;
        }
        output.groupBegin[group] = static_cast<uint16_t>(first);
        output.groupBytes[group] = endByte - beginByte;
        output.groupPitch[group] = endByte - beginByte;
        output.microInputStride[group] = beginByte;
        output.macroRows[group] = macro;
    }
    output.bufferPolicy = 3U;
    return true;
}

bool BuildSerpentinePlan(const ConcatStreamingTilingData& original,
                         ConcatStreamingTilingData& output)
{
    if (original.bufferPolicy != 3U) {
        return false;
    }
    output = original;
    for (uint32_t group = 0U; group < original.groupCount; ++group) {
        const uint32_t beginByte = original.microInputStride[group];
        const uint32_t endByte = beginByte + original.groupBytes[group];
        uint32_t allocation = 0U;
        uint32_t alignmentRows = 1U;
        for (uint32_t input = original.groupBegin[group]; input < original.inputCount; ++input) {
            const uint32_t prefix = original.prefixes[input];
            if (prefix >= endByte) {
                break;
            }
            const uint32_t segment = original.segments[input];
            if (segment == 0U) {
                continue;
            }
            allocation += StreamAlign32(S9_CONCAT_SERP_MACRO_ROWS * segment);
            const uint32_t lowBit = segment & (~segment + 1U);
            alignmentRows = std::max(alignmentRows, 32U / std::min(32U, lowBit));
        }
        if (allocation == 0U || allocation > S9_CONCAT_STREAM_WIDE_INPUT_BYTES ||
            S9_CONCAT_SERP_MACRO_ROWS % alignmentRows != 0U) {
            return false;
        }
        for (uint32_t rg = 0U; rg < original.rowGroups; ++rg) {
            const uint64_t start = original.outer * rg / original.rowGroups;
            const uint64_t rows = original.outer * (rg + 1U) / original.rowGroups - start;
            if (start % alignmentRows != 0U || rows < 2U * S9_CONCAT_SERP_MACRO_ROWS) {
                return false;
            }
        }
        output.macroRows[group] = S9_CONCAT_SERP_MACRO_ROWS;
    }
    output.bufferPolicy = 5U;
    return true;
}

bool BuildWideAspectPlan(const ConcatStreamingTilingData& original,
                         ConcatStreamingTilingData& output)
{
    constexpr uint32_t microRows = 4U;
    constexpr uint64_t budget = 186560ULL;
    if (original.bufferPolicy != 1U || original.elementBytes != 4U ||
        original.blockDim % 4U != 0U) { return false; }
    const uint32_t groups = original.blockDim / 4U;
    if (groups == 0U || groups >= original.groupCount || groups > S9_CONCAT_MAX_INPUT_GROUPS) {
        return false;
    }
    uint32_t largest = 0U, nonzero = 0U;
    for (uint32_t i = 0; i < original.inputCount; ++i) {
        largest = std::max(largest, original.segments[i]);
        nonzero += static_cast<uint32_t>(original.segments[i] != 0U);
    }
    if (nonzero < groups || StreamRequiredGroups(original.segments, original.inputCount,
                                                  original.rowBytes) > groups) { return false; }
    uint32_t low = std::max(largest, (original.rowBytes + groups - 1U) / groups);
    uint32_t high = original.rowBytes;
    while (low < high) {
        const uint32_t middle = low + (high - low) / 2U;
        if (StreamRequiredGroups(original.segments, original.inputCount, middle) <= groups) {
            high = middle;
        } else {
            low = middle + 1U;
        }
    }
    output = original;
    output.groupCount = groups;
    output.rowGroups = 4U;
    output.blockDim = groups * output.rowGroups;
    output.groupBegin[groups] = static_cast<uint16_t>(original.inputCount);
    uint32_t cursor = original.inputCount;
    uint32_t remaining = nonzero;
    for (uint32_t reverse = groups; reverse != 0U; --reverse) {
        const uint32_t group = reverse - 1U;
        uint32_t bytes = 0U, inputs = 0U;
        while (cursor != 0U) {
            const uint32_t segment = original.segments[cursor - 1U];
            if (segment == 0U) { --cursor; continue; }
            if (bytes + segment > low || inputs == S9_CONCAT_STREAM_MAX_GROUP_INPUTS ||
                remaining == group) { break; }
            bytes += segment;
            ++inputs;
            --remaining;
            --cursor;
        }
        if (inputs == 0U) { return false; }
        output.groupBegin[group] = static_cast<uint16_t>(cursor);
        output.groupBytes[group] = bytes;
        output.groupPitch[group] = StreamAlign32(bytes);
    }
    if (cursor != 0U || remaining != 0U) { return false; }
    for (uint32_t group = 0; group < groups; ++group) {
        const uint32_t pitch = output.groupPitch[group];
        const uint32_t columns = pitch / output.elementBytes;
        const uint64_t fixed = 2ULL * (microRows * pitch + 32U) +
            (microRows * columns * 4ULL + 32U) + (2ULL * columns + 8ULL) * 4ULL;
        if (fixed + 64U >= budget) { return false; }
        uint32_t capacity = static_cast<uint32_t>((budget - fixed) / 2ULL) & ~31U;
        if (capacity <= 32U) { return false; }
        capacity -= 32U;
        uint32_t stride = 0U;
        for (uint32_t i = output.groupBegin[group]; i < output.groupBegin[group + 1U]; ++i) {
            if (output.segments[i] != 0U) {
                stride += StreamAlign32(microRows * output.segments[i]);
            }
        }
        if (stride == 0U) { return false; }
        const uint32_t macro = microRows * (capacity / stride);
        if (macro == 0U || original.outer / output.rowGroups < 2ULL * macro) { return false; }
        output.macroRows[group] = macro;
        output.microInputStride[group] = capacity;
    }
    output.bufferPolicy = 8U;
    return true;
}

uint64_t WideStaticUb(const ConcatStreamingTilingData& p, uint32_t group,
                      uint32_t rows, uint32_t* inputBytes)
{
    uint32_t input = 0U;
    for (uint32_t i = p.groupBegin[group]; i < p.groupBegin[group + 1U]; ++i) {
        if (p.segments[i] != 0U) {
            input += StreamAlign32(rows * p.segments[i]);
        }
    }
    if (inputBytes != nullptr) { *inputBytes = input; }
    const uint64_t columns = p.groupPitch[group] / p.elementBytes;
    constexpr uint32_t microRows = 4U;
    return 2ULL * (input + 32U) +
           2ULL * (microRows * p.groupPitch[group] + 32U) +
           (rows * columns * 4ULL + 32U) +
           (2ULL * columns + 8ULL) * 4ULL;
}

bool BuildWideStaticPlan(const ConcatStreamingTilingData& original,
                         ConcatStreamingTilingData& output)
{
    constexpr uint64_t budget = 186560ULL;
    constexpr uint32_t microRows = 4U;
    if (original.bufferPolicy != 8U || original.elementBytes != 4U) { return false; }
    output = original;
    const uint64_t minimumRows = original.outer / original.rowGroups;
    for (uint32_t group = 0; group < original.groupCount; ++group) {
        uint32_t bestRows = 0U, bestInput = 0U;
        for (uint32_t rows = microRows; rows <= minimumRows / 2U; rows += microRows) {
            uint32_t requiredInput = 0U;
            if (WideStaticUb(original, group, rows, &requiredInput) > budget) { break; }
            bestRows = rows;
            bestInput = requiredInput;
        }
        if (bestRows < 2U * microRows || bestInput == 0U) { return false; }
        output.macroRows[group] = bestRows;
        output.microInputStride[group] = bestInput;
    }
    output.bufferPolicy = 9U;
    return true;
}

bool BuildFullVectorPitchPlan(const ConcatStreamingTilingData& original,
                              ConcatStreamingTilingData& output)
{
    if (original.bufferPolicy != 9U || original.elementBytes != 4U) { return false; }
    output = original;
    bool changed = false;
    for (uint32_t group = 0U; group < original.groupCount; ++group) {
        const uint32_t pitch = (original.groupPitch[group] + 63U) & ~63U;
        changed = changed || pitch != original.groupPitch[group];
        output.groupPitch[group] = pitch;
        if (WideStaticUb(output, group, original.macroRows[group], nullptr) > 186560ULL) {
            return false;
        }
    }
    if (!changed) { return false; }
    output.bufferPolicy = 12U;
    return true;
}

using GatherCostInt = unsigned __int128;
struct GatherPartitionCost {
    uint32_t begin = 0U, end = 0U, bytes = 0U, pitch = 0U, macro = 0U, input = 0U;

    std::array<GatherCostInt, 7> maximum{};

    std::array<GatherCostInt, 5> total{};
};

bool GatherPartitionGeometry(const ConcatStreamingTilingData& p, uint32_t begin,
                             uint32_t end, uint32_t macro, uint32_t pitch,
                             GatherPartitionCost& c)
{
    if (begin >= end || end > p.inputCount || macro == 0U || macro % 4U != 0U ||
        pitch == 0U || pitch % 64U != 0U || p.rowGroups == 0U) { return false; }
    uint64_t input = 0U, bytes = 0U;
    uint32_t nonzero = 0U;
    for (uint32_t i = begin; i < end; ++i) {
        bytes += p.segments[i];
        nonzero += static_cast<uint32_t>(p.segments[i] != 0U);
        input += (static_cast<uint64_t>(macro) * p.segments[i] + 31U) / 32U * 32U;
    }
    if (bytes == 0U || bytes > pitch || input > UINT32_MAX ||
        nonzero > S9_CONCAT_STREAM_MAX_GROUP_INPUTS) { return false; }
    const GatherCostInt live = 2U * (static_cast<GatherCostInt>(input) + 32U) +
        2U * (4U * static_cast<GatherCostInt>(pitch) + 32U) +
        static_cast<GatherCostInt>(macro) * pitch + 32U + 2U * pitch + 32U;
    c = {};
    c.begin = begin; c.end = end; c.bytes = static_cast<uint32_t>(bytes);
    c.pitch = pitch; c.macro = macro; c.input = static_cast<uint32_t>(input);
    const uint64_t rows = RowsCeilDiv(p.outer, p.rowGroups);
    c.maximum = {static_cast<GatherCostInt>(nonzero) * RowsCeilDiv(rows, macro),
        static_cast<GatherCostInt>(rows) * bytes,
        2U * static_cast<GatherCostInt>(rows) * pitch + bytes + (1U + 3U * static_cast<GatherCostInt>(macro)) * pitch,
        (static_cast<GatherCostInt>(rows) + 5U + 2U * macro) * pitch,
        nonzero, static_cast<GatherCostInt>(macro) * pitch, live};
    GatherCostInt macros = 0U, stores = 0U;
    for (uint32_t r = 0U; r < p.rowGroups; ++r) {
        const uint64_t first = static_cast<GatherCostInt>(p.outer) * r / p.rowGroups;
        const uint64_t last = static_cast<GatherCostInt>(p.outer) * (r + 1U) / p.rowGroups;
        macros += RowsCeilDiv(last - first, macro);
        stores += RowsCeilDiv(last - first, 4U);
    }
    c.total = {static_cast<GatherCostInt>(nonzero) * macros,
        2U * static_cast<GatherCostInt>(p.outer) * pitch + static_cast<GatherCostInt>(p.rowGroups) *
            (bytes + (1U + 3U * static_cast<GatherCostInt>(macro)) * pitch),
        static_cast<GatherCostInt>(p.outer) * pitch + static_cast<GatherCostInt>(p.rowGroups) *
            (5U + 2U * static_cast<GatherCostInt>(macro)) * pitch,
        4U * (macros + stores), static_cast<GatherCostInt>(macro) * pitch};
    return true;
}

bool BuildBalancedStaticGroups(const ConcatStreamingTilingData& p,
                               ConcatStreamingTilingData& output)
{
    if (p.bufferPolicy != 12U || p.elementBytes != 4U || p.groupCount < 2U ||
        p.groupCount > S9_CONCAT_MAX_INPUT_GROUPS || p.inputCount > S9_CONCAT_MAX_INPUTS ||
        p.rowGroups == 0U || p.blockDim != p.groupCount * p.rowGroups ||
        p.outer / p.rowGroups < 16U || p.groupBegin[0] != 0U ||
        p.groupBegin[p.groupCount] != p.inputCount) { return false; }
    std::array<GatherCostInt, 7> oldMax{};
    std::array<GatherCostInt, 5> oldTotal{};
    for (uint32_t g = 0U; g < p.groupCount; ++g) {
        GatherPartitionCost c;
        if (!GatherPartitionGeometry(p, p.groupBegin[g], p.groupBegin[g + 1U],
                                     p.macroRows[g], p.groupPitch[g], c) ||
            c.input != p.microInputStride[g] || c.bytes != p.groupBytes[g]) { return false; }
        for (uint32_t j = 0U; j < oldMax.size(); ++j) { oldMax[j] = std::max(oldMax[j], c.maximum[j]); }
        for (uint32_t j = 0U; j < oldTotal.size(); ++j) { oldTotal[j] += c.total[j]; }
    }
    std::vector<std::vector<GatherPartitionCost>> edges(p.inputCount);
    for (uint32_t begin = 0U; begin < p.inputCount; ++begin) {
        uint32_t bytes = 0U;
        for (uint32_t end = begin + 1U;
             end <= p.inputCount && end - begin <= S9_CONCAT_STREAM_MAX_GROUP_INPUTS; ++end) {
            bytes += p.segments[end - 1U];
            if (bytes == 0U) { continue; }
            if (static_cast<GatherCostInt>(RowsCeilDiv(p.outer, p.rowGroups)) * bytes > oldMax[1]) { break; }
            const uint32_t pitch = (bytes + 63U) & ~63U;
            const uint64_t capacityRows = 186560ULL / (2ULL * bytes + pitch);
            uint32_t low = 0U;
            uint32_t high = static_cast<uint32_t>(std::min<uint64_t>(p.outer / p.rowGroups / 2U, capacityRows) / 4U);
            while (low < high) {
                const uint32_t mid = low + (high - low + 1U) / 2U;
                GatherPartitionCost trial;
                if (GatherPartitionGeometry(p, begin, end, mid * 4U, pitch, trial) && trial.maximum[6] <= 186560U) {
                    low = mid;
                } else { high = mid - 1U; }
            }
            GatherPartitionCost c;
            if (low < 2U || !GatherPartitionGeometry(p, begin, end, low * 4U, pitch, c)) { continue; }
            bool within = true;
            for (uint32_t j = 1U; j < oldMax.size(); ++j) { within = within && c.maximum[j] <= oldMax[j]; }
            if (within) { edges[begin].push_back(c); }
        }
    }
    const GatherCostInt infinity = ~static_cast<GatherCostInt>(0U);
    std::vector<GatherCostInt> maximum(p.inputCount + 1U, infinity);
    maximum[0] = 0U;
    for (uint32_t g = 0U; g < p.groupCount; ++g) {
        std::vector<GatherCostInt> next(p.inputCount + 1U, infinity);
        for (uint32_t begin = 0U; begin < p.inputCount; ++begin) {
            if (maximum[begin] == infinity) { continue; }
            for (const auto& edge : edges[begin]) {
                next[edge.end] = std::min(next[edge.end], std::max(maximum[begin], edge.maximum[0]));
            }
        }
        maximum.swap(next);
    }
    const GatherCostInt ceiling = maximum[p.inputCount];
    if (ceiling == infinity || ceiling >= oldMax[0]) { return false; }
    using Additive = std::array<GatherCostInt, 5>;
    std::vector<Additive> values(p.inputCount + 1U);
    std::vector<bool> reachable(p.inputCount + 1U, false);
    std::vector<std::vector<int32_t>> previous(p.groupCount + 1U,
        std::vector<int32_t>(p.inputCount + 1U, -1));
    reachable[0] = true;
    for (uint32_t g = 0U; g < p.groupCount; ++g) {
        std::vector<Additive> nextValues(p.inputCount + 1U);
        std::vector<bool> nextReachable(p.inputCount + 1U, false);
        for (uint32_t begin = 0U; begin < p.inputCount; ++begin) {
            if (!reachable[begin]) { continue; }
            for (const auto& edge : edges[begin]) {
                if (edge.maximum[0] > ceiling) { continue; }
                Additive sum = values[begin];
                for (uint32_t j = 0U; j < sum.size(); ++j) { sum[j] += edge.total[j]; }
                if (!nextReachable[edge.end] || sum < nextValues[edge.end]) {
                    nextReachable[edge.end] = true; nextValues[edge.end] = sum;
                    previous[g + 1U][edge.end] = static_cast<int32_t>(begin);
                }
            }
        }
        values.swap(nextValues); reachable.swap(nextReachable);
    }
    if (!reachable[p.inputCount]) { return false; }
    for (uint32_t j = 0U; j < oldTotal.size(); ++j) {
        if (values[p.inputCount][j] > oldTotal[j]) { return false; }
    }
    ConcatStreamingTilingData candidate = p;
    uint32_t end = p.inputCount;
    for (uint32_t g = p.groupCount; g != 0U; --g) {
        const int32_t begin = previous[g][end];
        if (begin < 0) { return false; }
        bool found = false;
        for (const auto& edge : edges[static_cast<uint32_t>(begin)]) {
            if (edge.end != end) { continue; }
            const uint32_t index = g - 1U;
            candidate.groupBegin[index] = static_cast<uint16_t>(begin);
            candidate.groupBytes[index] = edge.bytes;
            candidate.groupPitch[index] = edge.pitch;
            candidate.macroRows[index] = edge.macro;
            candidate.microInputStride[index] = edge.input;
            found = true;
            break;
        }
        if (!found) { return false; }
        end = static_cast<uint32_t>(begin);
    }
    if (end != 0U) { return false; }

    output = candidate;
    return true;
}

bool BuildSmallResidentPlan(const ConcatTilingData& base, const uint64_t* segments,
                            uint32_t elementBytes, uint32_t cores,
                            ConcatStreamingTilingData& p)
{
    if (elementBytes != 2U || cores == 0U || base.outer < 16U ||
        base.inputCount < 2U || base.inputCount > S9_CONCAT_STREAM_MAX_GROUP_INPUTS ||
        base.inner == 0U || base.outputAxis > UINT64_MAX / base.inner / 2U) { return false; }
    const uint64_t rowBytes = base.outputAxis * base.inner * 2U;
    if (rowBytes == 0U || rowBytes % 32U != 0U || rowBytes > 131072U ||
        base.outer > 131072U / rowBytes) { return false; }
    uint64_t smallest = UINT64_MAX;
    uint32_t nonzero = 0U;
    for (uint32_t i = 0; i < base.inputCount; ++i) {
        if (segments[i] != 0U) { smallest = std::min(smallest, segments[i] * 2U); ++nonzero; }
    }
    if (nonzero < 2U) { return false; }
    const uint64_t rowUnits = RowsCeilDiv(base.outer, 16U);
    const uint64_t minimumUnits = RowsCeilDiv(RowsCeilDiv(128U, smallest), 16U);
    const uint32_t owners = static_cast<uint32_t>(std::min<uint64_t>(cores,
        std::max<uint64_t>(1U, rowUnits / minimumUnits)));
    const uint64_t rows = RowsCeilDiv(rowUnits, owners) * 16U;
    uint64_t input = 0U;
    for (uint32_t i = 0; i < base.inputCount; ++i) {
        input += RowsCeilDiv(rows * segments[i] * 2U, 32U) * 32U;
    }
    const uint64_t output = rows * rowBytes;
    const uint64_t live = (input + 32U) + (output + 32U) +
        (2U * output + 32U) + 4U * rowBytes + 32U;
    if (live > 186560U || output / 2U > UINT32_MAX) { return false; }
    p = {};
    p.outer = base.outer; p.rowBytes = static_cast<uint32_t>(rowBytes);
    p.elementBytes = 2U; p.inputCount = base.inputCount;
    p.groupCount = 1U; p.rowGroups = owners; p.blockDim = owners;
    p.groupBegin[1] = static_cast<uint16_t>(base.inputCount);
    p.groupBytes[0] = p.groupPitch[0] = p.rowBytes;
    p.macroRows[0] = static_cast<uint32_t>(rows);
    p.microInputStride[0] = static_cast<uint32_t>(input);
    p.bufferPolicy = 18U;
    uint32_t prefix = 0U;
    for (uint32_t i = 0; i < base.inputCount; ++i) {
        p.prefixes[i] = prefix;
        p.segments[i] = static_cast<uint32_t>(segments[i] * 2U);
        prefix += p.segments[i];
    }
    return prefix == rowBytes;
}

bool BuildRowCompactPlan(const ConcatTilingData& original, const uint64_t* segments,
                         uint32_t elementBytes, uint32_t cores,
                         ConcatRowCompactTilingData& output)
{
    if (elementBytes != 4U || original.inputCount < 2U || original.inputCount > 8U ||
        cores == 0U || original.outer < cores) { return false; }
    output = {};
    output.base = original;
    uint32_t pitch = 0U, rowBytes = 0U, nonzero = 0U;
    for (uint32_t i = 0U; i < original.inputCount; ++i) {
        if (segments[i] > 16384U) { return false; }
        const uint32_t bytes = static_cast<uint32_t>(segments[i]) * 4U;
        const uint32_t padded = (bytes + 31U) & ~31U;
        if (padded > 65536U - pitch) { return false; }
        output.segments[i] = bytes;
        output.offsets[i] = pitch;
        pitch += padded;
        rowBytes += bytes;
        nonzero += bytes != 0U;
    }
    if (nonzero < 2U || rowBytes < 4096U || pitch == 0U) { return false; }
    output.inputPitch = pitch;
    output.outputRowBytes = rowBytes;
    output.batchRows = 65536U / pitch;
    output.maskBytes = output.batchRows * pitch / 32U;
    if (output.maskBytes > sizeof(output.mask)) { return false; }
    output.base.blockDim = cores;
    for (uint32_t row = 0U; row < output.batchRows; ++row) {
        for (uint32_t i = 0U; i < original.inputCount; ++i) {
            const uint32_t start = (row * pitch + output.offsets[i]) / 4U;
            const uint32_t words = output.segments[i] / 4U;
            for (uint32_t word = 0U; word < words; ++word) {
                const uint32_t bit = start + word;
                output.mask[bit / 8U] |= static_cast<uint8_t>(1U << (bit % 8U));
            }
        }
    }
    return true;
}
bool BuildPackedLongPlan(const ConcatStreamingTilingData& original,
                          ConcatPackedBlockTilingData& output)
{
    if (original.elementBytes != 4U ||
        (original.bufferPolicy != 9U && original.bufferPolicy != 12U) ||
        original.rowGroups == 0U || original.groupCount == 0U || original.groupCount > S9_CONCAT_MAX_INPUT_GROUPS) {
        return false;
    }
    output = {};
    output.base = original;
    uint32_t cursor = 0U;
    for (uint32_t g = 0U; g < original.groupCount; ++g) {
        uint32_t macro = original.macroRows[g];
        const uint32_t pitch = original.groupPitch[g];
        const uint32_t useful = original.groupBytes[g];
        if (macro < 8U || macro % 4U != 0U || pitch < useful || pitch % 32U != 0U) { return false; }
        uint32_t bases[S9_CONCAT_MAX_INPUTS] = {};
        uint32_t counts[S9_CONCAT_MAX_INPUTS] = {};
        uint64_t sourceBytes = 0U, dataBlocks = 0U;
        const uint32_t begin = original.groupBegin[g], end = original.groupBegin[g + 1U];
        for (uint32_t i = begin; i < end; ++i) {
            const uint32_t width = original.segments[i];
            bases[i] = static_cast<uint32_t>(sourceBytes);
            if (width == 0U) { continue; }
            if (width % 4U != 0U) { return false; }
            const uint32_t words = width / 4U;
            const uint32_t low = words & (~words + 1U);
            const uint32_t phaseStep = low < 8U ? low : 8U;
            const uint64_t cover = (static_cast<uint64_t>(width) + (8U - phaseStep) * 4U + 31U) / 32U;
            if (cover > 510U) { return false; }
            counts[i] = static_cast<uint32_t>(cover);
            dataBlocks += cover;
            sourceBytes += (static_cast<uint64_t>(macro) * width + 31ULL) & ~31ULL;
            if (sourceBytes > 1048576ULL || dataBlocks > 510ULL) { return false; }
        }
        if (sourceBytes != original.microInputStride[g] || dataBlocks == 0U) { return false; }
        const uint32_t padding = (pitch - useful) / 4U;
        const uint64_t block64 = (dataBlocks + (padding + 7U) / 8U + 7ULL) & ~7ULL;
        if (block64 == 0U || block64 > 510U) { return false; }
        const uint32_t blocks = static_cast<uint32_t>(block64);

        const uint64_t fixed = 2ULL * 32U + 2ULL * (4ULL * pitch + 32U) +
            32ULL + 8ULL * blocks + 32U + 4ULL * blocks * 32U + 32U;
        const uint64_t perRow = 2ULL * useful + 4ULL * blocks;
        if (fixed >= 186560ULL || perRow == 0U) { return false; }
        const uint64_t capacityRows = (186560ULL - fixed) / perRow;
        const uint64_t maxRows = std::min<uint64_t>(capacityRows,
            original.outer / original.rowGroups / 2U) / 8U * 8U;
        if (maxRows <= macro || maxRows > UINT32_MAX) { return false; }
        macro = static_cast<uint32_t>(maxRows);
        sourceBytes = 0U;
        for (uint32_t i = begin; i < end; ++i) {
            bases[i] = static_cast<uint32_t>(sourceBytes);
            sourceBytes += static_cast<uint64_t>(macro) * original.segments[i];
        }
        if (sourceBytes > 1048576ULL) { return false; }
        output.base.macroRows[g] = macro;
        output.base.microInputStride[g] = static_cast<uint32_t>(sourceBytes);
        const uint64_t indexBytes = static_cast<uint64_t>(macro) * blocks * 2ULL;
        const uint64_t maskBytes = 8ULL * blocks;
        const uint64_t explicitUb = 2ULL * (sourceBytes + 32ULL) +
            2ULL * (4ULL * pitch + 32ULL) +
            ((static_cast<uint64_t>(macro) * blocks * 4ULL + 63ULL) & ~31ULL) +
            ((maskBytes + 63ULL) & ~31ULL) + 4ULL * blocks * 32ULL + 32ULL;
        if (explicitUb > 186560ULL || indexBytes + maskBytes > sizeof(output.payload) - cursor) {
            return false;
        }
        output.rowBlocks[g] = blocks;
        output.payloadOffset[g] = cursor;
        auto* masks = output.payload + cursor + indexBytes;
        for (uint32_t row = 0U; row < macro; ++row) {
            uint32_t firstBlock = 0U;
            for (uint32_t i = begin; i < end; ++i) {
                const uint32_t width = original.segments[i];
                if (width == 0U) { continue; }
                const uint64_t address = static_cast<uint64_t>(bases[i]) + static_cast<uint64_t>(row) * width;
                const uint32_t phase = static_cast<uint32_t>(address % 32U) / 4U;
                for (uint32_t b = 0U; b < counts[i]; ++b) {
                    uint8_t mask = 0U;
                    for (uint32_t lane = 0U; lane < 8U; ++lane) {
                        const uint32_t position = b * 8U + lane;
                        if (position >= phase && position - phase < width / 4U) {
                            mask |= static_cast<uint8_t>(1U << lane);
                        }
                    }
                    const uint64_t block = mask == 0U ? 0ULL : address / 32ULL + b;
                    if (block >= 32768ULL) { return false; }
                    const int16_t encoded = static_cast<int16_t>(block);
                    std::memcpy(output.payload + cursor + 2U * (row * blocks + firstBlock + b),
                                &encoded, sizeof(encoded));
                    if (row < 8U) { masks[row * blocks + firstBlock + b] = mask; }
                }
                firstBlock += counts[i];
            }
            if (row < 8U) {
                for (uint32_t word = 0U; word < padding; ++word) {
                    masks[row * blocks + firstBlock + word / 8U] |= static_cast<uint8_t>(1U << (word % 8U));
                }
            }
        }
        cursor += static_cast<uint32_t>(indexBytes + maskBytes);
    }
    output.base.bufferPolicy = 14U;
    return true;
}

}

namespace optiling {
// 校验输入形状，计算拼接前缀，并根据访存布局和缓冲容量选择分块路径。
static ge::graphStatus ConcatTiling(gert::TilingContext* context)
{
    ConcatManyTilingData computed = {};
    ConcatLinearTilingData linearComputed = {};
    auto* tiling = &computed.base;
    uint64_t segments[S9_CONCAT_MAX_INPUTS] = {};
    const auto* firstStorage = context->GetDynamicInputShape(0, 0);
    if (firstStorage == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& first = firstStorage->GetStorageShape();
    const auto* firstDesc = context->GetDynamicInputDesc(0, 0);
    if (firstDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int elementBytes = ge::GetSizeByDataType(firstDesc->GetDataType());
    if (elementBytes <= 0) {
        return ge::GRAPH_FAILED;
    }
    const size_t rank = first.GetDimNum();
    const size_t inputCount = context->GetComputeNodeInputNum();
    const int64_t* axisAttr = context->GetAttrs()->GetInt(0);
    const int64_t axis = NormalizeAxis(axisAttr == nullptr ? 0 : *axisAttr, rank);
    if (rank == 0 || rank > S9_CONCAT_MAX_RANK || inputCount == 0 ||
        inputCount > S9_CONCAT_MAX_INPUTS ||
        axis < 0 || axis >= static_cast<int64_t>(rank)) {
        return ge::GRAPH_FAILED;
    }

    tiling->outer = 1;
    tiling->inner = 1;
    tiling->outputAxis = 0;
    tiling->maxSegment = 0;
    tiling->inputCount = static_cast<uint32_t>(inputCount);
    tiling->axis = static_cast<uint32_t>(axis);
    tiling->rank = static_cast<uint32_t>(rank);
    for (int64_t i = 0; i < axis; ++i) {
        tiling->outer *= static_cast<uint64_t>(first.GetDim(i));
    }
    for (size_t i = static_cast<size_t>(axis + 1); i < rank; ++i) {
        tiling->inner *= static_cast<uint64_t>(first.GetDim(i));
    }

    for (size_t i = 0; i < inputCount; ++i) {
        const auto* storage = context->GetDynamicInputShape(0, i);
        if (storage == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape& shape = storage->GetStorageShape();
        if (shape.GetDimNum() != rank) {
            return ge::GRAPH_FAILED;
        }
        for (size_t dim = 0; dim < rank; ++dim) {
            if (dim != static_cast<size_t>(axis) && shape.GetDim(dim) != first.GetDim(dim)) {
                return ge::GRAPH_FAILED;
            }
        }
        const uint64_t axisLength = static_cast<uint64_t>(shape.GetDim(axis));
        const uint64_t segment = axisLength * tiling->inner;
        segments[i] = segment;
        computed.prefixes[i] = tiling->outputAxis * tiling->inner;
        tiling->outputAxis += axisLength;
        tiling->maxSegment = std::max<uint64_t>(tiling->maxSegment, segment);
    }
    tiling->taskCount = tiling->outer * tiling->inputCount;
    const uint32_t coreCount = GetAivCoreCount(context);

    ConcatStreamingTilingData resident = {};
    if (BuildSmallResidentPlan(*tiling, segments, static_cast<uint32_t>(elementBytes),
                               coreCount, resident)) {
        auto* output = context->GetTilingData<ConcatStreamingTilingData>();
        if (output != nullptr) {
            *output = resident;
            context->SetTilingKey(4);
            context->SetBlockDim(resident.blockDim);
            context->GetWorkspaceSizes(1)[0] = 0U;
            return ge::GRAPH_SUCCESS;
        }
    }
    ConcatRowCompactTilingData rowCompact = {};
    if (BuildRowCompactPlan(*tiling, segments, static_cast<uint32_t>(elementBytes), coreCount, rowCompact)) {
        auto* output = context->GetTilingData<ConcatRowCompactTilingData>();
        if (output != nullptr) {
            *output = rowCompact;

            context->SetTilingKey(24);
            context->SetBlockDim(rowCompact.base.blockDim);
            context->GetWorkspaceSizes(1)[0] = 0;
            return ge::GRAPH_SUCCESS;
        }
    }

    ConcatStreamingTilingData streaming = {};
    if (BuildStreamingPlan(tiling->outer, segments, tiling->inputCount,
                           static_cast<uint32_t>(elementBytes), coreCount, streaming)) {
        ConcatStreamingTilingData wider = {};
        if (BuildStreamingPlan(tiling->outer, segments, tiling->inputCount,
                               static_cast<uint32_t>(elementBytes), coreCount,
                               wider, true) && wider.groupCount < streaming.groupCount) {

            streaming = wider;
        }

        ConcatStreamingTilingData shorter = {};
        ConcatStreamingTilingData aligned = {};
        ConcatStreamingTilingData serpentine = {};
        if (BuildShortMicroPlan(streaming, shorter) &&
            BuildAlignedOwnershipPlan(shorter, aligned) &&
            BuildSerpentinePlan(aligned, serpentine)) {
            streaming = serpentine;
        }
        ConcatStreamingTilingData wideAspect = {};
        if (BuildWideAspectPlan(streaming, wideAspect)) { streaming = wideAspect; }
        ConcatStreamingTilingData wideStatic = {};
        if (BuildWideStaticPlan(streaming, wideStatic)) { streaming = wideStatic; }
        ConcatStreamingTilingData fullVectorPitch = {};
        if (BuildFullVectorPitchPlan(streaming, fullVectorPitch)) { streaming = fullVectorPitch; }
        ConcatStreamingTilingData balancedStatic = {};
        if (BuildBalancedStaticGroups(streaming, balancedStatic)) { streaming = balancedStatic; }
        if (firstDesc->GetDataType() == ge::DT_FLOAT &&
            context->GetRawTilingData()->GetCapacity() >= sizeof(ConcatPackedBlockTilingData)) {
            auto* packed = context->GetTilingData<ConcatPackedBlockTilingData>();
            if (packed != nullptr && BuildPackedLongPlan(streaming, *packed)) {
                const uint32_t last = packed->base.groupCount - 1U;
                const uint64_t used = 3776ULL + packed->payloadOffset[last] +
                    (2ULL * packed->base.macroRows[last] + 8U) * packed->rowBlocks[last];
                context->GetRawTilingData()->SetDataSize(used);
                context->SetTilingKey(4);
                context->SetBlockDim(streaming.blockDim);
                context->GetWorkspaceSizes(1)[0] = 0U;
                return ge::GRAPH_SUCCESS;
            }
        }
        auto* output = context->GetTilingData<ConcatStreamingTilingData>();
        if (output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *output = streaming;
        context->SetTilingKey(4);
        context->SetBlockDim(streaming.blockDim);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    constexpr uint32_t kPackAlignElements = 16;

    const uint32_t packPayloadElements = tiling->outer >= 4096 ? 1536U : 1224U;
    constexpr uint32_t kPackMaxSegmentElements = 80;
    uint16_t packGroupCount = 0;
    if (elementBytes == 2 && tiling->inner == 1 &&
        inputCount >= S9_CONCAT_PREFIX_THRESHOLD &&
        tiling->maxSegment <= kPackMaxSegmentElements) {
        uint32_t packPaddedElements = 0;
        uint32_t packRealElements = 0;
        computed.packGroupOffsets[0] = 0;
        for (uint16_t inputIndex = 0; inputIndex < inputCount; ++inputIndex) {
            const uint32_t segment = static_cast<uint32_t>(segments[inputIndex]);
            const uint32_t alignedSegment =
                (segment + kPackAlignElements - 1) & ~(kPackAlignElements - 1);
            if (segment != 0 && packRealElements != 0 &&
                packPaddedElements + alignedSegment > packPayloadElements) {
                computed.packGroupRealElements[packGroupCount] =
                    static_cast<uint16_t>(packRealElements);
                ++packGroupCount;
                computed.packGroupOffsets[packGroupCount] = inputIndex;
                packPaddedElements = 0;
                packRealElements = 0;
            }
            packPaddedElements += alignedSegment;
            packRealElements += segment;
        }
        if (packRealElements != 0) {
            computed.packGroupRealElements[packGroupCount] =
                static_cast<uint16_t>(packRealElements);
            ++packGroupCount;
            computed.packGroupOffsets[packGroupCount] =
                static_cast<uint16_t>(inputCount);
        }
        computed.packGroupCount = packGroupCount;
    }

    uint32_t bestRows = 1;
    uint32_t bestInputs = 1;
    uint32_t bestBlocks = 1;
    const uint32_t maxRows = static_cast<uint32_t>(
        std::min<uint64_t>(tiling->outer, coreCount));
    for (uint32_t rows = 1; rows <= std::max<uint32_t>(1, maxRows); ++rows) {
        const uint32_t inputGroups = std::max<uint32_t>(1,
            std::min<uint32_t>(tiling->inputCount, coreCount / rows));
        const uint32_t usedBlocks = rows * inputGroups;

        if (usedBlocks > bestBlocks) {
            bestRows = rows;
            bestInputs = inputGroups;
            bestBlocks = usedBlocks;
        }
    }

    constexpr uint64_t kSmallInputBytesPerCore = 32ULL * 1024ULL;
    const uint64_t maxInputBytes = tiling->outer * tiling->maxSegment *
        static_cast<uint64_t>(elementBytes);
    if (maxInputBytes <= kSmallInputBytesPerCore && inputCount <= coreCount) {
        bestRows = 1;
        bestInputs = static_cast<uint32_t>(inputCount);
        bestBlocks = static_cast<uint32_t>(inputCount);
    }

    constexpr uint64_t kElementsPerColumnCore = 8192;
    const uint32_t columnBlocks = static_cast<uint32_t>(
        std::min<uint64_t>(coreCount,
            std::max<uint64_t>(1,
                (tiling->maxSegment + kElementsPerColumnCore - 1) /
                    kElementsPerColumnCore)));
    const bool usePackedHalf = elementBytes == 2 && tiling->inner == 1 &&
        inputCount >= S9_CONCAT_PREFIX_THRESHOLD && tiling->outer >= 512 &&
        tiling->outer <= 4096 && tiling->maxSegment > 0 &&
        tiling->maxSegment <= kPackMaxSegmentElements &&
        tiling->outputAxis <=
            std::numeric_limits<uint32_t>::max() / (16U * elementBytes) &&
        packGroupCount > 0;
    if (usePackedHalf) {
        tiling->mode = 3;
        tiling->inputGroups = std::min<uint32_t>(coreCount, packGroupCount);
        tiling->rowGroups = std::max<uint32_t>(1U,
            std::min<uint32_t>(static_cast<uint32_t>(tiling->outer),
                               coreCount / tiling->inputGroups));
        tiling->blockDim = tiling->inputGroups * tiling->rowGroups;
    } else if (columnBlocks > bestBlocks) {
        tiling->mode = 1;
        tiling->rowGroups = 1;
        tiling->inputGroups = 1;
        tiling->blockDim = columnBlocks;
    } else if (inputCount <= 48 &&
               tiling->maxSegment > 32768ULL && tiling->taskCount >= coreCount) {

        tiling->mode = 2;
        tiling->rowGroups = 1;
        tiling->inputGroups = 1;
        tiling->blockDim = coreCount;
    } else {
        tiling->mode = 0;
        tiling->rowGroups = bestRows;
        tiling->inputGroups = bestInputs;
        tiling->blockDim = bestBlocks;
    }
    tiling->reserved = 0;
    ConcatRowsTilingData weighted = {};
    if (BuildWeightedRows(*tiling, segments, static_cast<uint32_t>(elementBytes), weighted) ||
        BuildBatchAlignedOwners(*tiling, segments, static_cast<uint32_t>(elementBytes), weighted)) {
        auto* output = context->GetTilingData<ConcatRowsTilingData>();
        if (output == nullptr) { return ge::GRAPH_FAILED; }
        *output = weighted;
        context->SetTilingKey(15);
        context->SetBlockDim(weighted.base.blockDim);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
    ConcatRowsTilingData exactGrid = {};
    if (BuildDescriptorFreeGrid(*tiling, segments, static_cast<uint32_t>(elementBytes), exactGrid)) {
        auto* output = context->GetTilingData<ConcatRowsTilingData>();
        if (output != nullptr) {
            *output = exactGrid;
            context->SetTilingKey(15);
            context->SetBlockDim(tiling->blockDim);
            context->GetWorkspaceSizes(1)[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

    }

    if (tiling->mode == 1U && tiling->outer == 1U && inputCount > 1U &&
        tiling->inner != 0U && tiling->outputAxis <= UINT64_MAX / tiling->inner / 32U) {
        auto* output = context->GetTilingData<ConcatFlatTilingData>();
        if (output == nullptr) { return ge::GRAPH_FAILED; }
        *output = {};
        output->base = *tiling;
        for (uint32_t i = 0; i < inputCount; ++i) { output->prefixes[i] = computed.prefixes[i]; }
        output->prefixes[inputCount] = tiling->outputAxis * tiling->inner;
        context->SetTilingKey(30);
        context->SetBlockDim(tiling->blockDim);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
    const bool useManyGrid = inputCount >= S9_CONCAT_PREFIX_THRESHOLD &&
        (tiling->mode == 0 || tiling->mode == 3);
    const bool useBalancedLinear =
        tiling->mode == 2 && tiling->taskCount <= S9_CONCAT_MAX_LINEAR_TASKS;
    if (useManyGrid || useBalancedLinear) {
        if (useManyGrid && tiling->mode == 3) {

        const uint32_t groupCount = tiling->blockDim;
        if (groupCount == 0 || groupCount > S9_CONCAT_MAX_INPUT_GROUPS) {
            return ge::GRAPH_FAILED;
        }
        if (packGroupCount <= tiling->inputGroups &&
            tiling->rowGroups > 1U) {
            uint16_t cursor = 0;
            for (uint32_t rowGroup = 0; rowGroup < tiling->rowGroups;
                 ++rowGroup) {
                for (uint16_t group = 0; group < packGroupCount; ++group) {
                    computed.groupOffsets[cursor] = cursor;
                    computed.inputOrder[cursor] = group;
                    ++cursor;
                }
            }
            computed.groupOffsets[cursor] = cursor;
        } else {
        uint16_t sortedGroups[S9_CONCAT_MAX_PACK_GROUPS] = {};
        uint16_t assignments[S9_CONCAT_MAX_PACK_GROUPS] = {};
        uint64_t costs[S9_CONCAT_MAX_PACK_GROUPS] = {};
        uint64_t groupLoads[S9_CONCAT_MAX_INPUT_GROUPS] = {};
        uint16_t groupSizes[S9_CONCAT_MAX_INPUT_GROUPS] = {};
        constexpr uint64_t kPackQueueElements = 64ULL * 1024ULL / 2ULL;
        for (uint16_t group = 0; group < packGroupCount; ++group) {
            sortedGroups[group] = group;
            uint64_t sourceRowElements = 0;
            uint64_t realElements = 0;
            uint64_t nonzeroInputs = 0;
            uint64_t firstTransposeCommands = 0;
            for (uint16_t inputIndex = computed.packGroupOffsets[group];
                 inputIndex < computed.packGroupOffsets[group + 1]; ++inputIndex) {
                const uint64_t segment = segments[inputIndex];
                if (segment == 0) {
                    continue;
                }
                const uint64_t alignedSegment = (segment + 15ULL) & ~15ULL;
                sourceRowElements += alignedSegment;
                firstTransposeCommands += alignedSegment / 16ULL;
                realElements += segment;
                ++nonzeroInputs;
            }
            const uint64_t alignedReal = (realElements + 15ULL) & ~15ULL;
            const uint64_t middleColumns = alignedReal + 16ULL;
            uint64_t rowsPerBatch = std::min<uint64_t>(
                kPackQueueElements / sourceRowElements,
                std::min<uint64_t>(kPackQueueElements / alignedReal,
                                   kPackQueueElements / middleColumns));
            rowsPerBatch = std::min<uint64_t>(4080, rowsPerBatch) & ~15ULL;
            rowsPerBatch = std::max<uint64_t>(16, rowsPerBatch);
            const uint64_t batches =
                (tiling->outer + rowsPerBatch - 1) / rowsPerBatch;
            const uint64_t secondTransposeCommands = alignedReal / 16ULL;

            costs[group] = batches * (nonzeroInputs + firstTransposeCommands +
                                      secondTransposeCommands + 1ULL);
        }
        std::sort(sortedGroups, sortedGroups + packGroupCount,
                  [&](uint16_t lhs, uint16_t rhs) {
                      return costs[lhs] > costs[rhs];
                  });
        for (uint16_t sorted = 0; sorted < packGroupCount; ++sorted) {
            const uint16_t group = sortedGroups[sorted];
            uint32_t bestGroup = 0;
            for (uint32_t core = 1; core < groupCount; ++core) {
                if (groupLoads[core] < groupLoads[bestGroup] ||
                    (groupLoads[core] == groupLoads[bestGroup] &&
                     groupSizes[core] < groupSizes[bestGroup])) {
                    bestGroup = core;
                }
            }
            assignments[group] = static_cast<uint16_t>(bestGroup);
            groupLoads[bestGroup] += costs[group];
            ++groupSizes[bestGroup];
        }
        uint16_t cursor = 0;
        for (uint32_t core = 0; core < groupCount; ++core) {
            computed.groupOffsets[core] = cursor;
            for (uint16_t group = 0; group < packGroupCount; ++group) {
                if (assignments[group] == core) {
                    computed.inputOrder[cursor++] = group;
                }
            }
        }
        computed.groupOffsets[groupCount] = cursor;
        }
        } else if (useManyGrid && tiling->mode == 0) {

        const uint32_t groupCount = tiling->inputGroups;
        if (groupCount == 0 || groupCount > S9_CONCAT_MAX_INPUT_GROUPS) {
            return ge::GRAPH_FAILED;
        }
        uint16_t sortedInputs[S9_CONCAT_MAX_INPUTS] = {};
        uint16_t assignments[S9_CONCAT_MAX_INPUTS] = {};
        uint64_t groupLoads[S9_CONCAT_MAX_INPUT_GROUPS] = {};
        uint16_t groupSizes[S9_CONCAT_MAX_INPUT_GROUPS] = {};
        for (uint16_t i = 0; i < inputCount; ++i) {
            sortedInputs[i] = i;
        }
        std::sort(sortedInputs, sortedInputs + inputCount,
                  [&](uint16_t lhs, uint16_t rhs) {
                      return segments[lhs] > segments[rhs];
                  });
        for (size_t sorted = 0; sorted < inputCount; ++sorted) {
            const uint16_t inputIndex = sortedInputs[sorted];
            if (segments[inputIndex] == 0) {
                continue;
            }
            uint32_t bestGroup = 0;
            for (uint32_t group = 1; group < groupCount; ++group) {
                if (groupLoads[group] < groupLoads[bestGroup] ||
                    (groupLoads[group] == groupLoads[bestGroup] &&
                     groupSizes[group] < groupSizes[bestGroup])) {
                    bestGroup = group;
                }
            }
            assignments[inputIndex] = static_cast<uint16_t>(bestGroup);
            groupLoads[bestGroup] += segments[inputIndex];
            ++groupSizes[bestGroup];
        }
        uint16_t cursor = 0;
        for (uint32_t group = 0; group < groupCount; ++group) {
            computed.groupOffsets[group] = cursor;
            for (uint16_t inputIndex = 0; inputIndex < inputCount; ++inputIndex) {
                if (segments[inputIndex] != 0 && assignments[inputIndex] == group) {
                    computed.inputOrder[cursor++] = inputIndex;
                }
            }
        }
        computed.groupOffsets[groupCount] = cursor;
        } else if (useBalancedLinear) {

            const uint32_t groupCount = tiling->blockDim;
            if (groupCount == 0 || groupCount > S9_CONCAT_MAX_INPUT_GROUPS) {
                return ge::GRAPH_FAILED;
            }
            uint16_t sortedTasks[S9_CONCAT_MAX_LINEAR_TASKS] = {};
            uint16_t assignments[S9_CONCAT_MAX_LINEAR_TASKS] = {};
            uint64_t groupLoads[S9_CONCAT_MAX_INPUT_GROUPS] = {};
            uint16_t groupSizes[S9_CONCAT_MAX_INPUT_GROUPS] = {};
            uint16_t scheduledTaskCount = 0;
            for (uint32_t task = 0; task < tiling->taskCount; ++task) {
                const uint32_t inputIndex = task % tiling->inputCount;
                if (segments[inputIndex] != 0) {
                    sortedTasks[scheduledTaskCount++] = static_cast<uint16_t>(task);
                }
            }
            std::sort(sortedTasks, sortedTasks + scheduledTaskCount,
                      [&](uint16_t lhs, uint16_t rhs) {
                          return segments[lhs % tiling->inputCount] >
                                 segments[rhs % tiling->inputCount];
                      });
            for (uint16_t sorted = 0; sorted < scheduledTaskCount; ++sorted) {
                const uint16_t task = sortedTasks[sorted];
                uint32_t bestGroup = 0;
                for (uint32_t group = 1; group < groupCount; ++group) {
                    if (groupLoads[group] < groupLoads[bestGroup] ||
                        (groupLoads[group] == groupLoads[bestGroup] &&
                         groupSizes[group] < groupSizes[bestGroup])) {
                        bestGroup = group;
                    }
                }
                assignments[task] = static_cast<uint16_t>(bestGroup);
                groupLoads[bestGroup] += segments[task % tiling->inputCount];
                ++groupSizes[bestGroup];
            }
            uint16_t cursor = 0;
            for (uint32_t group = 0; group < groupCount; ++group) {
                linearComputed.groupOffsets[group] = cursor;
                for (uint16_t sorted = 0; sorted < scheduledTaskCount; ++sorted) {
                    const uint16_t task = sortedTasks[sorted];
                    if (assignments[task] == group) {
                        linearComputed.linearTaskOrder[cursor++] = task;
                    }
                }
            }
            linearComputed.groupOffsets[groupCount] = cursor;
        }
        if (useManyGrid) {
            auto* output = context->GetTilingData<ConcatManyTilingData>();
            if (output == nullptr) {
                return ge::GRAPH_FAILED;
            }
            *output = computed;
            context->SetTilingKey(2);
        } else {
            linearComputed.base = computed.base;
            for (size_t inputIndex = 0; inputIndex < inputCount; ++inputIndex) {
                linearComputed.prefixes[inputIndex] = computed.prefixes[inputIndex];
            }
            auto* output = context->GetTilingData<ConcatLinearTilingData>();
            if (output == nullptr) {
                return ge::GRAPH_FAILED;
            }
            *output = linearComputed;
            context->SetTilingKey(3);
        }
    } else {
        auto* output = context->GetTilingData<ConcatTilingData>();
        if (output == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *output = computed.base;
        context->SetTilingKey(1);
    }
    context->SetBlockDim(tiling->blockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus ConcatInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* first = context->GetDynamicInputShape(0, 0);
    gert::Shape* output = context->GetOutputShape(0);
    if (first == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    const size_t rank = first->GetDimNum();
    const size_t inputCount = context->GetComputeNodeInputNum();
    const int64_t* axisAttr = context->GetAttrs()->GetInt(0);
    const int64_t axis = NormalizeAxis(axisAttr == nullptr ? 0 : *axisAttr, rank);
    if (rank == 0 || rank > S9_CONCAT_MAX_RANK || inputCount == 0 ||
        axis < 0 || axis >= static_cast<int64_t>(rank)) {
        return GRAPH_FAILED;
    }

    *output = *first;
    int64_t concatenated = 0;
    for (size_t i = 0; i < inputCount; ++i) {
        const gert::Shape* input = context->GetDynamicInputShape(0, i);
        if (input == nullptr) {
            return GRAPH_FAILED;
        }
        if (input->GetDimNum() != rank) {
            return GRAPH_FAILED;
        }
        for (size_t dim = 0; dim < rank; ++dim) {
            if (dim != static_cast<size_t>(axis) && input->GetDim(dim) != first->GetDim(dim)) {
                return GRAPH_FAILED;
            }
        }
        concatenated += input->GetDim(axis);
    }
    output->SetDim(axis, concatenated);
    return GRAPH_SUCCESS;
}

static ge::graphStatus ConcatInferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetDynamicInputDataType(0, 0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Concat : public OpDef {
public:
    explicit Concat(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> types = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("inputs").ParamType(DYNAMIC).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Output("y").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Attr("dim").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::ConcatInferShape)
            .SetInferDataType(ge::ConcatInferDataType);
        this->AICore().SetTiling(optiling::ConcatTiling).AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}
