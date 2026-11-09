#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace squaresum {
constexpr unsigned kMaxRank = 64;
constexpr unsigned kOutputGroup = 16;
struct Plan {
    uint64_t inputNum = 0, outputNum = 0, outerNum = 1, reduceNum = 1;
    uint64_t tailNum = 1, units = 0;
    uint32_t tile = 0, cores = 1, tailReduced = 0, keepRank = 0, reduceRank = 0;
    uint64_t keepDims[kMaxRank]{}, keepStrides[kMaxRank]{};
    uint64_t reduceDims[kMaxRank]{}, reduceStrides[kMaxRank]{};
};
inline bool Product(uint64_t a, uint64_t b, uint64_t &out) {
    if (b && a > static_cast<uint64_t>(INT64_MAX) / b) return false;
    out = a * b;
    return true;
}
inline uint64_t CeilDiv(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }

// A dense ND tensor becomes alternating kept/reduced dimension runs.
// The innermost run is contiguous.
inline bool MakePlan(const std::vector<int64_t> &shape, const std::vector<int64_t> &axes,
                     uint64_t ubBytes, uint32_t coreCount, uint32_t typeBytes, Plan &p) {
    p = Plan{};
    const size_t rank = shape.empty() ? 1 : shape.size();
    if (rank > kMaxRank || !coreCount || (typeBytes != 2 && typeBytes != 4)) return false;
    std::vector<bool> reduced(rank, axes.empty());
    for (auto a : axes) {
        if (a < 0) a += rank;
        if (a < 0 || a >= static_cast<int64_t>(rank) || reduced[a]) return false;
        reduced[a] = true;
    }
    std::vector<uint64_t> dims, strides;
    std::vector<bool> flags;
    p.inputNum = p.outputNum = 1;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t d = shape.empty() ? 1 : shape[i];
        if (d < 0 || !Product(p.inputNum, d, p.inputNum)) return false;
        if (!reduced[i] && !Product(p.outputNum, d, p.outputNum)) return false;
        // Unit dimensions have no address effect; eliminating them also lets
        // a reduction followed by kept singleton dimensions stream contiguously.
        if (d != 1) { dims.push_back(d); flags.push_back(reduced[i]); }
    }
    if (dims.empty()) { dims.push_back(1); flags.push_back(true); }
    strides.resize(dims.size());
    uint64_t stride = 1;
    for (size_t i = dims.size(); i-- > 0;) {
        strides[i] = stride;
        if (!Product(stride, dims[i], stride)) return false;
    }
    p.tailReduced = flags.back();
    size_t tailBegin = dims.size() - 1;
    while (tailBegin && flags[tailBegin - 1] == flags.back()) --tailBegin;
    for (size_t i = tailBegin; i < dims.size(); ++i)
        if (!Product(p.tailNum, dims[i], p.tailNum)) return false;
    for (size_t i = 0; i < tailBegin; ++i) {
        if (flags[i]) {
            p.reduceDims[p.reduceRank] = dims[i]; p.reduceStrides[p.reduceRank++] = strides[i];
            if (!Product(p.reduceNum, dims[i], p.reduceNum)) return false;
        } else {
            p.keepDims[p.keepRank] = dims[i]; p.keepStrides[p.keepRank++] = strides[i];
            if (!Product(p.outerNum, dims[i], p.outerNum)) return false;
        }
    }
    // Three float vectors plus reduction scratch, three T vectors, small scalars.
    if (ubBytes < 4096) return false;
    p.tile = static_cast<uint32_t>(std::min<uint64_t>(4096, (ubBytes - 1024) / (16 + 3 * typeBytes)));
    p.tile = p.tile / 64 * 64;
    if (!p.tile) return false;
    if (p.tailReduced) p.units = CeilDiv(p.outputNum, kOutputGroup);
    else if (!Product(p.outerNum, CeilDiv(p.tailNum, p.tile), p.units)) return false;
    p.cores = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(coreCount, p.units)));
    return true;
}
} // namespace squaresum
