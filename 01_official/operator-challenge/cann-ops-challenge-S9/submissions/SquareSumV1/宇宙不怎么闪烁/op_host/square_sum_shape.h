#pragma once
#include <cstdint>
#include <limits>
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kDimNum = 5U;

static uint32_t DataTypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2U;
        case ge::DT_FLOAT:
            return 4U;
        default:
            return 0U;
    }
}

static bool FillDims(const gert::Shape& shape, uint32_t (&dims)[kDimNum])
{
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        dims[i] = 1U;
    }
    const uint32_t rank = static_cast<uint32_t>(shape.GetDimNum());
    const uint32_t base = kDimNum - rank;
    for (uint32_t i = 0U; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0 || static_cast<uint64_t>(dim) > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        dims[base + i] = static_cast<uint32_t>(dim);
    }
    return true;
}

static bool CalcStrides(const uint32_t (&dims)[kDimNum], uint32_t (&strides)[kDimNum])
{
    uint64_t stride = 1ULL;
    for (int32_t i = static_cast<int32_t>(kDimNum) - 1; i >= 0; --i) {
        if (stride > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        strides[i] = static_cast<uint32_t>(stride);
        stride *= dims[i];
    }
    return stride <= std::numeric_limits<uint32_t>::max();
}

static bool CalcTotal(const uint32_t (&dims)[kDimNum], uint32_t& result)
{
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        if (dims[i] == 0U) {
            result = 0U;
            return true;
        }
    }
    uint64_t total = 1ULL;
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        if (total > std::numeric_limits<uint32_t>::max() / dims[i]) {
            return false;
        }
        total *= dims[i];
    }
    result = static_cast<uint32_t>(total);
    return true;
}

template <typename AttrT>
static bool FillReduceMask(const AttrT* attr, uint32_t rank, uint32_t (&mask)[kDimNum])
{
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        mask[i] = 0U;
    }
    if (attr == nullptr) {
        return false;
    }
    const int64_t* axes = attr->GetData();
    const uint32_t axisNum = static_cast<uint32_t>(attr->GetSize());
    const uint32_t base = kDimNum - rank;
    if (axisNum == 0U) {
        for (uint32_t i = 0U; i < rank; ++i) {
            mask[base + i] = 1U;
        }
        return true;
    }
    for (uint32_t i = 0U; i < axisNum; ++i) {
        int64_t axis = axes[i];
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
            return false;
        }
        const uint32_t maskIndex = base + static_cast<uint32_t>(axis);
        if (mask[maskIndex] != 0U) {
            return false;
        }
        mask[maskIndex] = 1U;
    }
    return true;
}

static bool IsSuffixReduce(const uint32_t (&reduceMask)[kDimNum])
{
    bool seenReduce = false;
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        if (reduceMask[i] != 0U) {
            seenReduce = true;
        } else if (seenReduce) {
            return false;
        }
    }
    return seenReduce;
}

static bool FindLastReduce(const uint32_t (&reduceMask)[kDimNum], uint32_t& lastReduce)
{
    for (int32_t i = static_cast<int32_t>(kDimNum) - 1; i >= 0; --i) {
        if (reduceMask[static_cast<uint32_t>(i)] != 0U) {
            lastReduce = static_cast<uint32_t>(i);
            return true;
        }
    }
    return false;
}

static bool FindPlaneReduce(const uint32_t (&dims)[kDimNum],
                            const uint32_t (&reduceMask)[kDimNum],
                            uint32_t& planeSize, uint32_t& outerOutputSize,
                            uint32_t& suffixReduceSize, uint32_t& prefixReduceSize)
{
    if (reduceMask[kDimNum - 1U] == 0U) {
        return false;
    }

    uint32_t suffixReduceStart = kDimNum - 1U;
    while (suffixReduceStart > 0U && reduceMask[suffixReduceStart - 1U] != 0U) {
        --suffixReduceStart;
    }
    uint32_t planeStart = suffixReduceStart;
    while (planeStart > 0U && reduceMask[planeStart - 1U] == 0U) {
        --planeStart;
    }
    if (planeStart == suffixReduceStart) {
        return false;
    }

    bool hasPrefixReduce = false;
    uint64_t plane = 1ULL;
    uint64_t outerOutput = 1ULL;
    uint64_t prefixReduce = 1ULL;
    uint64_t suffixReduce = 1ULL;
    for (uint32_t i = 0U; i < planeStart; ++i) {
        if (reduceMask[i] != 0U) {
            hasPrefixReduce = true;
            prefixReduce *= dims[i];
        } else {
            outerOutput *= dims[i];
        }
    }
    for (uint32_t i = planeStart; i < suffixReduceStart; ++i) {
        plane *= dims[i];
    }
    for (uint32_t i = suffixReduceStart; i < kDimNum; ++i) {
        suffixReduce *= dims[i];
    }
    if (!hasPrefixReduce || plane == 0ULL || prefixReduce == 0ULL || suffixReduce == 0ULL ||
        plane > std::numeric_limits<uint32_t>::max() ||
        outerOutput > std::numeric_limits<uint32_t>::max() ||
        prefixReduce > std::numeric_limits<uint32_t>::max() ||
        suffixReduce > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    planeSize = static_cast<uint32_t>(plane);
    outerOutputSize = static_cast<uint32_t>(outerOutput);
    prefixReduceSize = static_cast<uint32_t>(prefixReduce);
    suffixReduceSize = static_cast<uint32_t>(suffixReduce);
    return true;
}
}  // namespace
