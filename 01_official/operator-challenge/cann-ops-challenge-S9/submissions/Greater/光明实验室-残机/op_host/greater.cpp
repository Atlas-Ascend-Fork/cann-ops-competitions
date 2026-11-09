#include "greater_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr int64_t kMaxRank = 8;
constexpr int64_t kS9MaxRank = 4;
constexpr int64_t kCanonRank = 4;
constexpr int64_t kKernelTileLength = 1024;
constexpr int64_t kWorkspaceAlignBytes = 512;
constexpr int64_t kNoBroadcastCoreAlignBytes = 512;
constexpr int64_t kModeNoBroadcast = 0;
constexpr int64_t kModeWorkspaceBroadcast = 1;
constexpr int64_t kModeScalarBroadcast = 2;
constexpr int64_t kModeDirectBroadcast = 3;
constexpr int64_t kModeBatchBroadcast = 4;
constexpr uint32_t kGenericFallbackTilingKey = 0;
constexpr uint32_t kBf16RepeatedCacheTilingKey = 1;
constexpr uint32_t kNoBroadcastStaticTilingKey = 2;
constexpr uint32_t kDirectBroadcastStaticTilingKey = 3;
constexpr uint32_t kScalarBroadcastStaticTilingKey = 4;
constexpr uint32_t kBatchBroadcastStaticTilingKey = 5;
constexpr uint32_t kInt32DirectGenericTilingKey = 6;
constexpr uint32_t kTinyFp16NoBroadcastTilingKey = 7;
constexpr uint32_t kInt32CompactStaticTilingKey = 8;
constexpr int64_t kFp32NoBroadcastStaticMinBytes = 64LL << 20;
constexpr int64_t kTinyNoBroadcastMaxBytes = 64LL << 10;
constexpr int64_t kTinyNoBroadcastLaunchBlockDim = 8;
constexpr int64_t kDirectSegmentBaseThreshold = 128;
constexpr int64_t kInt32CompactRows = 8;
constexpr int64_t kInt32CompactBatchRows = 24;
constexpr int64_t kInt32CompactRowElems = 1023;
constexpr int64_t kDirectSegmentLowSizeUnit = 64;
constexpr int64_t kDirect2DSplitBytes = 512;
constexpr int64_t kDirect2DTargetBytesPerCore = 2LL << 20;
constexpr int64_t kFp16GatherTileLength = 8192;
constexpr int64_t kFp16VectorElems = 4096;
constexpr int64_t kBatchBroadcastMinElems = 1 << 20;

constexpr ge::DataType kSupportedTypes[] = {
    ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8};
constexpr ge::Format kSupportedFormats[] = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

int64_t AlignUp(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

int64_t CeilDiv(int64_t value, int64_t align)
{
    return (value + align - 1) / align;
}

int64_t GetDataTypeSize(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_INT8:
            return 1;
        default:
            return 0;
    }
}

struct DirectBroadcastPlan {
    bool valid = false;
    int64_t broadcastInputIndex = 0;
    int64_t lowSize = 1;
    int64_t totalSegments = 0;
    int64_t outerRank = 0;
    int64_t outerDims[kCanonRank] = {1, 1, 1, 1};
    int64_t outStrides[kCanonRank] = {1, 1, 1, 1};
    int64_t srcStrides[kCanonRank] = {0, 0, 0, 0};
};

struct BatchBroadcastPlan {
    bool valid = false;
    int64_t broadcastInputIndex = 0;
    int64_t splitAxis = 0;
    int64_t midFormer = 1;
    int64_t totalTiles = 0;
};

int64_t DecideBlockDim(int64_t mode, int64_t totalElems, int64_t totalBytes, int64_t maxBlockDim,
                       int64_t directTotalSegments = 0)
{
    if (totalElems <= 0 || maxBlockDim <= 1) {
        return 1;
    }

    if (mode == kModeNoBroadcast) {
        if (totalBytes <= kTinyNoBroadcastMaxBytes) {
            return 1;
        }
        return std::max<int64_t>(1, std::min(maxBlockDim, CeilDiv(totalElems, kKernelTileLength)));
    }

    int64_t targetElemsPerCore = kKernelTileLength;
    if (mode == kModeScalarBroadcast) {
        targetElemsPerCore = (totalElems <= 4096) ? 256 : 2048;
    } else {
        if (totalElems <= 2048) {
            targetElemsPerCore = 256;
        } else if (totalElems <= 16384) {
            targetElemsPerCore = 512;
        } else {
            targetElemsPerCore = 2048;
        }
    }

    int64_t blockDim = CeilDiv(totalElems, targetElemsPerCore);
    if (totalBytes <= 1024) {
        blockDim = 1;
    } else if (totalBytes <= 4096) {
        blockDim = std::max<int64_t>(blockDim, 2);
    }
    if ((mode == kModeDirectBroadcast || mode == kModeBatchBroadcast) && directTotalSegments > 0) {
        blockDim = std::min(blockDim, directTotalSegments);
    }
    return std::max<int64_t>(1, std::min(maxBlockDim, blockDim));
}

int64_t GetDimFromRight(const gert::Shape& shape, int64_t alignedIndex, int64_t rank)
{
    const int64_t shapeRank = static_cast<int64_t>(shape.GetDimNum());
    if (alignedIndex + shapeRank < rank) {
        return 1;
    }
    return shape.GetDim(alignedIndex + shapeRank - rank);
}

bool BuildBroadcastPlan(const gert::Shape& x1Shape, const gert::Shape& x2Shape, int64_t (&x1Dims)[kMaxRank],
                        int64_t (&x2Dims)[kMaxRank], int64_t (&dims)[kMaxRank], int64_t (&outStrides)[kMaxRank],
                        int64_t (&x1Strides)[kMaxRank], int64_t (&x2Strides)[kMaxRank], int64_t& rank,
                        int64_t& totalElems, int64_t& x1TotalElems, int64_t& x2TotalElems, bool& noBroadcast)
{
    const int64_t x1Rank = static_cast<int64_t>(x1Shape.GetDimNum());
    const int64_t x2Rank = static_cast<int64_t>(x2Shape.GetDimNum());
    rank = std::max(x1Rank, x2Rank);
    if (rank <= 0 || rank > kS9MaxRank || rank > kMaxRank) {
        return false;
    }

    totalElems = 1;
    x1TotalElems = 1;
    x2TotalElems = 1;
    noBroadcast = true;
    for (int64_t i = 0; i < kMaxRank; ++i) {
        x1Dims[i] = 1;
        x2Dims[i] = 1;
        dims[i] = 1;
        outStrides[i] = 1;
        x1Strides[i] = 0;
        x2Strides[i] = 0;
    }

    for (int64_t i = 0; i < rank; ++i) {
        const int64_t x1Dim = GetDimFromRight(x1Shape, i, rank);
        const int64_t x2Dim = GetDimFromRight(x2Shape, i, rank);
        if (x1Dim <= 0 || x2Dim <= 0) {
            return false;
        }
        if (x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1) {
            return false;
        }
        if (x1Dim != x2Dim) {
            noBroadcast = false;
        }
        x1Dims[i] = x1Dim;
        x2Dims[i] = x2Dim;
        dims[i] = std::max(x1Dim, x2Dim);
        if (totalElems > std::numeric_limits<int64_t>::max() / dims[i]) {
            return false;
        }
        if (x1TotalElems > std::numeric_limits<int64_t>::max() / x1Dim ||
            x2TotalElems > std::numeric_limits<int64_t>::max() / x2Dim) {
            return false;
        }
        totalElems *= dims[i];
        x1TotalElems *= x1Dim;
        x2TotalElems *= x2Dim;
    }

    int64_t outRunning = 1;
    int64_t x1Running = 1;
    int64_t x2Running = 1;
    for (int64_t i = rank - 1; i >= 0; --i) {
        outStrides[i] = outRunning;
        if (x1Dims[i] != 1 || dims[i] == 1) {
            x1Strides[i] = x1Running;
        }
        if (x2Dims[i] != 1 || dims[i] == 1) {
            x2Strides[i] = x2Running;
        }
        outRunning *= dims[i];
        x1Running *= x1Dims[i];
        x2Running *= x2Dims[i];
    }
    return totalElems > 0 && x1TotalElems > 0 && x2TotalElems > 0;
}

std::vector<int64_t> GetCanonDims(const int64_t (&dims)[kMaxRank], int64_t rank)
{
    std::vector<int64_t> canon(kCanonRank, 1);
    for (int64_t i = 0; i < rank; ++i) {
        canon[i] = dims[rank - i - 1];
    }
    while (!canon.empty() && canon.back() == 1) {
        canon.pop_back();
    }
    return canon;
}

void BuildFmaxCanon4D(const int64_t (&x1Dims)[kMaxRank], const int64_t (&x2Dims)[kMaxRank], int64_t rank,
                      int64_t (&canonX1)[kCanonRank], int64_t (&canonX2)[kCanonRank],
                      int64_t (&canonOut)[kCanonRank])
{
    std::vector<int64_t> x1Canon = GetCanonDims(x1Dims, rank);
    std::vector<int64_t> x2Canon = GetCanonDims(x2Dims, rank);
    if (x1Canon.size() != x2Canon.size()) {
        const size_t minSize = std::min(x1Canon.size(), x2Canon.size());
        const size_t maxSize = std::max(x1Canon.size(), x2Canon.size());
        std::vector<int64_t>& maxShape = (x1Canon.size() > x2Canon.size()) ? x1Canon : x2Canon;
        int64_t highDim = 1;
        for (size_t i = minSize; i < maxSize; ++i) {
            highDim *= maxShape[i];
        }
        maxShape.resize(minSize + 1);
        maxShape[minSize] = highDim;
    }

    x1Canon.resize(kCanonRank, 1);
    x2Canon.resize(kCanonRank, 1);
    for (int64_t i = 0; i < kCanonRank; ++i) {
        canonX1[i] = x1Canon[i];
        canonX2[i] = x2Canon[i];
        canonOut[i] = std::max(canonX1[i], canonX2[i]);
    }
}

void SavePlan(optiling::GreaterTilingData& tiling, const int64_t (&dims)[kMaxRank],
              const int64_t (&outStrides)[kMaxRank], const int64_t (&x1Strides)[kMaxRank],
              const int64_t (&x2Strides)[kMaxRank])
{
    tiling.set_dim0(dims[0]); tiling.set_dim1(dims[1]); tiling.set_dim2(dims[2]); tiling.set_dim3(dims[3]);
    tiling.set_dim4(dims[4]); tiling.set_dim5(dims[5]); tiling.set_dim6(dims[6]); tiling.set_dim7(dims[7]);
    tiling.set_out_stride0(outStrides[0]); tiling.set_out_stride1(outStrides[1]);
    tiling.set_out_stride2(outStrides[2]); tiling.set_out_stride3(outStrides[3]);
    tiling.set_out_stride4(outStrides[4]); tiling.set_out_stride5(outStrides[5]);
    tiling.set_out_stride6(outStrides[6]); tiling.set_out_stride7(outStrides[7]);
    tiling.set_x1_stride0(x1Strides[0]); tiling.set_x1_stride1(x1Strides[1]);
    tiling.set_x1_stride2(x1Strides[2]); tiling.set_x1_stride3(x1Strides[3]);
    tiling.set_x1_stride4(x1Strides[4]); tiling.set_x1_stride5(x1Strides[5]);
    tiling.set_x1_stride6(x1Strides[6]); tiling.set_x1_stride7(x1Strides[7]);
    tiling.set_x2_stride0(x2Strides[0]); tiling.set_x2_stride1(x2Strides[1]);
    tiling.set_x2_stride2(x2Strides[2]); tiling.set_x2_stride3(x2Strides[3]);
    tiling.set_x2_stride4(x2Strides[4]); tiling.set_x2_stride5(x2Strides[5]);
    tiling.set_x2_stride6(x2Strides[6]); tiling.set_x2_stride7(x2Strides[7]);
}

void SaveCanon(optiling::GreaterTilingData& tiling, const int64_t (&canonX1)[kCanonRank],
               const int64_t (&canonX2)[kCanonRank], const int64_t (&canonOut)[kCanonRank])
{
    tiling.set_canon_x10(canonX1[0]);
    tiling.set_canon_x11(canonX1[1]);
    tiling.set_canon_x12(canonX1[2]);
    tiling.set_canon_x13(canonX1[3]);
    tiling.set_canon_x20(canonX2[0]);
    tiling.set_canon_x21(canonX2[1]);
    tiling.set_canon_x22(canonX2[2]);
    tiling.set_canon_x23(canonX2[3]);
    tiling.set_canon_out0(canonOut[0]);
    tiling.set_canon_out1(canonOut[1]);
    tiling.set_canon_out2(canonOut[2]);
    tiling.set_canon_out3(canonOut[3]);
}

void SaveDirectPlan(optiling::GreaterTilingData& tiling, const DirectBroadcastPlan& plan)
{
    tiling.set_direct_broadcast_input_index(plan.broadcastInputIndex);
    tiling.set_direct_low_size(plan.lowSize);
    tiling.set_direct_total_segments(plan.totalSegments);
    tiling.set_direct_outer_rank(plan.outerRank);
    tiling.set_direct_outer_dim0(plan.outerDims[0]);
    tiling.set_direct_outer_dim1(plan.outerDims[1]);
    tiling.set_direct_outer_dim2(plan.outerDims[2]);
    tiling.set_direct_outer_dim3(plan.outerDims[3]);
    tiling.set_direct_out_stride0(plan.outStrides[0]);
    tiling.set_direct_out_stride1(plan.outStrides[1]);
    tiling.set_direct_out_stride2(plan.outStrides[2]);
    tiling.set_direct_out_stride3(plan.outStrides[3]);
    tiling.set_direct_src_stride0(plan.srcStrides[0]);
    tiling.set_direct_src_stride1(plan.srcStrides[1]);
    tiling.set_direct_src_stride2(plan.srcStrides[2]);
    tiling.set_direct_src_stride3(plan.srcStrides[3]);
}

bool BuildDirectBroadcastPlan(ge::DataType dt, int64_t elemSize, const int64_t (&x1Dims)[kMaxRank],
                              const int64_t (&x2Dims)[kMaxRank], const int64_t (&dims)[kMaxRank], int64_t rank,
                              int64_t totalElems, int64_t x1TotalElems, int64_t x2TotalElems,
                              DirectBroadcastPlan& plan)
{
    if (rank <= 0 || rank > kS9MaxRank) {
        return false;
    }
    if (x1TotalElems <= 1 || x2TotalElems <= 1) {
        return false;
    }

    const int64_t* bcastDims = nullptr;
    if (x1TotalElems != totalElems && x2TotalElems == totalElems) {
        plan.broadcastInputIndex = 1;
        bcastDims = x1Dims;
    } else if (x2TotalElems != totalElems && x1TotalElems == totalElems) {
        plan.broadcastInputIndex = 2;
        bcastDims = x2Dims;
    } else {
        return false;
    }

    int64_t broadcastAxisCount = 0;
    for (int64_t i = 0; i < rank; ++i) {
        if (bcastDims[i] != dims[i]) {
            ++broadcastAxisCount;
        }
    }
    if (broadcastAxisCount <= 0) {
        return false;
    }

    int64_t lowSize = 1;
    int64_t suffixLen = 0;
    for (int64_t i = rank - 1; i >= 0; --i) {
        if (bcastDims[i] != dims[i]) {
            break;
        }
        if (lowSize > std::numeric_limits<int64_t>::max() / dims[i]) {
            return false;
        }
        lowSize *= dims[i];
        ++suffixLen;
    }
    if (suffixLen <= 0 || lowSize <= 0) {
        return false;
    }
    if (lowSize * elemSize < 32) {
        return false;
    }
    if (totalElems % lowSize != 0) {
        return false;
    }

    const int64_t outerRank = rank - suffixLen;
    if (outerRank <= 0 || outerRank > kCanonRank) {
        return false;
    }

    plan.lowSize = lowSize;
    plan.totalSegments = totalElems / lowSize;
    plan.outerRank = outerRank;
    if (broadcastAxisCount > 1) {
        const int64_t segmentScale = CeilDiv(lowSize, kDirectSegmentLowSizeUnit);
        const int64_t segmentThreshold =
            (segmentScale > std::numeric_limits<int64_t>::max() / kDirectSegmentBaseThreshold)
                ? std::numeric_limits<int64_t>::max()
                : segmentScale * kDirectSegmentBaseThreshold;
        if (plan.totalSegments >= segmentThreshold) {
            return false;
        }
    }

    int64_t outStride = 1;
    int64_t srcStride = 1;
    for (int64_t i = kCanonRank - 1; i >= 0; --i) {
        plan.outerDims[i] = 1;
        plan.outStrides[i] = 1;
        plan.srcStrides[i] = 0;
    }
    for (int64_t i = outerRank - 1; i >= 0; --i) {
        plan.outerDims[i] = dims[i];
        plan.outStrides[i] = outStride;
        if (outStride > std::numeric_limits<int64_t>::max() / dims[i]) {
            return false;
        }
        outStride *= dims[i];

        if (bcastDims[i] != 1) {
            plan.srcStrides[i] = srcStride;
            if (srcStride > std::numeric_limits<int64_t>::max() / bcastDims[i]) {
                return false;
            }
            srcStride *= bcastDims[i];
        }
    }
    plan.valid = plan.totalSegments > 0;
    return plan.valid;
}

bool BuildBatchBroadcastPlan(ge::DataType dt, const int64_t (&x1Dims)[kMaxRank],
                             const int64_t (&x2Dims)[kMaxRank], const int64_t (&dims)[kMaxRank],
                             const int64_t (&outStrides)[kMaxRank], int64_t rank, int64_t totalElems,
                             int64_t x1TotalElems, int64_t x2TotalElems, BatchBroadcastPlan& plan)
{
    if (dt != ge::DT_FLOAT16 || rank <= 0 || rank > kS9MaxRank || totalElems < kBatchBroadcastMinElems) {
        return false;
    }
    const int64_t* broadcastDims = nullptr;
    if (x1TotalElems != totalElems && x2TotalElems == totalElems) {
        plan.broadcastInputIndex = 1;
        broadcastDims = x1Dims;
    } else if (x2TotalElems != totalElems && x1TotalElems == totalElems) {
        plan.broadcastInputIndex = 2;
        broadcastDims = x2Dims;
    } else {
        return false;
    }

    int64_t splitAxis = -1;
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
        if (broadcastDims[axis] != dims[axis]) {
            splitAxis = axis;
            break;
        }
    }
    if (splitAxis < 0) {
        return false;
    }

    const int64_t patternElems = outStrides[splitAxis];
    if (patternElems <= 0 || patternElems > kFp16VectorElems) {
        return false;
    }
    const int64_t rowStrideElems = AlignUp(patternElems * static_cast<int64_t>(sizeof(uint16_t)), 32) /
                                   static_cast<int64_t>(sizeof(uint16_t));
    const int64_t midFormer = std::min(kFp16GatherTileLength / rowStrideElems, dims[splitAxis]);
    if (midFormer <= 1) {
        return false;
    }

    int64_t outerTiles = 1;
    for (int64_t i = 0; i < splitAxis; ++i) {
        if (outerTiles > std::numeric_limits<int64_t>::max() / dims[i]) {
            return false;
        }
        outerTiles *= dims[i];
    }
    const int64_t splitTiles = CeilDiv(dims[splitAxis], midFormer);
    if (outerTiles > std::numeric_limits<int64_t>::max() / splitTiles) {
        return false;
    }

    plan.splitAxis = splitAxis;
    plan.midFormer = midFormer;
    plan.totalTiles = outerTiles * splitTiles;
    plan.valid = plan.totalTiles > 0;
    return plan.valid;
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GreaterTilingData tiling;

    if (context == nullptr || context->GetInputTensor(0) == nullptr || context->GetInputTensor(1) == nullptr ||
        context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dt = context->GetInputTensor(0)->GetDataType();
    const int64_t elemSize = GetDataTypeSize(dt);
    if (dt != context->GetInputTensor(1)->GetDataType() || elemSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& x1Shape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape& x2Shape = context->GetInputShape(1)->GetStorageShape();
    int64_t x1Dims[kMaxRank] = {};
    int64_t x2Dims[kMaxRank] = {};
    int64_t dims[kMaxRank] = {};
    int64_t outStrides[kMaxRank] = {};
    int64_t x1Strides[kMaxRank] = {};
    int64_t x2Strides[kMaxRank] = {};
    int64_t rank = 0;
    int64_t totalElems = 0;
    int64_t x1TotalElems = 0;
    int64_t x2TotalElems = 0;
    bool noBroadcast = false;
    if (!BuildBroadcastPlan(x1Shape, x2Shape, x1Dims, x2Dims, dims, outStrides, x1Strides, x2Strides, rank,
                            totalElems, x1TotalElems, x2TotalElems, noBroadcast)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t totalBytes = totalElems * elemSize;

    int64_t mode = kModeNoBroadcast;
    int64_t scalarInputIndex = 0;
    if (!noBroadcast) {
        if (x1TotalElems == 1 && x2TotalElems > 1) {
            mode = kModeScalarBroadcast;
            scalarInputIndex = 1;
        } else if (x2TotalElems == 1 && x1TotalElems > 1) {
            mode = kModeScalarBroadcast;
            scalarInputIndex = 2;
        } else {
            mode = kModeWorkspaceBroadcast;
        }
    }

    DirectBroadcastPlan directPlan;
    BatchBroadcastPlan batchPlan;
    if (mode == kModeWorkspaceBroadcast) {
        if (BuildDirectBroadcastPlan(dt, elemSize, x1Dims, x2Dims, dims, rank, totalElems, x1TotalElems, x2TotalElems,
                                     directPlan)) {
            mode = kModeDirectBroadcast;
        } else if (BuildBatchBroadcastPlan(dt, x1Dims, x2Dims, dims, outStrides, rank, totalElems,
                                           x1TotalElems, x2TotalElems, batchPlan)) {
            mode = kModeBatchBroadcast;
        }
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t maxBlockDim = static_cast<int64_t>(std::max(1U, ascendcPlatform.GetCoreNumAiv()));
    int64_t workItems = batchPlan.totalTiles;
    if (mode == kModeDirectBroadcast) {
        workItems = directPlan.totalSegments;
        const int64_t lowBytes = directPlan.lowSize * elemSize;
        const int64_t lowSlices = CeilDiv(lowBytes, kDirect2DSplitBytes);
        const int64_t maxDirectWorkItems =
            (directPlan.totalSegments > std::numeric_limits<int64_t>::max() / lowSlices)
                ? std::numeric_limits<int64_t>::max()
                : directPlan.totalSegments * lowSlices;
        const int64_t targetCores = CeilDiv(totalBytes, kDirect2DTargetBytesPerCore);
        workItems = std::min(maxDirectWorkItems, std::max(directPlan.totalSegments, targetCores));
    }
    const bool useTinyFp16NoBroadcastSingleWorker =
        dt == ge::DT_FLOAT16 && mode == kModeNoBroadcast &&
        totalBytes <= kTinyNoBroadcastMaxBytes;
    const int64_t decidedBlockDim =
        DecideBlockDim(mode, totalElems, totalBytes, maxBlockDim, workItems);
    const int64_t blockDim = useTinyFp16NoBroadcastSingleWorker
                                 ? std::min(maxBlockDim, kTinyNoBroadcastLaunchBlockDim)
                                 : decidedBlockDim;
    int64_t elemsPerCore = useTinyFp16NoBroadcastSingleWorker
                               ? totalElems
                               : CeilDiv(totalElems, blockDim);
    if (mode == kModeNoBroadcast) {
        elemsPerCore = AlignUp(elemsPerCore, kNoBroadcastCoreAlignBytes / elemSize);
    }

    tiling.set_total_elems(totalElems);
    tiling.set_x1_total_elems(x1TotalElems);
    tiling.set_x2_total_elems(x2TotalElems);
    tiling.set_elems_per_core(elemsPerCore);
    tiling.set_rank(rank);
    tiling.set_mode(mode);
    tiling.set_scalar_input_index(scalarInputIndex);
    tiling.set_direct_broadcast_input_index(0);
    tiling.set_elem_size(elemSize);
    tiling.set_x1_workspace_offset(-1);
    tiling.set_x2_workspace_offset(-1);
    tiling.set_scratch_workspace_offset(-1);
    tiling.set_direct_low_size(1);
    tiling.set_direct_total_segments(0);
    tiling.set_direct_outer_rank(0);
    tiling.set_direct_outer_dim0(1); tiling.set_direct_outer_dim1(1);
    tiling.set_direct_outer_dim2(1); tiling.set_direct_outer_dim3(1);
    tiling.set_direct_out_stride0(1); tiling.set_direct_out_stride1(1);
    tiling.set_direct_out_stride2(1); tiling.set_direct_out_stride3(1);
    tiling.set_direct_src_stride0(0); tiling.set_direct_src_stride1(0);
    tiling.set_direct_src_stride2(0); tiling.set_direct_src_stride3(0);
    tiling.set_batch_split_axis(0);
    tiling.set_batch_mid_former(1);
    tiling.set_batch_total_tiles(0);
    tiling.set_batch_pattern_reciprocal(1.0f);
    tiling.set_canon_x10(1); tiling.set_canon_x11(1); tiling.set_canon_x12(1); tiling.set_canon_x13(1);
    tiling.set_canon_x20(1); tiling.set_canon_x21(1); tiling.set_canon_x22(1); tiling.set_canon_x23(1);
    tiling.set_canon_out0(1); tiling.set_canon_out1(1); tiling.set_canon_out2(1); tiling.set_canon_out3(1);
    SavePlan(tiling, dims, outStrides, x1Strides, x2Strides);

    if (mode == kModeDirectBroadcast) {
        SaveDirectPlan(tiling, directPlan);
    } else if (mode == kModeBatchBroadcast) {
        tiling.set_direct_broadcast_input_index(batchPlan.broadcastInputIndex);
        tiling.set_batch_split_axis(batchPlan.splitAxis);
        tiling.set_batch_mid_former(batchPlan.midFormer);
        tiling.set_batch_total_tiles(batchPlan.totalTiles);
        tiling.set_batch_pattern_reciprocal(
            1.0f / static_cast<float>(outStrides[batchPlan.splitAxis]));
    } else if (mode == kModeWorkspaceBroadcast) {
        int64_t canonX1[kCanonRank] = {1, 1, 1, 1};
        int64_t canonX2[kCanonRank] = {1, 1, 1, 1};
        int64_t canonOut[kCanonRank] = {1, 1, 1, 1};
        BuildFmaxCanon4D(x1Dims, x2Dims, rank, canonX1, canonX2, canonOut);
        SaveCanon(tiling, canonX1, canonX2, canonOut);

        int64_t nextOffset = 0;
        const int64_t oneTensorBytes = AlignUp(totalBytes, kWorkspaceAlignBytes);
        if (x1TotalElems != totalElems) {
            tiling.set_x1_workspace_offset(nextOffset);
            nextOffset += oneTensorBytes;
        }
        if (x2TotalElems != totalElems) {
            tiling.set_x2_workspace_offset(nextOffset);
            nextOffset += oneTensorBytes;
        }
        tiling.set_scratch_workspace_offset(nextOffset);
        nextOffset += oneTensorBytes;

        size_t* workspaceSizes = context->GetWorkspaceSizes(1);
        if (workspaceSizes != nullptr) {
            workspaceSizes[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize() + nextOffset);
        }
    }

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    const bool useBf16RepeatedCache =
        dt == ge::DT_BF16 && mode == kModeDirectBroadcast && directPlan.outerRank > 0 &&
        directPlan.outerDims[directPlan.outerRank - 1] > 1 &&
        directPlan.srcStrides[directPlan.outerRank - 1] == 0;
    const int64_t directInnerRepeat =
        directPlan.outerRank > 0 ? directPlan.outerDims[directPlan.outerRank - 1] : 0;
    const bool useInt32CompactStatic =
        dt == ge::DT_INT32 && mode == kModeDirectBroadcast &&
        directPlan.totalSegments > kDirectSegmentBaseThreshold &&
        directPlan.lowSize == kInt32CompactRowElems &&
        directPlan.totalSegments % (blockDim * kInt32CompactBatchRows) == 0 &&
        directInnerRepeat >= kInt32CompactBatchRows &&
        directInnerRepeat % kInt32CompactBatchRows == 0 &&
        directPlan.srcStrides[directPlan.outerRank - 1] == 0;
    uint32_t tilingKey = kGenericFallbackTilingKey;
    if (useTinyFp16NoBroadcastSingleWorker) {
        tilingKey = kTinyFp16NoBroadcastTilingKey;
    } else if (useBf16RepeatedCache) {
        tilingKey = kBf16RepeatedCacheTilingKey;
    } else if (mode == kModeNoBroadcast) {
        const bool useStatic = dt != ge::DT_FLOAT || totalBytes >= kFp32NoBroadcastStaticMinBytes;
        tilingKey = useStatic ? kNoBroadcastStaticTilingKey : kGenericFallbackTilingKey;
    } else if (useInt32CompactStatic) {
        tilingKey = kInt32CompactStaticTilingKey;
    } else if (dt == ge::DT_INT32 && mode == kModeDirectBroadcast &&
               directPlan.totalSegments > kDirectSegmentBaseThreshold) {
        tilingKey = kInt32DirectGenericTilingKey;
    } else if (mode == kModeDirectBroadcast) {
        tilingKey = kDirectBroadcastStaticTilingKey;
    } else if (mode == kModeScalarBroadcast) {
        tilingKey = kScalarBroadcastStaticTilingKey;
    } else if (mode == kModeBatchBroadcast) {
        // The static path pads each FP16 row but does not compact the uint8 output rows.
        // Keep large multi-axis broadcasts on the generic path until that layout is fixed.
        tilingKey = kGenericFallbackTilingKey;
    }
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (x1Shape == nullptr || x2Shape == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t x1Dims[kMaxRank] = {};
    int64_t x2Dims[kMaxRank] = {};
    int64_t dims[kMaxRank] = {};
    int64_t outStrides[kMaxRank] = {};
    int64_t x1Strides[kMaxRank] = {};
    int64_t x2Strides[kMaxRank] = {};
    int64_t rank = 0;
    int64_t totalElems = 0;
    int64_t x1TotalElems = 0;
    int64_t x2TotalElems = 0;
    bool noBroadcast = false;
    if (!BuildBroadcastPlan(*x1Shape, *x2Shape, x1Dims, x2Dims, dims, outStrides, x1Strides, x2Strides, rank,
                            totalElems, x1TotalElems, x2TotalElems, noBroadcast)) {
        return ge::GRAPH_FAILED;
    }

    outputShape->SetDimNum(0);
    for (int64_t i = 0; i < rank; ++i) {
        outputShape->AppendDim(dims[i]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context->GetInputDataType(0) != context->GetInputDataType(1)) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({kSupportedTypes[0], kSupportedTypes[1], kSupportedTypes[2], kSupportedTypes[3],
                       kSupportedTypes[4]})
            .Format({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                     kSupportedFormats[4]})
            .UnknownShapeFormat({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2],
                                 kSupportedFormats[3], kSupportedFormats[4]});

        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({kSupportedTypes[0], kSupportedTypes[1], kSupportedTypes[2], kSupportedTypes[3],
                       kSupportedTypes[4]})
            .Format({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2], kSupportedFormats[3],
                     kSupportedFormats[4]})
            .UnknownShapeFormat({kSupportedFormats[0], kSupportedFormats[1], kSupportedFormats[2],
                                 kSupportedFormats[3], kSupportedFormats[4]});

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Greater);

}  // namespace ops
