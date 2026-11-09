#include "greater_tiling.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t kMaxRank = 8;
constexpr uint32_t kMaxCoreNum = 40;
constexpr uint32_t kMinElementsPerCore = 4096;
constexpr uint32_t kDefaultElementsPerCore = 8192;
constexpr uint32_t kAlignBytes = 32;
constexpr uint64_t kWorkspaceAlign = 512;
constexpr uint32_t kLinearKey = 0;
constexpr uint32_t kSingleAxisKey = 1;
constexpr uint32_t kWorkspaceKey = 2;
constexpr uint32_t kFullMiddleKey = 3;
constexpr uint32_t kOuterMiddleKey = 4;
constexpr uint32_t kCachedBroadcastKey = 5;
constexpr uint32_t kCachedInnerElements = 4096;
constexpr uint64_t kLargeStreamBytes = 1024U * 1024U;
constexpr uint32_t kLargeLinearCoreCap = 36;
constexpr uint32_t kBf16BatchedCoreCap = 37;
constexpr uint32_t kBf16RowsPerTile = 4;
constexpr uint64_t kBf16UbBudget = 184U * 1024U;

struct Axis {
    uint32_t x1;
    uint32_t x2;
    uint32_t out;
    uint8_t pattern;
};

struct BroadcastPlan {
    gert::Shape outputShape;
    uint32_t rank = 1;
    uint32_t totalLength = 1;
    uint32_t broadcastAxes = 0;
    uint32_t x1BroadcastAxes = 0;
    uint32_t x2BroadcastAxes = 0;
    uint32_t x1Dims[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t x2Dims[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
    uint32_t outDims[kMaxRank] = {1, 1, 1, 1, 1, 1, 1, 1};
};

uint32_t DataWidth(ge::DataType dtype)
{
    if (dtype == ge::DT_INT8) return 1U;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) return 2U;
    if (dtype == ge::DT_FLOAT || dtype == ge::DT_INT32) return 4U;
    return 0U;
}

uint64_t AlignUp(uint64_t value, uint64_t align)
{ return (value + align - 1U) / align * align; }

int64_t RightAlignedDim(const gert::Shape& shape, uint32_t axis, uint32_t rank)
{
    const uint32_t shapeRank = static_cast<uint32_t>(shape.GetDimNum());
    return axis + shapeRank < rank ? 1 : shape.GetDim(axis + shapeRank - rank);
}

bool PreparePlan(const gert::Shape& x1Shape, const gert::Shape& x2Shape, BroadcastPlan& plan)
{
    const uint32_t mergedRank = std::max(static_cast<uint32_t>(x1Shape.GetDimNum()),
                                         static_cast<uint32_t>(x2Shape.GetDimNum()));
    uint64_t total = 1;
    std::vector<Axis> axes;
    axes.reserve(std::min(mergedRank, kMaxRank));

    for (uint32_t axis = 0; axis < mergedRank; ++axis) {
        const int64_t a64 = RightAlignedDim(x1Shape, axis, mergedRank);
        const int64_t b64 = RightAlignedDim(x2Shape, axis, mergedRank);
        if (a64 < 0 || b64 < 0 || a64 > UINT32_MAX || b64 > UINT32_MAX ||
            (a64 != b64 && a64 != 1 && b64 != 1)) {
            return false;
        }
        const uint32_t a = static_cast<uint32_t>(a64);
        const uint32_t b = static_cast<uint32_t>(b64);
        const uint32_t out = a == b ? a : a == 1U ? b : a;
        const uint8_t pattern = static_cast<uint8_t>(a == out) |
                                static_cast<uint8_t>(static_cast<uint8_t>(b == out) << 1U);
        plan.outputShape.AppendDim(out);
        total *= out;
        if (total > UINT32_MAX) {
            return false;
        }
        if (!axes.empty() && axes.back().pattern == pattern) {
            axes.back().x1 *= a;
            axes.back().x2 *= b;
            axes.back().out *= out;
        } else {
            if (axes.size() == kMaxRank) {
                return false;
            }
            axes.push_back({a, b, out, pattern});
        }
    }

    plan.rank = std::max<uint32_t>(1U, static_cast<uint32_t>(axes.size()));
    plan.totalLength = static_cast<uint32_t>(total);
    for (uint32_t i = 0; i < axes.size(); ++i) {
        const Axis& axis = axes[axes.size() - i - 1U];
        plan.x1Dims[i] = axis.x1;
        plan.x2Dims[i] = axis.x2;
        plan.outDims[i] = axis.out;
        if (axis.pattern != 3U) {
            ++plan.broadcastAxes;
            plan.x1BroadcastAxes += axis.x1 != axis.out;
            plan.x2BroadcastAxes += axis.x2 != axis.out;
        }
    }
    return true;
}

uint32_t Product(const uint32_t* dims, uint32_t begin, uint32_t end)
{
    uint64_t result = 1;
    for (uint32_t i = begin; i < end; ++i) {
        result *= dims[i];
    }
    return static_cast<uint32_t>(result);
}

bool IsFull(const uint32_t* dims, const BroadcastPlan& plan)
{
    for (uint32_t i = 0; i < plan.rank; ++i) {
        if (dims[i] != plan.outDims[i]) return false;
    }
    return true;
}

bool IsMiddleOnly(const uint32_t* dims, const BroadcastPlan& plan)
{
    return plan.rank == 3U && dims[0] == 1U && dims[1] == plan.outDims[1] && dims[2] == 1U;
}

bool IsOuterTail(const uint32_t* dims, const BroadcastPlan& plan)
{
    return plan.rank == 3U && dims[0] == plan.outDims[0] && dims[1] == 1U &&
           dims[2] == plan.outDims[2];
}

bool IsOddAxes(const uint32_t* dims, const BroadcastPlan& plan)
{
    if (dims[0] != 1U || dims[1] != plan.outDims[1]) return false;
    return plan.rank == 2U ||
           (plan.rank == 4U && dims[2] == 1U && dims[3] == plan.outDims[3]);
}

bool IsEvenAxes(const uint32_t* dims, const BroadcastPlan& plan)
{
    if (dims[0] != plan.outDims[0] || dims[1] != 1U) return false;
    return plan.rank == 2U ||
           (plan.rank == 4U && dims[2] == plan.outDims[2] && dims[3] == 1U);
}

uint32_t Gcd(uint32_t a, uint32_t b)
{
    while (b != 0U) {
        const uint32_t next = a % b;
        a = b;
        b = next;
    }
    return a;
}

bool RowsMeetMinimum(uint32_t rows, uint32_t width, uint32_t cores)
{
    const uint32_t rowGroup = kAlignBytes / Gcd(width, kAlignBytes);
    const uint32_t rowsPerCore = static_cast<uint32_t>(
        AlignUp((rows + cores - 1U) / cores, rowGroup));
    const uint64_t assignedRows = static_cast<uint64_t>(rowsPerCore) * (cores - 1U);
    const uint64_t lastRows = rows > assignedRows ? rows - assignedRows : 0U;
    return lastRows * width >= kMinElementsPerCore;
}

bool OuterMeetMinimum(uint32_t outer, uint64_t rowElements, uint32_t cores)
{
    return static_cast<uint64_t>(outer / cores) * rowElements >= kMinElementsPerCore;
}

uint64_t CheckerboardPrefix(const BroadcastPlan& plan, uint64_t tasks)
{
    constexpr uint32_t tile = 64U;
    const uint32_t cols = plan.outDims[0];
    const uint32_t middle = plan.outDims[1];
    const uint64_t colTiles = (cols + tile - 1U) / tile;
    const uint64_t middleTiles = (middle + tile - 1U) / tile;
    const uint64_t tasksPerPair = middleTiles * colTiles;
    const uint64_t pairs = tasks / tasksPerPair;
    const uint64_t remain = tasks % tasksPerPair;
    const uint64_t middleTile = remain / colTiles;
    const uint64_t colTileCount = remain % colTiles;
    const uint64_t completedRows = std::min<uint64_t>(middle, middleTile * tile);
    const uint64_t currentRows = std::min<uint64_t>(tile, middle - completedRows);
    const uint64_t completedCols = std::min<uint64_t>(cols, colTileCount * tile);
    return pairs * middle * cols + completedRows * cols + currentRows * completedCols;
}

bool SplitMeetsMinimum(const BroadcastPlan& plan, uint32_t key,
                       uint32_t low, uint32_t middle, uint32_t high, uint32_t cores)
{
    if (cores == 1U || plan.totalLength < kMinElementsPerCore) return true;
    if (key == kCachedBroadcastKey) return cores <= high;
    if (key == kSingleAxisKey) {
        return RowsMeetMinimum(high * middle, low, cores);
    }
    if (key == kFullMiddleKey) {
        return OuterMeetMinimum(high, static_cast<uint64_t>(middle) * low, cores) &&
               RowsMeetMinimum(high * middle, low, cores);
    }
    if (key != kOuterMiddleKey) return true;
    if (plan.rank == 3U) {
        return OuterMeetMinimum(high, static_cast<uint64_t>(middle) * low, cores);
    }

    constexpr uint32_t tile = 64U;
    const uint64_t pairCount = static_cast<uint64_t>(plan.outDims[2]) * plan.outDims[3];
    const uint64_t tasksPerPair =
        ((plan.outDims[1] + tile - 1U) / tile) *
        ((plan.outDims[0] + tile - 1U) / tile);
    const uint64_t taskCount = pairCount * tasksPerPair;
    for (uint32_t core = 0; core < cores; ++core) {
        const uint64_t begin = taskCount * core / cores;
        const uint64_t end = taskCount * (core + 1U) / cores;
        if (CheckerboardPrefix(plan, end) - CheckerboardPrefix(plan, begin) <
            kMinElementsPerCore) return false;
    }
    return true;
}

bool UseBf16BatchedRows(ge::DataType dtype, uint32_t key, const BroadcastPlan& plan,
                        uint32_t low, uint32_t middle, uint32_t high)
{
    if (dtype != ge::DT_BF16 || key != kSingleAxisKey || low == 0U || high == 0U ||
        middle < kBf16RowsPerTile || middle % kBf16RowsPerTile != 0U ||
        static_cast<uint64_t>(low) * sizeof(uint16_t) % kAlignBytes == 0U ||
        static_cast<uint64_t>(plan.totalLength) * sizeof(uint16_t) <= kLargeStreamBytes) {
        return false;
    }
    const uint64_t tileLength = static_cast<uint64_t>(kBf16RowsPerTile) * low;
    const uint64_t fullStorage = AlignUp(tileLength, 256U / sizeof(uint16_t));
    const uint64_t compareLength = AlignUp(tileLength, 256U / sizeof(float));
    const uint64_t broadcastStorage = AlignUp(low, 256U / sizeof(uint16_t));
    const uint64_t maskBytes = AlignUp(compareLength, 256U) / 8U;
    const uint64_t ubBytes = 12U * fullStorage + 8U * compareLength +
                             4U * broadcastStorage + maskBytes + 768U;
    return ubBytes <= kBf16UbBudget;
}

}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto dtype = context->GetInputTensor(0)->GetDataType();
    if (dtype != context->GetInputTensor(1)->GetDataType()) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t dataWidth = DataWidth(dtype);
    const auto* x1Storage = context->GetInputShape(0);
    const auto* x2Storage = context->GetInputShape(1);
    if (dataWidth == 0 || x1Storage == nullptr || x2Storage == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x1Shape = x1Storage->GetStorageShape();
    const auto& x2Shape = x2Storage->GetStorageShape();
    BroadcastPlan plan;
    if (!PreparePlan(x1Shape, x2Shape, plan)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t key = kLinearKey;
    uint32_t lowSize = plan.totalLength;
    uint32_t middleSize = 1;
    uint32_t highSize = 1;
    uint32_t broadcastIsX1 = 0;
    int64_t offsets[3] = {-1, -1, -1};

    const bool x1Full = IsFull(plan.x1Dims, plan);
    const bool x2Full = IsFull(plan.x2Dims, plan);
    const bool x1Middle = IsMiddleOnly(plan.x1Dims, plan);
    const bool x2Middle = IsMiddleOnly(plan.x2Dims, plan);
    const bool x1OuterTail = IsOuterTail(plan.x1Dims, plan);
    const bool x2OuterTail = IsOuterTail(plan.x2Dims, plan);
    const bool x1Odd = IsOddAxes(plan.x1Dims, plan);
    const bool x2Odd = IsOddAxes(plan.x2Dims, plan);
    const bool x1Even = IsEvenAxes(plan.x1Dims, plan);
    const bool x2Even = IsEvenAxes(plan.x2Dims, plan);
    if (plan.totalLength != 0 && plan.rank == 3U &&
        ((x1Full && x2Middle) || (x2Full && x1Middle))) {
        key = kFullMiddleKey;
        lowSize = plan.outDims[0];
        middleSize = plan.outDims[1];
        highSize = plan.outDims[2];
        broadcastIsX1 = x1Middle;
    } else if (plan.totalLength != 0 && plan.rank == 3U &&
               ((x1Middle && x2OuterTail) || (x2Middle && x1OuterTail))) {
        key = kOuterMiddleKey;
        lowSize = plan.outDims[0];
        middleSize = plan.outDims[1];
        highSize = plan.outDims[2];
        broadcastIsX1 = x1Middle;
    } else if (plan.totalLength != 0 &&
               (plan.rank == 2U ||
                (static_cast<uint64_t>(plan.outDims[1]) * plan.outDims[0]) % kAlignBytes == 0U) &&
               ((x1Odd && x2Even) || (x2Odd && x1Even))) {
        key = kOuterMiddleKey;
        broadcastIsX1 = x1Odd;
    } else if (plan.totalLength != 0 && plan.broadcastAxes == 1) {
        uint32_t axis = 0;
        while (plan.x1Dims[axis] == plan.x2Dims[axis]) {
            ++axis;
        }
        lowSize = Product(plan.outDims, 0, axis);
        middleSize = plan.outDims[axis];
        highSize = Product(plan.outDims, axis + 1U, plan.rank);
        broadcastIsX1 = plan.x1Dims[axis] == 1U;
        key = kSingleAxisKey;
    } else if (plan.totalLength != 0 && plan.broadcastAxes > 1) {
        key = kWorkspaceKey;
        uint32_t innerRank = 0;
        uint32_t innerLength = 1;
        while (innerRank < plan.rank &&
               static_cast<uint64_t>(innerLength) * plan.outDims[innerRank] <=
                   kCachedInnerElements) {
            innerLength *= plan.outDims[innerRank++];
        }
        const uint32_t x1Elements = Product(plan.x1Dims, 0, plan.rank);
        const uint32_t x2Elements = Product(plan.x2Dims, 0, plan.rank);
        const uint32_t compareWidth = dtype == ge::DT_BF16 ? 4U :
                                      dtype == ge::DT_INT8 ? 2U : dataWidth;
        const uint32_t cacheWidth = dataWidth == compareWidth ? dataWidth :
                                    dataWidth + compareWidth;
        const uint64_t cacheBytes =
            (static_cast<uint64_t>(x1Elements) + x2Elements) * cacheWidth;
        const uint32_t cacheLimit = dtype == ge::DT_INT32 ? 32U * 1024U : 48U * 1024U;
        if (innerRank != 0U && innerLength >= 256U && cacheBytes <= cacheLimit) {
            key = kCachedBroadcastKey;
            lowSize = innerLength;
            middleSize = innerRank;
            highSize = plan.totalLength / innerLength;
        }
    }
    const bool useBf16BatchedRows =
        UseBf16BatchedRows(dtype, key, plan, lowSize, middleSize, highSize);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    if (key == kWorkspaceKey) {
        const uint64_t singleBytes = AlignUp(static_cast<uint64_t>(plan.totalLength) * dataWidth,
                                             kWorkspaceAlign);
        uint64_t nextOffset = 0;
        if (plan.x1BroadcastAxes != 0) {
            offsets[0] = static_cast<int64_t>(nextOffset);
            nextOffset += singleBytes;
        }
        if (plan.x2BroadcastAxes != 0) {
            offsets[1] = static_cast<int64_t>(nextOffset);
            nextOffset += singleBytes;
        }
        if (std::max(plan.x1BroadcastAxes, plan.x2BroadcastAxes) > 1U) {
            offsets[2] = static_cast<int64_t>(nextOffset);
            nextOffset += singleBytes;
        }
        auto workspaceSizes = context->GetWorkspaceSizes(1);
        workspaceSizes[0] = platform.GetLibApiWorkSpaceSize() + nextOffset;
    }

    GreaterTilingData tiling;
    float inverseStrides[kMaxRank] = {};
    float inversePeriods[kMaxRank] = {};
    uint64_t outputStride = 1U;
    for (uint32_t axis = 0; axis < plan.rank; ++axis) {
        const float inverse = 1.0f / static_cast<float>(outputStride);
        inverseStrides[axis] = (key == kSingleAxisKey && outputStride == lowSize) ||
                                       key == kCachedBroadcastKey ?
            std::nextafter(inverse, std::numeric_limits<float>::infinity()) : inverse;
        outputStride *= plan.outDims[axis];
        const float inversePeriod = 1.0f / static_cast<float>(outputStride);
        inversePeriods[axis] = (key == kCachedBroadcastKey || key == kOuterMiddleKey) ?
            std::nextafter(inversePeriod, std::numeric_limits<float>::infinity()) : inversePeriod;
    }
    tiling.set_totalLength(plan.totalLength);
    tiling.set_lowSize(lowSize);
    tiling.set_middleSize(middleSize);
    tiling.set_highSize(highSize);
    tiling.set_rank(plan.rank);
    tiling.set_broadcastIsX1(broadcastIsX1);
    tiling.set_x1Dims(plan.x1Dims);
    tiling.set_x2Dims(plan.x2Dims);
    tiling.set_outDims(plan.outDims);
    tiling.set_inverseStrides(inverseStrides);
    tiling.set_inversePeriods(inversePeriods);
    tiling.set_broadcastOffsets(offsets);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    const uint32_t availableCores = std::max(1U, platform.GetCoreNumAiv());
    const uint32_t workCores = std::max(1U, plan.totalLength / kMinElementsPerCore);
    const uint32_t legalCores = std::min({kMaxCoreNum, availableCores, workCores});
    uint32_t targetElements = kMinElementsPerCore;
    if (key == kLinearKey) {
        targetElements = kDefaultElementsPerCore;
        if (dtype == ge::DT_FLOAT || dtype == ge::DT_INT8) {
            targetElements = 4U * kMinElementsPerCore;
        }
    }
    const uint32_t targetCores = std::max(1U, plan.totalLength / targetElements);
    uint32_t blockDim = std::min(legalCores, targetCores);
    while (blockDim > 1U &&
           !SplitMeetsMinimum(plan, key, lowSize, middleSize, highSize, blockDim)) {
        --blockDim;
    }
    const uint64_t inputBytes = static_cast<uint64_t>(plan.totalLength) * dataWidth;
    if (key == kLinearKey && inputBytes > kLargeStreamBytes &&
        (dtype == ge::DT_FLOAT || dtype == ge::DT_INT8)) {
        blockDim = std::min(kLargeLinearCoreCap, legalCores);
    }
    if (useBf16BatchedRows) {
        blockDim = std::min({kBf16BatchedCoreCap, legalCores, highSize});
    }
    if (key == kLinearKey && blockDim == 1U && plan.totalLength != 0U &&
        plan.totalLength <= kMinElementsPerCore) {
        blockDim = std::min({8U, kMaxCoreNum, availableCores});
    }
    context->SetBlockDim(blockDim);
    context->SetTilingKey(key);
    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* x1 = context->GetInputShape(0);
    const auto* x2 = context->GetInputShape(1);
    auto* output = context->GetOutputShape(0);
    if (x1 == nullptr || x2 == nullptr || output == nullptr) {
        return ge::GRAPH_FAILED;
    }
    BroadcastPlan plan;
    if (!PreparePlan(*x1, *x2, plan)) {
        return ge::GRAPH_FAILED;
    }
    *output = plan.outputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
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

}
