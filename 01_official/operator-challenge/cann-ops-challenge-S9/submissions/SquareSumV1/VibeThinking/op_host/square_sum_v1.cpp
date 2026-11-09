#include "square_sum_v1_tiling.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint32_t kMaxRank = 8;

constexpr uint64_t kMinBytesPerCore = 512;
constexpr uint64_t kSmallInputBytes = 64 * 1024;
constexpr uint64_t kFloatsPerBlock = 8;
constexpr uint64_t kFlatElems = 16384;
constexpr uint64_t kMaxColumnWidth = 1024;
constexpr uint64_t kMaxRepeat = 255;
constexpr uint64_t kBurstBytes = 512;
constexpr uint64_t kLanes = 64;

constexpr uint64_t kMinBlockFoldRun = 256;
constexpr uint64_t kMaxBlockFoldRun = 4096;
constexpr uint64_t kMaxBlockFoldBatch = 15;
constexpr uint64_t kStageElems = 16384;
constexpr uint64_t kGroupRows = 8;
constexpr uint64_t kPackRows = 16;
constexpr uint64_t kMaxRepStride = 255;
constexpr uint64_t kUbBytes = 190 * 1024;
constexpr uint64_t kAccElems = 1024;
constexpr uint64_t kMaxPrefixPairs = 32;
constexpr uint64_t kMinPairBandBytes = 1 << 20;
constexpr size_t kReservedWorkspace = 16 * 1024 * 1024;

uint32_t ElementBytes(ge::DataType type)
{
    switch (type) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
            return 4;
        default:
            return 0;
    }
}

bool ReadAxes(const gert::TilingContext* context, uint32_t rank, std::vector<bool>& reduced)
{
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return false;
    }
    const auto* axes = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    if (axes == nullptr || axes->GetSize() == 0 || axes->GetSize() > rank) {
        return false;
    }
    const int64_t* values = static_cast<const int64_t*>(axes->GetData());
    reduced.assign(rank, false);
    for (size_t i = 0; i < axes->GetSize(); ++i) {
        int64_t axis = values[i];
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank) || reduced[static_cast<size_t>(axis)]) {
            return false;
        }
        reduced[static_cast<size_t>(axis)] = true;
    }
    return true;
}
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* storage = context->GetInputShape(0);
    const auto* desc = context->GetInputDesc(0);
    auto* tiling = context->GetRawTilingData();
    if (storage == nullptr || desc == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& shape = storage->GetStorageShape();
    const uint32_t rank = static_cast<uint32_t>(shape.GetDimNum());
    if (rank == 0 || rank > kMaxRank) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dataType = desc->GetDataType();
    const uint32_t elementBytes = ElementBytes(dataType);
    if (elementBytes == 0) {
        return ge::GRAPH_FAILED;
    }
    std::vector<bool> reduced;
    if (!ReadAxes(context, rank, reduced)) {
        return ge::GRAPH_FAILED;
    }

    // 合并相邻且归约属性相同的轴，统一表示连续行和跨行归约。
    std::vector<uint64_t> groupSize;
    std::vector<bool> groupReduced;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        const int64_t extent = shape.GetDim(axis);
        if (extent <= 0) {
            return ge::GRAPH_FAILED;
        }
        if (extent == 1) {
            continue;
        }
        if (!groupSize.empty() && groupReduced.back() == reduced[axis]) {
            groupSize.back() *= static_cast<uint64_t>(extent);
        } else {
            groupSize.push_back(static_cast<uint64_t>(extent));
            groupReduced.push_back(reduced[axis]);
        }
    }
    if (groupSize.empty()) {

        groupSize.push_back(1);
        groupReduced.push_back(true);
    } else if (std::find(groupReduced.begin(), groupReduced.end(), true) == groupReduced.end()) {

        groupSize.push_back(1);
        groupReduced.push_back(true);
    }
    const uint32_t groups = static_cast<uint32_t>(groupSize.size());
    if (groups > kMaxRank) {
        return ge::GRAPH_FAILED;
    }

    std::vector<uint64_t> groupStride(groups, 1);
    for (uint32_t i = groups - 1; i > 0; --i) {
        groupStride[i - 1] = groupStride[i] * groupSize[i];
    }
    uint64_t outputCount = 1;
    uint64_t reduceCount = 1;
    for (uint32_t i = 0; i < groups; ++i) {
        (groupReduced[i] ? reduceCount : outputCount) *= groupSize[i];
    }

    const bool tailReduced = groupReduced[groups - 1];
    const uint64_t runLength = tailReduced ? groupSize[groups - 1] : 1;
    const uint64_t innerWidth = tailReduced ? 1 : groupSize[groups - 1];

    uint64_t contiguousOutputs = 1;
    uint64_t contiguousRows = 1;
    if (tailReduced) {
        if (groups >= 2 && !groupReduced[groups - 2] && groupStride[groups - 2] == runLength) {
            contiguousOutputs = groupSize[groups - 2];
        }
    } else {
        if (groups >= 2 && groupReduced[groups - 2] && groupStride[groups - 2] == innerWidth) {
            contiguousRows = groupSize[groups - 2];
        }
    }

    if (outputCount * reduceCount > 0xFFFFFFFFULL) {
        return ge::GRAPH_FAILED;
    }
    SquareSumV1TilingData data;
    const uint64_t outerCount = outputCount / innerWidth;
    uint32_t flags = (tailReduced ? 1U : 0U) | (groups << 16);
    const uint32_t blockElems = 32 / elementBytes;

    data.set_outputCount(outputCount);
    data.set_reduceCount(reduceCount);
    data.set_runLength(runLength);
    data.set_innerWidth(innerWidth);
    data.set_contiguousOutputs(contiguousOutputs);
    data.set_contiguousRows(contiguousRows);
    uint32_t sizes[kMaxRank] = {0};
    for (uint32_t i = 0; i < groups; ++i) {
        sizes[i] = static_cast<uint32_t>(groupSize[i]);
        flags |= groupReduced[i] ? (1U << (8 + i)) : 0U;
    }
    data.set_groupSize(sizes);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t cores = std::max<uint64_t>(1, platform.GetCoreNumAiv());
    uint64_t outputBlocks = (outputCount + blockElems - 1) / blockElems;

    if (outputCount * reduceCount * elementBytes < kSmallInputBytes) {
        outputBlocks = (outputBlocks + 1) / 2;
    }
    uint64_t blockDim = std::min<uint64_t>(cores, std::max<uint64_t>(1, outputBlocks));
    const uint64_t bytes = outputCount * reduceCount * elementBytes;
    blockDim = std::min<uint64_t>(blockDim, std::max<uint64_t>(1, bytes / kMinBytesPerCore));

    // 输出列较少时沿归约维分核，各核计算部分和后汇总到输出。
    uint64_t reduceCores = 1;
    uint64_t partialPitch = 0;

    if (!tailReduced && outerCount == 1 && innerWidth <= kMaxColumnWidth &&
        bytes >= kMinBytesPerCore * 2) {
        reduceCores = std::min<uint64_t>(cores, reduceCount);
        reduceCores = std::min<uint64_t>(reduceCores, std::max<uint64_t>(1, bytes / kMinBytesPerCore));
        if (reduceCores > 1) {
            partialPitch = (innerWidth + kFloatsPerBlock - 1) / kFloatsPerBlock * kFloatsPerBlock;
            blockDim = reduceCores;
        } else {
            reduceCores = 1;
        }
    }
    if (!tailReduced && outerCount == 1 && reduceCores == 1) {

        blockDim = 1;
    }

    uint32_t rowOwned = 0;
    uint32_t rowSplit = 0;
    if (!tailReduced && reduceCores == 1 && contiguousRows > 1 && outerCount > 1 &&
        innerWidth <= kMaxColumnWidth && kGroupRows * innerWidth <= kStageElems &&
        kGroupRows * innerWidth <= kFlatElems) {
        rowOwned = 1;
        blockDim = std::min<uint64_t>(cores, outerCount);
        blockDim = std::min<uint64_t>(blockDim, std::max<uint64_t>(1, bytes / kMinBytesPerCore));

        const uint64_t peakRows = (outerCount + blockDim - 1) / blockDim;
        blockDim = (outerCount + peakRows - 1) / peakRows;
    }
    flags |= (rowOwned << 1) | (rowSplit << 2);

    uint64_t flatBatch = 0;
    bool packedRoute = false;
    uint64_t slotFloats = 0;
    uint64_t blockFold = 0;
    if (tailReduced && contiguousOutputs > 1 && runLength * elementBytes < kBurstBytes &&
        runLength <= kFlatElems) {

        const uint64_t paddedRun = (runLength + kFloatsPerBlock - 1) / kFloatsPerBlock * kFloatsPerBlock;

        const bool fusedSquare = dataType == ge::DT_FLOAT16 || dataType == ge::DT_FLOAT;
        const bool packedFold = runLength <= kMaxRepStride && runLength + kGroupRows <= 128;
        packedRoute = packedFold;

        const uint64_t stagePitch = (fusedSquare && packedFold) ? runLength : paddedRun;

        const bool narrowAccum = fusedSquare && packedFold && elementBytes == 2 && reduceCount > runLength &&
                                 (kPackRows * runLength * elementBytes) % 32 == 0 &&
                                 runLength + kPackRows <= 128;
        const uint64_t accumBytes = narrowAccum ? elementBytes * runLength : 4 * runLength;
        const uint64_t perRow = 2 * stagePitch * elementBytes + accumBytes +
                                ((fusedSquare && packedFold) ? 0 : 4 * paddedRun) + 3 * sizeof(float);
        const uint64_t budget = kUbBytes - kAccElems * sizeof(float) - 2048;

        const uint64_t batchCap = packedFold ? kAccElems : kMaxRepeat;
        flatBatch = std::min<uint64_t>(budget / perRow, batchCap);
        flatBatch = std::min<uint64_t>(flatBatch, contiguousOutputs);
        if (flatBatch > 1) {
            slotFloats = flatBatch * runLength;
        } else {
            flatBatch = 0;
        }
    } else if (tailReduced && contiguousOutputs > 1 && reduceCount == runLength &&
               runLength >= kMinBlockFoldRun && runLength <= kMaxBlockFoldRun &&
               runLength / kLanes % kFloatsPerBlock == 0 && runLength % kLanes > 0) {

        const uint64_t spill = runLength % kLanes;
        flatBatch = std::min<uint64_t>((kFlatElems - kLanes) / runLength, kMaxRepeat);
        flatBatch = std::min<uint64_t>(flatBatch, kStageElems / runLength);
        flatBatch = std::min<uint64_t>(flatBatch, kMaxRepeat * kLanes / runLength);
        flatBatch = std::min<uint64_t>(flatBatch, kLanes / spill);
        flatBatch = std::min<uint64_t>(flatBatch, contiguousOutputs);

        flatBatch = std::min<uint64_t>(flatBatch, kMaxBlockFoldBatch);
        if (flatBatch > 1) {
            blockFold = 1;
        } else {
            flatBatch = 0;
        }
    }

    if (!tailReduced && contiguousRows > 1 && innerWidth <= kMaxColumnWidth) {
        slotFloats = std::max<uint64_t>(slotFloats, kGroupRows * innerWidth);
    }
    slotFloats = (slotFloats + kFloatsPerBlock - 1) / kFloatsPerBlock * kFloatsPerBlock;

    uint64_t pairBand = 0;
    if (dataType == ge::DT_FLOAT16 && groups == 3 && groupReduced[0] && !groupReduced[1] && groupReduced[2] &&
        groupSize[0] >= 2 && groupSize[0] <= 2 * kMaxPrefixPairs && groupSize[2] > 64 &&
        groupSize[2] <= 113 && bytes >= kMinPairBandBytes) {
        const uint64_t w = groupSize[2];
        const uint64_t c = std::min<uint64_t>(cores, std::max<uint64_t>(1, groupSize[1] / 16));
        auto up = [](uint64_t n) { return (n + 31) / 32 * 32; };
        for (uint64_t band = 16 * (kUbBytes / (3 * w * elementBytes) / 16); band >= 16 && pairBand == 0; band -= 16) {
            const uint64_t slot = up(band * w * elementBytes) + 128 * elementBytes;
            const uint64_t held = std::min<uint64_t>(8, groupSize[1] / (c * band) + 1);
            if (3 * slot + held * up(band * elementBytes) + 1024 <= kUbBytes) {
                pairBand = band;
                blockDim = c;
            }
        }
    }
    data.set_pairBand(static_cast<uint32_t>(pairBand));

    uint64_t spanBatch = 0;
    if (elementBytes == 2 && tailReduced && reduceCount == runLength && contiguousOutputs == outputCount &&
        runLength > 64 && (runLength - 1) % 512 == 0 && (runLength - 1) / 64 <= 64 && outputCount >= 16) {
        const uint64_t m = (runLength - 1) / 64;
        for (uint64_t b = 16; b >= 8 && spanBatch == 0; b /= 2) {
            const uint64_t need = 2 * b * runLength * elementBytes + (b * runLength + 8) * 4 + b * m * 8 * 4 +
                                  (b * m + 64) * 4 + 2 * b * 64 * 4 + 6 * b * 4 + 2048;
            if (need <= kUbBytes) {
                spanBatch = b;
            }
        }
    }
    data.set_spanBatch(static_cast<uint32_t>(spanBatch));

    uint64_t shortRows = 0;
    if (dataType == ge::DT_FLOAT16 && tailReduced && reduceCount == runLength && contiguousOutputs == outputCount &&
        runLength <= 56 && bytes < kSmallInputBytes) {
        const uint64_t units = (outputCount + 15) / 16;
        blockDim = std::min<uint64_t>(cores, (units + 1) / 2);
        shortRows = ((units + blockDim - 1) / blockDim) * 16;
    }
    data.set_shortRows(static_cast<uint32_t>(shortRows));

    uint32_t laneCycle = 0;
    if (dataType == ge::DT_FLOAT && !tailReduced && outerCount == 1 && contiguousRows == reduceCount &&
        reduceCount >= 4096 && innerWidth % kFloatsPerBlock != 0) {
        const uint64_t w = innerWidth;
        const uint64_t pitch = (w + kFloatsPerBlock - 1) / kFloatsPerBlock * kFloatsPerBlock;
        const uint64_t need = (2 * 32 * w + 16 * w + 8 * pitch + w + pitch) * 4 + 320 + 2048;
        if (need <= kUbBytes) {
            laneCycle = 1;
            blockDim = std::min<uint64_t>(cores, reduceCount / 32);
        }
    }
    data.set_flatBatch(static_cast<uint32_t>(flatBatch));
    flags |= static_cast<uint32_t>(blockFold) << 3;
    data.set_flags(flags);
    data.set_slotFloats(static_cast<uint32_t>(slotFloats));
    data.set_partialBase(static_cast<uint32_t>(blockDim * slotFloats));
    data.set_reduceCores(static_cast<uint32_t>(reduceCores));
    data.set_partialPitch(static_cast<uint32_t>(partialPitch));
    data.SaveToBuffer(tiling->GetData(), tiling->GetCapacity());
    tiling->SetDataSize(data.GetDataSize());

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 按连续性、行宽和 UB 容量选择归约路径，分片参数随路径一同传入。
    TilingRoute route;
    if (pairBand != 0) {
        route = TilingRoute::kWaveSlice;
    } else if (spanBatch != 0) {
        route = TilingRoute::kSpanEdge;
    } else if (shortRows != 0) {
        route = TilingRoute::kShortRun;
    } else if (laneCycle != 0) {
        route = TilingRoute::kLaneCycle;
    } else if (reduceCores > 1) {
        route = partialPitch == innerWidth ? TilingRoute::kSplitAligned : TilingRoute::kSplitPadded;
    } else if (tailReduced && flatBatch != 0) {
        route = blockFold != 0 ? TilingRoute::kBlockFold
                               : (packedRoute ? TilingRoute::kFlatPacked : TilingRoute::kFlatStaged);
    } else if (tailReduced) {
        route = outputCount == 1 ? TilingRoute::kRowSingle : TilingRoute::kRowMany;
    } else if (rowOwned != 0) {
        route = TilingRoute::kColumnOwned;
    } else {
        route = TilingRoute::kColumnStrided;
    }
    context->SetTilingKey(static_cast<uint32_t>(route));

    size_t* workspace =
        (pairBand != 0 || spanBatch != 0 || shortRows != 0 || laneCycle != 0) ? nullptr : context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        const uint64_t partialSlots = reduceCores > 1 ? reduceCores : (rowSplit != 0 ? 2 * blockDim : 0);
        workspace[0] = kReservedWorkspace +
                       (blockDim * slotFloats + partialSlots * partialPitch) * sizeof(float);
    }
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* attrs = context->GetAttrs();
    const gert::Shape* input = context->GetInputShape(0);
    gert::Shape* output = context->GetOutputShape(0);
    if (attrs == nullptr || input == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    const auto* axes = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const auto* keepDims = attrs->GetAttrPointer<bool>(1);
    if (axes == nullptr || axes->GetSize() == 0) {
        return GRAPH_FAILED;
    }
    const uint32_t rank = static_cast<uint32_t>(input->GetDimNum());
    bool reduced[8] = {false};
    const int64_t* values = static_cast<const int64_t*>(axes->GetData());
    for (size_t i = 0; i < axes->GetSize(); ++i) {
        int64_t axis = values[i] < 0 ? values[i] + static_cast<int64_t>(rank) : values[i];
        if (axis < 0 || axis >= static_cast<int64_t>(rank) || reduced[axis]) {
            return GRAPH_FAILED;
        }
        reduced[axis] = true;
    }
    const bool keep = keepDims != nullptr && *keepDims;
    output->SetDimNum(0);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        if (keep) {
            output->AppendDim(reduced[axis] ? 1 : input->GetDim(axis));
        } else if (!reduced[axis]) {
            output->AppendDim(input->GetDim(axis));
        }
    }
    if (output->GetDimNum() == 0) {
        output->AppendDim(1);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        const std::vector<ge::DataType> types = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT};
        const std::vector<ge::Format> formats(types.size(), ge::FORMAT_ND);
        this->Input("x").ParamType(REQUIRED).DataType(types).Format(formats).UnknownShapeFormat(formats);
        this->Output("y").ParamType(REQUIRED).DataType(types).Format(formats).UnknownShapeFormat(formats);
        this->Attr("axis").AttrType(REQUIRED).ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
}
