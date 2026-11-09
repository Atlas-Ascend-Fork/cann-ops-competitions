#include "square_sum_v1_tiling.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr uint32_t kAlignBytes = 32;
constexpr uint32_t kRowsPerCoreForSmallReduce = 24;
constexpr uint32_t kGenericBatchMinReduce = 64;
constexpr uint32_t kGenericMinTileCols = 256;
constexpr uint32_t kGenericMaxTileCols = 1024;
constexpr uint32_t kMaxFastReduceLength = 256;
constexpr uint32_t kMaxSingleReduceLength = 128;
constexpr uint32_t kMaxRank = 8;
constexpr uint32_t kMaxCoreNum = 40;
constexpr uint64_t kTilingKeySuffixReduce = 0;
constexpr uint64_t kTilingKeyGenericVectorReduce = 2;
constexpr uint64_t kTilingKeySuffixSplitReduce = 3;
constexpr uint64_t kTilingKeyNarrowAxis2Reduce = 4;
constexpr uint64_t kTilingKeySuffixWideReduce = 5;
constexpr uint64_t kTilingKeySplitKeptReduce = 6;
constexpr uint64_t kTilingKeySuffixWideDoubleReduce = 7;
constexpr uint64_t kTilingKeySplitKeptFp32Pipeline = 8;
constexpr uint64_t kTilingKeyRank4Axis12Bf16 = 9;
constexpr uint64_t kTilingKeyRank4Axis03Batches = 10;
constexpr uint64_t kTilingKeyRank4Axis12Bf16Unaligned = 11;
constexpr uint64_t kTilingKeyRank3SuffixBf16Compact = 12;
constexpr uint64_t kTilingKeyRank4Axis12Bf16Cycle32 = 13;

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return ((value + align - 1U) / align) * align;
}

uint32_t BytesPerElement(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_FLOAT:
            return 4;
        default:
            return 0;
    }
}

bool ReadAxes(const gert::RuntimeAttrs* attrs, uint32_t rank, uint32_t& axisMask)
{
    if (attrs == nullptr || rank == 0 || rank > kMaxRank) {
        return false;
    }
    const auto* axisAttr = attrs->GetListInt(0);
    if (axisAttr == nullptr || axisAttr->GetSize() == 0U) {
        return false;
    }

    axisMask = 0;
    for (size_t idx = 0; idx < axisAttr->GetSize(); ++idx) {
        int64_t axis = axisAttr->GetData()[idx];
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
            return false;
        }
        axisMask |= 1U << static_cast<uint32_t>(axis);
    }
    return axisMask != 0;
}

bool ReadKeepDims(const gert::RuntimeAttrs* attrs, bool& keepDims)
{
    if (attrs == nullptr) {
        return false;
    }
    keepDims = false;
    const bool* keepDimsAttr = attrs->GetBool(1);
    if (keepDimsAttr != nullptr) {
        keepDims = *keepDimsAttr;
    }
    return true;
}

bool ReadReduceProfile(const gert::Shape& shape, const gert::RuntimeAttrs* attrs, uint32_t& rank,
                       uint32_t& axisMask, uint64_t& inputSize, uint64_t& outputSize,
                       uint64_t& suffixOuterLength, uint64_t& suffixReduceLength, bool& isSuffixReduce,
                       uint32_t dims[kMaxRank])
{
    rank = static_cast<uint32_t>(shape.GetDimNum());
    if (!ReadAxes(attrs, rank, axisMask)) {
        return false;
    }

    inputSize = 1;
    outputSize = 1;
    suffixOuterLength = 1;
    suffixReduceLength = 1;
    for (uint32_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
        const int64_t dim = shape.GetDim(static_cast<size_t>(dimIdx));
        if (dim <= 0 || static_cast<uint64_t>(dim) > std::numeric_limits<uint32_t>::max() ||
            inputSize > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
                            static_cast<uint64_t>(dim)) {
            return false;
        }
        dims[dimIdx] = static_cast<uint32_t>(dim);
        inputSize *= static_cast<uint64_t>(dim);
        if ((axisMask & (1U << dimIdx)) == 0) {
            outputSize *= static_cast<uint64_t>(dim);
        }
    }

    isSuffixReduce = true;
    bool seenKeptAfterSuffix = false;
    for (int64_t dimIdx = static_cast<int64_t>(rank) - 1; dimIdx >= 0; --dimIdx) {
        const bool reduced = (axisMask & (1U << static_cast<uint32_t>(dimIdx))) != 0;
        if (reduced) {
            if (seenKeptAfterSuffix) {
                isSuffixReduce = false;
            }
            suffixReduceLength *= dims[dimIdx];
        } else {
            seenKeptAfterSuffix = true;
            suffixOuterLength *= dims[dimIdx];
        }
    }

    for (uint32_t dimIdx = rank; dimIdx < kMaxRank; ++dimIdx) {
        dims[dimIdx] = 1;
    }

    return inputSize > 0 && outputSize > 0 && suffixOuterLength > 0 && suffixReduceLength > 0;
}

}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* inputShapeHolder = context->GetInputShape(0);
    const auto* inputTensor = context->GetInputTensor(0);
    if (inputShapeHolder == nullptr || inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = inputTensor->GetDataType();
    const uint32_t dataBytes = BytesPerElement(dtype);
    if (dataBytes == 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape inputShape = inputShapeHolder->GetStorageShape();
    uint32_t rank = 0;
    uint32_t axisMask = 0;
    uint64_t inputSize64 = 0;
    uint64_t outputSize64 = 0;
    uint64_t outerLength64 = 0;
    uint64_t reduceLength64 = 0;
    bool isSuffixReduce = false;
    uint32_t dims[kMaxRank] = {};
    if (!ReadReduceProfile(inputShape, context->GetAttrs(), rank, axisMask, inputSize64, outputSize64,
                           outerLength64, reduceLength64, isSuffixReduce, dims)) {
        return ge::GRAPH_FAILED;
    }
    bool keepDims = false;
    if (!ReadKeepDims(context->GetAttrs(), keepDims)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t compactRank = 0;
    uint32_t compactMask = 0;
    for (uint32_t dim = 0; dim < rank; ++dim) {
        if (dims[dim] != 1U) {
            compactMask |= ((axisMask >> dim) & 1U) << compactRank;
            dims[compactRank++] = dims[dim];
        }
    }
    if (compactRank < rank) {
        if (compactMask == 0U) {
            compactMask = 1U << compactRank;
            dims[compactRank++] = 1U;
        }
        rank = compactRank;
        axisMask = compactMask;
        for (uint32_t dim = rank; dim < kMaxRank; ++dim) dims[dim] = 1U;
        const uint32_t lowestAxis = axisMask & (~axisMask + 1U);
        isSuffixReduce = axisMask + lowestAxis == (1U << rank);
    }
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = std::min(std::max(1U, platform.GetCoreNumAiv()), kMaxCoreNum);
    if (inputSize64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / dataBytes) {
        return ge::GRAPH_FAILED;
    }

    if (inputSize64 > std::numeric_limits<uint32_t>::max() / dataBytes ||
        (dtype == ge::DT_FLOAT16 && reduceLength64 > 262144U)) {
        SquareSumV1TilingData tiling;
        tiling.set_rank(rank);
        tiling.set_axisMask(axisMask);
        tiling.set_dims(dims);
        tiling.set_wideInputSize(inputSize64);
        tiling.set_wideOutputSize(outputSize64);
        tiling.set_wideReduceLength(reduceLength64);
        context->SetBlockDim(blockDim);
        context->SetTilingKey(14U);
        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
        context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
                                         static_cast<size_t>(blockDim) * 8192U * sizeof(float);
        return ge::GRAPH_SUCCESS;
    }
    const uint32_t inputSize = static_cast<uint32_t>(inputSize64);
    const uint32_t outputSize = static_cast<uint32_t>(outputSize64);
    const uint32_t outerLength = static_cast<uint32_t>(outerLength64);
    const uint32_t reduceLength = static_cast<uint32_t>(reduceLength64);
    const bool isFloat16 = dtype == ge::DT_FLOAT16;
    const bool useSuffixFastPath = isFloat16 && isSuffixReduce && reduceLength <= kMaxSingleReduceLength;
    const bool useSuffixSplitPath = isFloat16 && isSuffixReduce && reduceLength > kMaxSingleReduceLength &&
                                    reduceLength <= kMaxFastReduceLength;
    const bool useSuffixPath = useSuffixFastPath || useSuffixSplitPath;
    const bool useSuffixWidePath = isSuffixReduce && reduceLength >= kGenericBatchMinReduce && !useSuffixPath;
    const bool useSuffixWideDoublePath = useSuffixWidePath && dataBytes == 2U && outerLength > blockDim &&
                                         (reduceLength > kGenericMaxTileCols ||
                                          (dtype == ge::DT_BF16 && rank == 3U && axisMask == 0x4U &&
                                           reduceLength > 512U));
    const bool useRank3SuffixBf16Compact = useSuffixWidePath && dtype == ge::DT_BF16 &&
                                           rank == 3U && axisMask == 0x4U &&
                                           reduceLength > 960U && reduceLength < 2048U &&
                                           outerLength > blockDim;
    const bool useNarrowAxis2Path = isFloat16 && rank == 4U && axisMask == (1U << 2U) &&
                                    dims[3] == 5U && reduceLength >= 8192U;
    const bool useGenericKeptRowBatch = !useSuffixPath && !useSuffixWidePath &&
                                        reduceLength >= kGenericBatchMinReduce &&
                                        (axisMask & (1U << (rank - 1U))) == 0;
    uint32_t genericTileCols = kGenericMaxTileCols;
    uint32_t genericKeptWorkItems = outputSize;
    bool useLargeFp32FullWidth = false;
    if (useGenericKeptRowBatch) {
        uint32_t lastReducedDim = rank - 1U;
        while ((axisMask & (1U << lastReducedDim)) == 0U) {
            --lastReducedDim;
        }
        uint32_t trailingKeptLength = 1;
        for (uint32_t dimIdx = lastReducedDim + 1U; dimIdx < rank; ++dimIdx) {
            trailingKeptLength *= dims[dimIdx];
        }
        const uint32_t innerDim = trailingKeptLength >= kGenericMinTileCols
                                      ? trailingKeptLength
                                      : dims[rank - 1U];
        const uint32_t outputRows = outputSize / innerDim;
        const uint32_t targetTiles = std::min((innerDim + kGenericMinTileCols - 1U) / kGenericMinTileCols,
                                              (blockDim + outputRows - 1U) / outputRows);
        genericTileCols = AlignUp((innerDim + targetTiles - 1U) / targetTiles, kGenericMinTileCols);
        genericTileCols = std::min(genericTileCols, kGenericMaxTileCols);
        genericKeptWorkItems = outputRows * ((innerDim + genericTileCols - 1U) / genericTileCols);
        if (dtype == ge::DT_FLOAT && rank == 3U && axisMask == 0x3U &&
            dims[2] > 512U && dims[2] <= 768U && reduceLength > 2048U) {
            useLargeFp32FullWidth = reduceLength > 4096U && dims[2] <= 640U;
            genericTileCols = useLargeFp32FullWidth
                                  ? dims[2]
                                  : AlignUp((dims[2] + 1U) / 2U,
                                            kAlignBytes / sizeof(float));
            genericKeptWorkItems = useLargeFp32FullWidth ? 1U : 2U;
        }
    }
    uint32_t firstReducedDim = rank;
    uint32_t lastReducedDim = 0;
    for (uint32_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
        if ((axisMask & (1U << dimIdx)) != 0U) {
            firstReducedDim = std::min(firstReducedDim, dimIdx);
            lastReducedDim = dimIdx;
        }
    }
    uint32_t contiguousReduceMask = 0;
    for (uint32_t dimIdx = firstReducedDim; dimIdx <= lastReducedDim; ++dimIdx) {
        contiguousReduceMask |= 1U << dimIdx;
    }
    const uint32_t maxBlockDim = blockDim;
    const uint32_t splitCount = !useNarrowAxis2Path && useGenericKeptRowBatch && axisMask == contiguousReduceMask &&
                                        genericKeptWorkItems < maxBlockDim && reduceLength >= 512U
                                    ? std::min(reduceLength, maxBlockDim / genericKeptWorkItems)
                                    : 1U;
    const bool useSplitKeptPath = splitCount > 1U;
    const bool useSplitKeptFp32Pipeline = useSplitKeptPath && dtype == ge::DT_FLOAT;
    const bool rank4Axis12Bf16 = dtype == ge::DT_BF16 && rank == 4U && axisMask == 0x6U &&
                                 reduceLength >= kGenericBatchMinReduce &&
                                 !useSplitKeptPath && dims[3] <= genericTileCols;
    const bool useRank4Axis12Bf16 = rank4Axis12Bf16 && dims[3] % 16U == 0U;
    const bool useRank4Axis12Bf16Unaligned = rank4Axis12Bf16 && dims[3] % 16U != 0U &&
                                             dims[3] <= 511U;
    const bool useRank4Axis12Bf16Cycle32 = useRank4Axis12Bf16Unaligned && dims[3] <= 255U;
    const bool useRank4Axis03Batches = isFloat16 && rank == 4U && axisMask == 0x9U &&
                                       dims[3] >= 65U && dims[3] <= 80U &&
                                       reduceLength <= 2048U && outputSize > 512U;
    const bool useRank4Axis03PhasePairs = useRank4Axis03Batches &&
                                          dims[0] >= 2U && dims[0] <= 22U &&
                                          dims[0] % 2U == 0U && dims[3] >= 65U && dims[3] <= 69U;
    const bool useRank4Axis03Batch32 = useRank4Axis03Batches && !useRank4Axis03PhasePairs &&
                                       dims[0] <= 16U && dims[3] % 8U == 0U;
    const uint32_t axis03BatchRows = useRank4Axis03PhasePairs ? 272U
                                      : (useRank4Axis03Batch32 ? 32U : 16U);
    const uint32_t workItems = useRank4Axis03Batches ? (outputSize + axis03BatchRows - 1U) / axis03BatchRows
                               : (useSplitKeptPath ? genericKeptWorkItems * splitCount
                               : ((useSuffixPath || useSuffixWidePath) ? outerLength
                                  : (useGenericKeptRowBatch ? genericKeptWorkItems : outputSize)));
    if (useSuffixPath && reduceLength <= 64U) {
        blockDim = std::min(blockDim, (workItems + kRowsPerCoreForSmallReduce - 1U) /
                                           kRowsPerCoreForSmallReduce);
    }
    if (useRank4Axis03Batch32 && workItems % 32U == 0U) {
        blockDim = std::min(blockDim, 32U);
    }
    blockDim = std::max(1U, std::min(blockDim, workItems));
    const uint32_t smallCoreItems = workItems / blockDim;
    const uint32_t bigCoreCount = workItems % blockDim;
    const uint32_t elemAlign = std::max(1U, kAlignBytes / dataBytes);
    const uint32_t alignedReduceLength = AlignUp(reduceLength, elemAlign);

    SquareSumV1TilingData tiling;
    tiling.set_outerLength(outerLength);
    tiling.set_reduceLength(reduceLength);
    tiling.set_smallCoreItems(smallCoreItems);
    tiling.set_bigCoreCount(bigCoreCount);
    tiling.set_alignedReduceLength(alignedReduceLength);
    tiling.set_inputSize(inputSize);
    tiling.set_outputSize(outputSize);
    tiling.set_rank(rank);
    tiling.set_axisMask(axisMask);
    tiling.set_genericTileCols(genericTileCols);
    tiling.set_dims(dims);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(useRank4Axis12Bf16Cycle32 ? kTilingKeyRank4Axis12Bf16Cycle32
                                            : (useRank4Axis12Bf16Unaligned ? kTilingKeyRank4Axis12Bf16Unaligned
                                            : (useRank4Axis03Batches ? kTilingKeyRank4Axis03Batches
                                            : (useRank4Axis12Bf16 ? kTilingKeyRank4Axis12Bf16
                                            : (useSuffixFastPath ? kTilingKeySuffixReduce
                                            : (useSuffixSplitPath ? kTilingKeySuffixSplitReduce
                                            : (useSplitKeptFp32Pipeline ? kTilingKeySplitKeptFp32Pipeline
                                            : (useSplitKeptPath ? kTilingKeySplitKeptReduce
                                            : (useNarrowAxis2Path ? kTilingKeyNarrowAxis2Reduce
                                            : (useRank3SuffixBf16Compact ? kTilingKeyRank3SuffixBf16Compact
                                            : (useSuffixWideDoublePath ? kTilingKeySuffixWideDoubleReduce
                                            : (useSuffixWidePath ? kTilingKeySuffixWideReduce
                                                              : kTilingKeyGenericVectorReduce))))))))))));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    const bool useBalancedBf16 = useRank4Axis12Bf16Cycle32;
    if (useSplitKeptPath || useBalancedBf16) {
        auto workspaceSizes = context->GetWorkspaceSizes(1);
        const size_t userWorkspaceBytes = useBalancedBf16
            ? static_cast<size_t>(2U * (blockDim - 1U)) *
                  AlignUp(dims[3], kAlignBytes / sizeof(uint16_t)) * sizeof(float)
            : static_cast<size_t>(blockDim) *
                  AlignUp(genericTileCols, kAlignBytes / sizeof(float)) * sizeof(float);
        workspaceSizes[0] = platform.GetLibApiWorkSpaceSize() + userWorkspaceBytes;
    }
    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t rank = 0;
    uint32_t axisMask = 0;
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    uint64_t outerLength = 0;
    uint64_t reduceLength = 0;
    bool isSuffixReduce = false;
    uint32_t dims[kMaxRank] = {};
    if (!ReadReduceProfile(*inputShape, context->GetAttrs(), rank, axisMask, inputSize, outputSize,
                           outerLength, reduceLength, isSuffixReduce, dims)) {
        return ge::GRAPH_FAILED;
    }
    (void)inputSize;
    (void)outputSize;
    (void)outerLength;
    (void)reduceLength;
    (void)isSuffixReduce;

    bool keepDims = false;
    if (!ReadKeepDims(context->GetAttrs(), keepDims)) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape resultShape;
    for (uint32_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
        if ((axisMask & (1U << dimIdx)) != 0) {
            if (keepDims) {
                resultShape.AppendDim(1);
            }
            continue;
        }
        resultShape.AppendDim(inputShape->GetDim(static_cast<size_t>(dimIdx)));
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *outputShape = resultShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType inputDtype = context->GetInputDataType(0);
    if (inputDtype != ge::DT_FLOAT16 && inputDtype != ge::DT_BF16 && inputDtype != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("axis").AttrType(REQUIRED).ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);

}
