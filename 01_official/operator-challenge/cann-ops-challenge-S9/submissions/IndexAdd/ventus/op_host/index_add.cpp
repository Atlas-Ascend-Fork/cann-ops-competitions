/**
 * IndexAdd 算子注册、形状推导与主机侧分块规划。
 * 根据索引数量、目标轴长度和数据类型划分累加任务，并计算所需工作区。
 */

#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace {

constexpr uint64_t kAccElements = 16384;

constexpr uint64_t kMaxInnerChunk = 4096;
constexpr uint64_t kUbBlockBytes = 32;

constexpr uint64_t kOwnAccCapBytes = 122880;
constexpr uint64_t kBf16RowElements = 7680;
constexpr uint64_t kResidentIndexElements = 8192;
constexpr uint64_t kResidentMaxRows = 2048;

uint32_t GetAivCoreCount(const gert::TilingContext* context)
{
    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    return std::max<uint32_t>(1U, platform.GetCoreNumAiv());
}

uint64_t DataTypeSize(ge::DataType type)
{
    switch (type) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_INT8:
            return 1;
        default:
            return 0;
    }
}

uint64_t RowPitch(uint64_t chunk, uint64_t elementSize)
{
    const uint64_t bytes = ((chunk * elementSize + kUbBlockBytes - 1) / kUbBlockBytes) * kUbBlockBytes;
    return bytes / elementSize;
}

uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return b == 0 ? 0 : (a + b - 1) / b;
}
}

namespace optiling {
// 按 outer、axis 和 inner 规划累加任务，并为索引分桶或跨核同步分配工作区。
static ge::graphStatus IndexAddTiling(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<IndexAddTilingData>();
    const auto* selfStorage = context->GetInputShape(0);
    const auto* indexStorage = context->GetInputShape(1);
    const auto* selfDesc = context->GetInputDesc(0);
    if (tiling == nullptr || selfStorage == nullptr || indexStorage == nullptr || selfDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& shape = selfStorage->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    if (rank == 0 || rank > S9_INDEX_ADD_MAX_RANK) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) <= 0) {
            return ge::GRAPH_FAILED;
        }
    }
    const uint64_t elementSize = DataTypeSize(selfDesc->GetDataType());
    if (elementSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t* dimAttr = context->GetAttrs()->GetInt(0);
    int64_t dim = (dimAttr == nullptr) ? 0 : *dimAttr;
    if (dim < 0) {
        dim += static_cast<int64_t>(rank);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outer = 1;
    uint64_t inner = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer *= static_cast<uint64_t>(shape.GetDim(i));
    }
    for (size_t i = static_cast<size_t>(dim) + 1; i < rank; ++i) {
        inner *= static_cast<uint64_t>(shape.GetDim(i));
    }
    tiling->outer = outer;
    tiling->axisLen = static_cast<uint64_t>(shape.GetDim(dim));
    tiling->inner = inner;

    uint64_t indexCount = 1;
    const gert::Shape& indexShape = indexStorage->GetStorageShape();
    for (size_t i = 0; i < indexShape.GetDimNum(); ++i) {
        indexCount *= static_cast<uint64_t>(indexShape.GetDim(i));
    }
    tiling->indexCount = indexCount;

    const uint64_t perRow = std::max<uint64_t>(1, kAccElements / std::max<uint64_t>(1, tiling->axisLen));
    const uint64_t elementsPerBlock = kUbBlockBytes / elementSize;
    uint64_t chunk = (perRow / elementsPerBlock) * elementsPerBlock;
    if (chunk < elementsPerBlock) {
        chunk = elementsPerBlock;
    }
    const uint64_t innerChunk = std::min<uint64_t>(std::min<uint64_t>(chunk, inner), kMaxInnerChunk);
    const uint64_t pitch = RowPitch(innerChunk, elementSize);
    tiling->innerChunk = innerChunk;
    tiling->innerChunkCount = CeilDiv(inner, innerChunk);

    uint64_t axisChunk = std::max<uint64_t>(1, std::min<uint64_t>(tiling->axisLen, kAccElements / pitch));
    uint64_t axisChunkCount = CeilDiv(tiling->axisLen, axisChunk);
    const uint64_t cores = GetAivCoreCount(context);

    const uint64_t denom =
        std::max<uint64_t>(1, outer * tiling->innerChunkCount);
    if (axisChunkCount * denom < cores) {
        uint64_t want = cores / denom;
        if (want < 1) {
            want = 1;
        }
        if (want > tiling->axisLen) {
            want = tiling->axisLen;
        }
        if (want > axisChunkCount) {
            axisChunk = CeilDiv(tiling->axisLen, want);
            if (axisChunk < 1) {
                axisChunk = 1;
            }
            axisChunkCount = CeilDiv(tiling->axisLen, axisChunk);
        }
    }
    tiling->axisChunk = axisChunk;
    tiling->axisChunkCount = axisChunkCount;
    tiling->taskCount = outer * tiling->innerChunkCount * axisChunkCount;

    const bool isInteger =
        selfDesc->GetDataType() == ge::DT_INT8 || selfDesc->GetDataType() == ge::DT_INT32;

    tiling->useBucket = 0;
    tiling->numSlices = 0;
    tiling->sliceRows = 0;
    tiling->sliceShift = 0;
    tiling->bucketCap = 0;
    tiling->bucketCores = 0;
    tiling->useResident = 0;
    tiling->singleSubblock = 0;

    if (isInteger) {


        uint64_t effCores = cores < 2 ? 48 : cores;
        tiling->blockDim = static_cast<uint32_t>(effCores);
        context->SetBlockDim(tiling->blockDim);
        context->GetWorkspaceSizes(1)[0] = 16UL * 1024 * 1024 + tiling->blockDim * 32 + 32;
        return ge::GRAPH_SUCCESS;
    }

    const uint64_t P = (cores < 2) ? 48 : cores;

    const ge::DataType dt = selfDesc->GetDataType();
    const bool isFloatBucket = (dt == ge::DT_FLOAT) || (dt == ge::DT_FLOAT16) || (dt == ge::DT_BF16);

    const uint64_t kMinBucketRowBytes = 64;
    const bool wideEnough = (inner * elementSize) >= kMinBucketRowBytes;

    const uint64_t residentPitch = RowPitch(inner, elementSize);
    const uint64_t residentBatch = residentPitch == 0 ? 0 : kMaxInnerChunk / residentPitch;
    const bool residentEligible = isFloatBucket && outer > 0 && tiling->axisLen > 0 && inner > 0 &&
        inner <= kMaxInnerChunk && tiling->axisLen * residentPitch <= kAccElements &&
        residentBatch >= 2 && indexCount > 0;

    const auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t bucketCores = dt == ge::DT_FLOAT16 ?
        std::max<uint32_t>(1U, platform.GetCoreNumAic()) : P;

    const uint64_t residentOuterThreshold = dt == ge::DT_BF16 ? P : 8;
    const bool preferResident = residentEligible &&
        (!wideEnough || indexCount < 1024 || outer >= residentOuterThreshold);
    if (preferResident) {
        tiling->useResident = 1;
        tiling->taskCount = outer;
        tiling->blockDim = static_cast<uint32_t>(
            std::min<uint64_t>(P, outer));
        const uint32_t launchDim = dt == ge::DT_FLOAT16 ?
            (tiling->blockDim + 1U) / 2U : tiling->blockDim;
        context->SetBlockDim(launchDim);
        const bool sharedBf16Order = dt == ge::DT_BF16 && outer >= P &&
            indexCount >= 128U && indexCount <= kResidentIndexElements &&
            inner * elementSize >= 128U &&
            ((inner * elementSize + 31U) / 32U) <= 255U &&
            tiling->axisLen <= kResidentMaxRows && residentPitch != 0U &&
            kBf16RowElements / residentPitch >= 8U;
        if (sharedBf16Order) {
            tiling->bucketCores = P;
            const uint64_t sortedBytes =
                ((indexCount * sizeof(int32_t) + 31U) / 32U) * 32U;
            const uint64_t startsBytes =
                (tiling->axisLen + 1U) * sizeof(int32_t);
            context->GetWorkspaceSizes(1)[0] = 16UL * 1024 * 1024 +
                sortedBytes + startsBytes + 32U;
            return ge::GRAPH_SUCCESS;
        }

        context->GetWorkspaceSizes(1)[0] =
            16UL * 1024 * 1024 +
            static_cast<uint64_t>(tiling->blockDim) * 32 + 32;
        return ge::GRAPH_SUCCESS;
    }

    const bool bucketAllowed = isFloatBucket && dt != ge::DT_FLOAT;
    if (bucketAllowed && wideEnough && inner >= 1 && inner <= 4096 &&
        indexCount >= 1024) {

        const uint64_t pitchElems = RowPitch(inner, elementSize);
        uint64_t sliceRows = 1;
        uint64_t shift = 0;
        while ((sliceRows * 2) * pitchElems <= kAccElements && (sliceRows * 2) <= tiling->axisLen) {
            sliceRows *= 2;
            ++shift;
        }
        uint64_t S = CeilDiv(tiling->axisLen, sliceRows);
        while (outer * S < bucketCores && sliceRows > 1) {
            sliceRows /= 2;
            --shift;
            S = CeilDiv(tiling->axisLen, sliceRows);
        }
        uint64_t bucketCap = CeilDiv(indexCount, bucketCores);
        bucketCap = ((bucketCap + 7) / 8) * 8;
        if (bucketCap < 8) {
            bucketCap = 8;
        }

        const uint64_t bucketBytes =
            bucketCores * S * bucketCap * 2ULL * sizeof(int32_t);

        const uint64_t wsBytes = 32UL * 1024 * 1024 + bucketCores * 32 +
            bucketCores * S * sizeof(int32_t) + bucketBytes +
            1UL * 1024 * 1024;
        if (bucketBytes <= 128UL * 1024 * 1024 && S >= 1 && S <= 256 &&
            sliceRows >= 1 && bucketCap <= 1024) {
            tiling->useBucket = 1;
            tiling->numSlices = S;
            tiling->sliceRows = sliceRows;
            tiling->sliceShift = shift;
            tiling->bucketCap = bucketCap;
            tiling->bucketCores = bucketCores;
            tiling->blockDim = static_cast<uint32_t>(bucketCores);
            tiling->singleSubblock = dt == ge::DT_FLOAT16 ? 1U : 0U;
            context->SetBlockDim(tiling->blockDim);
            context->GetWorkspaceSizes(1)[0] = wsBytes;
            return ge::GRAPH_SUCCESS;
        }
    }

    const uint64_t usefulTasks = std::max<uint64_t>(1, tiling->taskCount);
    tiling->blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(cores, usefulTasks));
    const uint32_t launchDim = dt == ge::DT_FLOAT16 ?
        (tiling->blockDim + 1U) / 2U : tiling->blockDim;
    context->SetBlockDim(launchDim);

    const uint64_t wsSync =
        16UL * 1024 * 1024 + static_cast<uint64_t>(tiling->blockDim) * 32 + 32;
    context->GetWorkspaceSizes(1)[0] =
        dt == ge::DT_FLOAT16 ? std::max<uint64_t>(wsSync, 4096UL) : wsSync;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus IndexAddInferShape(gert::InferShapeContext* context)
{
    const gert::Shape* self = context->GetInputShape(0);
    gert::Shape* output = context->GetOutputShape(0);
    if (self == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    *output = *self;
    return GRAPH_SUCCESS;
}

static ge::graphStatus IndexAddInferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char* name) : OpDef(name)
    {
        const std::initializer_list<ge::DataType> types = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8};
        const std::initializer_list<ge::DataType> indexTypes = {
            ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32};
        const std::initializer_list<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("self").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("index").ParamType(REQUIRED).DataType(indexTypes)
            .Format(formats).UnknownShapeFormat(formats);
        this->Input("source").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Output("output").ParamType(REQUIRED).DataType(types)
            .Format(formats).UnknownShapeFormat(formats);
        this->Attr("dim").AttrType(OPTIONAL).Int(0);
        this->SetInferShape(ge::IndexAddInferShape)
            .SetInferDataType(ge::IndexAddInferDataType);
        this->AICore().SetTiling(optiling::IndexAddTiling).AddConfig("ascend910b");
    }
};

OP_ADD(IndexAdd);
}
