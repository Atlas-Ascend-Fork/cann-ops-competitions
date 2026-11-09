
#include <cstdint>

#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kDefaultBlockDim = 32;
constexpr uint64_t kSharedPlanHeaderWords = 32;
constexpr uint64_t kSharedPlanWaveSlack = 64;
constexpr uint64_t kBatchedSharedWaveCapacity = 64;

static uint64_t AlignPlanWords(uint64_t words)
{
    return (words + 7U) & ~7ULL;
}

static uint64_t CalcSharedPlanWorkspaceBytes(
    uint32_t blockDim, uint64_t dimSize, uint64_t indexSize)
{
    const uint64_t alignedDim = (dimSize + 15U) & ~15ULL;
    const uint64_t alignedIndex = AlignPlanWords(indexSize);
    const uint64_t maxLocalIndex =
        (indexSize + static_cast<uint64_t>(blockDim) - 1U) /
        static_cast<uint64_t>(blockDim);
    const uint64_t maxOccupiedDims =
        maxLocalIndex < dimSize ? maxLocalIndex : dimSize;
    const uint64_t localPackedWords =
        (maxOccupiedDims * 8U + 15U) & ~15ULL;
    const uint64_t packedWords =
        static_cast<uint64_t>(blockDim) * localPackedWords;
    const uint64_t waveWords =
        alignedIndex + kSharedPlanWaveSlack * alignedDim;
    const uint64_t workspaceWords =
        kSharedPlanHeaderWords +
        static_cast<uint64_t>(blockDim) * alignedDim +
        alignedDim +
        static_cast<uint64_t>(blockDim) * alignedDim +
        packedWords + waveWords;
    return workspaceWords * sizeof(uint32_t);
}

static uint64_t CalcBatchedPlanWorkspaceBytes(
    uint32_t blockDim, uint64_t dimSize, uint64_t indexSize)
{
    constexpr uint64_t cacheFlagStride = 8;
    const uint64_t alignedIndex = AlignPlanWords(indexSize);
    const uint64_t alignedDim = AlignPlanWords(dimSize);
    const uint64_t alignedGroupCount = AlignPlanWords(dimSize / 8U);
    const uint64_t maxLocalIndex =
        (indexSize + static_cast<uint64_t>(blockDim) - 1U) /
        static_cast<uint64_t>(blockDim);
    const uint64_t maxOccupiedDims =
        maxLocalIndex < dimSize ? maxLocalIndex : dimSize;
    const uint64_t localPackedWords =
        (maxOccupiedDims * 8U + 15U) & ~15ULL;
    const uint64_t workspaceWords =
        kSharedPlanHeaderWords + 2U * alignedIndex +
        alignedGroupCount +
        static_cast<uint64_t>(blockDim) * cacheFlagStride +
        2U * static_cast<uint64_t>(blockDim) * alignedDim +
        static_cast<uint64_t>(blockDim) * localPackedWords +
        kBatchedSharedWaveCapacity * alignedDim;
    return workspaceWords * sizeof(uint32_t);
}

static int64_t GetDimAttr(const gert::RuntimeAttrs* attrs)
{
    if (attrs == nullptr) {
        return 0;
    }
    const int64_t* dim = attrs->GetAttrPointer<int64_t>(0);
    return dim == nullptr ? 0 : *dim;
}

static bool NormalizeDim(int64_t dimAttr, uint32_t rank, uint32_t* dim)
{
    if (rank == 0 || dim == nullptr) {
        return false;
    }
    if (dimAttr < 0) {
        dimAttr += static_cast<int64_t>(rank);
    }
    if (dimAttr < 0 || dimAttr >= static_cast<int64_t>(rank)) {
        return false;
    }
    *dim = static_cast<uint32_t>(dimAttr);
    return true;
}

static uint64_t CalcRangeSize(const gert::Shape& shape, uint32_t begin, uint32_t end)
{
    uint64_t size = 1;
    for (uint32_t i = begin; i < end; ++i) {
        const int64_t dimValue = shape.GetDim(i);
        if (dimValue < 0) {
            return 0;
        }
        size *= static_cast<uint64_t>(dimValue);
    }
    return size;
}

static uint64_t CalcTotalSize(const gert::Shape& shape)
{
    return CalcRangeSize(shape, 0, static_cast<uint32_t>(shape.GetDimNum()));
}

static uint32_t ToU32(uint64_t value)
{
    return static_cast<uint32_t>(value);
}

static uint32_t CalcBlockDim(uint64_t totalSize, uint32_t maxBlockDim)
{
    if (totalSize == 0) {
        return 1;
    }
    if (maxBlockDim == 0) {
        maxBlockDim = kDefaultBlockDim;
    }
    if (totalSize < maxBlockDim) {
        return ToU32(totalSize);
    }
    return maxBlockDim;
}

static bool CheckSourceShape(const gert::Shape& selfShape, const gert::Shape& sourceShape, uint32_t rank,
    uint32_t dim, uint64_t indexSize)
{
    if (static_cast<uint32_t>(sourceShape.GetDimNum()) != rank) {
        return false;
    }
    for (uint32_t i = 0; i < rank; ++i) {
        const int64_t sourceDim = sourceShape.GetDim(i);
        if (sourceDim < 0) {
            return false;
        }
        if (i == dim) {
            if (static_cast<uint64_t>(sourceDim) != indexSize) {
                return false;
            }
        } else if (sourceDim != selfShape.GetDim(i)) {
            return false;
        }
    }
    return true;
}
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* selfStorageShape = context->GetInputShape(0);
    const gert::StorageShape* indexStorageShape = context->GetInputShape(1);
    const gert::StorageShape* sourceStorageShape = context->GetInputShape(2);
    if (selfStorageShape == nullptr || indexStorageShape == nullptr || sourceStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& selfShape = selfStorageShape->GetStorageShape();
    const gert::Shape& indexShape = indexStorageShape->GetStorageShape();
    const gert::Shape& sourceShape = sourceStorageShape->GetStorageShape();
    const uint32_t rank = static_cast<uint32_t>(selfShape.GetDimNum());
    uint32_t dim = 0;
    if (!NormalizeDim(GetDimAttr(context->GetAttrs()), rank, &dim)) {
        return ge::GRAPH_FAILED;
    }
    if (indexShape.GetDimNum() != 1 || indexShape.GetDim(0) < 0 || selfShape.GetDim(dim) < 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t indexSize = static_cast<uint64_t>(indexShape.GetDim(0));
    if (!CheckSourceShape(selfShape, sourceShape, rank, dim, indexSize)) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t outerSize = CalcRangeSize(selfShape, 0, dim);
    const uint64_t innerSize = CalcRangeSize(selfShape, dim + 1, rank);
    const uint64_t dimSize = static_cast<uint64_t>(selfShape.GetDim(dim));
    const uint64_t totalSize = CalcTotalSize(selfShape);
    const uint64_t lineSize = outerSize * innerSize;
    const platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint64_t systemWorkspaceBytes =
        platform.GetLibApiWorkSpaceSize();
    uint32_t maxBlockDim = platform.GetCoreNumAiv();
    if (maxBlockDim == 0) {
        maxBlockDim = kDefaultBlockDim;
    }
    uint32_t blockDim = CalcBlockDim(totalSize, maxBlockDim);
    const auto* selfDesc = context->GetInputDesc(0);
    uint64_t workspaceBytes = 0;
    const bool atomicDataType = selfDesc != nullptr &&
        (selfDesc->GetDataType() == ge::DT_FLOAT ||
         selfDesc->GetDataType() == ge::DT_INT32);
    const bool groupedVectorShape =
        innerSize >= 8 && outerSize < 32 && indexSize >= 512 &&
        indexSize < 2048 && totalSize >= 4096;
    const bool scalarAtomicShape =
        innerSize == 1 && outerSize >= 32 && outerSize <= 256 &&
        dimSize > 512 && dimSize < 1024 && indexSize >= 2048 &&
        totalSize >= 4096;
    const bool batchedScalarAtomicShape =
        innerSize == 1 && outerSize > 256 && dimSize >= 64 &&
        dimSize < 512 && dimSize % 8 == 0 && indexSize >= 512 &&
        totalSize >= 4096;
    const bool useBatchedTargetGroup = selfDesc != nullptr &&
        selfDesc->GetDataType() == ge::DT_INT32 &&
        batchedScalarAtomicShape && indexSize < 2048;
    const bool directScalarAtomicShape =
        atomicDataType && dimSize % 8 == 0 &&
        (scalarAtomicShape || batchedScalarAtomicShape);
    const bool useAtomicVector = atomicDataType && groupedVectorShape &&
        innerSize % 8 == 0 && dimSize <= 65535;
    const bool useGroupedPaddedAtomic = atomicDataType &&
        groupedVectorShape && innerSize % 8 != 0 && dimSize <= 65535;
    const bool useParallelSharedPlan = selfDesc != nullptr &&
        selfDesc->GetDataType() == ge::DT_FLOAT && innerSize == 1 &&
        outerSize >= 512 && dimSize > 512 && dimSize < 1024 &&
        indexSize >= 4096 && indexSize <= 16384 && totalSize >= 4096;
    const bool useHalfBucketCache = selfDesc != nullptr &&
        selfDesc->GetDataType() == ge::DT_FLOAT16 && innerSize == 1 &&
        outerSize >= 2048 && dimSize >= 64 && dimSize < 512 &&
        indexSize >= 512 && indexSize < 2048 && totalSize >= 4096;
    if (selfDesc != nullptr && blockDim > kDefaultBlockDim) {
        const bool useCappedGroupedVector =
            atomicDataType &&
            groupedVectorShape && !useAtomicVector && !useGroupedPaddedAtomic;
        const ge::DataType dataType = selfDesc->GetDataType();
        if (dataType == ge::DT_INT8 || useCappedGroupedVector) {
            blockDim = kDefaultBlockDim;
        }
    }
    // GM scalar int8 stores can update an aligned transfer unit.  Use the
    // multi-core row-atomic route only for a general family whose complete rows
    // are transfer-unit aligned and whose two UB buffers retain explicit slack.
    // Other int8 shapes use the conservative single-core fallback.
    if (selfDesc != nullptr && selfDesc->GetDataType() == ge::DT_INT8) {
        constexpr uint64_t kInt8RouteUbBudgetBytes = 188416;
        const uint64_t alignedIndexBytes =
            (indexSize * sizeof(int32_t) + 31U) & ~31ULL;
        const uint64_t alignedRowBytes = (innerSize + 31U) & ~31ULL;
        const bool useAlignedInt8AtomicRows = outerSize == 1 && dimSize > 1 &&
            indexSize > 0 && innerSize >= 32 && innerSize % 32U == 0 &&
            alignedIndexBytes + alignedRowBytes <= kInt8RouteUbBudgetBytes;
        if (!useAlignedInt8AtomicRows) {
            blockDim = 1;
        }
    }
    const bool useDirectPaddedAtomic = useGroupedPaddedAtomic && atomicDataType;
    if (useGroupedPaddedAtomic && !useDirectPaddedAtomic) {
        const uint64_t paddedInner = ((innerSize + 7) / 8) * 8;
        workspaceBytes =
            outerSize * dimSize * paddedInner * sizeof(uint32_t);
    }
    if (directScalarAtomicShape && batchedScalarAtomicShape) {
        blockDim = maxBlockDim;
    } else if (directScalarAtomicShape) {
        blockDim = maxBlockDim < kDefaultBlockDim ? maxBlockDim : kDefaultBlockDim;
    } else if (scalarAtomicShape) {
        blockDim = 4;
    }
    if (useBatchedTargetGroup) {
        workspaceBytes = CalcBatchedPlanWorkspaceBytes(
            blockDim, dimSize, indexSize);
    }
    if (useParallelSharedPlan) {
        blockDim = maxBlockDim;
        workspaceBytes = CalcSharedPlanWorkspaceBytes(
            blockDim, dimSize, indexSize);
    }
    if (useHalfBucketCache) {
        blockDim = maxBlockDim;
        // Header + cached index + per-core verification flags + one full
        // UB-sized count-bucket plan.  Keep explicit GM headroom.
        workspaceBytes = 262144U;
        // V440_PRIVATE_HOST_BEGIN: isolated slots only for the FP16 family.
        if (dimSize >= 128U && blockDim > 0U && blockDim <= 65535U) {
            const uint64_t privateBytes = static_cast<uint64_t>(blockDim) * 65536ULL;
            if (privateBytes > workspaceBytes) {
                workspaceBytes = privateBytes;
            }
        }
        // V440_PRIVATE_HOST_END
    }
    const bool useBfloatWaveCache = selfDesc != nullptr &&
        selfDesc->GetDataType() == ge::DT_BF16 && outerSize > 1U &&
        outerSize <= blockDim && dimSize >= 32U && dimSize <= 255U &&
        indexSize > 0U && indexSize <= 4096U && innerSize > 1U;
    if (useBfloatWaveCache) {
        // Verified index copy, metadata, per-core flags and sixteen wave rows.
        workspaceBytes = 65536U;
    }

    IndexAddTilingData tiling;
    tiling.set_totalSize(ToU32(totalSize));
    tiling.set_blockSize(ToU32((lineSize + blockDim - 1) / blockDim));
    tiling.set_outerSize(ToU32(outerSize));
    tiling.set_dimSize(ToU32(dimSize));
    tiling.set_indexSize(ToU32(indexSize));
    tiling.set_innerSize(ToU32(innerSize));

    context->SetBlockDim(blockDim);
    auto* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] =
        workspaceBytes == 0 ? 0 : systemWorkspaceBytes + workspaceBytes;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* selfShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (selfShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    *outputShape = *selfShape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("source")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(IndexAdd);
}
