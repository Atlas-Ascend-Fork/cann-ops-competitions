#include <algorithm>
#include <cstdint>

#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kDataBlockBytes = 32U;
// Match the compiler-visible UB capacity in the Ascend 910B4 platform profile.
constexpr uint32_t kUsableUbBytes = 192U * 1024U;
// C220 Gather profiling shows severe bank-group conflicts when source-row
// strides repeat a small set of 512-byte UB address classes. Dephase a
// resonant stride by one data block before aligning work to its vector group.
constexpr uint32_t kGatherConflictStrideBytes = 512U;
// C220 DataCopyPad encodes blockCount in a 12-bit instruction field.
constexpr uint32_t kMaxDataCopyBlocks = 4095U;
constexpr uint32_t kMaxVectorRepeats = 255U;
constexpr uint32_t kTransposeTileElements = 16U;
constexpr uint32_t kGatherPipelineDepth = 6U;
constexpr uint32_t kBfloatPipelineDepth = 4U;
constexpr uint32_t kBfloatRawBufferCount = 2U;
constexpr uint32_t kAtomicRawBufferCount = 2U;
constexpr uint32_t kOuterSourcePipelineDepth = 2U;
constexpr uint32_t kBfloatTargetChunks = 12U;
// One vnchwconv address-table setup costs roughly as much scalar work as 32 Gathers.
constexpr uint32_t kTransposeSetupWeight = 32U;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1U) / alignment * alignment;
}

uint64_t SaturatingMultiply(uint64_t lhs, uint64_t rhs)
{
    return lhs != 0U && rhs > UINT64_MAX / lhs ? UINT64_MAX : lhs * rhs;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* selfStorageShape = context->GetInputShape(0);
    const auto* indexStorageShape = context->GetInputShape(1);
    const auto* sourceStorageShape = context->GetInputShape(2);
    const auto* attrs = context->GetAttrs();
    const auto* selfDesc = context->GetInputDesc(0);
    if (selfStorageShape == nullptr || indexStorageShape == nullptr ||
        sourceStorageShape == nullptr || attrs == nullptr || selfDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& selfShape = selfStorageShape->GetStorageShape();
    const auto& indexShape = indexStorageShape->GetStorageShape();
    const auto& sourceShape = sourceStorageShape->GetStorageShape();
    const uint32_t rank = selfShape.GetDimNum();
    const int64_t* dimAttr = attrs->GetAttrPointer<int64_t>(0);
    if (rank == 0U || dimAttr == nullptr || indexShape.GetDimNum() != 1U ||
        sourceShape.GetDimNum() != rank) {
        return ge::GRAPH_FAILED;
    }

    int64_t dim = *dimAttr;
    if (dim < 0) {
        dim += static_cast<int64_t>(rank);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t outerSize = 1U;
    uint64_t innerSize = 1U;
    for (int64_t axis = 0; axis < dim; ++axis) {
        const int64_t extent = selfShape.GetDim(static_cast<uint32_t>(axis));
        if (extent <= 0) {
            return ge::GRAPH_FAILED;
        }
        outerSize *= static_cast<uint64_t>(extent);
    }
    for (uint32_t axis = static_cast<uint32_t>(dim + 1); axis < rank; ++axis) {
        const int64_t extent = selfShape.GetDim(axis);
        if (extent <= 0) {
            return ge::GRAPH_FAILED;
        }
        innerSize *= static_cast<uint64_t>(extent);
    }

    const int64_t axisSize64 = selfShape.GetDim(static_cast<uint32_t>(dim));
    const int64_t indexSize64 = indexShape.GetDim(0);
    if (axisSize64 <= 0 || indexSize64 <= 0 || outerSize == 0U || innerSize == 0U ||
        sourceShape.GetDim(static_cast<uint32_t>(dim)) != indexSize64) {
        return ge::GRAPH_FAILED;
    }

    uint32_t typeBytes = 0U;
    const ge::DataType dtype = selfDesc->GetDataType();
    if (dtype == ge::DT_INT8) {
        typeBytes = 1U;
    } else if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        typeBytes = 2U;
    } else if (dtype == ge::DT_INT32 || dtype == ge::DT_FLOAT) {
        typeBytes = 4U;
    } else {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t availableCores = std::max<uint32_t>(1U, platform.GetCoreNumAiv());
    const uint32_t targetCores = dtype == ge::DT_BF16
        ? std::max<uint32_t>(1U, availableCores * 2U / 3U)
        : availableCores;
    const bool useAtomicAdd = dtype == ge::DT_INT32 && innerSize > 1U &&
        outerSize < availableCores && indexSize64 >= availableCores &&
        indexSize64 > axisSize64;
    uint32_t targetTiles = 1U;
    if (outerSize < targetCores) {
        targetTiles = static_cast<uint32_t>(targetCores / outerSize);
        targetTiles = std::max<uint32_t>(1U, std::min<uint64_t>(targetTiles, innerSize));
    }

    uint64_t tileElements64 = (innerSize + targetTiles - 1U) / targetTiles;
    if (dtype == ge::DT_BF16) {
        tileElements64 = AlignUp(tileElements64, kDataBlockBytes / typeBytes);
    }
    if (tileElements64 == 0U || tileElements64 > UINT32_MAX ||
        axisSize64 > kMaxDataCopyBlocks || indexSize64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t tileElements = static_cast<uint32_t>(tileElements64);
    const uint32_t tilesPerOuter =
        static_cast<uint32_t>((innerSize + tileElements - 1U) / tileElements);

    const uint64_t rawRowBytes = AlignUp(tileElements64 * typeBytes, kDataBlockBytes);
    const uint64_t indexBytes = AlignUp(static_cast<uint64_t>(indexSize64) * sizeof(int32_t),
                                        kDataBlockBytes);
    const bool emulateBfloat16Rounding = dtype == ge::DT_BF16;
    const uint64_t outputRowBytes = rawRowBytes;

    uint32_t sourceChunk = 1U;
    uint32_t outerBatch = 1U;
    uint32_t batchAxisChunk = 1U;
    uint32_t sourceTranspose = 0U;
    uint32_t sourceBufferCount = 1U;
    const bool canBatchOuter = innerSize == 1U && dtype != ge::DT_INT8 &&
                               dtype != ge::DT_BF16 && indexBytes < kUsableUbBytes;
    if (canBatchOuter) {
        const uint64_t maxBatchForCores =
            (outerSize + availableCores - 1U) / availableCores;
        const uint64_t maxCandidateBatch = std::min<uint64_t>(
            {outerSize, maxBatchForCores, static_cast<uint64_t>(kMaxVectorRepeats)});
        const uint64_t axisOffsetBytes =
            AlignUp(static_cast<uint64_t>(axisSize64) * sizeof(uint32_t), kDataBlockBytes);
        uint64_t bestWaves = UINT64_MAX;
        for (uint64_t candidateBatch = 2U;
             candidateBatch <= maxCandidateBatch; ++candidateBatch) {
            // The device allocates every staging row with the same aligned
            // batch extent used by its Gather vectors.  Account for that
            // aligned extent here as well; sizing with candidateBatch alone
            // can overcommit UB whenever the batch needs data-block padding.
            const uint64_t stagingBatch = AlignUp(
                candidateBatch, kDataBlockBytes / typeBytes);
            const uint64_t effectiveBatch = dtype == ge::DT_FLOAT
                ? candidateBatch : std::min<uint64_t>(outerSize, stagingBatch);
            const uint64_t batchVectorBytes = stagingBatch * typeBytes;
            const uint64_t candidateBufferCount = dtype == ge::DT_FLOAT16
                ? kOuterSourcePipelineDepth : 1U;
            const uint64_t outerOffsetBytes =
                AlignUp(effectiveBatch * sizeof(uint32_t), kDataBlockBytes);
            const uint64_t outputBytes =
                static_cast<uint64_t>(axisSize64) * batchVectorBytes;
            const uint64_t gatherValueBytes =
                (dtype == ge::DT_FLOAT ? 1U : kGatherPipelineDepth) * batchVectorBytes;
            const uint64_t fixedBytes = indexBytes + outputBytes + gatherValueBytes +
                2U * outerOffsetBytes + axisOffsetBytes;
            if (fixedBytes + candidateBufferCount * stagingBatch * kDataBlockBytes >
                kUsableUbBytes) {
                continue;
            }
            const uint64_t stagingRowBytes =
                ((kUsableUbBytes - fixedBytes) /
                 (candidateBufferCount * stagingBatch) / kDataBlockBytes) *
                kDataBlockBytes;
            const uint64_t stagingElements = stagingRowBytes / typeBytes;
            uint64_t candidateSourceChunk =
                std::min<uint64_t>(indexSize64, stagingElements);
            const uint64_t sourceRowBytes =
                AlignUp(candidateSourceChunk * typeBytes, kDataBlockBytes);
            const uint64_t dataBlockElements = kDataBlockBytes / typeBytes;
            const bool repeatsGatherAddressClass =
                sourceRowBytes % kGatherConflictStrideBytes == 0U ||
                (candidateBufferCount > 1U &&
                 kGatherConflictStrideBytes % sourceRowBytes == 0U);
            if (repeatsGatherAddressClass &&
                candidateSourceChunk > dataBlockElements) {
                candidateSourceChunk -= dataBlockElements;
            }
            if (candidateBufferCount > 1U &&
                candidateSourceChunk >= kGatherPipelineDepth) {
                candidateSourceChunk = candidateSourceChunk /
                    kGatherPipelineDepth * kGatherPipelineDepth;
            }
            const uint64_t candidateAxisChunk =
                std::min<uint64_t>(axisSize64,
                                   candidateBufferCount * stagingElements);
            if (candidateSourceChunk == 0U || candidateAxisChunk == 0U) {
                continue;
            }
            const uint64_t taskCount =
                (outerSize + effectiveBatch - 1U) / effectiveBatch;
            const uint64_t waves =
                (taskCount + availableCores - 1U) / availableCores;
            if (waves < bestWaves ||
                (waves == bestWaves && effectiveBatch < outerBatch)) {
                bestWaves = waves;
                outerBatch = static_cast<uint32_t>(effectiveBatch);
                sourceChunk = static_cast<uint32_t>(candidateSourceChunk);
                batchAxisChunk = static_cast<uint32_t>(candidateAxisChunk);
                sourceBufferCount = static_cast<uint32_t>(candidateBufferCount);
            }
        }

        const uint64_t transposeRowElements = typeBytes == sizeof(float)
            ? kTransposeTileElements / 2U : kTransposeTileElements;
        uint64_t transposeBestWaves = UINT64_MAX;
        uint64_t transposeBestCalls = UINT64_MAX;
        uint32_t transposeBatch = 1U;
        uint32_t transposeSourceChunk = 1U;
        uint32_t transposeAxisChunk = 1U;
        const uint64_t transposeAxisOffsetBytes = AlignUp(
            std::min<uint64_t>(static_cast<uint64_t>(axisSize64),
                               kTransposeTileElements) * sizeof(uint32_t),
            kDataBlockBytes);
        for (uint64_t candidateBatch = 2U;
             candidateBatch <= maxCandidateBatch; ++candidateBatch) {
            const uint64_t stagingBatch = AlignUp(candidateBatch, kTransposeTileElements);
            const uint64_t effectiveBatch = dtype == ge::DT_FLOAT
                ? candidateBatch : std::min<uint64_t>(outerSize, stagingBatch);
            const uint64_t rawBufferCount = dtype == ge::DT_FLOAT &&
                    (static_cast<uint64_t>(indexSize64) % transposeRowElements) == 0U
                ? kOuterSourcePipelineDepth : 1U;
            const uint64_t transposeBufferCount = rawBufferCount + 1U;
            const uint64_t transposeGroups = stagingBatch / kTransposeTileElements;
            const uint64_t transposeConfigBytes = rawBufferCount > 1U
                ? (rawBufferCount + 1U) * transposeGroups * 4U * sizeof(uint64_t)
                : 0U;
            const uint64_t batchVectorBytes = stagingBatch * typeBytes;
            const uint64_t outerOffsetBytes =
                AlignUp(effectiveBatch * sizeof(uint32_t), kDataBlockBytes);
            const uint64_t outputRows = dtype == ge::DT_FLOAT
                ? AlignUp(static_cast<uint64_t>(axisSize64), kTransposeTileElements)
                : static_cast<uint64_t>(axisSize64);
            const uint64_t outputBytes = outputRows * batchVectorBytes;
            const uint64_t fixedBytes = indexBytes + outputBytes + batchVectorBytes +
                2U * outerOffsetBytes + transposeAxisOffsetBytes + transposeConfigBytes;
            if (fixedBytes + transposeBufferCount * stagingBatch * kDataBlockBytes >
                kUsableUbBytes) {
                continue;
            }
            const uint64_t stagingRowBytes =
                ((kUsableUbBytes - fixedBytes) /
                 (transposeBufferCount * stagingBatch) / kDataBlockBytes) *
                kDataBlockBytes;
            const uint64_t stagingElements = stagingRowBytes / typeBytes;
            if (stagingElements < transposeRowElements) {
                continue;
            }
            uint64_t candidateSourceChunk = std::min<uint64_t>(
                {static_cast<uint64_t>(indexSize64), stagingElements,
                 transposeRowElements * kMaxVectorRepeats});
            candidateSourceChunk =
                candidateSourceChunk / transposeRowElements * transposeRowElements;
            const uint64_t candidateAxisChunk =
                std::min<uint64_t>(axisSize64, rawBufferCount * stagingElements);
            uint64_t candidateTransposeAxisChunk = candidateAxisChunk;
            if (dtype == ge::DT_FLOAT && candidateTransposeAxisChunk >= kTransposeTileElements) {
                candidateTransposeAxisChunk = candidateTransposeAxisChunk / kTransposeTileElements *
                    kTransposeTileElements;
            }
            if (candidateSourceChunk == 0U || candidateTransposeAxisChunk == 0U) {
                continue;
            }
            const uint64_t transposeCalls =
                ((static_cast<uint64_t>(indexSize64) + candidateSourceChunk - 1U) /
                 candidateSourceChunk) * (stagingBatch / kTransposeTileElements);
            if (transposeCalls * kTransposeSetupWeight >=
                static_cast<uint64_t>(indexSize64)) {
                continue;
            }
            const uint64_t taskCount =
                (outerSize + effectiveBatch - 1U) / effectiveBatch;
            const uint64_t waves =
                (taskCount + availableCores - 1U) / availableCores;
            if (waves < transposeBestWaves ||
                (waves == transposeBestWaves &&
                 (transposeCalls < transposeBestCalls ||
                  (transposeCalls == transposeBestCalls && effectiveBatch > transposeBatch)))) {
                transposeBestWaves = waves;
                transposeBestCalls = transposeCalls;
                transposeBatch = static_cast<uint32_t>(effectiveBatch);
                transposeSourceChunk = static_cast<uint32_t>(candidateSourceChunk);
                transposeAxisChunk = static_cast<uint32_t>(candidateTransposeAxisChunk);
            }
        }
        const uint64_t gatherCriticalWork =
            SaturatingMultiply(bestWaves, static_cast<uint64_t>(indexSize64));
        const uint64_t transposeCriticalWork = SaturatingMultiply(
            SaturatingMultiply(transposeBestWaves, transposeBestCalls),
            kTransposeSetupWeight);
        if (transposeBestWaves != UINT64_MAX &&
            (transposeBestWaves <= bestWaves ||
             transposeCriticalWork < gatherCriticalWork)) {
            outerBatch = transposeBatch;
            sourceChunk = transposeSourceChunk;
            batchAxisChunk = transposeAxisChunk;
            sourceTranspose = 1U;
            sourceBufferCount = dtype == ge::DT_FLOAT &&
                    (static_cast<uint64_t>(indexSize64) % transposeRowElements) == 0U
                ? kOuterSourcePipelineDepth : 1U;
        }
    }

    if (useAtomicAdd) {
        const uint64_t atomicRowBytes =
            AlignUp(innerSize * typeBytes, kDataBlockBytes);
        const uint64_t atomicAlignElements = kDataBlockBytes / typeBytes;
        const bool useAtomicPairStream =
            (innerSize % atomicAlignElements) * 2U == atomicAlignElements;
        const uint64_t atomicStorageRowBytes = atomicRowBytes +
            (useAtomicPairStream
                 ? kDataBlockBytes / 2U + atomicRowBytes / 2U : 0U);
        if (atomicStorageRowBytes + indexBytes > kUsableUbBytes) {
            return ge::GRAPH_FAILED;
        }
        sourceChunk = static_cast<uint32_t>(std::min<uint64_t>(
            {static_cast<uint64_t>(indexSize64),
             static_cast<uint64_t>(kMaxDataCopyBlocks),
             (kUsableUbBytes - indexBytes) /
                 (kAtomicRawBufferCount * atomicStorageRowBytes)}));
        if (useAtomicPairStream && sourceChunk > 1U) {
            sourceChunk &= ~1U;
        }
    } else if (outerBatch == 1U) {
        const uint64_t outputBytes = static_cast<uint64_t>(axisSize64) * outputRowBytes;
        if (outputBytes + indexBytes >= kUsableUbBytes) {
            return ge::GRAPH_FAILED;
        }
        if (emulateBfloat16Rounding) {
            const uint64_t floatRowBytes =
                AlignUp(tileElements64 * sizeof(float), kDataBlockBytes);
            const uint64_t baseBytes = outputBytes + indexBytes +
                kBfloatPipelineDepth * floatRowBytes;
            const uint64_t bytesPerSourceRow =
                kBfloatRawBufferCount * rawRowBytes + floatRowBytes;
            if (baseBytes + bytesPerSourceRow > kUsableUbBytes) {
                return ge::GRAPH_FAILED;
            }
            uint64_t candidateSourceChunk = std::min<uint64_t>(
                {static_cast<uint64_t>(indexSize64),
                 static_cast<uint64_t>(kMaxDataCopyBlocks),
                 (kUsableUbBytes - baseBytes) / bytesPerSourceRow});
            if (candidateSourceChunk < static_cast<uint64_t>(indexSize64)) {
                candidateSourceChunk = std::min<uint64_t>(
                    candidateSourceChunk,
                    (static_cast<uint64_t>(indexSize64) + kBfloatTargetChunks - 1U) /
                        kBfloatTargetChunks);
            }
            sourceChunk = static_cast<uint32_t>(
                std::max<uint64_t>(1U, candidateSourceChunk));
        } else {
            sourceChunk = static_cast<uint32_t>(std::max<uint64_t>(
                1U, std::min<uint64_t>(
                        {static_cast<uint64_t>(indexSize64),
                         static_cast<uint64_t>(kMaxDataCopyBlocks),
                         (kUsableUbBytes - outputBytes - indexBytes) / rawRowBytes})));
        }
    }

    const uint64_t totalTasks = useAtomicAdd
        ? SaturatingMultiply(outerSize, static_cast<uint64_t>(indexSize64))
        : outerBatch > 1U
        ? (outerSize + outerBatch - 1U) / outerBatch
        : outerSize * tilesPerOuter;
    const uint32_t blockDim =
        static_cast<uint32_t>(std::min<uint64_t>(availableCores, totalTasks));

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0U;

    IndexAddTilingData tiling;
    tiling.set_outerSize(outerSize);
    tiling.set_innerSize(innerSize);
    tiling.set_axisSize(static_cast<uint32_t>(axisSize64));
    tiling.set_indexSize(static_cast<uint32_t>(indexSize64));
    tiling.set_tilesPerOuter(tilesPerOuter);
    tiling.set_tileElements(tileElements);
    tiling.set_sourceChunk(sourceChunk);
    tiling.set_outerBatch(outerBatch);
    tiling.set_batchAxisChunk(batchAxisChunk);
    tiling.set_sourceTranspose(sourceTranspose);
    tiling.set_sourceBufferCount(sourceBufferCount);
    tiling.set_atomicAdd(useAtomicAdd ? 1U : 0U);
    tiling.set_typeBytes(typeBytes);
    tiling.set_promoteBfloat16(emulateBfloat16Rounding ? 1U : 0U);
    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

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

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("source")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT8, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IndexAdd);
}  // namespace ops
