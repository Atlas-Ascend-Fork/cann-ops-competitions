// Host responsibilities: validate shapes/dtypes, select an existing route,
// size its UB allocations, then serialize the matching kernel contract.
// Existing tuned routes retain their priority and budgets. Linear DMA admission
// is recorded once here; its generic kernel path does not rescan all widths.
#include "concat_tiling.h"
#include "concat_block_pack_plan.h"
#include "concat_asym_pack_plan.h"
#include "concat_slice_pack_plan.h"
#include "concat_input_stream_plan.h"
#include "concat_row_phase_prefix_plan.h"
#include "../op_kernel/concat_linear_route.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kMaxConcatInputs = 1792U;
constexpr uint32_t kAivCoreNum910B = 40U;
constexpr uint32_t kWorkBlockBytes = concat_linear_route::kWorkBlockBytes;
constexpr uint32_t kUbBytes910B = 192U * 1024U;
constexpr uint32_t kUbReservedBytes = 8U * 1024U;
constexpr uint32_t kMetaCacheMinInputs = 64U;
constexpr uint32_t kMetaCacheMaxInputs = 256U;
constexpr uint32_t kPeriodicGatherAxisElems = 32U;
constexpr uint32_t kPeriodicGatherFlag = 0x80000000U;
constexpr uint32_t kContinuousB32Flag = 0x40000000U;
constexpr uint32_t kWideGeometryFlag = 0x02000000U;
constexpr uint32_t kCompositeGatherB32Flag = 0x20000000U;
constexpr uint32_t kPhaseOwnedB32Flag = kCompositeGatherB32Flag;
constexpr uint32_t kRouteDefault = 0U;
constexpr uint32_t kRouteContinuousB32 = 1U;
constexpr uint32_t kRouteCompositeGatherB32 = 2U;
constexpr uint32_t kRoutePhaseOwnedB32 = kRouteCompositeGatherB32;
constexpr uint32_t kContinuousMaxWidth = 128U;
constexpr uint32_t kContinuousGatherWidthMax = 82U;
constexpr uint32_t kContinuousBufferPlanes = 7U;
constexpr uint32_t kContinuousMetaPlanes = 5U;
constexpr uint32_t kMaxDataCopyBlocks = 4095U;
constexpr uint32_t kPhaseCount = 8U;
constexpr uint32_t kPhaseOwnerBits = 6U;
constexpr uint32_t kPhaseColumnMask = 0x1FFFFFFFU;
constexpr uint32_t kPhaseBoundarySourcePitch = 64U;

struct HotRouteSelection {
    uint32_t key = kRouteDefault;
    uint32_t scheduleCount = 0U;
    uint32_t boundaryCount = 0U;
    uint32_t unitCount = 0U;
    uint32_t ownerWordCount = 0U;
};

static uint32_t DataTypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            return 1U;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2U;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4U;
        default:
            return 0U;
    }
}

static uint32_t CeilDivU32(uint32_t x, uint32_t y)
{
    return x / y + (x % y == 0U ? 0U : 1U);
}

static uint32_t AlignDownU32(uint32_t x, uint32_t alignment)
{
    return (x / alignment) * alignment;
}

static uint32_t SelectBlockDim(uint32_t totalBlocks)
{
    if (totalBlocks == 0U) {
        return 1U;
    }
    return totalBlocks < kAivCoreNum910B ? totalBlocks : kAivCoreNum910B;
}

static uint32_t CalcTileElems(uint32_t elemBytes, uint32_t blockElems)
{
    uint32_t usableBytes = kUbBytes910B - kUbReservedBytes;
    // TQueBind shares each VECIN/VECOUT slot, so only the two pipeline slots consume UB.
    uint32_t tileBytes = usableBytes / 2U;
    tileBytes = AlignDownU32(tileBytes, kWorkBlockBytes);
    if (tileBytes < kWorkBlockBytes) {
        tileBytes = kWorkBlockBytes;
    }
    uint32_t tileElems = tileBytes / elemBytes;
    if (tileElems < blockElems) {
        tileElems = blockElems;
    }
    tileElems = AlignDownU32(tileElems, blockElems);
    if (tileElems == 0U) {
        tileElems = blockElems;
    }
    return tileElems;
}

static uint32_t FindPeriodicGatherPattern(uint32_t elemBytes, uint32_t inputCount,
                                          uint32_t outerSize, uint32_t innerSize,
                                          uint32_t outputAxisSize, uint32_t blockDim,
                                          const uint32_t (&axisLens)[kMaxConcatInputs])
{
    if (elemBytes != 4U || innerSize != 1U || outerSize == 0U || outerSize > 4095U ||
        inputCount < kMetaCacheMinInputs || inputCount > kMetaCacheMaxInputs) {
        return 0U;
    }

    uint32_t patternInputs = 0U;
    uint32_t patternAxis = 0U;
    while (patternInputs < inputCount && patternAxis < kPeriodicGatherAxisElems) {
        const uint32_t width = axisLens[patternInputs];
        if (width == 0U || width > kPeriodicGatherAxisElems - patternAxis ||
            ((static_cast<uint64_t>(outerSize) * width * elemBytes) & 31ULL) != 0ULL) {
            return 0U;
        }
        patternAxis += width;
        ++patternInputs;
    }
    if (patternAxis != kPeriodicGatherAxisElems || patternInputs == 0U ||
        inputCount % patternInputs != 0U) {
        return 0U;
    }

    const uint32_t groupCount = inputCount / patternInputs;
    if (groupCount * kPeriodicGatherAxisElems != outputAxisSize ||
        groupCount * 4U < blockDim * 3U) {
        return 0U;
    }
    for (uint32_t inputIdx = patternInputs; inputIdx < inputCount; ++inputIdx) {
        if (axisLens[inputIdx] != axisLens[inputIdx % patternInputs]) {
            return 0U;
        }
    }

    const uint64_t periodDataElems =
        static_cast<uint64_t>(outerSize) * kPeriodicGatherAxisElems;
    const uint32_t metaStride = CeilDivU32(inputCount, 8U) * 8U;
    const uint64_t requiredBytes =
        periodDataElems * (2U * elemBytes + 2U * sizeof(uint32_t)) +
        static_cast<uint64_t>(metaStride) * 4U * sizeof(uint32_t) + kUbReservedBytes;
    return requiredBytes <= kUbBytes910B ? patternInputs : 0U;
}

static uint32_t NextContinuousInput(
    bool wantGather, uint32_t inputCount,
    const uint32_t (&axisLens)[kMaxConcatInputs], uint32_t& cursor)
{
    while (cursor < inputCount) {
        const uint32_t inputIdx = cursor++;
        const uint32_t width = axisLens[inputIdx];
        if (width == 0U) {
            continue;
        }
        const bool useGather =
            (width & 7U) != 0U && width <= kContinuousGatherWidthMax;
        if (useGather == wantGather) {
            return inputIdx;
        }
    }
    return inputCount;
}

static HotRouteSelection SelectHotRoute(
    uint32_t elemBytes, uint32_t inputCount, uint32_t outerSize,
    uint32_t innerSize, uint32_t outputAxisSize, uint32_t blockDim,
    const uint32_t (&axisLens)[kMaxConcatInputs],
    uint32_t (&scheduleOrder)[kMaxConcatInputs])
{
    HotRouteSelection route;
    if (elemBytes != 4U || blockDim != kAivCoreNum910B || outerSize == 0U ||
        outputAxisSize == 0U ||
        outputAxisSize > UINT32_MAX / elemBytes) {
        return route;
    }

    const uint32_t rowsPerCore = CeilDivU32(outerSize, blockDim);
    if (rowsPerCore == 0U || rowsPerCore > kMaxDataCopyBlocks) {
        return route;
    }

    // Contiguous long-row input; gather eight words for full-row prefixes,
    // or only the required 0--7 alignment words for physically clipped rows.
    concat_row_phase_prefix::Plan rowPhase;
    if (concat_row_phase_prefix::Build(elemBytes, inputCount, outerSize, innerSize,
                                       outputAxisSize, blockDim, axisLens, rowPhase)) {
        route.key = 4U;
        route.boundaryCount = rowPhase.slotWords;
        route.unitCount = rowPhase.indexWords;
        return route;
    }

    concat_input_stream::Plan stream;
    if (concat_input_stream::Build(elemBytes, inputCount, outerSize, innerSize,
                                    outputAxisSize, blockDim, axisLens, stream)) {
        concat_input_stream::Serialize(stream, scheduleOrder);
        route.key = 5U;
        return route;
    }
    concat_slice_pack::Plan sliced;
    if (concat_slice_pack::Build(elemBytes, inputCount, outerSize, innerSize,
                                 outputAxisSize, blockDim, axisLens, sliced)) {
        concat_slice_pack::Serialize(sliced, scheduleOrder);
        route.key = 5U;
        return route;
    }
    concat_asym_pack::Plan asymmetric;
    if (concat_asym_pack::Build(elemBytes, inputCount, outerSize, innerSize,
                                outputAxisSize, blockDim, axisLens, asymmetric)) {
        concat_asym_pack::Serialize(asymmetric, scheduleOrder);
        route.key = 5U;
        return route;
    }
    concat_block_pack::Plan pack;
    if (concat_block_pack::Build(elemBytes, inputCount, outerSize, innerSize,
                                 outputAxisSize, blockDim, axisLens, pack, true)) {
        route.key = 3U;
        route.boundaryCount = pack.sourceElems;
        route.unitCount = pack.outputElems;
        route.ownerWordCount = pack.expandedElems;
        return route;
    }

    // Proven continuous small-width B32 data path.  Admission is now a complete generic
    // geometry/capacity proof performed once in host tiling; the kernel keeps
    // the verified queue depths, copy order and 40-way row ownership intact.
    if (innerSize == 1U && inputCount >= 65U && inputCount <= 256U &&
        outerSize >= 64U && outputAxisSize >= 512U) {
        uint32_t nonZeroInputs = 0U;
        bool widthsFit = true;
        for (uint32_t inputIdx = 0U; inputIdx < inputCount; ++inputIdx) {
            const uint32_t width = axisLens[inputIdx];
            if (width == 0U) {
                continue;
            }
            if (width > kContinuousMaxWidth) {
                widthsFit = false;
                break;
            }
            ++nonZeroInputs;
        }

        const uint64_t maxLocalElems =
            static_cast<uint64_t>(rowsPerCore) * kContinuousMaxWidth;
        const uint32_t metaStride = CeilDivU32(inputCount, 8U) * 8U;
        const uint64_t requiredBytes =
            maxLocalElems * elemBytes * kContinuousBufferPlanes +
            static_cast<uint64_t>(metaStride) * kContinuousMetaPlanes *
                sizeof(uint32_t);
        if (widthsFit && nonZeroInputs >= 2U &&
            maxLocalElems <= UINT32_MAX && requiredBytes <= kUbBytes910B) {
            uint32_t gatherCursor = 0U;
            uint32_t directCursor = 0U;
            uint32_t scheduleStep = 0U;
            while (route.scheduleCount < nonZeroInputs) {
                const uint32_t schedulePhase = scheduleStep % 9U;
                const bool wantGather =
                    (schedulePhase & 1U) != 0U || schedulePhase == 8U;
                uint32_t inputIdx = NextContinuousInput(
                    wantGather, inputCount, axisLens,
                    wantGather ? gatherCursor : directCursor);
                if (inputIdx == inputCount) {
                    inputIdx = NextContinuousInput(
                        !wantGather, inputCount, axisLens,
                        wantGather ? directCursor : gatherCursor);
                }
                if (inputIdx == inputCount) {
                    return HotRouteSelection{};
                }
                scheduleOrder[route.scheduleCount++] = inputIdx;
                ++scheduleStep;
            }
            route.key = kRouteContinuousB32;
            return route;
        }
    }

    return HotRouteSelection{};
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ConcatTilingData tiling;
    const auto* firstShapePtr = context->GetDynamicInputShape(0, 0);
    const auto* firstDesc = context->GetDynamicInputDesc(0, 0);
    const auto* outputDesc = context->GetOutputDesc(0);
    if (firstShapePtr == nullptr || firstDesc == nullptr || outputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // The generated dtype-specific kernel copies raw words of one fixed width.
    // Reject mixed TensorLists before tiling instead of reinterpreting their data.
    const ge::DataType dtype = firstDesc->GetDataType();
    const uint32_t elemBytes = DataTypeBytes(dtype);
    if (elemBytes == 0U || outputDesc->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }

    const auto& firstShape = firstShapePtr->GetStorageShape();
    const int64_t rank = static_cast<int64_t>(firstShape.GetDimNum());
    if (rank <= 0) {
        return ge::GRAPH_FAILED;
    }

    int64_t dim = *context->GetAttrs()->GetInt(0);
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        return ge::GRAPH_FAILED;
    }

    for (int64_t i = 0; i < rank; ++i) {
        if (firstShape.GetDim(i) < 0) {
            return ge::GRAPH_FAILED;
        }
    }

    uint64_t outerSize = 1ULL;
    uint64_t innerSize = 1ULL;
    for (int64_t i = 0; i < dim; ++i) {
        const uint64_t extent = static_cast<uint64_t>(firstShape.GetDim(i));
        if (extent != 0ULL && outerSize > UINT64_MAX / extent) {
            return ge::GRAPH_FAILED;
        }
        outerSize *= extent;
    }
    for (int64_t i = dim + 1; i < rank; ++i) {
        const uint64_t extent = static_cast<uint64_t>(firstShape.GetDim(i));
        if (extent != 0ULL && innerSize > UINT64_MAX / extent) {
            return ge::GRAPH_FAILED;
        }
        innerSize *= extent;
    }

    uint32_t axisLens[kMaxConcatInputs] = {0U};
    uint32_t axisOffsets[kMaxConcatInputs] = {0U};
    uint32_t scheduleOrder[kMaxConcatInputs] = {0U};
    uint64_t outputAxisSize = 0ULL;
    uint32_t inputCount = 0U;
    uint32_t nonzeroInputs = 0U, firstNonzeroInput = 0U;
    uint32_t shortestAxis = UINT32_MAX;
    for (uint32_t i = 0U; i < kMaxConcatInputs; ++i) {
        const auto* inputShapePtr = context->GetDynamicInputShape(0, i);
        if (inputShapePtr == nullptr) {
            break;
        }
        const auto* inputDesc = context->GetDynamicInputDesc(0, i);
        if (inputDesc == nullptr || inputDesc->GetDataType() != dtype) {
            return ge::GRAPH_FAILED;
        }
        const auto& shape = inputShapePtr->GetStorageShape();
        if (static_cast<int64_t>(shape.GetDimNum()) != rank) {
            return ge::GRAPH_FAILED;
        }
        for (int64_t d = 0; d < rank; ++d) {
            if (shape.GetDim(d) < 0) {
                return ge::GRAPH_FAILED;
            }
            if (d == dim) {
                continue;
            }
            if (shape.GetDim(d) != firstShape.GetDim(d)) {
                return ge::GRAPH_FAILED;
            }
        }
        const uint64_t axisLen = static_cast<uint64_t>(shape.GetDim(dim));
        if (axisLen > UINT32_MAX || outputAxisSize > UINT32_MAX - axisLen) {
            return ge::GRAPH_FAILED;
        }
        axisLens[i] = static_cast<uint32_t>(axisLen);
        axisOffsets[i] = static_cast<uint32_t>(outputAxisSize);
        if (axisLen != 0U) {
            if (nonzeroInputs == 0U) firstNonzeroInput = i;
            ++nonzeroInputs;
            shortestAxis = std::min(shortestAxis, static_cast<uint32_t>(axisLen));
        }
        outputAxisSize += axisLen;
        ++inputCount;
    }
    if (context->GetDynamicInputShape(0, kMaxConcatInputs) != nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (inputCount == 0U || outputAxisSize > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalElems64 = outerSize;
    if (outputAxisSize != 0ULL && totalElems64 > UINT64_MAX / outputAxisSize) {
        return ge::GRAPH_FAILED;
    }
    totalElems64 *= outputAxisSize;
    if (innerSize != 0ULL && totalElems64 > UINT64_MAX / innerSize) {
        return ge::GRAPH_FAILED;
    }
    totalElems64 *= innerSize;
    if (totalElems64 > UINT64_MAX / elemBytes) return ge::GRAPH_FAILED;
    const bool wideGeometry = outerSize > UINT32_MAX || innerSize > UINT32_MAX ||
                              totalElems64 > UINT32_MAX;
    // The validated nonempty output bounds every input's byte extent, so the
    // shortest-fragment product is safe in U64. No payload values are inspected.
    uint32_t linearPlan = 0U;
    if (totalElems64 != 0U &&
        (nonzeroInputs == 1U || static_cast<uint64_t>(shortestAxis) * innerSize * elemBytes >=
                                    concat_linear_route::kMinBulkBytes)) {
        linearPlan = concat_linear_route::kBulkDmaFlag | firstNonzeroInput;
        if (nonzeroInputs == 1U) linearPlan |= concat_linear_route::kSingleInputFlag;
    }
    const uint32_t totalElems = static_cast<uint32_t>(totalElems64);
    uint32_t blockElems = kWorkBlockBytes / elemBytes;
    if (blockElems == 0U) {
        blockElems = 1U;
    }
    const uint32_t totalBlocks = wideGeometry ? 0U : CeilDivU32(totalElems, blockElems);
    const uint32_t tileElems = CalcTileElems(elemBytes, blockElems);
    uint32_t blockDim = wideGeometry ? (totalElems64 == 0U ? 1U : kAivCoreNum910B) : SelectBlockDim(totalBlocks);

    const uint32_t periodicPatternInputs = wideGeometry ? 0U : FindPeriodicGatherPattern(
        elemBytes, inputCount, static_cast<uint32_t>(outerSize),
        static_cast<uint32_t>(innerSize), static_cast<uint32_t>(outputAxisSize),
        blockDim, axisLens);
    uint32_t alignedGroupCount = 0U;
    HotRouteSelection hotRoute;
    const uint32_t alignElems = 32U / elemBytes;
    if (periodicPatternInputs != 0U) {
        const uint32_t periodicGroupCount = inputCount / periodicPatternInputs;
        blockDim = std::min(blockDim, periodicGroupCount);
        alignedGroupCount = kPeriodicGatherFlag | periodicPatternInputs;
    } else if (!wideGeometry) {
        hotRoute = SelectHotRoute(
            elemBytes, inputCount, static_cast<uint32_t>(outerSize),
            static_cast<uint32_t>(innerSize),
            static_cast<uint32_t>(outputAxisSize), blockDim, axisLens,
            scheduleOrder);
    }
    if (!wideGeometry && periodicPatternInputs == 0U && hotRoute.key == kRouteDefault &&
        elemBytes == 4U && innerSize == 1ULL && inputCount > 64U && inputCount <= 256U &&
        outerSize >= 64ULL && outerSize <= 4095ULL && outputAxisSize >= 512ULL &&
        (outputAxisSize % alignElems) == 0ULL) {
        uint32_t axisEnd = 0U;
        uint32_t lastGroupEnd = 0U;
        bool allInputsFit = true;
        for (uint32_t i = 0U; i < inputCount; ++i) {
            const uint32_t axisLen = axisLens[i];
            if (axisLen != 0U) {
                const uint32_t alignedLen = CeilDivU32(axisLen, alignElems) * alignElems;
                if (static_cast<uint64_t>(alignedLen) * outerSize > tileElems) {
                    allInputsFit = false;
                    break;
                }
            }
            axisEnd += axisLen;
            if (axisEnd > lastGroupEnd && (axisEnd % alignElems) == 0U) {
                ++alignedGroupCount;
                lastGroupEnd = axisEnd;
            }
        }
        if (!allInputsFit || axisEnd != outputAxisSize || alignedGroupCount < blockDim) {
            alignedGroupCount = 0U;
        }
    }

    if (wideGeometry) {
        alignedGroupCount = kWideGeometryFlag;
    } else if (hotRoute.key == kRouteContinuousB32) {
        alignedGroupCount = kContinuousB32Flag | hotRoute.scheduleCount;
    } else if (hotRoute.key == kRouteCompositeGatherB32) {
        alignedGroupCount = kCompositeGatherB32Flag;
    } else if (hotRoute.key == 3U) {
        alignedGroupCount = concat_block_pack::kFlag;
    } else if (hotRoute.key == 4U) {
        alignedGroupCount = 0x08000000U;
    } else if (hotRoute.key == 5U) {
        alignedGroupCount = concat_asym_pack::kFlag;
    }

    tiling.set_input_count(inputCount);
    tiling.set_outer_size(static_cast<uint32_t>(outerSize));
    tiling.set_inner_size(static_cast<uint32_t>(innerSize));
    tiling.set_output_axis_size(static_cast<uint32_t>(outputAxisSize));
    tiling.set_total_elems(totalElems);
    tiling.set_block_elems(blockElems);
    tiling.set_total_blocks(totalBlocks);
    tiling.set_tile_elems(tileElems);
    tiling.set_elem_bytes(elemBytes);
    tiling.set_ub_bytes(kUbBytes910B);
    tiling.set_aligned_group_count(alignedGroupCount);
    tiling.set_phase_boundary_count(hotRoute.boundaryCount);
    tiling.set_phase_unit_count(hotRoute.unitCount);
    tiling.set_phase_owner_word_count(hotRoute.ownerWordCount);
    tiling.set_axis_lens(axisLens);
    tiling.set_axis_offsets(axisOffsets);
    tiling.set_schedule_order(scheduleOrder);
    tiling.set_outer_size_hi(static_cast<uint32_t>(outerSize >> 32U));
    tiling.set_inner_size_hi(static_cast<uint32_t>(innerSize >> 32U));
    tiling.set_total_elems_hi(static_cast<uint32_t>(totalElems64 >> 32U));
    tiling.set_linear_plan(linearPlan);

    context->GetWorkspaceSizes(1)[0] = 32UL;
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
    const gert::Shape* inputShape = context->GetDynamicInputShape(0, 0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    int64_t rank = static_cast<int64_t>(inputShape->GetDimNum());
    int64_t dim = *context->GetAttrs()->GetInt(0);
    if (dim < 0) {
        dim += rank;
    }
    if (dim < 0 || dim >= rank) {
        return GRAPH_FAILED;
    }

    int64_t concatAxis = 0;
    for (uint32_t i = 0; i < kMaxConcatInputs; ++i) {
        const gert::Shape* curShape = context->GetDynamicInputShape(0, i);
        if (curShape == nullptr) {
            break;
        }
        if (static_cast<int64_t>(curShape->GetDimNum()) != rank) {
            return GRAPH_FAILED;
        }
        concatAxis += curShape->GetDim(dim);
    }
    if (context->GetDynamicInputShape(0, kMaxConcatInputs) != nullptr) {
        return GRAPH_FAILED;
    }

    gert::Shape yShape;
    for (int64_t i = 0; i < rank; ++i) {
        yShape.AppendDim(i == dim ? concatAxis : inputShape->GetDim(i));
    }
    *outputShape = yShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Concat : public OpDef {
public:
    explicit Concat(const char* name) : OpDef(name)
    {
        this->Input("inputs")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}  // namespace ops
