#include "square_sum_v1_tiling.h"

#include "square_sum_tiling_policy.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto& inputShape = context->GetInputShape(0)->GetStorageShape();
    const uint32_t rank = static_cast<uint32_t>(inputShape.GetDimNum());
    if (rank == 0U || rank > kDimNum) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    const uint32_t elemBytes = DataTypeBytes(dtype);
    if (elemBytes == 0U) {
        return ge::GRAPH_FAILED;
    }

    uint32_t inDims[kDimNum];
    uint32_t outDims[kDimNum];
    uint32_t inStrides[kDimNum];
    uint32_t outStrides[kDimNum];
    uint32_t reduceMask[kDimNum];
    if (!FillDims(inputShape, inDims)) {
        return ge::GRAPH_FAILED;
    }
    if (!FillReduceMask(context->GetAttrs()->GetListInt(0), rank, reduceMask)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t reduceSize64 = 1ULL;
    bool emptyReduce = false;
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        outDims[i] = reduceMask[i] == 0U ? inDims[i] : 1U;
        if (reduceMask[i] != 0U) {
            emptyReduce = emptyReduce || inDims[i] == 0U;
            if (!emptyReduce) {
                reduceSize64 *= inDims[i];
                if (reduceSize64 > std::numeric_limits<uint32_t>::max()) {
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    const uint32_t reduceSize = emptyReduce ? 0U : static_cast<uint32_t>(reduceSize64);
    if (!CalcStrides(inDims, inStrides) || !CalcStrides(outDims, outStrides)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t inputElems = 0U;
    uint32_t totalElems = 0U;
    if (!CalcTotal(inDims, inputElems) || !CalcTotal(outDims, totalElems)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockElems = 32U / elemBytes;
    uint32_t totalBlocks = CeilDivU32(totalElems, blockElems);
    uint32_t tileElems = CalcVectorTileElems(elemBytes);
    uint32_t accTileElems = 32U / elemBytes;
    uint32_t outerSize = totalElems;
    uint32_t innerSize = 1U;
    uint32_t taskNum = totalElems;
    uint32_t mode = kModeGenericFallback;
    uint64_t workspaceBytes = 32ULL;
    uint32_t blockDim = totalBlocks < kAivCoreNum910B ? totalBlocks : kAivCoreNum910B;

    uint32_t lastReduce = 0U;
    if (IsSuffixReduce(reduceMask)) {
        uint32_t batchRows = 0U;
        const bool useBlockReduce = FindContiguousBlockReduce(
            elemBytes, totalElems, reduceSize, blockElems, tileElems, batchRows);
        mode = useBlockReduce ? kModeContiguousBlockReduce : kModeSuffixReduce;
        outerSize = totalElems;
        innerSize = 1U;
        if (useBlockReduce) {
            accTileElems = batchRows;
            taskNum = CeilDivU32(totalElems, batchRows);
            blockDim = taskNum < kAivCoreNum910B ? taskNum : kAivCoreNum910B;
        } else {
            taskNum = totalElems;
            blockDim = totalBlocks < kAivCoreNum910B ? totalBlocks : kAivCoreNum910B;
        }
    } else if (dtype != ge::DT_BF16 && IsRank4Reduce02(inDims, reduceMask)) {
        const uint32_t dimB = inDims[2];
        const uint32_t dimC = inDims[3];
        const uint32_t dimD = inDims[4];
        const uint32_t dTileElems = CalcRank4Reduce02TileElems(elemBytes, dimB, dimD);
        if (dTileElems == 0U) {
            return ge::GRAPH_FAILED;
        }
        const bool coreRange = CanUseRank4Reduce02CoreRange(
            elemBytes, dimB, dimC, dimD, dTileElems);
        const bool batchB = !coreRange && CanBatchRank4Reduce02(
            elemBytes, dimB, dimC, dimD, dTileElems);
        const uint32_t dTiles = CeilDivU32(dimD, dTileElems);
        const uint64_t taskNum64 = coreRange
            ? dimB
            : (batchB ? CeilDivU32(dimB, kRank4Reduce02BatchB)
                      : static_cast<uint64_t>(dimB) * dTiles);
        if (taskNum64 == 0ULL || taskNum64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        mode = coreRange ? kModeRank4Reduce02CoreRange : kModeRank4Reduce02;
        innerSize = dimD;
        outerSize = dimB;
        accTileElems = dTileElems;
        taskNum = static_cast<uint32_t>(taskNum64);
        blockDim = taskNum < kAivCoreNum910B ? taskNum : kAivCoreNum910B;
    } else if (FindLastReduce(reduceMask, lastReduce) && lastReduce + 1U < kDimNum) {
        mode = kModeInnerAccum;
        innerSize = 1U;
        for (uint32_t i = lastReduce + 1U; i < kDimNum; ++i) {
            innerSize *= inDims[i];
        }
        outerSize = innerSize == 0U ? 0U : totalElems / innerSize;
        accTileElems = CalcInnerTileElems(elemBytes, totalElems);
        taskNum = CeilDivU32(totalElems, accTileElems);
        blockDim = taskNum < kAivCoreNum910B ? taskNum : kAivCoreNum910B;
        if (dtype == ge::DT_BF16) {
            const uint64_t userBytes = CalcWorkspaceBf16Bytes(
                innerSize, outerSize, reduceSize, inDims, reduceMask);
            if (userBytes != 0ULL) {
                mode = kModeWorkspaceBf16Inner;
                workspaceBytes = userBytes;
                taskNum = static_cast<uint32_t>(static_cast<uint64_t>(outerSize) * reduceSize);
                blockDim = kAivCoreNum910B;
            }
        }
        if (dtype == ge::DT_FLOAT && outerSize == 1U &&
            square_sum_policy::ContiguousInner(inDims, reduceMask) &&
            square_sum_policy::AtomicRows(innerSize) != 0U) {
            // Tight streaming needs no per-output owner granularity. Bound its
            // simultaneous input streams and atomic participants to 32 workers;
            // keep the existing 40-worker non-tight fallback unchanged.
            constexpr uint32_t tightWorkers = 32U;
            blockDim = square_sum_policy::TightAtomicFits(innerSize)
                ? tightWorkers : kAivCoreNum910B;
            taskNum = blockDim;
        }
    } else {
        uint32_t planeSize = 0U;
        uint32_t outerOutputSize = 0U;
        uint32_t suffixReduceSize = 0U;
        uint32_t prefixReduceSize = 0U;
        const bool isPlaneReduce = FindPlaneReduce(inDims, reduceMask, planeSize,
                                                    outerOutputSize, suffixReduceSize,
                                                    prefixReduceSize);
        const uint64_t oldCommands = isPlaneReduce
            ? static_cast<uint64_t>(totalElems) * (reduceSize / inDims[kDimNum - 1U]) *
                  CeilDivU32(inDims[kDimNum - 1U], tileElems)
            : 0ULL;
        const uint64_t newCommands = isPlaneReduce
            ? static_cast<uint64_t>(outerOutputSize) * CeilDivU32(planeSize, kPlaneTileRows) *
                  prefixReduceSize * CeilDivU32(suffixReduceSize, kPlaneReduceChunkElems)
            : 0ULL;
        const bool planeStrideFits = isPlaneReduce &&
            suffixReduceSize <= std::numeric_limits<uint32_t>::max() / elemBytes;
        if (planeStrideFits && newCommands < oldCommands) {
            mode = kModePlaneReduce;
            outerSize = outerOutputSize;
            innerSize = planeSize;
            accTileElems = dtype == ge::DT_FLOAT16 && outerOutputSize == 1U &&
                    square_sum_policy::DirectPlaneRows(suffixReduceSize) != 0U
                ? square_sum_policy::DirectPlaneRows(suffixReduceSize)
                : kPlaneTileRows;
            const uint32_t planeTiles = CeilDivU32(planeSize, accTileElems);
            const uint64_t taskNum64 = static_cast<uint64_t>(outerOutputSize) * planeTiles;
            if (taskNum64 > std::numeric_limits<uint32_t>::max()) {
                return ge::GRAPH_FAILED;
            }
            taskNum = static_cast<uint32_t>(taskNum64);
            blockDim = taskNum < kAivCoreNum910B ? taskNum : kAivCoreNum910B;
        } else {
            taskNum = totalElems;
            blockDim = totalBlocks < kAivCoreNum910B ? totalBlocks : kAivCoreNum910B;
        }
    }
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    uint64_t l2Bytes = 0ULL;
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Bytes);
    SquareSumV1TilingData tiling;
    tiling.set_bypass_l2(square_sum_policy::UseStreamingInput(
        static_cast<uint64_t>(inputElems) * elemBytes, l2Bytes) ? 1U : 0U);
    tiling.set_mode(mode);
    tiling.set_total_size(totalElems);
    tiling.set_input_size(inputElems);
    tiling.set_output_size(totalElems);
    tiling.set_block_elems(blockElems);
    tiling.set_total_blocks(totalBlocks);
    tiling.set_tile_elems(tileElems);
    tiling.set_acc_tile_elems(accTileElems);
    tiling.set_outer_size(outerSize);
    tiling.set_reduce_size(reduceSize);
    tiling.set_inner_size(innerSize);
    tiling.set_task_num(taskNum);
    tiling.set_in_dims(inDims);
    tiling.set_out_dims(outDims);
    tiling.set_in_strides(inStrides);
    tiling.set_out_strides(outStrides);
    tiling.set_reduce_mask(reduceMask);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    // Reserve the toolkit's system region as well for the workspace-bearing
    // MIX kernel. Do not infer a fixed offset or depend on allocator padding:
    // profiling can require a different system/user workspace layout.
    const uint64_t systemWorkspaceBytes = mode == kModeWorkspaceBf16Inner
        ? static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize()) : 0ULL;
    context->GetWorkspaceSizes(1)[0] = systemWorkspaceBytes + workspaceBytes;
    context->SetBlockDim(blockDim);
    if (dtype == ge::DT_FLOAT) {
        context->SetTilingKey(0U);
    } else if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(1U);
    } else {
        context->SetTilingKey(2U);
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    auto* axisAttr = context->GetAttrs()->GetListInt(0);
    const bool* keepDimsPtr = context->GetAttrs()->GetBool(1);
    if (axisAttr == nullptr || keepDimsPtr == nullptr) {
        return GRAPH_FAILED;
    }
    const bool keepDims = *keepDimsPtr;
    const uint32_t rank = static_cast<uint32_t>(inputShape->GetDimNum());
    if (rank == 0U || rank > kDimNum) {
        return GRAPH_FAILED;
    }
    uint32_t reduceMask[kDimNum];
    if (!FillReduceMask(axisAttr, rank, reduceMask)) {
        return GRAPH_FAILED;
    }
    const uint32_t base = kDimNum - rank;
    gert::Shape out;
    for (uint32_t i = 0U; i < rank; ++i) {
        if (reduceMask[base + i] != 0U) {
            if (keepDims) {
                out.AppendDim(1);
            }
        } else {
            out.AppendDim(inputShape->GetDim(i));
        }
    }
    if (out.GetDimNum() == 0) {
        out.AppendDim(1);
    }
    *outputShape = out;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("input").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").ListInt();
        this->Attr("keep_dims").Bool();
        this->Output("output").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
}  // namespace ops
