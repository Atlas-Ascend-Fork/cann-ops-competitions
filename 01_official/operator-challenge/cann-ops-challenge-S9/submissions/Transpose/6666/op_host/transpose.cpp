#include "transpose_tiling.h"
#include "transpose_tiling_rules.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {
bool MultiplyChecked(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool ReadPermutation(const gert::RuntimeAttrs *attrs, size_t rank,
                     uint32_t (&permutation)[optiling::TRANSPOSE_MAX_RANK])
{
    if (attrs == nullptr || rank == 0 ||
        rank > optiling::TRANSPOSE_MAX_RANK) {
        return false;
    }
    const gert::TypedContinuousVector<int64_t> *rawPermutation =
        attrs->GetListInt(0);
    if (rawPermutation == nullptr || rawPermutation->GetSize() != rank) {
        return false;
    }

    bool seen[optiling::TRANSPOSE_MAX_RANK] = {false};
    for (size_t index = 0; index < rank; ++index) {
        const int64_t axis = rawPermutation->GetData()[index];
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
            return false;
        }
        const uint32_t normalizedAxis = static_cast<uint32_t>(axis);
        if (seen[normalizedAxis]) {
            return false;
        }
        seen[normalizedAxis] = true;
        permutation[index] = normalizedAxis;
    }
    return true;
}

bool DescribeShape(
    const gert::Shape &shape,
    const uint32_t (&permutation)[optiling::TRANSPOSE_MAX_RANK],
    uint32_t elementSize,
    uint64_t (&inputDims)[optiling::TRANSPOSE_MAX_RANK],
    uint64_t (&outputDims)[optiling::TRANSPOSE_MAX_RANK],
    uint64_t (&sourceStrides)[optiling::TRANSPOSE_MAX_RANK],
    uint64_t &totalElements)
{
    const size_t rank = shape.GetDimNum();
    uint64_t inputStrides[optiling::TRANSPOSE_MAX_RANK] = {0};
    for (size_t axis = 0; axis < optiling::TRANSPOSE_MAX_RANK; ++axis) {
        inputDims[axis] = 1;
        outputDims[axis] = 1;
        sourceStrides[axis] = 1;
    }

    totalElements = 1;
    for (size_t axis = 0; axis < rank; ++axis) {
        const int64_t rawDim = shape.GetDim(axis);
        if (rawDim <= 0) {
            return false;
        }
        inputDims[axis] = static_cast<uint64_t>(rawDim);
        if (!MultiplyChecked(totalElements, inputDims[axis], totalElements)) {
            return false;
        }
    }
    uint64_t totalBytes = 0;
    if (!MultiplyChecked(totalElements, elementSize, totalBytes)) {
        return false;
    }
    (void)totalBytes;

    uint64_t stride = 1;
    for (size_t reverse = rank; reverse > 0; --reverse) {
        const size_t axis = reverse - 1;
        inputStrides[axis] = stride;
        if (!MultiplyChecked(stride, inputDims[axis], stride)) {
            return false;
        }
    }
    for (size_t outputAxis = 0; outputAxis < rank; ++outputAxis) {
        const uint32_t inputAxis = permutation[outputAxis];
        outputDims[outputAxis] = inputDims[inputAxis];
        sourceStrides[outputAxis] = inputStrides[inputAxis];
    }
    return true;
}

uint32_t DataTypeBytes(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_INT8:
            return 1;
        default:
            return 0;
    }
}

}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    const gert::CompileTimeTensorDesc *inputDesc = context->GetInputDesc(0);
    if (inputShape == nullptr || inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &shape = inputShape->GetOriginShape();
    const size_t rank = shape.GetDimNum();
    uint32_t permutation[TRANSPOSE_MAX_RANK] = {0};
    if (!ReadPermutation(context->GetAttrs(), rank, permutation)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t elementSize = DataTypeBytes(inputDesc->GetDataType());
    if (elementSize == 0) {
        return ge::GRAPH_FAILED;
    }
    uint64_t inputDims[TRANSPOSE_MAX_RANK];
    uint64_t outputDims[TRANSPOSE_MAX_RANK];
    uint64_t sourceStrides[TRANSPOSE_MAX_RANK];
    uint64_t totalElements = 0;
    if (!DescribeShape(shape, permutation, elementSize, inputDims, outputDims,
                       sourceStrides, totalElements)) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aivCores = std::max(1U, platform.GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    // Choose a UB-bounded tile plan, then distribute only nonempty tasks to AIV cores.
    const TransposePlan plan = SelectTransposePlan(
        outputDims, sourceStrides, static_cast<uint32_t>(rank), totalElements,
        elementSize, aivCores, ubSize);

    TransposeTilingData tiling;
    tiling.set_totalElements(totalElements);
    tiling.set_runElements(plan.runElements);
    tiling.set_runCount(plan.runCount);
    tiling.set_groupCount(plan.groupCount);
    tiling.set_blocksPerGroup(plan.blocksPerGroup);
    tiling.set_chunksPerUnit(plan.chunksPerUnit);
    tiling.set_taskCount(plan.taskCount);
    tiling.set_sourceStepBytes(plan.sourceStepBytes);
    tiling.set_batchCount(plan.batchCount);
    tiling.set_matrixRows(plan.matrixRows);
    tiling.set_matrixColumns(plan.matrixColumns);
    tiling.set_matrixPrefixRank(plan.matrixPrefixRank);
    tiling.set_outputMatrixRowStride(plan.outputMatrixRowStride);
    tiling.set_rank(static_cast<uint32_t>(rank));
    tiling.set_prefixRank(plan.prefixRank);
    tiling.set_tileElements(plan.tileElements);
    tiling.set_gatherBlocks(plan.gatherBlocks);
    tiling.set_elementSize(elementSize);
    tiling.set_matrixRowsPerTask(plan.matrixRowsPerTask);
    tiling.set_matrixColumnsPerTask(plan.matrixColumnsPerTask);
    tiling.set_matrixRowAxis(plan.matrixRowAxis);
    tiling.set_matrixColumnAxis(plan.matrixColumnAxis);
    tiling.set_runAxis(plan.runAxis);
    tiling.set_outputRunStride(plan.outputRunStride);
    tiling.set_outputDims(outputDims);
    tiling.set_sourceStrides(sourceStrides);
    if (plan.strategy == TransposeStrategy::SLAB_2D_FP16 ||
        plan.strategy == TransposeStrategy::SLAB_2D_BYTE ||
        plan.strategy == TransposeStrategy::SLAB_2D_WORD32 ||
        plan.strategy == TransposeStrategy::BATCHED_SLAB_WORD32 ||
        plan.strategy == TransposeStrategy::STRIDED_TILE_WORD32) {
        const uint64_t matrixColumns =
            plan.strategy == TransposeStrategy::SLAB_2D_WORD32 ||
                    plan.strategy == TransposeStrategy::SLAB_2D_BYTE ||
                    plan.strategy == TransposeStrategy::BATCHED_SLAB_WORD32 ||
                    plan.strategy == TransposeStrategy::STRIDED_TILE_WORD32
                ? plan.matrixColumnsPerTask
                : outputDims[0];
        const std::vector<int64_t> inputDims = {
            static_cast<int64_t>(plan.matrixRowsPerTask),
            static_cast<int64_t>(matrixColumns)};
        const ge::Shape input2dShape(inputDims);
        AscendC::GetTransposeTilingInfo(
            input2dShape, 0, plan.strategy == TransposeStrategy::SLAB_2D_BYTE ? 2 : elementSize,
            7, tiling.transposeTiling);
        if (plan.strategy == TransposeStrategy::BATCHED_SLAB_WORD32) {
            const std::vector<int64_t> tailDims = {
                static_cast<int64_t>(plan.matrixRowsPerTask),
                static_cast<int64_t>(TRANSPOSE_MATRIX_TILE)};
            const ge::Shape tail2dShape(tailDims);
            AscendC::GetTransposeTilingInfo(
                tail2dShape, 0, elementSize, 7,
                tiling.tailTransposeTiling);
        }
    }

    context->SetBlockDim(plan.blockDim);
    context->SetTilingKey(plan.tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *workspace = context->GetWorkspaceSizes(1);
    if (workspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    const size_t rank = inputShape->GetDimNum();
    uint32_t permutation[optiling::TRANSPOSE_MAX_RANK] = {0};
    if (!ReadPermutation(context->GetAttrs(), rank, permutation)) {
        return GRAPH_FAILED;
    }
    outputShape->SetDimNum(0);
    for (size_t axis = 0; axis < rank; ++axis) {
        outputShape->AppendDim(inputShape->GetDim(permutation[axis]));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char *name) : OpDef(name)
    {
        const std::vector<ge::DataType> dataTypes = {
            ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8};
        const std::vector<ge::Format> formats = {
            ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
        this->Input("inputs")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType(dataTypes)
            .Format(formats)
            .UnknownShapeFormat(formats);
        this->Attr("dims").ListInt();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Transpose);
}  // namespace ops
