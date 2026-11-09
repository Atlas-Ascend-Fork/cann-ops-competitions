/*
 * Greater compares two broadcast-compatible tensors elementwise and returns Boolean output.
 * The host derives collapsed dimensions, strides and per-core work for the kernel.
 */
#include "greater_tiling.h"
#include "greater_tiling_core.h"
#include "greater_diag.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <cstdint>
#include <vector>

namespace {

int DtypeIndex(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:   return 0;
        case ge::DT_FLOAT16: return 1;
        case ge::DT_BF16:    return 2;
        case ge::DT_INT32:   return 3;
        case ge::DT_INT8:    return 4;
        default:             return -1;
    }
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = platform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = platform.GetCoreNum();
    }
    // Apply the optional downward core-count cap before deriving the tiling plan.
    coreNum = s9::CoreLimit(coreNum);

    auto *inDesc = context->GetInputDesc(0);
    if (inDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int dtypeIdx = DtypeIndex(inDesc->GetDataType());
    if (dtypeIdx < 0) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *sh1 = context->GetInputShape(0);
    const gert::StorageShape *sh2 = context->GetInputShape(1);
    if (sh1 == nullptr || sh2 == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape &g1 = sh1->GetStorageShape();
    const gert::Shape &g2 = sh2->GetStorageShape();

    std::vector<int64_t> d1(g1.GetDimNum());
    std::vector<int64_t> d2(g2.GetDimNum());
    for (size_t i = 0; i < d1.size(); ++i) {
        d1[i] = g1.GetDim(i);
    }
    for (size_t i = 0; i < d2.size(); ++i) {
        d2[i] = g2.GetDim(i);
    }

    // The caller supplies input shapes and types before planning.
    const uint32_t variant = s9::Variant();

    greater_core::Plan plan{};
    uint32_t family = 0;
    if (!greater_core::MakePlanV(d1.data(), d1.size(), d2.data(), d2.size(), static_cast<uint32_t>(dtypeIdx),
                                 ubSize, coreNum, variant, &plan, &family)) {
        return ge::GRAPH_FAILED;
    }

    // The optional diagnostic code is zero when diagnostics are disabled.
    plan.diagCode = greater_diag::Code(plan, family, static_cast<uint32_t>(dtypeIdx));

    // Serialize the complete shared GT_TILING_FIELDS schema in its declared order.
    GreaterTilingData tiling;
#define GT_SET(type, name) tiling.set_##name(plan.name);
#define GT_SET_ARR(type, count, name) tiling.set_##name(plan.name);
    GT_TILING_FIELDS(GT_SET, GT_SET_ARR)
#undef GT_SET
#undef GT_SET_ARR

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());

    // Serialize the metadata required by the selected route. The tiling key selects
    // the matching kernel reader; enabled diagnostics append a route-specific word.
    const bool fullTilingArm = (variant == greater_core::GT_VAR_FULL_TILING);
    const greater_diag::BlobLayout blob =
        greater_diag::Layout(family, fullTilingArm, greater_diag::Enabled());
    uint64_t *raw = reinterpret_cast<uint64_t *>(context->GetRawTilingData()->GetData());
    if (family == greater_core::FAM_FAST && !fullTilingArm) {
        // The aligned route reads one uint64_t field: perCore.
        raw[0] = plan.perCore;
    }
    // Append the diagnostic word at the offset expected by this route.
    if (greater_diag::Enabled()) {
        raw[blob.diagOffset / 8] = plan.diagCode;
    }
    context->GetRawTilingData()->SetDataSize(static_cast<size_t>(blob.shippedBytes));

    context->SetBlockDim(plan.usedCores);
    context->SetTilingKey(plan.tilingKey);

    size_t *workspaces = context->GetWorkspaceSizes(1);
    // Allocate the diagnostic scratch sink only when diagnostics are enabled.
    workspaces[0] = greater_diag::Enabled() ? greater_diag::GT_DIAG_WORKSPACE : 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1 = context->GetInputShape(0);
    const gert::Shape *x2 = context->GetInputShape(1);
    gert::Shape *y = context->GetOutputShape(0);
    if (x1 == nullptr || x2 == nullptr || y == nullptr) {
        return GRAPH_FAILED;
    }
    const size_t r1 = x1->GetDimNum();
    const size_t r2 = x2->GetDimNum();
    const size_t rank = (r1 > r2) ? r1 : r2;
    y->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t a = (i + r1 >= rank) ? x1->GetDim(i + r1 - rank) : 1;
        const int64_t b = (i + r2 >= rank) ? x2->GetDim(i + r2 - rank) : 1;
        y->SetDim(i, (a > b) ? a : b);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Greater : public OpDef {
public:
    explicit Greater(const char *name) : OpDef(name)
    {
        this->Input("x1")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("x2")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8})
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

}  // namespace ops
