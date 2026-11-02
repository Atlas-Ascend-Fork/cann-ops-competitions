/*!
 * \file recompute_wu_infershape.cpp
 * \brief RecomputeWu 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeRecomputeWu(gert::InferShapeContext* context)
{
    const gert::Shape* kShape = context->GetInputShape(0);
    const gert::Shape* vShape = context->GetInputShape(1);
    const gert::Shape* aShape = context->GetInputShape(2);
    gert::Shape* wShape = context->GetOutputShape(0);
    gert::Shape* uShape = context->GetOutputShape(1);
    if (aShape == nullptr || kShape == nullptr || vShape == nullptr || wShape == nullptr || uShape == nullptr ||
        aShape->GetDimNum() != 4 || kShape->GetDimNum() != 4 || vShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }

    wShape->SetDimNum(4);
    wShape->SetDim(0, aShape->GetDim(0));
    wShape->SetDim(1, aShape->GetDim(1));
    wShape->SetDim(2, aShape->GetDim(2));
    wShape->SetDim(3, kShape->GetDim(3));
    uShape->SetDimNum(4);
    uShape->SetDim(0, aShape->GetDim(0));
    uShape->SetDim(1, aShape->GetDim(1));
    uShape->SetDim(2, aShape->GetDim(2));
    uShape->SetDim(3, vShape->GetDim(3));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RecomputeWu).InferShape(InferShapeRecomputeWu);

} // namespace ops
