/*!
 * \file group_norm_silu_quant_infershape.cpp
 * \brief GroupNormSiluQuant 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeGroupNormSiluQuant(gert::InferShapeContext* context)
{
    const gert::Shape* selfShape = context->GetInputShape(0);
    gert::Shape* outShape = context->GetOutputShape(0);
    gert::Shape* meanShape = context->GetOutputShape(1);
    gert::Shape* rstdShape = context->GetOutputShape(2);
    if (selfShape == nullptr || outShape == nullptr || meanShape == nullptr || rstdShape == nullptr ||
        selfShape->GetDimNum() != 4) {
        return ge::GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetAttrNum() == 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t* groupAttr = attrs->GetAttrPointer<int64_t>(0);
    if (groupAttr == nullptr || *groupAttr <= 0) {
        return ge::GRAPH_FAILED;
    }
    *outShape = *selfShape;
    meanShape->SetDimNum(2);
    meanShape->SetDim(0, selfShape->GetDim(0));
    meanShape->SetDim(1, *groupAttr);
    *rstdShape = *meanShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupNormSiluQuant).InferShape(InferShapeGroupNormSiluQuant);

} // namespace ops
