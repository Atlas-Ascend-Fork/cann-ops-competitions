/*!
 * \file inplace_update_infershape.cpp
 * \brief Shape inference for InplaceUpdate.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

namespace ops {

static ge::graphStatus InferShapeInplaceUpdate(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *yShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(InplaceUpdate).InferShape(InferShapeInplaceUpdate);

} // namespace ops
