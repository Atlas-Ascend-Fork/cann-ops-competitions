/*!
 * \file ctc_loss_infershape.cpp
 * \brief CtcLoss 算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

#include <cstring>

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeCtcLoss(gert::InferShapeContext* context)
{
    const gert::Shape* log_probs_shape = context->GetInputShape(0);
    gert::Shape* output_shape = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, log_probs_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_IF(log_probs_shape->GetDimNum() != 3,
                OP_LOGE(context->GetNodeName(), "log_probs must be rank 3"), return ge::GRAPH_FAILED);

    const char* reduction = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, reduction);
    if (std::strcmp(reduction, "none") == 0) {
        *output_shape = gert::Shape({log_probs_shape->GetDim(1)});
    } else if (std::strcmp(reduction, "mean") == 0 || std::strcmp(reduction, "sum") == 0) {
        *output_shape = gert::Shape({1});
    } else {
        OP_LOGE(context->GetNodeName(), "reduction must be none, mean or sum");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CtcLoss).InferShape(InferShapeCtcLoss);

} // namespace ops
