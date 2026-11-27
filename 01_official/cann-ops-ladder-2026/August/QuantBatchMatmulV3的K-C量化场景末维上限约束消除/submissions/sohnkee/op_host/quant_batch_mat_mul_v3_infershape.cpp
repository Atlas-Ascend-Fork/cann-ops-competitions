/*!
 * \file quant_batch_mat_mul_v3_infershape.cpp
 * \brief QuantBatchMatMulV3 shape inference implementation.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static bool GetTransposeAttr(const gert::RuntimeAttrs* attrs, size_t index)
{
    if (attrs == nullptr) {
        return false;
    }
    const bool* attr = attrs->GetBool(index);
    return attr != nullptr && *attr;
}

static ge::graphStatus InferShapeQuantBatchMatMulV3(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    if (x1Shape == nullptr || x2Shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (x1Shape->GetDimNum() != 2 || x2Shape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const bool transposeX1 = GetTransposeAttr(attrs, 0);
    const bool transposeX2 = GetTransposeAttr(attrs, 1);

    const int64_t x1Dim0 = x1Shape->GetDim(0);
    const int64_t x1Dim1 = x1Shape->GetDim(1);
    const int64_t x2Dim0 = x2Shape->GetDim(0);
    const int64_t x2Dim1 = x2Shape->GetDim(1);
    if (x1Dim0 <= 0 || x1Dim1 <= 0 || x2Dim0 <= 0 || x2Dim1 <= 0) {
        return ge::GRAPH_FAILED;
    }

    const int64_t m = transposeX1 ? x1Dim1 : x1Dim0;
    const int64_t k1 = transposeX1 ? x1Dim0 : x1Dim1;
    const int64_t k2 = transposeX2 ? x2Dim1 : x2Dim0;
    const int64_t n = transposeX2 ? x2Dim0 : x2Dim1;
    if (k1 != k2 || m <= 0 || n <= 0) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    outputShape->SetDimNum(2);
    outputShape->SetDim(0, m);
    outputShape->SetDim(1, n);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantBatchMatMulV3).InferShape(InferShapeQuantBatchMatMulV3);

} // namespace ops
