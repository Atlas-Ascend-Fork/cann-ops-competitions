/*!
 * \file confusion_matrix_infershape.cpp
 * \brief ConfusionMatrix shape inference implementation
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"

using namespace ge;

namespace ops {

static int64_t GetNumClassesAttr(gert::InferShapeContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return -1;
    }
    const int64_t* numClassesAttr = attrs->GetInt(0);
    if (numClassesAttr == nullptr) {
        return -1;
    }
    return *numClassesAttr;
}

static int64_t GetShapeSize(const gert::Shape* shape)
{
    if (shape == nullptr) {
        return -1;
    }
    if (shape->GetDimNum() == 0) {
        return 1;
    }
    int64_t size = 1;
    for (size_t i = 0; i < shape->GetDimNum(); ++i) {
        const int64_t dim = shape->GetDim(i);
        if (dim < 0) {
            return -1;
        }
        if (dim == 0) {
            return 0;
        }
        size *= dim;
    }
    return size;
}

static ge::graphStatus InferShapeConfusionMatrix(gert::InferShapeContext* context)
{
    const gert::Shape* labelsShape = context->GetInputShape(0);
    const gert::Shape* predictionsShape = context->GetInputShape(1);
    if (labelsShape == nullptr || predictionsShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t labelsNum = GetShapeSize(labelsShape);
    const int64_t predictionsNum = GetShapeSize(predictionsShape);
    if (labelsNum > 0 && predictionsNum > 0 && labelsNum != predictionsNum) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* weightsShape = context->GetInputShape(2);
    if (weightsShape != nullptr) {
        if (weightsShape->GetDimNum() == 0 && labelsNum > 1 && predictionsNum > 1) {
            weightsShape = nullptr;
        }
    }
    if (weightsShape != nullptr) {
        const int64_t weightsNum = GetShapeSize(weightsShape);
        if (labelsNum > 0 && weightsNum > 0 && labelsNum != weightsNum) {
            return ge::GRAPH_FAILED;
        }
        if (predictionsNum > 0 && weightsNum > 0 && predictionsNum != weightsNum) {
            return ge::GRAPH_FAILED;
        }
    }
    if (labelsNum == 0 || predictionsNum == 0) {
        return ge::GRAPH_FAILED;
    }

    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t numClasses = GetNumClassesAttr(context);
    outputShape->SetDimNum(2);
    if (numClasses > 0) {
        outputShape->SetDim(0, numClasses);
        outputShape->SetDim(1, numClasses);
    } else {
        outputShape->SetDim(0, -1);
        outputShape->SetDim(1, -1);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ConfusionMatrix).InferShape(InferShapeConfusionMatrix);

} // namespace ops
