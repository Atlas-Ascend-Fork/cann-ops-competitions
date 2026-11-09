// Graph metadata adapters for the S9 merge; no tensor computation runs here.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#ifndef S9_METADATA_TEST
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#endif

namespace s9merge {
using Shape = std::vector<int64_t>;
using Shapes = std::vector<Shape>;
static bool UnknownRank(const Shape& shape) { return shape.size() == 1 && shape[0] == -2; }
static bool Valid(const Shape& shape) {
    return std::all_of(shape.begin(), shape.end(), [](int64_t d) { return d >= -1; });
}
static bool Infer(const Shapes& inputs, const Shape& attributes, Shape& output)
{
    if (inputs.size() != 3 || attributes.size() != 1) return false;
    const auto& self = inputs[0];
    const auto& index = inputs[1];
    const auto& source = inputs[2];
    if (UnknownRank(self)) { output = {-2}; return true; }
    if (!Valid(self) || self.empty()) return false;
    int64_t axis = attributes[0];
    const int64_t rank = static_cast<int64_t>(self.size());
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return false;
    if (!UnknownRank(index) && (!Valid(index) || index.size() != 1)) return false;
    if (!UnknownRank(source)) {
        if (!Valid(source) || source.size() != self.size()) return false;
        for (int64_t i = 0; i < rank; ++i) {
            if (i != axis && self[i] >= 0 && source[i] >= 0 && self[i] != source[i]) return false;
        }
        if (!UnknownRank(index) && index[0] >= 0 && source[axis] >= 0 && index[0] != source[axis]) return false;
    }
    output = self;
    return true;
}

#ifndef S9_METADATA_TEST
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) return ge::GRAPH_FAILED;
    Shapes inputs;
    for (size_t i = 0; i < context->GetComputeNodeInputNum(); ++i) {
        const auto* shape = context->GetInputShape(i);
        if (shape == nullptr) return ge::GRAPH_FAILED;
        Shape dimensions;
        for (size_t axis = 0; axis < shape->GetDimNum(); ++axis) dimensions.push_back(shape->GetDim(axis));
        inputs.push_back(std::move(dimensions));
    }
    Shape attributes;
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const auto* axis = attrs->GetAttrPointer<int64_t>(0);
    if (axis == nullptr) return ge::GRAPH_FAILED;
    attributes.push_back(*axis);
    Shape result;
    auto* output = context->GetOutputShape(0);
    if (output == nullptr || !Infer(inputs, attributes, result)) return ge::GRAPH_FAILED;
    output->SetDimNum(result.size());
    for (size_t i = 0; i < result.size(); ++i) output->SetDim(i, result[i]);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr || context->GetComputeNodeInputNum() == 0) return ge::GRAPH_FAILED;
    if (context->GetComputeNodeInputNum() != 3 ||
        context->GetInputDataType(1) != ge::DT_INT32 ||
        context->GetInputDataType(0) != context->GetInputDataType(2)) return ge::GRAPH_FAILED;
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}
#endif
} // namespace s9merge
#ifndef S9_METADATA_TEST
IMPL_OP_INFERSHAPE(IndexAdd).InferShape(s9merge::InferShape).InferDataType(s9merge::InferDataType);
#endif
