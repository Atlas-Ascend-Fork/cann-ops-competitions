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
    if (inputs.size() != 1) return false;
    const auto& input = inputs[0];
    if (UnknownRank(input)) { output = {-2}; return true; }
    if (!Valid(input) || attributes.size() != input.size()) return false;
    const int64_t rank = static_cast<int64_t>(input.size());
    std::vector<bool> seen(input.size(), false);
    output.resize(input.size());
    for (size_t i = 0; i < attributes.size(); ++i) {
        int64_t axis = attributes[i];
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank || seen[axis]) return false;
        seen[axis] = true;
        output[i] = input[axis];
    }
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
    const auto* dims = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    if (dims == nullptr) return ge::GRAPH_FAILED;
    const auto* values = static_cast<const int64_t*>(dims->GetData());
    if (values == nullptr && dims->GetSize() != 0) return ge::GRAPH_FAILED;
    for (size_t i = 0; i < dims->GetSize(); ++i) attributes.push_back(values[i]);
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
    return context->SetOutputDataType(0, context->GetInputDataType(0));
}
#endif
} // namespace s9merge
#ifndef S9_METADATA_TEST
IMPL_OP_INFERSHAPE(Transpose).InferShape(s9merge::InferShape).InferDataType(s9merge::InferDataType);
#endif
