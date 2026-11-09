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
    if (inputs.empty() || attributes.size() != 1) return false;
    for (const auto& shape : inputs) {
        if (UnknownRank(shape)) { output = {-2}; return true; }
        if (!Valid(shape)) return false;
    }
    const int64_t rank = static_cast<int64_t>(inputs[0].size());
    if (rank == 0) return false;
    int64_t axis = attributes[0];
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return false;
    output = inputs[0];
    int64_t sum = 0;
    bool unknown = false;
    for (const auto& shape : inputs) {
        if (static_cast<int64_t>(shape.size()) != rank) return false;
        for (int64_t i = 0; i < rank; ++i) {
            if (i == axis) continue;
            if (output[i] >= 0 && shape[i] >= 0 && output[i] != shape[i]) return false;
            if (output[i] == -1) output[i] = shape[i];
        }
        if (shape[axis] == -1) unknown = true;
        else {
            if (sum > std::numeric_limits<int64_t>::max() - shape[axis]) return false;
            sum += shape[axis];
        }
    }
    output[axis] = unknown ? -1 : sum;
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
    const auto type = context->GetInputDataType(0);
    for (size_t i = 1; i < context->GetComputeNodeInputNum(); ++i) {
        if (context->GetInputDataType(i) != type) return ge::GRAPH_FAILED;
    }
    return context->SetOutputDataType(0, type);
}
#endif
} // namespace s9merge
#ifndef S9_METADATA_TEST
IMPL_OP_INFERSHAPE(Concat).InferShape(s9merge::InferShape).InferDataType(s9merge::InferDataType);
#endif
