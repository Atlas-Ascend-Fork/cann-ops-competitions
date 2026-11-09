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
    (void)attributes;
    if (inputs.size() != 2) return false;
    const auto& a = inputs[0];
    const auto& b = inputs[1];
    if (UnknownRank(a) || UnknownRank(b)) { output = {-2}; return true; }
    if (!Valid(a) || !Valid(b)) return false;
    const size_t rank = std::max(a.size(), b.size());
    output.assign(rank, 1);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t x = i < rank - a.size() ? 1 : a[i - (rank - a.size())];
        const int64_t y = i < rank - b.size() ? 1 : b[i - (rank - b.size())];
        if (x == y) output[i] = x;
        else if (x == 1) output[i] = y;
        else if (y == 1) output[i] = x;
        else if (x == -1 || y == -1) output[i] = -1;
        else return false;
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
    if (context->GetComputeNodeInputNum() != 2 ||
        context->GetInputDataType(0) != context->GetInputDataType(1)) return ge::GRAPH_FAILED;
    return context->SetOutputDataType(0, ge::DT_BOOL);
}
#endif
} // namespace s9merge
#ifndef S9_METADATA_TEST
IMPL_OP_INFERSHAPE(Greater).InferShape(s9merge::InferShape).InferDataType(s9merge::InferDataType);
#endif
