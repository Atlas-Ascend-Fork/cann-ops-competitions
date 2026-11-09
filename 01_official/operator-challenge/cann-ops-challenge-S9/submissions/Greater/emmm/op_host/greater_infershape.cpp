/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file greater_infer.cpp
 * \brief
 */
#include <algorithm>
#include <array>

#include "register/op_impl_registry.h"
#include "../op_kernel/greater_tiling_data.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShapeGreater(gert::InferShapeContext* context)
{
    const gert::Shape* x1Shape = context->GetInputShape(0);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (x1Shape == nullptr || x2Shape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }
    const size_t rank = std::max(x1Shape->GetDimNum(), x2Shape->GetDimNum());
    if (rank > GREATER_MAX_BROADCAST_DIM) {
        return GRAPH_FAILED;
    }

    std::array<int64_t, GREATER_MAX_BROADCAST_DIM> outputDims {};
    for (size_t reverseIndex = 0; reverseIndex < rank; ++reverseIndex) {
        const int64_t x1Dim = reverseIndex < x1Shape->GetDimNum() ?
                                  x1Shape->GetDim(x1Shape->GetDimNum() - 1 - reverseIndex) :
                                  1;
        const int64_t x2Dim = reverseIndex < x2Shape->GetDimNum() ?
                                  x2Shape->GetDim(x2Shape->GetDimNum() - 1 - reverseIndex) :
                                  1;
        if (x1Dim < 0 || x2Dim < 0 || (x1Dim != x2Dim && x1Dim != 1 && x2Dim != 1)) {
            return GRAPH_FAILED;
        }
        outputDims[rank - 1 - reverseIndex] =
            (x1Dim == 0 || x2Dim == 0) ? 0 : std::max(x1Dim, x2Dim);
    }
    outputShape->SetDimNum(rank);
    for (size_t index = 0; index < rank; ++index) {
        outputShape->SetDim(index, outputDims[index]);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Greater).InferShape(InferShapeGreater);
} // namespace ops
