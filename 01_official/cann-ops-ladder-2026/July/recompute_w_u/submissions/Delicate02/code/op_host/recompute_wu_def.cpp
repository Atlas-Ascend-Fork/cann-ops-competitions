/*
 * SPDX-License-Identifier: MIT
 */
/*!
 * \file recompute_wu_def.cpp
 * \brief RecomputeWu 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class RecomputeWu : public OpDef {
public:
    explicit RecomputeWu(const char* name) : OpDef(name)
    {
        const auto addInput = [this](const char* tensorName, ge::DataType dataType) {
            this->Input(tensorName)
                .ParamType(REQUIRED)
                .DataType({dataType})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
        };
        const auto addOutput = [this](const char* tensorName) {
            this->Output(tensorName)
                .ParamType(REQUIRED)
                .DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND})
                .AutoContiguous();
        };

        addInput("k", ge::DT_BF16);
        addInput("v", ge::DT_BF16);
        addInput("A", ge::DT_BF16);
        addInput("beta", ge::DT_BF16);
        addInput("g", ge::DT_FLOAT);
        addInput("global_chunk_offsets", ge::DT_INT32);
        addOutput("w");
        addOutput("u");

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(RecomputeWu);
} // namespace ops
