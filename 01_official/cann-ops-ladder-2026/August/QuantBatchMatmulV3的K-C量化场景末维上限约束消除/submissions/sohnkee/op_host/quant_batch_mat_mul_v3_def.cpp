/*!
 * \file quant_batch_mat_mul_v3_def.cpp
 * \brief QuantBatchMatMulV3 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {
class QuantBatchMatMulV3 : public OpDef {
public:
    explicit QuantBatchMatMulV3(const char* name) : OpDef(name)
    {
    this->Input("x1")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("x2")
        .ParamType(REQUIRED)
        .DataType({ge::DT_INT8})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("scale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Input("pertokenScale")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Output("out")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND})
        .AutoContiguous();
    this->Attr("transposeX1")
        .AttrType(OPTIONAL)
        .Bool();
    this->Attr("transposeX2")
        .AttrType(OPTIONAL)
        .Bool();
    this->AICore()
        .AddConfig("ascend910b")
        .AddConfig("ascend910_93");
    }
};
OP_ADD(QuantBatchMatMulV3);
} // namespace ops
