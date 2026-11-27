/*!
 * \file inplace_update_def.cpp
 * \brief InplaceUpdate 算子定义
 */
#include "register/op_def_registry.h"

namespace ops {

#define INPLACE_UPDATE_T_DTYPES \
    {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_INT64, \
     ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_BOOL}

#define INPLACE_UPDATE_I_DTYPES \
    {ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, \
     ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32}

#define INPLACE_UPDATE_ND_FORMATS \
    {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
     ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}

class InplaceUpdate : public OpDef {
public:
    explicit InplaceUpdate(const char* name) : OpDef(name)
    {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType(INPLACE_UPDATE_T_DTYPES)
        .Format(INPLACE_UPDATE_ND_FORMATS)
        .UnknownShapeFormat(INPLACE_UPDATE_ND_FORMATS)
        .AutoContiguous();
    this->Input("i")
        .ParamType(REQUIRED)
        .DataType(INPLACE_UPDATE_I_DTYPES)
        .Format(INPLACE_UPDATE_ND_FORMATS)
        .UnknownShapeFormat(INPLACE_UPDATE_ND_FORMATS)
        .AutoContiguous();
    this->Input("v")
        .ParamType(REQUIRED)
        .DataType(INPLACE_UPDATE_T_DTYPES)
        .Format(INPLACE_UPDATE_ND_FORMATS)
        .UnknownShapeFormat(INPLACE_UPDATE_ND_FORMATS)
        .AutoContiguous();
    this->Output("y")
        .ParamType(REQUIRED)
        .DataType(INPLACE_UPDATE_T_DTYPES)
        .Format(INPLACE_UPDATE_ND_FORMATS)
        .UnknownShapeFormat(INPLACE_UPDATE_ND_FORMATS)
        .AutoContiguous();

        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(InplaceUpdate);

#undef INPLACE_UPDATE_T_DTYPES
#undef INPLACE_UPDATE_I_DTYPES
#undef INPLACE_UPDATE_ND_FORMATS

} // namespace ops
