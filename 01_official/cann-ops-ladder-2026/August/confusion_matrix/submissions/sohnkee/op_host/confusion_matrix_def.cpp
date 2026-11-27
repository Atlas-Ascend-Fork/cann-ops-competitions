/*!
 * \file confusion_matrix_def.cpp
 * \brief ConfusionMatrix operator definition
 */

#include "register/op_def_registry.h"

namespace ops {

#define CM_LABEL_DTYPES \
    { \
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, \
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, \
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, \
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, \
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, \
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 \
    }

#define CM_PREDICTIONS_DTYPES \
    { \
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, \
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, \
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, \
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, \
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, \
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64 \
    }

#define CM_WEIGHTS_DTYPES \
    { \
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, \
        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, \
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, \
        ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, \
        ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, \
        ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT \
    }

#define CM_OUTPUT_DTYPES \
    { \
        ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, \
        ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, \
        ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, \
        ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, \
        ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, \
        ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_FLOAT \
    }

#define CM_ND_FORMATS \
    { \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, \
        ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND \
    }

class ConfusionMatrix : public OpDef {
public:
    explicit ConfusionMatrix(const char* name) : OpDef(name)
    {
        this->Input("labels")
            .ParamType(REQUIRED)
            .DataType(CM_LABEL_DTYPES)
            .Format(CM_ND_FORMATS)
            .UnknownShapeFormat(CM_ND_FORMATS)
            .AutoContiguous();
        this->Input("predictions")
            .ParamType(REQUIRED)
            .DataType(CM_PREDICTIONS_DTYPES)
            .Format(CM_ND_FORMATS)
            .UnknownShapeFormat(CM_ND_FORMATS)
            .AutoContiguous();
        this->Input("weights")
            .ParamType(OPTIONAL)
            .DataType(CM_WEIGHTS_DTYPES)
            .Format(CM_ND_FORMATS)
            .UnknownShapeFormat(CM_ND_FORMATS)
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(CM_OUTPUT_DTYPES)
            .Format(CM_ND_FORMATS)
            .UnknownShapeFormat(CM_ND_FORMATS)
            .InitValue(0)
            .AutoContiguous();
        this->Attr("num_classes")
            .AttrType(OPTIONAL)
            .Int(-1);
        this->Attr("dtype")
            .AttrType(OPTIONAL)
            .String("int32");
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(ConfusionMatrix);

#undef CM_LABEL_DTYPES
#undef CM_PREDICTIONS_DTYPES
#undef CM_WEIGHTS_DTYPES
#undef CM_OUTPUT_DTYPES
#undef CM_ND_FORMATS

} // namespace ops
