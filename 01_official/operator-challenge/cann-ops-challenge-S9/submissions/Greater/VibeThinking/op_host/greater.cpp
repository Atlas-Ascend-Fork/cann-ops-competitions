#include <cstring>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "greater_tiling.h"
#include "planner.h"

namespace s9g_host {

inline std::vector<int64_t> DimensionsOf(const gert::Shape& shape) {
    std::vector<int64_t> dimensions(shape.GetDimNum());
    for (size_t axis=0;axis<dimensions.size();++axis)
        dimensions[axis]=shape.GetDim(axis);
    return dimensions;
}

inline uint32_t TypeIndex(ge::DataType type) {
    const ge::DataType supported[]={
        ge::DT_FLOAT,ge::DT_BF16,ge::DT_FLOAT16,ge::DT_INT32,ge::DT_INT8
    };
    for (uint32_t index=0;index<sizeof(supported)/sizeof(supported[0]);++index)
        if (type==supported[index]) return index;
    return 99;
}

}

namespace s9g_host {

// 按所选路径只传递需要的分片参数，Host 与 Kernel 共用数据布局。
inline bool StoreTilingPayload(gert::TilingContext* context,
                               const s9g::Plan& plan) {
    auto* storage=context->GetRawTilingData();
    if (!storage || storage->GetCapacity()<plan.tilingBytes) return false;
    void* destination=storage->GetData();

    if (plan.key==s9g::EMPTY || plan.key==s9g::EXACT ||
        (plan.key>=16 && plan.key<=20)) {
        const GreaterExact payload{plan.data.perCore};
        std::memcpy(destination,&payload,sizeof(payload));
    } else if (plan.key==s9g::ONESHOT) {
        const GreaterOneShot payload{plan.data.total,plan.data.perCore};
        std::memcpy(destination,&payload,sizeof(payload));
    } else if (plan.key==s9g::FP32_PAIR) {
        std::memcpy(destination,&plan.pair,sizeof(plan.pair));
    } else if (plan.key==7 || plan.key==8) {
        std::memcpy(destination,&plan.linear,sizeof(plan.linear));
    } else {
        std::memcpy(destination,&plan.data,plan.tilingBytes);
    }
    storage->SetDataSize(plan.tilingBytes);
    return true;
}

}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto* leftShape=context->GetInputShape(0);
    const auto* rightShape=context->GetInputShape(1);
    const auto* outputShape=context->GetOutputShape(0);
    const auto* leftDesc=context->GetInputDesc(0);
    const auto* rightDesc=context->GetInputDesc(1);
    if (!leftShape || !rightShape || !outputShape || !leftDesc || !rightDesc)
        return ge::GRAPH_FAILED;
    if (leftDesc->GetDataType()!=rightDesc->GetDataType())
        return ge::GRAPH_FAILED;

    const std::vector<int64_t> left=
        s9g_host::DimensionsOf(leftShape->GetStorageShape());
    const std::vector<int64_t> right=
        s9g_host::DimensionsOf(rightShape->GetStorageShape());
    std::vector<int64_t> inferred;
    if (!s9g::InferBroadcastShape(left,right,inferred) ||
        inferred!=s9g_host::DimensionsOf(outputShape->GetStorageShape()))
        return ge::GRAPH_FAILED;

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ubBytes=0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ubBytes);

    // 结合广播布局、数据类型和 UB 容量确定分块大小及并行核数。
    s9g::Plan plan;
    if (!s9g::BuildPlan(left,right,s9g_host::TypeIndex(leftDesc->GetDataType()),
                        platform.GetCoreNumAiv(),ubBytes,plan))
        return ge::GRAPH_FAILED;
    if (!s9g_host::StoreTilingPayload(context,plan))
        return ge::GRAPH_FAILED;

    context->SetTilingKey(plan.key);
    context->SetBlockDim(plan.cores);
    context->GetWorkspaceSizes(1)[0]=0;
    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context) {
    const auto* left=context->GetInputShape(0);
    const auto* right=context->GetInputShape(1);
    auto* output=context->GetOutputShape(0);
    if (!left || !right || !output) return GRAPH_FAILED;

    std::vector<int64_t> dimensions;
    if (!s9g::InferBroadcastShape(s9g_host::DimensionsOf(*left),
                                  s9g_host::DimensionsOf(*right),dimensions))
        return GRAPH_FAILED;
    output->SetDimNum(dimensions.size());
    for (size_t axis=0;axis<dimensions.size();++axis)
        output->SetDim(axis,dimensions[axis]);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const ge::DataType left=context->GetInputDataType(0);
    if (left!=context->GetInputDataType(1) || s9g_host::TypeIndex(left)>4)
        return GRAPH_FAILED;
    context->SetOutputDataType(0,ge::DT_BOOL);
    return GRAPH_SUCCESS;
}

}

namespace ops {

class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name) {
        const std::vector<ge::DataType> inputTypes={
            ge::DT_FLOAT,ge::DT_BF16,ge::DT_FLOAT16,ge::DT_INT32,ge::DT_INT8
        };
        const std::vector<ge::Format> ndFormats(inputTypes.size(),ge::FORMAT_ND);
        const std::vector<ge::DataType> outputTypes(inputTypes.size(),ge::DT_BOOL);

        Input("x1").ParamType(REQUIRED).DataType(inputTypes)
            .Format(ndFormats).UnknownShapeFormat(ndFormats);
        Input("x2").ParamType(REQUIRED).DataType(inputTypes)
            .Format(ndFormats).UnknownShapeFormat(ndFormats);
        Output("y").ParamType(REQUIRED).DataType(outputTypes)
            .Format(ndFormats).UnknownShapeFormat(ndFormats);
        SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(Greater);

}
