#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "quant_batch_mat_mul_v3_tiling_data.h"

namespace QuantBatchMatMulV3UT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "QuantBatchMatMulV3";

struct QuantBatchMatMulV3TestParam {
    std::string caseName;
    std::initializer_list<int64_t> x1Shape;
    ge::DataType x1Dtype;
    ge::Format x1Format;
    std::initializer_list<int64_t> x2Shape;
    ge::DataType x2Dtype;
    ge::Format x2Format;
    std::initializer_list<int64_t> scaleShape;
    ge::DataType scaleDtype;
    ge::Format scaleFormat;
    std::initializer_list<int64_t> pertokenScaleShape;
    ge::DataType pertokenScaleDtype;
    ge::Format pertokenScaleFormat;
    std::initializer_list<int64_t> outShape;
    ge::DataType outDtype;
    ge::Format outFormat;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
    std::string expectTilingData;
    std::vector<size_t> expectWorkspaces;
    uint64_t maxAIVNum;
    uint64_t ubSize;
    uint64_t tilingDataMaxSize;
};

// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新：
//   expectTilingKey:  参考 op_kernel/quant_batch_mat_mul_v3_tiling_key.h 和 op_host/quant_batch_mat_mul_v3_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/quant_batch_mat_mul_v3_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/quant_batch_mat_mul_v3_tiling.cpp 中 GetWorkspaceSize 的逻辑
static QuantBatchMatMulV3TestParam testCases[] = {
    {"quant_batch_mat_mul_v3_0", {128, 65536}, ge::DT_INT8, ge::FORMAT_ND, {65536, 96}, ge::DT_INT8, ge::FORMAT_ND, {96}, ge::DT_FLOAT, ge::FORMAT_ND, {128}, ge::DT_FLOAT, ge::FORMAT_ND, {128, 96}, ge::DT_BF16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class QuantBatchMatMulV3TilingTest : public testing::TestWithParam<QuantBatchMatMulV3TestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "QuantBatchMatMulV3TilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "QuantBatchMatMulV3TilingTest TearDown." << std::endl;
    }
};

struct QuantBatchMatMulV3CompileInfo {} compileInfo;

static void TestOneParamCase(const QuantBatchMatMulV3TestParam &param)
{
    gert::StorageShape x1Shape = {param.x1Shape, param.x1Shape};
    gert::StorageShape x2Shape = {param.x2Shape, param.x2Shape};
    gert::StorageShape scaleShape = {param.scaleShape, param.scaleShape};
    gert::StorageShape pertokenScaleShape = {param.pertokenScaleShape, param.pertokenScaleShape};
    gert::StorageShape outShape = {param.outShape, param.outShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{x1Shape, param.x1Dtype, param.x1Format},
        {x2Shape, param.x2Dtype, param.x2Format},
        {scaleShape, param.scaleDtype, param.scaleFormat},
        {pertokenScaleShape, param.pertokenScaleDtype, param.pertokenScaleFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{outShape, param.outDtype, param.outFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("transposeX1", Ops::Math::AnyValue::CreateFrom<bool>(false)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("transposeX2", Ops::Math::AnyValue::CreateFrom<bool>(false)));
    gert::TilingContextPara tilingContextPara(
        OP_NAME,
        inputTensorDesc_,
        outputTensorDesc_,
        attrs_,
        &compileInfo,
        param.maxAIVNum,
        param.ubSize,
        param.tilingDataMaxSize);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey,
                    param.expectTilingData, param.expectWorkspaces);
}

TEST_P(QuantBatchMatMulV3TilingTest, tiling_test)
{
    const QuantBatchMatMulV3TestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    QuantBatchMatMulV3TilingTests,
    QuantBatchMatMulV3TilingTest,
    testing::ValuesIn(testCases));

}
