#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "ctc_loss_tiling_data.h"

namespace CtcLossUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "CtcLoss";

struct CtcLossTestParam {
    std::string caseName;
    std::initializer_list<int64_t> log_probsShape;
    ge::DataType log_probsDtype;
    ge::Format log_probsFormat;
    std::initializer_list<int64_t> targetsShape;
    ge::DataType targetsDtype;
    ge::Format targetsFormat;
    std::initializer_list<int64_t> input_lengthsShape;
    ge::DataType input_lengthsDtype;
    ge::Format input_lengthsFormat;
    std::initializer_list<int64_t> target_lengthsShape;
    ge::DataType target_lengthsDtype;
    ge::Format target_lengthsFormat;
    std::initializer_list<int64_t> neg_log_likelihoodShape;
    ge::DataType neg_log_likelihoodDtype;
    ge::Format neg_log_likelihoodFormat;
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
//   expectTilingKey:  参考 op_kernel/ctc_loss_tiling_key.h 和 op_host/ctc_loss_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/ctc_loss_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/ctc_loss_tiling.cpp 中 GetWorkspaceSize 的逻辑
static CtcLossTestParam testCases[] = {
    {"ctc_loss_0", {2, 1, 3}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 2}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::DT_INT32, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class CtcLossTilingTest : public testing::TestWithParam<CtcLossTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "CtcLossTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "CtcLossTilingTest TearDown." << std::endl;
    }
};

struct CtcLossCompileInfo {} compileInfo;

static void TestOneParamCase(const CtcLossTestParam &param)
{
    gert::StorageShape log_probsShape = {param.log_probsShape, param.log_probsShape};
    gert::StorageShape targetsShape = {param.targetsShape, param.targetsShape};
    gert::StorageShape input_lengthsShape = {param.input_lengthsShape, param.input_lengthsShape};
    gert::StorageShape target_lengthsShape = {param.target_lengthsShape, param.target_lengthsShape};
    gert::StorageShape neg_log_likelihoodShape = {param.neg_log_likelihoodShape, param.neg_log_likelihoodShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{log_probsShape, param.log_probsDtype, param.log_probsFormat},
        {targetsShape, param.targetsDtype, param.targetsFormat},
        {input_lengthsShape, param.input_lengthsDtype, param.input_lengthsFormat},
        {target_lengthsShape, param.target_lengthsDtype, param.target_lengthsFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{neg_log_likelihoodShape, param.neg_log_likelihoodDtype, param.neg_log_likelihoodFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("blank", Ops::Math::AnyValue::CreateFrom<int64_t>(0)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("reduction", Ops::Math::AnyValue::CreateFrom<std::string>("none")));
    attrs_.push_back(gert::TilingContextPara::OpAttr("zero_infinity", Ops::Math::AnyValue::CreateFrom<bool>(false)));
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

TEST_P(CtcLossTilingTest, tiling_test)
{
    const CtcLossTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    CtcLossTilingTests,
    CtcLossTilingTest,
    testing::ValuesIn(testCases));

}
