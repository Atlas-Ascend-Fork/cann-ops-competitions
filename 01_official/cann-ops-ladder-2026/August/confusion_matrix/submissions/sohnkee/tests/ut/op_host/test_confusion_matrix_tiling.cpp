#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "confusion_matrix_tiling_data.h"

namespace ConfusionMatrixUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "ConfusionMatrix";

struct ConfusionMatrixTestParam {
    std::string caseName;
    std::initializer_list<int64_t> labelsShape;
    ge::DataType labelsDtype;
    ge::Format labelsFormat;
    std::initializer_list<int64_t> predictionsShape;
    ge::DataType predictionsDtype;
    ge::Format predictionsFormat;
    std::initializer_list<int64_t> weightsShape;
    ge::DataType weightsDtype;
    ge::Format weightsFormat;
    std::initializer_list<int64_t> yShape;
    ge::DataType yDtype;
    ge::Format yFormat;
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
//   expectTilingKey:  参考 op_kernel/confusion_matrix_tiling_key.h 和 op_host/confusion_matrix_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/confusion_matrix_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/confusion_matrix_tiling.cpp 中 GetWorkspaceSize 的逻辑
static ConfusionMatrixTestParam testCases[] = {
    {"confusion_matrix_0", {5}, ge::DT_INT32, ge::FORMAT_ND, {5}, ge::DT_INT32, ge::FORMAT_ND, {}, ge::DT_UNDEFINED, ge::FORMAT_ND, {3, 3}, ge::DT_INT32, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class ConfusionMatrixTilingTest : public testing::TestWithParam<ConfusionMatrixTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "ConfusionMatrixTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "ConfusionMatrixTilingTest TearDown." << std::endl;
    }
};

struct ConfusionMatrixCompileInfo {} compileInfo;

static void TestOneParamCase(const ConfusionMatrixTestParam &param)
{
    gert::StorageShape labelsShape = {param.labelsShape, param.labelsShape};
    gert::StorageShape predictionsShape = {param.predictionsShape, param.predictionsShape};
    gert::StorageShape weightsShape = {param.weightsShape, param.weightsShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{labelsShape, param.labelsDtype, param.labelsFormat},
        {predictionsShape, param.predictionsDtype, param.predictionsFormat},
        {weightsShape, param.weightsDtype, param.weightsFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("num_classes", Ops::Math::AnyValue::CreateFrom<int64_t>(3)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("dtype", Ops::Math::AnyValue::CreateFrom<std::string>("int32")));
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

TEST_P(ConfusionMatrixTilingTest, tiling_test)
{
    const ConfusionMatrixTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    ConfusionMatrixTilingTests,
    ConfusionMatrixTilingTest,
    testing::ValuesIn(testCases));

}
