#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "group_norm_tiling_data.h"

namespace GroupNormUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "GroupNorm";

struct GroupNormTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> gammaShape;
    ge::DataType gammaDtype;
    ge::Format gammaFormat;
    std::initializer_list<int64_t> betaShape;
    ge::DataType betaDtype;
    ge::Format betaFormat;
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
//   expectTilingKey:  参考 op_kernel/group_norm_tiling_key.h 和 op_host/group_norm_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/group_norm_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/group_norm_tiling.cpp 中 GetWorkspaceSize 的逻辑
static GroupNormTestParam testCases[] = {
    {"group_norm_0", {1, 4, 2, 2}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {4}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 4, 2, 2}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class GroupNormTilingTest : public testing::TestWithParam<GroupNormTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "GroupNormTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "GroupNormTilingTest TearDown." << std::endl;
    }
};

struct GroupNormCompileInfo {} compileInfo;

static void TestOneParamCase(const GroupNormTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape gammaShape = {param.gammaShape, param.gammaShape};
    gert::StorageShape betaShape = {param.betaShape, param.betaShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
        {gammaShape, param.gammaDtype, param.gammaFormat},
        {betaShape, param.betaDtype, param.betaFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("num_groups", Ops::Math::AnyValue::CreateFrom<int64_t>(2)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("epsilon", Ops::Math::AnyValue::CreateFrom<float>(1e-05f)));
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

TEST_P(GroupNormTilingTest, tiling_test)
{
    const GroupNormTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    GroupNormTilingTests,
    GroupNormTilingTest,
    testing::ValuesIn(testCases));

}
