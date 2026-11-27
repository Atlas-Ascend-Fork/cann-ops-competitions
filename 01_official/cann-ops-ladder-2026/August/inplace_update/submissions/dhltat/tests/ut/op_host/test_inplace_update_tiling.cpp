#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "inplace_update_tiling_data.h"

namespace InplaceUpdateUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "InplaceUpdate";

struct InplaceUpdateTestParam {
    std::string caseName;
    std::initializer_list<int64_t> xShape;
    ge::DataType xDtype;
    ge::Format xFormat;
    std::initializer_list<int64_t> iShape;
    ge::DataType iDtype;
    ge::Format iFormat;
    std::initializer_list<int64_t> vShape;
    ge::DataType vDtype;
    ge::Format vFormat;
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
//   expectTilingKey:  参考 op_kernel/inplace_update_tiling_key.h 和 op_host/inplace_update_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/inplace_update_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/inplace_update_tiling.cpp 中 GetWorkspaceSize 的逻辑
static InplaceUpdateTestParam testCases[] = {
    {"inplace_update_0", {1024, 1280}, ge::DT_FLOAT, ge::FORMAT_ND, {1}, ge::DT_INT32, ge::FORMAT_ND, {1, 1280}, ge::DT_FLOAT, ge::FORMAT_ND, {1024, 1280}, ge::DT_FLOAT, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 1UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class InplaceUpdateTilingTest : public testing::TestWithParam<InplaceUpdateTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "InplaceUpdateTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "InplaceUpdateTilingTest TearDown." << std::endl;
    }
};

struct InplaceUpdateCompileInfo {} compileInfo;

static void TestOneParamCase(const InplaceUpdateTestParam &param)
{
    gert::StorageShape xShape = {param.xShape, param.xShape};
    gert::StorageShape iShape = {param.iShape, param.iShape};
    gert::StorageShape vShape = {param.vShape, param.vShape};
    gert::StorageShape yShape = {param.yShape, param.yShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{xShape, param.xDtype, param.xFormat},
        {iShape, param.iDtype, param.iFormat},
        {vShape, param.vDtype, param.vFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{yShape, param.yDtype, param.yFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;

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

TEST_P(InplaceUpdateTilingTest, tiling_test)
{
    const InplaceUpdateTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    InplaceUpdateTilingTests,
    InplaceUpdateTilingTest,
    testing::ValuesIn(testCases));

}
