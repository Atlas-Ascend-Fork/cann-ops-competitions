#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "group_norm_silu_quant_tiling_data.h"

namespace GroupNormSiluQuantUT {
using namespace std;
using namespace ge;
using namespace gert;
static const std::string OP_NAME = "GroupNormSiluQuant";

struct GroupNormSiluQuantTestParam {
    std::string caseName;
    std::initializer_list<int64_t> selfShape;
    ge::DataType selfDtype;
    ge::Format selfFormat;
    std::initializer_list<int64_t> gammaShape;
    ge::DataType gammaDtype;
    ge::Format gammaFormat;
    std::initializer_list<int64_t> betaShape;
    ge::DataType betaDtype;
    ge::Format betaFormat;
    std::initializer_list<int64_t> quantScaleShape;
    ge::DataType quantScaleDtype;
    ge::Format quantScaleFormat;
    std::initializer_list<int64_t> outShape;
    ge::DataType outDtype;
    ge::Format outFormat;
    std::initializer_list<int64_t> meanOutShape;
    ge::DataType meanOutDtype;
    ge::Format meanOutFormat;
    std::initializer_list<int64_t> rstdOutShape;
    ge::DataType rstdOutDtype;
    ge::Format rstdOutFormat;
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
//   expectTilingKey:  参考 op_kernel/group_norm_silu_quant_tiling_key.h 和 op_host/group_norm_silu_quant_tiling.cpp 中 tilingKey 的逻辑
//   expectTilingData: 参考 op_host/group_norm_silu_quant_tiling.cpp 中 TilingData 各字段的赋值
//   expectWorkspaces: 参考 op_host/group_norm_silu_quant_tiling.cpp 中 GetWorkspaceSize 的逻辑
static GroupNormSiluQuantTestParam testCases[] = {
    {"group_norm_silu_quant_0", {1, 4, 2, 2}, ge::DT_FLOAT16, ge::FORMAT_ND, {4}, ge::DT_FLOAT16, ge::FORMAT_ND, {4}, ge::DT_FLOAT16, ge::FORMAT_ND, {1}, ge::DT_FLOAT, ge::FORMAT_ND, {1, 4, 2, 2}, ge::DT_INT8, ge::FORMAT_ND, {1, 2}, ge::DT_FLOAT16, ge::FORMAT_ND, {1, 2}, ge::DT_FLOAT16, ge::FORMAT_ND, "Ascend910B", ge::GRAPH_SUCCESS, 0UL, "0 1 0 ", {0}, 64, 262144, 4096},
};

class GroupNormSiluQuantTilingTest : public testing::TestWithParam<GroupNormSiluQuantTestParam> {
protected:
    static void SetUpTestCase() {
        std::cout << "GroupNormSiluQuantTilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase() {
        std::cout << "GroupNormSiluQuantTilingTest TearDown." << std::endl;
    }
};

struct GroupNormSiluQuantCompileInfo {} compileInfo;

static void TestOneParamCase(const GroupNormSiluQuantTestParam &param)
{
    gert::StorageShape selfShape = {param.selfShape, param.selfShape};
    gert::StorageShape gammaShape = {param.gammaShape, param.gammaShape};
    gert::StorageShape betaShape = {param.betaShape, param.betaShape};
    gert::StorageShape quantScaleShape = {param.quantScaleShape, param.quantScaleShape};
    gert::StorageShape outShape = {param.outShape, param.outShape};
    gert::StorageShape meanOutShape = {param.meanOutShape, param.meanOutShape};
    gert::StorageShape rstdOutShape = {param.rstdOutShape, param.rstdOutShape};
    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{selfShape, param.selfDtype, param.selfFormat},
        {gammaShape, param.gammaDtype, param.gammaFormat},
        {betaShape, param.betaDtype, param.betaFormat},
        {quantScaleShape, param.quantScaleDtype, param.quantScaleFormat}});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{outShape, param.outDtype, param.outFormat},
        {meanOutShape, param.meanOutDtype, param.meanOutFormat},
        {rstdOutShape, param.rstdOutDtype, param.rstdOutFormat}});
    std::vector<gert::TilingContextPara::OpAttr> attrs_;
    attrs_.push_back(gert::TilingContextPara::OpAttr("group", Ops::Math::AnyValue::CreateFrom<int64_t>(2)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("eps", Ops::Math::AnyValue::CreateFrom<float>(1e-05f)));
    attrs_.push_back(gert::TilingContextPara::OpAttr("activateSilu", Ops::Math::AnyValue::CreateFrom<bool>(true)));
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

TEST_P(GroupNormSiluQuantTilingTest, tiling_test)
{
    const GroupNormSiluQuantTestParam &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    GroupNormSiluQuantTilingTests,
    GroupNormSiluQuantTilingTest,
    testing::ValuesIn(testCases));

}
