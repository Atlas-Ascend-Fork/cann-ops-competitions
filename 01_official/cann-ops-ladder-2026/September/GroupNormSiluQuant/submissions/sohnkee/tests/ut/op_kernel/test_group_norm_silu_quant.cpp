/*!
 * \file test_group_norm_silu_quant.cpp
 * \brief GroupNormSiluQuant 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "group_norm_silu_quant_tiling.h"
#include "../../../op_kernel/group_norm_silu_quant.cpp"

#include <array>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

using namespace std;

static uint16_t FloatToHalf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (exp << 10) | mant;
}

static uint16_t FloatToBFloat16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return (uint16_t)(bits >> 16);
}

class GroupNormSiluQuantKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "GroupNormSiluQuantKernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "GroupNormSiluQuantKernelTest TearDown" << endl;
    }
};

TEST_F(GroupNormSiluQuantKernelTest, test_kernel_run)
{
    constexpr size_t size = 16;
    constexpr size_t tilingDataSize = sizeof(GroupNormSiluQuantTilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t selfByteSize = 16 * 2;
    constexpr size_t gammaByteSize = 4 * 2;
    constexpr size_t betaByteSize = 4 * 2;
    constexpr size_t quantScaleByteSize = 1 * 4;
    constexpr size_t outByteSize = 16 * 1;
    constexpr size_t meanOutByteSize = 2 * 2;
    constexpr size_t rstdOutByteSize = 2 * 2;
    std::vector<float> selfHost(16, 1);
    std::vector<float> gammaHost(4, 1);
    std::vector<float> betaHost(4, 1);
    std::vector<float> quantScaleHost(1, 1);
    std::vector<int8_t> outHost(16, 0);
    std::vector<float> meanOutHost(2, 0);
    std::vector<float> rstdOutHost(2, 0);
    
    
    uint8_t* self = (uint8_t*)AscendC::GmAlloc(selfByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(betaByteSize);
    uint8_t* quantScale = (uint8_t*)AscendC::GmAlloc(quantScaleByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* meanOut = (uint8_t*)AscendC::GmAlloc(meanOutByteSize);
    uint8_t* rstdOut = (uint8_t*)AscendC::GmAlloc(rstdOutByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    for (size_t _i = 0; _i < 16; _i++) { uint16_t _h = FloatToHalf(selfHost[_i]); memcpy(self + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 4; _i++) { uint16_t _h = FloatToHalf(gammaHost[_i]); memcpy(gamma + _i * 2, &_h, 2); }
    for (size_t _i = 0; _i < 4; _i++) { uint16_t _h = FloatToHalf(betaHost[_i]); memcpy(beta + _i * 2, &_h, 2); }
    memcpy(quantScale, quantScaleHost.data(), quantScaleByteSize);
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/group_norm_silu_quant_tiling_data.h 中的字段定义
    //         参考 op_host/group_norm_silu_quant_tiling.cpp 中的 tiling 计算逻辑
    GroupNormSiluQuantTilingData* tilingData = reinterpret_cast<GroupNormSiluQuantTilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/group_norm_silu_quant_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((group_norm_silu_quant<0>), numBlocks, self, gamma, beta, quantScale, out, meanOut, rstdOut, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(outHost.data(), out, outByteSize);
    { std::ofstream _ofs("int8_output_group_norm_silu_quant_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(outHost.data()), outByteSize); }
    memcpy(meanOutHost.data(), meanOut, meanOutByteSize);
    { std::ofstream _ofs("float16_output_group_norm_silu_quant_1.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(meanOutHost.data()), meanOutByteSize); }
    memcpy(rstdOutHost.data(), rstdOut, rstdOutByteSize);
    { std::ofstream _ofs("float16_output_group_norm_silu_quant_2.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(rstdOutHost.data()), rstdOutByteSize); }
    
    AscendC::GmFree(self);
    AscendC::GmFree(gamma);
    AscendC::GmFree(beta);
    AscendC::GmFree(quantScale);
    AscendC::GmFree(out);
    AscendC::GmFree(meanOut);
    AscendC::GmFree(rstdOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
