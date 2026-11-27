/*!
 * \file test_quant_batch_mat_mul_v3.cpp
 * \brief QuantBatchMatMulV3 算子 kernel UT 测试
 * 
 * 独立运行，直接构造 tilingData，不依赖 op_host UT
 */

#include "quant_batch_mat_mul_v3_tiling.h"
#include "../../../op_kernel/quant_batch_mat_mul_v3.cpp"

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

class QuantBatchMatMulV3KernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "QuantBatchMatMulV3KernelTest SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "QuantBatchMatMulV3KernelTest TearDown" << endl;
    }
};

TEST_F(QuantBatchMatMulV3KernelTest, test_kernel_run)
{
    constexpr size_t size = 8388608;
    constexpr size_t tilingDataSize = sizeof(QuantBatchMatMulV3TilingData);
    constexpr uint32_t numBlocks = 1;

    constexpr size_t x1ByteSize = 8388608 * 1;
    constexpr size_t x2ByteSize = 6291456 * 1;
    constexpr size_t scaleByteSize = 96 * 4;
    constexpr size_t pertokenScaleByteSize = 128 * 4;
    constexpr size_t outByteSize = 12288 * 2;
    std::vector<int8_t> x1Host(8388608, 1);
    std::vector<int8_t> x2Host(6291456, 1);
    std::vector<float> scaleHost(96, 1);
    std::vector<float> pertokenScaleHost(128, 1);
    std::vector<float> outHost(12288, 0);
    
    
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(x1ByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(x2ByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* pertokenScale = (uint8_t*)AscendC::GmAlloc(pertokenScaleByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    
    memcpy(x1, x1Host.data(), x1ByteSize);
    memcpy(x2, x2Host.data(), x2ByteSize);
    memcpy(scale, scaleHost.data(), scaleByteSize);
    memcpy(pertokenScale, pertokenScaleHost.data(), pertokenScaleByteSize);
    
    // TODO: 以下 tilingData 字段基于初始模板的 TilingData 结构。
    //       修改 TilingData 结构后请更新字段名和赋值：
    //         参考 op_kernel/quant_batch_mat_mul_v3_tiling_data.h 中的字段定义
    //         参考 op_host/quant_batch_mat_mul_v3_tiling.cpp 中的 tiling 计算逻辑
    QuantBatchMatMulV3TilingData* tilingData = reinterpret_cast<QuantBatchMatMulV3TilingData*>(tiling);
    tilingData->totalNum = size;
    tilingData->blockFactor = size;
    tilingData->ubFactor = size;
    
    // TODO: tilingKey 应与 op_host/quant_batch_mat_mul_v3_tiling.cpp 中 SetTilingKey 设置的值一致
    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    
    ICPU_RUN_KF((quant_batch_mat_mul_v3<1>), numBlocks, x1, x2, scale, pertokenScale, out, workspace, tiling);
    
    // 将动态输出的 packed buffer 拆回 individual buffers
    
    
    // 将 output 数据保存到 bin 文件供 compare_data.py 比对
    memcpy(outHost.data(), out, outByteSize);
    { std::ofstream _ofs("bfloat16_output_quant_batch_mat_mul_v3_0.bin", std::ios::binary); _ofs.write(reinterpret_cast<const char*>(outHost.data()), outByteSize); }
    
    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(scale);
    AscendC::GmFree(pertokenScale);
    AscendC::GmFree(out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
