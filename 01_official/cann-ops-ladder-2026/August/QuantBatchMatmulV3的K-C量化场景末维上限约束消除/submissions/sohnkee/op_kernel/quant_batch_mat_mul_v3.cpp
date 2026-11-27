/*!
 * \file quant_batch_mat_mul_v3.cpp
 * \brief QuantBatchMatMulV3 算子 kernel 入口
 */

#include "quant_batch_mat_mul_v3.h"

enum class QuantBatchMatMulV3TilingKey : uint32_t
{
    TILING_KEY_QUANTBATCHMATMULV3_MODE_0 = 0,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_1 = 1,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_2 = 2,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_3 = 3,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_4 = 4,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_5 = 5,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_6 = 6,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_7 = 7,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_8 = 8,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_9 = 9,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_10 = 10,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_11 = 11,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_12 = 12,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_13 = 13,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_14 = 14,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_15 = 15,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_16 = 16,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_17 = 17,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_18 = 18,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_19 = 19,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_20 = 20,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_21 = 21,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_22 = 22,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_23 = 23,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_24 = 24,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_25 = 25,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_26 = 26,
    TILING_KEY_QUANTBATCHMATMULV3_MODE_27 = 27,
};

template <uint32_t schMode>
__global__ __aicore__ void quant_batch_mat_mul_v3(GM_ADDR x1, GM_ADDR x2, GM_ADDR scale, GM_ADDR pertokenScale, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    KERNEL_TASK_TYPE(6, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(7, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(8, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(9, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(10, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(11, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(14, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(20, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(24, KERNEL_TYPE_MIX_AIC_1_2);
    KERNEL_TASK_TYPE(27, KERNEL_TYPE_MIX_AIC_1_1);
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }
    AscendC::TPipe tPipe;
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_0)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, false> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_1)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_2)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<true, false> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_3)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<true, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_4)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Case05AicLocalA1<false, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_5)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransBDirectFusion<> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_6)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, false> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.ProcessSingleColumnVector();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_7)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransADirectFusion<true, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_8)) {
        if ASCEND_IS_AIV {
            NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, false> op;
            op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData, true);
            op.Process();
            tPipe.Destroy();
        }
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_9)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransBDirectFusion<> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_10)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransADirectFusion<true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_12)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, false, false, true, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_13)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<true, false, false, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_11)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransBDirectFusion<true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_14)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransADirectFusion<true, true, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_15)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, false, false, false, false, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_16)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Case05AicLocalA1<true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_17)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Case09AicLocalA1 op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_18)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, false, false, false, false, false, true> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.ProcessRowMajorMidNInt32Public();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_19)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Matmul<false, false> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm, &tilingData.matmulTiling);
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.ProcessRowMajorSingleRowInt32Public();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_20)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransADirectFusion<true, false, false, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_21)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Case05AicLocalA1<true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_22)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Case05AicLocalA1<false, false, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_23)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Case05AicLocalA1<false, false, false, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_24)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransADirectFusion<true, false, false, true, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_25)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransBDirectFusion<false, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_26)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3Case05AicLocalA1<false, false, false, false, true> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
    if constexpr (schMode == static_cast<uint32_t>(QuantBatchMatMulV3TilingKey::TILING_KEY_QUANTBATCHMATMULV3_MODE_27)) {
        NsQuantBatchMatMulV3::QuantBatchMatMulV3TransADirectFusion<true, true, true, false, false, false> op;
        op.Init(x1, x2, scale, pertokenScale, out, userWorkspace, &tPipe, &tilingData);
        op.Process();
        tPipe.Destroy();
    }
}
