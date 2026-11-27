/* CTCLoss kernel entry. */

#include "ctc_loss.h"

enum class CtcLossTilingKey : uint32_t {
    FP16_INT32 = 0,
    FP16_INT64 = 1,
    FP32_INT32 = 2,
    FP32_INT64 = 3,
};

template <uint32_t schMode>
__global__ __aicore__ void ctc_loss(GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths,
                                    GM_ADDR target_lengths, GM_ADDR neg_log_likelihood, GM_ADDR workspace,
                                    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(4, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(5, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(6, KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(7, KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(CtcLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(CtcLossTilingData, tiling_data, tiling);
    constexpr uint32_t schedule_family = schMode & 12U;
    constexpr bool pair_compiled = schedule_family == 8U;
    constexpr bool asym_pair_compiled = schedule_family == 12U;
    if constexpr (!pair_compiled && !asym_pair_compiled) {
        if (AscendC::GetBlockIdx() >= tiling_data.batchSize &&
            (tiling_data.reduction == 0 || tiling_data.batchSize == 1)) {
            return;
        }
    }
    GM_ADDR user_workspace = AscendC::GetUserWorkspace(workspace);
    constexpr uint32_t dtype_mode = schMode & 3U;
    if constexpr (dtype_mode == static_cast<uint32_t>(CtcLossTilingKey::FP16_INT32)) {
        NsCtcLoss::CtcLoss<half, int32_t, pair_compiled, asym_pair_compiled> op;
        op.Init(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, user_workspace, &tiling_data);
        op.Process();
    } else if constexpr (dtype_mode == static_cast<uint32_t>(CtcLossTilingKey::FP16_INT64)) {
        NsCtcLoss::CtcLoss<half, int64_t, pair_compiled, asym_pair_compiled> op;
        op.Init(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, user_workspace, &tiling_data);
        op.Process();
    } else if constexpr (dtype_mode == static_cast<uint32_t>(CtcLossTilingKey::FP32_INT32)) {
        NsCtcLoss::CtcLoss<float, int32_t, pair_compiled, asym_pair_compiled> op;
        op.Init(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, user_workspace, &tiling_data);
        op.Process();
    } else if constexpr (dtype_mode == static_cast<uint32_t>(CtcLossTilingKey::FP32_INT64)) {
        NsCtcLoss::CtcLoss<float, int64_t, pair_compiled, asym_pair_compiled> op;
        op.Init(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, user_workspace, &tiling_data);
        op.Process();
    }
}
