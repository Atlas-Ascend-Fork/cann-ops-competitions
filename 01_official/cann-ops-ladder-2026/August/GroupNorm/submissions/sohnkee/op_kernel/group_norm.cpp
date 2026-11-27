/*!
 * \file group_norm.cpp
 * \brief GroupNorm kernel entry.
 */

#include "group_norm.h"

template <uint32_t schMode>
__global__ __aicore__ void group_norm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(GroupNormTilingData);
    GET_TILING_DATA_WITH_STRUCT(GroupNormTilingData, tilingData, tiling);
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_2
    if constexpr (schMode == GROUPNORM_TPL_SCH_MODE_2) {
        AscendC::InitSocState();
        NsGroupNorm::TinyStaticGroupNorm<DTYPE_X> op;
        op.Init(x, gamma, beta, y, &tilingData);
        op.Process();
    }
#else
    NsGroupNorm::GroupNorm<DTYPE_X> op;
    op.Init(x, gamma, beta, y, &tilingData);
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_1
    if constexpr (schMode == GROUPNORM_TPL_SCH_MODE_1) {
        op.ProcessLowAFirstTile();
        return;
    }
#endif
#if TILING_KEY_VAR == GROUPNORM_TPL_SCH_MODE_3
    if constexpr (schMode == GROUPNORM_TPL_SCH_MODE_3) {
        op.ProcessTwoByteShiftedMoments();
        return;
    }
#endif
    op.Process();
#endif
}
