/*!
 * \file recompute_wu.cpp
 * \brief RecomputeWu 算子 kernel 入口
 */

#include "recompute_wu.h"
#include "recompute_wu_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void recompute_wu(GM_ADDR k, GM_ADDR v, GM_ADDR A, GM_ADDR beta, GM_ADDR g,
    GM_ADDR globalChunkOffsets, GM_ADDR w, GM_ADDR u, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(RecomputeWuTilingData);
    GET_TILING_DATA_WITH_STRUCT(RecomputeWuTilingData, tilingData, tiling);

    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    AscendC::TPipe pipe;
    constexpr bool useOfficialMmad = schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD ||
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_DIRECTV ||
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_GROUP4 ||
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_GROUP_PIPELINE ||
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_SMALL_GROUP_DIRECTV ||
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_STAGED_OUTPUT;
    constexpr bool useDirectValue = schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_DIRECTV ||
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_GROUP4 ||
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_GROUP_PIPELINE ||
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_SMALL_GROUP_DIRECTV;
    constexpr bool useSyncedDirectValue =
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_SMALL_GROUP_DIRECTV;
    constexpr bool useDirectD128HeadBatch =
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_GROUP4 ||
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_GROUP_PIPELINE;
    constexpr bool useD128GroupPipeline =
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_GROUP_PIPELINE;
    constexpr bool useStagedOutput =
        schMode == RECOMPUTEWU_TPL_OFFICIAL_MMAD_STAGED_OUTPUT;
    if ASCEND_IS_AIV {
        NsRecomputeWu::RecomputeWuVector<useOfficialMmad,
            schMode == RECOMPUTEWU_TPL_NORMAL_PACKED_WIDE,
            schMode == RECOMPUTEWU_TPL_BATCH_CHUNKS, useDirectValue,
            useDirectD128HeadBatch, useD128GroupPipeline, useSyncedDirectValue,
            useStagedOutput> vectorOp;
        vectorOp.Init(A, k, v, beta, g, globalChunkOffsets, w, u, userWorkspace, &tilingData, &pipe);
        vectorOp.Process();
    }

    if ASCEND_IS_AIC {
        if constexpr (useOfficialMmad) {
            NsRecomputeWu::RecomputeWuCubeMmad<useDirectValue,
                useDirectD128HeadBatch, useD128GroupPipeline, useSyncedDirectValue,
                useStagedOutput> cubeOp;
            cubeOp.Init(k, v, globalChunkOffsets, w, u, userWorkspace, &tilingData, &pipe);
            cubeOp.Process();
        } else if constexpr (schMode == RECOMPUTEWU_TPL_BATCH_CHUNKS) {
            NsRecomputeWu::RecomputeWuCubeBatchChunks cubeOp;
            cubeOp.Init(globalChunkOffsets, w, u, userWorkspace, &tilingData, &pipe);
            cubeOp.Process();
        } else {
            NsRecomputeWu::RecomputeWuCubeMatmul<schMode == RECOMPUTEWU_TPL_MDL,
                schMode == RECOMPUTEWU_TPL_NORMAL_PACKED_WIDE> cubeOp;
            cubeOp.Init(k, v, globalChunkOffsets, w, u, userWorkspace, &tilingData, &pipe);
            cubeOp.Process();
        }
    }
}
