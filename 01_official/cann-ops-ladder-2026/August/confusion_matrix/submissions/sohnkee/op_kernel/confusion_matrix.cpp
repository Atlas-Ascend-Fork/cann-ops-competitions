/*!
 * \file confusion_matrix.cpp
 * \brief ConfusionMatrix 算子 kernel 入口
 */

#include "confusion_matrix.h"

template <uint32_t schMode>
__global__ __aicore__ void confusion_matrix(GM_ADDR labels, GM_ADDR predictions, GM_ADDR weights, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // All specialized paths, including the direct sparse path, execute on AIV.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ConfusionMatrixTilingData);
#if TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_6
    const __gm__ ConfusionMatrixTilingData* tilingData =
        reinterpret_cast<const __gm__ ConfusionMatrixTilingData*>(tiling);
    const uint64_t packedShape = static_cast<uint64_t>(tilingData->blockFactor);
    NsConfusionMatrix::ConfusionMatrix op;
    op.InitAtomicSparse(
        labels,
        predictions,
        y,
        static_cast<int64_t>(static_cast<uint16_t>(packedShape)),
        static_cast<int64_t>(static_cast<uint16_t>(packedShape >> 16U)));
    op.ProcessAtomicSparse(static_cast<uint64_t>(static_cast<uint16_t>(packedShape >> 32U)));
#elif TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_0
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    const __gm__ ConfusionMatrixTilingData* tilingData =
        reinterpret_cast<const __gm__ ConfusionMatrixTilingData*>(tiling);
    NsConfusionMatrix::ProcessTinyInt32NoWeight(
        labels,
        predictions,
        y,
        static_cast<uint32_t>(tilingData->totalNum),
        static_cast<uint32_t>(tilingData->numClasses));
#elif TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_2
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
    NsConfusionMatrix::ProcessTinyInt32WeightInt32(labels, predictions, weights, y, &tilingData);
#elif TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_3
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }
    GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
    NsConfusionMatrix::ProcessTinyInt64WeightFloat(labels, predictions, weights, y, &tilingData);
#elif TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_4
    if (AscendC::GetBlockIdx() >= 3) {
        return;
    }
    const __gm__ ConfusionMatrixTilingData* tilingData =
        reinterpret_cast<const __gm__ ConfusionMatrixTilingData*>(tiling);
    NsConfusionMatrix::ProcessC3VectorHistogram<false>(
        labels, predictions, y, static_cast<uint64_t>(tilingData->totalNum));
#elif TILING_KEY_VAR == CONFUSIONMATRIX_TPL_SCH_MODE_5
    if (AscendC::GetBlockIdx() >= 3) {
        return;
    }
    const __gm__ ConfusionMatrixTilingData* tilingData =
        reinterpret_cast<const __gm__ ConfusionMatrixTilingData*>(tiling);
    NsConfusionMatrix::ProcessC3VectorHistogram<true>(
        labels, predictions, y, static_cast<uint64_t>(tilingData->totalNum));
#else
    GET_TILING_DATA_WITH_STRUCT(ConfusionMatrixTilingData, tilingData, tiling);
    NsConfusionMatrix::ConfusionMatrix op;
    op.Init(labels, predictions, weights, y, &tilingData);
    op.Process();
#endif
}
