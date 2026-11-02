/*!
 * \file recompute_wu.h
 * \brief RecomputeWu 算子 kernel 实现
 */

#ifndef RECOMPUTEWU_H
#define RECOMPUTEWU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "recompute_wu_tiling_data.h"

namespace NsRecomputeWu {

using namespace AscendC;

constexpr uint32_t CHUNK_SIZE = 64;
constexpr uint32_t DIM_TILE = 128;
constexpr uint32_t MAX_DIM = 256;
constexpr uint8_t FLOAT_ROW_BLOCKS = static_cast<uint8_t>(CHUNK_SIZE * sizeof(float) / 32U);
constexpr uint32_t READY_FLAG_BASE = 0;
constexpr uint32_t DONE_FLAG = 10;
constexpr uint32_t MATMUL_W_READY_FLAG_0 = 2;
constexpr uint32_t MATMUL_U_READY_FLAG_0 = 3;
constexpr uint32_t MATMUL_DONE_FLAG_0 = 4;
constexpr uint32_t MATMUL_W_READY_FLAG_1 = 6;
constexpr uint32_t MATMUL_U_READY_FLAG_1 = 7;
constexpr uint32_t MATMUL_DONE_FLAG_1 = 8;
constexpr uint8_t SYNC_MODE2 = 2;

constexpr uint32_t A_ELEMS = CHUNK_SIZE * CHUNK_SIZE;
constexpr uint32_t TILE_MATRIX_ELEMS = CHUNK_SIZE * MAX_DIM;
constexpr uint32_t D128_MATRIX_ELEMS = CHUNK_SIZE * DIM_TILE;
constexpr uint32_t D128_CUBE_SLOTS = 4U;
constexpr uint32_t WS_MATRIX_ELEMS = CHUNK_SIZE * MAX_DIM;
constexpr uint32_t WS_AU_OFFSET = 0;
constexpr uint32_t WS_AW_OFFSET = WS_AU_OFFSET + A_ELEMS;
constexpr uint32_t WS_BU_OFFSET = WS_AW_OFFSET + A_ELEMS;
constexpr uint32_t WS_BW_OFFSET = WS_BU_OFFSET + WS_MATRIX_ELEMS;
constexpr uint32_t WS_BUFFER_ELEMS = WS_BW_OFFSET + WS_MATRIX_ELEMS;
constexpr uint32_t MAX_CHUNKS_PER_TASK = 2U;
constexpr uint32_t MAX_BATCH_CHUNKS_PER_TASK = 4U;
constexpr uint32_t MAX_HEADS_PER_TASK = 4U;
constexpr uint32_t PIPELINE_DEPTH = 5U;
constexpr uint32_t GROUP_PIPELINE_DEPTH = 3U;
constexpr uint32_t GROUP_HEAD_COUNT = MAX_HEADS_PER_TASK;
constexpr uint32_t WS_BATCH_BUFFER_ELEMS = WS_BUFFER_ELEMS * MAX_CHUNKS_PER_TASK;
constexpr uint32_t WS_PER_CORE_ELEMS = WS_BATCH_BUFFER_ELEMS * PIPELINE_DEPTH;
constexpr uint32_t WS_PER_CORE_BYTES = WS_PER_CORE_ELEMS * sizeof(bfloat16_t);
constexpr uint32_t AIV_SLICE_ELEMS = A_ELEMS / 2U;
constexpr uint32_t BATCH_RIGHT_ELEMS = CHUNK_SIZE * CHUNK_SIZE;
constexpr uint32_t BATCH_AW_OFFSET = 0U;
constexpr uint32_t BATCH_BW_OFFSET = BATCH_AW_OFFSET + MAX_BATCH_CHUNKS_PER_TASK * A_ELEMS;
constexpr uint32_t BATCH_AU_OFFSET = BATCH_BW_OFFSET + MAX_BATCH_CHUNKS_PER_TASK * BATCH_RIGHT_ELEMS;
constexpr uint32_t BATCH_BU_OFFSET = BATCH_AU_OFFSET + MAX_BATCH_CHUNKS_PER_TASK * A_ELEMS;
constexpr uint32_t BATCH_BUFFER_ELEMS =
    BATCH_BU_OFFSET + MAX_BATCH_CHUNKS_PER_TASK * BATCH_RIGHT_ELEMS;

// The batch-chunk path lays out a whole head/chunk task in one workspace
// buffer.  A single W/U ready handshake then covers all heads, while K is
// stored once per chunk and reused by every head in the GQA batch.
constexpr uint32_t BATCH_TASK_MAX_DIM = 48U;
constexpr uint32_t BATCH_TASK_MAX_MATRICES = MAX_HEADS_PER_TASK * MAX_BATCH_CHUNKS_PER_TASK;
constexpr uint32_t BATCH_TASK_RIGHT_ELEMS = CHUNK_SIZE * BATCH_TASK_MAX_DIM;
constexpr uint32_t BATCH_TASK_AW_OFFSET = 0U;
constexpr uint32_t BATCH_TASK_BW_OFFSET =
    BATCH_TASK_AW_OFFSET + BATCH_TASK_MAX_MATRICES * A_ELEMS;
constexpr uint32_t BATCH_TASK_AU_OFFSET =
    BATCH_TASK_BW_OFFSET + MAX_BATCH_CHUNKS_PER_TASK * BATCH_TASK_RIGHT_ELEMS;
constexpr uint32_t BATCH_TASK_BU_OFFSET =
    BATCH_TASK_AU_OFFSET + BATCH_TASK_MAX_MATRICES * A_ELEMS;
constexpr uint32_t BATCH_TASK_BUFFER_ELEMS =
    BATCH_TASK_BU_OFFSET + BATCH_TASK_MAX_MATRICES * BATCH_TASK_RIGHT_ELEMS;

constexpr uint32_t SCRATCH_BYTES = 112 * 1024;
constexpr uint32_t RAW_BF16_BYTE_OFFSET = 0;
constexpr uint32_t RIGHT_BF16_BYTE_OFFSET = 8 * 1024;
constexpr uint32_t MATRIX_FLOAT_BYTE_OFFSET = 40 * 1024;
constexpr uint32_t AW_FLOAT_BYTE_OFFSET = 56 * 1024;
constexpr uint32_t A_LOCAL_BYTE_OFFSET = 72 * 1024;
constexpr uint32_t SMALL_BYTE_OFFSET = 80 * 1024;

using MatmulAType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using MatmulBType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using MatmulCType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using MatmulBiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
using BatchMatmulAType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false, LayoutMode::NORMAL>;
using BatchMatmulBType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false, LayoutMode::NORMAL>;
using BatchMatmulCType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false, LayoutMode::NORMAL>;

template <bool USE_MDL>
__aicore__ constexpr MatmulConfig GetRecomputeWuMatmulConfig()
{
    MatmulConfig config = USE_MDL ? GetMDLConfig(true) : GetNormalConfig();
    config.enableGetTensorC = false;
    config.enableSetBias = false;
    config.enableQuantVector = false;
    config.enableSetDefineData = false;
    config.iterateMode = IterateMode::ITERATE_MODE_ALL;
    return config;
}

template <bool USE_MDL>
constexpr MatmulConfig RECOMPUTE_WU_MATMUL_CONFIG = GetRecomputeWuMatmulConfig<USE_MDL>();

__aicore__ inline uint32_t AlignUp32(uint32_t value)
{
    return (value + 31U) & ~31U;
}

__aicore__ inline uint32_t AlignUp16(uint32_t value)
{
    return (value + 15U) & ~15U;
}

__aicore__ inline uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    if (divisor == 0U) {
        return 0U;
    }
    return (value + divisor - 1U) / divisor;
}

__aicore__ inline uint32_t SelectDimTile(uint32_t outputDim)
{
    if (outputDim <= 64U) {
        return 64U;
    }
    return outputDim <= DIM_TILE ? DIM_TILE : MAX_DIM;
}

template <bool PACKED_WIDE>
__aicore__ inline uint32_t SelectPackedDimTile(uint32_t outputDim)
{
    if constexpr (PACKED_WIDE) {
        return MAX_DIM;
    }
    return SelectDimTile(outputDim);
}

__aicore__ inline bool UseDirectKeyPath(const RecomputeWuTilingData* tiling)
{
    const uint32_t groupSize = tiling->headNum / tiling->groupHeadNum;
    if (groupSize < 4U) {
        return true;
    }
    const uint32_t shortTaskLimit = tiling->blockDim * 4U;
    const uint32_t schedulingTaskCount = tiling->taskCount * tiling->chunksPerTask;
    return schedulingTaskCount <= shortTaskLimit && tiling->keyDim >= CHUNK_SIZE;
}

__aicore__ inline bool UseCompactWorkspace(const RecomputeWuTilingData* tiling)
{
    return UseDirectKeyPath(tiling) && tiling->valueDim >= DIM_TILE;
}

template <bool PACKED_WIDE = false, bool FORCE_DIRECT_KEY = false>
__aicore__ inline uint32_t SelectValueTile(const RecomputeWuTilingData* tiling)
{
    return (FORCE_DIRECT_KEY || UseCompactWorkspace(tiling)) ?
        MAX_DIM : SelectPackedDimTile<PACKED_WIDE>(tiling->valueDim);
}

template <bool FORCE_DIRECT_KEY = false>
__aicore__ inline uint32_t GetWorkspaceBufferElems(const RecomputeWuTilingData* tiling)
{
    if (!FORCE_DIRECT_KEY && !UseCompactWorkspace(tiling)) {
        return WS_BUFFER_ELEMS;
    }
    return WS_BU_OFFSET + CHUNK_SIZE * SelectValueTile<false, FORCE_DIRECT_KEY>(tiling);
}

__aicore__ inline uint32_t GetOfficialMmadStagingBaseElems(
    const RecomputeWuTilingData* tiling)
{
    return GetWorkspaceBufferElems(tiling) * tiling->chunksPerTask * PIPELINE_DEPTH;
}

__aicore__ inline uint32_t GetOfficialMmadStagedOutputOffset(
    const RecomputeWuTilingData* tiling, uint32_t bufferIdx, uint32_t chunkOffset, bool isU)
{
    const uint32_t keyMatrixElems = CHUNK_SIZE * AlignUp16(tiling->keyDim);
    const uint32_t valueMatrixElems = CHUNK_SIZE * AlignUp16(tiling->valueDim);
    const uint32_t matrixElems = isU ? valueMatrixElems : keyMatrixElems;
    const uint32_t bufferElems =
        tiling->chunksPerTask * (keyMatrixElems + valueMatrixElems);
    const uint32_t outputBase = isU ? tiling->chunksPerTask * keyMatrixElems : 0U;
    return GetOfficialMmadStagingBaseElems(tiling) + bufferIdx * bufferElems +
        outputBase + chunkOffset * matrixElems;
}

__aicore__ inline bool UsePackedD128HeadBatchPath(const RecomputeWuTilingData* tiling)
{
    const uint32_t groupSize = tiling->headNum / tiling->groupHeadNum;
    return !UseDirectKeyPath(tiling) && tiling->keyDim == DIM_TILE &&
        tiling->valueDim == DIM_TILE && tiling->headsPerTask > 1U && groupSize >= 8U;
}

template <bool USE_OFFICIAL_MMAD>
__aicore__ inline uint32_t GetWReadyFlag(uint32_t bufferIdx)
{
    if constexpr (!USE_OFFICIAL_MMAD) {
        return bufferIdx == 0U ? MATMUL_W_READY_FLAG_0 : MATMUL_W_READY_FLAG_1;
    }
    return READY_FLAG_BASE + bufferIdx * 2U;
}

template <bool USE_OFFICIAL_MMAD>
__aicore__ inline uint32_t GetUReadyFlag(uint32_t bufferIdx)
{
    if constexpr (!USE_OFFICIAL_MMAD) {
        return bufferIdx == 0U ? MATMUL_U_READY_FLAG_0 : MATMUL_U_READY_FLAG_1;
    }
    return GetWReadyFlag<true>(bufferIdx) + 1U;
}

template <bool USE_OFFICIAL_MMAD>
__aicore__ inline uint32_t GetDoneFlag(uint32_t bufferIdx)
{
    if constexpr (USE_OFFICIAL_MMAD) {
        return DONE_FLAG;
    }
    return bufferIdx == 0U ? MATMUL_DONE_FLAG_0 : MATMUL_DONE_FLAG_1;
}

template <bool USE_OFFICIAL_MMAD, bool PACKED_WIDE = false, bool BATCH_CHUNKS = false,
    bool DIRECT_VALUE = false, bool DIRECT_D128_HEAD_BATCH = false,
    bool D128_GROUP_PIPELINE = false, bool SYNCED_DIRECT_VALUE = false,
    bool STAGED_OUTPUT = false>
class RecomputeWuVector {
public:
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR k, GM_ADDR v, GM_ADDR beta, GM_ADDR g,
        GM_ADDR globalChunkOffsets, GM_ADDR w, GM_ADDR u, GM_ADDR userWorkspace,
        const RecomputeWuTilingData* tilingData, TPipe* pipe)
    {
        tiling_ = tilingData;
        subBlockIdx_ = GetSubBlockIdx();
        blockIdx_ = GetBlockIdx() / 2U;

        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(A));
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
        betaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(beta));
        gGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(g));
        offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(globalChunkOffsets));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w));
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(u));

        const uint32_t workspacePerCoreElems = tiling_->workspacePerCore / sizeof(bfloat16_t);
        const uint64_t coreWsOffset = static_cast<uint64_t>(blockIdx_) * workspacePerCoreElems;
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWorkspace) + coreWsOffset,
            workspacePerCoreElems);
        pipe->InitBuffer(scratch_, SCRATCH_BYTES);
    }

    __aicore__ inline void Process()
    {
        if constexpr (BATCH_CHUNKS) {
            ProcessBatchChunks();
            return;
        }
        if constexpr (D128_GROUP_PIPELINE) {
            ProcessD128GroupPipeline();
            return;
        }
        const bool directKey = DIRECT_D128_HEAD_BATCH || UseDirectKeyPath(tiling_);
        bool packedD128HeadBatch = false;
        uint32_t pipelineDepth = 2U;
        if constexpr (USE_OFFICIAL_MMAD) {
            packedD128HeadBatch = UsePackedD128HeadBatchPath(tiling_);
            pipelineDepth = directKey || packedD128HeadBatch ? PIPELINE_DEPTH : 2U;
        }
        const uint32_t workspaceBufferElems =
            GetWorkspaceBufferElems<DIRECT_D128_HEAD_BATCH>(tiling_);
        uint32_t stagedHeadIdx[PIPELINE_DEPTH] = {0U, 0U, 0U, 0U, 0U};
        uint32_t stagedFirstChunk[PIPELINE_DEPTH] = {0U, 0U, 0U, 0U, 0U};
        uint32_t stagedChunkCount[PIPELINE_DEPTH] = {0U, 0U, 0U, 0U, 0U};
        uint32_t taskSeq = 0;
        for (uint32_t task = blockIdx_; task < tiling_->taskCount; task += tiling_->blockDim) {
            const uint32_t tasksPerChunkGroup = tiling_->groupHeadNum * tiling_->headBatchCount;
            const uint32_t chunkGroup = task / tasksPerChunkGroup;
            const uint32_t taskInChunkGroup = task - chunkGroup * tasksPerChunkGroup;
            const uint32_t groupHead = taskInChunkGroup / tiling_->headBatchCount;
            const uint32_t headBatch = taskInChunkGroup - groupHead * tiling_->headBatchCount;
            const uint32_t groupSize = tiling_->headNum / tiling_->groupHeadNum;
            const uint32_t headStart = groupHead * groupSize + headBatch * tiling_->headsPerTask;
            uint32_t headCount = groupSize - headBatch * tiling_->headsPerTask;
            if (headCount > tiling_->headsPerTask) {
                headCount = tiling_->headsPerTask;
            }
            const uint32_t firstChunk = chunkGroup * tiling_->chunksPerTask;
            uint32_t chunkCount = tiling_->numChunks - firstChunk;
            if (chunkCount > tiling_->chunksPerTask) {
                chunkCount = tiling_->chunksPerTask;
            }
            const bool splitByChunk = BATCH_CHUNKS || chunkCount == MAX_CHUNKS_PER_TASK;
            if constexpr (USE_OFFICIAL_MMAD) {
                if ((packedD128HeadBatch || DIRECT_D128_HEAD_BATCH) && headCount > 1U) {
                    const uint32_t batchBufferElems = workspaceBufferElems * tiling_->chunksPerTask;
                    const uint32_t commonBufferIdx = taskSeq % pipelineDepth;
                    const uint32_t commonBufferBase = commonBufferIdx * batchBufferElems;

                // Produce every W operand first. K is shared by all heads in this
                // GQA batch, so pack it only into the first head's workspace slot.
                for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                    const uint32_t headSeq = taskSeq + headOffset;
                    const uint32_t headIdx = headStart + headOffset;
                    const uint32_t bufferIdx = headSeq % pipelineDepth;
                    const uint32_t bufferBase = bufferIdx * batchBufferElems;
                    if (headSeq >= pipelineDepth) {
                        CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(GetDoneFlag<USE_OFFICIAL_MMAD>(bufferIdx));
                    }

                    for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                        if (splitByChunk && chunkOffset != subBlockIdx_) {
                            continue;
                        }
                        const uint32_t chunkIdx = firstChunk + chunkOffset;
                        const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
                        const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                        if (startValue < 0 || endValue <= startValue) {
                            continue;
                        }
                        const uint32_t chunkStart = static_cast<uint32_t>(startValue);
                        uint32_t chunkLen = static_cast<uint32_t>(endValue - startValue);
                        if (chunkLen > CHUNK_SIZE) {
                            chunkLen = CHUNK_SIZE;
                        }
                        const uint32_t firstSliceRows = (chunkLen + 1U) / 2U;
                        const uint32_t rowStart = splitByChunk ? 0U :
                            (subBlockIdx_ == 0U ? 0U : firstSliceRows);
                        const uint32_t rowCount = splitByChunk ? chunkLen :
                            (subBlockIdx_ == 0U ? firstSliceRows : chunkLen - firstSliceRows);
                        if (rowCount == 0U) {
                            continue;
                        }
                        const uint32_t chunkBufferBase = bufferBase + chunkOffset * workspaceBufferElems;
                        PrepareFactor(chunkStart, chunkLen, headIdx, true);
                        PrepareScaledARowSlice(chunkStart, chunkLen, headIdx, rowStart, rowCount,
                            chunkBufferBase + WS_AW_OFFSET, chunkBufferBase + WS_AU_OFFSET,
                            splitByChunk ? 0U : chunkOffset);

                        if (!DIRECT_D128_HEAD_BATCH && headOffset == 0U) {
                            const uint32_t keyTileWidth = SelectDimTile(tiling_->keyDim);
                            const uint32_t keyTileElems = CHUNK_SIZE * keyTileWidth;
                            const uint32_t keyTiles = CeilDivU32(tiling_->keyDim, keyTileWidth);
                            const uint32_t commonChunkBase =
                                commonBufferBase + chunkOffset * workspaceBufferElems;
                            for (uint32_t tileIdx = 0; tileIdx < keyTiles; ++tileIdx) {
                                uint32_t validDim = tiling_->keyDim - tileIdx * keyTileWidth;
                                if (validDim > keyTileWidth) {
                                    validDim = keyTileWidth;
                                }
                                PackRightMatrixSlice(kGm_, chunkStart, rowStart, rowCount,
                                    groupHead, tiling_->groupHeadNum, tiling_->keyDim,
                                    tileIdx * keyTileWidth, validDim, keyTileWidth,
                                    commonChunkBase + WS_BW_OFFSET + tileIdx * keyTileElems,
                                    keyTileWidth);
                            }
                        }
                    }
                    if (DIRECT_D128_HEAD_BATCH || headOffset != 0U) {
                        // The next head reuses the local A scratch through MTE2.
                        WaitMte3ToMte2();
                    }
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(GetWReadyFlag<USE_OFFICIAL_MMAD>(bufferIdx));
                }

                if constexpr (SYNCED_DIRECT_VALUE) {
                    WaitMte3ToMte2();
                }
                if constexpr (!DIRECT_VALUE || SYNCED_DIRECT_VALUE) {
                    // While Cube consumes the W batch, prepare every head's V operand.
                    for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                        const uint32_t headSeq = taskSeq + headOffset;
                        const uint32_t headIdx = headStart + headOffset;
                        const uint32_t bufferIdx = headSeq % pipelineDepth;
                        const uint32_t bufferBase = bufferIdx * batchBufferElems;
                        if constexpr (!DIRECT_VALUE) {
                            for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                                if (splitByChunk && chunkOffset != subBlockIdx_) {
                                    continue;
                                }
                                const uint32_t chunkIdx = firstChunk + chunkOffset;
                                const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
                                const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                                if (startValue < 0 || endValue <= startValue) {
                                    continue;
                                }
                                const uint32_t chunkStart = static_cast<uint32_t>(startValue);
                                uint32_t chunkLen = static_cast<uint32_t>(endValue - startValue);
                                if (chunkLen > CHUNK_SIZE) {
                                    chunkLen = CHUNK_SIZE;
                                }
                                const uint32_t firstSliceRows = (chunkLen + 1U) / 2U;
                                const uint32_t rowStart = splitByChunk ? 0U :
                                    (subBlockIdx_ == 0U ? 0U : firstSliceRows);
                                const uint32_t rowCount = splitByChunk ? chunkLen :
                                    (subBlockIdx_ == 0U ? firstSliceRows : chunkLen - firstSliceRows);
                                if (rowCount == 0U) {
                                    continue;
                                }
                                const uint32_t chunkBufferBase =
                                    bufferBase + chunkOffset * workspaceBufferElems;
                                const uint32_t valueTileWidth =
                                    SelectValueTile<false, DIRECT_D128_HEAD_BATCH>(tiling_);
                                const uint32_t valueTileElems = CHUNK_SIZE * valueTileWidth;
                                const uint32_t valueTiles =
                                    CeilDivU32(tiling_->valueDim, valueTileWidth);
                                for (uint32_t tileIdx = 0; tileIdx < valueTiles; ++tileIdx) {
                                    uint32_t validDim = tiling_->valueDim - tileIdx * valueTileWidth;
                                    if (validDim > valueTileWidth) {
                                        validDim = valueTileWidth;
                                    }
                                    PackRightMatrixSlice(vGm_, chunkStart, rowStart, rowCount,
                                        headIdx, tiling_->headNum, tiling_->valueDim,
                                        tileIdx * valueTileWidth, validDim, valueTileWidth,
                                        chunkBufferBase + WS_BU_OFFSET + tileIdx * valueTileElems,
                                        valueTileWidth);
                                }
                            }
                        }
                        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(
                            GetUReadyFlag<USE_OFFICIAL_MMAD>(bufferIdx));
                    }
                }
                taskSeq += headCount;
                continue;
                }
            }
            for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                const uint32_t headIdx = headStart + headOffset;
                const uint32_t bufferIdx = taskSeq % pipelineDepth;
                const uint32_t batchBufferElems = BATCH_CHUNKS ? BATCH_BUFFER_ELEMS :
                    workspaceBufferElems * tiling_->chunksPerTask;
                const uint32_t bufferBase = bufferIdx * batchBufferElems;
                if (taskSeq >= pipelineDepth) {
                    CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(GetDoneFlag<USE_OFFICIAL_MMAD>(bufferIdx));
                    if constexpr (STAGED_OUTPUT) {
                        CopyStagedTask(bufferIdx, stagedHeadIdx[bufferIdx],
                            stagedFirstChunk[bufferIdx], stagedChunkCount[bufferIdx]);
                    }
                }
                if constexpr (STAGED_OUTPUT) {
                    stagedHeadIdx[bufferIdx] = headIdx;
                    stagedFirstChunk[bufferIdx] = firstChunk;
                    stagedChunkCount[bufferIdx] = chunkCount;
                }

                for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                    if (splitByChunk && (chunkOffset & 1U) != subBlockIdx_) {
                        continue;
                    }
                    const uint32_t chunkIdx = firstChunk + chunkOffset;
                    const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
                    const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                    if (startValue < 0 || endValue <= startValue) {
                        continue;
                    }
                    const uint32_t chunkStart = static_cast<uint32_t>(startValue);
                    uint32_t chunkLen = static_cast<uint32_t>(endValue - startValue);
                    if (chunkLen > CHUNK_SIZE) {
                        chunkLen = CHUNK_SIZE;
                    }
                    const uint32_t firstSliceRows = (chunkLen + 1U) / 2U;
                    const uint32_t rowStart = splitByChunk ? 0U :
                        (subBlockIdx_ == 0U ? 0U : firstSliceRows);
                    const uint32_t rowCount = splitByChunk ? chunkLen :
                        (subBlockIdx_ == 0U ? firstSliceRows : chunkLen - firstSliceRows);
                    if (rowCount == 0U) {
                        continue;
                    }
                    const uint32_t chunkBufferBase = bufferBase + chunkOffset * workspaceBufferElems;
                    const uint32_t awOffset = BATCH_CHUNKS ?
                        bufferBase + BATCH_AW_OFFSET + chunkOffset * A_ELEMS :
                        chunkBufferBase + WS_AW_OFFSET;
                    const uint32_t auOffset = BATCH_CHUNKS ?
                        bufferBase + BATCH_AU_OFFSET + chunkOffset * A_ELEMS :
                        chunkBufferBase + WS_AU_OFFSET;
                    PrepareFactor(chunkStart, chunkLen, headIdx, true);
                    PrepareScaledARowSlice(
                        chunkStart, chunkLen, headIdx, rowStart, rowCount, awOffset, auOffset,
                        splitByChunk ? 0U : chunkOffset);
                    if constexpr (BATCH_CHUNKS) {
                        if (!directKey && headOffset >= 2U) {
                            WaitMte3ToMte2();
                        }
                    }
                    if (!splitByChunk && chunkOffset + 1U < chunkCount) {
                        WaitMte3ToMte2();
                        if (!directKey) {
                            WaitMte3ToVector();
                        }
                    }
                    if (!directKey && headOffset < 2U) {
                            const uint32_t keyTileWidth = BATCH_CHUNKS ?
                                AlignUp32(tiling_->keyDim * sizeof(bfloat16_t)) / sizeof(bfloat16_t) :
                                SelectPackedDimTile<PACKED_WIDE>(tiling_->keyDim);
                            const uint32_t keyStoreWidth = BATCH_CHUNKS ? tiling_->keyDim : keyTileWidth;
                            const uint32_t keyTileElems = CHUNK_SIZE * keyStoreWidth;
                            const uint32_t keyTiles = CeilDivU32(tiling_->keyDim, keyTileWidth);
                            for (uint32_t tileIdx = 0; tileIdx < keyTiles; ++tileIdx) {
                                uint32_t validDim = tiling_->keyDim - tileIdx * keyTileWidth;
                                if (validDim > keyTileWidth) {
                                    validDim = keyTileWidth;
                                }
                                const uint32_t bwOffset = BATCH_CHUNKS ?
                                    bufferBase + BATCH_BW_OFFSET + chunkOffset * keyTileElems :
                                    chunkBufferBase + WS_BW_OFFSET + tileIdx * keyTileElems;
                                PackRightMatrixSlice(kGm_, chunkStart, rowStart, rowCount,
                                    groupHead, tiling_->groupHeadNum,
                                    tiling_->keyDim, tileIdx * keyTileWidth, validDim, keyTileWidth,
                                    bwOffset, keyStoreWidth);
                            }
                    }
                }
                CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(GetWReadyFlag<USE_OFFICIAL_MMAD>(bufferIdx));

                if constexpr (SYNCED_DIRECT_VALUE) {
                    WaitMte3ToMte2();
                }
                if constexpr (!DIRECT_VALUE) {
                    for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                        if (splitByChunk && (chunkOffset & 1U) != subBlockIdx_) {
                            continue;
                        }
                        const uint32_t chunkIdx = firstChunk + chunkOffset;
                        const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
                        const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                        if (startValue < 0 || endValue <= startValue) {
                            continue;
                        }
                        const uint32_t chunkStart = static_cast<uint32_t>(startValue);
                        uint32_t chunkLen = static_cast<uint32_t>(endValue - startValue);
                        if (chunkLen > CHUNK_SIZE) {
                            chunkLen = CHUNK_SIZE;
                        }
                        const uint32_t firstSliceRows = (chunkLen + 1U) / 2U;
                        const uint32_t rowStart = splitByChunk ? 0U :
                            (subBlockIdx_ == 0U ? 0U : firstSliceRows);
                        const uint32_t rowCount = splitByChunk ? chunkLen :
                            (subBlockIdx_ == 0U ? firstSliceRows : chunkLen - firstSliceRows);
                        if (rowCount == 0U) {
                            continue;
                        }
                        const uint32_t chunkBufferBase =
                            bufferBase + chunkOffset * workspaceBufferElems;
                        const uint32_t valueTileWidth = BATCH_CHUNKS ?
                            AlignUp32(tiling_->valueDim * sizeof(bfloat16_t)) / sizeof(bfloat16_t) :
                            SelectValueTile<PACKED_WIDE>(tiling_);
                        const uint32_t valueStoreWidth =
                            BATCH_CHUNKS ? tiling_->valueDim : valueTileWidth;
                        const uint32_t valueTileElems = CHUNK_SIZE * valueStoreWidth;
                        const uint32_t valueTiles = CeilDivU32(tiling_->valueDim, valueTileWidth);
                        for (uint32_t tileIdx = 0; tileIdx < valueTiles; ++tileIdx) {
                            uint32_t validDim = tiling_->valueDim - tileIdx * valueTileWidth;
                            if (validDim > valueTileWidth) {
                                validDim = valueTileWidth;
                            }
                            const uint32_t buOffset = BATCH_CHUNKS ?
                                bufferBase + BATCH_BU_OFFSET + chunkOffset * valueTileElems :
                                chunkBufferBase + WS_BU_OFFSET + tileIdx * valueTileElems;
                            PackRightMatrixSlice(vGm_, chunkStart, rowStart, rowCount,
                                headIdx, tiling_->headNum, tiling_->valueDim,
                                tileIdx * valueTileWidth, validDim, valueTileWidth,
                                buOffset, valueStoreWidth);
                        }
                    }
                }
                if constexpr (!DIRECT_VALUE || SYNCED_DIRECT_VALUE) {
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(
                        GetUReadyFlag<USE_OFFICIAL_MMAD>(bufferIdx));
                }
                ++taskSeq;
            }
        }
        if constexpr (STAGED_OUTPUT) {
            const uint32_t firstPending = taskSeq > pipelineDepth ? taskSeq - pipelineDepth : 0U;
            for (uint32_t pendingSeq = firstPending; pendingSeq < taskSeq; ++pendingSeq) {
                const uint32_t bufferIdx = pendingSeq % pipelineDepth;
                CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(GetDoneFlag<USE_OFFICIAL_MMAD>(bufferIdx));
                CopyStagedTask(bufferIdx, stagedHeadIdx[bufferIdx],
                    stagedFirstChunk[bufferIdx], stagedChunkCount[bufferIdx]);
            }
        }
    }

private:
    __aicore__ inline void ProcessBatchChunks()
    {
        uint32_t taskSeq = 0U;
        for (uint32_t task = blockIdx_; task < tiling_->taskCount; task += tiling_->blockDim) {
            const uint32_t tasksPerChunkGroup = tiling_->groupHeadNum * tiling_->headBatchCount;
            const uint32_t chunkGroup = task / tasksPerChunkGroup;
            const uint32_t taskInChunkGroup = task - chunkGroup * tasksPerChunkGroup;
            const uint32_t groupHead = taskInChunkGroup / tiling_->headBatchCount;
            const uint32_t headBatch = taskInChunkGroup - groupHead * tiling_->headBatchCount;
            const uint32_t groupSize = tiling_->headNum / tiling_->groupHeadNum;
            const uint32_t headStart = groupHead * groupSize + headBatch * tiling_->headsPerTask;
            uint32_t headCount = groupSize - headBatch * tiling_->headsPerTask;
            if (headCount > tiling_->headsPerTask) {
                headCount = tiling_->headsPerTask;
            }
            const uint32_t firstChunk = chunkGroup * tiling_->chunksPerTask;
            uint32_t chunkCount = tiling_->numChunks - firstChunk;
            if (chunkCount > tiling_->chunksPerTask) {
                chunkCount = tiling_->chunksPerTask;
            }

            const uint32_t bufferIdx = taskSeq & 1U;
            const uint32_t bufferBase = bufferIdx * BATCH_TASK_BUFFER_ELEMS;
            if (taskSeq >= 2U) {
                CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(GetDoneFlag<false>(bufferIdx));
            }

            const uint32_t rightElems = CHUNK_SIZE * tiling_->keyDim;
            for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                if ((chunkOffset & 1U) != subBlockIdx_) {
                    continue;
                }
                const uint32_t chunkIdx = firstChunk + chunkOffset;
                const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
                const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                if (startValue < 0 || endValue <= startValue) {
                    continue;
                }
                const uint32_t chunkStart = static_cast<uint32_t>(startValue);
                uint32_t chunkLen = static_cast<uint32_t>(endValue - startValue);
                if (chunkLen > CHUNK_SIZE) {
                    chunkLen = CHUNK_SIZE;
                }

                for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                    const uint32_t headIdx = headStart + headOffset;
                    const uint32_t batchIdx = headOffset * chunkCount + chunkOffset;
                    PrepareFactor(chunkStart, chunkLen, headIdx, true);
                    PrepareScaledARowSlice(chunkStart, chunkLen, headIdx, 0U, chunkLen,
                        bufferBase + BATCH_TASK_AW_OFFSET + batchIdx * A_ELEMS,
                        bufferBase + BATCH_TASK_AU_OFFSET + batchIdx * A_ELEMS, 0U);
                    if (headOffset == 0U) {
                        PackRightMatrixSlice(kGm_, chunkStart, 0U, chunkLen,
                            groupHead, tiling_->groupHeadNum, tiling_->keyDim,
                            0U, tiling_->keyDim, tiling_->keyDim,
                            bufferBase + BATCH_TASK_BW_OFFSET + chunkOffset * rightElems,
                            tiling_->keyDim);
                    } else {
                        WaitMte3ToMte2();
                    }
                }
            }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(GetWReadyFlag<false>(bufferIdx));

            for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                if ((chunkOffset & 1U) != subBlockIdx_) {
                    continue;
                }
                const uint32_t chunkIdx = firstChunk + chunkOffset;
                const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
                const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                if (startValue < 0 || endValue <= startValue) {
                    continue;
                }
                const uint32_t chunkStart = static_cast<uint32_t>(startValue);
                uint32_t chunkLen = static_cast<uint32_t>(endValue - startValue);
                if (chunkLen > CHUNK_SIZE) {
                    chunkLen = CHUNK_SIZE;
                }

                for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                    const uint32_t headIdx = headStart + headOffset;
                    const uint32_t batchIdx = headOffset * chunkCount + chunkOffset;
                    PackRightMatrixSlice(vGm_, chunkStart, 0U, chunkLen,
                        headIdx, tiling_->headNum, tiling_->valueDim,
                        0U, tiling_->valueDim, tiling_->valueDim,
                        bufferBase + BATCH_TASK_BU_OFFSET + batchIdx * rightElems,
                        tiling_->valueDim);
                }
            }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(GetUReadyFlag<false>(bufferIdx));
            ++taskSeq;
        }
    }

    __aicore__ inline void ProcessD128GroupPipeline()
    {
        const uint32_t workspaceBufferElems = GetWorkspaceBufferElems<true>(tiling_);
        const uint32_t headBufferElems = workspaceBufferElems * tiling_->chunksPerTask;
        const uint32_t groupBufferElems = headBufferElems * GROUP_HEAD_COUNT;
        uint32_t taskSeq = 0U;
        for (uint32_t task = blockIdx_; task < tiling_->taskCount; task += tiling_->blockDim) {
            const uint32_t tasksPerChunkGroup = tiling_->groupHeadNum * tiling_->headBatchCount;
            const uint32_t chunkGroup = task / tasksPerChunkGroup;
            const uint32_t taskInChunkGroup = task - chunkGroup * tasksPerChunkGroup;
            const uint32_t groupHead = taskInChunkGroup / tiling_->headBatchCount;
            const uint32_t headBatch = taskInChunkGroup - groupHead * tiling_->headBatchCount;
            const uint32_t groupSize = tiling_->headNum / tiling_->groupHeadNum;
            const uint32_t headStart = groupHead * groupSize + headBatch * tiling_->headsPerTask;
            uint32_t headCount = groupSize - headBatch * tiling_->headsPerTask;
            if (headCount > tiling_->headsPerTask) {
                headCount = tiling_->headsPerTask;
            }
            const uint32_t firstChunk = chunkGroup * tiling_->chunksPerTask;
            uint32_t chunkCount = tiling_->numChunks - firstChunk;
            if (chunkCount > tiling_->chunksPerTask) {
                chunkCount = tiling_->chunksPerTask;
            }
            const bool splitByChunk = chunkCount == MAX_CHUNKS_PER_TASK;
            const uint32_t groupSlot = taskSeq % GROUP_PIPELINE_DEPTH;
            const uint32_t groupBufferBase = groupSlot * groupBufferElems;
            if (taskSeq >= GROUP_PIPELINE_DEPTH) {
                CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(GROUP_PIPELINE_DEPTH + groupSlot);
            }

            for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                const uint32_t headIdx = headStart + headOffset;
                const uint32_t headBufferBase = groupBufferBase + headOffset * headBufferElems;
                for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                    if (splitByChunk && chunkOffset != subBlockIdx_) {
                        continue;
                    }
                    const uint32_t chunkIdx = firstChunk + chunkOffset;
                    const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
                    const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                    if (startValue < 0 || endValue <= startValue) {
                        continue;
                    }
                    const uint32_t chunkStart = static_cast<uint32_t>(startValue);
                    uint32_t chunkLen = static_cast<uint32_t>(endValue - startValue);
                    if (chunkLen > CHUNK_SIZE) {
                        chunkLen = CHUNK_SIZE;
                    }
                    const uint32_t firstSliceRows = (chunkLen + 1U) / 2U;
                    const uint32_t rowStart = splitByChunk ? 0U :
                        (subBlockIdx_ == 0U ? 0U : firstSliceRows);
                    const uint32_t rowCount = splitByChunk ? chunkLen :
                        (subBlockIdx_ == 0U ? firstSliceRows : chunkLen - firstSliceRows);
                    if (rowCount == 0U) {
                        continue;
                    }
                    const uint32_t chunkBufferBase =
                        headBufferBase + chunkOffset * workspaceBufferElems;
                    PrepareFactor(chunkStart, chunkLen, headIdx, true);
                    PrepareScaledARowSlice(chunkStart, chunkLen, headIdx, rowStart, rowCount,
                        chunkBufferBase + WS_AW_OFFSET, chunkBufferBase + WS_AU_OFFSET,
                        splitByChunk ? 0U : chunkOffset);
                }
                WaitMte3ToMte2();
            }

            if constexpr (!DIRECT_VALUE) {
                for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                    const uint32_t headIdx = headStart + headOffset;
                    const uint32_t headBufferBase = groupBufferBase + headOffset * headBufferElems;
                    for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                        if (splitByChunk && chunkOffset != subBlockIdx_) {
                            continue;
                        }
                        const uint32_t chunkIdx = firstChunk + chunkOffset;
                        const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
                        const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
                        if (startValue < 0 || endValue <= startValue) {
                            continue;
                        }
                        const uint32_t chunkStart = static_cast<uint32_t>(startValue);
                        uint32_t chunkLen = static_cast<uint32_t>(endValue - startValue);
                        if (chunkLen > CHUNK_SIZE) {
                            chunkLen = CHUNK_SIZE;
                        }
                        const uint32_t firstSliceRows = (chunkLen + 1U) / 2U;
                        const uint32_t rowStart = splitByChunk ? 0U :
                            (subBlockIdx_ == 0U ? 0U : firstSliceRows);
                        const uint32_t rowCount = splitByChunk ? chunkLen :
                            (subBlockIdx_ == 0U ? firstSliceRows : chunkLen - firstSliceRows);
                        if (rowCount == 0U) {
                            continue;
                        }
                        const uint32_t chunkBufferBase =
                            headBufferBase + chunkOffset * workspaceBufferElems;
                        PackRightMatrixSlice(vGm_, chunkStart, rowStart, rowCount,
                            headIdx, tiling_->headNum, tiling_->valueDim, 0U, tiling_->valueDim,
                            DIM_TILE, chunkBufferBase + WS_BU_OFFSET, DIM_TILE);
                    }
                }
            }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(groupSlot);
            ++taskSeq;
        }
    }

    __aicore__ inline void WaitMte2ToVector()
    {
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void CopyStagedTask(uint32_t bufferIdx, uint32_t headIdx,
        uint32_t firstChunk, uint32_t chunkCount)
    {
        const bool splitByChunk = chunkCount == MAX_CHUNKS_PER_TASK;
        for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
            if (splitByChunk && chunkOffset != subBlockIdx_) {
                continue;
            }
            const uint32_t chunkIdx = firstChunk + chunkOffset;
            const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
            const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
            if (startValue < 0 || endValue <= startValue) {
                continue;
            }
            const uint32_t chunkStart = static_cast<uint32_t>(startValue);
            uint32_t chunkLen = static_cast<uint32_t>(endValue - startValue);
            if (chunkLen > CHUNK_SIZE) {
                chunkLen = CHUNK_SIZE;
            }
            const uint32_t firstSliceRows = (chunkLen + 1U) / 2U;
            const uint32_t rowStart = splitByChunk ? 0U :
                (subBlockIdx_ == 0U ? 0U : firstSliceRows);
            const uint32_t rowCount = splitByChunk ? chunkLen :
                (subBlockIdx_ == 0U ? firstSliceRows : chunkLen - firstSliceRows);
            if (rowCount == 0U) {
                continue;
            }
            CopyStagedOutputSlice(GetOfficialMmadStagedOutputOffset(
                tiling_, bufferIdx, chunkOffset, false), wGm_, chunkStart,
                headIdx, rowStart, rowCount, tiling_->keyDim);
            CopyStagedOutputSlice(GetOfficialMmadStagedOutputOffset(
                tiling_, bufferIdx, chunkOffset, true), uGm_, chunkStart,
                headIdx, rowStart, rowCount, tiling_->valueDim);
        }
    }

    __aicore__ inline void CopyStagedOutputSlice(uint32_t stagedOffset,
        GlobalTensor<bfloat16_t>& output, uint32_t chunkStart, uint32_t headIdx,
        uint32_t rowStart, uint32_t rowCount, uint32_t dim)
    {
        LocalTensor<bfloat16_t> raw =
            scratch_.Get<bfloat16_t>()[RIGHT_BF16_BYTE_OFFSET / sizeof(bfloat16_t)];
        const uint32_t stagedDim = AlignUp16(dim);
        const uint32_t blockBytes = dim * sizeof(bfloat16_t);
        const uint32_t alignedBytes = stagedDim * sizeof(bfloat16_t);
        const uint32_t rightPadding = (alignedBytes - blockBytes) / sizeof(bfloat16_t);
        const DataCopyExtParams loadParams = {
            static_cast<uint16_t>(rowCount), blockBytes,
            static_cast<uint32_t>((stagedDim - dim) * sizeof(bfloat16_t)), 0, 0,
        };
        const DataCopyPadExtParams<bfloat16_t> padParams = {
            rightPadding != 0U, 0, static_cast<uint8_t>(rightPadding),
            static_cast<bfloat16_t>(0),
        };
        DataCopyPad(raw, workspaceGm_[stagedOffset + rowStart * stagedDim],
            loadParams, padParams);
        WaitMte2ToMte3();
        const DataCopyExtParams storeParams = {
            static_cast<uint16_t>(rowCount), blockBytes, 0, 0, 0,
        };
        const uint64_t outputOffset =
            (static_cast<uint64_t>(headIdx) * tiling_->totalTokens + chunkStart + rowStart) * dim;
        DataCopyPad(output[outputOffset], raw, storeParams);
        WaitMte3ToMte2();
    }

    __aicore__ inline void WaitVectorToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void WaitMte2ToMte3()
    {
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void WaitMte3ToMte2()
    {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void WaitMte3ToVector()
    {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void PrepareScaledARowSlice(uint32_t chunkStart, uint32_t chunkLen,
        uint32_t headIdx, uint32_t rowStart, uint32_t rowCount, uint32_t awOffset,
        uint32_t auOffset, uint32_t localSlot)
    {
        LocalTensor<bfloat16_t> awLocal = scratch_.Get<bfloat16_t>()[RAW_BF16_BYTE_OFFSET / sizeof(bfloat16_t)];
        LocalTensor<bfloat16_t> aLocal = scratch_.Get<bfloat16_t>()[A_LOCAL_BYTE_OFFSET / sizeof(bfloat16_t)];
        LocalTensor<float> auFloat =
            scratch_.Get<float>()[MATRIX_FLOAT_BYTE_OFFSET / sizeof(float) + localSlot * AIV_SLICE_ELEMS];
        LocalTensor<float> awFloat = scratch_.Get<float>()[AW_FLOAT_BYTE_OFFSET / sizeof(float)];
        const uint32_t blockBytes = chunkLen * sizeof(bfloat16_t);
        const uint32_t alignedBytes = AlignUp32(blockBytes);
        const uint32_t rightPadding = (alignedBytes - blockBytes) / sizeof(bfloat16_t);
        const DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rowCount), blockBytes,
            static_cast<uint32_t>((CHUNK_SIZE - chunkLen) * sizeof(bfloat16_t)),
            static_cast<uint32_t>((CHUNK_SIZE * sizeof(bfloat16_t) - alignedBytes) / 32U), 0,
        };
        const DataCopyPadExtParams<bfloat16_t> padParams = {
            rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), static_cast<bfloat16_t>(0),
        };
        const uint64_t aOffset =
            (static_cast<uint64_t>(headIdx) * tiling_->totalTokens + chunkStart + rowStart) * CHUNK_SIZE;
        DataCopyPad(aLocal, aGm_[aOffset], copyParams, padParams);
        WaitMte2ToVector();

        const uint32_t matrixCount = rowCount * CHUNK_SIZE;
        Cast(auFloat, aLocal, RoundMode::CAST_NONE, matrixCount);
        PipeBarrier<PIPE_V>();
        const BinaryRepeatParams scaleParams = {
            1, 1, 1, FLOAT_ROW_BLOCKS, FLOAT_ROW_BLOCKS, 0,
        };
        Mul<float, true>(auFloat, auFloat, betaFloat_, static_cast<uint64_t>(chunkLen),
            static_cast<uint8_t>(rowCount), scaleParams);
        PipeBarrier<PIPE_V>();
        Mul<float, true>(awFloat, auFloat, gateFloat_, static_cast<uint64_t>(chunkLen),
            static_cast<uint8_t>(rowCount), scaleParams);
        PipeBarrier<PIPE_V>();
        Cast(awLocal, awFloat, RoundMode::CAST_RINT, matrixCount);
        WaitVectorToMte3();
        DataCopy(workspaceGm_[awOffset + rowStart * CHUNK_SIZE], awLocal, matrixCount);
        LocalTensor<bfloat16_t> auLocal = scratch_.Get<bfloat16_t>()[A_LOCAL_BYTE_OFFSET / sizeof(bfloat16_t)];
        Cast(auLocal, auFloat, RoundMode::CAST_RINT, matrixCount);
        WaitVectorToMte3();
        DataCopy(workspaceGm_[auOffset + rowStart * CHUNK_SIZE], auLocal, matrixCount);
    }

    __aicore__ inline void PrepareFactor(uint32_t chunkStart, uint32_t chunkLen, uint32_t headIdx, bool withGate)
    {
        LocalTensor<uint8_t> small = scratch_.Get<uint8_t>()[SMALL_BYTE_OFFSET];
        LocalTensor<bfloat16_t> betaVec = small[768].ReinterpretCast<bfloat16_t>();
        betaFloat_ = small[1024].ReinterpretCast<float>();
        gateFloat_ = small[1280].ReinterpretCast<float>();

        const uint32_t betaBytes = chunkLen * sizeof(bfloat16_t);
        const uint32_t betaAlignedBytes = AlignUp32(betaBytes);
        const uint32_t betaPadding = (betaAlignedBytes - betaBytes) / sizeof(bfloat16_t);
        DataCopyPadExtParams<bfloat16_t> betaPad = {
            betaPadding != 0, 0, static_cast<uint8_t>(betaPadding), static_cast<bfloat16_t>(0),
        };
        DataCopyExtParams betaCopy = {1, betaBytes, 0, 0, 0};
        const uint64_t factorOffset = static_cast<uint64_t>(headIdx) * tiling_->totalTokens + chunkStart;
        DataCopyPad(betaVec, betaGm_[factorOffset], betaCopy, betaPad);

        if (withGate) {
            const uint32_t gBytes = chunkLen * sizeof(float);
            const uint32_t gAlignedBytes = AlignUp32(gBytes);
            const uint32_t gPadding = (gAlignedBytes - gBytes) / sizeof(float);
            DataCopyPadExtParams<float> gPad = {
                gPadding != 0, 0, static_cast<uint8_t>(gPadding), 0.0f,
            };
            DataCopyExtParams gCopy = {1, gBytes, 0, 0, 0};
            DataCopyPad(gateFloat_, gGm_[factorOffset], gCopy, gPad);
        }
        WaitMte2ToVector();

        Cast(betaFloat_, betaVec, RoundMode::CAST_NONE, chunkLen);
        if (withGate) {
            Exp(gateFloat_, gateFloat_, chunkLen);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PackRightMatrixSlice(GlobalTensor<bfloat16_t>& input, uint32_t chunkStart,
        uint32_t rowStart, uint32_t rowCount, uint32_t inputHead, uint32_t inputHeadNum, uint32_t inputDim,
        uint32_t dimOffset, uint32_t validDim, uint32_t tileWidth, uint32_t wsOffset,
        uint32_t storeWidth)
    {
        LocalTensor<bfloat16_t> raw =
            scratch_.Get<bfloat16_t>()[RIGHT_BF16_BYTE_OFFSET / sizeof(bfloat16_t)];
        const uint32_t blockBytes = validDim * sizeof(bfloat16_t);
        const uint32_t alignedBytes = AlignUp32(blockBytes);
        const uint32_t rightPadding = (alignedBytes - blockBytes) / sizeof(bfloat16_t);
        const DataCopyExtParams copyParams = {
            static_cast<uint16_t>(rowCount), blockBytes,
            static_cast<uint32_t>((inputHeadNum * inputDim - validDim) * sizeof(bfloat16_t)),
            static_cast<uint32_t>((tileWidth * sizeof(bfloat16_t) - alignedBytes) / 32U), 0,
        };
        const DataCopyPadExtParams<bfloat16_t> padParams = {
            rightPadding != 0, 0, static_cast<uint8_t>(rightPadding), static_cast<bfloat16_t>(0),
        };
        const uint64_t inputOffset =
            (static_cast<uint64_t>(chunkStart + rowStart) * inputHeadNum + inputHead) * inputDim + dimOffset;
        DataCopyPad(raw, input[inputOffset], copyParams, padParams);
        WaitMte2ToMte3();
        if (storeWidth == tileWidth) {
            const uint32_t matrixCount = rowCount * tileWidth;
            DataCopy(workspaceGm_[wsOffset + rowStart * tileWidth], raw, matrixCount);
        } else {
            const DataCopyExtParams storeParams = {
                static_cast<uint16_t>(rowCount),
                static_cast<uint32_t>(validDim * sizeof(bfloat16_t)),
                0, 0, 0,
            };
            DataCopyPad(workspaceGm_[wsOffset + rowStart * storeWidth], raw, storeParams);
        }
        WaitMte3ToMte2();
    }

private:
    const RecomputeWuTilingData* tiling_ = nullptr;
    uint32_t subBlockIdx_ = 0;
    uint32_t blockIdx_ = 0;

    GlobalTensor<bfloat16_t> aGm_;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> vGm_;
    GlobalTensor<bfloat16_t> betaGm_;
    GlobalTensor<float> gGm_;
    GlobalTensor<int32_t> offsetsGm_;
    GlobalTensor<bfloat16_t> wGm_;
    GlobalTensor<bfloat16_t> uGm_;
    GlobalTensor<bfloat16_t> workspaceGm_;
    LocalTensor<float> betaFloat_;
    LocalTensor<float> gateFloat_;
    TBuf<TPosition::VECCALC> scratch_;
};

template <bool DIRECT_VALUE, bool DIRECT_D128_HEAD_BATCH = false,
    bool D128_GROUP_PIPELINE = false, bool SYNCED_DIRECT_VALUE = false,
    bool STAGED_OUTPUT = false>
class RecomputeWuCubeMmad {
public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR v, GM_ADDR globalChunkOffsets, GM_ADDR w, GM_ADDR u,
        GM_ADDR userWorkspace, const RecomputeWuTilingData* tilingData, TPipe* pipe)
    {
        tiling_ = tilingData;
        blockIdx_ = GetBlockIdx();
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
        offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(globalChunkOffsets));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w));
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(u));
        const uint32_t workspacePerCoreElems = tiling_->workspacePerCore / sizeof(bfloat16_t);
        const uint64_t coreWsOffset = static_cast<uint64_t>(blockIdx_) * workspacePerCoreElems;
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWorkspace) + coreWsOffset,
            workspacePerCoreElems);

        pipe->InitBuffer(aL1Buf_, 2U * A_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(bL1Buf_, 2U * TILE_MATRIX_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(aL0Buf_, 2U * A_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(bL0Buf_, 2U * TILE_MATRIX_ELEMS * sizeof(bfloat16_t));
        pipe->InitBuffer(cL0Buf_, 2U * TILE_MATRIX_ELEMS * sizeof(float));

        SetLoadDataPaddingValue<uint64_t>(0);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        InitSyncFlags();
    }

    __aicore__ inline void Process()
    {
        if constexpr (D128_GROUP_PIPELINE) {
            ProcessD128GroupPipeline();
            WaitSyncFlags();
            return;
        }
        const bool directKey = DIRECT_D128_HEAD_BATCH || UseDirectKeyPath(tiling_);
        const bool packedD128HeadBatch = UsePackedD128HeadBatchPath(tiling_);
        const uint32_t pipelineDepth =
            directKey || packedD128HeadBatch ? PIPELINE_DEPTH : 2U;
        const uint32_t workspaceBufferElems =
            GetWorkspaceBufferElems<DIRECT_D128_HEAD_BATCH>(tiling_);
        uint32_t taskSeq = 0;
        for (uint32_t task = blockIdx_; task < tiling_->taskCount; task += tiling_->blockDim) {
            const uint32_t tasksPerChunkGroup = tiling_->groupHeadNum * tiling_->headBatchCount;
            const uint32_t chunkGroup = task / tasksPerChunkGroup;
            const uint32_t taskInChunkGroup = task - chunkGroup * tasksPerChunkGroup;
            const uint32_t groupHead = taskInChunkGroup / tiling_->headBatchCount;
            const uint32_t headBatch = taskInChunkGroup - groupHead * tiling_->headBatchCount;
            const uint32_t groupSize = tiling_->headNum / tiling_->groupHeadNum;
            const uint32_t headStart = groupHead * groupSize + headBatch * tiling_->headsPerTask;
            uint32_t headCount = groupSize - headBatch * tiling_->headsPerTask;
            if (headCount > tiling_->headsPerTask) {
                headCount = tiling_->headsPerTask;
            }
            const uint32_t firstChunk = chunkGroup * tiling_->chunksPerTask;
            uint32_t chunkCount = tiling_->numChunks - firstChunk;
            if (chunkCount > tiling_->chunksPerTask) {
                chunkCount = tiling_->chunksPerTask;
            }
            const uint32_t valueTileWidth =
                SelectValueTile<false, DIRECT_D128_HEAD_BATCH>(tiling_);
            const uint32_t keyTileWidth = SelectDimTile(tiling_->keyDim);
            const uint32_t valueTileElems = CHUNK_SIZE * valueTileWidth;
            const uint32_t keyTileElems = CHUNK_SIZE * keyTileWidth;
            const uint32_t valueTiles = CeilDivU32(tiling_->valueDim, valueTileWidth);
            const uint32_t keyTiles = CeilDivU32(tiling_->keyDim, keyTileWidth);
            if ((packedD128HeadBatch || DIRECT_D128_HEAD_BATCH) && headCount > 1U) {
                const uint32_t batchBufferElems = workspaceBufferElems * tiling_->chunksPerTask;
                const uint32_t commonBufferIdx = taskSeq % pipelineDepth;
                const uint32_t commonBufferBase = commonBufferIdx * batchBufferElems;
                uint32_t chunkStarts[MAX_CHUNKS_PER_TASK] = {0U, 0U};
                uint32_t chunkLens[MAX_CHUNKS_PER_TASK] = {0U, 0U};
                bool chunkValid[MAX_CHUNKS_PER_TASK] = {false, false};
                for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                    chunkValid[chunkOffset] = GetChunkInfo(
                        firstChunk + chunkOffset, chunkStarts[chunkOffset], chunkLens[chunkOffset]);
                }
                if constexpr (DIRECT_D128_HEAD_BATCH) {
                    if (UseD128FourSlotPath()) {
                        // D128 needs only half of the generic B/C slot size. Keep both
                        // chunks' shared K matrices in slots 0/1 and use slots 2/3 for
                        // V, so each head can finish W then U without evicting K.
                        for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                            const uint32_t headSeq = taskSeq + headOffset;
                            const uint32_t headIdx = headStart + headOffset;
                            const uint32_t bufferIdx = headSeq % pipelineDepth;
                            const uint32_t bufferBase = bufferIdx * batchBufferElems;
                            WaitVectorReady(GetWReadyFlag<true>(bufferIdx));
                            for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                                if (!chunkValid[chunkOffset]) {
                                    continue;
                                }
                                const uint32_t chunkBufferBase =
                                    bufferBase + chunkOffset * workspaceBufferElems;
                                const uint64_t keyOffset =
                                    (static_cast<uint64_t>(chunkStarts[chunkOffset]) *
                                        tiling_->groupHeadNum + groupHead) * tiling_->keyDim;
                                RunCubeD128Slot(chunkOffset, headOffset == 0U,
                                    chunkBufferBase + WS_AW_OFFSET, kGm_, keyOffset,
                                    wGm_, chunkStarts[chunkOffset], headIdx, tiling_->keyDim,
                                    tiling_->groupHeadNum * tiling_->keyDim,
                                    chunkLens[chunkOffset]);
                            }
                            for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                                if (!chunkValid[chunkOffset]) {
                                    continue;
                                }
                                const uint32_t chunkBufferBase =
                                    bufferBase + chunkOffset * workspaceBufferElems;
                                const uint64_t valueOffset =
                                    (static_cast<uint64_t>(chunkStarts[chunkOffset]) *
                                        tiling_->headNum + headIdx) * tiling_->valueDim;
                                RunCubeD128Slot(2U + chunkOffset, true,
                                    chunkBufferBase + WS_AU_OFFSET, vGm_, valueOffset,
                                    uGm_, chunkStarts[chunkOffset], headIdx, tiling_->valueDim,
                                    tiling_->headNum * tiling_->valueDim,
                                    chunkLens[chunkOffset]);
                            }
                            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(GetDoneFlag<true>(bufferIdx));
                        }
                        taskSeq += headCount;
                        continue;
                    }
                }
                // Keep each chunk's shared K tile in a fixed L0B slot while all
                // heads replace only L0A and L0C for their W output.
                for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                    const uint32_t headSeq = taskSeq + headOffset;
                    const uint32_t headIdx = headStart + headOffset;
                    const uint32_t bufferIdx = headSeq % pipelineDepth;
                    const uint32_t bufferBase = bufferIdx * batchBufferElems;
                    WaitVectorReady(GetWReadyFlag<true>(bufferIdx));
                    for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                        if (!chunkValid[chunkOffset]) {
                            continue;
                        }
                        const uint32_t chunkBufferBase = bufferBase + chunkOffset * workspaceBufferElems;
                        if constexpr (DIRECT_D128_HEAD_BATCH) {
                            const uint64_t keyOffset =
                                (static_cast<uint64_t>(chunkStarts[chunkOffset]) *
                                    tiling_->groupHeadNum + groupHead) * tiling_->keyDim;
                            RunCubeCachedL0B(chunkOffset, headOffset == 0U,
                                chunkBufferBase + WS_AW_OFFSET,
                                kGm_, keyOffset,
                                wGm_, chunkStarts[chunkOffset], headIdx, tiling_->keyDim, 0U,
                                tiling_->groupHeadNum * tiling_->keyDim,
                                chunkLens[chunkOffset], tiling_->keyDim);
                        } else {
                            const uint32_t commonChunkBase =
                                commonBufferBase + chunkOffset * workspaceBufferElems;
                            RunCubeCachedL0B(chunkOffset, headOffset == 0U,
                                chunkBufferBase + WS_AW_OFFSET,
                                workspaceGm_, commonChunkBase + WS_BW_OFFSET,
                                wGm_, chunkStarts[chunkOffset], headIdx, tiling_->keyDim, 0U,
                                keyTileWidth, chunkLens[chunkOffset], tiling_->keyDim);
                        }
                    }
                }

                for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                    const uint32_t headSeq = taskSeq + headOffset;
                    const uint32_t headIdx = headStart + headOffset;
                    const uint32_t bufferIdx = headSeq % pipelineDepth;
                    const uint32_t bufferBase = bufferIdx * batchBufferElems;
                    if constexpr (!DIRECT_VALUE || SYNCED_DIRECT_VALUE) {
                        WaitVectorReady(GetUReadyFlag<true>(bufferIdx));
                    }
                    for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                        if (!chunkValid[chunkOffset]) {
                            continue;
                        }
                        const uint32_t chunkBufferBase = bufferBase + chunkOffset * workspaceBufferElems;
                        for (uint32_t tileIdx = 0; tileIdx < valueTiles; ++tileIdx) {
                            uint32_t validDim = tiling_->valueDim - tileIdx * valueTileWidth;
                            if (validDim > valueTileWidth) {
                                validDim = valueTileWidth;
                            }
                            if constexpr (DIRECT_VALUE) {
                                const uint64_t valueOffset =
                                    (static_cast<uint64_t>(chunkStarts[chunkOffset]) * tiling_->headNum +
                                        headIdx) * tiling_->valueDim +
                                    tileIdx * valueTileWidth;
                                RunCube(chunkBufferBase + WS_AU_OFFSET,
                                    vGm_, valueOffset,
                                    uGm_, chunkStarts[chunkOffset], headIdx, tiling_->valueDim,
                                    tileIdx * valueTileWidth, tiling_->headNum * tiling_->valueDim,
                                    chunkLens[chunkOffset], validDim);
                            } else {
                                RunCube(chunkBufferBase + WS_AU_OFFSET,
                                    workspaceGm_,
                                    chunkBufferBase + WS_BU_OFFSET + tileIdx * valueTileElems,
                                    uGm_, chunkStarts[chunkOffset], headIdx, tiling_->valueDim,
                                    tileIdx * valueTileWidth, valueTileWidth,
                                    chunkLens[chunkOffset], validDim);
                            }
                        }
                    }
                    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(GetDoneFlag<true>(bufferIdx));
                }
                taskSeq += headCount;
                continue;
            }
            for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                const uint32_t headIdx = headStart + headOffset;
                const uint32_t bufferIdx = taskSeq % pipelineDepth;
                const uint32_t batchBufferElems = workspaceBufferElems * tiling_->chunksPerTask;
                const uint32_t bufferBase = bufferIdx * batchBufferElems;
                WaitVectorReady(GetWReadyFlag<true>(bufferIdx));
                for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                    uint32_t chunkStart = 0;
                    uint32_t chunkLen = 0;
                    if (!GetChunkInfo(firstChunk + chunkOffset, chunkStart, chunkLen)) {
                        continue;
                    }
                    const uint32_t chunkBufferBase = bufferBase + chunkOffset * workspaceBufferElems;
                    for (uint32_t tileIdx = 0; tileIdx < keyTiles; ++tileIdx) {
                        uint32_t validDim = tiling_->keyDim - tileIdx * keyTileWidth;
                        if (validDim > keyTileWidth) {
                            validDim = keyTileWidth;
                        }
                        if (directKey) {
                            const uint64_t keyOffset =
                                (static_cast<uint64_t>(chunkStart) * tiling_->groupHeadNum + groupHead) *
                                    tiling_->keyDim +
                                tileIdx * keyTileWidth;
                            RunCube(chunkBufferBase + WS_AW_OFFSET,
                                kGm_, keyOffset,
                                wGm_, chunkStart, headIdx, tiling_->keyDim,
                                tileIdx * keyTileWidth, tiling_->groupHeadNum * tiling_->keyDim,
                                chunkLen, validDim, GetOfficialMmadStagedOutputOffset(
                                    tiling_, bufferIdx, chunkOffset, false));
                        } else {
                            RunCube(chunkBufferBase + WS_AW_OFFSET,
                                workspaceGm_, chunkBufferBase + WS_BW_OFFSET + tileIdx * keyTileElems,
                                wGm_, chunkStart, headIdx, tiling_->keyDim,
                                tileIdx * keyTileWidth, keyTileWidth, chunkLen, validDim,
                                GetOfficialMmadStagedOutputOffset(
                                    tiling_, bufferIdx, chunkOffset, false));
                        }
                    }
                }

                WaitVectorReady(GetUReadyFlag<true>(bufferIdx));
                for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                    uint32_t chunkStart = 0;
                    uint32_t chunkLen = 0;
                    if (!GetChunkInfo(firstChunk + chunkOffset, chunkStart, chunkLen)) {
                        continue;
                    }
                    const uint32_t chunkBufferBase = bufferBase + chunkOffset * workspaceBufferElems;
                    for (uint32_t tileIdx = 0; tileIdx < valueTiles; ++tileIdx) {
                        uint32_t validDim = tiling_->valueDim - tileIdx * valueTileWidth;
                        if (validDim > valueTileWidth) {
                            validDim = valueTileWidth;
                        }
                        if constexpr (DIRECT_VALUE) {
                            const uint64_t valueOffset =
                                (static_cast<uint64_t>(chunkStart) * tiling_->headNum + headIdx) *
                                    tiling_->valueDim +
                                tileIdx * valueTileWidth;
                            RunCube(chunkBufferBase + WS_AU_OFFSET,
                                vGm_, valueOffset, uGm_, chunkStart, headIdx, tiling_->valueDim,
                                tileIdx * valueTileWidth, tiling_->headNum * tiling_->valueDim,
                                chunkLen, validDim, GetOfficialMmadStagedOutputOffset(
                                    tiling_, bufferIdx, chunkOffset, true));
                        } else {
                            RunCube(chunkBufferBase + WS_AU_OFFSET,
                                workspaceGm_,
                                chunkBufferBase + WS_BU_OFFSET + tileIdx * valueTileElems,
                                uGm_, chunkStart, headIdx, tiling_->valueDim,
                                tileIdx * valueTileWidth, valueTileWidth, chunkLen, validDim,
                                GetOfficialMmadStagedOutputOffset(
                                    tiling_, bufferIdx, chunkOffset, true));
                        }
                    }
                }
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(GetDoneFlag<true>(bufferIdx));
                ++taskSeq;
            }
        }
        WaitSyncFlags();
    }

private:
    __aicore__ inline void ProcessD128GroupPipeline()
    {
        const uint32_t workspaceBufferElems = GetWorkspaceBufferElems<true>(tiling_);
        const uint32_t headBufferElems = workspaceBufferElems * tiling_->chunksPerTask;
        const uint32_t groupBufferElems = headBufferElems * GROUP_HEAD_COUNT;
        uint32_t taskSeq = 0U;
        for (uint32_t task = blockIdx_; task < tiling_->taskCount; task += tiling_->blockDim) {
            const uint32_t tasksPerChunkGroup = tiling_->groupHeadNum * tiling_->headBatchCount;
            const uint32_t chunkGroup = task / tasksPerChunkGroup;
            const uint32_t taskInChunkGroup = task - chunkGroup * tasksPerChunkGroup;
            const uint32_t groupHead = taskInChunkGroup / tiling_->headBatchCount;
            const uint32_t headBatch = taskInChunkGroup - groupHead * tiling_->headBatchCount;
            const uint32_t groupSize = tiling_->headNum / tiling_->groupHeadNum;
            const uint32_t headStart = groupHead * groupSize + headBatch * tiling_->headsPerTask;
            uint32_t headCount = groupSize - headBatch * tiling_->headsPerTask;
            if (headCount > tiling_->headsPerTask) {
                headCount = tiling_->headsPerTask;
            }
            const uint32_t firstChunk = chunkGroup * tiling_->chunksPerTask;
            uint32_t chunkCount = tiling_->numChunks - firstChunk;
            if (chunkCount > tiling_->chunksPerTask) {
                chunkCount = tiling_->chunksPerTask;
            }
            const uint32_t groupSlot = taskSeq % GROUP_PIPELINE_DEPTH;
            const uint32_t groupBufferBase = groupSlot * groupBufferElems;
            uint32_t chunkStarts[MAX_CHUNKS_PER_TASK] = {0U, 0U};
            uint32_t chunkLens[MAX_CHUNKS_PER_TASK] = {0U, 0U};
            bool chunkValid[MAX_CHUNKS_PER_TASK] = {false, false};
            for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                chunkValid[chunkOffset] = GetChunkInfo(
                    firstChunk + chunkOffset, chunkStarts[chunkOffset], chunkLens[chunkOffset]);
            }

            WaitVectorReady(groupSlot);
            for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                const uint32_t headIdx = headStart + headOffset;
                const uint32_t headBufferBase = groupBufferBase + headOffset * headBufferElems;
                for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                    if (!chunkValid[chunkOffset]) {
                        continue;
                    }
                    const uint32_t chunkBufferBase =
                        headBufferBase + chunkOffset * workspaceBufferElems;
                    const uint64_t keyOffset =
                        (static_cast<uint64_t>(chunkStarts[chunkOffset]) *
                            tiling_->groupHeadNum + groupHead) * tiling_->keyDim;
                    RunCubeCachedL0B(chunkOffset, headOffset == 0U,
                        chunkBufferBase + WS_AW_OFFSET, kGm_, keyOffset,
                        wGm_, chunkStarts[chunkOffset], headIdx, tiling_->keyDim, 0U,
                        tiling_->groupHeadNum * tiling_->keyDim,
                        chunkLens[chunkOffset], tiling_->keyDim);
                }
            }

            for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                const uint32_t headIdx = headStart + headOffset;
                const uint32_t headBufferBase = groupBufferBase + headOffset * headBufferElems;
                for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                    if (!chunkValid[chunkOffset]) {
                        continue;
                    }
                    const uint32_t chunkBufferBase =
                        headBufferBase + chunkOffset * workspaceBufferElems;
                    if constexpr (DIRECT_VALUE) {
                        const uint64_t valueOffset =
                            (static_cast<uint64_t>(chunkStarts[chunkOffset]) * tiling_->headNum +
                                headIdx) * tiling_->valueDim;
                        RunCube(chunkBufferBase + WS_AU_OFFSET, vGm_, valueOffset,
                            uGm_, chunkStarts[chunkOffset], headIdx, tiling_->valueDim,
                            0U, tiling_->headNum * tiling_->valueDim,
                            chunkLens[chunkOffset], tiling_->valueDim);
                    } else {
                        RunCube(chunkBufferBase + WS_AU_OFFSET, workspaceGm_,
                            chunkBufferBase + WS_BU_OFFSET, uGm_, chunkStarts[chunkOffset],
                            headIdx, tiling_->valueDim, 0U, DIM_TILE,
                            chunkLens[chunkOffset], tiling_->valueDim);
                    }
                }
            }
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(GROUP_PIPELINE_DEPTH + groupSlot);
            ++taskSeq;
        }
    }

    __aicore__ inline void WaitVectorReady(uint32_t flagId)
    {
        CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE2>(flagId);
    }

    __aicore__ inline bool UseD128FourSlotPath() const
    {
        if constexpr (!DIRECT_D128_HEAD_BATCH || D128_GROUP_PIPELINE) {
            return false;
        }
        const uint32_t groupSize = tiling_->headNum / tiling_->groupHeadNum;
        if (groupSize != GROUP_HEAD_COUNT) {
            return false;
        }
        const uint32_t chunkGroupCount = CeilDivU32(tiling_->numChunks, tiling_->chunksPerTask);
        return chunkGroupCount > tiling_->blockDim;
    }

    __aicore__ inline event_t SelectL1Event(uint32_t slot) const
    {
        if (slot == 0U) {
            return EVENT_ID0;
        }
        if (slot == 1U) {
            return EVENT_ID1;
        }
        return slot == 2U ? EVENT_ID2 : EVENT_ID3;
    }

    __aicore__ inline event_t SelectL0Event(uint32_t slot) const
    {
        if (slot == 0U) {
            return EVENT_ID6;
        }
        if (slot == 1U) {
            return EVENT_ID7;
        }
        return slot == 2U ? EVENT_ID4 : EVENT_ID5;
    }

    __aicore__ inline void InitSyncFlags()
    {
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID6);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID7);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        if (UseD128FourSlotPath()) {
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID5);
            SetFlag<HardEvent::FIX_M>(EVENT_ID2);
            SetFlag<HardEvent::FIX_M>(EVENT_ID3);
        }
    }

    __aicore__ inline void WaitSyncFlags()
    {
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID6);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID7);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        if (UseD128FourSlotPath()) {
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID5);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID2);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID3);
        }
    }

    __aicore__ inline bool GetChunkInfo(uint32_t chunkIdx, uint32_t& chunkStart, uint32_t& chunkLen)
    {
        const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
        const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
        if (startValue < 0 || endValue <= startValue) {
            return false;
        }
        chunkStart = static_cast<uint32_t>(startValue);
        chunkLen = static_cast<uint32_t>(endValue - startValue);
        if (chunkLen > CHUNK_SIZE) {
            chunkLen = CHUNK_SIZE;
        }
        return true;
    }

    __aicore__ inline void CopyNdToL1(LocalTensor<bfloat16_t>& dst,
        const GlobalTensor<bfloat16_t>& src, uint32_t rows, uint32_t cols, uint32_t srcRowStride)
    {
        Nd2NzParams params;
        params.ndNum = 1;
        params.nValue = rows;
        params.dValue = cols;
        params.srcNdMatrixStride = 0;
        params.srcDValue = srcRowStride;
        params.dstNzC0Stride = (rows + 15U) & ~15U;
        params.dstNzNStride = 1;
        params.dstNzMatrixStride = 0;
        DataCopy(dst, src, params);
    }

    __aicore__ inline void LoadA(LocalTensor<bfloat16_t>& dst,
        const LocalTensor<bfloat16_t>& src, uint32_t mRound, uint32_t kRound)
    {
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = kRound / 16U;
        params.srcStride = mRound / 16U;
        params.dstGap = 0;
        params.ifTranspose = false;
        for (uint32_t blockM = 0; blockM < mRound / 16U; ++blockM) {
            LoadData(dst[blockM * 16U * kRound], src[blockM * 16U * 16U], params);
        }
    }

    __aicore__ inline void LoadB(LocalTensor<bfloat16_t>& dst,
        const LocalTensor<bfloat16_t>& src, uint32_t kRound, uint32_t nRound)
    {
        LoadData2DParams params;
        params.startIndex = 0;
        params.repeatTimes = nRound / 16U;
        params.srcStride = kRound / 16U;
        params.dstGap = 0;
        params.ifTranspose = true;
        for (uint32_t blockK = 0; blockK < kRound / 16U; ++blockK) {
            LoadData(dst[blockK * 16U * nRound], src[blockK * 16U * 16U], params);
        }
    }

    __aicore__ inline void RunCube(
        uint32_t aOffset, GlobalTensor<bfloat16_t>& bInput, uint64_t bOffset,
        GlobalTensor<bfloat16_t>& output,
        uint32_t chunkStart, uint32_t headIdx, uint32_t outputDim, uint32_t dimOffset,
        uint32_t inputRowStride, uint32_t chunkLen, uint32_t validDim,
        uint32_t stagedOffset = 0U)
    {
        const uint32_t mRound = (chunkLen + 15U) & ~15U;
        const uint32_t kRound = mRound;
        const uint32_t nRound = (validDim + 15U) & ~15U;
        const uint32_t slot = cubeSeq_ & 1U;
        const event_t l1Event = SelectL1Event(slot);
        const event_t l0Event = SelectL0Event(slot);

        LocalTensor<bfloat16_t> aL1 = aL1Buf_.Get<bfloat16_t>()[slot * A_ELEMS];
        LocalTensor<bfloat16_t> bL1 = bL1Buf_.Get<bfloat16_t>()[slot * TILE_MATRIX_ELEMS];
        LocalTensor<bfloat16_t> aL0 = aL0Buf_.Get<bfloat16_t>()[slot * A_ELEMS];
        LocalTensor<bfloat16_t> bL0 = bL0Buf_.Get<bfloat16_t>()[slot * TILE_MATRIX_ELEMS];
        LocalTensor<float> cL0 = cL0Buf_.Get<float>()[slot * TILE_MATRIX_ELEMS];

        WaitFlag<HardEvent::MTE1_MTE2>(l1Event);
        CopyNdToL1(aL1, workspaceGm_[aOffset], chunkLen, chunkLen, CHUNK_SIZE);
        CopyNdToL1(bL1, bInput[bOffset], chunkLen, validDim, inputRowStride);
        SetFlag<HardEvent::MTE2_MTE1>(l1Event);
        WaitFlag<HardEvent::MTE2_MTE1>(l1Event);

        WaitFlag<HardEvent::M_MTE1>(l0Event);
        LoadA(aL0, aL1, mRound, kRound);
        LoadB(bL0, bL1, kRound, nRound);
        SetFlag<HardEvent::MTE1_MTE2>(l1Event);
        SetFlag<HardEvent::MTE1_M>(l0Event);
        WaitFlag<HardEvent::MTE1_M>(l0Event);

        WaitFlag<HardEvent::FIX_M>(l1Event);
        Mmad(cL0, aL0, bL0, MmadParams(mRound, nRound, kRound, 0, false, true));
        SetFlag<HardEvent::M_MTE1>(l0Event);
        SetFlag<HardEvent::M_FIX>(l1Event);
        WaitFlag<HardEvent::M_FIX>(l1Event);

        if constexpr (STAGED_OUTPUT) {
            const uint32_t stagedDim = AlignUp16(outputDim);
            FixpipeParamsV220 fixParams(stagedDim, chunkLen, mRound, stagedDim, false);
            fixParams.quantPre = QuantMode_t::F322BF16;
            Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(workspaceGm_[stagedOffset], cL0, fixParams);
        } else {
            const uint64_t outputOffset =
                (static_cast<uint64_t>(headIdx) * tiling_->totalTokens + chunkStart) *
                    outputDim + dimOffset;
            FixpipeParamsV220 fixParams(validDim, chunkLen, mRound, outputDim, false);
            fixParams.quantPre = QuantMode_t::F322BF16;
            Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(output[outputOffset], cL0, fixParams);
        }
        SetFlag<HardEvent::FIX_M>(l1Event);
        ++cubeSeq_;
    }

    __aicore__ inline void RunCubeCachedL0B(uint32_t slot, bool loadB,
        uint32_t aOffset, GlobalTensor<bfloat16_t>& bInput, uint64_t bOffset,
        GlobalTensor<bfloat16_t>& output,
        uint32_t chunkStart, uint32_t headIdx, uint32_t outputDim, uint32_t dimOffset,
        uint32_t inputRowStride, uint32_t chunkLen, uint32_t validDim)
    {
        const uint32_t mRound = (chunkLen + 15U) & ~15U;
        const uint32_t kRound = mRound;
        const uint32_t nRound = (validDim + 15U) & ~15U;
        const uint32_t fixedSlot = slot & 1U;
        const event_t l1Event = SelectL1Event(fixedSlot);
        const event_t l0Event = SelectL0Event(fixedSlot);

        LocalTensor<bfloat16_t> aL1 = aL1Buf_.Get<bfloat16_t>()[fixedSlot * A_ELEMS];
        LocalTensor<bfloat16_t> bL1 = bL1Buf_.Get<bfloat16_t>()[fixedSlot * TILE_MATRIX_ELEMS];
        LocalTensor<bfloat16_t> aL0 = aL0Buf_.Get<bfloat16_t>()[fixedSlot * A_ELEMS];
        LocalTensor<bfloat16_t> bL0 = bL0Buf_.Get<bfloat16_t>()[fixedSlot * TILE_MATRIX_ELEMS];
        LocalTensor<float> cL0 = cL0Buf_.Get<float>()[fixedSlot * TILE_MATRIX_ELEMS];

        WaitFlag<HardEvent::MTE1_MTE2>(l1Event);
        CopyNdToL1(aL1, workspaceGm_[aOffset], chunkLen, chunkLen, CHUNK_SIZE);
        if (loadB) {
            CopyNdToL1(bL1, bInput[bOffset], chunkLen, validDim, inputRowStride);
        }
        SetFlag<HardEvent::MTE2_MTE1>(l1Event);
        WaitFlag<HardEvent::MTE2_MTE1>(l1Event);

        WaitFlag<HardEvent::M_MTE1>(l0Event);
        LoadA(aL0, aL1, mRound, kRound);
        if (loadB) {
            LoadB(bL0, bL1, kRound, nRound);
        }
        SetFlag<HardEvent::MTE1_MTE2>(l1Event);
        SetFlag<HardEvent::MTE1_M>(l0Event);
        WaitFlag<HardEvent::MTE1_M>(l0Event);

        WaitFlag<HardEvent::FIX_M>(l1Event);
        Mmad(cL0, aL0, bL0, MmadParams(mRound, nRound, kRound, 0, false, true));
        SetFlag<HardEvent::M_MTE1>(l0Event);
        SetFlag<HardEvent::M_FIX>(l1Event);
        WaitFlag<HardEvent::M_FIX>(l1Event);

        const uint64_t outputOffset =
            (static_cast<uint64_t>(headIdx) * tiling_->totalTokens + chunkStart) * outputDim + dimOffset;
        FixpipeParamsV220 fixParams(validDim, chunkLen, mRound, outputDim, false);
        fixParams.quantPre = QuantMode_t::F322BF16;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(output[outputOffset], cL0, fixParams);
        SetFlag<HardEvent::FIX_M>(l1Event);
        ++cubeSeq_;
    }

    __aicore__ inline void RunCubeD128Slot(uint32_t slot, bool loadB,
        uint32_t aOffset, GlobalTensor<bfloat16_t>& bInput, uint64_t bOffset,
        GlobalTensor<bfloat16_t>& output,
        uint32_t chunkStart, uint32_t headIdx, uint32_t outputDim,
        uint32_t inputRowStride, uint32_t chunkLen)
    {
        const uint32_t mRound = (chunkLen + 15U) & ~15U;
        const uint32_t kRound = mRound;
        const uint32_t nRound = (outputDim + 15U) & ~15U;
        const uint32_t fixedSlot = slot & 3U;
        const uint32_t aSlot = fixedSlot & 1U;
        const event_t l1Event = SelectL1Event(fixedSlot);
        const event_t l0Event = SelectL0Event(fixedSlot);
        const event_t aL1Event = SelectL1Event(aSlot);
        const event_t aL0Event = SelectL0Event(aSlot);

        LocalTensor<bfloat16_t> aL1 = aL1Buf_.Get<bfloat16_t>()[aSlot * A_ELEMS];
        LocalTensor<bfloat16_t> bL1 = bL1Buf_.Get<bfloat16_t>()[fixedSlot * D128_MATRIX_ELEMS];
        LocalTensor<bfloat16_t> aL0 = aL0Buf_.Get<bfloat16_t>()[aSlot * A_ELEMS];
        LocalTensor<bfloat16_t> bL0 = bL0Buf_.Get<bfloat16_t>()[fixedSlot * D128_MATRIX_ELEMS];
        LocalTensor<float> cL0 = cL0Buf_.Get<float>()[fixedSlot * D128_MATRIX_ELEMS];

        if (fixedSlot < 2U) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1Event);
            CopyNdToL1(aL1, workspaceGm_[aOffset], chunkLen, chunkLen, CHUNK_SIZE);
            if (loadB) {
                CopyNdToL1(bL1, bInput[bOffset], chunkLen, outputDim, inputRowStride);
            }
            SetFlag<HardEvent::MTE2_MTE1>(l1Event);
            WaitFlag<HardEvent::MTE2_MTE1>(l1Event);

            WaitFlag<HardEvent::M_MTE1>(l0Event);
            LoadA(aL0, aL1, mRound, kRound);
            if (loadB) {
                LoadB(bL0, bL1, kRound, nRound);
            }
            SetFlag<HardEvent::MTE1_MTE2>(l1Event);
            SetFlag<HardEvent::MTE1_M>(l0Event);
            WaitFlag<HardEvent::MTE1_M>(l0Event);
        } else {
            WaitFlag<HardEvent::MTE1_MTE2>(aL1Event);
            CopyNdToL1(aL1, workspaceGm_[aOffset], chunkLen, chunkLen, CHUNK_SIZE);
            SetFlag<HardEvent::MTE2_MTE1>(aL1Event);

            WaitFlag<HardEvent::MTE1_MTE2>(l1Event);
            if (loadB) {
                CopyNdToL1(bL1, bInput[bOffset], chunkLen, outputDim, inputRowStride);
            }
            SetFlag<HardEvent::MTE2_MTE1>(l1Event);
            WaitFlag<HardEvent::MTE2_MTE1>(aL1Event);
            WaitFlag<HardEvent::MTE2_MTE1>(l1Event);

            WaitFlag<HardEvent::M_MTE1>(aL0Event);
            WaitFlag<HardEvent::M_MTE1>(l0Event);
            LoadA(aL0, aL1, mRound, kRound);
            if (loadB) {
                LoadB(bL0, bL1, kRound, nRound);
            }
            SetFlag<HardEvent::MTE1_MTE2>(aL1Event);
            SetFlag<HardEvent::MTE1_MTE2>(l1Event);
            SetFlag<HardEvent::MTE1_M>(l0Event);
            WaitFlag<HardEvent::MTE1_M>(l0Event);
        }

        WaitFlag<HardEvent::FIX_M>(l1Event);
        Mmad(cL0, aL0, bL0, MmadParams(mRound, nRound, kRound, 0, false, true));
        if (fixedSlot >= 2U) {
            SetFlag<HardEvent::M_MTE1>(aL0Event);
        }
        SetFlag<HardEvent::M_MTE1>(l0Event);
        SetFlag<HardEvent::M_FIX>(l1Event);
        WaitFlag<HardEvent::M_FIX>(l1Event);

        const uint64_t outputOffset =
            (static_cast<uint64_t>(headIdx) * tiling_->totalTokens + chunkStart) * outputDim;
        FixpipeParamsV220 fixParams(outputDim, chunkLen, mRound, outputDim, false);
        fixParams.quantPre = QuantMode_t::F322BF16;
        Fixpipe<bfloat16_t, float, CFG_ROW_MAJOR>(output[outputOffset], cL0, fixParams);
        SetFlag<HardEvent::FIX_M>(l1Event);
        ++cubeSeq_;
    }

private:
    const RecomputeWuTilingData* tiling_ = nullptr;
    uint32_t blockIdx_ = 0;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> vGm_;
    GlobalTensor<int32_t> offsetsGm_;
    GlobalTensor<bfloat16_t> wGm_;
    GlobalTensor<bfloat16_t> uGm_;
    GlobalTensor<bfloat16_t> workspaceGm_;
    uint32_t cubeSeq_ = 0;
    TBuf<TPosition::A1> aL1Buf_;
    TBuf<TPosition::B1> bL1Buf_;
    TBuf<TPosition::A2> aL0Buf_;
    TBuf<TPosition::B2> bL0Buf_;
    TBuf<TPosition::CO1> cL0Buf_;
};

class RecomputeWuCubeBatchChunks {
public:
    __aicore__ inline void Init(GM_ADDR globalChunkOffsets, GM_ADDR w, GM_ADDR u,
        GM_ADDR userWorkspace, const RecomputeWuTilingData* tilingData, TPipe* pipe)
    {
        tiling_ = tilingData;
        blockIdx_ = GetBlockIdx();
        offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(globalChunkOffsets));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w));
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(u));
        const uint32_t workspacePerCoreElems = tiling_->workspacePerCore / sizeof(bfloat16_t);
        const uint64_t coreWsOffset = static_cast<uint64_t>(blockIdx_) * workspacePerCoreElems;
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWorkspace) + coreWsOffset,
            workspacePerCoreElems);
        REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), mm_, &tiling_->cubeTiling);
    }

    __aicore__ inline void Process()
    {
        uint32_t taskSeq = 0;
        for (uint32_t task = blockIdx_; task < tiling_->taskCount; task += tiling_->blockDim) {
            const uint32_t tasksPerChunkGroup = tiling_->groupHeadNum * tiling_->headBatchCount;
            const uint32_t chunkGroup = task / tasksPerChunkGroup;
            const uint32_t taskInChunkGroup = task - chunkGroup * tasksPerChunkGroup;
            const uint32_t groupHead = taskInChunkGroup / tiling_->headBatchCount;
            const uint32_t headBatch = taskInChunkGroup - groupHead * tiling_->headBatchCount;
            const uint32_t groupSize = tiling_->headNum / tiling_->groupHeadNum;
            const uint32_t headStart = groupHead * groupSize + headBatch * tiling_->headsPerTask;
            uint32_t headCount = groupSize - headBatch * tiling_->headsPerTask;
            if (headCount > tiling_->headsPerTask) {
                headCount = tiling_->headsPerTask;
            }
            const uint32_t firstChunk = chunkGroup * tiling_->chunksPerTask;
            uint32_t chunkCount = tiling_->numChunks - firstChunk;
            if (chunkCount > tiling_->chunksPerTask) {
                chunkCount = tiling_->chunksPerTask;
            }
            const uint32_t bufferIdx = taskSeq & 1U;
            const uint32_t bufferBase = bufferIdx * BATCH_TASK_BUFFER_ELEMS;
            const uint32_t firstChunkStart = firstChunk * CHUNK_SIZE;

            CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE1>(GetWReadyFlag<false>(bufferIdx));
            for (uint32_t headOffset = 0U; headOffset < headCount; ++headOffset) {
                const uint32_t headIdx = headStart + headOffset;
                RunBatch(bufferBase + BATCH_TASK_AW_OFFSET + headOffset * chunkCount * A_ELEMS,
                    bufferBase + BATCH_TASK_BW_OFFSET, wGm_, firstChunkStart, headIdx,
                    chunkCount, A_ELEMS, CHUNK_SIZE * tiling_->keyDim);
            }

            CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE1>(GetUReadyFlag<false>(bufferIdx));
            for (uint32_t headOffset = 0U; headOffset < headCount; ++headOffset) {
                const uint32_t headIdx = headStart + headOffset;
                RunBatch(bufferBase + BATCH_TASK_AU_OFFSET + headOffset * chunkCount * A_ELEMS,
                    bufferBase + BATCH_TASK_BU_OFFSET +
                        headOffset * chunkCount * CHUNK_SIZE * tiling_->valueDim,
                    uGm_, firstChunkStart, headIdx, chunkCount,
                    A_ELEMS, CHUNK_SIZE * tiling_->valueDim);
            }

            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(GetDoneFlag<false>(bufferIdx));
            ++taskSeq;
        }
        mm_.End();
    }

private:
    __aicore__ inline void RunBatch(uint32_t aOffset, uint32_t bOffset,
        GlobalTensor<bfloat16_t>& output, uint32_t chunkStart, uint32_t headIdx,
        uint32_t chunkCount, uint32_t matrixStrideA, uint32_t matrixStrideB)
    {
#if defined(SPLIT_CORE_CUBE)
        const uint32_t outputDim = tiling_->keyDim;
        const uint64_t outputOffset =
            (static_cast<uint64_t>(headIdx) * tiling_->totalTokens + chunkStart) * outputDim;
        mm_.SetBatchNum(chunkCount, chunkCount);
        mm_.SetTensorA(workspaceGm_[aOffset]);
        mm_.SetTensorB(workspaceGm_[bOffset]);
        mm_.IterateBatch(output[outputOffset], false, 0, false, matrixStrideA, matrixStrideB, 0);
#endif
    }

private:
    const RecomputeWuTilingData* tiling_ = nullptr;
    uint32_t blockIdx_ = 0;
    GlobalTensor<int32_t> offsetsGm_;
    GlobalTensor<bfloat16_t> wGm_;
    GlobalTensor<bfloat16_t> uGm_;
    GlobalTensor<bfloat16_t> workspaceGm_;
    Matmul<BatchMatmulAType, BatchMatmulBType, BatchMatmulCType, MatmulBiasType,
        RECOMPUTE_WU_MATMUL_CONFIG<false>> mm_;
};

template <bool USE_MDL, bool PACKED_WIDE = false>
class RecomputeWuCubeMatmul {
public:
    __aicore__ inline void Init(GM_ADDR k, GM_ADDR v, GM_ADDR globalChunkOffsets, GM_ADDR w, GM_ADDR u,
        GM_ADDR userWorkspace, const RecomputeWuTilingData* tilingData, TPipe* pipe)
    {
        tiling_ = tilingData;
        blockIdx_ = GetBlockIdx();
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(k));
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(v));
        offsetsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(globalChunkOffsets));
        wGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w));
        uGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(u));
        const uint32_t workspacePerCoreElems = tiling_->workspacePerCore / sizeof(bfloat16_t);
        const uint64_t coreWsOffset = static_cast<uint64_t>(blockIdx_) * workspacePerCoreElems;
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWorkspace) + coreWsOffset,
            workspacePerCoreElems);
        REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), mm_, &tiling_->cubeTiling);
    }

    __aicore__ inline void Process()
    {
        const bool directKey = UseDirectKeyPath(tiling_);
        const uint32_t workspaceBufferElems = GetWorkspaceBufferElems(tiling_);
        uint32_t taskSeq = 0;
        for (uint32_t task = blockIdx_; task < tiling_->taskCount; task += tiling_->blockDim) {
            const uint32_t tasksPerChunkGroup = tiling_->groupHeadNum * tiling_->headBatchCount;
            const uint32_t chunkGroup = task / tasksPerChunkGroup;
            const uint32_t taskInChunkGroup = task - chunkGroup * tasksPerChunkGroup;
            const uint32_t groupHead = taskInChunkGroup / tiling_->headBatchCount;
            const uint32_t headBatch = taskInChunkGroup - groupHead * tiling_->headBatchCount;
            const uint32_t groupSize = tiling_->headNum / tiling_->groupHeadNum;
            const uint32_t headStart = groupHead * groupSize + headBatch * tiling_->headsPerTask;
            uint32_t headCount = groupSize - headBatch * tiling_->headsPerTask;
            if (headCount > tiling_->headsPerTask) {
                headCount = tiling_->headsPerTask;
            }
            const uint32_t firstChunk = chunkGroup * tiling_->chunksPerTask;
            uint32_t chunkCount = tiling_->numChunks - firstChunk;
            if (chunkCount > tiling_->chunksPerTask) {
                chunkCount = tiling_->chunksPerTask;
            }
            const uint32_t valueTileWidth = SelectValueTile<PACKED_WIDE>(tiling_);
            const uint32_t keyTileWidth = SelectPackedDimTile<PACKED_WIDE>(tiling_->keyDim);
            const uint32_t valueTileElems = CHUNK_SIZE * valueTileWidth;
            const uint32_t keyTileElems = CHUNK_SIZE * keyTileWidth;
            const uint32_t valueTiles = CeilDivU32(tiling_->valueDim, valueTileWidth);
            const uint32_t keyTiles = CeilDivU32(tiling_->keyDim, keyTileWidth);
            for (uint32_t headOffset = 0; headOffset < headCount; ++headOffset) {
                const uint32_t headIdx = headStart + headOffset;
                const uint32_t bufferIdx = taskSeq & 1U;
                const uint32_t batchBufferElems = workspaceBufferElems * tiling_->chunksPerTask;
                const uint32_t bufferBase = bufferIdx * batchBufferElems;
                CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE1>(GetWReadyFlag<false>(bufferIdx));
                for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                    uint32_t chunkStart = 0;
                    uint32_t chunkLen = 0;
                    if (!GetChunkInfo(firstChunk + chunkOffset, chunkStart, chunkLen)) {
                        continue;
                    }
                    const uint32_t chunkBufferBase = bufferBase + chunkOffset * workspaceBufferElems;
                    for (uint32_t tileIdx = 0; tileIdx < keyTiles; ++tileIdx) {
                        uint32_t validDim = tiling_->keyDim - tileIdx * keyTileWidth;
                        if (validDim > keyTileWidth) {
                            validDim = keyTileWidth;
                        }
                        if (directKey) {
                            const uint64_t keyOffset =
                                (static_cast<uint64_t>(chunkStart) * tiling_->groupHeadNum + groupHead) *
                                    tiling_->keyDim +
                                tileIdx * keyTileWidth;
                            RunCube(chunkBufferBase + WS_AW_OFFSET,
                                kGm_, keyOffset,
                                wGm_, chunkStart, headIdx, tiling_->keyDim,
                                tileIdx * keyTileWidth, tiling_->groupHeadNum * tiling_->keyDim,
                                chunkLen, validDim);
                        } else {
                            RunCube(chunkBufferBase + WS_AW_OFFSET,
                                workspaceGm_, chunkBufferBase + WS_BW_OFFSET + tileIdx * keyTileElems,
                                wGm_, chunkStart, headIdx, tiling_->keyDim,
                                tileIdx * keyTileWidth, keyTileWidth, chunkLen, validDim);
                        }
                    }
                }

                CrossCoreWaitFlag<SYNC_MODE2, PIPE_MTE1>(GetUReadyFlag<false>(bufferIdx));
                for (uint32_t chunkOffset = 0; chunkOffset < chunkCount; ++chunkOffset) {
                    uint32_t chunkStart = 0;
                    uint32_t chunkLen = 0;
                    if (!GetChunkInfo(firstChunk + chunkOffset, chunkStart, chunkLen)) {
                        continue;
                    }
                    const uint32_t chunkBufferBase = bufferBase + chunkOffset * workspaceBufferElems;
                    for (uint32_t tileIdx = 0; tileIdx < valueTiles; ++tileIdx) {
                        uint32_t validDim = tiling_->valueDim - tileIdx * valueTileWidth;
                        if (validDim > valueTileWidth) {
                            validDim = valueTileWidth;
                        }
                        RunCube(chunkBufferBase + WS_AU_OFFSET,
                            workspaceGm_, chunkBufferBase + WS_BU_OFFSET + tileIdx * valueTileElems,
                            uGm_, chunkStart, headIdx, tiling_->valueDim,
                            tileIdx * valueTileWidth, valueTileWidth, chunkLen, validDim);
                    }
                }
                CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(GetDoneFlag<false>(bufferIdx));
                ++taskSeq;
            }
        }
        mm_.End();
    }

private:
    __aicore__ inline bool GetChunkInfo(uint32_t chunkIdx, uint32_t& chunkStart, uint32_t& chunkLen)
    {
        const int32_t startValue = offsetsGm_.GetValue(chunkIdx);
        const int32_t endValue = offsetsGm_.GetValue(chunkIdx + 1U);
        if (startValue < 0 || endValue <= startValue) {
            return false;
        }
        chunkStart = static_cast<uint32_t>(startValue);
        chunkLen = static_cast<uint32_t>(endValue - startValue);
        if (chunkLen > CHUNK_SIZE) {
            chunkLen = CHUNK_SIZE;
        }
        return true;
    }

    __aicore__ inline void RunCube(
        uint32_t aOffset, GlobalTensor<bfloat16_t>& bInput, uint64_t bOffset,
        GlobalTensor<bfloat16_t>& output,
        uint32_t chunkStart, uint32_t headIdx, uint32_t outputDim, uint32_t dimOffset,
        uint32_t inputRowStride, uint32_t chunkLen, uint32_t validDim)
    {
        const uint64_t outputOffset =
            (static_cast<uint64_t>(headIdx) * tiling_->totalTokens + chunkStart) * outputDim + dimOffset;
        mm_.SetOrgShape(CHUNK_SIZE, inputRowStride, CHUNK_SIZE, CHUNK_SIZE, outputDim);
        mm_.SetTail(chunkLen, validDim, chunkLen);
        mm_.SetTensorA(workspaceGm_[aOffset]);
        mm_.SetTensorB(bInput[bOffset]);
        mm_.IterateAll(output[outputOffset]);
    }

private:
    const RecomputeWuTilingData* tiling_ = nullptr;
    uint32_t blockIdx_ = 0;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> vGm_;
    GlobalTensor<int32_t> offsetsGm_;
    GlobalTensor<bfloat16_t> wGm_;
    GlobalTensor<bfloat16_t> uGm_;
    GlobalTensor<bfloat16_t> workspaceGm_;
    Matmul<MatmulAType, MatmulBType, MatmulCType, MatmulBiasType,
        RECOMPUTE_WU_MATMUL_CONFIG<USE_MDL>> mm_;
};

} // namespace NsRecomputeWu

#endif // RECOMPUTEWU_H
