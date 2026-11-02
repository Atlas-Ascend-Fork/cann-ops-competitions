/*!
 * \file recompute_wu_tiling.cpp
 * \brief RecomputeWu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/recompute_wu_tiling_data.h"
#include "../op_kernel/recompute_wu_tiling_key.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {
namespace {

constexpr uint32_t CHUNK_SIZE = 64;
constexpr uint32_t DIM_TILE = 256;
constexpr uint32_t PACK_TILE = 128;
constexpr uint32_t MAX_CHUNKS_PER_TASK = 2;
constexpr uint32_t MAX_BATCH_CHUNKS_PER_TASK = 4;
constexpr uint32_t MAX_HEADS_PER_TASK = 4;
constexpr uint32_t MAX_BATCH_MATRICES_PER_TASK = MAX_BATCH_CHUNKS_PER_TASK * MAX_HEADS_PER_TASK;
constexpr uint32_t MAX_BATCH_OUTPUT_DIM = 48;
constexpr uint32_t SHORT_D128_HEAD_BATCH_CORE_NUMERATOR = 4;
constexpr uint32_t SHORT_D128_HEAD_BATCH_CORE_DENOMINATOR = 5;
constexpr uint32_t PIPELINE_DEPTH = 5;
constexpr uint32_t MATMUL_PIPELINE_DEPTH = 2;
constexpr uint32_t GROUP_PIPELINE_DEPTH = 3;
constexpr uint32_t A_ELEMS = CHUNK_SIZE * CHUNK_SIZE;
constexpr uint32_t FULL_WORKSPACE_BUFFER_ELEMS = 2U * A_ELEMS + 2U * CHUNK_SIZE * DIM_TILE;
constexpr uint32_t BF16_BYTES = sizeof(uint16_t);
constexpr uint32_t MDL_TASK_MIN = 256;

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    if (align == 0U) {
        return value;
    }
    return (value + align - 1U) / align * align;
}

bool UseDirectKeyPath(uint32_t groupSize, uint32_t taskCount, uint32_t blockDim, uint32_t keyDim)
{
    if (groupSize < 4U) {
        return true;
    }
    const uint32_t shortTaskLimit = blockDim * 4U;
    return taskCount <= shortTaskLimit && keyDim >= CHUNK_SIZE;
}

bool UseMdlMatmul(uint32_t keyDim, uint32_t valueDim, uint32_t groupSize, uint32_t taskCount)
{
    return keyDim == PACK_TILE && valueDim == PACK_TILE &&
        groupSize >= 4U && taskCount >= MDL_TASK_MIN;
}

bool UseOfficialMmadPath(bool directKey, uint32_t keyDim, uint32_t valueDim,
    uint32_t groupSize, uint32_t taskCount, uint32_t blockDim)
{
    const bool useDirectMmad = directKey && keyDim >= PACK_TILE && valueDim >= PACK_TILE;
    const bool usePackedD128Mmad = groupSize >= 8U && keyDim == PACK_TILE && valueDim == PACK_TILE;
    return (useDirectMmad || usePackedD128Mmad) && taskCount >= blockDim * 2U;
}

bool GetShape(const gert::StorageShape* storageShape, gert::Shape& shape)
{
    if (storageShape == nullptr) {
        return false;
    }
    shape = storageShape->GetStorageShape();
    return true;
}

bool IsPositiveRank(const gert::Shape& shape, size_t rank)
{
    if (shape.GetDimNum() != rank) {
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

ge::graphStatus ValidateDtypes(gert::TilingContext* context)
{
    const ge::DataType expectedInputs[] = {
        ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_INT32,
    };
    for (size_t i = 0; i < 6; ++i) {
        const auto* desc = context->GetInputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_IF(desc->GetDataType() != expectedInputs[i],
            OP_LOGE(context, "input %zu has invalid dtype", i), return ge::GRAPH_FAILED);
    }
    for (size_t i = 0; i < 2; ++i) {
        const auto* desc = context->GetOutputDesc(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        OP_CHECK_IF(desc->GetDataType() != ge::DT_BF16,
            OP_LOGE(context, "output %zu must be bfloat16", i), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildCubeTiling(const platform_ascendc::PlatformAscendC& platform,
    AscendC::tiling::TCubeTiling& cubeTiling)
{
    matmul_tiling::MatmulApiTiling tiling(platform);
    OP_CHECK_IF(tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16) != 0,
        OP_LOGE("RecomputeWu", "SetAType failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16) != 0,
        OP_LOGE("RecomputeWu", "SetBType failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16) != 0,
        OP_LOGE("RecomputeWu", "SetCType failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetShape(CHUNK_SIZE, DIM_TILE, CHUNK_SIZE) != 0 ||
        tiling.SetOrgShape(CHUNK_SIZE, DIM_TILE, CHUNK_SIZE) != 0,
        OP_LOGE("RecomputeWu", "SetShape failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetFixSplit(CHUNK_SIZE, DIM_TILE, CHUNK_SIZE) != 0,
        OP_LOGE("RecomputeWu", "SetFixSplit failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetBufferSpace(-1, -1, -1) != 0,
        OP_LOGE("RecomputeWu", "SetBufferSpace failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.GetTiling(cubeTiling) != 0,
        OP_LOGE("RecomputeWu", "GetTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildBatchChunkCubeTiling(const platform_ascendc::PlatformAscendC& platform,
    uint32_t outputDim, AscendC::tiling::TCubeTiling& cubeTiling)
{
    matmul_tiling::BatchMatmulTiling tiling(platform);
    OP_CHECK_IF(tiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16, false) != 0,
        OP_LOGE("RecomputeWu", "batch SetAType failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16, false) != 0,
        OP_LOGE("RecomputeWu", "batch SetBType failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
        matmul_tiling::DataType::DT_BF16) != 0,
        OP_LOGE("RecomputeWu", "batch SetCType failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetShape(CHUNK_SIZE, outputDim, CHUNK_SIZE) != 0 ||
        tiling.SetOrgShape(CHUNK_SIZE, outputDim, CHUNK_SIZE) != 0,
        OP_LOGE("RecomputeWu", "batch SetShape failed"), return ge::GRAPH_FAILED);
    const uint32_t baseN = (outputDim + 15U) & ~15U;
    OP_CHECK_IF(tiling.SetFixSplit(CHUNK_SIZE, baseN, CHUNK_SIZE) != 0,
        OP_LOGE("RecomputeWu", "batch SetFixSplit failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetBatchInfoForNormal(MAX_BATCH_CHUNKS_PER_TASK, MAX_BATCH_CHUNKS_PER_TASK,
        CHUNK_SIZE, outputDim, CHUNK_SIZE) != 0 ||
        tiling.SetBatchNum(MAX_BATCH_CHUNKS_PER_TASK) != 0,
        OP_LOGE("RecomputeWu", "batch SetBatchInfo failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.SetBufferSpace(-1, -1, -1) != 0,
        OP_LOGE("RecomputeWu", "batch SetBufferSpace failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tiling.GetTiling(cubeTiling) == -1,
        OP_LOGE("RecomputeWu", "batch GetTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus RecomputeWuTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(ValidateDtypes(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "dtype validation failed"), return ge::GRAPH_FAILED);

    gert::Shape aShape;
    gert::Shape kShape;
    gert::Shape vShape;
    gert::Shape betaShape;
    gert::Shape gShape;
    gert::Shape offsetsShape;
    OP_CHECK_IF(!GetShape(context->GetInputShape(0), kShape) || !IsPositiveRank(kShape, 4) ||
        !GetShape(context->GetInputShape(1), vShape) || !IsPositiveRank(vShape, 4) ||
        !GetShape(context->GetInputShape(2), aShape) || !IsPositiveRank(aShape, 4) ||
        !GetShape(context->GetInputShape(3), betaShape) || !IsPositiveRank(betaShape, 3) ||
        !GetShape(context->GetInputShape(4), gShape) || !IsPositiveRank(gShape, 3) ||
        !GetShape(context->GetInputShape(5), offsetsShape) || !IsPositiveRank(offsetsShape, 1),
        OP_LOGE(context, "invalid input rank or dimension"), return ge::GRAPH_FAILED);

    const int64_t batch = aShape.GetDim(0);
    const int64_t headNum = aShape.GetDim(1);
    const int64_t totalTokens = aShape.GetDim(2);
    const int64_t chunkSize = aShape.GetDim(3);
    const int64_t groupHeadNum = kShape.GetDim(2);
    const int64_t keyDim = kShape.GetDim(3);
    const int64_t valueDim = vShape.GetDim(3);
    const int64_t numChunks = offsetsShape.GetDim(0) - 1;

    OP_CHECK_IF(batch != 1 || kShape.GetDim(0) != batch || vShape.GetDim(0) != batch ||
        betaShape.GetDim(0) != batch || gShape.GetDim(0) != batch,
        OP_LOGE(context, "only B=1 is supported"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(kShape.GetDim(1) != totalTokens || vShape.GetDim(1) != totalTokens ||
        betaShape.GetDim(2) != totalTokens || gShape.GetDim(2) != totalTokens,
        OP_LOGE(context, "all inputs must share T"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(vShape.GetDim(2) != headNum || betaShape.GetDim(1) != headNum || gShape.GetDim(1) != headNum,
        OP_LOGE(context, "A, v, beta and g must share H"), return ge::GRAPH_FAILED);
    if (groupHeadNum <= 0) {
        OP_LOGE(context, "Hg must be positive");
        return ge::GRAPH_FAILED;
    }
    const int64_t validatedGroupHeadNum = groupHeadNum == 0 ? 1 : groupHeadNum;
    OP_CHECK_IF(headNum % validatedGroupHeadNum != 0,
        OP_LOGE(context, "H must be divisible by Hg"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(chunkSize != CHUNK_SIZE || numChunks <= 0,
        OP_LOGE(context, "BT must be 64 and offsets must contain at least two entries"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDim > 256 || valueDim > 256,
        OP_LOGE(context, "K and V must not exceed 256"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    const platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t cubeCoreNum = std::max(1U, platform.GetCoreNumAic());
    const uint32_t groupSize = static_cast<uint32_t>(headNum / validatedGroupHeadNum);
    const bool uniformFullChunks =
        static_cast<uint64_t>(totalTokens) == static_cast<uint64_t>(numChunks) * CHUNK_SIZE;
    uint32_t chunksPerTask = 1U;
    // BatchMatmul writes each matrix with an implicit Cube-aligned N stride.
    // Keep the task-batched layout only when the logical N already has that
    // alignment; other dimensions retain the explicit-stride regular path.
    const bool batchChunkCandidate = groupSize >= 4U && keyDim == valueDim &&
        keyDim <= MAX_BATCH_OUTPUT_DIM && (keyDim & 15) == 0 &&
        uniformFullChunks;
    if (batchChunkCandidate) {
        const uint32_t headBatchCountForBatchPath =
            (groupSize + MAX_HEADS_PER_TASK - 1U) / MAX_HEADS_PER_TASK;
        const uint32_t chunkGroups =
            (static_cast<uint32_t>(numChunks) + MAX_BATCH_CHUNKS_PER_TASK - 1U) /
            MAX_BATCH_CHUNKS_PER_TASK;
        const uint64_t candidateTasks = static_cast<uint64_t>(chunkGroups) *
            static_cast<uint64_t>(groupHeadNum) * headBatchCountForBatchPath;
        if (candidateTasks >= MDL_TASK_MIN) {
            chunksPerTask = MAX_BATCH_CHUNKS_PER_TASK;
        }
    }
    const uint32_t pairedChunkGroups =
        (static_cast<uint32_t>(numChunks) + MAX_CHUNKS_PER_TASK - 1U) / MAX_CHUNKS_PER_TASK;
    const uint64_t pairedTaskCount = static_cast<uint64_t>(pairedChunkGroups) *
        static_cast<uint64_t>(groupHeadNum) * groupSize;
    // For MHA/small GQA groups there is little K reuse across heads, so pairing chunks
    // amortizes task and CrossCore control. Keep at least two waves after pairing to
    // avoid sacrificing load balance on short or heavily tailed inputs.
    if (chunksPerTask == 1U && groupSize < 4U &&
        pairedTaskCount >= static_cast<uint64_t>(cubeCoreNum) * 2U) {
        chunksPerTask = MAX_CHUNKS_PER_TASK;
    } else if (chunksPerTask == 1U && groupSize >= 4U) {
        const uint32_t pairedHeadBatchCount =
            (groupSize + MAX_HEADS_PER_TASK - 1U) / MAX_HEADS_PER_TASK;
        const uint64_t pairedLargeGroupTaskCount = static_cast<uint64_t>(pairedChunkGroups) *
            static_cast<uint64_t>(groupHeadNum) * pairedHeadBatchCount;
        if (pairedLargeGroupTaskCount >= static_cast<uint64_t>(cubeCoreNum) * 2U) {
            chunksPerTask = MAX_CHUNKS_PER_TASK;
        }
    }
    const uint32_t chunkGroupCount =
        (static_cast<uint32_t>(numChunks) + chunksPerTask - 1U) / chunksPerTask;
    uint32_t headsPerTask = 1U;
    if (groupSize >= 4U) {
        // Large GQA groups benefit more from reusing the packed K matrix across four
        // heads than from coarsening the chunk granularity.
        const uint32_t batchCandidates[] = {MAX_HEADS_PER_TASK, 2U};
        for (const uint32_t candidate : batchCandidates) {
            const uint32_t batchCount = (groupSize + candidate - 1U) / candidate;
            const uint64_t candidateTasks = static_cast<uint64_t>(chunkGroupCount) *
                static_cast<uint64_t>(groupHeadNum) * batchCount;
            if (candidateTasks >= cubeCoreNum) {
                headsPerTask = candidate;
                break;
            }
        }
        // A short D128 group-of-four workload can be too small to select the
        // lower-overhead official-MMAD path. Use the largest head batch that
        // still reaches two Cube waves, preserving as much K reuse as possible.
        if (groupSize == MAX_HEADS_PER_TASK && keyDim == PACK_TILE && valueDim == PACK_TILE) {
            const uint64_t fourHeadTasks = static_cast<uint64_t>(chunkGroupCount) *
                static_cast<uint64_t>(groupHeadNum);
            const uint64_t twoHeadTasks = fourHeadTasks * 2U;
            const uint64_t twoWaveTasks = static_cast<uint64_t>(cubeCoreNum) * 2U;
            if (fourHeadTasks < twoWaveTasks && twoHeadTasks >= twoWaveTasks) {
                headsPerTask = 2U;
            }
        }
    } else if (groupSize == 2U && keyDim == PACK_TILE && valueDim == PACK_TILE) {
        // The two query heads share K. Batch them only when at least two Cube
        // waves remain. On uniform full chunks, using at least 80% of the Cube
        // cores still preserves two waves while retaining the shared D128 K tile.
        const uint64_t twoHeadTasks =
            static_cast<uint64_t>(chunkGroupCount) * groupHeadNum;
        const uint32_t reducedBlockFloor = std::max(1U,
            cubeCoreNum * SHORT_D128_HEAD_BATCH_CORE_NUMERATOR /
                SHORT_D128_HEAD_BATCH_CORE_DENOMINATOR);
        if (uniformFullChunks &&
            twoHeadTasks >= static_cast<uint64_t>(reducedBlockFloor) * 2U) {
            headsPerTask = 2U;
        }
    }
    const uint32_t headBatchCount = (groupSize + headsPerTask - 1U) / headsPerTask;
    const uint64_t taskCount64 = static_cast<uint64_t>(chunkGroupCount) *
        static_cast<uint64_t>(groupHeadNum) * headBatchCount;
    OP_CHECK_IF(taskCount64 == 0 || taskCount64 > std::numeric_limits<uint32_t>::max(),
        OP_LOGE(context, "task count overflow"), return ge::GRAPH_FAILED);

    const uint32_t taskCount = static_cast<uint32_t>(taskCount64);
    uint32_t blockDim = std::min(cubeCoreNum, taskCount);
    const uint32_t reducedBlockFloor = std::max(1U,
        cubeCoreNum * SHORT_D128_HEAD_BATCH_CORE_NUMERATOR /
            SHORT_D128_HEAD_BATCH_CORE_DENOMINATOR);
    const bool useReducedShortD128Grid =
        groupSize >= 2U && groupSize <= 4U &&
        keyDim == PACK_TILE && valueDim == PACK_TILE && headsPerTask > 1U;
    const bool useReducedShortD256Group4Grid =
        groupSize == MAX_HEADS_PER_TASK &&
        keyDim == DIM_TILE && valueDim == DIM_TILE &&
        headsPerTask == MAX_HEADS_PER_TASK;
    const bool useReducedShortHeadBatchGrid = uniformFullChunks &&
        (useReducedShortD128Grid || useReducedShortD256Group4Grid) &&
        taskCount >= reducedBlockFloor * 2U && taskCount < cubeCoreNum * 2U;
    if (useReducedShortHeadBatchGrid) {
        blockDim = taskCount / 2U;
    }
    const uint32_t schedulingTaskCount = groupSize >= 4U ? taskCount * chunksPerTask : taskCount;
    const bool useDirectD128HeadBatch = groupSize >= 2U && groupSize <= 4U &&
        keyDim == PACK_TILE &&
        valueDim == PACK_TILE && headsPerTask > 1U && taskCount >= blockDim * 2U;
    const bool useDirectD256Group4HeadBatch = groupSize == MAX_HEADS_PER_TASK &&
        keyDim == DIM_TILE && valueDim == DIM_TILE &&
        headsPerTask == MAX_HEADS_PER_TASK && taskCount >= blockDim * 2U;
    const bool useDirectHeadBatch =
        useDirectD128HeadBatch || useDirectD256Group4HeadBatch;
    const bool useD128GroupPipeline = useDirectD128HeadBatch && groupSize == 4U &&
        headsPerTask == MAX_HEADS_PER_TASK && taskCount >= blockDim * 4U;
    const bool useGroupPipeline =
        useD128GroupPipeline || useDirectD256Group4HeadBatch;
    const bool directKey = useDirectHeadBatch || UseDirectKeyPath(
        groupSize, schedulingTaskCount, blockDim, static_cast<uint32_t>(keyDim));
    const bool useOfficialMmad = UseOfficialMmadPath(directKey, static_cast<uint32_t>(keyDim),
        static_cast<uint32_t>(valueDim), groupSize, taskCount, blockDim);
    const bool usePackedD128DirectValue = useOfficialMmad && !directKey &&
        keyDim == PACK_TILE && valueDim == PACK_TILE && headsPerTask > 1U && groupSize >= 8U;
    const bool useSmallGroupDirectValue = useOfficialMmad && groupSize < 4U;
    const bool useBatchChunks = batchChunkCandidate && headsPerTask == MAX_HEADS_PER_TASK &&
        chunksPerTask >= MAX_CHUNKS_PER_TASK && chunksPerTask <= MAX_BATCH_CHUNKS_PER_TASK &&
        taskCount >= MDL_TASK_MIN;
    const bool useOfficialMmadStagedOutput = useOfficialMmad && groupSize >= 4U &&
        keyDim > 240 && keyDim < 256 && valueDim > 240 && valueDim < 256;
    const bool usePackedWide = !useBatchChunks && groupSize >= 4U && keyDim <= 48 && valueDim <= 48 &&
        taskCount >= MDL_TASK_MIN;
    const uint32_t pipelineDepth = useOfficialMmad ? PIPELINE_DEPTH : MATMUL_PIPELINE_DEPTH;
    const bool compactWorkspace = directKey && valueDim >= PACK_TILE;
    uint32_t workspacePerCore = AlignUp(
        FULL_WORKSPACE_BUFFER_ELEMS * chunksPerTask * pipelineDepth * BF16_BYTES, 32U);
    if (compactWorkspace) {
        const uint32_t valueElems = CHUNK_SIZE * DIM_TILE;
        const uint32_t workspaceBufferElems = 2U * A_ELEMS + valueElems;
        workspacePerCore = AlignUp(
            workspaceBufferElems * chunksPerTask * pipelineDepth * BF16_BYTES, 32U);
        if (useGroupPipeline) {
            workspacePerCore = AlignUp(workspaceBufferElems * chunksPerTask *
                MAX_HEADS_PER_TASK * GROUP_PIPELINE_DEPTH * BF16_BYTES, 32U);
        }
    }
    if (useBatchChunks) {
        const uint32_t maxBatchRightElems = CHUNK_SIZE * MAX_BATCH_OUTPUT_DIM;
        const uint32_t batchTaskBufferElems =
            2U * MAX_BATCH_MATRICES_PER_TASK * A_ELEMS +
            (MAX_BATCH_CHUNKS_PER_TASK + MAX_BATCH_MATRICES_PER_TASK) * maxBatchRightElems;
        const uint32_t batchWorkspacePerCore = AlignUp(
            batchTaskBufferElems * MATMUL_PIPELINE_DEPTH * BF16_BYTES, 32U);
        workspacePerCore = std::max(workspacePerCore, batchWorkspacePerCore);
    }
    if (useOfficialMmadStagedOutput) {
        const uint32_t stagedKeyDim = AlignUp(static_cast<uint32_t>(keyDim), 16U);
        const uint32_t stagedValueDim = AlignUp(static_cast<uint32_t>(valueDim), 16U);
        const uint32_t stagedBufferElems =
            chunksPerTask * CHUNK_SIZE * (stagedKeyDim + stagedValueDim);
        workspacePerCore += AlignUp(
            stagedBufferElems * PIPELINE_DEPTH * BF16_BYTES, 32U);
    }

    RecomputeWuTilingData* tiling = context->GetTilingData<RecomputeWuTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->batchSize = static_cast<uint32_t>(batch);
    tiling->totalTokens = static_cast<uint32_t>(totalTokens);
    tiling->headNum = static_cast<uint32_t>(headNum);
    tiling->groupHeadNum = static_cast<uint32_t>(groupHeadNum);
    tiling->keyDim = static_cast<uint32_t>(keyDim);
    tiling->valueDim = static_cast<uint32_t>(valueDim);
    tiling->chunkSize = static_cast<uint32_t>(chunkSize);
    tiling->numChunks = static_cast<uint32_t>(numChunks);
    tiling->taskCount = taskCount;
    tiling->blockDim = blockDim;
    tiling->workspacePerCore = workspacePerCore;
    tiling->headsPerTask = headsPerTask;
    tiling->headBatchCount = headBatchCount;
    tiling->chunksPerTask = chunksPerTask;
    tiling->chunkGroupCount = chunkGroupCount;

    OP_CHECK_IF((useBatchChunks ? BuildBatchChunkCubeTiling(platform, static_cast<uint32_t>(keyDim),
        tiling->cubeTiling) :
        BuildCubeTiling(platform, tiling->cubeTiling)) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "cube tiling failed"), return ge::GRAPH_FAILED);

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) +
        static_cast<size_t>(blockDim) * workspacePerCore;

    context->SetBlockDim(blockDim);
    if (useOfficialMmad) {
        if (useOfficialMmadStagedOutput) {
            context->SetTilingKey(GET_TPL_TILING_KEY(
                RECOMPUTEWU_TPL_OFFICIAL_MMAD_STAGED_OUTPUT));
        } else if (useGroupPipeline) {
            context->SetTilingKey(GET_TPL_TILING_KEY(
                RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_GROUP_PIPELINE));
        } else if (useDirectHeadBatch) {
            context->SetTilingKey(
                GET_TPL_TILING_KEY(RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_GROUP4));
        } else if (usePackedD128DirectValue) {
            context->SetTilingKey(
                GET_TPL_TILING_KEY(RECOMPUTEWU_TPL_OFFICIAL_MMAD_D128_DIRECTV));
        } else if (useSmallGroupDirectValue) {
            context->SetTilingKey(GET_TPL_TILING_KEY(
                RECOMPUTEWU_TPL_OFFICIAL_MMAD_SMALL_GROUP_DIRECTV));
        } else {
            context->SetTilingKey(GET_TPL_TILING_KEY(RECOMPUTEWU_TPL_OFFICIAL_MMAD));
        }
    } else {
        const bool useMdl = UseMdlMatmul(static_cast<uint32_t>(keyDim),
            static_cast<uint32_t>(valueDim), groupSize, schedulingTaskCount);
        if (useBatchChunks) {
            context->SetTilingKey(GET_TPL_TILING_KEY(RECOMPUTEWU_TPL_BATCH_CHUNKS));
        } else if (usePackedWide) {
            context->SetTilingKey(GET_TPL_TILING_KEY(RECOMPUTEWU_TPL_NORMAL_PACKED_WIDE));
        } else {
            context->SetTilingKey(useMdl ? GET_TPL_TILING_KEY(RECOMPUTEWU_TPL_MDL) :
                GET_TPL_TILING_KEY(RECOMPUTEWU_TPL_NORMAL));
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRecomputeWu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct RecomputeWuCompileInfo {};

IMPL_OP_OPTILING(RecomputeWu)
    .Tiling(RecomputeWuTilingFunc)
    .TilingParse<RecomputeWuCompileInfo>(TilingParseForRecomputeWu);

} // namespace optiling
