/*!
 * \file group_norm_tiling.cpp
 * \brief GroupNorm tiling for Ascend 910B.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/group_norm_tiling_data.h"
#include "../op_kernel/group_norm_tiling_key.h"

#include <cstdint>

namespace optiling {

namespace {

constexpr uint32_t WORKSPACE_SIZE = 0U;
constexpr int64_t MAX_OUTPUT_TILE_ELEMENTS = 16384;
constexpr int64_t MAX_TILE_ELEMENTS = 8192;
constexpr int64_t MID_TILE_ELEMENTS = 4096;
constexpr int64_t MIN_TILE_ELEMENTS = 2048;
constexpr int64_t FLOATS_PER_BLOCK = 8;
constexpr int64_t UB_SAFETY_BYTES = 8192;
constexpr int64_t SMALL_LAUNCH_CORES = 8;
constexpr int64_t TINY_STATIC_MAX_ELEMENTS = 64;
constexpr int64_t REDUNDANT_STATS_MIN_HALF_TILES = 4;

inline int64_t CeilDivInt(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

inline int64_t CeilAlignInt(int64_t value, int64_t align)
{
    return CeilDivInt(value, align) * align;
}

ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum = platform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "AIV core number must be positive"), return ge::GRAPH_FAILED);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "UB size must be positive"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetWorkspace(gert::TilingContext* context)
{
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

bool IsPositiveShape(const gert::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) <= 0) {
            return false;
        }
    }
    return true;
}

int64_t SelectTileElements(uint64_t ubSize, int64_t channelsAlign, int64_t xTypeBytes)
{
    const int64_t candidates[] = {MAX_TILE_ELEMENTS, MID_TILE_ELEMENTS, MIN_TILE_ELEMENTS};
    for (const int64_t tile : candidates) {
        // One input queue + one output queue, two FP32 work buffers,
        // FP32 gamma/beta, one reduce-result block and a safety reserve.
        const uint64_t required = static_cast<uint64_t>(2 * tile * xTypeBytes) +
            static_cast<uint64_t>(2 * tile * sizeof(float)) +
            static_cast<uint64_t>(2 * channelsAlign * sizeof(float)) + 32U + UB_SAFETY_BYTES;
        if (required <= ubSize) {
            return tile;
        }
    }
    return 0;
}

int64_t SelectOutputElements(
    uint64_t ubSize, int64_t channelsAlign, int64_t xTypeBytes, int64_t statisticsElements)
{
    // A 16K output tile is legal only for 2-byte inputs: DataCopyPad blockLen
    // remains 32768B and the larger input/output/x1 buffers must still fit UB.
    if (xTypeBytes != sizeof(uint16_t)) {
        return statisticsElements;
    }
    const uint64_t required = static_cast<uint64_t>(2 * MAX_OUTPUT_TILE_ELEMENTS * xTypeBytes) +
        static_cast<uint64_t>(MAX_OUTPUT_TILE_ELEMENTS * sizeof(float)) +
        static_cast<uint64_t>(statisticsElements * sizeof(float)) +
        static_cast<uint64_t>(2 * channelsAlign * sizeof(float)) + 32U + UB_SAFETY_BYTES;
    return required <= ubSize ? MAX_OUTPUT_TILE_ELEMENTS : statisticsElements;
}

} // namespace

static ge::graphStatus GroupNormTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "failed to query platform information"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetWorkspace(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "failed to set workspace"), return ge::GRAPH_FAILED);

    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    const gert::StorageShape* gammaStorageShape = context->GetInputShape(1);
    const gert::StorageShape* betaStorageShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaStorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaStorageShape);

    const gert::Shape& xShape = xStorageShape->GetStorageShape();
    const gert::Shape& gammaShape = gammaStorageShape->GetStorageShape();
    const gert::Shape& betaShape = betaStorageShape->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() != 4 && xShape.GetDimNum() != 5,
        OP_LOGE(context, "x must be a 4D or 5D ND tensor"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsPositiveShape(xShape), OP_LOGE(context, "x dimensions must be positive"),
        return ge::GRAPH_FAILED);

    const auto* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::DataType xDtype = xDesc->GetDataType();
    OP_CHECK_IF(xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16 && xDtype != ge::DT_FLOAT,
        OP_LOGE(context, "x dtype must be float16, bfloat16 or float32"), return ge::GRAPH_FAILED);
    const int64_t xTypeBytes = xDtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);

    const int64_t batch = xShape.GetDim(0);
    const int64_t channels = xShape.GetDim(1);
    OP_CHECK_IF(gammaShape.GetDimNum() != 1 || gammaShape.GetDim(0) != channels,
        OP_LOGE(context, "gamma must have shape [C]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(betaShape.GetDimNum() != 1 || betaShape.GetDim(0) != channels,
        OP_LOGE(context, "beta must have shape [C]"), return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* numGroupsAttr = attrs->GetAttrPointer<int64_t>(0);
    const float* epsilonAttr = attrs->GetAttrPointer<float>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, numGroupsAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilonAttr);
    const int64_t numGroups = *numGroupsAttr;
    OP_CHECK_IF(numGroups <= 0 || numGroups > 32,
        OP_LOGE(context, "num_groups must be in [1, 32]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(channels % numGroups != 0,
        OP_LOGE(context, "num_groups must divide C"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*epsilonAttr < 0.0f, OP_LOGE(context, "epsilon must be non-negative"),
        return ge::GRAPH_FAILED);

    int64_t spatial = 1;
    for (size_t i = 2; i < xShape.GetDimNum(); ++i) {
        spatial *= xShape.GetDim(i);
    }
    const int64_t channelsPerGroup = channels / numGroups;
    const int64_t groupElements = channelsPerGroup * spatial;
    const int64_t totalGroups = batch * numGroups;
    const int64_t channelsAlign = CeilAlignInt(channels, FLOATS_PER_BLOCK);
    const int64_t tileElements = SelectTileElements(ubSize, channelsAlign, xTypeBytes);
    OP_CHECK_IF(tileElements == 0, OP_LOGE(context, "UB is too small for the GroupNorm baseline"),
        return ge::GRAPH_FAILED);
    int64_t outputElements = SelectOutputElements(
        ubSize, channelsAlign, xTypeBytes, tileElements);
    int32_t statisticsPrefetchMode = 1;
    if (xTypeBytes == sizeof(float) && groupElements > tileElements) {
        // FP32 large-group statistics can consume the DeQue tensor directly.
        // A second input slot allows MTE2 of tile k+1 to overlap the Vector
        // reductions of tile k; all other buffers retain the O22 layout.
        const uint64_t prefetchRequired = static_cast<uint64_t>(
            3 * outputElements * xTypeBytes) +
            static_cast<uint64_t>(2 * outputElements * sizeof(float)) +
            static_cast<uint64_t>(2 * channelsAlign * sizeof(float)) + 32U + UB_SAFETY_BYTES;
        if (prefetchRequired <= ubSize) {
            statisticsPrefetchMode = 2;
            // The key1 bound queue removes the separate 8K output and x1
            // buffers.  Reinvest exactly that UB in two 16K input/output
            // slots: the main working-set budget stays unchanged while both
            // statistics prefetch depth and output reuse are preserved.
            outputElements = MAX_OUTPUT_TILE_ELEMENTS;
        }
    } else if (xTypeBytes == sizeof(uint16_t) && groupElements > tileElements) {
        // Cast finishes before the reduction and releases the only input
        // queue slot.  Refill that slot immediately so MTE2 of tile k+1 can
        // overlap the FP32 reductions of tile k without any extra UB.
        statisticsPrefetchMode = 3;
    }

    // Resident groups have no cross-core reduction.  Above 16 groups, cap the
    // logical workers at 16 so each launched core amortizes TPipe and scheduler
    // overhead across consecutive groups.  Large groups retain full parallelism.
    const int64_t schedulingCoreNum = groupElements <= tileElements &&
        totalGroups > 16 ? (coreNum < 16 ? coreNum : 16) : coreNum;
    const int64_t groupsPerCore = CeilDivInt(totalGroups, schedulingCoreNum);
    const int64_t usedCoreNum = CeilDivInt(totalGroups, groupsPerCore);
    const int64_t lastCoreGroups = totalGroups - (usedCoreNum - 1) * groupsPerCore;

    // Two independent workers may reproduce the same stable moments and then
    // normalize disjoint channel halves.  Duplicating statistics avoids all
    // cross-core synchronization and workspace-lifetime dependencies.
    const bool useRedundantStatisticsSplit = xTypeBytes == sizeof(float) &&
        statisticsPrefetchMode == 2 && groupsPerCore == 1 &&
        channelsPerGroup % 2 == 0 && spatial % FLOATS_PER_BLOCK == 0 &&
        totalGroups * 2 <= coreNum &&
        (channelsPerGroup / 2) * spatial >=
            REDUNDANT_STATS_MIN_HALF_TILES * outputElements;

    GroupNormTilingData* tiling = context->GetTilingData<GroupNormTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->outputFactor = outputElements;
    tiling->blockFactor = groupsPerCore;
    // For 2-byte inputs the O15 layout already provides a 16K input/x1
    // allocation.  Large-group statistics can consume that full tile by
    // squaring x1 in place while x2 remains an 8K reduction workspace.
    tiling->ubFactor = outputElements;
    tiling->totalGroups = totalGroups;
    tiling->lastCoreGroups = lastCoreGroups;
    tiling->numGroups = numGroups;
    tiling->channels = channels;
    tiling->channelsPerGroup = channelsPerGroup;
    tiling->spatial = spatial;
    tiling->groupElements = groupElements;
    tiling->channelsAlign = channelsAlign;
    tiling->epsilon = *epsilonAttr;
    tiling->statisticsPrefetchMode = statisticsPrefetchMode;
    tiling->outputSplitWorkers = useRedundantStatisticsSplit ? 2 : 1;
    // A launch with fewer than eight logical groups can enter a lower-overhead
    // scheduler bucket at physical blockDim=8.  Surplus cores return before
    // TPipe setup, so the data mapping and per-group work remain unchanged.
    const int64_t launchCoreNum = useRedundantStatisticsSplit ? totalGroups * 2 :
        (totalGroups < SMALL_LAUNCH_CORES && coreNum >= SMALL_LAUNCH_CORES ?
            SMALL_LAUNCH_CORES : usedCoreNum);
    context->SetBlockDim(static_cast<uint32_t>(launchCoreNum));

    // Isolate the low-A first-tile scheduling leaf from the tiny key0 binary;
    // this avoids changing instruction layout for Case01-03.
    const bool useLowAFirstTile = totalGroups < SMALL_LAUNCH_CORES &&
        groupElements > MAX_TILE_ELEMENTS && statisticsPrefetchMode == 3;
    // FP32 large groups use the same isolated binary key to bind the input and
    // output queues.  This preserves key0 instruction layout while removing
    // the two FP32 zero-copy vector passes in the mode-2 normalization leaf.
    const bool useFp32BoundQueue = xTypeBytes == sizeof(float) &&
        statisticsPrefetchMode == 2;
    // Isolate high-parallel two-byte mode-3 groups so their shifted first and
    // second moments can share one Vector-to-Scalar publication per tile.
    const bool useTwoByteShiftedMoments = xTypeBytes == sizeof(uint16_t) &&
        statisticsPrefetchMode == 3 && groupsPerCore == 1 &&
        totalGroups >= SMALL_LAUNCH_CORES;
    const bool useTinyStaticTensor = totalGroups < SMALL_LAUNCH_CORES &&
        groupElements <= TINY_STATIC_MAX_ELEMENTS;
    if (useTinyStaticTensor) {
        context->SetTilingKey(GET_TPL_TILING_KEY(GROUPNORM_TPL_SCH_MODE_2));
    } else if (useTwoByteShiftedMoments) {
        context->SetTilingKey(GET_TPL_TILING_KEY(GROUPNORM_TPL_SCH_MODE_3));
    } else if (useLowAFirstTile || useFp32BoundQueue) {
        context->SetTilingKey(GET_TPL_TILING_KEY(GROUPNORM_TPL_SCH_MODE_1));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(GROUPNORM_TPL_SCH_MODE_0));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGroupNorm([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct GroupNormCompileInfo {};

IMPL_OP_OPTILING(GroupNorm)
    .Tiling(GroupNormTilingFunc)
    .TilingParse<GroupNormCompileInfo>(TilingParseForGroupNorm);

} // namespace optiling
