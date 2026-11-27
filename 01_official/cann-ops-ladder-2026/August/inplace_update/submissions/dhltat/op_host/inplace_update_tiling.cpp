/*!
 * \file inplace_update_tiling.cpp
 * \brief Tiling implementation for InplaceUpdate.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/inplace_update_tiling_data.h"
#include "../op_kernel/inplace_update_tiling_key.h"

#include <algorithm>

namespace optiling {

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t UB_RESERVED_BYTES = 16 * 1024;
constexpr int64_t DATA_UB_BYTES_LIMIT = 64 * 1024;
constexpr int64_t LARGE_INDEX_DATA_UB_BYTES_LIMIT = 160 * 1024;
constexpr int64_t LARGE_WIDE_DATA_UB_BYTES_LIMIT = 144 * 1024;
constexpr int64_t INDEX_UB_COUNT_LIMIT = 1024;
constexpr int64_t SMALL_INDEX_SKIP_LIMIT = 8;
constexpr int64_t TINY_NARROW_MULTI_ROW_BYTES = 128;
constexpr int64_t PACKED_SINGLE_ROW_BYTES = 512;
constexpr int64_t SMALL_EIGHT_CORE_MIN_BYTES = 392;
constexpr int64_t ONE_WORKER_EIGHT_CORE_BYTES = 32 * 1024;
constexpr int64_t WIDE_STREAM_MIN_BYTES = 64 * 1024;
constexpr int64_t WIDE_STREAM_MAX_BYTES = 2 * 1024 * 1024;
constexpr uint32_t MAX_USED_CORES = 32;
constexpr uint32_t LARGE_INDEX_TARGET_CORES = 32;
constexpr uint32_t LARGE_WIDE_TARGET_CORES = 32;
constexpr uint32_t MODE3_TINY_LAUNCH_CORES = 8;
constexpr uint32_t SMALL_MULTI_ROW_TARGET_CORES = 8;
constexpr uint32_t SMALL_ONE_WORKER_TARGET_CORES = 8;

enum class ScheduleKind {
    EMPTY,
    LARGE_INDEX_ROW_FILTER,
    LARGE_WIDE_ROW,
    PACKED_SINGLE_ROW,
    TINY_NARROW_MULTI_ROW,
    TINY_SINGLE_CORE,
    SMALL_MULTI_ROW_SPLIT,
    SMALL_ONE_WORKER_LAUNCH,
    SINGLE_INDEX_BANDWIDTH,
    DEFAULT_BANDWIDTH,
};

struct TilingSchedule {
    uint32_t blockDim = 1;
    int64_t blockFactor = 0;
    uint64_t tilingKey = 0;
};

static const gert::Shape g_vec_1_shape = {1};

static inline gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static int64_t ShapeElementCount(const gert::Shape& shape)
{
    if (shape.GetDimNum() == 0) {
        return 1;
    }

    int64_t count = 1;
    for (size_t idx = 0; idx < shape.GetDimNum(); ++idx) {
        const int64_t dim = shape.GetDim(idx);
        if (dim < 0) {
            return -1;
        }
        count *= dim;
    }
    return count;
}

static int64_t GetTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
        case ge::DT_UINT16:
            return 2;
        case ge::DT_DOUBLE:
        case ge::DT_INT64:
        case ge::DT_UINT64:
            return 8;
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            return 1;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
        default:
            return 4;
    }
}

static uint32_t ClampBlockDim(uint32_t target, int64_t totalBytes, uint32_t maxCoreNum)
{
    if (totalBytes <= 0 || maxCoreNum == 0) {
        return 1;
    }

    const uint32_t usefulCoreNum = static_cast<uint32_t>(CeilDiv(totalBytes, BLOCK_BYTES));
    target = std::min(target, usefulCoreNum);
    target = std::min(target, maxCoreNum);
    return std::max<uint32_t>(target, 1);
}

static uint32_t SelectBandwidthBlockDim(int64_t totalBytes, uint32_t maxCoreNum)
{
    uint32_t target = MAX_USED_CORES;
    if (totalBytes <= 256 * 1024) {
        target = 8;
    } else if (totalBytes <= 1024 * 1024) {
        target = 16;
    } else if (totalBytes <= 6 * 1024 * 1024) {
        // Mid single-index bandwidth inputs get 16 cores: 320KiB of
        // contiguous work per core keeps the MTE2/MTE3 chain long instead
        // of splitting it across 32 cores.
        target = 16;
    }
    return ClampBlockDim(target, totalBytes, maxCoreNum);
}

static ScheduleKind SelectScheduleKind(
    int64_t totalBytes,
    int64_t rowNum,
    int64_t innerBytes,
    int64_t indexNum,
    int64_t typeSize)
{
    if (totalBytes <= 0) {
        return ScheduleKind::EMPTY;
    }
    if (indexNum >= 256 && innerBytes <= 4096 && totalBytes > ONE_WORKER_EIGHT_CORE_BYTES) {
        return ScheduleKind::LARGE_INDEX_ROW_FILTER;
    }
    if (innerBytes >= 8192 && totalBytes > ONE_WORKER_EIGHT_CORE_BYTES) {
        return ScheduleKind::LARGE_WIDE_ROW;
    }
    if (rowNum == 1 && totalBytes <= PACKED_SINGLE_ROW_BYTES && indexNum > 0 &&
        indexNum <= SMALL_INDEX_SKIP_LIMIT) {
        return ScheduleKind::PACKED_SINGLE_ROW;
    }
    if ((totalBytes <= TINY_NARROW_MULTI_ROW_BYTES && typeSize <= 2 && rowNum > 1 && indexNum > 1) ||
        (totalBytes > TINY_NARROW_MULTI_ROW_BYTES && totalBytes <= PACKED_SINGLE_ROW_BYTES && rowNum > 1 &&
         indexNum > 1 && indexNum <= SMALL_INDEX_SKIP_LIMIT && innerBytes <= TINY_NARROW_MULTI_ROW_BYTES)) {
        return ScheduleKind::TINY_NARROW_MULTI_ROW;
    }
    if (totalBytes <= SMALL_EIGHT_CORE_MIN_BYTES) {
        return ScheduleKind::TINY_SINGLE_CORE;
    }
    if (totalBytes <= ONE_WORKER_EIGHT_CORE_BYTES && rowNum > 1 && innerBytes <= 1024) {
        return ScheduleKind::SMALL_MULTI_ROW_SPLIT;
    }
    if (totalBytes <= ONE_WORKER_EIGHT_CORE_BYTES) {
        return ScheduleKind::SMALL_ONE_WORKER_LAUNCH;
    }
    if (indexNum == 1) {
        return ScheduleKind::SINGLE_INDEX_BANDWIDTH;
    }
    return ScheduleKind::DEFAULT_BANDWIDTH;
}

static int64_t CalcAlignedBlockFactor(int64_t totalNum, uint32_t blockDim, int64_t factorAlign)
{
    if (totalNum <= 0) {
        return 0;
    }
    return CeilAlign(CeilDiv(totalNum, static_cast<int64_t>(blockDim)), factorAlign);
}

static TilingSchedule MakeSchedule(
    int64_t totalNum,
    int64_t totalBytes,
    int64_t rowNum,
    int64_t innerNum,
    int64_t innerBytes,
    int64_t indexNum,
    int64_t typeSize,
    int64_t blockAlign,
    uint32_t maxCoreNum)
{
    TilingSchedule schedule;
    schedule.tilingKey = GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_1);
    const ScheduleKind kind = SelectScheduleKind(totalBytes, rowNum, innerBytes, indexNum, typeSize);

    switch (kind) {
        case ScheduleKind::EMPTY:
            schedule.blockDim = 1;
            schedule.blockFactor = 0;
            break;
        case ScheduleKind::LARGE_INDEX_ROW_FILTER:
            schedule.blockDim = ClampBlockDim(LARGE_INDEX_TARGET_CORES, totalBytes, maxCoreNum);
            schedule.blockFactor = CeilDiv(rowNum, static_cast<int64_t>(schedule.blockDim)) * innerNum;
            schedule.tilingKey = (innerBytes % BLOCK_BYTES == 0) ?
                (innerBytes == 2048 ? GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_20) :
                    GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_12)) :
                GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_4);
            break;
        case ScheduleKind::LARGE_WIDE_ROW:
            schedule.blockDim = ClampBlockDim(LARGE_WIDE_TARGET_CORES, totalBytes, maxCoreNum);
            schedule.blockFactor = CeilDiv(rowNum, static_cast<int64_t>(schedule.blockDim)) * innerNum;
            {
                const int64_t maxOwnedRows = CeilDiv(rowNum, static_cast<int64_t>(schedule.blockDim));
                const bool fixedIndex4Stream = indexNum == 4 && innerBytes >= WIDE_STREAM_MIN_BYTES &&
                    innerBytes <= WIDE_STREAM_MAX_BYTES && innerBytes % BLOCK_BYTES == 0 &&
                    maxOwnedRows <= WIDE_STREAM_MAX_BYTES / innerBytes;
                schedule.tilingKey = fixedIndex4Stream ?
                    GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_13) :
                    GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_6);
            }
            break;
        case ScheduleKind::PACKED_SINGLE_ROW:
            schedule.blockDim = std::min<uint32_t>(MODE3_TINY_LAUNCH_CORES, maxCoreNum);
            schedule.blockFactor = totalNum;
            schedule.tilingKey = GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_5);
            break;
        case ScheduleKind::TINY_NARROW_MULTI_ROW:
            schedule.blockDim = std::min<uint32_t>(MODE3_TINY_LAUNCH_CORES, maxCoreNum);
            schedule.blockFactor = totalNum;
            if (totalBytes <= TINY_NARROW_MULTI_ROW_BYTES && indexNum == 2 && typeSize <= 2) {
                schedule.tilingKey = innerBytes == 16 ? GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_16) :
                    (innerBytes == 24 ? GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_17) :
                        GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_8));
            } else if (totalBytes <= TINY_NARROW_MULTI_ROW_BYTES && indexNum == 3) {
                const bool ubCompose = typeSize == 4 && innerBytes >= 3 * typeSize &&
                    innerBytes <= 4 * typeSize && innerBytes % typeSize == 0;
                schedule.tilingKey = ubCompose ?
                    (innerBytes == 3 * typeSize ? GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_14) :
                        GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_15)) :
                    (totalBytes <= 64 ? GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_11) :
                        GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_10));
            } else if (totalBytes > TINY_NARROW_MULTI_ROW_BYTES && indexNum == 3) {
                schedule.tilingKey = GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_9);
            } else {
                schedule.tilingKey = GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_5);
            }
            break;
        case ScheduleKind::TINY_SINGLE_CORE:
            schedule.blockDim = 1;
            schedule.blockFactor = CalcAlignedBlockFactor(totalNum, schedule.blockDim, blockAlign);
            if (totalBytes <= TINY_NARROW_MULTI_ROW_BYTES) {
                schedule.blockDim = std::min<uint32_t>(MODE3_TINY_LAUNCH_CORES, maxCoreNum);
                schedule.blockFactor = totalNum;
                if (indexNum == 1 && rowNum > 1 && typeSize == 4 && innerBytes >= 4 &&
                    innerBytes <= 16 && innerBytes % typeSize == 0) {
                    schedule.tilingKey = innerBytes == 8 ?
                        GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_19) :
                        GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_18);
                } else if (indexNum == 2 && typeSize <= 4) {
                    schedule.tilingKey = innerBytes == 16 ?
                        GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_16) :
                        (innerBytes == 24 ? GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_17) :
                            GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_8));
                } else if (indexNum == 3 && rowNum > 1) {
                    const bool ubCompose = typeSize == 4 && innerBytes >= 3 * typeSize &&
                        innerBytes <= 4 * typeSize && innerBytes % typeSize == 0;
                    schedule.tilingKey = ubCompose ?
                        (innerBytes == 3 * typeSize ? GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_14) :
                            GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_15)) :
                        (totalBytes <= 64 ? GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_11) :
                            GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_10));
                } else {
                    schedule.tilingKey = GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_5);
                }
                break;
            }
            schedule.tilingKey = GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_2);
            break;
        case ScheduleKind::SMALL_MULTI_ROW_SPLIT:
            schedule.blockDim = ClampBlockDim(SMALL_MULTI_ROW_TARGET_CORES, totalBytes, maxCoreNum);
            schedule.blockFactor = CeilDiv(rowNum, static_cast<int64_t>(schedule.blockDim)) * innerNum;
            break;
        case ScheduleKind::SMALL_ONE_WORKER_LAUNCH:
            schedule.blockDim = ClampBlockDim(SMALL_ONE_WORKER_TARGET_CORES, totalBytes, maxCoreNum);
            schedule.blockFactor = totalNum;
            break;
        case ScheduleKind::SINGLE_INDEX_BANDWIDTH:
            schedule.blockDim = SelectBandwidthBlockDim(totalBytes, maxCoreNum);
            schedule.blockFactor = CalcAlignedBlockFactor(totalNum, schedule.blockDim, blockAlign);
            schedule.tilingKey = GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_7);
            break;
        case ScheduleKind::DEFAULT_BANDWIDTH:
        default:
            schedule.blockDim = SelectBandwidthBlockDim(totalBytes, maxCoreNum);
            schedule.blockFactor = CalcAlignedBlockFactor(totalNum, schedule.blockDim, blockAlign);
            break;
    }
    return schedule;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0) {
        coreNum = ascendcPlatform.GetCoreNumAic();
    }
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InplaceUpdateTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    InplaceUpdateTilingData* tiling = context->GetTilingData<InplaceUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape xShape = EnsureNotScalar(xStorageShape->GetStorageShape());
    OP_CHECK_IF(xShape.GetDimNum() > 8, OP_LOGE(context, "x dims must be <= 8"), return ge::GRAPH_FAILED);

    const int64_t totalNum = ShapeElementCount(xShape);
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "invalid x shape"), return ge::GRAPH_FAILED);

    const int64_t rowNum = xShape.GetDimNum() == 0 ? 1 : xShape.GetDim(0);
    OP_CHECK_IF(rowNum < 0, OP_LOGE(context, "invalid row number"), return ge::GRAPH_FAILED);
    const int64_t innerNum = rowNum > 0 ? totalNum / rowNum : 0;

    const gert::StorageShape* iStorageShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, iStorageShape);
    const int64_t indexNum = ShapeElementCount(iStorageShape->GetStorageShape());
    OP_CHECK_IF(indexNum < 0, OP_LOGE(context, "invalid index shape"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const int64_t typeSize = GetTypeSize(inputDesc->GetDataType());
    const int64_t blockAlign = std::max<int64_t>(BLOCK_BYTES / typeSize, 1);
    const int64_t totalBytes = totalNum * typeSize;
    const int64_t innerBytes = innerNum * typeSize;

    TilingSchedule schedule = MakeSchedule(
        totalNum,
        totalBytes,
        rowNum,
        innerNum,
        innerBytes,
        indexNum,
        typeSize,
        blockAlign,
        static_cast<uint32_t>(coreNum));

    int64_t availableUb = static_cast<int64_t>(ubSize);
    if (availableUb > UB_RESERVED_BYTES) {
        availableUb -= UB_RESERVED_BYTES;
    }
    int64_t indexUbCount = std::min<int64_t>(INDEX_UB_COUNT_LIMIT, std::max<int64_t>(indexNum, 8));
    indexUbCount = CeilAlign(indexUbCount, static_cast<int64_t>(8));
    int64_t indexUbBytes = indexUbCount * static_cast<int64_t>(sizeof(int32_t));
    if (indexUbBytes > availableUb / 4) {
        indexUbBytes = std::max<int64_t>(FloorAlign(availableUb / 4, BLOCK_BYTES), BLOCK_BYTES);
        indexUbCount = indexUbBytes / static_cast<int64_t>(sizeof(int32_t));
    }

    const bool useLargeIndexUb =
        schedule.tilingKey == GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_4) ||
        schedule.tilingKey == GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_12) ||
        schedule.tilingKey == GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_20);
    const bool useLargeWideUb =
        schedule.tilingKey == GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_6);
    const int64_t dataUbBytesLimit = useLargeIndexUb ? LARGE_INDEX_DATA_UB_BYTES_LIMIT :
        (useLargeWideUb ? LARGE_WIDE_DATA_UB_BYTES_LIMIT : DATA_UB_BYTES_LIMIT);
    int64_t dataUbBytes = std::min<int64_t>(availableUb - indexUbBytes, dataUbBytesLimit);
    dataUbBytes = FloorAlign(dataUbBytes, BLOCK_BYTES);
    if (dataUbBytes < BLOCK_BYTES) {
        dataUbBytes = BLOCK_BYTES;
    }
    if (schedule.tilingKey == GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_20) && dataUbBytes < innerBytes) {
        schedule.tilingKey = GET_TPL_TILING_KEY(INPLACEUPDATE_TPL_SCH_MODE_12);
    }
    int64_t ubFactor = FloorAlign(dataUbBytes / typeSize, blockAlign);
    if (ubFactor <= 0) {
        ubFactor = blockAlign;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = schedule.blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->rowNum = rowNum;
    tiling->innerNum = innerNum;
    tiling->indexNum = indexNum;
    tiling->indexUbCount = indexUbCount;
    tiling->typeSize = typeSize;

    context->SetBlockDim(schedule.blockDim);
    context->SetTilingKey(schedule.tilingKey);
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForInplaceUpdate([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct InplaceUpdateCompileInfo {};

IMPL_OP_OPTILING(InplaceUpdate).Tiling(InplaceUpdateTilingFunc).TilingParse<InplaceUpdateCompileInfo>(TilingParseForInplaceUpdate);

} // namespace optiling
