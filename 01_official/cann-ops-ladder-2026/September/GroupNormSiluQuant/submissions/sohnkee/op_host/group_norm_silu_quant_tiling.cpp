/*!
 * \file group_norm_silu_quant_tiling.cpp
 * \brief GroupNormSiluQuant tiling for Ascend 910B.
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/group_norm_silu_quant_tiling_data.h"
#include "../op_kernel/group_norm_silu_quant_tiling_key.h"

#include <cstdint>
#include <limits>

namespace optiling {
namespace {

constexpr uint32_t WORKSPACE_SIZE = 0U;
constexpr int64_t MAX_DIM_NC = 4096;
constexpr int64_t MAX_DIM_HW = 1024;
constexpr int64_t START_TILE_ELEMENTS = 4096;
constexpr int64_t MIN_TILE_ELEMENTS = 256;
constexpr int64_t UB_SAFETY_BYTES = 8192;
constexpr int64_t SMALL_LAUNCH_CORES = 8;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t BANK_CONFLICT_PADDING_BYTES = 256;
constexpr int64_t SHORT_RESIDENT_INPUT_PADDING_BYTES = 512;
constexpr int64_t MULTI_TILE_COMPUTE_BUFFER_COUNT = 3;
constexpr int64_t SINGLE_TILE_COMPUTE_BUFFER_COUNT = 4;
constexpr int64_t CLAMP_ELISION_MIN_GROUP_ELEMENTS = 32768;
constexpr int64_t NUMPY_PAIRWISE_TILE_ELEMENTS = 4096;
constexpr int64_t NUMPY_EXTERNAL_BUFFER_ELEMENTS = 8192;
constexpr int64_t FULL_GROUP_RESIDENT_MIN_TILES = 8;
constexpr int64_t FULL_GROUP_SCALAR_BUFFER_BYTES = 256;
constexpr uint64_t EXACT_FP32_RESIDENT_BYTES = 186240;
constexpr int64_t EXACT_FP32_RESIDENT_ELEMENTS = 8192;

inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

inline int64_t AlignBytes(int64_t value)
{
    return CeilDiv(value, BLOCK_BYTES) * BLOCK_BYTES;
}

bool SafeMultiply(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<int64_t>::max() / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
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

int64_t SelectTileElements(uint64_t ubSize, int64_t channelsPerGroup, int64_t groupElements, int64_t typeBytes,
                           int64_t quantScaleCount, int64_t& parameterBytes)
{
    const int64_t gammaBytes = AlignBytes(channelsPerGroup * typeBytes);
    const int64_t betaBytes = AlignBytes(channelsPerGroup * typeBytes);
    const int64_t scaleElements = quantScaleCount == 1 ? 1 : channelsPerGroup;
    const int64_t scaleBytes = AlignBytes(scaleElements * static_cast<int64_t>(sizeof(float)));
    parameterBytes = gammaBytes + betaBytes + scaleBytes;
    const int64_t alignedChannels = CeilDiv(channelsPerGroup, 8) * 8;

    for (int64_t tile = START_TILE_ELEMENTS; tile >= MIN_TILE_ELEMENTS; tile /= 2) {
        const bool enableDoubleBuffer = groupElements > tile;
        const bool useInputDirectly = enableDoubleBuffer &&
            typeBytes == static_cast<int64_t>(sizeof(float));
        const uint64_t computeBufferCount = useInputDirectly ?
            MULTI_TILE_COMPUTE_BUFFER_COUNT : SINGLE_TILE_COMPUTE_BUFFER_COUNT;
        const uint64_t bankPadding = enableDoubleBuffer ?
            static_cast<uint64_t>(MULTI_TILE_COMPUTE_BUFFER_COUNT * BANK_CONFLICT_PADDING_BYTES) : 0U;
        const uint64_t required = static_cast<uint64_t>(2 * tile * typeBytes) +
            static_cast<uint64_t>(2 * tile * sizeof(int8_t)) +
            computeBufferCount * static_cast<uint64_t>(tile * sizeof(float)) +
            static_cast<uint64_t>(tile * sizeof(float)) +
            static_cast<uint64_t>(parameterBytes) +
            static_cast<uint64_t>(2 * alignedChannels * sizeof(float)) + bankPadding + 64U + UB_SAFETY_BYTES;
        if (required <= ubSize) {
            return tile;
        }
    }
    return 0;
}

bool CanUseFullGroupResident(uint64_t ubSize, int64_t channelsPerGroup,
                             int64_t groupElements, int64_t parameterBytes)
{
    const uint64_t tileElements = static_cast<uint64_t>(NUMPY_PAIRWISE_TILE_ELEMENTS);
    const uint64_t residentInputBytes =
        static_cast<uint64_t>(groupElements) * sizeof(float);
    const uint64_t paddedComputeBytes = tileElements * sizeof(float) +
        static_cast<uint64_t>(BANK_CONFLICT_PADDING_BYTES);
    const uint64_t outputQueueBytes = tileElements * sizeof(int8_t);
    const uint64_t boundaryMaskBytes = AlignBytes(
        CeilDiv(NUMPY_PAIRWISE_TILE_ELEMENTS, 8));
    const uint64_t alignedChannels = static_cast<uint64_t>(
        CeilDiv(channelsPerGroup, 8) * 8);
    const uint64_t allocatedBytes = residentInputBytes +
        3U * paddedComputeBytes + outputQueueBytes + boundaryMaskBytes +
        static_cast<uint64_t>(parameterBytes) +
        2U * alignedChannels * sizeof(float) + FULL_GROUP_SCALAR_BUFFER_BYTES;
    return allocatedBytes + static_cast<uint64_t>(UB_SAFETY_BYTES) <= ubSize;
}

bool CanUseShortFullGroupResident(uint64_t ubSize, int64_t channelsPerGroup,
                                  int64_t groupElements, int64_t parameterBytes)
{
    const uint64_t tileElements = static_cast<uint64_t>(NUMPY_PAIRWISE_TILE_ELEMENTS);
    const uint64_t residentInputBytes =
        static_cast<uint64_t>(groupElements) * sizeof(float);
    const uint64_t paddedComputeBytes = tileElements * sizeof(float) +
        static_cast<uint64_t>(BANK_CONFLICT_PADDING_BYTES);
    const uint64_t outputQueueBytes = tileElements * sizeof(int8_t);
    const uint64_t boundaryMaskBytes = AlignBytes(
        CeilDiv(NUMPY_PAIRWISE_TILE_ELEMENTS, 8));
    const uint64_t roundedBytes = tileElements * sizeof(int32_t);
    const uint64_t alignedChannels = static_cast<uint64_t>(
        CeilDiv(channelsPerGroup, 8) * 8);
    const uint64_t allocatedBytes = residentInputBytes +
        static_cast<uint64_t>(SHORT_RESIDENT_INPUT_PADDING_BYTES) +
        3U * paddedComputeBytes + outputQueueBytes + boundaryMaskBytes + roundedBytes +
        static_cast<uint64_t>(parameterBytes) +
        2U * alignedChannels * sizeof(float) + FULL_GROUP_SCALAR_BUFFER_BYTES;
    return allocatedBytes + static_cast<uint64_t>(UB_SAFETY_BYTES) <= ubSize;
}

} // namespace

static ge::graphStatus GroupNormSiluQuantTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* selfStorage = context->GetInputShape(0);
    const gert::StorageShape* gammaStorage = context->GetInputShape(1);
    const gert::StorageShape* betaStorage = context->GetInputShape(2);
    const gert::StorageShape* scaleStorage = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaStorage);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleStorage);

    const gert::Shape& selfShape = selfStorage->GetStorageShape();
    const gert::Shape& gammaShape = gammaStorage->GetStorageShape();
    const gert::Shape& betaShape = betaStorage->GetStorageShape();
    const gert::Shape& scaleShape = scaleStorage->GetStorageShape();
    OP_CHECK_IF(selfShape.GetDimNum() != 4 || !IsPositiveShape(selfShape),
        OP_LOGE(context, "self must be a positive 4D tensor"), return ge::GRAPH_FAILED);

    const int64_t batch = selfShape.GetDim(0);
    const int64_t channels = selfShape.GetDim(1);
    const int64_t height = selfShape.GetDim(2);
    const int64_t width = selfShape.GetDim(3);
    OP_CHECK_IF(batch > MAX_DIM_NC || channels > MAX_DIM_NC || height > MAX_DIM_HW || width > MAX_DIM_HW,
        OP_LOGE(context, "input dimensions exceed the public problem range"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gammaShape.GetDimNum() != 1 || gammaShape.GetDim(0) != channels ||
                    betaShape.GetDimNum() != 1 || betaShape.GetDim(0) != channels,
        OP_LOGE(context, "gamma and beta must have shape [C]"), return ge::GRAPH_FAILED);

    int64_t quantScaleCount = 0;
    if (scaleShape.GetDimNum() == 0) {
        quantScaleCount = 1;
    } else if (scaleShape.GetDimNum() == 1 &&
               (scaleShape.GetDim(0) == 1 || scaleShape.GetDim(0) == channels)) {
        quantScaleCount = scaleShape.GetDim(0);
    } else {
        OP_LOGE(context, "quantScale must be scalar, [1], or [C]");
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* groupAttr = attrs->GetAttrPointer<int64_t>(0);
    const float* epsilonAttr = attrs->GetAttrPointer<float>(1);
    const bool* activateSiluAttr = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, groupAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, epsilonAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, activateSiluAttr);
    const int64_t numGroups = *groupAttr;
    OP_CHECK_IF(numGroups <= 0 || numGroups > channels || channels % numGroups != 0,
        OP_LOGE(context, "group must be positive and divide C"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*epsilonAttr < 0.0f, OP_LOGE(context, "eps must be non-negative"), return ge::GRAPH_FAILED);

    const auto* selfDesc = context->GetInputDesc(0);
    const auto* gammaDesc = context->GetInputDesc(1);
    const auto* betaDesc = context->GetInputDesc(2);
    const auto* scaleDesc = context->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, selfDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, gammaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, betaDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleDesc);
    const ge::DataType selfDtype = selfDesc->GetDataType();
    OP_CHECK_IF(selfDtype != ge::DT_FLOAT16 && selfDtype != ge::DT_FLOAT,
        OP_LOGE(context, "self dtype must be float16 or float32"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(gammaDesc->GetDataType() != selfDtype || betaDesc->GetDataType() != selfDtype ||
                    scaleDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE(context, "gamma/beta dtype must match self and quantScale must be float32"),
        return ge::GRAPH_FAILED);

    int64_t spatial = 0;
    int64_t groupElements = 0;
    int64_t totalGroups = 0;
    OP_CHECK_IF(!SafeMultiply(height, width, spatial), OP_LOGE(context, "H*W overflow"),
        return ge::GRAPH_FAILED);
    const int64_t channelsPerGroup = channels / numGroups;
    OP_CHECK_IF(!SafeMultiply(channelsPerGroup, spatial, groupElements),
        OP_LOGE(context, "group element count overflow"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(!SafeMultiply(batch, numGroups, totalGroups),
        OP_LOGE(context, "group task count overflow"), return ge::GRAPH_FAILED);

    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "failed to query platform resources"), return ge::GRAPH_FAILED);
    const int64_t typeBytes = selfDtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    int64_t parameterBytes = 0;
    const int64_t tileElements = SelectTileElements(
        ubSize, channelsPerGroup, groupElements, typeBytes, quantScaleCount, parameterBytes);
    OP_CHECK_IF(tileElements == 0, OP_LOGE(context, "UB is too small for the frozen design"),
        return ge::GRAPH_FAILED);

    int64_t workerNum = totalGroups < coreNum ? totalGroups : coreNum;
    // Pair resident FP32 groups only when the launch still occupies at least
    // eight vector cores. A worker then reduces and writes two groups with
    // shared vector instructions, without changing either group's arithmetic.
    if (selfDtype == ge::DT_FLOAT && tileElements == NUMPY_PAIRWISE_TILE_ELEMENTS &&
        ubSize >= EXACT_FP32_RESIDENT_BYTES + UB_SAFETY_BYTES &&
        groupElements >= 512 && groupElements <= 2048 &&
        spatial % BLOCK_BYTES == 0 &&
        totalGroups >= 2 * SMALL_LAUNCH_CORES && totalGroups <= coreNum &&
        totalGroups % 2 == 0) {
        workerNum = totalGroups / 2;
    }
    if (groupElements <= NUMPY_EXTERNAL_BUFFER_ELEMENTS && totalGroups > coreNum) {
        // Keep the same minimum number of waves while assigning an equal
        // number of groups to every active core when the division is exact.
        const int64_t waves = CeilDiv(totalGroups, coreNum);
        if (totalGroups % waves == 0) {
            workerNum = totalGroups / waves;
        }
    }
    const int64_t blockDim = totalGroups < SMALL_LAUNCH_CORES && coreNum >= SMALL_LAUNCH_CORES ?
        SMALL_LAUNCH_CORES : workerNum;

    GroupNormSiluQuantTilingData* tiling = context->GetTilingData<GroupNormSiluQuantTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->exactResidentCapacity = selfDtype == ge::DT_FLOAT &&
        tileElements == NUMPY_PAIRWISE_TILE_ELEMENTS &&
        ubSize >= EXACT_FP32_RESIDENT_BYTES + UB_SAFETY_BYTES ?
        EXACT_FP32_RESIDENT_ELEMENTS : 0;
    tiling->batch = batch;
    tiling->channels = channels;
    tiling->height = height;
    tiling->width = width;
    tiling->numGroups = numGroups;
    tiling->channelsPerGroup = channelsPerGroup;
    tiling->spatial = spatial;
    tiling->groupElements = groupElements;
    tiling->totalGroups = totalGroups;
    tiling->tileElements = tileElements;
    tiling->workerNum = workerNum;
    tiling->quantScaleCount = quantScaleCount;
    tiling->parameterBytes = parameterBytes;
    tiling->epsilon = *epsilonAttr;
    tiling->activateSilu = *activateSiluAttr ? 1 : 0;

    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = WORKSPACE_SIZE;
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    const bool useResidentFp32Reuse = selfDtype == ge::DT_FLOAT &&
        groupElements >= 512 && groupElements <= NUMPY_PAIRWISE_TILE_ELEMENTS &&
        groupElements <= tileElements &&
        spatial % (BLOCK_BYTES / static_cast<int64_t>(sizeof(float))) == 0;
    const bool useShortFullGroupResident = selfDtype == ge::DT_FLOAT && *activateSiluAttr &&
        tileElements == NUMPY_PAIRWISE_TILE_ELEMENTS &&
        groupElements == NUMPY_EXTERNAL_BUFFER_ELEMENTS &&
        spatial % (BLOCK_BYTES / static_cast<int64_t>(sizeof(float))) == 0 &&
        spatial <= NUMPY_PAIRWISE_TILE_ELEMENTS / 2 &&
        CanUseShortFullGroupResident(ubSize, channelsPerGroup, groupElements, parameterBytes);
    const bool useFullGroupResident = selfDtype == ge::DT_FLOAT && *activateSiluAttr &&
        tileElements == NUMPY_PAIRWISE_TILE_ELEMENTS &&
        groupElements % NUMPY_EXTERNAL_BUFFER_ELEMENTS == 0 &&
        spatial % NUMPY_PAIRWISE_TILE_ELEMENTS == 0 &&
        groupElements / NUMPY_PAIRWISE_TILE_ELEMENTS >= FULL_GROUP_RESIDENT_MIN_TILES &&
        CanUseFullGroupResident(ubSize, channelsPerGroup, groupElements, parameterBytes);
    const int64_t halfBlockElements = BLOCK_BYTES / static_cast<int64_t>(sizeof(uint16_t));
    const int64_t inputPaddedSpatial = CeilDiv(spatial, halfBlockElements) * halfBlockElements;
    const int64_t outputPaddedSpatial = CeilDiv(spatial, BLOCK_BYTES) * BLOCK_BYTES;
    int64_t inputPaddedGroupElements = 0;
    int64_t outputPaddedGroupElements = 0;
    const bool canUseFp16SingleTileResident = selfDtype == ge::DT_FLOAT16 &&
        channelsPerGroup <= 3 &&
        SafeMultiply(inputPaddedSpatial, channelsPerGroup, inputPaddedGroupElements) &&
        SafeMultiply(outputPaddedSpatial, channelsPerGroup, outputPaddedGroupElements) &&
        inputPaddedGroupElements <= tileElements && outputPaddedGroupElements <= tileElements;
    uint32_t selectedMode = 0;
    if (selfDtype == ge::DT_FLOAT16 && groupElements > tileElements) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_4;
    } else if (canUseFp16SingleTileResident) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_7;
    } else if (selfDtype == ge::DT_FLOAT16 && *activateSiluAttr) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_9;
    } else if (selfDtype == ge::DT_FLOAT16) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_0;
    } else if (useResidentFp32Reuse && *activateSiluAttr &&
        quantScaleCount == 1) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_10;
    } else if (useResidentFp32Reuse) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_5;
    } else if (useShortFullGroupResident && quantScaleCount == 1) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_12;
    } else if (useShortFullGroupResident) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_8;
    } else if (useFullGroupResident &&
        groupElements == 8 * NUMPY_PAIRWISE_TILE_ELEMENTS) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_16;
    } else if (useFullGroupResident) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_6;
    } else if (groupElements >= CLAMP_ELISION_MIN_GROUP_ELEMENTS && *activateSiluAttr) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_3;
    } else if (groupElements > tileElements) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_2;
    } else if (quantScaleCount == 1 && groupElements <= MIN_TILE_ELEMENTS) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_14;
    } else {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_1;
    }
    // Select the same complete resident algorithms previously dispatched
    // inside each kernel. Keys encode reduction size and storage alignment.
    const bool previousResidentMode = selectedMode == 1 || selectedMode == 2 ||
        selectedMode == 5 || selectedMode == 8 || selectedMode == 10 || selectedMode == 12;
    if (previousResidentMode && tiling->exactResidentCapacity != 0 && tileElements == 4096 &&
        groupElements >= 512 && groupElements <= tiling->exactResidentCapacity && spatial % 32 == 0) {
        if ((groupElements & (groupElements - 1)) == 0) {
            uint32_t sizeIndex = 0;
            for (int64_t elements = 512; elements < groupElements; elements *= 2) ++sizeIndex;
            const uint32_t padded = groupElements >= 1024 && spatial % 128 == 0 ? 1U : 0U;
            selectedMode = 32U + 2U * sizeIndex + padded;
        } else {
            selectedMode = 48U;
        }
    }
    // Preserve the existing one-group-per-worker FP16 tiny algorithm gate,
    // while allowing its entry to read only the metadata it consumes.
    if (selectedMode == GROUPNORMSILUQUANT_TPL_SCH_MODE_7 &&
        groupElements <= 128 && channelsPerGroup <= 2 && quantScaleCount == 1 &&
        *activateSiluAttr && workerNum == totalGroups) {
        selectedMode = GROUPNORMSILUQUANT_TPL_SCH_MODE_18;
    }
    context->SetTilingKey(GET_TPL_TILING_KEY(selectedMode));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGroupNormSiluQuant([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct GroupNormSiluQuantCompileInfo {};

IMPL_OP_OPTILING(GroupNormSiluQuant)
    .Tiling(GroupNormSiluQuantTilingFunc)
    .TilingParse<GroupNormSiluQuantCompileInfo>(TilingParseForGroupNormSiluQuant);

} // namespace optiling
