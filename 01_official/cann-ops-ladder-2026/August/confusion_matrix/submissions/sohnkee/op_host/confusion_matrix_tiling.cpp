/*!
 * \file confusion_matrix_tiling.cpp
 * \brief ConfusionMatrix tiling implementation
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/confusion_matrix_tiling_data.h"
#include "../op_kernel/confusion_matrix_tiling_key.h"

namespace optiling {

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr uint64_t SPLIT_OUTPUT_MIN_ELEMS_PER_CORE = 4096;
constexpr uint64_t TINY_FASTPATH_MAX_SAMPLES = 64;
constexpr uint64_t TINY_FASTPATH_MAX_OUTPUT = 256;
constexpr uint32_t SINGLE_WORKER_LAUNCH_CORES = 8;
constexpr uint32_t MAX_SPLIT_OUTPUT_CORES = 48;
constexpr uint32_t MAX_VECTOR_FILTER_SPARSE_CORES = 48;
constexpr uint64_t VECTOR_FILTER_SPARSE_MAX_SAMPLES = 10240;
constexpr uint64_t VECTOR_FILTER_SPARSE_MIN_SAMPLES = 512;
constexpr uint64_t UINT32_OUTPUT_OFFSET_MAX = 0xFFFFFFFFULL;
constexpr uint64_t C3_VECTOR_SINGLE_TILE_MAX_SAMPLES = 3072;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return inShape;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static uint32_t GetVectorCoreNum(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return 1;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    const int64_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const int64_t aicNum = ascendcPlatform.GetCoreNumAic();
    const int64_t coreNum = aivNum > 0 ? aivNum : aicNum;
    if (coreNum <= 0) {
        return 1;
    }
    if (coreNum > static_cast<int64_t>(MAX_SPLIT_OUTPUT_CORES)) {
        return MAX_SPLIT_OUTPUT_CORES;
    }
    return static_cast<uint32_t>(coreNum);
}

static bool ShouldSplitOutput(int64_t totalNum, int64_t outputNum, uint32_t blockDim)
{
    if (totalNum <= 0 || outputNum <= 0 || blockDim <= 1) {
        return false;
    }

    const uint64_t outputCount = static_cast<uint64_t>(outputNum);
    const uint64_t sampleCount = static_cast<uint64_t>(totalNum);
    const uint64_t coreCount = static_cast<uint64_t>(blockDim);
    const uint64_t perCoreOutputCount = (outputCount + coreCount - 1) / coreCount;

    if (perCoreOutputCount < SPLIT_OUTPUT_MIN_ELEMS_PER_CORE) {
        return false;
    }

    return perCoreOutputCount >= sampleCount;
}

static bool ShouldUseVectorFilterSparse(
    int64_t totalNum,
    int64_t outputNum,
    int32_t labelsDtype,
    int32_t predictionsDtype,
    int32_t outputDtype,
    int32_t hasWeights)
{
    if (totalNum < static_cast<int64_t>(VECTOR_FILTER_SPARSE_MIN_SAMPLES) || outputNum <= 0 ||
        totalNum > static_cast<int64_t>(VECTOR_FILTER_SPARSE_MAX_SAMPLES) ||
        static_cast<uint64_t>(outputNum) > UINT32_OUTPUT_OFFSET_MAX) {
        return false;
    }
    if (labelsDtype != CONFUSION_MATRIX_DTYPE_INT32 ||
        predictionsDtype != CONFUSION_MATRIX_DTYPE_INT32 ||
        outputDtype != CONFUSION_MATRIX_DTYPE_INT32 || hasWeights != 0) {
        return false;
    }
    const uint64_t sampleCount = static_cast<uint64_t>(totalNum);
    const uint64_t outputCount = static_cast<uint64_t>(outputNum);
    return outputCount / sampleCount >= sampleCount;
}

static bool IsTinyShape(int64_t totalNum, int64_t outputNum)
{
    return totalNum > 0 &&
        totalNum <= static_cast<int64_t>(TINY_FASTPATH_MAX_SAMPLES) &&
        outputNum > 0 &&
        outputNum <= static_cast<int64_t>(TINY_FASTPATH_MAX_OUTPUT);
}

static int32_t SelectTinyFastPathMode(
    int64_t totalNum,
    int64_t outputNum,
    int32_t labelsDtype,
    int32_t predictionsDtype,
    int32_t weightsDtype,
    int32_t outputDtype,
    int32_t hasWeights)
{
    if (!IsTinyShape(totalNum, outputNum)) {
        return CONFUSIONMATRIX_TPL_SCH_MODE_1;
    }
    if (labelsDtype == CONFUSION_MATRIX_DTYPE_INT32 &&
        predictionsDtype == CONFUSION_MATRIX_DTYPE_INT32 &&
        outputDtype == CONFUSION_MATRIX_DTYPE_INT32) {
        if (hasWeights == 0) {
            return CONFUSIONMATRIX_TPL_SCH_MODE_0;
        }
        if (weightsDtype == CONFUSION_MATRIX_DTYPE_INT32 && totalNum <= 6) {
            return CONFUSIONMATRIX_TPL_SCH_MODE_2;
        }
    }
    if (labelsDtype == CONFUSION_MATRIX_DTYPE_INT64 &&
        predictionsDtype == CONFUSION_MATRIX_DTYPE_INT64 &&
        weightsDtype == CONFUSION_MATRIX_DTYPE_FLOAT32 &&
        outputDtype == CONFUSION_MATRIX_DTYPE_FLOAT32 &&
        hasWeights != 0 && totalNum <= 6) {
        return CONFUSIONMATRIX_TPL_SCH_MODE_3;
    }
    return CONFUSIONMATRIX_TPL_SCH_MODE_1;
}

static uint32_t GetSingleWorkerLaunchCoreNum(uint32_t availableCoreNum)
{
    if (availableCoreNum == 0) {
        return 1;
    }
    if (availableCoreNum > SINGLE_WORKER_LAUNCH_CORES) {
        return SINGLE_WORKER_LAUNCH_CORES;
    }
    return availableCoreNum;
}

static ge::graphStatus ConvertDtype(gert::TilingContext* context, ge::DataType dtype, int32_t& dtypeCode)
{
    if (dtype == ge::DT_INT32) {
        dtypeCode = CONFUSION_MATRIX_DTYPE_INT32;
        return ge::GRAPH_SUCCESS;
    }
    if (dtype == ge::DT_INT64) {
        dtypeCode = CONFUSION_MATRIX_DTYPE_INT64;
        return ge::GRAPH_SUCCESS;
    }
    if (dtype == ge::DT_FLOAT) {
        dtypeCode = CONFUSION_MATRIX_DTYPE_FLOAT32;
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGE(context, "ConfusionMatrix unsupported dtype: %d", static_cast<int32_t>(dtype));
    return ge::GRAPH_FAILED;
}

static ge::graphStatus GetTensorNum(gert::TilingContext* context, const gert::StorageShape* storageShape, int64_t& num)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    const gert::Shape shape = storageShape->GetStorageShape();
    if (shape.GetDimNum() == 0) {
        num = 1;
        return ge::GRAPH_SUCCESS;
    }
    num = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        OP_CHECK_IF(dim <= 0, OP_LOGE(context, "input dims must be positive"), return ge::GRAPH_FAILED);
        num *= dim;
    }
    return ge::GRAPH_SUCCESS;
}

static bool IsOptionalWeightsPresent(gert::TilingContext* context, int64_t totalNum, int64_t& weightsNum)
{
    const auto weightsDesc = context->GetInputDesc(2);
    const gert::StorageShape* weightsStorageShape = context->GetInputShape(2);
    if (weightsDesc == nullptr || weightsStorageShape == nullptr || weightsDesc->GetDataType() == ge::DT_UNDEFINED) {
        return false;
    }
    const gert::Shape weightsShape = weightsStorageShape->GetStorageShape();
    if (weightsShape.GetDimNum() == 0 && totalNum != 1) {
        return false;
    }
    return GetTensorNum(context, weightsStorageShape, weightsNum) == ge::GRAPH_SUCCESS;
}

static int64_t GetNumClassesFromAttr(gert::TilingContext* context)
{
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return -1;
    }
    const int64_t* numClassesAttr = attrs->GetInt(0);
    if (numClassesAttr == nullptr) {
        return -1;
    }
    return *numClassesAttr;
}

static int64_t GetNumClassesFromOutput(gert::TilingContext* context)
{
    const gert::StorageShape* yStorageShape = context->GetOutputShape(0);
    if (yStorageShape == nullptr) {
        return -1;
    }
    const gert::Shape yShape = EnsureNotScalar(yStorageShape->GetStorageShape());
    if (yShape.GetDimNum() != 2) {
        return -1;
    }
    const int64_t rows = yShape.GetDim(0);
    const int64_t cols = yShape.GetDim(1);
    if (rows <= 0 || rows != cols) {
        return -1;
    }
    return rows;
}

static ge::graphStatus ConfusionMatrixTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    OP_CHECK_IF(
        GetTensorNum(context, context->GetInputShape(0), totalNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid labels shape"),
        return ge::GRAPH_FAILED);

    int64_t predictionsNum = 0;
    OP_CHECK_IF(
        GetTensorNum(context, context->GetInputShape(1), predictionsNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid predictions shape"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(predictionsNum != totalNum, OP_LOGE(context, "labels and predictions size mismatch"), return ge::GRAPH_FAILED);

    const auto labelsDesc = context->GetInputDesc(0);
    const auto predictionsDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, labelsDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, predictionsDesc);

    int32_t labelsDtype = CONFUSION_MATRIX_DTYPE_INT32;
    int32_t predictionsDtype = CONFUSION_MATRIX_DTYPE_INT32;
    OP_CHECK_IF(
        ConvertDtype(context, labelsDesc->GetDataType(), labelsDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid labels dtype"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ConvertDtype(context, predictionsDesc->GetDataType(), predictionsDtype) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "invalid predictions dtype"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        labelsDtype == CONFUSION_MATRIX_DTYPE_FLOAT32 || predictionsDtype == CONFUSION_MATRIX_DTYPE_FLOAT32,
        OP_LOGE(context, "labels and predictions must be int32 or int64"),
        return ge::GRAPH_FAILED);

    int32_t hasWeights = 0;
    int32_t weightsDtype = CONFUSION_MATRIX_DTYPE_INT32;
    int64_t weightsNum = 0;
    const auto weightsDesc = context->GetInputDesc(2);
    if (IsOptionalWeightsPresent(context, totalNum, weightsNum)) {
        hasWeights = 1;
        OP_CHECK_IF(
            ConvertDtype(context, weightsDesc->GetDataType(), weightsDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE(context, "invalid weights dtype"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(weightsNum != totalNum, OP_LOGE(context, "weights size mismatch"), return ge::GRAPH_FAILED);
    }

    int64_t numClasses = GetNumClassesFromAttr(context);
    if (numClasses <= 0) {
        numClasses = GetNumClassesFromOutput(context);
    }
    OP_CHECK_IF(numClasses <= 0, OP_LOGE(context, "num_classes must be positive"), return ge::GRAPH_FAILED);

    const gert::StorageShape* yStorageShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorageShape);
    const gert::Shape yShape = EnsureNotScalar(yStorageShape->GetStorageShape());
    OP_CHECK_IF(yShape.GetDimNum() != 2, OP_LOGE(context, "output must be 2-D"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        yShape.GetDim(0) != numClasses || yShape.GetDim(1) != numClasses,
        OP_LOGE(context, "output shape must be [num_classes, num_classes]"),
        return ge::GRAPH_FAILED);

    int32_t outputDtype = CONFUSION_MATRIX_DTYPE_INT32;
    const auto outputDesc = context->GetOutputDesc(0);
    if (outputDesc != nullptr) {
        OP_CHECK_IF(
            ConvertDtype(context, outputDesc->GetDataType(), outputDtype) != ge::GRAPH_SUCCESS,
            OP_LOGE(context, "invalid output dtype"),
            return ge::GRAPH_FAILED);
    }

    ConfusionMatrixTilingData* tiling = context->GetTilingData<ConfusionMatrixTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const int64_t outputNum = numClasses * numClasses;
    const uint32_t splitOutputBlockDim = GetVectorCoreNum(context);
    uint32_t blockDim = GetSingleWorkerLaunchCoreNum(splitOutputBlockDim);
    int64_t blockFactor = totalNum;
    int32_t reserved = 0;
    int32_t schMode = CONFUSIONMATRIX_TPL_SCH_MODE_1;
    if (ShouldSplitOutput(totalNum, outputNum, splitOutputBlockDim)) {
        blockDim = splitOutputBlockDim;
        blockFactor = (numClasses + static_cast<int64_t>(blockDim) - 1) / static_cast<int64_t>(blockDim);
        reserved |= CONFUSION_MATRIX_RESERVED_SPLIT_OUTPUT;
        if (ShouldUseVectorFilterSparse(
                totalNum, outputNum, labelsDtype, predictionsDtype, outputDtype, hasWeights)) {
            if (blockDim > MAX_VECTOR_FILTER_SPARSE_CORES) {
                blockDim = MAX_VECTOR_FILTER_SPARSE_CORES;
                blockFactor =
                    (numClasses + static_cast<int64_t>(blockDim) - 1) / static_cast<int64_t>(blockDim);
            }
            reserved |= CONFUSION_MATRIX_RESERVED_VECTOR_FILTER_SPARSE;
            schMode = CONFUSIONMATRIX_TPL_SCH_MODE_6;
        }
    } else {
        schMode = SelectTinyFastPathMode(
            totalNum, outputNum, labelsDtype, predictionsDtype, weightsDtype, outputDtype, hasWeights);
        if (labelsDtype == CONFUSION_MATRIX_DTYPE_INT32 &&
            predictionsDtype == CONFUSION_MATRIX_DTYPE_INT32 &&
            outputDtype == CONFUSION_MATRIX_DTYPE_INT32 && hasWeights == 0 && numClasses == 3 && totalNum >= 512) {
            schMode = totalNum <= static_cast<int64_t>(C3_VECTOR_SINGLE_TILE_MAX_SAMPLES) &&
                (static_cast<uint64_t>(totalNum) & 63UL) == 56UL ?
                CONFUSIONMATRIX_TPL_SCH_MODE_5 : CONFUSIONMATRIX_TPL_SCH_MODE_4;
            blockFactor = totalNum;
        }
    }
    OP_CHECK_IF(
        context->SetNeedAtomic(schMode == CONFUSIONMATRIX_TPL_SCH_MODE_6) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "failed to configure atomic clean"),
        return ge::GRAPH_FAILED);
    if (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_6) {
        const uint32_t sampleSpan =
            static_cast<uint32_t>((totalNum + static_cast<int64_t>(blockDim) - 1) / static_cast<int64_t>(blockDim));
        blockFactor = static_cast<int64_t>(
            static_cast<uint64_t>(static_cast<uint16_t>(totalNum)) |
            (static_cast<uint64_t>(static_cast<uint16_t>(numClasses)) << 16U) |
            (static_cast<uint64_t>(static_cast<uint16_t>(sampleSpan)) << 32U));
    }
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->numClasses = numClasses;
    tiling->labelsDtype = labelsDtype;
    tiling->predictionsDtype = predictionsDtype;
    tiling->weightsDtype = weightsDtype;
    tiling->outputDtype = outputDtype;
    tiling->hasWeights = hasWeights;
    tiling->reserved = reserved;

    context->SetBlockDim(blockDim);
    if (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_6) {
        context->SetScheduleMode(1);
    }
    if (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_0) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CONFUSIONMATRIX_TPL_SCH_MODE_0));
    } else if (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_2) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CONFUSIONMATRIX_TPL_SCH_MODE_2));
    } else if (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_3) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CONFUSIONMATRIX_TPL_SCH_MODE_3));
    } else if (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_4) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CONFUSIONMATRIX_TPL_SCH_MODE_4));
    } else if (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_5) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CONFUSIONMATRIX_TPL_SCH_MODE_5));
    } else if (schMode == CONFUSIONMATRIX_TPL_SCH_MODE_6) {
        context->SetTilingKey(GET_TPL_TILING_KEY(CONFUSIONMATRIX_TPL_SCH_MODE_6));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(CONFUSIONMATRIX_TPL_SCH_MODE_1));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForConfusionMatrix([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ConfusionMatrixCompileInfo {};

IMPL_OP_OPTILING(ConfusionMatrix).Tiling(ConfusionMatrixTilingFunc).TilingParse<ConfusionMatrixCompileInfo>(TilingParseForConfusionMatrix);

} // namespace optiling
