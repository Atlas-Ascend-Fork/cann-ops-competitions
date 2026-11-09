
#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* selfShape = context->GetInputShape(0);
    const gert::StorageShape* indexShape = context->GetInputShape(1);
    const auto* selfDesc = context->GetInputDesc(0);
    if (selfShape == nullptr || indexShape == nullptr || selfDesc == nullptr ||
        context->GetAttrs() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& shape = selfShape->GetStorageShape();
    const size_t rank = shape.GetDimNum();
    const int64_t* dimAttr = context->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (rank == 0 || dimAttr == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t dim = *dimAttr;
    if (dim < 0) {
        dim += static_cast<int64_t>(rank);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t totalLength = 1;
    uint64_t innerLength = 1;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t extent = shape.GetDim(i);
        if (extent <= 0) {
            return ge::GRAPH_FAILED;
        }
        totalLength *= static_cast<uint64_t>(extent);
        if (i > static_cast<size_t>(dim)) {
            innerLength *= static_cast<uint64_t>(extent);
        }
    }

    const int64_t axisLength = shape.GetDim(static_cast<size_t>(dim));
    const gert::Shape& indexStorageShape = indexShape->GetStorageShape();
    if (axisLength <= 0 || indexStorageShape.GetDimNum() != 1 || indexStorageShape.GetDim(0) <= 0) {
        return ge::GRAPH_FAILED;
    }

    IndexAddTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_innerLength(innerLength);
    tiling.set_axisLength(static_cast<uint32_t>(axisLength));
    tiling.set_indexLength(static_cast<uint32_t>(indexStorageShape.GetDim(0)));

    const uint64_t outerLength = totalLength /
                                 (static_cast<uint64_t>(axisLength) * innerLength);
    const int64_t indexLength = indexStorageShape.GetDim(0);
    const bool useSmallInt8AxisGroups =
        selfDesc->GetDataType() == ge::DT_INT8 && outerLength <= 16 &&
        axisLength <= 64 && innerLength <= 128 && indexLength <= 128;

    auto* platformInfo = context->GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t aivCoreCount = static_cast<uint32_t>(platform.GetCoreNumAiv());
    if (aivCoreCount == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t blockWorkCount =
        useSmallInt8AxisGroups
            ? outerLength *
                  ((static_cast<uint64_t>(axisLength) + 3U) / 4U)
            : totalLength;
    const uint32_t blockDim = static_cast<uint32_t>(
        std::min<uint64_t>(aivCoreCount, blockWorkCount));
    const uint64_t smallInnerWorkCount = outerLength * static_cast<uint64_t>(axisLength);
    const bool packedIndexScale = indexLength >= 512 ||
                                  (indexLength >= 256 && smallInnerWorkCount >= 1024);
    const bool usePackedInt8Indices =
        selfDesc->GetDataType() == ge::DT_INT8 && innerLength <= 128 &&
        packedIndexScale && smallInnerWorkCount >= 4 * blockDim;
    const bool useBalancedSmallEightWay =
        innerLength <= 128 && indexLength >= 512 && axisLength % 8 == 0 &&
        smallInnerWorkCount >= 8 * blockDim &&
        (smallInnerWorkCount % (8 * blockDim) == 0 ||
         smallInnerWorkCount >= 64 * blockDim);
    const uint64_t completeEightWayGroupCount =
        outerLength * (static_cast<uint64_t>(axisLength) / 8);
    const bool useAxisTailSmallEightWay =
        innerLength <= 128 && indexLength >= 512 && axisLength % 8 != 0 &&
        completeEightWayGroupCount >= 8 * blockDim;
    const bool useDeepSmallOneTwentyEightWay =
        innerLength <= 32 && axisLength > 128 && indexLength >= 512 &&
        smallInnerWorkCount >= 256 * blockDim;
    uint32_t deepRowsPerGroup = 1024;
    const uint64_t minimumDeepGroupCount = 2 * static_cast<uint64_t>(blockDim);
    while (deepRowsPerGroup > 128 &&
           outerLength *
                   ((static_cast<uint64_t>(axisLength) + deepRowsPerGroup - 1) /
                    deepRowsPerGroup) <
               minimumDeepGroupCount) {
        deepRowsPerGroup /= 2;
    }
    const uint32_t accumulatorElementBytes =
        selfDesc->GetDataType() == ge::DT_FLOAT ||
                selfDesc->GetDataType() == ge::DT_INT32 ||
                selfDesc->GetDataType() == ge::DT_BF16
            ? 4
            : 2;
    const uint64_t unusedDeepAccumulatorBytes =
        axisLength <= 1024
            ? (1024 - static_cast<uint64_t>(axisLength)) * 32 *
                  accumulatorElementBytes
            : 0;
    const bool useCachedDeepIndices =
        useDeepSmallOneTwentyEightWay && axisLength <= deepRowsPerGroup &&
        outerLength >= 4 * blockDim &&
        static_cast<uint64_t>(indexLength) * sizeof(int32_t) <=
            unusedDeepAccumulatorBytes;
    const bool useDeepScalarFloat =
        useCachedDeepIndices && selfDesc->GetDataType() == ge::DT_FLOAT &&
        innerLength == 1;
    const bool useChunkedOuterBatchFloat =
        selfDesc->GetDataType() == ge::DT_FLOAT && innerLength == 1 &&
        axisLength <= 1024 && indexLength >= 512 && indexLength <= 8000 &&
        outerLength >= 4 * blockDim;
    const bool useThirtyTwoOuterBatchFloat =
        useChunkedOuterBatchFloat && axisLength <= 672;
    const bool useThirtyTwoOuterBatchHalf =
        selfDesc->GetDataType() == ge::DT_FLOAT16 && innerLength == 1 &&
        axisLength <= 672 && indexLength >= 512 && indexLength <= 8000 &&
        outerLength >= 4 * blockDim;
    const bool useBalancedOuterBatchHalf =
        selfDesc->GetDataType() == ge::DT_FLOAT16 && innerLength == 1 &&
        axisLength <= 224 && indexLength >= 512 && indexLength <= 2040 &&
        outerLength >= 4 * blockDim;
    const uint64_t halfAxisTail = axisLength % 128;
    const uint64_t halfIndexTail = indexLength % 128;
    const bool useSingleRepeatHalfTail =
        (halfAxisTail > 0 && halfAxisTail <= 16) ||
        (halfIndexTail > 0 && halfIndexTail <= 16);
    const bool useVnchwOuterBatchHalf =
        useBalancedOuterBatchHalf && axisLength >= 128 &&
        indexLength >= 1024 && outerLength > 127 * blockDim &&
        outerLength <= 192 * blockDim && !useSingleRepeatHalfTail;
    const bool useWholeAxisStagedHalf =
        selfDesc->GetDataType() == ge::DT_FLOAT16 && innerLength == 1 &&
        axisLength <= 128 && indexLength >= 512 && indexLength <= 2040 &&
        outerLength >= 4 * blockDim;
    const bool useBalancedMediumEightWay =
        selfDesc->GetDataType() != ge::DT_BF16 && innerLength > 128 &&
        innerLength <= 256 && indexLength >= 512 && axisLength % 8 == 0 &&
        smallInnerWorkCount >= 8 * blockDim &&
        (smallInnerWorkCount % (8 * blockDim) == 0 ||
         smallInnerWorkCount >= 64 * blockDim);
    const bool useAxisTailMediumEightWay =
        selfDesc->GetDataType() != ge::DT_BF16 && innerLength > 128 &&
        innerLength <= 256 && indexLength >= 512 && axisLength % 8 != 0 &&
        completeEightWayGroupCount >= 8 * blockDim;
    const bool useBalancedWideInner =
        innerLength > 256 && innerLength <= 1024 &&
        smallInnerWorkCount >= 4 * blockDim &&
        (smallInnerWorkCount % (4 * blockDim) == 0 ||
         smallInnerWorkCount >= 32 * blockDim);
    const bool useWholeInt32Row =
        selfDesc->GetDataType() == ge::DT_INT32 && innerLength > 1024 &&
        innerLength <= 8192 && indexLength >= 512 && indexLength <= 8000 &&
        smallInnerWorkCount >= blockDim;
    const uint64_t fiveRowGroupCount =
        outerLength * ((static_cast<uint64_t>(axisLength) + 4) / 5);
    const bool useFiveWholeInt32Rows =
        useWholeInt32Row && innerLength <= 6144 &&
        fiveRowGroupCount >= blockDim;
    const bool useBalancedFourWholeInt32Rows =
        useFiveWholeInt32Rows && axisLength >= 8 && axisLength <= 16 &&
        axisLength % 4 == 0;
    const bool useFlatBalancedWholeInt32Rows =
        useBalancedFourWholeInt32Rows && innerLength <= 4096 &&
        smallInnerWorkCount > 4 * static_cast<uint64_t>(blockDim) &&
        smallInnerWorkCount <= 7 * static_cast<uint64_t>(blockDim);
    const uint64_t int32SourceElementCount =
        outerLength * static_cast<uint64_t>(indexLength) * innerLength;
    const bool useContiguousAtomicInt32 =
        useFlatBalancedWholeInt32Rows &&
        int32SourceElementCount >= 32ULL * 1024ULL * 1024ULL;
    constexpr uint64_t columnOwnedSourceBatchRows = 8U;
    constexpr uint64_t coalescedAtomicSourceChunkRows = 8U;
    constexpr uint64_t coalescedAtomicAuxiliaryBytes = 96U;
    constexpr uint64_t columnOwnedUbBudgetBytes = 188U * 1024U;
    const uint64_t flatColumnCount = outerLength * innerLength;
    const uint64_t maxColumnLength =
        ((flatColumnCount + blockDim - 1U) / blockDim + 7U) & ~7ULL;
    const uint64_t alignedIndexBytes =
        (static_cast<uint64_t>(indexLength) * sizeof(int32_t) + 31U) &
        ~31ULL;
    const uint64_t columnOwnedUbBytes =
        (static_cast<uint64_t>(axisLength) +
         2U * columnOwnedSourceBatchRows) *
            maxColumnLength * sizeof(int32_t) +
        alignedIndexBytes;
    const bool useColumnOwnedInt32 =
        useContiguousAtomicInt32 &&
        columnOwnedUbBytes <= columnOwnedUbBudgetBytes;
    const uint64_t coalescedAtomicBufferElements =
        coalescedAtomicSourceChunkRows * innerLength;
    const uint64_t coalescedAtomicUbBytes =
        2U * coalescedAtomicBufferElements * sizeof(int32_t) +
        alignedIndexBytes + coalescedAtomicAuxiliaryBytes;
    const bool useCoalescedAtomicInt32 =
        useContiguousAtomicInt32 && innerLength > 4U &&
        (innerLength & 7U) == 4U &&
        coalescedAtomicUbBytes <= columnOwnedUbBudgetBytes;
    const uint64_t coalescedAtomicAlignedRowBytes =
        ((innerLength + 7U) & ~7ULL) * sizeof(int32_t);
    const bool usePhaseZeroCachedCoalescedAtomicInt32 =
        useCoalescedAtomicInt32 &&
        coalescedAtomicUbBytes + coalescedAtomicAlignedRowBytes <=
            columnOwnedUbBudgetBytes;
    const bool useWideInt32AxisGroups =
        selfDesc->GetDataType() == ge::DT_INT32 && outerLength <= 4 &&
        axisLength >= 256 && axisLength <= 512 && innerLength > 4096 &&
        innerLength <= 5136 && indexLength >= 1024 && indexLength <= 1216;
    const bool useWideInt32InnerTiles =
        useWideInt32AxisGroups && axisLength <= 320;
    const bool useBfloatAxisGroups =
        selfDesc->GetDataType() == ge::DT_BF16 && outerLength >= 8 &&
        outerLength <= 64 && axisLength > 128 && axisLength <= 256 &&
        innerLength > 64 && innerLength <= 128 && indexLength >= 256 &&
        indexLength <= 512;
    const bool useLargeFloatOuterBatch =
        selfDesc->GetDataType() == ge::DT_FLOAT && outerLength >= 2048 &&
        axisLength >= 512 && axisLength <= 655 && innerLength == 1 &&
        indexLength >= 2048 && indexLength <= 5000;
    constexpr int64_t vnchwFloatColumnChunk = 368;
    const bool useFullGenerationSpecializedLargeFloat =
        useLargeFloatOuterBatch &&
        indexLength >= 4 * vnchwFloatColumnChunk;
    uint64_t tilingKey = 1;
    if (useSmallInt8AxisGroups) {
        tilingKey = 21;
    } else if (useFullGenerationSpecializedLargeFloat) {
        tilingKey = 35;
    } else if (useLargeFloatOuterBatch) {
        tilingKey = 22;
    } else if (useBfloatAxisGroups) {
        tilingKey = 19;
    } else if (useWideInt32InnerTiles) {
        tilingKey = 24;
    } else if (useWideInt32AxisGroups) {
        tilingKey = 18;
    } else if (usePhaseZeroCachedCoalescedAtomicInt32) {
        tilingKey = 33;
    } else if (useCoalescedAtomicInt32) {
        tilingKey = 30;
    } else if (useColumnOwnedInt32) {
        tilingKey = 29;
    } else if (useContiguousAtomicInt32) {
        tilingKey = 17;
    } else if (useFlatBalancedWholeInt32Rows) {
        tilingKey = 26;
    } else if (useBalancedFourWholeInt32Rows) {
        tilingKey = 25;
    } else if (useFiveWholeInt32Rows) {
        tilingKey = 16;
    } else if (useWholeInt32Row) {
        tilingKey = 15;
    } else if (useVnchwOuterBatchHalf) {
        tilingKey = 23;
    } else if (useBalancedOuterBatchHalf) {
        tilingKey = 14;
    } else if (useThirtyTwoOuterBatchFloat || useThirtyTwoOuterBatchHalf) {
        tilingKey = 12;
    } else if (useChunkedOuterBatchFloat) {
        tilingKey = 13;
    } else if (useWholeAxisStagedHalf) {
        tilingKey = 11;
    } else if (useDeepScalarFloat) {
        tilingKey = 10;
    } else if (useCachedDeepIndices) {
        tilingKey = 9;
    } else if (useDeepSmallOneTwentyEightWay) {
        tilingKey = 8;
    } else if (useBalancedSmallEightWay) {
        tilingKey = 4;
    } else if (useAxisTailSmallEightWay) {
        tilingKey = 5;
    } else if (useBalancedMediumEightWay) {
        tilingKey = 6;
    } else if (useAxisTailMediumEightWay) {
        tilingKey = 7;
    } else if (usePackedInt8Indices) {
        tilingKey = 2;
    } else if (useBalancedWideInner) {
        tilingKey = 3;
    }
    if (context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t launchBlockDim =
        useBfloatAxisGroups ? std::min<uint32_t>(blockDim, 24U) : blockDim;
    context->SetBlockDim(launchBlockDim);
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0U;
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("source")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(IndexAdd);
}
