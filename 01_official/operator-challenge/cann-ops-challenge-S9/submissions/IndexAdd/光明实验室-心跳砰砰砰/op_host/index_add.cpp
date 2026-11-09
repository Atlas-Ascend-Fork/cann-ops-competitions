#include "index_add_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr int64_t kTileAlignElems = 32;
constexpr int64_t kAlignBytes = 32;
constexpr int64_t kUbBytes = 192 * 1024;
constexpr int64_t kUbReserveBytes = 8 * 1024;
constexpr int64_t kCopyBytesPerBlock = 8 * 1024;
constexpr int64_t kAtomicBytesPerBlock = 1024;
constexpr int64_t kOutputOwnedMinBytes = 64 * 1024;
constexpr int64_t kNarrowArenaBytes = 184 * 1024;
constexpr int64_t kNarrowWorkspaceBytes = 2 * 1024 * 1024;
constexpr int64_t kNarrowChunkElems = 4096;
constexpr int64_t kMinParallelRowsPerCore = 16;
constexpr uint32_t kMaxBlockDim = 40;
constexpr uint32_t kTilingKeyAtomic = 1;
constexpr uint32_t kTilingKeyOrderedAtomic = 2;
constexpr uint32_t kTilingKeyOutputOwned = 3;
constexpr uint32_t kTilingKeyOuterOwned = 4;
constexpr uint32_t kTilingKeyHitList = 5;
constexpr uint32_t kTilingKeyNarrowOuter = 6;
constexpr uint32_t kTilingKeyColumnWave = 7;
constexpr uint32_t kTilingKeyBf16TargetOwned = 8;
constexpr uint32_t kTilingKeyReducedAtomic = 9;
constexpr uint32_t kTilingKeySmallInt8 = 10;
int64_t AlignUp(int64_t value, int64_t align)
{
    return ((value + align - 1) / align) * align;
}

int64_t AlignDown(int64_t value, int64_t align)
{
    return (value / align) * align;
}

bool ReadDimAttr(const gert::RuntimeAttrs* attrs, int64_t& dim)
{
    dim = 0;
    if (attrs == nullptr) {
        return true;
    }
    const int64_t* dimPtr = attrs->GetInt(0);
    if (dimPtr != nullptr) {
        dim = *dimPtr;
    }
    return true;
}

int64_t ProductDims(const gert::Shape& shape, size_t begin, size_t end)
{
    int64_t value = 1;
    for (size_t i = begin; i < end; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0 || value > INT64_MAX / dim) {
            return -1;
        }
        value *= dim;
    }
    return value;
}

bool IsSupportedDtype(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return true;
        default:
            return false;
    }
}

int64_t DtypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2;
        case ge::DT_INT8:
            return 1;
        default:
            return -1;
    }
}

int64_t UbBytesPerTileElem(ge::DataType dtype)
{
    return DtypeBytes(dtype);
}

int64_t ComputeTileElems(ge::DataType dtype, int64_t indexRows)
{
    const int64_t bytesPerElem = UbBytesPerTileElem(dtype);
    const int64_t indexAlignElems = kAlignBytes / static_cast<int64_t>(sizeof(int32_t));
    const int64_t indexBytes = AlignUp(indexRows, indexAlignElems) * static_cast<int64_t>(sizeof(int32_t));
    const int64_t tileBudgetBytes = kUbBytes - kUbReserveBytes - indexBytes;
    if (bytesPerElem <= 0 || tileBudgetBytes < kTileAlignElems * bytesPerElem) {
        return -1;
    }
    return AlignDown(tileBudgetBytes / bytesPerElem, kTileAlignElems);
}

int64_t ComputeOutputTileElems(ge::DataType dtype, int64_t dimSize, int64_t indexRows, bool hitList)
{
    const int64_t bytesPerElem =
        dtype == ge::DT_INT8 || dtype == ge::DT_BF16 ? 12 : 3 * DtypeBytes(dtype);
    const int64_t indexBytes = AlignUp(indexRows, kAlignBytes / sizeof(int32_t)) * sizeof(int32_t);
    const int64_t cursorBytes =
        hitList ? AlignUp(dimSize, kAlignBytes / sizeof(int32_t)) * sizeof(int32_t) : 0;
    const int64_t metadataBytes = hitList
                                      ? AlignUp(dimSize + 1 + indexRows, kAlignBytes / sizeof(int32_t)) *
                                            sizeof(int32_t)
                                      : 0;
    const int64_t tileBudgetBytes = kUbBytes - kUbReserveBytes - indexBytes - cursorBytes - metadataBytes;
    if (bytesPerElem <= 0 || tileBudgetBytes < kTileAlignElems * bytesPerElem) {
        return -1;
    }
    return AlignDown(tileBudgetBytes / bytesPerElem, kTileAlignElems);
}

int64_t BalanceTileElems(int64_t maxTileElems, int64_t innerSize)
{
    const int64_t tileCount = (innerSize + maxTileElems - 1) / maxTileElems;
    return std::min(maxTileElems,
                    AlignUp((innerSize + tileCount - 1) / tileCount, kTileAlignElems));
}

bool NeedsOrderedAtomic(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

bool NeedsOutputOwned(ge::DataType dtype)
{
    return IsSupportedDtype(dtype);
}

bool FitsOuterOwned(ge::DataType dtype, int64_t dimSize, int64_t innerSize, int64_t indexRows)
{
    const int64_t dtypeBytes = DtypeBytes(dtype);
    const int64_t paddedInner = AlignUp(innerSize, kAlignBytes / dtypeBytes);
    const int64_t outputElems = dimSize * paddedInner;
    const int64_t sourceElems = indexRows * innerSize;
    const int64_t indexBytes = AlignUp(indexRows * static_cast<int64_t>(sizeof(int32_t)), kAlignBytes);
    int64_t bytes = kUbReserveBytes + indexBytes;
    if (dtype == ge::DT_BF16) {
        bytes += AlignUp(sourceElems * 2, kAlignBytes) + AlignUp(outputElems * 2, kAlignBytes) +
                 AlignUp(outputElems * 4, kAlignBytes) + AlignUp(paddedInner * 4, kAlignBytes) +
                 AlignUp(paddedInner * 2, kAlignBytes) + AlignUp(paddedInner * 4, kAlignBytes);
    } else if (dtype == ge::DT_INT8) {
        bytes += AlignUp(sourceElems, kAlignBytes) + AlignUp(sourceElems * 2, kAlignBytes) +
                 AlignUp(outputElems, kAlignBytes) +
                 AlignUp(outputElems * 4, kAlignBytes) + AlignUp(paddedInner * 4, kAlignBytes) +
                 AlignUp(paddedInner * 2, kAlignBytes) + AlignUp(paddedInner, kAlignBytes) +
                 AlignUp(paddedInner * 4, kAlignBytes);
    } else {
        return false;
    }
    return bytes <= kUbBytes;
}

bool FitsNarrowOuter(ge::DataType dtype, int64_t dimSize, int64_t innerSize, int64_t indexRows)
{
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_INT32 && dtype != ge::DT_FLOAT16) || innerSize > 8) {
        return false;
    }
    const int64_t dtypeBytes = DtypeBytes(dtype);
    const int64_t dataAlignElems = kAlignBytes / dtypeBytes;
    const int64_t intAlignElems = kAlignBytes / static_cast<int64_t>(sizeof(int32_t));
    const int64_t paddedInner = AlignUp(innerSize, dataAlignElems);
    const int64_t metadataElems = dimSize + 1 + indexRows;
    const int64_t buildBytes = (AlignUp(indexRows, intAlignElems) + AlignUp(dimSize, intAlignElems) +
                                AlignUp(metadataElems, intAlignElems)) * sizeof(int32_t);
    const int64_t outputElems = dimSize * innerSize;
    const int64_t paddedOutput = AlignUp(outputElems, dataAlignElems);
    const int64_t denseStreamBytes = AlignUp(indexRows * innerSize, dataAlignElems) * dtypeBytes +
                                     kTileAlignElems * dtypeBytes + paddedOutput * (2 * dtypeBytes + 8);
    int64_t processBytes = denseStreamBytes;
    if (dtypeBytes == static_cast<int64_t>(sizeof(int32_t))) {
        const int64_t sparseElems = AlignUp(metadataElems, intAlignElems) +
                                    AlignUp(indexRows * innerSize, intAlignElems) + dimSize * paddedInner +
                                    2 * kNarrowChunkElems +
                                    AlignUp(innerSize * paddedInner + paddedInner, intAlignElems);
        processBytes = std::min(processBytes, sparseElems * static_cast<int64_t>(sizeof(int32_t)));
    }
    return buildBytes <= kNarrowArenaBytes && processBytes <= kNarrowArenaBytes;
}

bool FitsWideInt32Outer(ge::DataType dtype, int64_t dimSize, int64_t innerSize, int64_t indexRows)
{
    if (dtype != ge::DT_INT32 || innerSize <= 8) {
        return false;
    }
    const int64_t alignElems = kAlignBytes / static_cast<int64_t>(sizeof(int32_t));
    const int64_t paddedInner = AlignUp(innerSize, alignElems);
    const int64_t arenaElems = kNarrowArenaBytes / static_cast<int64_t>(sizeof(int32_t));
    return AlignUp(indexRows, alignElems) + dimSize * paddedInner + kTileAlignElems * paddedInner <= arenaElems;
}

int64_t WideInt32ChunkRows(ge::DataType dtype, int64_t dimSize, int64_t innerSize, int64_t indexRows)
{
    if (dtype != ge::DT_INT32 || innerSize <= 8) {
        return 0;
    }
    const int64_t alignElems = kAlignBytes / static_cast<int64_t>(sizeof(int32_t));
    const int64_t paddedInner = AlignUp(innerSize, alignElems);
    const int64_t arenaElems = kNarrowArenaBytes / static_cast<int64_t>(sizeof(int32_t));
    const int64_t availableRows = (arenaElems - AlignUp(indexRows, alignElems)) / paddedInner;
    if (availableRows <= kTileAlignElems) {
        return 0;
    }
    if (dimSize <= availableRows - kTileAlignElems) {
        return dimSize;
    }
    return availableRows - kTileAlignElems;
}

bool FitsColumnWave(ge::DataType dtype, int64_t dimSize, int64_t innerSize, int64_t indexRows)
{
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_INT32) || innerSize < 3 || innerSize > 8) {
        return false;
    }
    const int64_t alignElems = kAlignBytes / static_cast<int64_t>(sizeof(int32_t));
    const int64_t paddedDim = AlignUp(dimSize, alignElems);
    if (paddedDim / alignElems > UINT8_MAX) {
        return false;
    }
    const int64_t arenaElems = kNarrowArenaBytes / static_cast<int64_t>(sizeof(int32_t));
    const int64_t paddedOutput = AlignUp(dimSize * innerSize, alignElems);
    const int64_t tileRows = AlignDown(
        (arenaElems - 3 * paddedOutput - 2 * alignElems) / innerSize, kTileAlignElems);
    const int64_t sourceElems = AlignUp(tileRows * innerSize + 2 * alignElems, alignElems);
    const int64_t processElems = sourceElems + 3 * paddedOutput;
    const int64_t metadataElems = dimSize + 1 + indexRows;
    const int64_t buildElems = AlignUp(indexRows, alignElems) + AlignUp(dimSize, alignElems) +
                               AlignUp(metadataElems, alignElems);
    return tileRows >= kTileAlignElems && std::max(buildElems, processElems) <= arenaElems;
}

bool FitsBf16TargetOwned(int64_t dimSize, int64_t innerSize, int64_t indexRows,
                         int64_t outerCount, int64_t maxBlocks)
{
    if (outerCount > UINT16_MAX) {
        return false;
    }
    const int64_t blockDim = std::min(dimSize, maxBlocks);
    const int64_t maxTargets = (dimSize + blockDim - 1) / blockDim;
    const int64_t paddedInner = AlignUp(innerSize, kAlignBytes / 2);
    const int64_t stateTileElems = outerCount * paddedInner;
    const int64_t indexBytes = AlignUp(indexRows, 64) * sizeof(int32_t);
    const int64_t requiredBytes = indexBytes + stateTileElems * (2 * maxTargets + 12);
    const int64_t outputStrideBytes = (dimSize * innerSize - innerSize) * 2;
    const int64_t sourceStrideBytes = (indexRows * innerSize - innerSize) * 2;
    return requiredBytes <= kNarrowArenaBytes && outputStrideBytes <= UINT32_MAX &&
           sourceStrideBytes <= UINT32_MAX;
}

int64_t ComputeOrderedTileElems(int64_t maxTileElems, int64_t innerSize, int64_t outerCount)
{
    const int64_t targetTiles = std::max<int64_t>(
        1, std::min<int64_t>((innerSize + kTileAlignElems - 1) / kTileAlignElems,
                             (kMaxBlockDim + std::min<int64_t>(outerCount, kMaxBlockDim) - 1) /
                                 std::min<int64_t>(outerCount, kMaxBlockDim)));
    return std::min(maxTileElems, AlignUp((innerSize + targetTiles - 1) / targetTiles, kTileAlignElems));
}

}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto inputShape = context->GetInputShape(0)->GetStorageShape();
    const auto indexShape = context->GetInputShape(1)->GetStorageShape();
    const auto sourceShape = context->GetInputShape(2)->GetStorageShape();
    const size_t rank = inputShape.GetDimNum();
    if (rank == 0 || indexShape.GetDimNum() != 1 || sourceShape.GetDimNum() != rank) {
        return ge::GRAPH_FAILED;
    }

    int64_t dim = 0;
    if (!ReadDimAttr(context->GetAttrs(), dim)) {
        return ge::GRAPH_FAILED;
    }
    if (dim < 0) {
        dim += static_cast<int64_t>(rank);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(rank)) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType inputDtype = context->GetInputTensor(0)->GetDataType();
    const ge::DataType sourceDtype = context->GetInputTensor(2)->GetDataType();
    if (!IsSupportedDtype(inputDtype) || sourceDtype != inputDtype ||
        context->GetInputTensor(1)->GetDataType() != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }
    for (size_t axis = 0; axis < rank; ++axis) {
        if (inputShape.GetDim(axis) <= 0 || sourceShape.GetDim(axis) <= 0) {
            return ge::GRAPH_FAILED;
        }
        const bool isIndexedAxis = axis == static_cast<size_t>(dim);
        const int64_t expected = isIndexedAxis ? indexShape.GetDim(0) : inputShape.GetDim(axis);
        if (sourceShape.GetDim(axis) != expected) {
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t dimSize = inputShape.GetDim(static_cast<size_t>(dim));
    const int64_t indexRows = indexShape.GetDim(0);
    const int64_t innerSize = ProductDims(inputShape, static_cast<size_t>(dim) + 1, rank);
    const int64_t totalElems = ProductDims(inputShape, 0, rank);
    if (dimSize <= 0 || indexRows <= 0 || innerSize <= 0 || totalElems <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t outerCount = totalElems / (dimSize * innerSize);
    const int64_t dtypeBytes = DtypeBytes(inputDtype);
    const int64_t sourceElems = outerCount * indexRows * innerSize;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t coreNum = std::max(1U, platform.GetCoreNumAiv());
    const uint32_t maxUsableBlocks = std::max(1U, std::min(coreNum, kMaxBlockDim));
    const int64_t smallInt8Scan = AlignUp(indexRows, 64);
    const int64_t smallInt8Source = AlignUp(indexRows * innerSize, 128);
    const int64_t smallInt8Bytes = 12 * smallInt8Scan +
        4 * AlignUp(indexRows * innerSize / 16, 8) + AlignUp(smallInt8Scan / 8, 32) +
        9 * smallInt8Source + 9 * innerSize;
    const bool smallInt8 = inputDtype == ge::DT_INT8 && outerCount == 1 &&
        innerSize % 32 == 0 && dimSize <= maxUsableBlocks && smallInt8Bytes <= 184 * 1024;
    const int64_t reducedRowElems = AlignUp(innerSize, kTileAlignElems);
    const int64_t reducedGatherElems = AlignUp(innerSize + 7, 64);
    const int64_t reducedPackedElems = dimSize * reducedRowElems +
        AlignUp(indexRows, kTileAlignElems) + reducedGatherElems +
        8 * AlignUp(reducedGatherElems / 8, 32) / sizeof(int32_t) +
        AlignUp(4 * innerSize, 64) + 64;
    const bool reducedAtomic = inputDtype == ge::DT_INT32 && innerSize % 8 != 0 &&
        dimSize <= indexRows / 8 && sourceElems * dtypeBytes >= 128 * 1024 * 1024 &&
        reducedPackedElems <= 188 * 1024 / sizeof(int32_t);
    const bool outputOwned = NeedsOutputOwned(inputDtype) && sourceElems * dtypeBytes >= kOutputOwnedMinBytes;
    const bool bf16TargetOwned =
        inputDtype == ge::DT_BF16 &&
        FitsBf16TargetOwned(dimSize, innerSize, indexRows, outerCount, maxUsableBlocks);
    const bool orderedAtomic = !outputOwned && NeedsOrderedAtomic(inputDtype);
    const bool outerOwned = outputOwned && FitsOuterOwned(inputDtype, dimSize, innerSize, indexRows);
    const int64_t parallelRowThreshold =
        static_cast<int64_t>(maxUsableBlocks) * kMinParallelRowsPerCore;
    const bool smallInt32Narrow = inputDtype == ge::DT_INT32 && outerCount < kMaxBlockDim &&
                                  innerSize <= 8 && indexRows >= parallelRowThreshold;
    const int64_t wideInt32ChunkRows =
        outputOwned ? WideInt32ChunkRows(inputDtype, dimSize, innerSize, indexRows) : 0;
    const bool wideInt32Outer = wideInt32ChunkRows > 0;
    const bool narrowOuter = wideInt32Outer ||
                             (((outputOwned && outerCount >= kMaxBlockDim && indexRows >= kTileAlignElems) ||
                               smallInt32Narrow) && dimSize >= kTileAlignElems &&
                              FitsNarrowOuter(inputDtype, dimSize, innerSize, indexRows));
    const bool columnWave = outputOwned && !narrowOuter && outerCount >= kMaxBlockDim &&
                            indexRows >= kTileAlignElems && dimSize >= kTileAlignElems &&
                            FitsColumnWave(inputDtype, dimSize, innerSize, indexRows);
    const int64_t tileElems = ComputeTileElems(inputDtype, indexRows);
    if (tileElems <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t orderedTileElems =
        orderedAtomic ? ComputeOrderedTileElems(tileElems, innerSize, outerCount) : tileElems;
    const int64_t scanOutputTileElems =
        outputOwned ? ComputeOutputTileElems(inputDtype, dimSize, indexRows, false) : tileElems;
    if (scanOutputTileElems <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t scanInnerTileCount = (innerSize + scanOutputTileElems - 1) / scanOutputTileElems;
    const int64_t scanOutputTaskCount = outerCount * dimSize * scanInnerTileCount;
    const int64_t parallelCountStride = AlignUp(dimSize + 1, 64);
    const int64_t parallelScratchElems =
        AlignUp((indexRows + kMaxBlockDim - 1) / kMaxBlockDim, kAlignBytes / sizeof(int32_t)) +
        2 * parallelCountStride;
    const bool parallelInt32HitList = inputDtype == ge::DT_INT32 && indexRows >= parallelRowThreshold &&
                                      scanOutputTaskCount >= kMaxBlockDim &&
                                      parallelScratchElems <=
                                          AlignUp(dimSize + 1 + indexRows, kAlignBytes / sizeof(int32_t));
    const bool hitList = outputOwned && !outerOwned && !narrowOuter && !columnWave &&
                         (scanOutputTaskCount >= static_cast<int64_t>(maxUsableBlocks) * 8 ||
                          parallelInt32HitList);
    const int64_t maxOutputTileElems = hitList
                                           ? ComputeOutputTileElems(inputDtype, dimSize, indexRows, true)
                                           : scanOutputTileElems;
    if (maxOutputTileElems <= 0) {
        return ge::GRAPH_FAILED;
    }
    const int64_t outputTileElems = hitList && parallelInt32HitList
                                        ? BalanceTileElems(maxOutputTileElems, innerSize)
                                        : maxOutputTileElems;

    const int64_t workTileElems =
        outputOwned ? outputTileElems : (orderedAtomic ? orderedTileElems : tileElems);
    const int64_t innerTileCount = (innerSize + workTileElems - 1) / workTileElems;
    const int64_t copyTaskCount = (totalElems + tileElems - 1) / tileElems;
    const int64_t sourceTaskCount = outerCount * indexRows * innerTileCount;
    const int64_t outputTaskCount = outerCount * dimSize * innerTileCount;
    const int64_t copyBlocks = std::max<int64_t>(
        copyTaskCount, (totalElems * dtypeBytes + kCopyBytesPerBlock - 1) / kCopyBytesPerBlock);
    const int64_t atomicWorkBlocks =
        (sourceElems * dtypeBytes + kAtomicBytesPerBlock - 1) / kAtomicBytesPerBlock;
    const int64_t atomicBlocks = outputOwned
                                     ? outputTaskCount
                                     : (orderedAtomic
                                            ? outerCount * innerTileCount
                                            : std::min(sourceTaskCount, std::min(outputTaskCount, atomicWorkBlocks)));
    const int64_t wideInt32Tasks = wideInt32Outer
                                       ? outerCount * ((dimSize + wideInt32ChunkRows - 1) / wideInt32ChunkRows)
                                       : 0;
    const int64_t taskCount = smallInt8 ? dimSize : reducedAtomic ? outerCount * indexRows : bf16TargetOwned
                                  ? dimSize
                                  : (outerOwned || narrowOuter || columnWave)
                                  ? (wideInt32Outer ? wideInt32Tasks
                                                   : (smallInt32Narrow ? kMaxBlockDim : outerCount))
                                  : (outputOwned ? outputTaskCount : std::max(copyBlocks, atomicBlocks));
    const uint32_t blockDim = static_cast<uint32_t>(std::max<int64_t>(
        1, std::min<int64_t>(static_cast<int64_t>(maxUsableBlocks), taskCount)));

    IndexAddTilingData tiling;
    tiling.set_totalElems(totalElems);
    tiling.set_dimSize(dimSize);
    tiling.set_innerSize(innerSize);
    tiling.set_indexRows(indexRows);
    tiling.set_tileElems(tileElems);
    tiling.set_orderedTileElems(orderedTileElems);
    tiling.set_outputTileElems(outputTileElems);

    context->SetBlockDim(blockDim);
    context->SetTilingKey(
        smallInt8 ? kTilingKeySmallInt8 : reducedAtomic ? kTilingKeyReducedAtomic : bf16TargetOwned
            ? kTilingKeyBf16TargetOwned
            : (columnWave ? kTilingKeyColumnWave
                          : (narrowOuter ? kTilingKeyNarrowOuter
                                         : (outerOwned ? kTilingKeyOuterOwned
                                                       : (hitList ? kTilingKeyHitList
                                                                  : (outputOwned ? kTilingKeyOutputOwned
                                                                                 : (orderedAtomic
                                                                                        ? kTilingKeyOrderedAtomic
                                                                                        : kTilingKeyAtomic)))))));
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    if (hitList || narrowOuter || columnWave) {
        auto workspaceSizes = context->GetWorkspaceSizes(1);
        const int64_t hitListCountBytes =
            hitList ? kMaxBlockDim * AlignUp(dimSize + 1, 64) * static_cast<int64_t>(sizeof(int32_t)) : 0;
        workspaceSizes[0] = platform.GetLibApiWorkSpaceSize() +
                            static_cast<size_t>(AlignUp((dimSize + 1 + indexRows) * sizeof(int32_t), 512)) +
                            static_cast<size_t>(hitListCountBytes) +
                            ((narrowOuter || columnWave) ? kNarrowWorkspaceBytes : 0);
    }
    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType inputDtype = context->GetInputDataType(0);
    if (!IsSupportedDtype(inputDtype) || context->GetInputDataType(2) != inputDtype ||
        context->GetInputDataType(1) != ge::DT_INT32) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char* name) : OpDef(name)
    {
        this->Input("input")
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

        this->Attr("dim").AttrType(OPTIONAL).Int(0);

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(IndexAdd);

}
