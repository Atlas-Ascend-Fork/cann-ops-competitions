#include "../op_kernel/transpose_tiling.h"

#include <algorithm>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {

constexpr int64_t kScheduleGeneral2D = 0;
constexpr int64_t kScheduleAligned2D = 1;
constexpr int64_t kScheduleCopy2D = 2;
constexpr int64_t kScheduleRank3Perm120 = 12;
constexpr int64_t kScheduleRank4Perm0231 = 13;
constexpr int64_t kScheduleRank4Perm0312 = 14;
constexpr int64_t kScheduleByteTile2D = 15;
constexpr int64_t kScheduleTileGrid2D = 16;
constexpr int64_t kScheduleRank4Perm0231Wide = 17;
constexpr int64_t kScheduleRank4Perm0312Wide = 18;
constexpr int64_t kScheduleTileGrid32x32 = 19;
constexpr int64_t kScheduleTileGrid2DPad = 20;
constexpr int64_t kScheduleGenericNd = 21;
constexpr int64_t kSchedulePlaneNd = 22;
constexpr int64_t kScheduleReverseNd = 23;
constexpr int64_t kSchedulePlaneCompactB16 = 24;
constexpr int64_t kSchedulePlaneB32 = 25;
constexpr int64_t kSchedulePlaneB32Wide = 26;
constexpr int64_t kSchedulePlaneB32Grouped = 27;
constexpr int64_t kSchedulePlaneB32WideTail = 28;
constexpr int64_t kScheduleGenericRowsReordered = 29;
constexpr int64_t kSchedulePlaneB32FullRows = 30;
constexpr int64_t kScheduleRank5GroupedB32 = 31;
constexpr int64_t kAlignedTileRows = 16;
constexpr int64_t kAlignedTileCols = 32;
constexpr int64_t kWideTileRows = 64;
constexpr int64_t kWideTileCols = 32;
constexpr int64_t kRank4Perm0312GroupRows = 256;
constexpr int64_t kGrid32TileRows = 32;
constexpr int64_t kGrid32TileCols = 32;
constexpr int64_t kByteTileRows = 32;
constexpr int64_t kByteTileCols = 64;
constexpr int64_t kMediumPadTileCols = 64;
constexpr int64_t kLargePadTileColsB16 = 128;
constexpr int64_t kLargePadTileColsB32 = 184;
constexpr int64_t kWidePadThreshold = 2048;
constexpr int64_t kLargePadElements = 80 * 1024 * 1024;
constexpr int64_t kGenericTileRows = 32;
constexpr int64_t kGenericNdTileElems = 1024;
constexpr int64_t kGenericReorderedGroup = 20;
constexpr int64_t kGenericReorderedFusedGroup = 10;
constexpr int64_t kGenericReorderedUbBytes = 190 * 1024;
constexpr int64_t kFullRowsUbBytes = 192 * 1024;
constexpr int64_t kPlaneNdTileCols = 512;
constexpr int64_t kPlaneNdTileElemsB16 = 16 * 1024;
constexpr int64_t kPlaneNdTileElemsB32 = 10 * 1024;
constexpr int64_t kPlaneB32WideTileRows = 48;
constexpr int64_t kPlaneB32WideTileCols = 288;
constexpr int64_t kPlaneB32WideTailRows = 48;
constexpr int64_t kPlaneB32WideTailCols = 336;
constexpr int64_t kPlaneB32GroupedTileCols = 264;
constexpr int64_t kPlaneB32GroupedPlanes = 4;
constexpr int64_t kRank5InnerTile = 6;
constexpr int64_t kRank5OuterGroup = 6;
constexpr int64_t kPlaneB32WideTailMinDim = 513;
constexpr int64_t kPlaneB32WideTailMaxDim = 1024;
constexpr int64_t kPlaneB32FullRows = 48;
constexpr int64_t kPlaneB32FullTileElements = 48 * 336 * 3;
constexpr int64_t kCompactFullRows = 64;
constexpr int64_t kCompactTileCols = 80;
constexpr int64_t kCompactInputGuard = 32;
constexpr int64_t kPlaneB32GroupedMaxRows = 192;
constexpr int64_t kPlaneNdTileElemsB8 = 16 * 1024;
constexpr int64_t kPlaneCompactTileElemsB16 = 21 * 1024;
constexpr int64_t kReverseTileElemsB16 = 32 * 1024;
constexpr int64_t kReverseTileElemsB32 = 16 * 1024;
constexpr int64_t kCopyTileBytes = 32 * 1024;
constexpr int64_t kGenericNdInputBytes = 32 * 1024;
constexpr int64_t kGenericUbBudgetBytes = 48 * 1024;
constexpr int64_t kMaxCoreNum = 40;

int64_t PadTileRows(int64_t elementBytes)
{
    return kWideTileRows;
}

int64_t PadTileCols(int64_t cols, int64_t totalLength, int64_t elementBytes)
{
    if (cols < kWidePadThreshold) {
        return kAlignedTileCols;
    }
    if (totalLength < kLargePadElements) {
        return kMediumPadTileCols;
    }
    return elementBytes == 2 ? kLargePadTileColsB16 : kLargePadTileColsB32;
}

int64_t PlaneTileCols(int64_t elementBytes, int64_t contiguousDim)
{
    if (elementBytes == 4 && contiguousDim > kPlaneNdTileCols) {
        return contiguousDim >= kPlaneB32WideTailMinDim && contiguousDim <= kPlaneB32WideTailMaxDim ?
            kPlaneB32WideTailCols : kPlaneB32WideTileCols;
    }
    return std::min(contiguousDim, kPlaneNdTileCols);
}

int64_t ReorderedRowsUbBytes(int64_t rows, int64_t cols, int64_t elementBytes, int64_t groupSize)
{
    const int64_t inputRowBytes = (groupSize * cols * elementBytes + 31) / 32 * 32;
    const int64_t outputGroupBytes = (rows * cols * elementBytes + 31) / 32 * 32;
    const int64_t indexBytes = groupSize == kGenericReorderedFusedGroup ?
        groupSize * outputGroupBytes / elementBytes * static_cast<int64_t>(sizeof(uint32_t)) :
        rows * cols * static_cast<int64_t>(sizeof(uint32_t));
    const int64_t outputBuffers = groupSize == kGenericReorderedFusedGroup ? 2 : 1;
    return 2 * rows * inputRowBytes + outputBuffers * groupSize * outputGroupBytes + indexBytes;
}

int64_t ReorderedRowsGroup(int64_t rows, int64_t cols, int64_t elementBytes)
{
    return ReorderedRowsUbBytes(rows, cols, elementBytes, kGenericReorderedGroup) <=
        kGenericReorderedUbBytes ? kGenericReorderedGroup : kGenericReorderedFusedGroup;
}

bool CanUseCompactFullRows(int64_t transposedDim, int64_t contiguousDim, int64_t elementBytes)
{
    const int64_t ubBytes =
        (kCompactFullRows * contiguousDim + kCompactInputGuard) * elementBytes +
        2 * kCompactFullRows * kCompactTileCols * elementBytes +
        kCompactFullRows * kCompactTileCols * static_cast<int64_t>(sizeof(uint32_t));
    return elementBytes == 4 && transposedDim >= kCompactFullRows && ubBytes <= kFullRowsUbBytes;
}

int64_t BytesPerElement(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        case ge::DT_INT8:
            return 1;
        default:
            return 0;
    }
}

bool IsSupportedType(ge::DataType dtype)
{
    return BytesPerElement(dtype) > 0;
}

int64_t TilingKeyForType(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
            return 0;
        case ge::DT_FLOAT:
            return 1;
        case ge::DT_INT32:
            return 2;
        case ge::DT_INT8:
            return 3;
        default:
            return -1;
    }
}

bool ReadDims(const gert::RuntimeAttrs* attrs, int64_t rank, int64_t* dims)
{
    if (attrs == nullptr || rank <= 0) {
        return false;
    }
    const auto* dimsAttr = attrs->GetListInt(0);
    if (dimsAttr == nullptr || static_cast<int64_t>(dimsAttr->GetSize()) != rank) {
        return false;
    }
    bool seen[8] = {false, false, false, false, false, false, false, false};
    for (int64_t i = 0; i < rank; ++i) {
        int64_t dim = dimsAttr->GetData()[i];
        if (dim < 0) {
            dim += rank;
        }
        if (dim < 0 || dim >= rank || seen[dim]) {
            return false;
        }
        seen[dim] = true;
        dims[i] = dim;
    }
    return true;
}

int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return (value + divisor - 1) / divisor;
}

int64_t CyclicSplit(const int64_t dims[8], int64_t rank)
{
    for (int64_t split = 1; split < rank; ++split) {
        bool matches = true;
        for (int64_t axis = 0; axis < rank; ++axis) {
            matches = matches && dims[axis] == (axis + split) % rank;
        }
        if (matches) {
            return split;
        }
    }
    return -1;
}

bool IsFullReverse(const int64_t dims[8], int64_t rank)
{
    for (int64_t axis = 0; axis < rank; ++axis) {
        if (dims[axis] != rank - 1 - axis) {
            return false;
        }
    }
    return true;
}

int64_t ReverseTileB(int64_t elementBytes)
{
    return elementBytes == 2 ? 16 : 8;
}

int64_t ReverseTileC(int64_t elementBytes, int64_t dimA, int64_t dimC, int64_t tileB)
{
    const int64_t tileElements = elementBytes == 4 ? kReverseTileElemsB32 : kReverseTileElemsB16;
    const int64_t alignment = 32 / elementBytes;
    const int64_t maxCols = tileElements / (dimA * tileB);
    return std::min(dimC, std::max(alignment, maxCols / alignment * alignment));
}

bool IsLayoutPreservingPermutation(const int64_t inputDims[8], const int64_t dims[8], int64_t rank)
{
    int64_t nextNonSingletonAxis = 0;
    for (int64_t i = 0; i < rank; ++i) {
        if (inputDims[dims[i]] == 1) {
            continue;
        }
        while (nextNonSingletonAxis < rank && inputDims[nextNonSingletonAxis] == 1) {
            ++nextNonSingletonAxis;
        }
        if (nextNonSingletonAxis >= rank || dims[i] != nextNonSingletonAxis) {
            return false;
        }
        ++nextNonSingletonAxis;
    }
    return true;
}

bool CanUseTiled2D(int64_t elementBytes, int64_t rows, int64_t cols)
{
    if (elementBytes < 2 || rows % kAlignedTileRows != 0 || cols % kAlignedTileCols != 0) {
        return false;
    }
    if ((kAlignedTileRows * elementBytes) % 32 != 0 || (kAlignedTileCols * elementBytes) % 32 != 0) {
        return false;
    }
    const int64_t inputStrideBlocks = (cols - kAlignedTileCols) * elementBytes / 32;
    const int64_t outputStrideBlocks = (rows - kAlignedTileRows) * elementBytes / 32;
    return inputStrideBlocks >= 0 && inputStrideBlocks <= UINT16_MAX &&
           outputStrideBlocks >= 0 && outputStrideBlocks <= UINT16_MAX;
}

bool ShouldUseTileGrid2D(int64_t rows, int64_t cols)
{
    const int64_t rowTiles = CeilDiv(rows, kAlignedTileRows);
    const int64_t colTiles = CeilDiv(cols, kAlignedTileCols);
    const int64_t rowTasksPerCore = CeilDiv(rowTiles, kMaxCoreNum);
    const int64_t activeRowCores = CeilDiv(rowTiles, rowTasksPerCore);
    return activeRowCores < kMaxCoreNum && rowTiles * colTiles >= kMaxCoreNum;
}

bool ShouldUseTileGrid32x32(int64_t elementBytes, int64_t rows, int64_t cols)
{
    if (elementBytes != 2 || rows % kGrid32TileRows != 0 || cols % kGrid32TileCols != 0) {
        return false;
    }
    const int64_t inputStrideBlocks = (cols - kGrid32TileCols) * elementBytes / 32;
    const int64_t outputStrideBlocks = (rows - kGrid32TileRows) * elementBytes / 32;
    return rows * cols >= 512 * 1024 &&
           inputStrideBlocks >= 0 && inputStrideBlocks <= UINT16_MAX &&
           outputStrideBlocks >= 0 && outputStrideBlocks <= UINT16_MAX;
}

bool ShouldUsePadTile2D(int64_t elementBytes, int64_t rows, int64_t cols, int64_t totalLength)
{
    return elementBytes >= 2 && rows >= kAlignedTileRows && cols >= kAlignedTileCols &&
           totalLength >= 64 * 1024;
}

bool IsPerm(const int64_t dims[8], int64_t rank, int64_t p0, int64_t p1, int64_t p2, int64_t p3 = -1)
{
    if (rank == 3) {
        return dims[0] == p0 && dims[1] == p1 && dims[2] == p2;
    }
    return rank == 4 && dims[0] == p0 && dims[1] == p1 && dims[2] == p2 && dims[3] == p3;
}

bool CanUseRank3Perm120(int64_t elementBytes, const int64_t inputDims[8], const int64_t dims[8], int64_t rank)
{
    if (elementBytes < 2 || !IsPerm(dims, rank, 1, 2, 0)) {
        return false;
    }
    if (inputDims[0] % kAlignedTileRows != 0 || inputDims[2] % kAlignedTileCols != 0) {
        return false;
    }
    const int64_t inputRowStride = inputDims[1] * inputDims[2];
    const int64_t srcStrideBlocks = (inputRowStride - kAlignedTileCols) * elementBytes / 32;
    const int64_t dstStrideBlocks = (inputDims[0] - kAlignedTileRows) * elementBytes / 32;
    return (kAlignedTileRows * elementBytes) % 32 == 0 &&
           (kAlignedTileCols * elementBytes) % 32 == 0 &&
           srcStrideBlocks >= 0 && srcStrideBlocks <= UINT16_MAX &&
           dstStrideBlocks >= 0 && dstStrideBlocks <= UINT16_MAX;
}

bool CanUseRank4Perm0231(int64_t elementBytes, const int64_t inputDims[8], const int64_t dims[8], int64_t rank)
{
    if (elementBytes < 2 || !IsPerm(dims, rank, 0, 2, 3, 1)) {
        return false;
    }
    if (inputDims[1] % kAlignedTileRows != 0 || inputDims[3] % kAlignedTileCols != 0) {
        return false;
    }
    const int64_t inputChannelStride = inputDims[2] * inputDims[3];
    const int64_t srcStrideBlocks = (inputChannelStride - kAlignedTileCols) * elementBytes / 32;
    const int64_t dstStrideBlocks = (inputDims[1] - kAlignedTileRows) * elementBytes / 32;
    return (kAlignedTileRows * elementBytes) % 32 == 0 &&
           (kAlignedTileCols * elementBytes) % 32 == 0 &&
           srcStrideBlocks >= 0 && srcStrideBlocks <= UINT16_MAX &&
           dstStrideBlocks >= 0 && dstStrideBlocks <= UINT16_MAX;
}

bool CanUseRank4Perm0312(int64_t elementBytes, const int64_t inputDims[8], const int64_t dims[8], int64_t rank)
{
    if (elementBytes < 2 || !IsPerm(dims, rank, 0, 3, 1, 2)) {
        return false;
    }
    if (inputDims[2] % kAlignedTileRows != 0 || inputDims[3] % kAlignedTileCols != 0) {
        return false;
    }
    const int64_t srcStrideBlocks = (inputDims[3] - kAlignedTileCols) * elementBytes / 32;
    const int64_t dstStrideBlocks = (inputDims[1] * inputDims[2] - kAlignedTileRows) * elementBytes / 32;
    return (kAlignedTileRows * elementBytes) % 32 == 0 &&
           (kAlignedTileCols * elementBytes) % 32 == 0 &&
           srcStrideBlocks >= 0 && srcStrideBlocks <= UINT16_MAX &&
           dstStrideBlocks >= 0 && dstStrideBlocks <= UINT16_MAX;
}

bool CanUseRank4Perm0231Wide(int64_t elementBytes, const int64_t inputDims[8], const int64_t dims[8], int64_t rank)
{
    if (elementBytes != 2 || !IsPerm(dims, rank, 0, 2, 3, 1)) {
        return false;
    }
    if (inputDims[1] % kWideTileRows != 0 || inputDims[3] % kWideTileCols != 0) {
        return false;
    }
    const int64_t inputChannelStride = inputDims[2] * inputDims[3];
    const int64_t srcStrideBlocks = (inputChannelStride - kWideTileCols) * elementBytes / 32;
    const int64_t dstStrideBlocks = (inputDims[1] - kWideTileRows) * elementBytes / 32;
    return srcStrideBlocks >= 0 && srcStrideBlocks <= UINT16_MAX &&
           dstStrideBlocks >= 0 && dstStrideBlocks <= UINT16_MAX;
}

bool CanUseRank4Perm0312Wide(int64_t elementBytes, const int64_t inputDims[8], const int64_t dims[8], int64_t rank)
{
    if (elementBytes != 2 || !IsPerm(dims, rank, 0, 3, 1, 2)) {
        return false;
    }
    if ((inputDims[1] * inputDims[2] * elementBytes) % 32 != 0 ||
        inputDims[3] % kWideTileCols != 0) {
        return false;
    }
    const int64_t outputStride = inputDims[1] * inputDims[2];
    const int64_t dstStrideBlocks = outputStride * elementBytes / 32;
    return dstStrideBlocks >= 0 && dstStrideBlocks <= UINT16_MAX;
}

void FillDimTiling(optiling::TransposeTilingData& tiling, const int64_t inputDims[8], const int64_t dims[8])
{
    tiling.set_in0(inputDims[0]);
    tiling.set_in1(inputDims[1]);
    tiling.set_in2(inputDims[2]);
    tiling.set_in3(inputDims[3]);
    tiling.set_in4(inputDims[4]);
    tiling.set_in5(inputDims[5]);
    tiling.set_in6(inputDims[6]);
    tiling.set_in7(inputDims[7]);
    tiling.set_perm0(dims[0]);
    tiling.set_perm1(dims[1]);
    tiling.set_perm2(dims[2]);
    tiling.set_perm3(dims[3]);
    tiling.set_perm4(dims[4]);
    tiling.set_perm5(dims[5]);
    tiling.set_perm6(dims[6]);
    tiling.set_perm7(dims[7]);
}

}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* inputShapeHolder = context->GetInputShape(0);
    const auto* inputTensor = context->GetInputTensor(0);
    if (inputShapeHolder == nullptr || inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape inputShape = inputShapeHolder->GetStorageShape();
    const int64_t rank = inputShape.GetDimNum();
    if (rank <= 0 || rank > 8) {
        return ge::GRAPH_FAILED;
    }
    int64_t dims[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    if (!ReadDims(context->GetAttrs(), rank, dims)) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtype = inputTensor->GetDataType();
    const int64_t elementBytes = BytesPerElement(dtype);
    int64_t tilingKey = TilingKeyForType(dtype);
    if (!IsSupportedType(dtype) || tilingKey < 0) {
        return ge::GRAPH_FAILED;
    }
    int64_t inputDims[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t totalLength = 1;
    for (int64_t i = 0; i < rank; ++i) {
        inputDims[i] = inputShape.GetDim(i);
        if (inputDims[i] <= 0) {
            return ge::GRAPH_FAILED;
        }
        totalLength *= inputDims[i];
    }
    const int64_t cyclicSplit = CyclicSplit(dims, rank);
    int64_t rows = inputDims[0];
    if (cyclicSplit > 1) {
        for (int64_t axis = 1; axis < cyclicSplit; ++axis) {
            rows *= inputDims[axis];
        }
    }
    const int64_t cols = cyclicSplit > 0 ? totalLength / rows : rank >= 2 ? inputDims[1] : 1;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int64_t coreNum = std::max(1U, platform.GetCoreNumAiv());
    coreNum = std::min<int64_t>(coreNum, kMaxCoreNum);
    const bool isPhysicalCopy = IsLayoutPreservingPermutation(inputDims, dims, rank);
    const bool isTranspose2D = cyclicSplit > 0;
    const bool tiled2D = isTranspose2D && CanUseTiled2D(elementBytes, rows, cols);
    int64_t schedule = kScheduleGeneral2D;
    int64_t blockRows = kGenericTileRows;
    int64_t reverseTileB = 0;
    int64_t taskItems = CeilDiv(rows, blockRows);
    if (tiled2D) {
        if (elementBytes == 2 && rows >= 32 && rows <= 256 && cols <= 1024) {
            schedule = kScheduleAligned2D;
            blockRows = rows;
            taskItems = CeilDiv(cols, 32);
        } else if (ShouldUseTileGrid32x32(elementBytes, rows, cols)) {
            schedule = kScheduleTileGrid32x32;
            blockRows = kGrid32TileRows;
            taskItems = CeilDiv(rows, kGrid32TileRows) * CeilDiv(cols, kGrid32TileCols);
        } else {
            schedule = ShouldUseTileGrid2D(rows, cols) ? kScheduleTileGrid2D : kScheduleAligned2D;
            blockRows = kAlignedTileRows;
            taskItems = schedule == kScheduleTileGrid2D ? CeilDiv(rows, kAlignedTileRows) * CeilDiv(cols, kAlignedTileCols)
                                                        : CeilDiv(rows, blockRows);
        }
    } else if (isPhysicalCopy) {
        schedule = kScheduleCopy2D;
        blockRows = kCopyTileBytes / elementBytes;
        taskItems = CeilDiv(totalLength, blockRows);
    } else if (isTranspose2D && elementBytes == 1) {
        schedule = kScheduleByteTile2D;
        blockRows = kByteTileRows;
        taskItems = CeilDiv(rows, kByteTileRows) * CeilDiv(cols, kByteTileCols);
    } else if (isTranspose2D && ShouldUsePadTile2D(elementBytes, rows, cols, totalLength)) {
        schedule = kScheduleTileGrid2DPad;
        blockRows = PadTileRows(elementBytes);
        taskItems = CeilDiv(rows, blockRows) * CeilDiv(cols, PadTileCols(cols, totalLength, elementBytes));
    } else if (isTranspose2D && cols * elementBytes > kGenericUbBudgetBytes) {
        schedule = kScheduleGenericNd;
        blockRows = kGenericNdTileElems;
        const int64_t outputLast = inputDims[dims[rank - 1]];
        const int64_t tileRows = outputLast <= blockRows ? blockRows / outputLast : 1;
        taskItems = CeilDiv(totalLength / outputLast, tileRows) * CeilDiv(outputLast, blockRows);
    } else if (isTranspose2D) {
        const int64_t maxRowsByUb = std::max<int64_t>(1, kGenericUbBudgetBytes / (cols * elementBytes));
        blockRows = std::min<int64_t>(kGenericTileRows, maxRowsByUb);
        taskItems = CeilDiv(rows, blockRows);
    } else {
        if (elementBytes >= 2 && rank >= 3 && inputDims[0] <= 64 && IsFullReverse(dims, rank)) {
            schedule = kScheduleReverseNd;
            reverseTileB = ReverseTileB(elementBytes);
            blockRows = ReverseTileC(elementBytes, inputDims[0], inputDims[rank - 1], reverseTileB);
            const int64_t planes = totalLength / inputDims[0] / inputDims[1] / inputDims[rank - 1];
            taskItems = planes * CeilDiv(inputDims[1], reverseTileB) *
                        CeilDiv(inputDims[rank - 1], blockRows);
        } else if (CanUseRank3Perm120(elementBytes, inputDims, dims, rank)) {
            schedule = kScheduleRank3Perm120;
            blockRows = elementBytes == 2 && inputDims[0] % kWideTileRows == 0 ? kWideTileRows : kAlignedTileRows;
            taskItems = inputDims[1] * CeilDiv(inputDims[0], blockRows) *
                        CeilDiv(inputDims[2], kAlignedTileCols);
        } else if (CanUseRank4Perm0231Wide(elementBytes, inputDims, dims, rank)) {
            schedule = kScheduleRank4Perm0231Wide;
            taskItems = inputDims[0] * inputDims[2] * (inputDims[1] / kWideTileRows) *
                        (inputDims[3] / kWideTileCols);
        } else if (CanUseRank4Perm0231(elementBytes, inputDims, dims, rank)) {
            schedule = kScheduleRank4Perm0231;
            taskItems = inputDims[0] * inputDims[2] * CeilDiv(inputDims[1], kAlignedTileRows) *
                        CeilDiv(inputDims[3], kAlignedTileCols);
        } else if (CanUseRank4Perm0312Wide(elementBytes, inputDims, dims, rank)) {
            schedule = kScheduleRank4Perm0312Wide;
            blockRows = kRank4Perm0312GroupRows;
            taskItems = inputDims[0] * CeilDiv(inputDims[1] * inputDims[2], blockRows) *
                        (inputDims[3] / kWideTileCols);
        } else if (CanUseRank4Perm0312(elementBytes, inputDims, dims, rank)) {
            schedule = kScheduleRank4Perm0312;
            taskItems = inputDims[0] * inputDims[1] * CeilDiv(inputDims[2], kAlignedTileRows) *
                        CeilDiv(inputDims[3], kAlignedTileCols);
        } else if (dims[rank - 1] != rank - 1) {
            const int64_t transposedDim = inputDims[dims[rank - 1]];
            const int64_t tileCols = PlaneTileCols(elementBytes, inputDims[rank - 1]);
            const bool compactInputRows = dims[rank - 1] == rank - 2 && inputDims[rank - 1] <= tileCols;
            const bool groupedOffsets = elementBytes == 2 && compactInputRows && tileCols <= 32;
            const int64_t planes = totalLength / transposedDim / inputDims[rank - 1];
            const int64_t innerPlanes = inputDims[dims[rank - 2]];
            const int64_t groupPlanes = std::min(innerPlanes, kPlaneB32GroupedPlanes);
            const int64_t groupedRows = groupPlanes * transposedDim;
            const bool groupedB32 = elementBytes == 4 && !compactInputRows && dims[rank - 2] != rank - 1 &&
                innerPlanes > 1 && groupedRows >= 32 &&
                groupedRows <= kPlaneB32GroupedMaxRows && planes % innerPlanes == 0 &&
                2 * CeilDiv(groupedRows, 16) * 16 * kPlaneB32GroupedTileCols * elementBytes <=
                    kGenericReorderedUbBytes;
            const bool fullRowsB32 = elementBytes == 4 && rank == 3 && planes >= 96 &&
                IsPerm(dims, rank, 0, 2, 1) &&
                inputDims[rank - 1] >= kPlaneB32WideTailMinDim &&
                inputDims[rank - 1] <= kPlaneB32WideTailMaxDim &&

                kPlaneB32FullRows * (CeilDiv(inputDims[rank - 1], 8) * 8 + 2 * 8) <=
                    kPlaneB32FullTileElements;
            const int64_t rank5Rows = CeilDiv(inputDims[2], 16) * 16;
            const int64_t rank5Cols = CeilDiv(kRank5InnerTile * inputDims[4], 8) * 8;
            const int64_t rank5OutputRows = CeilDiv(kRank5OuterGroup * inputDims[2], 16) * 16;
            const bool rank5GroupedB32 = elementBytes == 4 && rank == 5 &&
                dims[0] == 3 && dims[1] == 4 && dims[2] == 1 && dims[3] == 0 && dims[4] == 2 &&
                inputDims[2] <= 32 &&
                2 * rank5OutputRows * rank5Cols * elementBytes <= kGenericReorderedUbBytes;
            if (rank5GroupedB32) {
                schedule = kScheduleRank5GroupedB32;
                blockRows = rank5Rows;
                taskItems = inputDims[1] * CeilDiv(inputDims[0], kRank5OuterGroup) *
                            CeilDiv(inputDims[3], kRank5InnerTile);
            } else if (groupedB32) {
                schedule = kSchedulePlaneB32Grouped;
                blockRows = groupPlanes * transposedDim;
                taskItems = planes / innerPlanes * CeilDiv(innerPlanes, groupPlanes) *
                            CeilDiv(inputDims[rank - 1], kPlaneB32GroupedTileCols);
            } else if (fullRowsB32) {
                schedule = kSchedulePlaneB32FullRows;
                blockRows = CanUseCompactFullRows(transposedDim, inputDims[rank - 1], elementBytes) ?
                    kCompactFullRows : kPlaneB32FullRows;
                taskItems = planes * CeilDiv(transposedDim, blockRows);
            } else {
                schedule = groupedOffsets ? kSchedulePlaneCompactB16 :
                           elementBytes == 4 && !compactInputRows && transposedDim >= 16 ?
                               (inputDims[rank - 1] >= kPlaneB32WideTailMinDim &&
                                inputDims[rank - 1] <= kPlaneB32WideTailMaxDim ?
                                   kSchedulePlaneB32WideTail :
                                inputDims[rank - 1] > kPlaneNdTileCols ?
                                   kSchedulePlaneB32Wide : kSchedulePlaneB32) : kSchedulePlaneNd;
                const int64_t localTileCols = compactInputRows ? tileCols :
                    CeilDiv(tileCols * elementBytes, 32) * 32 / elementBytes;
                const int64_t tileElements = groupedOffsets ? kPlaneCompactTileElemsB16 :
                    schedule == kSchedulePlaneB32WideTail ? kPlaneB32WideTailRows * kPlaneB32WideTailCols :
                    schedule == kSchedulePlaneB32Wide ? kPlaneB32WideTileRows * kPlaneB32WideTileCols :
                    elementBytes == 4 ? kPlaneNdTileElemsB32 :
                    elementBytes == 1 ? kPlaneNdTileElemsB8 : kPlaneNdTileElemsB16;
                const int64_t rowAlign = elementBytes == 4 ? 16 : 32 / elementBytes;
                const int64_t rowLimit = schedule == kSchedulePlaneB32WideTail ? kPlaneB32WideTailRows :
                    schedule == kSchedulePlaneB32Wide ? kPlaneB32WideTileRows : 4095;
                const int64_t maxRows = std::min(rowLimit, tileElements / localTileCols) / rowAlign * rowAlign;
                blockRows = std::min(maxRows, CeilDiv(transposedDim, rowAlign) * rowAlign);
                taskItems = planes * CeilDiv(transposedDim, blockRows) * CeilDiv(inputDims[rank - 1], tileCols);
            }
        } else {
            schedule = kScheduleGenericNd;
            blockRows = kGenericNdTileElems;
            const int64_t outputLast = inputDims[dims[rank - 1]];
            if (dims[rank - 1] == rank - 1 && outputLast * elementBytes <= kGenericNdInputBytes) {
                const int64_t reorderedGroup =
                    ReorderedRowsGroup(inputDims[0], outputLast, elementBytes);
                const bool reordered = elementBytes == 2 && rank == 4 &&
                    IsPerm(dims, rank, 2, 1, 0, 3) &&
                    ReorderedRowsUbBytes(inputDims[0], outputLast, elementBytes, reorderedGroup) <=
                        kGenericReorderedUbBytes;
                if (reordered) {
                    schedule = kScheduleGenericRowsReordered;
                    blockRows = reorderedGroup;
                    taskItems = inputDims[1] * CeilDiv(inputDims[2], reorderedGroup);
                } else {
                    const int64_t localRowBytes = CeilDiv(outputLast * elementBytes, 32) * 32;
                    int64_t inputRowStride = 1;
                    for (int64_t axis = dims[rank - 2] + 1; axis < rank; ++axis) {
                        inputRowStride *= inputDims[axis];
                    }

                    blockRows = inputRowStride - outputLast > UINT32_MAX / elementBytes ? 1 :
                        std::min<int64_t>(4095, kGenericNdInputBytes / localRowBytes);
                    const int64_t rowAxis = inputDims[dims[rank - 2]];
                    taskItems = totalLength / outputLast / rowAxis * CeilDiv(rowAxis, blockRows);
                }
            } else {
                const int64_t tileRows = outputLast <= blockRows ? blockRows / outputLast : 1;
                taskItems = CeilDiv(totalLength / outputLast, tileRows) * CeilDiv(outputLast, blockRows);
            }
        }
    }
    coreNum = std::min<int64_t>(coreNum, taskItems);
    coreNum = std::max<int64_t>(1, coreNum);
    if (schedule == kScheduleRank5GroupedB32) {
        const int64_t planes = inputDims[0] * inputDims[1];
        const int64_t paddedRows = CeilDiv(inputDims[2], 16) * 16;
        const int64_t packedUb = 8 * planes * (2 * paddedRows + 3 * inputDims[2]) * elementBytes;
        if (packedUb <= kGenericReorderedUbBytes) {
            tilingKey = dtype == ge::DT_FLOAT ? 6 : 7;
            context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
                planes * paddedRows * CeilDiv(inputDims[3] * inputDims[4], 480) * 480 * elementBytes;
        }
    }
    if (schedule == kSchedulePlaneB32FullRows) {
        tilingKey = dtype == ge::DT_FLOAT ? 4 : 5;
        if (blockRows == kCompactFullRows && inputDims[1] <= 768 && inputDims[1] % 8 != 0) {

            coreNum = CeilDiv(inputDims[0], CeilDiv(inputDims[0], coreNum));
            context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
                coreNum * CeilDiv(inputDims[1], 64) * 64 *
                CeilDiv(inputDims[2], kCompactTileCols) * kCompactTileCols * elementBytes;
        }
    }

    TransposeTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_dim0(schedule == kScheduleRank5GroupedB32 ? inputDims[2] :
                    schedule == kSchedulePlaneB32FullRows ? inputDims[1] :
                    schedule == kScheduleGenericRowsReordered ? inputDims[dims[rank - 2]] : rows);
    tiling.set_dim1(schedule == kScheduleReverseNd ? reverseTileB :
                    schedule == kScheduleRank5GroupedB32 ? inputDims[4] :
                    schedule == kSchedulePlaneB32FullRows || schedule == kScheduleGenericRowsReordered ?
                        inputDims[rank - 1] : cols);
    tiling.set_rank(rank);
    tiling.set_blockRows(blockRows);
    tiling.set_smallCoreTasks(taskItems / coreNum);
    tiling.set_bigCoreCount(taskItems % coreNum);
    tiling.set_schedule(schedule);
    FillDimTiling(tiling, inputDims, dims);

    context->SetBlockDim(static_cast<uint32_t>(coreNum));
    context->SetTilingKey(tilingKey);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t rank = inputShape->GetDimNum();
    int64_t dims[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    if (rank <= 0 || rank > 8 || !ReadDims(context->GetAttrs(), rank, dims)) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* outputShape = context->GetOutputShape(0);
    if (outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape resultShape;
    for (int64_t i = 0; i < rank; ++i) {
        resultShape.AppendDim(inputShape->GetDim(dims[i]));
    }
    *outputShape = resultShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const ge::DataType inputDtype = context->GetInputDataType(0);
    if (!IsSupportedType(inputDtype)) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, inputDtype);
    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class Transpose : public OpDef {
public:
    explicit Transpose(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("perm").AttrType(REQUIRED).ListInt();

        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Transpose);

}
