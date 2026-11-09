#include "square_sum_v1_tiling.h"
#include <cstdint>
#include <limits>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr uint32_t kDimNum = 5U;
constexpr uint32_t kAivCoreNum910B = 40U;
constexpr uint32_t kModeSuffixReduce = 0U;
constexpr uint32_t kModeInnerAccum = 1U;
constexpr uint32_t kModePlaneReduce = 2U;
constexpr uint32_t kModeGenericFallback = 3U;
constexpr uint32_t kModeContiguousBlockReduce = 4U;
constexpr uint32_t kModeRank4Reduce02 = 5U;
constexpr uint32_t kPlaneTileRows = 128U;
constexpr uint32_t kPlaneReduceChunkElems = 64U;
constexpr uint32_t kBlockReduceMaxRows = 64U;
constexpr uint32_t kBlockReduceTargetTasks = kAivCoreNum910B * 2U;
constexpr uint32_t kBlockReduceUbBudgetBytes = 176U * 1024U;
constexpr uint32_t kRank4Reduce02MaxTileElems = 192U;
constexpr uint32_t kRank4Reduce02BatchB = 2U;
constexpr uint32_t kRank4Reduce02MaxInputSlotBytes = 64U * 1024U;
constexpr uint32_t kPhaseMinReduceRows = 64U * 1024U;
constexpr uint32_t kPhaseMinInnerCols = 257U;
constexpr uint32_t kPhaseMaxInnerCols = 1024U;
constexpr uint32_t kRowShardMinInnerCols = 65U;
constexpr uint32_t kRowShardMaxInnerCols = 512U;
constexpr uint32_t kRowShardMinReduceRows = 512U;
constexpr uint32_t kRowShardMinPlanesPerCore = 2U;

static uint32_t DataTypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2U;
        case ge::DT_FLOAT:
            return 4U;
        default:
            return 0U;
    }
}

static uint32_t CeilDivU32(uint32_t x, uint32_t y)
{
    return x / y + (x % y != 0U ? 1U : 0U);
}

static uint32_t AlignDownU32(uint32_t x, uint32_t alignment)
{
    return (x / alignment) * alignment;
}

static bool FillDims(const gert::Shape& shape, uint32_t (&dims)[kDimNum])
{
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        dims[i] = 1U;
    }
    const uint32_t rank = static_cast<uint32_t>(shape.GetDimNum());
    const uint32_t base = kDimNum - rank;
    for (uint32_t i = 0U; i < rank; ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0 || static_cast<uint64_t>(dim) > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        dims[base + i] = static_cast<uint32_t>(dim);
    }
    return true;
}

static bool CalcStrides(const uint32_t (&dims)[kDimNum], uint32_t (&strides)[kDimNum])
{
    uint64_t stride = 1ULL;
    for (int32_t i = static_cast<int32_t>(kDimNum) - 1; i >= 0; --i) {
        if (stride > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        strides[i] = static_cast<uint32_t>(stride);
        stride *= dims[i];
    }
    return stride <= std::numeric_limits<uint32_t>::max();
}

static bool CalcTotal(const uint32_t (&dims)[kDimNum], uint32_t& result)
{
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        if (dims[i] == 0U) {
            result = 0U;
            return true;
        }
    }
    uint64_t total = 1ULL;
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        if (total > std::numeric_limits<uint32_t>::max() / dims[i]) {
            return false;
        }
        total *= dims[i];
    }
    result = static_cast<uint32_t>(total);
    return true;
}

static bool HasContiguousMiddleReduceLayout(
    const uint32_t (&dims)[kDimNum], const uint32_t (&reduceMask)[kDimNum],
    uint32_t reduceSize, uint32_t innerSize)
{
    uint32_t firstReduceDim = kDimNum;
    uint32_t lastReduceDim = kDimNum;
    for (uint32_t dim = 0U; dim < kDimNum; ++dim) {
        if (reduceMask[dim] != 0U) {
            if (firstReduceDim == kDimNum) {
                firstReduceDim = dim;
            }
            lastReduceDim = dim;
        }
    }
    if (firstReduceDim == kDimNum || lastReduceDim + 1U >= kDimNum) {
        return false;
    }
    for (uint32_t dim = firstReduceDim; dim <= lastReduceDim; ++dim) {
        if (reduceMask[dim] == 0U) {
            return false;
        }
    }
    uint64_t contiguousReduceRows = 1ULL;
    for (uint32_t dim = firstReduceDim; dim <= lastReduceDim; ++dim) {
        contiguousReduceRows *= dims[dim];
    }
    uint64_t suffixCols = 1ULL;
    for (uint32_t dim = lastReduceDim + 1U; dim < kDimNum; ++dim) {
        suffixCols *= dims[dim];
    }
    return contiguousReduceRows == reduceSize && suffixCols == innerSize;
}

static uint32_t CalcVectorTileElems(uint32_t elemBytes)
{
    const uint32_t alignElems = 32U / elemBytes;
    return AlignDownU32(16384U, alignElems);
}

static uint32_t CalcInnerTileElems(uint32_t elemBytes, uint32_t outputSize)
{
    const uint32_t alignElems = 32U / elemBytes;
    constexpr uint32_t kMaxInnerTileElems = 256U;
    constexpr uint32_t kTargetTasks = kAivCoreNum910B * 2U;
    if (outputSize == 0U) {
        return alignElems;
    }
    uint32_t tileElems = CeilDivU32(outputSize, kTargetTasks);
    tileElems = CeilDivU32(tileElems, alignElems) * alignElems;
    if (tileElems > kMaxInnerTileElems) {
        tileElems = kMaxInnerTileElems;
    }
    if (tileElems < alignElems) {
        tileElems = alignElems;
    }
    return tileElems;
}

static bool IsRank4Reduce02(const uint32_t (&dims)[kDimNum],
                            const uint32_t (&reduceMask)[kDimNum])
{
    return dims[0] == 1U && reduceMask[0] == 0U && reduceMask[1] != 0U &&
           reduceMask[2] == 0U && reduceMask[3] != 0U && reduceMask[4] == 0U &&
           dims[1] != 0U && dims[2] != 0U && dims[3] != 0U && dims[4] != 0U;
}

static uint32_t CalcRank4Reduce02TileElems(uint32_t elemBytes, uint32_t dimB,
                                            uint32_t dimD)
{
    if (elemBytes == 0U || dimB == 0U || dimD == 0U) {
        return 0U;
    }
    const uint32_t alignElems = 32U / elemBytes;
    const uint32_t tileCap = dimD < kRank4Reduce02MaxTileElems
        ? dimD : kRank4Reduce02MaxTileElems;
    uint32_t dTiles = CeilDivU32(kAivCoreNum910B * 2U, dimB);
    if (dTiles == 0U) {
        dTiles = 1U;
    }
    uint32_t tileElems = CeilDivU32(dimD, dTiles);
    if (dimD >= alignElems) {
        tileElems = CeilDivU32(tileElems, alignElems) * alignElems;
    }
    if (tileElems > tileCap) {
        tileElems = tileCap;
        if (dimD >= alignElems) {
            tileElems = AlignDownU32(tileElems, alignElems);
        }
    }
    if (tileElems == 0U) {
        tileElems = dimD < alignElems ? dimD : alignElems;
    }
    return tileElems;
}

static bool CanBatchRank4Reduce02(uint32_t elemBytes, uint32_t dimB,
                                  uint32_t dimC, uint32_t dimD,
                                  uint32_t dTileElems)
{
    if (elemBytes == 0U || dimB == 0U || dimC == 0U || dimC > 255U ||
        dTileElems != dimD || ((dimD * elemBytes) & 31U) != 0U) {
        return false;
    }
    const uint32_t batchTasks = CeilDivU32(dimB, kRank4Reduce02BatchB);
    const uint64_t inputSlotBytes = static_cast<uint64_t>(kRank4Reduce02BatchB) *
                                    dimC * dimD * elemBytes;
    return batchTasks >= kAivCoreNum910B &&
           inputSlotBytes <= kRank4Reduce02MaxInputSlotBytes;
}

static bool FindContiguousBlockReduce(uint32_t elemBytes, uint32_t outputSize,
                                      uint32_t reduceSize, uint32_t blockElems,
                                      uint32_t tileElems, uint32_t& batchRows)
{
    if (outputSize < blockElems * 2U || reduceSize <= kPlaneReduceChunkElems) {
        return false;
    }
    const uint64_t rowPitch = static_cast<uint64_t>(CeilDivU32(reduceSize, blockElems)) * blockElems;
    // Two input and output queue slots let MTE2 prefetch the next row batch
    // while vector reduction consumes the current batch and MTE3 retires the
    // preceding result. Account for both queue slots in the UB tile budget.
    const uint64_t rowBytes = rowPitch * (2U * elemBytes + 2U * sizeof(float)) +
                              2U * elemBytes + 2U * sizeof(float);
    if (rowPitch > std::numeric_limits<uint32_t>::max() || rowBytes == 0ULL ||
        reduceSize > std::numeric_limits<uint32_t>::max() / elemBytes) {
        return false;
    }

    uint32_t maxRows = static_cast<uint32_t>(kBlockReduceUbBudgetBytes / rowBytes);
    if (maxRows > kBlockReduceMaxRows) {
        maxRows = kBlockReduceMaxRows;
    }
    maxRows = AlignDownU32(maxRows, blockElems);
    if (maxRows < blockElems) {
        return false;
    }

    const uint32_t outputBlocks = CeilDivU32(outputSize, blockElems);
    uint32_t rows = CeilDivU32(outputBlocks, kBlockReduceTargetTasks) * blockElems;
    if (rows < blockElems) {
        rows = blockElems;
    }
    if (rows > maxRows) {
        rows = maxRows;
    }
    const uint64_t oldCommands = static_cast<uint64_t>(outputSize) *
                                 CeilDivU32(reduceSize, tileElems);
    const uint64_t newCommands = CeilDivU32(outputSize, rows);
    if (newCommands >= oldCommands) {
        return false;
    }
    batchRows = rows;
    return true;
}

template <typename AttrT>
static bool FillReduceMask(const AttrT* attr, uint32_t rank, uint32_t (&mask)[kDimNum])
{
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        mask[i] = 0U;
    }
    if (attr == nullptr) {
        return false;
    }
    const int64_t* axes = attr->GetData();
    const uint32_t axisNum = static_cast<uint32_t>(attr->GetSize());
    const uint32_t base = kDimNum - rank;
    if (axisNum == 0U) {
        for (uint32_t i = 0U; i < rank; ++i) {
            mask[base + i] = 1U;
        }
        return true;
    }
    for (uint32_t i = 0U; i < axisNum; ++i) {
        int64_t axis = axes[i];
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
            return false;
        }
        const uint32_t maskIndex = base + static_cast<uint32_t>(axis);
        if (mask[maskIndex] != 0U) {
            return false;
        }
        mask[maskIndex] = 1U;
    }
    return true;
}

static bool IsSuffixReduce(const uint32_t (&reduceMask)[kDimNum])
{
    bool seenReduce = false;
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        if (reduceMask[i] != 0U) {
            seenReduce = true;
        } else if (seenReduce) {
            return false;
        }
    }
    return seenReduce;
}

static bool FindLastReduce(const uint32_t (&reduceMask)[kDimNum], uint32_t& lastReduce)
{
    for (int32_t i = static_cast<int32_t>(kDimNum) - 1; i >= 0; --i) {
        if (reduceMask[static_cast<uint32_t>(i)] != 0U) {
            lastReduce = static_cast<uint32_t>(i);
            return true;
        }
    }
    return false;
}

static bool FindPlaneReduce(const uint32_t (&dims)[kDimNum],
                            const uint32_t (&reduceMask)[kDimNum],
                            uint32_t& planeSize, uint32_t& outerOutputSize,
                            uint32_t& suffixReduceSize, uint32_t& prefixReduceSize)
{
    if (reduceMask[kDimNum - 1U] == 0U) {
        return false;
    }

    uint32_t suffixReduceStart = kDimNum - 1U;
    while (suffixReduceStart > 0U && reduceMask[suffixReduceStart - 1U] != 0U) {
        --suffixReduceStart;
    }
    uint32_t planeStart = suffixReduceStart;
    while (planeStart > 0U && reduceMask[planeStart - 1U] == 0U) {
        --planeStart;
    }
    if (planeStart == suffixReduceStart) {
        return false;
    }

    bool hasPrefixReduce = false;
    uint64_t plane = 1ULL;
    uint64_t outerOutput = 1ULL;
    uint64_t prefixReduce = 1ULL;
    uint64_t suffixReduce = 1ULL;
    for (uint32_t i = 0U; i < planeStart; ++i) {
        if (reduceMask[i] != 0U) {
            hasPrefixReduce = true;
            prefixReduce *= dims[i];
        } else {
            outerOutput *= dims[i];
        }
    }
    for (uint32_t i = planeStart; i < suffixReduceStart; ++i) {
        plane *= dims[i];
    }
    for (uint32_t i = suffixReduceStart; i < kDimNum; ++i) {
        suffixReduce *= dims[i];
    }
    if (!hasPrefixReduce || plane == 0ULL || prefixReduce == 0ULL || suffixReduce == 0ULL ||
        plane > std::numeric_limits<uint32_t>::max() ||
        outerOutput > std::numeric_limits<uint32_t>::max() ||
        prefixReduce > std::numeric_limits<uint32_t>::max() ||
        suffixReduce > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    planeSize = static_cast<uint32_t>(plane);
    outerOutputSize = static_cast<uint32_t>(outerOutput);
    prefixReduceSize = static_cast<uint32_t>(prefixReduce);
    suffixReduceSize = static_cast<uint32_t>(suffixReduce);
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto& inputShape = context->GetInputShape(0)->GetStorageShape();
    const uint32_t rank = static_cast<uint32_t>(inputShape.GetDimNum());
    if (rank == 0U || rank > kDimNum) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    const uint32_t elemBytes = DataTypeBytes(dtype);
    if (elemBytes == 0U) {
        return ge::GRAPH_FAILED;
    }

    uint32_t inDims[kDimNum];
    uint32_t outDims[kDimNum];
    uint32_t inStrides[kDimNum];
    uint32_t outStrides[kDimNum];
    uint32_t reduceMask[kDimNum];
    if (!FillDims(inputShape, inDims)) {
        return ge::GRAPH_FAILED;
    }
    if (!FillReduceMask(context->GetAttrs()->GetListInt(0), rank, reduceMask)) {
        return ge::GRAPH_FAILED;
    }
    uint64_t reduceSize64 = 1ULL;
    bool emptyReduce = false;
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        outDims[i] = reduceMask[i] == 0U ? inDims[i] : 1U;
        if (reduceMask[i] != 0U) {
            emptyReduce = emptyReduce || inDims[i] == 0U;
            if (!emptyReduce) {
                reduceSize64 *= inDims[i];
                if (reduceSize64 > std::numeric_limits<uint32_t>::max()) {
                    return ge::GRAPH_FAILED;
                }
            }
        }
    }
    const uint32_t reduceSize = emptyReduce ? 0U : static_cast<uint32_t>(reduceSize64);
    if (!CalcStrides(inDims, inStrides) || !CalcStrides(outDims, outStrides)) {
        return ge::GRAPH_FAILED;
    }

    uint32_t inputElems = 0U;
    uint32_t totalElems = 0U;
    if (!CalcTotal(inDims, inputElems) || !CalcTotal(outDims, totalElems)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockElems = 32U / elemBytes;
    uint32_t totalBlocks = CeilDivU32(totalElems, blockElems);
    uint32_t tileElems = CalcVectorTileElems(elemBytes);
    uint32_t accTileElems = 32U / elemBytes;
    uint32_t outerSize = totalElems;
    uint32_t innerSize = 1U;
    uint32_t taskNum = totalElems;
    uint32_t mode = kModeGenericFallback;
    uint32_t blockDim = totalBlocks < kAivCoreNum910B ? totalBlocks : kAivCoreNum910B;

    uint32_t lastReduce = 0U;
    if (IsSuffixReduce(reduceMask)) {
        uint32_t batchRows = 0U;
        const bool useBlockReduce = FindContiguousBlockReduce(
            elemBytes, totalElems, reduceSize, blockElems, tileElems, batchRows);
        mode = useBlockReduce ? kModeContiguousBlockReduce : kModeSuffixReduce;
        outerSize = totalElems;
        innerSize = 1U;
        if (useBlockReduce) {
            accTileElems = batchRows;
            taskNum = CeilDivU32(totalElems, batchRows);
            blockDim = taskNum < kAivCoreNum910B ? taskNum : kAivCoreNum910B;
        } else {
            taskNum = totalElems;
            blockDim = totalBlocks < kAivCoreNum910B ? totalBlocks : kAivCoreNum910B;
        }
    } else if (dtype != ge::DT_BF16 && IsRank4Reduce02(inDims, reduceMask)) {
        const uint32_t dimB = inDims[2];
        const uint32_t dimC = inDims[3];
        const uint32_t dimD = inDims[4];
        const uint32_t dTileElems = CalcRank4Reduce02TileElems(elemBytes, dimB, dimD);
        if (dTileElems == 0U) {
            return ge::GRAPH_FAILED;
        }
        const bool batchB = CanBatchRank4Reduce02(elemBytes, dimB, dimC,
                                                   dimD, dTileElems);
        const uint32_t dTiles = CeilDivU32(dimD, dTileElems);
        const uint64_t taskNum64 = batchB
            ? CeilDivU32(dimB, kRank4Reduce02BatchB)
            : static_cast<uint64_t>(dimB) * dTiles;
        if (taskNum64 == 0ULL || taskNum64 > std::numeric_limits<uint32_t>::max()) {
            return ge::GRAPH_FAILED;
        }
        mode = kModeRank4Reduce02;
        innerSize = dimD;
        outerSize = dimB;
        accTileElems = dTileElems;
        taskNum = static_cast<uint32_t>(taskNum64);
        blockDim = taskNum < kAivCoreNum910B ? taskNum : kAivCoreNum910B;
    } else if (FindLastReduce(reduceMask, lastReduce) && lastReduce + 1U < kDimNum) {
        mode = kModeInnerAccum;
        innerSize = 1U;
        for (uint32_t i = lastReduce + 1U; i < kDimNum; ++i) {
            innerSize *= inDims[i];
        }
        outerSize = innerSize == 0U ? 0U : totalElems / innerSize;
        accTileElems = CalcInnerTileElems(elemBytes, totalElems);
        taskNum = CeilDivU32(totalElems, accTileElems);
        blockDim = taskNum < kAivCoreNum910B ? taskNum : kAivCoreNum910B;
    } else {
        uint32_t planeSize = 0U;
        uint32_t outerOutputSize = 0U;
        uint32_t suffixReduceSize = 0U;
        uint32_t prefixReduceSize = 0U;
        const bool isPlaneReduce = FindPlaneReduce(inDims, reduceMask, planeSize,
                                                    outerOutputSize, suffixReduceSize,
                                                    prefixReduceSize);
        const uint64_t oldCommands = isPlaneReduce
            ? static_cast<uint64_t>(totalElems) * (reduceSize / inDims[kDimNum - 1U]) *
                  CeilDivU32(inDims[kDimNum - 1U], tileElems)
            : 0ULL;
        const uint64_t newCommands = isPlaneReduce
            ? static_cast<uint64_t>(outerOutputSize) * CeilDivU32(planeSize, kPlaneTileRows) *
                  prefixReduceSize * CeilDivU32(suffixReduceSize, kPlaneReduceChunkElems)
            : 0ULL;
        const bool planeStrideFits = isPlaneReduce &&
            suffixReduceSize <= std::numeric_limits<uint32_t>::max() / elemBytes;
        if (planeStrideFits && newCommands < oldCommands) {
            mode = kModePlaneReduce;
            outerSize = outerOutputSize;
            innerSize = planeSize;
            accTileElems = kPlaneTileRows;
            const uint32_t planeTiles = CeilDivU32(planeSize, accTileElems);
            const uint64_t taskNum64 = static_cast<uint64_t>(outerOutputSize) * planeTiles;
            if (taskNum64 > std::numeric_limits<uint32_t>::max()) {
                return ge::GRAPH_FAILED;
            }
            taskNum = static_cast<uint32_t>(taskNum64);
            blockDim = taskNum < kAivCoreNum910B ? taskNum : kAivCoreNum910B;
        } else {
            taskNum = totalElems;
            blockDim = totalBlocks < kAivCoreNum910B ? totalBlocks : kAivCoreNum910B;
        }
    }
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    SquareSumV1TilingData tiling;
    tiling.set_mode(mode);
    tiling.set_total_size(totalElems);
    tiling.set_input_size(inputElems);
    tiling.set_output_size(totalElems);
    tiling.set_block_elems(blockElems);
    tiling.set_total_blocks(totalBlocks);
    tiling.set_tile_elems(tileElems);
    tiling.set_acc_tile_elems(accTileElems);
    tiling.set_outer_size(outerSize);
    tiling.set_reduce_size(reduceSize);
    tiling.set_inner_size(innerSize);
    tiling.set_task_num(taskNum);
    tiling.set_in_dims(inDims);
    tiling.set_out_dims(outDims);
    tiling.set_in_strides(inStrides);
    tiling.set_out_strides(outStrides);
    tiling.set_reduce_mask(reduceMask);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    const bool usePhaseWorkspace =
        dtype == ge::DT_FLOAT && mode == kModeInnerAccum &&
        totalElems == innerSize && reduceSize >= kPhaseMinReduceRows &&
        innerSize >= kPhaseMinInnerCols &&
        innerSize <= kPhaseMaxInnerCols && blockDim > 1U &&
        HasContiguousMiddleReduceLayout(
            inDims, reduceMask, reduceSize, innerSize);
    const uint64_t rowPitch =
        ((static_cast<uint64_t>(innerSize) + 7ULL) / 8ULL) * 8ULL;
    const bool useRowShardWorkspace =
        dtype == ge::DT_BF16 && mode == kModeInnerAccum && blockDim > 1U &&
        innerSize >= kRowShardMinInnerCols && innerSize <= kRowShardMaxInnerCols &&
        (innerSize & 15U) != 0U && reduceSize >= kRowShardMinReduceRows &&
        totalElems % innerSize == 0U &&
        totalElems / innerSize >= blockDim * kRowShardMinPlanesPerCore &&
        HasContiguousMiddleReduceLayout(inDims, reduceMask, reduceSize, innerSize);
    const uint64_t rowShardPitch =
        ((static_cast<uint64_t>(innerSize) + 15ULL) / 16ULL) * 16ULL;
    const uint64_t userWorkspaceBytes = usePhaseWorkspace
        ? 1ULL * blockDim * rowPitch * sizeof(float)
        : (useRowShardWorkspace
            ? 2ULL * (totalElems / innerSize) * rowShardPitch * sizeof(float)
            : 0ULL);
    const auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint64_t systemWorkspaceBytes =
        static_cast<uint64_t>(platform.GetLibApiWorkSpaceSize());
    context->GetWorkspaceSizes(1)[0] =
        systemWorkspaceBytes + userWorkspaceBytes;
    context->SetBlockDim(blockDim);
    if (dtype == ge::DT_FLOAT) {
        context->SetTilingKey(usePhaseWorkspace ? 3U : 0U);
    } else if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(1U);
    } else {
        context->SetTilingKey(useRowShardWorkspace ? 4U : 2U);
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    gert::Shape* outputShape = context->GetOutputShape(0);
    auto* axisAttr = context->GetAttrs()->GetListInt(0);
    const bool* keepDimsPtr = context->GetAttrs()->GetBool(1);
    if (axisAttr == nullptr || keepDimsPtr == nullptr) {
        return GRAPH_FAILED;
    }
    const bool keepDims = *keepDimsPtr;
    const uint32_t rank = static_cast<uint32_t>(inputShape->GetDimNum());
    if (rank == 0U || rank > kDimNum) {
        return GRAPH_FAILED;
    }
    uint32_t reduceMask[kDimNum];
    if (!FillReduceMask(axisAttr, rank, reduceMask)) {
        return GRAPH_FAILED;
    }
    const uint32_t base = kDimNum - rank;
    gert::Shape out;
    for (uint32_t i = 0U; i < rank; ++i) {
        if (reduceMask[base + i] != 0U) {
            if (keepDims) {
                out.AppendDim(1);
            }
        } else {
            out.AppendDim(inputShape->GetDim(i));
        }
    }
    if (out.GetDimNum() == 0) {
        out.AppendDim(1);
    }
    *outputShape = out;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("input").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").ListInt();
        this->Attr("keep_dims").Bool();
        this->Output("output").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
}  // namespace ops
