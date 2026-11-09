#pragma once
#include "square_sum_shape.h"
#include "../op_kernel/square_sum_policy.h"

namespace {

constexpr uint32_t kAivCoreNum910B = 40U;
constexpr uint32_t kModeSuffixReduce = 0U;
constexpr uint32_t kModeInnerAccum = 1U;
constexpr uint32_t kModePlaneReduce = 2U;
constexpr uint32_t kModeGenericFallback = 3U;
constexpr uint32_t kModeContiguousBlockReduce = 4U;
constexpr uint32_t kModeRank4Reduce02 = 5U;
constexpr uint32_t kModeRank4Reduce02CoreRange = 6U;
constexpr uint32_t kModeWorkspaceBf16Inner = 7U;
constexpr uint32_t kPlaneTileRows = 128U;
constexpr uint32_t kPlaneReduceChunkElems = 64U;
constexpr uint32_t kBlockReduceMaxRows = 64U;
constexpr uint32_t kBlockReduceTargetTasks = kAivCoreNum910B * 2U;
constexpr uint32_t kBlockReduceUbBudgetBytes = 176U * 1024U;
constexpr uint32_t kRank4Reduce02MaxTileElems = 192U;
constexpr uint32_t kRank4Reduce02BatchB = 2U;
constexpr uint32_t kRank4Reduce02MaxInputSlotBytes = 64U * 1024U;

static uint64_t CalcWorkspaceBf16Bytes(uint32_t innerSize, uint32_t outerSize,
                                        uint32_t reduceSize,
                                        const uint32_t* dims, const uint32_t* mask)
{
    if (innerSize < 64U || innerSize > 256U || reduceSize <= 512U ||
        !square_sum_policy::ContiguousInner(dims, mask)) {
        return 0ULL;
    }
    const uint64_t rows = static_cast<uint64_t>(outerSize) * reduceSize;
    const uint64_t pitch = ((static_cast<uint64_t>(innerSize) + 15ULL) / 16ULL) * 16ULL;
    const uint64_t slots = square_sum_policy::WorkspaceBf16Slots(
        outerSize, reduceSize, kAivCoreNum910B);
    const uint64_t bytes = static_cast<uint64_t>(kAivCoreNum910B) * slots * pitch * sizeof(float);
    if (rows < kAivCoreNum910B * 80ULL ||
        rows > std::numeric_limits<uint32_t>::max() || bytes > 1024ULL * 1024ULL ||
        square_sum_policy::WorkspaceBf16Bytes(innerSize) > square_sum_policy::kUbBytes) {
        return 0ULL;
    }
    return bytes;
}

static uint32_t CeilDivU32(uint32_t x, uint32_t y)
{
    return x / y + (x % y != 0U ? 1U : 0U);
}

static uint32_t AlignDownU32(uint32_t x, uint32_t alignment)
{
    return (x / alignment) * alignment;
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
    constexpr uint32_t kTargetTasks = kAivCoreNum910B;
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

static bool CanUseRank4Reduce02CoreRange(uint32_t elemBytes, uint32_t dimB,
                                         uint32_t dimC, uint32_t dimD,
                                         uint32_t dTileElems)
{
    if (elemBytes == 0U || dimB <= kAivCoreNum910B * kRank4Reduce02BatchB ||
        dimC == 0U || dimC > 255U || dTileElems != dimD ||
        ((dimD * elemBytes) & 31U) != 0U) {
        return false;
    }
    const uint32_t maxBPerCore = CeilDivU32(dimB, kAivCoreNum910B);
    const uint64_t inputSlotBytes = static_cast<uint64_t>(maxBPerCore) *
                                    dimC * dimD * elemBytes;
    return inputSlotBytes <= kRank4Reduce02MaxInputSlotBytes;
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

}  // namespace
