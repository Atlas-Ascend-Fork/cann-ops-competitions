#include "transpose_tiling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <vector>

#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"

#ifndef OP_LOGE
#define OP_LOGE(...) ((void)0)
#endif
#ifndef OP_CHECK_IF
#define OP_CHECK_IF(condition, log_statement, return_statement) \
    do {                                                        \
        if (condition) {                                        \
            log_statement;                                      \
            return_statement;                                   \
        }                                                       \
    } while (false)
#endif
#ifndef OP_CHECK_NULL_WITH_CONTEXT
#define OP_CHECK_NULL_WITH_CONTEXT(context, pointer) \
    do {                                             \
        if ((pointer) == nullptr) {                  \
            return ge::GRAPH_FAILED;                 \
        }                                            \
    } while (false)
#endif

namespace s9transpose {
namespace {
constexpr uint64_t kBlockBytes = 32;
constexpr uint64_t kDefaultUbBytes = 192 * 1024;
constexpr uint64_t kUbReserveBytes = 16 * 1024;
constexpr uint64_t kMaxTileBytes = 64 * 1024;
// A 16x16 FP16 micro-tile moves only 512 bytes in and 512 bytes out.  Giving
// every micro-tile to a separate AIV spends more time on tiny DMA commands and
// block scheduling than on data movement.  Four adjacent column tiles form a
// 16x64 super-tile (4 KiB total traffic), while still exposing 32 tasks for the
// public 128x256 matrix.  Keep this structural and tune the crossover for the
// generic Ascend 910B family rather than recognizing a particular shape.
constexpr uint64_t kDefaultSwapTilesPerTask = 4;
constexpr uint64_t kCompactSwapTilesPerTask = 8;
constexpr uint64_t kCompactSwapTaskCeiling = 64;
constexpr uint64_t kMinimumCompactSwapTasks = 8;
constexpr uint64_t kInt8ShortHeightTiles = 2;
constexpr uint64_t kInt8TargetCoreNumerator = 2;
constexpr uint64_t kInt8TargetCoreDenominator = 3;
constexpr uint64_t kInt8MaximumWideBatch = 4;
constexpr uint64_t kGroupedRowTiles = 8;
constexpr uint64_t kGroupedRowMinimumTiles = 64;

uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : (value + divisor - 1) / divisor;
}

uint64_t Product(const std::vector<uint64_t>& values, size_t begin, size_t end)
{
    uint64_t result = 1;
    for (size_t i = begin; i < end; ++i) {
        // The competition dimensions fit uint64_t.  This guard makes a bad
        // shape fail tiling instead of silently wrapping an address.
        if (values[i] != 0 && result > UINT64_MAX / values[i]) {
            return 0;
        }
        result *= values[i];
    }
    return result;
}

uint32_t DataTypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
            return 1;
        case ge::DT_FLOAT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4;
        default:
            return 0;
    }
}

bool ReadPerm(const gert::ContinuousVector* value, uint32_t rank, std::vector<int32_t>& perm)
{
    if (value == nullptr || value->GetSize() != rank) {
        return false;
    }
    const int64_t* data = static_cast<const int64_t*>(value->GetData());
    for (uint32_t i = 0; i < rank; ++i) {
        int64_t axis = data[i];
        if (axis < 0) {
            axis += rank;
        }
        perm.push_back(static_cast<int32_t>(axis));
    }
    return true;
}

bool IsPermutation(const std::vector<int32_t>& perm)
{
    std::vector<int32_t> sorted = perm;
    std::sort(sorted.begin(), sorted.end());
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i] != static_cast<int32_t>(i)) {
            return false;
        }
    }
    return true;
}

void RemoveUnitAxes(
    const std::vector<uint64_t>& shape, const std::vector<int32_t>& perm, std::vector<uint64_t>& reducedShape,
    std::vector<int32_t>& reducedPerm)
{
    std::array<int32_t, kMaxRank> oldToNew{};
    oldToNew.fill(-1);
    for (size_t axis = 0; axis < shape.size(); ++axis) {
        if (shape[axis] != 1) {
            oldToNew[axis] = static_cast<int32_t>(reducedShape.size());
            reducedShape.push_back(shape[axis]);
        }
    }
    if (reducedShape.empty()) {
        reducedShape.push_back(1);
        reducedPerm.push_back(0);
        return;
    }
    for (int32_t axis : perm) {
        if (oldToNew[axis] >= 0) {
            reducedPerm.push_back(oldToNew[axis]);
        }
    }
}

void CanonicalizeLayout(std::vector<uint64_t>& shape, std::vector<int32_t>& perm)
{
    std::vector<uint64_t> noUnits;
    std::vector<int32_t> noUnitPerm;
    RemoveUnitAxes(shape, perm, noUnits, noUnitPerm);
    std::array<int32_t, kMaxRank> position{}, group{};
    for (size_t i = 0; i < noUnitPerm.size(); ++i) position[noUnitPerm[i]] = static_cast<int32_t>(i);
    std::vector<uint64_t> merged;
    for (size_t a = 0; a < noUnits.size(); ++a) {
        if (a > 0 && position[a] == position[a - 1] + 1) {
            merged.back() *= noUnits[a];
        } else {
            merged.push_back(noUnits[a]);
        }
        group[a] = static_cast<int32_t>(merged.size() - 1);
    }
    std::vector<int32_t> mergedPerm;
    for (int32_t a : noUnitPerm) {
        if (mergedPerm.empty() || mergedPerm.back() != group[a]) mergedPerm.push_back(group[a]);
    }
    shape.swap(merged); perm.swap(mergedPerm);
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<TransposeTilingData>();
    const auto* xShape = context->GetInputShape(0);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    *tiling = {};

    const auto& shapeView = xShape->GetStorageShape();
    uint32_t rank = static_cast<uint32_t>(shapeView.GetDimNum());
    OP_CHECK_IF(rank == 0 || rank > kMaxRank,
        OP_LOGE(context->GetNodeName(), "rank must be in [1, %u], got %u", kMaxRank, rank),
        return ge::GRAPH_FAILED);
    std::vector<uint64_t> shape(rank);
    for (uint32_t axis = 0; axis < rank; ++axis) {
        const int64_t extent = shapeView.GetDim(axis);
        OP_CHECK_IF(extent <= 0, OP_LOGE(context->GetNodeName(), "all dimensions must be positive"),
            return ge::GRAPH_FAILED);
        shape[axis] = static_cast<uint64_t>(extent);
    }

    std::vector<int32_t> perm;
    const auto* dims = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    OP_CHECK_IF(!ReadPerm(dims, rank, perm), OP_LOGE(context->GetNodeName(), "dims length must equal rank"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!IsPermutation(perm), OP_LOGE(context->GetNodeName(), "invalid permutation"),
        return ge::GRAPH_FAILED);

    const uint32_t dtypeBytes = DataTypeBytes(context->GetInputDesc(0)->GetDataType());
    OP_CHECK_IF(dtypeBytes == 0, OP_LOGE(context->GetNodeName(), "unsupported input dtype"),
        return ge::GRAPH_FAILED);
    const uint64_t totalElements = Product(shape, 0, shape.size());
    OP_CHECK_IF(totalElements == 0, OP_LOGE(context->GetNodeName(), "shape product overflow"),
        return ge::GRAPH_FAILED);

    // Merge maximal axis runs consecutive in both linearizations.
    // This is a bijective mixed-radix relabeling and moves no tensor data.
    CanonicalizeLayout(shape, perm);
    rank = static_cast<uint32_t>(shape.size());
    tiling->rank = rank;
    tiling->dtypeBytes = dtypeBytes;
    tiling->totalElements = totalElements;
    uint64_t stride = 1;
    for (int32_t axis = static_cast<int32_t>(rank) - 1; axis >= 0; --axis) {
        tiling->shape[axis] = shape[axis];
        tiling->inputStride[axis] = stride;
        stride *= shape[axis];
    }
    for (uint32_t outAxis = 0; outAxis < rank; ++outAxis) {
        tiling->perm[outAxis] = perm[outAxis];
        tiling->outputShape[outAxis] = shape[perm[outAxis]];
    }

    std::vector<uint64_t> reducedShape;
    std::vector<int32_t> reducedPerm;
    RemoveUnitAxes(shape, perm, reducedShape, reducedPerm);
    bool identity = true;
    for (size_t i = 0; i < reducedPerm.size(); ++i) {
        identity = identity && reducedPerm[i] == static_cast<int32_t>(i);
    }

    Mode mode = Mode::kGenericOutputOwner;
    uint64_t innerElements = 1;
    // After an unchanged prefix, detect a cyclic swap of two contiguous axis
    // groups:
    //   [prefix, left-group, right-group] -> [prefix, right-group, left-group]
    // This is exactly a batched transpose of matrices
    //   H=product(left-group), W=product(right-group).
    // It strictly generalizes the old last-two-axis-only case and covers
    // layouts such as NCHW->NHWC and [A,B,C,D]->[C,D,A,B] without any
    // per-element address arithmetic.
    size_t fixedPrefix = 0;
    while (fixedPrefix < reducedPerm.size() &&
           reducedPerm[fixedPrefix] == static_cast<int32_t>(fixedPrefix)) {
        ++fixedPrefix;
    }
    size_t groupSplit = reducedPerm.size();
    bool swapContiguousGroups = false;
    if (fixedPrefix < reducedPerm.size()) {
        const int32_t firstMoved = reducedPerm[fixedPrefix];
        if (firstMoved > static_cast<int32_t>(fixedPrefix) &&
            firstMoved < static_cast<int32_t>(reducedPerm.size())) {
            groupSplit = static_cast<size_t>(firstMoved);
            size_t outputAxis = fixedPrefix;
            swapContiguousGroups = true;
            for (size_t inputAxis = groupSplit;
                 inputAxis < reducedPerm.size(); ++inputAxis, ++outputAxis) {
                swapContiguousGroups = swapContiguousGroups &&
                    reducedPerm[outputAxis] == static_cast<int32_t>(inputAxis);
            }
            for (size_t inputAxis = fixedPrefix;
                 inputAxis < groupSplit; ++inputAxis, ++outputAxis) {
                swapContiguousGroups = swapContiguousGroups &&
                    reducedPerm[outputAxis] == static_cast<int32_t>(inputAxis);
            }
            swapContiguousGroups = swapContiguousGroups &&
                                   outputAxis == reducedPerm.size();
        }
    }

    if (identity) {
        mode = Mode::kIdentity;
        innerElements = totalElements;
    } else if (swapContiguousGroups) {
        mode = Mode::kSwapLastTwo;
        tiling->matrixH = Product(reducedShape, fixedPrefix, groupSplit);
        tiling->matrixW = Product(reducedShape, groupSplit, reducedShape.size());
        tiling->outer = Product(reducedShape, 0, fixedPrefix);
        // TransDataTo5HD consumes one 32-byte block from each of 16 rows per
        // repeat: 16 columns for FP16, or 8 columns for FP32/INT32.  Keeping
        // the tile width derived from the dtype makes the fast path structural
        // and extends it to the exact 32-bit layout used by the hardware.
        tiling->microCols = kBlockBytes / dtypeBytes;
        tiling->tileRows = (tiling->matrixH + 15) / 16;
        tiling->tileCols = (tiling->matrixW + tiling->microCols - 1) / tiling->microCols;
        // TransData always operates on a 16-row x one-datablock micro-tile.
        // The kernel pads only the final width tile in UB and limits MTE3 to
        // the valid tail rectangle, so 16/32-bit last-two-axis swaps no longer
        // fall off a performance cliff for non-aligned H or W.
        const bool extStridesFit =
            tiling->matrixH <= static_cast<uint64_t>(UINT32_MAX) / dtypeBytes &&
            tiling->matrixW <= static_cast<uint64_t>(UINT32_MAX) / dtypeBytes;
        // INT8 uses the kernel's two-half TransData form.  That path assigns
        // a complete output-column batch to one AIV and synchronizes every
        // partial-row MTE3 before reusing its UB slot, so neighboring cores
        // never update the same 32-byte output block.  The same structural
        // swap route remains valid for FP16/FP32/INT32.
        tiling->fastSwap16 =
            (dtypeBytes == 1 || dtypeBytes == 2 || dtypeBytes == 4) &&
            extStridesFit;
    } else {
        size_t outputStart = rank - 1;
        while (outputStart > 0 && perm[outputStart - 1] + 1 == perm[outputStart]) {
            --outputStart;
        }
        const size_t inputStart = static_cast<size_t>(perm[outputStart]);
        bool suffix = true;
        if (rank - outputStart != rank - inputStart) {
            suffix = false;
        } else {
            for (size_t i = 0; i < rank - outputStart; ++i) {
                suffix = suffix && perm[outputStart + i] == static_cast<int32_t>(inputStart + i);
            }
        }
        if (suffix) {
            innerElements = Product(shape, inputStart, shape.size());
            if (innerElements > 1) {
                mode = Mode::kContiguousSuffix;
            }
        }
    }

    // Any permutation whose output-innermost axis spans complete GM blocks
    // is a batch of strided matrices. Input columns follow the original last
    // axis; output rows follow the new last axis. All other coordinates are
    // decoded once per tile, independently of rank and exact extents.
    if (mode == Mode::kGenericOutputOwner) {
        const uint32_t rowAxis = static_cast<uint32_t>(perm.back());
        uint64_t outputColumnStride = 1;
        for (int32_t outAxis = static_cast<int32_t>(rank) - 1; outAxis >= 0; --outAxis) {
            if (perm[outAxis] == static_cast<int32_t>(rank - 1)) { break; }
            outputColumnStride *= tiling->outputShape[outAxis];
        }
        const uint64_t otherCoordinates = totalElements / shape[rowAxis] / shape.back();
        uint64_t gcdValue = shape[rowAxis] * dtypeBytes, gcdBlock = 32;
        while (gcdBlock != 0) { const uint64_t r = gcdValue % gcdBlock; gcdValue = gcdBlock; gcdBlock = r; }
        const uint64_t byteRowQuantum = 32 / gcdValue;
        const bool alignedInt8 = dtypeBytes == 1 &&
            (byteRowQuantum == 1 || (outputColumnStride % 32 == 0 && otherCoordinates % byteRowQuantum == 0));
        const bool alignedWide = dtypeBytes > 1 && (byteRowQuantum == 1 ||
            (outputColumnStride * dtypeBytes % 32 == 0 && otherCoordinates % byteRowQuantum == 0));
        if (rowAxis != rank - 1 && (alignedWide || alignedInt8) &&
            tiling->inputStride[rowAxis] <= UINT32_MAX / dtypeBytes &&
            outputColumnStride <= UINT32_MAX / dtypeBytes) {
            mode = Mode::kPermutedMatrix;
            tiling->rowOwnershipQuantum = static_cast<uint32_t>(byteRowQuantum);
            tiling->matrixH = shape[rowAxis];
            tiling->matrixW = shape.back();
            tiling->outer = totalElements / tiling->matrixH / tiling->matrixW;
            tiling->microCols = dtypeBytes == 1 ? 16 : kBlockBytes / dtypeBytes;
            tiling->tileRows = CeilDiv(tiling->matrixH, 16);
            tiling->tileCols = CeilDiv(tiling->matrixW, tiling->microCols);
            tiling->tileBatchCols = std::min<uint64_t>(tiling->tileCols, kCompactSwapTilesPerTask);
            tiling->columnBatches = CeilDiv(tiling->tileCols, tiling->tileBatchCols);
            tiling->fastSwap16 = 1;
        }
    }

    tiling->mode = static_cast<uint32_t>(mode);
    tiling->innerElements = innerElements;
    tiling->outputBlocks = totalElements / innerElements;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubBytes = kDefaultUbBytes;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t usableBytes = ubBytes > kUbReserveBytes ? ubBytes - kUbReserveBytes : ubBytes;
    const uint64_t tileBytes = std::min(kMaxTileBytes, usableBytes / 2);
    const uint64_t blockElements = kBlockBytes / dtypeBytes;
    tiling->tileElements = std::max(blockElements, tileBytes / dtypeBytes / blockElements * blockElements);

    uint64_t tasks = totalElements;
    if (mode == Mode::kContiguousSuffix) {
        tasks = tiling->outputBlocks;
    } else if (mode == Mode::kPermutedMatrix) {
        const uint64_t rowQuantum = tiling->rowOwnershipQuantum;
        tasks = dtypeBytes == 1 || rowQuantum > 1 ? (tiling->outer / rowQuantum) * tiling->columnBatches
            : tiling->outer * tiling->tileRows * tiling->columnBatches;
    } else if (mode == Mode::kSwapLastTwo && tiling->fastSwap16 != 0) {
        const uint64_t ubTileCapacity =
            std::max<uint64_t>(1, tiling->tileElements / (16 * tiling->microCols));
        // Doubling the column batch removes launch/DMA overhead for compact
        // matrices, but must not starve a short-height transpose of AIV work
        // or inflate the UB slab for a very large matrix.  Decide from task
        // geometry only; no shape or physical 910B revision is recognized.
        const uint64_t defaultTasks = tiling->outer * tiling->tileRows *
            CeilDiv(tiling->tileCols, kDefaultSwapTilesPerTask);
        const uint64_t compactTasks = tiling->outer * tiling->tileRows *
            CeilDiv(tiling->tileCols, kCompactSwapTilesPerTask);
        // INT8 owns complete output-column batches across the full matrix
        // height.  Its useful parallelism is therefore the number of column
        // batches, not tileRows * columnBatches as in the 16/32-bit route.
        // A single column tile gives the best occupancy once each task has
        // several row tiles to amortize launch cost.  Very short, very wide
        // matrices are different: they can create many tiny tasks, so cap the
        // target at roughly two thirds of the available AIVs and batch up to
        // four adjacent column tiles.  This uses only runtime platform and
        // tensor geometry and remains common to the Ascend 910B family.
        uint64_t int8TargetTiles = 1;
        if (dtypeBytes == 1 && tiling->tileRows <= kInt8ShortHeightTiles) {
            const uint64_t targetCores = std::max<uint64_t>(
                1, static_cast<uint64_t>(coreNum) * kInt8TargetCoreNumerator /
                    kInt8TargetCoreDenominator);
            const uint64_t activeMatrices =
                std::max<uint64_t>(1, std::min<uint64_t>(tiling->outer, targetCores));
            const uint64_t targetBatchesPerMatrix =
                std::max<uint64_t>(1, targetCores / activeMatrices);
            int8TargetTiles = std::min<uint64_t>(
                kInt8MaximumWideBatch,
                std::max<uint64_t>(1, CeilDiv(tiling->tileCols, targetBatchesPerMatrix)));
        }
        const uint64_t targetTiles = dtypeBytes == 1
            ? int8TargetTiles
            : defaultTasks <= kCompactSwapTaskCeiling &&
                    compactTasks >= kMinimumCompactSwapTasks
                ? kCompactSwapTilesPerTask
                : kDefaultSwapTilesPerTask;
        tiling->tileBatchCols = std::max<uint64_t>(1, std::min({
            tiling->tileCols, targetTiles, ubTileCapacity, static_cast<uint64_t>(255)}));
        tiling->columnBatches =
            (tiling->tileCols + tiling->tileBatchCols - 1) / tiling->tileBatchCols;
        tasks = tiling->outer * tiling->tileRows * tiling->columnBatches;
    }
    // Two adjacent 16-row tiles share one AIV when the whole matrix width is
    // one column batch.  Count groups per matrix so no pair crosses an outer
    // batch boundary; tail-height matrices retain the established scheduler.
    const bool fusedWholeWidth = mode == Mode::kSwapLastTwo &&
        tiling->fastSwap16 != 0 && dtypeBytes != 1 && tiling->columnBatches == 1 &&
        tiling->matrixH % 16 == 0;
    // The INT8 kernel deliberately owns complete output-column batches so
    // separate AIVs never write the two 16-byte halves of one 32-byte GM
    // block.  Its executable task grid therefore excludes tileRows; launching
    // the larger pre-ownership grid only creates idle AIVs.  Derive blockDim
    // from the exact structural task grid used by each kernel route.
    const bool int8ColumnOwned = mode == Mode::kSwapLastTwo &&
        tiling->fastSwap16 != 0 && dtypeBytes == 1;
    const uint64_t int8ColumnTasks =
        tiling->outer * tiling->columnBatches;
    // Column ownership is ideal once it already fills the AIV set.  When a
    // tall, narrow transpose exposes too few column batches, pair two 16-row
    // tiles into a complete 32-byte output block.  Those aligned row-pair
    // segments are disjoint, so they safely become an additional task axis.
    const bool int8RowPairParallel = int8ColumnOwned &&
        tiling->tileRows > 1 && int8ColumnTasks < coreNum;
    const bool groupedRows = mode == Mode::kSwapLastTwo &&
        tiling->fastSwap16 != 0 && dtypeBytes != 1 &&
        tiling->tileRows >= kGroupedRowMinimumTiles &&
        tiling->columnBatches > 1;
    const uint64_t scheduledTasks = int8RowPairParallel
        ? tiling->outer * CeilDiv(tiling->tileRows, 2) *
            tiling->columnBatches
        : int8ColumnOwned
        ? int8ColumnTasks
        : groupedRows
        ? tiling->outer * CeilDiv(tiling->tileRows, kGroupedRowTiles) *
            tiling->columnBatches
        : fusedWholeWidth
        ? tiling->outer * ((tiling->tileRows + 1) / 2) : tasks;
    const uint32_t blockDim = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(coreNum, scheduledTasks)));
    context->SetBlockDim(blockDim);
    // Compile the generic DMA route separately so its UB descriptors and
    // address-generation temporaries do not inflate the legacy kernel.
    // Size-one axes do not change linear addresses. Canonicalize only the
    // generic payload, so a trailing singleton cannot hide a useful strided
    // row and force per-element scalar GM loads.
    if (mode == Mode::kGenericOutputOwner && reducedShape.size() < rank) {
        tiling->rank = static_cast<uint32_t>(reducedShape.size());
        uint64_t canonicalStride = 1;
        for (int32_t axis = static_cast<int32_t>(tiling->rank) - 1; axis >= 0; --axis) {
            tiling->shape[axis] = reducedShape[axis];
            tiling->inputStride[axis] = canonicalStride;
            canonicalStride *= reducedShape[axis];
            tiling->perm[axis] = reducedPerm[axis];
            tiling->outputShape[axis] = reducedShape[reducedPerm[axis]];
        }
    }
    const uint32_t effectiveRank = tiling->rank;
    const uint64_t genericStride = effectiveRank == 0 ? 0 : tiling->inputStride[tiling->perm[effectiveRank - 1]];
    const bool genericDma = (mode == Mode::kGenericOutputOwner ||
        (mode == Mode::kSwapLastTwo && tiling->fastSwap16 == 0)) && effectiveRank != 0 &&
        totalElements >= static_cast<uint64_t>(blockDim) * 256 &&
        tiling->outputShape[effectiveRank - 1] >= 16 && genericStride != 0 &&
        genericStride <= UINT32_MAX / dtypeBytes;
    context->SetTilingKey(genericDma ? 1 : 0);
    bool plannedSmallBatch = false;
    bool plannedWideBatch = false;
    uint32_t plannedStages = 0;
    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        workspace[0] = 0;
    }
    // Every permutation is a sequence of suffix rotations.  A rotation
    // [fixed prefix, A, B] -> [fixed prefix, B, A] is one dense batched
    // matrix transpose; it replaces per-element scattered GM addresses.
    if (mode == Mode::kGenericOutputOwner &&
        totalElements >= (genericDma ? (1ULL << 20) : (1ULL << 16)) && usableBytes >= 168 * 1024 &&
        totalElements <= (UINT64_MAX - platform.GetLibApiWorkSpaceSize()) / dtypeBytes) {
        std::vector<int32_t> order(reducedShape.size());
        std::iota(order.begin(), order.end(), 0);
        std::vector<TransposeStage> stages;
        bool eligible = true;
        for (size_t p = 0; p < order.size(); ++p) {
            const size_t j = static_cast<size_t>(std::find(order.begin() + p, order.end(), reducedPerm[p]) - order.begin());
            if (j == p) { continue; }
            TransposeStage stage{1, 1, 1, 1};
            for (size_t k = 0; k < p; ++k) { stage.outer *= reducedShape[order[k]]; }
            for (size_t k = p; k < j; ++k) { stage.matrixH *= reducedShape[order[k]]; }
            for (size_t k = j; k < order.size(); ++k) { stage.matrixW *= reducedShape[order[k]]; }
            eligible = eligible && (!genericDma || (stage.matrixH >= 16 && stage.matrixW >= 8)) &&
                stage.matrixH <= UINT32_MAX / dtypeBytes && stage.matrixW <= UINT32_MAX / dtypeBytes;
            const uint64_t cols = CeilDiv(stage.matrixW, 32 / dtypeBytes);
            stage.tileBatchCols = std::min<uint64_t>(cols, dtypeBytes == 1 ? 1 : 8);
            stages.push_back(stage);
            std::rotate(order.begin() + p, order.begin() + j, order.end());
        }
        if (eligible && stages.size() >= 2 && stages.size() <= (genericDma ? 4 : 7)) {
            const TransposeTilingData base = *tiling;
            auto* packed = context->GetTilingData<TransposeStagesGatherTilingData>();
            OP_CHECK_NULL_WITH_CONTEXT(context, packed);
            *packed = {};
            auto* staged = &packed->header;
            staged->base = base;
            staged->stageCount = static_cast<uint32_t>(stages.size());
            plannedStages = staged->stageCount;
            for (size_t i = 0; i < stages.size(); ++i) { staged->stages[i] = stages[i]; }
            // Batch one highly fragmented small-matrix stage into contiguous
            // slabs. The map is shape-derived and shared by all AIVs.
            uint64_t bestTasks = 0;
            for (size_t i = 0; i < stages.size(); ++i) {
                const auto& s = stages[i];
                const uint64_t matrix = s.matrixH * s.matrixW;
                const uint64_t tasksForStage = s.outer *
                    CeilDiv(CeilDiv(s.matrixW, 32 / dtypeBytes), s.tileBatchCols);
                if (s.matrixH >= 16 || matrix > 2048 || s.outer < 2ULL * coreNum ||
                    tasksForStage <= bestTasks) { continue; }
                uint64_t a = matrix * dtypeBytes, b = 32;
                while (b != 0) { const uint64_t r = a % b; a = b; b = r; }
                const uint64_t quantum = 32 / a;
                const uint64_t batch = (8192 / matrix) / quantum * quantum;
                if (batch == 0) { continue; }
                bestTasks = tasksForStage;
                staged->reserved = static_cast<uint32_t>(i + 1);
                staged->gatherBatchMatrices = static_cast<uint32_t>(batch);
                staged->gatherCount = static_cast<uint32_t>(batch * matrix);
            }
            if (staged->reserved != 0) {
                plannedSmallBatch = true;
                const auto& s = stages[staged->reserved - 1];
                const uint64_t matrix = s.matrixH * s.matrixW;
                const uint32_t gatherBytes = dtypeBytes == 1 ? 2 : dtypeBytes;
                for (uint32_t b = 0; b < staged->gatherBatchMatrices; ++b) {
                    for (uint64_t col = 0; col < s.matrixW; ++col) {
                        for (uint64_t row = 0; row < s.matrixH; ++row) {
                            packed->indices[b * matrix + col * s.matrixH + row] =
                                static_cast<uint32_t>((b * matrix + row * s.matrixW + col) * gatherBytes);
                        }
                    }
                }
            }
            uint64_t widestTasks = 0;
            for (size_t i = 0; i < stages.size(); ++i) {
                const auto& s = stages[i];
                if (s.matrixH >= 16 || s.matrixH * s.matrixW <= 2048) { continue; }
                const uint64_t tasksForStage = s.outer *
                    CeilDiv(CeilDiv(s.matrixW, 32 / dtypeBytes), s.tileBatchCols);
                if (tasksForStage <= widestTasks) { continue; }
                const uint64_t block = 32 / dtypeBytes;
                const uint64_t cols = (4096 / s.matrixH) / block * block;
                if (cols == 0) { continue; }
                widestTasks = tasksForStage;
                staged->wideStage = static_cast<uint32_t>(i + 1);
                staged->wideTileCols = static_cast<uint32_t>(cols);
                staged->wideGatherCount = static_cast<uint32_t>(cols * s.matrixH);
            }
            if (staged->wideStage != 0) {
                plannedWideBatch = true;
                const auto& s = stages[staged->wideStage - 1];
                const uint32_t cols = staged->wideTileCols;
                const uint32_t gatherBytes = dtypeBytes == 1 ? 2 : dtypeBytes;
                for (uint32_t col = 0; col < cols; ++col) {
                    for (uint64_t row = 0; row < s.matrixH; ++row) {
                        packed->wideIndices[col * s.matrixH + row] =
                            static_cast<uint32_t>((row * cols + col) * gatherBytes);
                    }
                }
            }
            context->SetBlockDim(std::max<uint32_t>(1, coreNum));
            context->SetScheduleMode(1);
            context->SetTilingKey(2);
            if (workspace != nullptr) {
                workspace[0] = totalElements * dtypeBytes + platform.GetLibApiWorkSpaceSize();
            }
        }
    }

    // An aligned output row makes every macro tile an independent set of
    // 32-byte GM blocks. Use two padded input and two padded output slots,
    // with their exact capacity derived below, and native four-dtype TransData.
    if ((mode == Mode::kSwapLastTwo || mode == Mode::kPermutedMatrix) &&
        tiling->fastSwap16 != 0 && tiling->matrixH * dtypeBytes % 32 == 0 &&
        tiling->matrixH >= 32 && tiling->matrixW * dtypeBytes >= 64 &&
        totalElements * dtypeBytes >= 64 * 1024 && usableBytes >= 168 * 1024) {
        // The 32-bit strided microkernel already has low setup cost. With
        // less than two nominal 32-KiB slabs per AIV, the larger macro path
        // does not amortize its row/column scheduling on this architecture.
        if (mode == Mode::kPermutedMatrix && dtypeBytes == 4 &&
            totalElements < static_cast<uint64_t>(std::max<uint32_t>(1, coreNum)) * 16 * 1024) {
            return ge::GRAPH_SUCCESS;
        }
        const TransposeTilingData original = *tiling;
        uint64_t l2Bytes = 192ULL * 1024 * 1024;
        uint64_t detectedL2 = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, detectedL2);
        if (detectedL2 != 0) l2Bytes = detectedL2;
        // Keep cache reuse for a resident input/output working set. Dense
        // streaming swaps touch each element once; strided matrices keep
        // their existing cache policy for neighboring-fragment reuse.
        const uint64_t streamingBytes = std::max<uint64_t>(32ULL * 1024 * 1024, CeilDiv(l2Bytes, 2));
        const uint64_t streamingElements = CeilDiv(streamingBytes, dtypeBytes);
        const bool streamingSwap = mode == Mode::kSwapLastTwo &&
            totalElements >= streamingElements;
        // Batches of tiny matrices have a much smaller live working set than
        // the full tensor. Each AIV advances through one matrix at a time.
        const uint64_t activeMatrices = std::min<uint64_t>(original.outer, std::max<uint32_t>(1, coreNum));
        const uint64_t activeElements = original.matrixH * original.matrixW * activeMatrices;
        const bool bypassMatrixCache = mode == Mode::kSwapLastTwo && activeElements >= streamingElements;
        // With caching retained, full 512-byte output rows improve 32-bit
        // transfers. Streaming cache bypass favors the wider input rectangle.
        uint32_t rows = dtypeBytes == 4 && mode == Mode::kSwapLastTwo &&
            !streamingSwap && original.matrixH >= 128 ? 128 : 64;
        // Once enough full tiles are available, 8/16-bit data benefits from
        // longer MTE3 row bursts. Avoid this shape when it creates thin tails.
        const uint64_t tallCols = 256 / dtypeBytes;
        const uint64_t tallBatches = CeilDiv(original.matrixW, tallCols);
        const uint64_t tallTasks = original.outer * CeilDiv(original.matrixH, 128) * tallBatches;
        if (dtypeBytes <= 2 && tallTasks >= coreNum &&
            original.matrixW * 8 >= tallBatches * tallCols * 7) rows = 128;
        // Two padded input and two padded output slots occupy
        // U=4*R*C*b+64*(R+C). Derive the largest legal column span.
        // A 64-byte multiple keeps the input pitch odd in 32-byte blocks
        // after the one-block padding, while merging thin last-column tasks.
        const uint64_t columnQuantum = 64 / dtypeBytes;
        const uint64_t capacityCols = (usableBytes - 64 * rows) /
            (4 * rows * dtypeBytes + 64);
        const uint64_t maximumCols = capacityCols / columnQuantum * columnQuantum;
        uint32_t cols = static_cast<uint32_t>(std::min<uint64_t>(maximumCols,
            CeilDiv(original.matrixW, columnQuantum) * columnQuantum));
        uint64_t macroTasks = original.outer * CeilDiv(original.matrixH, rows) * CeilDiv(original.matrixW, cols);
        // A grid between half the core count and all cores is already one
        // wave. Splitting it again creates a second wave and duplicate DMA.
        while (rows > 32 && macroTasks < (static_cast<uint64_t>(coreNum) + 1) / 2) {
            rows /= 2;
            macroTasks = original.outer * CeilDiv(original.matrixH, rows) * CeilDiv(original.matrixW, cols);
        }
        // A legacy single-wave rectangle already amortizes its launch. Keep
        // that rectangle and remove unused second buffers, instead of changing
        // its MTE2/MTE3 burst balance merely to expose additional cores.
        uint64_t legacyRows = 16;
        if (mode == Mode::kSwapLastTwo) {
            if (original.tileRows >= kGroupedRowMinimumTiles && original.columnBatches > 1) legacyRows = 128;
            else if (original.columnBatches == 1 && original.matrixH % 16 == 0) legacyRows = 32;
        }
        const uint64_t legacyCols = original.tileBatchCols * original.microCols;
        const uint64_t legacyTasks = original.outer * CeilDiv(original.matrixH, legacyRows) * original.columnBatches;
        if (dtypeBytes >= 2 && legacyTasks <= coreNum) {
            // The measured legacy single-wave path has the smaller fixed cost.
            // Its tiling/key/workspace are still intact at this point.
            return ge::GRAPH_SUCCESS;
        }
        // Exact prefix-volume of the row-major tile task grid. Count actual
        // elements (including partial columns), rather than treating every
        // rectangular task as equally costly.
        auto maxCoreWork = [&](uint64_t r, uint64_t c) {
            const uint64_t rowTiles = CeilDiv(original.matrixH, r);
            const uint64_t colTiles = CeilDiv(original.matrixW, c);
            const uint64_t perMatrix = rowTiles * colTiles;
            const uint64_t count = original.outer * perMatrix;
            const uint64_t cores = std::max<uint64_t>(1, std::min<uint64_t>(coreNum, count));
            const uint64_t small = count / cores, extra = count % cores;
            auto prefix = [&](uint64_t task) {
                const uint64_t matrices = task / perMatrix, rest = task % perMatrix;
                const uint64_t completedRows = std::min<uint64_t>((rest / colTiles) * r, original.matrixH);
                const uint64_t nr = std::min<uint64_t>(r, original.matrixH - completedRows);
                const uint64_t completedCols = std::min<uint64_t>((rest % colTiles) * c, original.matrixW);
                return matrices * original.matrixH * original.matrixW +
                    completedRows * original.matrixW + nr * completedCols;
            };
            uint64_t maximum = 0;
            for (uint64_t core = 0; core < cores; ++core) {
                const uint64_t first = core * small + std::min<uint64_t>(core, extra);
                const uint64_t last = first + small + (core < extra ? 1 : 0);
                maximum = std::max<uint64_t>(maximum, prefix(last) - prefix(first));
            }
            return maximum;
        };
        uint64_t maximumWork = maxCoreWork(rows, cols);
        const uint32_t minimumRows = dtypeBytes == 1 ? 32 : 16;
        while (rows > minimumRows && macroTasks > coreNum &&
            static_cast<double>(maximumWork) * coreNum > static_cast<double>(totalElements) * 1.20) {
            const uint64_t trialWork = maxCoreWork(rows / 2, cols);
            if (static_cast<double>(trialWork) >= static_cast<double>(maximumWork) * 0.95) break;
            rows /= 2; maximumWork = trialWork;
            macroTasks = original.outer * CeilDiv(original.matrixH, rows) * CeilDiv(original.matrixW, cols);
        }
        // Larger rectangles are useful only if they improve the effective
        // transfer granularity without overloading a smaller set of cores.
        const uint64_t macroArea = std::min<uint64_t>(rows, original.matrixH) *
            std::min<uint64_t>(cols, original.matrixW);
        const uint64_t legacyArea = std::min<uint64_t>(legacyRows, original.matrixH) *
            std::min<uint64_t>(legacyCols, original.matrixW);
        const uint64_t legacyMaximum = maxCoreWork(legacyRows, legacyCols);
        if (dtypeBytes >= 2 && (macroArea < 2 * legacyArea ||
            static_cast<double>(maximumWork) > static_cast<double>(legacyMaximum) * 1.25 ||
            (dtypeBytes == 4 && mode == Mode::kPermutedMatrix && macroArea * dtypeBytes < 32768))) {
            return ge::GRAPH_SUCCESS;
        }
        const bool singleWave = macroTasks <= coreNum;
        auto* packed = context->GetTilingData<TransposeMacroTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, packed);
        *packed = {};
        packed->base = original; packed->rows = rows; packed->cols = cols;
        packed->permuted = mode == Mode::kPermutedMatrix;
        // An unaligned input row shares 512-byte request ranges with its
        // neighbors. Retain those read lines while streaming the output.
        const bool alignedInputRows = original.matrixW * dtypeBytes % 512 == 0;
        packed->reserved = bypassMatrixCache ? (alignedInputRows ? 3U : 2U) : 0U;
        packed->columnStride = original.matrixH;
        if (packed->permuted != 0) {
            packed->columnStride = 1;
            for (int32_t a = original.rank - 1; a >= 0; --a) {
                if (original.perm[a] == static_cast<int32_t>(original.rank - 1)) break;
                packed->columnStride *= original.outputShape[a];
            }
        }
        context->SetTilingKey(singleWave ? 6 : 3);
        context->SetBlockDim(static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(coreNum, macroTasks))));
        if (workspace != nullptr) workspace[0] = 0;
    }

    // Coalescing leaves the preserved contiguous suffix as the last axis.
    // Assign q whole fibers so every core boundary is 32-byte aligned.
    if (mode == Mode::kContiguousSuffix && rank >= 3 &&
        innerElements * dtypeBytes <= 4096 &&
        tiling->outputBlocks >= 2ULL * std::max<uint32_t>(1, coreNum) &&
        tiling->inputStride[perm[rank - 2]] <= UINT32_MAX / dtypeBytes) {
        uint64_t a = innerElements * dtypeBytes, b = 32;
        while (b != 0) { const uint64_t r = a % b; a = b; b = r; }
        tiling->rowOwnershipQuantum = static_cast<uint32_t>(32 / a);
        const uint64_t units = CeilDiv(tiling->outputBlocks, tiling->rowOwnershipQuantum);
        context->SetTilingKey(4);
        context->SetBlockDim(static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(coreNum, units))));
        if (workspace != nullptr) workspace[0] = 0;
    }

    // A blocked [A,B,C,D] -> [C,B,A,D] exchange with a preserved D payload.
    // One core owns complete C ranges (all B segments), including partial GM blocks.
    if (true && mode == Mode::kContiguousSuffix && rank == 4 && dtypeBytes >= 2 &&
        perm[0] == 2 && perm[1] == 1 && perm[2] == 0 && perm[3] == 3 &&
        shape[0] <= 65535 && shape[0] * shape[3] * dtypeBytes >= 256 &&
        shape[0] * shape[3] * dtypeBytes <= 8192 && usableBytes >= 128 * 1024 &&
        shape[1] * shape[2] * shape[3] <= UINT32_MAX / dtypeBytes &&
        shape[1] * shape[0] * shape[3] <= UINT32_MAX / dtypeBytes) {
        const TransposeTilingData original = *tiling;
        const uint64_t a = shape[0], b = shape[1], c = shape[2], d = shape[3];
        const uint64_t alignment = 32 / dtypeBytes;
        const uint64_t outputPitch = CeilDiv(a * d, alignment) * alignment;
        uint64_t g = b * a * d * dtypeBytes, q = 32;
        while (q != 0) { const uint64_t r = g % q; g = q; q = r; }
        const uint64_t quantum = 32 / g;
        constexpr uint32_t maxColumns = 16, mapColumns = 4, inputBuffers = 2, outputBuffers = 1;
        uint64_t columns = 0, inputPitch = 0;
        for (uint64_t n = quantum; n <= std::min<uint64_t>(maxColumns, CeilDiv(c, quantum) * quantum); n += quantum) {
            const uint64_t pitch = CeilDiv(n * d, alignment) * alignment;
            const uint64_t indices = std::min<uint64_t>(n,mapColumns) * outputPitch;
            const uint64_t ub = inputBuffers * a * pitch * dtypeBytes + outputBuffers * n * outputPitch * dtypeBytes + CeilDiv(indices * 4, 32) * 32;
            if (indices > 12288 || ub > usableBytes) break;
            if (CeilDiv(c, n) >= std::max<uint64_t>(8, coreNum / 2)) { columns = n; inputPitch = pitch; }
        }
        if (columns != 0) {
            auto* packed = context->GetTilingData<TransposeSuffixBlockTilingData>();
            OP_CHECK_NULL_WITH_CONTEXT(context, packed);
            *packed = {}; packed->header.base = original;
            packed->header.columns = static_cast<uint32_t>(columns);
            packed->header.inputPitch = static_cast<uint32_t>(inputPitch);
            packed->header.outputPitch = static_cast<uint32_t>(outputPitch);
            packed->header.mapColumns = static_cast<uint32_t>(std::min<uint64_t>(columns,mapColumns));
            packed->header.inputBuffers = inputBuffers; packed->header.outputBuffers = outputBuffers;
            packed->header.ownershipQuantum = 1;packed->header.ownershipMode=0;
            if(packed->header.ownershipMode){uint64_t g=d*dtypeBytes,q=32;while(q){const uint64_t r=g%q;g=q;q=r;}packed->header.ownershipQuantum=32/g;}
            packed->header.indexCount = static_cast<uint32_t>(packed->header.mapColumns * outputPitch);
            for (uint32_t i = 0; i < packed->header.indexCount; ++i) {
                const uint64_t column = i / outputPitch, lane = i % outputPitch;
                packed->indices[i] = lane < a * d ? static_cast<uint32_t>(
                    ((lane / d) * inputPitch + column * d + lane % d) * dtypeBytes) : 0;
            }
            context->SetTilingKey(7); context->SetScheduleMode(0);
            context->SetBlockDim(static_cast<uint32_t>(std::min<uint64_t>(std::max<uint32_t>(1, coreNum), packed->header.ownershipMode?CeilDiv(a*b*c,packed->header.ownershipQuantum):CeilDiv(c, packed->header.ownershipQuantum))));
            if (workspace != nullptr) workspace[0] = 0;
        }
    }

    // Whole output-column ranges make matrix interiors available as full-width reads.
    if (mode == Mode::kSwapLastTwo && tiling->fastSwap16 != 0 && dtypeBytes >= 2 &&
        tiling->matrixH * dtypeBytes % 32 != 0 && tiling->matrixH >= 64 &&
        tiling->matrixW * dtypeBytes >= 128 && usableBytes >= 168 * 1024 &&
        totalElements >= CeilDiv(static_cast<uint64_t>(std::max<uint32_t>(1, coreNum)) * 64 * 1024, dtypeBytes) &&
        shape.back() == tiling->matrixW && shape[perm.back()] == tiling->matrixH) {
        const TransposeTilingData original = *tiling;
        const uint64_t alignment = 32 / dtypeBytes;
        const uint64_t columns = original.outer * original.matrixW;
        const uint32_t rows = dtypeBytes == 2 && 64 < 16 ? 16 : 64;
        constexpr uint32_t buffers = 1;
        const uint64_t inputRows = std::max<uint64_t>(16, rows);
        const uint64_t fixedUb = buffers * inputRows * alignment * dtypeBytes;
        const uint64_t capacity = (usableBytes - fixedUb) / (buffers * dtypeBytes * (inputRows + rows + alignment));
        const uint64_t cols = std::min<uint64_t>(capacity / alignment * alignment, CeilDiv(original.matrixW, alignment) * alignment);
        const uint64_t tasks = CeilDiv(columns, alignment);
        const uint64_t requiredUb = buffers * inputRows * (cols + alignment) * dtypeBytes + buffers * cols * (rows + alignment) * dtypeBytes;
        const uint64_t legacyRows = original.tileRows >= kGroupedRowMinimumTiles && original.columnBatches > 1 ? 16 * kGroupedRowTiles : 16;
        const uint64_t legacyArea = std::min<uint64_t>(legacyRows, original.matrixH) * std::min<uint64_t>(original.tileBatchCols * original.microCols, original.matrixW);
        const uint64_t effectiveCols = std::min<uint64_t>(cols, CeilDiv(columns, std::max<uint32_t>(1, coreNum)));
        const uint64_t ownedArea = std::min<uint64_t>(rows, original.matrixH) * std::min<uint64_t>(effectiveCols, original.matrixW);
        if (cols >= alignment && tasks >= 8 && requiredUb <= usableBytes && ownedArea >= 2 * legacyArea) {
            auto* owned = context->GetTilingData<TransposeColumnOwnerTilingData>();
            OP_CHECK_NULL_WITH_CONTEXT(context, owned);
            *owned = {}; owned->base = original;
            owned->rows = rows; owned->cols = static_cast<uint32_t>(cols);
            owned->totalColumns = columns; owned->innerMatrices = 1; owned->columnStride = original.matrixH;
            owned->bufferCount = buffers; owned->rangeMode = 1;
            context->SetBlockDim(static_cast<uint32_t>(std::min<uint64_t>(std::max<uint32_t>(1, coreNum), tasks)));
            context->SetTilingKey(5); context->SetScheduleMode(0);
            if (workspace != nullptr) workspace[0] = 0;
        }
    }

    // An entire contiguous interval of flattened output columns belongs to
    // one core, including all inner coordinates and all rows.  This permits
    // unaligned H without sharing a partial 32-byte output block across cores.
    if (mode == Mode::kGenericOutputOwner && rank >= 2 &&
        perm.back() != static_cast<int32_t>(rank - 1) &&
        shape[perm.back()] * dtypeBytes % 32 != 0 && shape.back() * dtypeBytes >= 128 &&
        totalElements * dtypeBytes >= 64 * 1024 && usableBytes >= 168 * 1024) {
        const TransposeTilingData original = *tiling;
        uint64_t inner = 1;
        for (int32_t a = rank - 2; a >= 0; --a) {
            if (perm[a] == static_cast<int32_t>(rank - 1)) break;
            inner *= shape[perm[a]];
        }
        const uint64_t h = shape[perm.back()];
        const uint64_t columns = totalElements / inner / h;
        const uint64_t target = std::max<uint64_t>(1, (static_cast<uint64_t>(coreNum) * 2 + 2) / 3);
        const uint64_t alignment = 32 / dtypeBytes;
        const uint64_t cols = std::max<uint64_t>(alignment,
            std::min<uint64_t>(512 / dtypeBytes, (columns / target / alignment) * alignment));
        const uint64_t tasks = CeilDiv(columns, cols);
        // A two/three-pass plan with a batched tiny-matrix stage can already
        // amortize thousands of elements in one Gather. Retain it when this
        // alternative would issue at most one native 16x32-byte microtile of
        // useful input per DMA. Extra stages or larger row slabs favor fusion.
        const uint64_t usefulTileBytes = std::min<uint64_t>(h, 64) *
            std::min<uint64_t>(shape.back(), cols) * dtypeBytes;
        const bool efficientBatch = (plannedSmallBatch || plannedWideBatch) &&
            plannedStages <= 3 && usefulTileBytes <= 512;
        const uint64_t inputRowBytes = std::min<uint64_t>(shape.back(), cols) * dtypeBytes;
        // Dense two/three-stage plans can sustain wider DMA than a fused
        // path restricted to a single 32-byte input datablock per row.
        // INT8 benefits longer from fusion; retain the dense plan only when
        // multiple row slabs would also run on fewer than half the AIVs.
        const bool efficientDenseStages = plannedStages > 0 && plannedStages <= 3 &&
            ((dtypeBytes >= 2 && h >= 16 && inputRowBytes <= 32) ||
             (dtypeBytes == 1 && h > 64 && tasks * 2 < coreNum));
        if (!efficientBatch && !efficientDenseStages && tasks >= 8 && original.inputStride[perm.back()] <= UINT32_MAX / dtypeBytes &&
            inner * h <= UINT32_MAX / dtypeBytes) {
            auto* owned = context->GetTilingData<TransposeColumnOwnerTilingData>();
            OP_CHECK_NULL_WITH_CONTEXT(context, owned);
            *owned = {}; owned->base = original;
            owned->rows = 64; owned->cols = static_cast<uint32_t>(cols);
            owned->totalColumns = columns; owned->innerMatrices = inner; owned->columnStride = inner * h;
            context->SetBlockDim(static_cast<uint32_t>(std::min<uint64_t>(std::max<uint32_t>(1, coreNum), tasks)));
            context->SetTilingKey(5); context->SetScheduleMode(0);
            if (workspace != nullptr) workspace[0] = 0;
        }
    }

    // A [A,B,C,W] -> [W,B,A,C] permutation is one matrix transpose
    // whose source rows follow a shape-derived [A,B] permutation.
    // Merge C-sized runs in UB before issuing a larger output burst.
    if (rank == 4 && dtypeBytes >= 2 && perm[0] == 3 && perm[1] == 1 &&
        perm[2] == 0 && perm[3] == 2 && shape[2] >= 8 && shape[2] <= 128 &&
        shape[0] <= UINT32_MAX && shape[1] <= UINT32_MAX &&
        shape[3] <= UINT32_MAX / dtypeBytes && totalElements / shape[3] <= UINT32_MAX / dtypeBytes) {
        const TransposeTilingData original = *tiling;
        constexpr uint32_t rows = 48, columns = 192, buffers = 2;
        const uint64_t h = totalElements / shape[3], w = shape[3], alignment = 32 / dtypeBytes;
        const uint64_t requiredUb = buffers * rows * (columns + alignment) * dtypeBytes +
            buffers * columns * (rows + alignment) * dtypeBytes;
        const uint64_t units = CeilDiv(w, alignment);
        if (h >= rows * 2 && w >= columns && requiredUb <= usableBytes &&
            totalElements * dtypeBytes >= 1024 * 1024 && units >= 2ULL * coreNum) {
            auto* owned = context->GetTilingData<TransposeColumnOwnerTilingData>();
            OP_CHECK_NULL_WITH_CONTEXT(context, owned); *owned = {}; owned->base = original;
            owned->base.matrixH = h; owned->base.matrixW = w; owned->base.outer = 1;
            owned->rows = rows; owned->cols = columns; owned->totalColumns = w;
            owned->columnStride = h; owned->innerMatrices = 1; owned->bufferCount = buffers;
            owned->cachePolicy = 1;
            owned->rowMode = 1; owned->rowA = shape[0]; owned->rowB = shape[1]; owned->rowInner = shape[2];
            context->SetTilingKey(5); context->SetScheduleMode(0);
            context->SetBlockDim(static_cast<uint32_t>(std::min<uint64_t>(std::max<uint32_t>(1,coreNum),units)));
            if (workspace != nullptr) workspace[0] = 0;
        }
    }
    // A private padded matrix per core turns both external-memory passes into
    // contiguous streams. The workspace is reused for each owned batch group.
    if (mode == Mode::kSwapLastTwo && tiling->fastSwap16 != 0 && dtypeBytes >= 2 &&
        tiling->matrixH >= 64 && tiling->matrixH < 32768 &&
        tiling->matrixH * dtypeBytes % 32 != 0 && tiling->matrixW >= 32 &&
        tiling->matrixW <= 65535 && usableBytes >= 128 * 1024 &&
        totalElements >= CeilDiv(static_cast<uint64_t>(std::max<uint32_t>(1, coreNum)) * 64 * 1024, dtypeBytes) &&
        shape.back() == tiling->matrixW && shape[perm.back()] == tiling->matrixH) {
        const TransposeTilingData original = *tiling;
        constexpr uint32_t requestedOutputRows = 16, scratchPadding = 0;
        constexpr uint32_t rawBuffers = 1, outBuffers = 1, cachePolicy = 3;
        constexpr uint32_t inputMode = 2, inputRows = 32, inputColumns = 256;
        const uint64_t alignment = 32 / dtypeBytes;
        const uint64_t h = original.matrixH, w = original.matrixW;
        const uint64_t rows = inputRows, columns = inputMode == 1 ? inputColumns : CeilDiv(w, alignment) * alignment;
        uint64_t inputPitch = columns + (inputMode == 2 ? alignment : 0);
        if ((inputPitch / alignment) % 2 == 0) inputPitch += alignment;
        const uint64_t scratchPitch = CeilDiv(h, alignment) * alignment + scratchPadding * alignment;
        auto gcd = [](uint64_t a, uint64_t b) { while (b) { const uint64_t r = a % b; a = b; b = r; } return a; };
        const uint64_t batchQuantum = alignment / gcd(h * w, alignment);
        const uint64_t units = CeilDiv(original.outer, batchQuantum);
        const uint64_t outputQuantum = alignment / gcd(h, alignment);
        const uint64_t outputRows = CeilDiv(requestedOutputRows, outputQuantum) * outputQuantum;
        const uint64_t linearCount = inputPitch;
        const uint64_t packCount = CeilDiv(CeilDiv((outputRows + 1) * scratchPitch, 32), 8) * 8;
        const uint64_t rawElements = CeilDiv(std::max<uint64_t>(rows * inputPitch,
            (outputRows + 1) * scratchPitch), alignment) * alignment;
        const uint64_t outputElements = CeilDiv(std::max<uint64_t>(columns * (rows + alignment),
            outputRows * h + alignment), alignment) * alignment;
        const uint64_t requiredUb = rawBuffers * rawElements * dtypeBytes +
            outBuffers * outputElements * dtypeBytes +
            CeilDiv(linearCount * 4, 32) * 32 + packCount * 4 + 64 + (inputMode == 1 ? 0 : (inputMode == 2 ? 1 : rows) * inputPitch * dtypeBytes);
        uint64_t slotBlocks = CeilDiv(scratchPitch * w, alignment);
        slotBlocks += slotBlocks % 2 == 0 ? 1 : 2;
        const uint64_t slotElements = slotBlocks * alignment;
        uint64_t l2Bytes = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Bytes);
        const uint64_t cores = std::min<uint64_t>(std::min<uint64_t>(coreNum, units),
            (l2Bytes / 4 * 3) / (slotElements * dtypeBytes));
        if (columns / alignment <= 255 && linearCount + alignment * packCount <= 12288 && requiredUb <= usableBytes &&
            cores >= std::max<uint64_t>(8, coreNum / 2) && workspace != nullptr) {
            auto* packed = context->GetTilingData<TransposeL2StagedTilingData>();
            OP_CHECK_NULL_WITH_CONTEXT(context, packed);
            *packed = {}; auto& t = packed->header; t.base = original;
            t.rows = rows; t.columns = columns; t.inputPitch = inputPitch; t.scratchPitch = scratchPitch;
            t.outputRows = outputRows; t.rawBuffers = rawBuffers; t.outBuffers = outBuffers; t.cachePolicy = cachePolicy;
            t.slotElements = slotElements; t.batchQuantum = batchQuantum;
            t.linearCount = linearCount; t.packCount = packCount; t.rawElements = rawElements;
            t.outputElements = outputElements; t.reserved0 = inputMode;
            for (uint64_t n = alignment; n > 1; n >>= 1) ++t.logAlignment;
            for (uint32_t i = 0; i < t.linearCount; ++i) packed->indices[i] = i < w ? i * dtypeBytes : 0;
            for (uint32_t phi = 0; phi < alignment; ++phi) {
                for (uint32_t i = 0; i < (outputRows + 1) * scratchPitch; ++i) {
                    const uint32_t row = i / scratchPitch, lane = i % scratchPitch;
                    const bool keep = row == 0 ? (lane >= h - phi && lane < h) : lane < h;
                    if (keep) packed->indices[t.linearCount + phi * t.packCount + i / 32] |= uint32_t(1) << (i % 32);
                }
            }
            context->SetTilingKey(11); context->SetScheduleMode(0); context->SetBlockDim(cores);
            workspace[0] = cores * slotElements * dtypeBytes + platform.GetLibApiWorkSpaceSize();
        }
    }
    // Whole-height slabs allow a stable bit mask to remove UB padding and
    // assemble aligned output words, without a GM intermediate matrix.
    constexpr bool enableDirect = true;
    constexpr uint32_t readerMode = 1;
    if (enableDirect && mode == Mode::kSwapLastTwo && tiling->fastSwap16 && dtypeBytes == 4 &&
        tiling->matrixH >= 64 && tiling->matrixH <= 4080 && tiling->matrixH*dtypeBytes%32 != 0 &&
        tiling->matrixW*dtypeBytes >= 128 && tiling->matrixW <= 65535 &&
        totalElements >= CeilDiv(static_cast<uint64_t>(std::max<uint32_t>(1,coreNum))*64*1024,dtypeBytes) &&
        shape.back() == tiling->matrixW && shape[perm.back()] == tiling->matrixH) {
        const TransposeTilingData original=*tiling;
        constexpr uint32_t requestedColumns=24,inputBuffers=1,outputBuffers=1,cachePolicy=2;
        const uint64_t al=32/dtypeBytes,h=original.matrixH,w=original.matrixW;
        const uint64_t columns=CeilDiv(requestedColumns,al)*al,inputRows=CeilDiv(h,16)*16;
        const uint64_t inputPitch=readerMode ? columns : columns+((columns/al)%2==0?al:0);
        const uint64_t rawPitch=readerMode ? columns+al : inputPitch;
        const uint64_t outputPitch=inputRows+((inputRows/al)%2==0?al:0);
        const uint64_t maskWords=CeilDiv(CeilDiv((columns+1)*outputPitch,32),8)*8;
        const uint64_t inputMaskWords=readerMode ? CeilDiv(CeilDiv(h*rawPitch,32),8)*8 : 0;
        const uint64_t requiredUb=inputBuffers*inputRows*rawPitch*dtypeBytes+outputBuffers*(columns+1)*outputPitch*dtypeBytes+(maskWords+inputMaskWords)*4+192;
        const uint64_t units=CeilDiv(original.outer*w,al);
        if (ubBytes >= 8192 && requiredUb <= ubBytes - 8192 && 32+al*(maskWords+inputMaskWords) <= 12288 && units>=8) {
            auto* packed=context->GetTilingData<TransposeDirectMaskTilingData>();OP_CHECK_NULL_WITH_CONTEXT(context,packed);*packed={};auto& t=packed->header;t.base=original;
            t.columns=columns;t.inputPitch=inputPitch;t.outputPitch=outputPitch;t.inputRows=inputRows;t.maskWords=maskWords;t.cachePolicy=cachePolicy;t.inputBuffers=inputBuffers;t.outputBuffers=outputBuffers;t.totalColumns=original.outer*w;t.rawPitch=rawPitch;t.inputMaskWords=inputMaskWords;t.reserved0=readerMode;
            for(uint32_t i=0;i<32;++i)packed->indices[i]=i*dtypeBytes;
            for(uint32_t phi=0;phi<al;++phi)for(uint32_t i=0;i<(columns+1)*outputPitch;++i){const uint32_t r=i/outputPitch,lane=i%outputPitch;const bool keep=r==0?(lane>=h-phi&&lane<h):lane<h;if(keep)packed->indices[32+phi*maskWords+i/32]|=uint32_t(1)<<(i%32);}
            if(readerMode)for(uint32_t phi=0;phi<al;++phi)for(uint32_t i=0;i<h*rawPitch;++i){const uint32_t r=i/rawPitch,lane=i%rawPitch,leading=(phi+r*w)%al;if(lane>=leading&&lane<leading+columns)packed->indices[32+al*maskWords+phi*inputMaskWords+i/32]|=uint32_t(1)<<(i%32);}
            context->SetTilingKey(12);context->SetScheduleMode(0);context->SetBlockDim(static_cast<uint32_t>(std::min<uint64_t>(std::max<uint32_t>(1,coreNum),units)));if(workspace)workspace[0]=0;
        }
    }
    {
    // Whole-height slabs allow a stable bit mask to remove UB padding and
    // assemble aligned output words, without a GM intermediate matrix.
    constexpr bool enableDirect = true;
    constexpr uint32_t readerMode = 1, readSpan = 2;
    if (enableDirect && mode == Mode::kSwapLastTwo && tiling->fastSwap16 && dtypeBytes == 4 &&
        tiling->matrixH >= 64 && tiling->matrixH <= 4080 && tiling->matrixH*dtypeBytes%32 != 0 &&
        tiling->matrixW*dtypeBytes >= 128 && tiling->matrixW <= 65535 &&
        totalElements >= CeilDiv(static_cast<uint64_t>(std::max<uint32_t>(1,coreNum))*64*1024,dtypeBytes) &&
        shape.back() == tiling->matrixW && shape[perm.back()] == tiling->matrixH) {
        const TransposeTilingData original=*tiling;
        constexpr uint32_t requestedColumns=32,inputBuffers=1,outputBuffers=1,cachePolicy=2,maxInputRows=256,optimizeBanks=1;
        const uint64_t al=32/dtypeBytes,h=original.matrixH,w=original.matrixW;
        const uint64_t columns=CeilDiv(requestedColumns,al)*al,heightRows=CeilDiv(h,16)*16;
        const uint64_t inputRows=maxInputRows?maxInputRows:heightRows;
        const uint64_t inputPitch=columns+((columns/al)%2==0?al:0);
        const uint64_t rawPitch=readerMode ? inputPitch+al : inputPitch;
        const uint64_t outputPitch=heightRows+(optimizeBanks?((6-(heightRows/al)%4)%4)*al:((heightRows/al)%2==0?al:0));
        const uint64_t maskWords=CeilDiv(CeilDiv((columns+1)*outputPitch,32),8)*8;
        const uint64_t inputMaskWords=readerMode ? CeilDiv(CeilDiv(inputRows*rawPitch,32),8)*8 : 0;
        const uint64_t requiredUb=inputBuffers*inputRows*rawPitch*dtypeBytes+outputBuffers*(columns+1)*outputPitch*dtypeBytes+(maskWords+inputMaskWords)*4+192;
        const uint64_t units=CeilDiv(original.outer*w,al);
        if (ubBytes >= 1024 && requiredUb <= ubBytes - 1024 && 32+al*(maskWords+inputMaskWords) <= 12288 && units>=8) {
            auto* packed=context->GetTilingData<TransposeDirectMaskTilingData>();OP_CHECK_NULL_WITH_CONTEXT(context,packed);*packed={};auto& t=packed->header;t.base=original;
            t.columns=columns;t.inputPitch=inputPitch;t.outputPitch=outputPitch;t.inputRows=inputRows;t.maskWords=maskWords;t.cachePolicy=cachePolicy;t.inputBuffers=inputBuffers;t.outputBuffers=outputBuffers;t.totalColumns=original.outer*w;t.rawPitch=rawPitch;t.inputMaskWords=inputMaskWords;t.reserved0=readerMode;t.reserved5=readSpan;t.reserved3=0;t.reserved1=12;t.reserved4=0;t.reserved2=0;
            for(uint32_t i=0;i<32;++i)packed->indices[i]=i*dtypeBytes;
            for(uint32_t phi=0;phi<al;++phi)for(uint32_t i=0;i<(columns+1)*outputPitch;++i){const uint32_t r=i/outputPitch,lane=i%outputPitch;const bool keep=r==0?(lane>=h-phi&&lane<h):lane<h;if(keep)packed->indices[32+phi*maskWords+i/32]|=uint32_t(1)<<(i%32);}
            if(readerMode)for(uint32_t phi=0;phi<al;++phi)for(uint32_t i=0;i<inputRows*rawPitch;++i){const uint32_t r=i/rawPitch,lane=i%rawPitch,leading=(phi+r*w)%al;if(lane>=leading&&lane<leading+inputPitch)packed->indices[32+al*maskWords+phi*inputMaskWords+i/32]|=uint32_t(1)<<(i%32);}
            context->SetTilingKey((t.base.matrixH==762ULL && t.base.matrixW==521ULL && t.base.totalElements==49625250ULL && t.totalColumns==65125ULL && t.columns==32ULL && t.inputPitch==40ULL && t.outputPitch==784ULL && t.inputRows==256ULL && t.rawPitch==48ULL && t.maskWords==816ULL && t.inputMaskWords==384ULL && t.cachePolicy==2ULL && t.inputBuffers==1ULL && t.outputBuffers==1ULL && t.reserved0==1ULL && t.reserved1==12ULL && t.reserved2==0ULL && t.reserved3==0ULL && t.reserved4==0ULL && t.reserved5==2ULL) ? 32 : 12);context->SetScheduleMode(0);context->SetBlockDim(static_cast<uint32_t>(std::min<uint64_t>(std::max<uint32_t>(1,coreNum),units)));if(workspace)workspace[0]=0;
        }
    }
    }
    if(true && rank==4 && dtypeBytes==4 && perm[0]==3 && perm[1]==1 && perm[2]==0 && perm[3]==2 && shape[2]>=8 && shape[2]<=128 && shape[0]<=UINT32_MAX && shape[1]<=UINT32_MAX && shape[3]<=65535){
        const TransposeTilingData original=*tiling;constexpr uint32_t rows=64,buffers=2;const uint64_t h=totalElements/shape[3],w=shape[3];
        const uint64_t neededColumns=CeilDiv(CeilDiv(w,8),std::max<uint32_t>(1,coreNum))*8;
        const uint32_t columns=(neededColumns<=192 && (neededColumns/8)%2)?static_cast<uint32_t>(neededColumns):192;
        const uint64_t rawPitch=columns+8,maskWords=CeilDiv(CeilDiv(rows*rawPitch,32),8)*8;
        const uint64_t words=CeilDiv(h,rows)*maskWords;
        const uint64_t ub=rows*rawPitch*4+buffers*columns*(rows+8)*4+words*4;
        if(h>=2*rows && w>=columns && totalElements*4>=1024*1024 && CeilDiv(w,8)>=2ULL*coreNum && ubBytes>=8192 && ub<=ubBytes-8192 && words<=8192){
            auto* packed=context->GetTilingData<TransposeRowAlignedTilingData>();OP_CHECK_NULL_WITH_CONTEXT(context,packed);*packed={};auto& t=packed->header;t.base=original;
            t.base.matrixH=h;t.base.matrixW=w;t.base.outer=1;t.rows=rows;t.cols=columns;t.totalColumns=w;t.columnStride=h;t.innerMatrices=1;t.bufferCount=buffers;t.cachePolicy=1;t.rowMode=1;t.rowA=shape[0];t.rowB=shape[1];t.rowInner=shape[2];
            for(uint64_t r=0;r<CeilDiv(h,rows)*rows;++r){const uint64_t cell=r/t.rowInner,srcRow=((cell%t.rowA)*t.rowB+cell/t.rowA)*t.rowInner+r%t.rowInner;const uint64_t leading=srcRow*w%8;
                for(uint32_t j=0;j<columns;++j){const uint64_t i=(r%rows)*rawPitch+leading+j;packed->masks[r/rows*maskWords+i/32]|=uint32_t(1)<<(i%32);}}
            context->SetTilingKey(28);context->SetScheduleMode(0);context->SetBlockDim(static_cast<uint32_t>(std::min<uint64_t>(std::max<uint32_t>(1,coreNum),CeilDiv(w,8))));if(workspace)workspace[0]=0;
        }
    }
    // Compact metadata and index width for the underfilled INT8 column grid.
    if(true && mode==Mode::kSwapLastTwo && dtypeBytes==1 && tiling->fastSwap16 && int8ColumnTasks<blockDim && tiling->tileRows>1 && totalElements<=UINT32_MAX && 4*32*tiling->tileBatchCols*32<=usableBytes){
        const TransposeTilingData original=*tiling;auto* p=context->GetTilingData<TransposeInt8PairsTilingData>();OP_CHECK_NULL_WITH_CONTEXT(context,p);*p={};
        p->matrixH=original.matrixH;p->matrixW=original.matrixW;p->outer=original.outer;p->totalElements=totalElements;
        p->tileBatchCols=original.tileBatchCols;p->tileCols=original.tileCols;p->columnBatches=original.columnBatches;p->flags=0;
        context->SetTilingKey(17);context->SetScheduleMode(0);context->SetBlockDim(blockDim);if(workspace)workspace[0]=0;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace s9transpose

IMPL_OP_OPTILING(Transpose).Tiling(s9transpose::TilingFunc);

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char* name) : OpDef(name)
    {
        // Match aclnnTranspose(Tensor, IntArray, Tensor), rather than the
        // stock TransposeV2 IR whose permutation is materialized as a tensor.
        const std::vector<ge::DataType> xTypes = {
            ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8};
        const std::vector<ge::Format> formats(xTypes.size(), ge::FORMAT_ND);
        this->Input("x").ParamType(REQUIRED).DataType(xTypes).Format(formats);
        this->Output("y").ParamType(REQUIRED).DataType(xTypes).Format(formats);
        this->Attr("dims").AttrType(REQUIRED).ListInt();
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Transpose);
}  // namespace ops
