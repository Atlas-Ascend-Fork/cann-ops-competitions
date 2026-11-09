#include "transpose_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {
static uint64_t CeilDiv(uint64_t a, uint64_t b) { return a / b + (a % b != 0); }

static ge::graphStatus TilingFunc(gert::TilingContext *ctx) {
  const auto &shape = ctx->GetInputShape(0)->GetStorageShape();
  const auto *perm = ctx->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0);
  const size_t rank = shape.GetDimNum();
  if (!perm || rank > 64 || perm->GetSize() != rank) return ge::GRAPH_FAILED;
  const auto *axes = static_cast<const int64_t *>(perm->GetData());
  uint64_t strides[64] = {}, dims[64] = {}, mapped[64] = {}, outStrides[64] = {};
  bool used[64] = {};
  uint64_t total = 1;
  for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
    if (shape.GetDim(i) < 0) return ge::GRAPH_FAILED;
    strides[i] = total;
    if (shape.GetDim(i) != 0 && total > std::numeric_limits<uint64_t>::max() / shape.GetDim(i))
      return ge::GRAPH_FAILED;
    total *= shape.GetDim(i);
  }
  uint32_t reducedRank = 0;
  for (size_t i = 0; i < rank; ++i) {
    int64_t axis = axes[i] < 0 ? axes[i] + rank : axes[i];
    if (axis < 0 || axis >= static_cast<int64_t>(rank) || used[axis]) return ge::GRAPH_FAILED;
    used[axis] = true;
    const uint64_t dim = shape.GetDim(axis);
    if (dim == 1) continue;
    // Merge only when both input and output describe the same contiguous run.
    if (reducedRank && mapped[reducedRank - 1] == strides[axis] * dim) {
      dims[reducedRank - 1] *= dim;
      mapped[reducedRank - 1] = strides[axis];
    } else {
      dims[reducedRank] = dim;
      mapped[reducedRank++] = strides[axis];
    }
  }
  if (reducedRank == 0 || total == 0) {
    reducedRank = 1;
    dims[0] = total == 0 ? 0 : 1;
    mapped[0] = 1;
  }
  uint64_t outputStride = 1;
  for (int32_t i = static_cast<int32_t>(reducedRank) - 1; i >= 0; --i) {
    outStrides[i] = outputStride;
    outputStride *= dims[i];
  }

  uint32_t elementBytes = 0;
  switch (ctx->GetInputDesc(0)->GetDataType()) {
    case ge::DT_FLOAT16: elementBytes = 2; break;
    case ge::DT_FLOAT:
    case ge::DT_INT32: elementBytes = 4; break;
    case ge::DT_INT8: elementBytes = 1; break;
    default: return ge::GRAPH_FAILED;
  }
  platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
  const uint32_t coreCount = std::max(1u, static_cast<uint32_t>(platform.GetCoreNumAiv()));
  uint64_t ubSize = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  // These kernels use only UB; retain a small margin for compiler/runtime needs.
  if (ubSize < 8192) return ge::GRAPH_FAILED;
  const uint64_t ubBudget = ubSize - 1024;

  const uint32_t last = reducedRank - 1;
  const uint32_t mode = mapped[last] == 1 ? 0 : 1;
  uint32_t axisRows = last, axisCols = last;
  uint64_t planeRows = 1, planeCols = dims[last];
  uint64_t inputRowStride = dims[last], outputRowStride = dims[last];
  if (mode == 0) {
    // A preserved trailing run is copied in batches of rows, including short rows.
    if (last > 0) {
      axisRows = last - 1;
      planeRows = dims[axisRows];
      inputRowStride = mapped[axisRows];
    }
  } else {
    // Every non-identity permutation contains an input-contiguous axis.
    axisCols = 0;
    while (axisCols < reducedRank && mapped[axisCols] != 1) ++axisCols;
    if (axisCols == reducedRank) return ge::GRAPH_FAILED;
    planeRows = dims[last];
    planeCols = dims[axisCols];
    inputRowStride = mapped[last];
    outputRowStride = outStrides[axisCols];
  }
  uint64_t outerDims[64] = {}, outerInput[64] = {}, outerOutput[64] = {};
  uint32_t outerRank = 0;
  uint64_t outerCount = 1;
  for (uint32_t i = 0; i < reducedRank; ++i) {
    if (i == axisRows || i == axisCols) continue;
    outerDims[outerRank] = dims[i];
    outerInput[outerRank] = mapped[i];
    outerOutput[outerRank++] = outStrides[i];
    outerCount *= dims[i];
  }

  // Matrix rows of 32-bit elements benefit from longer output DMA rows.
  // Large byte permutations benefit from consecutive work on each core, while
  // retaining both queue slots to overlap transfers. Other layouts use the baseline.
  uint32_t executionMode = mode;
  if (mode && elementBytes == 4 && outerRank == 0) executionMode = 2;
  else if (mode && elementBytes == 1 && outerRank > 0 && total >= 16ull * 1024 * 1024)
    executionMode = 3;
  const uint32_t buffers = executionMode == 2 ? 2 : 4;
  uint32_t tileRows = mode ? 32 : 1;
  uint32_t tileCols = mode ? 32 : static_cast<uint32_t>(std::min<uint64_t>(
      CeilDiv(std::max<uint64_t>(planeCols, 1), 32 / elementBytes) * (32 / elementBytes),
      std::min<uint64_t>(65536 / elementBytes, ubBudget / 32 * (32 / elementBytes))));
  if (mode == 0) {
    // Split large copies across cores; all tile boundaries remain 32-byte aligned.
    const uint64_t target = CeilDiv(std::max<uint64_t>(total, 1), coreCount);
    const uint64_t alignedTarget = CeilDiv(target, 32 / elementBytes) * (32 / elementBytes);
    tileCols = static_cast<uint32_t>(std::min<uint64_t>(tileCols, std::max<uint64_t>(32 / elementBytes, alignedTarget)));
    tileRows = static_cast<uint32_t>(std::min<uint64_t>(std::min<uint64_t>(planeRows, 4095),
        ubBudget / (tileCols * elementBytes)));
    tileRows = std::max(1u, tileRows);
    const uint64_t rowTasks = outerCount * CeilDiv(std::max<uint64_t>(planeCols, 1), tileCols);
    const uint64_t wanted = CeilDiv(coreCount, std::max<uint64_t>(rowTasks, 1));
    tileRows = static_cast<uint32_t>(std::min<uint64_t>(tileRows,
        std::max<uint64_t>(1, CeilDiv(planeRows, std::max<uint64_t>(wanted, 1)))));
  } else {
    const uint64_t desiredTasks = std::min<uint64_t>(coreCount, std::max<uint64_t>(1, CeilDiv(total, 1024)));
    // Grow the column tile first: each repeat handles another complete 32-byte
    // input segment. The selected queue depth bounds the combined input/output storage.
    bool changed = true;
    while (changed) {
      changed = false;
      for (uint32_t direction = 0; direction < 2; ++direction) {
        const uint32_t nextRows = direction ? tileRows * 2 : tileRows;
        const uint32_t nextCols = direction ? tileCols : tileCols * 2;
        if (nextRows > 256 || nextCols > 1024 ||
            static_cast<uint64_t>(buffers) * nextRows * nextCols * elementBytes > ubBudget) continue;
        if ((direction && tileRows >= planeRows) || (!direction && tileCols >= planeCols)) continue;
        const uint64_t tiles = outerCount * CeilDiv(planeRows, nextRows) * CeilDiv(planeCols, nextCols);
        if (tiles < desiredTasks) continue;
        tileRows = nextRows;
        tileCols = nextCols;
        changed = true;
      }
    }
  }
  if (mode && outerRank > 0 && total >= 1024ull * 1024) {
    // Keep a moderate-width input-contiguous axis in one DMA row. A single
    // input/output buffer pair can hold wider odd-sized planes than the
    // power-of-two double-buffer tiles, avoiding separate one-column tails.
    const uint64_t rowAlignment = elementBytes == 1 ? 32 : 16;
    const uint64_t columnAlignment = 32 / elementBytes;
    const uint64_t fullCols = CeilDiv(planeCols, columnAlignment) * columnAlignment;
    if (fullCols <= 1024) {
      const uint64_t rowsFit = ubBudget / (2 * fullCols * elementBytes) / rowAlignment * rowAlignment;
      const uint64_t candidateRows = std::min<uint64_t>(256,
          std::min<uint64_t>(rowsFit, CeilDiv(planeRows, rowAlignment) * rowAlignment));
      if (candidateRows >= planeRows &&
          outerCount * CeilDiv(planeRows, candidateRows) >= coreCount) {
        // Full-plane geometry does not always require single buffering.
        // Keep two input/output slots when all four buffers fit the actual UB.
        executionMode = 4 * candidateRows * fullCols * elementBytes <= ubBudget ? 3 : 2;
        tileRows = static_cast<uint32_t>(candidateRows);
        tileCols = static_cast<uint32_t>(fullCols);
      }
    }
  }
  if (mode && elementBytes == 4 && outerRank > 0 && outerCount >= coreCount &&
      total >= 1024ull * 1024 && inputRowStride == planeCols &&
      planeCols >= 64 && planeCols <= 1024 && (planeCols * elementBytes) % 32 != 0) {
    const uint64_t fullRows = CeilDiv(planeRows, 8) * 8;
    if (fullRows <= 256) {
      // Keep a whole compact input plane in UB; gather full output rows in
      // column groups. One input, two output slots, and a reusable index table.
      const uint64_t inputBytes = fullRows * planeCols * elementBytes;
      if (inputBytes < ubBudget && 2 * inputBytes > ubBudget) {
        const uint64_t colsFit = (ubBudget - inputBytes) / (3 * fullRows * elementBytes) / 8 * 8;
        const uint64_t outputCols = std::min<uint64_t>(128,
            std::min<uint64_t>(colsFit, CeilDiv(planeCols, 8) * 8));
        if (outputCols >= 8) {
          executionMode = 6;
          tileRows = static_cast<uint32_t>(fullRows);
          tileCols = static_cast<uint32_t>(outputCols);
        }
      }
    }
  }
  // Narrow compact rows waste both UB and DMA issue bandwidth when each row
  // is padded independently. Gather from a contiguous window instead. Keep
  // the entire plane on one core and bound every buffer by the actual UB.
  if (mode && (elementBytes == 2 || elementBytes == 4) && outerCount >= coreCount &&
      total >= 1024ull * 1024 && inputRowStride == planeCols &&
      planeCols >= 8 && planeCols < 64 && planeRows >= 512) {
    const uint64_t paddedCols = CeilDiv(planeCols, 8) * 8;
    const uint64_t rowsFit = ubBudget / (2 * planeCols * elementBytes + (2 * elementBytes + 4) * paddedCols) / 32 * 32;
    const uint64_t windowRows = std::min<uint64_t>(1024, rowsFit);
    if (windowRows >= 128) {
      executionMode = 7;
      tileRows = static_cast<uint32_t>(windowRows);
      tileCols = static_cast<uint32_t>(paddedCols);
    }
  }
  if (mode && elementBytes == 1 && outerCount >= coreCount &&
      total >= 1024ull * 1024 && inputRowStride == planeCols &&
      planeCols >= 8 && planeCols < 64 && planeRows >= 512) {
    const uint64_t paddedCols = CeilDiv(planeCols, 8) * 8;
    // Two byte inputs, one half input, two byte outputs, one half output,
    // and one uint32 index table. Promotion is exact for every int8 value.
    const uint64_t rowsFit = ubBudget / (4 * planeCols + 8 * paddedCols) / 32 * 32;
    const uint64_t windowRows = std::min<uint64_t>(1024, rowsFit);
    if (windowRows >= 128) {
      executionMode = 8;
      tileRows = static_cast<uint32_t>(windowRows);
      tileCols = static_cast<uint32_t>(paddedCols);
    }
  }
  // On a matrix, favor longer output rows while fitting one input/output
  // slot pair. Keep enough independent tiles to occupy the available cores.
  if (mode && elementBytes == 2 && outerRank == 0 &&
      total >= 1024ull * 1024 && planeRows >= 128 && planeCols >= 128) {
    const uint64_t rows = std::min<uint64_t>(256, CeilDiv(planeRows, 16) * 16);
    const uint64_t columnAlignment = 32 / elementBytes;
    const uint64_t colsFit = ubBudget / (2 * rows * elementBytes) / columnAlignment * columnAlignment;
    const uint64_t cols = std::min<uint64_t>(1024,
        std::min<uint64_t>(CeilDiv(planeCols, columnAlignment) * columnAlignment, colsFit));
    if (cols >= 64 && CeilDiv(planeRows, rows) * CeilDiv(planeCols, cols) >= coreCount) {
      executionMode = 2;
      tileRows = static_cast<uint32_t>(rows);
      tileCols = static_cast<uint32_t>(cols);
    }
  }
  // On memory-sized 32-bit matrices, retain the baseline when output
  // row starts share a 512-byte transaction boundary. Otherwise use longer
  // output spans, selected by input stride and bounded row-tail overhead.
  // This is based on independent aligned, mixed and odd-layout measurements.
  if (mode && elementBytes == 4 && outerRank == 0 &&
      total >= 16ull * 1024 * 1024 && planeRows % (512 / elementBytes) != 0) {
    uint64_t rows = planeCols % (512 / elementBytes) == 0 ? 256 : 512;
    while (rows > 128 && (rows - planeRows % rows) % rows > planeRows / 4) {
      rows /= 2;
    }
    const uint64_t columns = 16384 / rows;
    if (planeRows >= rows && planeCols >= columns &&
        2 * rows * columns * elementBytes <= ubBudget &&
        CeilDiv(planeRows, rows) * CeilDiv(planeCols, columns) >= coreCount) {
      executionMode = 2;
      tileRows = static_cast<uint32_t>(rows);
      tileCols = static_cast<uint32_t>(columns);
    }
  }
  // Concatenate output-adjacent planes into a taller local matrix. This
  // increases each output DMA length while preserving exact GM ownership.
  uint32_t planeGroup = 1;
  uint64_t groupedTasks = 0;
  // A fully resident double-buffered plane already overlaps all stages;
  // preserve it instead of splitting each input plane into column windows.
  if (mode && executionMode != 3 && elementBytes == 4 && total >= 1024ull * 1024 &&
      outerRank > 0 && outerOutput[outerRank - 1] == planeRows &&
      planeRows >= 32 && planeRows <= 256 && planeCols >= 64 && planeCols <= 1024) {
    const uint64_t adjacentPlanes = outerDims[outerRank - 1];
    const uint64_t prefixPlanes = outerCount / adjacentPlanes;
    for (uint32_t group = 8; group >= 2; group /= 2) {
      if (adjacentPlanes < group) continue;
      const uint64_t groups = prefixPlanes * CeilDiv(adjacentPlanes, group);
      if (groups < coreCount) continue;
      const uint64_t mergedRows = CeilDiv(planeRows * group, 16) * 16;
      const uint64_t colsFit = ubBudget / (2 * mergedRows * elementBytes) / 8 * 8;
      const uint64_t columns = std::min<uint64_t>(CeilDiv(planeCols, 8) * 8,
          std::min<uint64_t>(1024, colsFit));
      if (columns < 64) continue;
      executionMode = 9;
      planeGroup = group;
      tileRows = static_cast<uint32_t>(mergedRows);
      tileCols = static_cast<uint32_t>(columns);
      groupedTasks = groups * CeilDiv(planeCols, tileCols);
      break;
    }
  }
  // Byte permutations benefit from consecutive task ownership down to
  // moderate sizes; wider element types retain the previous scheduling.
  if (executionMode == 1 && elementBytes == 1 && outerRank > 0 &&
      total >= 1024ull * 1024) executionMode = 3;
  // A compact batch with aligned row starts can issue longer output
  // transactions without introducing a tail. Keep one input/output pair;
  // measured rectangular batches benefit from 128x128 rather than 64x128.
  // Batch coordinates may themselves be permuted; only the two local axes
  // need compact row strides. Capacity and useful task count are checked.
  if (mode && elementBytes == 4 && outerRank > 0 && total >= 1024ull * 1024 &&
      planeRows >= 256 && planeCols >= 256 && planeRows % 128 == 0 &&
      planeCols % 128 == 0 && inputRowStride == planeCols &&
      outputRowStride == planeRows && 2ull * 128 * 128 * elementBytes <= ubBudget &&
      outerCount * CeilDiv(planeRows, 128) * CeilDiv(planeCols, 128) >= coreCount) {
    executionMode = 2;
    tileRows = 128;
    tileCols = 128;
    planeGroup = 1;
  }
  // A preserved, non-aligned trailing run can be gathered without UB
  // row padding and written as one contiguous DMA. Retain exact input and
  // output ownership, and reserve both slots of both queues in the budget.
  if (mode == 0 && (elementBytes == 2 || elementBytes == 4) &&
      total >= 1024ull * 1024 && planeRows >= 2 && planeCols >= 2 &&
      planeCols <= 1024 && planeCols % (32 / elementBytes) != 0 &&
      outputRowStride == planeCols) {
    const uint64_t cols = CeilDiv(planeCols, 32 / elementBytes) * (32 / elementBytes);
    uint64_t rows = std::min<uint64_t>(255,
        std::min<uint64_t>(planeRows, ubBudget / (4 * cols * elementBytes)));
    const uint64_t wanted = CeilDiv(coreCount, std::max<uint64_t>(outerCount, 1));
    rows = std::min<uint64_t>(rows, CeilDiv(planeRows, wanted));
    if (rows >= 2 && outerCount * CeilDiv(planeRows, rows) >= coreCount) {
      executionMode = 10;
      tileRows = static_cast<uint32_t>(rows);
      tileCols = static_cast<uint32_t>(cols);
    }
  }
  // Longer output spans reduce fragmented writes for unaligned rows.
  // On half matrices the input row must remain transaction-aligned. On
  // 32-bit matrices extend the long-row strategy to narrower, medium-sized
  // matrices. At least 2048 rows bounds the extra final tile to under 25%.
  const bool longUnalignedMatrix = mode && outerRank == 0 &&
      inputRowStride == planeCols && outputRowStride == planeRows &&
      total >= 1024ull * 1024 && planeRows >= 2048 && planeCols >= 256 &&
      planeRows % 128 != 0 &&
      ((elementBytes == 2 && planeCols % 128 == 0) ||
       (elementBytes == 4 && planeCols <= 1024 && total < 16ull * 1024 * 1024));
  if (longUnalignedMatrix && 2ull * 512 * 32 * elementBytes <= ubBudget &&
      CeilDiv(planeRows, 512) * CeilDiv(planeCols, 32) >= coreCount) {
    executionMode = 2;
    tileRows = 512;
    tileCols = 32;
    planeGroup = 1;
  }
  // Compact half-precision batches benefit from longer output DMA spans.
  // Aligned row strides bound padding overhead, while batch coordinates
  // may be reordered independently. Check UB capacity and useful tile count.
  if (mode && elementBytes == 2 && outerRank > 0 &&
      inputRowStride == planeCols && outputRowStride == planeRows &&
      total >= 1024ull * 1024 && planeRows >= 256 && planeCols >= 256 &&
      planeRows % 128 == 0 && planeCols % 128 == 0 &&
      2ull * 256 * 64 * elementBytes <= ubBudget &&
      outerCount * CeilDiv(planeRows, 256) * CeilDiv(planeCols, 64) >= coreCount) {
    executionMode = 2;
    tileRows = 256;
    tileCols = 64;
    planeGroup = 1;
  }
  // A large narrow matrix can have only one or a few outer planes.
  // Read compact row windows and distribute those windows across cores.
  // Aligned windows own disjoint output row spans, including exact tails.
  if (mode && outerCount < coreCount && total >= 1024ull * 1024 &&
      inputRowStride == planeCols && planeRows >= 512 &&
      planeCols >= 8 && planeCols < 64 && planeCols % (32 / elementBytes) != 0) {
    const uint64_t paddedCols = CeilDiv(planeCols, 8) * 8;
    const uint64_t bytesPerRow = elementBytes == 1 ?
        4 * planeCols + 8 * paddedCols :
        2 * planeCols * elementBytes + (2 * elementBytes + 4) * paddedCols;
    const uint64_t windowRows = std::min<uint64_t>(1024, ubBudget / bytesPerRow / 32 * 32);
    if (windowRows >= 128) {
      executionMode = elementBytes == 1 ? 12 : 11;
      tileRows = static_cast<uint32_t>(windowRows);
      tileCols = static_cast<uint32_t>(paddedCols);
      planeGroup = 1;
    }
  }
  // Compact 32-bit batches with exactly one aligned plane dimension
  // benefit from tiles oriented toward the unaligned GM row stride.
  // Bound both UB storage and available tasks; keep fully aligned and
  // doubly unaligned planes on their existing paths.
  if (mode && elementBytes == 4 && outerRank > 0 &&
      inputRowStride == planeCols && outputRowStride == planeRows &&
      total >= 1024ull * 1024 && planeRows >= 256 && planeCols >= 256 &&
      ((planeRows % 128 == 0) != (planeCols % 128 == 0)) &&
      // The 512x32 path loses input bandwidth at large GM row strides.
      // Bound that empirical policy to <=8 KiB; the 128x128 orientation
      // retains its independent selection for aligned plane rows.
      (planeRows % 128 == 0 || inputRowStride <= 8192 / elementBytes)) {
    const uint32_t rows = planeRows % 128 == 0 ? 128 : 512;
    const uint32_t cols = planeRows % 128 == 0 ? 128 : 32;
    if (2ull * rows * cols * elementBytes <= ubBudget &&
        outerCount * CeilDiv(planeRows, rows) * CeilDiv(planeCols, cols) >= coreCount) {
      executionMode = 2;
      tileRows = rows;
      tileCols = cols;
      planeGroup = 1;
    }
  }
  // Compact 32-bit batches with unaligned rows benefit from reading a
  // complete input row window once. Gather output columns in reusable groups.
  // The policy depends on layout, transfer size and available UB, and also
  // supports reordered outer dimensions through the normal stride mapping.
  if (mode && elementBytes == 4 && outerRank > 0 && total >= 1024ull * 1024 &&
      inputRowStride == planeCols && outputRowStride == planeRows &&
      planeRows >= 128 && planeCols >= 128 && planeCols <= 1024 &&
      (planeRows % 8 != 0 || planeCols % 8 != 0)) {
    const uint64_t columns = std::min<uint64_t>(64, CeilDiv(planeCols, 8) * 8);
    // One input window, two output groups, one uint32 offset table.
    const uint64_t bytesPerRow = (planeCols + 3 * columns) * elementBytes;
    const uint64_t rows = std::min<uint64_t>(64,
        std::min<uint64_t>(CeilDiv(planeRows, 8) * 8, ubBudget / bytesPerRow / 8 * 8));
    if (rows >= 8) {
      executionMode = 13; tileRows = rows; tileCols = columns; planeGroup = 1;
    }
  }
  // Join output-adjacent short planes to issue longer DMA output rows.
  // Column tasks, rather than only outer groups, provide core parallelism.
  if (mode && elementBytes == 4 && total >= 1024ull * 1024 && outerRank > 0 &&
      outerOutput[outerRank - 1] == planeRows && planeRows >= 2 && planeRows <= 32 &&
      planeCols >= 1024 && planeCols <= 16384) {
    const uint64_t adjacent = outerDims[outerRank - 1];
    // Balance the last group instead of leaving one or two planes behind.
    const uint64_t group = CeilDiv(adjacent, CeilDiv(adjacent, 16));
    const uint64_t rows = CeilDiv(planeRows * group, 16) * 16;
    const uint64_t columns = std::min<uint64_t>(1024,
        std::min<uint64_t>(CeilDiv(planeCols, 8) * 8,
                          ubBudget / (2 * rows * elementBytes) / 8 * 8));
    const uint64_t groups = outerCount / adjacent * CeilDiv(adjacent, group);
    if (group >= 2 && columns >= 8 && groups * CeilDiv(planeCols, columns) >= coreCount) {
      executionMode = 9; planeGroup = group; tileRows = rows; tileCols = columns;
      groupedTasks = groups * CeilDiv(planeCols, columns);
    }
  }
  // Coalesce preserved trailing runs across an input-contiguous outer axis.
  // Small runs retain the existing path to amortize the offset-table setup.
  uint64_t blockTasks = 0;
  if (mode == 0 && (elementBytes == 2 || elementBytes == 4) &&
      total >= 1024ull * 1024 && outerRank > 0 && planeRows >= 8 &&
      planeCols >= 16 && planeCols <= 256 && outputRowStride == planeCols &&
      planeCols % (32 / elementBytes) != 0) {
    uint32_t adjacentAxis = outerRank;
    for (uint32_t axis = 0; axis < outerRank; ++axis)
      if (outerInput[axis] == planeCols && outerDims[axis] >= 64) { adjacentAxis = axis; break; }
    if (adjacentAxis < outerRank) {
      const uint64_t alignment = 32 / elementBytes;
      uint64_t r = 8, c = 8;
      // A short retained row group can be written in one longer transaction.
      // Aligned UB pitches permit exact retained row counts; resource/task guards remain.
      // Independent tests support full-row coalescing for16-bit elements;
      // retain the previous tile for32-bit elements.
      if (elementBytes == 2 && planeRows > 16 && planeRows <= 32) {
        const uint64_t fullRows = planeRows;
        const uint64_t fullInPitch = CeilDiv(4 * planeCols, alignment) * alignment;
        const uint64_t fullOutPitch = CeilDiv(fullRows * planeCols, alignment) * alignment;
        const uint64_t fullBytes = 2 * fullRows * fullInPitch * elementBytes +
            4 * fullOutPitch * (2 * elementBytes + 4);
        const uint64_t fullTasks = outerCount / outerDims[adjacentAxis] *
            CeilDiv(outerDims[adjacentAxis], 4);
        if (fullBytes <= ubBudget && fullTasks >= 32ull * coreCount) {
          r = fullRows; c = 4;
        }
      }
      const uint64_t inputPitch = CeilDiv(c * planeCols, alignment) * alignment;
      const uint64_t outputPitch = CeilDiv(r * planeCols, alignment) * alignment;
      const uint64_t bytes = 2 * r * inputPitch * elementBytes + c * outputPitch * (2 * elementBytes + 4);
      const uint64_t count = outerCount / outerDims[adjacentAxis] *
          CeilDiv(planeRows, r) * CeilDiv(outerDims[adjacentAxis], c);
      // Enough work per core amortizes the reusable gather offset table.
      if (bytes <= ubBudget && count >= 32ull * coreCount) {
        executionMode = 16; tileRows = r; tileCols = c;
        planeGroup = adjacentAxis + 1; blockTasks = count;
      }
    }
  }
  // Memory-sized compact 32-bit batches use padded private row windows.
  // Each core owns its workspace; smaller tensors avoid scratch overhead.
  const bool staged = mode && elementBytes == 4 && outerRank > 0 &&
      total >= 16ull * 1024 * 1024 && total <= 128ull * 1024 * 1024 &&
      inputRowStride == planeCols && outputRowStride == planeRows &&
      planeRows >= 128 && planeCols >= 128 && planeCols <= 768 &&
      // Wider input rows retain the direct window path measured in holdouts.
      // Bound redundant work when the last row window is very short.
      CeilDiv(planeRows, 256) * 256 - planeRows <= planeRows / 8 &&
      (planeRows % 8 != 0 || planeCols % 8 != 0) && ubBudget >= 163840;
  if (staged) {
    executionMode = 15; tileRows = 256; tileCols = 64; planeGroup = 2;
  }
  // Queue overlap keeps scratch private to the core. Resource and window
  // efficiency checks preserve the existing staged path on other devices.
  const bool stagedOverlap = staged && 131072ull <= ubBudget &&
      CeilDiv(planeRows, 256) * 256 - planeRows <= planeRows / 8;
  if (stagedOverlap) { tileRows = 256; tileCols = 64; }
  // Compact complete output rows when their padded pitch fits the mask
  // instruction and the two UB buffers. Larger rows keep input staging.
  const uint64_t outputPitch = CeilDiv(planeRows, 128) * 128;
  // This one-pass output policy is reserved for tensors above128MiB.
  // Smaller cache-resident tensors and aligned output rows retain staging,
  // as measured by the independent size and alignment controls.
  const uint64_t outputCapacity = (128ull * 176);
  const bool outputCompact = staged && total >= 32ull * 1024 * 1024 &&
      planeRows % 8 != 0 && 2 * outputCapacity * 4 <= ubBudget &&
      outputPitch <= 1024 && outputCapacity / outputPitch > 0 &&
      outputCapacity / outputPitch <= 255;
  if (outputCompact) {
    executionMode = 22; tileRows = 128; tileCols = 176;
    // A column range smaller than one full window retains grouped input DMA.
    planeGroup = planeCols >= tileCols ? 0u : 1u;
  }
  // Physical pitches spread vector accesses; retain v77 if UB is smaller.
  const bool bankPadding = outputCompact && 195584ull <= ubSize-512;
  // Wider retained runs only when the original row tile and occupancy fit.
  if (executionMode==16) {
    const uint64_t columns=16;
    const uint64_t alignment=32/elementBytes;
    const uint64_t inPitch=CeilDiv(columns*planeCols,alignment)*alignment;
    const uint64_t outPitch=CeilDiv(tileRows*planeCols,alignment)*alignment;
    const uint64_t patternWords=CeilDiv(
        CeilDiv(planeCols+alignment-1,elementBytes*8),alignment)*alignment;
    const uint64_t bytes=2*tileRows*inPitch*elementBytes+
        2*columns*outPitch*elementBytes+columns*patternWords*elementBytes;
    const uint32_t axis=planeGroup-1;
    const uint64_t count=outerCount/outerDims[axis]*
        CeilDiv(planeRows,tileRows)*CeilDiv(outerDims[axis],columns);
    if (bytes<=ubBudget && tileRows<=255 && inPitch/alignment<=128 &&
        count>=8ull*coreCount) {tileCols=columns;blockTasks=count;}
  }
  const uint64_t tasks = outputCompact ? outerCount * CeilDiv(planeCols, tileCols) : executionMode == 16 ? blockTasks : total == 0 ? 0 : ((executionMode == 11 || executionMode == 12 || executionMode == 13 || executionMode == 15) ?
      outerCount * CeilDiv(planeRows, tileRows) : (executionMode == 9 ? groupedTasks :
      ((executionMode == 6 || executionMode == 7 || executionMode == 8) ? outerCount :
      outerCount * CeilDiv(planeRows, tileRows) * CeilDiv(planeCols, tileCols))));
  // Streaming reads can avoid polluting L2 when contiguous transactions
  // are sufficiently long. Short or irregular rows retain normal caching.
  const bool compactPlane = mode != 0 && inputRowStride == planeCols &&
      outputRowStride == planeRows;
  const bool aligned32Matrix = compactPlane && elementBytes == 4 &&
      planeRows >= 256 && planeCols >= 256 && planeRows % 128 == 0 &&
      planeCols % 128 == 0 && total >= 33554432ull / elementBytes;
  const bool alignedByteMatrix = compactPlane && outerRank == 0 && elementBytes == 1 &&
      planeRows >= 256 && planeCols >= 256 && planeCols % 32 == 0 &&
      total >= 16777216ull;
  const bool alignedCopy = mode == 0 && planeCols >= 256 / elementBytes &&
      planeCols % (32 / elementBytes) == 0 &&
      total >= 16777216ull / elementBytes;
  const bool widePackedCopy = mode == 0 && outerRank <= 1 &&
      elementBytes >= 2 && planeCols >= 1020 / elementBytes &&
      total >= 67108864ull / elementBytes;
  // If both row traversal and adjacent output planes jump over input rows,
  // neighboring tasks cannot promptly reuse input cache lines. Stream these
  // scattered planes; compact adjacent planes retain normal aggregation.
  const bool scatteredPlanes = mode != 0 && outerRank > 0 &&
      inputRowStride > planeCols && outerInput[outerRank - 1] > planeCols &&
      planeRows >= 16 && planeCols >= 64 && total >= 16777216ull / elementBytes;
  bool inputCacheBypass = aligned32Matrix || alignedByteMatrix ||
      alignedCopy || widePackedCopy || scatteredPlanes;
  // Byte matrices with aligned input and output rows also benefit from
  // streaming writes. An unaligned output row keeps normal write caching.
  bool outputCacheBypass = alignedByteMatrix && planeRows % 32 == 0;
  // Compact retained runs with aligned bit masks when counter strides fit.
  if (executionMode==16) {
    const uint64_t alignment=32/elementBytes;
    const uint64_t inPitch=CeilDiv(tileCols*planeCols,alignment)*alignment;
    if (tileRows<=255 && inPitch/alignment<=128) executionMode=26;
  }
  // Cache hints apply only to the retained-run family.
  if (executionMode==26) {
    inputCacheBypass=true;
    outputCacheBypass=false;
  }
  TransposeTilingData data;
  data.set_total(total);
  data.set_taskCount(tasks);
  data.set_planeRows(planeRows);
  data.set_planeCols(planeCols);
  data.set_inputRowStride(inputRowStride);
  data.set_outputRowStride(outputRowStride);
  data.set_rank(outerRank);
  data.set_mode(executionMode);
  data.set_tileRows(tileRows);
  data.set_tileCols(tileCols);
  data.set_planeGroup(planeGroup);
  data.set_dims(outerDims);
  data.set_inputStrides(outerInput);
  data.set_outputStrides(outerOutput);
  uint32_t blocks = static_cast<uint32_t>(std::max<uint64_t>(1, std::min<uint64_t>(tasks, coreCount)));
  // Resource-derived occupancy for private output-window tasks.
  if (outputCompact) {
    const uint32_t desired=CeilDiv(tasks,CeilDiv(tasks,blocks));
    blocks=std::max<uint32_t>(1,std::min<uint64_t>(tasks,desired));
  }
  ctx->SetBlockDim(blocks);
  // Three complete kernels retain runtime algorithm selection, with
  // cache attributes constant throughout every DMA helper and loop.
  uint64_t tilingKey = bankPadding ? 16 : outputCompact ? 7 : stagedOverlap ? 6 : staged ? 5 : (executionMode == 11 || executionMode == 12) ? 4 :
      (inputCacheBypass ? (outputCacheBypass ? 3 : 1) : 0);
  constexpr uint32_t staticPolicy=3;
  const bool useStatic = ((staticPolicy&1) && (executionMode==1 || executionMode==2 || executionMode==3)) ||
      ((staticPolicy&2) && executionMode==9) || ((staticPolicy&4) && executionMode==26);
  if (useStatic && (tilingKey==0 || tilingKey==1 || tilingKey==2 || tilingKey==3)) tilingKey+=20;
  ctx->SetTilingKey(tilingKey);
  data.SaveToBuffer(ctx->GetRawTilingData()->GetData(), ctx->GetRawTilingData()->GetCapacity());
  ctx->GetRawTilingData()->SetDataSize(data.GetDataSize());
  // CANN 8.5 requires nonzero workspace for a pure-vector MIX kernel.
  const uint64_t paddedCols = CeilDiv(planeCols, 128) * 128;
  ctx->GetWorkspaceSizes(1)[0] = outputCompact ? platform.GetLibApiWorkSpaceSize() +
      blocks * tileCols * outputPitch * 4 : staged ? platform.GetLibApiWorkSpaceSize() +
      blocks * CeilDiv(tileRows * paddedCols * 4, 512) * 512 : 32;
  return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *ctx) {
  const auto *shape = ctx->GetInputShape(0);
  const auto *perm = ctx->GetAttrs()->GetAttrPointer<gert::ContinuousVector>(0);
  if (!perm || shape->GetDimNum() != perm->GetSize()) return GRAPH_FAILED;
  const auto *axes = static_cast<const int64_t *>(perm->GetData());
  auto *out = ctx->GetOutputShape(0);
  const int64_t rank = shape->GetDimNum();
  if (rank > 64) return GRAPH_FAILED;
  bool used[64] = {};
  out->SetDimNum(rank);
  for (int64_t i = 0; i < rank; ++i) {
    int64_t axis = axes[i] < 0 ? axes[i] + rank : axes[i];
    if (axis < 0 || axis >= rank || used[axis]) return GRAPH_FAILED;
    used[axis] = true;
    out->SetDim(i, shape->GetDim(axis));
  }
  return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *ctx) {
  ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
  return GRAPH_SUCCESS;
}
}

namespace ops {
class Transpose : public OpDef {
 public:
  explicit Transpose(const char *name) : OpDef(name) {
    this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Output("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
      .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
    this->Attr("dims").AttrType(REQUIRED).ListInt();
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};
OP_ADD(Transpose);
}
