#include "kernel_operator.h"
using namespace AscendC;

__aicore__ inline uint64_t TransposeCeilDiv(uint64_t a, uint32_t b) {
  return a / b + (a % b != 0);
}

// Each task owns one rectangular plane tile. Other coordinates are decoded
// once per tile, rather than once for every element.
__aicore__ inline void DecodeTask(const TransposeTilingData &t, uint64_t task,
    uint64_t &src, uint64_t &dst, uint32_t &rows, uint32_t &cols) {
  const uint64_t columnTiles = TransposeCeilDiv(t.planeCols, t.tileCols);
  const uint64_t rowTiles = TransposeCeilDiv(t.planeRows, t.tileRows);
  const uint64_t col = (task % columnTiles) * t.tileCols;
  task /= columnTiles;
  const uint64_t row = (task % rowTiles) * t.tileRows;
  task /= rowTiles;
  src = row * t.inputRowStride + col;
  dst = (t.mode == 0 || t.mode == 10) ? row * t.outputRowStride + col : col * t.outputRowStride + row;
  for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
    const uint64_t coord = task % t.dims[d];
    task /= t.dims[d];
    src += coord * t.inputStrides[d];
    dst += coord * t.outputStrides[d];
  }
  rows = static_cast<uint32_t>(t.planeRows - row < t.tileRows ? t.planeRows - row : t.tileRows);
  cols = static_cast<uint32_t>(t.planeCols - col < t.tileCols ? t.planeCols - col : t.tileCols);
}

// A core owns a contiguous interval of plane tiles. Decode its first tile
// once, then advance a mixed-radix cursor with additions and carry only.
class TransposeTaskCursor {
 public:
  __aicore__ inline TransposeTaskCursor(const TransposeTilingData &data, uint64_t task)
      : t(data) {
    const uint64_t columnTiles = TransposeCeilDiv(t.planeCols, t.tileCols);
    const uint64_t rowTiles = TransposeCeilDiv(t.planeRows, t.tileRows);
    col = (task % columnTiles) * t.tileCols;
    task /= columnTiles;
    row = (task % rowTiles) * t.tileRows;
    task /= rowTiles;
    src = row * t.inputRowStride + col;
    dst = t.mode ? col * t.outputRowStride + row : row * t.outputRowStride + col;
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      coordinates[d] = task % t.dims[d];
      task /= t.dims[d];
      src += coordinates[d] * t.inputStrides[d];
      dst += coordinates[d] * t.outputStrides[d];
    }
  }
  __aicore__ inline void Get(uint64_t &source, uint64_t &destination,
                            uint32_t &rows, uint32_t &cols) const {
    source = src;
    destination = dst;
    rows = static_cast<uint32_t>(t.planeRows - row < t.tileRows ? t.planeRows - row : t.tileRows);
    cols = static_cast<uint32_t>(t.planeCols - col < t.tileCols ? t.planeCols - col : t.tileCols);
  }
  __aicore__ inline void Next() {
    col += t.tileCols;
    src += t.tileCols;
    dst += t.mode ? t.tileCols * t.outputRowStride : t.tileCols;
    if (col < t.planeCols) return;
    src -= col;
    dst -= t.mode ? col * t.outputRowStride : col;
    col = 0;
    row += t.tileRows;
    src += t.tileRows * t.inputRowStride;
    dst += t.mode ? t.tileRows : t.tileRows * t.outputRowStride;
    if (row < t.planeRows) return;
    src -= row * t.inputRowStride;
    dst -= t.mode ? row : row * t.outputRowStride;
    row = 0;
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      ++coordinates[d];
      src += t.inputStrides[d];
      dst += t.outputStrides[d];
      if (coordinates[d] < t.dims[d]) return;
      src -= coordinates[d] * t.inputStrides[d];
      dst -= coordinates[d] * t.outputStrides[d];
      coordinates[d] = 0;
    }
  }
 private:
  const TransposeTilingData &t;
  uint64_t coordinates[64];
  uint64_t src, dst, row, col;
};

__aicore__ inline void TransposeTaskRange(uint64_t tasks, uint64_t &begin, uint64_t &end) {
  const uint32_t core = GetBlockIdx(), cores = GetBlockNum();
  const uint64_t perCore = tasks / cores, extra = tasks % cores;
  begin = perCore * core + (core < extra ? core : extra);
  end = begin + perCore + (core < extra);
}

// DataCopyPad uses bytes on the GM side and 32-byte blocks on the UB side.
// The gap starts after the implicitly aligned UB block length, including tails.
template <typename T>
__aicore__ inline void ReadRows(const LocalTensor<T> &local, const GlobalTensor<T> &gm,
    uint64_t offset, uint32_t rows, uint32_t cols, uint32_t pitch, uint64_t gmStride,
    bool groupAlignment = true) {
  const uint32_t bytes = cols * sizeof(T);
  const uint32_t gap = (pitch * sizeof(T) - ((bytes + 31) / 32) * 32) / 32;
  const uint64_t gmGap = (gmStride - cols) * sizeof(T);
  DataCopyPadExtParams<T> pad{false, 0, 0, 0};
  // Ordinary DMA counts and gaps are in 32-byte units. GM base addresses
  // need only the element alignment guaranteed by a typed tensor.
  if (rows > 0 && rows <= 4095 && bytes > 0 && bytes % 32 == 0 &&
      bytes / 32 <= 65535 && gap <= 65535 &&
      (rows == 1 || (gmGap % 32 == 0 && gmGap / 32 <= 65535))) {
    DataCopyParams params{static_cast<uint16_t>(rows), static_cast<uint16_t>(bytes / 32),
        static_cast<uint16_t>(rows == 1 ? 0 : gmGap / 32), static_cast<uint16_t>(gap)};
    DataCopy(local, gm[offset], params);
    return;
  }
  const uint64_t gmStrideBytes = gmStride * sizeof(T);
  // Group rows with the same alignment residue. Advancing by period rows
  // makes both GM and UB gaps representable in ordinary DMA units.
  // Prefixes and tails are disjoint: no over-read or overlapping writes.
  if (groupAlignment && bytes >= 256 && rows <= 4095 && gmStrideBytes % 32 != 0 &&
      gmStrideBytes <= 2097151) {
    const uint32_t period = 32 / static_cast<uint32_t>(gmStrideBytes & (~gmStrideBytes + 1));
    const uint32_t prefixBytes = bytes / 32 * 32;
    const uint64_t groupGmGap = period * gmStrideBytes - prefixBytes;
    const uint64_t groupUbGap = static_cast<uint64_t>(period) * pitch * sizeof(T) - prefixBytes;
    if (period <= 16 && rows >= 2 * period && prefixBytes / 32 <= 65535 &&
        groupGmGap / 32 <= 65535 && groupUbGap / 32 <= 65535) {
      for (uint32_t residue = 0; residue < period; ++residue) {
        const uint32_t count = (rows - residue + period - 1) / period;
        DataCopyParams params{static_cast<uint16_t>(count), static_cast<uint16_t>(prefixBytes / 32),
            static_cast<uint16_t>(groupGmGap / 32), static_cast<uint16_t>(groupUbGap / 32)};
        DataCopy(local[residue * pitch], gm[offset + static_cast<uint64_t>(residue) * gmStride], params);
      }
      const uint32_t tailBytes = bytes - prefixBytes;
      if (tailBytes) {
        DataCopyExtParams tail{static_cast<uint16_t>(rows), tailBytes,
            static_cast<uint32_t>(gmStrideBytes - tailBytes), static_cast<uint32_t>(pitch * sizeof(T) / 32 - 1), 0};
        DataCopyPad(local[prefixBytes / sizeof(T)], gm[offset + prefixBytes / sizeof(T)], tail, pad);
      }
      return;
    }
  }
  if (gmGap <= 0xffffffffull) {
    DataCopyExtParams params{static_cast<uint16_t>(rows), bytes, static_cast<uint32_t>(gmGap), gap, 0};
    DataCopyPad(local, gm[offset], params, pad);
  } else {
    // Very large tensor strides cannot be represented by DataCopyExtParams.
    DataCopyExtParams params{1, bytes, 0, 0, 0};
    for (uint32_t r = 0; r < rows; ++r)
      DataCopyPad(local[r * pitch], gm[offset + r * gmStride], params, pad);
  }
}

template <typename T>
__aicore__ inline void WriteRows(const GlobalTensor<T> &gm, const LocalTensor<T> &local,
    uint64_t offset, uint32_t rows, uint32_t cols, uint32_t pitch, uint64_t gmStride,
    bool groupAlignment = false) {
  const uint32_t bytes = cols * sizeof(T);
  const uint32_t gap = (pitch * sizeof(T) - ((bytes + 31) / 32) * 32) / 32;
  const uint64_t gmGap = (gmStride - cols) * sizeof(T);
  if (rows > 0 && rows <= 4095 && bytes > 0 && bytes % 32 == 0 &&
      bytes / 32 <= 65535 && gap <= 65535 &&
      (rows == 1 || (gmGap % 32 == 0 && gmGap / 32 <= 65535))) {
    DataCopyParams params{static_cast<uint16_t>(rows), static_cast<uint16_t>(bytes / 32),
        static_cast<uint16_t>(gap), static_cast<uint16_t>(rows == 1 ? 0 : gmGap / 32)};
    DataCopy(gm[offset], local, params);
    return;
  }
  const uint64_t gmStrideBytes = gmStride * sizeof(T);
  // Same-residue rows have exactly representable ordinary-DMA strides.
  // Copy only the aligned prefix; byte tails occupy disjoint GM intervals.
  if (groupAlignment && bytes >= 256 && rows <= 4095 && gmStrideBytes % 32 != 0 &&
      gmStrideBytes <= 2097151) {
    const uint32_t period = 32 / static_cast<uint32_t>(gmStrideBytes & (~gmStrideBytes + 1));
    const uint32_t prefixBytes = bytes / 32 * 32;
    const uint64_t groupGmGap = period * gmStrideBytes - prefixBytes;
    const uint64_t groupUbGap = static_cast<uint64_t>(period) * pitch * sizeof(T) - prefixBytes;
    if (period <= 16 && rows >= 2 * period && prefixBytes / 32 <= 65535 &&
        groupGmGap / 32 <= 65535 && groupUbGap / 32 <= 65535) {
      for (uint32_t residue = 0; residue < period; ++residue) {
        const uint32_t count = (rows - residue + period - 1) / period;
        DataCopyParams params{static_cast<uint16_t>(count), static_cast<uint16_t>(prefixBytes / 32),
            static_cast<uint16_t>(groupUbGap / 32), static_cast<uint16_t>(groupGmGap / 32)};
        DataCopy(gm[offset + static_cast<uint64_t>(residue) * gmStride], local[residue * pitch], params);
      }
      const uint32_t tailBytes = bytes - prefixBytes;
      if (tailBytes) {
        DataCopyExtParams tail{static_cast<uint16_t>(rows), tailBytes,
            static_cast<uint32_t>(pitch * sizeof(T) / 32 - 1),
            static_cast<uint32_t>(gmStrideBytes - tailBytes), 0};
        DataCopyPad(gm[offset + prefixBytes / sizeof(T)], local[prefixBytes / sizeof(T)], tail);
      }
      return;
    }
  }
  if (gmGap <= 0xffffffffull) {
    DataCopyExtParams params{static_cast<uint16_t>(rows), bytes, gap, static_cast<uint32_t>(gmGap), 0};
    DataCopyPad(gm[offset], local, params);
  } else {
    DataCopyExtParams params{1, bytes, 0, 0, 0};
    for (uint32_t r = 0; r < rows; ++r)
      DataCopyPad(gm[offset + r * gmStride], local[r * pitch], params);
  }
}

template <typename T, bool InputBypass, bool OutputBypass>
__aicore__ inline void CopyPreservedLastAxis(GM_ADDR x, GM_ADDR y, const TransposeTilingData &t) {
  GlobalTensor<T> input, output;
  input.SetGlobalBuffer((__gm__ T *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  TPipe pipe;
  TBuf<TPosition::VECIN> buffer;
  pipe.InitBuffer(buffer, t.tileRows * t.tileCols * sizeof(T));
  auto local = buffer.Get<T>();
  for (uint64_t task = GetBlockIdx(); task < t.taskCount; task += GetBlockNum()) {
    uint64_t src, dst;
    uint32_t rows, cols;
    DecodeTask(t, task, src, dst, rows, cols);
    ReadRows(local, input, src, rows, cols, t.tileCols, t.inputRowStride);
    SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
    WriteRows(output, local, dst, rows, cols, t.tileCols, t.outputRowStride);
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
  }
}

// GatherMask counter mode on A2 consumes cols elements from each padded
// input row and appends them densely. Both tensors start on a UB block;
// repeat count <=255 and the row stride <=128 blocks are bounded by tiling.
// The gather performs no arithmetic on the values, including NaN bit patterns.
template <typename T, bool InputBypass, bool OutputBypass>
__aicore__ inline void CopyPackedPreservedRows(GM_ADDR x, GM_ADDR y,
                                              const TransposeTilingData &t) {
  if constexpr (sizeof(T) == 2 || sizeof(T) == 4) {
    GlobalTensor<T> input, output;
    input.SetGlobalBuffer((__gm__ T *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    TPipe pipe;
    TQue<TPosition::VECIN, 2> inQueue;
    TQue<TPosition::VECOUT, 2> outQueue;
    const uint32_t bytes = t.tileRows * t.tileCols * sizeof(T);
    pipe.InitBuffer(inQueue, 2, bytes);
    pipe.InitBuffer(outQueue, 2, bytes);
    for (uint64_t task = GetBlockIdx(); task < t.taskCount; task += GetBlockNum()) {
      uint64_t src, dst;
      uint32_t rows, cols;
      DecodeTask(t, task, src, dst, rows, cols);
      auto xLocal = inQueue.AllocTensor<T>();
      ReadRows(xLocal, input, src, rows, cols, t.tileCols, t.inputRowStride);
      inQueue.EnQue(xLocal);
      auto xReady = inQueue.DeQue<T>();
      auto yLocal = outQueue.AllocTensor<T>();
      uint64_t retained = 0;
      GatherMask(yLocal, xReady, static_cast<uint8_t>(7), true, cols,
          {1, static_cast<uint16_t>(rows),
           static_cast<uint16_t>(t.tileCols * sizeof(T) / 32), 0}, retained);
      outQueue.EnQue(yLocal);
      inQueue.FreeTensor(xReady);
      auto yReady = outQueue.DeQue<T>();
      DataCopyExtParams write{1, rows * cols * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
      DataCopyPad(output[dst], yReady, write);
      outQueue.FreeTensor(yReady);
    }
  }
}

template <typename T>
__aicore__ inline void TransposeTile(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    uint32_t rows, uint32_t cols, uint32_t tileRows, uint32_t tileCols) {
  uint64_t srcList[16], dstList[16];
  const uint64_t srcBase = (uint64_t)(src.GetPhyAddr());
  const uint64_t dstBase = (uint64_t)(dst.GetPhyAddr());
  constexpr uint32_t blockElements = 32 / sizeof(T);
  const uint8_t repeats = static_cast<uint8_t>((cols + blockElements - 1) / blockElements);
  const uint16_t dstRepeatStride = repeats == 1 ? 0 : static_cast<uint16_t>(tileRows);
  const uint16_t srcRepeatStride = repeats == 1 ? 0 : 1;

  if constexpr (sizeof(T) != 1) {
    // Repeat along rows when that needs fewer address lists than repeating
    // along columns. Discarded edge lanes stay inside the allocated tile.
    const uint32_t rowRepeats = (rows + 15) / 16;
    if (rowRepeats <= 255 && static_cast<uint32_t>(repeats) < rowRepeats) {
      for (uint32_t c = 0; c < cols; c += blockElements) {
        for (uint32_t i = 0; i < 16; ++i) {
          srcList[i] = srcBase + (i * tileCols + c) * sizeof(T);
          if constexpr (sizeof(T) == 2) {
            dstList[i] = dstBase + ((c + i) * tileRows) * sizeof(T);
          } else {
            dstList[i] = dstBase + ((c + i / 2) * tileRows + (i % 2) * 8) * sizeof(T);
          }
        }
        TransDataTo5HDParams params(false, false, static_cast<uint8_t>(rowRepeats),
            static_cast<uint16_t>(rowRepeats == 1 ? 0 : 16 * sizeof(T) / 32),
            static_cast<uint16_t>(rowRepeats == 1 ? 0 : 16 * tileCols * sizeof(T) / 32));
        TransDataTo5HD<T>(dstList, srcList, params);
      }
      return;
    }
  }

  if constexpr (sizeof(T) == 1) {
    // vnchwconv.b8 transposes one 16x16 half-block at a time. Four calls
    // cover each 32-row block without converting signed integer values.
    for (uint32_t r = 0; r < rows; r += 32) {
      for (uint32_t dstHalf = 0; dstHalf < 2; ++dstHalf) {
        for (uint32_t i = 0; i < 16; ++i) {
          const uint32_t sourceRow = r + dstHalf * 16 + i;
          srcList[i] = srcBase + (sourceRow < rows ? sourceRow * tileCols : 0);
        }
        for (uint32_t srcHalf = 0; srcHalf < 2; ++srcHalf) {
          for (uint32_t i = 0; i < 16; ++i) {
            dstList[i] = dstBase + (i + srcHalf * 16) * tileRows + r;
          }
          TransDataTo5HDParams params(dstHalf != 0, srcHalf != 0, repeats, dstRepeatStride, srcRepeatStride);
          TransDataTo5HD<T>(dstList, srcList, params);
        }
        // Column halves write disjoint output rows. Only the following byte
        // half reuses their 32-byte blocks; the next r-group is disjoint again.
        if (dstHalf == 0) PipeBarrier<PIPE_V>();
      }
    }
  } else {
    for (uint32_t r = 0; r < rows; r += 16) {
      for (uint32_t i = 0; i < 16; ++i) {
        srcList[i] = srcBase + (r + i < rows ? (r + i) * tileCols * sizeof(T) : 0);
        if constexpr (sizeof(T) == 2) {
          dstList[i] = dstBase + (i * tileRows + r) * sizeof(T);
        } else {
          // A 32-bit output row has two 32-byte halves for its 16 elements.
          dstList[i] = dstBase + ((i / 2) * tileRows + r + (i % 2) * 8) * sizeof(T);
        }
      }
      TransDataTo5HDParams params(false, false, repeats, dstRepeatStride, srcRepeatStride);
      TransDataTo5HD<T>(dstList, srcList, params);
    }
  }
}

template <typename T, bool InputBypass, bool OutputBypass>
__aicore__ inline void CopyTransposedPlanes(GM_ADDR x, GM_ADDR y, const TransposeTilingData &t) {
  GlobalTensor<T> input, output;
  input.SetGlobalBuffer((__gm__ T *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  TPipe pipe;
  TQue<TPosition::VECIN, 2> inQueue;
  TQue<TPosition::VECOUT, 2> outQueue;
  const uint32_t tileBytes = t.tileRows * t.tileCols * sizeof(T);
  pipe.InitBuffer(inQueue, 2, tileBytes);
  pipe.InitBuffer(outQueue, 2, tileBytes);
  for (uint64_t task = GetBlockIdx(); task < t.taskCount; task += GetBlockNum()) {
    uint64_t src, dst;
    uint32_t rows, cols;
    DecodeTask(t, task, src, dst, rows, cols);
    auto xLocal = inQueue.AllocTensor<T>();
    ReadRows(xLocal, input, src, rows, cols, t.tileCols, t.inputRowStride);
    inQueue.EnQue(xLocal);
    auto xReady = inQueue.DeQue<T>();
    auto yLocal = outQueue.AllocTensor<T>();
    TransposeTile(yLocal, xReady, rows, cols, t.tileRows, t.tileCols);
    outQueue.EnQue(yLocal);
    inQueue.FreeTensor(xReady);
    auto yReady = outQueue.DeQue<T>();
    WriteRows(output, yReady, dst, cols, rows, t.tileRows, t.outputRowStride);
    outQueue.FreeTensor(yReady);
  }
}

template <typename T, uint32_t Depth, bool InputBypass, bool OutputBypass>
__aicore__ inline void CopyTransposedPlanesCursor(GM_ADDR x, GM_ADDR y, const TransposeTilingData &t) {
  GlobalTensor<T> input, output;
  input.SetGlobalBuffer((__gm__ T *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  TPipe pipe;
  TQue<TPosition::VECIN, Depth> inQueue;
  TQue<TPosition::VECOUT, Depth> outQueue;
  const uint32_t tileBytes = t.tileRows * t.tileCols * sizeof(T);
  pipe.InitBuffer(inQueue, Depth, tileBytes);
  pipe.InitBuffer(outQueue, Depth, tileBytes);
  uint64_t begin, end;
  TransposeTaskRange(t.taskCount, begin, end);
  if (begin == end) return;
  TransposeTaskCursor cursor(t, begin);
  for (uint64_t task = begin; task < end; ++task) {
    uint64_t src, dst;
    uint32_t rows, cols;
    cursor.Get(src, dst, rows, cols);
    auto xLocal = inQueue.template AllocTensor<T>();
    ReadRows(xLocal, input, src, rows, cols, t.tileCols, t.inputRowStride);
    inQueue.EnQue(xLocal);
    auto xReady = inQueue.template DeQue<T>();
    auto yLocal = outQueue.template AllocTensor<T>();
    TransposeTile(yLocal, xReady, rows, cols, t.tileRows, t.tileCols);
    outQueue.EnQue(yLocal);
    inQueue.FreeTensor(xReady);
    auto yReady = outQueue.template DeQue<T>();
    WriteRows(output, yReady, dst, cols, rows, t.tileRows, t.outputRowStride);
    outQueue.FreeTensor(yReady);
    if (task + 1 < end) cursor.Next();
  }
}

// Read a compact input plane once, then gather complete output rows in groups.
// A task owns either a full plane or an exact, disjoint row window.
template <bool InputBypass, bool OutputBypass, bool Windowed = false>
__aicore__ inline void CopyResidentPackedInputPlanes(GM_ADDR x, GM_ADDR y,
                                                   const TransposeTilingData &t) {
  GlobalTensor<uint32_t> input, output;
  input.SetGlobalBuffer((__gm__ uint32_t *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ uint32_t *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  const uint32_t planeCols = static_cast<uint32_t>(t.planeCols);
  const uint32_t inputBytes = t.tileRows * planeCols * sizeof(uint32_t);
  const uint32_t outputBytes = t.tileRows * t.tileCols * sizeof(uint32_t);
  TPipe pipe;
  TQue<TPosition::VECIN, 1> inQueue;
  TQue<TPosition::VECOUT, 2> outQueue;
  TBuf<TPosition::VECCALC> offsets;
  pipe.InitBuffer(inQueue, 1, inputBytes);
  pipe.InitBuffer(outQueue, 2, outputBytes);
  pipe.InitBuffer(offsets, outputBytes);
  auto indices = offsets.Get<int32_t>();
  CreateVecIndex(indices, static_cast<int32_t>(0), t.tileRows);
  PipeBarrier<PIPE_V>();
  Muls(indices, indices, static_cast<int32_t>(planeCols * sizeof(uint32_t)), t.tileRows);
  PipeBarrier<PIPE_V>();
  for (uint32_t filled = 1; filled < t.tileCols; filled *= 2) {
    const uint32_t columns = t.tileCols - filled < filled ? t.tileCols - filled : filled;
    Adds(indices[filled * t.tileRows], indices,
         static_cast<int32_t>(filled * sizeof(uint32_t)), columns * t.tileRows);
    PipeBarrier<PIPE_V>();
  }
  auto byteOffsets = indices.ReinterpretCast<uint32_t>();
  const uint64_t windows = TransposeCeilDiv(t.planeRows, t.tileRows);
  for (uint64_t task = GetBlockIdx(); task < t.taskCount; task += GetBlockNum()) {
    uint64_t coordinate = Windowed ? task / windows : task;
    uint64_t src = 0, dst = 0;
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint64_t rowStart = Windowed ? (task % windows) * t.tileRows : 0;
    const uint32_t planeRows = static_cast<uint32_t>(Windowed && t.planeRows - rowStart > t.tileRows ?
        t.tileRows : t.planeRows - rowStart);
    src += rowStart * planeCols;
    dst += rowStart;
    auto xLocal = inQueue.AllocTensor<uint32_t>();
    DataCopyExtParams copy{1, planeRows * planeCols * static_cast<uint32_t>(sizeof(uint32_t)), 0, 0, 0};
    DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
    DataCopyPad(xLocal, input[src], copy, pad);
    inQueue.EnQue(xLocal);
    auto xReady = inQueue.DeQue<uint32_t>();
    for (uint32_t col = 0; col < planeCols; col += t.tileCols) {
      const uint32_t cols = planeCols - col < t.tileCols ? planeCols - col : t.tileCols;
      auto yLocal = outQueue.AllocTensor<uint32_t>();
      Gather(yLocal, xReady, byteOffsets, col * static_cast<uint32_t>(sizeof(uint32_t)),
             cols * t.tileRows);
      outQueue.EnQue(yLocal);
      auto yReady = outQueue.DeQue<uint32_t>();
      WriteRows(output, yReady, dst + static_cast<uint64_t>(col) * t.outputRowStride,
                cols, planeRows, t.tileRows, t.outputRowStride, Windowed);
      outQueue.FreeTensor(yReady);
    }
    inQueue.FreeTensor(xReady);
  }
}

// Compact narrow input windows keep DMA lengths large without assuming a
// particular rank, permutation, or shape. Padded Gather lanes remain in UB;
// only valid rows and columns are copied to GM.
template <typename T, bool InputBypass, bool OutputBypass, bool SplitRows = false>
__aicore__ inline void CopyNarrowPackedInputWindows(GM_ADDR x, GM_ADDR y,
                                                   const TransposeTilingData &t) {
  GlobalTensor<T> input, output;
  input.SetGlobalBuffer((__gm__ T *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  const uint32_t cols = static_cast<uint32_t>(t.planeCols);
  const uint32_t inputBytes = t.tileRows * cols * sizeof(T);
  const uint32_t outputBytes = t.tileRows * t.tileCols * sizeof(T);
  TPipe pipe;
  TQue<TPosition::VECIN, 2> inQueue;
  TQue<TPosition::VECOUT, 2> outQueue;
  TBuf<TPosition::VECCALC> offsets;
  pipe.InitBuffer(inQueue, 2, inputBytes);
  pipe.InitBuffer(outQueue, 2, outputBytes);
  pipe.InitBuffer(offsets, t.tileRows * t.tileCols * sizeof(uint32_t));
  auto indices = offsets.Get<int32_t>();
  CreateVecIndex(indices, static_cast<int32_t>(0), t.tileRows);
  PipeBarrier<PIPE_V>();
  Muls(indices, indices, static_cast<int32_t>(cols * sizeof(T)), t.tileRows);
  PipeBarrier<PIPE_V>();
  for (uint32_t filled = 1; filled < t.tileCols; filled *= 2) {
    const uint32_t count = t.tileCols - filled < filled ? t.tileCols - filled : filled;
    Adds(indices[filled * t.tileRows], indices,
         static_cast<int32_t>(filled * sizeof(T)), count * t.tileRows);
    PipeBarrier<PIPE_V>();
  }
  auto byteOffsets = indices.ReinterpretCast<uint32_t>();
  uint64_t begin, end;
  TransposeTaskRange(t.taskCount, begin, end);
  const uint64_t windowsPerPlane = TransposeCeilDiv(t.planeRows, t.tileRows);
  for (uint64_t task = begin; task < end; ++task) {
    uint64_t coordinate = SplitRows ? task / windowsPerPlane : task;
    uint64_t src = 0, dst = 0;
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint64_t rowBegin = SplitRows ? (task % windowsPerPlane) * t.tileRows : 0;
    const uint64_t rowEnd = SplitRows ? rowBegin + t.tileRows : t.planeRows;
    for (uint64_t row = rowBegin; row < rowEnd && row < t.planeRows; row += t.tileRows) {
      const uint32_t rows = static_cast<uint32_t>(
          t.planeRows - row < t.tileRows ? t.planeRows - row : t.tileRows);
      auto xLocal = inQueue.AllocTensor<T>();
      DataCopyExtParams copy{1, rows * cols * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
      DataCopyPadExtParams<T> pad{false, 0, 0, 0};
      DataCopyPad(xLocal, input[src + row * cols], copy, pad);
      inQueue.EnQue(xLocal);
      auto xReady = inQueue.DeQue<T>();
      auto yLocal = outQueue.AllocTensor<T>();
      Gather(yLocal, xReady, byteOffsets, static_cast<uint32_t>(0), cols * t.tileRows);
      outQueue.EnQue(yLocal);
      inQueue.FreeTensor(xReady);
      auto yReady = outQueue.DeQue<T>();
      WriteRows(output, yReady, dst + row, cols, rows, t.tileRows, t.outputRowStride);
      outQueue.FreeTensor(yReady);
    }
  }
}

// A2 Gather accepts half but not int8. Every signed byte is exactly
// representable as half, so the two casts preserve all 256 input values.
template <bool InputBypass, bool OutputBypass, bool SplitRows = false>
__aicore__ inline void CopyBytePackedInputWindows(GM_ADDR x, GM_ADDR y,
                                                   const TransposeTilingData &t) {
  GlobalTensor<int8_t> input, output;
  input.SetGlobalBuffer((__gm__ int8_t *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ int8_t *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  const uint32_t cols = static_cast<uint32_t>(t.planeCols);
  const uint32_t inputBytes = t.tileRows * cols * sizeof(int8_t);
  const uint32_t outputBytes = t.tileRows * t.tileCols * sizeof(int8_t);
  TPipe pipe;
  TQue<TPosition::VECIN, 2> inQueue;
  TQue<TPosition::VECOUT, 2> outQueue;
  TBuf<TPosition::VECCALC> offsets, promotedInput, promotedOutput;
  pipe.InitBuffer(inQueue, 2, inputBytes);
  pipe.InitBuffer(outQueue, 2, outputBytes);
  pipe.InitBuffer(offsets, t.tileRows * t.tileCols * sizeof(uint32_t));
  pipe.InitBuffer(promotedInput, inputBytes * sizeof(half));
  pipe.InitBuffer(promotedOutput, outputBytes * sizeof(half));
  auto indices = offsets.Get<int32_t>();
  CreateVecIndex(indices, static_cast<int32_t>(0), t.tileRows);
  PipeBarrier<PIPE_V>();
  Muls(indices, indices, static_cast<int32_t>(cols * sizeof(half)), t.tileRows);
  PipeBarrier<PIPE_V>();
  for (uint32_t filled = 1; filled < t.tileCols; filled *= 2) {
    const uint32_t count = t.tileCols - filled < filled ? t.tileCols - filled : filled;
    Adds(indices[filled * t.tileRows], indices,
         static_cast<int32_t>(filled * sizeof(half)), count * t.tileRows);
    PipeBarrier<PIPE_V>();
  }
  auto byteOffsets = indices.ReinterpretCast<uint32_t>();
  auto expanded = promotedInput.Get<half>();
  auto gathered = promotedOutput.Get<half>();
  uint64_t begin, end;
  TransposeTaskRange(t.taskCount, begin, end);
  const uint64_t windowsPerPlane = TransposeCeilDiv(t.planeRows, t.tileRows);
  for (uint64_t task = begin; task < end; ++task) {
    uint64_t coordinate = SplitRows ? task / windowsPerPlane : task;
    uint64_t src = 0, dst = 0;
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint64_t rowBegin = SplitRows ? (task % windowsPerPlane) * t.tileRows : 0;
    const uint64_t rowEnd = SplitRows ? rowBegin + t.tileRows : t.planeRows;
    for (uint64_t row = rowBegin; row < rowEnd && row < t.planeRows; row += t.tileRows) {
      const uint32_t rows = static_cast<uint32_t>(
          t.planeRows - row < t.tileRows ? t.planeRows - row : t.tileRows);
      auto xLocal = inQueue.AllocTensor<int8_t>();
      DataCopyExtParams copy{1, rows * cols * static_cast<uint32_t>(sizeof(int8_t)), 0, 0, 0};
      DataCopyPadExtParams<int8_t> pad{false, 0, 0, 0};
      DataCopyPad(xLocal, input[src + row * cols], copy, pad);
      inQueue.EnQue(xLocal);
      auto xReady = inQueue.DeQue<int8_t>();
      Cast(expanded, xReady, RoundMode::CAST_NONE, cols * t.tileRows);
      PipeBarrier<PIPE_V>();
      Gather(gathered, expanded, byteOffsets, static_cast<uint32_t>(0), cols * t.tileRows);
      PipeBarrier<PIPE_V>();
      auto yLocal = outQueue.AllocTensor<int8_t>();
      Cast(yLocal, gathered, RoundMode::CAST_NONE, cols * t.tileRows);
      PipeBarrier<PIPE_V>();
      outQueue.EnQue(yLocal);
      inQueue.FreeTensor(xReady);
      auto yReady = outQueue.DeQue<int8_t>();
      WriteRows(output, yReady, dst + row, cols, rows, t.tileRows, t.outputRowStride);
      outQueue.FreeTensor(yReady);
    }
  }
}

// The last outer axis has output stride planeRows. Joining its planes
// forms a longer contiguous output segment; all input reads stay exact.
template <bool InputBypass, bool OutputBypass>
__aicore__ inline void CopyGroupedOutputPlanes(GM_ADDR x, GM_ADDR y,
                                               const TransposeTilingData &t) {
  GlobalTensor<uint32_t> input, output;
  input.SetGlobalBuffer((__gm__ uint32_t *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ uint32_t *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  TPipe pipe;
  TQue<TPosition::VECIN, 1> inQueue;
  TQue<TPosition::VECOUT, 1> outQueue;
  const uint32_t tileBytes = t.tileRows * t.tileCols * sizeof(uint32_t);
  pipe.InitBuffer(inQueue, 1, tileBytes);
  pipe.InitBuffer(outQueue, 1, tileBytes);
  const uint32_t adjacentAxis = t.rank - 1;
  const uint64_t groupsPerPrefix = TransposeCeilDiv(t.dims[adjacentAxis], t.planeGroup);
  const uint64_t columnTiles = TransposeCeilDiv(t.planeCols, t.tileCols);
  uint64_t begin, end;
  TransposeTaskRange(t.taskCount, begin, end);
  for (uint64_t task = begin; task < end; ++task) {
    const uint64_t col = (task % columnTiles) * t.tileCols;
    uint64_t coordinate = task / columnTiles;
    const uint64_t firstPlane = (coordinate % groupsPerPrefix) * t.planeGroup;
    coordinate /= groupsPerPrefix;
    uint64_t src = firstPlane * t.inputStrides[adjacentAxis] + col;
    uint64_t dst = firstPlane * t.planeRows + col * t.outputRowStride;
    for (int32_t d = static_cast<int32_t>(adjacentAxis) - 1; d >= 0; --d) {
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint32_t planes = static_cast<uint32_t>(t.dims[adjacentAxis] - firstPlane < t.planeGroup ?
        t.dims[adjacentAxis] - firstPlane : t.planeGroup);
    const uint32_t rows = planes * static_cast<uint32_t>(t.planeRows);
    const uint32_t cols = static_cast<uint32_t>(t.planeCols - col < t.tileCols ?
        t.planeCols - col : t.tileCols);
    auto xLocal = inQueue.AllocTensor<uint32_t>();
    for (uint32_t plane = 0; plane < planes; ++plane) {
      ReadRows(xLocal[plane * static_cast<uint32_t>(t.planeRows) * t.tileCols],
          input, src + plane * t.inputStrides[adjacentAxis],
          static_cast<uint32_t>(t.planeRows), cols, t.tileCols, t.inputRowStride, false);
    }
    inQueue.EnQue(xLocal);
    auto xReady = inQueue.DeQue<uint32_t>();
    auto yLocal = outQueue.AllocTensor<uint32_t>();
    TransposeTile(yLocal, xReady, rows, cols, t.tileRows, t.tileCols);
    outQueue.EnQue(yLocal);
    inQueue.FreeTensor(xReady);
    auto yReady = outQueue.DeQue<uint32_t>();
    WriteRows(output, yReady, dst, cols, rows, t.tileRows, t.outputRowStride);
    outQueue.FreeTensor(yReady);
  }
}

// Coalesce a preserved trailing run K across input-adjacent outer positions.
// A task maps [R,C,K] to [C,R,K] inside UB. Remaining axes use their ordinary
// input/output strides, and every task writes exact nonoverlapping segments.
template <typename T, bool InputBypass, bool OutputBypass>
__aicore__ inline void CopyPreservedBlocks(GM_ADDR x, GM_ADDR y,
                                         const TransposeTilingData &t) {
  const uint32_t k = static_cast<uint32_t>(t.planeCols);
  const uint32_t axis = t.planeGroup - 1;
  const uint32_t alignment = 32 / sizeof(T);
  const uint32_t inPitch = TransposeCeilDiv(t.tileCols * k, alignment) * alignment;
  const uint32_t outPitch = TransposeCeilDiv(t.tileRows * k, alignment) * alignment;
  TPipe pipe;
  TQue<TPosition::VECIN, 2> inputQueue;
  TQue<TPosition::VECOUT, 2> outputQueue;
  TBuf<TPosition::VECCALC> indexBuffer;
  pipe.InitBuffer(inputQueue, 2, t.tileRows * inPitch * sizeof(T));
  pipe.InitBuffer(outputQueue, 2, t.tileCols * outPitch * sizeof(T));
  pipe.InitBuffer(indexBuffer, t.tileCols * outPitch * sizeof(uint32_t));
  auto indices = indexBuffer.Get<int32_t>();
  // Vector destinations are 32-byte aligned. Descending R repairs the
  // aligned prefix overwritten by the following logical segment.
  for (uint32_t c = 0; c < t.tileCols; ++c) {
    for (int32_t r = static_cast<int32_t>(t.tileRows) - 1; r >= 0; --r) {
      const uint32_t start = c * outPitch + r * k;
      const uint32_t residue = start % 8;
      CreateVecIndex(indices[start - residue],
          static_cast<int32_t>(r * inPitch + c * k) - static_cast<int32_t>(residue),
          k + residue);
      PipeBarrier<PIPE_V>();
    }
  }
  // Unused output-row padding receives a safe offset instead of undefined
  // gather addresses. Only R*K valid elements per row are written to GM.
  for (uint32_t c = 0; c < t.tileCols; ++c)
    for (uint32_t pad = t.tileRows * k; pad < outPitch; ++pad)
      indices.SetValue(c * outPitch + pad, 0);
  SetFlag<HardEvent::S_V>(EVENT_ID0);
  WaitFlag<HardEvent::S_V>(EVENT_ID0);
  Muls(indices, indices, static_cast<int32_t>(sizeof(T)), t.tileCols * outPitch);
  PipeBarrier<PIPE_V>();
  GlobalTensor<T> input, output;
  input.SetGlobalBuffer((__gm__ T *)x);
  output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  const uint64_t rowTiles = TransposeCeilDiv(t.planeRows, t.tileRows);
  const uint64_t colTiles = TransposeCeilDiv(t.dims[axis], t.tileCols);
  uint64_t begin, end;
  TransposeTaskRange(t.taskCount, begin, end);
  for (uint64_t task = begin; task < end; ++task) {
    const uint64_t col = task % colTiles * t.tileCols;
    const uint64_t row = task / colTiles % rowTiles * t.tileRows;
    uint64_t coordinate = task / colTiles / rowTiles;
    uint64_t src = row * t.inputRowStride + col * t.inputStrides[axis];
    uint64_t dst = row * t.outputRowStride + col * t.outputStrides[axis];
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      if (static_cast<uint32_t>(d) == axis) continue;
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint32_t rows = static_cast<uint32_t>(t.planeRows - row < t.tileRows ?
        t.planeRows - row : t.tileRows);
    const uint32_t cols = static_cast<uint32_t>(t.dims[axis] - col < t.tileCols ?
        t.dims[axis] - col : t.tileCols);
    auto xLocal = inputQueue.template AllocTensor<T>();
    ReadRows(xLocal, input, src, rows, cols * k, inPitch, t.inputRowStride);
    inputQueue.EnQue(xLocal);
    auto xReady = inputQueue.template DeQue<T>();
    auto yLocal = outputQueue.template AllocTensor<T>();
    Gather(yLocal, xReady, indices.ReinterpretCast<uint32_t>(), 0, cols * outPitch);
    outputQueue.EnQue(yLocal);
    inputQueue.FreeTensor(xReady);
    auto yReady = outputQueue.template DeQue<T>();
    WriteRows(output, yReady, dst, cols, rows * k, outPitch, t.outputStrides[axis]);
    outputQueue.FreeTensor(yReady);
  }
}

template <typename T, bool InputBypass, bool OutputBypass, bool Async=true>
__aicore__ inline void CopyPreservedBlocksMask(GM_ADDR x, GM_ADDR y,
                                         const TransposeTilingData &t) {
  const uint32_t k = static_cast<uint32_t>(t.planeCols);
  const uint32_t axis = t.planeGroup - 1;
  const uint32_t alignment = 32 / sizeof(T);
  const uint32_t inPitch = TransposeCeilDiv(t.tileCols * k, alignment) * alignment;
  const uint32_t outPitch = TransposeCeilDiv(t.tileRows * k, alignment) * alignment;
  TPipe pipe;
  TQue<TPosition::VECIN, 2> inputQueue;
  TQue<TPosition::VECOUT, 2> outputQueue;
  TBuf<TPosition::VECCALC> patternBuffer;
  const uint32_t wordBits=sizeof(T)*8;
  const uint32_t patternWords=TransposeCeilDiv(
      TransposeCeilDiv(k+alignment-1,wordBits),alignment)*alignment;
  pipe.InitBuffer(inputQueue,2,t.tileRows*inPitch*sizeof(T));
  pipe.InitBuffer(outputQueue,2,t.tileCols*outPitch*sizeof(T));
  pipe.InitBuffer(patternBuffer,t.tileCols*patternWords*sizeof(T));
  auto patterns=patternBuffer.Get<T>();
  for (uint32_t c=0;c<t.tileCols;++c) {
    const uint32_t residue=(c*k)%alignment;
    for (uint32_t word=0;word<patternWords;++word) {
      const uint32_t startBit=word*wordBits;
      uint32_t lo=residue>startBit ? residue-startBit:0;
      uint32_t hi=residue+k>startBit ? residue+k-startBit:0;
      lo=lo<wordBits ? lo:wordBits;hi=hi<wordBits ? hi:wordBits;
      const uint64_t bits=hi>lo ? ((1ull<<(hi-lo))-1ull)<<lo:0;
      patterns.SetValue(c*patternWords+word,static_cast<T>(bits));
    }
  }
  SetFlag<HardEvent::S_V>(EVENT_ID0);
  WaitFlag<HardEvent::S_V>(EVENT_ID0);
  GlobalTensor<T> input, output;
  input.SetGlobalBuffer((__gm__ T *)x);
  output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  const uint64_t rowTiles = TransposeCeilDiv(t.planeRows, t.tileRows);
  const uint64_t colTiles = TransposeCeilDiv(t.dims[axis], t.tileCols);
  uint64_t begin, end;
  TransposeTaskRange(t.taskCount, begin, end);
  for (uint64_t task = begin; task < end; ++task) {
    const uint64_t col = task % colTiles * t.tileCols;
    const uint64_t row = task / colTiles % rowTiles * t.tileRows;
    uint64_t coordinate = task / colTiles / rowTiles;
    uint64_t src = row * t.inputRowStride + col * t.inputStrides[axis];
    uint64_t dst = row * t.outputRowStride + col * t.outputStrides[axis];
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      if (static_cast<uint32_t>(d) == axis) continue;
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint32_t rows = static_cast<uint32_t>(t.planeRows - row < t.tileRows ?
        t.planeRows - row : t.tileRows);
    const uint32_t cols = static_cast<uint32_t>(t.dims[axis] - col < t.tileCols ?
        t.dims[axis] - col : t.tileCols);
    auto xLocal = inputQueue.template AllocTensor<T>();
    ReadRows(xLocal, input, src, rows, cols * k, inPitch, t.inputRowStride);
    inputQueue.EnQue(xLocal);
    auto xReady = inputQueue.template DeQue<T>();
    auto yLocal = outputQueue.template AllocTensor<T>();
    for (uint32_t c=0;c<cols;++c) {
      const uint32_t residue=(c*k)%alignment;
      uint64_t retained=0;
      // All three tensor bases are32B aligned. Skip only the leading residue,
      // then append exactly K valid values from each retained input row.
      if constexpr (Async) {
        // CANN8.5 supplies this count-free overload. Valid count is rows*K.
        // It is deprecated for future CANN releases; this build targets8.5.
        SetMaskCount();
        SetVectorMask<T,MaskMode::COUNTER>(k+residue);
        GatherMask(yLocal[c*outPitch],xReady[c*k-residue],patterns[c*patternWords],
                   static_cast<uint8_t>(0),
                   {1,static_cast<uint16_t>(rows),static_cast<uint16_t>(inPitch/alignment),0});
        SetMaskNorm();
        ResetMask();
      } else {
      GatherMask(yLocal[c*outPitch],xReady[c*k-residue],patterns[c*patternWords],
                 true,k+residue,
                 {1,static_cast<uint16_t>(rows),static_cast<uint16_t>(inPitch/alignment),0},retained);
      }
      PipeBarrier<PIPE_V>();
    }
    outputQueue.EnQue(yLocal);
    inputQueue.FreeTensor(xReady);
    auto yReady = outputQueue.template DeQue<T>();
    WriteRows(output, yReady, dst, cols, rows * k, outPitch, t.outputStrides[axis]);
    outputQueue.FreeTensor(yReady);
  }
}

template <bool InputBypass, bool OutputBypass>
__aicore__ inline void DispatchTranspose(GM_ADDR x, GM_ADDR y, const TransposeTilingData &t) {
  if (t.mode == 26) {
    if constexpr (sizeof(DTYPE_X)==2) CopyPreservedBlocksMask<uint16_t,InputBypass,OutputBypass>(x,y,t);
    else CopyPreservedBlocksMask<uint32_t,InputBypass,OutputBypass>(x,y,t);
  }
  else if (t.mode == 16) {
    if constexpr (sizeof(DTYPE_X) == 2) CopyPreservedBlocks<uint16_t, InputBypass, OutputBypass>(x, y, t);
    else CopyPreservedBlocks<uint32_t, InputBypass, OutputBypass>(x, y, t);
  }
  else if (t.mode == 0) CopyPreservedLastAxis<DTYPE_X, InputBypass, OutputBypass>(x, y, t);
  else if (t.mode == 10) CopyPackedPreservedRows<DTYPE_X, InputBypass, OutputBypass>(x, y, t);
  else if (t.mode == 2) CopyTransposedPlanesCursor<DTYPE_X, 1, InputBypass, OutputBypass>(x, y, t);
  else if (t.mode == 3) CopyTransposedPlanesCursor<DTYPE_X, 2, InputBypass, OutputBypass>(x, y, t);
  else if (t.mode == 9) CopyGroupedOutputPlanes<InputBypass, OutputBypass>(x, y, t);
  else if (t.mode == 13) CopyResidentPackedInputPlanes<InputBypass, OutputBypass, true>(x, y, t);
  else if (t.mode == 6) CopyResidentPackedInputPlanes<InputBypass, OutputBypass>(x, y, t);
  else if (t.mode == 8) CopyBytePackedInputWindows<InputBypass, OutputBypass>(x, y, t);
  else if (t.mode == 7) {
    if constexpr (sizeof(DTYPE_X) == 2) CopyNarrowPackedInputWindows<uint16_t, InputBypass, OutputBypass>(x, y, t);
    else CopyNarrowPackedInputWindows<uint32_t, InputBypass, OutputBypass>(x, y, t);
  }
  else CopyTransposedPlanes<DTYPE_X, InputBypass, OutputBypass>(x, y, t);
}

// Each core owns its scratch slot and an exact input-row window. Padding
// and transpose happen on that core, so no cross-core barrier is needed.
// Scratch contains no state from another invocation: every read follows a
// completed write in this task. Output windows are disjoint byte intervals.
__aicore__ inline void CopyStagedInputWindows(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                             const TransposeTilingData &t) {
  constexpr uint32_t capacity = 16384;
  constexpr uint32_t indexCapacity = 8192;
  const uint32_t columns = static_cast<uint32_t>(t.planeCols);
  const uint32_t alignment = (t.planeGroup & 2u) ? 128 : 8;
  const uint32_t pitch = TransposeCeilDiv(columns, alignment) * alignment;
  const bool gatherPadding = (t.planeGroup & 1u) != 0;
  const uint32_t padRows = (gatherPadding ? indexCapacity : capacity) / pitch;
  const uint64_t slotElements = TransposeCeilDiv(static_cast<uint64_t>(t.tileRows) * pitch, 128) * 128;
  GlobalTensor<uint32_t> input, output, scratch;
  input.SetGlobalBuffer((__gm__ uint32_t *)x);
  // The staged family has long, one-pass input rows. Keep the private
  // scratch window in cache instead of admitting the full input tensor.
  input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ uint32_t *)y);
  scratch.SetGlobalBuffer((__gm__ uint32_t *)GetUserWorkspace(workspace) + GetBlockIdx() * slotElements);
  TPipe pipe;
  TBuf<TPosition::VECIN> inputBuffer;
  TBuf<TPosition::VECOUT> outputBuffer;
  TBuf<TPosition::VECCALC> indexBuffer;
  pipe.InitBuffer(inputBuffer, capacity * 4);
  pipe.InitBuffer(outputBuffer, capacity * 4);
  pipe.InitBuffer(indexBuffer, indexCapacity * 4);
  auto xLocal = inputBuffer.Get<uint32_t>();
  auto yLocal = outputBuffer.Get<uint32_t>();
  auto indices = indexBuffer.Get<int32_t>();
  if (gatherPadding) {
    for (uint32_t row = 0; row < padRows; ++row)
      CreateVecIndex(indices[row * pitch], static_cast<int32_t>(row * columns), pitch);
    PipeBarrier<PIPE_V>();
    Muls(indices, indices, static_cast<int32_t>(4), padRows * pitch);
    PipeBarrier<PIPE_V>();
  }
  const uint64_t windows = TransposeCeilDiv(t.planeRows, t.tileRows);
  for (uint64_t task = GetBlockIdx(); task < t.taskCount; task += GetBlockNum()) {
    uint64_t coordinate = task / windows, src = 0, dst = 0;
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint64_t rowStart = (task % windows) * t.tileRows;
    const uint32_t rows = static_cast<uint32_t>(t.planeRows - rowStart < t.tileRows ?
        t.planeRows - rowStart : t.tileRows);
    src += rowStart * columns;
    dst += rowStart;
    for (uint32_t row = 0; row < rows; row += padRows) {
      const uint32_t count = rows - row < padRows ? rows - row : padRows;
      if (gatherPadding) {
        DataCopyExtParams copy{1, count * columns * 4, 0, 0, 0};
        DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        DataCopyPad(xLocal, input[src + static_cast<uint64_t>(row) * columns], copy, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Gather(yLocal, xLocal, indices.ReinterpretCast<uint32_t>(), 0, count * pitch);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(scratch[static_cast<uint64_t>(row) * pitch], yLocal, count * pitch);
      } else {
        ReadRows(xLocal, input, src + static_cast<uint64_t>(row) * columns,
                 count, columns, pitch, columns);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(scratch[static_cast<uint64_t>(row) * pitch], xLocal, count * pitch);
      }
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t col = 0; col < columns; col += t.tileCols) {
      const uint32_t cols = columns - col < t.tileCols ? columns - col : t.tileCols;
      ReadRows(xLocal, scratch, col, rows, cols, t.tileCols, pitch);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      TransposeTile(yLocal, xLocal, rows, cols, t.tileRows, t.tileCols);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WriteRows(output, yLocal, dst + static_cast<uint64_t>(col) * t.outputRowStride,
                cols, rows, t.tileRows, t.outputRowStride);
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
  }
}

// Padded per-core windows with three UB queue slots. The transpose loop
// can overlap the next input transfer with the preceding output transfer.
template <int InputDepth, int OutputDepth, bool InputStream = true, bool GroupOutput = false>
__aicore__ inline void CopyStagedOverlap(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                         const TransposeTilingData &t) {
  const uint32_t capacity = t.tileRows * t.tileCols;
  const uint32_t columns = static_cast<uint32_t>(t.planeCols);
  const uint32_t alignment = (t.planeGroup & 2u) ? 128 : 8;
  const uint32_t pitch = TransposeCeilDiv(columns, alignment) * alignment;
  const uint32_t padRows = capacity / pitch;
  const uint64_t slotElements = TransposeCeilDiv(static_cast<uint64_t>(t.tileRows) * pitch, 128) * 128;
  GlobalTensor<uint32_t> input, output, scratch;
  input.SetGlobalBuffer((__gm__ uint32_t *)x);
  if constexpr (InputStream) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ uint32_t *)y);
  scratch.SetGlobalBuffer((__gm__ uint32_t *)GetUserWorkspace(workspace) + GetBlockIdx() * slotElements);
  TPipe pipe;
  TQue<TPosition::VECIN, InputDepth> inQueue;
  TQue<TPosition::VECOUT, OutputDepth> outQueue;
  pipe.InitBuffer(inQueue, InputDepth, capacity * 4);
  pipe.InitBuffer(outQueue, OutputDepth, capacity * 4);
  const uint64_t windows = TransposeCeilDiv(t.planeRows, t.tileRows);
  for (uint64_t task = GetBlockIdx(); task < t.taskCount; task += GetBlockNum()) {
    uint64_t coordinate = task / windows, src = 0, dst = 0;
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint64_t rowStart = (task % windows) * t.tileRows;
    const uint32_t rows = static_cast<uint32_t>(t.planeRows - rowStart < t.tileRows ?
        t.planeRows - rowStart : t.tileRows);
    src += rowStart * columns;
    dst += rowStart;
    for (uint32_t row = 0; row < rows; row += padRows) {
      const uint32_t count = rows - row < padRows ? rows - row : padRows;
      auto local = inQueue.template AllocTensor<uint32_t>();
      ReadRows(local, input, src + static_cast<uint64_t>(row) * columns,
               count, columns, pitch, columns);
      inQueue.EnQue(local);
      auto ready = inQueue.template DeQue<uint32_t>();
      SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
      DataCopy(scratch[static_cast<uint64_t>(row) * pitch], ready, count * pitch);
      SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      inQueue.FreeTensor(ready);
    }
    PipeBarrier<PIPE_ALL>();
    for (uint32_t col = 0; col < columns; col += t.tileCols) {
      const uint32_t cols = columns - col < t.tileCols ? columns - col : t.tileCols;
      auto xLocal = inQueue.template AllocTensor<uint32_t>();
      ReadRows(xLocal, scratch, col, rows, cols, t.tileCols, pitch);
      inQueue.EnQue(xLocal);
      auto xReady = inQueue.template DeQue<uint32_t>();
      auto yLocal = outQueue.template AllocTensor<uint32_t>();
      TransposeTile(yLocal, xReady, rows, cols, t.tileRows, t.tileCols);
      outQueue.EnQue(yLocal);
      inQueue.FreeTensor(xReady);
      auto yReady = outQueue.template DeQue<uint32_t>();
      WriteRows(output, yReady, dst + static_cast<uint64_t>(col) * t.outputRowStride,
                cols, rows, t.tileRows, t.outputRowStride, GroupOutput);
      outQueue.FreeTensor(yReady);
    }
  }
}

// Each core owns a column window and a private aligned output scratch slot.
// Scratch is produced and consumed within one task; no cross-launch state or
// cross-core synchronization exists. Final GM stores cover exact output rows.
template <bool StageOverlap, bool PackOverlap>
__aicore__ inline void CopyOutputOverlap(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                      const TransposeTilingData &t) {
  const uint32_t capacity = t.tileRows * t.tileCols;
  const uint32_t pitch = TransposeCeilDiv(t.planeRows, 128) * 128;
  const uint32_t compactRows = capacity / pitch;
  const uint64_t slot = TransposeCeilDiv(static_cast<uint64_t>(t.tileCols) * pitch, 128) * 128;
  GlobalTensor<uint32_t> input, output, scratch;
  input.SetGlobalBuffer((__gm__ uint32_t *)x);
  output.SetGlobalBuffer((__gm__ uint32_t *)y);
  output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  scratch.SetGlobalBuffer((__gm__ uint32_t *)GetUserWorkspace(workspace) + GetBlockIdx() * slot);
  TPipe pipe;
  TBuf<TPosition::VECIN> inputBuffer;
  TBuf<TPosition::VECOUT> outputBuffer;
  pipe.InitBuffer(inputBuffer, capacity * 4);
  pipe.InitBuffer(outputBuffer, capacity * 4);
  auto xLocal = inputBuffer.Get<uint32_t>();
  auto yLocal = outputBuffer.Get<uint32_t>();
  const uint64_t windows = TransposeCeilDiv(t.planeCols, t.tileCols);
  for (uint64_t task = GetBlockIdx(); task < t.taskCount; task += GetBlockNum()) {
    uint64_t coordinate = task / windows, src = 0, dst = 0;
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint64_t columnStart = (task % windows) * t.tileCols;
    const uint32_t cols = static_cast<uint32_t>(t.planeCols - columnStart < t.tileCols ?
        t.planeCols - columnStart : t.tileCols);
    src += columnStart;
    dst += columnStart * t.outputRowStride;
    if constexpr (StageOverlap) {
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    for (uint32_t row = 0; row < t.planeRows; row += t.tileRows) {
      const uint32_t rows = t.planeRows - row < t.tileRows ? t.planeRows - row : t.tileRows;
      if constexpr (StageOverlap) WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      ReadRows(xLocal, input, src + static_cast<uint64_t>(row) * t.inputRowStride,
               rows, cols, t.tileCols, t.inputRowStride, (t.planeGroup & 1u) != 0);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      if constexpr (StageOverlap) WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      TransposeTile(yLocal, xLocal, rows, cols, t.tileRows, t.tileCols);
      if constexpr (StageOverlap) SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WriteRows(scratch, yLocal, row, cols, rows, t.tileRows, pitch);
      if constexpr (StageOverlap) SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      else {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
    }
    if constexpr (StageOverlap) {
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
    if constexpr (PackOverlap) {
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    for (uint32_t col = 0; col < cols; col += compactRows) {
      const uint32_t count = cols - col < compactRows ? cols - col : compactRows;
      // Read the full owned padded rows contiguously. GatherMask selects
      // only the R elements produced by this task; all padding is discarded.
      if constexpr (PackOverlap) WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      DataCopy(xLocal, scratch[static_cast<uint64_t>(col) * pitch], count * pitch);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      if constexpr (PackOverlap) WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      uint64_t retained = 0;
      GatherMask(yLocal, xLocal, static_cast<uint8_t>(7), true,
          static_cast<uint32_t>(t.planeRows),
          {1, static_cast<uint16_t>(count), static_cast<uint16_t>(pitch / 8), 0}, retained);
      if constexpr (PackOverlap) SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopyExtParams copy{1, static_cast<uint32_t>(count * t.planeRows * 4), 0, 0, 0};
      DataCopyPad(output[dst + static_cast<uint64_t>(col) * t.outputRowStride], yLocal, copy);
      if constexpr (PackOverlap) SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      else {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
    }
    if constexpr (PackOverlap) {
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
  }
}

// Every transpose operand remains inside its padded UB allocation.
// GM writes retain exact valid rows and columns; padded values are discarded.
template <uint32_t InputPitch,uint32_t OutputPitch>
__aicore__ inline void TransposePaddedFixed32(const LocalTensor<uint32_t> &dst,
                                             const LocalTensor<uint32_t> &src) {
  constexpr uint32_t Rows=128, Cols=176;
  uint64_t srcList[16],dstList[16];
  const uint64_t srcBase=(uint64_t)src.GetPhyAddr();
  const uint64_t dstBase=(uint64_t)dst.GetPhyAddr();
  TransDataTo5HDParams params(false,false,static_cast<uint8_t>(Cols/8),
      static_cast<uint16_t>(OutputPitch),static_cast<uint16_t>(1));
  for (uint32_t r=0;r<Rows;r+=16) {
    for (uint32_t i=0;i<16;++i) {
      srcList[i]=srcBase+((r+i)*InputPitch)*4;
      dstList[i]=dstBase+((i/2)*OutputPitch+r+(i%2)*8)*4;
    }
    TransDataTo5HD<uint32_t>(dstList,srcList,params);
  }
}

// Each core owns a column window and a private aligned output scratch slot.
// Scratch is produced and consumed within one task; no cross-launch state or
// cross-core synchronization exists. Final GM stores cover exact output rows.
template <uint32_t InputPitch, uint32_t OutputPitch, bool Fixed=false>
__aicore__ inline void CopyOutputBankPadded(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                      const TransposeTilingData &t) {
  constexpr uint32_t logicalRows=128, logicalCols=176;
  constexpr uint32_t inputCapacity=logicalRows*InputPitch;
  constexpr uint32_t outputCapacity=OutputPitch*logicalCols;
  constexpr uint32_t capacity=inputCapacity<outputCapacity ? inputCapacity:outputCapacity;
  constexpr bool StageOverlap=true, PackOverlap=true;
  static_assert(InputPitch>=logicalCols && OutputPitch>=logicalRows, "physical pitch");
  static_assert(InputPitch%8==0 && OutputPitch%8==0, "block alignment");
  const uint32_t pitch = TransposeCeilDiv(t.planeRows, 128) * 128;
  const uint32_t compactRows = capacity / pitch;
  const uint64_t slot = TransposeCeilDiv(static_cast<uint64_t>(logicalCols) * pitch, 128) * 128;
  GlobalTensor<uint32_t> input, output, scratch;
  input.SetGlobalBuffer((__gm__ uint32_t *)x);
  output.SetGlobalBuffer((__gm__ uint32_t *)y);
  output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  scratch.SetGlobalBuffer((__gm__ uint32_t *)GetUserWorkspace(workspace) + GetBlockIdx() * slot);
  TPipe pipe;
  TBuf<TPosition::VECIN> inputBuffer;
  TBuf<TPosition::VECOUT> outputBuffer;
  pipe.InitBuffer(inputBuffer, inputCapacity * 4);
  pipe.InitBuffer(outputBuffer, outputCapacity * 4);
  auto xLocal = inputBuffer.Get<uint32_t>();
  auto yLocal = outputBuffer.Get<uint32_t>();
  const uint64_t windows = TransposeCeilDiv(t.planeCols, logicalCols);
  for (uint64_t task = GetBlockIdx(); task < t.taskCount; task += GetBlockNum()) {
    uint64_t coordinate = task / windows, src = 0, dst = 0;
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint64_t columnStart = (task % windows) * logicalCols;
    const uint32_t cols = static_cast<uint32_t>(t.planeCols - columnStart < logicalCols ?
        t.planeCols - columnStart : logicalCols);
    src += columnStart;
    dst += columnStart * t.outputRowStride;
    if constexpr (StageOverlap) {
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    for (uint32_t row = 0; row < t.planeRows; row += logicalRows) {
      const uint32_t rows = t.planeRows - row < logicalRows ? t.planeRows - row : logicalRows;
      if constexpr (StageOverlap) WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      ReadRows(xLocal, input, src + static_cast<uint64_t>(row) * t.inputRowStride,
               rows, cols, InputPitch, t.inputRowStride, (t.planeGroup & 1u) != 0);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      if constexpr (StageOverlap) WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      if constexpr (Fixed) TransposePaddedFixed32<InputPitch,OutputPitch>(yLocal,xLocal);
      else TransposeTile(yLocal, xLocal, rows, cols, OutputPitch, InputPitch);
      if constexpr (StageOverlap) SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WriteRows(scratch, yLocal, row, cols, rows, OutputPitch, pitch);
      if constexpr (StageOverlap) SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      else {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
    }
    if constexpr (StageOverlap) {
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    PipeBarrier<PIPE_ALL>();
    if constexpr (PackOverlap) {
      SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
    for (uint32_t col = 0; col < cols; col += compactRows) {
      const uint32_t count = cols - col < compactRows ? cols - col : compactRows;
      // Read the full owned padded rows contiguously. GatherMask selects
      // only the R elements produced by this task; all padding is discarded.
      if constexpr (PackOverlap) WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      DataCopy(xLocal, scratch[static_cast<uint64_t>(col) * pitch], count * pitch);
      SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
      WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
      if constexpr (PackOverlap) WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
      uint64_t retained = 0;
      GatherMask(yLocal, xLocal, static_cast<uint8_t>(7), true,
          static_cast<uint32_t>(t.planeRows),
          {1, static_cast<uint16_t>(count), static_cast<uint16_t>(pitch / 8), 0}, retained);
      if constexpr (PackOverlap) SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
      SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
      WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
      DataCopyExtParams copy{1, static_cast<uint32_t>(count * t.planeRows * 4), 0, 0, 0};
      DataCopyPad(output[dst + static_cast<uint64_t>(col) * t.outputRowStride], yLocal, copy);
      if constexpr (PackOverlap) SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
      else {
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
      }
    }
    if constexpr (PackOverlap) {
      WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
      WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
  }
}

template <typename T, bool InputBypass, bool OutputBypass>
__aicore__ inline void CopyStaticPlanes(GM_ADDR x, GM_ADDR y, const TransposeTilingData &t) {
  GlobalTensor<T> input, output;
  input.SetGlobalBuffer((__gm__ T *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  constexpr uint32_t Depth=2;
  const uint32_t tileBytes=t.tileRows*t.tileCols*sizeof(T);
  uint32_t iteration=0;
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  if constexpr (Depth == 2) {
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }
  for (uint64_t task = GetBlockIdx(); task < t.taskCount; task += GetBlockNum(), ++iteration) {
    const uint32_t slot=iteration%Depth;
    uint64_t src, dst;
    uint32_t rows, cols;
    DecodeTask(t, task, src, dst, rows, cols);
    if (slot == 0) WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    else WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    LocalTensor<T> xLocal(TPosition::VECCALC, slot*tileBytes, t.tileRows*t.tileCols);
    ReadRows(xLocal, input, src, rows, cols, t.tileCols, t.inputRowStride);
    if (slot == 0) SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    else SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    if (slot == 0) WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    else WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    auto xReady = xLocal;
    if (slot == 0) WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    else WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    LocalTensor<T> yLocal(TPosition::VECCALC, (Depth+slot)*tileBytes, t.tileRows*t.tileCols);
    TransposeTile(yLocal, xReady, rows, cols, t.tileRows, t.tileCols);
    if (slot == 0) SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    else SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    if (slot == 0) SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    else SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    if (slot == 0) WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    else WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    auto yReady = yLocal;
    WriteRows(output, yReady, dst, cols, rows, t.tileRows, t.outputRowStride);
    if (slot == 0) SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    else SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  if constexpr (Depth == 2) {
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }
}

template <typename T, uint32_t Depth, bool InputBypass, bool OutputBypass>
__aicore__ inline void CopyStaticPlanesCursor(GM_ADDR x, GM_ADDR y, const TransposeTilingData &t) {
  GlobalTensor<T> input, output;
  input.SetGlobalBuffer((__gm__ T *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  const uint32_t tileBytes = t.tileRows * t.tileCols * sizeof(T);
  uint64_t begin, end;
  TransposeTaskRange(t.taskCount, begin, end);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  if constexpr (Depth == 2) {
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }
  TransposeTaskCursor cursor(t, begin);
  for (uint64_t task = begin; task < end; ++task) {
    const uint32_t slot = (task - begin) % Depth;
    uint64_t src, dst;
    uint32_t rows, cols;
    cursor.Get(src, dst, rows, cols);
    if (slot == 0) WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    else WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    LocalTensor<T> xLocal(TPosition::VECCALC, slot * tileBytes, t.tileRows * t.tileCols);
    ReadRows(xLocal, input, src, rows, cols, t.tileCols, t.inputRowStride);
    if (slot == 0) SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    else SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    if (slot == 0) WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    else WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    auto xReady = xLocal;
    if (slot == 0) WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    else WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    LocalTensor<T> yLocal(TPosition::VECCALC, (Depth + slot) * tileBytes, t.tileRows * t.tileCols);
    TransposeTile(yLocal, xReady, rows, cols, t.tileRows, t.tileCols);
    if (slot == 0) SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    else SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    if (slot == 0) SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    else SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    if (slot == 0) WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    else WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    auto yReady = yLocal;
    WriteRows(output, yReady, dst, cols, rows, t.tileRows, t.outputRowStride);
    if (slot == 0) SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    else SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
    if (task + 1 < end) cursor.Next();
  }
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  if constexpr (Depth == 2) {
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }
}

template <bool InputBypass, bool OutputBypass>
__aicore__ inline void CopyStaticGroupedPlanes(GM_ADDR x, GM_ADDR y,
                                               const TransposeTilingData &t) {
  GlobalTensor<uint32_t> input, output;
  input.SetGlobalBuffer((__gm__ uint32_t *)x);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  output.SetGlobalBuffer((__gm__ uint32_t *)y);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  const uint32_t tileBytes = t.tileRows * t.tileCols * sizeof(uint32_t);
  const uint32_t adjacentAxis = t.rank - 1;
  const uint64_t groupsPerPrefix = TransposeCeilDiv(t.dims[adjacentAxis], t.planeGroup);
  const uint64_t columnTiles = TransposeCeilDiv(t.planeCols, t.tileCols);
  uint64_t begin, end;
  TransposeTaskRange(t.taskCount, begin, end);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  for (uint64_t task = begin; task < end; ++task) {
    const uint64_t col = (task % columnTiles) * t.tileCols;
    uint64_t coordinate = task / columnTiles;
    const uint64_t firstPlane = (coordinate % groupsPerPrefix) * t.planeGroup;
    coordinate /= groupsPerPrefix;
    uint64_t src = firstPlane * t.inputStrides[adjacentAxis] + col;
    uint64_t dst = firstPlane * t.planeRows + col * t.outputRowStride;
    for (int32_t d = static_cast<int32_t>(adjacentAxis) - 1; d >= 0; --d) {
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint32_t planes = static_cast<uint32_t>(t.dims[adjacentAxis] - firstPlane < t.planeGroup ?
        t.dims[adjacentAxis] - firstPlane : t.planeGroup);
    const uint32_t rows = planes * static_cast<uint32_t>(t.planeRows);
    const uint32_t cols = static_cast<uint32_t>(t.planeCols - col < t.tileCols ?
        t.planeCols - col : t.tileCols);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    LocalTensor<uint32_t> xLocal(TPosition::VECCALC, 0, t.tileRows * t.tileCols);
    for (uint32_t plane = 0; plane < planes; ++plane) {
      ReadRows(xLocal[plane * static_cast<uint32_t>(t.planeRows) * t.tileCols],
          input, src + plane * t.inputStrides[adjacentAxis],
          static_cast<uint32_t>(t.planeRows), cols, t.tileCols, t.inputRowStride, false);
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    auto xReady = xLocal;
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    LocalTensor<uint32_t> yLocal(TPosition::VECCALC, tileBytes, t.tileRows * t.tileCols);
    TransposeTile(yLocal, xReady, rows, cols, t.tileRows, t.tileCols);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    auto yReady = yLocal;
    WriteRows(output, yReady, dst, cols, rows, t.tileRows, t.outputRowStride);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  }
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
}

template <typename T, bool InputBypass, bool OutputBypass, uint32_t Depth=2>
__aicore__ inline void CopyStaticBlocksMask(GM_ADDR x, GM_ADDR y,
                                         const TransposeTilingData &t) {
  const uint32_t k = static_cast<uint32_t>(t.planeCols);
  const uint32_t axis = t.planeGroup - 1;
  const uint32_t alignment = 32 / sizeof(T);
  const uint32_t inPitch = TransposeCeilDiv(t.tileCols * k, alignment) * alignment;
  const uint32_t outPitch = TransposeCeilDiv(t.tileRows * k, alignment) * alignment;
  const uint32_t wordBits=sizeof(T)*8;
  const uint32_t patternWords=TransposeCeilDiv(
      TransposeCeilDiv(k+alignment-1,wordBits),alignment)*alignment;
  const uint32_t inputBytes=t.tileRows*inPitch*sizeof(T);
  const uint32_t outputBytes=t.tileCols*outPitch*sizeof(T);
  LocalTensor<T> patterns(TPosition::VECCALC, Depth*(inputBytes+outputBytes),
                          t.tileCols*patternWords);
  for (uint32_t c=0;c<t.tileCols;++c) {
    const uint32_t residue=(c*k)%alignment;
    for (uint32_t word=0;word<patternWords;++word) {
      const uint32_t startBit=word*wordBits;
      uint32_t lo=residue>startBit ? residue-startBit:0;
      uint32_t hi=residue+k>startBit ? residue+k-startBit:0;
      lo=lo<wordBits ? lo:wordBits;hi=hi<wordBits ? hi:wordBits;
      const uint64_t bits=hi>lo ? ((1ull<<(hi-lo))-1ull)<<lo:0;
      patterns.SetValue(c*patternWords+word,static_cast<T>(bits));
    }
  }
  SetFlag<HardEvent::S_V>(EVENT_ID0);
  WaitFlag<HardEvent::S_V>(EVENT_ID0);
  GlobalTensor<T> input, output;
  input.SetGlobalBuffer((__gm__ T *)x);
  output.SetGlobalBuffer((__gm__ T *)y);
  if constexpr (InputBypass) input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  if constexpr (OutputBypass) output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  const uint64_t rowTiles = TransposeCeilDiv(t.planeRows, t.tileRows);
  const uint64_t colTiles = TransposeCeilDiv(t.dims[axis], t.tileCols);
  uint64_t begin, end;
  TransposeTaskRange(t.taskCount, begin, end);
  SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
  SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
  if constexpr (Depth == 2) {
    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }
  for (uint64_t task = begin; task < end; ++task) {
    const uint32_t slot = (task - begin) % Depth;
    const uint64_t col = task % colTiles * t.tileCols;
    const uint64_t row = task / colTiles % rowTiles * t.tileRows;
    uint64_t coordinate = task / colTiles / rowTiles;
    uint64_t src = row * t.inputRowStride + col * t.inputStrides[axis];
    uint64_t dst = row * t.outputRowStride + col * t.outputStrides[axis];
    for (int32_t d = static_cast<int32_t>(t.rank) - 1; d >= 0; --d) {
      if (static_cast<uint32_t>(d) == axis) continue;
      const uint64_t index = coordinate % t.dims[d];
      coordinate /= t.dims[d];
      src += index * t.inputStrides[d];
      dst += index * t.outputStrides[d];
    }
    const uint32_t rows = static_cast<uint32_t>(t.planeRows - row < t.tileRows ?
        t.planeRows - row : t.tileRows);
    const uint32_t cols = static_cast<uint32_t>(t.dims[axis] - col < t.tileCols ?
        t.dims[axis] - col : t.tileCols);
    if (slot == 0) WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    else WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    LocalTensor<T> xLocal(TPosition::VECCALC, slot*inputBytes, t.tileRows*inPitch);
    ReadRows(xLocal, input, src, rows, cols * k, inPitch, t.inputRowStride);
    if (slot == 0) SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    else SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    if (slot == 0) WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    else WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    auto xReady = xLocal;
    if (slot == 0) WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    else WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
    LocalTensor<T> yLocal(TPosition::VECCALC, Depth*inputBytes+slot*outputBytes, t.tileCols*outPitch);
    for (uint32_t c=0;c<cols;++c) {
      const uint32_t residue=(c*k)%alignment;
      uint64_t retained=0;
      // All three tensor bases are32B aligned. Skip only the leading residue,
      // then append exactly K valid values from each retained input row.
      SetMaskCount();
      SetVectorMask<T,MaskMode::COUNTER>(k+residue);
      GatherMask(yLocal[c*outPitch],xReady[c*k-residue],patterns[c*patternWords],
                 static_cast<uint8_t>(0),
                 {1,static_cast<uint16_t>(rows),static_cast<uint16_t>(inPitch/alignment),0});
      SetMaskNorm();
      ResetMask();
      PipeBarrier<PIPE_V>();
    }
    if (slot == 0) SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    else SetFlag<HardEvent::V_MTE3>(EVENT_ID1);
    if (slot == 0) SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
    else SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    if (slot == 0) WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    else WaitFlag<HardEvent::V_MTE3>(EVENT_ID1);
    auto yReady = yLocal;
    WriteRows(output, yReady, dst, cols, rows * k, outPitch, t.outputStrides[axis]);
    if (slot == 0) SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
    else SetFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }
  WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
  WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
  if constexpr (Depth == 2) {
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);
  }
}


template <bool InputBypass, bool OutputBypass>
__aicore__ inline void DispatchStaticTensor(GM_ADDR x, GM_ADDR y,
                                            const TransposeTilingData &t) {
  if (t.mode == 1) CopyStaticPlanes<DTYPE_X,InputBypass,OutputBypass>(x,y,t);
  else if (t.mode == 2) CopyStaticPlanesCursor<DTYPE_X,1,InputBypass,OutputBypass>(x,y,t);
  else if (t.mode == 3) CopyStaticPlanesCursor<DTYPE_X,2,InputBypass,OutputBypass>(x,y,t);
  else if (t.mode == 9) CopyStaticGroupedPlanes<InputBypass,OutputBypass>(x,y,t);
  else if (t.mode == 26) {
    if constexpr (sizeof(DTYPE_X)==2) CopyStaticBlocksMask<uint16_t,InputBypass,OutputBypass>(x,y,t);
    else CopyStaticBlocksMask<uint32_t,InputBypass,OutputBypass>(x,y,t);
  }
}


extern "C" __global__ __aicore__ void transpose(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
  GET_TILING_DATA(t, tiling);
  if (t.total == 0) return;
  if (TILING_KEY_IS(20)) {
    InitSocState();
    DispatchStaticTensor<false,false>(x,y,t);
    PipeBarrier<PIPE_ALL>();
  } else if (TILING_KEY_IS(21)) {
    InitSocState();
    DispatchStaticTensor<true,false>(x,y,t);
    PipeBarrier<PIPE_ALL>();
  } else if (TILING_KEY_IS(23)) {
    InitSocState();
    DispatchStaticTensor<true,true>(x,y,t);
    PipeBarrier<PIPE_ALL>();
  } else if (TILING_KEY_IS(16)) CopyOutputBankPadded<184,144,true>(x,y,workspace,t);
  else if (TILING_KEY_IS(7)) CopyOutputOverlap<true,true>(x,y,workspace,t);
  else if (TILING_KEY_IS(6)) CopyStagedOverlap<1,1,true,false>(x,y,workspace,t);
  else if (TILING_KEY_IS(5)) CopyStagedInputWindows(x, y, workspace, t);
  else if (TILING_KEY_IS(4)) {
    if constexpr (sizeof(DTYPE_X) == 1)
      CopyBytePackedInputWindows<false, false, true>(x, y, t);
    else if constexpr (sizeof(DTYPE_X) == 2)
      CopyNarrowPackedInputWindows<uint16_t, false, false, true>(x, y, t);
    else
      CopyNarrowPackedInputWindows<uint32_t, false, false, true>(x, y, t);
  }
  else if (TILING_KEY_IS(0)) DispatchTranspose<false, false>(x, y, t);
  else if (TILING_KEY_IS(1)) DispatchTranspose<true, false>(x, y, t);
  else if (TILING_KEY_IS(3)) DispatchTranspose<true, true>(x, y, t);
}
