#include "concat_layout.h"
#include "kernel_operator.h"

namespace {
using namespace compact;
template <class T> __aicore__ inline T LowerBoundValue(T a, T b) { return a < b ? a : b; }

template <class Local, class Global>
__aicore__ inline __attribute__((always_inline)) void
LoadSpanBytes(__ubuf__ Local *to, __gm__ Global *from, uint32_t bytes,
          uint16_t rows = 1) {
  copy_gm_to_ubuf_align_b8(reinterpret_cast<__ubuf__ int8_t *>(to),
                           reinterpret_cast<const __gm__ int8_t *>(from), 0,
                           rows, bytes, 0, 0, 0, 0);
}
template <class Global, class Local>
__aicore__ inline __attribute__((always_inline)) void
StoreSpanBytes(__gm__ Global *to, __ubuf__ Local *from, uint32_t bytes,
           uint16_t rows = 1, uint32_t gmGap = 0, uint32_t ubGap = 0) {
  copy_ubuf_to_gm_align_b8(reinterpret_cast<__gm__ int8_t *>(to),
                           reinterpret_cast<__ubuf__ int8_t *>(from), 0, rows,
                           bytes, 0, 0, ubGap, gmGap);
}

#define CONCAT_DMA_FENCE(PRODUCER, CONSUMER, EVENT) \
  do { \
    set_flag(PRODUCER, CONSUMER, EVENT); \
    wait_flag(PRODUCER, CONSUMER, EVENT); \
  } while (false)

template <AscendC::HardEvent Event>
__aicore__ inline __attribute__((always_inline)) void CompleteVectorTransfer() {
  AscendC::SetFlag<Event>(EVENT_ID0);
  AscendC::WaitFlag<Event>(EVENT_ID0);
}

__aicore__ void ReadSlab(__ubuf__ int8_t *buffer, __ubuf__ Plan *p,
                         __ubuf__ uint64_t *ptr, uint32_t const first, uint32_t const last,
                         uint64_t const row, uint32_t const n, uint32_t const slot,
                         uint32_t const batch) {
  uint32_t offset = 0;
  for (uint32_t i = first; (i < last); ++i) {
    uint32_t const width = p->width[i];
    if (width) {
      auto *src = AscendC::L2CacheAlter<int8_t, AscendC::CacheRwMode::READ>(
          reinterpret_cast<__gm__ int8_t *>(ptr[i]),
          AscendC::CacheMode::CACHE_MODE_DISABLE);
      auto *from = src + row * width;
      auto *to = buffer + offset;
      const uint32_t bytes = n * width;
      const uint64_t alignment = reinterpret_cast<uint64_t>(from) | reinterpret_cast<uint64_t>(to) | bytes;
      if ((alignment & 31U) == 0)
        copy_gm_to_ubuf(reinterpret_cast<__ubuf__ void *>(to), reinterpret_cast<__gm__ void *>(from),
                        0, 1, bytes / 32, 0, 0);
      else
        LoadSpanBytes(to, from, bytes);
    }
    offset += batch * width;
  }
  set_flag(PIPE_MTE2, PIPE_V, slot ? event_t::EVENT_ID1 : event_t::EVENT_ID0);
}

__aicore__ void ReadSelectedSlab(__ubuf__ int8_t *buffer, const __ubuf__ uint64_t *widths,
                         __ubuf__ uint64_t *ptr, uint32_t const first, uint32_t const last,
                         uint64_t const row, uint32_t const n, uint32_t const slot,
                         uint32_t const batch) {
  uint32_t offset = 0;
  for (uint32_t i = first; (i < last); ++i) {
    uint32_t const width = widths[i];
    if (width) {
      auto *src = AscendC::L2CacheAlter<int8_t, AscendC::CacheRwMode::READ>(
          reinterpret_cast<__gm__ int8_t *>(ptr[i]),
          AscendC::CacheMode::CACHE_MODE_DISABLE);
      auto *from = src + row * width;
      auto *to = buffer + offset;
      const uint32_t bytes = n * width;
      const uint64_t alignment = reinterpret_cast<uint64_t>(from) | reinterpret_cast<uint64_t>(to) | bytes;
      if ((alignment & 31U) == 0)
        copy_gm_to_ubuf(reinterpret_cast<__ubuf__ void *>(to), reinterpret_cast<__gm__ void *>(from),
                        0, 1, bytes / 32, 0, 0);
      else
        LoadSpanBytes(to, from, bytes);
    }
    offset += batch * width;
  }
  set_flag(PIPE_MTE2, PIPE_V, slot ? event_t::EVENT_ID1 : event_t::EVENT_ID0);
}

__aicore__ void Narrow(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling) {
  auto const *plan = reinterpret_cast<__gm__ Plan *>(tiling);
  uint32_t const batch = plan->quantum;
  uint32_t const inputSlot = plan->tile, outputSlot = plan->reserved;
  uint32_t const maxPitch = outputSlot / batch;
  auto *memory = reinterpret_cast<__ubuf__ int8_t *>(get_imm(0));
  auto *index = reinterpret_cast<__ubuf__ int32_t *>(memory + 2 * inputSlot + 2 * outputSlot);
  auto *stride = reinterpret_cast<__ubuf__ int32_t *>(memory + 2 * inputSlot + 3 * outputSlot);
  auto *lanes = reinterpret_cast<__ubuf__ int32_t *>(memory + 2 * inputSlot + 3 * outputSlot + maxPitch);
  auto *p = reinterpret_cast<__ubuf__ Plan *>(reinterpret_cast<__ubuf__ int8_t *>(lanes) + 512);
  auto *ptr = reinterpret_cast<__ubuf__ uint64_t *>(reinterpret_cast<__ubuf__ int8_t *>(p) +
                                                    (sizeof(Plan) + 31) / 32 * 32);
  auto const *header = reinterpret_cast<__gm__ uint64_t *>(inputs);
  LoadSpanBytes(p, reinterpret_cast<__gm__ int8_t *>(tiling), sizeof(Plan));
  LoadSpanBytes(ptr, reinterpret_cast<__gm__ int8_t *>(inputs) + header[0],
            plan->count * 8);
  CONCAT_DMA_FENCE(PIPE_MTE2, PIPE_S, event_t::EVENT_ID0);
  for (uint32_t k = 0; (k < 64); ++k) {
    lanes[k] = 4 * k;
    lanes[64 + k] = 0;
  }
  CONCAT_DMA_FENCE(PIPE_S, PIPE_V, event_t::EVENT_ID0);
  uint64_t const units = (p->rows + batch - 1) / batch;
  uint64_t const core = AscendC::GetBlockIdx();
  uint64_t const begin = units * core / p->cores * batch;
  uint64_t const end = LowerBoundValue(p->rows, units * (core + 1) / p->cores * batch);
  auto *dst = reinterpret_cast<__gm__ int8_t *>(output);
  uint64_t outputPrefix = 0;
  for (uint32_t first = 0; (first < p->count);) {
    uint32_t last = first, bytes = 0;
    while (last < p->count && bytes + p->width[last] <= inputSlot / batch)
      bytes += p->width[last++];
    uint32_t const pitch = (bytes + 255) / 256 * 256;
    if (!bytes) {
      first = last;
      continue;
    }

    set_mask_norm();
    set_vector_mask(0, ~uint64_t(0));

    vadds(index, lanes + 64, int32_t(0), pitch / 256, 1, 1, 8, 0);
    vadds(stride, lanes + 64, int32_t(0), pitch / 256, 1, 1, 8, 0);
    pipe_barrier(PIPE_V);
    uint32_t column = 0;
    for (uint32_t i = first; (i < last); ++i) {
      uint32_t const w = p->width[i] / 4;
      for (uint32_t block = column / 64 * 64; (block < column + w); block += 64) {
        uint32_t const lo = column > block ? column - block : 0;
        uint32_t const hi = LowerBoundValue(64U, column + w - block);
        uint64_t const mask =
            (hi == 64 ? ~uint64_t(0) : (uint64_t(1) << hi) - 1) &
            (~uint64_t(0) << lo);
        set_vector_mask(0, mask);
        vadds(index + block, lanes,
              static_cast<int32_t>(column * 4 * batch + (block - column) * 4),
              1, 1, 1, 8, 8);
        vadds(stride + block, lanes + 64, static_cast<int32_t>(w * 4), 1, 1, 1,
              8, 8);
      }
      column += w;
    }
    set_vector_mask(0, ~uint64_t(0));
    pipe_barrier(PIPE_V);
    for (uint32_t r = 1; (r < batch); ++r) {
      vadd(index + r * pitch / 4, index + (r - 1) * pitch / 4, stride,
           pitch / 256, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
    }
    ReadSlab(memory, p, ptr, first, last, begin,
             LowerBoundValue<uint64_t>(batch, end - begin), 0, batch);

    for (uint64_t row = begin; (row < end); row += batch) {
      uint32_t const slot = ((row - begin) / batch) & 1;
      uint32_t const n = LowerBoundValue<uint64_t>(batch, end - row);
      auto *stage = memory + slot * inputSlot;
      auto *packed =
          reinterpret_cast<__ubuf__ uint32_t *>(memory + 2 * inputSlot + slot * outputSlot);
      auto const release = slot ? event_t::EVENT_ID3 : event_t::EVENT_ID2;
      if (row + batch < end) {
        if (row > begin)
          wait_flag(PIPE_V, PIPE_MTE2,
                    slot ? event_t::EVENT_ID2 : event_t::EVENT_ID3);
        ReadSlab(memory + (slot ^ 1) * inputSlot, p, ptr, first, last, row + batch,
                 LowerBoundValue<uint64_t>(batch, end - row - batch), slot ^ 1, batch);
      }
      wait_flag(PIPE_MTE2, PIPE_V,
                slot ? event_t::EVENT_ID1 : event_t::EVENT_ID0);
      if (row >= begin + 2 * batch)
        wait_flag(PIPE_MTE3, PIPE_V, release);
      uint32_t const address =
          static_cast<uint32_t>(reinterpret_cast<uint64_t>(stage));
      set_vector_mask(0, ~uint64_t(0));
      vgather(packed, reinterpret_cast<__ubuf__ uint32_t *>(index), address, 8,
              n * pitch / 256);
      set_flag(PIPE_V, PIPE_MTE2, release);
      CONCAT_DMA_FENCE(PIPE_V, PIPE_MTE3, event_t::EVENT_ID0);
      StoreSpanBytes(dst + row * p->rowBytes + outputPrefix,
                 reinterpret_cast<__ubuf__ int8_t *>(packed), bytes, n,
                 p->rowBytes - bytes, (pitch - (bytes + 31) / 32 * 32) / 32);
      set_flag(PIPE_MTE3, PIPE_V, release);
    }
    uint32_t const used = LowerBoundValue<uint64_t>(2, (end - begin + batch - 1) / batch);
    for (uint32_t slot = 0; (slot < used); ++slot) {
      auto const release = slot ? event_t::EVENT_ID3 : event_t::EVENT_ID2;
      wait_flag(PIPE_V, PIPE_MTE2, release);
      wait_flag(PIPE_MTE3, PIPE_V, release);
    }
    first = last;
    outputPrefix += bytes;
  }
}

template <bool ShapeTable = false>
__aicore__ void SwingNarrow(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling) {
  constexpr uint32_t R = compact::kReadRows, W = compact::kWriteRows;
  auto const *plan = reinterpret_cast<__gm__ Plan *>(tiling);
  const uint32_t sourceBytes = plan->tile, outputBytes = plan->reserved;
  const uint32_t pitchCapacity = outputBytes / W;
  auto *memory = reinterpret_cast<__ubuf__ int8_t *>(get_imm(0));
  auto *index = reinterpret_cast<__ubuf__ int32_t *>(memory + sourceBytes + 2 * outputBytes);
  auto *stride = index + W * pitchCapacity / 4 + 32;
  auto *lanes = stride + W * pitchCapacity / 4;
  auto *widths = reinterpret_cast<__ubuf__ uint64_t *>(lanes + 128);
  auto *ptr = reinterpret_cast<__ubuf__ uint64_t *>(reinterpret_cast<__ubuf__ int8_t *>(widths) +
                                                   (sizeof(Plan) + 31) / 32 * 32);
  auto const *header = reinterpret_cast<__gm__ uint64_t *>(inputs);
  auto const *job = &reinterpret_cast<__gm__ const SwingPlan *>(tiling)->jobs[AscendC::GetBlockIdx()];
  const uint64_t begin = job->begin, end = job->end;
  const uint32_t first = job->first, last = job->last, prefix = job->prefix;
  const uint64_t rowBytes = plan->rowBytes;
  const uint32_t alignedFirst = first / 4 * 4;

  LoadSpanBytes(widths + alignedFirst, plan->width + alignedFirst, (last - alignedFirst) * 8);
  LoadSpanBytes(ptr + alignedFirst, reinterpret_cast<__gm__ int8_t *>(inputs) + header[0] + alignedFirst * 8,
                (last - alignedFirst) * 8);
  CONCAT_DMA_FENCE(PIPE_MTE2, PIPE_S, event_t::EVENT_ID0);

  uint32_t bytes = 0;
  for (uint32_t i = first; i < last; ++i) bytes += widths[i];
  const uint32_t pitch = (bytes + 255) / 256 * 256;
  auto *shape16 = reinterpret_cast<__ubuf__ int16_t *>(memory + sourceBytes);
  auto *shape8 = reinterpret_cast<__ubuf__ int8_t *>(memory + sourceBytes + outputBytes);
  if constexpr (ShapeTable) {
    auto *shaped = reinterpret_cast<__gm__ ShapeTablePlan *>(tiling);
    LoadSpanBytes(shape16, shaped->firstRow + prefix/4, pitch/2);
    LoadSpanBytes(shape8, shaped->stepMinusOne + prefix/4, pitch/4);
    set_flag(PIPE_MTE2, PIPE_V, event_t::EVENT_ID1);
  }
  ReadSelectedSlab(memory, widths, ptr, first, last, begin,
           LowerBoundValue<uint64_t>(R, end - begin), 0, R);
  if constexpr (ShapeTable) {
    wait_flag(PIPE_MTE2, PIPE_V, event_t::EVENT_ID1);
    set_mask_norm(); set_vector_mask(0, ~uint64_t(0));
    const uint32_t repeats=pitch/256;
    vconv_s162f32(reinterpret_cast<__ubuf__ float *>(index), shape16, repeats, 1, 1, 8, 4);
    pipe_barrier(PIPE_V);
    vconv_f322s32z(index, reinterpret_cast<__ubuf__ float *>(index), repeats, 1, 1, 8, 8);
    vconv_s82f16(reinterpret_cast<__ubuf__ half *>(shape16), shape8, repeats, 1, 1, 4, 2);
    pipe_barrier(PIPE_V);
    vconv_f162s32z(stride, reinterpret_cast<__ubuf__ half *>(shape16), repeats, 1, 1, 8, 4);
    vadd(index,index,index,repeats,1,1,1,8,8,8);
    pipe_barrier(PIPE_V);
    vadds(stride,stride,int32_t(1),repeats,1,1,8,8);
    vadd(index,index,index,repeats,1,1,1,8,8,8);
    pipe_barrier(PIPE_V);
    vadd(stride,stride,stride,repeats,1,1,1,8,8,8);
    pipe_barrier(PIPE_V);
    vadd(stride,stride,stride,repeats,1,1,1,8,8,8);
    pipe_barrier(PIPE_V);
  } else {
  for (uint32_t i = 0; i < 64; ++i) { lanes[i] = i * 4; lanes[i + 64] = 0; }
  CONCAT_DMA_FENCE(PIPE_S, PIPE_V, event_t::EVENT_ID0);
  set_mask_norm(); set_vector_mask(0, ~uint64_t(0));
  vadds(index, lanes + 64, int32_t(0), pitch / 256, 1, 1, 8, 0);
  vadds(stride, lanes + 64, int32_t(0), pitch / 256, 1, 1, 8, 0);
  pipe_barrier(PIPE_V);
  uint32_t column = 0;
  for (uint32_t i = first; i < last; ++i) {
    const uint32_t width = widths[i] / 4;
    for (uint32_t block = column / 64 * 64; block < column + width; block += 64) {
      const uint32_t lo = column > block ? column - block : 0;
      const uint32_t hi = LowerBoundValue(64U, column + width - block);
      const uint64_t mask = (hi == 64 ? ~uint64_t(0) : (uint64_t(1) << hi) - 1) &
                             (~uint64_t(0) << lo);
      set_vector_mask(0, mask);
      vadds(index + block, lanes, static_cast<int32_t>(column * 4 * R + (block - column) * 4),
            1, 1, 1, 8, 8);
      vadds(stride + block, lanes + 64, static_cast<int32_t>(width * 4), 1, 1, 1, 8, 8);
    }
    column += width;
  }
  set_vector_mask(0, ~uint64_t(0)); pipe_barrier(PIPE_V);
  }
  for (uint32_t row = 1; row < W; ++row) {
    vadd(index + row * pitch / 4, index + (row - 1) * pitch / 4, stride,
         pitch / 256, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
  }

  for (uint32_t scale = 1; scale < W; scale *= 2) {
    vadd(stride, stride, stride, pitch / 256, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
  }
  for (uint32_t r = 1; r < W; ++r)
    vadds(stride + r * pitch / 4, stride, int32_t(0), pitch / 256, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  uint32_t cursorRow = 0;
  auto *dst = reinterpret_cast<__gm__ int8_t *>(output);
  uint32_t issued = 0;
  for (uint64_t row = begin; row < end; row += R) {
    const uint32_t n = LowerBoundValue<uint64_t>(R, end - row);
    wait_flag(PIPE_MTE2, PIPE_V, event_t::EVENT_ID0);
    const bool reverse = ((row - begin) / R) & 1;
    const uint32_t lastRow = (n - 1) / W * W;

    while (cursorRow > lastRow) {
      vsub(index, index, stride, W * pitch / 256, 1, 1, 1, 8, 8, 8);
      pipe_barrier(PIPE_V);
      cursorRow -= W;
    }
    const uint32_t tiles = (n + W - 1) / W;
    for (uint32_t t = 0; t < tiles; ++t) {
      const uint32_t r = cursorRow;
      const uint32_t slot = issued & 1, count = LowerBoundValue(W, n - r);
      const auto event = slot ? event_t::EVENT_ID3 : event_t::EVENT_ID2;
      if (issued >= 2) wait_flag(PIPE_MTE3, PIPE_V, event);
      auto *packed = reinterpret_cast<__ubuf__ uint32_t *>(memory + sourceBytes + slot * outputBytes);
      vgather(packed, reinterpret_cast<__ubuf__ uint32_t *>(index), 0,
              8, count * pitch / 256);
      CONCAT_DMA_FENCE(PIPE_V, PIPE_MTE3, event_t::EVENT_ID0);
      StoreSpanBytes(dst + (row + r) * rowBytes + prefix,
                     reinterpret_cast<__ubuf__ int8_t *>(packed), bytes, count,
                     rowBytes - bytes, (pitch - (bytes + 31) / 32 * 32) / 32);
      set_flag(PIPE_MTE3, PIPE_V, event);
      ++issued;
      if (t + 1 < tiles) {
        if (reverse) {
          vsub(index, index, stride, W * pitch / 256, 1, 1, 1, 8, 8, 8);
          cursorRow -= W;
        } else {
          vadd(index, index, stride, W * pitch / 256, 1, 1, 1, 8, 8, 8);
          cursorRow += W;
        }
        pipe_barrier(PIPE_V);
      }
    }
    if (row + R < end) {
      CONCAT_DMA_FENCE(PIPE_V, PIPE_MTE2, event_t::EVENT_ID0);
      ReadSelectedSlab(memory, widths, ptr, first, last, row + R,
               LowerBoundValue<uint64_t>(R, end - row - R), 0, R);
    }
  }
  for (uint32_t slot = 0; slot < LowerBoundValue(2U, issued); ++slot)
    wait_flag(PIPE_MTE3, PIPE_V, slot ? event_t::EVENT_ID3 : event_t::EVENT_ID2);
}

// 两个 UB 缓冲交替搬运，复用前等待写回完成。
class Copier {
public:
  __aicore__ Copier(GM_ADDR output, uint32_t tile)
      : out_(reinterpret_cast<__gm__ int8_t *>(output)), tile_(tile) {}
  __aicore__ void Copy(__gm__ int8_t *input, uint64_t dst, uint32_t bytes,
                       uint16_t rows = 1, uint32_t gap = 0) {
    auto const event = slot_ ? event_t::EVENT_ID1 : event_t::EVENT_ID0;
    if (used_[slot_])
      wait_flag(PIPE_MTE3, PIPE_MTE2, event);
    auto *ub = reinterpret_cast<__ubuf__ int8_t *>(get_imm(0)) + slot_ * tile_;
    LoadSpanBytes(ub, input, bytes, rows);
    CONCAT_DMA_FENCE(PIPE_MTE2, PIPE_MTE3, event);
    StoreSpanBytes(out_ + dst, ub, bytes, rows, gap);
    set_flag(PIPE_MTE3, PIPE_MTE2, event);
    used_[slot_] = true;
    slot_ ^= 1;
  }
  __aicore__ void Finish() {
    if (used_[0])
      wait_flag(PIPE_MTE3, PIPE_MTE2, event_t::EVENT_ID0);
    if (used_[1])
      wait_flag(PIPE_MTE3, PIPE_MTE2, event_t::EVENT_ID1);
  }

private:
  __gm__ int8_t *out_;
  uint32_t tile_, slot_ = 0;
  bool used_[2] = {false, false};
};

__aicore__ void Execute(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling,
                        bool byRows) {
  auto *memory = reinterpret_cast<__ubuf__ int8_t *>(get_imm(0));
  auto const *gmPlan = reinterpret_cast<__gm__ Plan *>(tiling);
  auto *p = reinterpret_cast<__ubuf__ Plan *>(memory + 2 * gmPlan->tile);
  auto *pointers =
      reinterpret_cast<__ubuf__ uint64_t *>(memory + 2 * gmPlan->tile + 2112);
  auto const *header = reinterpret_cast<__gm__ uint64_t *>(inputs);
  LoadSpanBytes(p, reinterpret_cast<__gm__ int8_t *>(tiling),
            48 + gmPlan->count * 8);
  LoadSpanBytes(pointers, reinterpret_cast<__gm__ int8_t *>(inputs) + header[0],
            gmPlan->count * 8);
  CONCAT_DMA_FENCE(PIPE_MTE2, PIPE_S, event_t::EVENT_ID0);
  if (!p->total)
    return;
  Copier copy(output, p->tile);
  uint64_t const quantum = byRows ? p->quantum : 32;
  uint64_t const length = byRows ? p->rows : p->total;
  uint64_t const units = (length + quantum - 1) / quantum;
  uint64_t const core = AscendC::GetBlockIdx(), cores = p->cores;
  uint64_t const begin = (units * core / cores) * quantum;
  uint64_t const end = LowerBoundValue(length, (units * (core + 1) / cores) * quantum);
  if (byRows) {
    uint64_t prefix = 0;
    for (uint32_t i = 0; (i < p->count); ++i) {
      uint32_t const width = p->width[i];
      if (!width)
        continue;
      uint32_t const batch = LowerBoundValue(255U, p->tile / ((width + 31) / 32 * 32));
      auto *src = reinterpret_cast<__gm__ int8_t *>(pointers[i]);
      for (uint64_t row = begin; (row < end); row += batch)
        copy.Copy(src + row * width, row * p->rowBytes + prefix, width,
                  LowerBoundValue<uint64_t>(batch, end - row), p->rowBytes - width);
      prefix += width;
    }
  } else {

    for (uint64_t row = begin / p->rowBytes; (row * p->rowBytes < end); ++row) {
      uint64_t prefix = row * p->rowBytes;
      for (uint32_t i = 0; (i < p->count); ++i) {
        uint64_t const width = p->width[i];
        uint64_t at = begin > prefix ? begin : prefix;
        uint64_t const stop = LowerBoundValue(end, prefix + width);
        auto *src = reinterpret_cast<__gm__ int8_t *>(pointers[i]);
        while (at < stop) {
          uint32_t const bytes = LowerBoundValue<uint64_t>(p->tile, stop - at);
          copy.Copy(src + row * width + at - prefix, at, bytes);
          at += bytes;
        }
        prefix += width;
      }
    }
  }
  copy.Finish();
}

__aicore__ inline void Small(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling) {
  using Task = SmallMove;
  auto const *t =
      reinterpret_cast<__gm__ Task *>(tiling) + AscendC::GetBlockIdx();
  auto *header = reinterpret_cast<__gm__ uint64_t *>(inputs);
  dc_preload(header + header[0] / 8, 0);
  dc_preload(reinterpret_cast<__gm__ uint64_t *>(tiling) +
                 AscendC::GetBlockIdx() * sizeof(Task) / 8,
             0);
  if (!t->bytes)
    return;
  auto *src =
      reinterpret_cast<__gm__ int8_t *>(header[header[0] / 8 + t->input]);
  Copier copy(output, 4096);
  copy.Copy(src + t->src, t->dst, t->bytes, t->rows, t->gap);
  copy.Finish();
}

__aicore__ void Scheduled(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling) {
  auto const *s = reinterpret_cast<__gm__ Schedule *>(tiling);
  uint32_t const core = AscendC::GetBlockIdx();
  uint32_t const begin = s->begin[core], count = s->begin[core + 1] - begin;
  if (!count)
    return;
  auto *memory = reinterpret_cast<__ubuf__ int8_t *>(get_imm(0));
  auto *moves = reinterpret_cast<__ubuf__ Move *>(memory + 180224);
  auto *ptr = reinterpret_cast<__ubuf__ uint64_t *>(memory + 180224 + 4096);
  auto const *header = reinterpret_cast<__gm__ uint64_t *>(inputs);
  LoadSpanBytes(moves,
            reinterpret_cast<__gm__ int8_t *>(tiling) + sizeof(Schedule) +
                begin * sizeof(Move),
            count * sizeof(Move));
  LoadSpanBytes(ptr, reinterpret_cast<__gm__ int8_t *>(inputs) + header[0],
            s->inputs * 8);
  CONCAT_DMA_FENCE(PIPE_MTE2, PIPE_S, event_t::EVENT_ID0);
  for (uint32_t i = 0; (i < s->inputs); ++i)
    ptr[i] = reinterpret_cast<uint64_t>(
        AscendC::L2CacheAlter<int8_t, AscendC::CacheRwMode::READ>(
            reinterpret_cast<__gm__ int8_t *>(ptr[i]),
            AscendC::CacheMode::CACHE_MODE_DISABLE));
  uint32_t const slots = s->reserved;
  for (uint32_t at = 0; (at < count); at += slots) {
    uint32_t const n = LowerBoundValue(slots, count - at);
    for (uint32_t slot = 0; (slot < n); ++slot) {
      auto const event = static_cast<event_t>(slot);
      if (at)
        wait_flag(PIPE_MTE3, PIPE_MTE2, event);
      auto const &m = moves[at + slot];
      LoadSpanBytes(memory + m.reserved,
                reinterpret_cast<__gm__ int8_t *>(ptr[m.input]) + m.src,
                m.bytes, m.rows);
      set_flag(PIPE_MTE2, PIPE_MTE3, event);
    }
    for (uint32_t slot = 0; (slot < n); ++slot) {
      auto const event = static_cast<event_t>(slot);
      wait_flag(PIPE_MTE2, PIPE_MTE3, event);
      auto const &m = moves[at + slot];
      StoreSpanBytes(reinterpret_cast<__gm__ int8_t *>(output) + m.dst,
                 memory + m.reserved, m.bytes, m.rows, m.gap);
      set_flag(PIPE_MTE3, PIPE_MTE2, event);
    }
  }
  for (uint32_t slot = 0; (slot < LowerBoundValue(slots, count)); ++slot)
    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(slot));
}
template <class T>
__aicore__ inline void CacheRecords(GM_ADDR inputs, const __gm__ T *records,
                                    uint32_t count, uint32_t inputCount,
                                    __ubuf__ T *local,
                                    __ubuf__ uint64_t *pointers) {
  auto const *header = reinterpret_cast<__gm__ uint64_t *>(inputs);
  LoadSpanBytes(local, records, count * sizeof(T));
  LoadSpanBytes(pointers, reinterpret_cast<__gm__ int8_t *>(inputs) + header[0],
            inputCount * 8);
  CONCAT_DMA_FENCE(PIPE_MTE2, PIPE_S, event_t::EVENT_ID0);
}

__aicore__ void Planned(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling) {
  auto const *schedule = reinterpret_cast<__gm__ Schedule *>(tiling);
  uint32_t const core = AscendC::GetBlockIdx();
  uint32_t const begin = schedule->begin[core];
  uint32_t const count = schedule->begin[core + 1] - begin;
  if (!count)
    return;
  auto *memory = reinterpret_cast<__ubuf__ int8_t *>(get_imm(0));
  auto const *tasks =
      reinterpret_cast<__gm__ Transfer *>(
          reinterpret_cast<__gm__ int8_t *>(tiling) + sizeof(Schedule)) +
      begin;
  auto const *list = reinterpret_cast<__gm__ uint64_t *>(inputs);
  auto const *pointers = list + list[0] / 8;
  dc_preload((const_cast<__gm__ uint64_t *>(pointers)), 0);
  dc_preload(
      reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ Transfer *>(tasks)),
      0);
  auto *destination = reinterpret_cast<__gm__ int8_t *>(output);
  for (uint32_t i = 0; (i < count); ++i) {
    auto const &t = tasks[i];
    auto *source = AscendC::L2CacheAlter<int8_t, AscendC::CacheRwMode::READ>(
        reinterpret_cast<__gm__ int8_t *>(pointers[t.input]),
        AscendC::CacheMode::CACHE_MODE_DISABLE);
    copy_gm_to_ubuf_align_b8(memory + t.ubOffset, source + t.src, 0, 1, t.bytes,
                             0, 0, 0, 0);
  }
  CONCAT_DMA_FENCE(PIPE_MTE2, PIPE_MTE3, event_t::EVENT_ID0);
  for (uint32_t i = 0; (i < count); ++i) {
    auto const &t = tasks[i];
    copy_ubuf_to_gm_align_b8(destination + t.dst, memory + t.ubOffset, 0, 1,
                             t.bytes, 0, 0, 0, 0);
  }
  CONCAT_DMA_FENCE(PIPE_MTE3, PIPE_MTE2, event_t::EVENT_ID0);
}

__aicore__ void Streamed(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling) {
  auto const *s = reinterpret_cast<__gm__ Schedule *>(tiling);
  uint32_t const core = AscendC::GetBlockIdx(), begin = s->begin[core];
  uint32_t const count = s->begin[core + 1] - begin;
  if (!count)
    return;
  auto *records =
      reinterpret_cast<__gm__ StreamSpan *>(
          reinterpret_cast<__gm__ int8_t *>(tiling) + sizeof(Schedule)) +
      begin;
  auto *memory = reinterpret_cast<__ubuf__ int8_t *>(get_imm(0));
  auto *tasks =
      reinterpret_cast<__ubuf__ StreamSpan *>(memory + 2 * kStreamArena);
  auto *pointers =
      reinterpret_cast<__ubuf__ uint64_t *>(memory + 2 * kStreamArena + 2048);
  CacheRecords(inputs, records, count, s->inputs, tasks, pointers);
  auto *dst = reinterpret_cast<__gm__ int8_t *>(output);
  uint32_t at = 0, waves = 0;
  while (at < count) {
    uint32_t stop = at, footprint = 0;
    while (stop < count) {
      uint32_t const bytes = (tasks[stop].bytes + 31) / 32 * 32;
      if (footprint + bytes > kStreamArena)
        break;
      footprint += bytes;
      ++stop;
    }
    uint32_t const slot = waves & 1;
    auto const event = static_cast<event_t>(slot);
    if (waves >= 2)
      wait_flag(PIPE_MTE3, PIPE_MTE2, event);
    auto *buffer = memory + slot * kStreamArena;
    uint32_t offset = 0;
    for (uint32_t i = at; (i < stop); ++i) {
      auto const &t = tasks[i];
      auto *src = AscendC::L2CacheAlter<int8_t, AscendC::CacheRwMode::READ>(
          reinterpret_cast<__gm__ int8_t *>(pointers[t.input]),
          AscendC::CacheMode::CACHE_MODE_DISABLE);
      LoadSpanBytes(buffer + offset, src + t.src, t.bytes);
      offset += (t.bytes + 31) / 32 * 32;
    }
    CONCAT_DMA_FENCE(PIPE_MTE2, PIPE_MTE3, event);
    offset = 0;
    for (uint32_t i = at; (i < stop); ++i) {
      auto const &t = tasks[i];
      StoreSpanBytes(dst + t.dst, buffer + offset, t.bytes);
      offset += (t.bytes + 31) / 32 * 32;
    }
    set_flag(PIPE_MTE3, PIPE_MTE2, event);
    at = stop;
    ++waves;
  }
  for (uint32_t slot = 0; (slot < LowerBoundValue(2U, waves)); ++slot)
    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(slot));
}

__aicore__ void Strided(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling) {
  auto const *schedule = reinterpret_cast<__gm__ Schedule *>(tiling);
  uint32_t const core = AscendC::GetBlockIdx();
  uint32_t const begin = schedule->begin[core];
  uint32_t const count = schedule->begin[core + 1] - begin;
  if (!count)
    return;
  auto *memory = reinterpret_cast<__ubuf__ int8_t *>(get_imm(0));
  auto const *tasks =
      reinterpret_cast<__gm__ Transfer *>(
          reinterpret_cast<__gm__ int8_t *>(tiling) + sizeof(Schedule)) +
      begin;
  auto const *list = reinterpret_cast<__gm__ uint64_t *>(inputs);
  auto const *pointers = list + list[0] / 8;
  dc_preload((const_cast<__gm__ uint64_t *>(pointers)), 0);
  dc_preload(
      reinterpret_cast<__gm__ uint64_t *>(const_cast<__gm__ Transfer *>(tasks)),
      0);
  auto *destination = reinterpret_cast<__gm__ int8_t *>(output);
  constexpr uint32_t slots = 3;
  uint32_t const tile = schedule->tile;
  for (uint32_t at = 0; (at < count); at += slots) {
    uint32_t const n = LowerBoundValue(slots, count - at);
    for (uint32_t slot = 0; (slot < n); ++slot) {
      auto const event = static_cast<event_t>(slot);
      if (at)
        wait_flag(PIPE_MTE3, PIPE_MTE2, event);
      auto const &t = tasks[at + slot];
      auto *source = AscendC::L2CacheAlter<int8_t, AscendC::CacheRwMode::READ>(
          reinterpret_cast<__gm__ int8_t *>(pointers[t.input]),
          AscendC::CacheMode::CACHE_MODE_DISABLE);
      LoadSpanBytes(memory + slot * tile, source + t.src, t.bytes, t.rows);
      set_flag(PIPE_MTE2, PIPE_MTE3, event);
    }
    for (uint32_t slot = 0; (slot < n); ++slot) {
      auto const event = static_cast<event_t>(slot);
      wait_flag(PIPE_MTE2, PIPE_MTE3, event);
      auto const &t = tasks[at + slot];
      StoreSpanBytes(destination + t.dst, memory + slot * tile, t.bytes, t.rows,
                 t.gap);
      set_flag(PIPE_MTE3, PIPE_MTE2, event);
    }
  }
  for (uint32_t slot = 0; (slot < LowerBoundValue(slots, count)); ++slot)
    wait_flag(PIPE_MTE3, PIPE_MTE2, static_cast<event_t>(slot));
}
// 先连续读入各输入，再按 Host 生成的索引重排为完整输出行。
__aicore__ void AffineGatherRows(GM_ADDR inputs, GM_ADDR output, GM_ADDR tiling) {
  using namespace AscendC;
  auto const *p = reinterpret_cast<__gm__ AffineRowPlan *>(tiling);
  uint32_t const tileRows = p->tileRows, pitch = p->pitch, lanes = pitch / 2;
  auto const *list = reinterpret_cast<__gm__ uint64_t *>(inputs);
  auto const *pointers = list + list[0] / 8;
  TPipe pipe;
  TBuf<QuePosition::VECCALC> staging, result, indexStorage;
  pipe.InitBuffer(staging, p->stageBytes);
  pipe.InitBuffer(result, tileRows * pitch);
  pipe.InitBuffer(indexStorage, tileRows * lanes * sizeof(int32_t));
  auto indices = indexStorage.Get<uint32_t>();
  GlobalTensor<uint32_t> map;
  map.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tiling + sizeof(AffineRowPlan)));
  DataCopy(indices, map, tileRows * lanes);
  GlobalTensor<uint16_t> dst;
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(output));
  for (uint32_t first = GetBlockIdx() * tileRows; first < p->rows;
       first += p->cores * tileRows) {
    uint16_t const n = LowerBoundValue(tileRows, p->rows - first);
    for (uint32_t i = 0; i < p->count; ++i) {
      uint32_t const width = p->parts[i].width;
      if (!width)
        continue;
      GlobalTensor<uint16_t> src;
      src.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(pointers[i]));
      DataCopyExtParams copy{1, n * width, 0, 0, 0};
      DataCopyPad(staging.Get<uint16_t>()[p->parts[i].offset / 2],
                  src[static_cast<uint64_t>(first) * width / 2], copy,
                  DataCopyPadExtParams<uint16_t>{false, 0, 0, 0});
    }
    CompleteVectorTransfer<HardEvent::MTE2_V>();
    Gather(result.Get<uint16_t>(), staging.Get<uint16_t>(),
           indices.ReinterpretCast<uint32_t>(), uint32_t(0), n * lanes);
    CompleteVectorTransfer<HardEvent::V_MTE3>();
    DataCopyExtParams copy{n, p->rowBytes, 0, 0, 0};
    DataCopyPad(dst[static_cast<uint64_t>(first) * p->rowBytes / 2],
                result.Get<uint16_t>(), copy);
    CompleteVectorTransfer<HardEvent::MTE3_MTE2>();
  }
}
// 按对齐周期分组搬运行主体，并单独补齐行首未对齐部分。
__aicore__ inline void PhaseRows(GM_ADDR inputs, GM_ADDR output,
                                 GM_ADDR tiling) {
  using namespace AscendC;
  auto const *p = reinterpret_cast<__gm__ compact::Plan *>(tiling);
  auto *list = reinterpret_cast<__gm__ uint64_t *>(inputs);
  auto *ptr = list + list[0] / 8;
  TPipe pipe;
  TBuf<QuePosition::VECCALC> source, heads;
  pipe.InitBuffer(source, compact::kPhaseArena);
  pipe.InitBuffer(heads, 2048);
  AscendC::GlobalTensor<uint32_t> dst;
  dst.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(output));
  uint64_t const units = (p->rows + p->quantum - 1) / p->quantum;
  uint64_t const first = units * GetBlockIdx() / p->cores * p->quantum;
  uint64_t const limit = units * (GetBlockIdx() + 1) / p->cores * p->quantum;
  uint64_t const end = p->rows < limit ? p->rows : limit;
  uint64_t prefix = 0;
  for (uint32_t i = 0; (i < p->count); ++i) {
    uint32_t const width = p->width[i];
    if (!width)
      continue;
    uint32_t period = 1;
    while (period * width % 32)
      period *= 2;
    uint32_t const capacity = compact::kPhaseArena / width;
    uint32_t const batch = (capacity < 64 ? capacity : 64) / period * period;
    AscendC::GlobalTensor<uint32_t> src;
    src.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(*(ptr + i)));
    src.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    for (uint64_t row = first; (row < end); row += batch) {
      uint32_t const n = end - row < batch ? end - row : batch;
      AscendC::DataCopyExtParams read{1, n * width, 0, 0, 0};
      AscendC::DataCopyPad(source.Get<uint32_t>(), src[row * width / 4], read,
                  AscendC::DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
      AscendC::DataCopyExtParams edges{static_cast<uint16_t>(n), 32, width - 32, 0, 0};
      AscendC::DataCopyPad(heads.Get<uint32_t>(), src[row * width / 4], edges,
                  AscendC::DataCopyPadExtParams<uint32_t>{false, 0, 0, 0});
      CompleteVectorTransfer<HardEvent::MTE2_MTE3>();
      for (uint32_t phase = 0; (phase < period && phase < n); ++phase) {
        uint16_t const rows = (n - 1 - phase) / period + 1;
        uint32_t const lead = (32 - phase * width % 32) % 32;
        uint32_t const bytes = width - lead;
        uint64_t const start = (row + phase) * p->rowBytes + prefix;
        AscendC::DataCopyExtParams body{
            rows, bytes, (period * width - (bytes + 31) / 32 * 32) / 32,
            static_cast<uint32_t>(period * p->rowBytes - bytes), 0};
        AscendC::DataCopyPad(dst[(start + lead) / 4],
                    source.Get<uint32_t>()[(phase * width + lead) / 4], body);
        if (lead) {
          AscendC::DataCopyExtParams head{
              rows, lead, period - 1,
              static_cast<uint32_t>(period * p->rowBytes - lead), 0};
          AscendC::DataCopyPad(dst[start / 4], heads.Get<uint32_t>()[phase * 8], head);
        }
      }
      CompleteVectorTransfer<HardEvent::MTE3_MTE2>();
    }
    prefix += width;
  }
}
}

extern "C" __global__ __aicore__ void
concat(GM_ADDR const inputListAddress, GM_ADDR const resultAddress, GM_ADDR const workspaceAddress, GM_ADDR const planAddress) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  if (TILING_KEY_IS(12))
    SwingNarrow<true>(inputListAddress, resultAddress, planAddress);
  if (TILING_KEY_IS(11))
    SwingNarrow(inputListAddress, resultAddress, planAddress);
  if (TILING_KEY_IS(10))
    PhaseRows(inputListAddress, resultAddress, planAddress);
  if (TILING_KEY_IS(9))
    AffineGatherRows(inputListAddress, resultAddress, planAddress);
  if (TILING_KEY_IS(1))
    Execute(inputListAddress, resultAddress, planAddress, false);
  if (TILING_KEY_IS(2))
    Execute(inputListAddress, resultAddress, planAddress, true);
  if (TILING_KEY_IS(3))
    Narrow(inputListAddress, resultAddress, planAddress);
  if (TILING_KEY_IS(4))
    Small(inputListAddress, resultAddress, planAddress);
  if (TILING_KEY_IS(5))
    Scheduled(inputListAddress, resultAddress, planAddress);
  if (TILING_KEY_IS(6))
    Streamed(inputListAddress, resultAddress, planAddress);
  if (TILING_KEY_IS(7))
    Planned(inputListAddress, resultAddress, planAddress);
  if (TILING_KEY_IS(8))
    Strided(inputListAddress, resultAddress, planAddress);
}

#undef CONCAT_DMA_FENCE
