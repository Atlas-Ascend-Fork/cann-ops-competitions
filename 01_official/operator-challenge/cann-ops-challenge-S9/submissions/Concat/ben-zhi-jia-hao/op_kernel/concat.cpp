#include "kernel_operator.h"
using namespace AscendC;

// A DMA-only implementation. No vector gathers or intermediate permutations.
class ConcatCopyEngine {
 public:
  template <class Tiling>
  __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const Tiling &data) {
    count_ = data.inputCount; axis_ = data.axis; rank_ = data.rank;
    outer_ = data.outer; innerBytes_ = data.innerBytes;
    rowBytes_ = data.rowBytes; totalBytes_ = data.totalBytes;
    for (uint32_t j = 0; j < count_ && j < 96; ++j) cached_[j] = data.inputRowBytes[j];
    descriptor_ = reinterpret_cast<__gm__ uint64_t *>(inputs);
    pointers_ = descriptor_ + descriptor_[0] / 8;
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(output));
    pipe_.InitBuffer(queue_, 2, kTile);
  }

  __aicore__ inline void Flat() {
    if (!totalBytes_) return;
    const uint64_t blocks = (totalBytes_ + 31) / 32;
    const uint64_t begin = blocks * GetBlockIdx() / GetBlockNum() * 32;
    const uint64_t rawEnd = blocks * (GetBlockIdx() + 1) / GetBlockNum() * 32;
    const uint64_t end = rawEnd < totalBytes_ ? rawEnd : totalBytes_;
    uint64_t prefix = 0;
    for (uint32_t j = 0; j < count_; ++j) {
      const uint64_t bytes = Segment(j);
      const uint64_t lo = begin > prefix ? begin : prefix;
      const uint64_t hi = end < prefix + bytes ? end : prefix + bytes;
      if (lo < hi) {
        GlobalTensor<DTYPE_X> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(pointers_[j]));
        Copy(source, lo - prefix, lo, hi - lo);
      }
      prefix += bytes;
      if (prefix >= end) break;
    }
  }

  __aicore__ inline void SmallRows() {
    if (!rowBytes_ || !outer_) return;
    const uint32_t period = Period(rowBytes_);
    const uint64_t groups = (outer_ + period - 1) / period;
    const uint64_t begin = groups * GetBlockIdx() / GetBlockNum() * period;
    const uint64_t rawEnd = groups * (GetBlockIdx() + 1) / GetBlockNum() * period;
    const uint64_t end = rawEnd < outer_ ? rawEnd : outer_;
    uint64_t prefix = 0;
    for (uint32_t j = 0; j < count_; ++j) {
      const uint32_t bytes = Segment(j);
      if (bytes && begin < end) {
        GlobalTensor<DTYPE_X> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(pointers_[j]));
        const uint32_t capacity = kTile / Align(bytes);
        const uint32_t limit = capacity < 4095 ? capacity : 4095;
        for (uint64_t row = begin; row < end; row += limit) {
          const uint16_t rows = end - row < limit ? end - row : limit;
          auto local = queue_.AllocTensor<DTYPE_X>();
          DataCopyExtParams in{rows, bytes, 0, 0, 0};
          DataCopyPadExtParams<DTYPE_X> pad{false, 0, 0, 0};
          DataCopyPad(local, source[row * bytes / sizeof(DTYPE_X)], in, pad);
          queue_.EnQue(local);
          local = queue_.DeQue<DTYPE_X>();
          DataCopyExtParams out{rows, bytes, 0, static_cast<uint32_t>(rowBytes_ - bytes), 0};
          DataCopyPad(output_[(row * rowBytes_ + prefix) / sizeof(DTYPE_X)], local, out);
          queue_.FreeTensor(local);
        }
      }
      prefix += bytes;
    }
  }

  __aicore__ inline void PhaseViews() {
    const uint64_t core = GetBlockIdx(), cores = GetBlockNum();
    uint64_t prefix = 0, taskBase = 0;
    for (uint32_t j = 0; j < count_; ++j) {
      const uint32_t bytes = Segment(j);
      if (bytes) {
        const uint32_t period = Period(bytes);
        uint32_t rowsPerTask = kTile / bytes;
        if (rowsPerTask >= period) rowsPerTask = rowsPerTask / period * period;
        const uint64_t tasks = (outer_ + rowsPerTask - 1) / rowsPerTask;
        const uint64_t first = (core + cores - taskBase) % cores;
        GlobalTensor<DTYPE_X> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(pointers_[j]));
        for (uint64_t task = first; task < tasks; task += cores) {
          const uint64_t begin = task * rowsPerTask;
          const uint32_t rows = outer_ - begin < rowsPerTask ? outer_ - begin : rowsPerTask;
          const uint32_t phases = rows < period ? rows : period;
          for (uint32_t phase = 0; phase < phases; ++phase) {
            const uint16_t count = (rows - 1 - phase) / period + 1;
            const uint32_t span = (count - 1) * period * bytes + bytes;
            // Repeated overlapping reads stay in L2. A single wide read can stream.
            const bool streaming = (period == 1 || count == 1) && span >= 8192;
            source.SetL2CacheHint(streaming ? CacheMode::CACHE_MODE_DISABLE : CacheMode::CACHE_MODE_NORMAL);
            auto local = queue_.AllocTensor<DTYPE_X>();
            DataCopyExtParams in{1, span, 0, 0, 0};
            DataCopyPadExtParams<DTYPE_X> pad{false, 0, 0, 0};
            DataCopyPad(local, source[(begin + phase) * bytes / sizeof(DTYPE_X)], in, pad);
            queue_.EnQue(local);
            local = queue_.DeQue<DTYPE_X>();
            DataCopyExtParams out{count, bytes, (period * bytes - Align(bytes)) / 32,
                static_cast<uint32_t>(period * rowBytes_ - bytes), 0};
            DataCopyPad(output_[((begin + phase) * rowBytes_ + prefix) / sizeof(DTYPE_X)], local, out);
            queue_.FreeTensor(local);
          }
        }
        taskBase = (taskBase + tasks) % cores;
      }
      prefix += bytes;
    }
  }

  __aicore__ inline void General() {
    if (!totalBytes_ || !rowBytes_) return;
    const uint64_t blocks = (totalBytes_ + 31) / 32;
    const uint64_t begin = blocks * GetBlockIdx() / GetBlockNum() * 32;
    const uint64_t rawEnd = blocks * (GetBlockIdx() + 1) / GetBlockNum() * 32;
    const uint64_t end = rawEnd < totalBytes_ ? rawEnd : totalBytes_;
    if (begin >= end) return;
    uint64_t prefix = 0;
    for (uint32_t j = 0; j < count_; ++j) {
      const uint64_t bytes = Segment(j);
      if (bytes) {
        GlobalTensor<DTYPE_X> source;
        source.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(pointers_[j]));
        for (uint64_t row = begin / rowBytes_; row <= (end - 1) / rowBytes_; ++row) {
          const uint64_t start = row * rowBytes_ + prefix;
          const uint64_t lo = begin > start ? begin : start;
          const uint64_t hi = end < start + bytes ? end : start + bytes;
          if (lo < hi) Copy(source, row * bytes + lo - start, lo, hi - lo);
        }
      }
      prefix += bytes;
    }
  }

 private:
  __aicore__ inline uint32_t Align(uint32_t bytes) { return (bytes + 31) / 32 * 32; }
  __aicore__ inline uint32_t Period(uint64_t bytes) {
    uint32_t alignment = 32;
    while (bytes % alignment) alignment /= 2;
    return 32 / alignment;
  }
  __aicore__ inline uint64_t Segment(uint32_t j) {
    return j < 96 ? cached_[j] : descriptor_[2 + static_cast<uint64_t>(j) * (rank_ + 1) + axis_] * innerBytes_;
  }
  __aicore__ inline void Copy(GlobalTensor<DTYPE_X> &source, uint64_t src, uint64_t dst, uint64_t bytes) {
    while (bytes) {
      const uint32_t n = bytes < kTile ? bytes : kTile;
      auto local = queue_.AllocTensor<DTYPE_X>();
      DataCopyExtParams params{1, n, 0, 0, 0};
      DataCopyPadExtParams<DTYPE_X> pad{false, 0, 0, 0};
      DataCopyPad(local, source[src / sizeof(DTYPE_X)], params, pad);
      queue_.EnQue(local); local = queue_.DeQue<DTYPE_X>();
      DataCopyPad(output_[dst / sizeof(DTYPE_X)], local, params);
      queue_.FreeTensor(local);
      src += n; dst += n; bytes -= n;
    }
  }
  static constexpr uint32_t kTile = 65536;
  TPipe pipe_;
  TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> queue_;
  GlobalTensor<DTYPE_X> output_;
  __gm__ uint64_t *descriptor_, *pointers_;
  uint64_t cached_[96], outer_, innerBytes_, rowBytes_, totalBytes_;
  uint32_t count_, rank_, axis_;
};

// Read each input tile contiguously, then expand rows to DMA alignment in UB.
// uint32_t is used only for bit-preserving movement, never numeric conversion.
class ConcatGatherInputs {
 public:
  template <class Tiling>
  __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const Tiling &data) {
    count_ = data.inputCount;
    axis_ = data.axis;
    rank_ = data.rank;
    innerBytes_ = data.innerBytes;
    rowBytes_ = data.rowBytes;
    outer_ = data.outer;
    for (uint32_t j = 0; j < 96 && j < count_; ++j)
      cached_[j] = data.inputRowBytes[j];
    descriptor_ = reinterpret_cast<__gm__ uint64_t *>(inputs);
    pointers_ = descriptor_ + descriptor_[0] / sizeof(uint64_t);
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(output));
    // rows*bytes + final padding <= rows*padded <= kStage.
    pipe_.InitBuffer(inQueue_, kBuffers, kStage);
    pipe_.InitBuffer(outQueue_, kBuffers, kStage);
    if (true) {
      pipe_.InitBuffer(scratchA_, 8 * kMaskStride);
      if (false) pipe_.InitBuffer(scratchB_, kStage);
      MakeMasks();
    } else {
      pipe_.InitBuffer(scratchA_, kStage);
      pipe_.InitBuffer(scratchB_, kStage);
    }
  }

  __aicore__ inline void Process() {
    const uint32_t descriptorRank = static_cast<uint32_t>(descriptor_[1]);
    if (count_ > 96 && descriptorRank != rank_) return;
    const uint64_t core = GetBlockIdx(), cores = GetBlockNum();
    uint64_t rowBegin = 0, rowEnd = outer_;
    if (false) {
      uint32_t alignment = 32;
      while (rowBytes_ % alignment) alignment /= 2;
      const uint32_t period = 32 / alignment;
      const uint64_t groups = outer_ / period + (outer_ % period != 0);
      rowBegin = groups * core / cores * period;
      rowEnd = groups * (core + 1) / cores * period;
      if (rowEnd > outer_) rowEnd = outer_;
      if (rowBegin >= rowEnd) return;
    }
    const uint64_t workRows = rowEnd - rowBegin;
    uint64_t taskOffset = 0, segmentStart = 0;
    for (uint32_t j = 0; j < count_; ++j) {
      const uint64_t bytes = j < 96 ? cached_[j] :
        descriptor_[2 + static_cast<uint64_t>(j) * (descriptorRank + 1) + axis_] * innerBytes_;
      if (bytes) {
        const uint32_t padded = (static_cast<uint32_t>(bytes) + 31) / 32 * 32;
        const uint32_t capacity = kStage / padded;
        const uint32_t rowLimit = capacity > 4095 ? 4095 : capacity;
        const uint64_t tasks = workRows / rowLimit + (workRows % rowLimit != 0);
        const uint64_t first = false ? 0 : (core + cores - taskOffset % cores) % cores;
        if (first < tasks) {
          GlobalTensor<uint32_t> source;
          source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(pointers_[j]));
          if (true) source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
          const uint32_t indexRows = workRows < rowLimit ? static_cast<uint32_t>(workRows) : rowLimit;
          if (!true && padded != bytes) MakeOffsets(padded / 4, bytes / 4, indexRows);
          const uint64_t firstBegin=rowBegin+first*rowLimit;
          Prefetch(source,firstBegin,rowEnd-firstBegin<rowLimit?rowEnd-firstBegin:rowLimit,bytes,padded);
          for (uint64_t task = first; task < tasks; task += false ? 1 : cores) {
            const uint64_t begin = rowBegin + task * rowLimit;
            const uint16_t rows = static_cast<uint16_t>(rowEnd - begin < rowLimit ? rowEnd - begin : rowLimit);
            auto input=inQueue_.DeQue<uint32_t>();
            if (task+cores<tasks) {
              const uint64_t nextBegin=rowBegin+(task+cores)*rowLimit;
              Prefetch(source,nextBegin,rowEnd-nextBegin<rowLimit?rowEnd-nextBegin:rowLimit,bytes,padded);
            }
            auto result = outQueue_.AllocTensor<uint32_t>();
            if (padded == bytes) {
              DataCopy(result, input, rows * padded / 4);
            } else if (true) {
              if (false && padded <= 1024) {
                auto grouped = scratchB_.Get<uint32_t>();
                RepackMasks(grouped, input, rows, bytes, padded);
                PipeBarrier<PIPE_V>();
                RestoreRows(result, grouped, rows, bytes, padded);
              } else {
                RepackMasks(result, input, rows, bytes, padded);
              }
            } else {
              Gather(result, input, scratchB_.Get<uint32_t>(), 0, rows * padded / 4);
            }
            outQueue_.EnQue(result);
            inQueue_.FreeTensor(input);
            result = outQueue_.DeQue<uint32_t>();
            const uint32_t period = true &&
                !(false && padded <= 1024) ? RowPeriod(bytes) : 1;
            uint32_t sourceOffset = 0;
            for (uint32_t phase = 0; phase < period && phase < rows; ++phase) {
              const uint16_t phaseRows = (rows - 1 - phase) / period + 1;
              DataCopyExtParams out{phaseRows, static_cast<uint32_t>(bytes), 0,
                  static_cast<uint32_t>(period * rowBytes_ - bytes), 0};
              DataCopyPad(output_[((begin + phase) * rowBytes_ + segmentStart) / 4],
                  result[sourceOffset / 4], out);
              sourceOffset += phaseRows * padded;
            }
            outQueue_.FreeTensor(result);
          }
        }
        taskOffset = (taskOffset + tasks) % cores;
      }
      segmentStart += bytes;
    }
  }

 protected:
  __aicore__ inline void Prefetch(GlobalTensor<uint32_t> &source,uint64_t begin,
                                  uint32_t rows,uint32_t bytes,uint32_t padded) {
    auto input=inQueue_.AllocTensor<uint32_t>();
    DataCopyExtParams in{1,rows*bytes,0,0,0};
    DataCopyPadExtParams<uint32_t> pad{true,0,static_cast<uint8_t>((padded-bytes)/4),0};
    DataCopyPad(input,source[begin*bytes/4],in,pad);
    inQueue_.EnQue(input);
  }

  __aicore__ inline uint32_t RowPeriod(uint32_t bytes) {
    uint32_t alignment = 32;
    while (bytes % alignment) alignment /= 2;
    return 32 / alignment;
  }

  __aicore__ inline void MakeMasks() {
    auto masks = scratchA_.Get<uint32_t>();
    Duplicate(masks, UINT32_MAX, 8 * kMaskStride / 4);
    auto vs = static_cast<event_t>(pipe_.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(vs);
    WaitFlag<HardEvent::V_S>(vs);
    for (uint32_t leading = 0; leading < 8; ++leading)
      masks.SetValue(leading * kMaskStride / 4, UINT32_MAX << leading);
    auto sv = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(sv);
    WaitFlag<HardEvent::S_V>(sv);
  }

  __aicore__ inline void RepackMasks(LocalTensor<uint32_t> &dst,
      LocalTensor<uint32_t> &src, uint32_t rows, uint32_t bytes, uint32_t padded) {
    const uint32_t period = RowPeriod(bytes);
    uint32_t outOffset = 0;
    for (uint32_t phase = 0; phase < period && phase < rows; ++phase) {
      const uint16_t n = (rows - 1 - phase) / period + 1;
      const uint32_t offset = phase * bytes;
      const uint32_t leading = offset % 32 / 4;
      auto pattern = scratchA_.Get<uint32_t>()[leading * kMaskStride / 4];
      uint64_t retained = 0;
      GatherMask(dst[outOffset / 4], src[(offset / 32 * 32) / 4], pattern,
          true, padded / 4 + leading,
          {1, n, static_cast<uint16_t>(period * bytes / 32), 0}, retained);
      outOffset += n * padded;
    }
  }

  __aicore__ inline void RestoreRows(LocalTensor<uint32_t> &dst,
      LocalTensor<uint32_t> &src, uint32_t rows, uint32_t bytes, uint32_t padded) {
    const uint32_t period = RowPeriod(bytes);
    uint32_t sourceOffset = 0;
    CopyRepeatParams params{1, 1, static_cast<uint16_t>(period * padded / 32),
        static_cast<uint16_t>(padded / 32)};
    for (uint32_t phase = 0; phase < period && phase < rows; ++phase) {
      const uint32_t count = (rows - 1 - phase) / period + 1;
      for (uint32_t first = 0; first < count; first += 255) {
        const uint8_t repeat = static_cast<uint8_t>(count - first < 255 ? count - first : 255);
        for (uint32_t column = 0; column < padded; column += 256) {
          const uint64_t elements = (padded - column < 256 ? padded - column : 256) / 4;
          Copy(dst[((phase + first * period) * padded + column) / 4],
              src[(sourceOffset + first * padded + column) / 4], elements, repeat, params);
        }
      }
      sourceOffset += count * padded;
    }
    PipeBarrier<PIPE_V>();
  }

  __aicore__ inline void MakeOffsets(uint32_t paddedElements, uint32_t elements, uint32_t rows) {
    const uint32_t n = rows * paddedElements;
    auto a = scratchA_.Get<float>();
    auto b = scratchB_.Get<float>();
    CreateVecIndex(a, 0.0f, n);
    PipeBarrier<PIPE_V>();
    Muls(b, a, 1.0f / static_cast<float>(static_cast<int32_t>(paddedElements)), n);
    PipeBarrier<PIPE_V>();
    Adds(b, b, 1.0f / 8192.0f, n);
    PipeBarrier<PIPE_V>();
    Cast(a.ReinterpretCast<int32_t>(), b, RoundMode::CAST_FLOOR, n);
    PipeBarrier<PIPE_V>();
    Cast(b, a.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, n);
    PipeBarrier<PIPE_V>();
    Muls(b, b, static_cast<float>(static_cast<int32_t>(paddedElements - elements)), n);
    PipeBarrier<PIPE_V>();
    CreateVecIndex(a, 0.0f, n);
    PipeBarrier<PIPE_V>();
    Sub(a, a, b, n);
    PipeBarrier<PIPE_V>();
    Muls(a, a, 4.0f, n);
    PipeBarrier<PIPE_V>();
    Cast(b.ReinterpretCast<int32_t>(), a, RoundMode::CAST_RINT, n);
    PipeBarrier<PIPE_V>();
  }

  static constexpr uint32_t kStage = 32768;
  static constexpr uint32_t kBuffers = 2;
  // Up to 4096 output words plus seven leading words; one bit per word.
  static constexpr uint32_t kMaskStride = 544;
  TPipe pipe_;
  TQue<QuePosition::VECIN, kBuffers> inQueue_;
  TQue<QuePosition::VECOUT, kBuffers> outQueue_;
  TBuf<QuePosition::VECCALC> scratchA_, scratchB_;
  GlobalTensor<uint32_t> output_;
  __gm__ uint64_t *descriptor_, *pointers_;
  uint64_t cached_[96];
  uint64_t innerBytes_, rowBytes_, outer_;
  uint32_t count_, axis_, rank_;
};


// ABI: four u32, four u64, 256 u64 row lengths, two 256-u32 planning arrays.
// Read the fixed header and only the input lengths the selected engine needs.
struct ConcatTilingView {
  uint32_t inputCount, axis, rank, reserved;
  uint64_t outer, innerBytes, rowBytes, totalBytes;
  __gm__ uint64_t *inputRowBytes;
  __gm__ uint32_t *inputOffsets, *columnFirst;
};
__aicore__ inline ConcatTilingView LoadConcatTiling(GM_ADDR raw) {
  auto narrow = reinterpret_cast<__gm__ uint32_t *>(raw);
  auto wide = reinterpret_cast<__gm__ uint64_t *>(raw);
  ConcatTilingView d;
  d.inputCount = narrow[0]; d.axis = narrow[1]; d.rank = narrow[2]; d.reserved = narrow[3];
  d.outer = wide[2]; d.innerBytes = wide[3]; d.rowBytes = wide[4]; d.totalBytes = wide[5];
  d.inputRowBytes = wide + 6;
  d.inputOffsets = narrow + 524;
  d.columnFirst = narrow + 780;
  return d;
}

class ConcatPlannedDirect {
 public:
  template<class Tiling>
  __aicore__ inline void Init(GM_ADDR inputs, GM_ADDR output, const Tiling &d) {
    count_=d.inputCount; groups_=d.reserved; outer_=d.outer; rowBytes_=d.rowBytes;
    const uint32_t core=GetBlockIdx();
    for(uint32_t g=0;g<groups_;++g) {
      const uint32_t first=d.columnFirst[128+g],last=d.columnFirst[129+g];
      if(core<first || core>=last)continue;
      own_.begin=d.columnFirst[g]; own_.end=g+1<groups_?d.columnFirst[g+1]:count_;
      own_.offset=d.inputOffsets[own_.begin];
      own_.width=(own_.end<count_?d.inputOffsets[own_.end]:rowBytes_)-own_.offset;
      own_.stride=(own_.width+31)/32*32; own_.rows=16;
      worker_=core-first; workers_=last-first;
      for(uint32_t j=own_.begin;j<own_.end;++j) {
        bytes_[j]=d.inputRowBytes[j]; starts_[j]=d.inputOffsets[j];
      }
      break;
    }
    auto descriptor=reinterpret_cast<__gm__ uint64_t *>(inputs);
    auto pointers=descriptor+descriptor[0]/8;
    for(uint32_t j=own_.begin;j<own_.end;++j) addresses_[j]=pointers[j];
    for(uint32_t k=0;k<24;++k) {
      const uint32_t j=own_.begin+k;
      lens_[k]=j<own_.end?bytes_[j]:0;
      dests_[k]=j<own_.end?16*(starts_[j]-own_.offset):0;
      if(lens_[k]) {
        sources_[k].SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(addresses_[j]));
        sources_[k].SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
      }
    }
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(output));
    pipe_.InitBuffer(input_,kBuffers*kInput);
    pipe_.InitBuffer(map_,65536);
    pipe_.InitBuffer(outQueue_,1,65536);
    for(uint32_t slot=0;slot<kBuffers;++slot) {
      ready_[slot]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());
      free_[slot]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE2>());
    }
    auto row=input_.Get<int32_t>()[0],col=input_.Get<int32_t>()[512],work=input_.Get<int32_t>()[1024];
    CreateVecIndex(col,int32_t(0),512);
    PipeBarrier<PIPE_V>();
    ShiftRight(row,col,int32_t(6),512);
    PipeBarrier<PIPE_V>();
    Muls(work,row,int32_t(64),512);
    PipeBarrier<PIPE_V>();
    Sub(col,col,work,512);
    PipeBarrier<PIPE_V>();
    Muls(col,col,int32_t(4),512);
    PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Process() {
    BuildMap(own_);
    for(uint32_t slot=0;slot<kBuffers;++slot) SetFlag<HardEvent::V_MTE2>(free_[slot]);
    const uint32_t microBytes=own_.rows*own_.width;
    const uint32_t superRows=16;
    const uint32_t tiles=(outer_+7)/8;
    const uint32_t begin=static_cast<uint64_t>(tiles)*worker_/workers_*8;
    const uint32_t rawEnd=static_cast<uint64_t>(tiles)*(worker_+1)/workers_*8;
    const uint32_t end=rawEnd<outer_?rawEnd:outer_;
    uint32_t iteration=0;
    if(kBuffers==2 && begin<end) ReadSuper(begin,end-begin<superRows?end-begin:superRows,0);
    for(uint32_t row=begin;row<end;row+=superRows,++iteration) {
      const uint32_t slot=iteration%kBuffers;
      const uint32_t rows=end-row<superRows?end-row:superRows;
      if(kBuffers==1) ReadSuper(row,rows,slot);
      WaitFlag<HardEvent::MTE2_V>(ready_[slot]);
      const bool hasNext=kBuffers==2 && row+superRows<end;
      const uint32_t nextRow=row+superRows,nextRows=hasNext?(end-nextRow<superRows?end-nextRow:superRows):0;
      uint32_t nextInput=own_.begin;
      if(hasNext && kInterleaved) WaitFlag<HardEvent::V_MTE2>(free_[slot^1]);
      if(hasNext && !kInterleaved) {
        const uint32_t next=row+superRows;
        ReadSuper(next,end-next<superRows?end-next:superRows,slot^1);
      }
      auto result=outQueue_.AllocTensor<uint32_t>();
      Gather(result,input_.Get<uint32_t>(),map_.Get<uint32_t>(),uint32_t(0),rows*own_.stride/4);
      outQueue_.EnQue(result);result=outQueue_.DeQue<uint32_t>();
      DataCopyExtParams out{static_cast<uint16_t>(rows),own_.width,0,rowBytes_-own_.width,0};
      DataCopyPad(output_[(static_cast<uint64_t>(row)*rowBytes_+own_.offset)/4],result,out);
      outQueue_.FreeTensor(result);
      SetFlag<HardEvent::V_MTE2>(free_[slot]);
    }
    for(uint32_t slot=0;slot<kBuffers;++slot) WaitFlag<HardEvent::V_MTE2>(free_[slot]);
    for(uint32_t slot=0;slot<kBuffers;++slot) {
      pipe_.ReleaseEventID<HardEvent::MTE2_V>(ready_[slot]);
      pipe_.ReleaseEventID<HardEvent::V_MTE2>(free_[slot]);
    }
  }
 private:
  struct Plan { uint32_t begin=0,end=0,offset=0,width=0,stride=0,rows=8; };
  __aicore__ inline void ReadSuper(uint32_t row,uint32_t rows,uint32_t slot) {
    WaitFlag<HardEvent::V_MTE2>(free_[slot]);
    if(own_.end-own_.begin<=24) {
      #pragma unroll
      for(uint32_t k=0;k<24;++k) {
        const uint32_t bytes=lens_[k];
        if(!bytes)continue;
        const uint32_t dest=dests_[k];
        DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
        DataCopyExtParams in{1,rows*bytes,0,0,0};
        DataCopyPad(input_.Get<uint32_t>()[dest/4],sources_[k][static_cast<uint64_t>(row)*bytes/4],in,pad);
      }
    } else {
      for(uint32_t j=own_.begin;j<own_.end;++j) ReadOne(row,rows,slot,j);
    }
    SetFlag<HardEvent::MTE2_V>(ready_[slot]);
  }
  __aicore__ inline void ReadOne(uint32_t row,uint32_t rows,uint32_t slot,uint32_t j) {
    const uint32_t blocks=rows/own_.rows,tail=rows%own_.rows;
      const uint32_t bytes=bytes_[j]; if(!bytes)return;
      GlobalTensor<uint32_t> source;
      source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(addresses_[j]));
      source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
      const uint32_t blockBytes=own_.rows*bytes,dest=slot*kInput+own_.rows*(starts_[j]-own_.offset);
      DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
      if(blocks) {
        DataCopyExtParams in{static_cast<uint16_t>(blocks),blockBytes,0,(own_.rows*own_.width-blockBytes)/32,0};
        DataCopyPad(input_.Get<uint32_t>()[dest/4],source[static_cast<uint64_t>(row)*bytes/4],in,pad);
      }
      if(tail) {
        DataCopyExtParams in{1,tail*bytes,0,0,0};
        DataCopyPad(input_.Get<uint32_t>()[(blocks*own_.rows*own_.width+dest)/4],
                    source[static_cast<uint64_t>(row+blocks*own_.rows)*bytes/4],in,pad);
      }
  }
  __aicore__ inline uint64_t Bits(uint32_t n) { return n>=64?UINT64_MAX:((uint64_t(1)<<n)-1); }
  __aicore__ inline void BuildMap(const Plan &p) {
    auto map=map_.Get<int32_t>(), work=input_.Get<int32_t>()[1024];
    PipeBarrier<PIPE_V>();
    Duplicate(map,int32_t(0),p.rows*p.stride/4);
    PipeBarrier<PIPE_V>();
    for (uint32_t block=0; block<p.rows; block+=kIndexRows) {
      const uint32_t rows=p.rows-block<kIndexRows?p.rows-block:kIndexRows;
      for (uint32_t j=p.begin; j<p.end; ++j) {
        if (!bytes_[j]) continue;
        const uint32_t dest=starts_[j]-p.offset, leading=dest%32;
        Muls(work,input_.Get<int32_t>()[0],static_cast<int32_t>(bytes_[j]),rows*64);
        PipeBarrier<PIPE_V>();
        Add(work,work,input_.Get<int32_t>()[512],rows*64);
        PipeBarrier<PIPE_V>();
        Adds(work,work,static_cast<int32_t>(p.rows*dest+block*bytes_[j]-leading),rows*64);
        PipeBarrier<PIPE_V>();
        for (uint32_t chunk=0; chunk<leading+bytes_[j]; chunk+=256) {
          if (chunk) { Adds(work,work,int32_t(256),rows*64); PipeBarrier<PIPE_V>(); }
          const uint32_t first=chunk<leading?(leading-chunk)/4:0;
          const uint32_t last=leading+bytes_[j]-chunk<256?(leading+bytes_[j]-chunk)/4:64;
          uint64_t mask[2]={Bits(last)&~Bits(first),0};
          CopyRepeatParams params{1,1,static_cast<uint16_t>(p.stride/32),8};
          Copy(map[(block*p.stride+dest-leading+chunk)/4],work,mask,static_cast<uint8_t>(rows),params);
          PipeBarrier<PIPE_V>();
        }
      }
    }
  }

  static constexpr uint32_t kIndexRows=8,kInput=65280,kSlice=32768,kBuffers=1,kOutputTiles=1;
  static constexpr bool kInterleaved=false;
  TPipe pipe_;
  TBuf<QuePosition::VECCALC> input_,map_;
  TQue<QuePosition::VECOUT,2> outQueue_;
  GlobalTensor<uint32_t> output_;
  Plan own_;
  uint32_t count_,groups_,outer_,rowBytes_,worker_=0,workers_=1;
  uint32_t bytes_[256],starts_[256];
  uint64_t addresses_[256];
  uint32_t lens_[24],dests_[24];
  GlobalTensor<uint32_t> sources_[24];
  event_t ready_[kBuffers],free_[kBuffers];
};

// Small row concatenation: contiguous source panels, one complete output tile.
class ConcatSmallPanel {
 public:
  template<class Tiling>
  __aicore__ inline void Init(GM_ADDR inputs,GM_ADDR output,GM_ADDR raw,const Tiling &d) {
    count_=d.inputCount;outer_=d.outer;width_=d.rowBytes;stride_=(width_+31)/32*32;
    auto descriptor=reinterpret_cast<__gm__ uint64_t *>(inputs);
    auto pointers=descriptor+descriptor[0]/8;
    uint32_t prefix=0;
    for(uint32_t k=0;k<16;++k) {
      lens_[k]=k<count_?d.inputRowBytes[k]:0;offsets_[k]=16*prefix;
      if(lens_[k])sources_[k].SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(pointers[k]));
      prefix+=lens_[k];
    }
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t *>(output));
    pipe_.InitBuffer(in_,2,16384);pipe_.InitBuffer(out_,2,16384);pipe_.InitBuffer(map_,32768);
    GlobalTensor<uint32_t> table;
    table.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(raw)+1036);
    DataCopy(map_.Get<uint32_t>(),table,16*stride_/2);
    const event_t ready=static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(ready);WaitFlag<HardEvent::MTE2_V>(ready);
  }
  __aicore__ inline void Process() {
    const uint32_t core=GetBlockIdx(),cores=GetBlockNum(),tiles=(outer_+15)/16;
    if(core>=tiles)return;
    Read(core*16,outer_-core*16<16?outer_-core*16:16);
    for(uint32_t tile=core;tile<tiles;tile+=cores) {
      const uint32_t row=tile*16,rows=outer_-row<16?outer_-row:16;
      auto input=in_.DeQue<uint16_t>();
      if(tile+cores<tiles) {
        const uint32_t next=(tile+cores)*16;Read(next,outer_-next<16?outer_-next:16);
      }
      auto result=out_.AllocTensor<uint16_t>();
      Gather(result,input,map_.Get<uint32_t>(),uint32_t(0),rows*stride_/2);
      out_.EnQue(result);in_.FreeTensor(input);result=out_.DeQue<uint16_t>();
      DataCopyExtParams p{static_cast<uint16_t>(rows),width_,0,0,0};
      DataCopyPad(output_[static_cast<uint64_t>(row)*width_/2],result,p);
      out_.FreeTensor(result);
    }
  }
 private:
  __aicore__ inline void Read(uint32_t row,uint32_t rows) {
    auto local=in_.AllocTensor<uint16_t>();
    #pragma unroll
    for(uint32_t k=0;k<16;++k) {
      if(!lens_[k])continue;
      DataCopyExtParams p{1,rows*lens_[k],0,0,0};
      DataCopyPadExtParams<uint16_t> pad{false,0,0,0};
      DataCopyPad(local[offsets_[k]/2],sources_[k][static_cast<uint64_t>(row)*lens_[k]/2],p,pad);
    }
    in_.EnQue(local);
  }
  TPipe pipe_;TQue<TPosition::VECIN,2> in_;TQue<TPosition::VECOUT,2> out_;TBuf<TPosition::VECCALC> map_;
  GlobalTensor<uint16_t> output_,sources_[16];
  uint32_t count_,outer_,width_,stride_,lens_[16],offsets_[16];
};

// Flat concatenation: only this core's commands are read; prefetch crosses tensors.
class ConcatFlatCommands {
 public:
  template<class Tiling>
  __aicore__ inline void Init(GM_ADDR inputs,GM_ADDR output,GM_ADDR raw,const Tiling &d) {
    auto desc=reinterpret_cast<__gm__ uint64_t *>(inputs);
    pointers_=desc+desc[0]/8;
    auto plan=reinterpret_cast<__gm__ uint32_t *>(raw)+1036;
    first_=plan[GetBlockIdx()];last_=plan[GetBlockIdx()+1];commands_=plan+65;
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(output));
    pipe_.InitBuffer(queue_,2,65536);
  }
  __aicore__ inline void Process() {
    if(first_==last_)return;
    Task current=Get(first_);Read(current);
    for(uint32_t index=first_;index<last_;++index) {
      auto local=queue_.DeQue<uint8_t>();
      Task next;
      if(index+1<last_) {next=Get(index+1);Read(next);}
      DataCopyExtParams p{1,current.bytes,0,0,0};
      DataCopyPad(output_[current.dest],local,p);
      queue_.FreeTensor(local);
      if(index+1<last_)current=next;
    }
  }
 private:
  struct Task {uint32_t input,source,dest,bytes;};
  __aicore__ inline Task Get(uint32_t n) {
    return {commands_[4*n],commands_[4*n+1],commands_[4*n+2],commands_[4*n+3]};
  }
  __aicore__ inline void Read(const Task &t) {
    auto local=queue_.AllocTensor<uint8_t>();
    GlobalTensor<uint8_t> source;
    source.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(pointers_[t.input]));
    DataCopyExtParams p{1,t.bytes,0,0,0};DataCopyPadExtParams<uint8_t> pad{false,0,0,0};
    DataCopyPad(local,source[t.source],p,pad);queue_.EnQue(local);
  }
  TPipe pipe_;TQueBind<TPosition::VECIN,TPosition::VECOUT,2> queue_;
  GlobalTensor<uint8_t> output_;__gm__ uint64_t *pointers_;__gm__ uint32_t *commands_;
  uint32_t first_,last_;
};

// Exact row-segment tasks with the next GM read submitted before the current write.
class ConcatRowCommands {
 public:
  template<class Tiling>
  __aicore__ inline void Init(GM_ADDR inputs,GM_ADDR output,GM_ADDR raw,const Tiling &d) {
    auto descriptor=reinterpret_cast<__gm__ uint64_t *>(inputs);pointers_=descriptor+descriptor[0]/8;
    plan_=reinterpret_cast<__gm__ uint32_t *>(raw)+1036;commands_=plan_+65;
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(output));pipe_.InitBuffer(queue_,2,kTile);
  }
  __aicore__ inline void Process() {
    const uint32_t core=GetBlockIdx(),first=plan_[core],last=plan_[core+1];if(first==last)return;
    Work current=Decode(first);Read(current);
    for(uint32_t c=first;c<last;++c) {
      auto local=queue_.DeQue<DTYPE_X>();Work next;
      if(c+1<last){next=Decode(c+1);Read(next);}
      DataCopyExtParams move{1,current.bytes,0,0,0};DataCopyPad(output_[current.dest/sizeof(DTYPE_X)],local,move);
      queue_.FreeTensor(local);if(c+1<last)current=next;
    }
  }
 private:
  struct Work{uint32_t input,src,dest,bytes;};
  __aicore__ inline Work Decode(uint32_t c){return {commands_[4*c],commands_[4*c+1],commands_[4*c+2],commands_[4*c+3]};}
  __aicore__ inline void Read(const Work&w) {
    GlobalTensor<DTYPE_X> source;source.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(pointers_[w.input]));source.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    auto local=queue_.AllocTensor<DTYPE_X>();DataCopyExtParams move{1,w.bytes,0,0,0};DataCopyPadExtParams<DTYPE_X> pad{false,0,0,0};
    DataCopyPad(local,source[w.src/sizeof(DTYPE_X)],move,pad);queue_.EnQue(local);
  }
  static constexpr uint32_t kTile=65536;
  TPipe pipe_;TQueBind<QuePosition::VECIN,QuePosition::VECOUT,2> queue_;GlobalTensor<DTYPE_X> output_;
  __gm__ uint64_t *pointers_;__gm__ uint32_t *plan_,*commands_;
};

extern "C" __global__ __aicore__ void concat(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  auto data = LoadConcatTiling(tiling);
  if (TILING_KEY_IS(5)) {
    ConcatGatherInputs gather;
    gather.Init(x, y, data);
    gather.Process();
    return;
  }
  if(TILING_KEY_IS(8)) {
    ConcatFlatCommands op;op.Init(x,y,tiling,data);op.Process();return;
  }
  if(TILING_KEY_IS(9)) {
    ConcatRowCommands op;op.Init(x,y,tiling,data);op.Process();return;
  }
  if(TILING_KEY_IS(7)) {
    ConcatSmallPanel op;op.Init(x,y,tiling,data);op.Process();return;
  }
  if (TILING_KEY_IS(6)) {
    ConcatPlannedDirect direct;
    direct.Init(x, y, data);
    direct.Process();
    return;
  }
  ConcatCopyEngine op;
  op.Init(x, y, data);
  if (TILING_KEY_IS(1)) op.Flat();
  else if (TILING_KEY_IS(2)) op.SmallRows();
  else if (TILING_KEY_IS(3)) op.PhaseViews();
  else if (TILING_KEY_IS(4)) op.General();
}
