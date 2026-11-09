#include "kernel_operator.h"
using namespace AscendC;
#include "row_gather.h"
#include "row_batch.h"
#include "slice_pipeline.h"
#include "int32_atomic.h"
#include "int32_packed_atomic.h"
#include "int8_resident.h"
#include "parallel_index_plan.h"
#include "int32_partial_outer.h"
#include "bf16_outer_batch.h"

// General IndexAdd path. Each core owns whole 32-byte output blocks, and
// destination lists retain ascending source-index order.
template<class T, uint32_t KIND> class IndexAddBaseline {
 public:
  template<class D> __aicore__ inline void Init(GM_ADDR self, GM_ADDR index,
      GM_ADDR source, GM_ADDR output, const D &d) {
    axis_ = d.axisSize; inner_ = d.inner; count_ = d.indexCount; total_ = d.total;
    useList_ = d.listBytes != 0;
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(self));
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(source));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    pipe_.InitBuffer(outBuf_, TILE * sizeof(T));
    pipe_.InitBuffer(srcBuf_, TILE * sizeof(T));
    if constexpr (KIND == 3) {
      pipe_.InitBuffer(fpA_, TILE * sizeof(float));
      pipe_.InitBuffer(fpB_, TILE * sizeof(float));
    }
    if constexpr (KIND == 5) {
      pipe_.InitBuffer(intAcc_, TILE * sizeof(int32_t));
      pipe_.InitBuffer(intSrc_, TILE * sizeof(int32_t));
      pipe_.InitBuffer(intTmp_, TILE * sizeof(half));

    }
    if (useList_) pipe_.InitBuffer(listBuf_, d.listBytes);
    useIndexCache_ = count_ > 0 && count_ <= 8000;
    if(useIndexCache_)pipe_.InitBuffer(indexBuf_, (count_+7)/8*32);
  }
  __aicore__ inline void Process() {
    if (!total_ || !inner_ || !axis_) return;
    if(useIndexCache_) {
      DataCopyExtParams ip{1,static_cast<uint32_t>(count_*4),0,0,0};
      DataCopyPadExtParams<int32_t> pad{false,0,0,0};
      DataCopyPad(indexBuf_.Get<int32_t>(),index_,ip,pad);PipeBarrier<PIPE_ALL>();
    }
    if (useList_) {
      auto list = listBuf_.Get<int32_t>();
      for (uint64_t a = 0; a < axis_; ++a) list.SetValue(a, -1);
      for (uint64_t j = count_; j != 0;) {
        --j;
        const int32_t a = ReadIndex(j);
        if (a < 0 || static_cast<uint64_t>(a) >= axis_) return;
        list.SetValue(axis_ + j, list.GetValue(a));
        list.SetValue(a, static_cast<int32_t>(j));
      }
    }
    constexpr uint32_t ALIGN = 32 / sizeof(T);
    const uint64_t blocks = total_ / ALIGN + (total_ % ALIGN != 0);
    uint64_t pos = blocks * GetBlockIdx() / GetBlockNum() * ALIGN;
    uint64_t end = blocks * (GetBlockIdx() + 1) / GetBlockNum() * ALIGN;
    if (end > total_) end = total_;
    while (pos < end) {
      const uint64_t lane = pos % inner_;
      const uint64_t row = pos / inner_;
      const uint64_t a = row % axis_, outer = row / axis_;
      uint64_t n64 = inner_ - lane;
      if (n64 > end - pos) n64 = end - pos;
      if (n64 > TILE) n64 = TILE;
      const uint32_t n = static_cast<uint32_t>(n64);
      auto out = outBuf_.Get<T>();
      DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
      DataCopyPadExtParams<T> pad{false, 0, 0, 0};
      DataCopyPad(out, self_[pos], p, pad);
      PipeBarrier<PIPE_ALL>();
      if constexpr (KIND == 5) {
        Cast(intTmp_.Get<half>(),out,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
        Cast(intAcc_.Get<int32_t>(),intTmp_.Get<half>(),RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();
      }
      if (useList_) {
        auto list = listBuf_.Get<int32_t>();
        for (int32_t j = list.GetValue(a); j >= 0; j = list.GetValue(axis_ + j))
          Accumulate(out, (outer * count_ + j) * inner_ + lane, n);
      } else {
        for (uint64_t j = 0; j < count_; ++j)
          if (ReadIndex(j) == static_cast<int64_t>(a))
            Accumulate(out, (outer * count_ + j) * inner_ + lane, n);
      }
      if constexpr (KIND == 5) {
        auto signedAcc=intAcc_.Get<int32_t>();auto bits=intAcc_.Get<uint32_t>();
        Adds(signedAcc,signedAcc,static_cast<int32_t>(128),n);PipeBarrier<PIPE_V>();
        ShiftLeft(bits,bits,24U,n);PipeBarrier<PIPE_V>();ShiftRight(bits,bits,24U,n);PipeBarrier<PIPE_V>();
        Adds(signedAcc,signedAcc,static_cast<int32_t>(-128),n);PipeBarrier<PIPE_V>();
        SetDeqScale(static_cast<half>(1.0));
        Cast(intTmp_.Get<half>(),intAcc_.Get<int32_t>(),RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
        Cast(out,intTmp_.Get<half>(),RoundMode::CAST_NONE,n);
      }
      PipeBarrier<PIPE_ALL>();
      DataCopyPad(output_[pos], out, p);
      PipeBarrier<PIPE_ALL>();
      pos += n;
    }
  }
 private:
  __aicore__ inline int32_t ReadIndex(uint64_t j) {
    return useIndexCache_?indexBuf_.Get<int32_t>().GetValue(j):index_.GetValue(j);
  }
  __aicore__ inline void Accumulate(LocalTensor<T> out, uint64_t offset, uint32_t n) {
    auto src = srcBuf_.Get<T>();
    DataCopyExtParams p{1, static_cast<uint32_t>(n * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(src, source_[offset], p, pad);
    PipeBarrier<PIPE_ALL>();
    if constexpr (KIND == 5) {
      // Accumulate modulo 2^32, then sign-extend the low byte once at output.
      // Every int8-to-half-to-int32 conversion is exact.
      Cast(intTmp_.Get<half>(),src,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
      Cast(intSrc_.Get<int32_t>(),intTmp_.Get<half>(),RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();
      Add(intAcc_.Get<int32_t>(),intAcc_.Get<int32_t>(),intSrc_.Get<int32_t>(),n);
    } else if constexpr (KIND == 3) {
      auto a = fpA_.Get<float>(); auto b = fpB_.Get<float>();
      Cast(a, out, RoundMode::CAST_NONE, n);
      Cast(b, src, RoundMode::CAST_NONE, n);
      PipeBarrier<PIPE_V>();
      Add(a, a, b, n);
      PipeBarrier<PIPE_V>();
      Cast(out, a, RoundMode::CAST_RINT, n);
    } else {
      // int32 vector addition uses modulo 2^32; no C++ signed arithmetic.
      // half is rounded on each addition, matching int32-index CPU slices.
      Add(out, out, src, n);
    }
    PipeBarrier<PIPE_ALL>();
  }
  static constexpr uint32_t TILE = 2048;
  TPipe pipe_;
  TBuf<TPosition::VECCALC> outBuf_, srcBuf_, listBuf_, fpA_, fpB_;
  TBuf<TPosition::VECCALC> intAcc_, intSrc_, intTmp_, intMask_;
  GlobalTensor<T> self_, source_, output_;
  GlobalTensor<int32_t> index_;
  uint64_t axis_, inner_, count_, total_;
  bool useList_, useIndexCache_;
  TBuf<TPosition::VECCALC> indexBuf_;
};

template<class T,uint32_t KIND,class D> __aicore__ inline void RunIndexAdd(
    GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR output,GM_ADDR workspace,const D &d) {
  if constexpr(KIND==5){
    if(IndexAddInt8Resident::Eligible(d)){IndexAddInt8Resident k;k.Init(self,index,source,output,d);k.Process();return;}
  }
  if constexpr(KIND==1 || KIND==2) {
    if(d.parallelPlan!=0) {
      IndexAddParallelPlan<T> k;k.Init(self,index,source,output,GetUserWorkspace(workspace),d);k.Process();return;
    }
  }
  if constexpr(KIND==3) {
    if(d.bf16Batch!=0) {
      IndexAddBf16OuterBatch k;k.Init(self,index,source,output,GetUserWorkspace(workspace),d);k.Process();return;
    }
  }
  if constexpr(KIND==4) {
    if(IndexAddInt32PackedAtomic::Eligible(d)) {
      IndexAddInt32PackedAtomic k;k.Init(self,index,source,output,d);k.Process();return;
    }

    if(IndexAddInt32PartialOuter::Eligible(d)) {
      IndexAddInt32PartialOuter k;k.Init(self,index,source,output,d);k.Process();return;
    }

    if(IndexAddInt32Atomic::Eligible(d)) {
      IndexAddInt32Atomic k;k.Init(self,index,source,output,d);k.Process();return;
    }
  }
  if constexpr(KIND==1 || KIND==2 || KIND==4) {
    if(IndexAddSlicePipeline<T>::Eligible(d)) {
      IndexAddSlicePipeline<T> k;k.Init(self,index,source,output,d);k.Process();return;
    }
    if(IndexAddRowBatch<T>::Eligible(d)) {
      IndexAddRowBatch<T> k;k.Init(self,index,source,output,d);k.Process();return;
    }
    if(IndexAddRowGather<T>::Eligible(d)) {
      IndexAddRowGather<T> k;k.Init(self,index,source,output,d);k.Process();return;
    }
  }
  IndexAddBaseline<T,KIND> k;k.Init(self,index,source,output,d);k.Process();
}

extern "C" __global__ __aicore__ void index_add(GM_ADDR self, GM_ADDR index,
    GM_ADDR source, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(d, tiling);
  if (TILING_KEY_IS(1)) { RunIndexAdd<float, 1>(self,index,source,output,workspace,d); }
  if (TILING_KEY_IS(2)) { RunIndexAdd<half, 2>(self,index,source,output,workspace,d); }
  if (TILING_KEY_IS(3)) { RunIndexAdd<bfloat16_t, 3>(self,index,source,output,workspace,d); }
  if (TILING_KEY_IS(4)) { RunIndexAdd<int32_t, 4>(self,index,source,output,workspace,d); }
  if (TILING_KEY_IS(5)) { RunIndexAdd<int8_t, 5>(self,index,source,output,workspace,d); }
}
