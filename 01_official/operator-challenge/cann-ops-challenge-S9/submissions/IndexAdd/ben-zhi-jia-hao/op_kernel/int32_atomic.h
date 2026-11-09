#pragma once
// Integer modulo addition is associative. Initialize the output, synchronize
// every participating core, then stream contiguous source slices with atomic
// int32 additions. Floating point types never enter this path.
class IndexAddInt32Atomic {
 public:
  template<class D> __aicore__ static inline bool Eligible(const D &d) {
    return d.inner>=256 && d.indexCount>0 && d.indexCount<=8000 && d.total>0;
  }
  template<class D> __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR output,const D &d) {
    axis_=d.axisSize;inner_=d.inner;count_=d.indexCount;outer_=d.outer;total_=d.total;
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(self));
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(source));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output));
    index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    pipe_.InitBuffer(queue_,2,TILE*4);pipe_.InitBuffer(indexCache_,(count_+7)/8*32);
  }
  __aicore__ inline void Process() {
    auto indices=indexCache_.Get<int32_t>();DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyExtParams ip{1,static_cast<uint32_t>(count_*4),0,0,0};
    DataCopyPad(indices,index_,ip,pad);PipeBarrier<PIPE_ALL>();
    for(uint32_t j=0;j<count_;++j){int32_t a=indices.GetValue(j);if(a<0 || static_cast<uint64_t>(a)>=axis_)return;}
    SetAtomicNone();
    uint64_t blocks=(total_+7)/8,begin=blocks*GetBlockIdx()/GetBlockNum()*8;
    uint64_t end=blocks*(GetBlockIdx()+1)/GetBlockNum()*8;if(end>total_)end=total_;
    for(uint64_t p=begin;p<end;){
      uint32_t n=end-p>TILE?TILE:static_cast<uint32_t>(end-p);
      auto x=queue_.AllocTensor<int32_t>();DataCopyExtParams cp{1,n*4,0,0,0};
      DataCopyPad(x,self_[p],cp,pad);queue_.EnQue(x);auto ready=queue_.DeQue<int32_t>();
      DataCopyPad(output_[p],ready,cp);queue_.FreeTensor(ready);p+=n;
    }
    PipeBarrier<PIPE_ALL>();SyncAll<true>();
    SetAtomicAdd<int32_t>();
    const uint64_t rows=outer_*count_,rowBegin=rows*GetBlockIdx()/GetBlockNum(),rowEnd=rows*(GetBlockIdx()+1)/GetBlockNum();
    for(uint64_t row=rowBegin;row<rowEnd;++row){
      const uint64_t dstRow=(row/count_)*axis_+indices.GetValue(row%count_);
      for(uint64_t lane=0;lane<inner_;){
        const uint32_t n=inner_-lane>TILE?TILE:static_cast<uint32_t>(inner_-lane);
        auto x=queue_.AllocTensor<int32_t>();DataCopyExtParams cp{1,n*4,0,0,0};
        DataCopyPad(x,source_[row*inner_+lane],cp,pad);queue_.EnQue(x);
        auto ready=queue_.DeQue<int32_t>();DataCopyPad(output_[dstRow*inner_+lane],ready,cp);
        queue_.FreeTensor(ready);lane+=n;
      }
    }
    PipeBarrier<PIPE_ALL>();SetAtomicNone();
  }
 private:
  static constexpr uint32_t TILE=16384;
  TPipe pipe_;TQueBind<QuePosition::VECIN,QuePosition::VECOUT,2> queue_;
  TBuf<TPosition::VECCALC> indexCache_;
  GlobalTensor<int32_t> self_,source_,output_,index_;
  uint64_t axis_,inner_,count_,outer_,total_;
};
