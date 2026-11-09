#pragma once
// Partition source rows across all cores. Integer partial sums are accumulated
// in UB and each core atomically contributes one completed slab per outer.
class IndexAddInt32PartialOuter {
 public:
  template<class D> __aicore__ static inline bool Eligible(const D &d){
    if(d.inner<256 || d.indexCount==0 || d.indexCount>8000 || d.axisSize==0 || d.total==0)return false;
    const uint64_t stride=(d.inner+7)/8*8;
    if(stride>40000 || d.axisSize>40000/stride)return false;
    return (d.axisSize+2)*stride*4+(d.indexCount+7)/8*32<=190000;
  }
  template<class D> __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR output,const D &d){
    axis_=d.axisSize;inner_=d.inner;count_=d.indexCount;outer_=d.outer;total_=d.total;stride_=(inner_+7)/8*8;
    const uint32_t accBytes=axis_*stride_*4,indexBytes=(count_+7)/8*32;
    batch_=(190000-accBytes-indexBytes)/(2*stride_*4);if(batch_>128)batch_=128;
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(self));
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(source));
    source_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output));
    index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    pipe_.InitBuffer(acc_,accBytes);pipe_.InitBuffer(indices_,indexBytes);pipe_.InitBuffer(queue_,2,batch_*stride_*4);
  }
  __aicore__ inline void Process(){
    auto acc=acc_.Get<int32_t>();auto indices=indices_.Get<int32_t>();
    DataCopyPadExtParams<int32_t> pad{false,0,0,0};SetAtomicNone();
    const uint64_t blocks=(total_+7)/8;
    uint64_t pos=blocks*GetBlockIdx()/GetBlockNum()*8,end=blocks*(GetBlockIdx()+1)/GetBlockNum()*8;
    if(end>total_)end=total_;
    const uint32_t tile=axis_*stride_>4096?4096:static_cast<uint32_t>(axis_*stride_);
    for(;pos<end;){
      const uint32_t n=end-pos>tile?tile:static_cast<uint32_t>(end-pos);DataCopyExtParams cp{1,n*4,0,0,0};
      DataCopyPad(acc,self_[pos],cp,pad);PipeBarrier<PIPE_ALL>();DataCopyPad(output_[pos],acc,cp);PipeBarrier<PIPE_ALL>();pos+=n;
    }
    SyncAll<true>();
    DataCopyExtParams ip{1,static_cast<uint32_t>(count_*4),0,0,0};DataCopyPad(indices,index_,ip,pad);PipeBarrier<PIPE_ALL>();
    const uint64_t rows=outer_*count_;
    uint64_t first=rows*GetBlockIdx()/GetBlockNum(),last=rows*(GetBlockIdx()+1)/GetBlockNum();
    for(;first<last;){
      const uint64_t outer=first/count_;const uint32_t begin=first%count_;
      const uint64_t boundary=(outer+1)*count_;const uint64_t finish=last<boundary?last:boundary;
      const uint32_t stop=finish-outer*count_;
      Duplicate(acc,static_cast<int32_t>(0),axis_*stride_);PipeBarrier<PIPE_ALL>();Load(outer,begin,stop);
      for(uint32_t j=begin;j<stop;j+=batch_){
        if(j+batch_<stop)Load(outer,j+batch_,stop);
        auto src=queue_.DeQue<int32_t>();const uint32_t n=stop-j>batch_?batch_:stop-j;
        for(uint32_t k=0;k<n;++k){
          const int32_t a=indices.GetValue(j+k);if(a<0 || static_cast<uint64_t>(a)>=axis_)continue;
          Add(acc[static_cast<uint64_t>(a)*stride_],acc[static_cast<uint64_t>(a)*stride_],src[k*stride_],inner_);PipeBarrier<PIPE_V>();
        }
        queue_.FreeTensor(src);
      }
      PipeBarrier<PIPE_ALL>();SetAtomicAdd<int32_t>();
      DataCopyExtParams cp{static_cast<uint16_t>(axis_),static_cast<uint32_t>(inner_*4),0,0,0};
      DataCopyPad(output_[outer*axis_*inner_],acc,cp);PipeBarrier<PIPE_ALL>();SetAtomicNone();first=finish;
    }
  }
 private:
  __aicore__ inline void Load(uint64_t outer,uint32_t j,uint32_t stop){
    const uint32_t n=stop-j>batch_?batch_:stop-j;auto src=queue_.AllocTensor<int32_t>();
    DataCopyExtParams cp{static_cast<uint16_t>(n),static_cast<uint32_t>(inner_*4),0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(src,source_[(outer*count_+j)*inner_],cp,pad);queue_.EnQue(src);
  }
  TPipe pipe_;TBuf<TPosition::VECCALC> acc_,indices_;TQue<QuePosition::VECIN,2> queue_;
  GlobalTensor<int32_t> self_,source_,output_,index_;uint64_t axis_,inner_,stride_,count_,outer_,total_;uint32_t batch_;
};
