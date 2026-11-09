#pragma once
// Read multiple complete source rows as one contiguous transfer. Gather each
// row into aligned UB storage, then atomically add it to the actual destination.
class IndexAddInt32PackedAtomic {
 public:
  template<class D> __aicore__ static inline bool Eligible(const D &d){
    if(d.inner<256 || d.inner>16384 || d.indexCount==0 || d.indexCount>8000 || d.total==0)return false;
    const uint64_t stride=(d.inner+7)/8*8,offsets=(d.inner+63)/64*64;
    return 2*(SOURCE_BYTES+32)+2*stride*4+offsets*4+(d.indexCount+7)/8*32<=190000;
  }
  template<class D> __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR output,const D &d){
    axis_=d.axisSize;inner_=d.inner;count_=d.indexCount;outer_=d.outer;total_=d.total;
    stride_=(inner_+7)/8*8;offsetN_=(inner_+63)/64*64;batch_=SOURCE_BYTES/(inner_*4);
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(self));source_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(source));
    source_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(output));index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    pipe_.InitBuffer(input_,2*(SOURCE_BYTES+32));pipe_.InitBuffer(aligned_,2,stride_*4);
    pipe_.InitBuffer(indices_,(count_+7)/8*32);pipe_.InitBuffer(offsets_,offsetN_*4);
  }
  __aicore__ inline void Process(){
    const auto toStore=static_cast<event_t>(pipe_.AllocEventID<HardEvent::V_MTE3>());
    for(uint32_t b=0;b<2;++b){in_[b]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE2_V>());reuse_[b]=static_cast<event_t>(pipe_.AllocEventID<HardEvent::MTE3_MTE2>());}
    SetAtomicNone();DataCopyPadExtParams<int32_t> pad{false,0,0,0};const uint64_t blocks=(total_+7)/8;
    uint64_t pos=blocks*GetBlockIdx()/GetBlockNum()*8,end=blocks*(GetBlockIdx()+1)/GetBlockNum()*8;if(end>total_)end=total_;
    for(;pos<end;){
      const uint32_t n=end-pos>SOURCE_BYTES/4?SOURCE_BYTES/4:static_cast<uint32_t>(end-pos);
      auto t=input_.Get<int32_t>();DataCopyExtParams cp{1,n*4,0,0,0};DataCopyPad(t,self_[pos],cp,pad);
      PipeBarrier<PIPE_ALL>();DataCopyPad(output_[pos],t,cp);PipeBarrier<PIPE_ALL>();pos+=n;
    }
    PipeBarrier<PIPE_ALL>();SyncAll<true>();
    auto indices=indices_.Get<int32_t>();auto offsets=offsets_.Get<int32_t>();DataCopyExtParams ip{1,static_cast<uint32_t>(count_*4),0,0,0};
    DataCopyPad(indices,index_,ip,pad);PipeBarrier<PIPE_ALL>();
    const uint64_t rows=outer_*count_,begin=rows*GetBlockIdx()/GetBlockNum(),last=rows*(GetBlockIdx()+1)/GetBlockNum();
    if(begin==last){pipe_.ReleaseEventID<HardEvent::V_MTE3>(toStore);for(uint32_t b=0;b<2;++b){pipe_.ReleaseEventID<HardEvent::MTE2_V>(in_[b]);pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(reuse_[b]);}return;}CreateVecIndex(offsets,static_cast<int32_t>(0),offsetN_);PipeBarrier<PIPE_V>();Muls(offsets,offsets,static_cast<int32_t>(4),offsetN_);PipeBarrier<PIPE_V>();Load(begin,last,0);if(begin+batch_<last)Load(begin+batch_,last,1);SetAtomicAdd<int32_t>();
    uint32_t bank=0;
    for(uint64_t first=begin;first<last;first+=batch_){
      if(first!=begin && first+batch_<last){const uint32_t nb=1-bank;WaitFlag<HardEvent::MTE3_MTE2>(reuse_[nb]);Load(first+batch_,last,nb);}
      WaitFlag<HardEvent::MTE2_V>(in_[bank]);auto src=input_.Get<int32_t>()[bank*(SOURCE_BYTES+32)/4];SetFlag<HardEvent::V_MTE3>(toStore);WaitFlag<HardEvent::V_MTE3>(toStore);const uint32_t nr=last-first>batch_?batch_:static_cast<uint32_t>(last-first);
      
      for(uint32_t k=0;k<nr;++k){
        const uint64_t row=first+k;const int32_t a=indices.GetValue(row%count_);
        if(a>=0 && static_cast<uint64_t>(a)<axis_){
          DataCopyExtParams cp{1,static_cast<uint32_t>(inner_*4),0,0,0};
          if(((k*inner_)&7)==0){
            DataCopyPad(output_[((row/count_)*axis_+a)*inner_],src[k*inner_],cp);
          }else{
            auto dst=aligned_.AllocTensor<int32_t>();Gather(dst,src,offsets.template ReinterpretCast<uint32_t>(),static_cast<uint32_t>(k*inner_*4),stride_);aligned_.EnQue(dst);
            auto ready=aligned_.DeQue<int32_t>();DataCopyPad(output_[((row/count_)*axis_+a)*inner_],ready,cp);aligned_.FreeTensor(ready);
          }
        }
        
      }
      if(first+2*batch_<last)SetFlag<HardEvent::MTE3_MTE2>(reuse_[bank]);bank=1-bank;
    }
    PipeBarrier<PIPE_ALL>();SetAtomicNone();pipe_.ReleaseEventID<HardEvent::V_MTE3>(toStore);for(uint32_t b=0;b<2;++b){pipe_.ReleaseEventID<HardEvent::MTE2_V>(in_[b]);pipe_.ReleaseEventID<HardEvent::MTE3_MTE2>(reuse_[b]);}
  }
 private:
  __aicore__ inline void Load(uint64_t row,uint64_t last,uint32_t bank){
    const uint32_t n=last-row>batch_?batch_:static_cast<uint32_t>(last-row);auto src=input_.Get<int32_t>()[bank*(SOURCE_BYTES+32)/4];
    DataCopyExtParams cp{1,static_cast<uint32_t>(n*inner_*4),0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(src,source_[row*inner_],cp,pad);SetFlag<HardEvent::MTE2_V>(in_[bank]);
  }
  static constexpr uint32_t SOURCE_BYTES=65536;
  TPipe pipe_;TBuf<TPosition::VECCALC> input_;TQue<QuePosition::VECOUT,2> aligned_;
  TBuf<TPosition::VECCALC> indices_,offsets_;GlobalTensor<int32_t> self_,source_,output_,index_;
  uint64_t axis_,inner_,count_,outer_,total_;uint32_t stride_,offsetN_,batch_;event_t in_[2],reuse_[2];
};
