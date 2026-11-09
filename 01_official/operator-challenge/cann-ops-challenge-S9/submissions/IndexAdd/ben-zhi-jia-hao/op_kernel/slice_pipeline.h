#pragma once
// Each core owns aligned output spans. Two source buffers overlap DMA and Add,
// while following each destination's original ascending source-index order.
template<class T> class IndexAddSlicePipeline {
 public:
  template<class D> __aicore__ static inline bool Eligible(const D &d) {
    return d.inner>=256 && d.listBytes && d.indexCount>0 && d.indexCount<=8000;
  }
  template<class D> __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR output,const D &d) {
    axis_=d.axisSize;inner_=d.inner;count_=d.indexCount;total_=d.total;
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(self));
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(source));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    pipe_.InitBuffer(out_,TILE*sizeof(T));pipe_.InitBuffer(inputQueue_,2,TILE*sizeof(T));
    pipe_.InitBuffer(list_,d.listBytes);pipe_.InitBuffer(indexCache_,(count_+7)/8*32);
    // fp32/int32 maximum: 98304 list + 32000 index + 3*16384 = 179456 bytes.
  }
  __aicore__ inline void Process() {
    if(!total_ || !axis_)return;
    auto list=list_.Get<int32_t>();auto indices=indexCache_.Get<int32_t>();
    DataCopyExtParams ip{1,static_cast<uint32_t>(count_*4),0,0,0};
    DataCopyPadExtParams<int32_t> ipPad{false,0,0,0};
    DataCopyPad(indices,index_,ip,ipPad);PipeBarrier<PIPE_ALL>();
    for(uint64_t a=0;a<axis_;++a)list.SetValue(a,-1);
    for(uint64_t j=count_;j;){
      --j;const int32_t a=indices.GetValue(j);if(a<0 || static_cast<uint64_t>(a)>=axis_)return;
      list.SetValue(axis_+j,list.GetValue(a));list.SetValue(a,static_cast<int32_t>(j));
    }
    const uint64_t blocks=(total_+ALIGN-1)/ALIGN;
    uint64_t pos=blocks*GetBlockIdx()/GetBlockNum()*ALIGN;
    uint64_t end=blocks*(GetBlockIdx()+1)/GetBlockNum()*ALIGN;if(end>total_)end=total_;
    auto out=out_.Get<T>();DataCopyPadExtParams<T> pad{false,0,0,0};
    while(pos<end){
      const uint64_t lane=pos%inner_,row=pos/inner_,a=row%axis_,outer=row/axis_;
      uint64_t n64=inner_-lane;if(n64>end-pos)n64=end-pos;if(n64>TILE)n64=TILE;
      const uint32_t n=n64;
      DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
      DataCopyPad(out,self_[pos],p,pad);PipeBarrier<PIPE_ALL>();
      int32_t j=list.GetValue(a);
      if(j>=0)Load((outer*count_+j)*inner_+lane,n);
      while(j>=0){
        auto src=inputQueue_.template DeQue<T>();
        const int32_t next=list.GetValue(axis_+j);
        if(next>=0)Load((outer*count_+next)*inner_+lane,n);
        Add(out,out,src,n);PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(src);j=next;
      }
      PipeBarrier<PIPE_ALL>();DataCopyPad(output_[pos],out,p);PipeBarrier<PIPE_ALL>();pos+=n;
    }
  }
 private:
  __aicore__ inline void Load(uint64_t offset,uint32_t n) {
    auto src=inputQueue_.template AllocTensor<T>();
    DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
    DataCopyPadExtParams<T> pad{false,0,0,0};
    DataCopyPad(src,source_[offset],p,pad);inputQueue_.EnQue(src);
  }
  static constexpr uint32_t TILE=4096,ALIGN=32/sizeof(T);
  TPipe pipe_;TQue<QuePosition::VECIN,2> inputQueue_;
  TBuf<TPosition::VECCALC> out_,list_,indexCache_;
  GlobalTensor<T> self_,source_,output_;GlobalTensor<int32_t> index_;
  uint64_t axis_,inner_,count_,total_;
};
