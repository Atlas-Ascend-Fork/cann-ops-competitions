#pragma once
// Ordered inverse-list gather for rows whose source fits in UB. No atomics.
template<class T> class IndexAddRowGather {
 public:
  template<class D> __aicore__ static inline bool Eligible(const D &d) {
    if(d.inner!=1 || !d.axisSize || d.axisSize>10000 || !d.indexCount || d.indexCount>8000)return false;
    uint32_t a=32/sizeof(T),b=d.axisSize;
    while(b){uint32_t r=a%b;a=b;b=r;}
    const uint32_t rows=(32/sizeof(T))/a;
    return d.outer>=static_cast<uint64_t>(rows)*GetBlockNum();
  }
  template<class D> __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR output,const D &d) {
    axis_=d.axisSize;count_=d.indexCount;outer_=d.outer;
    uint32_t a=ALIGN,b=axis_;while(b){uint32_t r=a%b;a=b;b=r;}groupRows_=ALIGN/a;
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(self));
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(source));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    headN_=(axis_+7)/8*8;
    uint32_t nextN=count_+1;if(nextN<TILE)nextN=TILE;nextN=(nextN+7)/8*8;
    uint32_t srcN=count_+ALIGN;if(srcN<TILE)srcN=TILE;srcN=(srcN+ALIGN-1)/ALIGN*ALIGN;
    pipe_.InitBuffer(head_,headN_*4);pipe_.InitBuffer(depth_,headN_*4);
    pipe_.InitBuffer(next_,nextN*4);pipe_.InitBuffer(src_,srcN*sizeof(T));
    pipe_.InitBuffer(out_,TILE*sizeof(T));pipe_.InitBuffer(gathered_,TILE*sizeof(T));
    pipe_.InitBuffer(cur0_,TILE*4);pipe_.InitBuffer(cur1_,TILE*4);pipe_.InitBuffer(offset_,TILE*4);
    // This route requires its aligned source, index, list and temporary buffers
    // to fit within the available UB.
  }
  __aicore__ inline void Process() {
    auto head=head_.Get<uint32_t>();auto depth=depth_.Get<uint32_t>();auto next=next_.Get<uint32_t>();
    Duplicate(head,0U,headN_);Duplicate(depth,0U,headN_);PipeBarrier<PIPE_ALL>();
    next.SetValue(0,0);
    for(uint32_t j=count_;j;--j){
      const int32_t a=index_.GetValue(j-1);
      if(a<0 || static_cast<uint32_t>(a)>=axis_)return;
      next.SetValue(j,head.GetValue(a));head.SetValue(a,j*4);
      const uint32_t length=depth.GetValue(a)+1;depth.SetValue(a,length);
      if(length>maxDepth_[a/TILE])maxDepth_[a/TILE]=length;
    }
    PipeBarrier<PIPE_ALL>();
    const uint64_t groups=(outer_+groupRows_-1)/groupRows_;
    const uint64_t begin=groups*GetBlockIdx()/GetBlockNum()*groupRows_;
    uint64_t end=groups*(GetBlockIdx()+1)/GetBlockNum()*groupRows_;if(end>outer_)end=outer_;
    auto source=src_.Get<T>();auto out=out_.Get<T>();auto gathered=gathered_.Get<T>();
    auto offset=offset_.Get<uint32_t>();
    Duplicate(source,static_cast<T>(0),ALIGN);PipeBarrier<PIPE_ALL>();
    DataCopyPadExtParams<T> pad{false,0,0,0};
    for(uint64_t row=begin;row<end;++row){
      DataCopyExtParams sp{1,static_cast<uint32_t>(count_*sizeof(T)),0,0,0};
      DataCopyPad(source[ALIGN],source_[row*count_],sp,pad);PipeBarrier<PIPE_ALL>();
      for(uint32_t a=0;a<axis_;a+=TILE){
        uint32_t n=axis_-a;if(n>TILE)n=TILE;
        DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
        DataCopyPad(out,self_[row*axis_+a],p,pad);
        DataCopy(cur0_.Get<uint32_t>(),head[a],(n+7)/8*8);PipeBarrier<PIPE_ALL>();
        for(uint32_t level=0;level<maxDepth_[a/TILE];++level){
          auto cur=(level&1)?cur1_.Get<uint32_t>():cur0_.Get<uint32_t>();
          auto dst=(level&1)?cur0_.Get<uint32_t>():cur1_.Get<uint32_t>();
          if constexpr(sizeof(T)==2){ShiftRight(offset,cur,1U,n);PipeBarrier<PIPE_V>();Adds(offset.template ReinterpretCast<int32_t>(),offset.template ReinterpretCast<int32_t>(),30,n);}
          else Adds(offset.template ReinterpretCast<int32_t>(),cur.template ReinterpretCast<int32_t>(),28,n);
          PipeBarrier<PIPE_V>();
          Gather(gathered,source,offset,0U,n);
          Gather(dst,next,cur,0U,n);PipeBarrier<PIPE_V>();
          Add(out,out,gathered,n);PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();DataCopyPad(output_[row*axis_+a],out,p);PipeBarrier<PIPE_ALL>();
      }
    }
  }
 private:
  static constexpr uint32_t TILE=2048,ALIGN=32/sizeof(T);
  TPipe pipe_;TBuf<TPosition::VECCALC> head_,depth_,next_,src_,out_,gathered_,cur0_,cur1_,offset_;
  GlobalTensor<T> self_,source_,output_;GlobalTensor<int32_t> index_;
  uint32_t axis_,count_,headN_,groupRows_,maxDepth_[5]={};uint64_t outer_;
};
