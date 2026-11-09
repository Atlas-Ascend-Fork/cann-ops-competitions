#pragma once
// Process several complete rows per vector instruction. List order is unchanged.
template<class T> class IndexAddRowBatch {
 public:
  template<class D> __aicore__ static inline bool Eligible(const D &d) {
    return d.inner==1 && d.axisSize>0 && d.axisSize<=1024 && d.indexCount>0 &&
      d.indexCount<=8000 && IndexAddRowGather<T>::Eligible(d);
  }
  template<class D> __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR output,const D &d) {
    axis_=d.axisSize;count_=d.indexCount;outer_=d.outer;
    axisPad_=(axis_+ALIGN-1)/ALIGN*ALIGN;
    srcStride_=(count_+ALIGN-1)/ALIGN*ALIGN+ALIGN;
    batch_=TILE/axisPad_;if(batch_>16)batch_=16;
    uint32_t nextN=0,srcBytes=0,n=0;
    for(;batch_;--batch_){
      n=batch_*axisPad_;
      nextN=(count_+8)/8*8;if(nextN<n)nextN=n;
      srcBytes=batch_*srcStride_*sizeof(T);
      if(srcBytes<n*sizeof(T))srcBytes=n*sizeof(T);
      if(srcBytes<count_*4)srcBytes=count_*4;
      srcBytes=(srcBytes+31)/32*32;
      const uint32_t bytes=axisPad_*4+nextN*4+srcBytes+n*(2*sizeof(T)+16);
      if(bytes<=190000)break;
    }
    uint32_t a=ALIGN,b=axis_;while(b){uint32_t r=a%b;a=b;b=r;}groupRows_=ALIGN/a;
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(self));
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(source));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));
    index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    pipe_.InitBuffer(head_,axisPad_*4);
    pipe_.InitBuffer(next_,nextN*4);pipe_.InitBuffer(src_,srcBytes);
    pipe_.InitBuffer(out_,n*sizeof(T));pipe_.InitBuffer(gathered_,n*sizeof(T));
    pipe_.InitBuffer(cur0_,n*4);pipe_.InitBuffer(cur1_,n*4);
    pipe_.InitBuffer(offset_,n*4);pipe_.InitBuffer(rowBase_,n*4);
    // Actual aligned allocations determine batch size. The source buffer holds
    // indices only during list construction, before any row source is loaded.
  }
  __aicore__ inline void Process() {
    auto head=head_.Get<uint32_t>();auto next=next_.Get<uint32_t>();
    Duplicate(head,0U,axisPad_);PipeBarrier<PIPE_ALL>();
    DataCopyExtParams ip{1,static_cast<uint32_t>(count_*4),0,0,0};
    DataCopyPadExtParams<int32_t> ipPad{false,0,0,0};
    DataCopyPad(src_.Get<int32_t>(),index_,ip,ipPad);PipeBarrier<PIPE_ALL>();
    next.SetValue(0,0);
    uint32_t maxDepth=0;
    for(uint32_t j=count_;j;--j){
      const int32_t a=src_.Get<int32_t>().GetValue(j-1);if(a<0 || static_cast<uint32_t>(a)>=axis_)return;
      // M<=8000: source byte indices fit in 15 bits; depth fits in 13.
      const uint32_t packed=head.GetValue(a),n=(packed>>15)+1;
      next.SetValue(j,packed&32767U);head.SetValue(a,(n<<15)|(j*4));
      if(n>maxDepth)maxDepth=n;
    }
    PipeBarrier<PIPE_ALL>();
    ShiftLeft(head,head,17U,axisPad_);PipeBarrier<PIPE_V>();
    ShiftRight(head,head,17U,axisPad_);PipeBarrier<PIPE_ALL>();
    auto src=src_.Get<T>();auto out=out_.Get<T>();auto gathered=gathered_.Get<T>();
    auto offset=offset_.Get<uint32_t>();auto rowBase=rowBase_.Get<int32_t>();
    PipeBarrier<PIPE_ALL>();
    Duplicate(src,static_cast<T>(0),batch_*srcStride_);
    for(uint32_t r=0;r<batch_;++r)
      Duplicate(rowBase[r*axisPad_],static_cast<int32_t>(r*srcStride_*sizeof(T)+32-sizeof(T)),axisPad_);
    PipeBarrier<PIPE_ALL>();
    const uint64_t groups=(outer_+groupRows_-1)/groupRows_;
    const uint64_t begin=groups*GetBlockIdx()/GetBlockNum()*groupRows_;
    uint64_t end=groups*(GetBlockIdx()+1)/GetBlockNum()*groupRows_;if(end>outer_)end=outer_;
    DataCopyPadExtParams<T> pad{false,0,0,0};
    DataCopyExtParams sp{1,static_cast<uint32_t>(count_*sizeof(T)),0,0,0};
    DataCopyExtParams op{1,static_cast<uint32_t>(axis_*sizeof(T)),0,0,0};
    for(uint64_t row=begin;row<end;){
      uint32_t rows=end-row>batch_?batch_:static_cast<uint32_t>(end-row);const uint32_t n=rows*axisPad_;
      for(uint32_t r=0;r<rows;++r){
        DataCopyPad(src[r*srcStride_+ALIGN],source_[(row+r)*count_],sp,pad);
        DataCopyPad(out[r*axisPad_],self_[(row+r)*axis_],op,pad);
        DataCopy(cur0_.Get<uint32_t>()[r*axisPad_],head,axisPad_);
      }
      PipeBarrier<PIPE_ALL>();
      for(uint32_t level=0;level<maxDepth;++level){
        auto cur=(level&1)?cur1_.Get<uint32_t>():cur0_.Get<uint32_t>();
        auto dst=(level&1)?cur0_.Get<uint32_t>():cur1_.Get<uint32_t>();
        if constexpr(sizeof(T)==2){
          ShiftRight(offset,cur,1U,n);PipeBarrier<PIPE_V>();
          Add(offset.template ReinterpretCast<int32_t>(),offset.template ReinterpretCast<int32_t>(),rowBase,n);
        }else Add(offset.template ReinterpretCast<int32_t>(),cur.template ReinterpretCast<int32_t>(),rowBase,n);
        PipeBarrier<PIPE_V>();Gather(gathered,src,offset,0U,n);Gather(dst,next,cur,0U,n);
        PipeBarrier<PIPE_V>();Add(out,out,gathered,n);PipeBarrier<PIPE_V>();
      }
      PipeBarrier<PIPE_ALL>();
      for(uint32_t r=0;r<rows;++r)DataCopyPad(output_[(row+r)*axis_],out[r*axisPad_],op);
      PipeBarrier<PIPE_ALL>();row+=rows;
    }
  }
 private:
  static constexpr uint32_t TILE=4096,ALIGN=32/sizeof(T);
  TPipe pipe_;TBuf<TPosition::VECCALC> head_,next_,src_,out_,gathered_,cur0_,cur1_,offset_,rowBase_;
  GlobalTensor<T> self_,source_,output_;GlobalTensor<int32_t> index_;
  uint32_t axis_,count_,axisPad_,srcStride_,batch_,groupRows_;uint64_t outer_;
};
