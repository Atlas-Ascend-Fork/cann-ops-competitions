#pragma once
// Small int8 source slabs stay in UB. Two independent bytes are added per
// uint16 lane; remove the carry from the low byte before accepting the sum.
class IndexAddInt8Resident {
 public:
  template<class D> __aicore__ static inline bool Eligible(const D &d){
    if(d.inner<32 || d.inner>4096 || d.inner%32 || !d.total || !d.axisSize || d.axisSize>2000 || !d.indexCount || d.indexCount>8000)return false;
    if(d.indexCount>128000/d.inner)return false;
    return d.indexCount*d.inner+(d.indexCount+7)/8*32+16384+4*d.inner<=190000;
  }
  template<class D> __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR output,const D &d){
    inner_=d.inner;axis_=d.axisSize;count_=d.indexCount;outer_=d.outer;
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(self));
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(source));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(output));
    index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    pipe_.InitBuffer(sourceBuf_,count_*inner_);
    pipe_.InitBuffer(indexBuf_,(count_+7)/8*32);
    pipe_.InitBuffer(outBuf_,16384);pipe_.InitBuffer(aBuf_,inner_);pipe_.InitBuffer(bBuf_,inner_);
    pipe_.InitBuffer(loBuf_,inner_);pipe_.InitBuffer(carryBuf_,inner_);
  }
  __aicore__ inline void Process(){
    auto index=indexBuf_.Get<int32_t>();
    DataCopyExtParams ip{1,static_cast<uint32_t>(count_*4),0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(index,index_,ip,pad);PipeBarrier<PIPE_ALL>();
    auto lo=loBuf_.Get<uint16_t>();auto carry=carryBuf_.Get<uint16_t>();
    Duplicate(lo,static_cast<uint16_t>(0x00ff),inner_/2);Duplicate(carry,static_cast<uint16_t>(0x0100),inner_/2);PipeBarrier<PIPE_ALL>();
    const uint64_t rows=outer_*axis_,first=rows*GetBlockIdx()/GetBlockNum(),last=rows*(GetBlockIdx()+1)/GetBlockNum();
    uint64_t loaded=outer_;auto source=sourceBuf_.Get<int8_t>();auto out=outBuf_.Get<int8_t>();
    auto a=aBuf_.Get<uint16_t>();auto b=bBuf_.Get<uint16_t>();
    for(uint64_t row=first;row<last;){
      const uint64_t outer=row/axis_;const uint32_t firstAxis=row%axis_;
      uint32_t nr=axis_-firstAxis;if(nr>last-row)nr=static_cast<uint32_t>(last-row);if(nr>16384/inner_)nr=16384/inner_;
      if(loaded!=outer){DataCopy(source,source_[outer*count_*inner_],static_cast<uint32_t>(count_*inner_));PipeBarrier<PIPE_ALL>();loaded=outer;}
      DataCopy(out,self_[row*inner_],nr*inner_);PipeBarrier<PIPE_ALL>();
      for(uint32_t j=0;j<count_;++j){
        const int32_t axis=index.GetValue(j);if(axis<0 || static_cast<uint32_t>(axis)>=axis_)return;
        if(static_cast<uint32_t>(axis)<firstAxis || static_cast<uint32_t>(axis)>=firstAxis+nr)continue;
        auto x=out[(static_cast<uint32_t>(axis)-firstAxis)*inner_].template ReinterpretCast<uint16_t>();
        auto y=source[j*inner_].template ReinterpretCast<uint16_t>();
        And(a,x,lo,inner_/2);And(b,y,lo,inner_/2);PipeBarrier<PIPE_V>();
        Add(a.template ReinterpretCast<int16_t>(),a.template ReinterpretCast<int16_t>(),b.template ReinterpretCast<int16_t>(),inner_/2);
        Add(x.template ReinterpretCast<int16_t>(),x.template ReinterpretCast<int16_t>(),y.template ReinterpretCast<int16_t>(),inner_/2);PipeBarrier<PIPE_V>();
        And(a,a,carry,inner_/2);PipeBarrier<PIPE_V>();
        Sub(x.template ReinterpretCast<int16_t>(),x.template ReinterpretCast<int16_t>(),a.template ReinterpretCast<int16_t>(),inner_/2);PipeBarrier<PIPE_V>();
      }
      PipeBarrier<PIPE_ALL>();DataCopy(output_[row*inner_],out,nr*inner_);PipeBarrier<PIPE_ALL>();row+=nr;
    }
  }
 private:
  TPipe pipe_;TBuf<TPosition::VECCALC> sourceBuf_,indexBuf_,outBuf_,aBuf_,bBuf_,loBuf_,carryBuf_;
  GlobalTensor<int8_t> self_,source_,output_;GlobalTensor<int32_t> index_;
  uint64_t outer_;uint32_t axis_,count_,inner_;
};
