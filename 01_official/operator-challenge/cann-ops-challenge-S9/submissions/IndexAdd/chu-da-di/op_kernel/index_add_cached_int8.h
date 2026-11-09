// Cache a complete INT8 source slab and reuse its widened representation.
class KernelIndexAddCachedInt8Rows {
public:
 __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR out,const IndexAddTilingData* t,TPipe* p){
  t_=t;p_=p;self_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(self),t->selfElements);src_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(source),t->sourceElements);out_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out),t->selfElements);idx_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),t->indexCount);
  p->InitBuffer(indexBuf_,t->indexBufferBytes);p->InitBuffer(rawBuf_,t->rawBufferBytes);p->InitBuffer(bridgeBuf_,t->castBufferBytes);p->InitBuffer(sourceBuf_,t->sourceCacheBufferBytes);p->InitBuffer(accBuf_,t->accumulatorBufferBytes);
 }
 __aicore__ inline void Process(){
  const uint32_t core=GetBlockIdx(),cores=t_->usedCoreNum,total=t_->outer*t_->destinationAxis,per=total/cores,extra=total%cores;
  uint32_t first=core*per+(core<extra?core:extra),end=first+per+(core<extra?1:0);
  const uint32_t M=t_->indexCount,N=t_->indexVectorElements,I=t_->inner,D=t_->destinationAxis;
  auto ix=indexBuf_.Get<int32_t>();auto raw=rawBuf_.Get<int8_t>();auto bridge=bridgeBuf_.Get<half>();auto source=sourceBuf_.Get<int16_t>();auto acc=accBuf_.Get<int16_t>();
  Duplicate(ix,int32_t(-1),N);Sync<HardEvent::V_MTE2>();DataCopyExtParams ip{1,M*4,0,0,0};DataCopyPadExtParams<int32_t> pi{false,0,0,0};DataCopyPad(ix,idx_,ip,pi);
  while(first<end){uint32_t outer=t_->outer==1?0:first/D,lo=first-outer*D,rows=D-lo;if(rows>end-first)rows=end-first;
   DataCopyExtParams sp{1,M*I,0,0,0};DataCopyPadExtParams<int8_t> px{false,0,0,0};DataCopyPad(raw,src_[outer*M*I],sp,px);Sync<HardEvent::MTE2_V>();Cast(bridge,raw,RoundMode::CAST_NONE,M*I);PipeBarrier<PIPE_V>();Cast(source,bridge,RoundMode::CAST_RINT,M*I);Sync<HardEvent::V_MTE2>();
   const uint32_t n=rows*I;DataCopyExtParams dp{1,n,0,0,0};DataCopyPad(raw,self_[first*I],dp,px);Sync<HardEvent::MTE2_V>();Cast(bridge,raw,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();Cast(acc,bridge,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();
   auto fl=bridgeBuf_.Get<float>();auto low=bridgeBuf_.Get<uint8_t>()[N*4],high=low[N/4];auto expanded=bridgeBuf_.Get<uint32_t>()[N+N/8],positions=expanded[N/8],selected=positions[N];
   Cast(fl,ix,RoundMode::CAST_RINT,N);PipeBarrier<PIPE_V>();CompareScalar(low,fl,float(int32_t(lo)),CMPMODE::GE,N);CompareScalar(high,fl,float(int32_t(lo+rows)),CMPMODE::LT,N);CreateVecIndex(positions.template ReinterpretCast<int32_t>(),int32_t(0),N);PipeBarrier<PIPE_V>();And(low.template ReinterpretCast<uint16_t>(),low.template ReinterpretCast<uint16_t>(),high.template ReinterpretCast<uint16_t>(),N/16);Sync<HardEvent::V_S>();auto words=low.template ReinterpretCast<uint32_t>();for(uint32_t r=0;r<N/64;++r){uint32_t a=words.GetValue(r*2),b=words.GetValue(r*2+1);for(uint32_t k=0;k<4;++k){expanded.SetValue(r*8+2*k,a);expanded.SetValue(r*8+2*k+1,b);}}Sync<HardEvent::S_V>();uint64_t kept=0;GatherMask(selected,positions,expanded,false,0,{1,uint16_t(N/64),8,1},kept);Sync<HardEvent::V_S>();
   for(uint32_t k=0;k<uint32_t(kept);++k){uint32_t m=selected.GetValue(k);if(m>=M)continue;uint32_t d=uint32_t(ix.GetValue(m));Add(acc[(d-lo)*I],acc[(d-lo)*I],source[m*I],I);PipeBarrier<PIPE_V>();}
   ShiftLeft(acc,acc,int16_t(8),n);PipeBarrier<PIPE_V>();ShiftRight(acc,acc,int16_t(8),n);PipeBarrier<PIPE_V>();Cast(bridge,acc,RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();Cast(raw,bridge,RoundMode::CAST_NONE,n);Sync<HardEvent::V_MTE3>();DataCopyPad(out_[first*I],raw,dp);Sync<HardEvent::MTE3_V>();Sync<HardEvent::MTE3_MTE2>();first+=rows;
  }
 }
private:
 template<HardEvent E>__aicore__ inline void Sync(){auto e=static_cast<event_t>(p_->FetchEventID(E));SetFlag<E>(e);WaitFlag<E>(e);}
 const IndexAddTilingData* t_;TPipe* p_;GlobalTensor<int8_t> self_,src_,out_;GlobalTensor<int32_t> idx_;
 TBuf<QuePosition::VECCALC> indexBuf_,rawBuf_,bridgeBuf_,sourceBuf_,accBuf_;
};
