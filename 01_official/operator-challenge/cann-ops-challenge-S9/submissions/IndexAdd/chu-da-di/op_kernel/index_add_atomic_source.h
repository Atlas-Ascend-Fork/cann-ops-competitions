// Source-owned contiguous DMA with atomic output accumulation.
// Initialization and updates are separated by an AIV-wide barrier.
template<typename T> class KernelIndexAddAtomicSource {
public:
 __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,GM_ADDR out,const IndexAddTilingData* t,TPipe* p){
  t_=t;p_=p;self_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(self),t->selfElements);
  src_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(source),t->sourceElements);
  out_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out),t->selfElements);
  index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(index),t->indexCount);
  p_->InitBuffer(indexBuffer_,t->indexBufferBytes);indices_=indexBuffer_.Get<int32_t>();
  p_->InitBuffer(queue_,2,t->csrSourceSlotBytes);
 }
 __aicore__ inline void Process(){
  if constexpr(IsSame<T,int8_t>::value || IsSame<T,int32_t>::value){
   SetAtomicNone();
   const uint64_t unit=32/sizeof(T),blocks=(t_->selfElements+unit-1)/unit;
   const uint64_t per=blocks/t_->usedCoreNum,extra=blocks%t_->usedCoreNum,core=GetBlockIdx();
   uint64_t pos=(core*per+(core<extra?core:extra))*unit;
   uint64_t end=pos+(per+(core<extra?1:0))*unit;if(end>t_->selfElements)end=t_->selfElements;
   const uint32_t cap=t_->csrSourceSlotBytes/sizeof(T);
   while(pos<end){
    const uint32_t n=end-pos<cap?end-pos:cap;auto x=queue_.AllocTensor<T>();
    DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
    DataCopyPad(x,self_[pos],p,pad);queue_.EnQue(x);x=queue_.DeQue<T>();
    Sync<HardEvent::V_MTE3>();DataCopyPad(out_[pos],x,p);Sync<HardEvent::MTE3_V>();queue_.FreeTensor(x);pos+=n;
   }
   DataCopyExtParams ip{1,static_cast<uint32_t>(t_->indexCount*4),0,0,0};
   DataCopyPadExtParams<int32_t> ipad{true,0,static_cast<uint8_t>(AlignUp(t_->indexCount,8)-t_->indexCount),0};
   DataCopyPad(indices_,index_,ip,ipad);Sync<HardEvent::MTE2_S>();
   SyncAll<true>();
   const uint64_t chunks=(t_->indexCount+t_->csrBatchSize-1)/t_->csrBatchSize;
   const uint64_t tasks=t_->outer*chunks*t_->innerTileCount;
   for(uint64_t task=core;task<tasks;task+=t_->usedCoreNum){
    const uint64_t tile=task%t_->innerTileCount;const uint64_t rowtask=task/t_->innerTileCount;
    const uint64_t firstM=(rowtask%chunks)*t_->csrBatchSize;const uint64_t outer=rowtask/chunks;
    const uint64_t firstI=tile*t_->tileInner;
    const uint32_t n=t_->inner-firstI<t_->tileInner?t_->inner-firstI:t_->tileInner;
    const uint32_t rows=t_->indexCount-firstM<t_->csrBatchSize?t_->indexCount-firstM:t_->csrBatchSize;
    auto x=queue_.AllocTensor<T>();
    DataCopyExtParams p{static_cast<uint16_t>(rows),static_cast<uint32_t>(n*sizeof(T)),static_cast<uint32_t>((t_->inner-n)*sizeof(T)),0,0};
    DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
    DataCopyPad(x,src_[(outer*t_->indexCount+firstM)*t_->inner+firstI],p,pad);queue_.EnQue(x);x=queue_.DeQue<T>();
    Sync<HardEvent::V_MTE3>();SetAtomicAdd<T>();
    const DataCopyExtParams op{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
    for(uint32_t j=0;j<rows;++j){
     const uint32_t d=static_cast<uint32_t>(indices_.GetValue(firstM+j));
     if(d<t_->destinationAxis)DataCopyPad(out_[(outer*t_->destinationAxis+d)*t_->inner+firstI],x[j*n],op);
    }
    SetAtomicNone();Sync<HardEvent::MTE3_V>();queue_.FreeTensor(x);
   }
   PipeBarrier<PIPE_ALL>();
  }
 }
private:
 template<HardEvent E> __aicore__ inline void Sync(){auto e=static_cast<event_t>(p_->FetchEventID(E));SetFlag<E>(e);WaitFlag<E>(e);}
 const IndexAddTilingData* t_;TPipe* p_;
 GlobalTensor<T> self_,src_,out_;GlobalTensor<int32_t> index_;
 TBuf<QuePosition::VECCALC> indexBuffer_;TQue<QuePosition::VECIN,1> queue_;LocalTensor<int32_t> indices_;
};
