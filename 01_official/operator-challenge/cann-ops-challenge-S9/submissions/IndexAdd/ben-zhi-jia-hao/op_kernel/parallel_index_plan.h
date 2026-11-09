#pragma once
// Construct ordered occurrence lists with vector compare/compaction, sharing
// work across cores. Then stream a depth-major index plan into row-batched adds.
template<class T> class IndexAddParallelPlan {
 public:
  template<class D> __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,
      GM_ADDR output,GM_ADDR workspace,const D &d){
    axis_=d.axisSize;count_=d.indexCount;outer_=d.outer;
    axisPad_=(axis_+ALIGN-1)/ALIGN*ALIGN;countPad_=(count_+63)/64*64;
    listStride_=(count_+15)/8*8;
    coreBase_=static_cast<uint64_t>(axis_)*listStride_;
    denseBase_=coreBase_+GetBlockNum()*8;
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(self));source_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(source));
    if constexpr(E==4 || E==2) source_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(output));index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    work_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(workspace));atomicWork_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace));pipe_.InitBuffer(arena_,190016);
  }
  __aicore__ inline void Process(){
    SetAtomicNone();if(TryPhaseAlias8Plan())return;if(TryLocalPlan())return;BuildLists();PipeBarrier<PIPE_ALL>();SyncAll<true>();
    auto data=At<uint32_t>(0);DataCopy(data,work_[coreBase_],GetBlockNum()*8);PipeBarrier<PIPE_ALL>();
    depth_=0;for(uint32_t c=0;c<GetBlockNum();++c){uint32_t n=data.GetValue(c*8);if(n>depth_)depth_=n;}
    const uint64_t size=static_cast<uint64_t>(depth_)*axisPad_,blocks=size/8;
    const uint64_t first=blocks*GetBlockIdx()/GetBlockNum()*8,last=blocks*(GetBlockIdx()+1)/GetBlockNum()*8;
    Duplicate(data,0U,4096);PipeBarrier<PIPE_ALL>();
    for(uint64_t p=first;p<last;){uint32_t n=last-p>4096?4096:static_cast<uint32_t>(last-p);DataCopy(work_[denseBase_+p],data,n);p+=n;}
    PipeBarrier<PIPE_ALL>();SyncAll<true>();TransposeLists();PipeBarrier<PIPE_ALL>();SyncAll<true>();if(!ComputeHalfBlockPlan() && !ComputePipelinedPlan() && !ComputeCachedPlan())Compute();
  }
 private:
  static constexpr uint32_t E=sizeof(T),ALIGN=32/E;
  template<class U> __aicore__ inline LocalTensor<U> At(uint32_t offset){return arena_.Get<uint8_t>()[offset].template ReinterpretCast<U>();}
  __aicore__ inline void Compact8Forward(LocalTensor<float> dst,LocalTensor<float> src,uint32_t stride,uint32_t cols){
    uint64_t sa[16],da[16];const uint32_t halfCols=cols/2;
    for(uint32_t k=0;k<16;++k){
      sa[k]=reinterpret_cast<uint64_t>(src[(k%8)*stride+(k/8)*halfCols].GetPhyAddr());
      da[k]=reinterpret_cast<uint64_t>(dst[((k%2)*halfCols+k/2)*8].GetPhyAddr());
    }
    TransDataTo5HDParams p{false,false,static_cast<uint8_t>(cols/16),8,1};
    if(cols==16){p.dstRepStride=0;p.srcRepStride=0;}
    TransDataTo5HD<float>(da,sa,p);
  }
  __aicore__ inline void Compact8Back(LocalTensor<float> dst,LocalTensor<float> src,uint32_t cols){
    uint64_t sa[16],da[16];
    for(uint32_t k=0;k<16;++k){
      sa[k]=reinterpret_cast<uint64_t>(src[k*8].GetPhyAddr());
      da[k]=reinterpret_cast<uint64_t>(dst[(k/2)*cols+(k%2)*8].GetPhyAddr());
    }
    TransDataTo5HDParams p{false,false,static_cast<uint8_t>(cols/16),2,16};
    if(cols==16){p.dstRepStride=0;p.srcRepStride=0;}
    TransDataTo5HD<float>(da,sa,p);
  }
  __aicore__ inline bool TryPhaseAlias8Plan(){
    if constexpr(E!=4){return false;}else{
    const uint32_t ap=(axis_+15)/16*16;if(ap>1008)return false;
    // During accumulation raw is dead. Its low region holds Gather output;
    // its high region holds the plan. The accumulator remains separate.
    constexpr uint32_t RAW_PAD=8;
    uint32_t chunkLimit=(190000-ap*32-32-8*RAW_PAD*4)/64/16*16;
    if(chunkLimit>2560)chunkLimit=2560;
    if(!chunkLimit)return false;
    const uint32_t chunk=count_>=chunkLimit?chunkLimit:(count_+15)/16*16;
    const uint32_t chunks=(count_+chunk-1)/chunk,headers=(chunks+7)/8*8;
    const uint64_t dense=static_cast<uint64_t>(GetBlockNum())*headers;
    const uint64_t needed=dense+static_cast<uint64_t>(chunks)*chunk*ap;
    const uint64_t oldWords=static_cast<uint64_t>(axis_)*listStride_+GetBlockNum()*8+static_cast<uint64_t>(count_)*axisPad_;
    if(chunks>16 || needed>oldWords)return false;
    const uint32_t inputStride=chunk+RAW_PAD;
    const uint32_t computeBytes=8*inputStride*4+(chunk+1)*8*4+ap*8*4;
    if(computeBytes>190000 || ap*8*4>8*chunk*4)return false;
    const uint32_t groups=ap/8,totalGroups=groups*chunks;
    const uint32_t begin=totalGroups*GetBlockIdx()/GetBlockNum(),end=totalGroups*(GetBlockIdx()+1)/GetBlockNum();
    const uint32_t maxKeys=((totalGroups+GetBlockNum()-1)/GetBlockNum())*8;
    const uint32_t cp=(chunk+63)/64*64,maskBytes=(cp/8+31)/32*32;
    const uint32_t keepStart=cp*12+maskBytes,keepWords=(count_+maxKeys*8+7)/8*8;
    const uint32_t metaStart=keepStart+keepWords*4;
    if(metaStart+maxKeys*8+headers*4>190000 || keepStart<GetBlockNum()*headers*4)return false;
    auto values=At<float>(0);auto positions=At<uint32_t>(cp*4);auto packed=At<uint32_t>(cp*8);
    auto mask=At<uint8_t>(cp*12);auto kept=At<uint32_t>(keepStart);
    auto starts=At<uint32_t>(metaStart);auto lengths=At<uint32_t>(metaStart+maxKeys*4);
    auto localDepth=At<uint32_t>(metaStart+maxKeys*8);
    Duplicate(localDepth,0U,headers);PipeBarrier<PIPE_ALL>();
    uint32_t cursor=0,loaded=chunks;
    DataCopyPadExtParams<int32_t> ipPad{false,0,0,0};
    for(uint32_t g=begin;g<end;++g){
      const uint32_t ch=g/groups,first=ch*chunk,n=count_-first>chunk?chunk:count_-first;
      if(loaded!=ch){
        Duplicate(values,-1.0f,cp);PipeBarrier<PIPE_ALL>();
        DataCopyExtParams ip{1,n*4,0,0,0};DataCopyPad(positions.template ReinterpretCast<int32_t>(),index_[first],ip,ipPad);PipeBarrier<PIPE_ALL>();
        Cast(values,positions.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,n);PipeBarrier<PIPE_V>();
        CreateVecIndex(positions.template ReinterpretCast<int32_t>(),static_cast<int32_t>(1),cp);PipeBarrier<PIPE_V>();
        ShiftLeft(positions,positions,5U,cp);PipeBarrier<PIPE_ALL>();loaded=ch;
      }
      for(uint32_t c=0;c<8;++c){
        const uint32_t axis=(g%groups)*8+c,key=(g-begin)*8+c;
        Duplicate(packed,0U,cp);CompareScalar(mask,values,static_cast<float>(static_cast<int32_t>(axis)),CMPMODE::EQ,cp);PipeBarrier<PIPE_V>();
        uint64_t retained=0;GatherMask(packed,positions,mask.template ReinterpretCast<uint32_t>(),true,n,{1,1,0,0},retained);PipeBarrier<PIPE_ALL>();
        const uint32_t len=static_cast<uint32_t>(retained),pad=(len+7)/8*8;
        starts.SetValue(key,cursor);lengths.SetValue(key,len);
        if(len>localDepth.GetValue(ch))localDepth.SetValue(ch,len);
        if(pad)DataCopy(kept[cursor],packed,pad);PipeBarrier<PIPE_ALL>();cursor+=pad;
      }
    }
    PipeBarrier<PIPE_ALL>();DataCopy(work_[GetBlockIdx()*headers],localDepth,headers);PipeBarrier<PIPE_ALL>();SyncAll<true>();
    auto allDepth=At<uint32_t>(0);DataCopy(allDepth,work_,GetBlockNum()*headers);PipeBarrier<PIPE_ALL>();
    uint32_t depths[16];bool admitted=true;
    for(uint32_t ch=0;ch<chunks;++ch){
      uint32_t depth=0;for(uint32_t core=0;core<GetBlockNum();++core){const uint32_t n=allDepth.GetValue(core*headers+ch);if(n>depth)depth=n;}
      depths[ch]=depth;if(((depth+7)/8*8)*128>keepStart)admitted=false;
      if(ap*8*4+static_cast<uint64_t>(depth)*ap*4>8*chunk*4)admitted=false;
    }
    if(!admitted)return false;
    for(uint32_t g=begin;g<end;++g){
      const uint32_t ch=g/groups,depth=depths[ch];if(!depth)continue;
      const uint32_t dp=(depth+7)/8*8,matrix=dp*8;
      auto tmp=At<uint32_t>(0);auto grid=At<uint32_t>(matrix*4);auto offsets=At<uint32_t>(matrix*8);auto levels=At<uint32_t>(matrix*12);
      Duplicate(tmp,0U,matrix);PipeBarrier<PIPE_ALL>();
      for(uint32_t c=0;c<8;++c){const uint32_t key=(g-begin)*8+c,len=lengths.GetValue(key);if(len)DataCopy(tmp[c*dp],kept[starts.GetValue(key)],(len+7)/8*8);}
      PipeBarrier<PIPE_ALL>();CreateVecIndex(offsets.template ReinterpretCast<int32_t>(),static_cast<int32_t>(0),matrix);PipeBarrier<PIPE_V>();
      ShiftRight(levels,offsets,3U,matrix);ShiftLeft(offsets,offsets,29U,matrix);PipeBarrier<PIPE_V>();
      ShiftRight(offsets,offsets,29U,matrix);ShiftLeft(levels,levels,2U,matrix);PipeBarrier<PIPE_V>();
      Muls(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),static_cast<int32_t>(dp*4),matrix);PipeBarrier<PIPE_V>();
      Add(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),levels.template ReinterpretCast<int32_t>(),matrix);PipeBarrier<PIPE_V>();
      Gather(grid,tmp,offsets,0U,matrix);PipeBarrier<PIPE_ALL>();
      DataCopyExtParams out{static_cast<uint16_t>(depth),32,0,(ap-8)*4,0};
      DataCopyPad(work_[dense+static_cast<uint64_t>(ch)*chunk*ap+(g%groups)*8],grid,out);PipeBarrier<PIPE_ALL>();
    }
    SyncAll<true>();
    uint32_t off=0;auto raw=At<float>(off);off+=8*inputStride*4;
    auto transposed=At<float>(off);off+=(chunk+1)*8*4;
    auto acc=At<float>(off);off+=ap*8*4;
    auto gathered=At<float>(0);
    auto resident=At<uint32_t>(ap*8*4);
    const uint64_t tiles=(outer_+7)/8,firstTile=tiles*GetBlockIdx()/GetBlockNum(),lastTile=tiles*(GetBlockIdx()+1)/GetBlockNum();
    DataCopyPadExtParams<float> pad{false,0,0,0};GatherRepeatParams gp{1,8};
    for(uint64_t tile=firstTile;tile<lastTile;++tile){
      const uint64_t row=tile*8;const uint32_t nr=outer_-row>8?8:static_cast<uint32_t>(outer_-row);
      if(nr!=8){Duplicate(gathered,0.0f,ap*8);PipeBarrier<PIPE_ALL>();}
      DataCopyExtParams selfCopy{static_cast<uint16_t>(nr),axis_*4,0,(ap-axisPad_)/8,0};
      DataCopyPad(gathered,self_[row*axis_],selfCopy,pad);PipeBarrier<PIPE_ALL>();
      Compact8Forward(acc,gathered,ap,ap);PipeBarrier<PIPE_ALL>();
      for(uint32_t ch=0;ch<chunks;++ch){
        const uint32_t depth=depths[ch];if(!depth)continue;
        const uint32_t j=ch*chunk,n=count_-j>chunk?chunk:count_-j,np=(n+7)/8*8;
        if(n!=chunk || nr!=8){Duplicate(raw,0.0f,8*inputStride);PipeBarrier<PIPE_ALL>();}
        DataCopyExtParams cp{static_cast<uint16_t>(nr),n*4,(count_-n)*4,(inputStride-np)/8,0};
        DataCopyPad(raw,source_[row*count_+j],cp,pad);PipeBarrier<PIPE_ALL>();
        Duplicate(transposed,0.0f,8);PipeBarrier<PIPE_V>();Compact8Forward(transposed[8],raw,inputStride,chunk);PipeBarrier<PIPE_ALL>();
        // The transpose finished reading raw. Plan and Gather output occupy
        // disjoint subregions until all additions finish before the next DMA.
        DataCopy(resident,work_[dense+static_cast<uint64_t>(ch)*chunk*ap],depth*ap);PipeBarrier<PIPE_ALL>();
        for(uint32_t level=0;level<depth;++level){
          Gatherb<uint32_t>(gathered.template ReinterpretCast<uint32_t>(),transposed.template ReinterpretCast<uint32_t>(),resident[level*ap],static_cast<uint8_t>(ap/8),gp);PipeBarrier<PIPE_V>();
          Add(acc,acc,gathered,ap*8);PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_ALL>();
      }
      Compact8Back(gathered,acc,ap);PipeBarrier<PIPE_ALL>();
      DataCopyExtParams cp{static_cast<uint16_t>(nr),axis_*4,(ap-axisPad_)/8,0,0};
      DataCopyPad(output_[row*axis_],gathered,cp);PipeBarrier<PIPE_ALL>();
    }
    return true;
    }
  }
  __aicore__ inline bool TryLocalPlan(){
    const uint32_t groups=(axis_+7)/8,maxCols=((groups+GetBlockNum()-1)/GetBlockNum())*8;
    const uint32_t keepStart=countPad_*12+(countPad_/8+31)/32*32;
    const uint32_t metaStart=keepStart+(countPad_+maxCols*8)*4;
    if(keepStart<GetBlockNum()*32 || metaStart+maxCols*8>190000)return false;
    const uint32_t begin=groups*GetBlockIdx()/GetBlockNum()*8;
    uint32_t end=groups*(GetBlockIdx()+1)/GetBlockNum()*8;if(end>axis_)end=axis_;
    auto values=At<float>(0);auto positions=At<uint32_t>(countPad_*4);auto packed=At<uint32_t>(countPad_*8);
    auto mask=At<uint8_t>(countPad_*12);auto kept=At<uint32_t>(keepStart);
    auto starts=At<uint32_t>(metaStart);auto lengths=At<uint32_t>(metaStart+maxCols*4);
    Duplicate(values,static_cast<float>(static_cast<int32_t>(axis_+1)),countPad_);PipeBarrier<PIPE_ALL>();
    DataCopyExtParams ip{1,count_*4,0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(positions.template ReinterpretCast<int32_t>(),index_,ip,pad);PipeBarrier<PIPE_ALL>();
    Cast(values,positions.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count_);PipeBarrier<PIPE_V>();
    CreateVecIndex(positions.template ReinterpretCast<int32_t>(),static_cast<int32_t>(1),countPad_);PipeBarrier<PIPE_V>();
    Muls(positions.template ReinterpretCast<int32_t>(),positions.template ReinterpretCast<int32_t>(),static_cast<int32_t>(4),countPad_);PipeBarrier<PIPE_ALL>();
    uint32_t cursor=0,localDepth=0;
    for(uint32_t a=begin;a<end;++a){
      Duplicate(packed,0U,countPad_);CompareScalar(mask,values,static_cast<float>(static_cast<int32_t>(a)),CMPMODE::EQ,countPad_);PipeBarrier<PIPE_V>();
      uint64_t retained=0;GatherMask(packed,positions,mask.template ReinterpretCast<uint32_t>(),true,count_,{1,1,0,0},retained);PipeBarrier<PIPE_ALL>();
      const uint32_t n=static_cast<uint32_t>(retained),np=(n+7)/8*8;
      starts.SetValue(a-begin,cursor);lengths.SetValue(a-begin,n);if(n>localDepth)localDepth=n;
      if(np)DataCopy(kept[cursor],packed,np);PipeBarrier<PIPE_ALL>();cursor+=np;
    }
    {PipeBarrier<PIPE_ALL>();if(begin<end){DataCopyExtParams cp{1,(end-begin)*4,0,0,0};DataCopyPad(work_[begin],lengths,cp);}PipeBarrier<PIPE_ALL>();}
    auto header=At<uint32_t>(0);Duplicate(header,0U,8);PipeBarrier<PIPE_ALL>();header.SetValue(0,localDepth);PipeBarrier<PIPE_ALL>();
    DataCopy(work_[coreBase_+GetBlockIdx()*8],header,8);PipeBarrier<PIPE_ALL>();SyncAll<true>();
    DataCopy(header,work_[coreBase_],GetBlockNum()*8);PipeBarrier<PIPE_ALL>();
    depth_=0;for(uint32_t c=0;c<GetBlockNum();++c){uint32_t n=header.GetValue(c*8);if(n>depth_)depth_=n;}
    uint32_t gridStride=8,shift=3;while(gridStride<maxCols){gridStride*=2;++shift;}
    const uint32_t depthPad=(depth_+7)/8*8;
    const uint64_t matrix64=static_cast<uint64_t>(depthPad)*gridStride;
    // The decision is identical on every core. Deep/skewed distributions use
    // the original complete general path instead of truncating the index plan.
    if(matrix64>16320 || matrix64*16>keepStart)return false;
    const uint32_t matrix=static_cast<uint32_t>(matrix64),zeroCount=keepStart/4>1024?1024:keepStart/4/8*8;
    Duplicate(header,0U,zeroCount);PipeBarrier<PIPE_ALL>();
    const uint64_t denseN=static_cast<uint64_t>(depth_)*axisPad_,blocks=denseN/8;
    const uint64_t first=blocks*GetBlockIdx()/GetBlockNum()*8,last=blocks*(GetBlockIdx()+1)/GetBlockNum()*8;
    for(uint64_t p=first;p<last;){uint32_t n=last-p>zeroCount?zeroCount:static_cast<uint32_t>(last-p);DataCopy(work_[denseBase_+p],header,n);p+=n;}
    PipeBarrier<PIPE_ALL>();SyncAll<true>();
    auto tmp=At<uint32_t>(0);auto grid=At<uint32_t>(matrix*4);auto offsets=At<uint32_t>(matrix*8);auto levels=At<uint32_t>(matrix*12);
    Duplicate(tmp,0U,matrix);PipeBarrier<PIPE_ALL>();
    for(uint32_t a=begin;a<end;++a){const uint32_t n=lengths.GetValue(a-begin);if(n)DataCopy(tmp[(a-begin)*depthPad],kept[starts.GetValue(a-begin)],(n+7)/8*8);}
    PipeBarrier<PIPE_ALL>();
    CreateVecIndex(offsets.template ReinterpretCast<int32_t>(),static_cast<int32_t>(0),matrix);PipeBarrier<PIPE_V>();
    ShiftRight(levels,offsets,shift,matrix);ShiftLeft(offsets,offsets,32-shift,matrix);PipeBarrier<PIPE_V>();
    ShiftRight(offsets,offsets,32-shift,matrix);ShiftLeft(levels,levels,2U,matrix);PipeBarrier<PIPE_V>();
    Muls(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),static_cast<int32_t>(depthPad*4),matrix);PipeBarrier<PIPE_V>();
    Add(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),levels.template ReinterpretCast<int32_t>(),matrix);PipeBarrier<PIPE_V>();
    Gather(grid,tmp,offsets,0U,matrix);PipeBarrier<PIPE_ALL>();
    if(begin<end){const uint32_t cols=(end-begin+7)/8*8;DataCopyExtParams cp{static_cast<uint16_t>(depth_),cols*4,(gridStride-cols)/8,(axisPad_-cols)*4,0};DataCopyPad(work_[denseBase_+begin],grid,cp);}
    PipeBarrier<PIPE_ALL>();SyncAll<true>();
    if(!ComputeHalfBlockPlan() && !ComputeSortedPlan() && !ComputePipelinedPlan() && !ComputeCachedPlan())Compute();return true;
  }

  __aicore__ inline void BuildLists(){
    auto values=At<float>(0);auto positions=At<uint32_t>(countPad_*4);auto packed=At<uint32_t>(countPad_*8);
    const uint32_t maskBytes=(countPad_/8+31)/32*32;
    auto mask=At<uint8_t>(countPad_*12);auto header=At<uint32_t>(countPad_*12+maskBytes);
    Duplicate(values,static_cast<float>(static_cast<int32_t>(axis_+1)),countPad_);PipeBarrier<PIPE_ALL>();
    DataCopyExtParams ip{1,count_*4,0,0,0};DataCopyPadExtParams<int32_t> pad{false,0,0,0};
    DataCopyPad(positions.template ReinterpretCast<int32_t>(),index_,ip,pad);PipeBarrier<PIPE_ALL>();
    Cast(values,positions.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,count_);PipeBarrier<PIPE_V>();
    CreateVecIndex(positions.template ReinterpretCast<int32_t>(),static_cast<int32_t>(1),countPad_);PipeBarrier<PIPE_V>();
    Muls(positions.template ReinterpretCast<int32_t>(),positions.template ReinterpretCast<int32_t>(),static_cast<int32_t>(4),countPad_);PipeBarrier<PIPE_ALL>();
    const uint32_t groups=(axis_+7)/8,begin=groups*GetBlockIdx()/GetBlockNum()*8;
    uint32_t end=groups*(GetBlockIdx()+1)/GetBlockNum()*8;if(end>axis_)end=axis_;
    uint32_t maxDepth=0;
    for(uint32_t a=begin;a<end;++a){
      CompareScalar(mask,values,static_cast<float>(static_cast<int32_t>(a)),CMPMODE::EQ,countPad_);PipeBarrier<PIPE_V>();
      uint64_t retained=0;
      GatherMask(packed,positions,mask.template ReinterpretCast<uint32_t>(),true,count_,{1,1,0,0},retained);PipeBarrier<PIPE_ALL>();
      const uint32_t n=static_cast<uint32_t>(retained);if(n>maxDepth)maxDepth=n;
      Duplicate(header,0U,8);PipeBarrier<PIPE_ALL>();header.SetValue(0,n);PipeBarrier<PIPE_ALL>();
      DataCopy(work_[static_cast<uint64_t>(a)*listStride_],header,8);
      if(n){DataCopyExtParams cp{1,n*4,0,0,0};DataCopyPad(work_[static_cast<uint64_t>(a)*listStride_+8],packed,cp);}
      PipeBarrier<PIPE_ALL>();
    }
    Duplicate(header,0U,8);PipeBarrier<PIPE_ALL>();header.SetValue(0,maxDepth);PipeBarrier<PIPE_ALL>();
    DataCopy(work_[coreBase_+GetBlockIdx()*8],header,8);
  }
  __aicore__ inline void TransposeLists(){
    auto packed=At<uint32_t>(0);auto offsets=At<uint32_t>(countPad_*4);auto expanded=At<uint32_t>(countPad_*4+16384);
    CreateVecIndex(offsets.template ReinterpretCast<int32_t>(),static_cast<int32_t>(0),4096);PipeBarrier<PIPE_V>();
    ShiftRight(offsets,offsets,3U,4096);PipeBarrier<PIPE_V>();ShiftLeft(offsets,offsets,2U,4096);PipeBarrier<PIPE_ALL>();
    const uint32_t groups=(axis_+7)/8,begin=groups*GetBlockIdx()/GetBlockNum()*8;
    uint32_t end=groups*(GetBlockIdx()+1)/GetBlockNum()*8;if(end>axis_)end=axis_;
    DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
    for(uint32_t a=begin;a<end;++a){
      const uint32_t n=work_.GetValue(static_cast<uint64_t>(a)*listStride_);if(!n)continue;
      DataCopyExtParams cp{1,n*4,0,0,0};DataCopyPad(packed,work_[static_cast<uint64_t>(a)*listStride_+8],cp,pad);PipeBarrier<PIPE_ALL>();
      for(uint32_t first=0;first<n;first+=512){
        uint32_t len=n-first>512?512:n-first;Gather(expanded,packed[first],offsets,0U,len*8);PipeBarrier<PIPE_ALL>();
        DataCopyExtParams out{static_cast<uint16_t>(len),4,0,axisPad_*4-4,0};
        DataCopyPad(work_[denseBase_+static_cast<uint64_t>(first)*axisPad_+a],expanded,out);PipeBarrier<PIPE_ALL>();
      }
    }
  }
  __aicore__ inline void HalfBlockTranspose(LocalTensor<half> dst,LocalTensor<half> src,uint32_t stride,uint32_t cols){
    uint64_t sa[16],da[16];
    for(uint32_t k=0;k<16;++k){sa[k]=reinterpret_cast<uint64_t>(src[k*stride].GetPhyAddr());da[k]=reinterpret_cast<uint64_t>(dst[k*16].GetPhyAddr());}
    TransDataTo5HDParams p{false,false,static_cast<uint8_t>(cols/16),static_cast<uint8_t>(cols==16?0:16),static_cast<uint8_t>(cols==16?0:1)};TransDataTo5HD<half>(da,sa,p);
  }
  __aicore__ inline void HalfBlockTransposeBack(LocalTensor<half> dst,LocalTensor<half> src,uint32_t stride,uint32_t cols){
    uint64_t sa[16],da[16];
    for(uint32_t k=0;k<16;++k){sa[k]=reinterpret_cast<uint64_t>(src[k*16].GetPhyAddr());da[k]=reinterpret_cast<uint64_t>(dst[k*stride].GetPhyAddr());}
    TransDataTo5HDParams p{false,false,static_cast<uint8_t>(cols/16),static_cast<uint8_t>(cols==16?0:1),static_cast<uint8_t>(cols==16?0:16)};TransDataTo5HD<half>(da,sa,p);
  }
  __aicore__ inline bool ComputeHalfBlockPlan(){
    if constexpr(E!=2){return false;}else{
      const uint32_t cp=(count_+15)/16*16,ap=axisPad_,rawCols=cp>ap?cp:ap;
      if(cp>4080 || ap>2040)return false;
      const uint64_t needed=static_cast<uint64_t>(rawCols)*32+(cp+1ULL)*32+2ULL*ap*32+static_cast<uint64_t>(depth_)*ap*4;
      if(needed>190000)return false;
      uint32_t off=0;auto raw=At<half>(off);off+=rawCols*32;
      auto src=At<half>(off);off+=(cp+1)*32;
      auto acc=At<half>(off);off+=ap*32;auto gathered=At<half>(off);off+=ap*32;
      auto plan=At<uint32_t>(off);
      if(depth_){DataCopy(plan,work_[denseBase_],depth_*ap);PipeBarrier<PIPE_ALL>();ShiftLeft(plan,plan,3U,depth_*ap);PipeBarrier<PIPE_V>();}
      Duplicate(src,static_cast<half>(0),16);PipeBarrier<PIPE_ALL>();
      const uint64_t tiles=(outer_+15)/16,begin=tiles*GetBlockIdx()/GetBlockNum(),end=tiles*(GetBlockIdx()+1)/GetBlockNum();
      DataCopyPadExtParams<half> pad{false,0,0,0};GatherRepeatParams gp{1,8};
      if(begin<end){
        const uint64_t row=begin*16;const uint32_t nr=outer_-row>16?16:static_cast<uint32_t>(outer_-row);
        if(nr!=16){Duplicate(raw,static_cast<half>(0),rawCols*16);PipeBarrier<PIPE_ALL>();}
        DataCopyExtParams initial{static_cast<uint16_t>((count_&15)==0?1:nr),((count_&15)==0?nr:1)*count_*2,0,0,0};
        DataCopyPad(raw,source_[row*count_],initial,pad);
      }
      for(uint64_t tile=begin;tile<end;++tile){
        const uint64_t row=tile*16;const uint32_t nr=outer_-row>16?16:static_cast<uint32_t>(outer_-row);
        if(nr!=16){Duplicate(gathered,static_cast<half>(0),ap*16);PipeBarrier<PIPE_ALL>();}
        DataCopyExtParams selfCopy{static_cast<uint16_t>(nr),axis_*2,0,0,0};
        DataCopyPad(gathered,self_[row*axis_],selfCopy,pad);PipeBarrier<PIPE_ALL>();
        HalfBlockTranspose(acc,gathered,ap,ap);PipeBarrier<PIPE_V>();
        HalfBlockTranspose(src[16],raw,cp,cp);PipeBarrier<PIPE_ALL>();
        // The completed source transpose releases raw. The current Adds read
        // src, so the next tile can fill raw concurrently without extra UB.
        if(tile+1<end){
          const uint64_t nextRow=(tile+1)*16;const uint32_t nextRows=outer_-nextRow>16?16:static_cast<uint32_t>(outer_-nextRow);
          if(nextRows!=16){Duplicate(raw,static_cast<half>(0),rawCols*16);PipeBarrier<PIPE_ALL>();}
          DataCopyExtParams nextCopy{static_cast<uint16_t>((count_&15)==0?1:nextRows),((count_&15)==0?nextRows:1)*count_*2,0,0,0};
          DataCopyPad(raw,source_[nextRow*count_],nextCopy,pad);
        }
        for(uint32_t level=0;level<depth_;++level){
          Gatherb<uint32_t>(gathered.template ReinterpretCast<uint32_t>(),src.template ReinterpretCast<uint32_t>(),plan[level*ap],static_cast<uint8_t>(ap/8),gp);PipeBarrier<PIPE_V>();
          Add(acc,acc,gathered,ap*16);PipeBarrier<PIPE_V>();
        }
        HalfBlockTransposeBack(gathered,acc,ap,ap);PipeBarrier<PIPE_ALL>();
        DataCopyExtParams outCopy{static_cast<uint16_t>(nr),axis_*2,0,0,0};
        DataCopyPad(output_[row*axis_],gathered,outCopy);PipeBarrier<PIPE_ALL>();
      }
      return true;
    }
  }

  __aicore__ inline bool ComputeSortedPlan(){
    if constexpr(E==4){
      if(axisPad_<GetBlockNum()*8 || depth_>=GetBlockNum() || 3ULL*axisPad_+GetBlockNum()*8>coreBase_)return false;
    }
    if(depth_==0 || depth_>64)return false;
    const uint32_t sourceStride=(count_+ALIGN-1)/ALIGN*ALIGN+ALIGN;
    const uint32_t fixed=axisPad_*12+512;
    const uint64_t perRow=2ULL*sourceStride*E+axisPad_*(4ULL*E+16)+4ULL*count_;
    if(fixed+28ULL*depth_+perRow>190000)return false;
    uint32_t batch=1,logBatch=0;
    while(batch<64 && batch*2*axisPad_<=16320 && fixed+28ULL*depth_+perRow*batch*2<=190000){batch*=2;++logBatch;}
    const uint32_t nmax=batch*axisPad_,sourceBank=batch*sourceStride;
    const uint32_t sortN=(axisPad_+63)/64*64;
    if(fixed+sortN*12+(sortN/8+31)/32*32>190000)return false;
    auto perm=At<uint32_t>(0);auto active=At<uint32_t>(axisPad_*4);auto starts=At<uint32_t>(axisPad_*4+256);
    auto head=At<uint32_t>(axisPad_*4+512);auto sortedHead=At<uint32_t>(axisPad_*8+512);
    auto values=At<float>(fixed);auto positions=At<uint32_t>(fixed+sortN*4);auto packed=At<uint32_t>(fixed+sortN*8);auto mask=At<uint8_t>(fixed+sortN*12);
    Duplicate(values,0.0f,sortN);Duplicate(head,0U,axisPad_);PipeBarrier<PIPE_ALL>();
    DataCopyExtParams countsCopy{1,axis_*4,0,0,0};DataCopyPadExtParams<uint32_t> countPad{false,0,0,0};
    DataCopyPad(head,work_,countsCopy,countPad);PipeBarrier<PIPE_ALL>();
    Cast(values,head.template ReinterpretCast<int32_t>(),RoundMode::CAST_NONE,axisPad_);PipeBarrier<PIPE_V>();
    CreateVecIndex(positions.template ReinterpretCast<int32_t>(),static_cast<int32_t>(0),sortN);PipeBarrier<PIPE_V>();
    ShiftLeft(positions,positions,2U,sortN);PipeBarrier<PIPE_V>();
    uint32_t cursor=0;
    if constexpr(E==4){
      const uint32_t core=GetBlockIdx(),histBase=axisPad_,permBase=histBase+GetBlockNum()*8,invBase=permBase+axisPad_;
      uint64_t retained=0;
      if(core<=depth_){
        CompareScalar(mask,values,static_cast<float>(static_cast<int32_t>(depth_-core)),CMPMODE::EQ,sortN);PipeBarrier<PIPE_V>();
        GatherMask(packed,positions,mask.template ReinterpretCast<uint32_t>(),true,axisPad_,{1,1,0,0},retained);PipeBarrier<PIPE_ALL>();
      }
      const uint32_t kept=static_cast<uint32_t>(retained);
      Duplicate(head,0U,axisPad_);Duplicate(sortedHead,0U,axisPad_);PipeBarrier<PIPE_ALL>();
      sortedHead.SetValue(0,kept);PipeBarrier<PIPE_ALL>();DataCopy(work_[histBase+core*8],sortedHead,8);
      const uint32_t zbegin=(axisPad_/8)*core/GetBlockNum()*8,zend=(axisPad_/8)*(core+1)/GetBlockNum()*8;
      if(zend>zbegin)DataCopy(atomicWork_[invBase+zbegin],head.template ReinterpretCast<int32_t>(),zend-zbegin);
      PipeBarrier<PIPE_ALL>();SyncAll<true>();
      DataCopy(sortedHead,work_[histBase],GetBlockNum()*8);PipeBarrier<PIPE_ALL>();
      uint32_t prefix=0;
      for(uint32_t c=0;c<=depth_;++c){const uint32_t n=sortedHead.GetValue(c*8);if(c<core)prefix+=n;cursor+=n;if(c<depth_)active.SetValue(depth_-c-1,cursor);}
      if(kept){
        for(uint32_t q=0;q<kept;++q){const uint32_t col=packed.GetValue(q)/4;head.SetValue(col,(prefix+q)*batch*E);}
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams cp{1,kept*4,0,0,0};DataCopyPad(work_[permBase+prefix],packed,cp);
        SetAtomicAdd<int32_t>();DataCopy(atomicWork_[invBase],head.template ReinterpretCast<int32_t>(),axisPad_);PipeBarrier<PIPE_ALL>();SetAtomicNone();
      }
      PipeBarrier<PIPE_ALL>();SyncAll<true>();
      DataCopy(perm,work_[permBase],axisPad_);DataCopy(head,work_[invBase],axisPad_);PipeBarrier<PIPE_ALL>();
    }else{
    for(int32_t d=static_cast<int32_t>(depth_);d>=0;--d){
      CompareScalar(mask,values,static_cast<float>(d),CMPMODE::EQ,sortN);PipeBarrier<PIPE_V>();
      uint64_t retained=0;GatherMask(packed,positions,mask.template ReinterpretCast<uint32_t>(),true,axisPad_,{1,1,0,0},retained);PipeBarrier<PIPE_ALL>();
      const uint32_t n=static_cast<uint32_t>(retained);
      for(uint32_t q=0;q<n;++q){const uint32_t a=packed.GetValue(q);perm.SetValue(cursor+q,a);head.SetValue(a/4,(cursor+q)*batch*E);}PipeBarrier<PIPE_ALL>();
      cursor+=n;if(d)active.SetValue(d-1,cursor);
    }
    }
    if(cursor!=axisPad_)return false;
    uint32_t planN=0;
    for(uint32_t d=0;d<depth_;++d){starts.SetValue(d,planN);planN+=(active.GetValue(d)*batch+7)/8*8;}
    uint32_t off=fixed;auto plan=At<uint32_t>(off);off+=planN*4;
    auto mapIn=At<uint32_t>(off);off+=nmax*4;auto mapOut=At<uint32_t>(off);off+=nmax*4;
    auto colidx=At<uint32_t>(off);off+=nmax*4;auto rowBase=At<uint32_t>(off);off+=nmax*4;
    auto src=At<T>(off);off+=2*sourceBank*E;auto out=At<T>(off);off+=2*nmax*E;
    auto sortedOut=At<T>(off);off+=nmax*E;auto gathered=At<T>(off);off+=nmax*E;
    if(off>190000)return false;
    CreateVecIndex(colidx.template ReinterpretCast<int32_t>(),static_cast<int32_t>(0),nmax);PipeBarrier<PIPE_V>();
    if(logBatch){
      ShiftLeft(rowBase,colidx,32-logBatch,nmax);PipeBarrier<PIPE_V>();ShiftRight(rowBase,rowBase,32-logBatch,nmax);
      ShiftRight(colidx,colidx,logBatch,nmax);
    }else Duplicate(rowBase,0U,nmax);
    PipeBarrier<PIPE_V>();ShiftLeft(colidx,colidx,2U,nmax);PipeBarrier<PIPE_V>();
    Gather(mapIn,perm,colidx,0U,nmax);PipeBarrier<PIPE_V>();
    if constexpr(E==2){ShiftRight(mapIn,mapIn,1U,nmax);PipeBarrier<PIPE_V>();}
    Muls(rowBase.template ReinterpretCast<int32_t>(),rowBase.template ReinterpretCast<int32_t>(),static_cast<int32_t>(axisPad_*E),nmax);PipeBarrier<PIPE_V>();
    Add(mapIn.template ReinterpretCast<int32_t>(),mapIn.template ReinterpretCast<int32_t>(),rowBase.template ReinterpretCast<int32_t>(),nmax);PipeBarrier<PIPE_V>();
    CreateVecIndex(colidx.template ReinterpretCast<int32_t>(),static_cast<int32_t>(0),axisPad_);PipeBarrier<PIPE_V>();
    ShiftLeft(colidx,colidx,2U,axisPad_);PipeBarrier<PIPE_V>();
    for(uint32_t rr=0;rr<batch;++rr){Gather(mapOut[rr*axisPad_],head,colidx,0U,axisPad_);PipeBarrier<PIPE_V>();Adds(mapOut[rr*axisPad_].template ReinterpretCast<int32_t>(),mapOut[rr*axisPad_].template ReinterpretCast<int32_t>(),static_cast<int32_t>(rr*E),axisPad_);PipeBarrier<PIPE_V>();}
    CreateVecIndex(colidx.template ReinterpretCast<int32_t>(),static_cast<int32_t>(0),nmax);PipeBarrier<PIPE_V>();
    if(logBatch){
      ShiftLeft(rowBase,colidx,32-logBatch,nmax);PipeBarrier<PIPE_V>();ShiftRight(rowBase,rowBase,32-logBatch,nmax);
      ShiftRight(colidx,colidx,logBatch,nmax);
    }else Duplicate(rowBase,0U,nmax);
    PipeBarrier<PIPE_V>();ShiftLeft(colidx,colidx,2U,nmax);
    Muls(rowBase.template ReinterpretCast<int32_t>(),rowBase.template ReinterpretCast<int32_t>(),static_cast<int32_t>(sourceStride*E),nmax);PipeBarrier<PIPE_V>();
    Adds(rowBase.template ReinterpretCast<int32_t>(),rowBase.template ReinterpretCast<int32_t>(),static_cast<int32_t>(32-E),nmax);PipeBarrier<PIPE_ALL>();
    for(uint32_t d=0;d<depth_;++d){
      const uint32_t n=active.GetValue(d)*batch,p=starts.GetValue(d);
      DataCopy(head,work_[denseBase_+static_cast<uint64_t>(d)*axisPad_],axisPad_);PipeBarrier<PIPE_ALL>();
      Gather(sortedHead,head,perm,0U,axisPad_);PipeBarrier<PIPE_V>();
      Gather(plan[p],sortedHead,colidx,0U,n);PipeBarrier<PIPE_V>();
      if constexpr(E==2){ShiftRight(plan[p],plan[p],1U,n);PipeBarrier<PIPE_V>();}
      Add(plan[p].template ReinterpretCast<int32_t>(),plan[p].template ReinterpretCast<int32_t>(),rowBase.template ReinterpretCast<int32_t>(),n);PipeBarrier<PIPE_ALL>();
    }
    Duplicate(src,static_cast<T>(0),2*sourceBank);PipeBarrier<PIPE_ALL>();
    uint32_t aa=ALIGN,bb=axis_;while(bb){uint32_t rr=aa%bb;aa=bb;bb=rr;}const uint32_t groupRows=ALIGN/aa;
    const uint64_t groups=(outer_+groupRows-1)/groupRows;
    uint64_t row=groups*GetBlockIdx()/GetBlockNum()*groupRows,end=groups*(GetBlockIdx()+1)/GetBlockNum()*groupRows;if(end>outer_)end=outer_;
    if(row==end)return true;
    const auto in0=static_cast<event_t>(pipe_.template AllocEventID<HardEvent::MTE2_V>());
    const auto in1=static_cast<event_t>(pipe_.template AllocEventID<HardEvent::MTE2_V>());
    const auto toOut=static_cast<event_t>(pipe_.template AllocEventID<HardEvent::V_MTE3>());
    const auto reuse=static_cast<event_t>(pipe_.template AllocEventID<HardEvent::MTE3_MTE2>());
    DataCopyPadExtParams<T> pad{false,0,0,0};
    uint32_t firstRows=end-row>batch?batch:static_cast<uint32_t>(end-row);
    DataCopyExtParams firstSource{static_cast<uint16_t>(firstRows),count_*E,0,1,0},firstOutput{static_cast<uint16_t>(firstRows),axis_*E,0,0,0};
    DataCopyPad(src[ALIGN],source_[row*count_],firstSource,pad);DataCopyPad(out,self_[row*axis_],firstOutput,pad);SetFlag<HardEvent::MTE2_V>(in0);
    uint32_t bank=0;
    for(;row<end;){
      const uint32_t rows=end-row>batch?batch:static_cast<uint32_t>(end-row);
      if(row+rows<end){
        const uint32_t nb=1-bank,nr=end-row-rows>batch?batch:static_cast<uint32_t>(end-row-rows);
        DataCopyExtParams sp{static_cast<uint16_t>(nr),count_*E,0,1,0},op{static_cast<uint16_t>(nr),axis_*E,0,0,0};
        DataCopyPad(src[nb*sourceBank+ALIGN],source_[(row+rows)*count_],sp,pad);DataCopyPad(out[nb*nmax],self_[(row+rows)*axis_],op,pad);SetFlag<HardEvent::MTE2_V>(nb?in1:in0);
      }
      WaitFlag<HardEvent::MTE2_V>(bank?in1:in0);
      Gather(sortedOut,out[bank*nmax],mapIn,0U,nmax);PipeBarrier<PIPE_V>();
      for(uint32_t d=0;d<depth_;++d){const uint32_t n=active.GetValue(d)*batch,p=starts.GetValue(d);Gather(gathered,src[bank*sourceBank],plan[p],0U,n);PipeBarrier<PIPE_V>();Add(sortedOut,sortedOut,gathered,n);PipeBarrier<PIPE_V>();}
      Gather(out[bank*nmax],sortedOut,mapOut,0U,nmax);PipeBarrier<PIPE_V>();
      SetFlag<HardEvent::V_MTE3>(toOut);WaitFlag<HardEvent::V_MTE3>(toOut);
      DataCopyExtParams op{static_cast<uint16_t>(rows),axis_*E,0,0,0};DataCopyPad(output_[row*axis_],out[bank*nmax],op);
      SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);row+=rows;bank=1-bank;
    }
    PipeBarrier<PIPE_ALL>();pipe_.template ReleaseEventID<HardEvent::MTE2_V>(in0);pipe_.template ReleaseEventID<HardEvent::MTE2_V>(in1);
    pipe_.template ReleaseEventID<HardEvent::V_MTE3>(toOut);pipe_.template ReleaseEventID<HardEvent::MTE3_MTE2>(reuse);return true;
  }

  __aicore__ inline bool ComputePipelinedPlan(){
    const uint32_t sourceStride=(count_+ALIGN-1)/ALIGN*ALIGN+ALIGN;
    const uint64_t perRow=2ULL*sourceStride*E+static_cast<uint64_t>(axisPad_)*(3*E+4*depth_);
    if(perRow>190000-axisPad_*4)return false;
    uint32_t batch=(190000-axisPad_*4)/perRow;if(batch>64)batch=64;if(batch>16320/axisPad_)batch=16320/axisPad_;if(!batch)return false;
    const uint32_t nmax=batch*axisPad_,sourceBank=batch*sourceStride;
    auto head=At<uint32_t>(0);uint32_t off=axisPad_*4;
    auto plan=At<uint32_t>(off);off+=depth_*nmax*4;
    auto src=At<T>(off);off+=2*sourceBank*E;auto out=At<T>(off);off+=2*nmax*E;auto gathered=At<T>(off);
    for(uint32_t level=0;level<depth_;++level){
      DataCopy(head,work_[denseBase_+static_cast<uint64_t>(level)*axisPad_],axisPad_);PipeBarrier<PIPE_ALL>();
      if constexpr(E==2){ShiftRight(head,head,1U,axisPad_);PipeBarrier<PIPE_V>();}
      for(uint32_t r=0;r<batch;++r)Adds(plan[level*nmax+r*axisPad_].template ReinterpretCast<int32_t>(),head.template ReinterpretCast<int32_t>(),static_cast<int32_t>(r*sourceStride*E+32-E),axisPad_);
      PipeBarrier<PIPE_ALL>();
    }
    Duplicate(src,static_cast<T>(0),2*sourceBank);PipeBarrier<PIPE_ALL>();
    uint32_t aa=ALIGN,bb=axis_;while(bb){uint32_t rr=aa%bb;aa=bb;bb=rr;}const uint32_t groupRows=ALIGN/aa;
    const uint64_t groups=(outer_+groupRows-1)/groupRows;
    uint64_t row=groups*GetBlockIdx()/GetBlockNum()*groupRows,end=groups*(GetBlockIdx()+1)/GetBlockNum()*groupRows;if(end>outer_)end=outer_;
    if(row==end)return true;
    const auto in0=static_cast<event_t>(pipe_.template AllocEventID<HardEvent::MTE2_V>());
    const auto in1=static_cast<event_t>(pipe_.template AllocEventID<HardEvent::MTE2_V>());
    const auto toOut=static_cast<event_t>(pipe_.template AllocEventID<HardEvent::V_MTE3>());
    const auto reuse=static_cast<event_t>(pipe_.template AllocEventID<HardEvent::MTE3_MTE2>());
    DataCopyPadExtParams<T> pad{false,0,0,0};
    uint32_t firstRows=end-row>batch?batch:static_cast<uint32_t>(end-row);
    DataCopyExtParams firstSource{static_cast<uint16_t>(firstRows),count_*E,0,1,0};
    DataCopyExtParams firstOutput{static_cast<uint16_t>(firstRows),axis_*E,0,0,0};
    DataCopyPad(src[ALIGN],source_[row*count_],firstSource,pad);DataCopyPad(out,self_[row*axis_],firstOutput,pad);SetFlag<HardEvent::MTE2_V>(in0);
    uint32_t bank=0;
    for(;row<end;){
      const uint32_t rows=end-row>batch?batch:static_cast<uint32_t>(end-row),n=rows*axisPad_;
      if(row+rows<end){
        const uint32_t nb=1-bank,nr=end-row-rows>batch?batch:static_cast<uint32_t>(end-row-rows);
        DataCopyExtParams sp{static_cast<uint16_t>(nr),count_*E,0,1,0},op{static_cast<uint16_t>(nr),axis_*E,0,0,0};
        DataCopyPad(src[nb*sourceBank+ALIGN],source_[(row+rows)*count_],sp,pad);DataCopyPad(out[nb*nmax],self_[(row+rows)*axis_],op,pad);
        SetFlag<HardEvent::MTE2_V>(nb?in1:in0);
      }
      WaitFlag<HardEvent::MTE2_V>(bank?in1:in0);
      for(uint32_t level=0;level<depth_;++level){
        Gather(gathered,src[bank*sourceBank],plan[level*nmax],0U,n);PipeBarrier<PIPE_V>();Add(out[bank*nmax],out[bank*nmax],gathered,n);PipeBarrier<PIPE_V>();
      }
      SetFlag<HardEvent::V_MTE3>(toOut);WaitFlag<HardEvent::V_MTE3>(toOut);
      DataCopyExtParams op{static_cast<uint16_t>(rows),axis_*E,0,0,0};DataCopyPad(output_[row*axis_],out[bank*nmax],op);
      SetFlag<HardEvent::MTE3_MTE2>(reuse);WaitFlag<HardEvent::MTE3_MTE2>(reuse);
      row+=rows;bank=1-bank;
    }
    PipeBarrier<PIPE_ALL>();
    pipe_.template ReleaseEventID<HardEvent::MTE2_V>(in0);pipe_.template ReleaseEventID<HardEvent::MTE2_V>(in1);
    pipe_.template ReleaseEventID<HardEvent::V_MTE3>(toOut);pipe_.template ReleaseEventID<HardEvent::MTE3_MTE2>(reuse);
    return true;
  }

  __aicore__ inline bool ComputeCachedPlan(){
    const uint32_t sourceStride=(count_+ALIGN-1)/ALIGN*ALIGN+ALIGN;
    const uint64_t perRow=static_cast<uint64_t>(sourceStride)*E+static_cast<uint64_t>(axisPad_)*(2*E+4*depth_);
    if(perRow>190000-axisPad_*4)return false;
    uint32_t batch=(190000-axisPad_*4)/perRow;if(batch>64)batch=64;if(batch>16320/axisPad_)batch=16320/axisPad_;
    if(!batch)return false;
    const uint32_t nmax=batch*axisPad_;
    auto head=At<uint32_t>(0);uint32_t off=axisPad_*4;
    auto plan=At<uint32_t>(off);off+=depth_*nmax*4;
    auto src=At<T>(off);off+=batch*sourceStride*E;
    auto out=At<T>(off);off+=nmax*E;auto gathered=At<T>(off);
    // Expand the actual ordered index plan once. All subsequent row batches
    // reuse these offsets, avoiding per-level GM reads and repeated broadcasts.
    for(uint32_t level=0;level<depth_;++level){
      DataCopy(head,work_[denseBase_+static_cast<uint64_t>(level)*axisPad_],axisPad_);PipeBarrier<PIPE_ALL>();
      if constexpr(E==2){ShiftRight(head,head,1U,axisPad_);PipeBarrier<PIPE_V>();}
      for(uint32_t r=0;r<batch;++r){
        Adds(plan[level*nmax+r*axisPad_].template ReinterpretCast<int32_t>(),head.template ReinterpretCast<int32_t>(),static_cast<int32_t>(r*sourceStride*E+32-E),axisPad_);
      }
      PipeBarrier<PIPE_ALL>();
    }
    Duplicate(src,static_cast<T>(0),batch*sourceStride);PipeBarrier<PIPE_ALL>();
    uint32_t aa=ALIGN,bb=axis_;while(bb){uint32_t rr=aa%bb;aa=bb;bb=rr;}const uint32_t groupRows=ALIGN/aa;
    const uint64_t groups=(outer_+groupRows-1)/groupRows;
    uint64_t row=groups*GetBlockIdx()/GetBlockNum()*groupRows,end=groups*(GetBlockIdx()+1)/GetBlockNum()*groupRows;if(end>outer_)end=outer_;
    DataCopyPadExtParams<T> pad{false,0,0,0};
    for(;row<end;){
      const uint32_t rows=end-row>batch?batch:static_cast<uint32_t>(end-row),n=rows*axisPad_;
      DataCopyExtParams sourceBatch{static_cast<uint16_t>(rows),count_*E,0,1,0};
      DataCopyExtParams outputBatch{static_cast<uint16_t>(rows),axis_*E,0,0,0};
      DataCopyPad(src[ALIGN],source_[row*count_],sourceBatch,pad);DataCopyPad(out,self_[row*axis_],outputBatch,pad);PipeBarrier<PIPE_ALL>();
      for(uint32_t level=0;level<depth_;++level){
        Gather(gathered,src,plan[level*nmax],0U,n);PipeBarrier<PIPE_V>();Add(out,out,gathered,n);PipeBarrier<PIPE_V>();
      }
      PipeBarrier<PIPE_ALL>();DataCopyPad(output_[row*axis_],out,outputBatch);PipeBarrier<PIPE_ALL>();row+=rows;
    }
    return true;
  }

  __aicore__ inline void Compute(){
    const uint32_t sourceStride=(count_+ALIGN-1)/ALIGN*ALIGN+ALIGN;
    uint32_t batch=16320/axisPad_;if(batch>64)batch=64;
    for(;batch>1;--batch){const uint64_t n=static_cast<uint64_t>(batch)*axisPad_;if(axisPad_*4+static_cast<uint64_t>(batch)*sourceStride*E+n*(2*E+8)<=190000)break;}
    const uint32_t nmax=batch*axisPad_;
    auto head=At<uint32_t>(0);uint32_t off=axisPad_*4;
    auto src=At<T>(off);off+=batch*sourceStride*E;
    auto out=At<T>(off);off+=nmax*E;auto gathered=At<T>(off);off+=nmax*E;
    auto offsets=At<uint32_t>(off);off+=nmax*4;auto rowBase=At<int32_t>(off);
    Duplicate(src,static_cast<T>(0),batch*sourceStride);
    for(uint32_t r=0;r<batch;++r)Duplicate(rowBase[r*axisPad_],static_cast<int32_t>(r*sourceStride*E+32-E),axisPad_);
    PipeBarrier<PIPE_ALL>();
    uint32_t aa=ALIGN,bb=axis_;while(bb){uint32_t rr=aa%bb;aa=bb;bb=rr;}const uint32_t groupRows=ALIGN/aa;
    const uint64_t groups=(outer_+groupRows-1)/groupRows;
    uint64_t row=groups*GetBlockIdx()/GetBlockNum()*groupRows,end=groups*(GetBlockIdx()+1)/GetBlockNum()*groupRows;if(end>outer_)end=outer_;
    DataCopyPadExtParams<T> pad{false,0,0,0};DataCopyExtParams sp{1,count_*E,0,0,0},op{1,axis_*E,0,0,0};
    for(;row<end;){
      const uint32_t rows=end-row>batch?batch:static_cast<uint32_t>(end-row),n=rows*axisPad_;
      for(uint32_t r=0;r<rows;++r){DataCopyPad(src[r*sourceStride+ALIGN],source_[(row+r)*count_],sp,pad);DataCopyPad(out[r*axisPad_],self_[(row+r)*axis_],op,pad);}
      PipeBarrier<PIPE_ALL>();
      for(uint32_t level=0;level<depth_;++level){
        DataCopy(head,work_[denseBase_+static_cast<uint64_t>(level)*axisPad_],axisPad_);PipeBarrier<PIPE_ALL>();
        for(uint32_t r=0;r<rows;++r)DataCopy(offsets[r*axisPad_],head,axisPad_);
        PipeBarrier<PIPE_ALL>();if constexpr(E==2){ShiftRight(offsets,offsets,1U,n);PipeBarrier<PIPE_V>();}
        Add(offsets.template ReinterpretCast<int32_t>(),offsets.template ReinterpretCast<int32_t>(),rowBase,n);PipeBarrier<PIPE_V>();
        Gather(gathered,src,offsets,0U,n);PipeBarrier<PIPE_V>();Add(out,out,gathered,n);PipeBarrier<PIPE_V>();
      }
      PipeBarrier<PIPE_ALL>();for(uint32_t r=0;r<rows;++r)DataCopyPad(output_[(row+r)*axis_],out[r*axisPad_],op);
      PipeBarrier<PIPE_ALL>();row+=rows;
    }
  }
  TPipe pipe_;TBuf<TPosition::VECCALC> arena_;GlobalTensor<T> self_,source_,output_;GlobalTensor<int32_t> index_,atomicWork_;GlobalTensor<uint32_t> work_;
  uint32_t axis_,axisPad_,count_,countPad_,listStride_,depth_;uint64_t outer_,coreBase_,denseBase_;
};
