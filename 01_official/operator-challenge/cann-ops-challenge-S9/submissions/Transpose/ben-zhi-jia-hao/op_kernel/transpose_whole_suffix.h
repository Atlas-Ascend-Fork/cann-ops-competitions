template<typename T> class TransposeWholeSuffixKernel {
public:
 template<typename P> __aicore__ inline void Run(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe){if constexpr(sizeof(T)==2||sizeof(T)==4){if(t->tilePrefetch==146)Blocked(x,y,t,pipe);else Impl(x,y,t,pipe);}}
private:
 template<typename P> __aicore__ inline void Blocked(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe) {
  const uint32_t ds=sizeof(T),m=t->foldExt,b=t->foldLen,k=t->rowCount,l=t->blkLenElems;
  const uint32_t mt=t->tileMT,bt=t->rowsPerBatch,kt=t->tileKT,slot=t->slotBytes;
  const uint32_t rows=mt*bt,n=rows*l,stage=rows*slot,cores=t->usedCores;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
#if !defined(ASCENDC_CPU_DEBUG)
  if(t->chunkBlks==151)x_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
#endif
  pipe->InitBuffer(in_,2,stage);pipe->InitBuffer(out_,1,n*kt*ds);pipe->InitBuffer(idx_,(n*4+31)/32*32);
  auto index=idx_.template Get<uint32_t>();auto tab=index.template ReinterpretCast<int32_t>();
  auto tmp=in_.template AllocTensor<T>();in_.EnQue(tmp);tmp=in_.template DeQue<T>();
  auto ti=tmp.template ReinterpretCast<int32_t>();auto tf=tmp.template ReinterpretCast<float>();
  uint32_t step=stage/4/64*64;if(step>8192)step=8192;SetMaskNorm();
  for(uint32_t off=0;off<n;off+=step) {
   const uint32_t count=static_cast<uint32_t>(MinU64(step,n-off));auto dst=tab[off];
   ArithProgression(ti,static_cast<int32_t>(off),1,count);PipeBarrier<PIPE_V>();
   Cast(tf,ti,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();Adds(tf,tf,0.5f,count);PipeBarrier<PIPE_V>();
   Muls(tf,tf,1.0f/l,count);PipeBarrier<PIPE_V>();Cast(ti,tf,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
   Muls(ti,ti,static_cast<int32_t>(slot-l*ds),count);PipeBarrier<PIPE_V>();
   ArithProgression(dst,static_cast<int32_t>(off*ds),static_cast<int32_t>(ds),count);PipeBarrier<PIPE_V>();
   Add(dst,dst,ti,count);PipeBarrier<PIPE_V>();
  }
  in_.FreeTensor(tmp);
  const uint64_t ntm=(m+mt-1)/mt,ntb=(b+bt-1)/bt,ntk=(k+kt-1)/kt,total=ntm*ntb*ntk;
  const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};

  for(uint64_t unit=GetBlockIdx();unit<total;unit+=cores) {
   const uint64_t mi=unit%ntm,bi=unit/ntm%ntb,ki=unit/ntm/ntb;
   const uint64_t m0=mi*mt,b0=bi*bt,k0=ki*kt;
   const uint32_t vm=MinU64(mt,m-m0),vb=MinU64(bt,b-b0),vk=MinU64(kt,k-k0);
   const DataCopyExtParams load{static_cast<uint16_t>(vm),vk*l*ds,
    static_cast<uint32_t>((static_cast<uint64_t>(b)*k-vk)*l*ds),(slot-(vk*l*ds+31)/32*32)/32,0};
   const DataCopyExtParams store{static_cast<uint16_t>(vb),vm*l*ds,
    (mt*l*ds-(vm*l*ds+31)/32*32)/32,(m-vm)*l*ds,0};
   auto in=in_.template AllocTensor<T>();
   for(uint32_t bb=0;bb<vb;++bb)
    DataCopyPad(in[bb*mt*(slot/ds)],x_[((m0*b+b0+bb)*k+k0)*l],load,pad);
   in_.EnQue(in);in=in_.template DeQue<T>();auto out=out_.template AllocTensor<T>();
   for(uint32_t tt=0;tt<vk;++tt)for(uint32_t off=0;off<n;off+=8192)
    Gather(out[tt*n+off],in,index[off],tt*l*ds,static_cast<uint32_t>(MinU64(8192,n-off)));
   in_.FreeTensor(in);out_.EnQue(out);out=out_.template DeQue<T>();
   for(uint32_t tt=0;tt<vk;++tt)
    DataCopyPad(y_[((k0+tt)*b+b0)*m*l+m0*l],out[tt*n],store);
   out_.FreeTensor(out);
  }
 }

 template<typename P> __aicore__ inline void Impl(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe){
  const uint32_t ds=sizeof(T),m=t->foldExt,b=t->foldLen,l=t->blkLenElems,rows=t->rowsPerBatch,slot=t->slotBytes,n=rows*l,stage=rows*slot,cores=t->usedCores;
  const uint32_t span=t->tileMT?t->tileMT:1,outputs=t->tileKT?t->tileKT:1,outPitch=(n*ds+31)/32*32;
  const uint64_t k=t->rowCount;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
  pipe->InitBuffer(in_,2,stage);pipe->InitBuffer(out_,outputs,span*outPitch);pipe->InitBuffer(idx_,(n*4+31)/32*32);
  auto index=idx_.template Get<uint32_t>();auto tab=index.template ReinterpretCast<int32_t>();
  auto temp=in_.template AllocTensor<T>();in_.EnQue(temp);temp=in_.template DeQue<T>();
  auto ti=temp.template ReinterpretCast<int32_t>();auto tf=temp.template ReinterpretCast<float>();
  uint32_t chunk=(stage/4)/64*64;if(chunk>8192)chunk=8192;
  SetMaskNorm();
  for(uint32_t off=0;off<n;off+=chunk){
   const uint32_t count=static_cast<uint32_t>(MinU64(chunk,n-off));auto dst=tab[off];
   ArithProgression(dst,static_cast<int32_t>(off),1,count);PipeBarrier<PIPE_V>();
   Cast(tf,dst,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();Adds(tf,tf,0.5f,count);PipeBarrier<PIPE_V>();Muls(tf,tf,1.0f/l,count);PipeBarrier<PIPE_V>();Cast(ti,tf,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
   Muls(ti,ti,static_cast<int32_t>(b*slot-l*ds),count);PipeBarrier<PIPE_V>();
   ArithProgression(dst,static_cast<int32_t>(off*ds),static_cast<int32_t>(ds),count);PipeBarrier<PIPE_V>();Add(dst,dst,ti,count);PipeBarrier<PIPE_V>();
   ArithProgression(ti,static_cast<int32_t>(off),1,count);PipeBarrier<PIPE_V>();
   Cast(tf,ti,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();Adds(tf,tf,0.5f,count);PipeBarrier<PIPE_V>();Muls(tf,tf,1.0f/(l*m),count);PipeBarrier<PIPE_V>();Cast(ti,tf,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
   Muls(ti,ti,static_cast<int32_t>((1-static_cast<int32_t>(m*b))*static_cast<int32_t>(slot)),count);PipeBarrier<PIPE_V>();Add(dst,dst,ti,count);PipeBarrier<PIPE_V>();
  }
  in_.FreeTensor(temp);
  const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
  const DataCopyExtParams load{static_cast<uint16_t>(rows),span*l*ds,static_cast<uint32_t>((k-span)*l*ds),0,0};
  const DataCopyExtParams store{static_cast<uint16_t>(span),n*ds,0,0,0};
  for(uint64_t unit=GetBlockIdx();unit<(k+span-1)/span;unit+=cores){
   const uint64_t j=MinU64(unit*span,k-span);
   auto in=in_.template AllocTensor<T>();DataCopyPad(in,x_[j*l],load,pad);in_.EnQue(in);auto out=out_.template AllocTensor<T>();in=in_.template DeQue<T>();
   for(uint32_t u=0;u<span;++u)for(uint32_t off=0;off<n;off+=8192)Gather(out[u*(outPitch/ds)+off],in,index[off],u*l*ds,static_cast<uint32_t>(MinU64(8192,n-off)));
   in_.FreeTensor(in);out_.EnQue(out);out=out_.template DeQue<T>();DataCopyPad(y_[j*n],out,store);out_.FreeTensor(out);
  }
 }
 TQue<QuePosition::VECIN,2> in_;TQue<QuePosition::VECOUT,2> out_;TBuf<QuePosition::VECCALC> idx_;GlobalTensor<T> x_,y_;
};

template<typename T> class TransposeCompactSuffixKernel {
public:
 template<typename P> __aicore__ inline void Run(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe){if constexpr(sizeof(T)==2||sizeof(T)==4)Impl(x,y,t,pipe);}
private:
 template<typename P> __aicore__ inline void Impl(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe) {
  const uint32_t ds=sizeof(T),m=t->foldExt,b=t->foldLen,k=t->rowCount,l=t->blkLenElems;
  const uint32_t mt=t->tileMT,bt=t->rowsPerBatch,kt=t->tileKT,slot=t->slotBytes;
  const uint32_t rows=mt*bt,n=rows*l,stage=rows*slot,cores=t->usedCores,outPitch=(n*ds+31)/32*32/ds;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
#if !defined(ASCENDC_CPU_DEBUG)
  if(t->chunkBlks==151)x_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
#endif
  pipe->InitBuffer(in_,2,stage);pipe->InitBuffer(out_,1,outPitch*kt*ds);pipe->InitBuffer(idx_,(n*4+31)/32*32);
  auto index=idx_.template Get<uint32_t>();auto tab=index.template ReinterpretCast<int32_t>();
  auto tmp=in_.template AllocTensor<T>();in_.EnQue(tmp);tmp=in_.template DeQue<T>();
  auto ti=tmp.template ReinterpretCast<int32_t>();auto tf=tmp.template ReinterpretCast<float>();
  uint32_t step=stage/4/64*64;if(step>8192)step=8192;SetMaskNorm();
  for(uint32_t off=0;off<n;off+=step) {
   const uint32_t count=static_cast<uint32_t>(MinU64(step,n-off));auto dst=tab[off];
   ArithProgression(ti,static_cast<int32_t>(off),1,count);PipeBarrier<PIPE_V>();
   Cast(tf,ti,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();Adds(tf,tf,0.5f,count);PipeBarrier<PIPE_V>();
   Muls(tf,tf,1.0f/l,count);PipeBarrier<PIPE_V>();Cast(ti,tf,RoundMode::CAST_FLOOR,count);PipeBarrier<PIPE_V>();
   Muls(ti,ti,static_cast<int32_t>(slot-l*ds),count);PipeBarrier<PIPE_V>();
   ArithProgression(dst,static_cast<int32_t>(off*ds),static_cast<int32_t>(ds),count);PipeBarrier<PIPE_V>();
   Add(dst,dst,ti,count);PipeBarrier<PIPE_V>();
  }
  in_.FreeTensor(tmp);
  const uint64_t ntm=(m+mt-1)/mt,ntb=(b+bt-1)/bt,ntk=(k+kt-1)/kt,total=ntm*ntb*ntk;
  const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};

  for(uint64_t unit=GetBlockIdx();unit<total;unit+=cores) {
   const uint64_t mi=unit%ntm,bi=unit/ntm%ntb,ki=unit/ntm/ntb;
   const uint64_t m0=mi*mt,b0=bi*bt,k0=ki*kt;
   const uint32_t vm=MinU64(mt,m-m0),vb=MinU64(bt,b-b0),vk=MinU64(kt,k-k0);
   const DataCopyExtParams load{static_cast<uint16_t>(vm),vk*l*ds,
    static_cast<uint32_t>((static_cast<uint64_t>(b)*k-vk)*l*ds),(slot-(vk*l*ds+31)/32*32)/32,0};
   const DataCopyExtParams store{static_cast<uint16_t>(vb),vm*l*ds,
    (outPitch*ds-(vm*l*ds+31)/32*32)/32,(m-vm)*l*ds,0};
   auto in=in_.template AllocTensor<T>();
   for(uint32_t bb=0;bb<vb;++bb)
    DataCopyPad(in[bb*mt*(slot/ds)],x_[((m0*b+b0+bb)*k+k0)*l],load,pad);
   in_.EnQue(in);in=in_.template DeQue<T>();auto out=out_.template AllocTensor<T>();
   for(uint32_t tt=0;tt<vk;++tt)for(uint32_t off=0;off<n;off+=8192)
    Gather(out[tt*outPitch+off],in,index[off],tt*l*ds,static_cast<uint32_t>(MinU64(8192,n-off)));
   in_.FreeTensor(in);out_.EnQue(out);out=out_.template DeQue<T>();
   DataCopyPad(y_[(k0*b+b0)*m*l],out,DataCopyExtParams{static_cast<uint16_t>(vk),m*l*ds,0,(b-1)*m*l*ds,0});
   out_.FreeTensor(out);
  }
 }
 TQue<QuePosition::VECIN,2> in_;TQue<QuePosition::VECOUT,2> out_;TBuf<QuePosition::VECCALC> idx_;GlobalTensor<T> x_,y_;
};
