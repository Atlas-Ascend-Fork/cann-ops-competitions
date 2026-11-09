template<typename T> class TransposeGroupKernel {
public:
 template<typename P> __aicore__ inline void Run(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe){if constexpr(sizeof(T)==4)Impl(x,y,t,pipe);}
private:
 template<typename P> __aicore__ inline void Impl(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe){
  const uint64_t bcount=t->rowCount,m=t->rowLen,k=t->tileK,c=t->foldExt,asp=t->foldSrcStride,bsp=t->foldDstRowStride,dp=t->dstRowStride;
  const uint32_t mt=t->tileMT,kt=t->tileKT,cores=t->usedCores,mi=m<mt?m:mt;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
#if !defined(ASCENDC_CPU_DEBUG)
  if(t->tilePrefetch==2)x_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
#endif
  pipe->InitBuffer(in_,2,mt*kt*4);pipe->InitBuffer(out_,1,mt*kt*4);
  const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
  const DataCopyExtParams store{static_cast<uint16_t>(kt),mi*4,static_cast<uint32_t>((mt-(mi+7)/8*8)*4/32),static_cast<uint32_t>((dp-mi)*4),0};
  const TransDataTo5HDParams cv{false,false,static_cast<uint8_t>(kt/8),static_cast<uint16_t>(mt),1};
  const uint64_t nm=(m+mt-1)/mt,nk=(k+kt-1)/kt,per=nm*nk;
  for(uint64_t u=GetBlockIdx();u<bcount*per;u+=cores){
   const uint64_t b=u/per,local=u%per,m0=mt>=m?0:MinU64((local%nm)*mt,m-mt),k0=MinU64((local/nm)*kt,k-kt);
   auto in=in_.template AllocTensor<T>();
   for(uint64_t i=0;i<mi;){
    const uint64_t row=m0+i,a=row/c,j=row%c,len=MinU64(mi-i,c-j);
    const DataCopyExtParams load{static_cast<uint16_t>(len),kt*4,static_cast<uint32_t>((k-kt)*4),0,0};
    DataCopyPad(in[i*kt],x_[a*asp+b*bsp+j*k+k0],load,pad);i+=len;
   }
   in_.EnQue(in);auto out=out_.template AllocTensor<T>();in=in_.template DeQue<T>();
   for(uint32_t i=0;i<mt;i+=16){
    uint64_t src[16],dst[16];for(uint32_t r=0;r<16;++r)src[r]=reinterpret_cast<uint64_t>(in[(i+r)*kt].GetPhyAddr());
    for(uint32_t r=0;r<8;++r){dst[2*r]=reinterpret_cast<uint64_t>(out[r*mt+i].GetPhyAddr());dst[2*r+1]=dst[2*r]+32;}
    TransDataTo5HD<T>(dst,src,cv);
   }
   in_.FreeTensor(in);out_.EnQue(out);out=out_.template DeQue<T>();
   DataCopyPad(y_[k0*dp+b*m+m0],out,store);out_.FreeTensor(out);
  }
 }
 TQue<QuePosition::VECIN,2> in_;TQue<QuePosition::VECOUT,1> out_;GlobalTensor<T> x_,y_;
};

template<typename T> class TransposeSlabKernel {
public:
 template<typename P> __aicore__ inline void Run(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe){if constexpr(sizeof(T)==4)Impl(x,y,t,pipe);}
private:
 template<typename P> __aicore__ inline void Impl(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe){
  const uint64_t bcount=t->rowCount,m=t->rowLen,k=t->tileK,dp=t->dstRowStride;
  const uint32_t mt=t->tileMT,ks=t->tileKT,gr=t->rowsPerBatch,cores=t->usedCores,n=t->outerRank,rawRows=t->foldLen==8?8:16;
  uint64_t ext[8],si[8],so[8];for(uint32_t a=0;a<n;++a){ext[a]=t->outerShape[a];si[a]=t->outerInStride[a];so[a]=t->outerOutStride[a];}
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
#if !defined(ASCENDC_CPU_DEBUG)
  x_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
#endif
  pipe->InitBuffer(raw_,1,rawRows*ks*4);pipe->InitBuffer(padded_,16*ks*4);pipe->InitBuffer(index_,gr*ks*4);pipe->InitBuffer(out_,1,mt*ks*4);
  auto aligned=padded_.template Get<T>();auto idx=index_.template Get<uint32_t>();auto ii=idx.template ReinterpretCast<int32_t>();
  for(uint32_t r=0;r<gr;++r)ArithProgression(ii[r*ks],static_cast<int32_t>(r*k*4),static_cast<int32_t>(4),ks);
  PipeBarrier<PIPE_V>();
  const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
  const DataCopyExtParams load{1,static_cast<uint32_t>(rawRows*k*4),0,0,0};
  const DataCopyExtParams store{static_cast<uint16_t>(k),mt*4,0,static_cast<uint32_t>((dp-mt)*4),0};
  const TransDataTo5HDParams cv{false,false,static_cast<uint8_t>(ks/8),static_cast<uint16_t>(mt),1};
  const uint64_t nm=(m+mt-1)/mt;
  for(uint64_t u=GetBlockIdx();u<bcount*nm;u+=cores){
   uint64_t b=u/nm,ib=0,ob=0;for(int a=n-1;a>=0;--a){const uint64_t j=b%ext[a];b/=ext[a];ib+=j*si[a];ob+=j*so[a];}
   const uint64_t m0=MinU64((u%nm)*mt,m-mt);auto out=out_.template AllocTensor<T>();
   for(uint32_t s=0;s<mt;s+=16){
    for(uint32_t half=0;half<16;half+=rawRows) {
     auto raw=raw_.template AllocTensor<T>();DataCopyPad(raw,x_[ib+(m0+s+half)*k],load,pad);raw_.EnQue(raw);raw=raw_.template DeQue<T>();
     for(uint32_t r=0;r<rawRows;r+=gr)Gather(aligned[(half+r)*ks],raw,idx,static_cast<uint32_t>(r*k*4),gr*ks);
     PipeBarrier<PIPE_V>();raw_.FreeTensor(raw);
    }
    uint64_t src[16],dst[16];for(uint32_t r=0;r<16;++r)src[r]=reinterpret_cast<uint64_t>(aligned[r*ks].GetPhyAddr());
    for(uint32_t r=0;r<8;++r){dst[2*r]=reinterpret_cast<uint64_t>(out[r*mt+s].GetPhyAddr());dst[2*r+1]=dst[2*r]+32;}
    TransDataTo5HD<T>(dst,src,cv);PipeBarrier<PIPE_V>();
   }
   out_.EnQue(out);out=out_.template DeQue<T>();DataCopyPad(y_[ob+m0],out,store);out_.FreeTensor(out);
  }
 }
 TQue<QuePosition::VECIN,1> raw_;TBuf<QuePosition::VECCALC> padded_,index_;TQue<QuePosition::VECOUT,1> out_;GlobalTensor<T> x_,y_;
};

template <typename T>
class TransposeAlignedWideKernel {
public:
    __aicore__ inline TransposeAlignedWideKernel() {}

    template <typename PtrT>
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        if constexpr (sizeof(T) == 4) {
            RunImpl(x, y, t, pipe);
        }
    }

private:
    template <typename PtrT>
    __aicore__ inline void RunImpl(GM_ADDR x, GM_ADDR y, PtrT t, TPipe* pipe)
    {
        // Read the key-3 field set once from the GM tiling pointer.
        numel_ = t->numel;
        rowCount_ = t->rowCount;
        rowLen_ = t->rowLen;
        srcBlkPitch_ = t->srcBlkPitch;
        usedCores_ = t->usedCores;
        tileK_ = t->tileK;
        dstRowStride_ = t->dstRowStride;
        tileMT_ = t->tileMT;
        tileKT_ = t->tileKT;
        // On key 9 tilePrefetch carries the SLAB BUFFER COUNT (2 = double
        // buffered loads, 1 = single slab traded for panel width); the route
        // has no prefetch arm of its own.
        tilePrefetch_ = t->tilePrefetch;
        outerRank_ = t->outerRank;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            outerShape_[a] = t->outerShape[a];
            outerInStride_[a] = t->outerInStride[a];
            outerOutStride_[a] = t->outerOutStride[a];
        }
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), numel_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), numel_);
#if defined(ASCENDC_CPU_DEBUG)
#else
        // Disable input allocation in L2 for this streaming read path.
        xGm_.template SetL2CacheHint<AscendC::CacheRwMode::READ>(
            AscendC::CacheMode::CACHE_MODE_DISABLE);
#endif

        if (numel_ == 0 || rowLen_ == 0 || tileK_ == 0 || tileMT_ == 0 || tileKT_ == 0 ||
            rowLen_ < tileMT_) {
            return;
        }

        const uint32_t ds = static_cast<uint32_t>(sizeof(T));
        const uint32_t mt = tileMT_;
        const uint32_t kSnap = tileKT_;
        // Mode 3 pairs double-buffered eight-row slabs to supply the 16 source
        // pointers of each vnchwconv operation.
        const bool pair8 =
            (tilePrefetch_ == 3u || tilePrefetch_ == 4u || tilePrefetch_ == 6u);
        // Mode 5 uses double-buffered 16-row slabs with wave scheduling.
        // Mode 4 pairs eight-row slabs and assigns units round-robin so neighboring
        // cores write adjacent output strips.
        const bool wave =
            (tilePrefetch_ == 4u || tilePrefetch_ == 5u || tilePrefetch_ == 6u ||
             tilePrefetch_ == 7u || tilePrefetch_ == 8u || tilePrefetch_ == 10u || tilePrefetch_ == 11u);
        // Mode 7 retains the mode-5 cache behavior.
        // Mode 8 disables output writeback caching through set_cache_notwriteback.
#if !defined(ASCENDC_CPU_DEBUG)
        if (tilePrefetch_ == 7u) {
            yGm_.template SetL2CacheHint<AscendC::CacheRwMode::WRITE>(
                AscendC::CacheMode::CACHE_MODE_NORMAL);
        }
        if (tilePrefetch_ == 8u) {
            yGm_.template SetL2CacheHint<AscendC::CacheRwMode::WRITE>(
                AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
#endif
        // Mode 10 loads a slab as one contiguous block and uses the native packed
        // row pitch for vnchwconv source pointers. Source row addresses may lack
        // 32-byte alignment; this optional mode relies on the intrinsic accepting them.
        const bool repairPacked = (tilePrefetch_ == 11u && kSnap != tileK_);
        const uint32_t padRows=t->rowsPerBatch==2u?2u:4u;
        const bool packedSlab = (tilePrefetch_ == 10u || tilePrefetch_ == 11u);
        const uint32_t slabRows = pair8 ? 8u : 16u;   // tp5/tp7 keep 16-row slabs
        const uint32_t slabBytes = slabRows * kSnap * ds;
        const uint32_t panelBytes = kSnap * mt * ds;  // kSnap rows: vnchwconv may write rows [bigK, kSnap).
        // Mode 6 uses three eight-row slab buffers with wave scheduling.
        const int32_t slabBufs = (t->foldExt==173 || tilePrefetch_ == 1u) ? 1
                                 : (tilePrefetch_ == 6u) ? 3
                                                         : BUFFER_NUM;
        pipe->InitBuffer(slabQue_, slabBufs, slabBytes);
        pipe->InitBuffer(panelQue_, 1, panelBytes);
        LocalTensor<T> padTmp;
        LocalTensor<uint32_t> padIndex;
        if(repairPacked) {
            // The host checks that four-row staging, the panel, and both slabs fit UB.
            pipe->InitBuffer(padTmpBuf_,padRows*kSnap*ds);
            pipe->InitBuffer(padIndexBuf_,padRows*kSnap*4u);
            padTmp=padTmpBuf_.template Get<T>();
            padIndex=padIndexBuf_.template Get<uint32_t>();
            auto idx=padIndex.template ReinterpretCast<int32_t>();
            SetMaskNorm();
            for(uint32_t r=0;r<padRows;++r)
                ArithProgression(idx[r*kSnap],static_cast<int32_t>(r*tileK_*ds),static_cast<int32_t>(ds),kSnap);
            PipeBarrier<PIPE_V>();
        }

        const uint64_t tilesM = (rowLen_ + mt - 1) / mt;
        const uint64_t workAll = rowCount_ * tilesM;
        uint64_t begin;
        uint64_t end;
        uint64_t step = 1;
        if (wave) {
            uint64_t cores = usedCores_ == 0 ? GetBlockNum() : usedCores_;
            if (cores == 0) {
                cores = 1;
            }
            begin = GetBlockIdx();
            end = workAll;
            step = cores;
            if (begin >= end) {
                return;
            }
        } else {
            SplitRange(workAll, begin, end);
            if (begin >= end) {
                return;
            }
        }

        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        const uint32_t loadGap = static_cast<uint32_t>((srcBlkPitch_ - tileK_) * ds);
        const uint32_t storeGap = static_cast<uint32_t>((dstRowStride_ - mt) * ds);
        const uint32_t subSlabs = mt / 16u;
        TransDataTo5HDParams cnv{false, false, static_cast<uint8_t>(kSnap / 8u),
                                 static_cast<uint16_t>(mt), 1};

        uint64_t batch = begin / tilesM;
        OdoSet(batch);
        uint64_t batchFirst = batch * tilesM;

        for (uint64_t unit = begin; unit < end; unit += step) {
            if (unit >= batchFirst + tilesM) {
                batch = unit / tilesM;
                OdoSet(batch);   // wave steps can cross several batches
                batchFirst = batch * tilesM;
            }
            uint64_t i0 = (unit - batchFirst) * mt;
            const bool alignedWide=t->foldExt==173;
            uint32_t width=mt;
            if (i0 + mt > rowLen_) {
                if(alignedWide)width=static_cast<uint32_t>((rowLen_-i0+15)/16*16);
                i0 = rowLen_ - width;
            }
            if(alignedWide) {
                auto first=slabQue_.template AllocTensor<T>();
                DataCopyExtParams in{1,static_cast<uint32_t>(16ull*tileK_*ds),0,0,0};
                DataCopyPad(first,xGm_[odoBase_+i0*srcBlkPitch_],in,pad);slabQue_.EnQue(first);
            }
            LocalTensor<T> panel = panelQue_.template AllocTensor<T>();
            for (uint32_t s = 0; s < width/16; ++s) {
                LocalTensor<T> slab;
                LocalTensor<T> slabB;
                if(alignedWide) {
                    slab=slabQue_.template DeQue<T>();
                } else if (pair8) {
                    DataCopyExtParams inA{8, static_cast<uint32_t>(tileK_ * ds),
                                          loadGap, 0, 0};
                    slab = slabQue_.template AllocTensor<T>();
                    DataCopyPad(slab, xGm_[odoBase_ + (i0 + 16u * s) * srcBlkPitch_],
                                inA, pad);
                    slabQue_.EnQue(slab);
                    slabB = slabQue_.template AllocTensor<T>();
                    DataCopyPad(slabB,
                                xGm_[odoBase_ + (i0 + 16u * s + 8u) * srcBlkPitch_],
                                inA, pad);
                    slabQue_.EnQue(slabB);
                    slab = slabQue_.template DeQue<T>();
                    slabB = slabQue_.template DeQue<T>();
                } else if (packedSlab) {
                    slab = slabQue_.template AllocTensor<T>();
                    // One fat contiguous block: 16 rows x tileK elems, packed.
                    DataCopyExtParams in{1,
                                         static_cast<uint32_t>(16ull * tileK_ * ds), 0,
                                         0, 0};
                    DataCopyPad(slab, xGm_[odoBase_ + (i0 + 16u * s) * srcBlkPitch_], in,
                                pad);
                    slabQue_.EnQue(slab);
                    slab = slabQue_.template DeQue<T>();
                } else {
                    slab = slabQue_.template AllocTensor<T>();
                    DataCopyExtParams in{16, static_cast<uint32_t>(tileK_ * ds), loadGap, 0, 0};
                    DataCopyPad(slab, xGm_[odoBase_ + (i0 + 16u * s) * srcBlkPitch_], in, pad);
                    slabQue_.EnQue(slab);
                    slab = slabQue_.template DeQue<T>();
                }

                if(repairPacked) {
                    const uint32_t count=padRows*kSnap, full=count/64u, tail=count%64u;
                    const CopyRepeatParams cp(1,1,8,8);
                    // Expand from the end like memmove. Stage each group before
                    // writing it; a destination never overlaps an unread group.
                    for(int32_t r=16-static_cast<int32_t>(padRows);r>=0;r-=static_cast<int32_t>(padRows)) {
                        Gather(padTmp,slab,padIndex,static_cast<uint32_t>(r)*tileK_*ds,count);
                        PipeBarrier<PIPE_V>();
                        auto dst=slab[static_cast<uint32_t>(r)*kSnap];
                        if(full) Copy(dst,padTmp,static_cast<uint64_t>(64),static_cast<uint8_t>(full),cp);
                        if(tail) Copy(dst[full*64],padTmp[full*64],static_cast<uint64_t>(tail),static_cast<uint8_t>(1),cp);
                        PipeBarrier<PIPE_V>();
                    }
                }
                uint64_t srcL[16];
                uint64_t dstL[16];
                for (uint32_t r = 0; r < 16u; ++r) {
                    if (pair8) {
                        const LocalTensor<T>& half = (r < 8u) ? slab : slabB;
                        srcL[r] = reinterpret_cast<uint64_t>(
                            half[static_cast<uint32_t>(r & 7u) * kSnap].GetPhyAddr());
                    } else if (packedSlab && !repairPacked) {
                        // Packed pitch is tileK elements; individual source rows may be unaligned.
                        srcL[r] = reinterpret_cast<uint64_t>(slab.GetPhyAddr()) +
                                  static_cast<uint64_t>(r) * tileK_ * ds;
                    } else {
                        srcL[r] = reinterpret_cast<uint64_t>(
                            slab[static_cast<uint32_t>(r) * kSnap].GetPhyAddr());
                    }
                }
                for (uint32_t r = 0; r < 8u; ++r) {
                    const uint64_t rowAddr = reinterpret_cast<uint64_t>(
                        panel[r * mt + 16u * s].GetPhyAddr());
                    dstL[2u * r] = rowAddr;
                    dstL[2u * r + 1u] = rowAddr + DATA_BLOCK_BYTES;
                }
                TransDataTo5HD<T>(dstL, srcL, cnv);
                slabQue_.FreeTensor(slab);
                if(alignedWide && s+1<width/16) {
                    auto next=slabQue_.template AllocTensor<T>();
                    DataCopyExtParams in{1,static_cast<uint32_t>(16ull*tileK_*ds),0,0,0};
                    DataCopyPad(next,xGm_[odoBase_+(i0+16u*(s+1))*srcBlkPitch_],in,pad);slabQue_.EnQue(next);
                }
                if (pair8) {
                    slabQue_.FreeTensor(slabB);
                }
            }
            panelQue_.EnQue(panel);
            panel = panelQue_.template DeQue<T>();
            DataCopyExtParams out{static_cast<uint16_t>(tileK_),
                                  width*ds, (mt-width)*ds/32, static_cast<uint32_t>((dstRowStride_-width)*ds), 0};
            DataCopyPad(yGm_[odoOutBase_ + i0], panel, out);
            panelQue_.FreeTensor(panel);
        }
    }

    __aicore__ inline void SplitRange(uint64_t work, uint64_t& begin, uint64_t& end)
    {
        uint64_t cores = usedCores_;
        if (cores == 0) {
            cores = GetBlockNum();
        }
        if (cores == 0) {
            cores = 1;
        }
        const uint64_t me = GetBlockIdx();
        if (me >= cores) {
            begin = 0;
            end = 0;
            return;
        }
        const uint64_t base = work / cores;
        const uint64_t rem = work % cores;
        if (me < rem) {
            begin = me * (base + 1);
            end = begin + base + 1;
        } else {
            begin = rem * (base + 1) + (me - rem) * base;
            end = begin + base;
        }
    }

    __aicore__ inline void OdoSet(uint64_t row)
    {
        odoBase_ = 0;
        odoOutBase_ = 0;
        for (uint32_t a = 0; a < outerRank_ && a < MAX_ODO; ++a) {
            odoIdx_[a] = 0;
        }
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (ext == 0) {
                continue;
            }
            const uint64_t idx = row % ext;
            row /= ext;
            odoIdx_[a] = idx;
            odoBase_ += idx * outerInStride_[a];
            odoOutBase_ += idx * outerOutStride_[a];
        }
    }

    __aicore__ inline void OdoNext()
    {
        for (int32_t a = static_cast<int32_t>(outerRank_) - 1; a >= 0; --a) {
            const uint64_t ext = outerShape_[a];
            if (odoIdx_[a] + 1 < ext) {
                ++odoIdx_[a];
                odoBase_ += outerInStride_[a];
                odoOutBase_ += outerOutStride_[a];
                return;
            }
            odoBase_ -= odoIdx_[a] * outerInStride_[a];
            odoOutBase_ -= odoIdx_[a] * outerOutStride_[a];
            odoIdx_[a] = 0;
        }
    }

    TBuf<QuePosition::VECCALC> padTmpBuf_,padIndexBuf_;
    TQue<QuePosition::VECIN, BUFFER_NUM> slabQue_;
    TQue<QuePosition::VECOUT, 1> panelQue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    uint64_t numel_;
    uint64_t rowCount_;
    uint64_t rowLen_;
    uint64_t srcBlkPitch_;
    uint64_t tileK_;
    uint64_t dstRowStride_;
    uint32_t tileMT_;
    uint32_t tileKT_;
    uint32_t tilePrefetch_;
    uint32_t outerRank_;
    uint32_t usedCores_;
    uint64_t outerShape_[MAX_ODO];
    uint64_t outerInStride_[MAX_ODO];
    uint64_t outerOutStride_[MAX_ODO];
    uint64_t odoIdx_[MAX_ODO];
    uint64_t odoBase_;
    uint64_t odoOutBase_;
};


// Packed M-slab with dense Gather into double-buffered K slices.
// Every Gather lane contributes to compact output.
template<typename T> class TransposePackedDenseKernel {
public:
 template<typename P> __aicore__ inline void Run(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe) {
  if constexpr(sizeof(T)==4) Impl(x,y,t,pipe);
 }
private:
 template<typename P> __aicore__ inline void Impl(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe) {
  const uint64_t m=t->rowLen,k=t->tileK,dp=t->dstRowStride,batches=t->rowCount;
  const uint32_t mt=t->tileMT,kc=t->tileKT,cores=t->usedCores,n=t->outerRank;
  uint64_t ext[8],si[8],so[8];
  for(uint32_t a=0;a<n;++a){ext[a]=t->outerShape[a];si[a]=t->outerInStride[a];so[a]=t->outerOutStride[a];}
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);
  y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
#if !defined(ASCENDC_CPU_DEBUG)
  x_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
#endif
  pipe->InitBuffer(in_,1,static_cast<uint32_t>((mt*k*4+31)/32*32));
  pipe->InitBuffer(out_,2,mt*kc*4);
  pipe->InitBuffer(index_,mt*kc*4);
  auto index=index_.template Get<uint32_t>();
  auto ix=index.template ReinterpretCast<int32_t>();
  for(uint32_t c=0;c<kc;++c) ArithProgression(ix[c*mt],static_cast<int32_t>(c*4),static_cast<int32_t>(k*4),mt);
  PipeBarrier<PIPE_V>();
  const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
  const uint64_t nm=(m+mt-1)/mt;
  for(uint64_t u=GetBlockIdx();u<batches*nm;u+=cores) {
   uint64_t b=u/nm,ib=0,ob=0;
   for(int a=n-1;a>=0;--a){const uint64_t j=b%ext[a];b/=ext[a];ib+=j*si[a];ob+=j*so[a];}
   const uint64_t m0=(u%nm)*mt;
   const uint32_t mi=static_cast<uint32_t>(MinU64(mt,m-m0));
   auto raw=in_.template AllocTensor<T>();
   DataCopyExtParams load{1,static_cast<uint32_t>(mi*k*4),0,0,0};
   DataCopyPad(raw,x_[ib+m0*k],load,pad);in_.EnQue(raw);raw=in_.template DeQue<T>();
   for(uint32_t k0=0;k0<k;k0+=kc) {
    const uint32_t kk=static_cast<uint32_t>(MinU64(kc,k-k0));
    auto out=out_.template AllocTensor<T>();
    Gather(out,raw,index,k0*4,kk*mt);
    out_.EnQue(out);out=out_.template DeQue<T>();
    const DataCopyExtParams store{static_cast<uint16_t>(kk),mi*4,
     static_cast<uint32_t>((mt-(mi+7)/8*8)/8),static_cast<uint32_t>((dp-mi)*4),0};
    DataCopyPad(y_[ob+k0*dp+m0],out,store);out_.FreeTensor(out);
   }
   in_.FreeTensor(raw);
  }
 }
 TQue<QuePosition::VECIN,1> in_;
 TQue<QuePosition::VECOUT,2> out_;
 TBuf<QuePosition::VECCALC> index_;
 GlobalTensor<T> x_,y_;
};

// Per-core blocked L2 image: [ceil(M/64), K, 64]. Both phases use long stores.
// The phase arenas alias only after all outstanding DMA flags are consumed.
template<typename T> class TransposeDenseBlockedL2Kernel {
public:
 template<typename P> __aicore__ inline void Run(GM_ADDR x,GM_ADDR y,GM_ADDR ws,P t,TPipe* pipe) {
  if constexpr(sizeof(T)==4) Impl(x,y,ws,t,pipe);
 }
private:
 __aicore__ inline void LoadRows(LocalTensor<T> raw,uint64_t ib,uint32_t row,uint32_t m,uint32_t k) {
  const uint32_t real=static_cast<uint32_t>(MinU64(64,m-row));
  DataCopyPad(raw,x_[ib+static_cast<uint64_t>(row)*k],DataCopyExtParams{1,real*k*4,0,0,0},DataCopyPadExtParams<T>{false,0,0,static_cast<T>(0)});
 }
 __aicore__ inline void LoadBlocks(LocalTensor<T> raw,uint32_t c,uint32_t k,uint32_t ng,uint32_t kc) {
  const uint32_t kk=static_cast<uint32_t>(MinU64(kc,k-c));
  DataCopyPad(raw,scratch_[c*64],DataCopyExtParams{static_cast<uint16_t>(ng),kk*256,(k-kk)*256,(kc-kk)*8,0},DataCopyPadExtParams<T>{false,0,0,static_cast<T>(0)});
 }
 template<typename P> __aicore__ inline void Impl(GM_ADDR x,GM_ADDR y,GM_ADDR ws,P t,TPipe* pipe) {
  const uint32_t m=t->rowLen,k=t->tileK,kc=t->tileKT,mp=(m+63)/64*64,ng=mp/64,mb=mp/8,qc=64;
  const uint32_t stage1=(64*k+2*64*qc)*4,packed=(kc*m+7)/8*8,stage2=(2*kc*mp+2*packed)*4;
  const uint32_t arenaBytes=stage1>stage2?stage1:stage2,maskWords=(kc*mp+255)/256*8;
  pipe->InitBuffer(arena_,arenaBytes);pipe->InitBuffer(index1_,64*qc*4);
  pipe->InitBuffer(index2_,kc*mb*4);pipe->InitBuffer(mask_,maskWords*4);
  auto arena=arena_.template Get<T>();auto ix1=index1_.template Get<int32_t>();
  auto ix2=index2_.template Get<uint32_t>();auto mask=mask_.template Get<uint32_t>();
  for(uint32_t c=0;c<qc;++c)ArithProgression(ix1[c*64],static_cast<int32_t>(c*4),static_cast<int32_t>(k*4),64);
  auto tmp=arena.template ReinterpretCast<int32_t>();
  ArithProgression(tmp,0,32,mb);PipeBarrier<PIPE_V>();
  ShiftRight(ix2,tmp.template ReinterpretCast<uint32_t>(),static_cast<uint32_t>(8),mb);PipeBarrier<PIPE_V>();
  Muls(ix2.template ReinterpretCast<int32_t>(),ix2.template ReinterpretCast<int32_t>(),static_cast<int32_t>((kc-1)*256),mb);PipeBarrier<PIPE_V>();
  Add(ix2.template ReinterpretCast<int32_t>(),ix2.template ReinterpretCast<int32_t>(),tmp,mb);PipeBarrier<PIPE_V>();
  for(uint32_t c=1;c<kc;++c)Adds(ix2[c*mb].template ReinterpretCast<int32_t>(),ix2.template ReinterpretCast<int32_t>(),static_cast<int32_t>(c*256),mb);
  Duplicate(mask.template ReinterpretCast<int32_t>(),-1,maskWords);
  SetFlag<HardEvent::V_S>(EVENT_ID0);WaitFlag<HardEvent::V_S>(EVENT_ID0);
  if(m%32)for(uint32_t c=0;c<kc;++c)mask.SetValue((c+1)*mp/32-1,(1u<<(m%32))-1);
  SetFlag<HardEvent::S_V>(EVENT_ID0);WaitFlag<HardEvent::S_V>(EVENT_ID0);
  PipeBarrier<PIPE_ALL>();
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
  scratch_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(GetUserWorkspace(ws))+static_cast<uint64_t>(GetBlockIdx())*mp*k,static_cast<uint64_t>(mp)*k);
#if !defined(ASCENDC_CPU_DEBUG)
  x_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
  y_.template SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
#endif

  for(uint64_t batch=GetBlockIdx();batch<t->rowCount;batch+=t->usedCores) {
   uint64_t b=batch,ib=0,ob=0;
   for(int a=static_cast<int>(t->outerRank)-1;a>=0;--a){const uint64_t j=b%t->outerShape[a];b/=t->outerShape[a];ib+=j*t->outerInStride[a];ob+=j*t->outerOutStride[a];}
   auto denseRaw=arena;bool panelBusy[2]={false,false};
   LoadRows(denseRaw,ib,0,m,k);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
   uint32_t serial=0;
   for(uint32_t g=0;g<ng;++g) {
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    for(uint32_t c0=0;c0<k;c0+=qc,++serial) {
     const uint32_t cc=static_cast<uint32_t>(MinU64(qc,k-c0)),slot=serial%2;
     if(panelBusy[slot])WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(slot));
     auto panel=arena[64*k+slot*64*qc];
     Gather(panel,denseRaw,ix1.template ReinterpretCast<uint32_t>(),c0*4,cc*64);
     SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(slot));WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(slot));
     DataCopyPad(scratch_[static_cast<uint64_t>(g)*k*64+c0*64],panel,DataCopyExtParams{1,cc*256,0,0,0});
     SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(slot));panelBusy[slot]=true;
    }
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if(g+1<ng){LoadRows(denseRaw,ib,(g+1)*64,m,k);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);}
   }
   for(uint32_t slot=0;slot<2;++slot)if(panelBusy[slot])WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(slot));
   PipeBarrier<PIPE_ALL>();
   auto raw=arena;auto ordered=arena[kc*mp];bool busy[2]={false,false};
   LoadBlocks(raw,0,k,ng,kc);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
   for(uint32_t c0=0,step=0;c0<k;c0+=kc,++step) {
    const uint32_t kk=static_cast<uint32_t>(MinU64(kc,k-c0)),slot=step%2;
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    // Host bounds MP<=768, KC=12 and MP%64==0: at most144 repeats.
    Gatherb(ordered.template ReinterpretCast<uint32_t>(),raw.template ReinterpretCast<uint32_t>(),ix2,static_cast<uint8_t>(kk*mb/8),GatherRepeatParams(1,8));
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    if(c0+kc<k){LoadBlocks(raw,c0+kc,k,ng,kc);SetFlag<HardEvent::MTE2_V>(EVENT_ID0);}
    if(busy[slot])WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(slot));
    auto out=arena[2*kc*mp+slot*packed];uint64_t reserved=0;
    GatherMask(out,ordered,mask,true,kk*mp,{1,1,8,0},reserved);
    SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(slot));WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(slot));
    DataCopyPad(y_[ob+static_cast<uint64_t>(c0)*m],out,DataCopyExtParams{1,kk*m*4,0,0,0});
    SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(slot));busy[slot]=true;
   }
   for(uint32_t s=0;s<2;++s)if(busy[s])WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(s));
   PipeBarrier<PIPE_ALL>();
  }
 }
 TBuf<QuePosition::VECCALC> arena_,index1_,index2_,mask_;GlobalTensor<T> x_,y_,scratch_;
};

template<typename T> class TransposeLongGroupKernel {
public:
 template<typename P> __aicore__ inline void Run(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe){if constexpr(sizeof(T)==4)Impl(x,y,t,pipe);}
private:
 template<typename P> __aicore__ inline void Impl(GM_ADDR x,GM_ADDR y,P t,TPipe* pipe){
  const uint64_t bcount=t->rowCount,m=t->rowLen,k=t->tileK,c=t->foldExt,asp=t->foldSrcStride,bsp=t->foldDstRowStride,dp=t->dstRowStride;
  const uint32_t mt=t->tileMT,kt=t->tileKT,cores=t->usedCores,mi=m<mt?m:mt;
  x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
#if !defined(ASCENDC_CPU_DEBUG)
  if(t->tilePrefetch==2)x_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
#endif
  pipe->InitBuffer(in_,1,mt*kt*4);pipe->InitBuffer(out_,1,mt*kt*4);
  const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
  const DataCopyExtParams store{static_cast<uint16_t>(kt),mi*4,static_cast<uint32_t>((mt-(mi+7)/8*8)*4/32),static_cast<uint32_t>((dp-mi)*4),0};
  const TransDataTo5HDParams cv{false,false,static_cast<uint8_t>(kt/8),static_cast<uint16_t>(mt),1};
  const uint64_t nm=(m+mt-1)/mt,nk=(k+kt-1)/kt,per=nm*nk;
  for(uint64_t u=GetBlockIdx();u<bcount*per;u+=cores){
   const uint64_t b=u/per,local=u%per,m0=mt>=m?0:MinU64((local%nm)*mt,m-mt),k0=MinU64((local/nm)*kt,k-kt);
   auto in=in_.template AllocTensor<T>();
   for(uint64_t i=0;i<mi;){
    const uint64_t row=m0+i,a=row/c,j=row%c,len=MinU64(mi-i,c-j);
    const DataCopyExtParams load{static_cast<uint16_t>(len),kt*4,static_cast<uint32_t>((k-kt)*4),0,0};
    DataCopyPad(in[i*kt],x_[a*asp+b*bsp+j*k+k0],load,pad);i+=len;
   }
   in_.EnQue(in);auto out=out_.template AllocTensor<T>();in=in_.template DeQue<T>();
   for(uint32_t i=0;i<mt;i+=16){
    uint64_t src[16],dst[16];for(uint32_t r=0;r<16;++r)src[r]=reinterpret_cast<uint64_t>(in[(i+r)*kt].GetPhyAddr());
    for(uint32_t r=0;r<8;++r){dst[2*r]=reinterpret_cast<uint64_t>(out[r*mt+i].GetPhyAddr());dst[2*r+1]=dst[2*r]+32;}
    TransDataTo5HD<T>(dst,src,cv);
   }
   in_.FreeTensor(in);out_.EnQue(out);out=out_.template DeQue<T>();
   DataCopyPad(y_[k0*dp+b*m+m0],out,store);out_.FreeTensor(out);
  }
 }
 TQue<QuePosition::VECIN,1> in_;TQue<QuePosition::VECOUT,1> out_;GlobalTensor<T> x_,y_;
};

