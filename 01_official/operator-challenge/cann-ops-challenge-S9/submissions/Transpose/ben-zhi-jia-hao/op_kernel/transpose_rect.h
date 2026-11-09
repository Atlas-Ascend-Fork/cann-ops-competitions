// Balanced rectangular panels, independent of the older Gather tile routes.
template<typename T>
class TransposeRectKernel {
public:
    template<typename PtrT>
    __aicore__ inline void Run(GM_ADDR x,GM_ADDR y,PtrT t,TPipe* pipe) {
        if constexpr(sizeof(T)==4)RunImpl(x,y,t,pipe);
    }
private:
    template<typename PtrT>
    __aicore__ inline void RunImpl(GM_ADDR x,GM_ADDR y,PtrT t,TPipe* pipe) {
        const uint64_t batches=t->rowCount,m=t->rowLen,k=t->tileK,sp=t->srcBlkPitch,dp=t->dstRowStride;
        const uint32_t mt=t->tileMT,kt=t->tileKT,mode=t->tilePrefetch,cores=t->usedCores,n=t->outerRank;
        uint64_t ext[8],si[8],so[8];
        for(uint32_t a=0;a<n;++a){ext[a]=t->outerShape[a];si[a]=t->outerInStride[a];so[a]=t->outerOutStride[a];}
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
#if !defined(ASCENDC_CPU_DEBUG)
        if(mode==1||mode==3)x_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        if(mode==3||mode==4)y_.template SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
#endif
        if(!cores||!kt||!mt)return;
        pipe->InitBuffer(in_,2,mt*kt*4);pipe->InitBuffer(out_,1,mt*kt*4);
        const DataCopyExtParams load{static_cast<uint16_t>(mt),kt*4,static_cast<uint32_t>((sp-kt)*4),0,0};
        const DataCopyExtParams store{static_cast<uint16_t>(kt),mt*4,0,static_cast<uint32_t>((dp-mt)*4),0};
        const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
        const TransDataTo5HDParams cv{false,false,static_cast<uint8_t>(kt/8),static_cast<uint16_t>(mt),1};
        const uint64_t nm=(m+mt-1)/mt,nk=(k+kt-1)/kt,perBatch=nm*nk;
        for(uint64_t unit=GetBlockIdx();unit<batches*perBatch;unit+=cores) {
            uint64_t b=unit/perBatch,local=unit-b*perBatch,mi=local%nm,ki=local/nm;
            uint64_t inBase=0,outBase=0;
            for(int a=static_cast<int>(n)-1;a>=0;--a){uint64_t j=b%ext[a];b/=ext[a];inBase+=j*si[a];outBase+=j*so[a];}
            const uint64_t m0=mi*mt+mt>m?m-mt:mi*mt;
            const uint64_t k0=ki*kt+kt>k?k-kt:ki*kt;
            auto in=in_.template AllocTensor<T>();DataCopyPad(in,x_[inBase+m0*sp+k0],load,pad);in_.EnQue(in);
            auto out=out_.template AllocTensor<T>();in=in_.template DeQue<T>();
            for(uint32_t i=0;i<mt;i+=16) {
                uint64_t src[16],dst[16];
                for(uint32_t r=0;r<16;++r)src[r]=reinterpret_cast<uint64_t>(in[(i+r)*kt].GetPhyAddr());
                for(uint32_t r=0;r<8;++r){dst[2*r]=reinterpret_cast<uint64_t>(out[r*mt+i].GetPhyAddr());dst[2*r+1]=dst[2*r]+32;}
                TransDataTo5HD<T>(dst,src,cv);
            }
            in_.FreeTensor(in);out_.EnQue(out);out=out_.template DeQue<T>();
            DataCopyPad(y_[outBase+k0*dp+m0],out,store);out_.FreeTensor(out);
        }
    }
    TQue<QuePosition::VECIN,2> in_;TQue<QuePosition::VECOUT,1> out_;GlobalTensor<T> x_,y_;
};
