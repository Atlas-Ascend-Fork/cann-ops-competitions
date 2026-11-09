// Output panels: all M rows, a strip of K columns, one contiguous output region.
// Input remains cacheable; modes 1/3 bypass the output cache, 2/4 are controls.
template<typename T>
class TransposeOutputKernel {
public:
    template<typename PtrT>
    __aicore__ inline void Run(GM_ADDR x,GM_ADDR y,PtrT t,TPipe* pipe) {
        if constexpr(sizeof(T)==4) RunImpl(x,y,t,pipe);
    }
private:
    template<typename PtrT>
    __aicore__ inline void RunImpl(GM_ADDR x,GM_ADDR y,PtrT t,TPipe* pipe) {
        const uint64_t batches=t->rowCount, m=t->rowLen, k=t->tileK;
        const uint32_t mp=t->tileMT, kt=t->tileKT, mode=t->tilePrefetch;
        const uint32_t cores=t->usedCores;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x),t->numel);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y),t->numel);
#if !defined(ASCENDC_CPU_DEBUG)
        if(mode==1 || mode==3) y_.template SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
#endif
        if(!cores || !kt)return;
        const uint32_t bytes=mp*kt*4;
        pipe->InitBuffer(in_,mode>=3?2:1,bytes);
        pipe->InitBuffer(out_,1,bytes);
        const DataCopyExtParams load{static_cast<uint16_t>(m),kt*4,
                                    static_cast<uint32_t>((k-kt)*4),0,0};
        const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
        const uint32_t roundedM=(m+7)/8*8;
        const DataCopyExtParams store{static_cast<uint16_t>(kt),static_cast<uint32_t>(m*4),
                                     (mp-roundedM)*4/32,0,0};
        const TransDataTo5HDParams cv{false,false,static_cast<uint8_t>(kt/8),
                                     static_cast<uint16_t>(mp),1};
        const uint64_t tiles=(k+kt-1)/kt;
        for(uint64_t b=GetBlockIdx();b<batches;b+=cores) {
            const uint64_t base=b*m*k;
            for(uint64_t tile=0;tile<tiles;++tile) {
                const uint64_t k0=tile*kt+kt>k ? k-kt : tile*kt;
                auto in=in_.template AllocTensor<T>();
                DataCopyPad(in,x_[base+k0],load,pad);
                in_.EnQue(in);
                auto out=out_.template AllocTensor<T>();
                in=in_.template DeQue<T>();
                for(uint32_t i=0;i<mp;i+=16) {
                    uint64_t src[16],dst[16];
                    for(uint32_t r=0;r<16;++r)
                        src[r]=reinterpret_cast<uint64_t>(in[(i+r)*kt].GetPhyAddr());
                    for(uint32_t r=0;r<8;++r) {
                        dst[2*r]=reinterpret_cast<uint64_t>(out[r*mp+i].GetPhyAddr());
                        dst[2*r+1]=dst[2*r]+32;
                    }
                    TransDataTo5HD<T>(dst,src,cv);
                }
                in_.FreeTensor(in);
                out_.EnQue(out);
                out=out_.template DeQue<T>();
                DataCopyPad(y_[base+k0*m],out,store);
                out_.FreeTensor(out);
            }
        }
    }
    TQue<QuePosition::VECIN,2> in_;
    TQue<QuePosition::VECOUT,1> out_;
    GlobalTensor<T> x_,y_;
};
