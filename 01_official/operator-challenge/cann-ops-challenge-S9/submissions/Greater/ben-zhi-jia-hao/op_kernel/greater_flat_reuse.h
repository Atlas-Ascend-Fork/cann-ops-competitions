#ifndef GREATER_FLAT_REUSE_H
#define GREATER_FLAT_REUSE_H

// Flat fp32: after Compare consumes both inputs, reuse x1 for half Select and
// x2 for bool output. TQueBind carries the MTE3 -> next MTE2 reuse dependency.
class KernelGreaterFlatReuse {
    TQue<QuePosition::VECIN,2> xQ_;
    TQueBind<QuePosition::VECIN,QuePosition::VECOUT,2> yQ_;
    TBuf<QuePosition::VECCALC> onesB_,maskB_;
    GlobalTensor<float> xG_,yG_;
    GlobalTensor<int8_t> outG_;
    LocalTensor<half> ones_;
    LocalTensor<uint8_t> mask_;
    uint64_t start_,length_;
    uint32_t tile_;
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,GM_ADDR out,GtTiling t,TPipe* pipe)
    {
        start_=static_cast<uint64_t>(GetBlockIdx())*t->perCore;
        length_=start_<t->totalNum?((t->totalNum-start_<t->perCore)?t->totalNum-start_:t->perCore):0;
        xG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        yG_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        outG_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(out));
        xG_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        yG_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        if(t->pad0 & 2048u) outG_.template SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
        const uint32_t slots=(t->pad0 & 128u)?1:2;
        const uint32_t avail=t->ubSize-8192,bytes=8*slots+2;
        tile_=(avail/bytes/512)*512;
        while(tile_*bytes+RoundUpU32((tile_+7)/8,256)>avail) tile_-=512;
        pipe->InitBuffer(xQ_,slots,tile_*4);
        pipe->InitBuffer(yQ_,slots,tile_*4);
        pipe->InitBuffer(onesB_,tile_*2);
        pipe->InitBuffer(maskB_,RoundUpU32((tile_+7)/8,256));
        ones_=onesB_.Get<half>(); mask_=maskB_.Get<uint8_t>();
        Duplicate(ones_,(half)1,tile_);
    }
    __aicore__ inline void Process()
    {
        for(uint64_t off=0;off<length_;off+=tile_) {
            const uint64_t g=start_+off;
            const uint32_t n=static_cast<uint32_t>(length_-off<tile_?length_-off:tile_);
            const uint32_t na=RoundUpU32(n,256);
            auto x=xQ_.AllocTensor<float>();
            auto y=yQ_.AllocTensor<float>();
            if(g%8==0 && n%8==0) {
                DataCopy(x,xG_[g],n); DataCopy(y,yG_[g],n);
            } else {
                DataCopyExtParams p{1,n*4,0,0,0};
                DataCopyPadExtParams<float> pad;
                DataCopyPad(x,xG_[g],p,pad); DataCopyPad(y,yG_[g],p,pad);
            }
            xQ_.EnQue(x);
            yQ_.EnQue<QuePosition::GM,QuePosition::VECIN,float>(y);
            x=xQ_.DeQue<float>();
            y=yQ_.DeQue<QuePosition::GM,QuePosition::VECIN,float>();
            Compare(mask_,x,y,CMPMODE::GT,na);
            PipeBarrier<PIPE_V>();
            auto selected=x.ReinterpretCast<half>();
            auto result=y.ReinterpretCast<int8_t>();
            Select(selected,mask_,ones_,(half)0,SELMODE::VSEL_TENSOR_SCALAR_MODE,na);
            Cast(result,selected,RoundMode::CAST_NONE,na);
            xQ_.FreeTensor(x);
            yQ_.EnQue<QuePosition::VECOUT,QuePosition::GM,int8_t>(result);
            result=yQ_.DeQue<QuePosition::VECOUT,QuePosition::GM,int8_t>();
            if(g%32==0 && n%32==0) DataCopy(outG_[g],result,n);
            else {
                DataCopyExtParams p{1,n,0,0,0};
                DataCopyPad(outG_[g],result,p);
            }
            yQ_.FreeTensor(result);
        }
        PipeBarrier<PIPE_ALL>();
    }
};
#endif
