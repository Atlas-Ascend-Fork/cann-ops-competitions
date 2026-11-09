#ifndef GREATER_PHASE_CACHE_H
#define GREATER_PHASE_CACHE_H

// A fixed-size tile advances the repeated row's phase by TILE%K. When that
// advance is aligned and small, one expanded row segment plus an overlap
// serves every tile of the broadcast batch. Dense DMA remains contiguous.
class KernelGreaterPhaseCache {
    static constexpr uint32_t TILE=8192,CACHED=9216;
    TQue<QuePosition::VECIN,2> denseQ_;
    TQue<QuePosition::VECIN,1> rowQ_;
    TQue<QuePosition::VECOUT,2> outQ_;
    TBuf<QuePosition::VECCALC> cacheB_,indexB_,pairB_,constantB_,maskB_;
    GlobalTensor<int32_t> denseG_,rowG_;
    GlobalTensor<int8_t> outG_;
    LocalTensor<int32_t> cache_,pair_;
    LocalTensor<uint32_t> index_;
    LocalTensor<half> constant_;
    LocalTensor<uint8_t> mask_;
    uint64_t start_,length_,span_;
    uint32_t k_,step_;
    bool leftShared_;

    __aicore__ inline void MakeIndices()
    {
        auto seq=index_.ReinterpretCast<int32_t>();
        auto tmpU=cache_.ReinterpretCast<uint32_t>();
        const int32_t magic=static_cast<int32_t>(((1u<<24)+k_-1)/k_);
        SetMaskNorm();
        ArithProgression(seq,static_cast<int32_t>(0),static_cast<int32_t>(1),CACHED);
        PipeBarrier<PIPE_V>();
        Muls(cache_,seq,magic,CACHED);
        PipeBarrier<PIPE_V>();
        ShiftRight(tmpU,tmpU,static_cast<uint32_t>(24),CACHED);
        PipeBarrier<PIPE_V>();
        Muls(cache_,cache_,static_cast<int32_t>(k_),CACHED);
        PipeBarrier<PIPE_V>();
        Sub(seq,seq,cache_,CACHED);
        PipeBarrier<PIPE_V>();
        ShiftLeft(index_,index_,static_cast<uint32_t>(2),CACHED);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void Prepare(uint64_t g)
    {
        auto raw=rowQ_.AllocTensor<int32_t>();
        DataCopyExtParams p{1,k_*4,0,0,0};
        DataCopyPadExtParams<int32_t> pad;
        DataCopyPad(raw,rowG_[(g/span_)*k_],p,pad);
        rowQ_.EnQue(raw); raw=rowQ_.DeQue<int32_t>();
        PipeBarrier<PIPE_V>();
        Gather(pair_,raw,index_,static_cast<uint32_t>(0),2*k_);
        PipeBarrier<PIPE_V>();
        Gather(cache_,pair_,index_,static_cast<uint32_t>((g%k_)*4),CACHED);
        PipeBarrier<PIPE_V>();
        rowQ_.FreeTensor(raw);
    }
public:
    __aicore__ inline void Init(GM_ADDR x1,GM_ADDR x2,GM_ADDR y,GtTiling t,TPipe* p)
    {
        k_=static_cast<uint32_t>(t->innerLen); step_=TILE%k_;
        span_=static_cast<uint64_t>(t->shape[1])*k_;
        start_=static_cast<uint64_t>(GetBlockIdx())*t->perCore;
        length_=start_<t->totalNum?((t->totalNum-start_<t->perCore)?t->totalNum-start_:t->perCore):0;
        leftShared_=t->strides[2]==0;
        denseG_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(leftShared_?x2:x1));
        rowG_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(leftShared_?x1:x2));
        outG_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));
        denseG_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        if(t->pad0 & 2048u) outG_.template SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
        p->InitBuffer(denseQ_,2,TILE*4); p->InitBuffer(rowQ_,1,RoundUpU32(k_,256)*4);
        p->InitBuffer(outQ_,2,TILE);
        p->InitBuffer(cacheB_,CACHED*4); p->InitBuffer(indexB_,CACHED*4);
        p->InitBuffer(pairB_,RoundUpU32(2*k_,256)*4);
        p->InitBuffer(constantB_,TILE*2); p->InitBuffer(maskB_,TILE/8);
        cache_=cacheB_.Get<int32_t>(); pair_=pairB_.Get<int32_t>(); index_=indexB_.Get<uint32_t>();
        constant_=constantB_.Get<half>(); mask_=maskB_.Get<uint8_t>();
        Duplicate(constant_,(half)0,TILE);
        MakeIndices();
    }
    __aicore__ inline void Process()
    {
        uint64_t g=start_,left=length_,resident=~0ULL;
        uint32_t shift=0;
        while(left) {
            const uint64_t batch=g/span_;
            uint64_t n64=(batch+1)*span_-g;
            if(n64>left) n64=left;
            if(n64>TILE) n64=TILE;
            const uint32_t n=static_cast<uint32_t>(n64),na=RoundUpU32(n,256);
            auto dense=denseQ_.AllocTensor<int32_t>();
            DataCopy(dense,denseG_[g],n);  // host guarantees g and n are multiples of8
            denseQ_.EnQue(dense);
            if(batch!=resident) { Prepare(g); resident=batch; shift=0; }
            dense=denseQ_.DeQue<int32_t>();
            auto shared=cache_[shift];
            if(leftShared_) Min(dense,shared,dense,na);
            else Max(dense,dense,shared,na);
            Compare(mask_,dense,shared,CMPMODE::EQ,na);
            PipeBarrier<PIPE_V>();
            auto selected=dense.ReinterpretCast<half>();
            Select(selected,mask_,constant_,(half)1,SELMODE::VSEL_TENSOR_SCALAR_MODE,na);
            auto out=outQ_.AllocTensor<int8_t>();
            Cast(out,selected,RoundMode::CAST_NONE,na);
            outQ_.EnQue(out); denseQ_.FreeTensor(dense);
            out=outQ_.DeQue<int8_t>();
            if(g%32==0 && n%32==0) DataCopy(outG_[g],out,n);
            else { DataCopyExtParams p{1,n,0,0,0}; DataCopyPad(outG_[g],out,p); }
            outQ_.FreeTensor(out);
            g+=n; left-=n; shift+=step_;
        }
        PipeBarrier<PIPE_ALL>();
    }
};
#endif
