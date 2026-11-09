#ifndef GREATER_ROW_CACHED_H
#define GREATER_ROW_CACHED_H

// [B,R,K] versus [B,1,K]. Expand a padded shared row once per B; keep it
// unchanged across all chunks. Dense input is the temporary Select storage.
template<typename T>
class KernelGreaterRowCached {
    static constexpr bool FP=GtTraits<T>::kNeedCast;
    static constexpr uint32_t CAP=12032;
    using CT=typename GtTraits<T>::CT;
    TQue<QuePosition::VECIN,2> denseQ_;
    TQue<QuePosition::VECIN,1> rowQ_;
    TQue<QuePosition::VECOUT,1> outQ_;
    TBuf<QuePosition::VECCALC> sharedB_,scratchB_,constantB_,maskB_;
    GlobalTensor<T> denseG_,rowG_;
    GlobalTensor<int8_t> outG_;
    LocalTensor<CT> shared_,scratch_;
    LocalTensor<half> constant_;
    LocalTensor<uint8_t> mask_;
    uint64_t start_,length_,span_;
    uint32_t k_,pitch_,maxRows_,gap_;
    bool leftShared_,prefetch_;

    __aicore__ inline void LoadDense(uint64_t g,uint32_t rows,uint32_t n)
    {
        auto t=denseQ_.template AllocTensor<T>();
        DataCopyExtParams p{static_cast<uint16_t>(rows),static_cast<uint32_t>(n*sizeof(T)),0,rows>1?gap_:0,0};
        DataCopyPadExtParams<T> pad;
        DataCopyPad(t,denseG_[g],p,pad);
        denseQ_.EnQue(t);
    }
    __aicore__ inline void LoadRow(uint64_t off,uint32_t n)
    {
        auto t=rowQ_.template AllocTensor<T>();
        DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad;
        DataCopyPad(t,rowG_[off],p,pad);
        rowQ_.EnQue(t);
    }
    __aicore__ inline void Compute(const LocalTensor<T>& dense,uint32_t n)
    {
        if constexpr(FP) {
            Cast(scratch_,dense,RoundMode::CAST_NONE,n);
            if(leftShared_) Compare(mask_,shared_,scratch_,CMPMODE::GT,n);
            else Compare(mask_,scratch_,shared_,CMPMODE::GT,n);
        } else {
            // Preserve shared_: the equality mask is x1<=x2 in either order.
            if(leftShared_) Min(dense,shared_,dense,n);
            else Max(dense,dense,shared_,n);
            Compare(mask_,dense,shared_,CMPMODE::EQ,n);
        }
        PipeBarrier<PIPE_V>();
        auto selected=dense.template ReinterpretCast<half>();
        Select(selected,mask_,constant_,FP?(half)0:(half)1,SELMODE::VSEL_TENSOR_SCALAR_MODE,n);
        auto out=outQ_.template AllocTensor<int8_t>();
        Cast(out,selected,RoundMode::CAST_NONE,n);
        outQ_.EnQue(out);
    }
    __aicore__ inline void Store(uint64_t g,uint32_t rows,uint32_t n)
    {
        auto out=outQ_.template DeQue<int8_t>();
        DataCopyExtParams p{static_cast<uint16_t>(rows),n,0,0,0};
        DataCopyPad(outG_[g],out,p);
        outQ_.FreeTensor(out);
    }
    __aicore__ inline void Direct(uint64_t g,uint32_t n)
    {
        LoadDense(g,1,n);
        LoadRow((g/span_)*k_+g%k_,n);
        auto row=rowQ_.template DeQue<T>();
        const uint32_t na=RoundUpU32(n,256);
        PipeBarrier<PIPE_V>();
        if constexpr(FP) Cast(shared_,row,RoundMode::CAST_NONE,na);
        else DataCopy(shared_,row,na);
        rowQ_.FreeTensor(row);
        auto dense=denseQ_.template DeQue<T>();
        Compute(dense,na);
        denseQ_.FreeTensor(dense);
        Store(g,1,n);
    }
    __aicore__ inline void PrepareShared(uint64_t batch)
    {
        LoadRow(batch*k_,k_);
        auto row=rowQ_.template DeQue<T>();
        PipeBarrier<PIPE_V>();
        if constexpr(FP) Cast(shared_,row,RoundMode::CAST_NONE,pitch_);
        else DataCopy(shared_,row,pitch_);
        PipeBarrier<PIPE_V>();
        // Each instruction copies one vector slice into all other row slots.
        // Host limits pitch*4/32 to the supported repeat-stride range.
        for(uint32_t off=0;off<pitch_;off+=64) {
            const uint32_t m=pitch_-off<64?pitch_-off:64;
            Copy(shared_[pitch_+off],shared_[off],static_cast<uint64_t>(m),
                 static_cast<uint8_t>(maxRows_-1),
                 CopyRepeatParams(1,1,static_cast<uint16_t>(pitch_/8),0));
        }
        rowQ_.FreeTensor(row);
    }
public:
    __aicore__ inline void Init(GM_ADDR x1,GM_ADDR x2,GM_ADDR y,GtTiling t,TPipe* p)
    {
        k_=static_cast<uint32_t>(t->innerLen); pitch_=RoundUpU32(k_,32);
        maxRows_=CAP/pitch_; gap_=(pitch_*sizeof(T)-RoundUpU32(k_*sizeof(T),32))/32;
        span_=static_cast<uint64_t>(t->shape[1])*k_;
        start_=static_cast<uint64_t>(GetBlockIdx())*t->perCore;
        length_=start_<t->totalNum?((t->totalNum-start_<t->perCore)?t->totalNum-start_:t->perCore):0;
        leftShared_=t->strides[2]==0;
        prefetch_=(t->pad0 & 16384u)!=0;
        denseG_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(leftShared_?x2:x1));
        rowG_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(leftShared_?x1:x2));
        outG_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));
        if(t->pad0 & 2u) denseG_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        if(t->pad0 & 2048u) outG_.template SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
        p->InitBuffer(denseQ_,2,CAP*sizeof(T));
        p->InitBuffer(rowQ_,1,RoundUpU32(k_,256)*sizeof(T));
        p->InitBuffer(outQ_,1,CAP);
        p->InitBuffer(sharedB_,CAP*4);
        if constexpr(FP) p->InitBuffer(scratchB_,CAP*4);
        p->InitBuffer(constantB_,CAP*2);
        p->InitBuffer(maskB_,RoundUpU32(CAP/8,256));
        shared_=sharedB_.template Get<CT>();
        if constexpr(FP) scratch_=scratchB_.template Get<CT>();
        constant_=constantB_.template Get<half>(); mask_=maskB_.template Get<uint8_t>();
        Duplicate(constant_,FP?(half)1:(half)0,CAP);
    }
    __aicore__ inline void Process()
    {
        uint64_t g=start_,left=length_,resident=~0ULL;
        if(!left) return;
        const uint32_t pos=g%k_;
        if(pos) {
            const uint32_t n=left<k_-pos?left:k_-pos;
            Direct(g,n); g+=n; left-=n;
        }
        bool prefetched=false;
        while(left>=k_) {
            const uint64_t batch=g/span_;
            uint32_t rows=static_cast<uint32_t>(((batch+1)*span_-g)/k_);
            if(rows>maxRows_) rows=maxRows_;
            if(rows>left/k_) rows=static_cast<uint32_t>(left/k_);
            if(!prefetched) LoadDense(g,rows,k_);
            if(batch!=resident) { PrepareShared(batch); resident=batch; }
            const uint32_t n=rows*k_;
            prefetched=prefetch_ && left-n>=k_;
            if(prefetched) {
                const uint64_t next=g+n;
                uint32_t nr=static_cast<uint32_t>(((next/span_+1)*span_-next)/k_);
                if(nr>maxRows_) nr=maxRows_;
                if(nr>(left-n)/k_) nr=static_cast<uint32_t>((left-n)/k_);
                LoadDense(next,nr,k_);
            }
            auto dense=denseQ_.template DeQue<T>();
            Compute(dense,RoundUpU32(rows*pitch_,256));
            denseQ_.FreeTensor(dense);
            Store(g,rows,k_);
            g+=n; left-=n;
        }
        if(left) Direct(g,static_cast<uint32_t>(left));
        PipeBarrier<PIPE_ALL>();
    }
};
#endif
