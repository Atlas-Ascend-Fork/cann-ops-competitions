#ifndef GREATER_REPEAT_H
#define GREATER_REPEAT_H
#include "lib/index/arithprogression.h"
#ifndef GT_REPEAT_INDEX_SCALAR
#define GT_REPEAT_INDEX_SCALAR 0
#endif

// Dense [B,R,K] versus [B,1,K], in either operand order. Each transfer covers
// several contiguous rows; the shared row stays resident, widened once for bf16.
template <typename T, bool FIXED = false>
class KernelGreaterRepeat {
    static constexpr bool FP = GtTraits<T>::kNeedCast;
    using CT = typename GtTraits<T>::CT;
    static constexpr uint32_t CAP = FP ? 7680 : 8192;
    static constexpr uint32_t MASK = ((CAP/8 + 255)/256)*256;
    TQue<QuePosition::VECIN, 2> denseQ_;
    TQue<QuePosition::VECIN, 1> rowQ_;
    TQue<QuePosition::VECOUT, 2> outQ_;
    TBuf<QuePosition::VECCALC> packedB_, indexB_, scratchB_, selectB_, constantB_, maskB_, rowFloatB_, skewB_, doubleRowB_;
    GlobalTensor<T> denseG_, rowG_;
    GlobalTensor<int8_t> outG_;
    LocalTensor<CT> packed_, scratch_, row_, doubleRow_;
    LocalTensor<uint32_t> index_;
    LocalTensor<half> select_, constant_;
    LocalTensor<uint8_t> mask_;
    LocalTensor<T> heldRaw_;
    uint64_t start_, length_, span_,total_,gridStride_;
    uint32_t k_, paddedK_, tile_;
    bool leftShared_, alignedCopy_,keepPacked_,roundRobin_;

    __aicore__ inline void Load(TQue<QuePosition::VECIN,2>& q, const GlobalTensor<T>& gm,
                               uint64_t off, uint32_t n)
    {
        auto t=q.template AllocTensor<T>();
        DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad;
        if constexpr(FIXED) {
            if(alignedCopy_ && ((off*sizeof(T))%32==0) && ((n*sizeof(T))%32==0)) DataCopy(t,gm[off],n);
            else DataCopyPad(t,gm[off],p,pad);
        } else DataCopyPad(t,gm[off],p,pad);
        q.EnQue(t);
    }
    __aicore__ inline void LoadRow(uint64_t off,uint32_t n)
    {
        auto t=rowQ_.template AllocTensor<T>();
        DataCopyExtParams p{1,static_cast<uint32_t>(n*sizeof(T)),0,0,0};
        DataCopyPadExtParams<T> pad;
        DataCopyPad(t,rowG_[off],p,pad);
        rowQ_.EnQue(t);
    }
    __aicore__ inline void Store(uint64_t off,uint32_t n)
    {
        auto t=outQ_.template DeQue<int8_t>();
        DataCopyExtParams p{1,n,0,0,0};
        if constexpr(FIXED) {
            if(alignedCopy_ && off%32==0 && n%32==0) DataCopy(outG_[off],t,n);
            else DataCopyPad(outG_[off],t,p);
        } else DataCopyPad(outG_[off],t,p);
        outQ_.FreeTensor(t);
    }
    __aicore__ inline void CompareAndPack(const LocalTensor<CT>& repeated,
                                         const LocalTensor<CT>& dense,uint32_t n)
    {
        auto a=leftShared_?repeated:dense;
        auto b=leftShared_?dense:repeated;
        if constexpr(FP) {
            Compare(mask_,a,b,CMPMODE::GT,n);
            Select(select_,mask_,constant_,(half)0,SELMODE::VSEL_TENSOR_SCALAR_MODE,n);
        } else {
            if(keepPacked_) {
                if(leftShared_) Min(dense,repeated,dense,n);
                else Max(dense,dense,repeated,n);
                Compare(mask_,dense,repeated,CMPMODE::EQ,n);
            } else {
                // Only x1 is needed after Min, so overwrite x2's local copy.
                Min(b,a,b,n);
                Compare(mask_,b,a,CMPMODE::EQ,n);
            }
            // The mask means <=. Select 0 for true and 1 otherwise: no Not pass.
            Select(select_,mask_,constant_,(half)1,SELMODE::VSEL_TENSOR_SCALAR_MODE,n);
        }
        auto o=outQ_.template AllocTensor<int8_t>();
        Cast(o,select_,RoundMode::CAST_NONE,n);
        outQ_.EnQue(o);
    }
    __aicore__ inline void MakeIndices()
    {
#if GT_REPEAT_INDEX_SCALAR
        uint32_t byteOffset=0;
        for(uint32_t i=0;i<CAP;++i) {
            index_.SetValue(i,byteOffset);
            byteOffset+=4;
            if(byteOffset==k_*4) byteOffset=0;
        }
        const auto e=GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(e);
        WaitFlag<HardEvent::S_V>(e);
#else
        if(k_<256) {
            uint32_t off=0;
            for(uint32_t i=0;i<CAP;++i) { index_.SetValue(i,off); off+=4; if(off==k_*4) off=0; }
            const auto e=GetTPipePtr()->FetchEventID(HardEvent::S_V);
            SetFlag<HardEvent::S_V>(e); WaitFlag<HardEvent::S_V>(e);
            return;
        }
        // ceil(2^24/K) gives exact quotients on 0..8191 for K=256..2048.
        // The largest product is below 2^29, so signed int32 never overflows.
        auto seq=index_.template ReinterpretCast<int32_t>();
        auto tmp=packed_.template ReinterpretCast<int32_t>();
        auto tmpU=tmp.template ReinterpretCast<uint32_t>();
        const int32_t magic=static_cast<int32_t>(((1u<<24)+k_-1)/k_);
        SetMaskNorm();
        ArithProgression(seq,static_cast<int32_t>(0),static_cast<int32_t>(1),CAP);
        PipeBarrier<PIPE_V>();
        Muls(tmp,seq,magic,CAP);
        PipeBarrier<PIPE_V>();
        ShiftRight(tmpU,tmpU,static_cast<uint32_t>(24),CAP);
        PipeBarrier<PIPE_V>();
        Muls(tmp,tmp,static_cast<int32_t>(k_),CAP);
        PipeBarrier<PIPE_V>();
        Sub(seq,seq,tmp,CAP);
        PipeBarrier<PIPE_V>();
        ShiftLeft(index_,index_,static_cast<uint32_t>(2),CAP);
        PipeBarrier<PIPE_V>();
#endif
    }
    __aicore__ inline void ProcessFixed()
    {
        uint64_t g=start_,left=length_,resident=~0ULL;
        if(roundRobin_) { g=static_cast<uint64_t>(GetBlockIdx())*CAP; left=g<total_?total_-g:0; }
        while(left) {
            const uint32_t n=static_cast<uint32_t>(left<CAP?left:CAP);
            Load(denseQ_,denseG_,g,n);
            uint32_t written=0;
            while(written<n) {
                const uint64_t here=g+written,batch=here/span_;
                if(batch!=resident) {
                    LoadRow(batch*k_,k_);
                    auto raw=rowQ_.template DeQue<T>();
                    // Finish readers of the old pair before replacing it.
                    PipeBarrier<PIPE_V>();
                    Gather(doubleRow_,raw,index_,static_cast<uint32_t>(0),2*k_);
                    PipeBarrier<PIPE_V>();
                    rowQ_.FreeTensor(raw);
                    resident=batch;
                }
                const uint64_t remain=(batch+1)*span_-here;
                const uint32_t count=static_cast<uint32_t>(remain<n-written?remain:n-written);
                // A doubled row permits any starting phase with one fixed
                // periodic index table. span%8 keeps each destination aligned.
                Gather(packed_[written],doubleRow_,index_,static_cast<uint32_t>((here%k_)*4),count);
                written+=count;
            }
            auto dense=denseQ_.template DeQue<T>();
            CompareAndPack(packed_,dense,RoundUpU32(n,256));
            denseQ_.FreeTensor(dense);
            Store(g,n);
            if(roundRobin_) { g+=gridStride_; left=g<total_?total_-g:0; }
            else { g+=n; left-=n; }
        }
        PipeBarrier<PIPE_ALL>();
    }
public:
    __aicore__ inline void Init(GM_ADDR x1,GM_ADDR x2,GM_ADDR y,GtTiling t,TPipe* p)
    {
        k_=static_cast<uint32_t>(t->innerLen);
        paddedK_=RoundUpU32(k_,256);
        tile_=(CAP/k_)*k_;
        span_=static_cast<uint64_t>(t->shape[1])*k_;
        start_=static_cast<uint64_t>(GetBlockIdx())*t->perCore;
        length_=start_<t->totalNum ? ((t->totalNum-start_<t->perCore)?t->totalNum-start_:t->perCore) : 0;
        leftShared_=t->strides[2]==0;
        alignedCopy_=(t->pad0 & 1024u)!=0;
        keepPacked_=(t->pad0 & 65536u)!=0;
        if constexpr(FIXED) {
            roundRobin_=(t->pad0 & 131072u)!=0;
            total_=t->totalNum; gridStride_=static_cast<uint64_t>(t->usedCores)*CAP;
        }
        denseG_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(leftShared_?x2:x1));
        rowG_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(leftShared_?x1:x2));
        if(t->pad0 & 2u) {
            denseG_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_DISABLE);
        }
        outG_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y));
        if(t->pad0 & 2048u) outG_.template SetL2CacheHint<CacheRwMode::WRITE>(CacheMode::CACHE_MODE_DISABLE);
        p->InitBuffer(denseQ_,2,CAP*sizeof(T));
        p->InitBuffer(rowQ_,1,paddedK_*sizeof(T));
        p->InitBuffer(outQ_,FIXED?1:2,CAP);
        if constexpr(!FP) { if(t->pad0 & 64u) p->InitBuffer(skewB_,256); }
        p->InitBuffer(packedB_,CAP*4);
        p->InitBuffer(indexB_,CAP*4);
        if constexpr(FP) p->InitBuffer(scratchB_,CAP*4);
        p->InitBuffer(selectB_,CAP*2);
        p->InitBuffer(constantB_,CAP*2);
        p->InitBuffer(maskB_,MASK);
        if constexpr(FP) p->InitBuffer(rowFloatB_,paddedK_*4);
        if constexpr(FIXED) {
            p->InitBuffer(doubleRowB_,RoundUpU32(2*k_,256)*4);
            doubleRow_=doubleRowB_.template Get<CT>();
        }
        packed_=packedB_.template Get<CT>();
        if constexpr(FP) scratch_=scratchB_.template Get<CT>();
        index_=indexB_.template Get<uint32_t>();
        select_=selectB_.template Get<half>();
        constant_=constantB_.template Get<half>();
        mask_=maskB_.template Get<uint8_t>();
        Duplicate(constant_,FP?(half)1:(half)0,CAP);
        MakeIndices();
    }
    __aicore__ inline void Process()
    {
        if constexpr(FIXED) { ProcessFixed(); return; }
        uint64_t g=start_,left=length_;
        if(left==0) return;
        const uint32_t pos=static_cast<uint32_t>(g%k_);
        // Keep the original 32-element core partition. One partial first row
        // is read directly; all subsequent pieces start on a row boundary.
        if(pos) {
            const uint32_t n=static_cast<uint32_t>(left<k_-pos?left:k_-pos);
            const uint32_t na=RoundUpU32(n,256);
            Load(denseQ_,denseG_,g,n);
            LoadRow((g/span_)*k_+pos,n);
            auto d=denseQ_.template DeQue<T>();
            auto r=rowQ_.template DeQue<T>();
            if constexpr(FP) {
                Cast(packed_,r,RoundMode::CAST_NONE,na);
                Cast(scratch_,d,RoundMode::CAST_NONE,na);
                CompareAndPack(packed_,scratch_,na);
            } else {
                CompareAndPack(r,d,na);
            }
            denseQ_.FreeTensor(d);
            rowQ_.FreeTensor(r);
            Store(g,n);
            g+=n; left-=n;
        }
        uint64_t resident=~0ULL;
        bool packedReady=false;
        while(left) {
            const uint64_t batch=g/span_;
            if(batch!=resident) {
                if constexpr(!FP) { if(resident!=~0ULL) rowQ_.FreeTensor(heldRaw_); }
                LoadRow(batch*k_,k_);
                heldRaw_=rowQ_.template DeQue<T>();
                if constexpr(FP) {
                    row_=rowFloatB_.template Get<CT>();
                    Cast(row_,heldRaw_,RoundMode::CAST_NONE,paddedK_);
                    rowQ_.FreeTensor(heldRaw_);
                } else row_=heldRaw_;
                resident=batch;
                packedReady=false;
            }
            uint64_t n64=(batch+1)*span_-g;
            if(n64>left) n64=left;
            if(n64>tile_) n64=tile_;
            const uint32_t n=static_cast<uint32_t>(n64),na=RoundUpU32(n,256);
            Load(denseQ_,denseG_,g,n);
            auto d=denseQ_.template DeQue<T>();
            if(!keepPacked_ || !packedReady) {
                Gather(packed_,row_,index_,static_cast<uint32_t>(0),keepPacked_?CAP:na);
                packedReady=true;
            }
            if constexpr(FP) {
                Cast(scratch_,d,RoundMode::CAST_NONE,na);
                CompareAndPack(packed_,scratch_,na);
            } else CompareAndPack(packed_,d,na);
            denseQ_.FreeTensor(d);
            Store(g,n);
            g+=n; left-=n;
        }
        if constexpr(!FP) { if(resident!=~0ULL) rowQ_.FreeTensor(heldRaw_); }
        PipeBarrier<PIPE_ALL>();
    }
};
#endif
