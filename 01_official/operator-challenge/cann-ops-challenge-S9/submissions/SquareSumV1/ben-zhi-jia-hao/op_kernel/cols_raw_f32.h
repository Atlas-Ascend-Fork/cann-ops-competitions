#pragma once

namespace ss_cols_raw_f32 {
using namespace AscendC;
class Kernel {
public:
    __aicore__ inline void Run(GM_ADDR input,GM_ADDR output,GM_ADDR workspace,GM_ADDR tiling)
    {
        auto t=reinterpret_cast<const __gm__ SquareSumV1TilingData*>(tiling);
        const uint32_t cores=t->usedCores;
        const uint32_t chunks=t->redChunks;
        const uint32_t lanes=t->runLanes;
        const uint32_t fold=t->runFold;
        const uint32_t width=static_cast<uint32_t>(t->innerLen);
        const uint32_t row=(width+7U)&~7U;
        const uint64_t rows=t->redPerTile;
        const uint64_t matrices=t->outTiles;
        const uint32_t inputCount=t->ubChunk;
        const uint32_t slab=inputCount/lanes*lanes;
        const uint32_t accLen=t->accLen, outCap=t->outStage;
        const uint32_t accAt=8U*inputCount;
        const uint32_t outAt=accAt+4U*accLen;
        TPipe pipe;
        pipe.InitBuffer(arena_,outAt+4U*outCap+4U*width*fold);
        in0_=arena_.GetWithOffset<float>(inputCount,0);
        in1_=arena_.GetWithOffset<float>(inputCount,4U*inputCount);
        acc_=arena_.GetWithOffset<float>(accLen,accAt);
        tmp_=in0_;
        index_=arena_.GetWithOffset<uint32_t>(width*fold,outAt+4U*outCap);
        outF_=arena_.GetWithOffset<float>(outCap,outAt);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        x_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        ws_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace)));
        Duplicate(outF_,0.0f,row);
        SetFlag<HardEvent::V_MTE2>(0); SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::MTE3_V>(0);
        const uint64_t id=GetBlockIdx();
        if (id==0) {
            WaitFlag<HardEvent::MTE3_V>(0);
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyExtParams clear{1,width*4U,0,0,0};
            DataCopyPad(y_,outF_,clear);
            SetFlag<HardEvent::MTE3_V>(0);
        }
        // Only output initialization must finish here. Unlike SyncAll, this
        // MTE3 notification leaves the two input reads free to overlap it.
        CrossCoreSetFlag<0,PIPE_MTE3>(14);
        bool initialized=false;
        const uint64_t tasks=matrices*chunks;
        const uint64_t lo=tasks*id/cores, hi=tasks*(id+1U)/cores;
        for (uint64_t task=lo;task<hi;++task) {
            const uint64_t matrix=task/chunks;
            const uint32_t chunk=static_cast<uint32_t>(task%chunks);
            const uint64_t total=rows*width;
            const uint64_t base=matrix*total;
            const uint64_t step=static_cast<uint64_t>(slab)*chunks;
            uint64_t pos=static_cast<uint64_t>(chunk)*slab;
            uint32_t current=0;
            if (pos<total) {
                current=static_cast<uint32_t>(total-pos<slab ? total-pos : slab);
                WaitFlag<HardEvent::V_MTE2>(0);
                Copy(in0_,base+pos,current);
                SetFlag<HardEvent::MTE2_V>(0);
                pos+=step;
            }
            BuildIndex(fold,width,width*fold);
            Duplicate(acc_,0.0f,lanes);
            SetMaskCount();
            SetVectorMask<float,MaskMode::COUNTER>(0,lanes);
            uint32_t slot=0;
            while (current) {
                uint32_t next=0;
                if (pos<total) {
                    next=static_cast<uint32_t>(total-pos<slab ? total-pos : slab);
                    if (slot==0) {
                        WaitFlag<HardEvent::V_MTE2>(1); Copy(in1_,base+pos,next);
                        SetFlag<HardEvent::MTE2_V>(1);
                    } else {
                        WaitFlag<HardEvent::V_MTE2>(0); Copy(in0_,base+pos,next);
                        SetFlag<HardEvent::MTE2_V>(0);
                    }
                    pos+=step;
                }
                if (!initialized) {
                    CrossCoreWaitFlag(14);
                    initialized=true;
                }
                if (slot==0) {
                    WaitFlag<HardEvent::MTE2_V>(0); Accumulate(in0_,current,lanes);
                    SetFlag<HardEvent::V_MTE2>(0);
                } else {
                    WaitFlag<HardEvent::MTE2_V>(1); Accumulate(in1_,current,lanes);
                    SetFlag<HardEvent::V_MTE2>(1);
                }
                slot^=1U; current=next;
            }
            SetMaskNorm();
            ResetMask();
            pipe_barrier(PIPE_V);
            Fold(width,lanes,fold);
            WaitFlag<HardEvent::MTE3_V>(0);
            Adds(outF_,acc_,0.0f,width);
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            if (!initialized) { CrossCoreWaitFlag(14); initialized=true; }
            SetAtomicAdd<float>();
            DataCopyExtParams cp{1,width*4U,0,0,0};
            DataCopyPad(y_[matrix*width],outF_,cp);
            SetAtomicNone();
            SetFlag<HardEvent::MTE3_V>(0);
        }
        WaitFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::MTE3_V>(0);
    }
private:
    __aicore__ inline void Copy(LocalTensor<float> dst,uint64_t offset,uint32_t count)
    {
        if ((count&7U)==0) DataCopy(dst,x_[offset],count);
        else {
            DataCopyExtParams cp{1,count*4U,0,0,0};
            DataCopyPadExtParams<float> pad{false,0,0,static_cast<float>(0)};
            DataCopyPad(dst,x_[offset],cp,pad);
        }
    }
    __aicore__ inline void Accumulate(LocalTensor<float> source,uint32_t count,uint32_t lanes)
    {
        for (uint32_t k=0;k<count;k+=lanes) {
            const uint32_t n=count-k<lanes ? count-k : lanes;
            if (n!=lanes) SetVectorMask<float,MaskMode::COUNTER>(0,n);
            const BinaryRepeatParams rp{1,1,1,8,8,8};
            MulAddDst<float,float,false>(acc_,source[k],source[k],uint64_t{0},1,rp);
            pipe_barrier(PIPE_V);
        }
    }
    __aicore__ inline void BuildIndex(uint32_t span,uint32_t stride,uint32_t count)
    {
        auto index=index_.ReinterpretCast<int32_t>();
        auto q=acc_.ReinterpretCast<int32_t>(), r=acc_[count].ReinterpretCast<int32_t>();
        uint32_t shift=0;
        while ((1U<<shift)<span) ++shift;
        ArithProgression<int32_t>(index,0,1,count);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(q,index,shift,count);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(r,q,shift,count);
        pipe_barrier(PIPE_V);
        Sub(index,index,r,count);
        pipe_barrier(PIPE_V);
        Muls<int32_t>(index,index,static_cast<int32_t>(stride),count);
        pipe_barrier(PIPE_V);
        Add(index,index,q,count);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(index,index,2,count);
        pipe_barrier(PIPE_V);
    }
    __aicore__ inline void Fold(uint32_t width,uint32_t lanes,uint32_t fold)
    {
        const uint32_t stop=width*fold;
        for (uint32_t n=lanes>>1U;n>=stop;n>>=1U) {
            Add(acc_,acc_,acc_[n],n);
            pipe_barrier(PIPE_V);
        }
        Gather(tmp_,acc_,index_,0U,stop);
        pipe_barrier(PIPE_V);
        uint32_t off=0;
        while (off+64U<=stop) {
            const uint32_t reps=(stop-off)/64U<255 ? (stop-off)/64U : 255;
            BlockReduceSum<float>(acc_[off/8U],tmp_[off],reps,64,1,1,8);
            off+=reps*64U;
        }
        if (off<stop) BlockReduceSum<float>(acc_[off/8U],tmp_[off],1,stop-off,1,1,8);
        pipe_barrier(PIPE_V);
        uint32_t groups=fold/8U, count=width*groups;
        while (groups>1U) {
            off=0;
            while (off+64U<=count) {
                const uint32_t reps=(count-off)/64U<255 ? (count-off)/64U : 255;
                PairReduceSum<float>(acc_[off/2U],acc_[off],reps,64,1,1,8);
                off+=reps*64U;
            }
            if (off<count) PairReduceSum<float>(acc_[off/2U],acc_[off],1,count-off,1,1,8);
            pipe_barrier(PIPE_V);
            groups>>=1U; count>>=1U;
        }
    }
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<float> x_,y_;
    GlobalTensor<float> ws_;
    LocalTensor<float> in0_,in1_;
    LocalTensor<float> acc_,tmp_,outF_;
    LocalTensor<uint32_t> index_;
};
}
