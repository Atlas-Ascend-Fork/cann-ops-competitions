#pragma once

namespace ss_cols_raw_bf16 {
using namespace AscendC;
class Kernel {
public:
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
    {
        auto t = reinterpret_cast<const __gm__ SquareSumV1TilingData*>(tiling);
        const uint32_t cores=t->usedCores, chunks=t->redChunks;
        const uint32_t lanes=t->runLanes, width=static_cast<uint32_t>(t->innerLen);
        const uint32_t row=(width+15U)&~15U, inputCount=t->ubChunk;
        const uint32_t slab=inputCount/lanes*lanes, math=t->tmpLen;
        const uint64_t total=t->redPerTile*width, matrices=t->outTiles;
        const uint32_t accAt=4U*inputCount;
        const uint32_t tmpAt=accAt+4U*t->accLen;
        const uint32_t idxAt=tmpAt+4U*t->tmpLen;
        const uint32_t outAt=idxAt+4U*t->workLen;
        TPipe pipe;
        pipe.InitBuffer(arena_,outAt+4U*t->outStage);
        in0_=arena_.GetWithOffset<bfloat16_t>(inputCount,0);
        in1_=arena_.GetWithOffset<bfloat16_t>(inputCount,2U*inputCount);
        acc_=arena_.GetWithOffset<float>(t->accLen,accAt);
        tmp_=arena_.GetWithOffset<float>(t->tmpLen,tmpAt);
        index_=arena_.GetWithOffset<uint32_t>(t->workLen,idxAt);
        out_=arena_.GetWithOffset<float>(t->outStage,outAt);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(input));
        x_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
        ws_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace)));
        Duplicate(out_,0.0f,row);
        SetFlag<HardEvent::V_MTE2>(0); SetFlag<HardEvent::V_MTE2>(1);
        SetFlag<HardEvent::MTE3_V>(0);
        const uint64_t id=GetBlockIdx(), tasks=matrices*chunks;
        const uint64_t lo=tasks*id/cores, hi=tasks*(id+1U)/cores;
        bool indexReady=false;
        uint32_t carryCount=0, carrySlot=0;
        for (uint64_t task=lo;task<hi;++task) {
            const uint64_t matrix=task/chunks, base=matrix*total;
            const uint32_t chunk=static_cast<uint32_t>(task%chunks);
            const uint64_t step=static_cast<uint64_t>(slab)*chunks;
            uint64_t pos=static_cast<uint64_t>(chunk)*slab;
            uint32_t current=carryCount, slot=carrySlot;
            if (carryCount) {
                pos+=step;
                carryCount=0;
            } else if (pos<total) {
                slot=0;
                current=static_cast<uint32_t>(total-pos<slab ? total-pos : slab);
                WaitFlag<HardEvent::V_MTE2>(0);
                Copy(in0_,base+pos,current);
                SetFlag<HardEvent::MTE2_V>(0);
                pos+=step;
            }
            if (!indexReady) { BuildIndex(width); indexReady=true; }
            Duplicate(acc_,0.0f,lanes);
            SetMaskCount();
            SetVectorMask<float,MaskMode::COUNTER>(0,math);
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
                } else if (task+1U<hi) {
                    const uint64_t futureTask=task+1U;
                    const uint64_t futurePos=(futureTask%chunks)*slab;
                    if (futurePos<total) {
                        const uint64_t futureBase=(futureTask/chunks)*total;
                        carryCount=static_cast<uint32_t>(total-futurePos<slab ? total-futurePos : slab);
                        carrySlot=slot^1U;
                        if (carrySlot==0) {
                            WaitFlag<HardEvent::V_MTE2>(0); Copy(in0_,futureBase+futurePos,carryCount);
                            SetFlag<HardEvent::MTE2_V>(0);
                        } else {
                            WaitFlag<HardEvent::V_MTE2>(1); Copy(in1_,futureBase+futurePos,carryCount);
                            SetFlag<HardEvent::MTE2_V>(1);
                        }
                    }
                }
                if (slot==0) {
                    WaitFlag<HardEvent::MTE2_V>(0); Accumulate(in0_,current,lanes,math);
                    SetFlag<HardEvent::V_MTE2>(0);
                } else {
                    WaitFlag<HardEvent::MTE2_V>(1); Accumulate(in1_,current,lanes,math);
                    SetFlag<HardEvent::V_MTE2>(1);
                }
                slot^=1U; current=next;
            }
            SetMaskNorm(); ResetMask();
            Fold(width,lanes);
            WaitFlag<HardEvent::MTE3_V>(0);
            Adds(out_,acc_,0.0f,width);
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            DataCopy(ws_[(static_cast<uint64_t>(chunk)*matrices+matrix)*row],out_,row);
            SetFlag<HardEvent::MTE3_V>(0);
        }
        WaitFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::V_MTE2>(1);
        WaitFlag<HardEvent::MTE3_V>(0);
        SyncAll<true>();
        SetFlag<HardEvent::V_MTE2>(0); SetFlag<HardEvent::MTE3_V>(0);
        auto merge=arena_.GetWithOffset<float>(inputCount,0);
        const uint64_t firstMatrix=matrices*id/cores;
        const uint32_t batch=static_cast<uint32_t>(matrices*(id+1U)/cores-firstMatrix);
        if (batch) {
            const uint32_t plane=batch*row;
            WaitFlag<HardEvent::V_MTE2>(0);
            DataCopyExtParams load{static_cast<uint16_t>(chunks),plane*4U,
                static_cast<uint32_t>((matrices-batch)*row*4ULL),0,0};
            DataCopyPadExtParams<float> noPad{false,0,0,0.0f};
            DataCopyPad(merge,ws_[firstMatrix*row],load,noPad);
            SetFlag<HardEvent::MTE2_V>(0); WaitFlag<HardEvent::MTE2_V>(0);
            uint32_t power=1;
            while ((power<<1U)<=chunks) power<<=1U;
            if (chunks>power) {
                Add(merge,merge,merge[power*plane],(chunks-power)*plane);
                pipe_barrier(PIPE_V);
            }
            for (uint32_t half=power>>1U;half;half>>=1U) {
                Add(merge,merge,merge[half*plane],half*plane);
                pipe_barrier(PIPE_V);
            }
            WaitFlag<HardEvent::MTE3_V>(0);
            auto dst=tmp_.ReinterpretCast<bfloat16_t>();
            Cast(dst,merge,RoundMode::CAST_RINT,plane);
            SetFlag<HardEvent::V_MTE2>(0);
            SetFlag<HardEvent::V_MTE3>(0); WaitFlag<HardEvent::V_MTE3>(0);
            DataCopyExtParams cp{static_cast<uint16_t>(batch),width*2U,0,0,0};
            DataCopyPad(y_[firstMatrix*width],dst,cp);
            SetFlag<HardEvent::MTE3_V>(0);
        }
        WaitFlag<HardEvent::V_MTE2>(0); WaitFlag<HardEvent::MTE3_V>(0);
    }
private:
    __aicore__ inline void Copy(LocalTensor<bfloat16_t> dst,uint64_t offset,uint32_t count)
    {
        if ((count&15U)==0) DataCopy(dst,x_[offset],count);
        else {
            DataCopyExtParams cp{1,count*2U,0,0,0};
            DataCopyPadExtParams<bfloat16_t> pad{false,0,0,static_cast<bfloat16_t>(0)};
            DataCopyPad(dst,x_[offset],cp,pad);
        }
    }
    __aicore__ inline void Accumulate(LocalTensor<bfloat16_t> source,uint32_t count,
                                     uint32_t lanes,uint32_t math)
    {
        uint32_t accOffset=0;
        for (uint32_t k=0;k<count;k+=math) {
            const uint32_t n=count-k<math ? count-k : math;
            if (n!=math) SetVectorMask<float,MaskMode::COUNTER>(0,n);
            const UnaryRepeatParams cvt{1,1,8,4};
            const BinaryRepeatParams fma{1,1,1,8,8,8};
            Cast<float,bfloat16_t,false>(tmp_,source[k],RoundMode::CAST_NONE,uint64_t{0},1,cvt);
            pipe_barrier(PIPE_V);
            MulAddDst<float,float,false>(acc_[accOffset],tmp_,tmp_,uint64_t{0},1,fma);
            pipe_barrier(PIPE_V);
            accOffset+=math;
            if (accOffset==lanes) accOffset=0;
        }
    }
    __aicore__ inline void BuildIndex(uint32_t width)
    {
        const uint32_t count=width*8U;
        auto index=index_.ReinterpretCast<int32_t>();
        auto q=tmp_.ReinterpretCast<int32_t>(), r=tmp_[count].ReinterpretCast<int32_t>();
        ArithProgression<int32_t>(index,0,1,count);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(q,index,3,count);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(r,q,3,count);
        pipe_barrier(PIPE_V);
        Sub(index,index,r,count);
        pipe_barrier(PIPE_V);
        Muls<int32_t>(index,index,static_cast<int32_t>(width),count);
        pipe_barrier(PIPE_V);
        Add(index,index,q,count);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(index,index,2,count);
        pipe_barrier(PIPE_V);
    }
    __aicore__ inline void Fold(uint32_t width,uint32_t lanes)
    {
        const uint32_t stop=width*8U;
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
    }
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<bfloat16_t> x_,y_;
    GlobalTensor<float> ws_;
    LocalTensor<bfloat16_t> in0_,in1_;
    LocalTensor<float> acc_,tmp_,out_;
    LocalTensor<uint32_t> index_;
};
}
