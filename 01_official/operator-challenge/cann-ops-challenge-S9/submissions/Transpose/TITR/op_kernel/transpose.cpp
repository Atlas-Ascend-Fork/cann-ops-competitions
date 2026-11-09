#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h" // Required by CANN's generated mixed-core wrapper.
using namespace AscendC;
template<class A,class B> constexpr bool SameElementType=false;
template<class A> constexpr bool SameElementType<A,A> = true;
__aicore__ inline uint32_t UMin(uint32_t a,uint32_t b){return a<b?a:b;}
__aicore__ inline uint32_t Align(uint32_t n,uint32_t a){return (n+a-1)/a*a;}

// Small-matrix routes consume only total and their compact p[] metadata.
// The SDK-generated pointer macro preserves the generated tiling field layout
// while avoiding InitTilingData's temporary UB copy and entry PIPE_ALL.
// Payload accesses and their synchronization remain in the original classes.
template<uint32_t ParamCount>
__aicore__ inline __attribute__((always_inline))
void LoadSmallMatrixTiling(GM_ADDR tiling,TransposeTilingData& data) {
    static_assert(ParamCount==3 || ParamCount==5,"small-matrix metadata only");
    GET_TILING_DATA_PTR_WITH_STRUCT(TransposeTilingData,source,tiling);
    data.total=source->total;
#pragma unroll
    for(uint32_t i=0;i<ParamCount;++i)data.p[i]=source->p[i];
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_TILING_DATA));
#endif
}

// Aligned half matrices: fixed 16x64 primitive, shape-derived GM pitches.
class AlignedHalfTranspose {
    const TransposeTilingData& t_;
    TPipe pipe_;TBuf<TPosition::VECCALC> arena_;GlobalTensor<half> x_,y_;
public:
    __aicore__ explicit AlignedHalfTranspose(const TransposeTilingData& t):t_(t){}
    __aicore__ void Init(GM_ADDR input,GM_ADDR output) {
        x_.SetGlobalBuffer((__gm__ half*)input,t_.total);y_.SetGlobalBuffer((__gm__ half*)output,t_.total);
        pipe_.InitBuffer(arena_,2*16*64*sizeof(half));
    }
    __aicore__ void Process() {
        const uint32_t n=t_.p[0],h=t_.p[1],w=t_.p[2],hg=h/16,cg=w/64;
        for(uint64_t task=GetBlockIdx();task<uint64_t(n)*hg*cg;task+=GetBlockNum()) {
            const uint64_t base=task/(hg*cg)*h*w;
            const uint32_t aStart=task%hg*16,cStart=(task/hg)%cg*64;
            auto input=arena_.Get<half>();
            const DataCopyParams read{16U,4U,static_cast<uint16_t>(w/16-4),0U};
            DataCopy(input,x_[base+aStart*w+cStart],read);
            SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVENT_ID0));
            auto output=input[16*64];uint64_t src[16],dst[16];
            const uint64_t sb=reinterpret_cast<uint64_t>(input.GetPhyAddr());
            const uint64_t db=reinterpret_cast<uint64_t>(output.GetPhyAddr());
#pragma unroll
            for(uint32_t i=0;i<16;++i){src[i]=sb+i*64*sizeof(half);dst[i]=db+i*16*sizeof(half);}
            TransDataTo5HDParams p;p.dstHighHalf=false;p.srcHighHalf=false;p.repeatTimes=4;p.srcRepStride=1;p.dstRepStride=16;
            const DataCopyParams write{64U,1U,0U,static_cast<uint16_t>(h/16-1)};
            const uint64_t offset=base+cStart*h+aStart;
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(EVENT_ID0));
            TransDataTo5HD<half>(dst,src,p);
            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVENT_ID0));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(EVENT_ID0));
            DataCopy(y_[offset],output,write);
            if(task+GetBlockNum()<uint64_t(n)*hg*cg) PipeBarrier<PIPE_ALL>();
        }
    }
};

// Balanced byte matrices: keep the measured 112x128 tile / 160-byte UB pitch.
class BalancedByteTranspose {
    const TransposeTilingData& t_;
    TPipe pipe_;GlobalTensor<int8_t> x_,y_;
    TQue<TPosition::VECIN,1> in_;TQue<TPosition::VECOUT,1> out_;
public:
    __aicore__ explicit BalancedByteTranspose(const TransposeTilingData& t):t_(t){}
    __aicore__ void Init(GM_ADDR input,GM_ADDR output) {
        x_.SetGlobalBuffer((__gm__ int8_t*)input,t_.total);y_.SetGlobalBuffer((__gm__ int8_t*)output,t_.total);
        pipe_.InitBuffer(in_,1,112*160+32);pipe_.InitBuffer(out_,1,128*160);
    }
    __aicore__ void Process() {
        const uint32_t n=t_.p[0],h=t_.p[1],w=t_.p[2],hg=t_.p[3],step=t_.p[4],cg=(w+127)/128;
        for(uint64_t task=GetBlockIdx();task<uint64_t(n)*hg*cg;task+=GetBlockNum()) {
            const uint64_t base=task/(hg*cg)*h*w;
            uint32_t hs=(task/cg)%hg*step,cs=task%cg*128;
            // Ceil-balanced partition can leave an empty last group at tiny H;
            // it is skipped before forming any DMA descriptor.
            if(hs>=h)continue;
            uint32_t hv=UMin(step,h-hs),cv=UMin(128,w-cs);
            auto input=in_.AllocTensor<int8_t>();
            const DataCopyExtParams read{static_cast<uint16_t>(hv),cv,w-cv,(160-Align(cv,32))/32,0U};
            const DataCopyPadExtParams<int8_t> pad{false,0,0,0};
            DataCopyPad(input,x_[base+hs*w+cs],read,pad);in_.EnQue(input);
            auto ready=in_.DeQue<int8_t>();auto output=out_.AllocTensor<int8_t>();
            DirectTranspose(output,ready);out_.EnQue(output);in_.FreeTensor(ready);
            auto result=out_.DeQue<int8_t>();
            const DataCopyExtParams write{static_cast<uint16_t>(cv),hv,(160-Align(hv,32))/32,h-hv,0U};
            DataCopyPad(y_[base+cs*h+hs],result,write);out_.FreeTensor(result);
        }
    }
private:
    __aicore__ void DirectTranspose(const LocalTensor<int8_t>& dst,const LocalTensor<int8_t>& src) {
        uint64_t sl[16],dl[16];TransDataTo5HDParams p;p.repeatTimes=4;p.srcRepStride=1;p.dstRepStride=160;
        for(uint32_t part=0;part<7;++part) {
            uint32_t start=(part>>1)*32,high=part&1;if(high)PipeBarrier<PIPE_V>();p.dstHighHalf=high!=0;
            for(uint32_t i=0;i<16;++i)sl[i]=reinterpret_cast<uint64_t>(src[(start+high*16+i)*160].GetPhyAddr());
            for(uint32_t half=0;half<2;++half) {
                p.srcHighHalf=half!=0;
                for(uint32_t i=0;i<16;++i)dl[i]=reinterpret_cast<uint64_t>(dst[(half*16+i)*160+start].GetPhyAddr());
                TransDataTo5HD<int8_t>(dl,sl,p);
            }
        }
    }
};

// Separate edge-capable 16x16 half / 16x8 word / 16x32 byte routes.
// Padding exists only in UB; GM reads and writes contain valid elements only.
template<class T> class PaddedMatrixTranspose {
    static constexpr uint32_t W=32/sizeof(T),OutPitch=sizeof(T)==1?32:16;
    TPipe pipe_;TQue<TPosition::VECIN,1> in_;TQue<TPosition::VECOUT,1> out_;
    GlobalTensor<T> x_,y_;const TransposeTilingData& t_;
public:
    __aicore__ explicit PaddedMatrixTranspose(const TransposeTilingData& t):t_(t){}
    __aicore__ void Init(GM_ADDR input,GM_ADDR output) {
        x_.SetGlobalBuffer((__gm__ T*)input,t_.total);y_.SetGlobalBuffer((__gm__ T*)output,t_.total);
        pipe_.InitBuffer(in_,1,16*W*sizeof(T));pipe_.InitBuffer(out_,1,W*OutPitch*sizeof(T));
    }
    __aicore__ void Process() {
        uint32_t h=t_.p[1],w=t_.p[2],hg=(h+15)/16,cg=(w+W-1)/W;
        uint64_t tasks=uint64_t(t_.p[0])*hg*cg;
        for(uint64_t task=GetBlockIdx();task<tasks;task+=GetBlockNum()) {
            uint64_t base=task/(uint64_t(hg)*cg)*h*w;
            uint32_t hs=(task/cg)%hg*16,cs=task%cg*W,hv=UMin(16,h-hs),cv=UMin(W,w-cs);
            auto input=in_.template AllocTensor<T>();
            const DataCopyExtParams read{static_cast<uint16_t>(hv),static_cast<uint32_t>(cv*sizeof(T)),static_cast<uint32_t>((w-cv)*sizeof(T)),0U,0U};
            const DataCopyPadExtParams<T> pad{false,0,0,0};
            DataCopyPad(input,x_[base+hs*w+cs],read,pad);in_.EnQue(input);
            auto ready=in_.template DeQue<T>();auto output=out_.template AllocTensor<T>();
            uint64_t sl[16],dl[16];TransDataTo5HDParams p;p.repeatTimes=1;p.srcRepStride=0;p.dstRepStride=0;p.dstHighHalf=false;p.srcHighHalf=false;
            for(uint32_t i=0;i<16;++i)sl[i]=reinterpret_cast<uint64_t>(ready[(i<hv?i:0)*W].GetPhyAddr());
            if constexpr(sizeof(T)==1) {
                for(uint32_t part=0;part<2;++part) {
                    p.srcHighHalf=part!=0;
                    for(uint32_t i=0;i<16;++i)dl[i]=reinterpret_cast<uint64_t>(output[(part*16+i)*OutPitch].GetPhyAddr());
                    TransDataTo5HD<T>(dl,sl,p);
                }
            } else {
                for(uint32_t i=0;i<16;++i) {
                    uint32_t offset=sizeof(T)==4 ? (i/2)*OutPitch+(i%2)*8 : i*OutPitch;
                    dl[i]=reinterpret_cast<uint64_t>(output[offset].GetPhyAddr());
                }
                TransDataTo5HD<T>(dl,sl,p);
            }
            out_.EnQue(output);in_.FreeTensor(ready);auto result=out_.template DeQue<T>();
            const DataCopyExtParams write{static_cast<uint16_t>(cv),static_cast<uint32_t>(hv*sizeof(T)),(OutPitch-Align(hv,W))/W,static_cast<uint32_t>((h-hv)*sizeof(T)),0U};
            DataCopyPad(y_[base+cs*h+hs],result,write);out_.FreeTensor(result);
        }
    }
};

// Bulk linear/suffix copy and correctness-first strided rows. No scalar
// GetValue/SetValue on tensor payload, including the generic fallback.
template<class T,uint32_t Strategy> class RowTransfer {
    TPipe pipe_;TQue<TPosition::VECIN,1> queue_;GlobalTensor<T> x_,y_;
    const TransposeTilingData& t_;
    __aicore__ uint64_t Source(uint64_t output) {
        uint64_t source=0;
        for(uint32_t r=t_.rank;r>0;--r) {
            uint32_t axis=r-1;source+=(output%t_.outShape[axis])*t_.stride[t_.perm[axis]];output/=t_.outShape[axis];
        }
        return source;
    }
public:
    __aicore__ explicit RowTransfer(const TransposeTilingData& t):t_(t){}
    __aicore__ void Init(GM_ADDR input,GM_ADDR output) {
        x_.SetGlobalBuffer((__gm__ T*)input,t_.total);y_.SetGlobalBuffer((__gm__ T*)output,t_.total);
        pipe_.InitBuffer(queue_,1,32768);
    }
    __aicore__ void Process() {
        if(!t_.total)return;
        const uint64_t row=Strategy==6?t_.total:t_.inner;
        const uint64_t capacity=Strategy==11 ? ((t_.innerStride-1)>UINT32_MAX/sizeof(T)?1:256) :32768/sizeof(T);
        const uint64_t chunks=(row+capacity-1)/capacity,tasks=(t_.total/row)*chunks;
        for(uint64_t task=GetBlockIdx();task<tasks;task+=GetBlockNum()) {
            uint64_t col=(task%chunks)*capacity,out=(task/chunks)*row+col;
            uint32_t count=static_cast<uint32_t>((row-col)<capacity?row-col:capacity);
            uint64_t src=Strategy==6?out:Source(out);auto input=queue_.template AllocTensor<T>();
            DataCopyExtParams read{1U,static_cast<uint32_t>(count*sizeof(T)),0U,0U,0U};
            if constexpr(Strategy==11) {
                read.blockCount=count;read.blockLen=sizeof(T);
                read.srcStride=count==1?0:static_cast<uint32_t>((t_.innerStride-1)*sizeof(T));
            }
            const DataCopyPadExtParams<T> pad{false,0,0,0};
            DataCopyPad(input,x_[src],read,pad);queue_.EnQue(input);auto ready=queue_.template DeQue<T>();
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyExtParams write{1U,static_cast<uint32_t>(count*sizeof(T)),0U,0U,0U};
            if constexpr(Strategy==11) {write.blockCount=count;write.blockLen=sizeof(T);}
            DataCopyPad(y_[out],ready,write);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            queue_.FreeTensor(ready);
        }
    }
};

// Native Cube reorders, preserving all payload bits.

class ReorderedCubeProducer {
    const TransposeTilingData& t_;

    const uint32_t kA=t_.p[0], kB=t_.p[1], kC=t_.p[2], kS=kA*kB*kC;
    const uint32_t kDE=t_.p[3], kSlots=t_.p[4], kSlotDE=t_.p[5];
    static constexpr uint32_t kC0=8;
    const uint32_t kMaxC1=kSlotDE/8, kWorkspaceH=t_.p[6], kTileA=t_.p[7];
    const uint32_t kTileH=kTileA*kC, kTileCount=kS/kTileH;
    const uint32_t kL1SlotElems=kMaxC1*kTileH*8, kL1SlotBytes=kL1SlotElems*4;
    const uint32_t kWorkspaceSlotElems=kMaxC1*kWorkspaceH*8;

public:
    __aicore__ explicit ReorderedCubeProducer(const TransposeTilingData& t):t_(t){}
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR userWorkspace, TPipe* pipe,
        bool inputBypass = true)
    {
        pipe_ = pipe;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(input));
        if (inputBypass) x_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace));
        pipe->InitBuffer(l1_, 2U * kL1SlotBytes);
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            ready_[slot] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
            free_[slot] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        }
        drained_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_S>());
    }

    __aicore__ inline void Finish()
    {
        if (pipe_ == nullptr) return;
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(ready_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(free_[slot]);
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_S>(drained_);
        pipe_ = nullptr;
    }

    __aicore__ inline void Process(uint32_t workspaceSlot)
    {
        if (workspaceSlot >= kSlots) return;
        const uint32_t deStart = workspaceSlot * kSlotDE;
        const uint32_t remaining = kDE - deStart;
        const uint32_t deCur = remaining < kSlotDE ? remaining : kSlotDE;
        const uint32_t c1Count = (deCur + kC0 - 1U) / kC0;

        SetFlag<HardEvent::MTE3_MTE2>(free_[0]);
        SetFlag<HardEvent::MTE3_MTE2>(free_[1]);
        CopyIn(deStart, deCur, 0U);
        CopyIn(deStart, deCur, 1U);
        for (uint32_t tile = 0U; tile < kTileCount; ++tile) {
            CopyOut(workspaceSlot, c1Count, tile);
            if (tile + 2U < kTileCount) CopyIn(deStart, deCur, tile + 2U);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(free_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(free_[1]);
        SetFlag<HardEvent::MTE2_S>(drained_);
        WaitFlag<HardEvent::MTE2_S>(drained_);
    }

private:
    __aicore__ inline void CopyIn(uint32_t deStart, uint32_t deCur, uint32_t tile)
    {
        const uint32_t slot = tile & 1U;
        WaitFlag<HardEvent::MTE3_MTE2>(free_[slot]);
        auto l1 = l1_.Get<int32_t>();
        const uint32_t b = tile / (kA / kTileA);
        const uint32_t aBegin = (tile % (kA / kTileA)) * kTileA;
        Nd2NzParams p;
        p.ndNum = 1U;
        p.nValue = kC;
        p.dValue = static_cast<uint16_t>(deCur);
        p.srcNdMatrixStride = 0U;
        p.srcDValue = kDE;
        p.dstNzC0Stride = kTileH;
        p.dstNzNStride = 1U;
        p.dstNzMatrixStride = 1U;
#pragma unroll
        for (uint32_t a = 0U; a < kTileA; ++a) {
            const uint64_t src = (static_cast<uint64_t>(aBegin + a) * kB + b) *
                kC * kDE + deStart;

            DataCopy(l1[slot * kL1SlotElems + a * kC * kC0], x_[src], p);
        }
        SetFlag<HardEvent::MTE2_MTE3>(ready_[slot]);
    }

    __aicore__ inline void CopyOut(uint32_t workspaceSlot, uint32_t c1Count,
        uint32_t tile)
    {
        const uint32_t slot = tile & 1U;
        WaitFlag<HardEvent::MTE2_MTE3>(ready_[slot]);
        auto l1 = l1_.Get<int32_t>();
        const DataCopyParams p{static_cast<uint16_t>(c1Count),
            static_cast<uint16_t>(kTileH), 0U,
            static_cast<uint16_t>(kWorkspaceH - kTileH)};
        const uint64_t dst = static_cast<uint64_t>(workspaceSlot) * kWorkspaceSlotElems +
            tile * kTileH * kC0;
        DataCopy(workspace_[dst], l1[slot * kL1SlotElems], p);
        SetFlag<HardEvent::MTE3_MTE2>(free_[slot]);
    }

    GlobalTensor<int32_t> x_;
    GlobalTensor<int32_t> workspace_;
    TBuf<TPosition::A1> l1_;
    TPipe* pipe_ = nullptr;
    event_t ready_[2];
    event_t free_[2];
    event_t drained_;
};

template<bool PackedOutput=false>
class ReorderedCubeConsumerImpl {
    const TransposeTilingData& t_;

    const uint32_t kS=t_.p[0]*t_.p[1]*t_.p[2], kDE=t_.p[3], kSlotDE=t_.p[5];
    const uint32_t kC1=kSlotDE/8, kWorkspaceH=t_.p[6], OutStride=kWorkspaceH;
    static constexpr uint32_t kTileDE=16, InputBuffers=PackedOutput?1:2;
    const uint32_t kInputElems=kS*kTileDE, kOutputElems=OutStride*kTileDE;
    const uint32_t kOutputOffset=InputBuffers*kInputElems;
    const uint32_t kPackedOffset=kOutputOffset+kOutputElems+(PackedOutput?64:0);
    const uint32_t kArenaBytes=(kPackedOffset+(PackedOutput?kInputElems:0))*4;

public:
    __aicore__ explicit ReorderedCubeConsumerImpl(const TransposeTilingData& t):t_(t){}
public:
    __aicore__ inline void Init(GM_ADDR output, GM_ADDR userWorkspace)
    {
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
        work_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace));
        pipe_.InitBuffer(arena_, kArenaBytes);
        for (uint32_t slot = 0U; slot < InputBuffers; ++slot) {
            inputReady_[slot] = pipe_.AllocEventID<HardEvent::MTE2_V>();
            inputFree_[slot] = pipe_.AllocEventID<HardEvent::V_MTE2>();
        }
        outputReady_ = pipe_.AllocEventID<HardEvent::V_MTE3>();
        outputFree_ = pipe_.AllocEventID<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Process(uint32_t workspaceSlot, uint32_t lane)
    {
        const uint32_t slotStart = workspaceSlot * kSlotDE;
        const uint32_t slotCount = Min(kSlotDE, kDE - slotStart);
        const uint32_t deBegin = lane * (kSlotDE / 32U * 16U);
        const uint32_t deEnd = lane == 0U ? kSlotDE / 32U * 16U : slotCount;
        const uint32_t groups = (deEnd - deBegin + kTileDE - 1U) / kTileDE;
        for (uint32_t slot=0;slot<InputBuffers;++slot)
            ReadChunk(workspaceSlot, deBegin, deEnd, slot, slot);
        LocalTensor<int32_t> out = arena_.Get<int32_t>()[kOutputOffset];
        for (uint32_t group = 0U; group < groups; ++group) {
            if constexpr (!PackedOutput) {
                if (group != 0U) WaitFlag<HardEvent::MTE3_V>(outputFree_);
            }
            const uint32_t inputSlot = group % InputBuffers;
            const uint32_t deStart = deBegin + group * kTileDE;
            const uint32_t channels = Min(kTileDE, deEnd - deStart);
            WaitFlag<HardEvent::MTE2_V>(inputReady_[inputSlot]);
            Transpose(out, arena_.Get<int32_t>()[inputSlot * kInputElems], channels);
            SetFlag<HardEvent::V_MTE2>(inputFree_[inputSlot]);
            if (group + InputBuffers < groups) {
                WaitFlag<HardEvent::V_MTE2>(inputFree_[inputSlot]);
                ReadChunk(workspaceSlot, deBegin, deEnd, group + InputBuffers, inputSlot);
            }
            if constexpr (PackedOutput) {
                // Padded transpose is separate from the pending GM output;
                // postpone packed-slot reuse until after the next transpose.
                if (group != 0U) WaitFlag<HardEvent::MTE3_V>(outputFree_);
                auto packed=arena_.Get<int32_t>()[kPackedOffset];
                PipeBarrier<PIPE_V>();
                SetMaskCount();
                set_vector_mask(0U,kS);
                // Builtin7 does not read src1. All-ones selection copies exactly
                // S raw words per row; the reserved guard covers a final fetch.
                vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(packed.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(out.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(out.GetPhyAddr()),
                    static_cast<uint16_t>(channels),1U,7U,OutStride/8U,0U);
                SetMaskNorm();
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(outputReady_);
                WaitFlag<HardEvent::V_MTE3>(outputReady_);
                const DataCopyExtParams copy{1U,channels*kS*4U,0U,0U,0U};
                DataCopyPad(y_[(slotStart+deStart)*kS],packed,copy);
            } else {
                SetFlag<HardEvent::V_MTE3>(outputReady_);
                WaitFlag<HardEvent::V_MTE3>(outputReady_);
                const DataCopyExtParams copy{static_cast<uint16_t>(channels), kS * 4U,
                    (OutStride - Align(kS, 8U)) / 8U, 0U, 0U};
                DataCopyPad(y_[(slotStart + deStart) * kS], out, copy);
            }
            SetFlag<HardEvent::MTE3_V>(outputFree_);
        }

        WaitFlag<HardEvent::MTE3_V>(outputFree_);
        for (uint32_t slot=0;slot<InputBuffers;++slot)
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
    }

    __aicore__ inline void Release()
    {
        for (uint32_t slot = 0U; slot < InputBuffers; ++slot) {
            pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady_[slot]);
            pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputFree_[slot]);
        }
        pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady_);
        pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputFree_);
    }

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const
    {
        return a < b ? a : b;
    }

    __aicore__ inline void ReadChunk(uint32_t workspaceSlot, uint32_t deBegin,
        uint32_t deEnd, uint32_t group, uint32_t inputSlot)
    {
        const uint32_t deStart = deBegin + group * kTileDE;
        const uint32_t channels = Min(kTileDE, deEnd - deStart);
        const uint32_t blocksC1 = (channels + 7U) / 8U;
        const uint32_t source = (workspaceSlot * kC1 + deStart / 8U) * kWorkspaceH * 8U;

        const DataCopyParams copy{static_cast<uint16_t>(blocksC1),
            static_cast<uint16_t>(kS), static_cast<uint16_t>(kWorkspaceH - kS), 0U};
        DataCopy(arena_.Get<int32_t>()[inputSlot * kInputElems], work_[source], copy);
        SetFlag<HardEvent::MTE2_V>(inputReady_[inputSlot]);
    }

    __aicore__ inline void Transpose(const LocalTensor<int32_t>& out,
        const LocalTensor<int32_t>& input, uint32_t channels)
    {
        uint64_t src[16], dst[16];
        TransDataTo5HDParams p;
        p.dstHighHalf = false;
        p.srcHighHalf = false;
        for (uint32_t c = 0U; c < channels; c += 8U) {
            const uint32_t inputBase = (c / 8U) * kS * 8U;
#pragma unroll
            for (uint32_t i = 0U; i < 16U; ++i) {
                src[i] = reinterpret_cast<uint64_t>(input[inputBase + i * 8U].GetPhyAddr());
                dst[i] = reinterpret_cast<uint64_t>(out[
                    (c + i / 2U) * OutStride + (i % 2U) * 8U].GetPhyAddr());
            }

            p.repeatTimes = kS / 16U;
            p.srcRepStride = 16U;
            p.dstRepStride = 2U;
            TransDataTo5HD<int32_t>(dst, src, p);
            if (kS % 16U != 0U) {
#pragma unroll
            for (uint32_t i = 0U; i < 16U; ++i) {

                const uint32_t safeRow = kS / 16U * 16U + (i < kS % 16U ? i : 0U);
                src[i] = reinterpret_cast<uint64_t>(input[inputBase + safeRow * 8U].GetPhyAddr());
                dst[i] = reinterpret_cast<uint64_t>(out[
                    (c + i / 2U) * OutStride + kS / 16U * 16U + (i % 2U) * 8U].GetPhyAddr());
            }
            p.repeatTimes = 1U;
            p.srcRepStride = 0U;
            p.dstRepStride = 0U;
            TransDataTo5HD<int32_t>(dst, src, p);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<int32_t> y_, work_;
    TEventID inputReady_[2], inputFree_[2];
    TEventID outputReady_, outputFree_;
};

using ReorderedCubeConsumer=ReorderedCubeConsumerImpl<false>;
using PackedReorderedCubeConsumer=ReorderedCubeConsumerImpl<true>;

class MaskedBlockTranspose {
    const TransposeTilingData& t_;

    static constexpr uint32_t InputBuffers=2, OutputBuffers=1, Rotation=1, CachePolicy=1, CounterMode=0;
    const uint32_t kA=t_.p[0], kB=t_.p[1], kC=t_.p[2], kD=t_.p[3], Tasks=t_.p[4];
    const uint32_t kGroups=Tasks/kB, kExtraGroups=Tasks%kB;
    const uint32_t kMaxC=(kC+kGroups-1)/kGroups, kSmallC=kMaxC-1;
    const uint32_t kInStride=Align(kMaxC*kD,16), kOutStride=Align(kA*kD,16);
    const uint32_t kInputBytes=kA*kInStride*2+256, kOutputBytes=kMaxC*kOutStride*2;
    const bool kTwoLengths=kC/(kGroups+1)==kSmallC;
    struct Tile {uint32_t b,cStart,cCur;};

public:
    __aicore__ explicit MaskedBlockTranspose(const TransposeTilingData& t):t_(t){}
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, TPipe* pipe)
    {
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        if constexpr ((CachePolicy & 1U) != 0U) x_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if constexpr ((CachePolicy & 2U) != 0U) y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        pipe->InitBuffer(in_, InputBuffers, kInputBytes);
        pipe->InitBuffer(out_, OutputBuffers, kOutputBytes);
        pipe->InitBuffer(mask_, 512U);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blocks = GetBlockNum();
        const uint32_t lane = (GetBlockIdx() * Rotation) % blocks;
        const uint32_t count = lane < Tasks ? (Tasks - lane + blocks - 1U) / blocks : 0U;
        for (uint32_t i = 0U; i < InputBuffers && i < count; ++i) {
            CopyIn(Decode(lane + i * blocks));
        }
        InitMasks();
        for (uint32_t i = 0U; i < count; ++i) {
            const Tile t = Decode(lane + i * blocks);
            Compute(t);
            if (i + InputBuffers < count) {
                CopyIn(Decode(lane + (i + InputBuffers) * blocks));
            }
            CopyOut(t);
        }
    }

private:
    __aicore__ inline Tile Decode(uint32_t task) const
    {
        Tile t{};
        if (kTwoLengths) {

            const uint32_t hiLong = kC - (kGroups + 1U) * kSmallC;
            const uint32_t loLong = kC - kGroups * kSmallC;
            const uint32_t hiLongTasks = kExtraGroups * hiLong;
            const uint32_t longTasks = hiLongTasks + (kB - kExtraGroups) * loLong;
            const uint32_t hiShort = kGroups + 1U - hiLong;
            const uint32_t loShort = kGroups - loLong;

            if (task < longTasks) {
                t.cCur = kMaxC;
                if (task < hiLongTasks) {

                    if (hiLong != 0U) {
                        t.b = task / hiLong;
                        t.cStart = (task % hiLong) * kMaxC;
                    }
                } else {
                    const uint32_t q = task - hiLongTasks;
                    t.b = kExtraGroups + q / loLong;
                    t.cStart = (q % loLong) * kMaxC;
                }
            } else {
                const uint32_t q = task - longTasks;
                t.cCur = kSmallC;
                if (q < kExtraGroups * hiShort) {
                    t.b = q / hiShort;
                    t.cStart = hiLong * kMaxC + (q % hiShort) * kSmallC;
                } else {
                    const uint32_t r = q - kExtraGroups * hiShort;
                    t.b = kExtraGroups + r / loShort;
                    t.cStart = loLong * kMaxC + (r % loShort) * kSmallC;
                }
            }
        } else {
            uint32_t local;
            uint32_t groups;
            if (task < kExtraGroups * (kGroups + 1U)) {
                groups = kGroups + 1U;
                t.b = task / groups;
                local = task % groups;
            } else {
                groups = kGroups;
                const uint32_t q = task - kExtraGroups * (kGroups + 1U);
                t.b = kExtraGroups + q / groups;
                local = q % groups;
            }
            t.cStart = local * kC / groups;
            t.cCur = (local + 1U) * kC / groups - t.cStart;
        }
        return t;
    }

    __aicore__ inline void InitMasks()
    {
        auto m = mask_.Get<uint64_t>();
        for (uint32_t lead = 0U; lead < 16U; ++lead) {
            m.SetValue(lead * 4U, ~0ULL << lead);
            m.SetValue(lead * 4U + 1U, (lead + kD == 128U ? ~0ULL : (1ULL << (lead + kD - 64U)) - 1ULL));
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
    }

    __aicore__ inline void CopyIn(const Tile& t)
    {
        auto in = in_.template AllocTensor<half>();
        const uint32_t elems = t.cCur * kD;
        const uint32_t alignedElems = (elems + 15U) / 16U * 16U;
        const DataCopyExtParams p{static_cast<uint16_t>(kA), elems * 2U,
            (kB * kC * kD - elems) * 2U, (kInStride - alignedElems) / 16U, 0U};
        const DataCopyPadExtParams<half> pad{false, 0U, 0U, 0};
        const uint64_t offset = (static_cast<uint64_t>(t.b) * kC + t.cStart) * kD;
        DataCopyPad(in, x_[offset], p, pad);
        in_.EnQue(in);
    }

    __aicore__ inline void Compute(const Tile& t)
    {
        auto in = in_.template DeQue<half>();
        auto out = out_.template AllocTensor<half>();
        auto patterns = mask_.Get<uint16_t>();
        if constexpr (CounterMode != 0U) {
            SetMaskCount();
            if constexpr (CounterMode == 2U) set_vector_mask(0U, 84U);
        } else {
            SetMaskNorm();
            set_vector_mask(0U, 0U);
        }
        for (uint32_t c = 0U; c < t.cCur; ++c) {
            const uint32_t start = c * kD;
            const uint32_t lead = start & 15U;
            if constexpr (CounterMode == 1U) set_vector_mask(0U, lead + kD);
            vreducev2(reinterpret_cast<__ubuf__ uint16_t*>(out[c * kOutStride].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(in[start - lead].GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint16_t*>(patterns[lead * 16U].GetPhyAddr()),
                static_cast<uint16_t>(kA), 1U, 0U,
                static_cast<uint16_t>(kInStride / 16U), 0U);
        }
        if constexpr (CounterMode != 0U) SetMaskNorm();
        out_.EnQue(out);
        in_.FreeTensor(in);
    }

    __aicore__ inline void CopyOut(const Tile& t)
    {
        auto out = out_.template DeQue<half>();
        const DataCopyExtParams p{static_cast<uint16_t>(t.cCur), kA * kD * 2U,
            (kOutStride - Align(kA*kD,16U)) / 16U, (kB - 1U) * kA * kD * 2U, 0U};
        const uint64_t offset = (static_cast<uint64_t>(t.cStart) * kB + t.b) * kA * kD;
        DataCopyPad(y_[offset], out, p);
        out_.FreeTensor(out);
    }

    GlobalTensor<half> x_;
    GlobalTensor<half> y_;
    TQue<TPosition::VECIN, InputBuffers> in_;
    TQue<TPosition::VECOUT, 1> out_;
    TBuf<TPosition::VECCALC> mask_;
};

class PackedCubeProducer {
    const TransposeTilingData& t_;

    using CopyT=int32_t;
    static constexpr uint32_t kC0=8, L1PitchH=112, kTileH=112;
    const uint32_t kN=t_.p[0], kH=t_.p[1], kC=t_.p[2], kSourceC=kC, kSourcePitchC=kC;
    const uint32_t kC1=(kC+7)/8, kWorkspaceH=t_.p[4], kWorkspaceSlots=t_.p[3];
    const uint32_t kTileCount=(kH+kTileH-1)/kTileH;
    const uint32_t kL1SlotElems=kC1*L1PitchH*kC0, kL1SlotBytes=kL1SlotElems*4;
    const uint32_t kWorkspaceSlotElems=kC1*kWorkspaceH*kC0;

public:
    __aicore__ explicit PackedCubeProducer(const TransposeTilingData& t):t_(t){}
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR userWorkspace, TPipe* pipe,
        bool inputBypass = false)
    {
        pipe_ = pipe;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ CopyT*>(input));
        if (inputBypass) x_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ CopyT*>(userWorkspace));
        pipe->InitBuffer(l1_, 2U * kL1SlotBytes);
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            ready_[slot] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
            free_[slot] = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        }
        drained_ = static_cast<event_t>(pipe->AllocEventID<HardEvent::MTE2_S>());
    }

    __aicore__ inline void Finish()
    {
        if (pipe_ == nullptr) return;
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            pipe_->ReleaseEventID<HardEvent::MTE2_MTE3>(ready_[slot]);
            pipe_->ReleaseEventID<HardEvent::MTE3_MTE2>(free_[slot]);
        }
        pipe_->ReleaseEventID<HardEvent::MTE2_S>(drained_);
        pipe_ = nullptr;
    }

    __aicore__ inline void PrefetchN(uint32_t n)
    {
        if (n >= kN) return;
        SetFlag<HardEvent::MTE3_MTE2>(free_[0]);
        SetFlag<HardEvent::MTE3_MTE2>(free_[1]);
        CopyIn(n, 0U);
        CopyIn(n, 1U);
    }

    __aicore__ inline void ProcessN(uint32_t n, uint32_t workspaceSlot, uint32_t bank = 0U,
        bool prefetched = false)
    {
        if (n >= kN || workspaceSlot >= kWorkspaceSlots || bank >= 2U) return;

        const uint32_t absoluteSlot = workspaceSlot + bank * kWorkspaceSlots;

        if (!prefetched) PrefetchN(n);
        for (uint32_t tile = 0U; tile < kTileCount; ++tile) {
            CopyOut(absoluteSlot, tile);
            if (tile + 2U < kTileCount) CopyIn(n, tile + 2U);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(free_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(free_[1]);
        SetFlag<HardEvent::MTE2_S>(drained_);
        WaitFlag<HardEvent::MTE2_S>(drained_);
    }

private:
    __aicore__ inline uint32_t Rows(uint32_t tile) const
    {
        const uint32_t remaining = kH - tile * kTileH;
        return remaining < kTileH ? remaining : kTileH;
    }

    __aicore__ inline void CopyIn(uint32_t n, uint32_t tile)
    {
        CopyInAt(n, tile * kTileH, Rows(tile), tile & 1U);
    }

    __aicore__ inline void CopyInAt(uint32_t n, uint32_t hStart, uint32_t rows,
        uint32_t slot)
    {
        WaitFlag<HardEvent::MTE3_MTE2>(free_[slot]);
        auto l1 = l1_.Get<CopyT>();
        Nd2NzParams p;
        p.ndNum = 1U;
        p.nValue = static_cast<uint16_t>(rows);
        p.dValue = kSourceC;
        p.srcNdMatrixStride = 0U;
        p.srcDValue = kSourcePitchC;
        p.dstNzC0Stride = L1PitchH;
        p.dstNzNStride = 1U;
        p.dstNzMatrixStride = 1U;
        const uint64_t src = (static_cast<uint64_t>(n) * kH + hStart) * kSourcePitchC;
        DataCopy(l1[slot * kL1SlotElems], x_[src], p);
        SetFlag<HardEvent::MTE2_MTE3>(ready_[slot]);
    }

    __aicore__ inline void CopyOut(uint32_t workspaceSlot, uint32_t tile)
    {
        CopyOutAt(workspaceSlot, tile * kTileH, Rows(tile), tile & 1U);
    }

    __aicore__ inline void CopyOutAt(uint32_t workspaceSlot, uint32_t hStart,
        uint32_t rows, uint32_t slot)
    {
        WaitFlag<HardEvent::MTE2_MTE3>(ready_[slot]);
        auto l1 = l1_.Get<CopyT>();

        const DataCopyParams p{static_cast<uint16_t>(kC1),
            static_cast<uint16_t>(rows), static_cast<uint16_t>(L1PitchH - rows),
            static_cast<uint16_t>(kWorkspaceH - rows)};
        const uint64_t dst = static_cast<uint64_t>(workspaceSlot) * kWorkspaceSlotElems +
            hStart * kC0;
        DataCopy(workspace_[dst], l1[slot * kL1SlotElems], p);
        SetFlag<HardEvent::MTE3_MTE2>(free_[slot]);
    }

public:

    __aicore__ inline void ProcessFinalSlice(uint32_t n, uint32_t workspaceSlot,
        uint32_t slice)
    {
        if (n >= kN || workspaceSlot >= kN % kWorkspaceSlots || slice >= t_.p[5]) return;
        const uint32_t hStart = slice * t_.p[6];
        const uint32_t rows = UMin(t_.p[6], kH - hStart);
        SetFlag<HardEvent::MTE3_MTE2>(free_[0]);
        SetFlag<HardEvent::MTE3_MTE2>(free_[1]);
        CopyInAt(n, hStart, 112U, 0U);
        CopyInAt(n, hStart + 112U, rows - 112U, 1U);
        CopyOutAt(workspaceSlot, hStart, 112U, 0U);
        CopyOutAt(workspaceSlot, hStart + 112U, rows - 112U, 1U);
        WaitFlag<HardEvent::MTE3_MTE2>(free_[0]);
        WaitFlag<HardEvent::MTE3_MTE2>(free_[1]);
        SetFlag<HardEvent::MTE2_S>(drained_);
        WaitFlag<HardEvent::MTE2_S>(drained_);
    }

private:

    GlobalTensor<CopyT> x_;
    GlobalTensor<CopyT> workspace_;
    TBuf<TPosition::A1> l1_;
    TPipe* pipe_ = nullptr;
    event_t ready_[2];
    event_t free_[2];
    event_t drained_;
};

class PackedCubeConsumer {
    const TransposeTilingData& t_;

    static constexpr uint32_t kChunkH=352, kTileC=16, InputBuffers=2, PackedBuffers=2;
    static constexpr bool OutputBypass=false, DropPattern=false, BuiltinMask=false;
    const uint32_t kH=t_.p[1], kC=t_.p[2], kC1=(kC+7)/8, kPaddedH=t_.p[4], PadStride=kPaddedH;
    const uint32_t kHGroups=(kH+kChunkH-1)/kChunkH;
    const uint32_t kInputElems=kTileC*kChunkH, kPaddedElems=kTileC*PadStride, kPackedElems=kTileC*kPaddedH;
    const uint32_t kPaddedOffset=InputBuffers*kInputElems, kPackedOffset=kPaddedOffset+kPaddedElems;
    const uint32_t kPatternOffset=kPackedOffset+PackedBuffers*kPackedElems, kPatternWords=(PadStride+255)/256*8;
    const uint32_t kArenaBytes=(kPatternOffset+kPatternWords)*4;

public:
    __aicore__ explicit PackedCubeConsumer(const TransposeTilingData& t):t_(t){}
public:


    __aicore__ inline TPipe* GetPipe() { return &pipe_; }

    __aicore__ inline void Init(GM_ADDR output, GM_ADDR userWorkspace)
    {
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
        if constexpr (OutputBypass)
            y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        work_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace));
        pipe_.InitBuffer(arena_, kArenaBytes);
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (slot < InputBuffers) {
                inputReady_[slot] = pipe_.AllocEventID<HardEvent::MTE2_V>();
                inputFree_[slot] = pipe_.AllocEventID<HardEvent::V_MTE2>();
            }
            if (slot < PackedBuffers) {
                outputReady_[slot] = pipe_.AllocEventID<HardEvent::V_MTE3>();
                outputFree_[slot] = pipe_.AllocEventID<HardEvent::MTE3_V>();
            }
        }
        if constexpr (!DropPattern) {
            auto pattern = arena_.Get<uint32_t>()[kPatternOffset];

            for(uint32_t word=0;word<kPatternWords;++word) {
                const uint32_t start=word*32U;
                pattern.SetValue(word, start>=kH ? 0U : kH-start>=32U ? 0xffffffffU : (1U<<(kH-start))-1U);
            }
            const auto patternReady = pipe_.AllocEventID<HardEvent::S_V>();
            SetFlag<HardEvent::S_V>(patternReady);
            WaitFlag<HardEvent::S_V>(patternReady);
            pipe_.ReleaseEventID<HardEvent::S_V>(patternReady);
        }
    }

    __aicore__ inline void ProcessN(uint32_t n, uint32_t nSlot, uint32_t lane)
    {
        const uint32_t cBegin = lane * (kC / 32U * 16U);
        const uint32_t cEnd = lane == 0U ? kC / 32U * 16U : kC;
        ProcessRange(n, nSlot, cBegin, cEnd);
    }

    __aicore__ inline void ProcessRange(uint32_t n, uint32_t nSlot,
        uint32_t cBegin, uint32_t cEnd)
    {
        const uint32_t groups = (cEnd - cBegin + kTileC - 1U) / kTileC;
        const uint32_t chunks = groups * kHGroups;
        for (uint32_t slot = 0U; slot < InputBuffers; ++slot)
            ReadChunk(nSlot, cBegin, cEnd, slot, slot);
        auto padded = arena_.Get<int32_t>()[kPaddedOffset];

        const auto pattern = arena_.Get<uint32_t>()[DropPattern ? kPaddedOffset : kPatternOffset];
        for (uint32_t cg = 0U; cg < groups; ++cg) {
            const uint32_t cStart = cBegin + cg * kTileC;
            const uint32_t channels = Min(kTileC, cEnd - cStart);
            for (uint32_t hg = 0U; hg < kHGroups; ++hg) {
                const uint32_t chunk = cg * kHGroups + hg;
                const uint32_t slot = chunk % InputBuffers;
                const uint32_t countH = Min(kChunkH, kH - hg * kChunkH);
                WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
                TransposeChunk(padded, arena_.Get<int32_t>()[slot * kInputElems],
                    hg * kChunkH, countH, channels);
                SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
                if (chunk + InputBuffers < chunks) {
                    WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
                    ReadChunk(nSlot, cBegin, cEnd, chunk + InputBuffers, slot);
                }
            }

            const uint32_t packedSlot = cg % PackedBuffers;

            if (cg >= PackedBuffers) WaitFlag<HardEvent::MTE3_V>(outputFree_[packedSlot]);
            auto packed = arena_.Get<int32_t>()[kPackedOffset + packedSlot * kPackedElems];
            PipeBarrier<PIPE_V>();

            SetMaskCount();

            set_vector_mask(0U, BuiltinMask ? kH : PadStride);
            vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(packed.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(padded.GetPhyAddr()),
                reinterpret_cast<__ubuf__ uint32_t*>(pattern.GetPhyAddr()),
                static_cast<uint16_t>(channels), 1U, BuiltinMask ? 7U : 0U,
                PadStride / 8U, 0U);
            SetMaskNorm();

            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(outputReady_[packedSlot]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[packedSlot]);
            const DataCopyExtParams copy{1U, channels * kH * 4U, 0U, 0U, 0U};
            DataCopyPad(y_[(n * kC + cStart) * kH], packed, copy);
            SetFlag<HardEvent::MTE3_V>(outputFree_[packedSlot]);
        }

        for (uint32_t slot = 0U; slot < PackedBuffers; ++slot)
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        for (uint32_t slot = 0U; slot < InputBuffers; ++slot)
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
    }

    __aicore__ inline void Release()
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (slot < InputBuffers) {
                pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady_[slot]);
                pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputFree_[slot]);
            }
            if (slot < PackedBuffers) {
                pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady_[slot]);
                pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputFree_[slot]);
            }
        }
    }

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const
    {
        return a < b ? a : b;
    }

    __aicore__ inline void ReadChunk(uint32_t nSlot, uint32_t cBegin,
        uint32_t cEnd, uint32_t chunk, uint32_t slot)
    {
        const uint32_t cStart = cBegin + (chunk / kHGroups) * kTileC;
        const uint32_t channels = Min(kTileC, cEnd - cStart);
        const uint32_t blocksC1 = (channels + 7U) / 8U;
        const uint32_t hStart = (chunk % kHGroups) * kChunkH;
        const uint32_t rows = Min(kChunkH, kH - hStart);
        const uint32_t source = ((nSlot * kC1 + cStart / 8U) * kPaddedH + hStart) * 8U;

        const DataCopyParams copy{static_cast<uint16_t>(blocksC1),
            static_cast<uint16_t>(rows), static_cast<uint16_t>(kPaddedH - rows),
            static_cast<uint16_t>(kChunkH - rows)};
        DataCopy(arena_.Get<int32_t>()[slot * kInputElems], work_[source], copy);
        SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
    }

    __aicore__ inline void TransposeChunk(const LocalTensor<int32_t>& output,
        const LocalTensor<int32_t>& input, uint32_t hBase, uint32_t rows,
        uint32_t channels)
    {
        const uint32_t repeats = rows / 16U;
        const uint32_t fullRows = repeats * 16U;
        const uint32_t tailRows = rows - fullRows;
        TransDataTo5HDParams main;
        main.dstHighHalf = false;
        main.srcHighHalf = false;
        main.repeatTimes = static_cast<uint8_t>(repeats);
        main.srcRepStride = repeats == 1U ? 0U : 16U;
        main.dstRepStride = repeats == 1U ? 0U : 2U;
        TransDataTo5HDParams tail;
        tail.dstHighHalf = false;
        tail.srcHighHalf = false;
        tail.repeatTimes = 1U;
        tail.srcRepStride = 0U;
        tail.dstRepStride = 0U;
        uint64_t src[16], dst[16];
        for (uint32_t c = 0U; c < channels; c += 8U) {
            const uint32_t c1Base = (c / 8U) * kChunkH * 8U;
#pragma unroll
            for (uint32_t i = 0U; i < 16U; ++i) {
                src[i] = reinterpret_cast<uint64_t>(input[c1Base + i * 8U].GetPhyAddr());
                dst[i] = reinterpret_cast<uint64_t>(output[
                    (c + i / 2U) * PadStride + hBase + (i % 2U) * 8U].GetPhyAddr());
            }
            if (repeats != 0U) TransDataTo5HD<int32_t>(dst, src, main);
            if (tailRows != 0U) {
#pragma unroll
                for (uint32_t i = 0U; i < 16U; ++i) {
                    const uint32_t safeH = fullRows + (i < tailRows ? i : 0U);
                    src[i] = reinterpret_cast<uint64_t>(input[c1Base + safeH * 8U].GetPhyAddr());
                    dst[i] = reinterpret_cast<uint64_t>(output[(c + i / 2U) * PadStride +
                        hBase + fullRows + (i % 2U) * 8U].GetPhyAddr());
                }
                TransDataTo5HD<int32_t>(dst, src, tail);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<int32_t> y_, work_;
    TEventID inputReady_[2], inputFree_[2];
    TEventID outputReady_[2], outputFree_[2];
};

class Split8FullHeightConsumer {
    const TransposeTilingData& t_;

    // Input/transpose use8 channels; only the packed-output ownership uses16.
    static constexpr uint32_t kChunkH=768, kInputTileC=8, kPaddedTileC=8;
    static constexpr uint32_t kTileC=16, InputBuffers=2, PackedBuffers=2;
    static constexpr bool OutputBypass=false, DropPattern=false, BuiltinMask=false;
    const uint32_t kH=t_.p[1], kC=t_.p[2], kC1=(kC+7)/8, kPaddedH=t_.p[4], PadStride=kPaddedH;
    const uint32_t kInputElems=kInputTileC*kChunkH, kPaddedElems=kPaddedTileC*PadStride, kPackedElems=kTileC*kPaddedH;
    const uint32_t kPaddedOffset=InputBuffers*kInputElems, kPackedOffset=kPaddedOffset+kPaddedElems;
    const uint32_t kPatternOffset=kPackedOffset+PackedBuffers*kPackedElems, kPatternWords=(PadStride+255)/256*8;
    const uint32_t kArenaBytes=(kPatternOffset+kPatternWords)*4;

public:
    __aicore__ explicit Split8FullHeightConsumer(const TransposeTilingData& t):t_(t){}
public:


    __aicore__ inline TPipe* GetPipe() { return &pipe_; }

    __aicore__ inline void Init(GM_ADDR output, GM_ADDR userWorkspace)
    {
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
        if constexpr (OutputBypass)
            y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        work_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace));
        pipe_.InitBuffer(arena_, kArenaBytes);
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (slot < InputBuffers) {
                inputReady_[slot] = pipe_.AllocEventID<HardEvent::MTE2_V>();
                inputFree_[slot] = pipe_.AllocEventID<HardEvent::V_MTE2>();
            }
            if (slot < PackedBuffers) {
                outputReady_[slot] = pipe_.AllocEventID<HardEvent::V_MTE3>();
                outputFree_[slot] = pipe_.AllocEventID<HardEvent::MTE3_V>();
            }
        }
        if constexpr (!DropPattern) {
            auto pattern = arena_.Get<uint32_t>()[kPatternOffset];

            for(uint32_t word=0;word<kPatternWords;++word) {
                const uint32_t start=word*32U;
                pattern.SetValue(word, start>=kH ? 0U : kH-start>=32U ? 0xffffffffU : (1U<<(kH-start))-1U);
            }
            const auto patternReady = pipe_.AllocEventID<HardEvent::S_V>();
            SetFlag<HardEvent::S_V>(patternReady);
            WaitFlag<HardEvent::S_V>(patternReady);
            pipe_.ReleaseEventID<HardEvent::S_V>(patternReady);
        }
    }

    __aicore__ inline void ProcessN(uint32_t n, uint32_t nSlot, uint32_t lane)
    {
        const uint32_t cBegin = lane * (kC / 32U * 16U);
        const uint32_t cEnd = lane == 0U ? kC / 32U * 16U : kC;
        ProcessRange(n, nSlot, cBegin, cEnd);
    }

    __aicore__ inline void ProcessRange(uint32_t n, uint32_t nSlot,
        uint32_t cBegin, uint32_t cEnd)
    {
        const uint32_t groups=(cEnd-cBegin+kTileC-1U)/kTileC;
        const uint32_t chunks=(cEnd-cBegin+kInputTileC-1U)/kInputTileC;
        // Existing normal/balanced ownership guarantees at least32 channels.
        for (uint32_t slot=0U;slot<InputBuffers;++slot)
            ReadChunk(nSlot,cBegin,cEnd,slot,slot);
        auto padded=arena_.Get<int32_t>()[kPaddedOffset];
        const auto pattern=arena_.Get<uint32_t>()[kPatternOffset];

        for (uint32_t cg=0U;cg<groups;++cg) {
            const uint32_t cStart=cBegin+cg*kTileC;
            const uint32_t channels=Min(kTileC,cEnd-cStart);
            const uint32_t packedSlot=cg%PackedBuffers;
            auto packed=arena_.Get<int32_t>()[kPackedOffset+packedSlot*kPackedElems];
            for (uint32_t localC=0U;localC<channels;localC+=kInputTileC) {
                const uint32_t chunk=cg*(kTileC/kInputTileC)+localC/kInputTileC;
                const uint32_t slot=chunk%InputBuffers;
                const uint32_t subChannels=Min(kInputTileC,channels-localC);
                WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
                TransposeChunk(padded,arena_.Get<int32_t>()[slot*kInputElems],
                    0U,kH,subChannels);
                // Only TransData reads input; VReduce reads independent padded UB.
                SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
                if (chunk+InputBuffers<chunks) {
                    WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
                    ReadChunk(nSlot,cBegin,cEnd,chunk+InputBuffers,slot);
                }
                // Retain old overlap: wait for packed reuse after first TransData.
                if (localC==0U && cg>=PackedBuffers)
                    WaitFlag<HardEvent::MTE3_V>(outputFree_[packedSlot]);
                PipeBarrier<PIPE_V>();
                SetMaskCount();
                set_vector_mask(0U,PadStride);
                // localC is0 or8, so localC*H words is always32B aligned.
                vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(packed[localC*kH].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(padded.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(pattern.GetPhyAddr()),
                    static_cast<uint16_t>(subChannels),1U,0U,PadStride/8U,0U);
                SetMaskNorm();
                // Critical: next C8 TransData reuses all padded rows. Join VReduce
                // before allowing that overwrite, while MTE3 only reads packed.
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(outputReady_[packedSlot]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[packedSlot]);
            const DataCopyExtParams copy{1U,channels*kH*4U,0U,0U,0U};
            DataCopyPad(y_[(n*kC+cStart)*kH],packed,copy);
            SetFlag<HardEvent::MTE3_V>(outputFree_[packedSlot]);
        }
        for (uint32_t slot=0U;slot<PackedBuffers;++slot)
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        for (uint32_t slot=0U;slot<InputBuffers;++slot)
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
    }

    __aicore__ inline void Release()
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (slot < InputBuffers) {
                pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady_[slot]);
                pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputFree_[slot]);
            }
            if (slot < PackedBuffers) {
                pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady_[slot]);
                pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputFree_[slot]);
            }
        }
    }

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const
    {
        return a < b ? a : b;
    }

    __aicore__ inline void ReadChunk(uint32_t nSlot, uint32_t cBegin,
        uint32_t cEnd, uint32_t chunk, uint32_t slot)
    {
        const uint32_t cStart=cBegin+chunk*kInputTileC;
        const uint32_t source=(nSlot*kC1+cStart/8U)*kPaddedH*8U;
        const DataCopyParams copy{1U,static_cast<uint16_t>(kH),
            static_cast<uint16_t>(kPaddedH-kH),static_cast<uint16_t>(kChunkH-kH)};
        DataCopy(arena_.Get<int32_t>()[slot*kInputElems],work_[source],copy);
        SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
    }

    __aicore__ inline void TransposeChunk(const LocalTensor<int32_t>& output,
        const LocalTensor<int32_t>& input, uint32_t hBase, uint32_t rows,
        uint32_t channels)
    {
        const uint32_t repeats = rows / 16U;
        const uint32_t fullRows = repeats * 16U;
        const uint32_t tailRows = rows - fullRows;
        TransDataTo5HDParams main;
        main.dstHighHalf = false;
        main.srcHighHalf = false;
        main.repeatTimes = static_cast<uint8_t>(repeats);
        main.srcRepStride = repeats == 1U ? 0U : 16U;
        main.dstRepStride = repeats == 1U ? 0U : 2U;
        TransDataTo5HDParams tail;
        tail.dstHighHalf = false;
        tail.srcHighHalf = false;
        tail.repeatTimes = 1U;
        tail.srcRepStride = 0U;
        tail.dstRepStride = 0U;
        uint64_t src[16], dst[16];
        for (uint32_t c = 0U; c < channels; c += 8U) {
            const uint32_t c1Base = (c / 8U) * kChunkH * 8U;
#pragma unroll
            for (uint32_t i = 0U; i < 16U; ++i) {
                src[i] = reinterpret_cast<uint64_t>(input[c1Base + i * 8U].GetPhyAddr());
                dst[i] = reinterpret_cast<uint64_t>(output[
                    (c + i / 2U) * PadStride + hBase + (i % 2U) * 8U].GetPhyAddr());
            }
            if (repeats != 0U) TransDataTo5HD<int32_t>(dst, src, main);
            if (tailRows != 0U) {
#pragma unroll
                for (uint32_t i = 0U; i < 16U; ++i) {
                    const uint32_t safeH = fullRows + (i < tailRows ? i : 0U);
                    src[i] = reinterpret_cast<uint64_t>(input[c1Base + safeH * 8U].GetPhyAddr());
                    dst[i] = reinterpret_cast<uint64_t>(output[(c + i / 2U) * PadStride +
                        hBase + fullRows + (i % 2U) * 8U].GetPhyAddr());
                }
                TransDataTo5HD<int32_t>(dst, src, tail);
            }
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<int32_t> y_, work_;
    TEventID inputReady_[2], inputFree_[2];
    TEventID outputReady_[2], outputFree_[2];
};

class MergedTailSplit8Consumer {
    const TransposeTilingData& t_;

    // Input/transpose use8 channels; only the packed-output ownership uses16.
    static constexpr uint32_t kChunkH=768, kInputTileC=8, kPaddedTileC=8;
    static constexpr uint32_t kTileC=16, InputBuffers=2, PackedBuffers=2;
    static constexpr bool OutputBypass=false, DropPattern=false, BuiltinMask=false;
    const uint32_t kH=t_.p[1], kC=t_.p[2], kC1=(kC+7)/8, kPaddedH=t_.p[4], PadStride=kPaddedH;
    const uint32_t kInputElems=kInputTileC*kChunkH, kPaddedElems=kPaddedTileC*PadStride, kPackedElems=kTileC*kPaddedH;
    const uint32_t kPaddedOffset=InputBuffers*kInputElems, kPackedOffset=kPaddedOffset+kPaddedElems;
    const uint32_t kPatternOffset=kPackedOffset+PackedBuffers*kPackedElems, kPatternWords=(PadStride+255)/256*8;
    const uint32_t kArenaBytes=(kPatternOffset+kPatternWords)*4;

public:
    __aicore__ explicit MergedTailSplit8Consumer(const TransposeTilingData& t):t_(t){}
public:


    __aicore__ inline TPipe* GetPipe() { return &pipe_; }

    __aicore__ inline void Init(GM_ADDR output, GM_ADDR userWorkspace)
    {
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(output));
        if constexpr (OutputBypass)
            y_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        work_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(userWorkspace));
        pipe_.InitBuffer(arena_, kArenaBytes);
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (slot < InputBuffers) {
                inputReady_[slot] = pipe_.AllocEventID<HardEvent::MTE2_V>();
                inputFree_[slot] = pipe_.AllocEventID<HardEvent::V_MTE2>();
            }
            if (slot < PackedBuffers) {
                outputReady_[slot] = pipe_.AllocEventID<HardEvent::V_MTE3>();
                outputFree_[slot] = pipe_.AllocEventID<HardEvent::MTE3_V>();
            }
        }
        // MERGED_TAIL_INIT: only padding rows, once for each persistent input slot.
        // V -> MTE2 -> first inputReady (MTE2 -> V) orders zero before TransData.
        for (uint32_t slot=0U;slot<InputBuffers;++slot) {
            auto tail=arena_.Get<int32_t>()[slot*kInputElems+kH*8U];
            const int32_t tailWords=static_cast<int32_t>(((kH+15U)/16U*16U-kH)*8U);
            Duplicate<int32_t>(tail,static_cast<int32_t>(0),tailWords);
            SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
        }
        if constexpr (!DropPattern) {
            auto pattern = arena_.Get<uint32_t>()[kPatternOffset];

            for(uint32_t word=0;word<kPatternWords;++word) {
                const uint32_t start=word*32U;
                pattern.SetValue(word, start>=kH ? 0U : kH-start>=32U ? 0xffffffffU : (1U<<(kH-start))-1U);
            }
            const auto patternReady = pipe_.AllocEventID<HardEvent::S_V>();
            SetFlag<HardEvent::S_V>(patternReady);
            WaitFlag<HardEvent::S_V>(patternReady);
            pipe_.ReleaseEventID<HardEvent::S_V>(patternReady);
        }
    }

    __aicore__ inline void ProcessN(uint32_t n, uint32_t nSlot, uint32_t lane)
    {
        const uint32_t cBegin = lane * (kC / 32U * 16U);
        const uint32_t cEnd = lane == 0U ? kC / 32U * 16U : kC;
        ProcessRange(n, nSlot, cBegin, cEnd);
    }

    __aicore__ inline void ProcessRange(uint32_t n, uint32_t nSlot,
        uint32_t cBegin, uint32_t cEnd)
    {
        const uint32_t groups=(cEnd-cBegin+kTileC-1U)/kTileC;
        const uint32_t chunks=(cEnd-cBegin+kInputTileC-1U)/kInputTileC;
        // Existing normal/balanced ownership guarantees at least32 channels.
        for (uint32_t slot=0U;slot<InputBuffers;++slot)
            ReadChunk(nSlot,cBegin,cEnd,slot,slot);
        auto padded=arena_.Get<int32_t>()[kPaddedOffset];
        const auto pattern=arena_.Get<uint32_t>()[kPatternOffset];

        for (uint32_t cg=0U;cg<groups;++cg) {
            const uint32_t cStart=cBegin+cg*kTileC;
            const uint32_t channels=Min(kTileC,cEnd-cStart);
            const uint32_t packedSlot=cg%PackedBuffers;
            auto packed=arena_.Get<int32_t>()[kPackedOffset+packedSlot*kPackedElems];
            for (uint32_t localC=0U;localC<channels;localC+=kInputTileC) {
                const uint32_t chunk=cg*(kTileC/kInputTileC)+localC/kInputTileC;
                const uint32_t slot=chunk%InputBuffers;
                const uint32_t subChannels=Min(kInputTileC,channels-localC);
                WaitFlag<HardEvent::MTE2_V>(inputReady_[slot]);
                TransposeChunk(padded,arena_.Get<int32_t>()[slot*kInputElems],
                    0U,kH,subChannels);
                // Only TransData reads input; VReduce reads independent padded UB.
                SetFlag<HardEvent::V_MTE2>(inputFree_[slot]);
                if (chunk+InputBuffers<chunks) {
                    WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
                    ReadChunk(nSlot,cBegin,cEnd,chunk+InputBuffers,slot);
                }
                // Retain old overlap: wait for packed reuse after first TransData.
                if (localC==0U && cg>=PackedBuffers)
                    WaitFlag<HardEvent::MTE3_V>(outputFree_[packedSlot]);
                PipeBarrier<PIPE_V>();
                SetMaskCount();
                set_vector_mask(0U,PadStride);
                // localC is0 or8, so localC*H words is always32B aligned.
                vreducev2(reinterpret_cast<__ubuf__ uint32_t*>(packed[localC*kH].GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(padded.GetPhyAddr()),
                    reinterpret_cast<__ubuf__ uint32_t*>(pattern.GetPhyAddr()),
                    static_cast<uint16_t>(subChannels),1U,0U,PadStride/8U,0U);
                SetMaskNorm();
                // Critical: next C8 TransData reuses all padded rows. Join VReduce
                // before allowing that overwrite, while MTE3 only reads packed.
                PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(outputReady_[packedSlot]);
            WaitFlag<HardEvent::V_MTE3>(outputReady_[packedSlot]);
            const DataCopyExtParams copy{1U,channels*kH*4U,0U,0U,0U};
            DataCopyPad(y_[(n*kC+cStart)*kH],packed,copy);
            SetFlag<HardEvent::MTE3_V>(outputFree_[packedSlot]);
        }
        for (uint32_t slot=0U;slot<PackedBuffers;++slot)
            WaitFlag<HardEvent::MTE3_V>(outputFree_[slot]);
        for (uint32_t slot=0U;slot<InputBuffers;++slot)
            WaitFlag<HardEvent::V_MTE2>(inputFree_[slot]);
    }

    __aicore__ inline void Release()
    {
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            if (slot < InputBuffers) {
                pipe_.ReleaseEventID<HardEvent::MTE2_V>(inputReady_[slot]);
                pipe_.ReleaseEventID<HardEvent::V_MTE2>(inputFree_[slot]);
            }
            if (slot < PackedBuffers) {
                pipe_.ReleaseEventID<HardEvent::V_MTE3>(outputReady_[slot]);
                pipe_.ReleaseEventID<HardEvent::MTE3_V>(outputFree_[slot]);
            }
        }
    }

private:
    __aicore__ inline uint32_t Min(uint32_t a, uint32_t b) const
    {
        return a < b ? a : b;
    }

    __aicore__ inline void ReadChunk(uint32_t nSlot, uint32_t cBegin,
        uint32_t cEnd, uint32_t chunk, uint32_t slot)
    {
        const uint32_t cStart=cBegin+chunk*kInputTileC;
        const uint32_t source=(nSlot*kC1+cStart/8U)*kPaddedH*8U;
        const DataCopyParams copy{1U,static_cast<uint16_t>(kH),
            static_cast<uint16_t>(kPaddedH-kH),static_cast<uint16_t>(kChunkH-kH)};
        DataCopy(arena_.Get<int32_t>()[slot*kInputElems],work_[source],copy);
        SetFlag<HardEvent::MTE2_V>(inputReady_[slot]);
    }

    __aicore__ inline void TransposeChunk(const LocalTensor<int32_t>& output,
        const LocalTensor<int32_t>& input, uint32_t hBase, uint32_t rows,
        uint32_t channels)
    {
        // Input padding was initialized once; ReadChunk never overwrites it.
        // Original VReduce retains only kH valid rows from every padded channel.
        const uint32_t repeats = (rows + 15U) / 16U;
        TransDataTo5HDParams main;
        main.dstHighHalf = false;
        main.srcHighHalf = false;
        main.repeatTimes = static_cast<uint8_t>(repeats);
        main.srcRepStride = repeats == 1U ? 0U : 16U;
        main.dstRepStride = repeats == 1U ? 0U : 2U;
        uint64_t src[16], dst[16];
        for (uint32_t c = 0U; c < channels; c += 8U) {
            const uint32_t c1Base = (c / 8U) * kChunkH * 8U;
#pragma unroll
            for (uint32_t i = 0U; i < 16U; ++i) {
                src[i] = reinterpret_cast<uint64_t>(input[c1Base + i * 8U].GetPhyAddr());
                dst[i] = reinterpret_cast<uint64_t>(output[
                    (c + i / 2U) * PadStride + hBase + (i % 2U) * 8U].GetPhyAddr());
            }
            if (repeats != 0U) TransDataTo5HD<int32_t>(dst, src, main);
        }
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> arena_;
    GlobalTensor<int32_t> y_, work_;
    TEventID inputReady_[2], inputFree_[2];
    TEventID outputReady_[2], outputFree_[2];
};


template<class Consumer=ReorderedCubeConsumer>
__aicore__ inline void RunReorderedCube(GM_ADDR input,GM_ADDR output,GM_ADDR workspace,const TransposeTilingData& t) {
    if(GetBlockNum()!=t.p[4])return;
    if ASCEND_IS_AIC {
        TPipe pipe;ReorderedCubeProducer producer(t);producer.Init(input,workspace,&pipe,true);
        producer.Process(GetBlockIdx());PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2U,PIPE_MTE3>(8U);CrossCoreWaitFlag<2U>(9U);producer.Finish();
    }
    if ASCEND_IS_AIV {
        Consumer consumer(t);consumer.Init(output,workspace);
        const uint32_t core=GetBlockIdx();CrossCoreWaitFlag<2U>(8U);
        consumer.Process(core/2,core%2);PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<2U,PIPE_MTE3>(9U);consumer.Release();
    }
}

// All pairs participate in every handshake, including idle final-wave pairs.
// Cross-owner writes in a balanced final wave require the preceding global
// reader join, and all producer Ready signals must be joined before reading.
template<typename ConsumerT>
__aicore__ inline void RunPackedCubeTyped(GM_ADDR input,GM_ADDR output,GM_ADDR workspace,const TransposeTilingData& t) {
    const uint32_t nCount=t.p[0],pairs=t.p[3],producers=t.p[5];
    const uint32_t rounds=(nCount+pairs-1)/pairs,last=rounds-1,base=last*pairs;
    if(GetBlockNum()!=pairs)return;
    if ASCEND_IS_AIC {
        TPipe pipe;PackedCubeProducer producer(t);producer.Init(input,workspace,&pipe,true);
        const uint32_t slot=GetBlockIdx();
        for(uint32_t round=0;round<rounds;++round) {
            const uint32_t n=round*pairs+slot;
            if(producers && round==last)producer.ProcessFinalSlice(base+slot/producers,slot/producers,slot%producers);
            else if(n<nCount)producer.ProcessN(n,slot);
            PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2U,PIPE_MTE3>(8U);CrossCoreWaitFlag<2U>(9U);
        }
        producer.Finish();
    }
    if ASCEND_IS_AIV {
        ConsumerT consumer(t);consumer.Init(output,workspace);
        const uint32_t core=GetBlockIdx(),slot=core/2,lane=core%2;
        for(uint32_t round=0;round<rounds;++round) {
            const uint32_t n=round*pairs+slot;CrossCoreWaitFlag<2U>(8U);
            if(producers && round==last) {
                SyncAll<true>();
                uint32_t readers=producers*2,owner=core/readers,part=core%readers,step=t.p[2]/readers/16*16;
                consumer.ProcessRange(base+owner,owner,part*step,part+1==readers?t.p[2]:(part+1)*step);
                SyncAll<true>();
            } else {
                if(n<nCount)consumer.ProcessN(n,slot,lane);
                if(producers && round+1==last)SyncAll<true>();
            }
            PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2U,PIPE_MTE3>(9U);
        }
        consumer.Release();
    }
}

// Prefetch the next normal wave into L1 while AIV consumes this wave.
template<typename ConsumerT>
__aicore__ inline void RunL1PrefetchSplit8(GM_ADDR input,GM_ADDR output,GM_ADDR workspace,const TransposeTilingData& t) {
    const uint32_t nCount=t.p[0],pairs=t.p[3],producers=t.p[5];
    const uint32_t rounds=(nCount+pairs-1)/pairs,last=rounds-1,base=last*pairs;
    if(GetBlockNum()!=pairs)return;
    if ASCEND_IS_AIC {
        TPipe pipe;PackedCubeProducer producer(t);producer.Init(input,workspace,&pipe,true);
        const uint32_t slot=GetBlockIdx();
        for(uint32_t round=0;round<rounds;++round) {
            const uint32_t n=round*pairs+slot;
            if(producers && round==last)producer.ProcessFinalSlice(base+slot/producers,slot/producers,slot%producers);
            else if(n<nCount)producer.ProcessN(n,slot,0U,round>0U && round<last);
            PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2U,PIPE_MTE3>(8U);
            // Only the next NORMAL wave is prefetched; final begins with empty local credits.
            if(round+1U<last)producer.PrefetchN((round+1U)*pairs+slot);
            CrossCoreWaitFlag<2U>(9U);
        }
        producer.Finish();
    }
    if ASCEND_IS_AIV {
        ConsumerT consumer(t);consumer.Init(output,workspace);
        const uint32_t core=GetBlockIdx(),slot=core/2,lane=core%2;
        for(uint32_t round=0;round<rounds;++round) {
            const uint32_t n=round*pairs+slot;CrossCoreWaitFlag<2U>(8U);
            if(producers && round==last) {
                SyncAll<true>();
                uint32_t readers=producers*2,owner=core/readers,part=core%readers,step=t.p[2]/readers/16*16;
                consumer.ProcessRange(base+owner,owner,part*step,part+1==readers?t.p[2]:(part+1)*step);
                SyncAll<true>();
            } else {
                if(n<nCount)consumer.ProcessN(n,slot,lane);
                if(producers && round+1==last)SyncAll<true>();
            }
            PipeBarrier<PIPE_ALL>();CrossCoreSetFlag<2U,PIPE_MTE3>(9U);
        }
        consumer.Release();
    }
}

__aicore__ inline void RunPackedCube(GM_ADDR input,GM_ADDR output,GM_ADDR workspace,const TransposeTilingData& t) {
    if(t.p[11]!=0U) {
        RunL1PrefetchSplit8<MergedTailSplit8Consumer>(input,output,workspace,t);
        return;
    }
    if(t.p[10]!=0U) {
        RunL1PrefetchSplit8<Split8FullHeightConsumer>(input,output,workspace,t);
        return;
    }
    // p9 is set by the paired host after full-height, 64-word pitch and actual UB checks.
    if(t.p[9]!=0U)
        RunPackedCubeTyped<Split8FullHeightConsumer>(input,output,workspace,t);
    else
        RunPackedCubeTyped<PackedCubeConsumer>(input,output,workspace,t);
}

extern "C" __global__ __aicore__ void transpose(GM_ADDR inputs,GM_ADDR output,
    __kfc_workspace__ GM_ADDR workspace,GM_ADDR tiling) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if(TILING_KEY_IS(1)) {
        if constexpr(SameElementType<DTYPE_INPUTS,half>) {
            TransposeTilingData data;LoadSmallMatrixTiling<3>(tiling,data);
            AlignedHalfTranspose op(data);op.Init(inputs,output);op.Process();
        }
        return;
    }
    // Only key1 uses compact metadata; key2 retains the full generated GET.
    GET_TILING_DATA(data,tiling);
    if(TILING_KEY_IS(2)) {
        if constexpr(SameElementType<DTYPE_INPUTS,int8_t>) {BalancedByteTranspose op(data);op.Init(inputs,output);op.Process();}
    } else if(TILING_KEY_IS(3)) {
        KERNEL_TASK_TYPE(3,KERNEL_TYPE_MIX_AIC_1_2);
        if constexpr(SameElementType<DTYPE_INPUTS,int32_t>) {
            if(data.p[8])RunReorderedCube<PackedReorderedCubeConsumer>(inputs,output,GetUserWorkspace(workspace),data);
            else RunReorderedCube<>(inputs,output,GetUserWorkspace(workspace),data);
        }
    } else if(TILING_KEY_IS(4)) {
        if constexpr(SameElementType<DTYPE_INPUTS,half>) {TPipe pipe;MaskedBlockTranspose op(data);op.Init(inputs,output,&pipe);op.Process();}
    } else if(TILING_KEY_IS(5)) {
        KERNEL_TASK_TYPE(5,KERNEL_TYPE_MIX_AIC_1_2);
        if constexpr(SameElementType<DTYPE_INPUTS,float>)RunPackedCube(inputs,output,GetUserWorkspace(workspace),data);
    } else if(TILING_KEY_IS(6)) {
        RowTransfer<DTYPE_INPUTS,6> op(data);op.Init(inputs,output);op.Process();
    } else if(TILING_KEY_IS(7)) {
        RowTransfer<DTYPE_INPUTS,7> op(data);op.Init(inputs,output);op.Process();
    } else if(TILING_KEY_IS(8)) {
        if constexpr(SameElementType<DTYPE_INPUTS,half>) {PaddedMatrixTranspose<half> op(data);op.Init(inputs,output);op.Process();}
    } else if(TILING_KEY_IS(9)) {
        if constexpr(sizeof(DTYPE_INPUTS)==4) {PaddedMatrixTranspose<int32_t> op(data);op.Init(inputs,output);op.Process();}
    } else if(TILING_KEY_IS(10)) {
        if constexpr(SameElementType<DTYPE_INPUTS,int8_t>) {PaddedMatrixTranspose<int8_t> op(data);op.Init(inputs,output);op.Process();}
    } else if(TILING_KEY_IS(11)) {
        RowTransfer<DTYPE_INPUTS,11> op(data);op.Init(inputs,output);op.Process();
    }
}
