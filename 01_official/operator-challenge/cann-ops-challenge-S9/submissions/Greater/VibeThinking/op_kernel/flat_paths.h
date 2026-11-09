#ifndef S9_GREATER_FLAT_PATHS_H
#define S9_GREATER_FLAT_PATHS_H

#include "runtime.h"

template<uint32_t ELEMENTS>
S9G_INLINE void RunFixedHalfPacket(GM_ADDR x1,GM_ADDR x2,GM_ADDR output) {
    const uint64_t offset=uint64_t(GetBlockIdx())*ELEMENTS;
    GlobalTensor<half> left;
    GlobalTensor<half> right;
    GlobalTensor<uint8_t> result;
    left.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x1)+offset);
    right.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x2)+offset);
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output)+offset);
    GreaterCompareEngine<half> engine;
    engine.Configure(ELEMENTS,1);
    DataCopy(engine.RawInput(1),left,ELEMENTS);
    DataCopy(engine.RawInput(2),right,ELEMENTS);
    engine.PrepareConstants();
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    engine.CompareInputs(ELEMENTS,0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    DataCopy(result,engine.ByteOutput(),ELEMENTS);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

template<class Element,bool EXACT_SIZE>
S9G_INLINE void RunSmallTensor(GM_ADDR x1,GM_ADDR x2,GM_ADDR output,
                               uint64_t perCore,uint64_t total=0) {
    if constexpr (EXACT_SIZE && GreaterTypeInfo<Element>::index==2) {
        switch (perCore) {
            case 128: RunFixedHalfPacket<128>(x1,x2,output); return;
            case 256: RunFixedHalfPacket<256>(x1,x2,output); return;
            case 512: RunFixedHalfPacket<512>(x1,x2,output); return;
            case 1024: RunFixedHalfPacket<1024>(x1,x2,output); return;
            case 2048: RunFixedHalfPacket<2048>(x1,x2,output); return;
            default: break;
        }
    }

    const uint64_t offset=uint64_t(GetBlockIdx())*perCore;
    uint32_t count=static_cast<uint32_t>(perCore);
    if constexpr (!EXACT_SIZE) {
        if (offset>=total) return;
        count=s9g::BoundedCount(offset,total,count);
    }
    GlobalTensor<Element> left;
    GlobalTensor<Element> right;
    GlobalTensor<uint8_t> result;
    left.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(x1)+offset);
    right.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(x2)+offset);
    result.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output)+offset);
    GreaterCompareEngine<Element> engine;
    engine.Configure(s9g::SMALL,1);
    if constexpr (EXACT_SIZE) {
        DataCopy(engine.RawInput(1),left,count);
        DataCopy(engine.RawInput(2),right,count);
    } else {
        s9g::ReadGlobal(engine.RawInput(1),left,0,count);
        s9g::ReadGlobal(engine.RawInput(2),right,0,count);
    }
    engine.PrepareConstants();
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    engine.CompareInputs(count,0);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    if constexpr (EXACT_SIZE)
        DataCopy(result,engine.ByteOutput(),count);
    else
        s9g::WriteGlobal(result,engine.ByteOutput(),0,count);
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

struct GreaterDenseRange {
    uint64_t begin;
    uint64_t end;
    uint64_t step;
    uint32_t tile;

    S9G_INLINE uint32_t Count(uint64_t position) const {
        return s9g::BoundedCount(position,end,tile);
    }
};

template<class Element,bool HAS_SCALAR>
class GreaterDensePipeline {
    using CompareType=typename GreaterTypeInfo<Element>::Compare;
public:
    S9G_INLINE void Run(GM_ADDR x1,GM_ADDR x2,GM_ADDR output,
                        const GreaterFlat& plan) {
        const bool wave=!HAS_SCALAR && (plan.flags&s9g::FLAG_WAVE);
        const GreaterDenseRange range=MakeRange(plan,wave);
        if (range.begin>=range.end) return;

        left_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(x1));
        right_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(x2));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
        scalarSide_=plan.scalarSide;
        ConfigureCache(plan.flags);

        const uint32_t resident=HAS_SCALAR ? ScalarCacheElements(plan.tile):0;
        vector_.Configure(plan.tile,plan.depth,resident);
        CompareType scalar=static_cast<CompareType>(0);
        if constexpr (HAS_SCALAR)
            scalar=vector_.LoadResidentScalar(scalarSide_==1 ? left_:right_);
        Execute(range,plan.depth,scalar);
    }

private:
    S9G_INLINE GreaterDenseRange MakeRange(const GreaterFlat& plan,bool wave) const {
        GreaterDenseRange range{};
        range.tile=plan.tile;
        range.begin=uint64_t(GetBlockIdx())*(wave ? plan.tile:plan.perCore);
        range.end=plan.total;
        if (!wave && range.begin<range.end && plan.perCore<range.end-range.begin)
            range.end=range.begin+plan.perCore;
        range.step=wave ? uint64_t(GetBlockNum())*plan.tile:plan.tile;
        return range;
    }

    S9G_INLINE uint32_t ScalarCacheElements(uint32_t tile) const {
        return GreaterTypeInfo<Element>::index==0 || GreaterTypeInfo<Element>::index==2 ? 128:tile;
    }

    S9G_INLINE void ConfigureCache(uint32_t flags) {
        if (!(flags&s9g::FLAG_BYPASS)) return;
        if (!HAS_SCALAR || scalarSide_!=1)
            left_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if (!HAS_SCALAR || scalarSide_!=2)
            right_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        output_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }

    S9G_INLINE void LoadBank(uint32_t bank,uint64_t position,uint32_t count) {
        if (!HAS_SCALAR || scalarSide_!=1)
            s9g::ReadGlobal(vector_.RawInput(1,bank),left_,position,count);
        if (!HAS_SCALAR || scalarSide_!=2)
            s9g::ReadGlobal(vector_.RawInput(2,bank),right_,position,count);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(bank));
    }

    S9G_INLINE void AcquireBank(uint32_t bank) {
        slots_.Reuse(bank);
    }

    S9G_INLINE void Execute(const GreaterDenseRange& range,uint32_t depth,
                            CompareType scalar) {
        uint64_t offsets[2]={range.begin,0};
        uint32_t counts[2]={range.Count(range.begin),0};
        LoadBank(0,offsets[0],counts[0]);
        vector_.PrepareConstants();

        uint32_t bank=0;
        for (uint64_t position=range.begin;position<range.end;position+=range.step) {
            const uint64_t nextPosition=position+range.step;
            const uint32_t nextBank=depth==2 ? bank^1:bank;
            if (depth==2 && nextPosition<range.end) {
                AcquireBank(nextBank);
                offsets[nextBank]=nextPosition;
                counts[nextBank]=range.Count(nextPosition);
                LoadBank(nextBank,offsets[nextBank],counts[nextBank]);
            }

            const event_t event=static_cast<event_t>(bank);
            WaitFlag<HardEvent::MTE2_V>(event);
            if constexpr (HAS_SCALAR)
                vector_.CompareScalarInput(counts[bank],bank,scalarSide_,scalar);
            else
                vector_.CompareInputs(counts[bank],bank);
            slots_.MarkInputReader(bank);
            SetFlag<HardEvent::V_MTE3>(event);
            WaitFlag<HardEvent::V_MTE3>(event);
            s9g::WriteGlobal(output_,vector_.ByteOutput(bank),offsets[bank],counts[bank]);
            slots_.MarkOutputReader(bank);

            if (depth==1 && nextPosition<range.end) {
                AcquireBank(0);
                offsets[0]=nextPosition;
                counts[0]=range.Count(nextPosition);
                LoadBank(0,offsets[0],counts[0]);
            }
            bank=nextBank;
        }
        AcquireBank(0);
        if (depth==2) AcquireBank(1);
    }

    GlobalTensor<Element> left_;
    GlobalTensor<Element> right_;
    GlobalTensor<uint8_t> output_;
    GreaterCompareEngine<Element> vector_;
    uint32_t scalarSide_=0;
    GreaterPipelineSlots<2> slots_;
};

class GreaterFp32PairPipeline {
public:
    S9G_INLINE void Run(GM_ADDR x1,GM_ADDR x2,GM_ADDR output,
                        const GreaterPairF32& plan) {
        tile_=plan.macro;
        emit_=s9g::PairEmitCapacity(tile_);
        maskBytes_=plan.scratch;
        layout_=s9g::MakePairLayout(tile_,plan.scratch);
        const uint64_t begin=uint64_t(GetBlockIdx())*plan.perCore;
        const uint64_t end=begin+plan.perCore<plan.total ? begin+plan.perCore:plan.total;
        if(begin>=end)return;
        left_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x1));
        right_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x2));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
        if(plan.flags&s9g::FLAG_BYPASS){
            left_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            right_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        Duplicate(Ones(),static_cast<half>(1),128);
        PipeBarrier<PIPE_V>();
        const uint64_t windowCapacity=uint64_t(plan.scratch)*8;
        for(uint64_t window=begin;window<end;){
            const uint64_t length=end-window<windowCapacity ? end-window:windowCapacity;
            const uint64_t windowEnd=window+length;
            Load(window,s9g::BoundedCount(window,windowEnd,tile_));

            for(uint64_t position=window;position<windowEnd;){
                const uint32_t count=s9g::BoundedCount(position,windowEnd,tile_);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                Compare(Predicate(position-window),LeftInput(),RightInput(),CMPMODE::GT,
                        static_cast<uint32_t>(s9g::Up(count,64)));
                PipeBarrier<PIPE_V>();
                position+=count;
                if(position<windowEnd){
                    ReleaseOperands();
                    Load(position,s9g::BoundedCount(position,windowEnd,tile_));
                }
            }

            for(uint64_t position=window;position<windowEnd;position+=emit_){
                const uint32_t count=s9g::BoundedCount(position,windowEnd,emit_);
                const BinaryRepeatParams select{1,1,0,8,0,0};
                Select(HalfWork(),Predicate(position-window),Ones(),static_cast<half>(0),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(128),
                       static_cast<uint8_t>((count+127)/128),select);
                PipeBarrier<PIPE_V>();
                slots_.WaitOutputReader(0);
                const UnaryRepeatParams pack{1,1,4,8};
                Cast(ByteOutput(),HalfWork(),RoundMode::CAST_NONE,uint64_t(128),
                     static_cast<uint8_t>((count+127)/128),pack);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                s9g::WriteGlobal(output_,ByteOutput(),position,count);
                slots_.MarkOutputReader(0);
            }
            slots_.Drain();
            window=windowEnd;
            if(window<end)ReleaseOperands();
        }
    }
private:
    S9G_INLINE LocalTensor<float> LeftInput(){return s9g::Local<float,TPosition::VECIN>(layout_.a,tile_);}
    S9G_INLINE LocalTensor<float> RightInput(){return s9g::Local<float,TPosition::VECIN>(layout_.b,tile_);}
    S9G_INLINE LocalTensor<uint8_t> Predicate(uint64_t offset){return s9g::Local<uint8_t>(layout_.predicate+static_cast<uint32_t>(offset/8),maskBytes_-static_cast<uint32_t>(offset/8));}
    S9G_INLINE LocalTensor<half> HalfWork(){return s9g::Local<half,TPosition::VECOUT>(layout_.halfWork,emit_);}
    S9G_INLINE LocalTensor<half> Ones(){return s9g::Local<half>(layout_.ones,128);}
    S9G_INLINE LocalTensor<uint8_t> ByteOutput(){return s9g::Local<uint8_t,TPosition::VECOUT>(layout_.output,emit_);}
    S9G_INLINE void ReleaseOperands(){
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    S9G_INLINE void Load(uint64_t position,uint32_t count){
        s9g::ReadGlobal(LeftInput(),left_,position,count);
        s9g::ReadGlobal(RightInput(),right_,position,count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    uint32_t tile_=0,emit_=0,maskBytes_=0;
    s9g::PairLayout layout_{};
    GreaterPipelineSlots<1> slots_;
    GlobalTensor<float> left_,right_;
    GlobalTensor<uint8_t> output_;
};

class GreaterInt8Stream {
public:
    S9G_INLINE bool Run(GM_ADDR x1,GM_ADDR x2,GM_ADDR y,const GreaterFlat& t) {
        if(t.flags&s9g::FLAG_WAVE)return false;
        plan_=s9g::MakeInt8WindowPlan(t.ubBytes,t.perCore);
        if(!plan_.tile)return false;
        const uint64_t begin=uint64_t(GetBlockIdx())*t.perCore;
        const uint64_t end=begin+t.perCore<t.total ? begin+t.perCore:t.total;
        if(begin>=end)return true;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x1));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x2));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if(t.flags&s9g::FLAG_BYPASS){
            a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        }
        Duplicate(Ones(),static_cast<half>(1),128);
        PipeBarrier<PIPE_V>();
        const uint64_t windowCapacity=uint64_t(plan_.maskBytes)*8;
        for(uint64_t window=begin;window<end;){
            const uint64_t length=end-window<windowCapacity ? end-window:windowCapacity;
            const uint64_t windowEnd=window+length;
            Load(window,s9g::BoundedCount(window,windowEnd,plan_.tile));
            for(uint64_t position=window;position<windowEnd;){
                const uint32_t count=s9g::BoundedCount(position,windowEnd,plan_.tile);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                PipeBarrier<PIPE_V>();
                Cast(Left(),RawA(),RoundMode::CAST_NONE,count);
                Cast(Right(),RawB(),RoundMode::CAST_NONE,count);
                const uint64_t next=position+count;
                if(next<windowEnd){

                    ReleaseInputs();
                    Load(next,s9g::BoundedCount(next,windowEnd,plan_.tile));
                }
                PipeBarrier<PIPE_V>();
                Compare(Mask(position-window),Left(),Right(),CMPMODE::GT,
                        static_cast<uint32_t>(s9g::Up(count,128)));
                PipeBarrier<PIPE_V>();
                position=next;
            }
            for(uint64_t position=window;position<windowEnd;position+=plan_.emit){
                const uint32_t count=s9g::BoundedCount(position,windowEnd,plan_.emit);
                const BinaryRepeatParams repeat{1,1,1,8,0,0};
                Select(Selection(),Mask(position-window),Ones(),static_cast<half>(0),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(128),
                       static_cast<uint8_t>((count+127)/128),repeat);
                PipeBarrier<PIPE_V>();
                slots_.WaitOutputReader(0);
                Cast(Output(),Selection(),RoundMode::CAST_NONE,count);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
                s9g::WriteGlobal(y_,Output(),position,count);
                slots_.MarkOutputReader(0);
            }
            slots_.Drain();
            window=windowEnd;
            if(window<end)ReleaseInputs();
        }
        return true;
    }
private:
    S9G_INLINE void ReleaseInputs(){
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }
    S9G_INLINE void Load(uint64_t offset,uint32_t count){
        s9g::ReadGlobal(RawA(),a_,offset,count);
        s9g::ReadGlobal(RawB(),b_,offset,count);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    S9G_INLINE LocalTensor<int8_t> RawA(){return s9g::Local<int8_t>(0,plan_.tile);}
    S9G_INLINE LocalTensor<int8_t> RawB(){return s9g::Local<int8_t>(plan_.tile,plan_.tile);}
    S9G_INLINE LocalTensor<half> Left(){return s9g::Local<half>(2*plan_.tile,plan_.tile);}
    S9G_INLINE LocalTensor<half> Right(){return s9g::Local<half>(4*plan_.tile,plan_.tile);}
    S9G_INLINE LocalTensor<half> Selection(){return s9g::Local<half>(0,plan_.emit);}
    S9G_INLINE LocalTensor<uint8_t> Output(){return s9g::Local<uint8_t,TPosition::VECOUT>(4*plan_.tile,plan_.emit);}
    S9G_INLINE LocalTensor<uint8_t> Mask(uint64_t offset){return s9g::Local<uint8_t>(6*plan_.tile+static_cast<uint32_t>(offset/8),plan_.maskBytes-static_cast<uint32_t>(offset/8));}
    S9G_INLINE LocalTensor<half> Ones(){return s9g::Local<half>(6*plan_.tile+plan_.maskBytes,128);}
    s9g::Int8WindowPlan plan_{};
    GreaterPipelineSlots<1> slots_;
    GlobalTensor<int8_t> a_,b_;
    GlobalTensor<uint8_t> y_;
};

#endif
