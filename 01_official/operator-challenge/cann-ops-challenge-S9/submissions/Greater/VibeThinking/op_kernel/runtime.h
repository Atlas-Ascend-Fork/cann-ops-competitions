#ifndef S9_GREATER_RUNTIME_H
#define S9_GREATER_RUNTIME_H

#include "greater_tiling.h"

template<class CompareElement,uint32_t TYPE_INDEX>
struct GreaterTypeDescriptor {
    using Compare=CompareElement;
    static constexpr uint32_t index=TYPE_INDEX;
};

template<class Element>
struct GreaterTypeInfo;

template<>
struct GreaterTypeInfo<float> : GreaterTypeDescriptor<float,0> {};

template<>
struct GreaterTypeInfo<bfloat16_t> : GreaterTypeDescriptor<float,1> {};

template<>
struct GreaterTypeInfo<half> : GreaterTypeDescriptor<half,2> {};

template<>
struct GreaterTypeInfo<int32_t> : GreaterTypeDescriptor<int32_t,3> {};

template<>
struct GreaterTypeInfo<int8_t> : GreaterTypeDescriptor<half,4> {};

namespace s9g {

template<class Element,TPosition Region=TPosition::VECCALC>
S9G_INLINE LocalTensor<Element> Local(uint32_t byteAddress,uint32_t elements) {
    return LocalTensor<Element>(Region,byteAddress,elements);
}

template<class Element>
S9G_INLINE void ReadGlobal(const LocalTensor<Element>& local,
                           const GlobalTensor<Element>& global,
                           uint64_t elementOffset,uint32_t elements) {
    const uint64_t byteOffset=elementOffset*sizeof(Element);
    const uint32_t byteCount=elements*sizeof(Element);
    if ((byteOffset&31)==0 && (byteCount&31)==0) {
        DataCopy(local,global[elementOffset],elements);
        return;
    }
    const DataCopyExtParams transfer{1,byteCount,0,0,0};
    const DataCopyPadExtParams<Element> padding{false,0,0,static_cast<Element>(0)};
    DataCopyPad(local,global[elementOffset],transfer,padding);
}

S9G_INLINE void WriteGlobal(const GlobalTensor<uint8_t>& global,
                            const LocalTensor<uint8_t>& local,
                            uint64_t byteOffset,uint32_t bytes) {
    if ((byteOffset&31)==0 && (bytes&31)==0) {
        DataCopy(global[byteOffset],local,bytes);
        return;
    }
    const DataCopyExtParams transfer{1,bytes,0,0,0};
    DataCopyPad(global[byteOffset],local,transfer);
}

S9G_INLINE uint32_t BoundedCount(uint64_t position,uint64_t limit,uint32_t capacity) {
    const uint64_t remaining=limit-position;
    return remaining<capacity ? static_cast<uint32_t>(remaining):capacity;
}

}

// 缓冲复用前等待对应的计算和写回结束，防止流水间覆盖数据。
template<uint32_t SLOT_COUNT>
class GreaterPipelineSlots {
public:
    S9G_INLINE void MarkInputReader(uint32_t slot) {
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(slot));
        inputReader_[slot]=true;
    }

    S9G_INLINE void MarkOutputReader(uint32_t slot) {
        SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(slot));
        outputReader_[slot]=true;
    }

    S9G_INLINE void WaitInputReader(uint32_t slot) {
        if (!inputReader_[slot]) return;
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(slot));
        inputReader_[slot]=false;
    }

    S9G_INLINE void WaitOutputReader(uint32_t slot) {
        if (!outputReader_[slot]) return;
        WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(slot));
        outputReader_[slot]=false;
    }

    S9G_INLINE void Reuse(uint32_t slot) {
        WaitInputReader(slot);
        WaitOutputReader(slot);
    }

    S9G_INLINE void Drain() {
        for (uint32_t slot=0;slot<SLOT_COUNT;++slot) Reuse(slot);
    }

private:
    bool inputReader_[SLOT_COUNT]{};
    bool outputReader_[SLOT_COUNT]{};
};

class GreaterBroadcastCursor {
public:
    S9G_INLINE void Reset(const GreaterRows& plan,uint64_t rowIndex) {
        plan_=&plan;
        left=0;
        right=0;
        for (int32_t axis=static_cast<int32_t>(plan.rank)-2;axis>=0;--axis) {
            coordinate_[axis]=rowIndex%plan.dims[axis];
            rowIndex/=plan.dims[axis];
            left+=coordinate_[axis]*plan.stride1[axis];
            right+=coordinate_[axis]*plan.stride2[axis];
        }
    }

    S9G_INLINE uint64_t RowsBeforeWrap() const {
        if (plan_->rank<2) return 1;
        const uint32_t rowAxis=plan_->rank-2;
        return plan_->dims[rowAxis]-coordinate_[rowAxis];
    }

    S9G_INLINE void Move(uint32_t rows) {
        for (int32_t axis=static_cast<int32_t>(plan_->rank)-2;
             axis>=0 && rows;--axis) {
            const uint64_t beforeWrap=
                plan_->dims[axis]-coordinate_[axis];
            if (rows<beforeWrap) {
                coordinate_[axis]+=rows;
                left+=uint64_t(rows)*plan_->stride1[axis];
                right+=uint64_t(rows)*plan_->stride2[axis];
                return;
            }

            left-=coordinate_[axis]*plan_->stride1[axis];
            right-=coordinate_[axis]*plan_->stride2[axis];
            coordinate_[axis]=0;
            rows=1;
        }
    }

    uint64_t left=0;
    uint64_t right=0;

private:
    const GreaterRows* plan_=nullptr;
    uint64_t coordinate_[8]{};
};

class GreaterLinearRowCursor {
public:
    S9G_INLINE void Reset(const GreaterRows& t,uint64_t row){
        const uint32_t axis=t.rank-2;
        repeats_=t.dims[axis];remaining_=repeats_-row%repeats_;
        const bool leftShared=t.stride1[axis]==0;
        const uint64_t dense=row*t.inner,shared=(row/repeats_)*t.inner;
        left=leftShared ? shared:dense;right=leftShared ? dense:shared;
        denseA_=leftShared ? 0:t.inner;denseB_=leftShared ? t.inner:0;
        groupA_=leftShared ? t.inner:0;groupB_=leftShared ? 0:t.inner;
    }
    S9G_INLINE uint64_t RowsBeforeWrap() const{return remaining_;}
    S9G_INLINE void Move(uint32_t rows){
        left+=uint64_t(rows)*denseA_;right+=uint64_t(rows)*denseB_;remaining_-=rows;
        if(!remaining_){left+=groupA_;right+=groupB_;remaining_=repeats_;}
    }
    uint64_t left=0,right=0;
private:
    uint64_t repeats_=0,remaining_=0,denseA_=0,denseB_=0,groupA_=0,groupB_=0;
};

template<class Element>
class GreaterCompareEngine {
public:
    using CompareType=typename GreaterTypeInfo<Element>::Compare;
    static constexpr uint32_t TYPE_ID=GreaterTypeInfo<Element>::index;

    S9G_INLINE void Configure(uint32_t tile,uint32_t depth,
                              uint32_t cacheElements=0,bool rawCache=false,
                              bool rowStageInWork=false) {
        tile_=tile;
        cacheElements_=cacheElements;
        layout_=s9g::MakeLayout(TYPE_ID,tile,depth,cacheElements,
                                rawCache,rowStageInWork);
    }

    S9G_INLINE LocalTensor<Element> RawInput(uint32_t side,uint32_t slot=0) {
        const uint32_t base=side==1 ? layout_.in1:layout_.in2;
        return s9g::Local<Element,TPosition::VECIN>(
            base+slot*tile_*sizeof(Element),tile_);
    }

    S9G_INLINE LocalTensor<uint8_t> ByteOutput(uint32_t slot=0) {
        return s9g::Local<uint8_t,TPosition::VECOUT>(layout_.out+slot*tile_,tile_);
    }

    S9G_INLINE void PrepareConstants() {
        if constexpr(TYPE_ID==3)SetDeqScale(static_cast<half>(1));
        else Duplicate(ConstantTensor(),static_cast<half>(1),tile_);
        PipeBarrier<PIPE_V>();
    }

    S9G_INLINE LocalTensor<CompareType> ConvertInput(uint32_t side,uint32_t count,
                                                     uint32_t slot=0) {
        if constexpr (TYPE_ID==1 || TYPE_ID==4) {
            auto converted=s9g::Local<CompareType>(
                layout_.work+(side-1)*tile_*sizeof(CompareType),tile_);
            Cast(converted,RawInput(side,slot),RoundMode::CAST_NONE,count);
            return converted;
        }
        else return RawInput(side,slot);
    }

    S9G_INLINE void ExpandScalarRows(LocalTensor<CompareType>& values,
                                     uint32_t rows,uint32_t pitch) {
        SetFlag<HardEvent::V_S>(EVENT_ID4);
        WaitFlag<HardEvent::V_S>(EVENT_ID4);
        for (int32_t row=static_cast<int32_t>(rows)-1;row>=0;--row) {
            const CompareType scalar=values.GetValue(row*(32/sizeof(Element)));
            SetFlag<HardEvent::S_V>(EVENT_ID4);
            WaitFlag<HardEvent::S_V>(EVENT_ID4);
            Duplicate(values[row*pitch],scalar,pitch);
        }
        PipeBarrier<PIPE_V>();
    }

    S9G_INLINE void ComparePrepared(const LocalTensor<CompareType>& left,
                                    const LocalTensor<CompareType>& right,
                                    uint32_t count,uint32_t slot=0) {
        BuildPredicate(left,right,count);
        EncodePredicate(count,ByteOutput(slot));
    }

    S9G_INLINE void CompareInputs(uint32_t count,uint32_t slot) {
        auto left=ConvertInput(1,count,slot);
        auto right=ConvertInput(2,count,slot);
        ComparePrepared(left,right,count,slot);
    }

    S9G_INLINE void CompareScalarInput(uint32_t count,uint32_t slot,
                                       uint32_t scalarSide,CompareType scalar) {
        auto dense=ConvertInput(scalarSide==1 ? 2:1,count,slot);
        PipeBarrier<PIPE_V>();
        if constexpr (TYPE_ID==0 || TYPE_ID==2) {
            auto predicate=PredicateTensor();
            const uint32_t rounded=RoundPredicateCount(count);
            const auto mode=scalarSide==1 ? CMPMODE::LT:CMPMODE::GT;
            CompareScalar(predicate,dense,scalar,mode,rounded);
            EncodePredicate(count,ByteOutput(slot));
        } else if (scalarSide==1) {
            ComparePrepared(ScalarCache(),dense,count,slot);
        } else {
            ComparePrepared(dense,ScalarCache(),count,slot);
        }
    }

    S9G_INLINE CompareType LoadResidentScalar(const GlobalTensor<Element>& source) {
        auto raw=ScalarRaw();
        s9g::ReadGlobal(raw,source,0,1);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        auto cache=ScalarCache();
        if constexpr (TYPE_ID==1 || TYPE_ID==4)
            Cast(cache,raw,RoundMode::CAST_NONE,1);
        else
            DataCopy(cache,raw,32/sizeof(Element));
        SetFlag<HardEvent::V_S>(EVENT_ID4);
        WaitFlag<HardEvent::V_S>(EVENT_ID4);
        const CompareType scalar=cache.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID4);
        WaitFlag<HardEvent::S_V>(EVENT_ID4);
        if constexpr (TYPE_ID!=0 && TYPE_ID!=2) {
            Duplicate(cache,scalar,tile_);
            PipeBarrier<PIPE_V>();
        }
        return scalar;
    }

private:
    S9G_INLINE uint32_t RoundPredicateCount(uint32_t count) const {
        return static_cast<uint32_t>(s9g::Up(count,(TYPE_ID==3 || sizeof(CompareType)==2) ? 128:64));
    }

    S9G_INLINE LocalTensor<uint8_t> PredicateTensor() {
        return s9g::Local<uint8_t>(layout_.mask,(tile_+7)/8);
    }
    S9G_INLINE LocalTensor<half> ConstantTensor() {
        return s9g::Local<half>(layout_.constants,tile_);
    }
    S9G_INLINE LocalTensor<CompareType> ScalarCache() {
        return s9g::Local<CompareType>(layout_.cache,cacheElements_);
    }
    S9G_INLINE LocalTensor<Element> ScalarRaw() {
        return s9g::Local<Element,TPosition::VECIN>(layout_.scalarRaw,tile_);
    }

    S9G_INLINE void BuildPredicate(const LocalTensor<CompareType>& left,
                                   const LocalTensor<CompareType>& right,
                                   uint32_t count) {
        PipeBarrier<PIPE_V>();
        auto predicate=PredicateTensor();
        const uint32_t rounded=RoundPredicateCount(count);
        // int32 比较结合差值和原数符号，避免减法溢出改变比较结果。
        if constexpr (TYPE_ID==3) {

            auto word=s9g::Local<int32_t>(layout_.work,tile_);
            auto temporary=word.template ReinterpretCast<half>();
            auto signs=ConstantTensor();
            Sub(word,right,left,rounded);
            PipeBarrier<PIPE_V>();
            ShiftRight(word,word,int32_t(31),rounded);
            PipeBarrier<PIPE_V>();
            Cast(signs,word,RoundMode::CAST_NONE,rounded);
            PipeBarrier<PIPE_V>();
            ShiftRight(word,right,int32_t(31),rounded);
            PipeBarrier<PIPE_V>();
            Cast(temporary,word,RoundMode::CAST_NONE,rounded);
            PipeBarrier<PIPE_V>();
            Add(signs,signs,temporary,rounded);
            PipeBarrier<PIPE_V>();
            ShiftRight(word,left,int32_t(31),rounded);
            PipeBarrier<PIPE_V>();
            Cast(temporary,word,RoundMode::CAST_NONE,rounded);
            PipeBarrier<PIPE_V>();
            Sub(signs,signs,temporary,rounded);
            PipeBarrier<PIPE_V>();
            CompareScalar(predicate,signs,static_cast<half>(0),CMPMODE::LT,rounded);
        } else {
            Compare(predicate,left,right,CMPMODE::GT,rounded);
        }
    }

    // 将比较掩码展开为逐元素的 0 或 1，按 bool 的字节布局写回。
    S9G_INLINE void EncodePredicate(uint32_t count,
                                    const LocalTensor<uint8_t>& output) {
        PipeBarrier<PIPE_V>();
        auto selected=s9g::Local<half>(layout_.work,tile_);
        if constexpr(TYPE_ID==3){
            Duplicate(ConstantTensor(),static_cast<half>(1),128);
            PipeBarrier<PIPE_V>();
            const BinaryRepeatParams repeat{1,1,1,8,0,0};
            Select(selected,PredicateTensor(),ConstantTensor(),static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(128),
                static_cast<uint8_t>((count+127)/128),repeat);
        }else Select(selected,PredicateTensor(),ConstantTensor(),static_cast<half>(0),
                     SELMODE::VSEL_TENSOR_SCALAR_MODE,count);
        PipeBarrier<PIPE_V>();
        Cast(output,selected,RoundMode::CAST_NONE,count);
    }

    s9g::Layout layout_{};
    uint32_t tile_=0;
    uint32_t cacheElements_=0;
};

#endif
