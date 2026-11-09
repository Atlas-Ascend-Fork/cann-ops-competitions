#ifndef S9_GREATER_ROW_PATHS_H
#define S9_GREATER_ROW_PATHS_H

#include "runtime.h"

template<class T>
class GreaterRowSlabs {
    using C=typename GreaterTypeInfo<T>::Compare;
    static constexpr uint32_t D=GreaterTypeInfo<T>::index;
    static constexpr uint32_t W=64;
public:
    __aicore__ inline bool Init(GM_ADDR x1,GM_ADDR x2,GM_ADDR y,const GreaterRows& t) {
        if constexpr (!(D==0 || D==1)) return false;
        if (t.rank<2 || t.innerBits || t.inner<32 || t.inner>UINT32_MAX/8) return false;
        const uint32_t axis=t.rank-2;
        if (t.stride1[axis]==0 && t.stride2[axis]==t.inner) side_=1;
        else if (t.stride2[axis]==0 && t.stride1[axis]==t.inner) side_=2;
        else return false;
        if (t.dims[axis]<2 || (t.inner-1)*sizeof(T)>UINT32_MAX) return false;
        row_=static_cast<uint32_t>(s9g::Up(t.inner,W));

        if (uint64_t(row_)*(sizeof(C)+2*sizeof(T))>t.ubBytes) return false;
        t_=&t;

        rows_=t.panelRows; pitch_=t.panelPitch;
        packed_=(t.reserved&s9g::ROW_PACKED_PREDICATE)!=0;
        if (packed_ && (!(D==0 || D==1) || pitch_%256 || uint64_t(rows_)*pitch_>32640)) return false;
        if (rows_<2 || rows_>128 || (!packed_ && rows_%2) || !pitch_ || pitch_%W || pitch_>32640) return false;
        n_=packed_ ? rows_*pitch_:rows_*W;
        const uint32_t horizontal=static_cast<uint32_t>(s9g::Up(pitch_,128));
        if (n_<horizontal) n_=horizontal;
        panelElements_=rows_*pitch_;
        raw_=row_*sizeof(C);
        rawCount_=panelElements_>row_ ? panelElements_:row_;
        converted_=raw_+2*rawCount_*sizeof(T);
        work_=converted_+(D==1 ? panelElements_*sizeof(C):0);
        mask_=work_+n_*2;
        output_=mask_+s9g::Up(n_/8,32);
        panelOutput_=output_;
        ones_=panelOutput_+panelElements_;
        if (ones_+256>t.ubBytes) return false;

        a_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x1));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x2));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if (t.policy==1) a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if (t.policy==2) b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        return true;
    }
    __aicore__ inline void Invalidate() { valid_=false; }
    __aicore__ inline bool Run(GreaterBroadcastCursor& cursor,uint64_t& offset,uint64_t end) {
        if (!t_ || offset%t_->inner || end-offset<t_->inner*2 || cursor.RowsBeforeWrap()<2) return false;
        uint64_t available=(end-offset)/t_->inner;
        if (available>cursor.RowsBeforeWrap()) available=cursor.RowsBeforeWrap();
        if (available>UINT32_MAX) available=UINT32_MAX;
        const uint32_t totalRows=static_cast<uint32_t>(available);
        const uint64_t base=side_==1 ? cursor.left:cursor.right;
        Refresh(base);
        const uint64_t denseBase=side_==1 ? cursor.right:cursor.left;
        uint32_t done=0,panel=0,current=0;
        uint32_t batch=totalRows<rows_ ? totalRows:rows_;
        LoadPanel(denseBase,0,0,batch,0);
        bool used[2]={false,false};
        bool outputPending=false;
        while (done<totalRows) {
            const uint32_t width=t_->inner-panel<pitch_ ? static_cast<uint32_t>(t_->inner-panel):pitch_;
            const uint32_t count=batch*W;
            uint32_t nextDone=done,nextPanel=panel+pitch_;
            if (nextPanel>=t_->inner) { nextPanel=0; nextDone+=batch; }
            const bool next=nextDone<totalRows;
            const uint32_t nextBatch=next ? (totalRows-nextDone<rows_ ? totalRows-nextDone:rows_):0;
            const uint32_t nextSlot=current^1;
            if (next) {
                if (used[nextSlot]) {
                    WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(6+nextSlot));
                    used[nextSlot]=false;
                }
                LoadPanel(denseBase,nextDone,nextPanel,nextBatch,nextSlot);
            }
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(6+current));
            PipeBarrier<PIPE_V>();
            auto dense=Dense(current);
            if constexpr (D==1) { Cast(dense,Raw(current),RoundMode::CAST_NONE,batch*pitch_); PipeBarrier<PIPE_V>(); }
            auto panelOutput=s9g::Local<uint8_t,TPosition::VECOUT>(panelOutput_,panelElements_);

            const bool compact=packed_ && pitch_>t_->inner && batch>1;
            if (packed_) {
                if constexpr (D==0 || D==1) Packed(panel,batch,width,dense,panelOutput,outputPending,current,compact);
            } else if (pitch_>1984 || batch<=(width+W-1)/W) {

                for (uint32_t row=0;row<batch;++row) {
                    auto resident=Resident()[panel];
                    auto changing=dense[row*pitch_];
                    auto left=side_==1 ? resident:changing;
                    auto right=side_==1 ? changing:resident;
                    const uint32_t compared=static_cast<uint32_t>(s9g::Up(width,W));
                    Compare(Mask(),left,right,CMPMODE::GT,compared);
                    PipeBarrier<PIPE_V>();
                    auto halfWork=s9g::Local<half>(work_,n_);
                    const BinaryRepeatParams emit{1,1,1,8,0,0};
                    Select(halfWork,Mask(),Ones(),static_cast<half>(0),
                        SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(128),
                        static_cast<uint8_t>((width+127)/128),emit);
                    PipeBarrier<PIPE_V>();
                    AcquireOutput(outputPending);
                    Cast(panelOutput[row*pitch_],halfWork,RoundMode::CAST_NONE,width);
                    PipeBarrier<PIPE_V>();
                }
            } else {
            for (uint32_t col=0;col<width;col+=W) {
                auto resident=Resident()[panel+col];
                auto changing=dense[col];
                const uint8_t rowStep=static_cast<uint8_t>(pitch_/8);
                const BinaryRepeatParams order{1,1,1,8,
                    static_cast<uint8_t>(side_==1 ? 0:rowStep),static_cast<uint8_t>(side_==1 ? rowStep:0)};
                auto left=side_==1 ? resident:changing;
                auto right=side_==1 ? changing:resident;
                {
                    Compare(Mask(),left,right,CMPMODE::GT,uint64_t(W),static_cast<uint8_t>(batch),order);
                }
                PipeBarrier<PIPE_V>();
                auto halfWork=s9g::Local<half>(work_,n_);
                const BinaryRepeatParams emit{1,1,1,8,0,0};
                Select(halfWork,Mask(),Ones(),static_cast<half>(0),
                    SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(128),
                    static_cast<uint8_t>((count+127)/128),emit);
                PipeBarrier<PIPE_V>();
                AcquireOutput(outputPending);

                const UnaryRepeatParams place{1,1,static_cast<uint8_t>(pitch_/32),4};
                Cast(panelOutput[col],halfWork,RoundMode::CAST_NONE,uint64_t(W),
                    static_cast<uint8_t>(batch),place);
                PipeBarrier<PIPE_V>();
            }
            }
            SetFlag<HardEvent::V_MTE3>(EVENT_ID6); WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
            const DataCopyExtParams out{static_cast<uint16_t>(batch),width,
                static_cast<uint32_t>((pitch_-s9g::Up(width,32))/32),static_cast<uint32_t>(t_->inner-width),0};
            if (compact)
                s9g::WriteGlobal(y_,panelOutput,offset+uint64_t(done)*t_->inner,batch*width);
            else if (pitch_==t_->inner && width==pitch_)
                s9g::WriteGlobal(y_,panelOutput,offset+uint64_t(done)*t_->inner,batch*pitch_);
            else DataCopyPad(y_[offset+uint64_t(done)*t_->inner+panel],panelOutput,out);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID6); outputPending=true;
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(6+current)); used[current]=true;
            if (!next) break;
            done=nextDone; panel=nextPanel; batch=nextBatch; current=nextSlot;
        }
        AcquireOutput(outputPending);
        for (uint32_t slot=0;slot<2;++slot)
            if (used[slot]) WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(6+slot));
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6); WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
        offset+=uint64_t(totalRows)*t_->inner; cursor.Move(totalRows);
        return true;
    }

private:

    __aicore__ inline void Packed(uint32_t panel,uint32_t batch,uint32_t width,
                                  LocalTensor<C> dense,LocalTensor<uint8_t> panelOutput,
                                  bool& outputPending,uint32_t inputSlot,bool compact) {
        const uint32_t compared=static_cast<uint32_t>(s9g::Up(width,W));
        for (uint32_t row=0;row<batch;++row) {
            auto resident=Resident()[panel],changing=dense[row*pitch_];

            auto left=side_==1 ? resident:changing;
            auto right=side_==1 ? changing:resident;
            Compare(Mask()[row*(pitch_/8)],left,right,CMPMODE::GT,compared);
        }
        PipeBarrier<PIPE_V>();
        const uint32_t count=batch*pitch_;
        auto halfWork=s9g::Local<half>(work_,n_);
        const BinaryRepeatParams emit{1,1,1,8,0,0};
        Select(halfWork,Mask(),Ones(),static_cast<half>(0),
            SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(128),static_cast<uint8_t>(count/128),emit);
        PipeBarrier<PIPE_V>();
        if (compact) {

            auto tight=s9g::Local<half>(raw_+inputSlot*rawCount_*sizeof(T),rawCount_*sizeof(T)/2);
            const GatherMaskParams rows{1,static_cast<uint16_t>(batch),
                static_cast<uint16_t>(pitch_/16),0};
            uint64_t retained=0;
            GatherMask(tight,halfWork,static_cast<uint8_t>(7),true,width,rows,retained);
            PipeBarrier<PIPE_V>();
            AcquireOutput(outputPending);
            Cast(panelOutput,tight,RoundMode::CAST_NONE,batch*width);
        } else {
            AcquireOutput(outputPending);
            Cast(panelOutput,halfWork,RoundMode::CAST_NONE,count);
        }
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void AcquireOutput(bool& pending) {
        if (pending) { WaitFlag<HardEvent::MTE3_V>(EVENT_ID6); pending=false; }
    }
    __aicore__ inline void LoadPanel(uint64_t base,uint32_t rowsDone,uint32_t panel,
                                     uint32_t batch,uint32_t slot) {
        const uint32_t width=t_->inner-panel<pitch_ ? static_cast<uint32_t>(t_->inner-panel):pitch_;
        const DataCopyExtParams p{static_cast<uint16_t>(batch),static_cast<uint32_t>(width*sizeof(T)),
            static_cast<uint32_t>((t_->inner-width)*sizeof(T)),
            static_cast<uint32_t>((pitch_*sizeof(T)-s9g::Up(width*sizeof(T),32))/32),0};
        const DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
        if (pitch_==t_->inner && width==pitch_)
            s9g::ReadGlobal(Raw(slot),side_==1 ? b_:a_,base+uint64_t(rowsDone)*t_->inner,batch*pitch_);
        else DataCopyPad(Raw(slot),(side_==1 ? b_:a_)[base+uint64_t(rowsDone)*t_->inner+panel],p,pad);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(6+slot));
    }
    __aicore__ inline LocalTensor<T> Raw(uint32_t slot=0) { return s9g::Local<T,TPosition::VECIN>(raw_+slot*rawCount_*sizeof(T),rawCount_); }
    __aicore__ inline LocalTensor<C> Resident() { return s9g::Local<C>(0,row_); }
    __aicore__ inline LocalTensor<C> Dense(uint32_t slot=0) { return s9g::Local<C>(D==1 ? converted_:raw_+slot*rawCount_*sizeof(T),panelElements_); }
    __aicore__ inline LocalTensor<uint8_t> Mask() { return s9g::Local<uint8_t>(mask_,s9g::Up(n_/8,32)); }
    __aicore__ inline LocalTensor<half> Ones() { return s9g::Local<half>(ones_,128); }
    __aicore__ inline void Refresh(uint64_t base) {
        if (valid_ && base_==base) return;
        if constexpr (D==1) {
            s9g::ReadGlobal(Raw(),side_==1 ? a_:b_,base,static_cast<uint32_t>(t_->inner));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID6); WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
            Cast(Resident(),Raw(),RoundMode::CAST_NONE,static_cast<uint32_t>(t_->inner));
        } else {
            s9g::ReadGlobal(Resident(),side_==1 ? a_:b_,base,static_cast<uint32_t>(t_->inner));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID6); WaitFlag<HardEvent::MTE2_V>(EVENT_ID6);
        }

        if (packed_ && !valid_) Duplicate(s9g::Local<int32_t>(mask_,n_/32),int32_t(0),n_/32);
        Duplicate(Ones(),static_cast<half>(1),128);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID6); WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
        base_=base; valid_=true;
    }
    const GreaterRows* t_=nullptr;
    uint32_t side_=0,row_=0,rows_=0,n_=0,raw_=0,rawCount_=0,converted_=0,work_=0,mask_=0,output_=0,ones_=0;
    uint32_t pitch_=0,panelElements_=0,panelOutput_=0;
    uint64_t base_=0; bool valid_=false,packed_=false;
    GlobalTensor<T> a_,b_; GlobalTensor<uint8_t> y_;
};

template<uint32_t SharedSide=0>
class GreaterCompactInt32Rows {
    static_assert(s9g::INT32_INPUTS==1,"Role rotation requires one logical input slot");
public:
    __aicore__ inline bool Init(GM_ADDR x1,GM_ADDR x2,GM_ADDR y,const GreaterRows& t){
        if(t.dtype!=3 || !(t.reserved&s9g::ROW_COMPACT_RESIDENT) || t.rank<2 || t.innerBits)return false;
        const uint32_t axis=t.rank-2;
        if(t.stride1[axis]==0 && t.stride2[axis]==t.inner)side_=1;
        else if(t.stride2[axis]==0 && t.stride1[axis]==t.inner)side_=2;
        else return false;
        if(t.inner<256 || t.inner>32640)return false;
        l_=s9g::MakeCompactRowLayout(static_cast<uint32_t>(t.inner),t.panelRows,t.dims[axis]);
        if(!l_.bytes || l_.bytes>t.ubBytes || l_.fixes>256)return false;
        inner_=static_cast<uint32_t>(t.inner);panelRows_=t.panelRows;
        PrepareConstants();
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x1));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x2));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if(t.policy==1)a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if(t.policy==2)b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        return true;
    }
    __aicore__ inline bool Init(GM_ADDR x1,GM_ADDR x2,GM_ADDR y,const GreaterLinearRows& t){
        side_=t.side;
        l_=s9g::MakeCompactRowLayout(t.inner,t.panelRows,t.groupRows);
        if(!l_.bytes || l_.bytes>t.ubBytes)return false;
        inner_=t.inner;panelRows_=t.panelRows;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x1));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(x2));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if(t.policy==1)a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if(t.policy==2)b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        PrepareConstants();
        return true;
    }
    __aicore__ inline void Invalidate(){Drain();valid_=false;}
    __aicore__ inline void Drain(){
        if(!inner_ || !(active_ || rowRead_))return;
        for(uint32_t s=0;s<s9g::INT32_OUTPUT_BANKS;++s)if(pending_[s]){
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(6+s));pending_[s]=false;
        }
        for(uint32_t s=0;s<s9g::INT32_INPUTS;++s)if(used_[s]){
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(5+s));used_[s]=false;
        }
        if(rowRead_){WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);rowRead_=false;}
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        nextSlot_=0;nextOutput_=0;active_=false;
    }
    template<class Cursor>
    __aicore__ inline bool Run(Cursor& cursor,uint64_t& offset,uint64_t end){
        if(!inner_ || offset>=end || offset%inner_)return false;
        active_=true;
        const uint32_t inner=static_cast<uint32_t>(inner_),col=static_cast<uint32_t>(offset%inner);
        uint64_t total=cursor.RowsBeforeWrap()*inner_-col;
        if(total>end-offset)total=end-offset;
        const uint64_t maxChunk=(uint64_t(UINT32_MAX)/inner)*inner;
        if(total>maxChunk)total=maxChunk;
        const uint64_t reuseBase=SharedLeft() ? cursor.left:cursor.right;
        const bool updating=BeginResidentUpdate(reuseBase);

        if(rowRead_){WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);rowRead_=false;}
        const uint64_t denseBase=(SharedLeft() ? cursor.right:cursor.left)+col;
        uint64_t done=0;
        uint32_t slot=nextSlot_,outSlot=nextOutput_,count=StreamCount(total);
        uint64_t submitted=0;
        uint32_t loadSlot=slot;
        for(uint32_t k=0;k<s9g::INT32_INPUTS && submitted<total;++k){
            const uint32_t take=StreamCount(total-submitted);
            if(used_[loadSlot]){WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(5+loadSlot));used_[loadSlot]=false;}
            Load(denseBase,submitted,take,loadSlot);
            submitted+=take;loadSlot=(loadSlot+1)%s9g::INT32_INPUTS;
        }
        if(updating)CompleteResidentUpdate(reuseBase);
        while(done<total){
            const uint64_t nextDone=done+count;
            const bool next=nextDone<total;
            const uint32_t nextCount=next ? StreamCount(total-nextDone):0;
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(5+slot));PipeBarrier<PIPE_V>();
            const uint32_t vectorCount=static_cast<uint32_t>(s9g::Up(count,128));
            const bool wholeRepeat=vectorCount<=16320;
            auto dense=Raw(slot);
            auto signWork=s9g::Local<half>(l_.maximum,l_.n);
            const uint32_t maskBytes=static_cast<uint32_t>(s9g::Up(l_.n/8,32));
            auto mask=s9g::Local<uint8_t>(l_.mask,maskBytes);
            auto sharedMask=s9g::Local<uint8_t>(l_.mask+maskBytes,maskBytes);
            auto differenceMask=s9g::Local<uint8_t>(l_.mask+2*maskBytes,maskBytes);

            AcquireSignArena();
            Cast(signWork,dense,RoundMode::CAST_NONE,vectorCount);
            PipeBarrier<PIPE_V>();
            CompareScalar(mask,signWork,static_cast<half>(0),
                          SharedLeft() ? CMPMODE::LT:CMPMODE::GE,vectorCount);
            PipeBarrier<PIPE_V>();
            auto shared=Expanded();
            if(SharedLeft())Sub(dense,dense,shared,vectorCount);
            else Sub(dense,shared,dense,vectorCount);
            PipeBarrier<PIPE_V>();
            const bool rotated=submitted<total;
            if(rotated){

                SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(5+slot));
                WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(5+slot));
                rawAtExpanded_=!rawAtExpanded_;
                const uint32_t take=StreamCount(total-submitted);
                Load(denseBase,submitted,take,loadSlot);
                submitted+=take;
            }

            Cast(signWork,dense,RoundMode::CAST_NONE,vectorCount);
            PipeBarrier<PIPE_V>();
            if(rotated){

                ExpandResidentIntegers();
            }else{
                SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(5+slot));used_[slot]=true;
            }
            CompareScalar(differenceMask,signWork,static_cast<half>(0),CMPMODE::LT,vectorCount);
            PipeBarrier<PIPE_V>();

            auto combined=signWork.template ReinterpretCast<int16_t>();
            auto dynamicBits=mask.template ReinterpretCast<int16_t>();
            auto sharedBits=sharedMask.template ReinterpretCast<int16_t>();
            auto differenceBits=differenceMask.template ReinterpretCast<int16_t>();
            const uint32_t words=vectorCount/16;
            Or(combined,dynamicBits,sharedBits,words);
            PipeBarrier<PIPE_V>();
            And(combined,combined,differenceBits,words);
            PipeBarrier<PIPE_V>();
            And(dynamicBits,dynamicBits,sharedBits,words);
            PipeBarrier<PIPE_V>();
            Or(dynamicBits,dynamicBits,combined,words);
            PipeBarrier<PIPE_V>();
            if(!next){SetFlag<HardEvent::V_MTE2>(EVENT_ID4);rowRead_=true;}
            auto halfWork=s9g::Local<half>(l_.halfWork,l_.n);
            const BinaryRepeatParams emit{1,1,1,8,0,0};
            Select(halfWork,mask,s9g::Local<half>(l_.ones,128),static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(128),static_cast<uint8_t>((count+127)/128),emit);
            PipeBarrier<PIPE_V>();

            auto output=s9g::Local<uint8_t,TPosition::VECOUT>(l_.output,l_.n);
            if(wholeRepeat){
                const UnaryRepeatParams contiguousCast{1,1,4,8};
                Cast(output,halfWork,RoundMode::CAST_NONE,uint64_t(128),static_cast<uint8_t>(vectorCount/128),contiguousCast);
            }else Cast(output,halfWork,RoundMode::CAST_NONE,count);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(6+outSlot));
            WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(6+outSlot));
            s9g::WriteGlobal(y_,output,offset+done,count);
            SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(6+outSlot));pending_[outSlot]=true;
            outSlot=0;
            if(!next)break;
            done=nextDone;count=nextCount;slot=(slot+1)%s9g::INT32_INPUTS;
        }
        nextSlot_=loadSlot;nextOutput_=outSlot;
        offset+=total;cursor.Move(static_cast<uint32_t>((col+total)/inner));
        return true;
    }
private:
    S9G_INLINE bool SharedLeft() const {
        if constexpr(SharedSide!=0)return SharedSide==1;
        return side_==1;
    }

    S9G_INLINE void AcquireSignArena(){
        if(pending_[0]){WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);pending_[0]=false;}
    }
    S9G_INLINE void PrepareConstants(){
        SetDeqScale(static_cast<half>(1));
        PipeBarrier<PIPE_V>();
        Duplicate(s9g::Local<half>(l_.ones,128),static_cast<half>(1),128);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline LocalTensor<int32_t> Raw(uint32_t slot){
        (void)slot;
        return s9g::Local<int32_t,TPosition::VECIN>(rawAtExpanded_ ? l_.expanded:l_.raw,l_.n);
    }
    __aicore__ inline LocalTensor<int32_t> Expanded(){
        return s9g::Local<int32_t>(rawAtExpanded_ ? l_.raw:l_.expanded,l_.n);
    }
    S9G_INLINE uint32_t StreamCount(uint64_t remaining) { return s9g::BoundedCount(0,remaining,l_.logical); }
    __aicore__ inline void Load(uint64_t base,uint64_t done,uint32_t count,uint32_t slot){
        const DataCopyExtParams p{1,static_cast<uint32_t>(count*sizeof(int32_t)),0,0,0};
        const DataCopyPadExtParams<int32_t> pad{false,0,0,0};
        DataCopyPad(Raw(slot),(SharedLeft() ? b_:a_)[base+done],p,pad);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(5+slot));
    }

    __aicore__ inline bool BeginResidentUpdate(uint64_t base){
        if(valid_ && base_==base)return false;
        if(rowRead_){WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);rowRead_=false;}
        s9g::ReadGlobal(s9g::Local<int32_t>(l_.resident,l_.row),SharedLeft() ? a_:b_,base,
                  static_cast<uint32_t>(inner_));
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);return true;
    }
    __aicore__ inline void ExpandResidentIntegers(){
        const GatherMaskParams repeatRow(1,static_cast<uint16_t>(panelRows_),0,0);
        uint64_t retained=0;
        auto row=s9g::Local<int32_t>(l_.resident,l_.row);
        GatherMask<int32_t>(Expanded(),row,7,true,inner_,repeatRow,retained);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void CompleteResidentUpdate(uint64_t base){
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        ExpandResidentIntegers();

        auto signWork=s9g::Local<half>(l_.maximum,l_.n);
        AcquireSignArena();
        Cast(signWork,Expanded(),RoundMode::CAST_NONE,l_.n);
        PipeBarrier<PIPE_V>();
        const uint32_t maskBytes=static_cast<uint32_t>(s9g::Up(l_.n/8,32));
        auto sharedMask=s9g::Local<uint8_t>(l_.mask+maskBytes,maskBytes);
        CompareScalar(sharedMask,signWork,static_cast<half>(0),
                      SharedLeft() ? CMPMODE::GE:CMPMODE::LT,l_.n);
        PipeBarrier<PIPE_V>();
        base_=base;valid_=true;
    }
    uint32_t inner_=0,panelRows_=0;
    s9g::CompactRowLayout l_{};
    bool used_[s9g::INT32_INPUTS]={},pending_[s9g::INT32_OUTPUT_BANKS]={},active_=false;
    uint32_t nextSlot_=0,nextOutput_=0;
    uint32_t side_=0;uint64_t base_=0;bool valid_=false,rowRead_=false,rawAtExpanded_=false;
    GlobalTensor<int32_t> a_,b_;GlobalTensor<uint8_t> y_;
};

template<class Cursor>
S9G_INLINE bool RunGreaterCompactRows(GM_ADDR x1,GM_ADDR x2,GM_ADDR output,
                                      const GreaterRows& plan) {
    GreaterCompactInt32Rows<> path;
    if (!path.Init(x1,x2,output,plan)) return false;

    if (plan.reserved&s9g::ROW_CYCLIC_MACROS) {
        const uint64_t macroRows=
            uint64_t(plan.panelRows)*s9g::COMPACT_MACRO_PANELS;
        const uint64_t jobs=(plan.total/plan.inner+macroRows-1)/macroRows;
        for (uint64_t job=GetBlockIdx();job<jobs;job+=GetBlockNum()) {
            const auto interval=s9g::MacroRowInterval(
                plan.total,static_cast<uint32_t>(plan.inner),plan.panelRows,job);
            uint64_t position=interval.begin;
            Cursor cursor;
            cursor.Reset(plan,position/plan.inner);
            while (position<interval.end)
                path.Run(cursor,position,interval.end);
        }
        path.Drain();
        return true;
    }

    const auto interval=s9g::BalancedRowInterval(
        plan.total,static_cast<uint32_t>(plan.inner),GetBlockIdx(),GetBlockNum());
    if (interval.begin<interval.end) {
        uint64_t position=interval.begin;
        Cursor cursor;
        cursor.Reset(plan,position/plan.inner);
        while (position<interval.end)
            path.Run(cursor,position,interval.end);
        path.Drain();
    }
    return true;
}

template<uint32_t SharedSide>
class GreaterGeometryCursor {
public:
    S9G_INLINE void Reset(const GreaterLinearRows& t,uint64_t row){
        inner_=t.inner;repeats_=t.groupRows;remaining_=repeats_-row%repeats_;
        const uint64_t dense=row*inner_,shared=(row/repeats_)*inner_;
        left=SharedSide==1 ? shared:dense;right=SharedSide==1 ? dense:shared;
    }
    S9G_INLINE uint64_t RowsBeforeWrap()const{return remaining_;}
    S9G_INLINE void Move(uint32_t rows){
        if constexpr(SharedSide==1)right+=uint64_t(rows)*inner_;
        else left+=uint64_t(rows)*inner_;
        remaining_-=rows;
        if(!remaining_){
            if constexpr(SharedSide==1)left+=inner_;else right+=inner_;
            remaining_=repeats_;
        }
    }
    uint64_t left=0,right=0;
private:
    uint64_t repeats_=0,remaining_=0;
    uint32_t inner_=0;
};

template<uint32_t SharedSide>
S9G_INLINE void RunGreaterGeometry(GM_ADDR x1,GM_ADDR x2,GM_ADDR output,
                                   const GreaterLinearRows& plan){
    const auto span=s9g::BalancedRowInterval(plan.total,plan.inner,GetBlockIdx(),GetBlockNum());
    if(span.begin>=span.end)return;
    GreaterCompactInt32Rows<SharedSide> path;
    if(!path.Init(x1,x2,output,plan))return;
    GreaterGeometryCursor<SharedSide> cursor;
    uint64_t position=span.begin;
    cursor.Reset(plan,position/plan.inner);
    while(position<span.end)path.Run(cursor,position,span.end);
    path.Drain();
}

template<uint32_t SharedSide=0>
class GreaterCompactBf16Rows {
public:
    __aicore__ inline bool Init(GM_ADDR x1,GM_ADDR x2,GM_ADDR y,const GreaterLinearRows& t){
        side_=t.side;
        l_=s9g::MakeBf16RowLayout(t.inner,t.panelRows,t.groupRows);
        if(!l_.bytes || l_.bytes>t.ubBytes)return false;
        inner_=t.inner;panelRows_=t.panelRows;
        a_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x1));
        b_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x2));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y));
        if(t.policy==1)a_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if(t.policy==2)b_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        PrepareConstants();
        return true;
    }
    __aicore__ inline void Invalidate(){Drain();valid_=false;}
    __aicore__ inline void Drain(){
        if(!inner_ || !(active_ || rowRead_))return;
        for(uint32_t s=0;s<s9g::COMPACT_OUTPUT_BANKS;++s)if(pending_[s]){
            WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(6+s));pending_[s]=false;
        }
        for(uint32_t s=0;s<s9g::COMPACT_INPUTS;++s)if(used_[s]){
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(5+s));used_[s]=false;
        }
        if(rowRead_){WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);rowRead_=false;}
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID4);
        nextSlot_=0;nextOutput_=0;active_=false;
    }
    template<class Cursor>
    __aicore__ inline bool Run(Cursor& cursor,uint64_t& offset,uint64_t end){
        if(!inner_ || offset>=end || offset%inner_)return false;
        active_=true;
        const uint32_t inner=static_cast<uint32_t>(inner_),col=static_cast<uint32_t>(offset%inner);
        uint64_t total=cursor.RowsBeforeWrap()*inner_-col;
        if(total>end-offset)total=end-offset;
        const uint64_t maxChunk=(uint64_t(UINT32_MAX)/inner)*inner;
        if(total>maxChunk)total=maxChunk;
        const uint64_t reuseBase=SharedLeft() ? cursor.left:cursor.right;
        const bool updating=BeginResidentUpdate(reuseBase);
        const uint64_t denseBase=(SharedLeft() ? cursor.right:cursor.left)+col;
        uint64_t done=0;
        uint32_t slot=nextSlot_,outSlot=nextOutput_,count=StreamCount(total);
        uint64_t submitted=0,outStart=heldStart_;
        uint32_t bankFill=heldFill_;
        uint32_t loadSlot=slot;
        for(uint32_t k=0;k<s9g::COMPACT_INPUTS && submitted<total;++k){
            const uint32_t take=StreamCount(total-submitted);
            if(used_[loadSlot]){WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(5+loadSlot));used_[loadSlot]=false;}
            Load(denseBase,submitted,take,loadSlot);
            submitted+=take;loadSlot=(loadSlot+1)%s9g::COMPACT_INPUTS;
        }
        if(updating)CompleteResidentUpdate(reuseBase);
        while(done<total){
            const uint64_t nextDone=done+count;
            const bool next=nextDone<total;
            const uint32_t nextCount=next ? StreamCount(total-nextDone):0;
            WaitFlag<HardEvent::MTE2_V>(static_cast<event_t>(5+slot));PipeBarrier<PIPE_V>();
            const uint32_t vectorCount=static_cast<uint32_t>(s9g::Up(count,128));
            const bool wholeRepeat=vectorCount<=16320;
            auto dense=s9g::Local<float>(l_.maximum,l_.n);
            Cast(dense,Raw(slot),RoundMode::CAST_NONE,count);
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(5+slot));used_[slot]=true;
            if(submitted<total){
                const uint32_t take=StreamCount(total-submitted);
                if(used_[loadSlot]){WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(5+loadSlot));used_[loadSlot]=false;}
                Load(denseBase,submitted,take,loadSlot);
                submitted+=take;loadSlot=(loadSlot+1)%s9g::COMPACT_INPUTS;
            }
            auto resident=s9g::Local<float>(l_.expanded,l_.n);
            auto mask=s9g::Local<uint8_t>(l_.mask,static_cast<uint32_t>(s9g::Up(l_.n/8,32)));
            if(SharedLeft())Compare(mask,resident,dense,CMPMODE::GT,static_cast<uint32_t>(s9g::Up(count,64)));
            else Compare(mask,dense,resident,CMPMODE::GT,static_cast<uint32_t>(s9g::Up(count,64)));
            PipeBarrier<PIPE_V>();
            auto halfWork=s9g::Local<half>(l_.halfWork,l_.n);
            const BinaryRepeatParams emit{1,1,1,8,0,0};
            Select(halfWork,mask,s9g::Local<half>(l_.ones,128),static_cast<half>(0),
                SELMODE::VSEL_TENSOR_SCALAR_MODE,uint64_t(128),static_cast<uint8_t>((count+127)/128),emit);
            PipeBarrier<PIPE_V>();
            if(bankFill==0 && pending_[outSlot]){WaitFlag<HardEvent::MTE3_V>(static_cast<event_t>(6+outSlot));pending_[outSlot]=false;}
            auto output=s9g::Local<uint8_t,TPosition::VECOUT>(l_.output+(outSlot*s9g::COMPACT_OUTPUT_BATCH+bankFill)*l_.n,l_.n);
            if(wholeRepeat){
                const UnaryRepeatParams contiguousCast{1,1,4,8};
                Cast(output,halfWork,RoundMode::CAST_NONE,uint64_t(128),static_cast<uint8_t>(vectorCount/128),contiguousCast);
            }else Cast(output,halfWork,RoundMode::CAST_NONE,count);
            PipeBarrier<PIPE_V>();

            if(bankFill==0)outStart=offset+done;
            ++bankFill;
            if(bankFill==s9g::COMPACT_OUTPUT_BATCH || (!next && (offset+total==end || count!=l_.logical))){
                auto bank=s9g::Local<uint8_t,TPosition::VECOUT>(l_.output+outSlot*s9g::COMPACT_OUTPUT_BATCH*l_.n,
                                                       s9g::COMPACT_OUTPUT_BATCH*l_.n);
                const uint32_t full=count==l_.logical ? bankFill:bankFill-1;
                SetFlag<HardEvent::V_MTE3>(static_cast<event_t>(6+outSlot));
                WaitFlag<HardEvent::V_MTE3>(static_cast<event_t>(6+outSlot));
                if(full){

                    const DataCopyExtParams p{static_cast<uint16_t>(full),l_.logical,
                        static_cast<uint32_t>((l_.n-s9g::Up(l_.logical,32))/32),0,0};
                    DataCopyPad(y_[outStart],bank,p);
                }
                if(count!=l_.logical)s9g::WriteGlobal(y_,bank[(bankFill-1)*l_.n],outStart+uint64_t(full)*l_.logical,count);
                SetFlag<HardEvent::MTE3_V>(static_cast<event_t>(6+outSlot));pending_[outSlot]=true;
                bankFill=0;outSlot=(outSlot+1)%s9g::COMPACT_OUTPUT_BANKS;
            }
            if(!next)break;
            done=nextDone;count=nextCount;slot=(slot+1)%s9g::COMPACT_INPUTS;
        }
        heldFill_=bankFill;heldStart_=outStart;
        nextSlot_=loadSlot;nextOutput_=outSlot;
        offset+=total;cursor.Move(static_cast<uint32_t>((col+total)/inner));
        return true;
    }
private:
    S9G_INLINE bool SharedLeft() const {
        if constexpr(SharedSide!=0)return SharedSide==1;
        return side_==1;
    }
    S9G_INLINE void PrepareConstants(){
        Duplicate(s9g::Local<half>(l_.ones,128),static_cast<half>(1),128);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline LocalTensor<bfloat16_t> Raw(uint32_t slot){
        return s9g::Local<bfloat16_t,TPosition::VECIN>(l_.raw+slot*2*l_.n,l_.n);
    }
    S9G_INLINE uint32_t StreamCount(uint64_t remaining) { return s9g::BoundedCount(0,remaining,l_.logical); }
    __aicore__ inline void Load(uint64_t base,uint64_t done,uint32_t count,uint32_t slot){
        const DataCopyExtParams p{1,static_cast<uint32_t>(count*sizeof(bfloat16_t)),0,0,0};
        const DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
        DataCopyPad(Raw(slot),(SharedLeft() ? b_:a_)[base+done],p,pad);
        SetFlag<HardEvent::MTE2_V>(static_cast<event_t>(5+slot));
    }

    __aicore__ inline bool BeginResidentUpdate(uint64_t base){
        if(valid_ && base_==base)return false;
        if(rowRead_){WaitFlag<HardEvent::V_MTE2>(EVENT_ID4);rowRead_=false;}
        s9g::ReadGlobal(s9g::Local<bfloat16_t>(l_.resident+4*l_.row,l_.row),SharedLeft() ? a_:b_,base,
                  static_cast<uint32_t>(inner_));
        SetFlag<HardEvent::MTE2_V>(EVENT_ID4);return true;
    }
    __aicore__ inline void CompleteResidentUpdate(uint64_t base){
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
        Cast(s9g::Local<float>(l_.resident,l_.row),
             s9g::Local<bfloat16_t>(l_.resident+4*l_.row,l_.row),RoundMode::CAST_NONE,inner_);
        PipeBarrier<PIPE_V>();
        const GatherMaskParams repeatRow(1,static_cast<uint16_t>(panelRows_),0,0);
        uint64_t retained=0;
        GatherMask<int32_t>(s9g::Local<int32_t>(l_.expanded,l_.n),s9g::Local<int32_t>(l_.resident,l_.row),
                           7,true,static_cast<uint32_t>(inner_),repeatRow,retained);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID4);rowRead_=true;
        base_=base;valid_=true;
    }
    uint32_t inner_=0,panelRows_=0;
    s9g::CompactRowLayout l_{};
    bool used_[s9g::COMPACT_INPUTS]={},pending_[s9g::COMPACT_OUTPUT_BANKS]={},active_=false;
    uint32_t nextSlot_=0,nextOutput_=0,heldFill_=0;
    uint64_t heldStart_=0;
    uint32_t side_=0;uint64_t base_=0;bool valid_=false,rowRead_=false;
    GlobalTensor<bfloat16_t> a_,b_;GlobalTensor<uint8_t> y_;
};

template<uint32_t SharedSide>
S9G_INLINE void RunGreaterBf16Geometry(GM_ADDR x1,GM_ADDR x2,GM_ADDR output,const GreaterLinearRows& plan){
    const auto span=s9g::BalancedRowInterval(plan.total,plan.inner,GetBlockIdx(),GetBlockNum());
    if(span.begin>=span.end)return;
    GreaterCompactBf16Rows<SharedSide> path;
    if(!path.Init(x1,x2,output,plan))return;
    GreaterGeometryCursor<SharedSide> cursor;
    uint64_t position=span.begin;cursor.Reset(plan,position/plan.inner);
    while(position<span.end)path.Run(cursor,position,span.end);
    path.Drain();
}

// 按广播步长推进输入地址，并批量处理可以连续访问的输出行。
template<class Element>
class GreaterBroadcastPath {
    using CompareType=typename GreaterTypeInfo<Element>::Compare;
    static constexpr uint32_t TYPE_ID=GreaterTypeInfo<Element>::index;

public:
    S9G_INLINE void Run(GM_ADDR x1,GM_ADDR x2,GM_ADDR output,
                        const GreaterRows& plan) {
        if constexpr (TYPE_ID==3) {
            if (plan.reserved&s9g::ROW_COMPACT_RESIDENT) {
                const bool linear=plan.reserved&s9g::ROW_LINEAR_BROADCAST;
                const bool accepted=linear ?
                    RunGreaterCompactRows<GreaterLinearRowCursor>(
                        x1,x2,output,plan):
                    RunGreaterCompactRows<GreaterBroadcastCursor>(
                        x1,x2,output,plan);
                if (accepted) return;
            }
        }
        RunGeneric(x1,x2,output,plan);
    }

private:
    S9G_INLINE void RunGeneric(GM_ADDR x1,GM_ADDR x2,GM_ADDR output,
                               const GreaterRows& plan) {
        plan_=&plan;
        left_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(x1));
        right_.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(x2));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
        if (plan.policy==1)
            left_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        if (plan.policy==2)
            right_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        compare_.Configure(plan.tile,1);

        GreaterRowSlabs<Element> slabs;
        bool useSlabs=false;
        if constexpr (TYPE_ID==0 || TYPE_ID==1)
            useSlabs=slabs.Init(x1,x2,output,plan);

        uint64_t position=uint64_t(GetBlockIdx())*plan.perCore;
        const uint64_t ownedEnd=position+plan.perCore;
        const uint64_t end=ownedEnd<plan.total ? ownedEnd:plan.total;
        if (position>=end) return;

        uint64_t column=position%plan.inner;
        GreaterBroadcastCursor cursor;
        cursor.Reset(plan,position/plan.inner);
        while (position<end) {
            if constexpr (TYPE_ID==0 || TYPE_ID==1) {
                if (useSlabs && slabs.Run(cursor,position,end)) {
                    column=0;
                    continue;
                }
            }

            slabs.Invalidate();
            uint32_t rows=1;
            uint32_t pitch=0;
            uint32_t count=0;
            const bool batched=column==0 && plan.inner<=plan.tile &&
                               end-position>=plan.inner;
            if (batched) {
                uint64_t availableRows=cursor.RowsBeforeWrap();
                if (availableRows>plan.batchRows) availableRows=plan.batchRows;
                const uint64_t ownedRows=(end-position)/plan.inner;
                if (availableRows>ownedRows) availableRows=ownedRows;
                rows=static_cast<uint32_t>(availableRows);
                pitch=plan.pitch;
                count=rows*pitch;
            } else {
                uint64_t available=plan.inner-column;
                if (available>end-position) available=end-position;
                if (available>plan.tile) available=plan.tile;
                count=static_cast<uint32_t>(available);
                pitch=count;
            }

            LoadOperand(1,cursor.left,column,rows,pitch,batched);
            LoadOperand(2,cursor.right,column,rows,pitch,batched);
            compare_.PrepareConstants();
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            auto left=PrepareOperand(1,rows,pitch,count,batched);
            auto right=PrepareOperand(2,rows,pitch,count,batched);
            compare_.ComparePrepared(left,right,count);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            if (batched) {
                const DataCopyExtParams copy{
                    static_cast<uint16_t>(rows),
                    static_cast<uint32_t>(plan.inner),
                    static_cast<uint32_t>(
                        (pitch-s9g::Up(plan.inner,32))/32),
                    0,0
                };
                DataCopyPad(output_[position],compare_.ByteOutput(),copy);
                position+=uint64_t(rows)*plan.inner;
                cursor.Move(rows);
                column=0;
            } else {
                s9g::WriteGlobal(output_,compare_.ByteOutput(),position,count);
                position+=count;
                column+=count;
                if (column==plan.inner) {
                    column=0;
                    cursor.Move(1);
                }
            }
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    S9G_INLINE bool OperandIsScalar(uint32_t side) const {
        return plan_->innerBits&(1U<<(side-1));
    }

    S9G_INLINE void LoadOperand(uint32_t side,uint64_t base,uint64_t column,
                                uint32_t rows,uint32_t pitch,bool batched) {
        const auto& source=side==1 ? left_:right_;
        auto input=compare_.RawInput(side);
        const bool scalar=OperandIsScalar(side);
        if (!batched) {
            s9g::ReadGlobal(input,source,base+(scalar ? 0:column),
                            scalar ? 1:pitch);
            return;
        }

        const uint64_t sourcePitch=plan_->rank>1 ?
            (side==1 ? plan_->stride1[plan_->rank-2]:
                       plan_->stride2[plan_->rank-2]):0;
        const uint32_t columns=scalar ? 1:static_cast<uint32_t>(plan_->inner);
        const uint32_t localPitch=scalar ? 32/sizeof(Element):pitch;
        if (sourcePitch>=columns &&
            (sourcePitch-columns)*sizeof(Element)<=UINT32_MAX) {
            const DataCopyExtParams copy{
                static_cast<uint16_t>(rows),
                static_cast<uint32_t>(columns*sizeof(Element)),
                static_cast<uint32_t>((sourcePitch-columns)*sizeof(Element)),
                static_cast<uint32_t>(
                    (localPitch*sizeof(Element)-
                     s9g::Up(columns*sizeof(Element),32))/32),
                0
            };
            const DataCopyPadExtParams<Element> padding{
                false,0,0,static_cast<Element>(0)
            };
            DataCopyPad(input,source[base],copy,padding);
            return;
        }
        for (uint32_t row=0;row<rows;++row)
            s9g::ReadGlobal(input[row*localPitch],source,
                            base+uint64_t(row)*sourcePitch,columns);
    }

    S9G_INLINE LocalTensor<CompareType> PrepareOperand(
            uint32_t side,uint32_t rows,uint32_t pitch,uint32_t count,
            bool batched) {
        const bool scalar=OperandIsScalar(side);
        const uint32_t converted=scalar ?
            (batched ? rows*(32/sizeof(Element)):1):count;
        auto data=compare_.ConvertInput(side,converted);
        PipeBarrier<PIPE_V>();
        if (scalar) compare_.ExpandScalarRows(data,rows,pitch);
        return data;
    }

    const GreaterRows* plan_=nullptr;
    GreaterCompareEngine<Element> compare_;
    GlobalTensor<Element> left_;
    GlobalTensor<Element> right_;
    GlobalTensor<uint8_t> output_;
};

#endif
