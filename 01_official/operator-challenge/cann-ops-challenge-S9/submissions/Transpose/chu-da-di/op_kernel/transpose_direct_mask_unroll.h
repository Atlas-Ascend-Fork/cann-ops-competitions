#ifndef S9_TRANSPOSE_DIRECT_MASK_UNROLL_H
#define S9_TRANSPOSE_DIRECT_MASK_UNROLL_H
namespace s9transpose {
template<typename T, bool Exact=false> class KernelTransposeDirectMaskUnroll {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,const TransposeDirectMaskHeader& t,GM_ADDR map){
        data_=t;x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if((Exact ? 2U : t.cachePolicy)&1U)x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if((Exact ? 2U : t.cachePolicy)&2U)y_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(in_,(Exact ? 1U : t.inputBuffers),(Exact ? 256U : t.inputRows)*(Exact ? 48U : t.rawPitch)*sizeof(T));
        pipe_.InitBuffer(out_,(Exact ? 1U : t.outputBuffers),((Exact ? 32U : t.columns)+1)*(Exact ? 784U : t.outputPitch)*sizeof(T));
        pipe_.InitBuffer(maskQ_,1,(Exact ? 816U : t.maskWords)*4);if((Exact ? 1U : t.reserved0))pipe_.InitBuffer(inputMaskQ_,1,(Exact ? 384U : t.inputMaskWords)*4);pipe_.InitBuffer(linear_,128);pipe_.InitBuffer(carry_,64);
        linearIdx_=linear_.template Get<uint32_t>();AscendC::GlobalTensor<uint32_t> g;g.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(map));
        map_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(map)+32);
        inputMap_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(map)+32+(32/sizeof(T))*(Exact ? 816U : t.maskWords));AscendC::DataCopy(linearIdx_,g,32);
        const event_t e=static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(e);AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(e);
    }
    __aicore__ inline void LoadMask(uint32_t phi){
        if(loaded_&&lastPhi_==phi)return;if(loaded_)maskQ_.FreeTensor(mask_);
        mask_=maskQ_.template AllocTensor<uint32_t>();AscendC::DataCopy(mask_,map_[phi*(Exact ? 816U : data_.maskWords)],(Exact ? 816U : data_.maskWords));
        maskQ_.EnQue(mask_);mask_=maskQ_.template DeQue<uint32_t>();loaded_=true;lastPhi_=phi;
    }
    __aicore__ inline void LoadInputMask(uint32_t phi){
        if(inputLoaded_&&lastInputPhi_==phi)return;if(inputLoaded_)inputMaskQ_.FreeTensor(inputMask_);
        inputMask_=inputMaskQ_.template AllocTensor<uint32_t>();AscendC::DataCopy(inputMask_,inputMap_[phi*(Exact ? 384U : data_.inputMaskWords)],(Exact ? 384U : data_.inputMaskWords));
        inputMaskQ_.EnQue(inputMask_);inputMask_=inputMaskQ_.template DeQue<uint32_t>();inputLoaded_=true;lastInputPhi_=phi;
    }
    __aicore__ inline void LoadAligned(AscendC::LocalTensor<T>& input,uint64_t matrix,uint64_t col,uint64_t row,uint32_t rows,uint32_t nc){
        constexpr uint32_t al=32/sizeof(T);
        const uint64_t h=rows,w=(Exact ? 521U : data_.base.matrixW),base=matrix*(Exact ? 762U : data_.base.matrixH)*w+row*w;
        const uint32_t pitch=(Exact ? 48U : data_.rawPitch);
        const uint32_t phaseShift=((Exact ? 0U : data_.reserved4)&1U)?AscendC::GetBlockIdx()%al:0;
        #pragma unroll
        for(uint32_t pi=0;pi<al;++pi){
            const uint32_t p=(pi+phaseShift)%al;if(p>=h)continue;
            const uint32_t leading=static_cast<uint32_t>((base+p*w+col)%al);
            const uint32_t width=(Exact ? 2U : data_.reserved5)==0?pitch:(nc+((Exact ? 2U : data_.reserved5)==1?al-1:leading)+al-1)/al*al;
            uint32_t n=static_cast<uint32_t>((h-p+al-1)/al);
            const uint64_t start=(base+p*w+col)/al*al;
            const uint64_t tailRow=p+(n-1)*al,tailStart=(base+tailRow*w+col)/al*al;
            const uint64_t remaining=(Exact ? 49625250U : data_.base.totalElements)-tailStart;
            const bool clipped=remaining<width;
            if(clipped)--n;
            if(n){AscendC::DataCopyParams ip{static_cast<uint16_t>(n),static_cast<uint16_t>(width/al),static_cast<uint16_t>(w-width/al),static_cast<uint16_t>((al*pitch-width)/al)};AscendC::DataCopy(input[p*pitch],x_[start],ip);}
            if(clipped){AscendC::DataCopyExtParams ip{1,static_cast<uint32_t>(remaining*sizeof(T)),0,0,0};AscendC::DataCopyPadExtParams<T> pp{false,0,0,static_cast<T>(0)};AscendC::DataCopyPad(input[tailRow*pitch],x_[tailStart],ip,pp);}
        }
    }
    __aicore__ inline void GatherLinear(const AscendC::LocalTensor<T>& dst,const AscendC::LocalTensor<T>& src,uint32_t offset,uint32_t count){
        if constexpr(sizeof(T)==2)AscendC::Gather(dst.template ReinterpretCast<half>(),src.template ReinterpretCast<half>(),linearIdx_,offset,count);
        else if constexpr(sizeof(T)==4)AscendC::Gather(dst.template ReinterpretCast<float>(),src.template ReinterpretCast<float>(),linearIdx_,offset,count);
    }
    __aicore__ inline void Process(){
        constexpr uint32_t al=32/sizeof(T),groups=16/al;
        const uint64_t h=(Exact ? 762U : data_.base.matrixH),w=(Exact ? 521U : data_.base.matrixW);
        const uint64_t units=((Exact ? 65125U : data_.totalColumns)+al-1)/al,cores=AscendC::GetBlockNum(),bid=AscendC::GetBlockIdx();
        const uint64_t rowBlocks=(h+(Exact ? 256U : data_.inputRows)-1)/(Exact ? 256U : data_.inputRows);
        const uint64_t firstRowBlock=((Exact ? 0U : data_.reserved4)&2U)?bid%rowBlocks:0;
        const uint64_t small=units/cores,extra=units%cores;
        const uint64_t packets=((Exact ? 65125U : data_.totalColumns)+(Exact ? 32U : data_.columns)-1)/(Exact ? 32U : data_.columns);
        const uint64_t requestedRanges=cores*(Exact ? 12U : data_.reserved1);
        const uint64_t balancedRanges=(Exact ? 12U : data_.reserved1)?(requestedRanges<packets?requestedRanges:packets):0;
        const uint64_t ranges=balancedRanges?balancedRanges:((Exact ? 0U : data_.reserved2)?(units+(Exact ? 0U : data_.reserved2)-1)/(Exact ? 0U : data_.reserved2):cores);
        const uint64_t packetSmall=balancedRanges?packets/ranges:0,packetExtra=balancedRanges?packets%ranges:0;
        for(uint64_t range=bid;range<ranges;range+=cores){
        const uint64_t first=balancedRanges?(range*packetSmall+(range<packetExtra?range:packetExtra))*(Exact ? 32U : data_.columns):((Exact ? 0U : data_.reserved2)?range*(Exact ? 0U : data_.reserved2)*al:(bid*small+(bid<extra?bid:extra))*al);
        const uint64_t end0=balancedRanges?first+(packetSmall+(range<packetExtra?1:0))*(Exact ? 32U : data_.columns):((Exact ? 0U : data_.reserved2)?first+(Exact ? 0U : data_.reserved2)*al:first+(small+(bid<extra?1:0))*al),last=end0<(Exact ? 65125U : data_.totalColumns)?end0:(Exact ? 65125U : data_.totalColumns);
        auto carry=carry_.template Get<T>();uint32_t carryCount=0;const uint32_t tailStart=static_cast<uint32_t>((h-al)/al*al);
        for(uint64_t flat=first;flat<last;){
            const uint64_t matrix=flat/w,col=flat%w;uint64_t span=w-col<last-flat?w-col:last-flat;if(span>(Exact ? 32U : data_.columns))span=(Exact ? 32U : data_.columns);
            const uint32_t nc=static_cast<uint32_t>(span),phi=static_cast<uint32_t>(flat*h%al);if(!(Exact ? 0U : data_.reserved3))LoadMask(phi);
            if((Exact ? 1U : data_.reserved0))LoadInputMask(static_cast<uint32_t>((matrix*h*w+col)%al));
            AscendC::LocalTensor<T> output;
            uint64_t rowBlock=firstRowBlock;
            #pragma unroll
            for(uint64_t ri=0;ri<rowBlocks;++ri){
                const uint64_t row=rowBlock*(Exact ? 256U : data_.inputRows);
                const uint32_t nr=static_cast<uint32_t>(h-row<(Exact ? 256U : data_.inputRows)?h-row:(Exact ? 256U : data_.inputRows));
                auto input=in_.template AllocTensor<T>();
                if((Exact ? 1U : data_.reserved0)){
                    LoadAligned(input,matrix,col,row,nr,nc);in_.EnQue(input);input=in_.template DeQue<T>();uint64_t inputSelected=0;
                    if constexpr(sizeof(T)==4)AscendC::GatherMask(input.template ReinterpretCast<uint32_t>(),input.template ReinterpretCast<uint32_t>(),inputMask_,true,nr*(Exact ? 48U : data_.rawPitch),{1,1,8,0},inputSelected);
                    AscendC::PipeBarrier<PIPE_V>();
                }else{
                    const uint32_t alignedCols=(nc+al-1)/al*al;
                    AscendC::DataCopyExtParams ip{static_cast<uint16_t>(nr),static_cast<uint32_t>(nc*sizeof(T)),static_cast<uint32_t>((w-nc)*sizeof(T)),((Exact ? 40U : data_.inputPitch)-alignedCols)/al,0};AscendC::DataCopyPadExtParams<T> pp{false,0,0,static_cast<T>(0)};
                    AscendC::DataCopyPad(input,x_[matrix*h*w+row*w+col],ip,pp);in_.EnQue(input);input=in_.template DeQue<T>();
                }
                if(ri==0){output=out_.template AllocTensor<T>();if(phi&&!(Exact ? 0U : data_.reserved3))GatherLinear(output[tailStart],carry,0,2*al);}
                if constexpr(sizeof(T)>=2){
                    uint64_t src[16],dst[16];const uint32_t reps=(nr+15)/16;
                    for(uint32_t ct=0;ct<(nc+al-1)/al;++ct){
                        for(uint32_t i=0;i<16;++i)src[i]=reinterpret_cast<uint64_t>(input[i*(Exact ? 40U : data_.inputPitch)+ct*al].GetPhyAddr());
                        for(uint32_t c=0;c<al;++c)for(uint32_t g=0;g<groups;++g)dst[c*groups+g]=reinterpret_cast<uint64_t>(output[(ct*al+c+1)*(Exact ? 784U : data_.outputPitch)+row+g*al].GetPhyAddr());
                        AscendC::TransDataTo5HDParams tp{false,false,static_cast<uint8_t>(reps),static_cast<uint16_t>(reps==1?0:16/al),static_cast<uint16_t>(reps==1?0:16*(Exact ? 40U : data_.inputPitch)/al)};AscendC::TransDataTo5HD<T>(dst,src,tp);
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();in_.FreeTensor(input);
                if(++rowBlock==rowBlocks)rowBlock=0;
            }
            if((Exact ? 0U : data_.reserved3)){
                out_.EnQue(output);output=out_.template DeQue<T>();
                const uint32_t alignedH=static_cast<uint32_t>((h+al-1)/al*al);
                AscendC::DataCopyExtParams cp{static_cast<uint16_t>(nc),static_cast<uint32_t>(h*sizeof(T)),static_cast<uint32_t>(((Exact ? 784U : data_.outputPitch)-alignedH)/al),0,0};
                AscendC::DataCopyPad(y_[flat*h],output[(Exact ? 784U : data_.outputPitch)],cp);out_.FreeTensor(output);flat+=nc;continue;
            }
            AscendC::PipeBarrier<PIPE_V>();carryCount=static_cast<uint32_t>((phi+nc*h)%al);
            if(carryCount)GatherLinear(carry,output,static_cast<uint32_t>((nc*(Exact ? 784U : data_.outputPitch)+tailStart)*sizeof(T)),2*al);
            AscendC::PipeBarrier<PIPE_V>();uint64_t selected=0;
            if constexpr(sizeof(T)==2)AscendC::GatherMask(output.template ReinterpretCast<uint16_t>(),output.template ReinterpretCast<uint16_t>(),mask_.template ReinterpretCast<uint16_t>(),true,(nc+1)*(Exact ? 784U : data_.outputPitch),{1,1,8,0},selected);
            else if constexpr(sizeof(T)==4)AscendC::GatherMask(output.template ReinterpretCast<uint32_t>(),output.template ReinterpretCast<uint32_t>(),mask_,true,(nc+1)*(Exact ? 784U : data_.outputPitch),{1,1,8,0},selected);
            out_.EnQue(output);output=out_.template DeQue<T>();
            AscendC::DataCopy(y_[flat*h-phi],output,static_cast<uint32_t>((phi+nc*h)/al*al));out_.FreeTensor(output);flat+=nc;
        }
        if(carryCount){auto output=out_.template AllocTensor<T>();GatherLinear(output,carry,static_cast<uint32_t>((h-carryCount-tailStart)*sizeof(T)),carryCount);
            out_.EnQue(output);output=out_.template DeQue<T>();AscendC::DataCopyExtParams op{1,static_cast<uint32_t>(carryCount*sizeof(T)),0,0,0};
            AscendC::DataCopyPad(y_[last*h-carryCount],output,op);out_.FreeTensor(output);}
        }
        if(loaded_)maskQ_.FreeTensor(mask_);if(inputLoaded_)inputMaskQ_.FreeTensor(inputMask_);
    }
private:
    TransposeDirectMaskHeader data_{};AscendC::TPipe pipe_;AscendC::TQue<AscendC::TPosition::VECIN,1> in_,maskQ_,inputMaskQ_;AscendC::TQue<AscendC::TPosition::VECOUT,1> out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> linear_,carry_;AscendC::LocalTensor<uint32_t> linearIdx_,mask_,inputMask_;AscendC::GlobalTensor<uint32_t> map_,inputMap_;AscendC::GlobalTensor<T>x_,y_;bool loaded_=false,inputLoaded_=false;uint32_t lastPhi_=0,lastInputPhi_=0;
};
}
#endif
