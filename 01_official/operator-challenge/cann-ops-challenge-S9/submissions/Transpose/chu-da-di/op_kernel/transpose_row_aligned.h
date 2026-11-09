#ifndef S9_TRANSPOSE_ROW_ALIGNED_H
#define S9_TRANSPOSE_ROW_ALIGNED_H
namespace s9transpose {
template<typename T, uint32_t InputBuffers=0> class KernelTransposeRowAligned {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,const TransposeColumnOwnerTilingData& t,GM_ADDR map){
        d_=t;x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if(t.cachePolicy&1U)x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if(t.cachePolicy&2U)y_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(in_,InputBuffers?InputBuffers:t.bufferCount,t.rows*(t.cols+8)*4);pipe_.InitBuffer(out_,t.bufferCount,t.cols*(t.rows+8)*4);
        maskWords_=((t.rows*(t.cols+8)+31)/32+7)/8*8;const uint32_t totalWords=(t.base.matrixH+t.rows-1)/t.rows*maskWords_;pipe_.InitBuffer(maskBuf_,totalWords*4);
        map_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(map));maskAll_=maskBuf_.template Get<uint32_t>();
        AscendC::DataCopyExtParams mp{1,totalWords*4,0,0,0};AscendC::DataCopyPadExtParams<uint32_t> pp{false,0,0,0};AscendC::DataCopyPad(maskAll_,map_,mp,pp);
        const event_t e=static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(e);AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(e);
    }
    __aicore__ inline void Process(){
        const uint64_t h=d_.base.matrixH,w=d_.base.matrixW,units=(w+7)/8,cores=AscendC::GetBlockNum(),bid=AscendC::GetBlockIdx();
        const uint64_t small=units/cores,extra=units%cores,first=(bid*small+(bid<extra?bid:extra))*8;
        const uint64_t end0=first+(small+(bid<extra?1:0))*8,last=end0<w?end0:w;
        const uint32_t rawPitch=d_.cols+8,inputPitch=d_.cols,outputPitch=d_.rows+8;
        for(uint64_t col=first;col<last;col+=d_.cols){
            const uint32_t nc=last-col<d_.cols?last-col:d_.cols;
            for(uint64_t row=0;row<h;row+=d_.rows){
                const uint32_t nr=h-row<d_.rows?h-row:d_.rows;
                auto mask=maskAll_[row/d_.rows*maskWords_];
                auto input=in_.template AllocTensor<T>();
                for(uint32_t done=0;done<nr;){
                    const uint64_t rr=row+done,cell=rr/d_.rowInner,srcRow=((cell%d_.rowA)*d_.rowB+cell/d_.rowA)*d_.rowInner+rr%d_.rowInner;
                    const uint32_t run=d_.rowInner-rr%d_.rowInner<nr-done?d_.rowInner-rr%d_.rowInner:nr-done;
                    for(uint32_t p=0;p<8&&p<run;++p){
                        const uint64_t logical=(srcRow+p)*w+col,start=logical/8*8;
                        const uint32_t width=(nc+logical%8+7)/8*8;uint32_t n=(run-p+7)/8;
                        const uint64_t tailStart=start+(n-1)*8*w,remaining=d_.base.totalElements-tailStart;
                        const bool clipped=remaining<width;if(clipped)--n;
                        if(n){AscendC::DataCopyParams cp{static_cast<uint16_t>(n),static_cast<uint16_t>(width/8),static_cast<uint16_t>(w-width/8),static_cast<uint16_t>(rawPitch-width/8)};AscendC::DataCopy(input[(done+p)*rawPitch],x_[start],cp);}
                        if(clipped){AscendC::DataCopyExtParams cp{1,static_cast<uint32_t>(remaining*4),0,0,0};AscendC::DataCopyPadExtParams<T> pp{false,0,0,static_cast<T>(0)};AscendC::DataCopyPad(input[(done+p+n*8)*rawPitch],x_[tailStart],cp,pp);}
                    }
                    done+=run;
                }
                in_.EnQue(input);input=in_.template DeQue<T>();uint64_t selected=0;
                AscendC::GatherMask(input.template ReinterpretCast<uint32_t>(),input.template ReinterpretCast<uint32_t>(),mask,true,nr*rawPitch,{1,1,8,0},selected);AscendC::PipeBarrier<PIPE_V>();
                auto output=out_.template AllocTensor<T>();uint64_t src[16],dst[16];const uint32_t reps=(nc+7)/8;
                for(uint32_t rt=0;rt<(nr+15)/16;++rt){
                    for(uint32_t i=0;i<16;++i)src[i]=reinterpret_cast<uint64_t>(input[(rt*16+i)*inputPitch].GetPhyAddr());
                    for(uint32_t c=0;c<8;++c)for(uint32_t g=0;g<2;++g)dst[c*2+g]=reinterpret_cast<uint64_t>(output[c*outputPitch+rt*16+g*8].GetPhyAddr());
                    AscendC::TransDataTo5HDParams tp{false,false,static_cast<uint8_t>(reps),static_cast<uint16_t>(reps==1?0:outputPitch),static_cast<uint16_t>(reps==1?0:1)};AscendC::TransDataTo5HD<T>(dst,src,tp);
                }
                out_.EnQue(output);in_.FreeTensor(input);output=out_.template DeQue<T>();
                const uint32_t alignedRows=(nr+7)/8*8;AscendC::DataCopyExtParams cp{static_cast<uint16_t>(nc),nr*4,(outputPitch-alignedRows)/8,static_cast<uint32_t>((h-nr)*4),0};AscendC::DataCopyPad(y_[col*h+row],output,cp);out_.FreeTensor(output);
            }
        }
    }
private:
    TransposeColumnOwnerTilingData d_{};AscendC::TPipe pipe_;AscendC::TQue<AscendC::TPosition::VECIN,1>in_;AscendC::TQue<AscendC::TPosition::VECOUT,1>out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC>maskBuf_;AscendC::LocalTensor<uint32_t>maskAll_;
    AscendC::GlobalTensor<T>x_,y_;AscendC::GlobalTensor<uint32_t>map_;uint32_t maskWords_=0;
};
}
#endif
