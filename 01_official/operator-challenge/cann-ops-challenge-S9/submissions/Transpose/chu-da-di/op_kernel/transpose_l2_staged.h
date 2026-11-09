#ifndef S9_TRANSPOSE_L2_STAGED_H
#define S9_TRANSPOSE_L2_STAGED_H
namespace s9transpose {
template<typename T> class KernelTransposeL2Staged {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const TransposeL2StagedHeader& t, GM_ADDR map) {
        data_ = t; x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x)); y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        if (t.cachePolicy & 1U) x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        if (t.cachePolicy & 2U) y_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        tempGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + AscendC::GetBlockIdx() * t.slotElements);
        tempGm_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
        pipe_.InitBuffer(raw_, t.rawBuffers, t.rawElements * sizeof(T));
        pipe_.InitBuffer(out_, t.outBuffers, t.outputElements * sizeof(T));
        if (t.reserved0 != 1) pipe_.InitBuffer(pad_, (t.reserved0 == 2 ? 1 : t.rows) * t.inputPitch * sizeof(T));
        pipe_.InitBuffer(linear_, (t.linearCount * 4 + 31) / 32 * 32);
        pipe_.InitBuffer(packQ_, 1, (t.packCount * 4 + 31) / 32 * 32);
        pipe_.InitBuffer(carry_, 64);
        linearIdx_ = linear_.template Get<uint32_t>();
        AscendC::GlobalTensor<uint32_t> lg; lg.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(map));
        packMap_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(map) + t.linearCount);
        AscendC::DataCopyExtParams ip{1, t.linearCount * 4, 0, 0, 0}; AscendC::DataCopyPadExtParams<uint32_t> pad{false,0,0,0};
        AscendC::DataCopyPad(linearIdx_, lg, ip, pad);
        const event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ready); AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ready);

    }
    __aicore__ inline void GatherValues(const AscendC::LocalTensor<T>& dst, const AscendC::LocalTensor<T>& src,
        const AscendC::LocalTensor<uint32_t>& idx, uint32_t base, uint32_t count) {
        for (uint32_t off=0;off<count;off+=8192) {
            const uint32_t n=count-off<8192?count-off:8192;
            if constexpr(sizeof(T)==2) AscendC::Gather(dst.template ReinterpretCast<half>()[off],src.template ReinterpretCast<half>(),idx[off],base,n);
            else if constexpr(sizeof(T)==4) AscendC::Gather(dst.template ReinterpretCast<float>()[off],src.template ReinterpretCast<float>(),idx[off],base,n);
        }
    }
    __aicore__ inline void LoadPack(uint32_t phi) {
        if (packLoaded_ && phi == lastPhi_) return;
        if (packLoaded_) packQ_.FreeTensor(packIdx_);
        packIdx_ = packQ_.template AllocTensor<uint32_t>();
        AscendC::DataCopy(packIdx_,packMap_[phi*data_.packCount],data_.packCount);
        packQ_.EnQue(packIdx_);packIdx_=packQ_.template DeQue<uint32_t>();
        packLoaded_=true;lastPhi_=phi;
    }
    __aicore__ inline void FillScratchBlocked(uint64_t base) {
        constexpr uint32_t alignment = 32 / sizeof(T);
        const uint64_t h = data_.base.matrixH, w = data_.base.matrixW;
        const uint32_t inputPitch = data_.inputPitch, outputPitch = data_.rows + alignment;
        for (uint64_t column = 0; column < w; column += data_.columns) {
            const uint32_t nc = static_cast<uint32_t>(w-column < data_.columns ? w-column : data_.columns);
            const uint32_t alignedCols = (nc + alignment - 1) / alignment * alignment;
            for (uint64_t row = 0; row < h; row += data_.rows) {
                const uint32_t nr = static_cast<uint32_t>(h-row < data_.rows ? h-row : data_.rows);
                auto input = raw_.template AllocTensor<T>();
                AscendC::DataCopyExtParams ip{static_cast<uint16_t>(nr),static_cast<uint32_t>(nc*sizeof(T)),
                    static_cast<uint32_t>((w-nc)*sizeof(T)),(inputPitch-alignedCols)/alignment,0};
                AscendC::DataCopyPadExtParams<T> pp{false,0,0,static_cast<T>(0)};
                AscendC::DataCopyPad(input,x_[base+row*w+column],ip,pp);
                raw_.EnQue(input);input=raw_.template DeQue<T>();auto output=out_.template AllocTensor<T>();
            uint64_t src[16], dst[16];
            const uint32_t repeats = (nc + alignment - 1) / alignment;
            const uint32_t rtiles = (nr + 15) / 16;
            for (uint32_t rt = 0; rt < rtiles; ++rt) {
                for (uint32_t i = 0; i < 16; ++i)
                    src[i] = reinterpret_cast<uint64_t>(input[(rt * 16 + i) * inputPitch].GetPhyAddr());
                if constexpr (sizeof(T) == 1) {
                    // Native INT8 low/high source halves; paired row tiles fill
                    // both halves of each destination datablock without casts.
                    for (uint32_t halfCol = 0; halfCol < 2; ++halfCol) {
                        for (uint32_t i = 0; i < 16; ++i)
                            dst[i] = reinterpret_cast<uint64_t>(output[(halfCol * 16 + i) * outputPitch + (rt / 2) * 32].GetPhyAddr());
                        AscendC::TransDataTo5HDParams tp{rt % 2 != 0, halfCol != 0,
                            static_cast<uint8_t>(repeats), static_cast<uint16_t>(repeats == 1 ? 0 : outputPitch),
                            static_cast<uint16_t>(repeats == 1 ? 0 : 1)};
                        AscendC::TransDataTo5HD<T>(dst, src, tp);
                    }
                } else {
                    constexpr uint32_t groups = 16 / alignment;
                    for (uint32_t c = 0; c < alignment; ++c)
                        for (uint32_t g = 0; g < groups; ++g)
                            dst[c * groups + g] = reinterpret_cast<uint64_t>(output[c * outputPitch + rt * 16 + g * alignment].GetPhyAddr());
                    AscendC::TransDataTo5HDParams tp{false, false, static_cast<uint8_t>(repeats),
                        static_cast<uint16_t>(repeats == 1 ? 0 : outputPitch),
                        static_cast<uint16_t>(repeats == 1 ? 0 : 1)};
                    AscendC::TransDataTo5HD<T>(dst, src, tp);
                }
            }
                out_.EnQue(output);raw_.FreeTensor(input);output=out_.template DeQue<T>();
                const uint32_t alignedRows = (nr + alignment - 1) / alignment * alignment;
                AscendC::DataCopyExtParams op{static_cast<uint16_t>(nc),static_cast<uint32_t>(nr*sizeof(T)),
                    (outputPitch-alignedRows)/alignment,static_cast<uint32_t>((data_.scratchPitch-nr)*sizeof(T)),0};
                AscendC::DataCopyPad(tempGm_[column*data_.scratchPitch+row],output,op);out_.FreeTensor(output);
            }
        }
    }
    __aicore__ inline void Process() {
        constexpr uint32_t al=32/sizeof(T),groups=16/al;
        const uint64_t h=data_.base.matrixH,w=data_.base.matrixW,matrixElements=h*w;
        const uint64_t units=(data_.base.outer+data_.batchQuantum-1)/data_.batchQuantum;
        const uint64_t blocks=AscendC::GetBlockNum(),bid=AscendC::GetBlockIdx(),small=units/blocks,extra=units%blocks;
        const uint64_t firstUnit=bid*small+(bid<extra?bid:extra),lastUnit=firstUnit+small+(bid<extra?1:0);
        const uint64_t first=firstUnit*data_.batchQuantum,last=lastUnit*data_.batchQuantum<data_.base.outer?lastUnit*data_.batchQuantum:data_.base.outer;
        AscendC::LocalTensor<T> padded; if (data_.reserved0 != 1) padded=pad_.template Get<T>();auto carry=carry_.template Get<T>();uint32_t carryCount=0;
        const uint32_t tailStart=static_cast<uint32_t>((h-al)/al*al);
        for(uint64_t matrix=first;matrix<last;++matrix){
            const uint64_t base=matrix*matrixElements;
            if (data_.reserved0 == 1) FillScratchBlocked(base);
            else for(uint64_t row=0;row<h;row+=data_.rows){
                const uint32_t nr=static_cast<uint32_t>(h-row<data_.rows?h-row:data_.rows);
                auto raw=raw_.template AllocTensor<T>();const uint64_t logical=base+row*w,start=logical/al*al;
                const uint32_t leading=static_cast<uint32_t>(logical-start);
                const uint32_t wanted=static_cast<uint32_t>(leading+nr*w);
                const uint32_t aligned=(wanted+al-1)/al*al;
                if(start+aligned<=data_.base.totalElements) AscendC::DataCopy(raw,x_[start],aligned);
                else {AscendC::DataCopyExtParams ip{1,static_cast<uint32_t>(wanted*sizeof(T)),0,0,0};AscendC::DataCopyPadExtParams<T> pp{false,0,0,static_cast<T>(0)};AscendC::DataCopyPad(raw,x_[start],ip,pp);}
                raw_.EnQue(raw);raw=raw_.template DeQue<T>();
                if (data_.reserved0 == 2) {
                    for (uint32_t rr = nr; rr > 0; --rr) {
                        const uint32_t r = rr - 1;
                        GatherValues(padded,raw,linearIdx_,static_cast<uint32_t>((leading+r*w)*sizeof(T)),data_.inputPitch);
                        AscendC::PipeBarrier<PIPE_V>();
                        GatherValues(raw[r*data_.inputPitch],padded,linearIdx_,0,data_.inputPitch);
                        AscendC::PipeBarrier<PIPE_V>();
                    }
                } else {
                    for(uint32_t r=0;r<nr;++r) GatherValues(padded[r*data_.inputPitch],raw,linearIdx_,static_cast<uint32_t>((leading+r*w)*sizeof(T)),data_.inputPitch);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                auto transInput = data_.reserved0 == 2 ? raw : padded;
                const uint32_t outputPitch = data_.reserved0 == 2 ? data_.rows + al : data_.rows;
                auto output=out_.template AllocTensor<T>();
                if constexpr(sizeof(T)>=2){
                    uint64_t src[16],dst[16];
                    const uint32_t reps=data_.columns/al;
                    for (uint32_t rt=0;rt<(nr+15)/16;++rt) {
                        for(uint32_t i=0;i<16;++i)src[i]=reinterpret_cast<uint64_t>(transInput[(rt*16+i)*data_.inputPitch].GetPhyAddr());
                        for(uint32_t c=0;c<al;++c)for(uint32_t g=0;g<groups;++g)dst[c*groups+g]=reinterpret_cast<uint64_t>(output[c*outputPitch+rt*16+g*al].GetPhyAddr());
                        AscendC::TransDataTo5HDParams tp{false,false,static_cast<uint8_t>(reps),static_cast<uint16_t>(reps==1?0:outputPitch),static_cast<uint16_t>(reps==1?0:1)};
                        AscendC::TransDataTo5HD<T>(dst,src,tp);
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();out_.EnQue(output);raw_.FreeTensor(raw);output=out_.template DeQue<T>();
                if(nr%al==0){AscendC::DataCopyParams op{static_cast<uint16_t>(w),static_cast<uint16_t>(nr/al),static_cast<uint16_t>((outputPitch-nr)/al),static_cast<uint16_t>((data_.scratchPitch-nr)/al)};AscendC::DataCopy(tempGm_[row],output,op);}
                else {AscendC::DataCopyExtParams op{static_cast<uint16_t>(w),static_cast<uint32_t>(nr*sizeof(T)),static_cast<uint32_t>((outputPitch-(nr+al-1)/al*al)/al),static_cast<uint32_t>((data_.scratchPitch-nr)*sizeof(T)),0};AscendC::DataCopyPad(tempGm_[row],output,op);}
                out_.FreeTensor(output);
            }
            const event_t stored=static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(stored);AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(stored);
            const uint32_t phi=static_cast<uint32_t>(base%al);LoadPack(phi);
            for(uint64_t column=0;column<w;column+=data_.outputRows){
                const uint32_t nc=static_cast<uint32_t>(w-column<data_.outputRows?w-column:data_.outputRows);
                auto raw=raw_.template AllocTensor<T>();
                if(column==0) AscendC::DataCopy(raw[data_.scratchPitch],tempGm_,nc*data_.scratchPitch);
                else AscendC::DataCopy(raw,tempGm_[(column-1)*data_.scratchPitch],(nc+1)*data_.scratchPitch);
                raw_.EnQue(raw);raw=raw_.template DeQue<T>();
                if(phi&&column==0){GatherValues(raw[tailStart],carry,linearIdx_,0,2*al);AscendC::PipeBarrier<PIPE_V>();}
                const uint32_t count=static_cast<uint32_t>((nc*h+phi)/al*al);
                auto output=out_.template AllocTensor<T>();uint64_t selected=0;
                if constexpr (sizeof(T)==2) AscendC::GatherMask(output.template ReinterpretCast<uint16_t>(),raw.template ReinterpretCast<uint16_t>(),packIdx_.template ReinterpretCast<uint16_t>(),true,(nc+1)*data_.scratchPitch,{1,1,8,0},selected);
                else if constexpr (sizeof(T)==4) AscendC::GatherMask(output.template ReinterpretCast<uint32_t>(),raw.template ReinterpretCast<uint32_t>(),packIdx_,true,(nc+1)*data_.scratchPitch,{1,1,8,0},selected);
                if(column+nc==w){
                    carryCount=static_cast<uint32_t>((nc*h+phi)%al);
                    if(carryCount)GatherValues(carry,raw,linearIdx_,static_cast<uint32_t>((nc*data_.scratchPitch+tailStart)*sizeof(T)),2*al);
                }
                AscendC::PipeBarrier<PIPE_V>();out_.EnQue(output);raw_.FreeTensor(raw);output=out_.template DeQue<T>();
                AscendC::DataCopy(y_[base+column*h-phi],output,count);out_.FreeTensor(output);
            }
            const event_t consumed=static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(consumed);AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(consumed);
        }
        if(carryCount){
            auto output=out_.template AllocTensor<T>();GatherValues(output,carry,linearIdx_,static_cast<uint32_t>((h-carryCount-tailStart)*sizeof(T)),carryCount);
            out_.EnQue(output);output=out_.template DeQue<T>();AscendC::DataCopyExtParams op{1,static_cast<uint32_t>(carryCount*sizeof(T)),0,0,0};
            AscendC::DataCopyPad(y_[last*matrixElements-carryCount],output,op);out_.FreeTensor(output);
        }
        if(packLoaded_)packQ_.FreeTensor(packIdx_);
    }
private:
    TransposeL2StagedHeader data_{};AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN,1> raw_,packQ_;
    AscendC::TQue<AscendC::TPosition::VECOUT,1> out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> pad_,linear_,carry_;
    AscendC::LocalTensor<uint32_t> linearIdx_,packIdx_;AscendC::GlobalTensor<uint32_t> packMap_;
    AscendC::GlobalTensor<T> x_,y_,tempGm_;bool packLoaded_=false;uint32_t lastPhi_=0;
};
}
#endif
