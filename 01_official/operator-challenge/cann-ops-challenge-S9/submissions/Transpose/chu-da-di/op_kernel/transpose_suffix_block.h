#ifndef S9_TRANSPOSE_SUFFIX_BLOCK_H
#define S9_TRANSPOSE_SUFFIX_BLOCK_H
namespace s9transpose {
// [A,B,C,D] -> [C,B,A,D]. Keep all B segments of a C range on one core.
// MTE2 reads adjacent C*D values; UB Gather changes their order before MTE3.
template<typename T> class KernelTransposeSuffixBlock {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeSuffixBlockHeader& t, GM_ADDR map) {
        data_ = t;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        x_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        pipe_.InitBuffer(in_, t.inputBuffers, t.base.shape[0] * t.inputPitch * sizeof(T));
        pipe_.InitBuffer(out_, t.outputBuffers, t.columns * t.outputPitch * sizeof(T));
        pipe_.InitBuffer(index_, (t.indexCount * sizeof(uint32_t) + 31) / 32 * 32);
        indices_ = index_.template Get<uint32_t>();
        AscendC::GlobalTensor<uint32_t> globalMap;
        globalMap.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(map));
        AscendC::DataCopyExtParams p{1, static_cast<uint32_t>(t.indexCount * sizeof(uint32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<uint32_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(indices_, globalMap, p, pad);
        const event_t ready = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(ready);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(ready);
    }
    __aicore__ inline void CopyFibers(uint64_t first,uint64_t last){
        const uint64_t a=data_.base.shape[0],b=data_.base.shape[1],c=data_.base.shape[2],d=data_.base.shape[3];constexpr uint32_t align=32/sizeof(T);
        const uint32_t alignedD=(d+align-1)/align*align;
        for(uint64_t fiber=first;fiber<last;){
            const uint64_t column=fiber/(a*b),middle=(fiber/a)%b,aa=fiber%a;
            const uint32_t run=a-aa<last-fiber?a-aa:last-fiber;auto input=in_.template AllocTensor<T>();
            AscendC::DataCopyExtParams ip{static_cast<uint16_t>(run),static_cast<uint32_t>(d*sizeof(T)),static_cast<uint32_t>((b*c*d-d)*sizeof(T)),(data_.inputPitch-alignedD)/align,0};AscendC::DataCopyPadExtParams<T> pp{false,0,0,static_cast<T>(0)};
            AscendC::DataCopyPad(input,x_[((aa*b+middle)*c+column)*d],ip,pp);in_.EnQue(input);input=in_.template DeQue<T>();auto output=out_.template AllocTensor<T>();
            const uint32_t count=run*d;
            for(uint32_t offset=0;offset<count;offset+=8192){const uint32_t once=count-offset<8192?count-offset:8192;
                if constexpr(sizeof(T)==2)AscendC::Gather(output.template ReinterpretCast<half>()[offset],input.template ReinterpretCast<half>(),indices_[offset],0,once);
                else if constexpr(sizeof(T)==4)AscendC::Gather(output.template ReinterpretCast<float>()[offset],input.template ReinterpretCast<float>(),indices_[offset],0,once);
            }
            out_.EnQue(output);in_.FreeTensor(input);output=out_.template DeQue<T>();AscendC::DataCopyExtParams op{1,static_cast<uint32_t>(count*sizeof(T)),0,0,0};AscendC::DataCopyPad(y_[fiber*d],output,op);out_.FreeTensor(output);fiber+=run;
        }
    }
    __aicore__ inline void ProcessColumns(uint64_t first,uint64_t last){
        const uint64_t a=data_.base.shape[0],b=data_.base.shape[1],c=data_.base.shape[2],d=data_.base.shape[3];constexpr uint32_t align=32/sizeof(T);
        for (uint64_t column = first; column < last; column += data_.columns) {
            const uint32_t nc = static_cast<uint32_t>(last - column < data_.columns ? last - column : data_.columns);
            const uint32_t alignedInput = (nc * d + align - 1) / align * align;
            for (uint64_t middle = 0; middle < b; ++middle) {
                auto input = in_.template AllocTensor<T>();
                AscendC::DataCopyExtParams ip{static_cast<uint16_t>(a), static_cast<uint32_t>(nc * d * sizeof(T)),
                    static_cast<uint32_t>((b * c * d - nc * d) * sizeof(T)),
                    (data_.inputPitch - alignedInput) / align, 0};
                AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
                AscendC::DataCopyPad(input, x_[(middle * c + column) * d], ip, pad);
                in_.EnQue(input); input = in_.template DeQue<T>();
                auto output = out_.template AllocTensor<T>();
                for (uint32_t mapColumn=0;mapColumn<nc;mapColumn+=data_.mapColumns) {
                    const uint32_t n = nc-mapColumn<data_.mapColumns?nc-mapColumn:data_.mapColumns;
                    const uint32_t count = n * data_.outputPitch;
                    for (uint32_t offset = 0; offset < count; offset += 8192) {
                        const uint32_t once = count - offset < 8192 ? count - offset : 8192;
                        if constexpr (sizeof(T) == 2) {
                            AscendC::Gather(output.template ReinterpretCast<half>()[mapColumn*data_.outputPitch+offset],
                                input.template ReinterpretCast<half>(), indices_[offset], static_cast<uint32_t>(mapColumn*d*sizeof(T)), once);
                        } else if constexpr (sizeof(T) == 4) {
                            AscendC::Gather(output.template ReinterpretCast<float>()[mapColumn*data_.outputPitch+offset],
                                input.template ReinterpretCast<float>(), indices_[offset], static_cast<uint32_t>(mapColumn*d*sizeof(T)), once);
                        }
                    }
                }
                out_.EnQue(output); in_.FreeTensor(input);
                output = out_.template DeQue<T>();
                AscendC::DataCopyExtParams op{static_cast<uint16_t>(nc), static_cast<uint32_t>(a * d * sizeof(T)),
                    0, static_cast<uint32_t>((b - 1) * a * d * sizeof(T)), 0};
                AscendC::DataCopyPad(y_[(column * b + middle) * a * d], output, op);
                out_.FreeTensor(output);
            }
        }
    }
    __aicore__ inline void Process(){
        const uint64_t a=data_.base.shape[0],b=data_.base.shape[1],c=data_.base.shape[2];
        const uint64_t extent=data_.ownershipMode?a*b*c:c,tasks=(extent+data_.ownershipQuantum-1)/data_.ownershipQuantum;
        const uint64_t cores=AscendC::GetBlockNum(),bid=AscendC::GetBlockIdx(),small=tasks/cores,extra=tasks%cores;
        const uint64_t first=(bid*small+(bid<extra?bid:extra))*data_.ownershipQuantum,end0=first+(small+(bid<extra?1:0))*data_.ownershipQuantum,last=end0<extent?end0:extent;
        if(!data_.ownershipMode){ProcessColumns(first,last);return;}
        const uint64_t firstColumn=(first+a*b-1)/(a*b),lastColumn=last/(a*b);
        if(firstColumn>=lastColumn){CopyFibers(first,last);return;}
        CopyFibers(first,firstColumn*a*b);ProcessColumns(firstColumn,lastColumn);CopyFibers(lastColumn*a*b,last);
    }
private:
    TransposeSuffixBlockHeader data_{};
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> in_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> out_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> index_;
    AscendC::LocalTensor<uint32_t> indices_;
    AscendC::GlobalTensor<T> x_, y_;
};
}
#endif
