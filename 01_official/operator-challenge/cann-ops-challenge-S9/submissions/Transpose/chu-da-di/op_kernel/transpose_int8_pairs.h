#ifndef S9_TRANSPOSE_INT8_PAIRS_H
#define S9_TRANSPOSE_INT8_PAIRS_H
namespace s9transpose {
class KernelTransposeInt8Pairs {
public:
    __aicore__ inline void Init(GM_ADDR x,GM_ADDR y,const TransposeInt8PairsTilingData& t){
        d_=t;x_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(x),t.totalElements);y_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(y),t.totalElements);
        const uint32_t elements=32*t.tileBatchCols*32;pipe_.InitBuffer(in_,2,elements);pipe_.InitBuffer(out_,2,elements);
    }
    __aicore__ inline void Process(){
        ProcessTyped<uint64_t,true>();
    }
private:
    template<typename I,bool Drain>
    __aicore__ inline void ProcessTyped(){
        const I h=static_cast<I>(d_.matrixH),w=static_cast<I>(d_.matrixW),outer=static_cast<I>(d_.outer);
        const I batchCols=d_.tileBatchCols,tileCols=d_.tileCols,batches=d_.columnBatches;
        const I pairs=h/32+(h%32!=0),perMatrix=pairs*batches,tasks=outer*perMatrix;
        const I cores=AscendC::GetBlockNum(),bid=AscendC::GetBlockIdx(),small=tasks/cores,extra=tasks%cores;
        const I start=bid*small+(bid<extra?bid:extra),count=small+(bid<extra?1:0);
        for(I task=start;task<start+count;++task){
            const I matrix=task/perMatrix,matrixTask=task%perMatrix,rowPair=matrixTask/batches,columnBatch=matrixTask%batches;
            const I firstTile=columnBatch*batchCols,remainingTiles=tileCols-firstTile,currentTiles=remainingTiles<batchCols?remainingTiles:batchCols;
            const I batchWidth=currentTiles*32,column=firstTile*32,nc=w-column<batchWidth?w-column:batchWidth;
            const I base=matrix*h*w,row=rowPair*32,nr=h-row<32?h-row:32;
            auto input=in_.AllocTensor<int8_t>();AscendC::DataCopyExtParams ip{static_cast<uint16_t>(nr),static_cast<uint32_t>(nc),static_cast<uint32_t>(w-nc),0,0};
            AscendC::DataCopyPadExtParams<int8_t> pp{nc!=batchWidth,0,static_cast<uint8_t>(batchWidth-nc),0};AscendC::DataCopyPad(input,x_[base+row*w+column],ip,pp);in_.EnQue(input);input=in_.DeQue<int8_t>();
            auto output=out_.AllocTensor<int8_t>();uint64_t src[16],dst[16];
            const uint16_t dstStride=currentTiles==1?0:32,srcStride=currentTiles==1?0:1;
            const I halves=nr>16?2:1;
            for(I half=0;half<halves;++half){
                for(I i=0;i<16;++i){src[i]=reinterpret_cast<uint64_t>(input[(half*16+i)*batchWidth].GetPhyAddr());dst[i]=reinterpret_cast<uint64_t>(output[i*32].GetPhyAddr());}
                AscendC::TransDataTo5HDParams lo{half!=0,false,static_cast<uint8_t>(currentTiles),dstStride,srcStride};AscendC::TransDataTo5HD<int8_t>(dst,src,lo);AscendC::PipeBarrier<PIPE_V>();
                for(I i=0;i<16;++i)dst[i]=reinterpret_cast<uint64_t>(output[(16+i)*32].GetPhyAddr());
                AscendC::TransDataTo5HDParams hi{half!=0,true,static_cast<uint8_t>(currentTiles),dstStride,srcStride};AscendC::TransDataTo5HD<int8_t>(dst,src,hi);AscendC::PipeBarrier<PIPE_V>();
            }
            out_.EnQue(output);in_.FreeTensor(input);output=out_.DeQue<int8_t>();
            AscendC::DataCopyExtParams op{static_cast<uint16_t>(nc),static_cast<uint32_t>(nr),0,static_cast<uint32_t>(h-nr),0};AscendC::DataCopyPad(y_[base+column*h+row],output,op);
            if constexpr(Drain)AscendC::PipeBarrier<PIPE_MTE3>();
            out_.FreeTensor(output);
        }
    }
    TransposeInt8PairsTilingData d_{};AscendC::TPipe pipe_;AscendC::TQue<AscendC::TPosition::VECIN,1>in_;AscendC::TQue<AscendC::TPosition::VECOUT,1>out_;AscendC::GlobalTensor<int8_t>x_,y_;
};
}
#endif
