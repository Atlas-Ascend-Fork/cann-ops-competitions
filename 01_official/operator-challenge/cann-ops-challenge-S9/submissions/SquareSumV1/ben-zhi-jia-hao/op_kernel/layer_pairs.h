#pragma once

namespace ss_layer_pairs {
using namespace AscendC;
__aicore__ inline uint32_t Round8(uint32_t n) { return (n+7U)&~7U; }
__aicore__ inline uint32_t Round16(uint32_t n) { return (n+15U)&~15U; }

class Kernel {
public:
    __aicore__ inline void Run(GM_ADDR input,GM_ADDR output,GM_ADDR tiling)
    {
        auto t=reinterpret_cast<const __gm__ SquareSumV1TilingData*>(tiling);
        const uint64_t rows=t->outTiles,layers=t->redPerTile;
        const uint32_t width=static_cast<uint32_t>(t->innerLen),cores=t->usedCores;
        const uint32_t batch=t->rowBatch,rf=Round8(batch),rh=Round16(batch);
        const uint32_t inputCount=t->ubChunk,pairCap=t->accLen;
        const uint32_t groupStride=t->rowPitch;
        const uint32_t edgeAt=pairCap*4U;
        const uint32_t edgeHalfAt=edgeAt+rf*4U;
        const uint32_t edgeIndexAt=edgeHalfAt+rh*2U;
        const uint32_t outIndexAt=edgeIndexAt+rf*4U;
        const uint32_t sumAt=outIndexAt+rf*4U;
        const uint32_t resultAt=sumAt+16U*groupStride*4U;
        const uint32_t oneAt=resultAt+rf*4U;
        TPipe pipe;
        pipe.InitBuffer(input_,2,inputCount*2U);
        pipe.InitBuffer(calc_,oneAt+32U);
        pipe.InitBuffer(output_,1,rh*2U);
        pairs_=calc_.GetWithOffset<float>(pairCap,0);
        edges_=calc_.GetWithOffset<float>(rf,edgeAt);
        edgeHalf_=calc_.GetWithOffset<half>(rh,edgeHalfAt);
        edgeIndex_=calc_.GetWithOffset<uint32_t>(rf,edgeIndexAt);
        outIndex_=calc_.GetWithOffset<uint32_t>(rf,outIndexAt);
        sums_=calc_.GetWithOffset<float>(16U*groupStride,sumAt);
        result_=calc_.GetWithOffset<float>(rf,resultAt);
        ones_=calc_.GetWithOffset<half>(16,oneAt);
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input));
        x_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        y_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output));
        Duplicate(ones_,static_cast<half>(1),16);
        const uint64_t id=GetBlockIdx();
        const uint64_t step=static_cast<uint64_t>(batch)*cores;
        bool ready=false;
        for (uint64_t first=id*batch;first<rows;first+=step) {
            const uint32_t count=static_cast<uint32_t>(rows-first<batch ? rows-first : batch);
            const uint32_t elements=count*width;
            const uint32_t rounded=Round16(elements);
            uint64_t layer=id%layers;
            Load((layer*rows+first)*width,elements);
            if (++layer==layers) layer=0;
            if (!ready) { Indices(width,batch,groupStride); ready=true; }
            Duplicate(pairs_,0.0f,pairCap);
            Duplicate(edges_,0.0f,rf);
            for (uint64_t j=0;j<layers;++j) {
                if (j+1U<layers) {
                    Load((layer*rows+first)*width,elements);
                    if (++layer==layers) layer=0;
                }
                auto in=input_.DeQue<half>();
                Mul(in,in,in,rounded);
                pipe_barrier(PIPE_V);
                Gather(edgeHalf_,in,edgeIndex_,0U,count);
                pipe_barrier(PIPE_V);
                AddHalf(edges_,edgeHalf_,count);
                Compact(in,rounded);
                AddHalf(pairs_,in,(elements+1U)/2U);
                input_.FreeTensor(in);
            }
            const uint32_t halfWidth=(width-1U)/2U;
            for (uint32_t residue=0;residue<16U && residue<count;++residue) {
                const uint32_t offset=(residue/2U)*width+((residue&1U) ? halfWidth+1U : 0U);
                const uint32_t head=offset&7U;
                const uint64_t bits=((uint64_t{1}<<halfWidth)-1U)<<head;
                uint64_t mask[2]={bits,0};
                const uint32_t repeat=(count-residue+15U)/16U;
                WholeReduceSum<float>(sums_[residue*groupStride],pairs_[offset&~7U],
                    mask,repeat,1,1,width);
            }
            pipe_barrier(PIPE_V);
            Gather(result_,sums_,outIndex_,0U,count);
            pipe_barrier(PIPE_V);
            Add(result_,result_,edges_,count);
            pipe_barrier(PIPE_V);
            auto out=output_.AllocTensor<half>();
            Cast(out,result_,RoundMode::CAST_NONE,count);
            output_.EnQue(out);
            out=output_.DeQue<half>();
            DataCopyExtParams cp{1,count*2U,0,0,0};
            DataCopyPad(y_[first],out,cp);
            output_.FreeTensor(out);
        }
    }
private:
    __aicore__ inline void Load(uint64_t offset,uint32_t count)
    {
        auto in=input_.AllocTensor<half>();
        if ((count&15U)==0) DataCopy(in,x_[offset],count);
        else {
            DataCopyExtParams cp{1,count*2U,0,0,0};
            DataCopyPadExtParams<half> pad{true,0,static_cast<uint8_t>(Round16(count)-count),static_cast<half>(0)};
            DataCopyPad(in,x_[offset],cp,pad);
        }
        input_.EnQue(in);
    }
    __aicore__ inline void AddHalf(LocalTensor<float> acc,LocalTensor<half> source,uint32_t count)
    {
        const BinaryRepeatParams rp{1,1,0,8,4,0};
        SetMaskCount();
        SetVectorMask<float,MaskMode::COUNTER>(0,count);
        MulAddDst<float,half,false>(acc,source,ones_,uint64_t{0},1,rp);
        pipe_barrier(PIPE_V);
        SetMaskNorm(); ResetMask();
    }
    __aicore__ inline void Compact(LocalTensor<half> data,uint32_t count)
    {
        uint32_t off=0;
        while (off+128U<=count) {
            const uint32_t reps=(count-off)/128U<255 ? (count-off)/128U : 255;
            PairReduceSum<half>(data[off/2U],data[off],reps,128,1,1,8);
            off+=reps*128U;
        }
        if (off<count) PairReduceSum<half>(data[off/2U],data[off],1,count-off,1,1,8);
        pipe_barrier(PIPE_V);
    }
    __aicore__ inline void Indices(uint32_t width,uint32_t rows,uint32_t groupStride)
    {
        auto q=pairs_.ReinterpretCast<int32_t>();
        auto r=result_.ReinterpretCast<int32_t>();
        auto edge=edgeIndex_.ReinterpretCast<int32_t>();
        ArithProgression<int32_t>(edge,0,1,rows);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(q,edge,1,rows);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(r,q,1,rows);
        pipe_barrier(PIPE_V);
        Sub(edge,edge,r,rows);
        Muls<int32_t>(q,q,2U*width,rows);
        pipe_barrier(PIPE_V);
        Add(edge,edge,q,rows);
        pipe_barrier(PIPE_V);
        Adds<int32_t>(edge,edge,width-1U,rows);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(edge,edge,1,rows);
        auto output=outIndex_.ReinterpretCast<int32_t>();
        ArithProgression<int32_t>(output,0,1,rows);
        pipe_barrier(PIPE_V);
        ShiftRight<int32_t>(q,output,4,rows);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(r,q,4,rows);
        pipe_barrier(PIPE_V);
        Sub(output,output,r,rows);
        pipe_barrier(PIPE_V);
        Muls<int32_t>(output,output,groupStride,rows);
        pipe_barrier(PIPE_V);
        Add(output,output,q,rows);
        pipe_barrier(PIPE_V);
        ShiftLeft<int32_t>(output,output,2,rows);
        pipe_barrier(PIPE_V);
    }
    TQue<TPosition::VECIN,2> input_;
    TQue<TPosition::VECOUT,1> output_;
    TBuf<TPosition::VECCALC> calc_;
    GlobalTensor<half> x_,y_;
    LocalTensor<float> pairs_,edges_,sums_,result_;
    LocalTensor<half> edgeHalf_,ones_;
    LocalTensor<uint32_t> edgeIndex_,outIndex_;
};
}
