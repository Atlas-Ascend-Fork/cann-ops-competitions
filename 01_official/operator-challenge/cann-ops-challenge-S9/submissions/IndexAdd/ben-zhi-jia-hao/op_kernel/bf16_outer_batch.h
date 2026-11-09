#pragma once
// Original j order and bf16 rounding after every add. Temporary padded output
// slabs make parallel outer ownership safe even at unaligned outer boundaries.
class IndexAddBf16OuterBatch {
 public:
  template<class D> __aicore__ inline void Init(GM_ADDR self,GM_ADDR index,GM_ADDR source,
      GM_ADDR output,GM_ADDR workspace,const D &d){
    axis_=d.axisSize;inner_=d.inner;count_=d.indexCount;outer_=d.outer;total_=d.total;
    stride_=(inner_+15)/16*16;batch_=d.bf16Batch;slab_=axis_*inner_;slabPad_=(slab_+15)/16*16;
    self_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(self));
    source_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(source));
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(output));
    work_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(workspace));
    index_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
    pipe_.InitBuffer(acc_,axis_*stride_*2);pipe_.InitBuffer(indices_,(count_+7)/8*32);
    pipe_.InitBuffer(fp0_,stride_*4);pipe_.InitBuffer(fp1_,stride_*4);
    pipe_.InitBuffer(packedRow_,stride_*2);pipe_.InitBuffer(rowOffsets_,((stride_+63)/64*64)*4);
    pipe_.InitBuffer(queue_,2,batch_*stride_*2);
  }
  __aicore__ inline void Process(){
    auto acc=acc_.Get<bfloat16_t>();auto indices=indices_.Get<int32_t>();
    auto a=fp0_.Get<float>();auto b=fp1_.Get<float>();
    auto packed=packedRow_.Get<bfloat16_t>();auto offsets=rowOffsets_.Get<int32_t>();
    const uint32_t offsetN=(stride_+63)/64*64;
    CreateVecIndex(offsets,static_cast<int32_t>(0),offsetN);PipeBarrier<PIPE_V>();
    Muls(offsets,offsets,static_cast<int32_t>(2),offsetN);PipeBarrier<PIPE_V>();
    uint64_t gcd=16,v=inner_;while(v){const uint64_t rem=gcd%v;gcd=v;v=rem;}
    const uint64_t axisGroup=16/gcd,groups=(axis_+axisGroup-1)/axisGroup;
    uint64_t workers=GetBlockNum()/outer_;if(workers<1)workers=1;if(workers>groups)workers=groups;
    const uint64_t begin=GetBlockIdx(),end=outer_*workers;
    if(begin<end){
      DataCopyExtParams ip{1,static_cast<uint32_t>(count_*4),0,0,0};
      DataCopyPadExtParams<int32_t> ipPad{false,0,0,0};DataCopyPad(indices,index_,ip,ipPad);PipeBarrier<PIPE_ALL>();
      DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
      for(uint64_t task=begin;task<end;task+=GetBlockNum()){
        const uint64_t outer=task/workers,part=task%workers;
        const uint64_t axisBegin=groups*part/workers*axisGroup;
        uint64_t axisEnd=groups*(part+1)/workers*axisGroup;if(axisEnd>axis_)axisEnd=axis_;
        DataCopyExtParams cp{static_cast<uint16_t>(axisEnd-axisBegin),static_cast<uint32_t>(inner_*2),0,0,0};
        DataCopyPad(acc[axisBegin*stride_],self_[outer*slab_+axisBegin*inner_],cp,pad);PipeBarrier<PIPE_ALL>();
        Load(outer,0);
        for(uint32_t first=0;first<count_;first+=batch_){
          if(first+batch_<count_)Load(outer,first+batch_);
          auto src=queue_.DeQue<bfloat16_t>();const uint32_t n=count_-first>batch_?batch_:static_cast<uint32_t>(count_-first);
          for(uint32_t j=0;j<n;++j){
            const int32_t ix=indices.GetValue(first+j);
            // Valid indices are part of the operator contract. Do not return
            // before the global barrier on data-dependent invalid input.
            if(ix<0 || static_cast<uint64_t>(ix)<axisBegin || static_cast<uint64_t>(ix)>=axisEnd)continue;
            auto dst=acc[static_cast<uint64_t>(ix)*stride_];
            auto value=src[j*inner_];
            if((j*inner_)&15){Gather(packed,src,offsets.template ReinterpretCast<uint32_t>(),static_cast<uint32_t>(j*inner_*2),static_cast<uint32_t>(inner_));PipeBarrier<PIPE_V>();value=packed;}
            Cast(a,dst,RoundMode::CAST_NONE,inner_);Cast(b,value,RoundMode::CAST_NONE,inner_);
            PipeBarrier<PIPE_V>();Add(a,a,b,inner_);PipeBarrier<PIPE_V>();
            Cast(dst,a,RoundMode::CAST_RINT,inner_);PipeBarrier<PIPE_V>();
          }
          queue_.FreeTensor(src);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(work_[outer*slabPad_+axisBegin*inner_],acc[axisBegin*stride_],cp);PipeBarrier<PIPE_ALL>();
      }
    }
    PipeBarrier<PIPE_ALL>();SyncAll<true>();
    const uint64_t blocks=(total_+15)/16;
    uint64_t pos=blocks*GetBlockIdx()/GetBlockNum()*16,limit=blocks*(GetBlockIdx()+1)/GetBlockNum()*16;
    if(limit>total_)limit=total_;
    const uint32_t tile=axis_*stride_>4096?4096:static_cast<uint32_t>(axis_*stride_);
    DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
    for(;pos<limit;){
      const uint64_t outer=pos/slab_,lane=pos%slab_;
      uint64_t n=limit-pos;if(n>tile)n=tile;if(n>slab_-lane)n=slab_-lane;
      DataCopyExtParams cp{1,static_cast<uint32_t>(n*2),0,0,0};
      DataCopyPad(acc,work_[outer*slabPad_+lane],cp,pad);PipeBarrier<PIPE_ALL>();
      DataCopyPad(output_[pos],acc,cp);PipeBarrier<PIPE_ALL>();pos+=n;
    }
  }
 private:
  __aicore__ inline void Load(uint64_t outer,uint32_t j){
    const uint32_t n=count_-j>batch_?batch_:static_cast<uint32_t>(count_-j);
    auto src=queue_.AllocTensor<bfloat16_t>();
    DataCopyExtParams cp{1,static_cast<uint32_t>(n*inner_*2),0,0,0};
    DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
    DataCopyPad(src,source_[(outer*count_+j)*inner_],cp,pad);queue_.EnQue(src);
  }
  TPipe pipe_;TBuf<TPosition::VECCALC> acc_,indices_,fp0_,fp1_,packedRow_,rowOffsets_;TQue<QuePosition::VECIN,2> queue_;
  GlobalTensor<bfloat16_t> self_,source_,output_,work_;GlobalTensor<int32_t> index_;
  uint64_t axis_,inner_,count_,outer_,total_,slab_,slabPad_;uint32_t stride_,batch_;
};
