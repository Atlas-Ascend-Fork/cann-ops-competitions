// Members of KernelIndexAddContiguousAxisOwner. Rank only destinations;
// each destination still consumes its source occurrences in original order.
    TBuf<QuePosition::VECCALC> rankBuffer_;
    LocalTensor<uint32_t> rankForward_,rankBackward_,rankInverse_,rankRows_;
    uint32_t rankCounts_[32],rankOffsets_[32];
    __aicore__ inline void PrepareRanked(){
        const uint32_t lanes=tiling_->inverseAlignedDestinations,batch=tiling_->inverseBatchRows,n=lanes*batch,capacity=tiling_->inverseWaveCapacity;
        auto order=countBuffer_.Get<uint32_t>();auto packed=waveBuffer_.Get<uint32_t>();auto backup=sourceBuffer_.Get<uint32_t>();
        DataCopy(backup,packed,capacity*lanes);PipeBarrier<PIPE_V>();
        uint32_t sortN=1;while(sortN<lanes)sortN*=2;
        auto keys=packed.template ReinterpretCast<float>();auto ids=packed[sortN];auto pairs=packed[2*sortN].template ReinterpretCast<float>();auto tmp=packed[4*sortN].template ReinterpretCast<float>();
        uint32_t keyBits=0;while((1u<<keyBits)<lanes)++keyBits;
        auto keyWords=keys.template ReinterpretCast<uint32_t>();auto keyInts=keys.template ReinterpretCast<int32_t>();Duplicate(keyWords,uint32_t(0),sortN);CreateVecIndex(ids.template ReinterpretCast<int32_t>(),int32_t(0),sortN);PipeBarrier<PIPE_V>();ShiftLeft(keyWords,order,keyBits,lanes);PipeBarrier<PIPE_V>();Adds(keyInts,keyInts,int32_t(0x3f800000u+(1u<<keyBits)-1),lanes);PipeBarrier<PIPE_V>();Sub(keyInts,keyInts,ids.template ReinterpretCast<int32_t>(),lanes);PipeBarrier<PIPE_V>();
        LocalTensor<float> concat;Concat(concat,keys,tmp,sortN/32);Sort<float,true>(pairs,concat,ids,tmp,sortN/32);PipeBarrier<PIPE_V>();Extract(keys,ids,pairs,sortN/32);PipeBarrier<PIPE_V>();DataCopy(order,ids,lanes);SharedSync<HardEvent::V_S>();
        rankForward_=rankBuffer_.Get<uint32_t>();rankBackward_=rankForward_[Blocked?lanes:n];rankInverse_=rankBackward_[Blocked?lanes:n];rankRows_=rankInverse_[lanes];

        uint32_t offset=0;for(uint32_t w=0;w<waveCount_;++w){uint32_t lo=0,hi=lanes;const uint32_t threshold=0x3f800000u+((w+1)<<keyBits);while(lo<hi){uint32_t mid=(lo+hi)/2;if(keyWords.GetValue(mid)>=threshold)lo=mid+1;else hi=mid;}rankCounts_[w]=lo*batch;rankOffsets_[w]=offset;offset+=AlignUp(Blocked?lo:lo*batch,8);}
        SharedSync<HardEvent::S_V>();Duplicate(keyWords,uint32_t(0),sortN);PipeBarrier<PIPE_V>();Muls(keyInts,order.template ReinterpretCast<int32_t>(),int32_t(-1),lanes);PipeBarrier<PIPE_V>();Adds(keyInts,keyInts,int32_t(0x3f800000u+(1u<<keyBits)-1),lanes);CreateVecIndex(ids.template ReinterpretCast<int32_t>(),int32_t(0),sortN);PipeBarrier<PIPE_V>();Concat(concat,keys,tmp,sortN/32);Sort<float,true>(pairs,concat,ids,tmp,sortN/32);PipeBarrier<PIPE_V>();Extract(keys,ids,pairs,sortN/32);PipeBarrier<PIPE_V>();DataCopy(rankInverse_,ids,lanes);PipeBarrier<PIPE_V>();
        if constexpr(Blocked){
            Muls(rankForward_.template ReinterpretCast<int32_t>(),order.template ReinterpretCast<int32_t>(),int32_t(32),lanes);Muls(rankBackward_.template ReinterpretCast<int32_t>(),rankInverse_.template ReinterpretCast<int32_t>(),int32_t(32),lanes);Muls(rankRows_.template ReinterpretCast<int32_t>(),order.template ReinterpretCast<int32_t>(),int32_t(4),lanes);PipeBarrier<PIPE_V>();
            for(uint32_t w=0;w<waveCount_;++w){const uint32_t active=rankCounts_[w]/16,off=rankOffsets_[w];if(active){Duplicate(packed[off],uint32_t(0),AlignUp(active,8));PipeBarrier<PIPE_V>();Gather(packed[off],backup[w*lanes],rankRows_,0,active);PipeBarrier<PIPE_V>();Muls(packed[off].template ReinterpretCast<int32_t>(),packed[off].template ReinterpretCast<int32_t>(),int32_t(16),active);PipeBarrier<PIPE_V>();}}
            const uint64_t selfBase=reinterpret_cast<uint64_t>(destinationLocal_.GetPhyAddr()),srcBase=reinterpret_cast<uint64_t>(sourceLocal_.GetPhyAddr()),transBase=reinterpret_cast<uint64_t>(this->rankTranspose_.template Get<uint16_t>().GetPhyAddr()),outBase=reinterpret_cast<uint64_t>(gatherBuffer_.Get<T>().GetPhyAddr());
            auto va=this->blockVA_.template Get<uint64_t>();
            for(uint32_t group=0;group<4;++group){uint64_t a=0,b=0,c=0,d=0,e=0;for(uint32_t j=0;j<4;++j){uint32_t r=group*4+j,shift=j*16;a|=(((selfBase+r*lanes*2)>>5)&0x1fffULL)<<shift;b|=(((transBase+r*32)>>5)&0x1fffULL)<<shift;c|=(((srcBase+r*tiling_->inverseSourceRowElements*2)>>5)&0x1fffULL)<<shift;d|=(((srcBase+r*32)>>5)&0x1fffULL)<<shift;e|=(((outBase+r*lanes*2)>>5)&0x1fffULL)<<shift;}va.SetValue(group,a);va.SetValue(4+group,b);va.SetValue(8+group,c);va.SetValue(12+group,d);va.SetValue(16+group,e);}SharedSync<HardEvent::S_V>();
            SharedSync<HardEvent::V_MTE2>();return;
        }
        auto fwd=rankForward_.template ReinterpretCast<int32_t>();auto back=rankBackward_.template ReinterpretCast<int32_t>();auto row=rankRows_.template ReinterpretCast<int32_t>();auto ff=rankForward_.template ReinterpretCast<float>();
        CreateVecIndex(back,int32_t(0),n);PipeBarrier<PIPE_V>();Cast(ff,back,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();Muls(ff,ff,1.0f/float(int32_t(batch)),n);PipeBarrier<PIPE_V>();Adds(ff,ff,0.0001f,n);PipeBarrier<PIPE_V>();Cast(row,ff,RoundMode::CAST_FLOOR,n);PipeBarrier<PIPE_V>();Muls(fwd,row,int32_t(batch),n);PipeBarrier<PIPE_V>();Sub(back,back,fwd,n);PipeBarrier<PIPE_V>();Muls(row,row,int32_t(4),n);PipeBarrier<PIPE_V>();Gather(rankForward_,order,rankRows_,0,n);PipeBarrier<PIPE_V>();
        Muls(row,back,int32_t(tiling_->inverseSourceRowElements*sizeof(T)),n);PipeBarrier<PIPE_V>();Muls(back,back,int32_t(lanes*sizeof(T)),n);Muls(fwd,fwd,int32_t(sizeof(T)),n);PipeBarrier<PIPE_V>();Add(fwd,fwd,back,n);PipeBarrier<PIPE_V>();Sub(back,fwd,back,n);PipeBarrier<PIPE_V>();if constexpr(sizeof(T)==2){Muls(back,back,int32_t(2),n);PipeBarrier<PIPE_V>();}
        for(uint32_t w=0;w<waveCount_;++w){const uint32_t count=rankCounts_[w],off=rankOffsets_[w];if(count){Gather(packed[off],backup[w*lanes],rankBackward_,0,count);PipeBarrier<PIPE_V>();Add(packed[off].template ReinterpretCast<int32_t>(),packed[off].template ReinterpretCast<int32_t>(),row,count);PipeBarrier<PIPE_V>();}}
        // Original wave table is now dead; reuse the source cache as scratch.
        auto temp=sourceBuffer_.Get<float>();auto ti=temp.template ReinterpretCast<int32_t>();CreateVecIndex(back,int32_t(0),n);PipeBarrier<PIPE_V>();Cast(temp,back,RoundMode::CAST_RINT,n);PipeBarrier<PIPE_V>();Muls(temp,temp,1.0f/float(int32_t(lanes)),n);PipeBarrier<PIPE_V>();Adds(temp,temp,0.0001f,n);PipeBarrier<PIPE_V>();Cast(row,temp,RoundMode::CAST_FLOOR,n);PipeBarrier<PIPE_V>();Muls(ti,row,int32_t(lanes),n);PipeBarrier<PIPE_V>();Sub(back,back,ti,n);PipeBarrier<PIPE_V>();Muls(back,back,int32_t(4),n);PipeBarrier<PIPE_V>();Gather(temp.template ReinterpretCast<uint32_t>(),rankInverse_,rankBackward_,0,n);PipeBarrier<PIPE_V>();Muls(ti,ti,int32_t(batch),n);PipeBarrier<PIPE_V>();Add(back,ti,row,n);PipeBarrier<PIPE_V>();Muls(back,back,int32_t(sizeof(T)),n);SharedSync<HardEvent::V_MTE2>();
    }
    __aicore__ inline void ProcessRanked(){
        const uint64_t core=GetBlockIdx(),tasks=tiling_->tasksPerCore+(core<tiling_->extraCoreTasks?1:0),first=core*tiling_->tasksPerCore+(core<tiling_->extraCoreTasks?core:tiling_->extraCoreTasks);uint64_t end=(first+tasks)*tiling_->tileInner;if(end>tiling_->outer)end=tiling_->outer;
        for(uint64_t pos=first*tiling_->tileInner;pos<end;pos+=tiling_->inverseBatchRows){uint32_t rows=end-pos<tiling_->inverseBatchRows?end-pos:tiling_->inverseBatchRows;RankedBatch(pos,rows);}
    }
    __aicore__ inline void RankedBatch(uint64_t first,uint32_t rows){
        if constexpr(Blocked){BlockedBatch(first,rows);return;}
        const uint32_t lanes=tiling_->inverseAlignedDestinations,batch=tiling_->inverseBatchRows,n=lanes*batch,pitch=tiling_->inverseSourceRowElements,dbytes=tiling_->destinationAxis*sizeof(T),sbytes=tiling_->indexCount*sizeof(T);
        DataCopyExtParams dp{static_cast<uint16_t>(rows),dbytes,0,static_cast<uint32_t>((lanes*sizeof(T)-AlignUp(dbytes,32))/32),0},sp{static_cast<uint16_t>(rows),sbytes,0,static_cast<uint32_t>((pitch*sizeof(T)-AlignUp(sbytes,32))/32),0};DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
        DataCopyPad(destinationLocal_,selfGm_[first*tiling_->destinationAxis],dp,pad);DataCopyPad(sourceLocal_,sourceGm_[first*tiling_->indexCount],sp,pad);PipeBarrier<PIPE_ALL>();auto gathered=gatherBuffer_.Get<T>();auto waves=waveBuffer_.Get<uint32_t>();
        Gather(gathered,destinationLocal_,rankForward_,0,n);PipeBarrier<PIPE_V>();DataCopy(destinationLocal_,gathered,n);PipeBarrier<PIPE_V>();
        for(uint32_t w=0;w<waveCount_;++w){uint32_t count=rankCounts_[w];if(count){Gather(gathered,sourceLocal_,waves[rankOffsets_[w]],0,count);PipeBarrier<PIPE_V>();Add(destinationLocal_,destinationLocal_,gathered,count);PipeBarrier<PIPE_V>();}}
        SharedSync<HardEvent::V_S>();SharedSync<HardEvent::MTE2_S>();auto tails=indexLocal_.template ReinterpretCast<uint32_t>();for(uint32_t t=0;t<tailCount_;++t){uint32_t value=tails.GetValue(t);if constexpr(Prebuilt){if(value==0xffffffffu)continue;}uint32_t d=rankInverse_.GetValue(value>>16),m=value&65535u;for(uint32_t r=0;r<rows;++r)AddSingle(d*batch+r,r*pitch+m);}
        SharedSync<HardEvent::S_V>();Gather(gathered,destinationLocal_,rankBackward_,0,n);SharedSync<HardEvent::V_MTE3>();DataCopyExtParams outp{static_cast<uint16_t>(rows),dbytes,static_cast<uint32_t>((lanes*sizeof(T)-AlignUp(dbytes,32))/32),0,0};DataCopyPad(outGm_[first*tiling_->destinationAxis],gathered,outp);PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void BlockedBatch(uint64_t first,uint32_t rows){
        const uint32_t lanes=tiling_->inverseAlignedDestinations,pitch=tiling_->inverseSourceRowElements,dbytes=tiling_->destinationAxis*2,sbytes=tiling_->indexCount*2;
        DataCopyExtParams dp{static_cast<uint16_t>(rows),dbytes,0,static_cast<uint32_t>((lanes*2-AlignUp(dbytes,32))/32),0},sp{static_cast<uint16_t>(rows),sbytes,0,static_cast<uint32_t>((pitch*2-AlignUp(sbytes,32))/32),0};DataCopyPadExtParams<T> pad{false,0,0,static_cast<T>(0)};
        DataCopyPad(destinationLocal_,selfGm_[first*tiling_->destinationAxis],dp,pad);DataCopyPad(sourceLocal_,sourceGm_[first*tiling_->indexCount],sp,pad);PipeBarrier<PIPE_ALL>();
        auto trans=this->rankTranspose_.template Get<uint16_t>();auto raw=sourceLocal_.template ReinterpretCast<uint16_t>();auto dest=destinationLocal_.template ReinterpretCast<uint16_t>();auto gathered=gatherBuffer_.Get<T>();auto g=gathered.template ReinterpretCast<uint16_t>();auto wave=waveBuffer_.Get<uint32_t>();
        BlockTranspose(1,0,static_cast<uint8_t>(lanes/16),16,1);PipeBarrier<PIPE_V>();Gatherb(dest,trans,rankForward_,static_cast<uint8_t>(lanes/8),GatherRepeatParams{1,8});PipeBarrier<PIPE_V>();
        BlockTranspose(1,2,static_cast<uint8_t>(pitch/16),16,1);PipeBarrier<PIPE_V>();
        for(uint32_t w=0;w<waveCount_;++w){const uint32_t active=rankCounts_[w]/16;if(active){Gatherb(g,trans,wave[rankOffsets_[w]],static_cast<uint8_t>(AlignUp(active,8)/8),GatherRepeatParams{1,8});PipeBarrier<PIPE_V>();Add(destinationLocal_,destinationLocal_,gathered,active*16);PipeBarrier<PIPE_V>();}}
        SharedSync<HardEvent::V_S>();SharedSync<HardEvent::MTE2_S>();auto tails=indexLocal_.template ReinterpretCast<uint32_t>();for(uint32_t t=0;t<tailCount_;++t){uint32_t value=tails.GetValue(t);if constexpr(Prebuilt){if(value==0xffffffffu)continue;}uint32_t d=rankInverse_.GetValue(value>>16),m=value&65535u;for(uint32_t r=0;r<rows;++r)AddSingle(d*16+r,r*pitch+m);}
        SharedSync<HardEvent::S_V>();Gatherb(raw,dest,rankBackward_,static_cast<uint8_t>(lanes/8),GatherRepeatParams{1,8});PipeBarrier<PIPE_V>();
        BlockTranspose(4,3,static_cast<uint8_t>(lanes/16),1,16);SharedSync<HardEvent::V_MTE3>();DataCopyExtParams outp{static_cast<uint16_t>(rows),dbytes,static_cast<uint32_t>((lanes*2-AlignUp(dbytes,32))/32),0,0};DataCopyPad(outGm_[first*tiling_->destinationAxis],gathered,outp);PipeBarrier<PIPE_ALL>();
    }

    // CANN 8.5.0 dav_c220 packed-VA entry used by its public TransDataTo5HD wrapper.
    // Packing is invariant within this call; store it once in UB.
    __aicore__ inline void BlockTranspose(uint32_t dstSlot,uint32_t srcSlot,uint8_t repeat,uint16_t dstStride,uint16_t srcStride){
        auto va=this->blockVA_.template Get<uint64_t>();
        TransDataTo5HDVldVaRegImpl<uint16_t>((__ubuf__ uint64_t*)va[dstSlot*4].GetPhyAddr(),(__ubuf__ uint64_t*)va[srcSlot*4].GetPhyAddr(),{false,false,repeat,dstStride,srcStride});
    }
