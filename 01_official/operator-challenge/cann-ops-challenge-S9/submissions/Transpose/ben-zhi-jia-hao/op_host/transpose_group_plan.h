#ifndef TRANSPOSE_GROUP_PLAN_H
#define TRANSPOSE_GROUP_PLAN_H
#include "transpose_plan.h"
namespace s9 { namespace transpose {
// A,B,C,D -> D,B,A,C: fuse output-adjacent A,C, splitting source loads at C.
inline bool BuildGroupPlan(const int64_t* shape,int rank,const int64_t* perm,int permRank,
                          uint32_t ds,uint64_t ub,uint32_t cores,bool disableRead,
                          uint32_t reqMt,TransposePlan& p) {
    Geometry g;
    if(!BuildGeometry(shape,rank,perm,permRank,g)||g.collapsedRank!=4||ds!=4||!cores||ub<=12288)return false;
    const int order[4]={3,1,0,2};for(int i=0;i<4;++i)if(g.collapsedPerm[i]!=order[i])return false;
    const uint64_t a=g.collapsedShape[0],b=g.collapsedShape[1],c=g.collapsedShape[2],k=g.collapsedShape[3],m=a*c;
    if(m<16||k<8||c==0)return false;
    uint64_t mt=(reqMt>0&&reqMt!=65535)?reqMt:(m+15)/16*16;
    if(mt%16||mt>4095||mt==0)return false;
    if(mt>m)mt=(m+15)/16*16;
    uint64_t kt=(ub-12288)/(3*mt*4)/8*8;
    if(kt>k/8*8)kt=k/8*8;if(kt>2040)kt=2040;
    if(kt<8||(k-kt)*4>0xffffffffULL||(b*m)*4>0xffffffffULL)return false;
    p=TransposePlan{};p.valid=true;p.tilingKey=15;p.dtypeSize=ds;p.numel=g.numel;
    p.rowCount=b;p.rowLen=m;p.tileK=k;p.tileMT=mt;p.tileKT=kt;p.tilePrefetch=disableRead?2:1;
    p.foldExt=c;p.srcBlkPitch=k;p.dstRowStride=b*m;
    p.foldSrcStride=b*c*k;p.foldDstRowStride=c*k;
    const uint64_t units=b*((m+mt-1)/mt)*((k+kt-1)/kt);p.blockDim=units<cores?units:cores;
    p.ubBufBytes=3*mt*kt*4;return true;
}
inline bool BuildSlabPlan(uint64_t ub,uint32_t mode,TransposePlan& p) {
    if(p.tilingKey!=9||p.dtypeSize!=4||p.srcBlkPitch!=p.tileK||ub<=12288)return false;
    uint64_t budget=ub-12288,ks=p.tileKT,mt=p.tileMT;
    uint32_t g=mode==2?2:8;
    while(g>1&&(32+mt+g)*ks*4>budget)g/=2;
    if((32+mt+g)*ks*4>budget)return false;
    p.tilingKey=16;p.rowsPerBatch=g;p.foldLen=16;p.ubBufBytes=(32+mt+g)*ks*4;
    return true;
}

inline bool BuildHalfSlabPlan(uint64_t ub,TransposePlan& p) {
    if(p.tilingKey!=9||p.dtypeSize!=4||p.srcBlkPitch!=p.tileK||ub<=8192)return false;
    const uint64_t ks=p.tileKT,budget=ub-8192;
    if(budget/(ks*4)<=25)return false;
    uint64_t mt=(budget/(ks*4)-25)/16*16;
    if(mt>p.rowLen/16*16)mt=p.rowLen/16*16;
    if(mt>64)mt=64;if(mt<32)return false;
    const uint64_t units=p.rowCount*((p.rowLen+mt-1)/mt);
    p.tilingKey=16;p.foldLen=8;p.rowsPerBatch=1;p.tileMT=mt;p.ubBufBytes=(25+mt)*ks*4;
    if(units<p.blockDim)p.blockDim=units;
    return true;
}
inline bool BuildWholeSuffixPlan(const int64_t* shape,int rank,const int64_t* perm,int permRank,
                                 uint32_t ds,uint64_t ub,uint32_t cores,TransposePlan& p) {
    Geometry g;
    if(!BuildGeometry(shape,rank,perm,permRank,g)||g.collapsedRank!=4||(ds!=2&&ds!=4)||ub<=12288||!cores)return false;
    const int order[4]={2,1,0,3};for(int i=0;i<4;++i)if(g.collapsedPerm[i]!=order[i])return false;
    const uint64_t m=g.collapsedShape[0],b=g.collapsedShape[1],k=g.collapsedShape[2],l=g.collapsedShape[3],rows=m*b;
    const uint64_t slot=(l*ds+31)/32*32,stage=rows*slot,out=(rows*l*ds+31)/32*32,index=(rows*l*4+31)/32*32;
    if(!rows||rows>4095||!k||2*stage+out+index>ub-12288||stage<256||rows*b*slot>0x7fffffffULL||(k-1)*l*ds>0xffffffffULL)return false;
    p=TransposePlan{};p.valid=true;p.tilingKey=17;p.dtypeSize=ds;p.numel=g.numel;p.rowCount=k;p.rowLen=rows*l;
    p.foldExt=m;p.foldLen=b;p.blkLenElems=l;p.rowsPerBatch=rows;p.slotBytes=slot;p.srcBlkPitch=k*l;
    p.chunkElems=stage;p.ubBufBytes=2*stage+out+index;p.blockDim=k<cores?k:cores;return true;
}
}}
#endif
