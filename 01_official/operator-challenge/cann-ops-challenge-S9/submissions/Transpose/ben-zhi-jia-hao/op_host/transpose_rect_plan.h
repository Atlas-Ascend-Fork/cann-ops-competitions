#ifndef TRANSPOSE_RECT_PLAN_H
#define TRANSPOSE_RECT_PLAN_H
#include "transpose_plan.h"
namespace s9 { namespace transpose {
inline bool BuildRectPlan(const int64_t* shape,int rank,const int64_t* perm,int permRank,
                          uint32_t ds,uint64_t ub,uint32_t cores,uint32_t mode,
                          uint32_t reqMt,uint32_t reqKt,TransposePlan& p) {
    Geometry g;
    if(!BuildGeometry(shape,rank,perm,permRank,g)||ds!=4||g.numel<=0||!mode||!cores)return false;
    const int r=g.collapsedRank;
    if(r<2)return false;
    const int mAxis=g.collapsedPerm[r-1];
    if(mAxis==r-1)return false;
    uint64_t si[kMaxRank],so[kMaxRank],outShape[kMaxRank];
    uint64_t stride=1;
    for(int i=r-1;i>=0;--i){si[i]=stride;stride*=g.collapsedShape[i];}
    for(int i=0;i<r;++i)outShape[i]=g.collapsedShape[g.collapsedPerm[i]];
    stride=1;
    for(int i=r-1;i>=0;--i){so[i]=stride;stride*=outShape[i];}
    const uint64_t m=g.collapsedShape[mAxis],k=g.collapsedShape[r-1];
    uint64_t mt=(reqMt>0&&reqMt!=65535)?reqMt:128;
    if(mt>m/16*16)mt=m/16*16;
    if(mt<16||mt%16||mt>4095||ub<=12288)return false;
    const uint64_t budget=ub-12288;
    uint64_t kt=budget/(3*mt*4)/8*8;
    if(kt>k/8*8)kt=k/8*8;
    if(kt>2040)kt=2040;
    if(reqKt>0&&reqKt!=65535){if(reqKt%8||reqKt>kt)return false;kt=reqKt;}
    if(kt<8||(si[mAxis]-kt)*4>0xffffffffULL)return false;
    int q=0;while(g.collapsedPerm[q]!=r-1)++q;
    if((so[q]-mt)*4>0xffffffffULL)return false;
    p=TransposePlan{};p.valid=true;p.tilingKey=14;p.dtypeSize=ds;p.numel=g.numel;
    p.rowLen=m;p.tileK=k;p.tileMT=mt;p.tileKT=kt;p.tilePrefetch=mode;
    p.srcBlkPitch=si[mAxis];p.dstRowStride=so[q];p.rowCount=1;
    for(int i=0;i<r-1;++i)if(i!=q){
        const int a=p.outerRank++;
        p.outerShape[a]=outShape[i];p.outerInStride[a]=si[g.collapsedPerm[i]];p.outerOutStride[a]=so[i];
        p.rowCount*=outShape[i];
    }
    const uint64_t units=p.rowCount*((m+mt-1)/mt)*((k+kt-1)/kt);
    p.blockDim=units<cores?units:cores;p.ubBufBytes=3*mt*kt*4;
    return true;
}
}}
#endif
