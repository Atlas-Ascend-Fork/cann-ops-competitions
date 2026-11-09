#ifndef TRANSPOSE_SUFFIX_PLAN_H
#define TRANSPOSE_SUFFIX_PLAN_H
#include "transpose_plan.h"
namespace s9 { namespace transpose {
// Preserve complete output rows and always fold the adjacent output axis.
inline bool BuildSuffixPlan(const int64_t* shape,int rank,const int64_t* perm,int permRank,
                            uint32_t ds,uint64_t ub,uint32_t cores,bool packed,bool disableRead,
                            TransposePlan& p) {
    Geometry g;
    if(!BuildGeometry(shape,rank,perm,permRank,g)||g.numel<=0||!cores||ub<=12416)return false;
    const int r=g.collapsedRank;
    if(r<3||g.collapsedPerm[r-1]!=r-1||(packed&&ds!=2&&ds!=4))return false;
    uint64_t si[8],so[8],outShape[8],s=1;
    for(int a=r-1;a>=0;--a){si[a]=s;s*=g.collapsedShape[a];}
    for(int a=0;a<r;++a)outShape[a]=g.collapsedShape[g.collapsedPerm[a]];
    s=1;for(int a=r-1;a>=0;--a){so[a]=s;s*=outShape[a];}
    const uint64_t l=outShape[r-1],f=outShape[r-2],sp=si[g.collapsedPerm[r-2]];
    const uint64_t bytes=l*ds,slot=(bytes+31)/32*32;
    if(!bytes||bytes>0xffffffffULL||sp<l||(sp-l)*ds>0xffffffffULL)return false;
    const uint64_t perRow=packed?2*slot+2*bytes+8*l:2*slot;
    uint64_t rows=(ub-12416)/perRow;
    if(rows>4095)rows=4095;
    if(packed&&rows>8192/l)rows=8192/l;
    if(!rows)return false;
    uint64_t fl=f<rows?f:rows;
    rows=rows/fl*fl;
    if(!rows||rows*slot>0xffffffffULL)return false;
    p=TransposePlan{};p.valid=true;p.tilingKey=packed?8:4;p.dtypeSize=ds;p.numel=g.numel;
    p.rowLen=l;p.rowCount=p.numel/l;p.foldExt=f;p.foldLen=fl;p.foldSrcStride=sp;p.foldDstRowStride=1;
    p.rowsPerBatch=rows;p.chunkBlks=1;p.slotBytes=slot;p.blkPerRow=1;p.blkLenElems=l;
    p.tilePrefetch=disableRead?101:1;p.srcBlkPitch=l;
    for(int a=0;a<r-2;++a){p.outerShape[a]=outShape[a];p.outerInStride[a]=si[g.collapsedPerm[a]];p.outerOutStride[a]=so[a]/l;}
    p.outerRank=r-2;
    const uint64_t units=p.rowCount/f*((f+fl-1)/fl);
    p.blockDim=units<cores?units:cores;p.ubBufBytes=rows*perRow+128;
    return true;
}
}}
#endif
