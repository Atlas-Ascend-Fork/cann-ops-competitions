#ifndef TRANSPOSE_CLOCK_PROBE_H
#define TRANSPOSE_CLOCK_PROBE_H
#include "transpose_geometry.h"
#ifndef TP_CLOCK_ENABLE
#define TP_CLOCK_ENABLE 0
#endif
#ifndef TP_CLOCK_FIELD
#define TP_CLOCK_FIELD 0
#endif
#ifndef TP_CLOCK_AXIS
#define TP_CLOCK_AXIS 0
#endif
#ifndef TP_CLOCK_SHIFT
#define TP_CLOCK_SHIFT 0
#endif
namespace transpose_clock {
inline uint64_t Log2(uint64_t n) { uint64_t e=0;while(n>1){n>>=1;++e;}return e; }
inline uint64_t Code(const int64_t* dims,uint32_t rank,const int64_t* perm,
                     uint32_t permRank,uint32_t dtypeSize) {
    uint64_t n=1;for(uint32_t i=0;i<rank;++i)n*=static_cast<uint64_t>(dims[i]);
#if TP_CLOCK_FIELD==0
    // Encode rank, permutation, element size, and floor(log2 N) for ranks 1..4.
    // The value 8191 denotes overflow.
    if(rank<1 || rank>4 || permRank!=rank || n==0 || Log2(n)>31)return 8191;
    const uint64_t factorial[5]={1,1,2,6,24};
    uint64_t index=0;
    for(uint32_t r=1;r<rank;++r)index+=factorial[r];
    for(uint32_t i=0;i<rank;++i) {
        uint64_t smaller=0;
        for(uint32_t j=i+1;j<rank;++j)if(perm[j]<perm[i])++smaller;
        index+=smaller*factorial[rank-i-1];
    }
    const uint64_t sizeCode=dtypeSize==1?0:dtypeSize==2?1:dtypeSize==4?2:3;
    if(sizeCode==3)return 8191;
    return (index*3+sizeCode)*32+Log2(n);
#elif TP_CLOCK_FIELD==1
    return (n>>TP_CLOCK_SHIFT)&8191ULL;
#elif TP_CLOCK_FIELD==2
    return TP_CLOCK_AXIS<rank ? (static_cast<uint64_t>(dims[TP_CLOCK_AXIS])>>TP_CLOCK_SHIFT)&8191ULL : 0;
#elif TP_CLOCK_FIELD>=3 && TP_CLOCK_FIELD<=7
    s9::transpose::Geometry g;
    if(!s9::transpose::BuildGeometry(dims,rank,perm,permRank,g))return 8191;
    constexpr uint32_t phase=TP_CLOCK_FIELD-3;
    if(rank>4 && phase==0) {
        // Canonical rank1..5, permutation, size, and log2 N in [15,31].
        // 153 permutations * 3 sizes * 17 exponents fit in 13 bits.
        if(g.collapsedRank<1 || g.collapsedRank>5 || Log2(n)<15 || Log2(n)>31)return 8191;
        const uint64_t fac[6]={1,1,2,6,24,120};
        uint64_t index=0;
        for(int r=1;r<g.collapsedRank;++r)index+=fac[r];
        for(int i=0;i<g.collapsedRank;++i) {
            uint64_t smaller=0;
            for(int j=i+1;j<g.collapsedRank;++j)if(g.collapsedPerm[j]<g.collapsedPerm[i])++smaller;
            index+=smaller*fac[g.collapsedRank-i-1];
        }
        const uint64_t sc=dtypeSize==1?0:dtypeSize==2?1:dtypeSize==4?2:3;
        return sc<3 ? (index*3+sc)*17+Log2(n)-15 : 8191;
    }
    const uint32_t axis=rank>4 ? phase-1 : phase;
    // Observe an axis modulo 8192 with a phase-dependent bias.
    // The observation does not bound the original axis extent.
    const uint64_t d=axis<static_cast<uint32_t>(g.collapsedRank) ? g.collapsedShape[axis] : 0;
    return (d+phase)&8191ULL;
#else
    return 8191;
#endif
}
}
#endif
