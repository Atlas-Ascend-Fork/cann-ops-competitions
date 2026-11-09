#ifndef S9_GREATER_TILING_H
#define S9_GREATER_TILING_H
#include <cstdint>
#include <cstddef>
#ifndef S9G_INLINE
#define S9G_INLINE inline
#endif
struct GreaterExact { uint64_t perCore; };
struct GreaterOneShot { uint64_t total; uint64_t perCore; };
struct GreaterFlat {
    uint64_t total;
    uint64_t perCore;
    uint32_t tile, depth, flags, scalarSide, ubBytes, dtype, policy, reserved;
};

struct GreaterLinearRows {
    uint64_t total,groupRows;
    uint32_t inner,panelRows,ubBytes,side,policy,reserved;
};
static_assert(sizeof(GreaterLinearRows)==40,"Linear row geometry ABI");
struct GreaterRows {
    uint64_t total;
    uint64_t perCore;
    uint32_t tile, depth, flags, scalarSide, ubBytes, dtype, policy, reserved;
    uint64_t inner;
    uint32_t rank, innerBits, cacheSide, batchRows, cacheElems, pitch;
    uint64_t dims[8];
    uint64_t stride1[8];
    uint64_t stride2[8];
    uint32_t panelRows,panelPitch;
};
static_assert(sizeof(GreaterExact) == 8, "FastExact ABI");
static_assert(sizeof(GreaterOneShot) == 16, "OneShot ABI");
static_assert(sizeof(GreaterFlat) == 48, "Flat ABI");
static_assert(sizeof(GreaterRows) == 280, "Row ABI");
static_assert(offsetof(GreaterRows, inner) == sizeof(GreaterFlat), "Flat prefix");
namespace s9g {
constexpr uint32_t RESERVE = 8192;
constexpr uint32_t SMALL = 2048;
constexpr uint32_t FLAG_BYPASS = 1, FLAG_WAVE = 2, FLAG_RAW_CACHE = 4;

constexpr uint32_t ROW_STAGE_IN_WORK = 1;
constexpr uint32_t ROW_PACKED_PREDICATE = 2;
enum Key : uint64_t { EMPTY=0, EXACT=1, ONESHOT=2, FLAT=3, SCALAR=4, ROW=5, FP32_PAIR=6 };
S9G_INLINE uint64_t Up(uint64_t n, uint64_t a) { return (n+a-1)/a*a; }
S9G_INLINE uint32_t TypeBytes(uint32_t d) { return d==4 ? 1 : (d==0 || d==3 ? 4 : 2); }
S9G_INLINE uint32_t CompareBytes(uint32_t d) { return d==0 || d==1 || d==3 ? 4 : 2; }
struct Layout { uint32_t in1,in2,out,mask,work,constants,cache,scalarRaw,bytes; };

S9G_INLINE Layout MakeLayout(uint32_t d, uint32_t tile, uint32_t depth,
                              uint32_t cacheElems, uint32_t rawCache=0, bool rowStageInWork=false) {
    const uint32_t s=TypeBytes(d), c=CompareBytes(d);
    Layout l{};
    l.in2=Up(tile*s,32)*depth;
    l.out=l.in2+Up(tile*s,32)*depth;
    const uint32_t outputTiles=depth==1 && cacheElems && !rawCache && (d==0 || d==3) ? 4:depth;
    l.mask=l.out+Up(tile,32)*outputTiles;
    l.work=l.mask+Up((tile+7)/8,32);
    l.constants=l.work+Up((d==1 ? 8 : (d==3 || d==4 ? 4 : 2))*tile,32);
    l.cache=l.constants+Up(2*tile,32);
    l.scalarRaw=l.cache+Up(cacheElems*(rawCache ? s:c),32);
    l.bytes=l.scalarRaw+(cacheElems ? Up((cacheElems>tile ? cacheElems:tile)*s,32):0);

    if (rowStageInWork && d==3 && cacheElems && !rawCache) {
        l.bytes=l.scalarRaw;
        l.scalarRaw=l.work;
    }
    return l;
}
}

struct GreaterPairF32 {
 uint64_t total,perCore;
 uint32_t macro,scratch,requiredBytes,flags;
};
static_assert(sizeof(GreaterPairF32)==32,"FP32 pair ABI");
static_assert(offsetof(GreaterPairF32,macro)==16,"FP32 pair fields");
namespace s9g {
struct PairLayout {uint32_t a,b,predicate,halfWork,ones,output,bytes;};

S9G_INLINE uint32_t PairEmitCapacity(uint32_t n) {
 const uint32_t extent=2*n<255*128 ? 2*n:255*128;
 return extent/256*256;
}
struct Int8WindowPlan {uint32_t tile=0,maskBytes=0,emit=0,bytes=0;};
S9G_INLINE Int8WindowPlan MakeInt8WindowPlan(uint64_t budget,uint64_t perCore) {
 Int8WindowPlan p{};
 if(budget<4096 || budget>UINT32_MAX || !perCore)return p;
 const uint64_t maskBudget=budget/4/32*32;
 const uint64_t capacity=(budget-maskBudget-256)/6;
 p.tile=static_cast<uint32_t>((capacity<32512 ? capacity:32512)/256*256);
 if(!p.tile)return {};
 const uint64_t needed=perCore/p.tile+(perCore%p.tile!=0);
 const uint64_t available=maskBudget/(p.tile/8);
 const uint64_t pages=needed<available ? needed:available;
 if(!pages)return {};
 p.maskBytes=static_cast<uint32_t>(pages*(p.tile/8));
 p.emit=PairEmitCapacity(p.tile);p.bytes=6*p.tile+p.maskBytes+256;
 if(p.bytes>budget)return {};
 return p;
}
S9G_INLINE PairLayout MakePairLayout(uint32_t n,uint32_t maskBytes) {
 PairLayout l{};
 l.b=4*n;l.halfWork=l.a;l.output=l.b;
 l.predicate=8*n;l.ones=l.predicate+maskBytes;l.bytes=l.ones+256;
 return l;
}
}

namespace s9g {
constexpr uint32_t ROW_COMPACT_RESIDENT=4,ROW_LINEAR_BROADCAST=16;
constexpr uint32_t COMPACT_INPUTS=2,COMPACT_OUTPUT_BANKS=1,COMPACT_OUTPUT_BATCH=8;
constexpr uint32_t INT32_OUTPUT_BANKS=1;
constexpr uint32_t INT32_INPUTS=1;
S9G_INLINE uint32_t RowGcd(uint32_t a,uint32_t b) {
    while(b){const uint32_t r=a%b;a=b;b=r;}return a;
}
struct CompactRowInterval {uint64_t begin=0,end=0;};
S9G_INLINE CompactRowInterval BalancedRowInterval(uint64_t total,uint32_t inner,uint32_t core,uint32_t cores){
    CompactRowInterval span{};if(!inner || !cores || core>=cores)return span;
    const uint64_t rows=total/inner,q=32/RowGcd(inner,32),units=(rows+q-1)/q;
    const uint64_t whole=units/cores,extra=units%cores;

    const uint64_t first=core*whole+uint64_t(core)*extra/cores;
    const uint64_t last=(core+1)*whole+uint64_t(core+1)*extra/cores;
    span.begin=(first*q<rows ? first*q:rows)*inner;
    span.end=(last*q<rows ? last*q:rows)*inner;
    return span;
}
constexpr uint32_t ROW_CYCLIC_MACROS=8,COMPACT_MACRO_PANELS=8;
S9G_INLINE CompactRowInterval MacroRowInterval(uint64_t total,uint32_t inner,uint32_t panelRows,uint64_t job){
    const uint64_t rows=total/inner,step=uint64_t(panelRows)*COMPACT_MACRO_PANELS;
    CompactRowInterval span{};const uint64_t first=job*step;
    if(first>=rows)return span;
    span.begin=first*inner;span.end=(rows-first<step ? rows:first+step)*inner;return span;
}
struct CompactRowLayout {
    uint32_t row=0,logical=0,n=0,extent=0,expanded=0,raw=0,maximum=0,halfWork=0,mask=0,output=0,ones=0,resident=0,bytes=0,fixes=0;
};
S9G_INLINE CompactRowLayout MakeCompactRowLayout(uint32_t inner,uint32_t rows,uint64_t groupRows) {
    CompactRowLayout l{};
    if(inner<256 || inner>32640 || !rows || rows>128 || rows>groupRows ||
       uint64_t(inner)*rows>16256 ||
       !groupRows || groupRows>UINT32_MAX)return l;

    l.logical=inner*rows;l.n=static_cast<uint32_t>(Up(l.logical,128));l.extent=l.n;
    l.row=static_cast<uint32_t>(Up(inner,64));

    l.expanded=0;l.raw=4*l.extent;
    l.maximum=l.raw+4*INT32_INPUTS*l.n;l.halfWork=l.maximum;l.mask=l.maximum+2*l.n;

    l.output=l.maximum;
    l.ones=l.mask+3*static_cast<uint32_t>(Up(l.n/8,32));l.resident=l.ones+256;
    l.bytes=l.resident+4*l.row;
    l.fixes=0;
    return l;
}
S9G_INLINE CompactRowLayout MakeBf16RowLayout(uint32_t inner,uint32_t rows,uint64_t groupRows){
    CompactRowLayout l{};
    if(inner<32 || inner>32640 || !rows || rows>128 || rows>groupRows ||
       uint64_t(inner)*rows>32512 ||
       !groupRows || groupRows>UINT32_MAX)return l;
    l.logical=inner*rows;l.n=static_cast<uint32_t>(Up(l.logical,128));l.extent=l.n;
    l.row=static_cast<uint32_t>(Up(inner,64));
    l.expanded=0;l.raw=4*l.n;l.maximum=l.raw+4*l.n;l.halfWork=l.maximum;
    l.mask=l.maximum+4*l.n;l.output=l.mask+static_cast<uint32_t>(Up(l.n/8,32));
    l.ones=l.output+COMPACT_OUTPUT_BANKS*COMPACT_OUTPUT_BATCH*l.n;l.resident=l.ones+256;
    l.bytes=l.resident+6*l.row;
    return l;
}
}

#endif
