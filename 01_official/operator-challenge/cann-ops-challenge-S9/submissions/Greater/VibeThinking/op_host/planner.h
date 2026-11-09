#ifndef S9_GREATER_PLANNER_H
#define S9_GREATER_PLANNER_H

#include <algorithm>
#include <array>
#include <vector>

#include "../op_kernel/greater_tiling.h"

namespace s9g {

struct Options {
    uint32_t depth=2;
    uint32_t coreCap=0;
    bool batch=true;
    bool rawCache=false;
    bool wave=false;
    bool bypass=false;
};

struct Plan {
    GreaterRows data{};
    GreaterPairF32 pair{};
    GreaterLinearRows linear{};
    uint64_t key=EMPTY;
    uint32_t cores=1;
    uint32_t tilingBytes=8;
};

}

namespace s9g {

inline bool CheckedProduct(uint64_t left,uint64_t right,uint64_t& product) {
    if (right && left>uint64_t(INT64_MAX)/right) return false;
    product=left*right;
    return true;
}

inline bool ResolveAxis(int64_t left,int64_t right,int64_t& output) {
    if (left<0 || right<0) return false;
    if (left==right) output=left;
    else if (left==1) output=right;
    else if (right==1) output=left;
    else return false;
    return true;
}

inline bool InferBroadcastShape(const std::vector<int64_t>& left,
                                const std::vector<int64_t>& right,
                                std::vector<int64_t>& output) {
    const size_t rank=std::max(left.size(),right.size());
    if (rank>8) return false;
    output.resize(rank);
    for (size_t axis=0;axis<rank;++axis) {
        const int64_t leftDim=axis<rank-left.size() ? 1:left[axis-(rank-left.size())];
        const int64_t rightDim=axis<rank-right.size() ? 1:right[axis-(rank-right.size())];
        if (!ResolveAxis(leftDim,rightDim,output[axis])) return false;
    }
    const std::vector<int64_t>* shapes[]={&left,&right,&output};
    for (const auto* shape:shapes) {
        if (std::find(shape->begin(),shape->end(),0)!=shape->end()) continue;
        uint64_t elements=1;
        for (int64_t dim:*shape)
            if (!CheckedProduct(elements,static_cast<uint64_t>(dim),elements)) return false;
    }
    return true;
}

// 广播维度使用零步长；相邻且访问模式相同的维度合并，减少地址计算。
struct BroadcastGeometry {
    std::vector<int64_t> output;
    std::array<uint64_t,8> leftStride{};
    std::array<uint64_t,8> rightStride{};
    uint64_t leftElements=1;
    uint64_t rightElements=1;

    bool Build(const std::vector<int64_t>& left,const std::vector<int64_t>& right) {
        if (!InferBroadcastShape(left,right,output)) return false;
        if (std::find(output.begin(),output.end(),0)!=output.end()) return true;
        const size_t rank=output.size();
        for (int32_t axis=static_cast<int32_t>(rank)-1;axis>=0;--axis) {
            const uint64_t leftDim=axis<static_cast<int32_t>(rank-left.size()) ?
                1:static_cast<uint64_t>(left[axis-(rank-left.size())]);
            const uint64_t rightDim=axis<static_cast<int32_t>(rank-right.size()) ?
                1:static_cast<uint64_t>(right[axis-(rank-right.size())]);
            leftStride[axis]=leftDim==1 && output[axis]!=1 ? 0:leftElements;
            rightStride[axis]=rightDim==1 && output[axis]!=1 ? 0:rightElements;
            if (!CheckedProduct(leftElements,leftDim,leftElements) ||
                !CheckedProduct(rightElements,rightDim,rightElements)) return false;
        }
        return true;
    }

    bool Empty() const {
        return std::find(output.begin(),output.end(),0)!=output.end();
    }

    bool Pack(GreaterRows& rows) const {
        rows.total=1;
        for (size_t axis=0;axis<output.size();++axis) {
            const uint64_t extent=static_cast<uint64_t>(output[axis]);
            if (!CheckedProduct(rows.total,extent,rows.total)) return false;
            if (extent==1) continue;
            const bool samePattern=rows.rank &&
                (rows.stride1[rows.rank-1]==0)==(leftStride[axis]==0) &&
                (rows.stride2[rows.rank-1]==0)==(rightStride[axis]==0);
            const bool leftContiguous=samePattern && (leftStride[axis]==0 ||
                rows.stride1[rows.rank-1]==leftStride[axis]*extent);
            const bool rightContiguous=samePattern && (rightStride[axis]==0 ||
                rows.stride2[rows.rank-1]==rightStride[axis]*extent);
            if (samePattern && leftContiguous && rightContiguous) {
                rows.dims[rows.rank-1]*=extent;
                rows.stride1[rows.rank-1]=leftStride[axis];
                rows.stride2[rows.rank-1]=rightStride[axis];
            } else {
                rows.dims[rows.rank]=extent;
                rows.stride1[rows.rank]=leftStride[axis];
                rows.stride2[rows.rank]=rightStride[axis];
                ++rows.rank;
            }
        }
        if (!rows.rank) {
            rows.rank=1;
            rows.dims[0]=1;
            rows.stride1[0]=1;
            rows.stride2[0]=1;
        }
        return true;
    }
};

}

namespace s9g {

struct PanelWork { double control=0,repeats=0,readSegments=0,writeSegments=0; };
inline uint64_t PanelBytes(uint64_t inner,uint32_t pitch,uint32_t batch,bool packed,uint32_t dtype) {
    const uint64_t row=Up(inner,64),panel=uint64_t(batch)*pitch;
    const uint64_t n=std::max(packed ? panel:uint64_t(batch)*64,Up(pitch,128));
    const bool linear=dtype==3 && packed && pitch>inner && batch>=4;
    return row*4+2*std::max(panel,row)*TypeBytes(dtype)+(dtype==1 ? panel*4:0)+
        n*(dtype==3 && !packed ? 4:2)+Up(n/8,32)+panel+256+(linear ? uint64_t(pitch)*4:0);
}
inline PanelWork CountPanelWork(uint64_t inner,uint64_t rows,uint32_t pitch,uint32_t batch,bool packed,uint32_t dtype) {
    PanelWork out;
    if (!inner || !rows || !pitch || !batch) return out;
    const bool converted=dtype==1,integer=dtype==3;

    const bool linear=integer && packed && pitch>inner && batch>=4;
    const uint32_t service=11+(converted ? 2:0);
    const uint64_t nb[]={rows/batch,rows%batch ? 1ULL:0ULL};
    const uint32_t bs[]={batch,static_cast<uint32_t>(rows%batch)};
    const uint64_t np[]={inner/pitch,inner%pitch ? 1ULL:0ULL};
    const uint32_t ws[]={pitch,static_cast<uint32_t>(inner%pitch)};
    for (uint32_t i=0;i<2;++i) for (uint32_t j=0;j<2;++j) {
        if (!nb[i] || !np[j]) continue;
        const uint32_t b=bs[i],w=ws[j],cols=(w+63)/64;
        const double count=double(nb[i])*double(np[j]);
        const bool horizontal=pitch>1984 || b<=cols;
        const bool compact=packed && pitch>inner && b>1;
        const bool whole=pitch==inner && w==pitch;
        const uint32_t groups=horizontal ? b:cols;
        out.readSegments+=count*(linear || whole ? 1:b);
        out.writeSegments+=count*(compact || whole ? 1:b);
        if (converted) out.repeats+=count*((uint64_t(b)*pitch+63)/64);
        if (packed) {
            out.control+=count*((integer ? 4.0:1.0)*b+5+service);
            out.repeats+=count*((integer ? 2.0:1.0)*b*cols+double(b)*pitch/128);
            if (compact) {

                out.control+=count*2;
                out.repeats+=count*(double(b)*((w+127)/128)+(uint64_t(b)*w+127)/128);
            } else out.repeats+=count*double(b)*pitch/128;
            if (linear) { out.control+=count*2*b;out.repeats+=count*double(b)*cols; }
        } else {
            out.control+=count*((integer ? 8.0:6.0)*groups+service);
            out.repeats+=count*((integer ? 2.0:1.0)*b*cols+
                (horizontal ? 2.0*b*((w+127)/128):double(cols)*((b+1)/2+b)));
        }
    }
    if (packed) {

        out.control+=1;out.repeats+=(uint64_t(batch)*pitch+2047)/2048;
    }
    if (linear) {
        uint32_t steps=0;
        for (uint32_t have=8;have<pitch;have*=2) ++steps;
        out.control+=10+2*steps;out.repeats+=(pitch-8+63)/64+steps;
    }
    return out;
}

inline double PanelScore(const PanelWork& w,const PanelWork& reference) {
    return w.control/std::max(1.0,reference.control)+w.repeats/std::max(1.0,reference.repeats)+
        w.readSegments/std::max(1.0,reference.readSegments)+w.writeSegments/std::max(1.0,reference.writeSegments);
}
inline void TryPackedPanel(GreaterRows& t) {
    if (!(t.dtype==0 || t.dtype==1 || t.dtype==3) || !t.panelRows || !t.panelPitch) return;
    const uint64_t rows=std::min(t.dims[t.rank-2],std::max(uint64_t(2),t.perCore/t.inner));
    if (rows<2) return;
    const uint32_t row=static_cast<uint32_t>(Up(t.inner,64));
    const uint32_t plain[]={row<=32640 ? row:1984,std::min(row,1984U)};
    const auto reference=CountPanelWork(t.inner,rows,t.panelPitch,t.panelRows,false,t.dtype);
    double best=PanelScore(reference,reference);
    uint32_t bestPitch=t.panelPitch,bestRows=t.panelRows;bool bestPacked=false;
    for (uint32_t family=0;family<2;++family) for (uint32_t i=0;i<2;++i) {
        if (i && plain[i]==plain[0]) continue;
        const bool packed=family!=0;
        const uint32_t pitch=packed ? static_cast<uint32_t>(Up(plain[i],256)):plain[i];
        if (pitch>(packed ? 32512U:32640U)) continue;
        const uint32_t cap=packed ? static_cast<uint32_t>(std::min<uint64_t>({128,32640/pitch,rows})):
            static_cast<uint32_t>(std::min<uint64_t>(128,rows+rows%2));
        for (uint32_t b=2;b<=cap;++b) {
            if (!packed && b%2) continue;
            if (PanelBytes(t.inner,pitch,b,packed,t.dtype)>t.ubBytes) continue;
            const auto work=CountPanelWork(t.inner,rows,pitch,b,packed,t.dtype);
            const double score=PanelScore(work,reference);
            if (score<best-1e-12) { best=score;bestPitch=pitch;bestRows=b;bestPacked=packed; }
        }
    }
    t.panelPitch=bestPitch;t.panelRows=bestRows;
    t.reserved=(t.reserved&~ROW_PACKED_PREDICATE)|(bestPacked ? ROW_PACKED_PREDICATE:0);
}
}

namespace s9g {
inline void TryCompactResident(GreaterRows& t) {
    if(t.dtype!=3 || t.rank<2 || t.innerBits || t.inner<256 || t.inner>32640 ||
       !(t.reserved&ROW_PACKED_PREDICATE) || t.panelPitch<=t.inner)return;
    const uint32_t axis=t.rank-2;
    if(!((t.stride1[axis]==0 && t.stride2[axis]==t.inner) ||
         (t.stride2[axis]==0 && t.stride1[axis]==t.inner)))return;
    const uint64_t r=std::min(t.dims[axis],std::max(uint64_t(2),t.perCore/t.inner));
    const uint32_t quantum=8/RowGcd(static_cast<uint32_t>(t.inner),8);
    uint32_t best=0;uint64_t bestPanels=UINT64_MAX,bestWork=UINT64_MAX;

    for(uint32_t b=quantum;b<=128 && b<=r;b+=quantum){
        const auto l=MakeCompactRowLayout(static_cast<uint32_t>(t.inner),b,t.dims[axis]);
        if(!l.bytes || l.bytes>t.ubBytes || l.fixes>256)continue;
        const uint64_t panels=(r+b-1)/b,work=panels*l.n;
        if(panels<bestPanels || (panels==bestPanels && work<bestWork)){
            bestPanels=panels;bestWork=work;best=b;
        }
    }
    if(best){
        t.panelRows=best;t.panelPitch=static_cast<uint32_t>(Up(t.inner,64));
        t.reserved=(t.reserved&~ROW_PACKED_PREDICATE)|ROW_COMPACT_RESIDENT;
    }
}
}

namespace s9g {

inline bool ConfigureFp32Pair(uint64_t total,uint64_t perCore,uint64_t ub,
                              uint32_t flags,GreaterPairF32& pair) {
    if (ub<=RESERVE+256 || ub-RESERVE>UINT32_MAX || (flags&FLAG_WAVE)) return false;
    const uint64_t budget=ub-RESERVE;
    const uint64_t maskBudget=(budget/4)/32*32;
    const uint64_t capacity=(budget-maskBudget-256)/8;
    const uint32_t tile=static_cast<uint32_t>(
        std::min<uint64_t>(capacity,255*128)/256*256);
    if (tile<256 || perCore<=tile) return false;
    const uint64_t packets=std::min<uint64_t>(maskBudget/(tile/8),
                                             perCore/tile+(perCore%tile!=0));
    if(!packets)return false;
    const uint32_t maskBytes=static_cast<uint32_t>(packets*(tile/8));
    const PairLayout layout=MakePairLayout(tile,maskBytes);
    if(layout.bytes>budget)return false;
    pair={total,perCore,tile,maskBytes,layout.bytes,flags};
    return true;
}

inline void ConfigureRowPanel(GreaterRows& rows) {
    rows.panelRows=0;
    rows.panelPitch=0;
    const uint32_t type=rows.dtype;
    const uint32_t elementBytes=TypeBytes(type);
    if (!(type==0 || type==1 || type==3) || rows.rank<2 || rows.innerBits ||
        rows.inner<32 || rows.inner>UINT32_MAX/8) return;
    const uint32_t rowAxis=rows.rank-2;
    const bool reusable=(rows.stride1[rowAxis]==0 && rows.stride2[rowAxis]==rows.inner) ||
                        (rows.stride2[rowAxis]==0 && rows.stride1[rowAxis]==rows.inner);
    if (!reusable || rows.dims[rowAxis]<2 ||
        (rows.inner-1)*elementBytes>UINT32_MAX) return;

    const uint32_t paddedRow=static_cast<uint32_t>(Up(rows.inner,64));
    if (uint64_t(paddedRow)*(4+2*elementBytes)>rows.ubBytes) return;
    const uint32_t pitchChoices[]={
        paddedRow<=32640 ? paddedRow:1984,
        std::min(paddedRow,1984U)
    };
    for (uint32_t pitch:pitchChoices) {
        for (uint32_t panelRows=128;panelRows>=2;panelRows-=2) {
            const uint32_t vectorElements=std::max(
                panelRows*64,static_cast<uint32_t>(Up(pitch,128)));
            const uint32_t panelElements=panelRows*pitch;
            const uint32_t rawElements=std::max(panelElements,paddedRow);
            const uint64_t bytes=uint64_t(paddedRow)*4+
                uint64_t(2)*rawElements*elementBytes+
                (type==1 ? uint64_t(panelElements)*4:0)+
                uint64_t(vectorElements)*(type==3 ? 4:2)+
                Up(vectorElements/8,32)+panelElements+256;
            if (bytes<=rows.ubBytes) {
                rows.panelRows=panelRows;
                rows.panelPitch=pitch;
                return;
            }
        }
    }
}

}

namespace s9g {

inline bool SelectSmallTensor(uint32_t type,uint32_t availableCores,uint64_t ub,
                              Plan& plan) {
    GreaterRows& rows=plan.data;
    if (rows.total>uint64_t(availableCores)*SMALL ||
        MakeLayout(type,SMALL,1,0).bytes+RESERVE>ub) return false;

    const uint32_t wanted=static_cast<uint32_t>(std::min<uint64_t>(
        availableCores,std::max<uint64_t>(1,rows.total/512)));
    for (uint32_t cores=wanted;cores>0;--cores) {
        if (rows.total%cores || rows.total/cores>SMALL ||
            (rows.total/cores)%32) continue;
        plan.key=EXACT;
        plan.cores=cores;
        rows.perCore=rows.total/cores;
        plan.tilingBytes=sizeof(GreaterExact);
        if (type==2 && rows.perCore>=128 && rows.perCore<=2048 &&
            (rows.perCore&(rows.perCore-1))==0) {
            plan.key=16;
            for (uint64_t packet=128;packet<rows.perCore;packet*=2) ++plan.key;
        }
        rows.tile=SMALL;
        rows.depth=1;
        rows.ubBytes=MakeLayout(type,SMALL,1,0).bytes;
        return true;
    }

    rows.perCore=Up((rows.total+wanted-1)/wanted,32);
    if (rows.perCore>SMALL) return false;
    plan.key=ONESHOT;
    plan.cores=static_cast<uint32_t>((rows.total+rows.perCore-1)/rows.perCore);
    plan.tilingBytes=sizeof(GreaterOneShot);
    rows.tile=SMALL;
    rows.depth=1;
    rows.ubBytes=MakeLayout(type,SMALL,1,0).bytes;
    return true;
}

inline void AssignGeneralCores(uint32_t type,uint32_t availableCores,Plan& plan) {
    GreaterRows& rows=plan.data;
    const uint32_t elementBytes=TypeBytes(type);
    const uint64_t workUnits=rows.total/(32768/(2*elementBytes+1))+1;
    plan.cores=static_cast<uint32_t>(std::min<uint64_t>(
        availableCores,std::max<uint64_t>(1,workUnits)));
    rows.perCore=Up((rows.total+plan.cores-1)/plan.cores,256);
    plan.cores=static_cast<uint32_t>((rows.total+rows.perCore-1)/rows.perCore);
}

inline void AssignCompactRowCores(Plan& plan) {
    GreaterRows& rows=plan.data;
    constexpr uint32_t LOGICAL_CORE_CAP=40;
    uint32_t wanted=plan.cores;
    if (wanted>LOGICAL_CORE_CAP &&
        rows.perCore>=uint64_t(4)*rows.panelRows*rows.inner)
        wanted=LOGICAL_CORE_CAP;
    const uint64_t totalRows=rows.total/rows.inner;
    const uint32_t outputAlignmentRows=32/RowGcd(static_cast<uint32_t>(rows.inner),32);
    const uint64_t ownedRows=Up((totalRows+wanted-1)/wanted,outputAlignmentRows);
    rows.perCore=ownedRows*rows.inner;
    plan.cores=static_cast<uint32_t>((totalRows+ownedRows-1)/ownedRows);
}

inline bool SelectDirectRepeatedRows(GreaterRows& t) {
    if(t.dtype!=3 || t.rank<2 || t.innerBits || t.inner<256 || t.inner>32640)
        return false;
    const uint32_t axis=t.rank-2;
    if(!((t.stride1[axis]==0 && t.stride2[axis]==t.inner) ||
         (t.stride2[axis]==0 && t.stride1[axis]==t.inner)))return false;
    const uint64_t available=std::min(t.dims[axis],std::max(uint64_t(2),t.perCore/t.inner));

    const uint32_t quantum=1;
    uint32_t selected=0;
    uint64_t selectedCalls=UINT64_MAX,selectedElements=UINT64_MAX;
    const uint32_t limit=static_cast<uint32_t>(std::min<uint64_t>({128,available,32512/t.inner}));
    for(uint32_t rows=quantum;rows<=limit;rows+=quantum){
        const auto layout=MakeCompactRowLayout(static_cast<uint32_t>(t.inner),rows,t.dims[axis]);
        if(!layout.bytes || layout.bytes>t.ubBytes)continue;
        const uint64_t calls=(available+rows-1)/rows,elements=calls*layout.n;
        if(calls<selectedCalls || (calls==selectedCalls && elements<selectedElements)){
            selected=rows;selectedCalls=calls;selectedElements=elements;
        }
    }
    if(!selected)return false;
    t.panelRows=selected;t.panelPitch=static_cast<uint32_t>(Up(t.inner,64));
    t.reserved=(t.reserved&~ROW_PACKED_PREDICATE)|ROW_COMPACT_RESIDENT;
    return true;
}

inline bool SelectBf16RepeatedRows(GreaterRows& t){
    if(t.rank<2 || t.innerBits || t.inner<32 || t.inner>32640)return false;
    const uint32_t axis=t.rank-2;
    if(!((t.stride1[axis]==0 && t.stride2[axis]==t.inner) ||
         (t.stride2[axis]==0 && t.stride1[axis]==t.inner)))return false;
    const uint64_t available=std::min(t.dims[axis],std::max(uint64_t(2),t.perCore/t.inner));

    const uint32_t quantum=1;
    const uint32_t limit=static_cast<uint32_t>(std::min<uint64_t>({128,available,32512/t.inner}));
    uint32_t selected=0;uint64_t bestCalls=UINT64_MAX,bestElements=UINT64_MAX;
    for(uint32_t rows=quantum;rows<=limit;rows+=quantum){
        const auto l=MakeBf16RowLayout(static_cast<uint32_t>(t.inner),rows,t.dims[axis]);
        if(!l.bytes || l.bytes>t.ubBytes)continue;
        const uint64_t calls=(available+rows-1)/rows,elements=calls*l.n;
        if(calls<bestCalls || (calls==bestCalls && elements<bestElements)){
            selected=rows;bestCalls=calls;bestElements=elements;
        }
    }
    t.panelRows=selected;return selected!=0;
}

inline bool BuildPlan(const std::vector<int64_t>& left,
                      const std::vector<int64_t>& right,
                      uint32_t type,uint32_t hardwareCores,uint64_t ub,
                      Plan& plan,const Options& options={}) {
    plan={};
    BroadcastGeometry geometry;
    if (type>4 || !geometry.Build(left,right) || ub<RESERVE+16384) return false;
    if (geometry.Empty()) return true;

    GreaterRows& rows=plan.data;
    rows.dtype=type;
    if (!geometry.Pack(rows)) return false;
    const uint32_t elementBytes=TypeBytes(type);
    if (rows.total>uint64_t(INT64_MAX)/elementBytes) return false;

    uint32_t availableCores=std::max(1U,hardwareCores);
    if (options.coreCap) availableCores=std::min(availableCores,options.coreCap);
    rows.inner=rows.dims[rows.rank-1];
    rows.innerBits=(rows.stride1[rows.rank-1]==0 ? 1U:0U) |
                   (rows.stride2[rows.rank-1]==0 ? 2U:0U);
    rows.flags=(options.wave ? FLAG_WAVE:0) |
               (options.bypass ? FLAG_BYPASS:0) |
               (options.rawCache ? FLAG_RAW_CACHE:0);

    const bool flat=geometry.leftElements==rows.total &&
                    geometry.rightElements==rows.total;
    if (flat && SelectSmallTensor(type,availableCores,ub,plan)) return true;

    AssignGeneralCores(type,availableCores,plan);
    plan.key=flat ? FLAT:
        ((geometry.leftElements==1 || geometry.rightElements==1) ? SCALAR:ROW);
    if (plan.key==FLAT) rows.flags|=FLAG_BYPASS;
    plan.tilingBytes=plan.key==ROW ? sizeof(GreaterRows):sizeof(GreaterFlat);
    rows.scalarSide=geometry.leftElements==1 ? 1:
                    (geometry.rightElements==1 ? 2:0);
    rows.depth=plan.key==ROW ? 1:std::min(2U,std::max(1U,options.depth));
    rows.tile=rows.depth==2 ? 8192:12288;
    while (rows.tile>=128 &&
           MakeLayout(type,rows.tile,rows.depth,
               plan.key==SCALAR ? (type==0 || type==2 ? 128:rows.tile):0)
                   .bytes+RESERVE>ub)
        rows.tile-=128;
    if (rows.tile<128) return false;
    if (rows.perCore<=rows.tile && rows.depth==2) rows.depth=1;

    if (plan.key==ROW) {
        rows.policy=geometry.leftElements==rows.total ? 1:
                    (geometry.rightElements==rows.total ? 2:0);
        rows.pitch=static_cast<uint32_t>(Up(std::min<uint64_t>(rows.inner,rows.tile),32));
        rows.batchRows=1;
        if (options.batch && rows.inner<=rows.tile && rows.rank>1)
            rows.batchRows=static_cast<uint32_t>(std::min<uint64_t>({
                rows.tile/rows.pitch,rows.dims[rows.rank-2],255}));
    }
    if (plan.key==SCALAR)
        rows.cacheElems=type==0 || type==2 ? 128:rows.tile;

    rows.ubBytes=MakeLayout(type,rows.tile,rows.depth,rows.cacheElems,
        plan.key==ROW && options.rawCache,
        plan.key==ROW && (rows.reserved&ROW_STAGE_IN_WORK)).bytes;
    if (plan.key==FLAT && type==4 && ub-RESERVE<=UINT32_MAX)
        rows.ubBytes=static_cast<uint32_t>(ub-RESERVE);
    if (plan.key==FLAT && type==0 &&
        ConfigureFp32Pair(rows.total,rows.perCore,ub,rows.flags,plan.pair)) {
        plan.key=FP32_PAIR;
        plan.tilingBytes=sizeof(GreaterPairF32);
    }
    if (plan.key==ROW && ub-RESERVE<=UINT32_MAX)
        rows.ubBytes=static_cast<uint32_t>(ub-RESERVE);
    if(plan.key==ROW && type==1 && rows.rank>=2){
        const uint64_t group=rows.dims[rows.rank-2];
        const bool linear=(geometry.leftElements==rows.total && geometry.rightElements==rows.total/group) ||
                          (geometry.rightElements==rows.total && geometry.leftElements==rows.total/group);
        if(linear && SelectBf16RepeatedRows(rows)){
            AssignCompactRowCores(plan);
            plan.key=8;plan.tilingBytes=sizeof(GreaterLinearRows);
            plan.linear={rows.total,group,static_cast<uint32_t>(rows.inner),rows.panelRows,
                         rows.ubBytes,rows.stride1[rows.rank-2]==0 ? 1U:2U,rows.policy,0};
            return rows.ubBytes+RESERVE<=ub;
        }
    }
    if (plan.key==ROW) {
        if(!SelectDirectRepeatedRows(rows)) {
            ConfigureRowPanel(rows);
            TryPackedPanel(rows);
            TryCompactResident(rows);
        }
    }

    if (plan.key==ROW && (rows.reserved&ROW_COMPACT_RESIDENT)) {
        const uint64_t repeatRows=rows.dims[rows.rank-2];
        if ((geometry.leftElements==rows.total &&
             geometry.rightElements==rows.total/repeatRows) ||
            (geometry.rightElements==rows.total &&
             geometry.leftElements==rows.total/repeatRows))
            rows.reserved|=ROW_LINEAR_BROADCAST;
        AssignCompactRowCores(plan);
    }
    if (plan.key==ROW && (rows.reserved&ROW_COMPACT_RESIDENT) &&
        (rows.reserved&ROW_LINEAR_BROADCAST))
    {
        plan.key=7;
        plan.tilingBytes=sizeof(GreaterLinearRows);
        const uint32_t axis=rows.rank-2;
        plan.linear={rows.total,rows.dims[axis],static_cast<uint32_t>(rows.inner),
            rows.panelRows,rows.ubBytes,rows.stride1[axis]==0 ? 1U:2U,rows.policy,0};
    }
    return rows.ubBytes+RESERVE<=ub;
}

}

#endif
