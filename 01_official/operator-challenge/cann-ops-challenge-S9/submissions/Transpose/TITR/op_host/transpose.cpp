#ifndef TRANSPOSE_PLAN_TEST
#include "transpose_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#endif
#include <algorithm>
#include <cstdint>
#include <limits>

// Shape-only planning: no environment overrides, probe markers, or tensor reads.
namespace transpose_plan {
enum class Type { Half, Byte, Word, Float };
enum Key : uint32_t {
    AlignedHalfMatrix = 1, BalancedByteMatrix = 2, ReorderedCube = 3,
    MaskedBlockRuns = 4, PackedCubeMatrix = 5, LinearCopy = 6,
    ContiguousSuffix = 7, PaddedHalfMatrix = 8, PaddedWordMatrix = 9,
    CompactByteMatrix = 10, StridedRows = 11
};
struct Plan {
    uint32_t key = StridedRows, cores = 1, rank = 0, bytes = 0;
    uint64_t total = 1, inner = 1, innerStride = 1, workspace = 0;
    uint32_t shape[5]{1,1,1,1,1}, perm[5]{0,1,2,3,4}, outShape[5]{1,1,1,1,1};
    uint64_t stride[5]{1,1,1,1,1};
    // Matrix: N,H,W,H-groups,H-step. Reordered Cube: A,B,C,DE,pairs,
    // slot-DE,workspace-H,tile-A,packed-output. Block runs: A,B,C,D,tasks.
    // Packed Cube: N,H,W,pairs,workspace-H,tail-producers,tail-H-step.
    uint32_t p[16]{};
};
static uint64_t Up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }
static uint64_t Div(uint64_t n, uint64_t d) { return (n + d - 1) / d; }
static bool Perm(const Plan& t, const uint32_t* order, uint32_t n) {
    if (t.rank != n) return false;
    for (uint32_t i=0;i<n;++i) if (t.perm[i]!=order[i]) return false;
    return true;
}
// Recognize [prefix, left contiguous axes, right contiguous axes] ->
// [prefix, right, left]. Singleton axes are removed before recognition.
static bool Matrix(const Plan& t, uint32_t& n, uint32_t& h, uint32_t& w) {
    uint32_t s[5]{}, order[5]{}, map[5]{}, rank=0, out=0;
    for (uint32_t i=0;i<t.rank;++i) if (t.shape[i]!=1) { map[i]=rank; s[rank++]=t.shape[i]; }
    for (uint32_t i=0;i<t.rank;++i) if (t.shape[t.perm[i]]!=1) order[out++]=map[t.perm[i]];
    for (uint32_t prefix=0;prefix+1<rank;++prefix) {
        for (uint32_t split=prefix+1;split<rank;++split) {
            uint32_t k=0; bool ok=true;
            for (uint32_t i=0;i<prefix;++i) ok &= order[k++]==i;
            for (uint32_t i=split;i<rank;++i) ok &= order[k++]==i;
            for (uint32_t i=prefix;i<split;++i) ok &= order[k++]==i;
            if (!ok) continue;
            uint64_t nn=1,hh=1,ww=1;
            for(uint32_t i=0;i<prefix;++i) nn*=s[i];
            for(uint32_t i=prefix;i<split;++i) hh*=s[i];
            for(uint32_t i=split;i<rank;++i) ww*=s[i];
            if(nn>UINT32_MAX || hh>UINT32_MAX || ww>UINT32_MAX) return false;
            n=nn;h=hh;w=ww;return true;
        }
    }
    return false;
}
static bool Build(Plan& t, uint32_t rank, const int64_t* shape,
    const int64_t* perm, Type type, uint32_t vectors, uint32_t cubes,
    uint64_t ub=192*1024ULL, uint64_t l1=512*1024ULL) {
    if(rank>5 || !shape || !perm || vectors==0) return false;
    t=Plan{}; t.rank=rank; t.bytes=type==Type::Byte?1:type==Type::Half?2:4;
    bool seen[5]{};
    for(uint32_t i=0;i<rank;++i) {
        int64_t axis=perm[i]<0?perm[i]+rank:perm[i];
        if(shape[i]<0 || uint64_t(shape[i])>UINT32_MAX || axis<0 || axis>=rank || seen[axis]) return false;
        seen[axis]=true; t.shape[i]=shape[i];t.perm[i]=axis;
    }
    uint64_t product=1;
    for(uint32_t r=rank;r>0;--r) {
        uint32_t i=r-1;t.stride[i]=product;
        if(t.shape[i] && product>uint64_t(INT64_MAX)/t.bytes/t.shape[i]) return false;
        product*=t.shape[i];
    }
    t.total=product;
    for(uint32_t i=0;i<rank;++i)t.outShape[i]=t.shape[t.perm[i]];
    t.cores=std::min(40U,vectors);
    if(!product) {t.key=LinearCopy;t.cores=1;return true;}
    // Maximal contiguous suffix in the *output* traversal, ignoring unit axes.
    for(uint32_t r=rank;r>0;--r) {
        uint32_t i=r-1,axis=t.perm[i];
        if(t.shape[axis]==1) continue;
        if(t.stride[axis]!=t.inner) break;
        t.inner*=t.shape[axis];
    }
    if(t.inner==t.total) {t.key=LinearCopy;t.cores=std::min<uint64_t>(t.cores,Div(t.total*t.bytes,32768));return true;}
    uint32_t n=0,h=0,w=0;
    const bool matrix=product<=UINT32_MAX && Matrix(t,n,h,w) && uint64_t(std::max(h,w))*t.bytes<=UINT32_MAX;
    if(matrix) {t.p[0]=n;t.p[1]=h;t.p[2]=w;}
    if(matrix && type==Type::Half && h%16==0 && w%64==0 &&
       h/16-1<=65535 && w/16-4<=65535) {
        t.key=AlignedHalfMatrix;t.cores=std::min<uint64_t>(t.cores,uint64_t(n)*(h/16)*(w/64));return true;
    }
    if(matrix && type==Type::Byte && h>=112 && w>128) {
        t.key=BalancedByteMatrix;t.p[3]=Div(h,112);t.p[4]=Div(h,t.p[3]);
        t.cores=std::min<uint64_t>(t.cores,uint64_t(n)*t.p[3]*Div(w,128));return true;
    }
    const uint32_t bacOrder[5]{3,4,1,0,2};
    const uint32_t pairs=std::min({20U,cubes,vectors/2});
    if(type==Type::Word && product<=UINT32_MAX && pairs && Perm(t,bacOrder,5)) {
        uint64_t a=t.shape[0],b=t.shape[1],c=t.shape[2],de=uint64_t(t.shape[3])*t.shape[4],s=a*b*c;
        uint64_t slot=Up(Div(de,pairs),8),pitch=Up(s,16),last=de-(pairs-1)*std::min(de,slot);
        // Every producer and both readers must have >=2 local tiles. Bound
        // descriptor fields and physical UB/L1 spans before accepting.
        if(de>(pairs-1)*slot && de<=65535 && slot/8<=4095 && s>=16 && s<=4095 &&
           last>slot/32*16+16 && slot/32*16>=32 &&
           (2*s+pitch)*16*4<=ub) {
            for(uint32_t tile=std::min<uint64_t>(6,a);tile;--tile) {
                if(a%tile || a*b/tile<2 || 2*(slot/8)*(tile*c)*8*4>l1 || tile*c>65535)continue;
                t.key=ReorderedCube;t.cores=pairs;
                t.p[0]=a;t.p[1]=b;t.p[2]=c;t.p[3]=de;t.p[4]=pairs;t.p[5]=slot;t.p[6]=pitch;t.p[7]=tile;
                // Trade one input slot for one packed output. Reserve a full
                // 64-word source-fetch guard after the padded transpose.
                t.p[8]=(pitch/8<=255 && (2*s+pitch)*16*4+64*4<=ub)?1:0;
                t.workspace=uint64_t(pairs)*(slot/8)*pitch*8*4;return true;
            }
        }
    }
    const uint32_t blockOrder[4]{2,1,0,3};
    if(type==Type::Half && product<=UINT32_MAX && Perm(t,blockOrder,4)) {
        uint64_t a=t.shape[0],b=t.shape[1],c=t.shape[2],d=t.shape[3],tasks=t.cores*14;
        if(a<=255 && d>=64 && d<=113 && b<=tasks && b*c*d*2<=UINT32_MAX && (b-1)*a*d*2<=UINT32_MAX) {
            for(;tasks<=uint64_t(t.cores)*256 && tasks/b+1<=c;tasks+=t.cores) {
                uint64_t maxC=Div(c,tasks/b),in=Up(maxC*d,16),out=Up(a*d,16);
                if(in/16>255 || maxC>4095 || 2*(a*in*2+256)+maxC*out*2+512>ub)continue;
                t.key=MaskedBlockRuns;t.p[0]=a;t.p[1]=b;t.p[2]=c;t.p[3]=d;t.p[4]=tasks;return true;
            }
        }
    }
    if(matrix && type==Type::Float && pairs && h>352 && w>=64) {
        uint64_t pitch=Up(h,16),c1=Div(w,8),pattern=Div(pitch,256)*8;
        if(pitch/8<=255 && w<=65535 && 2*c1*112*8*4<=l1 &&
           (2*16*352+3*16*pitch+pattern)*4<=ub) {
            t.key=PackedCubeMatrix;t.cores=pairs;t.p[3]=pairs;t.p[4]=pitch;
            // Preserve existing key5 eligibility; only change consumer geometry
            // when two C8 full-height inputs and two C16 packed outputs fit.
            const uint64_t split8Bytes=(2*8*768+8*pitch+2*16*pitch+pattern)*4;
            // Conservatively require whole 64-word vector fetch rows.
            t.p[9]=h<=768 && pitch%64==0 && split8Bytes<=ub;
            t.p[10]=t.p[9] && n>2ULL*pairs;
            const uint64_t transposeH=Up(h,16);
            t.p[11]=t.p[10] && h%16!=0 && transposeH<=768 && transposeH<=pitch;
            // Rebalance an incomplete final wave only when all producer slices
            // fit two existing L1 buffers and all readers own >=2 C tiles.
            uint32_t tail=n%pairs;
            if(tail && pairs%tail==0) {
                uint32_t producers=pairs/tail,step=Up(Div(h,producers),8),readers=2*producers;
                if(producers>1 && uint64_t(producers-1)*step<h && step<=224 &&
                   h-(producers-1)*step>112 && w/readers/16*16>=32) {
                    t.p[5]=producers;t.p[6]=step;
                }
            }
            t.workspace=uint64_t(pairs)*c1*pitch*8*4;return true;
        }
    }
    if(t.inner>1) {t.key=ContiguousSuffix;t.cores=std::min<uint64_t>(t.cores,Div(t.total, std::min<uint64_t>(t.inner,32768/t.bytes)));return true;}
    if(matrix && type==Type::Half) {t.key=PaddedHalfMatrix;return true;}
    if(matrix && (type==Type::Float || type==Type::Word)) {t.key=PaddedWordMatrix;return true;}
    if(matrix && type==Type::Byte) {t.key=CompactByteMatrix;return true;}
    // DMA gathers a strided output row into 32-byte UB slots, then DMA
    // compacts valid bytes. Scalar handles addresses only, never payload.
    uint32_t last=rank-1;
    while(last && t.outShape[last]==1)--last;
    t.inner=t.outShape[last];t.innerStride=t.stride[t.perm[last]];
    t.cores=std::min<uint64_t>(t.cores,Div(t.total,std::min<uint64_t>(t.inner,256)));
    return true;
}
}

#ifndef TRANSPOSE_PLAN_TEST
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto* desc=context->GetInputDesc(0);auto* storage=context->GetInputShape(0);
    auto* attrs=context->GetAttrs();if(!desc || !storage || !attrs)return ge::GRAPH_FAILED;
    auto* order=attrs->GetListInt(0);const auto& shape=storage->GetStorageShape();
    uint32_t rank=shape.GetDimNum();if(!order || rank>5 || order->GetSize()!=rank)return ge::GRAPH_FAILED;
    int64_t sizes[5]{},axes[5]{};for(uint32_t i=0;i<rank;++i){sizes[i]=shape.GetDim(i);axes[i]=order->GetData()[i];}
    transpose_plan::Type type;
    switch(desc->GetDataType()) {
        case ge::DT_FLOAT16:type=transpose_plan::Type::Half;break;
        case ge::DT_INT8:type=transpose_plan::Type::Byte;break;
        case ge::DT_INT32:type=transpose_plan::Type::Word;break;
        case ge::DT_FLOAT:type=transpose_plan::Type::Float;break;
        default:return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint64_t ub=0,l1=0;platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
    transpose_plan::Plan p;
    if(!transpose_plan::Build(p,rank,sizes,axes,type,platform.GetCoreNumAiv(),platform.GetCoreNumAic(),ub,l1))return ge::GRAPH_FAILED;
    TransposeTilingData data;data.set_rank(p.rank);data.set_total(p.total);data.set_inner(p.inner);data.set_innerStride(p.innerStride);
    data.set_shape(p.shape);data.set_perm(p.perm);data.set_outShape(p.outShape);data.set_stride(p.stride);data.set_p(p.p);
    context->SetTilingKey(p.key);context->SetBlockDim(p.cores);
    data.SaveToBuffer(context->GetRawTilingData()->GetData(),context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    context->GetWorkspaceSizes(1)[0]=p.workspace ? p.workspace+platform.GetLibApiWorkSpaceSize():0;
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    auto* x=context->GetInputShape(0);auto* y=context->GetOutputShape(0);auto* attrs=context->GetAttrs();
    if(!x || !y || !attrs)return GRAPH_FAILED;
    auto* order=attrs->GetListInt(0);const uint32_t rank=x->GetDimNum();
    if(!order || rank>5 || order->GetSize()!=rank)return GRAPH_FAILED;
    bool seen[5]{};y->SetDimNum(0);
    for(uint32_t i=0;i<rank;++i) {int64_t a=order->GetData()[i];if(a<0)a+=rank;
        if(a<0 || a>=rank || seen[a])return GRAPH_FAILED;seen[a]=true;y->AppendDim(x->GetDim(a));}
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* c) {c->SetOutputDataType(0,c->GetInputDataType(0));return GRAPH_SUCCESS;}
}
namespace ops {
class Transpose: public OpDef {
public:explicit Transpose(const char* name):OpDef(name) {
    this->Input("inputs").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_INT32,ge::DT_INT8})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
    this->Output("output").ParamType(REQUIRED).DataType({ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_INT32,ge::DT_INT8})
        .Format({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND,ge::FORMAT_ND});
    this->Attr("dims").ListInt();this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
}};
OP_ADD(Transpose);
}
#endif
