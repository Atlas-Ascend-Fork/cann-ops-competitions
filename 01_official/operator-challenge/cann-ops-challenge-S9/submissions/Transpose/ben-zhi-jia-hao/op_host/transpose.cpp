// Transpose (torch.permute) host implementation: shape and dtype inference,
// operator registration, and conversion of planner output into tiling data.

#include "transpose_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "transpose_plan.h"
#include "s9_variant.h"
#include "transpose_clock_probe.h"
#include "transpose_diag.h"
#include "transpose_rect_plan.h"
#include "transpose_suffix_plan.h"
#include "transpose_group_plan.h"
#include "transpose_defaults.h"
#include <cstdio>
#include <cstdlib>

namespace {

// Byte width of every dtype this operator advertises.
inline uint32_t DtypeSizeOf(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            return 1;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
        case ge::DT_UINT32:
            return 4;
        default:
            return 0;
    }
}

constexpr uint32_t kBufferNum = 2;

// Diagnostic version 7's word: the pack gate's conjuncts as BuildPlan
// evaluated them (bits 0-6, set inside the planner) plus the outcome bit and
// the size, completed here.  Read by the delay probe only.
inline uint64_t PackGateWord(const s9::transpose::TransposePlan& p)
{
    uint64_t w = p.packGateWord;
    if (p.tilingKey == s9::transpose::kKeyTilePack ||
        p.tilingKey == s9::transpose::kKeyFoldPack) {
        w |= 4ull;
    }
    uint64_t lg = 0;
    uint64_t n = p.numel;
    while (n > 1ull) {
        n >>= 1ull;
        ++lg;
    }
    if (lg > 31ull) {
        lg = 31ull;
    }
    return w | (lg << 7ull);
}

}  // namespace

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto* inputDesc = context->GetInputDesc(0);
    auto* inputShapePtr = context->GetInputShape(0);
    auto* attrs = context->GetAttrs();
    if (inputDesc == nullptr || inputShapePtr == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t dtypeSize = DtypeSizeOf(inputDesc->GetDataType());
    if (dtypeSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& shape = inputShapePtr->GetStorageShape();
    const int rank = static_cast<int>(shape.GetDimNum());
    if (rank < 1 || rank > s9::transpose::kMaxRank) {
        return ge::GRAPH_FAILED;
    }
    int64_t dims[s9::transpose::kMaxRank] = {};
    for (int i = 0; i < rank; ++i) {
        dims[i] = shape.GetDim(i);
    }

    if (attrs->GetAttrNum() < 1) {
        return ge::GRAPH_FAILED;
    }
    const auto* permVec = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    if (permVec == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int permRank = static_cast<int>(permVec->GetSize());
    if (permRank < 1 || permRank > s9::transpose::kMaxRank) {
        return ge::GRAPH_FAILED;
    }
    const auto* permData = static_cast<const int64_t*>(permVec->GetData());
    if (permData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t perm[s9::transpose::kMaxRank] = {};
    for (int i = 0; i < permRank; ++i) {
        perm[i] = permData[i];
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = platform.GetCoreNum();
    if (coreNum == 0) {
        coreNum = 1;
    }
    // Apply the optional core-budget limit before BuildPlan derives its fields.
    // Changing blockDim after derivation could invalidate dependent tile geometry.
    coreNum = s9::CoreLimit(coreNum);
    if (ubSize == 0) {
        ubSize = 192 * 1024;
    }

    // Pass the per-core work floor into BuildPlan. Runtime environment handling
    // stays outside the planner header.
    const uint64_t floorElems = (s9::Variant() == s9::transpose::kVarCoreFloor)
                                    ? s9::transpose::kFloorElemsPerCore
                                    : 0u;
    const uint32_t wideArm = (s9::Variant() == s9::transpose::kVarTileWide)      ? 1u
                             : (s9::Variant() == s9::transpose::kVarTileWideOff) ? 2u
                                                                                 : 0u;
    const uint32_t viaArm = (s9::Variant() == s9::transpose::kVarTileVia)      ? 1u
                            : (s9::Variant() == s9::transpose::kVarTileViaOff) ? 2u
                                                                               : 0u;
    const uint32_t cubeArm = (s9::Variant() == s9::transpose::kVarTileCube) ? 1u : 0u;
    const uint32_t l2Arm = (s9::Variant() == s9::transpose::kVarTileL2) ? 1u : 0u;

    s9::transpose::TransposePlan plan;
    if (!s9::transpose::BuildPlan(dims, rank, perm, permRank, dtypeSize, ubSize,
                                  coreNum, kBufferNum, plan, true,
                                  s9::transpose::kTileEdgeFloorBlocks, 2, 1, 0, 0, 1, 1,
                                  floorElems,
                                  s9::TileMtOverride(), s9::TileKtOverride(),
                                  wideArm, viaArm, cubeArm, l2Arm) ||
        !plan.valid) {
        return ge::GRAPH_FAILED;
    }

    if(s9::Variant()>=28 && s9::Variant()<=31) {
        s9::transpose::BuildOutputPlan(dims,rank,perm,permRank,dtypeSize,ubSize,coreNum,
                                      s9::Variant()-27,s9::TileKtOverride(),plan);
    }

    if(s9::Variant()>=32 && s9::Variant()<=35) {
        s9::transpose::BuildRectPlan(dims,rank,perm,permRank,dtypeSize,ubSize,coreNum,
                                    s9::Variant()-31,s9::TileMtOverride(),s9::TileKtOverride(),plan);
    }

    if(s9::Variant()>=36 && s9::Variant()<=39) {
        s9::transpose::BuildSuffixPlan(dims,rank,perm,permRank,dtypeSize,ubSize,coreNum,
                                      s9::Variant()%2==1,s9::Variant()>=38,plan);
    }
    if(s9::Variant()==40 && (plan.tilingKey==4 || plan.tilingKey==8))plan.tilePrefetch=101;

    if(s9::Variant()==41 || s9::Variant()==42) {
        s9::transpose::BuildGroupPlan(dims,rank,perm,permRank,dtypeSize,ubSize,coreNum,
                                     s9::Variant()==42,s9::TileMtOverride(),plan);
    }
    if(s9::Variant()==43 || s9::Variant()==44)s9::transpose::BuildSlabPlan(ubSize,s9::Variant()-42,plan);

    if(s9::Variant()==45)s9::transpose::BuildWholeSuffixPlan(dims,rank,perm,permRank,dtypeSize,ubSize,coreNum,plan);
    if(s9::Variant()==46)s9::transpose::BuildHalfSlabPlan(ubSize,plan);

    if(s9::Variant()==0 && s9::TileMtOverride()==0 && s9::TileKtOverride()==0) {
        if(TP_DEFAULT_GROUP && plan.numel*dtypeSize>=1048576)
            s9::transpose::BuildGroupPlan(dims,rank,perm,permRank,dtypeSize,ubSize,coreNum,false,0,plan);
        if(TP_DEFAULT_SUFFIX && plan.tilingKey==4 && dtypeSize==2 &&
           plan.foldDstRowStride>1 && plan.rowLen*dtypeSize>=64 && plan.rowLen*dtypeSize<=256 &&
           plan.numel*dtypeSize>=1048576)
            s9::transpose::BuildSuffixPlan(dims,rank,perm,permRank,dtypeSize,ubSize,coreNum,true,false,plan);
    }

    if(s9::Variant()==0 && s9::TileMtOverride()==0 && s9::TileKtOverride()==0 &&
       plan.tilingKey==8 && dtypeSize==2 && plan.numel>=16777216) {
        s9::transpose::Geometry g;
        if(s9::transpose::BuildGeometry(dims,rank,perm,permRank,g) && g.collapsedRank==4 &&
           g.collapsedPerm[0]==2 && g.collapsedPerm[1]==1 && g.collapsedPerm[2]==0 && g.collapsedPerm[3]==3) {
            const uint64_t m=g.collapsedShape[0],b=g.collapsedShape[1],k=g.collapsedShape[2],l=g.collapsedShape[3];
            const uint64_t mt=m;
            const uint64_t bt=1;
            if(m>=16 && b && k>=32 && l>=32 && l<=256 && b*k*l*dtypeSize<=0xffffffffULL && m*l*dtypeSize<=0xffffffffULL && b*m*l*dtypeSize<=0xffffffffULL) {
                for(uint64_t span=k<32?k:32;span>=4;--span) {
                    const uint64_t rows=mt*bt,n=rows*l,slot=(span*l*dtypeSize+31)/32*32;
                    const uint64_t stage=rows*slot,out=((n*dtypeSize+31)/32*32)*span,index=(n*4+31)/32*32;
                    if(2*stage+out+index+12288<=ubSize) {
                        s9::transpose::TransposePlan q{};q.valid=true;q.tilingKey=17;q.dtypeSize=dtypeSize;q.numel=g.numel;
                        q.foldExt=m;q.foldLen=b;q.rowCount=k;q.blkLenElems=l;q.rowLen=mt;
                        q.tileMT=mt;q.tileKT=span;q.rowsPerBatch=bt;q.slotBytes=slot;q.tilePrefetch=204;
                        q.chunkElems=stage;q.ubBufBytes=2*stage+out+index;
                        const uint64_t units=((m+mt-1)/mt)*((b+bt-1)/bt)*((k+span-1)/span);
                        q.blockDim=units<coreNum?units:coreNum;plan=q;break;
                    }
                }
            }
        }
    }

    if(s9::Variant()==0 && s9::TileMtOverride()==0 && s9::TileKtOverride()==0 &&
       plan.tilingKey==9 && dtypeSize==4 && plan.numel>=33554432 &&
       plan.srcBlkPitch==plan.tileK && plan.tileKT==plan.tileK && plan.tileKT%256==0 && plan.tileKT<=512) {
        const uint64_t rows=ubSize/(plan.tileKT*4),mt=rows>16?(rows-16)/16*16:0;
        const uint64_t bytes=(mt+16)*plan.tileKT*4;
        if(mt>plan.tileMT && mt>=32 && mt<=65520 && bytes<=ubSize && plan.rowLen>=mt &&
           plan.rowCount*((plan.rowLen+mt-1)/mt)>=plan.blockDim) {
            plan.tilingKey=16;plan.tileMT=mt;plan.tilePrefetch=10;plan.foldExt=173;plan.ubBufBytes=bytes;
        }
    }

    // Dense Gather is reserved for large packed four-byte matrices with
    // odd row pitch; aligned pitches cause Gather bank conflicts.
    if(s9::Variant()==0 && s9::TileMtOverride()==0 && s9::TileKtOverride()==0 &&
       plan.tilingKey==9 && dtypeSize==4 && plan.numel>=33554432 &&
       plan.srcBlkPitch==plan.tileK && plan.tileK>=512 && (plan.tileK&1) && plan.rowLen>=64) {
        const uint64_t mt=64,kc=64;
        const uint64_t bytes=(mt*plan.tileK*4+31)/32*32+3*mt*kc*4;
        if(plan.tileK<=0xffffffffULL/(mt*4) && bytes+4096<=ubSize) {
            plan.tilingKey=16;plan.tileMT=mt;plan.tileKT=kc;plan.tilePrefetch=174;
            plan.foldExt=174;plan.ubBufBytes=bytes;plan.rowsPerBatch=1;
        }
    }

    if(s9::Variant()==0 && s9::TileMtOverride()==0 && s9::TileKtOverride()==0 &&
       dtypeSize==4 && plan.numel>=33554432 && (plan.tileK&1) &&
       (plan.tilingKey==9 || plan.tilingKey==3 || (plan.tilingKey==16 && plan.tilePrefetch==174)) &&
       plan.srcBlkPitch==plan.tileK && plan.dstRowStride==plan.rowLen) {
        const uint64_t m=plan.rowLen,k=plan.tileK,mp=(m+63)/64*64,kc=12;
        const uint64_t a=(64*k+2*64*64)*4,b=(2*kc*mp+2*((kc*m+7)/8*8))*4;
        const uint64_t bytes=(a>b?a:b)+64*64*4+kc*(mp/8)*4+((kc*mp+255)/256*8)*4;
        const uint32_t original=plan.blockDim,per=(plan.rowCount+original-1)/original;
        const uint32_t cores=(plan.rowCount+per-1)/per;
        if(m>=64 && m<=768 && m%64>32 && k>=128 && k<=1024 && bytes+2048<=ubSize &&
           static_cast<uint64_t>(cores)*mp*k*4<=96ull*1024*1024) {
            plan.tilingKey=16;plan.tileMT=64;plan.tileKT=kc;plan.tilePrefetch=197;
            plan.foldExt=197;plan.rowsPerBatch=1;plan.ubBufBytes=bytes;plan.blockDim=cores;
        }
    }

    // Group transpose uses the same native layout with longer source bursts.
    if(s9::Variant()==0 && s9::TileMtOverride()==0 && s9::TileKtOverride()==0 && plan.tilingKey==15) {
        uint64_t kt=(ubSize-12288)/(2*plan.tileMT*4)/8*8;
        if(kt>plan.tileK/8*8)kt=plan.tileK/8*8;if(kt>2040)kt=2040;
        plan.tileKT=kt;plan.tilePrefetch=205;plan.ubBufBytes=2*plan.tileMT*kt*4;
    }

    // Copy plan fields through the shared field list to keep host serialization
    // consistent with the kernel layout.
    TransposeTilingData tiling;
    if(std::getenv("TP_PLAN_LOG")) {
        std::fprintf(stderr,"TP_PLAN key=%llu cores=%u mt=%u kt=%u prefetch=%u rows=%llu M=%llu K=%llu padrows=%u ub=%u\n",
                     static_cast<unsigned long long>(plan.tilingKey),plan.blockDim,plan.tileMT,plan.tileKT,plan.tilePrefetch,
                     static_cast<unsigned long long>(plan.rowCount),static_cast<unsigned long long>(plan.rowLen),
                     static_cast<unsigned long long>(plan.tileK),plan.rowsPerBatch,plan.ubBufBytes);
    }
#define S9_SET(type, name) tiling.set_##name(plan.name);
    S9_TRANSPOSE_TILING_SCALARS(S9_SET)
    S9_TRANSPOSE_TILING_ARRAYS(S9_SET)
#undef S9_SET
    tiling.set_usedCores(plan.blockDim);
    {
        // The trailing diagnostic delay word is zero when kDiagVersion is zero.
        // It is separate from the shared planner field list.
#if TP_CLOCK_ENABLE
        tiling.set_diagSpin(transpose_clock::Code(dims,rank,perm,permRank,dtypeSize));
#else
        tiling.set_diagSpin(
            s9::transpose_diag::DiagSpin(dims, rank, perm, permRank, dtypeSize,
                                         plan.tilingKey, plan.blockDim, plan.tileK,
                                         PackGateWord(plan), plan.rowLen,
                                         plan.tilePrefetch));
#endif
    }

    // Every selected key needs a matching TILING_KEY_IS(N) kernel variant.
    // Reject unsupported keys before launch. Keep this range synchronized
    // with the entry-point dispatch.
    if (plan.tilingKey < s9::transpose::kKeyBulk ||
        plan.tilingKey > 17u) {
        return ge::GRAPH_FAILED;
    }

    context->SetTilingKey(plan.tilingKey);
    context->SetBlockDim(plan.blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    if (workspace != nullptr) {
        if(plan.tilingKey==16 && (plan.tilePrefetch==197 || plan.tilePrefetch==198)) {
            workspace[0]=s9::transpose::kSysWorkspaceBytes+static_cast<uint64_t>(plan.blockDim)*((plan.rowLen+63)/64*64)*plan.tileK*dtypeSize;
        } else if (plan.tilingKey == s9::transpose::kKeyTileL2) {
            // TWO packed batch images per launched core (the pipeline
            // alternates them by batch parity): bigK x bigM elems each.
            workspace[0] = static_cast<size_t>(
                s9::transpose::kSysWorkspaceBytes +
                static_cast<uint64_t>(plan.blockDim) * 2ull * plan.tileK *
                    plan.rowLen * dtypeSize);
        } else if (plan.tilingKey == s9::transpose::kKeyTileCube) {
            // Two batch-parity regions per AIC group, each holding one batch
            // of uniform 192-row chunks in padded-NZ layout.
            const uint64_t chunksC =
                (plan.rowLen + plan.tileMT - 1) / plan.tileMT;
            const uint64_t regionBytes = chunksC * plan.tileMT *
                                         plan.tileKT * dtypeSize;
            const uint64_t perGroup =
                (plan.rowCount + plan.blockDim - 1) / plan.blockDim;
            workspace[0] = static_cast<size_t>(
                s9::transpose::kSysWorkspaceBytes +
                perGroup * regionBytes * plan.blockDim);
        } else if (plan.tilingKey == s9::transpose::kKeyTileVia) {
            // One unit-sized scratch per launched core: ceil(J/16) panels of
            // kSnap x 16 elements, plus the runtime's own reserved prefix
            // that GetUserWorkspace skips.
            const uint64_t unitWs = (static_cast<uint64_t>(plan.tileMT) / 16u) *
                                    plan.tileKT * 16u * dtypeSize;
            workspace[0] = static_cast<size_t>(
                s9::transpose::kSysWorkspaceBytes +
                unitWs * plan.blockDim);
        } else {
            workspace[0] = 0;
        }
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    auto* attrs = context->GetAttrs();
    if (xShape == nullptr || yShape == nullptr || attrs == nullptr) {
        return GRAPH_FAILED;
    }
    if (attrs->GetAttrNum() < 1) {
        return GRAPH_FAILED;
    }
    const auto* permVec = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    if (permVec == nullptr) {
        return GRAPH_FAILED;
    }
    const auto* permData = static_cast<const int64_t*>(permVec->GetData());
    if (permData == nullptr) {
        return GRAPH_FAILED;
    }
    const size_t rank = xShape->GetDimNum();
    if (permVec->GetSize() != rank || rank == 0) {
        return GRAPH_FAILED;
    }
    yShape->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        int64_t axis = permData[i];
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
            return GRAPH_FAILED;
        }
        yShape->SetDim(i, xShape->GetDim(static_cast<size_t>(axis)));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Transpose : public OpDef {
public:
    explicit Transpose(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("perm").ListInt();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Transpose);

}  // namespace ops
