/*
 * SquareSumV1: y = sum(x * x, axis, keep_dims).
 * The host bridges GE/gert inputs to square_sum_v1_plan.h and implements
 * InferShape, InferDataType, and operator registration.
 */

#include "diag_probe.h"
#include "square_sum_v1_plan.h"
#include "square_sum_v1_tiling.h"

#include <array>
#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
using squaresum::MAX_DIM;

// Normalise the axis attribute into a boolean per-axis mask.  Negative axes are
// wrapped, duplicates collapse.  An empty list means "reduce every axis".
bool NormalizeAxes(const gert::TypedContinuousVector<int64_t>* attr, uint32_t rank,
                   std::array<bool, MAX_DIM>& reduced, uint32_t& reducedCount)
{
    reduced.fill(false);
    reducedCount = 0;
    if (rank > MAX_DIM) {
        return false;
    }
    if (attr == nullptr || attr->GetSize() == 0) {
        for (uint32_t i = 0; i < rank; ++i) {
            reduced[i] = true;
        }
        reducedCount = rank;
        return true;
    }
    for (size_t i = 0; i < attr->GetSize(); ++i) {
        int64_t axis = *(attr->GetData() + i);
        if (axis < 0) {
            axis += static_cast<int64_t>(rank);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
            return false;
        }
        if (!reduced[axis]) {
            reduced[axis] = true;
            ++reducedCount;
        }
    }
    return true;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* storage = context->GetInputShape(0);
    const auto* attrs = context->GetAttrs();
    const auto* desc = context->GetInputDesc(0);
    if (storage == nullptr || attrs == nullptr || desc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& shape = storage->GetStorageShape();
    const uint32_t rank = static_cast<uint32_t>(shape.GetDimNum());
    if (rank > MAX_DIM) {
        return ge::GRAPH_FAILED;
    }

    std::array<bool, MAX_DIM> reduced{};
    uint32_t reducedCount = 0;
    if (!NormalizeAxes(attrs->GetListInt(0), rank, reduced, reducedCount)) {
        return ge::GRAPH_FAILED;
    }
    int64_t dims[MAX_DIM] = {0};
    for (uint32_t i = 0; i < rank; ++i) {
        dims[i] = shape.GetDim(i);
    }

    uint32_t dtypeSize = 4;
    switch (desc->GetDataType()) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            dtypeSize = 2;
            break;
        case ge::DT_FLOAT:
            dtypeSize = 4;
            break;
        default:
            return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        ubSize = 192 * 1024;
    }
    uint32_t hwCores = platform.GetCoreNumAiv();
    if (hwCores == 0) {
        hwCores = platform.GetCoreNum();
    }
    // S9_CORES clamps this DOWNWARD only, and it is applied HERE - to the input
    // MakePlan derives from - not to p.usedCores afterwards. Unset, it is the
    // identity. See op_host/s9_variant.h.
    hwCores = s9::CoreLimit(hwCores);

    squaresum::Plan p =
        squaresum::MakePlan(dims, rank, reduced.data(), dtypeSize, ubSize, hwCores);
    if (!p.ok) {
        return ge::GRAPH_FAILED;
    }

    if (p.tilingKey == squaresum::KEY_ROWS && p.innerLen >= 64 &&
        p.innerLen <= 8192 && (p.innerLen & 15U) != 0) {
        const bool packed = desc->GetDataType() == ge::DT_BF16;
        const uint64_t unit = packed ? 16 : 8;
        const uint64_t blocks = p.innerLen / unit - 1U;
        const uint64_t edges = p.innerLen - blocks*unit;
        const uint64_t usable = ubSize > squaresum::UB_RESERVE ? ubSize-squaresum::UB_RESERVE : 32768;
        for (uint32_t batch = static_cast<uint32_t>(squaresum::PlanMin(64, p.totalOut)); batch; --batch) {
            uint32_t rows = 8;
            while (rows < batch) rows <<= 1;
            const uint64_t count = squaresum::PlanAlignUp(batch*p.innerLen, 16);
            const uint64_t partial = packed ? ((count/unit+63U)/64U)*72U
                                            : squaresum::PlanAlignUp(count/unit,8);
            const uint64_t temp = squaresum::PlanMax(blocks*rows,
                partial + (edges+1)*rows +
                    (packed ? squaresum::PlanAlignUp(edges*rows/2,8) : 0)) + 8;
            const uint64_t table = (blocks+edges)*rows;
            const uint64_t scratch = packed
                ? squaresum::PlanMax(squaresum::PlanMax(blocks,edges)*rows,
                                     squaresum::PlanAlignUp(count/4,8)) : count;
            const uint64_t bytes = count*2*dtypeSize + 4*(scratch+temp+table+rows);
            if (blocks*rows > 8192 || bytes > usable) continue;
            p.tilingKey = 14;
            p.rowBatch = batch;
            p.rowPitch = rows;
            p.ubChunk = static_cast<uint32_t>(count);
            p.accLen = static_cast<uint32_t>(scratch);
            p.tmpLen = static_cast<uint32_t>(temp);
            p.workLen = static_cast<uint32_t>(table);
            p.outStage = rows;
            break;
        }
    }

    if (dtypeSize==2 && (p.tilingKey==squaresum::KEY_COLS_CONTIG ||
        p.tilingKey==squaresum::KEY_COLS_POLLED) && p.runFold>8 &&
        p.innerLen*8ULL<=p.tmpLen && p.innerLen*8ULL<=8192) {
        // Input slabs keep their original alignment. The final reduction is
        // already FP32, so eight rows suffice before the transpose.
        p.runFold=8;
        p.workLen=static_cast<uint32_t>(squaresum::PlanMax(p.innerLen*8ULL,
            squaresum::PlanMax(128,p.redChunks*8ULL)));
    }

    if (desc->GetDataType()==ge::DT_BF16 &&
        (p.tilingKey==squaresum::KEY_COLS_CONTIG || p.tilingKey==squaresum::KEY_COLS_POLLED) &&
        p.runFold>=8 && p.innerLen*p.runFold<=8192) {
        const uint64_t table=p.innerLen*p.runFold;
        const uint32_t half=p.runLanes/2U;
        p.tmpLen=(p.runLanes%32U==0 && half>=table) ? half : p.runLanes;
        const uint64_t usable=ubSize>squaresum::UB_RESERVE ? ubSize-squaresum::UB_RESERVE : 32768;
        const uint64_t fixed=4ULL*(p.accLen+p.tmpLen+p.workLen+p.outStage);
        const uint64_t capacity=(usable-fixed)/(2ULL*dtypeSize);
        p.ubChunk=static_cast<uint32_t>((capacity/p.runLanes)*p.runLanes);
    }

    if (desc->GetDataType()==ge::DT_FLOAT && p.tilingKey==squaresum::KEY_COLS_POLLED &&
        p.nOuterR==1 && p.nOuterK==0 && p.outTiles==1 && p.colChunks==1 &&
        p.rStride[0]==p.innerLen && p.runFold==8 && p.innerLen*p.runFold<=8192 &&
        p.innerLen*p.runFold*2ULL<=p.ubChunk &&
        p.innerLen*p.runFold*2ULL<=p.accLen &&
        8ULL*p.ubChunk+4ULL*(p.accLen+p.outStage+p.innerLen*p.runFold)<=ubSize-squaresum::UB_RESERVE &&
        p.redChunks*squaresum::PlanAlignUp(p.innerLen,8)<=p.ubChunk*2ULL) {
        p.tilingKey=25;
        p.userWsBytes=p.redChunks*squaresum::PlanAlignUp(p.innerLen,8)*4ULL;
    }

    if (desc->GetDataType()==ge::DT_BF16 && p.tilingKey==squaresum::KEY_COLS_POLLED &&
        p.nOuterR==1 && p.nOuterK<=1 && p.colChunks==1 &&
        p.rStride[0]==p.innerLen &&
        (p.nOuterK==0 || p.kStride[0]==p.redPerTile*p.innerLen) &&
        p.runFold==8 && p.innerLen*8ULL<=8192 && p.innerLen*16ULL<=p.tmpLen &&
        p.runLanes%p.tmpLen==0 && p.tmpLen%16U==0 &&
        p.redChunks*squaresum::PlanAlignUp(p.innerLen,16)*squaresum::PlanCeilDiv(p.outTiles,p.usedCores)<=p.ubChunk &&
        squaresum::PlanAlignUp(p.innerLen,16)*squaresum::PlanCeilDiv(p.outTiles,p.usedCores)<=2ULL*p.tmpLen &&
        4ULL*(p.ubChunk+p.accLen+p.tmpLen+p.workLen+squaresum::PlanAlignUp(p.innerLen,16))<=ubSize) {
        p.tilingKey=26;
        p.outStage=static_cast<uint32_t>(squaresum::PlanAlignUp(p.innerLen,16));
        p.userWsBytes=p.outTiles*p.redChunks*p.outStage*4ULL;
    }

    if (desc->GetDataType()==ge::DT_FLOAT16 && p.tilingKey==squaresum::KEY_LAYERED &&
        p.nOuterR==1 && p.nOuterK==1 && p.kStride[0]==p.innerLen &&
        p.rStride[0]==p.outTiles*p.innerLen && p.innerLen>=3 && (p.innerLen&1U) &&
        (p.innerLen-1U)/2U+7U<=64U && p.outTiles>=64ULL*p.usedCores) {
        const uint64_t usable=ubSize>squaresum::UB_RESERVE ? ubSize-squaresum::UB_RESERVE : 32768;
        uint64_t capacity=512;
        for (;capacity>=16;capacity-=2) {
            const uint64_t rf=squaresum::PlanAlignUp(capacity,8),rh=squaresum::PlanAlignUp(capacity,16);
            const uint64_t input=squaresum::PlanAlignUp(capacity*p.innerLen,16);
            const uint64_t pair=squaresum::PlanAlignUp((capacity*p.innerLen+1U)/2U,8);
            const uint64_t stride=squaresum::PlanAlignUp(squaresum::PlanCeilDiv(capacity,16),8);
            const uint64_t bytes=4U*input+4U*pair+16U*rf+4U*rh+64U*stride+32U;
            if (bytes<=usable) break;
        }
        if (capacity>=16) {
            const uint64_t waves=squaresum::PlanCeilDiv(p.outTiles,capacity*p.usedCores);
            const uint64_t batch=squaresum::PlanAlignUp(squaresum::PlanCeilDiv(p.outTiles,waves*p.usedCores),2);
            p.tilingKey=29;
            p.rowBatch=static_cast<uint32_t>(batch);
            p.rowPitch=static_cast<uint32_t>(squaresum::PlanAlignUp(squaresum::PlanCeilDiv(batch,16),8));
            p.ubChunk=static_cast<uint32_t>(squaresum::PlanAlignUp(batch*p.innerLen,16));
            p.accLen=static_cast<uint32_t>(squaresum::PlanAlignUp((batch*p.innerLen+1U)/2U,8));
            p.outStage=static_cast<uint32_t>(squaresum::PlanAlignUp(batch,16));
            p.userWsBytes=0;
        }
    }

    SquareSumV1TilingData tiling;
    tiling.set_innerIsReduce(p.innerIsReduce);
    tiling.set_nOuterK(p.nOuterK);
    tiling.set_nOuterR(p.nOuterR);
    tiling.set_usedCores(p.usedCores);
    tiling.set_redChunks(p.redChunks);
    tiling.set_splitRuns(p.splitRuns);
    tiling.set_rowBatch(p.rowBatch);
    tiling.set_rowPitch(p.rowPitch);
    tiling.set_accLen(p.accLen);
    tiling.set_ubChunk(p.ubChunk);
    tiling.set_workLen(p.workLen);
    tiling.set_outStage(p.outStage);
    tiling.set_runGroup(p.runGroup);
    tiling.set_runLanes(p.runLanes);
    tiling.set_runFold(p.runFold);
    tiling.set_tmpLen(p.tmpLen);
    tiling.set_innerLen(p.innerLen);
    tiling.set_colPiece(p.colPiece);
    tiling.set_colChunks(p.colChunks);
    tiling.set_outTiles(p.outTiles);
    tiling.set_redPerTile(p.redPerTile);
    tiling.set_totalOut(p.totalOut);
    tiling.set_indepUnits(p.indepUnits);
    tiling.set_outAligned(p.outAligned);
    tiling.set_kSize(const_cast<uint64_t*>(p.kSize));
    tiling.set_kStride(const_cast<uint64_t*>(p.kStride));
    tiling.set_rSize(const_cast<uint64_t*>(p.rSize));
    tiling.set_rStride(const_cast<uint64_t*>(p.rStride));
    tiling.set_oneHdr(p.oneHdr);
    {
        // The diagnostic spin word is zero when kDiagVersion is zero;
        // other versions encode diagnostic values through diag_probe.h.
        squaresum_diag::DiagShapeInfo di{};
        di.dtypeIdx = (desc->GetDataType() == ge::DT_FLOAT16)
                          ? 1U
                          : ((desc->GetDataType() == ge::DT_BF16) ? 2U : 3U);
        di.rank = rank;
        const bool* keepPtr = (attrs->GetAttrNum() > 1) ? attrs->GetBool(1) : nullptr;
        di.keepDims = (keepPtr != nullptr && *keepPtr) ? 1U : 0U;
        di.innerIsReduce = p.innerIsReduce;
        di.axisMask = 0U;
        di.totalElems = 1ULL;
        for (uint32_t i = 0; i < rank; ++i) {
            if (reduced[i]) {
                di.axisMask |= (1U << i);
            }
            di.totalElems *= static_cast<uint64_t>(dims[i]);
        }
        di.keptInnerLen = (p.innerIsReduce != 0U) ? 0ULL : p.innerLen;
        di.tilingKey = static_cast<uint32_t>(p.tilingKey);
        di.usedCores = p.usedCores;
        di.runGroupOn = (p.runGroup != 0U) ? 1U : 0U;
        di.runsContig = (p.rStride[0] == p.innerLen) ? 1U : 0U;
        di.nOuterR = p.nOuterR;
        di.nOuterK = p.nOuterK;
        di.outTiles = p.outTiles;
        di.redChunksGt1 = (p.redChunks > 1U) ? 1U : 0U;
        di.indepLtCores = (p.indepUnits < static_cast<uint64_t>(p.usedCores)) ? 1U : 0U;
        di.innerLenRaw = p.innerLen;
        tiling.set_diagSpin(squaresum_diag::DiagSpin(di));
    }

    context->SetTilingKey(p.tilingKey);
    context->SetBlockDim(p.usedCores);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = (p.userWsBytes > 0)
                ? static_cast<size_t>(p.userWsBytes + squaresum::SYS_WORKSPACE)
                : 0U;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* input = context->GetInputShape(0);
    const auto* attrs = context->GetAttrs();
    gert::Shape* output = context->GetOutputShape(0);
    if (input == nullptr || attrs == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    const uint32_t rank = static_cast<uint32_t>(input->GetDimNum());
    std::array<bool, MAX_DIM> reduced{};
    uint32_t reducedCount = 0;
    if (!NormalizeAxes(attrs->GetListInt(0), rank, reduced, reducedCount)) {
        return GRAPH_FAILED;
    }
    const bool* keepPtr = (attrs->GetAttrNum() > 1) ? attrs->GetBool(1) : nullptr;
    const bool keepDims = (keepPtr != nullptr) ? *keepPtr : false;

    if (keepDims) {
        output->SetDimNum(rank);
        for (uint32_t i = 0; i < rank; ++i) {
            output->SetDim(i, reduced[i] ? 1 : input->GetDim(i));
        }
    } else {
        output->SetDimNum(rank - reducedCount);
        uint32_t o = 0;
        for (uint32_t i = 0; i < rank; ++i) {
            if (!reduced[i]) {
                output->SetDim(o++, input->GetDim(i));
            }
        }
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SquareSumV1 : public OpDef {
public:
    explicit SquareSumV1(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND}).AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("axis").ListInt();
        this->Attr("keep_dims").AttrType(OPTIONAL).Bool(false);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SquareSumV1);
}  // namespace ops
