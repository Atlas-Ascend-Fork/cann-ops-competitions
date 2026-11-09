#include "greater_tiling.h"
#include <cstdint>
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kDimNum = 5U;
constexpr uint32_t kAivCoreNum910B = 40U;
constexpr uint32_t kWorkBlockBytes = 1024U;
constexpr uint32_t kUbBytes910B = 192U * 1024U;
constexpr uint32_t kUbReservedBytes = 8U * 1024U;
constexpr uint32_t kCScalarCacheBytes = 1024U;

static uint32_t DataTypeBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT8:
        case ge::DT_UINT8:
        case ge::DT_BOOL:
            return 1U;
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
        case ge::DT_INT16:
            return 2U;
        case ge::DT_FLOAT:
        case ge::DT_INT32:
            return 4U;
        default:
            return 0U;
    }
}

static uint32_t CeilDivU32(uint32_t x, uint32_t y)
{
    return x / y + (x % y == 0U ? 0U : 1U);
}

static uint32_t AlignDownU32(uint32_t x, uint32_t alignment)
{
    return (x / alignment) * alignment;
}

static bool ReadExtent(const gert::Shape& shape, int64_t index, uint32_t& extent)
{
    if (index < 0) {
        extent = 1U;
        return true;
    }
    const int64_t value = shape.GetDim(index);
    if (value < 0 || static_cast<uint64_t>(value) > UINT32_MAX) {
        return false;
    }
    extent = static_cast<uint32_t>(value);
    return true;
}

static bool BroadcastExtent(uint32_t lhs, uint32_t rhs, uint32_t& out)
{
    if (lhs == rhs) {
        out = lhs;
        return true;
    }
    if (lhs == 1U) {
        out = rhs;
        return true;
    }
    if (rhs == 1U) {
        out = lhs;
        return true;
    }
    return false;
}

struct CanonicalGroup {
    uint32_t lhs;
    uint32_t rhs;
    uint32_t out;
    uint32_t pattern;
};

static bool MultiplyExtent(uint32_t lhs, uint32_t rhs, uint32_t& result)
{
    const uint64_t product = static_cast<uint64_t>(lhs) * rhs;
    if (product > UINT32_MAX) {
        return false;
    }
    result = static_cast<uint32_t>(product);
    return true;
}

static bool CanonicalizeShapes(const gert::Shape& lhsShape, const gert::Shape& rhsShape,
                               uint32_t (&lhsDims)[kDimNum], uint32_t (&rhsDims)[kDimNum],
                               uint32_t (&outDims)[kDimNum])
{
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        lhsDims[i] = 1U;
        rhsDims[i] = 1U;
        outDims[i] = 1U;
    }

    const int64_t lhsRank = static_cast<int64_t>(lhsShape.GetDimNum());
    const int64_t rhsRank = static_cast<int64_t>(rhsShape.GetDimNum());
    const int64_t outRank = lhsRank > rhsRank ? lhsRank : rhsRank;
    if (outRank <= static_cast<int64_t>(kDimNum)) {
        for (int64_t i = 0; i < outRank; ++i) {
            uint32_t lhs = 1U;
            uint32_t rhs = 1U;
            uint32_t out = 1U;
            if (!ReadExtent(lhsShape, lhsRank - outRank + i, lhs) ||
                !ReadExtent(rhsShape, rhsRank - outRank + i, rhs) ||
                !BroadcastExtent(lhs, rhs, out)) {
                return false;
            }
            const uint32_t dst = kDimNum - static_cast<uint32_t>(outRank) + static_cast<uint32_t>(i);
            lhsDims[dst] = lhs;
            rhsDims[dst] = rhs;
            outDims[dst] = out;
        }
        return true;
    }

    CanonicalGroup groups[kDimNum];
    uint32_t groupCount = 0U;
    for (int64_t i = 0; i < outRank; ++i) {
        uint32_t lhs = 1U;
        uint32_t rhs = 1U;
        uint32_t out = 1U;
        if (!ReadExtent(lhsShape, lhsRank - outRank + i, lhs) ||
            !ReadExtent(rhsShape, rhsRank - outRank + i, rhs) ||
            !BroadcastExtent(lhs, rhs, out)) {
            return false;
        }
        // Dimensions of extent one are neutral. Every non-neutral dimension
        // in a merged group must have the same dense/broadcast pattern for
        // both inputs; otherwise one 5-D stride cannot represent the mapping.
        uint32_t pattern = 0U;
        if (out != 1U) {
            pattern = (lhs == out ? 1U : 0U) | (rhs == out ? 2U : 0U);
        }
        if (groupCount != 0U &&
            (groups[groupCount - 1U].pattern == 0U || pattern == 0U ||
             groups[groupCount - 1U].pattern == pattern)) {
            CanonicalGroup& group = groups[groupCount - 1U];
            if (!MultiplyExtent(group.lhs, lhs, group.lhs) ||
                !MultiplyExtent(group.rhs, rhs, group.rhs) ||
                !MultiplyExtent(group.out, out, group.out)) {
                return false;
            }
            if (group.pattern == 0U) {
                group.pattern = pattern;
            }
        } else {
            if (groupCount == kDimNum) {
                return false;
            }
            groups[groupCount++] = CanonicalGroup{lhs, rhs, out, pattern};
        }
    }
    const uint32_t base = kDimNum - groupCount;
    for (uint32_t i = 0U; i < groupCount; ++i) {
        lhsDims[base + i] = groups[i].lhs;
        rhsDims[base + i] = groups[i].rhs;
        outDims[base + i] = groups[i].out;
    }
    return true;
}

static bool CalcStrides(const uint32_t (&dims)[kDimNum], uint32_t (&strides)[kDimNum],
                        bool broadcastZero)
{
    uint64_t stride = 1ULL;
    for (int32_t i = static_cast<int32_t>(kDimNum) - 1; i >= 0; --i) {
        if (stride > UINT32_MAX) {
            return false;
        }
        strides[i] = (broadcastZero && dims[i] == 1U) ? 0U : static_cast<uint32_t>(stride);
        stride *= dims[i];
    }
    return stride <= UINT32_MAX;
}

static bool CalcTotal(const uint32_t (&dims)[kDimNum], uint32_t& result)
{
    uint64_t total = 1ULL;
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        total *= dims[i];
        if (total > UINT32_MAX) {
            return false;
        }
    }
    result = static_cast<uint32_t>(total);
    return true;
}

static uint32_t CalcTileElems(ge::DataType dtype, uint32_t elemBytes, uint32_t blockElems,
                              bool cScalarOther, bool cRowOther)
{
    uint32_t usableBytes = kUbBytes910B - kUbReservedBytes;
    uint32_t bytesPerElem = (cScalarOther ? elemBytes : 2U * elemBytes) + 2U;
    switch (dtype) {
        case ge::DT_FLOAT:
            bytesPerElem += 4U;  // two half buffers for bool materialization.
            break;
        case ge::DT_FLOAT16:
            if (cScalarOther || cRowOther) {
                bytesPerElem += 2U;  // one half zero buffer; self queue is reused for one.
            }
            break;  // half path reuses input queues for 0/1 materialization.
        case ge::DT_BF16:
            bytesPerElem += 8U;  // two float cast buffers.
            break;
        case ge::DT_INT32:
            bytesPerElem += 5U;  // int32 max buffer and second byte mask.
            break;
        case ge::DT_INT8:
            bytesPerElem += 4U;  // two half cast buffers.
            break;
        default:
            return blockElems;
    }
    uint32_t tileBytes = (usableBytes / bytesPerElem) * elemBytes;
    tileBytes = AlignDownU32(tileBytes, kWorkBlockBytes);
    if (tileBytes < kWorkBlockBytes) {
        tileBytes = kWorkBlockBytes;
    }
    uint32_t tileElems = tileBytes / elemBytes;
    tileElems = AlignDownU32(tileElems, blockElems);
    return tileElems == 0U ? blockElems : tileElems;
}

static bool IsRank3CScalarOther(ge::DataType dtype, const uint32_t (&selfDims)[kDimNum],
                                const uint32_t (&otherDims)[kDimNum],
                                const uint32_t (&outDims)[kDimNum])
{
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return false;
    }
    if (outDims[0] != 1U || outDims[1] != 1U || outDims[4] < 64U ||
        static_cast<uint64_t>(outDims[3]) * DataTypeBytes(dtype) > kCScalarCacheBytes) {
        return false;
    }
    return selfDims[0] == outDims[0] && selfDims[1] == outDims[1] &&
           selfDims[2] == outDims[2] && selfDims[3] == outDims[3] &&
           selfDims[4] == outDims[4] &&
           otherDims[0] == 1U && otherDims[1] == 1U && otherDims[2] == 1U &&
           otherDims[3] == outDims[3] && otherDims[4] == 1U;
}

static bool IsRank4CRowOther(ge::DataType dtype, const uint32_t (&selfDims)[kDimNum],
                             const uint32_t (&otherDims)[kDimNum],
                             const uint32_t (&outDims)[kDimNum])
{
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return false;
    }
    if (outDims[0] != 1U || outDims[3] < 2U || outDims[4] < 2U) {
        return false;
    }
    return selfDims[0] == outDims[0] && selfDims[1] == outDims[1] &&
           selfDims[2] == outDims[2] && selfDims[3] == outDims[3] &&
           selfDims[4] == outDims[4] &&
           otherDims[0] == 1U && otherDims[1] == 1U &&
           otherDims[2] == outDims[2] && otherDims[3] == 1U &&
           otherDims[4] == outDims[4];
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto& selfShape = context->GetInputShape(0)->GetStorageShape();
    const auto& otherShape = context->GetInputShape(1)->GetStorageShape();

    uint32_t selfDims[kDimNum];
    uint32_t otherDims[kDimNum];
    uint32_t outDims[kDimNum];
    uint32_t selfStrides[kDimNum];
    uint32_t otherStrides[kDimNum];
    uint32_t outStrides[kDimNum];

    if (!CanonicalizeShapes(selfShape, otherShape, selfDims, otherDims, outDims) ||
        !CalcStrides(selfDims, selfStrides, true) ||
        !CalcStrides(otherDims, otherStrides, true) ||
        !CalcStrides(outDims, outStrides, false)) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
    const uint32_t elemBytes = DataTypeBytes(dtype);
    if (elemBytes == 0U) {
        return ge::GRAPH_FAILED;
    }

    uint32_t totalElems = 0U;
    if (!CalcTotal(outDims, totalElems)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t blockElems = kWorkBlockBytes / elemBytes;
    if (blockElems == 0U) {
        blockElems = 1U;
    }
    uint32_t totalBlocks = CeilDivU32(totalElems, blockElems);
    const bool cScalarOther = IsRank3CScalarOther(dtype, selfDims, otherDims, outDims);
    const bool cRowOther = IsRank4CRowOther(dtype, selfDims, otherDims, outDims);
    uint32_t tileElems = CalcTileElems(dtype, elemBytes, blockElems, cScalarOther, cRowOther);

    bool sameShape = true;
    bool selfDense = true;
    bool otherDense = true;
    for (uint32_t i = 0U; i < kDimNum; ++i) {
        sameShape = sameShape && selfDims[i] == outDims[i] && otherDims[i] == outDims[i];
        selfDense = selfDense && selfDims[i] == outDims[i];
        otherDense = otherDense && otherDims[i] == outDims[i];
    }

    GreaterTilingData tiling;
    tiling.set_total_size(totalElems);
    tiling.set_block_elems(blockElems);
    tiling.set_total_blocks(totalBlocks);
    tiling.set_tile_elems(tileElems);
    tiling.set_elem_bytes(elemBytes);
    tiling.set_same_shape(sameShape ? 1U : 0U);
    tiling.set_self_dense(selfDense ? 1U : 0U);
    tiling.set_other_dense(otherDense ? 1U : 0U);
    tiling.set_out_dims(outDims);
    tiling.set_self_dims(selfDims);
    tiling.set_other_dims(otherDims);
    tiling.set_out_strides(outStrides);
    tiling.set_self_strides(selfStrides);
    tiling.set_other_strides(otherStrides);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    const uint32_t blockDim = totalBlocks == 0U ? 1U :
        (totalBlocks < kAivCoreNum910B ? totalBlocks : kAivCoreNum910B);
    context->SetBlockDim(blockDim);
    switch (dtype) {
        case ge::DT_FLOAT:
            context->SetTilingKey(0U);
            break;
        case ge::DT_FLOAT16:
            context->SetTilingKey(1U);
            break;
        case ge::DT_BF16:
            context->SetTilingKey(2U);
            break;
        case ge::DT_INT32:
            context->SetTilingKey(3U);
            break;
        case ge::DT_INT8:
            context->SetTilingKey(4U);
            break;
        default:
            break;
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* selfShape = context->GetInputShape(0);
    const gert::Shape* otherShape = context->GetInputShape(1);
    gert::Shape* yShape = context->GetOutputShape(0);

    const int64_t lhsRank = static_cast<int64_t>(selfShape->GetDimNum());
    const int64_t rhsRank = static_cast<int64_t>(otherShape->GetDimNum());
    const int64_t outRank = lhsRank > rhsRank ? lhsRank : rhsRank;
    gert::Shape out;
    for (int64_t i = 0; i < outRank; ++i) {
        int64_t lhsIdx = lhsRank - outRank + i;
        int64_t rhsIdx = rhsRank - outRank + i;
        int64_t lhs = lhsIdx < 0 ? 1 : selfShape->GetDim(lhsIdx);
        int64_t rhs = rhsIdx < 0 ? 1 : otherShape->GetDim(rhsIdx);
        int64_t outDim = 0;
        if (lhs == rhs) {
            outDim = lhs;
        } else if (lhs == 1) {
            outDim = rhs;
        } else if (rhs == 1) {
            outDim = lhs;
        } else {
            return GRAPH_FAILED;
        }
        out.AppendDim(outDim);
    }
    *yShape = out;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310b");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(Greater);
}  // namespace ops
